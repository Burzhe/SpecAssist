from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Hashable, Iterable

import pandas as pd

from config import DEFAULT_TOL_MM
from tools.db import ensure_schema, insert_items, rebuild_fts, set_meta

NAME_KEYWORDS = (
    "шкаф",
    "пенал",
    "гардероб",
    "встро",
    "стол",
    "бенч",
    "бар",
    "перил",
    "зеркал",
    "перегород",
    "двер",
    "панел",
)

MATERIAL_KEYWORDS = (
    "лдсп",
    "egger",
    "ламинирован",
    "мдф",
    "mdf",
    "шпон",
    "veneer",
    "стекл",
    "зеркал",
    "металл",
    "нерж",
    "сталь",
    "алюм",
    "порошк",
)

DIM_PATTERN = re.compile(r"\d{2,5}\s*[x×*х]\s*\d{2,5}\s*[x×*х]\s*\d{2,5}", re.IGNORECASE)
HEADER_SCAN_ROWS = 50


def compute_source_version(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()[:10]
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{path.stem}_{timestamp}_{digest}"


def import_workbook(path: Path, conn) -> dict[str, Any]:
    ensure_schema(conn)
    source_version = compute_source_version(path)

    conn.execute("DELETE FROM items WHERE source_version = ?", (source_version,))
    conn.execute("DELETE FROM sheet_schemas WHERE source_version = ?", (source_version,))
    conn.commit()

    excel = pd.ExcelFile(path)
    sheet_names = excel.sheet_names
    total_inserted = 0
    detected_sheets = 0
    skipped_sheets = 0

    for sheet_name in sheet_names:
        header_row = detect_header_row(path, sheet_name)
        if header_row is None:
            mapping, confidence, stats = _empty_mapping()
            stats["header_row"] = None
            detected = 0
        else:
            sample_df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row,
                nrows=300,
                engine="openpyxl",
            )
            mapping, confidence, stats = detect_sheet_mapping(sample_df)
            stats["header_row"] = header_row + 1
            detected = int(confidence >= 0.35 and mapping.get("name_col") is not None)
        conn.execute(
            """
            INSERT INTO sheet_schemas (source_version, sheet_name, detected, confidence, map_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                source_version,
                sheet_name,
                detected,
                confidence,
                json.dumps({"mapping": mapping, "stats": stats}, ensure_ascii=False),
            ),
        )
        conn.commit()

        if not detected:
            skipped_sheets += 1
            continue

        detected_sheets += 1
        if header_row is None:
            skipped_sheets += 1
            continue
        df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
        rows_to_insert: list[tuple] = []
        for row_idx, row in df.iterrows():
            source_row = header_row + 2 + row_idx
            parsed = parse_row(row, mapping, source_version, sheet_name, source_row)
            if parsed is None:
                continue
            rows_to_insert.append(parsed)
        if rows_to_insert:
            insert_items(conn, rows_to_insert)
            total_inserted += len(rows_to_insert)

    rebuild_fts(conn)
    set_meta(conn, "active_version", source_version)
    set_meta(conn, f"version_path:{source_version}", str(path))
    set_meta(conn, "last_import_at", datetime.utcnow().isoformat())
    set_meta(conn, "default_tol_mm", str(DEFAULT_TOL_MM))

    return {
        "source_version": source_version,
        "inserted": total_inserted,
        "detected_sheets": detected_sheets,
        "skipped_sheets": skipped_sheets,
        "total_sheets": len(sheet_names),
    }


def detect_header_row(path: Path, sheet_name: str) -> int | None:
    preview = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        nrows=HEADER_SCAN_ROWS,
        engine="openpyxl",
    )
    for row_idx, row in preview.iterrows():
        values = [value for value in row.tolist() if value is not None]
        if not values:
            continue
        normalized = [_normalize_header_text(value) for value in values]
        if not any(text and text != "nan" for text in normalized):
            continue
        if _is_header_match(values, normalized):
            return int(row_idx)
    return None


def detect_sheet_mapping(df: pd.DataFrame) -> tuple[dict[str, Hashable | None], float, dict[str, Any]]:
    stats: dict[str, Any] = {}
    scores: dict[str, dict[int, float]] = {
        "dims": {},
        "qty": {},
        "price": {},
        "name": {},
        "desc": {},
    }

    for col in df.columns:
        series = df[col]
        col_stats = analyze_column(series)
        stats[str(col)] = col_stats
        scores["dims"][col] = col_stats["dims_score"]
        scores["qty"][col] = col_stats["qty_score"]
        scores["price"][col] = col_stats["price_score"]
        scores["name"][col] = col_stats["name_score"]
        scores["desc"][col] = col_stats["desc_score"]

    mapping: dict[str, Hashable | None] = {
        "dims_col": _best_col(scores["dims"]),
        "qty_col": _best_col(scores["qty"]),
        "price_unit_col": None,
        "price_total_col": None,
        "name_col": _best_col(scores["name"]),
        "desc_col": _best_col(scores["desc"]),
    }

    price_sorted = sorted(scores["price"].items(), key=lambda item: item[1], reverse=True)
    if price_sorted:
        mapping["price_unit_col"] = price_sorted[0][0]
    if len(price_sorted) > 1:
        mapping["price_total_col"] = price_sorted[1][0]

    name_score = scores["name"].get(mapping["name_col"], 0.0) if mapping["name_col"] is not None else 0.0
    dims_score = scores["dims"].get(mapping["dims_col"], 0.0) if mapping["dims_col"] is not None else 0.0
    price_score = scores["price"].get(mapping["price_unit_col"], 0.0) if mapping["price_unit_col"] is not None else 0.0
    desc_score = scores["desc"].get(mapping["desc_col"], 0.0) if mapping["desc_col"] is not None else 0.0

    confidence = min(1.0, (name_score + max(dims_score, price_score) + (desc_score * 0.5)) / 2.2)
    return mapping, confidence, stats


def analyze_column(series: pd.Series) -> dict[str, float]:
    total = len(series)
    if total == 0:
        return {
            "dims_score": 0.0,
            "qty_score": 0.0,
            "price_score": 0.0,
            "name_score": 0.0,
            "desc_score": 0.0,
        }

    text_values: list[str] = []
    numeric_values: list[float] = []
    dims_hits = 0
    name_hits = 0
    material_hits = 0

    for value in series.tolist():
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if DIM_PATTERN.search(text):
            dims_hits += 1
        text_values.append(text)
        if _to_float(value) is not None:
            numeric_values.append(_to_float(value))  # type: ignore[arg-type]
        lowered = text.lower()
        if any(token in lowered for token in NAME_KEYWORDS):
            name_hits += 1
        if any(token in lowered for token in MATERIAL_KEYWORDS):
            material_hits += 1

    text_ratio = len(text_values) / total
    numeric_ratio = len(numeric_values) / total
    dims_ratio = dims_hits / total

    avg_len = sum(len(text) for text in text_values) / len(text_values) if text_values else 0.0
    name_ratio = name_hits / len(text_values) if text_values else 0.0
    material_ratio = material_hits / len(text_values) if text_values else 0.0

    qty_score = 0.0
    price_score = 0.0
    if numeric_values:
        mean_val = sum(numeric_values) / len(numeric_values)
        median_val = sorted(numeric_values)[len(numeric_values) // 2]
        if 1 <= mean_val <= 100:
            qty_score = numeric_ratio * 0.9
        elif 0 < mean_val <= 200:
            qty_score = numeric_ratio * 0.5

        if 100 <= median_val <= 1e7:
            price_score = numeric_ratio * (1.0 if median_val >= 1000 else 0.6)

    name_score = text_ratio * (0.6 if 3 <= avg_len <= 60 else 0.3) + name_ratio * 0.8
    desc_score = text_ratio * min(avg_len / 120, 1.0) + material_ratio * 0.6

    return {
        "dims_score": dims_ratio,
        "qty_score": qty_score,
        "price_score": price_score,
        "name_score": name_score,
        "desc_score": desc_score,
    }


def _empty_mapping() -> tuple[dict[str, Hashable | None], float, dict[str, Any]]:
    return (
        {
            "dims_col": None,
            "qty_col": None,
            "price_unit_col": None,
            "price_total_col": None,
            "name_col": None,
            "desc_col": None,
        },
        0.0,
        {},
    )


def _normalize_header_text(value: Any) -> str:
    text = str(value).strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text)


def _is_header_match(values: list[Any], normalized: list[str]) -> bool:
    normalized_clean = [text for text in normalized if text and text != "nan"]
    has_name = any("наименование" in text for text in normalized_clean)
    has_position = any(
        "позиция" in text or "№" in str(raw) for raw, text in zip(values, normalized)
    )
    has_price_qty = any(
        any(token in text for token in ("кол-во", "кол во", "итого", "цена"))
        for text in normalized_clean
    )
    return has_name and has_position and has_price_qty


def _best_col(score_map: dict[Hashable, float]) -> Hashable | None:
    if not score_map:
        return None
    best_col, best_score = max(score_map.items(), key=lambda item: item[1])
    if best_score <= 0:
        return None
    return best_col


def parse_row(
    row: pd.Series,
    mapping: dict[str, Hashable | None],
    source_version: str,
    sheet_name: str,
    source_row: int,
) -> tuple | None:
    name = _normalize_text(_get_value(row, mapping.get("name_col")))
    description = _normalize_text(_get_value(row, mapping.get("desc_col")))

    if not name:
        return None

    dims_raw = _get_value(row, mapping.get("dims_col"))
    w_mm, d_mm, h_mm = _parse_dimensions(dims_raw)
    if w_mm is None and d_mm is None and h_mm is None:
        w_mm, d_mm, h_mm = _parse_dimensions(" ".join(filter(None, (name, description))))

    qty = _to_float(_get_value(row, mapping.get("qty_col")))
    price_unit_val = _get_value(row, mapping.get("price_unit_col"))
    price_total_val = _get_value(row, mapping.get("price_total_col"))
    price_unit_ex_vat, price_total_ex_vat = _extract_price(price_unit_val, price_total_val, qty)

    if _is_invalid_row(name, description, qty, price_unit_ex_vat, price_total_ex_vat):
        return None

    if not _has_signal(name, description, w_mm, d_mm, h_mm, price_unit_ex_vat, price_total_ex_vat):
        return None

    flags_text = _normalize_for_flags(f"{name or ''} {description or ''}")
    has_led = int(any(token in flags_text for token in ("подсвет", "подсветка", "led", "лента")))
    mat_ldsp = int(any(token in flags_text for token in ("лдсп", "egger", "ламинирован")))
    mat_mdf = int(any(token in flags_text for token in ("мдф", "mdf")))
    mat_veneer = int(any(token in flags_text for token in ("шпон", "veneer")))
    has_glass = int(any(token in flags_text for token in ("стекл", "зеркал")))
    has_metal = int(any(token in flags_text for token in ("металл", "нерж", "сталь", "алюм", "порошк")))

    raw_json = json.dumps(row.to_dict(), ensure_ascii=False)

    return (
        source_version,
        sheet_name,
        source_row,
        name,
        description,
        w_mm,
        d_mm,
        h_mm,
        qty,
        price_unit_ex_vat,
        price_total_ex_vat,
        has_led,
        mat_ldsp,
        mat_mdf,
        mat_veneer,
        has_glass,
        has_metal,
        raw_json,
    )


def _get_value(row: pd.Series, col: Hashable | None) -> Any:
    if col is None:
        return None
    return row.get(col)


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "nan" or lowered == "-":
        return None
    return text


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_dimensions(value: Any) -> tuple[int | None, int | None, int | None]:
    if value is None:
        return (None, None, None)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return (None, None, None)
    normalized = _normalize_for_flags(text).replace("мм", "").replace("mm", "")
    normalized = re.sub(r"[×*хx]", "x", normalized)
    w_mm = _extract_prefixed_dim(normalized, ("w", "ш", "ширин"))
    d_mm = _extract_prefixed_dim(normalized, ("d", "г", "глубин"))
    h_mm = _extract_prefixed_dim(normalized, ("h", "в", "высот"))
    numbers = re.findall(r"\d{2,5}", normalized)
    if len(numbers) >= 3 and not all((w_mm, d_mm, h_mm)):
        w_mm = w_mm or int(numbers[0])
        d_mm = d_mm or int(numbers[1])
        h_mm = h_mm or int(numbers[2])
    return (w_mm, d_mm, h_mm)


def _extract_price(
    price_unit_value: Any,
    price_total_value: Any,
    qty_value: float | None,
) -> tuple[float | None, float | None]:
    price_unit = _to_float(price_unit_value)
    price_total = _to_float(price_total_value)

    if price_unit and price_unit > 0:
        return (price_unit, price_total)
    if price_total and qty_value and qty_value > 0:
        return (price_total / qty_value, price_total)
    return (price_unit, price_total)


def _normalize_for_flags(text: str) -> str:
    lowered = text.lower().replace("ё", "е")
    lowered = re.sub(r"[-–—]+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _extract_prefixed_dim(text: str, stems: tuple[str, ...]) -> int | None:
    for stem in stems:
        match = re.search(rf"{stem}\s*[:=]?\s*(\d{{2,5}})", text)
        if match:
            return int(match.group(1))
    return None


def _looks_like_header(name: str) -> bool:
    lowered = _normalize_for_flags(name)
    header_tokens = (
        "наименование",
        "материал",
        "материалы",
        "описание",
        "ед изм",
        "единиц",
        "кол во",
        "количество",
        "цена",
        "итого",
        "стоимость",
        "позиция",
        "номер",
    )
    return any(token in lowered for token in header_tokens)


def _is_invalid_row(
    name: str,
    description: str | None,
    qty: float | None,
    price_unit: float | None,
    price_total: float | None,
) -> bool:
    if name.strip().lower() in {"nan", "-"}:
        return True
    if _looks_like_header(name) and not (qty or price_unit or price_total):
        return True
    if not description and not (qty or price_unit or price_total):
        return True
    return False


def _has_signal(
    name: str | None,
    description: str | None,
    w_mm: int | None,
    d_mm: int | None,
    h_mm: int | None,
    price_unit: float | None,
    price_total: float | None,
) -> bool:
    if name or description:
        keywords = (name or "") + (description or "")
        lowered = keywords.lower()
        if any(token in lowered for token in NAME_KEYWORDS):
            return True
    if any(val is not None for val in (w_mm, d_mm, h_mm)):
        return True
    if price_unit or price_total:
        return True
    return False
