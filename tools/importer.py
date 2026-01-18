from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import DEFAULT_TOL_MM
from tools.db import ensure_schema, insert_items, rebuild_fts, set_meta

NAME_KEYWORDS = (
    "шкаф",
    "кухн",
    "тумб",
    "гардероб",
    "стойк",
    "кровать",
    "панел",
    "двер",
)

MATERIAL_KEYWORDS = (
    "лдсп",
    "egger",
    "мдф",
    "mdf",
    "шпон",
    "veneer",
    "стекл",
    "зеркал",
    "металл",
    "нерж",
    "сталь",
)

DIM_PATTERN = re.compile(r"\d{2,5}\s*[x×*х]\s*\d{2,5}\s*[x×*х]\s*\d{2,5}", re.IGNORECASE)


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
        sample_df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=None,
            nrows=300,
            engine="openpyxl",
        )
        mapping, confidence, stats = detect_sheet_mapping(sample_df)
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
        df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
        rows_to_insert: list[tuple] = []
        for row_idx, row in df.iterrows():
            parsed = parse_row(row, mapping, source_version, sheet_name, row_idx + 1)
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


def detect_sheet_mapping(df: pd.DataFrame) -> tuple[dict[str, int | None], float, dict[str, Any]]:
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

    mapping: dict[str, int | None] = {
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


def _best_col(score_map: dict[int, float]) -> int | None:
    if not score_map:
        return None
    best_col, best_score = max(score_map.items(), key=lambda item: item[1])
    if best_score <= 0:
        return None
    return best_col


def parse_row(
    row: pd.Series,
    mapping: dict[str, int | None],
    source_version: str,
    sheet_name: str,
    source_row: int,
) -> tuple | None:
    name = _normalize_text(_get_value(row, mapping.get("name_col")))
    description = _normalize_text(_get_value(row, mapping.get("desc_col")))

    if not name and not description:
        return None

    dims_raw = _get_value(row, mapping.get("dims_col"))
    w_mm, d_mm, h_mm = _parse_dimensions(dims_raw)

    qty = _to_float(_get_value(row, mapping.get("qty_col")))
    price_unit_val = _get_value(row, mapping.get("price_unit_col"))
    price_total_val = _get_value(row, mapping.get("price_total_col"))
    price_unit_ex_vat, price_total_ex_vat = _extract_price(price_unit_val, price_total_val, qty)

    if not _has_signal(name, description, w_mm, d_mm, h_mm, price_unit_ex_vat, price_total_ex_vat):
        return None

    flags_text = f"{name or ''} {description or ''}".lower()
    has_led = int(any(token in flags_text for token in ("подсвет", "led", "лента")))
    mat_ldsp = int(any(token in flags_text for token in ("лдсп", "egger")))
    mat_mdf = int(any(token in flags_text for token in ("мдф", "mdf")))
    mat_veneer = int(any(token in flags_text for token in ("шпон", "veneer")))
    has_glass = int(any(token in flags_text for token in ("стекл", "зеркал")))
    has_metal = int(any(token in flags_text for token in ("металл", "нерж", "сталь")))

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


def _get_value(row: pd.Series, col: int | None) -> Any:
    if col is None:
        return None
    return row.get(col)


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
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
    normalized = text.lower().replace("мм", "").replace("mm", "")
    normalized = re.sub(r"[×*хx]", "x", normalized)
    numbers = re.findall(r"\d{2,5}", normalized)
    if len(numbers) < 3:
        return (None, None, None)
    w_mm, d_mm, h_mm = (int(num) for num in numbers[:3])
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
