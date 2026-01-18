from __future__ import annotations

import hashlib
import json
import logging
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
CANONICAL_FIELDS = (
    "name",
    "dims",
    "desc",
    "qty",
    "unit",
    "price_material",
    "price_install",
    "price_unit",
    "total",
    "comment",
)
LOGGER = logging.getLogger(__name__)


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
    skipped_rows = 0
    sheet_reports: list[dict[str, Any]] = []
    summary = {
        "sheets_total": 0,
        "sheets_ok": 0,
        "sheets_missing_price_unit": 0,
        "sheets_missing_qty": 0,
        "rows_total": 0,
        "rows_inserted": 0,
        "rows_skipped": 0,
        "rows_unit_price_from_price_unit": 0,
        "rows_unit_price_from_material_install": 0,
        "rows_unit_price_from_total_div_qty": 0,
        "sheets_problematic": 0,
    }

    for sheet_name in sheet_names:
        summary["sheets_total"] += 1
        header_row = detect_header_row(path, sheet_name)
        if header_row is None:
            mapping, confidence, stats = _empty_mapping()
            stats["header_row"] = None
            detected = 0
            columns: list[Hashable] = []
            rows_total = 0
            rows_inserted = 0
            rows_skipped = 0
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
            columns = list(sample_df.columns)
            rows_total = 0
            rows_inserted = 0
            rows_skipped = 0
        LOGGER.info(
            "Detected sheet mapping: %s -> %s",
            sheet_name,
            _format_mapping_for_log(mapping),
        )
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

        missing_critical = _missing_critical_fields(mapping)
        if mapping.get("price_unit_col") is None:
            summary["sheets_missing_price_unit"] += 1
        if mapping.get("qty_col") is None:
            summary["sheets_missing_qty"] += 1
        if missing_critical:
            summary["sheets_problematic"] += 1
        else:
            summary["sheets_ok"] += 1

        if not detected:
            skipped_sheets += 1
            sheet_reports.append(
                _build_sheet_report(
                    sheet_name=sheet_name,
                    header_row=stats["header_row"],
                    rows_total=rows_total,
                    rows_inserted=rows_inserted,
                    rows_skipped=rows_skipped,
                    mapping=mapping,
                    columns=columns,
                )
            )
            continue

        detected_sheets += 1
        if header_row is None:
            skipped_sheets += 1
            sheet_reports.append(
                _build_sheet_report(
                    sheet_name=sheet_name,
                    header_row=stats["header_row"],
                    rows_total=rows_total,
                    rows_inserted=rows_inserted,
                    rows_skipped=rows_skipped,
                    mapping=mapping,
                    columns=columns,
                )
            )
            continue
        df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
        columns = list(df.columns)
        rows_to_insert: list[tuple] = []
        for row_idx, row in df.iterrows():
            rows_total += 1
            source_row = header_row + 2 + row_idx
            parsed, price_source = parse_row(
                row,
                mapping,
                source_version,
                sheet_name,
                source_row,
            )
            if parsed is None:
                skipped_rows += 1
                rows_skipped += 1
                continue
            rows_to_insert.append(parsed)
            rows_inserted += 1
            if price_source == "price_unit":
                summary["rows_unit_price_from_price_unit"] += 1
            elif price_source == "material_install":
                summary["rows_unit_price_from_material_install"] += 1
            elif price_source == "total_div_qty":
                summary["rows_unit_price_from_total_div_qty"] += 1
        if rows_to_insert:
            insert_items(conn, rows_to_insert)
            total_inserted += len(rows_to_insert)
        summary["rows_total"] += rows_total
        summary["rows_inserted"] += rows_inserted
        summary["rows_skipped"] += rows_skipped
        sheet_reports.append(
            _build_sheet_report(
                sheet_name=sheet_name,
                header_row=stats["header_row"],
                rows_total=rows_total,
                rows_inserted=rows_inserted,
                rows_skipped=rows_skipped,
                mapping=mapping,
                columns=columns,
            )
        )

    rebuild_fts(conn)
    stats = _compute_import_stats(conn, source_version)
    set_meta(conn, "active_version", source_version)
    set_meta(conn, f"version_path:{source_version}", str(path))
    set_meta(conn, "last_import_at", datetime.utcnow().isoformat())
    set_meta(conn, "default_tol_mm", str(DEFAULT_TOL_MM))
    set_meta(conn, "stats:total_items", str(stats["total_items"]))
    set_meta(conn, "stats:valid_items", str(stats["valid_items"]))
    set_meta(conn, "stats:sheets_detected", str(detected_sheets))
    set_meta(conn, "stats:skipped_rows", str(skipped_rows))
    set_meta(conn, "stats:rows_with_price_unit", str(stats["rows_with_price_unit"]))
    set_meta(conn, "stats:rows_with_total_and_qty", str(stats["rows_with_total_and_qty"]))
    set_meta(conn, "stats:sheets_total", str(summary["sheets_total"]))
    set_meta(conn, "stats:sheets_ok", str(summary["sheets_ok"]))
    set_meta(conn, "stats:sheets_missing_price_unit", str(summary["sheets_missing_price_unit"]))
    set_meta(conn, "stats:sheets_missing_qty", str(summary["sheets_missing_qty"]))
    set_meta(conn, "stats:rows_total", str(summary["rows_total"]))
    set_meta(conn, "stats:rows_inserted", str(summary["rows_inserted"]))
    set_meta(conn, "stats:rows_skipped", str(summary["rows_skipped"]))
    set_meta(
        conn,
        "stats:rows_unit_price_from_price_unit",
        str(summary["rows_unit_price_from_price_unit"]),
    )
    set_meta(
        conn,
        "stats:rows_unit_price_from_material_install",
        str(summary["rows_unit_price_from_material_install"]),
    )
    set_meta(
        conn,
        "stats:rows_unit_price_from_total_div_qty",
        str(summary["rows_unit_price_from_total_div_qty"]),
    )
    set_meta(conn, "stats:sheets_problematic", str(summary["sheets_problematic"]))

    return {
        "source_version": source_version,
        "inserted": total_inserted,
        "detected_sheets": detected_sheets,
        "skipped_sheets": skipped_sheets,
        "total_sheets": len(sheet_names),
        "skipped_rows": skipped_rows,
        "stats": stats,
        "sheet_reports": sheet_reports,
        "summary": summary,
    }


def detect_header_row(path: Path, sheet_name: str, *, max_rows_scan: int = HEADER_SCAN_ROWS) -> int | None:
    preview = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        nrows=max_rows_scan,
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
    mapping: dict[str, Hashable | None] = {
        "name_col": None,
        "dims_col": None,
        "desc_col": None,
        "qty_col": None,
        "unit_col": None,
        "price_material_col": None,
        "price_install_col": None,
        "price_unit_col": None,
        "total_col": None,
        "comment_col": None,
    }

    for col in df.columns:
        normalized = _normalize_header_text(col)
        stats[str(col)] = {"normalized": normalized}
        canonical = _match_header_to_field(normalized)
        if canonical is None:
            continue
        key = f"{canonical}_col"
        if mapping.get(key) is None:
            mapping[key] = col

    confidence = 1.0 if mapping.get("name_col") is not None else 0.0
    return mapping, confidence, stats


def debug_workbook_mapping(
    path: Path,
    *,
    limit_sheets: int | None = None,
    max_rows_scan: int = HEADER_SCAN_ROWS,
) -> dict[str, Any]:
    excel = pd.ExcelFile(path)
    sheet_names = excel.sheet_names
    if limit_sheets is not None:
        sheet_names = sheet_names[:limit_sheets]
    sheet_reports: list[dict[str, Any]] = []
    for sheet_name in sheet_names:
        header_row = detect_header_row(path, sheet_name, max_rows_scan=max_rows_scan)
        if header_row is None:
            mapping, _, stats = _empty_mapping()
            stats["header_row"] = None
            columns: list[Hashable] = []
            rows_total = 0
            rows_inserted = 0
            rows_skipped = 0
        else:
            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row,
                nrows=max_rows_scan,
                engine="openpyxl",
            )
            mapping, _, stats = detect_sheet_mapping(df)
            stats["header_row"] = header_row + 1
            columns = list(df.columns)
            rows_total = 0
            rows_inserted = 0
            rows_skipped = 0
            for row_idx, row in df.iterrows():
                rows_total += 1
                source_row = header_row + 2 + row_idx
                parsed, _ = parse_row(row, mapping, "debug", sheet_name, source_row)
                if parsed is None:
                    rows_skipped += 1
                else:
                    rows_inserted += 1
        sheet_reports.append(
            _build_sheet_report(
                sheet_name=sheet_name,
                header_row=stats["header_row"],
                rows_total=rows_total,
                rows_inserted=rows_inserted,
                rows_skipped=rows_skipped,
                mapping=mapping,
                columns=columns,
            )
        )
    return {"sheet_reports": sheet_reports}


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
            "name_col": None,
            "dims_col": None,
            "desc_col": None,
            "qty_col": None,
            "unit_col": None,
            "price_material_col": None,
            "price_install_col": None,
            "price_unit_col": None,
            "total_col": None,
            "comment_col": None,
        },
        0.0,
        {},
    )


def _normalize_header_text(value: Any) -> str:
    text = str(value).strip().lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _match_header_to_field(normalized: str) -> str | None:
    if not normalized or normalized == "nan":
        return None
    if "артикул" in normalized:
        return None
    has_price = "цена" in normalized
    if "итого" in normalized:
        return "total"
    if "наимен" in normalized:
        return "name"
    if "размер" in normalized or "шхгхв" in normalized:
        return "dims"
    if "материал" in normalized or "описан" in normalized:
        return "desc"
    if "колво" in normalized or "кол" in normalized:
        return "qty"
    if "комментар" in normalized:
        return "comment"
    if "ед" in normalized and ("изм" in normalized or "едизм" in normalized) and not has_price:
        return "unit"
    if has_price and "материал" in normalized:
        return "price_material"
    if has_price and ("монтаж" in normalized or "достав" in normalized):
        return "price_install"
    if has_price and ("издел" in normalized or ("ед" in normalized and "изм" in normalized)):
        return "price_unit"
    return None


def _is_header_match(values: list[Any], normalized: list[str]) -> bool:
    normalized_clean = [text for text in normalized if text and text != "nan"]
    has_name = any("наимен" in text for text in normalized_clean)
    has_position = any(
        "позиция" in text or "№" in str(raw) for raw, text in zip(values, normalized)
    )
    has_price_qty = any(
        any(token in text for token in ("кол", "итого", "цена"))
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
) -> tuple[tuple | None, str | None]:
    name = _normalize_text(_get_value(row, mapping.get("name_col")))
    description = _normalize_text(_get_value(row, mapping.get("desc_col")))

    if not name:
        return None, None

    dims_raw = _get_value(row, mapping.get("dims_col"))
    w_mm, d_mm, h_mm = _parse_dimensions(dims_raw)
    if w_mm is None and d_mm is None and h_mm is None:
        w_mm, d_mm, h_mm = _parse_dimensions(" ".join(filter(None, (name, description))))

    qty = _to_float(_get_value(row, mapping.get("qty_col")))
    price_unit_val = _get_value(row, mapping.get("price_unit_col"))
    price_material_val = _get_value(row, mapping.get("price_material_col"))
    price_install_val = _get_value(row, mapping.get("price_install_col"))
    price_total_val = _get_value(row, mapping.get("total_col"))
    price_total_ex_vat = _to_float(price_total_val)
    price_unit_ex_vat, unit_price_source = compute_unit_price_with_source(
        price_unit=price_unit_val,
        price_material=price_material_val,
        price_install=price_install_val,
        total=price_total_ex_vat,
        qty=qty,
    )

    if _is_invalid_row(name, description, qty, price_unit_ex_vat, price_total_ex_vat):
        return None, None

    if not _has_signal(name, description, w_mm, d_mm, h_mm, price_unit_ex_vat, price_total_ex_vat):
        return None, None

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
    ), unit_price_source


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


def compute_unit_price(
    *,
    price_unit: Any,
    price_material: Any,
    price_install: Any,
    total: float | None,
    qty: float | None,
) -> float | None:
    value, _ = compute_unit_price_with_source(
        price_unit=price_unit,
        price_material=price_material,
        price_install=price_install,
        total=total,
        qty=qty,
    )
    return value


def compute_unit_price_with_source(
    *,
    price_unit: Any,
    price_material: Any,
    price_install: Any,
    total: float | None,
    qty: float | None,
) -> tuple[float | None, str | None]:
    unit_value = _to_float(price_unit)
    if unit_value is not None and unit_value > 0:
        return unit_value, "price_unit"
    material_value = _to_float(price_material)
    install_value = _to_float(price_install)
    if material_value is not None or install_value is not None:
        return (material_value or 0.0) + (install_value or 0.0), "material_install"
    if total is not None and qty is not None and qty > 0:
        return total / qty, "total_div_qty"
    return None, None


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


def _format_mapping_for_log(mapping: dict[str, Hashable | None]) -> dict[str, str | None]:
    formatted: dict[str, str | None] = {}
    for field in CANONICAL_FIELDS:
        key = f"{field}_col"
        value = mapping.get(key)
        formatted[field] = None if value is None else str(value)
    return formatted


def _missing_price_source(mapping: dict[str, Hashable | None]) -> bool:
    has_unit = mapping.get("price_unit_col") is not None
    has_material = mapping.get("price_material_col") is not None
    has_install = mapping.get("price_install_col") is not None
    has_total_qty = mapping.get("total_col") is not None and mapping.get("qty_col") is not None
    return not (has_unit or has_material or has_install or has_total_qty)


def _missing_critical_fields(mapping: dict[str, Hashable | None]) -> list[str]:
    missing: list[str] = []
    if mapping.get("name_col") is None:
        missing.append("name")
    if _missing_price_source(mapping):
        missing.append("price_source")
    return missing


def _unused_headers(columns: list[Hashable], mapping: dict[str, Hashable | None]) -> list[str]:
    used = {str(value) for value in mapping.values() if value is not None}
    unused = [str(col) for col in columns if str(col) not in used]
    return unused


def _build_sheet_report(
    *,
    sheet_name: str,
    header_row: int | None,
    rows_total: int,
    rows_inserted: int,
    rows_skipped: int,
    mapping: dict[str, Hashable | None],
    columns: list[Hashable],
) -> dict[str, Any]:
    return {
        "sheet_name": sheet_name,
        "header_row": header_row,
        "rows_total": rows_total,
        "rows_inserted": rows_inserted,
        "rows_skipped": rows_skipped,
        "mapping": _format_mapping_for_log(mapping),
        "unused_headers": _unused_headers(columns, mapping),
        "missing_critical_fields": _missing_critical_fields(mapping),
    }


def _compute_import_stats(conn, source_version: str) -> dict[str, int]:
    total_items = conn.execute(
        "SELECT COUNT(*) FROM items WHERE source_version = ?",
        (source_version,),
    ).fetchone()[0]
    valid_items = conn.execute(
        "SELECT COUNT(*) FROM items WHERE source_version = ? AND is_valid = 1",
        (source_version,),
    ).fetchone()[0]
    rows_with_price_unit = conn.execute(
        """
        SELECT COUNT(*) FROM items
        WHERE source_version = ?
          AND price_unit_ex_vat IS NOT NULL
          AND price_unit_ex_vat > 0
        """,
        (source_version,),
    ).fetchone()[0]
    rows_with_total_and_qty = conn.execute(
        """
        SELECT COUNT(*) FROM items
        WHERE source_version = ?
          AND price_total_ex_vat IS NOT NULL
          AND qty IS NOT NULL
          AND qty > 0
        """,
        (source_version,),
    ).fetchone()[0]
    return {
        "total_items": int(total_items),
        "valid_items": int(valid_items),
        "rows_with_price_unit": int(rows_with_price_unit),
        "rows_with_total_and_qty": int(rows_with_total_and_qty),
    }


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
