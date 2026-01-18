from __future__ import annotations

import re
import sqlite3
from typing import Any

import pandas as pd

from config import DB_PATH, EXCEL_PATH
from schema_map import get_sheet_mapping, get_sheet_names

LIGHT_TOKENS = (
    "light",
    "подсвет",
    "led",
    "свет",
    "лента",
    "illum",
)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(" ", "")
    cleaned = cleaned.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_dimensions(value: Any) -> tuple[float | None, float | None, float | None]:
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
    w_mm, h_mm, d_mm = (float(num) for num in numbers[:3])
    return (w_mm, h_mm, d_mm)


def _detect_light(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).lower()
    return int(any(token in text for token in LIGHT_TOKENS))


def _ensure_db_directory() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT,
            dims_raw TEXT,
            w_mm REAL,
            h_mm REAL,
            d_mm REAL,
            description TEXT,
            photo TEXT,
            qty REAL,
            price_unit REAL,
            total REAL,
            price REAL,
            light INTEGER,
            source_sheet TEXT,
            source_row INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX idx_items_type ON items(item_type)")
    conn.execute("CREATE INDEX idx_items_price ON items(price)")
    conn.execute("CREATE INDEX idx_items_sheet_row ON items(source_sheet, source_row)")


def _get_value(row: pd.Series, column: str | None) -> Any:
    if not column:
        return None
    return row.get(column)


def _load_sheet(sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(EXCEL_PATH, sheet_name=sheet_name, engine="openpyxl")


def _extract_price(
    price_unit_value: Any,
    total_value: Any,
    qty_value: Any,
) -> tuple[float | None, float | None, float | None, float | None]:
    price_unit = _to_float(price_unit_value)
    total = _to_float(total_value)
    qty = _to_float(qty_value)

    if price_unit and price_unit > 0:
        return (price_unit, total, qty, price_unit)

    if total and qty:
        return (price_unit, total, qty, total / qty)

    return (price_unit, total, qty, None)


def import_excel(recreate: bool = False) -> int:
    _ensure_db_directory()
    if recreate and DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    if recreate or not _table_exists(conn, "items"):
        _create_schema(conn)

    inserted = 0
    for sheet_name in get_sheet_names():
        mapping = get_sheet_mapping(sheet_name)
        if not mapping:
            continue
        df = _load_sheet(sheet_name)

        for idx, row in df.iterrows():
            type_value = _get_value(row, mapping.get("type_col"))
            dims_value = _get_value(row, mapping.get("dims_col"))
            desc_value = _get_value(row, mapping.get("desc_col"))

            if not is_meaningful_text(type_value) and not is_meaningful_text(
                desc_value
            ):
                continue

            if str(type_value).strip().lower() == "nan":
                continue

            w_mm, h_mm, d_mm = _parse_dimensions(dims_value)
            light = _detect_light(desc_value)

            price_unit_value = _get_value(row, mapping.get("price_unit_col"))
            total_value = _get_value(row, mapping.get("total_col"))
            qty_value = _get_value(row, mapping.get("qty_col"))
            price_unit, total, qty, price = _extract_price(
                price_unit_value, total_value, qty_value
            )

            photo = _get_value(row, mapping.get("photo_col"))

            conn.execute(
                """
                INSERT INTO items (
                    item_type,
                    dims_raw,
                    w_mm,
                    h_mm,
                    d_mm,
                    description,
                    photo,
                    qty,
                    price_unit,
                    total,
                    price,
                    light,
                    source_sheet,
                    source_row
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _normalize_text(type_value),
                    _normalize_text(dims_value),
                    w_mm,
                    h_mm,
                    d_mm,
                    _normalize_text(desc_value),
                    _normalize_text(photo),
                    qty,
                    price_unit,
                    total,
                    price,
                    light,
                    sheet_name,
                    idx + 2,
                ),
            )
            inserted += 1

    conn.commit()
    conn.close()
    return inserted


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone() is not None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_empty(value: Any) -> bool:
    return _normalize_text(value) is None


def is_meaningful_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() != "nan"


def _format_dims(w_mm: float | None, h_mm: float | None, d_mm: float | None) -> str:
    dims = [val for val in (w_mm, h_mm, d_mm) if val is not None]
    if not dims:
        return "-"
    formatted: list[str] = []
    for val in dims:
        if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
            formatted.append(f"{val:.0f}")
        else:
            formatted.append(f"{val:.2f}")
    return "x".join(formatted)


def sample_items(limit: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
    SELECT item_type, w_mm, h_mm, d_mm, price, light, source_sheet, source_row
    FROM items
    WHERE price IS NOT NULL AND price > 0
      AND (w_mm IS NOT NULL OR h_mm IS NOT NULL OR d_mm IS NOT NULL)
    ORDER BY id
    LIMIT ?
    """,
    (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    print(f"sample rows: {len(rows)}")
    for item_type, w_mm, h_mm, d_mm, price, light, sheet, row in rows:
        dims = _format_dims(w_mm, h_mm, d_mm)
        light_flag = "yes" if light else "no"
        print(
            f"type={item_type or '-'} | dims={dims} | price={price:.2f} | light={light_flag} | "
            f"sheet={sheet} row={row}"
        )


def main(recreate: bool = False) -> None:
    inserted = import_excel(recreate=recreate)
    print(f"Imported {inserted} rows into {DB_PATH}")


if __name__ == "__main__":
    main(recreate=True)
