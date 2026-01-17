from __future__ import annotations

from typing import Dict

SHEET_COLUMN_MAP: dict[str, Dict[str, str]] = {
    "1": {
        "type_col": "Unnamed: 2",
        "dims_col": "Unnamed: 3",
        "desc_col": "Unnamed: 4",
        "photo_col": "Unnamed: 5",
        "qty_col": "Unnamed: 7",
        "price_unit_col": "Unnamed: 10",
        "total_col": "Unnamed: 11",
    },
    "вар2": {
        "type_col": "Unnamed: 2",
        "dims_col": "Unnamed: 3",
        "desc_col": "Unnamed: 4",
    },
    "КаБубАво 23": {
        "type_col": "Unnamed: 2",
        "dims_col": "Unnamed: 3",
        "desc_col": "Unnamed: 4",
    },
}

SHEET_NAMES = list(SHEET_COLUMN_MAP.keys())


def get_sheet_mapping(sheet_name: str) -> Dict[str, str]:
    return SHEET_COLUMN_MAP.get(sheet_name, {})


def get_sheet_names() -> list[str]:
    return SHEET_NAMES.copy()
