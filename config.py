from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

EXCEL_PATH = BASE_DIR / "base" / "summary.xlsx"
DB_PATH = BASE_DIR / "data" / "app.db"

try:
    from config_local import *  # noqa: F403
except ModuleNotFoundError:
    pass
