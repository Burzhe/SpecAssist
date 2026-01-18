from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- DATA DIR ---
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- PATHS ---
EXCEL_PATH = BASE_DIR / "base" / "simpe1.xlsx"
DB_PATH = DATA_DIR / "app.db"


try:
    from config_local import *  # noqa
except ModuleNotFoundError:
    pass
