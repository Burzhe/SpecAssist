from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- DATA DIR ---
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- PATHS ---
EXCEL_PATH = os.getenv("EXCEL_PATH", "")
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "app.db")))

# --- BOT CONFIG ---
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "")
ADMIN_IDS: list[int] = []
for chunk in ADMIN_IDS_RAW.replace(";", ",").split(","):
    chunk = chunk.strip()
    if not chunk:
        continue
    try:
        ADMIN_IDS.append(int(chunk))
    except ValueError:
        continue

DEFAULT_TOL_MM = int(os.getenv("DEFAULT_TOL_MM", "50"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))
MAX_TG_FILE_MB = int(os.getenv("MAX_TG_FILE_MB", "20"))


try:
    from config_local import *  # noqa
except ModuleNotFoundError:
    pass
