from __future__ import annotations

import json
import sqlite3
from typing import Iterable

from config import DB_PATH

ITEM_COLUMNS: dict[str, str] = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "source_version": "TEXT",
    "source_sheet": "TEXT",
    "source_row": "INTEGER",
    "name": "TEXT",
    "description": "TEXT",
    "w_mm": "INTEGER",
    "d_mm": "INTEGER",
    "h_mm": "INTEGER",
    "qty": "REAL",
    "price_unit_ex_vat": "REAL",
    "price_total_ex_vat": "REAL",
    "has_led": "INTEGER",
    "mat_ldsp": "INTEGER",
    "mat_mdf": "INTEGER",
    "mat_veneer": "INTEGER",
    "has_glass": "INTEGER",
    "has_metal": "INTEGER",
    "is_valid": "INTEGER DEFAULT 1",
    "raw_json": "TEXT",
}


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    _ensure_table(conn, "items", ITEM_COLUMNS)
    _ensure_table(
        conn,
        "sheet_schemas",
        {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "source_version": "TEXT",
            "sheet_name": "TEXT",
            "detected": "INTEGER",
            "confidence": "REAL",
            "map_json": "TEXT",
        },
    )
    _ensure_table(conn, "meta", {"key": "TEXT PRIMARY KEY", "value": "TEXT"})
    _ensure_table(
        conn,
        "allowed_users",
        {
            "user_id": "INTEGER PRIMARY KEY",
            "username": "TEXT",
            "first_name": "TEXT",
            "last_name": "TEXT",
            "added_by": "INTEGER",
            "added_at": "TEXT",
        },
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_items_version ON items(source_version)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_items_sheet ON items(source_sheet)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_items_dims ON items(w_mm, d_mm, h_mm)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_items_price ON items(price_unit_ex_vat)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sheet_schema_version ON sheet_schemas(source_version)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sheet_schema_sheet ON sheet_schemas(sheet_name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_allowed_users_added_by ON allowed_users(added_by)"
    )
    conn.commit()

    if not _table_exists(conn, "items_fts"):
        create_fts(conn)


def _ensure_table(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    if not _table_exists(conn, table):
        cols = ", ".join([f"{name} {definition}" for name, definition in columns.items()])
        conn.execute(f"CREATE TABLE {table} ({cols})")
        conn.commit()
        return

    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone() is not None


def create_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS items_fts")
    conn.execute(
        """
        CREATE VIRTUAL TABLE items_fts
        USING fts5(name, description, content='items', content_rowid='id')
        """
    )
    conn.commit()


def rebuild_fts(conn: sqlite3.Connection) -> None:
    create_fts(conn)
    conn.execute(
        "INSERT INTO items_fts(rowid, name, description) SELECT id, name, description FROM items"
    )
    conn.commit()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    cursor = conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
    row = cursor.fetchone()
    if not row:
        return None
    return str(row["value"])


def list_versions(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.execute("SELECT DISTINCT source_version FROM items ORDER BY source_version")
    return [row[0] for row in cursor.fetchall() if row[0]]


def list_allowed_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    cursor = conn.execute(
        "SELECT user_id, username, first_name, last_name, added_by, added_at FROM allowed_users"
    )
    return cursor.fetchall()


def add_allowed_user(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    username: str | None,
    first_name: str | None,
    last_name: str | None,
    added_by: int | None,
    added_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO allowed_users (user_id, username, first_name, last_name, added_by, added_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            username=excluded.username,
            first_name=excluded.first_name,
            last_name=excluded.last_name,
            added_by=excluded.added_by,
            added_at=excluded.added_at
        """,
        (user_id, username, first_name, last_name, added_by, added_at),
    )
    conn.commit()


def remove_allowed_user(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("DELETE FROM allowed_users WHERE user_id = ?", (user_id,))
    conn.commit()


def is_user_allowed(conn: sqlite3.Connection, user_id: int) -> bool:
    cursor = conn.execute("SELECT 1 FROM allowed_users WHERE user_id = ? LIMIT 1", (user_id,))
    return cursor.fetchone() is not None


def dump_json(value: dict | list | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def insert_items(
    conn: sqlite3.Connection,
    rows: Iterable[tuple],
) -> None:
    conn.executemany(
        """
        INSERT INTO items (
            source_version,
            source_sheet,
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
            raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
