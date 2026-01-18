from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import ADMIN_IDS, BOT_TOKEN, UPLOAD_DIR
from tools.db import get_connection, get_meta, list_versions, set_meta
from tools.importer import import_workbook
from tools.search import search_items


class UploadState:
    def __init__(self) -> None:
        self.pending: set[int] = set()

    def mark_pending(self, user_id: int) -> None:
        self.pending.add(user_id)

    def clear_pending(self, user_id: int) -> None:
        self.pending.discard(user_id)

    def is_pending(self, user_id: int) -> bool:
        return user_id in self.pending


UPLOAD_STATE = UploadState()


def is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in ADMIN_IDS


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "SpecAssist bot.\n"
        "Use /s <query> to search.\n"
        "Admins: /upload, /versions, /use <version>, /reindex."
    )


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Usage: /s <query>")
        return
    conn = get_connection()
    result = search_items(conn, query)
    conn.close()
    lines = [f"Found {len(result['results'])} (tol={result['tol']}, relaxed: {', '.join(result['relaxed']) or 'no'})"]
    for idx, item in enumerate(result["results"], start=1):
        dims = _format_dims(item.get("w_mm"), item.get("d_mm"), item.get("h_mm"))
        price = item.get("price_unit_ex_vat")
        price_text = f"{price:.2f}" if price else "-"
        flags = _format_flags(item)
        lines.append(f"{idx}) {item.get('name') or '-'}")
        lines.append(f"dims: {dims} | price ex VAT: {price_text}")
        if flags:
            lines.append(f"flags: {flags}")
        lines.append(
            f"source: sheet={item.get('source_sheet')} row={item.get('source_row')} | version={item.get('source_version')}"
        )
    await update.message.reply_text("\n".join(lines))


async def upload_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await update.message.reply_text("Not authorized.")
        return
    UPLOAD_STATE.mark_pending(update.effective_user.id)
    await update.message.reply_text("Send Excel file as document to reindex.")


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.document is None:
        return
    user_id = update.effective_user.id if update.effective_user else None
    if not is_admin(user_id) or user_id is None:
        await update.message.reply_text("Not authorized.")
        return
    if not UPLOAD_STATE.is_pending(user_id):
        return
    UPLOAD_STATE.clear_pending(user_id)

    doc = update.message.document
    await update.message.reply_text("Download ok. Processing...")
    file = await context.bot.get_file(doc.file_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    dest = UPLOAD_DIR / f"{timestamp}_{doc.file_name}"
    await file.download_to_drive(custom_path=str(dest))
    await update.message.reply_text("Scanning sheets...")

    result = await asyncio.to_thread(_import_file, dest)
    await update.message.reply_text(
        "Imported {inserted} rows, detected sheets {detected_sheets}, skipped {skipped_sheets}.\n"
        "Active version set to {version}".format(
            inserted=result["inserted"],
            detected_sheets=result["detected_sheets"],
            skipped_sheets=result["skipped_sheets"],
            version=result["source_version"],
        )
    )


def _import_file(path: Path) -> dict:
    conn = get_connection()
    result = import_workbook(path, conn)
    conn.close()
    return result


async def versions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await update.message.reply_text("Not authorized.")
        return
    conn = get_connection()
    versions = list_versions(conn)
    active = get_meta(conn, "active_version")
    conn.close()
    if not versions:
        await update.message.reply_text("No versions imported yet.")
        return
    lines = ["Versions:"]
    for version in versions:
        marker = " (active)" if version == active else ""
        lines.append(f"- {version}{marker}")
    await update.message.reply_text("\n".join(lines))


async def use_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await update.message.reply_text("Not authorized.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /use <version>")
        return
    version = context.args[0]
    conn = get_connection()
    set_meta(conn, "active_version", version)
    conn.close()
    await update.message.reply_text(f"Active version set to {version}")


async def reindex_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await update.message.reply_text("Not authorized.")
        return
    conn = get_connection()
    active = get_meta(conn, "active_version")
    if not active:
        conn.close()
        await update.message.reply_text("No active version set.")
        return
    path = get_meta(conn, f"version_path:{active}")
    conn.close()
    if not path:
        await update.message.reply_text("Path for active version not found.")
        return
    await update.message.reply_text("Reindexing current active file...")
    result = await asyncio.to_thread(_import_file, Path(path))
    await update.message.reply_text(
        f"Reindexed {result['inserted']} rows. Active version {result['source_version']}"
    )


def _format_dims(w_mm: int | None, d_mm: int | None, h_mm: int | None) -> str:
    dims = [val for val in (w_mm, d_mm, h_mm) if val is not None]
    return "x".join(str(val) for val in dims) if dims else "-"


def _format_flags(item: dict) -> str:
    flags = []
    if item.get("has_led"):
        flags.append("LED")
    if item.get("mat_ldsp"):
        flags.append("ЛДСП")
    if item.get("mat_mdf"):
        flags.append("МДФ")
    if item.get("mat_veneer"):
        flags.append("ШПОН")
    if item.get("has_glass"):
        flags.append("СТЕКЛО")
    if item.get("has_metal"):
        flags.append("МЕТАЛЛ")
    return "/".join(flags)


def build_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not configured.")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("s", search_handler))
    app.add_handler(CommandHandler("upload", upload_handler))
    app.add_handler(CommandHandler("versions", versions_handler))
    app.add_handler(CommandHandler("use", use_handler))
    app.add_handler(CommandHandler("reindex", reindex_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    return app


def run_bot() -> None:
    app = build_app()
    app.run_polling()
