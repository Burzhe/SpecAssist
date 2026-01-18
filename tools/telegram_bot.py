from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import ADMIN_IDS, BOT_TOKEN, UPLOAD_DIR
from tools.db import get_connection, get_meta, list_versions, set_meta
from tools.importer import import_workbook
from tools.search import ParsedQuery, search_items, search_items_with_params


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

MAX_RESULTS_DEFAULT = 10
TOO_MANY_THRESHOLD = 60
SHOW_ALL_LIMIT = 200
PAGE_SIZE = 10

LOGGER = logging.getLogger(__name__)


@dataclass
class SearchState:
    query: str
    parsed: ParsedQuery
    keywords: list[str]
    flags: dict[str, bool | None]
    tol: int
    price_min: float | None = None
    price_max: float | None = None
    offset: int = 0
    limit: int = PAGE_SIZE
    total: int = 0
    relaxed: list[str] | None = None


SEARCH_STATE: dict[int, SearchState] = {}


def is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in ADMIN_IDS


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await send_split_message(
        context,
        update.effective_chat.id,
        "SpecAssist bot.\n"
        "Use /s <query> to search.\n"
        "Admins: /upload, /versions, /use <version>, /reindex.",
    )


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    query = " ".join(context.args).strip()
    if not query:
        await send_split_message(context, update.effective_chat.id, "Usage: /s <query>")
        return
    conn = get_connection()
    result = search_items(conn, query)
    conn.close()
    parsed = result["parsed"]
    state = SearchState(
        query=query,
        parsed=parsed,
        keywords=result["keywords"],
        flags=result["flags"],
        tol=result["tol"],
        offset=0,
        limit=PAGE_SIZE,
        total=result["total"],
        relaxed=result.get("relaxed", []),
    )
    SEARCH_STATE[update.effective_chat.id] = state
    await _render_search_results(
        context,
        update.effective_chat.id,
        result,
        state,
    )


async def upload_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Not allowed.")
        return
    UPLOAD_STATE.mark_pending(update.effective_user.id)
    await send_split_message(context, update.effective_chat.id, "Send Excel file as a document.")


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.document is None:
        return
    user_id = update.effective_user.id if update.effective_user else None
    if not is_admin(user_id) or user_id is None:
        await send_split_message(context, update.effective_chat.id, "Not allowed.")
        return
    if not UPLOAD_STATE.is_pending(user_id):
        return
    UPLOAD_STATE.clear_pending(user_id)

    doc = update.message.document
    await send_split_message(context, update.effective_chat.id, "Download ok. Processing...")
    file = await context.bot.get_file(doc.file_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    file_name = doc.file_name or "upload.xlsx"
    if not file_name.endswith(".xlsx"):
        file_name = f"{file_name}.xlsx"
    dest = UPLOAD_DIR / f"{timestamp}_{file_name}"
    await file.download_to_drive(custom_path=str(dest))
    await send_split_message(context, update.effective_chat.id, "Scanning sheets...")

    result = await asyncio.to_thread(_import_file, dest)
    conn = get_connection()
    versions = list_versions(conn)
    conn.close()
    await send_split_message(
        context,
        update.effective_chat.id,
        "Import report:\n"
        "- version: {version}\n"
        "- versions: {versions}\n"
        "- detected sheets: {detected_sheets}\n"
        "- skipped sheets: {skipped_sheets}\n"
        "- rows inserted: {inserted}\n"
        "- errors: {errors}\n"
        "Active version set to {version}".format(
            inserted=result["inserted"],
            detected_sheets=result["detected_sheets"],
            skipped_sheets=result["skipped_sheets"],
            version=result["source_version"],
            versions=", ".join(versions) if versions else "-",
            errors=0,
        ),
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
        await send_split_message(context, update.effective_chat.id, "Not allowed.")
        return
    conn = get_connection()
    versions = list_versions(conn)
    active = get_meta(conn, "active_version")
    conn.close()
    if not versions:
        await send_split_message(context, update.effective_chat.id, "No versions imported yet.")
        return
    lines = ["Versions:"]
    for version in versions:
        marker = " (active)" if version == active else ""
        lines.append(f"- {version}{marker}")
    await send_split_message(context, update.effective_chat.id, "\n".join(lines))


async def use_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Not allowed.")
        return
    if not context.args:
        await send_split_message(context, update.effective_chat.id, "Usage: /use <version>")
        return
    version = context.args[0]
    conn = get_connection()
    set_meta(conn, "active_version", version)
    conn.close()
    await send_split_message(context, update.effective_chat.id, f"Active version set to {version}")


async def reindex_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Not allowed.")
        return
    conn = get_connection()
    active = get_meta(conn, "active_version")
    if not active:
        conn.close()
        await send_split_message(context, update.effective_chat.id, "No active version set.")
        return
    path = get_meta(conn, f"version_path:{active}")
    conn.close()
    if not path:
        await send_split_message(context, update.effective_chat.id, "Path for active version not found.")
        return
    await send_split_message(context, update.effective_chat.id, "Reindexing current active file...")
    result = await asyncio.to_thread(_import_file, Path(path))
    await send_split_message(
        context,
        update.effective_chat.id,
        f"Reindexed {result['inserted']} rows. Active version {result['source_version']}",
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


def split_message(text: str, max_len: int = 3800) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        line_len = len(line)
        if line_len > max_len:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            start = 0
            while start < line_len:
                chunks.append(line[start : start + max_len])
                start += max_len
            continue
        pending = line_len + (1 if current else 0)
        if current_len + pending > max_len:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            if current:
                current.append(line)
            else:
                current = [line]
            current_len += pending
    if current:
        chunks.append("\n".join(current))
    return chunks


async def send_split_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    parts = split_message(text)
    for idx, part in enumerate(parts):
        markup = reply_markup if idx == len(parts) - 1 else None
        await context.bot.send_message(chat_id=chat_id, text=part, reply_markup=markup)


def _format_price(item: dict) -> str:
    unit = item.get("price_unit_ex_vat")
    total = item.get("price_total_ex_vat")
    qty = item.get("qty")
    parts = []
    if unit is not None:
        parts.append(f"unit ex VAT: {unit:.2f}")
    if total is not None:
        parts.append(f"total ex VAT: {total:.2f}")
    if qty is not None:
        parts.append(f"qty: {qty:.2f}")
    return " | ".join(parts) if parts else "-"


def _format_description(description: str | None, limit: int = 200) -> str | None:
    if not description:
        return None
    cleaned = description.strip().replace("\r\n", "\n")
    if not cleaned:
        return None
    lines = cleaned.split("\n")
    snippet = "\n".join(line.strip() for line in lines[:2] if line.strip())
    if len(snippet) > limit:
        return snippet[: limit - 1].rstrip() + "…"
    return snippet


def _format_item(item: dict, item_type: str | None) -> list[str]:
    name = item.get("name") or "-"
    item_type = item_type or "item"
    lines = [f"{name} ({item_type})"]
    dims = _format_dims(item.get("w_mm"), item.get("d_mm"), item.get("h_mm"))
    if dims != "-":
        lines.append(f"dims: {dims}")
    price = _format_price(item)
    if price != "-":
        lines.append(f"price: {price}")
    flags = _format_flags(item)
    if flags:
        lines.append(f"flags: {flags}")
    description = _format_description(item.get("description"))
    if description:
        lines.append(description)
    lines.append(
        "source: sheet={sheet} row={row} version={version}".format(
            sheet=item.get("source_sheet") or "-",
            row=item.get("source_row") or "-",
            version=item.get("source_version") or "-",
        )
    )
    return lines


def _format_query_summary(state: SearchState) -> str:
    dims = _format_dims(*state.parsed.dims)
    flags = _format_flags_from_filters(state.flags)
    keywords = ", ".join(state.keywords) if state.keywords else "-"
    parts = [
        f"category: {state.parsed.category or '-'}",
        f"dims: {dims}",
        f"flags: {flags or '-'}",
        f"keywords: {keywords}",
        f"tol: {state.tol}",
    ]
    if state.relaxed:
        parts.append(f"relaxed: {_format_relaxed_steps(state.relaxed)}")
    if state.price_min is not None or state.price_max is not None:
        parts.append(f"price: {_format_price_range(state.price_min, state.price_max)}")
    return "\n".join(parts)


def _format_flags_from_filters(flags: dict[str, bool | None]) -> str:
    labels = []
    for key, label in (
        ("has_led", "LED"),
        ("mat_mdf", "МДФ"),
        ("mat_ldsp", "ЛДСП"),
        ("mat_veneer", "ШПОН"),
        ("has_glass", "СТЕКЛО"),
        ("has_metal", "МЕТАЛЛ"),
    ):
        state = flags.get(key)
        if state is True:
            labels.append(label)
        elif state is False:
            labels.append(f"{label}✕")
    return "/".join(labels)


def _format_price_range(price_min: float | None, price_max: float | None) -> str:
    if price_min is not None and price_max is not None:
        return f"{int(price_min)}-{int(price_max)}"
    if price_min is not None:
        return f"{int(price_min)}+"
    if price_max is not None:
        return f"<= {int(price_max)}"
    return "-"


def _format_relaxed_steps(steps: list[str]) -> str:
    label_map = {
        "drop:has_led": "LED ignored",
        "drop:mat_ldsp": "ЛДСП ignored",
        "drop:mat_mdf": "МДФ ignored",
        "drop:mat_veneer": "ШПОН ignored",
        "drop:has_glass": "СТЕКЛО ignored",
        "drop:has_metal": "МЕТАЛЛ ignored",
    }
    formatted = []
    for step in steps:
        if step in label_map:
            formatted.append(label_map[step])
        elif step.startswith("tol="):
            formatted.append(step.replace("tol=", "tol="))
        elif step == "keywords:shortened":
            formatted.append("keywords shortened")
        elif step == "fallback:text-only":
            formatted.append("text-only fallback")
        else:
            formatted.append(step)
    return ", ".join(formatted)


def _next_flag_state(current: bool | None) -> bool | None:
    if current is None:
        return True
    if current is True:
        return False
    return None


def _flag_button_label(label: str, state: bool | None) -> str:
    suffix = "any"
    if state is True:
        suffix = "yes"
    elif state is False:
        suffix = "no"
    return f"{label}: {suffix}"


def _build_overflow_keyboard(
    state: SearchState,
    enable_show_all: bool,
    available_flags: dict[str, bool],
) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("➕ Add filters", callback_data="s:filters")],
        [InlineKeyboardButton("📄 Show more (10)", callback_data="s:more")],
    ]
    if enable_show_all:
        rows.append([InlineKeyboardButton("📄 Show all (careful)", callback_data="s:all")])
    if state.parsed.dims != (None, None, None):
        rows.append(
            [
                InlineKeyboardButton("↔️ Increase tol", callback_data="s:tol_up"),
                InlineKeyboardButton("↔️ Decrease tol", callback_data="s:tol_down"),
            ]
        )
    flag_row = []
    for key, label in (
        ("has_led", "LED"),
        ("mat_mdf", "МДФ"),
        ("mat_ldsp", "ЛДСП"),
        ("mat_veneer", "ШПОН"),
        ("has_glass", "СТЕКЛО"),
        ("has_metal", "МЕТАЛЛ"),
    ):
        if not available_flags.get(key):
            continue
        flag_row.append(
            InlineKeyboardButton(
                _flag_button_label(label, state.flags.get(key)),
                callback_data=f"s:toggle:{key}",
            )
        )
        if len(flag_row) == 3:
            rows.append(flag_row)
            flag_row = []
    if flag_row:
        rows.append(flag_row)
    rows.append([InlineKeyboardButton("❌ Clear filters", callback_data="s:clear")])
    return InlineKeyboardMarkup(rows)


def _build_refine_keyboard(state: SearchState) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                _flag_button_label("LED", state.flags.get("has_led")),
                callback_data="s:toggle:has_led",
            )
        ],
        [
            InlineKeyboardButton(
                _flag_button_label("МДФ", state.flags.get("mat_mdf")),
                callback_data="s:toggle:mat_mdf",
            ),
            InlineKeyboardButton(
                _flag_button_label("ЛДСП", state.flags.get("mat_ldsp")),
                callback_data="s:toggle:mat_ldsp",
            ),
            InlineKeyboardButton(
                _flag_button_label("ШПОН", state.flags.get("mat_veneer")),
                callback_data="s:toggle:mat_veneer",
            ),
        ],
        [
            InlineKeyboardButton(
                _flag_button_label("СТЕКЛО", state.flags.get("has_glass")),
                callback_data="s:toggle:has_glass",
            ),
            InlineKeyboardButton(
                _flag_button_label("МЕТАЛЛ", state.flags.get("has_metal")),
                callback_data="s:toggle:has_metal",
            ),
        ],
        [
            InlineKeyboardButton("<=100k", callback_data="s:price:max:100000"),
            InlineKeyboardButton("100-300k", callback_data="s:price:range:100000:300000"),
            InlineKeyboardButton("300-700k", callback_data="s:price:range:300000:700000"),
            InlineKeyboardButton("700k+", callback_data="s:price:min:700000"),
        ],
        [InlineKeyboardButton("Price: any", callback_data="s:price:clear")],
    ]
    if state.parsed.dims != (None, None, None):
        rows.append(
            [
                InlineKeyboardButton("tol 20", callback_data="s:tol:20"),
                InlineKeyboardButton("tol 50", callback_data="s:tol:50"),
                InlineKeyboardButton("tol 100", callback_data="s:tol:100"),
                InlineKeyboardButton("tol 200", callback_data="s:tol:200"),
            ]
        )
    rows.append(
        [
            InlineKeyboardButton("Apply", callback_data="s:apply"),
            InlineKeyboardButton("Cancel", callback_data="s:cancel"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _build_no_results_keyboard(state: SearchState) -> InlineKeyboardMarkup:
    rows = []
    if state.parsed.dims != (None, None, None):
        rows.append([InlineKeyboardButton("Increase tol", callback_data="s:tol_up")])
    rows.append(
        [
            InlineKeyboardButton("Clear flags", callback_data="s:clear_flags"),
            InlineKeyboardButton("Search text only", callback_data="s:text_only"),
        ]
    )
    return InlineKeyboardMarkup(rows)


async def _render_search_results(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    result: dict,
    state: SearchState,
    *,
    show_page: bool = False,
) -> None:
    total = result["total"]
    lines: list[str] = []
    if total == 0:
        lines.append("No exact match.")
        lines.append("Try removing keywords or using a bigger tolerance.")
        await send_split_message(
            context,
            chat_id,
            "\n".join(lines),
            reply_markup=_build_no_results_keyboard(state),
        )
        return

    header = f"Found ~{total} matches."
    if state.relaxed:
        header = "Exact match not found. Showing closest ({details}).".format(
            details=_format_relaxed_steps(state.relaxed)
        )
    if total > TOO_MANY_THRESHOLD:
        header = f"{header} Too many to display."
    lines.append(header)
    lines.append(_format_query_summary(state))
    lines.append("")

    if total <= MAX_RESULTS_DEFAULT or show_page:
        if total <= MAX_RESULTS_DEFAULT:
            lines.append("Results:")
        else:
            start = state.offset + 1
            end = min(state.offset + state.limit, total)
            lines.append(f"Results {start}-{end}:")
        for item in result["results"]:
            lines.extend(_format_item(item, state.parsed.category))
            lines.append("")
        reply_markup = None
        if total > MAX_RESULTS_DEFAULT:
            available_flags = {
                key: any(item.get(key) for item in result["results"])
                for key in state.flags.keys()
            }
            reply_markup = _build_overflow_keyboard(
                state,
                enable_show_all=total <= SHOW_ALL_LIMIT,
                available_flags=available_flags,
            )
        await send_split_message(context, chat_id, "\n".join(lines).strip(), reply_markup=reply_markup)
        return

    lines.append("Top matches preview:")
    for item in result["results"][:5]:
        lines.extend(_format_item(item, state.parsed.category))
        lines.append("")

    available_flags = {
        key: any(item.get(key) for item in result["results"])
        for key in state.flags.keys()
    }
    keyboard = _build_overflow_keyboard(
        state,
        enable_show_all=total <= SHOW_ALL_LIMIT,
        available_flags=available_flags,
    )
    await send_split_message(context, chat_id, "\n".join(lines).strip(), reply_markup=keyboard)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query is None:
        return
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id if query.message else None
    if chat_id is None:
        return
    state = SEARCH_STATE.get(chat_id)
    if state is None:
        await send_split_message(context, chat_id, "Search state expired. Please run /s again.")
        return

    action = query.data or ""
    in_refine_menu = bool(query.message and query.message.text and "Refine filters" in query.message.text)
    if action == "s:filters":
        await send_split_message(
            context,
            chat_id,
            "Refine filters:",
            reply_markup=_build_refine_keyboard(state),
        )
        return
    if action == "s:cancel":
        await send_split_message(context, chat_id, "Canceled.")
        return
    if action == "s:clear":
        state.flags = {key: None for key in state.flags}
        state.price_min = None
        state.price_max = None
        state.offset = 0
    if action.startswith("s:toggle:"):
        flag = action.split(":", 2)[2]
        state.flags[flag] = _next_flag_state(state.flags.get(flag))
        if in_refine_menu:
            await send_split_message(
                context,
                chat_id,
                "Refine filters:",
                reply_markup=_build_refine_keyboard(state),
            )
            return
    if action.startswith("s:price:"):
        parts = action.split(":")
        mode = parts[2]
        if mode == "clear":
            state.price_min = None
            state.price_max = None
        elif mode == "max":
            state.price_min = None
            state.price_max = float(parts[3])
        elif mode == "min":
            state.price_min = float(parts[3])
            state.price_max = None
        elif mode == "range":
            state.price_min = float(parts[3])
            state.price_max = float(parts[4])
        if in_refine_menu:
            await send_split_message(
                context,
                chat_id,
                "Refine filters:",
                reply_markup=_build_refine_keyboard(state),
            )
            return
    if action.startswith("s:tol:"):
        value = int(action.split(":")[2])
        state.tol = value
        if in_refine_menu:
            await send_split_message(
                context,
                chat_id,
                "Refine filters:",
                reply_markup=_build_refine_keyboard(state),
            )
            return
    if action == "s:tol_up":
        state.tol = min(state.tol + 50, 500)
    if action == "s:tol_down":
        state.tol = max(state.tol - 50, 10)
    if action == "s:clear_flags":
        state.flags = {key: None for key in state.flags}
    if action == "s:text_only":
        state.flags = {key: None for key in state.flags}
        state.parsed = ParsedQuery(None, (None, None, None), state.flags, state.keywords)
    if action == "s:more":
        if state.offset + state.limit >= state.total:
            await send_split_message(context, chat_id, "No more results.")
            return
        state.offset += state.limit
    if action == "s:all":
        if state.total > SHOW_ALL_LIMIT:
            await send_split_message(
                context,
                chat_id,
                "Too many results to show all. Please refine.",
            )
            return
        state.offset = 0
        state.limit = SHOW_ALL_LIMIT
    if action == "s:apply":
        state.offset = 0
        state.limit = PAGE_SIZE

    show_page = action in {"s:more", "s:all"}
    conn = get_connection()
    result = search_items_with_params(
        conn,
        state.query,
        parsed=state.parsed,
        keywords=state.keywords,
        flags=state.flags,
        tol=state.tol,
        price_min=state.price_min,
        price_max=state.price_max,
        limit=state.limit,
        offset=state.offset,
    )
    conn.close()
    state.total = result["total"]
    state.parsed = result["parsed"]
    state.flags = result["flags"]
    state.keywords = result["keywords"]
    state.relaxed = []
    if state.offset >= state.total:
        state.offset = 0

    await _render_search_results(
        context,
        chat_id,
        result,
        state,
        show_page=show_page,
    )


def build_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN not set")
    if not ADMIN_IDS:
        LOGGER.warning("ADMIN_IDS is empty. Uploads will be disabled.")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("s", search_handler))
    app.add_handler(CommandHandler("upload", upload_handler))
    app.add_handler(CommandHandler("versions", versions_handler))
    app.add_handler(CommandHandler("use", use_handler))
    app.add_handler(CommandHandler("reindex", reindex_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    return app


def run_bot() -> None:
    app = build_app()
    app.run_polling()
