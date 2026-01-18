from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import Forbidden
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import ADMIN_IDS, BOT_TOKEN, EXCEL_PATH
from tools.db import (
    add_allowed_user,
    get_latest_import_summary,
    get_connection,
    is_user_allowed,
    list_allowed_users,
    remove_allowed_user,
)
from tools.import_excel import reindex_with_report
from tools.importer import debug_workbook_mapping
from tools.search import ParsedQuery, find_similar, search_items, search_items_with_params

MAX_RESULTS_DEFAULT = 10
TOO_MANY_THRESHOLD = 60
SHOW_ALL_LIMIT = 200
PAGE_SIZE = 10
VAT_RATE = 0.22

LOGGER = logging.getLogger(__name__)


@dataclass
class SearchState:
    query: str
    parsed: ParsedQuery
    keywords: list[str]
    flags: dict[str, bool | None]
    tol_by_dim: tuple[int | None, int | None, int | None]
    items: list[dict]
    price_min: float | None = None
    price_max: float | None = None
    offset: int = 0
    limit: int = PAGE_SIZE
    total: int = 0


SEARCH_STATE: dict[int, SearchState] = {}


def is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in ADMIN_IDS


def is_authorized(user_id: int | None) -> bool:
    if is_admin(user_id):
        return True
    if user_id is None:
        return False
    conn = get_connection()
    allowed = is_user_allowed(conn, user_id)
    conn.close()
    return allowed


def _get_admin_summary_message() -> str | None:
    conn = get_connection()
    summary = get_latest_import_summary(conn)
    conn.close()
    if not summary:
        return None
    return _build_admin_summary_message(
        sheets_detected=int(summary["sheets_detected"] or 0),
        rows_scanned=int(summary["rows_scanned"] or 0),
        rows_inserted=int(summary["rows_inserted"] or 0),
        rows_skipped=int(summary["rows_skipped"] or 0),
        rows_unit_price_from_unit=int(summary["rows_unit_price_from_unit"] or 0),
        rows_unit_price_from_total_qty=int(summary["rows_unit_price_from_total_qty"] or 0),
    )


async def _send_startup_summary(app: Application) -> None:
    if not ADMIN_IDS:
        return
    message = _get_admin_summary_message()
    if not message:
        return
    for admin_id in ADMIN_IDS:
        try:
            await app.bot.send_message(chat_id=admin_id, text=message)
        except Forbidden:
            LOGGER.warning("Cannot send startup summary to admin %s: forbidden", admin_id)
        except Exception:
            LOGGER.exception("Failed to send startup summary to admin %s", admin_id)


def _build_admin_summary_message(
    *,
    sheets_detected: int,
    rows_scanned: int,
    rows_inserted: int,
    rows_skipped: int,
    rows_unit_price_from_unit: int,
    rows_unit_price_from_total_qty: int,
) -> str:
    message = (
        "Сводка импорта:\n"
        "Листов обнаружено: {sheets}\n"
        "Строк просмотрено: {scanned}\n"
        "Строк добавлено: {inserted}\n"
        "Строк пропущено: {skipped}\n"
        "Цена из колонки: {unit}\n"
        "Цена из итога/кол-ва: {total_qty}"
    ).format(
        sheets=sheets_detected,
        scanned=rows_scanned,
        inserted=rows_inserted,
        skipped=rows_skipped,
        unit=rows_unit_price_from_unit,
        total_qty=rows_unit_price_from_total_qty,
    )
    return message


def _build_admin_start_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("Поиск", callback_data="a:search")],
        [
            InlineKeyboardButton("Статус", callback_data="a:status"),
            InlineKeyboardButton("Очистить фильтры", callback_data="a:clear"),
        ],
    ]
    return InlineKeyboardMarkup(rows)


async def _send_admin_summary(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
) -> None:
    message = _get_admin_summary_message()
    if message:
        await send_split_message(context, chat_id, message)
    else:
        await send_split_message(context, chat_id, "Сводка недоступна. Запустите /reindex.")


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    user_id = update.effective_user.id if update.effective_user else None
    if not is_authorized(user_id):
        await send_split_message(
            context,
            update.effective_chat.id,
            "Нет доступа. Обратитесь к администратору.\n"
            f"Ваш user_id: {user_id}",
        )
        return
    is_admin_user = is_admin(user_id)
    message = (
        "SpecAssist — поиск по базе Excel.\n"
        "Просто отправьте текстовый запрос.\n"
        "Для подсказок: /help."
    )
    if is_admin_user:
        message = (
            "SpecAssist — поиск по базе Excel.\n"
            "Просто отправьте текстовый запрос.\n"
            "Для подсказок: /help.\n\n"
            "Команды администратора: /status, /users, /allow, /deny, /reindex, /debug_mapping."
        )
    await send_split_message(
        context,
        update.effective_chat.id,
        message,
        reply_markup=_build_admin_start_keyboard() if is_admin_user else None,
    )
    if is_admin_user:
        await _send_admin_summary(context, update.effective_chat.id)


async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Нет доступа.")
        return
    await _send_admin_summary(context, update.effective_chat.id)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    user_id = update.effective_user.id if update.effective_user else None
    if not is_authorized(user_id):
        await send_split_message(
            context,
            update.effective_chat.id,
            "Нет доступа. Обратитесь к администратору.",
        )
        return
    await send_split_message(
        context,
        update.effective_chat.id,
        "Примеры запросов:\n"
        "- шкаф лдсп с подсветкой\n"
        "- шкаф h2700 d400\n"
        "- бенч стол металл\n"
        "- перила нерж\n"
        "\n"
        "Можно указывать размеры частично: h2700, d400, w3000.\n"
        "Можно указывать габариты: 3000x400x2800.",
    )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or not update.message.text:
        return
    user_id = update.effective_user.id if update.effective_user else None
    if not is_authorized(user_id):
        await send_split_message(
            context,
            update.effective_chat.id,
            "Нет доступа. Обратитесь к администратору.",
        )
        return
    query = update.message.text.strip()
    if not query:
        return
    await _handle_search(context, update.effective_chat.id, query)


async def _handle_search(context: ContextTypes.DEFAULT_TYPE, chat_id: int, query: str) -> None:
    conn = get_connection()
    result = search_items(conn, query)
    conn.close()
    parsed = result["parsed"]
    state = SearchState(
        query=query,
        parsed=parsed,
        keywords=result["keywords"],
        flags=result["flags"],
        tol_by_dim=result["tol"],
        items=result["results"],
        offset=0,
        limit=PAGE_SIZE,
        total=result["total"],
    )
    SEARCH_STATE[chat_id] = state
    await _render_search_results(
        context,
        chat_id,
        result,
        state,
    )


async def reindex_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Нет доступа.")
        return
    if not EXCEL_PATH:
        await send_split_message(context, update.effective_chat.id, "Путь к Excel не настроен.")
        return
    path = Path(EXCEL_PATH)
    if not path.exists():
        await send_split_message(context, update.effective_chat.id, "Файл Excel не найден.")
        return
    await send_split_message(context, update.effective_chat.id, "Переиндексация Excel...")
    result = await asyncio.to_thread(reindex_with_report, str(path))
    summary = result.get("summary") or {}
    message = _build_admin_summary_message(
        sheets_detected=int(result.get("detected_sheets", 0)),
        rows_scanned=int(summary.get("rows_total", 0)),
        rows_inserted=int(summary.get("rows_inserted", 0)),
        rows_skipped=int(summary.get("rows_skipped", 0)),
        rows_unit_price_from_unit=int(summary.get("rows_unit_price_from_unit", 0)),
        rows_unit_price_from_total_qty=int(summary.get("rows_unit_price_from_total_qty", 0)),
    )
    try:
        await send_split_message(context, update.effective_chat.id, message)
    except Forbidden:
        LOGGER.warning("Cannot send reindex summary to admin %s: forbidden", update.effective_chat.id)
    except Exception:
        LOGGER.exception("Failed to send reindex summary to admin %s", update.effective_chat.id)


async def debug_mapping_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Нет доступа.")
        return
    if not EXCEL_PATH:
        await send_split_message(context, update.effective_chat.id, "Путь к Excel не настроен.")
        return
    path = Path(EXCEL_PATH)
    if not path.exists():
        await send_split_message(context, update.effective_chat.id, "Файл Excel не найден.")
        return
    report = debug_workbook_mapping(path, max_rows_scan=50)
    sheet_reports = report.get("sheet_reports", [])
    for sheet in sheet_reports:
        LOGGER.info("Debug mapping sheet: %s", sheet.get("sheet_name"))
        LOGGER.info("  header_row: %s", sheet.get("header_row"))
        LOGGER.info("  rows_total: %s", sheet.get("rows_total"))
        LOGGER.info("  rows_inserted: %s", sheet.get("rows_inserted"))
        LOGGER.info("  rows_skipped: %s", sheet.get("rows_skipped"))
        LOGGER.info("  mapping: %s", sheet.get("mapping"))
        LOGGER.info("  unused_headers: %s", sheet.get("unused_headers"))
        LOGGER.info("  missing_critical_fields: %s", sheet.get("missing_critical_fields"))
    await send_split_message(
        context,
        update.effective_chat.id,
        "Отчет сформирован. Смотрите логи сервера.",
    )


async def users_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Нет доступа.")
        return
    conn = get_connection()
    rows = list_allowed_users(conn)
    conn.close()
    if not rows:
        await send_split_message(context, update.effective_chat.id, "Список допущенных пользователей пуст.")
        return
    lines = ["Допущенные пользователи:"]
    for row in rows:
        display = str(row["user_id"])
        if row["username"]:
            display += f" (@{row['username']})"
        name_parts = " ".join(filter(None, [row["first_name"], row["last_name"]]))
        if name_parts:
            display += f" — {name_parts}"
        lines.append(display)
    await send_split_message(context, update.effective_chat.id, "\n".join(lines))


async def allow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    admin_id = update.effective_user.id if update.effective_user else None
    if not is_admin(admin_id):
        await send_split_message(context, update.effective_chat.id, "Нет доступа.")
        return
    if not context.args:
        await send_split_message(context, update.effective_chat.id, "Использование: /allow <user_id>")
        return
    try:
        user_id = int(context.args[0])
    except ValueError:
        await send_split_message(context, update.effective_chat.id, "user_id должен быть числом.")
        return
    conn = get_connection()
    add_allowed_user(
        conn,
        user_id=user_id,
        username=None,
        first_name=None,
        last_name=None,
        added_by=admin_id,
        added_at=datetime.utcnow().isoformat(),
    )
    conn.close()
    await send_split_message(context, update.effective_chat.id, f"Доступ выдан: {user_id}")


async def deny_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Нет доступа.")
        return
    if not context.args:
        await send_split_message(context, update.effective_chat.id, "Использование: /deny <user_id>")
        return
    try:
        user_id = int(context.args[0])
    except ValueError:
        await send_split_message(context, update.effective_chat.id, "user_id должен быть числом.")
        return
    conn = get_connection()
    remove_allowed_user(conn, user_id)
    conn.close()
    await send_split_message(context, update.effective_chat.id, f"Доступ отозван: {user_id}")


def _format_dims(w_mm: int | None, d_mm: int | None, h_mm: int | None) -> str | None:
    if w_mm is not None and d_mm is not None and h_mm is not None:
        return f"{w_mm}×{d_mm}×{h_mm} мм"
    parts = []
    if w_mm is not None:
        parts.append(f"Ш: {w_mm} мм")
    if d_mm is not None:
        parts.append(f"Г: {d_mm} мм")
    if h_mm is not None:
        parts.append(f"В: {h_mm} мм")
    return ", ".join(parts) if parts else None


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
    *,
    parse_mode: str | None = None,
) -> None:
    parts = split_message(text)
    for idx, part in enumerate(parts):
        markup = reply_markup if idx == len(parts) - 1 else None
        await context.bot.send_message(
            chat_id=chat_id,
            text=part,
            reply_markup=markup,
            parse_mode=parse_mode,
        )


def _format_currency(value: float) -> str:
    rounded = int(round(value))
    return f"{rounded:,}".replace(",", " ")


def _format_price_lines(item: dict) -> list[str]:
    unit = item.get("price_unit_ex_vat")
    total = item.get("price_total_ex_vat")
    qty = item.get("qty")
    lines: list[str] = []
    if unit is not None:
        lines.append(f"Цена за 1 шт (без НДС): {_format_currency(unit)} ₽")
        lines.append(f"Цена за 1 шт (с НДС 22%): {_format_currency(unit * (1 + VAT_RATE))} ₽")
    if qty is not None:
        qty_display = int(qty) if isinstance(qty, float) and qty.is_integer() else qty
        lines.append(f"Кол-во: {qty_display}")
    if total is not None:
        lines.append(f"Итого (без НДС): {_format_currency(total)} ₽")
        lines.append(f"Итого (с НДС 22%): {_format_currency(total * (1 + VAT_RATE))} ₽")
    return lines


def _format_description(description: str | None, limit: int = 200) -> tuple[str | None, bool]:
    if not description:
        return (None, False)
    cleaned = description.strip().replace("\r\n", "\n")
    if not cleaned:
        return (None, False)
    lines = cleaned.split("\n")
    snippet = "\n".join(line.strip() for line in lines[:2] if line.strip())
    if len(snippet) > limit:
        return (snippet[: limit - 1].rstrip() + "…", True)
    return (snippet, False)


def _format_item(item: dict, item_type: str | None, index: int) -> tuple[list[str], bool]:
    name = item.get("name") or "-"
    lines = [f"{index}. <b>{_escape_html(name)}</b>"]
    dims = _format_dims(item.get("w_mm"), item.get("d_mm"), item.get("h_mm"))
    if dims:
        lines.append(f"Габариты: {dims}")
    lines.extend(_escape_lines(_format_price_lines(item)))
    description, truncated = _format_description(item.get("description"))
    if description:
        lines.append(f"Описание: {_escape_html(description)}")
    sheet = item.get("source_sheet") or "-"
    row = item.get("source_row") or "-"
    lines.append(f'Excel: лист "{_escape_html(str(sheet))}", строка {row}')
    return lines, truncated or bool(description)


def _format_price_range(price_min: float | None, price_max: float | None) -> str:
    if price_min is not None and price_max is not None:
        return f"{int(price_min)}-{int(price_max)}"
    if price_min is not None:
        return f"{int(price_min)}+"
    if price_max is not None:
        return f"<= {int(price_max)}"
    return "-"


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_lines(lines: list[str]) -> list[str]:
    return [_escape_html(line) for line in lines]


def _next_flag_state(current: bool | None) -> bool | None:
    if current is None:
        return True
    if current is True:
        return False
    return None


def _flag_button_label(label: str, state: bool | None) -> str:
    suffix = "люб."
    if state is True:
        suffix = "да"
    elif state is False:
        suffix = "нет"
    return f"{label}: {suffix}"


def _build_overflow_keyboard(
    state: SearchState,
    available_flags: dict[str, bool],
    *,
    include_flags: bool = True,
) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("Уточнить фильтры", callback_data="s:filters")],
        [InlineKeyboardButton("Ещё 10", callback_data="s:more")],
    ]
    if include_flags:
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
    rows.append([InlineKeyboardButton("Сброс", callback_data="s:clear")])
    return InlineKeyboardMarkup(rows)


def _build_results_keyboard(
    state: SearchState,
    action_items: list[tuple[int, bool]],
    *,
    available_flags: dict[str, bool] | None = None,
    include_overflow: bool = False,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for idx, (item_id, has_desc) in enumerate(action_items, start=1):
        row = [InlineKeyboardButton(f"Похожие #{idx}", callback_data=f"s:similar:{item_id}")]
        if has_desc:
            row.append(
                InlineKeyboardButton(
                    f"Показать полностью #{idx}", callback_data=f"s:desc:{item_id}"
                )
            )
        rows.append(row)
    if include_overflow and available_flags is not None:
        overflow = _build_overflow_keyboard(
            state,
            available_flags,
            include_flags=state.total <= TOO_MANY_THRESHOLD,
        )
        rows.extend(overflow.inline_keyboard)
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
        [InlineKeyboardButton("Цена: любая", callback_data="s:price:clear")],
    ]
    rows.extend(_build_dim_tolerance_rows(state))
    rows.append(
        [
            InlineKeyboardButton("Применить", callback_data="s:apply"),
            InlineKeyboardButton("Отмена", callback_data="s:cancel"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _build_dim_tolerance_rows(state: SearchState) -> list[list[InlineKeyboardButton]]:
    rows: list[list[InlineKeyboardButton]] = []
    w_mm, d_mm, h_mm = state.parsed.dims
    if h_mm is not None:
        rows.append(
            [
                InlineKeyboardButton("Допуск H: 50", callback_data="s:tol:h:50"),
                InlineKeyboardButton("Допуск H: 100", callback_data="s:tol:h:100"),
                InlineKeyboardButton("Допуск H: 150", callback_data="s:tol:h:150"),
                InlineKeyboardButton("Допуск H: 200", callback_data="s:tol:h:200"),
            ]
        )
    if d_mm is not None:
        rows.append(
            [
                InlineKeyboardButton("Допуск D: 10", callback_data="s:tol:d:10"),
                InlineKeyboardButton("Допуск D: 20", callback_data="s:tol:d:20"),
                InlineKeyboardButton("Допуск D: 50", callback_data="s:tol:d:50"),
            ]
        )
    if w_mm is not None:
        rows.append(
            [
                InlineKeyboardButton("Допуск W: 50", callback_data="s:tol:w:50"),
                InlineKeyboardButton("Допуск W: 100", callback_data="s:tol:w:100"),
                InlineKeyboardButton("Допуск W: 200", callback_data="s:tol:w:200"),
            ]
        )
    return rows


def _build_no_results_keyboard(state: SearchState) -> InlineKeyboardMarkup:
    rows = []
    if state.parsed.dims != (None, None, None):
        rows.append([InlineKeyboardButton("Добавить допуск", callback_data="s:filters")])
    rows.append(
        [
            InlineKeyboardButton("Сбросить материалы", callback_data="s:clear_flags"),
            InlineKeyboardButton("Искать только по тексту", callback_data="s:text_only"),
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
        lines.append("Ничего не найдено.")
        lines.append("Попробуйте убрать ключевые слова или увеличить допуск.")
        await send_split_message(
            context,
            chat_id,
            "\n".join(lines),
            reply_markup=_build_no_results_keyboard(state),
        )
        return

    header = f"Найдено ~{total} вариантов."
    if total > TOO_MANY_THRESHOLD:
        header = f"Найдено ~{total} вариантов. Слишком много, показываю лучшие."
    lines.append(header)
    if total > TOO_MANY_THRESHOLD:
        lines.append("Совет: добавьте ключевые слова или фильтры.")
    lines.append("")

    action_items: list[tuple[int, bool]] = []
    if total <= MAX_RESULTS_DEFAULT or show_page:
        if total <= MAX_RESULTS_DEFAULT:
            lines.append("Результаты:")
        else:
            start = state.offset + 1
            end = min(state.offset + state.limit, total)
            lines.append(f"Результаты {start}-{end}:")
        state.items = result["results"]
        for idx, item in enumerate(result["results"], start=1):
            item_lines, has_desc = _format_item(item, state.parsed.category, idx)
            lines.extend(item_lines)
            lines.append("")
            if item.get("id") is not None:
                action_items.append((int(item["id"]), has_desc))
        reply_markup = None
        available_flags = {
            key: any(item.get(key) for item in result["results"])
            for key in state.flags.keys()
        }
        if action_items:
            reply_markup = _build_results_keyboard(
                state,
                action_items,
                available_flags=available_flags,
                include_overflow=total > MAX_RESULTS_DEFAULT,
            )
        await send_split_message(
            context,
            chat_id,
            "\n".join(lines).strip(),
            reply_markup=reply_markup,
            parse_mode="HTML",
        )
        return

    lines.append("Лучшие совпадения:")
    preview_items = result["results"][:MAX_RESULTS_DEFAULT]
    state.items = preview_items
    for idx, item in enumerate(preview_items, start=1):
        item_lines, has_desc = _format_item(item, state.parsed.category, idx)
        lines.extend(item_lines)
        lines.append("")
        if item.get("id") is not None:
            action_items.append((int(item["id"]), has_desc))

    available_flags = {
        key: any(item.get(key) for item in preview_items)
        for key in state.flags.keys()
    }
    keyboard = _build_results_keyboard(
        state,
        action_items,
        available_flags=available_flags,
        include_overflow=True,
    )
    await send_split_message(
        context,
        chat_id,
        "\n".join(lines).strip(),
        reply_markup=keyboard,
        parse_mode="HTML",
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query is None:
        return
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id if query.message else None
    if chat_id is None:
        return
    user_id = update.effective_user.id if update.effective_user else None
    if not is_authorized(user_id):
        await send_split_message(
            context,
            chat_id,
            "Нет доступа. Обратитесь к администратору.",
        )
        return
    action = query.data or ""
    if action == "a:search":
        await send_split_message(context, chat_id, "Отправьте текстовый запрос для поиска.")
        return
    if action == "a:status":
        if not is_admin(user_id):
            await send_split_message(context, chat_id, "Нет доступа.")
            return
        await _send_admin_summary(context, chat_id)
        return
    if action == "a:clear":
        state = SEARCH_STATE.get(chat_id)
        if state is None:
            await send_split_message(context, chat_id, "Фильтры уже сброшены.")
            return
        state.flags = {key: None for key in state.flags}
        state.price_min = None
        state.price_max = None
        state.offset = 0
        state.limit = PAGE_SIZE
        await send_split_message(context, chat_id, "Фильтры сброшены.")
        return

    state = SEARCH_STATE.get(chat_id)
    if state is None:
        await send_split_message(context, chat_id, "Состояние поиска устарело. Повторите запрос.")
        return
    in_refine_menu = bool(query.message and query.message.text and "Уточните фильтры" in query.message.text)
    if action.startswith("s:desc:"):
        item_id = int(action.split(":")[2])
        conn = get_connection()
        row = conn.execute(
            "SELECT name, description FROM items WHERE id = ? AND is_valid = 1",
            (item_id,),
        ).fetchone()
        conn.close()
        if not row:
            await send_split_message(context, chat_id, "Описание не найдено.")
            return
        description = row["description"] or "Описание отсутствует."
        message = f"<b>{_escape_html(row['name'] or '-')}</b>\n{_escape_html(description)}"
        await send_split_message(context, chat_id, message, parse_mode="HTML")
        return
    if action.startswith("s:similar:"):
        item_id = int(action.split(":")[2])
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM items WHERE id = ? AND is_valid = 1",
            (item_id,),
        ).fetchone()
        if not row:
            conn.close()
            await send_split_message(context, chat_id, "Элемент не найден.")
            return
        base_item = dict(row)
        similar = find_similar(conn, base_item, limit=10)
        conn.close()
        if not similar:
            await send_split_message(context, chat_id, "Похожие варианты не найдены.")
            return
        header_dims = _format_dims(base_item.get("w_mm"), base_item.get("d_mm"), base_item.get("h_mm"))
        header_name = base_item.get("name") or "-"
        header = (
            f"Похожие варианты для: <b>{_escape_html(header_name)}</b>"
            + (f" ({header_dims})" if header_dims else "")
        )
        lines = [header, ""]
        for idx, item in enumerate(similar, start=1):
            item_lines, _ = _format_item(item, state.parsed.category, idx)
            lines.extend(item_lines)
            lines.append("")
        await send_split_message(
            context,
            chat_id,
            "\n".join(lines).strip(),
            parse_mode="HTML",
        )
        return
    if action == "s:filters":
        await send_split_message(
            context,
            chat_id,
            "Уточните фильтры:",
            reply_markup=_build_refine_keyboard(state),
        )
        return
    if action == "s:cancel":
        await send_split_message(context, chat_id, "Отменено.")
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
                "Уточните фильтры:",
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
                "Уточните фильтры:",
                reply_markup=_build_refine_keyboard(state),
            )
            return
    if action.startswith("s:tol:"):
        parts = action.split(":")
        dim = parts[2]
        value = int(parts[3])
        tol_w, tol_d, tol_h = state.tol_by_dim
        if dim == "w":
            tol_w = value
        elif dim == "d":
            tol_d = value
        elif dim == "h":
            tol_h = value
        state.tol_by_dim = (tol_w, tol_d, tol_h)
        if in_refine_menu:
            await send_split_message(
                context,
                chat_id,
                "Уточните фильтры:",
                reply_markup=_build_refine_keyboard(state),
            )
            return
    if action == "s:clear_flags":
        state.flags = {key: None for key in state.flags}
    if action == "s:text_only":
        state.flags = {key: None for key in state.flags}
        state.parsed = ParsedQuery(
            None,
            False,
            (None, None, None),
            (None, None, None),
            state.flags,
            state.keywords,
        )
        state.tol_by_dim = (None, None, None)
    if action == "s:more":
        if state.offset + state.limit >= state.total:
            await send_split_message(context, chat_id, "Больше результатов нет.")
            return
        state.offset += state.limit
    if action == "s:all":
        if state.total > SHOW_ALL_LIMIT:
            await send_split_message(
                context,
                chat_id,
                "Слишком много результатов. Уточните фильтры.",
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
        tol_by_dim=state.tol_by_dim,
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
    state.tol_by_dim = result["tol"]
    state.items = result["results"]
    if state.offset >= state.total:
        state.offset = 0

    await _render_search_results(
        context,
        chat_id,
        result,
        state,
        show_page=show_page,
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.exception("Unhandled error in Telegram bot", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try:
            await send_split_message(
                context,
                update.effective_chat.id,
                "Что-то пошло не так. Попробуйте позже.",
            )
        except Exception:
            LOGGER.exception("Failed to send error message to user.")


def build_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN not set")
    app = Application.builder().token(BOT_TOKEN).post_init(_send_startup_summary).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("status", status_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("users", users_handler))
    app.add_handler(CommandHandler("allow", allow_handler))
    app.add_handler(CommandHandler("deny", deny_handler))
    app.add_handler(CommandHandler("reindex", reindex_handler))
    app.add_handler(CommandHandler("debug_mapping", debug_mapping_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_error_handler(error_handler)
    return app


def run_bot() -> None:
    app = build_app()
    LOGGER.info("Starting polling...")
    app.run_polling()
