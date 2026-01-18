from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import ADMIN_IDS, BOT_TOKEN
from tools.db import (
    add_allowed_user,
    get_connection,
    get_meta,
    is_allowed_user,
    list_allowed_users,
    list_versions,
    remove_allowed_user,
    set_meta,
)
from tools.search import ParsedQuery, search_items, search_items_with_params


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


def _is_authorized(user_id: int | None) -> bool:
    if user_id is None:
        return False
    if is_admin(user_id):
        return True
    conn = get_connection()
    allowed = is_allowed_user(conn, user_id)
    conn.close()
    return allowed


async def _reject_unauthorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id if update.effective_user else None
    if _is_authorized(user_id):
        return False
    if update.message:
        if update.message.text and update.message.text.startswith("/start"):
            message = "Нет доступа. Напишите администратору."
            if user_id is not None:
                message = f"{message}\nВаш ID: {user_id}"
            await send_split_message(context, update.effective_chat.id, message)
            return True
        await send_split_message(context, update.effective_chat.id, "Нет доступа. Напишите администратору.")
    elif update.callback_query and update.callback_query.message:
        await send_split_message(
            context,
            update.callback_query.message.chat_id,
            "Нет доступа. Напишите администратору.",
        )
    return True


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if await _reject_unauthorized(update, context):
        return
    await send_split_message(
        context,
        update.effective_chat.id,
        "SpecAssist — поиск по спецификациям.\n"
        "Просто напишите запрос, /s тоже поддерживается.\n"
        "\n"
        "Примеры:\n"
        "• шкаф из лдсп с подсветкой\n"
        "• тумба 1200х600х800\n"
        "• ресепшн мдф до 300k",
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start_handler(update, context)


async def versions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if await _reject_unauthorized(update, context):
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Нет доступа. Напишите администратору.")
        return
    conn = get_connection()
    versions = list_versions(conn)
    active = get_meta(conn, "active_version")
    conn.close()
    if not active and not versions:
        await send_split_message(
            context,
            update.effective_chat.id,
            "Нет активной версии. Выполните reindex через CLI.",
        )
        return
    lines = []
    if not active:
        lines.append("Нет активной версии. Выполните reindex через CLI.")
        lines.append("")
    lines.append("Версии:")
    for version in versions:
        marker = " (активная)" if version == active else ""
        lines.append(f"- {version}{marker}")
    await send_split_message(context, update.effective_chat.id, "\n".join(lines))


async def use_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if await _reject_unauthorized(update, context):
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Нет доступа. Напишите администратору.")
        return
    if not context.args:
        await send_split_message(context, update.effective_chat.id, "Использование: /use <version>")
        return
    version = context.args[0]
    conn = get_connection()
    set_meta(conn, "active_version", version)
    conn.close()
    await send_split_message(context, update.effective_chat.id, f"Активная версия: {version}")


async def users_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if await _reject_unauthorized(update, context):
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Нет доступа. Напишите администратору.")
        return
    conn = get_connection()
    users = list_allowed_users(conn)
    conn.close()
    if not users:
        await send_split_message(context, update.effective_chat.id, "Список пуст.")
        return
    lines = ["Разрешённые пользователи:"]
    for user in users:
        name_parts = [user["first_name"], user["last_name"]]
        name = " ".join(part for part in name_parts if part)
        username = f"@{user['username']}" if user["username"] else ""
        meta = " ".join(part for part in (name, username) if part)
        if meta:
            meta = f" — {meta}"
        added_at = user["added_at"]
        if added_at:
            lines.append(f"- {user['user_id']}{meta} (добавлен {added_at})")
        else:
            lines.append(f"- {user['user_id']}{meta}")
    await send_split_message(context, update.effective_chat.id, "\n".join(lines))


async def allow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if await _reject_unauthorized(update, context):
        return
    admin_id = update.effective_user.id if update.effective_user else None
    if not is_admin(admin_id):
        await send_split_message(context, update.effective_chat.id, "Нет доступа. Напишите администратору.")
        return
    if not context.args or not context.args[0].isdigit():
        await send_split_message(context, update.effective_chat.id, "Использование: /allow <user_id>")
        return
    user_id = int(context.args[0])
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
    await send_split_message(context, update.effective_chat.id, f"Пользователь {user_id} добавлен.")


async def deny_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if await _reject_unauthorized(update, context):
        return
    if not is_admin(update.effective_user.id if update.effective_user else None):
        await send_split_message(context, update.effective_chat.id, "Нет доступа. Напишите администратору.")
        return
    if not context.args or not context.args[0].isdigit():
        await send_split_message(context, update.effective_chat.id, "Использование: /deny <user_id>")
        return
    user_id = int(context.args[0])
    conn = get_connection()
    removed = remove_allowed_user(conn, user_id)
    conn.close()
    if removed:
        await send_split_message(context, update.effective_chat.id, f"Пользователь {user_id} удалён.")
        return
    await send_split_message(context, update.effective_chat.id, f"Пользователь {user_id} не найден.")


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if await _reject_unauthorized(update, context):
        return
    query = " ".join(context.args).strip()
    if not query:
        await send_split_message(context, update.effective_chat.id, "Введите запрос после /s.")
        return
    await _run_search(update, context, query)


async def text_search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if await _reject_unauthorized(update, context):
        return
    query = update.message.text.strip()
    if not query:
        return
    await _run_search(update, context, query)


async def _run_search(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
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


def _format_dims(w_mm: int | None, d_mm: int | None, h_mm: int | None) -> str:
    dims = [val for val in (w_mm, d_mm, h_mm) if val is not None]
    return "×".join(str(val) for val in dims) if dims else ""


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
    if unit is None and total is not None and qty and qty > 0:
        unit = total / qty
    if unit is None:
        return ""
    return f"{unit:.2f} ₽"


def _format_description(description: str | None, limit: int = 200) -> str | None:
    if not description:
        return None
    if str(description).strip().lower() == "nan":
        return None
    cleaned = description.strip().replace("\r\n", "\n")
    if not cleaned:
        return None
    lines = cleaned.split("\n")
    snippet = " ".join(line.strip() for line in lines if line.strip())
    if len(snippet) > limit:
        return snippet[: limit - 1].rstrip() + "…"
    return snippet


def _format_item(item: dict, item_type: str | None) -> list[str]:
    raw_name = item.get("name")
    name = raw_name if raw_name and str(raw_name).lower() != "nan" else None
    title = name or item_type or "Без названия"
    lines = [title]
    dims = _format_dims(item.get("w_mm"), item.get("d_mm"), item.get("h_mm"))
    if dims:
        lines.append(f"Габариты: {dims} мм")
    price = _format_price(item)
    if price:
        lines.append(f"Цена за 1 шт: {price}")
    qty = item.get("qty")
    if qty is not None:
        qty_label = int(qty) if isinstance(qty, (int, float)) and qty == int(qty) else qty
        lines.append(f"Кол-во: {qty_label}")
    description = _format_description(item.get("description"))
    if description:
        lines.append(f"Описание: {description}")
    lines.append(
        'Excel: лист "{sheet}", строка {row}'.format(
            sheet=item.get("source_sheet") or "?",
            row=item.get("source_row") or "?",
        )
    )
    return lines


def _format_query_summary(state: SearchState) -> str:
    dims = _format_dims(*state.parsed.dims)
    flags = _format_flags_from_filters(state.flags)
    keywords = ", ".join(state.keywords) if state.keywords else "-"
    parts = [
        f"Категория: {state.parsed.category or '-'}",
        f"Габариты: {dims or '-'}",
        f"Фильтры: {flags or '-'}",
        f"Ключевые слова: {keywords}",
        f"Допуск: {state.tol} мм",
    ]
    if state.relaxed:
        parts.append(f"Смягчение: {_format_relaxed_steps(state.relaxed)}")
    if state.price_min is not None or state.price_max is not None:
        parts.append(f"Цена: {_format_price_range(state.price_min, state.price_max)}")
    return "\n".join(parts)


def _format_flags_from_filters(flags: dict[str, bool | None]) -> str:
    labels = []
    for key, label in (
        ("has_led", "Подсветка"),
        ("mat_mdf", "МДФ"),
        ("mat_ldsp", "ЛДСП"),
        ("mat_veneer", "Шпон"),
        ("has_glass", "Стекло"),
        ("has_metal", "Металл"),
    ):
        state = flags.get(key)
        if state is True:
            labels.append(label)
        elif state is False:
            labels.append(f"{label}✕")
    return "/".join(labels)


def _format_price_range(price_min: float | None, price_max: float | None) -> str:
    if price_min is not None and price_max is not None:
        return f"{int(price_min)}–{int(price_max)}"
    if price_min is not None:
        return f"{int(price_min)}+"
    if price_max is not None:
        return f"до {int(price_max)}"
    return "-"


def _format_relaxed_steps(steps: list[str]) -> str:
    label_map = {
        "drop:has_led": "подсветка игнорируется",
        "drop:mat_ldsp": "ЛДСП игнорируется",
        "drop:mat_mdf": "МДФ игнорируется",
        "drop:mat_veneer": "шпон игнорируется",
        "drop:has_glass": "стекло игнорируется",
        "drop:has_metal": "металл игнорируется",
    }
    formatted = []
    for step in steps:
        if step in label_map:
            formatted.append(label_map[step])
        elif step.startswith("tol="):
            formatted.append(step.replace("tol=", "допуск="))
        elif step == "keywords:shortened":
            formatted.append("ключевые слова сокращены")
        elif step == "fallback:text-only":
            formatted.append("поиск только по тексту")
        else:
            formatted.append(step)
    return ", ".join(formatted)


def _next_flag_state(current: bool | None) -> bool | None:
    if current is None:
        return True
    if current is True:
        return False
    return None


def _flag_button_label(label: str, state: bool | None, *, any_label: str = "Любая") -> str:
    suffix = any_label
    if state is True:
        suffix = "Да"
    elif state is False:
        suffix = "Нет"
    return f"{label} — {suffix}"


def _build_overflow_keyboard(
    state: SearchState,
    enable_show_all: bool,
    available_flags: dict[str, bool],
) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("➕ Добавить фильтры", callback_data="s:filters")],
        [InlineKeyboardButton("📄 Показать ещё (10)", callback_data="s:more")],
    ]
    if enable_show_all:
        rows.append([InlineKeyboardButton("📄 Показать все (осторожно)", callback_data="s:all")])
    if state.parsed.dims != (None, None, None):
        rows.append(
            [
                InlineKeyboardButton("↔️ Увеличить допуск", callback_data="s:tol_up"),
                InlineKeyboardButton("↔️ Уменьшить допуск", callback_data="s:tol_down"),
            ]
        )
    flag_row = []
    for key, label in (
        ("has_led", "Подсветка"),
        ("mat_mdf", "Материал: МДФ"),
        ("mat_ldsp", "Материал: ЛДСП"),
        ("mat_veneer", "Материал: Шпон"),
        ("has_glass", "Стекло"),
        ("has_metal", "Металл"),
    ):
        if not available_flags.get(key):
            continue
        flag_row.append(
            InlineKeyboardButton(
                _flag_button_label(
                    label,
                    state.flags.get(key),
                    any_label="Любой" if label.startswith("Материал") else "Любая",
                ),
                callback_data=f"s:toggle:{key}",
            )
        )
        if len(flag_row) == 3:
            rows.append(flag_row)
            flag_row = []
    if flag_row:
        rows.append(flag_row)
    rows.append([InlineKeyboardButton("❌ Очистить фильтры", callback_data="s:clear")])
    return InlineKeyboardMarkup(rows)


def _build_refine_keyboard(state: SearchState) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                _flag_button_label("Подсветка", state.flags.get("has_led")),
                callback_data="s:toggle:has_led",
            )
        ],
        [
            InlineKeyboardButton(
                _flag_button_label("Материал: МДФ", state.flags.get("mat_mdf"), any_label="Любой"),
                callback_data="s:toggle:mat_mdf",
            ),
            InlineKeyboardButton(
                _flag_button_label("Материал: ЛДСП", state.flags.get("mat_ldsp"), any_label="Любой"),
                callback_data="s:toggle:mat_ldsp",
            ),
            InlineKeyboardButton(
                _flag_button_label("Материал: Шпон", state.flags.get("mat_veneer"), any_label="Любой"),
                callback_data="s:toggle:mat_veneer",
            ),
        ],
        [
            InlineKeyboardButton(
                _flag_button_label("Стекло", state.flags.get("has_glass")),
                callback_data="s:toggle:has_glass",
            ),
            InlineKeyboardButton(
                _flag_button_label("Металл", state.flags.get("has_metal")),
                callback_data="s:toggle:has_metal",
            ),
        ],
        [
            InlineKeyboardButton("Цена: до 100k", callback_data="s:price:max:100000"),
            InlineKeyboardButton("Цена: 100–300", callback_data="s:price:range:100000:300000"),
            InlineKeyboardButton("Цена: 300–700", callback_data="s:price:range:300000:700000"),
            InlineKeyboardButton("Цена: 700+", callback_data="s:price:min:700000"),
        ],
        [InlineKeyboardButton("Цена: любая", callback_data="s:price:clear")],
    ]
    if state.parsed.dims != (None, None, None):
        rows.append(
            [
                InlineKeyboardButton("Допуск: 20", callback_data="s:tol:20"),
                InlineKeyboardButton("Допуск: 50", callback_data="s:tol:50"),
                InlineKeyboardButton("Допуск: 100", callback_data="s:tol:100"),
                InlineKeyboardButton("Допуск: 200", callback_data="s:tol:200"),
            ]
        )
    rows.append(
        [
            InlineKeyboardButton("Применить", callback_data="s:apply"),
            InlineKeyboardButton("Отмена", callback_data="s:cancel"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _build_no_results_keyboard(state: SearchState) -> InlineKeyboardMarkup:
    rows = []
    if state.parsed.dims != (None, None, None):
        rows.append([InlineKeyboardButton("Увеличить допуск", callback_data="s:tol_up")])
    rows.append(
        [
            InlineKeyboardButton("Очистить фильтры", callback_data="s:clear_flags"),
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
        lines.append("Попробуйте убрать фильтры или увеличить допуск.")
        await send_split_message(
            context,
            chat_id,
            "\n".join(lines),
            reply_markup=_build_no_results_keyboard(state),
        )
        return

    header = f"Найдено ≈{total} вариантов."
    if state.relaxed:
        header = "Точное совпадение не найдено. Показываю ближайшее ({details}).".format(
            details=_format_relaxed_steps(state.relaxed)
        )
    if total > TOO_MANY_THRESHOLD:
        header = f"{header} Слишком много, покажу лучшие."
    lines.append(header)
    lines.append(_format_query_summary(state))
    lines.append("")

    if total <= MAX_RESULTS_DEFAULT or show_page:
        if total <= MAX_RESULTS_DEFAULT:
            lines.append("Результаты:")
        else:
            start = state.offset + 1
            end = min(state.offset + state.limit, total)
            lines.append(f"Результаты {start}–{end}:")
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

    lines.append("Лучшие варианты:")
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
    if await _reject_unauthorized(update, context):
        return
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id if query.message else None
    if chat_id is None:
        return
    state = SEARCH_STATE.get(chat_id)
    if state is None:
        await send_split_message(context, chat_id, "Состояние поиска истекло. Запустите поиск снова.")
        return

    action = query.data or ""
    in_refine_menu = bool(query.message and query.message.text and "Фильтры" in query.message.text)
    if action == "s:filters":
        await send_split_message(
            context,
            chat_id,
            "Фильтры:",
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
                "Фильтры:",
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
                "Фильтры:",
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
                "Фильтры:",
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
            await send_split_message(context, chat_id, "Больше результатов нет.")
            return
        state.offset += state.limit
    if action == "s:all":
        if state.total > SHOW_ALL_LIMIT:
            await send_split_message(
                context,
                chat_id,
                "Слишком много результатов. Уточните запрос.",
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
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("s", search_handler))
    app.add_handler(CommandHandler("versions", versions_handler))
    app.add_handler(CommandHandler("use", use_handler))
    app.add_handler(CommandHandler("users", users_handler))
    app.add_handler(CommandHandler("allow", allow_handler))
    app.add_handler(CommandHandler("deny", deny_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_search_handler))
    app.add_error_handler(error_handler)
    return app


def run_bot() -> None:
    app = build_app()
    LOGGER.info("Starting polling...")
    app.run_polling()
