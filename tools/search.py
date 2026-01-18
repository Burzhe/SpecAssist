from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

from config import DEFAULT_TOL_MM, MAX_RESULTS
from tools.db import get_meta

CATEGORY_STEMS = {
    "шкаф": ("шкаф", "пенал", "гардероб"),
    "встроенный": ("встро", "встраив", "встройк"),
    "стол": ("стол", "столешн"),
    "бенч": ("бенч", "bench"),
    "бар": ("бар", "стойк"),
    "перила": ("перил", "поручн"),
    "зеркало": ("зеркал",),
    "перегородка": ("перегород", "стеклян"),
    "дверь": ("двер",),
    "панель": ("панел", "стенов"),
}

FLAG_TOKENS = {
    "has_led": ("подсвет", "подсветка", "led", "лента"),
    "mat_ldsp": ("лдсп", "egger", "ламинирован"),
    "mat_mdf": ("мдф", "mdf"),
    "mat_veneer": ("шпон", "veneer"),
    "has_glass": ("стекл", "зеркал"),
    "has_metal": ("металл", "сталь", "нерж", "алюм", "порошк"),
}


@dataclass
class ParsedQuery:
    category: str | None
    dims: tuple[int | None, int | None, int | None]
    flags: dict[str, bool]
    keywords: list[str]


def parse_query(query: str) -> ParsedQuery:
    normalized = _normalize_text(query)
    dims = _extract_dims(normalized)
    category = _extract_category(normalized)
    flags = {key: any(token in normalized for token in tokens) for key, tokens in FLAG_TOKENS.items()}
    keywords = _extract_keywords(normalized)
    return ParsedQuery(category=category, dims=dims, flags=flags, keywords=keywords)


def search_items(conn: sqlite3.Connection, query: str) -> dict[str, Any]:
    parsed = parse_query(query)
    active_version = get_meta(conn, "active_version")

    tol_by_dim = _default_tol_by_dim(parsed.dims)
    keywords = parsed.keywords
    flags = _normalize_flag_filters(parsed.flags)
    relax_steps: list[str] = []

    results = _run_search(conn, active_version, parsed, tol_by_dim, keywords, flags)
    if len(results) < MAX_RESULTS:
        for next_tol in (100, 200):
            if len(results) >= MAX_RESULTS or parsed.dims == (None, None, None):
                break
            tol_by_dim = _increase_tol_by_dim(parsed.dims, tol_by_dim, next_tol)
            relax_steps.append(f"tol={next_tol}")
            results = _run_search(conn, active_version, parsed, tol_by_dim, keywords, flags)
            if results:
                break

    if len(results) < MAX_RESULTS and any(value is True for value in flags.values()):
        drop_order = ["has_glass", "has_metal", "mat_veneer", "mat_mdf", "mat_ldsp", "has_led"]
        for flag in drop_order:
            if flags.get(flag) is True:
                flags[flag] = None
                relax_steps.append(f"drop:{flag}")
                results = _run_search(conn, active_version, parsed, tol_by_dim, keywords, flags)
                if results:
                    break

    if len(results) < MAX_RESULTS and keywords:
        keywords = sorted(keywords, key=len, reverse=True)[:2]
        relax_steps.append("keywords:shortened")
        results = _run_search(conn, active_version, parsed, tol_by_dim, keywords, flags)

    if not results:
        relax_steps.append("fallback:text-only")
        parsed = ParsedQuery(parsed.category, (None, None, None), flags, keywords)
        results = _run_search(conn, active_version, parsed, tol_by_dim, keywords, flags)

    ranked = _rank_results(results, parsed.dims)

    return {
        "results": ranked[:MAX_RESULTS],
        "total": len(ranked),
        "tol": tol_by_dim,
        "relaxed": relax_steps,
        "active_version": active_version,
        "parsed": parsed,
        "flags": flags,
        "keywords": keywords,
    }


def search_items_with_params(
    conn: sqlite3.Connection,
    query: str,
    *,
    parsed: ParsedQuery | None = None,
    keywords: list[str] | None = None,
    flags: dict[str, bool | None] | None = None,
    tol_by_dim: tuple[int | None, int | None, int | None] | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    limit: int = MAX_RESULTS,
    offset: int = 0,
) -> dict[str, Any]:
    parsed = parsed or parse_query(query)
    keywords = keywords or parsed.keywords
    flag_filters = _normalize_flag_filters(parsed.flags, flags)
    tol_by_dim = tol_by_dim or _default_tol_by_dim(parsed.dims)
    active_version = get_meta(conn, "active_version")
    rows = _run_search(
        conn,
        active_version,
        parsed,
        tol_by_dim,
        keywords,
        flag_filters,
        price_min=price_min,
        price_max=price_max,
    )
    ranked = _rank_results(rows, parsed.dims)
    return {
        "results": ranked[offset : offset + limit],
        "total": len(ranked),
        "tol": tol_by_dim,
        "active_version": active_version,
        "parsed": parsed,
        "flags": flag_filters,
        "keywords": keywords,
    }


def _run_search(
    conn: sqlite3.Connection,
    active_version: str | None,
    parsed: ParsedQuery,
    tol_by_dim: tuple[int | None, int | None, int | None],
    keywords: list[str],
    flags: dict[str, bool | None],
    *,
    price_min: float | None = None,
    price_max: float | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = ["1=1", "items.is_valid = 1"]
    if active_version:
        where.append("items.source_version = ?")
        params.append(active_version)

    if parsed.category:
        synonyms = CATEGORY_STEMS.get(parsed.category, (parsed.category,))
        like_clauses = " OR ".join(["items.name LIKE ?"] * len(synonyms))
        where.append(f"({like_clauses})")
        params.extend([f"%{term}%" for term in synonyms])

    w_mm, d_mm, h_mm = parsed.dims
    tol_w, tol_d, tol_h = tol_by_dim
    if w_mm is not None:
        where.append("items.w_mm IS NOT NULL AND ABS(items.w_mm - ?) <= ?")
        params.extend([w_mm, tol_w or DEFAULT_TOL_MM])
    if d_mm is not None:
        where.append("items.d_mm IS NOT NULL AND ABS(items.d_mm - ?) <= ?")
        params.extend([d_mm, tol_d or DEFAULT_TOL_MM])
    if h_mm is not None:
        where.append("items.h_mm IS NOT NULL AND ABS(items.h_mm - ?) <= ?")
        params.extend([h_mm, tol_h or DEFAULT_TOL_MM])

    for flag, enabled in flags.items():
        if enabled is True:
            where.append(f"items.{flag} = 1")
        elif enabled is False:
            where.append(f"items.{flag} = 0")

    if price_min is not None:
        where.append("items.price_unit_ex_vat >= ?")
        params.append(price_min)
    if price_max is not None:
        where.append("items.price_unit_ex_vat <= ?")
        params.append(price_max)

    if keywords:
        sql = """
            SELECT items.*, fts.rank AS fts_rank
            FROM items
            LEFT JOIN (
                SELECT rowid, bm25(items_fts) AS rank
                FROM items_fts
                WHERE items_fts MATCH ?
                LIMIT 200
            ) AS fts ON items.id = fts.rowid
            WHERE {where_clause}
        """
        fts_query = " ".join(keywords)
        params_with_fts = [fts_query] + params
        sql = sql.format(where_clause=" AND ".join(where))
        rows = conn.execute(sql, params_with_fts).fetchall()
    else:
        sql = f"SELECT items.*, NULL AS fts_rank FROM items WHERE {' AND '.join(where)}"
        rows = conn.execute(sql, params).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        results.append(dict(row))
    return results


def _normalize_flag_filters(
    parsed_flags: dict[str, bool],
    overrides: dict[str, bool | None] | None = None,
) -> dict[str, bool | None]:
    flag_filters: dict[str, bool | None] = {
        key: True if value else None for key, value in parsed_flags.items()
    }
    if overrides:
        for key, value in overrides.items():
            flag_filters[key] = value
    for key in FLAG_TOKENS:
        flag_filters.setdefault(key, None)
    return flag_filters


def _rank_results(rows: list[dict[str, Any]], dims: tuple[int | None, int | None, int | None]) -> list[dict[str, Any]]:
    w_mm, d_mm, h_mm = dims

    def score(row: dict[str, Any]) -> tuple:
        fts_rank = row.get("fts_rank")
        fts_value = fts_rank if fts_rank is not None else 1e6
        dim_score = 0
        for key, value in zip(("w_mm", "d_mm", "h_mm"), (w_mm, d_mm, h_mm)):
            if value is not None and row.get(key) is not None:
                dim_score += abs(row.get(key) - value)
        flag_score = sum(
            1 for flag in FLAG_TOKENS.keys() if row.get(flag) and row.get(flag) == 1
        )
        return (fts_value, dim_score, -flag_score)

    return sorted(rows, key=score)


def find_similar(
    conn: sqlite3.Connection,
    item: dict[str, Any],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    name = item.get("name") or ""
    normalized = _normalize_text(name)
    category = _extract_category(normalized)
    keywords = sorted(_extract_keywords(normalized), key=len, reverse=True)[:4]
    parsed = ParsedQuery(
        category=category,
        dims=(item.get("w_mm"), item.get("d_mm"), item.get("h_mm")),
        flags={key: bool(item.get(key)) for key in FLAG_TOKENS},
        keywords=keywords,
    )
    tol_by_dim = _default_tol_by_dim(parsed.dims, multiplier=2.0)
    flags = _normalize_flag_filters(parsed.flags)
    active_version = get_meta(conn, "active_version")
    rows = _run_search(
        conn,
        active_version,
        parsed,
        tol_by_dim,
        keywords,
        flags,
    )
    rows = [row for row in rows if row.get("id") != item.get("id")]
    ranked = _rank_similar_results(rows, parsed)
    return ranked[:limit]


def _rank_similar_results(rows: list[dict[str, Any]], parsed: ParsedQuery) -> list[dict[str, Any]]:
    w_mm, d_mm, h_mm = parsed.dims

    def score(row: dict[str, Any]) -> tuple:
        fts_rank = row.get("fts_rank")
        fts_value = fts_rank if fts_rank is not None else 1e6
        dim_score = 0
        dim_hits = 0
        for key, value in zip(("w_mm", "d_mm", "h_mm"), (w_mm, d_mm, h_mm)):
            if value is not None and row.get(key) is not None:
                dim_score += abs(row.get(key) - value)
                dim_hits += 1
        material_hits = sum(
            1 for flag in FLAG_TOKENS.keys() if row.get(flag) and parsed.flags.get(flag)
        )
        category_hit = 0
        if parsed.category and row.get("name"):
            category_hit = 1 if any(
                stem in _normalize_text(row["name"]) for stem in CATEGORY_STEMS.get(parsed.category, ())
            ) else 0
        return (-category_hit, -material_hits, dim_score if dim_hits else 1e9, fts_value)

    return sorted(rows, key=score)


def _normalize_text(text: str) -> str:
    lowered = text.lower().replace("ё", "е")
    lowered = re.sub(r"[-–—]+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _extract_dims(text: str) -> tuple[int | None, int | None, int | None]:
    w_mm = _extract_prefixed_dim(text, ("w", "ш", "ширин"))
    d_mm = _extract_prefixed_dim(text, ("d", "г", "глубин"))
    h_mm = _extract_prefixed_dim(text, ("h", "в", "высот"))

    match = re.search(r"(\d{2,5})\s*[x×*х]\s*(\d{2,5})\s*[x×*х]\s*(\d{2,5})", text)
    if match:
        w_mm = w_mm or int(match.group(1))
        d_mm = d_mm or int(match.group(2))
        h_mm = h_mm or int(match.group(3))

    return (w_mm, d_mm, h_mm)


def _extract_category(text: str) -> str | None:
    for category, tokens in CATEGORY_STEMS.items():
        if any(token in text for token in tokens):
            return category
    return None


def _extract_keywords(text: str) -> list[str]:
    words = text.split()
    keywords = [word for word in words if len(word) >= 3 and not word.isdigit()]
    return keywords[:10]


def _extract_prefixed_dim(text: str, stems: Iterable[str]) -> int | None:
    for stem in stems:
        match = re.search(rf"{stem}\s*[:=]?\s*(\d{{2,5}})", text)
        if match:
            return int(match.group(1))
    return None


def _default_tol_by_dim(
    dims: tuple[int | None, int | None, int | None],
    *,
    multiplier: float = 1.0,
) -> tuple[int | None, int | None, int | None]:
    count = sum(1 for value in dims if value is not None)
    if count == 0:
        return (None, None, None)
    if count == 1:
        tol = int(100 * multiplier)
    elif count == 2:
        tol = int(50 * multiplier)
    else:
        tol = int(20 * multiplier)
    return tuple(tol if value is not None else None for value in dims)  # type: ignore[return-value]


def _increase_tol_by_dim(
    dims: tuple[int | None, int | None, int | None],
    current: tuple[int | None, int | None, int | None],
    next_tol: int,
) -> tuple[int | None, int | None, int | None]:
    return tuple(
        next_tol if value is not None else None
        for value in dims
    )  # type: ignore[return-value]
