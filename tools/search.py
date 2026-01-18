from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

from config import DEFAULT_TOL_MM, MAX_RESULTS

CATEGORY_STEMS = {
    "шкаф": ("шкаф", "пенал", "гардероб", "купе"),
    "стеллаж": ("стеллаж", "стелаж", "стелл"),
    "кухня": ("кухн",),
    "стол": ("стол", "столешн", "столик"),
    "бенч-стол": ("бенч", "bench", "бенч-стол"),
    "бар": ("бар", "барн", "стойк"),
    "дверь": ("двер", "дверн"),
    "перила": ("перил", "поручн"),
    "зеркало": ("зеркал",),
}

FLAG_TOKENS = {
    "has_led": ("подсвет", "подсветка", "led", "лента", "светодиод"),
    "mat_ldsp": ("лдсп", "egger", "ламинирован", "ламинир"),
    "mat_mdf": ("мдф", "mdf"),
    "mat_veneer": ("шпон", "veneer"),
    "has_glass": ("стекл", "зеркал"),
    "has_metal": ("металл", "сталь", "нерж", "нержав", "inox", "алюм", "порошк"),
}

WORD_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)


@dataclass
class ParsedQuery:
    category: str | None
    category_confident: bool
    dims: tuple[int | None, int | None, int | None]
    tol_by_dim: tuple[int | None, int | None, int | None]
    flags: dict[str, bool]
    keywords: list[str]


def parse_query(query: str) -> ParsedQuery:
    raw_lower = query.lower().replace("ё", "е")
    normalized = _normalize_text(query)
    dims, tol_by_dim = _extract_dims_with_tolerance(raw_lower)
    category, confident, category_tokens = _extract_category(normalized)
    flags, flag_tokens = _extract_flags(normalized)
    keywords = _extract_keywords(normalized, category_tokens, flag_tokens)
    return ParsedQuery(
        category=category,
        category_confident=confident,
        dims=dims,
        tol_by_dim=tol_by_dim,
        flags=flags,
        keywords=keywords,
    )


def search_items(conn: sqlite3.Connection, query: str) -> dict[str, Any]:
    parsed = parse_query(query)
    tol_by_dim = parsed.tol_by_dim
    keywords = parsed.keywords
    flags = _normalize_flag_filters(parsed.flags)

    results = _run_search(conn, parsed, tol_by_dim, flags)
    ranked = _rank_results(results, parsed.dims, keywords)

    return {
        "results": ranked[:MAX_RESULTS],
        "total": len(ranked),
        "tol": tol_by_dim,
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
    tol_by_dim = tol_by_dim or parsed.tol_by_dim
    rows = _run_search(
        conn,
        parsed,
        tol_by_dim,
        flag_filters,
        price_min=price_min,
        price_max=price_max,
    )
    ranked = _rank_results(rows, parsed.dims, keywords)
    return {
        "results": ranked[offset : offset + limit],
        "total": len(ranked),
        "tol": tol_by_dim,
        "parsed": parsed,
        "flags": flag_filters,
        "keywords": keywords,
    }


def _run_search(
    conn: sqlite3.Connection,
    parsed: ParsedQuery,
    tol_by_dim: tuple[int | None, int | None, int | None],
    flags: dict[str, bool | None],
    *,
    price_min: float | None = None,
    price_max: float | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = ["1=1", "items.is_valid = 1"]

    if parsed.category and parsed.category_confident:
        where_clause, clause_params = _build_category_clause(parsed.category)
        where.append(where_clause)
        params.extend(clause_params)

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
            clause, clause_params = _build_flag_clause(flag)
            where.append(clause)
            params.extend(clause_params)
        elif enabled is False:
            where.append(f"(items.{flag} = 0 OR items.{flag} IS NULL)")

    if price_min is not None:
        where.append("items.price_unit_ex_vat >= ?")
        params.append(price_min)
    if price_max is not None:
        where.append("items.price_unit_ex_vat <= ?")
        params.append(price_max)

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


def _rank_results(
    rows: list[dict[str, Any]],
    dims: tuple[int | None, int | None, int | None],
    keywords: list[str],
) -> list[dict[str, Any]]:
    w_mm, d_mm, h_mm = dims

    def score(row: dict[str, Any]) -> tuple:
        name = row.get("name") or ""
        description = row.get("description") or ""
        normalized_text = _normalize_text(f"{name} {description}")
        keyword_score = _keyword_score(normalized_text, keywords)
        dim_score = _dimension_score(row, dims, use_min=not keywords)
        flag_score = sum(1 for flag in FLAG_TOKENS.keys() if row.get(flag) == 1)
        if keywords:
            return (-keyword_score, dim_score, -flag_score)
        return (dim_score, -flag_score, name)

    return sorted(rows, key=score)


def find_similar(
    conn: sqlite3.Connection,
    item: dict[str, Any],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    name = item.get("name") or ""
    normalized = _normalize_text(name)
    category, confident, _ = _extract_category(normalized)
    keywords = sorted(_extract_keywords(normalized, set(), set()), key=len, reverse=True)[:4]
    parsed = ParsedQuery(
        category=category,
        category_confident=confident,
        dims=(item.get("w_mm"), item.get("d_mm"), item.get("h_mm")),
        tol_by_dim=_default_tol_by_dim((item.get("w_mm"), item.get("d_mm"), item.get("h_mm"))),
        flags={key: bool(item.get(key)) for key in FLAG_TOKENS},
        keywords=keywords,
    )
    tol_by_dim = _default_tol_by_dim(parsed.dims, multiplier=2.0)
    flags = _normalize_flag_filters(parsed.flags)
    rows = _run_search(
        conn,
        parsed,
        tol_by_dim,
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
            category_hit = 1 if _matches_category_text(parsed.category, _normalize_text(row["name"])) else 0
        return (-category_hit, -material_hits, dim_score if dim_hits else 1e9, fts_value)

    return sorted(rows, key=score)


def _normalize_text(text: str) -> str:
    lowered = text.lower().replace("ё", "е")
    lowered = re.sub(r"[-–—]+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _extract_dims_with_tolerance(
    text: str,
) -> tuple[tuple[int | None, int | None, int | None], tuple[int | None, int | None, int | None]]:
    w_mm, tol_w = _extract_prefixed_dim_with_tol(text, ("w", "ш", "ширин"))
    d_mm, tol_d = _extract_prefixed_dim_with_tol(text, ("d", "г", "глубин"))
    h_mm, tol_h = _extract_prefixed_dim_with_tol(text, ("h", "в", "высот"))

    match = re.search(r"(\d{2,5})\s*[x×*х]\s*(\d{2,5})\s*[x×*х]\s*(\d{2,5})", text)
    if match:
        w_mm = w_mm or int(match.group(1))
        d_mm = d_mm or int(match.group(2))
        h_mm = h_mm or int(match.group(3))

    dims = (w_mm, d_mm, h_mm)
    tol_by_dim = _default_tol_by_dim(dims)
    tol_by_dim = (
        tol_w or tol_by_dim[0],
        tol_d or tol_by_dim[1],
        tol_h or tol_by_dim[2],
    )
    return dims, tol_by_dim


def _extract_category(text: str) -> tuple[str | None, bool, set[str]]:
    tokens = _tokenize(text)
    best_category: str | None = None
    best_score = 0
    best_tokens: set[str] = set()
    for category, stems in CATEGORY_STEMS.items():
        score = 0
        matched: set[str] = set()
        for token in tokens:
            for stem in stems:
                if token.startswith(stem):
                    score += len(stem)
                    matched.add(token)
        if score > best_score:
            best_category = category
            best_score = score
            best_tokens = matched
    confident = best_category is not None and best_score > 0
    return best_category, confident, best_tokens


def _extract_flags(text: str) -> tuple[dict[str, bool], set[str]]:
    flags: dict[str, bool] = {}
    matched_tokens: set[str] = set()
    tokens = _tokenize(text)
    for key, flag_tokens in FLAG_TOKENS.items():
        enabled = False
        for token in tokens:
            if any(token.startswith(flag_token) for flag_token in flag_tokens):
                enabled = True
                matched_tokens.add(token)
        flags[key] = enabled
    return flags, matched_tokens


def _extract_keywords(text: str, category_tokens: set[str], flag_tokens: set[str]) -> list[str]:
    keywords: list[str] = []
    for token in _tokenize(text):
        if token in category_tokens or token in flag_tokens:
            continue
        if any(char.isdigit() for char in token):
            continue
        if len(token) < 3:
            continue
        keywords.append(token)
    return keywords[:10]


def _extract_prefixed_dim_with_tol(
    text: str,
    stems: Iterable[str],
) -> tuple[int | None, int | None]:
    for stem in stems:
        match = re.search(rf"{stem}\s*[:=]?\s*(\d{{2,5}})(?:\s*(?:мм|mm)?)", text)
        if not match:
            continue
        value = int(match.group(1))
        tail = text[match.end() : match.end() + 20]
        tol_match = re.search(r"(?:±|\+/-|\+-)\s*(\d{1,4})", tail)
        tol = int(tol_match.group(1)) if tol_match else None
        return value, tol
    return None, None


def _default_tol_by_dim(
    dims: tuple[int | None, int | None, int | None],
    *,
    multiplier: float = 1.0,
) -> tuple[int | None, int | None, int | None]:
    tol = int(DEFAULT_TOL_MM * multiplier)
    if all(value is None for value in dims):
        return (None, None, None)
    return tuple(tol if value is not None else None for value in dims)  # type: ignore[return-value]


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _build_category_clause(category: str) -> tuple[str, list[Any]]:
    tokens = CATEGORY_STEMS.get(category, (category,))
    fields = [
        "LOWER(COALESCE(items.name, ''))",
        "LOWER(COALESCE(items.description, ''))",
        "LOWER(COALESCE(items.source_sheet, ''))",
    ]
    clauses: list[str] = []
    params: list[Any] = []
    for token in tokens:
        for field in fields:
            clauses.append(f"{field} LIKE ?")
            params.append(f"%{token}%")
    return f"({' OR '.join(clauses)})", params


def _build_flag_clause(flag: str) -> tuple[str, list[Any]]:
    tokens = FLAG_TOKENS.get(flag, ())
    fields = [
        "LOWER(COALESCE(items.name, ''))",
        "LOWER(COALESCE(items.description, ''))",
    ]
    clauses = [f"items.{flag} = 1"]
    params: list[Any] = []
    for token in tokens:
        for field in fields:
            clauses.append(f"{field} LIKE ?")
            params.append(f"%{token}%")
    return f"({' OR '.join(clauses)})", params


def _keyword_score(text: str, keywords: list[str]) -> int:
    score = 0
    for keyword in keywords:
        if keyword in text:
            score += 1 + text.count(keyword)
    return score


def _dimension_score(
    row: dict[str, Any],
    dims: tuple[int | None, int | None, int | None],
    *,
    use_min: bool,
) -> int:
    diffs: list[int] = []
    for key, value in zip(("w_mm", "d_mm", "h_mm"), dims):
        if value is not None and row.get(key) is not None:
            diffs.append(abs(row.get(key) - value))
    if not diffs:
        return 10**9
    return min(diffs) if use_min else sum(diffs)


def _matches_category_text(category: str, text: str) -> bool:
    for stem in CATEGORY_STEMS.get(category, (category,)):
        if stem in text:
            return True
    return False
