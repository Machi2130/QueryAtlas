"""Plain-language answers, chart choice and friendly messages for people who never see SQL."""

from __future__ import annotations

import re
from typing import Any

from .validation import AUTO_LIMIT

# ═══════════════════════════════════════════════════════════════
#  PLAIN-LANGUAGE ANSWERS — for people who never see SQL (no LLM)
# ═══════════════════════════════════════════════════════════════

FRIENDLY_ERRORS = {
    "invalid":   "I couldn't work out a safe way to answer that. Try asking it a little differently.",
    "database":  "Something went wrong while reading the data. Try a simpler question, or ask again in a moment.",
    "llm":       "The assistant is unavailable right now. Please try again in a minute.",
    "no_llm":    "The assistant isn't set up yet. Please ask your administrator.",
}
_NO_TOTAL = re.compile(r"avg|average|mean|rate|ratio|percent|pct|%|price|min|max|median|id\b", re.IGNORECASE)


def nice_label(name: str) -> str:
    """trx_amt -> 'Trx amt', revenue -> 'Revenue'; aliases the LLM already wrote are kept."""
    if " " in name or not re.search(r"_|^[A-Z0-9_]+$|^[a-z0-9]+$", name):
        return name
    words = name.replace("_", " ").strip().lower()
    return words[:1].upper() + words[1:]


def fmt_value(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return "-" if v is None else str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v):,}"
    return str(v)


def summarize_result(columns: list, rows: list, truncated: bool = False) -> str:
    """One sentence a non-technical reader understands, built only from the returned rows."""
    if not rows:
        return "No matching data found."
    cols = [nice_label(str(c)) for c in columns]
    if len(rows) == 1:
        pairs = [f"{c}: {fmt_value(v)}" for c, v in zip(cols, rows[0])][:4]
        return " · ".join(pairs)

    def numeric(i: int) -> bool:
        vals = [r[i] for r in rows if r[i] is not None]
        return bool(vals) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)

    # An ID is a label, never the value ("Highest product id: Hair Oil (25)" was the result).
    is_id = [bool(re.search(r"(^|[_\s])id$", str(c), re.IGNORECASE)) for c in columns]
    nums = [i for i in range(len(columns)) if numeric(i) and not is_id[i]]
    labels = [i for i in range(len(columns)) if i not in nums and not is_id[i]] or \
             [i for i in range(len(columns)) if i not in nums]
    count = f"{len(rows):,}{'+' if truncated else ''} results"
    if not nums or not labels:
        return f"{count}."
    li, vi = labels[0], nums[0]
    is_time = bool(re.search(r"date|month|year|day|week|period|_dt$", str(columns[li]), re.IGNORECASE)) or \
        all(re.match(r"^\d{4}-\d{2}", str(r[li])) for r in rows)
    if is_time and len(labels) > 1:   # several series over time (month x channel): no single trend to state
        si = labels[1]
        periods, series = len({r[li] for r in rows}), len({r[si] for r in rows})
        text = f"{count} across {periods} periods and {series} {cols[si].lower()} values."
        if not _NO_TOTAL.search(str(columns[vi])):
            text += f" Total {cols[vi].lower()}: {fmt_value(sum(r[vi] or 0 for r in rows))}."
        return text
    if is_time:
        first, last = rows[0], rows[-1]
        a, b = first[vi] or 0, last[vi] or 0
        change = f" ({(b - a) / a * 100:+.0f}%)" if a else ""
        return (f"{cols[vi]} went from {fmt_value(a)} in {first[li]} to {fmt_value(b)} in {last[li]}{change} "
                f"across {count}.")
    top = max(rows, key=lambda r: r[vi] if r[vi] is not None else float("-inf"))
    text = f"{count}. Highest {cols[vi].lower()}: {top[li]} ({fmt_value(top[vi])})."
    if not _NO_TOTAL.search(str(columns[vi])):
        text += f" Total: {fmt_value(sum(r[vi] or 0 for r in rows))}."
    return text


def hit_auto_limit(sql_before_validation: str, rows: list) -> bool:
    """The validator capped an unlimited query: the user must not read '100 results' as all of them."""
    return len(rows) >= AUTO_LIMIT and not re.search(r"\bLIMIT\b|\bTOP\b", sql_before_validation, re.IGNORECASE)


def topic_suggestions(schema: dict, context: dict, limit: int = 5) -> list[str]:
    """What a non-technical user can ask about: the business analysis's table roles when
    there is one, otherwise readable table names. Busiest tables first is not known here."""
    roles = [m.get("role") for m in (context.get("tables") or {}).values() if m.get("role")]
    names = [re.sub(r"\b(tbl|mst|master|mast|log|dtl|hdr|xref)\b", "", t.lower().replace("_", " ")).strip()
             for t in schema]
    seen: list[str] = []
    for s in roles + names:
        if s and s not in seen:
            seen.append(s)
    return seen[:limit]

CLARIFY_MESSAGE = ("This question does not name anything in the connected database. "
                   "Say which data you mean (e.g. a table or a business term), or add a glossary note.")
CANT_FIND = "I couldn't find anything about that in your data."

# ═══════════════════════════════════════════════════════════════
#  CHART SUGGESTION — rules, no LLM
# ═══════════════════════════════════════════════════════════════

_DATE_NAME  = re.compile(r"date|_dt$|^dt|month|year|day|week|time|period", re.IGNORECASE)
_DATE_VALUE = re.compile(r"^\d{4}-\d{2}")


def suggest_chart_rule(columns: list, rows: list) -> dict | None:
    """The same rules the LLM prompt used to spell out, applied in code."""
    if len(rows) < 2 or not columns:
        return None
    sample = rows[:50]

    def is_num(i: int) -> bool:
        vals = [r[i] for r in sample if r[i] is not None]
        return bool(vals) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)

    def is_date(i: int) -> bool:
        vals = [r[i] for r in sample if r[i] is not None]
        return bool(_DATE_NAME.search(str(columns[i]))) or (bool(vals) and all(_DATE_VALUE.match(str(v)) for v in vals))

    nums  = [i for i in range(len(columns)) if is_num(i) and not is_date(i)]
    dates = [i for i in range(len(columns)) if is_date(i)]
    texts = [i for i in range(len(columns)) if i not in nums and i not in dates]
    if not nums:
        return None
    value = columns[nums[0]]
    if dates:
        label, kind = columns[dates[0]], "line"
    elif texts:
        label = columns[texts[0]]
        long_labels = max(len(str(r[texts[0]])) for r in sample) > 15
        kind = "doughnut" if len(rows) <= 6 else ("horizontalBar" if long_labels else "bar")
    elif len(nums) >= 2:
        label, value, kind = columns[nums[0]], columns[nums[1]], "scatter"
    else:
        return None
    return {"chart_type": kind, "label_column": label, "value_column": value,
            "x_column": label if kind in ("line", "scatter") else "", "title": f"{value} by {label}"}
