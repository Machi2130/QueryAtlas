"""Candidate joins from column names, for databases without declared foreign keys.

Deliberately broad: these are only guesses. ProfileService keeps a guess only after the
data confirms it (values of the column found in the target key), so no naming rule here
has to be right for any particular database. Indexed by name word, so a database with
thousands of tables costs one lookup per column, not a scan of every table."""

from __future__ import annotations

import re


_TEXT_TYPES = ("char", "text", "guid", "uniqueidentifier", "string")
_KEYLIKE = re.compile(r"(id|ref|reference|no|num|number|code|key|guid|email|mail|user|acct|account|to|from)$")


def key_like_text_columns(columns: dict, all_text: bool = False) -> list[str]:
    """Text columns whose name says identifier/reference/number/code/email/recipient, ordered first;
    when all_text=True, remaining text columns follow so non-standard names can still be checked."""
    text_cols = [c for c, cp in columns.items() if any(w in cp.type.lower() for w in _TEXT_TYPES)]
    primary = [c for c in text_cols if _KEYLIKE.search(c.lower())]
    return primary + ([c for c in text_cols if c not in primary] if all_text else [])


def _words(name: str) -> list[str]:
    """Letters-only words of 3+ letters, plural 's' dropped: CE_EMAIL_DETAILS07032022 -> email, detail."""
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z]+", name.lower()) if len(w) >= 3]


def _same(a: str, b: str) -> bool:
    return a == b or a.startswith(b) or b.startswith(a)   # cust ~ customer


def infer_joins(schema_dict: dict) -> list:
    """x_id -> a table named by x: all of x's words are in the table's name (member_id -> MEMBER_DETAILS),
    or all of the table's words are in x (order_item_id -> ORDER), joined to its id column or a
    same-named column. Original column casing kept."""
    table_words = {t: _words(t) for t in schema_dict}
    by_word: dict[str, set[str]] = {}
    buckets: dict[str, set[str]] = {}
    for t, words in table_words.items():
        for w in words:
            by_word.setdefault(w, set()).add(t)
            buckets.setdefault(w[:3], set()).add(w)

    def tables_named(p: str) -> set[str]:
        return {t for w in buckets.get(p[:3], ()) if _same(p, w) for t in by_word[w]}

    columns: dict[str, dict[str, str]] = {}   # table -> lower name -> real name, built on first use

    def column(table: str, lower: str) -> str | None:
        if table not in columns:
            columns[table] = {c.split(" ")[0].lower(): c.split(" ")[0] for c in schema_dict[table]}
        return columns[table].get(lower)

    joins = []
    for tbl, cols in schema_dict.items():
        for c in cols:
            col = c.split(" ")[0]
            low = col.lower()
            if not low.endswith("_id") or low == "_id":
                continue
            prefix = _words(low[:-3])
            if not prefix:
                continue
            per_word = [tables_named(p) for p in prefix]
            candidates = set.intersection(*per_word)                      # table name contains every word
            for t in set.union(*per_word):                                # or the whole table name is in the column
                if all(any(_same(w, p) for p in prefix) for w in table_words[t]) and table_words[t]:
                    candidates.add(t)
            for other in sorted(candidates - {tbl}):
                target = column(other, "id") or column(other, low)
                if target:
                    entry = {"from": f"{tbl}.{col}", "to": f"{other}.{target}", "type": "LEFT"}
                    if entry not in joins:
                        joins.append(entry)
    return joins
