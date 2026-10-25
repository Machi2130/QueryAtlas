"""How much context the knowledge layer saves: tokens sent to the model versus the whole schema.

Token counts are estimates (about four characters per token, the usual rule for English and SQL);
the ratio is what matters and it is insensitive to the exact tokenizer.
"""

from __future__ import annotations

import json
from typing import Any

CHARS_PER_TOKEN = 4


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return max(1, len(text) // CHARS_PER_TOKEN)


def schema_tokens(schema: dict) -> int:
    """What sending the entire schema would cost, as 'table: col (type), ...' lines."""
    return estimate_tokens("\n".join(f"{t}: {', '.join(cols)}" for t, cols in schema.items()))


_totals: dict[str, dict[str, int]] = {}  # per connection: questions, tokens sent, tokens a full dump would be


def record(sid: str, sent: Any, schema: dict) -> dict[str, int]:
    """Count one question's context. Returns this question's own numbers."""
    sent_t, full_t = estimate_tokens(sent), schema_tokens(schema)
    bucket = _totals.setdefault(sid, {"questions": 0, "sent": 0, "full_schema": 0})
    bucket["questions"] += 1
    bucket["sent"] += sent_t
    bucket["full_schema"] += full_t
    return {"sent": sent_t, "full_schema": full_t, "saved_pct": _pct(sent_t, full_t)}


def summary(sid: str) -> dict:
    b = _totals.get(sid)
    if not b or not b["questions"]:
        return {"questions": 0}
    return {
        **b,
        "saved_pct": _pct(b["sent"], b["full_schema"]),
        "note": f"estimated tokens, ~{CHARS_PER_TOKEN} chars each; 'sent' = the context the model got, "
        "'full_schema' = what sending every table would cost",
    }


def forget(sid: str) -> None:
    _totals.pop(sid, None)


def _pct(sent: int, full: int) -> float:
    return round(100 * (1 - sent / full), 1) if full else 0.0
