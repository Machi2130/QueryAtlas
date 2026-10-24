"""Matching question words to schema words (no LLM, no embeddings)."""

from __future__ import annotations

import re

_MATCH_ALIASES = {
    "unit": {"qty", "quantity"},
    "quantity": {"qty", "unit"},
    "qty": {"quantity", "unit"},
    "shipment": {"fulfillment"},
    "fulfillment": {"shipment"},
}


def normalize_word(word: str) -> str:
    """Normalize common English plurals and past-tense verbs for schema matching."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        if stem.endswith(("s", "v", "c")):
            return stem + "e"
        return stem
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def matching_words(words: set[str]) -> set[str]:
    """Add narrow domain-neutral synonyms used for common schema naming variants."""
    expanded = set(words)
    for word in words:
        expanded.update(_MATCH_ALIASES.get(word, ()))
    return expanded


def question_words(question: str) -> set[str]:
    """Content words in a normalized form: 'categories' meets 'category'."""
    return {normalize_word(w)
            for w in re.findall(r"[a-z0-9_]+", question.lower()) if len(w) > 3}


def names_anything(question: str, schema: dict, notes: list[dict] = ()) -> bool:
    """Does the question name anything in this database: a table/column word (prefix
    match, so 'customer' meets CUST_MASTER) or a word from a glossary note? Embedding
    scores could not separate 'flight bookings' from a real question on the eval fixture.
    ponytail: abbreviations no word prefixes (TRX for transaction) rely on glossary notes."""
    words = question_words(question)
    if any(words & question_words(n.get("note", "")) for n in notes):
        return True
    vocab = {p for t, cols in schema.items()
             for name in (t, *(c.split(" ")[0] for c in cols))
             for raw in re.split(r"[_\s.]+", name.lower())
             for p in (normalize_word(raw),) if len(p) >= 3}
    return any(w.startswith(v) or v.startswith(w) for w in words for v in vocab)
