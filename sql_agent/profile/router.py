"""Which open database a question is about, decided from each database's own knowledge.

No embeddings: measured on this project, embedding scores could not even separate
'flight bookings' from real questions. Each database instead contributes the words it
is made of (table names and descriptions count double, then columns, category values
and glossary notes) and the question is scored by how many of its words each database
recognises. A clear winner is used; a close call is put to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .vocab import question_words

MIN_PART = 3


class PrefixVocabulary:
    """Words with weights, matched by prefix in either direction ('customer' meets CUST,
    'txn' meets TXN_REF) in O(len(word)) per lookup: every word is indexed under each of
    its prefixes, so no scan over the vocabulary is needed (a hash-map trie)."""

    def __init__(self):
        self._words: dict[str, int] = {}       # whole word -> weight
        self._prefixes: dict[str, int] = {}    # prefix of some word -> best weight below it

    def add(self, text: str, weight: int) -> None:
        for part in re.split(r"[^a-z0-9]+", str(text).lower()):
            if len(part) < MIN_PART:
                continue
            self._words[part] = max(weight, self._words.get(part, 0))
            for i in range(MIN_PART, len(part) + 1):
                self._prefixes[part[:i]] = max(weight, self._prefixes.get(part[:i], 0))

    def match(self, word: str) -> int:
        """Weight of the best vocabulary word that is a prefix of `word` or starts with it."""
        best = self._prefixes.get(word, 0)                    # some vocabulary word starts with `word`
        for i in range(MIN_PART, len(word) + 1):              # `word` starts with a vocabulary word
            best = max(best, self._words.get(word[:i], 0))
        return best

    def __len__(self) -> int:
        return len(self._words)


def database_vocabulary(schema: dict, profile=None, notes: list[dict] = ()) -> PrefixVocabulary:
    vocab = PrefixVocabulary()
    for table, cols in schema.items():
        vocab.add(table, 2)
        for c in cols:
            vocab.add(str(c).split(" ")[0], 1)
    if profile is not None:
        for t in profile.tables.values():
            vocab.add(t.description, 2)
            for col in t.columns.values():
                for v in col.values:
                    vocab.add(v, 1)
    for n in notes:
        vocab.add(n.get("note", ""), 1)
    return vocab


@dataclass
class RouteDecision:
    key: str | None                        # the database to use; None = ask the user
    reason: str                            # only-one | matched | no-match | ambiguous
    scores: list[tuple[str, int]] = field(default_factory=list)   # best first


class DatabaseRouter:
    def __init__(self, margin: float = 1.5):
        self._margin = margin               # winner must beat the runner-up by this factor

    @staticmethod
    def score(question: str, vocab: PrefixVocabulary) -> int:
        return sum(vocab.match(w) for w in question_words(question))

    def route(self, question: str, vocabularies: dict[str, PrefixVocabulary], active: str | None) -> RouteDecision:
        if len(vocabularies) == 1:
            return RouteDecision(next(iter(vocabularies)), "only-one")
        scores = sorted(((k, self.score(question, v)) for k, v in vocabularies.items()),
                        key=lambda kv: (-kv[1], kv[0] != active))
        (best, top), (_, second) = scores[0], scores[1]
        if top == 0:
            return RouteDecision(active, "no-match", scores)        # nothing recognised: stay where we are
        if second == 0 or top >= self._margin * second:
            return RouteDecision(best, "matched", scores)
        return RouteDecision(None, "ambiguous", scores)
