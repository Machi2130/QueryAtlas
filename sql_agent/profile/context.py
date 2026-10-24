"""ContextAssembler: from a question to the slice of the database the model should see."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .models import ColumnKind, Relationship
from .service import COPY_WORDS, ProfileService, name_tokens
from .vocab import question_words

MAX_PROMPT_TABLES = 6
MAX_PICKED_TABLES = 24   # picker guidance only; graph paths are not capped at this value
RRF_K = 10               # rank fusion: a table near the top of either ranking stays near the top
_RELATIONAL_CUES = {"bought", "each", "per", "purchased", "sold", "with", "whose"}
_COPY_ALIASES = {
    "arch": {"archive", "archived"},
    "hist": {"historical", "history"},
    "stg": {"stage", "staging"},
}


@dataclass
class PromptContext:
    tables: dict[str, list[str]]                   # table -> column names to show, in order
    joins: list[Relationship] = field(default_factory=list)
    terminals: list[str] = field(default_factory=list)
    unconnected_terminals: list[str] = field(default_factory=list)
    preferred_table_limit: int | None = None

    @property
    def complete(self) -> bool:
        return not self.unconnected_terminals

    @property
    def preferred_limit_exceeded(self) -> bool:
        return self.preferred_table_limit is not None and len(self.tables) > self.preferred_table_limit

    def render(self, service: ProfileService, hints: bool = True) -> str:
        """hints=False under a tenant scope: sampled values, ranges and row counts come from every
        tenant's rows, so none of them may reach a prompt that is limited to one tenant."""
        lines = ["Tables (use ONLY these tables and columns):"]
        for name, cols in self.tables.items():
            t = service.table(name)
            about = t.description or t.role.value
            size = f", ~{t.row_estimate:,} rows" if hints and t.row_estimate is not None else ""
            lines.append(f"  {name}  -- {about}{size}")
            shown = []
            for c in cols:
                cp = t.columns[c]
                hint = cp.hint() if hints else ""
                shown.append(f"{c} ({cp.type})" + (f" [{hint}]" if hint else "") + (" key" if cp.kind is ColumnKind.KEY else "")
                             + (" indexed" if cp.indexed else ""))
            lines.append("    " + ", ".join(shown))
        if self.joins:
            lines.append("Joins:")
            lines += [f"  {j.describe()}" for j in self.joins]
        return "\n".join(lines)


class ContextAssembler:
    """Chooses the tables (the most relevant ones, joined through the relationship graph)
    and, per table, the columns: join keys always, the rest ranked by relevance."""

    def __init__(self, service: ProfileService,
                 rank_columns: Callable[[dict[str, list[str]], str, int], dict[str, list[str]]] | None = None,
                 max_tables: int | None = None, max_columns: int = 12):
        self._svc, self._rank = service, rank_columns
        self._max_tables, self._max_columns = max_tables, max_columns

    def _word_order(self, words: set[str], rank: dict[str, int]) -> list[str]:
        """Tables whose name, description or columns share words with the question, by greedy
        word cover: a table is worth a slot for words no earlier table covers (KYC_DOCS adds
        nothing once CUST_MASTER covers 'customer'; PRODUCTS still adds 'price'). A word
        matching a table's name counts double one matching a column. Ties: the retriever's rank."""
        hits = self._svc.word_hits(words)
        named = {t: h[0] for t, h in hits.items()}
        cols = {t: h[1] for t, h in hits.items()}
        fallback = len(rank)

        def gain(t: str, covered: set[str]) -> int:
            return 2 * len(named[t] - covered) + len(cols[t] - covered)

        order, covered, left = [], set(), set(hits)
        while left:
            best = max(left, key=lambda t: (gain(t, covered), gain(t, set()), -rank.get(t, fallback)))
            if gain(best, covered) == 0:
                break
            order.append(best)
            left.discard(best)
            covered |= named[best] | cols[best]
        return order + sorted(left, key=lambda t: (-gain(t, set()), rank.get(t, fallback)))

    def _asked_for(self, table: str, question: str) -> bool:
        """A backup/dated copy is wanted only when the question names its stamp or says 'backup', 'old', ..."""
        said = set(re.findall(r"[a-z0-9]+", question.lower()))
        return any(
            t in said or bool(_COPY_ALIASES.get(t, set()) & said)
            for t in name_tokens(table) if t.isdigit() or t in COPY_WORDS
        )

    def ranked(self, question: str, ranked_tables: list[str]) -> list[str]:
        """Every table worth considering, best first: the retriever's ranking (meaning) fused
        with the question-word ranking (names, descriptions, columns), copies hidden. Neither
        signal alone was enough on a 65-table database: word overlap picked a '*_HISTORY' table
        for 'billing history'; the retriever alone ignores a table named in the question."""
        tables, copies = self._svc.profile.tables, self._svc.copies
        retr = [t for t in dict.fromkeys(ranked_tables) if t in tables]
        rank = {t: i for i, t in enumerate(retr)}
        score: dict[str, float] = {}
        for ordering in (retr, self._word_order(question_words(question), rank)):
            for i, t in enumerate(ordering):
                score[t] = score.get(t, 0.0) + 1.0 / (RRF_K + i)
        shown = [t for t in score if t not in copies or self._asked_for(t, question)]
        return sorted(shown, key=lambda t: -score[t])

    def terminals(self, question: str, ranked_tables: list[str]) -> list[str]:
        """Choose supported candidates for each table-name concept in a complex question."""
        ranked = self.ranked(question, ranked_tables)
        hits = self._svc.word_hits(question_words(question))
        named = {t: value[0] for t, value in hits.items()}
        columns = {t: value[1] for t, value in hits.items()}
        rank = {t: i for i, t in enumerate(ranked)}
        named_concepts = set().union(*named.values()) if named else set()
        matched_concepts = named_concepts | (set().union(*columns.values()) if columns else set())

        if len(named_concepts) < 3:
            selected = ranked[:4]
            words = set(re.findall(r"[a-z0-9]+", question.lower()))
            if selected and words & _RELATIONAL_CUES:
                distances = self._svc.graph.distances_from(selected[0])
                return sorted(selected, key=lambda t: (distances.get(t, float("inf")), rank[t]))
            return selected

        roots = ranked[:min(12, len(ranked))]
        best_distances: dict[str, float] = {}
        best_score = (-1, 0)
        for root_rank, root in enumerate(roots):
            distances = self._svc.graph.distances_from(root)
            covered = sum(
                any(concept in named.get(table, set()) and table in distances for table in ranked)
                for concept in named_concepts
            )
            score = (covered, -root_rank)
            if score > best_score:
                best_distances, best_score = distances, score

        selected: list[str] = []
        for concept in sorted(named_concepts):
            candidates = [
                table for table in ranked
                if concept in named.get(table, set())
            ]
            supported = [table for table in candidates if best_distances.get(table, float("inf")) < float("inf")]
            selected.extend((supported or candidates)[:2])

        for concept in matched_concepts - named_concepts:
            candidates = [
                table for table in ranked
                if concept in columns.get(table, set())
            ]
            supported = [table for table in candidates if best_distances.get(table, float("inf")) < float("inf")]
            selected.extend((supported or candidates)[:1])

        return list(dict.fromkeys(selected)) or ranked[:1]

    def catalog_tables(self, question: str, ranked_tables: list[str], limit: int) -> list[str]:
        """The tables to describe to a model that picks: all of them when the database is small,
        else the `limit` best (a database with thousands of tables is narrowed first)."""
        best = self.ranked(question, ranked_tables)
        visible = [t for t in self._svc.profile.tables
                   if t not in self._svc.copies or self._asked_for(t, question)]
        if len(visible) <= limit:
            return best + [t for t in visible if t not in set(best)]
        return best[:limit]

    def catalog_line(self, name: str, brief: bool = False) -> str:
        """One line a model can choose from: name, what it is, its first columns. No sampling, no query.
        brief: name and description only (enough to decide whether to ask for the table)."""
        t = self._svc.profile.tables[name]
        if brief:
            return f"{name} -- {t.description or t.role.value}"
        cols = list(t.primary_key) + [c for c in t.columns if c not in t.primary_key]
        return f"{name} -- {t.description or t.role.value}: " + ", ".join(cols[:8]) + (" ..." if len(cols) > 8 else "")

    def assemble(self, question: str, ranked_tables: list[str], picked: list[str] | None = None) -> PromptContext:
        """`picked`: tables chosen by a model that read the catalog; all are kept, joined where the
        graph knows how. Otherwise the code's own choice (terminals), with the same rule: a table
        chosen is never silently dropped because no join path to it is known."""
        known = self._svc.profile.tables
        terminals = list(dict.fromkeys(t for t in picked or () if t in known)) \
            or self.terminals(question, ranked_tables)
        self._svc.learn_joins(terminals)   # a join no name suggests may be in the data
        names, _tree_joins = self._svc.graph.connect(terminals, self._max_tables)
        unconnected = [t for t in terminals if t not in names]
        names = names + [t for t in terminals if t not in names]
        if not names:
            names = [t for t in ranked_tables if t in self._svc.profile.tables][:1]
        included = set(names)
        joins = [
            relationship for relationship in self._svc.profile.relationships
            if relationship.from_table in included and relationship.to_table in included
        ]
        join_cols = {(t, c) for j in joins for t, c in ((j.from_table, j.from_column), (j.to_table, j.to_column))}
        candidates = {n: list(self._svc.table(n).columns) for n in names}
        ranked = self._rank(candidates, question, self._max_columns) if self._rank else \
            {n: cols[:self._max_columns] for n, cols in candidates.items()}
        tables = {}
        for n in names:
            keep = [c for c in candidates[n] if (n, c) in join_cols]
            tables[n] = keep + [c for c in ranked.get(n, []) if c not in keep]
        return PromptContext(tables, joins, terminals, unconnected, self._max_tables)
