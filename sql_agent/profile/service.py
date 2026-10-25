"""ProfileService: the one object a connection holds to know its database."""

from __future__ import annotations

import re
import sys
import threading
from functools import cached_property

from ..database import fetch_primary_keys
from .builder import ProfileBuilder, ProfileStore, schema_fingerprint
from .inference import key_like_text_columns
from .models import DatabaseProfile, Relationship, TableProfile
from .sampler import JoinChecker, ValueSampler
from .vocab import matching_words, normalize_word

MAX_LEARN_CHECKS = 40  # column pairs examined in the data per question, however many tables it names

COPY_WORDS = {
    "arch",
    "archive",
    "archived",
    "bak",
    "backup",
    "bkp",
    "copy",
    "dump",
    "hist",
    "old",
    "stg",
    "temp",
    "tmp",
}


def name_tokens(name: str) -> list[str]:
    """CE_EMAIL_TEMPLATE07032022 -> ce, email, template, 07032022."""
    return re.findall(r"[a-z]+|\d+", name.lower())


class ProfileService:
    """Owns a DatabaseProfile: reuses the stored one while the schema is unchanged, builds
    a fresh one otherwise, and profiles each table's values the first time a question
    touches it (so a 2,000-table database costs nothing until it is asked about)."""

    def __init__(
        self, profile: DatabaseProfile, sampler: ValueSampler | None = None, store: ProfileStore | None = None
    ):
        self.profile = profile
        self._sampler, self._store = sampler, store
        self._lock = threading.Lock()
        self.picks: dict = {}  # question -> (tables, missing topics) the table picker chose
        self._examined: set[frozenset] = set()  # table pairs already searched for a join in the data
        self._keys = {
            c for r in profile.relationships for c in ((r.from_table, r.from_column), (r.to_table, r.to_column))
        }

    @classmethod
    def open(
        cls,
        *,
        flavor: str,
        schema: dict,
        foreign_keys: list[dict],
        business_context: dict | None = None,
        engine=None,
        folder: str | None = None,
        refresh: bool = False,
        sampler=None,
        confirm=None,
        primary_keys: dict[str, list[str]] | None = None,
    ) -> ProfileService:
        """engine: an SQL database. Without one (MongoDB, Cassandra), sampler / confirm / primary_keys
        supply the same facts: value hints, data-confirmed joins and keys."""
        store = ProfileStore(folder) if folder else None
        profile = None if refresh or store is None else store.load(schema_fingerprint(schema))
        if profile is None:
            if engine is not None:
                # Guessed joins are kept only when the data confirms them (any schema, no rules to tune).
                with engine.connect() as conn:
                    primary_keys = fetch_primary_keys(conn, flavor, schema)
                checker = JoinChecker(engine, flavor)
                profile = ProfileBuilder(
                    flavor, schema, foreign_keys, business_context, primary_keys, confirm=checker
                ).build()
                if checker.rejected:
                    print(
                        f"[profile] {len(checker.rejected)} name-guessed joins not confirmed by data, dropped",
                        file=sys.stderr,
                    )
            else:
                profile = ProfileBuilder(
                    flavor, schema, foreign_keys, business_context, primary_keys, confirm=confirm
                ).build()
            if store:
                store.save(profile)
        return cls(profile, sampler or (ValueSampler(engine, flavor) if engine is not None else None), store)

    @property
    def graph(self):
        return self.profile.graph

    def learn_joins(self, tables: list[str]) -> bool:
        """Tables a question needs that the graph cannot connect: look for how they join in the data
        itself (CE_EMAIL_DETAILS.Member_Id holds MEMBER_RELATIONS.Relation_Reference values; no name
        says so). Text identifiers only; at most MAX_LEARN_CHECKS column pairs per call. A join found
        is kept in the profile and saved, so each pair of tables is examined once. True if the graph changed."""
        engine = getattr(self._sampler, "engine", None)
        known = [t for t in dict.fromkeys(tables) if t in self.profile.tables]
        if engine is None or len(known) < 2:
            return False
        with self._lock:
            connected, _ = self.graph.connect(known, len(known) + 3)
            bridges = [
                nb for t in known for nb in self.graph.neighbours(t) if nb in self.profile.tables and nb not in known
            ]
            targets = list(dict.fromkeys([*known, *bridges]))
            pairs = list(
                dict.fromkeys(frozenset((a, b)) for a in known if a not in connected for b in targets if b != a)
            )
            pairs = [p for p in pairs if p not in self._examined]
            if not pairs:
                return False
            checker = JoinChecker(engine, self.profile.flavor, min_values=10)
            unique: dict[tuple[str, str], bool] = {}
            candidates: list[tuple[str, str, str, str]] = []
            for pair in pairs:
                if len(candidates) >= MAX_LEARN_CHECKS:
                    break
                self._examined.add(pair)
                a, b = sorted(pair)
                for src, dst in ((a, b), (b, a)):
                    for sc in key_like_text_columns(self.profile.tables[src].columns):
                        for dc in key_like_text_columns(self.profile.tables[dst].columns):
                            if len(candidates) >= MAX_LEARN_CHECKS:
                                break
                            if (dst, dc) not in unique:
                                unique[(dst, dc)] = checker.unique(dst, dc)
                            if unique[(dst, dc)]:
                                candidates.append((src, sc, dst, dc))
            linked: set[frozenset] = set()
            found = []
            for ft, fc, tt, tc in checker(candidates):
                if frozenset((ft, tt)) not in linked:  # one join per pair of tables
                    linked.add(frozenset((ft, tt)))
                    found.append(Relationship(ft, fc, tt, tc, "learned"))
            if not found:
                return False
            self.profile.relationships.extend(found)
            self.profile.__dict__.pop("graph", None)  # rebuilt with the new joins on next use
            self._keys |= {c for r in found for c in ((r.from_table, r.from_column), (r.to_table, r.to_column))}
            if self._store:
                self._store.save(self.profile)
            print(
                f"[profile] learned {len(found)} join(s) from the data: "
                + ", ".join(f"{r.from_table}.{r.from_column} -> {r.to_table}.{r.to_column}" for r in found),
                file=sys.stderr,
            )
            return True

    @cached_property
    def copies(self) -> dict[str, str]:
        """Backup / dated copies -> the table they copy, found from structure alone: the same
        name once stamps (digits) and copy words (bkp, old, ...) are removed, a clean original
        beside them, and nearly the same columns. Hidden from table choice unless asked for."""
        groups: dict[tuple, list[tuple[str, bool]]] = {}
        for name in self.profile.tables:
            tokens = name_tokens(name)
            base = tuple(t for t in tokens if not t.isdigit() and t not in COPY_WORDS)
            if base:
                groups.setdefault(base, []).append((name, len(base) == len(tokens)))
        found: dict[str, str] = {}
        for members in groups.values():
            originals = [n for n, clean in members if clean]
            if len(members) < 2 or len(originals) != 1:
                continue
            cols = lambda n: {c.lower() for c in self.profile.tables[n].columns}  # noqa: E731
            for name, _ in members:
                a, b = cols(name), cols(originals[0])
                if name != originals[0] and len(a & b) >= 0.8 * len(a | b):
                    found[name] = originals[0]
        return found

    @cached_property
    def _word_index(self) -> tuple[dict, dict, dict]:
        """(parts of table names + descriptions -> tables, parts of column names -> tables,
        first three letters -> parts): lets a question's words find tables without scanning them all."""
        named: dict[str, set[str]] = {}
        cols: dict[str, set[str]] = {}
        for name, t in self.profile.tables.items():
            for raw in re.findall(r"[a-z0-9]+", f"{name} {t.description}".lower()):
                p = normalize_word(raw)
                if len(p) >= 3:
                    named.setdefault(p, set()).add(name)
            for c in t.columns:
                for raw in re.findall(r"[a-z0-9]+", c.lower()):
                    p = normalize_word(raw)
                    if len(p) >= 3:
                        cols.setdefault(p, set()).add(name)
        buckets: dict[str, set[str]] = {}
        for p in (*named, *cols):
            buckets.setdefault(p[:3], set()).add(p)
        return named, cols, buckets

    def word_hits(self, words: set[str]) -> dict[str, tuple[set[str], set[str]]]:
        """table -> (question words matching its name/description, words matching its columns);
        a prefix match either way, so 'customer' meets CUST_MASTER. Only tables with a hit appear."""
        named, cols, buckets = self._word_index
        hits: dict[str, tuple[set[str], set[str]]] = {}
        for w in matching_words(words):
            for p in buckets.get(w[:3], ()):
                if w.startswith(p) or p.startswith(w):
                    for t in named.get(p, ()):
                        hits.setdefault(t, (set(), set()))[0].add(w)
                    for t in cols.get(p, ()):
                        hits.setdefault(t, (set(), set()))[1].add(w)
        return hits

    def table(self, name: str) -> TableProfile:
        t = self.profile.tables[name]
        if not t.sampled and self._sampler is not None:
            with self._lock:
                if not t.sampled:
                    self._sampler.profile(t, {c for tbl, c in self._keys if tbl == name})
                    if self._store:
                        self._store.save(self.profile)
        return t
