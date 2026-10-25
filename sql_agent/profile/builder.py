"""Builds a DatabaseProfile from what connect already read (schema + foreign keys),
and stores it per database so a reconnect reuses it."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable

from .inference import infer_joins
from .models import ColumnProfile, DatabaseProfile, Relationship, TableProfile, TableRole

PROFILE_FORMAT = 3  # bump when the profile gains fields: stored profiles are then rebuilt


def schema_fingerprint(schema: dict) -> str:
    """Changes whenever a table or column is added, removed or retyped (or the format changes)."""
    canon = json.dumps({"v": PROFILE_FORMAT, "t": {t: sorted(cols) for t, cols in sorted(schema.items())}})
    return hashlib.sha1(canon.encode()).hexdigest()[:16]


class ProfileBuilder:
    """Structure only, no queries: tables, columns, relationships and each table's role.
    Declared foreign keys come first; name-based inference fills in for databases that
    never declared them, marked as guessed."""

    def __init__(
        self,
        flavor: str,
        schema: dict,
        foreign_keys: list[dict],
        business_context: dict | None = None,
        primary_keys: dict[str, list[str]] | None = None,
        confirm: Callable[[list[tuple[str, str, str, str]]], set] | None = None,
    ):
        """confirm([(from_table, from_column, to_table, to_column), ...]) -> the guesses the data
        confirms (declared foreign keys are never second-guessed). None = keep guesses."""
        self._flavor, self._schema = flavor, schema
        self._fks, self._context = foreign_keys, business_context or {}
        self._pks = primary_keys or {}
        self._confirm = confirm

    def build(self) -> DatabaseProfile:
        relationships = self._relationships()
        tables = {
            t: TableProfile(name=t, columns=self._columns(cols), primary_key=list(self._pks.get(t, [])))
            for t, cols in self._schema.items()
        }
        outgoing = {r.from_table for r in relationships}
        incoming = {r.to_table for r in relationships}
        roles = self._context.get("tables") if isinstance(self._context.get("tables"), dict) else {}
        for name, table in tables.items():
            table.role = (
                TableRole.HEADER
                if name in outgoing and name in incoming
                else TableRole.DETAIL
                if name in outgoing
                else TableRole.LOOKUP
                if name in incoming
                else TableRole.STANDALONE
            )
            table.description = str((roles.get(name) or {}).get("role") or "")
        return DatabaseProfile(self._flavor, schema_fingerprint(self._schema), tables, relationships)

    @staticmethod
    def _columns(cols: list[str]) -> dict[str, ColumnProfile]:
        out = {}
        for c in cols:  # "name (TYPE)" as the schema readers produce it
            name, _, rest = c.partition(" ")
            out[name] = ColumnProfile(name=name, type=rest.strip("()"))
        return out

    def _key_name_joins(self) -> list[dict]:
        """A column named exactly like another table's single-column primary key references
        that table (ACCOUNT.CUST_NO -> CUSTOMER.CUST_NO): how legacy databases without
        declared foreign keys join, whatever their naming convention."""
        owners: dict[str, list[str]] = {}
        for table, pk in self._pks.items():
            if len(pk) == 1:
                owners.setdefault(pk[0].lower(), []).append(table)
        joins = []
        for table, cols in self._schema.items():
            for c in cols:
                name = c.split(" ")[0]
                if name.lower() == "id":
                    continue  # every table's own key: two tables sharing the name 'id' say nothing about each other
                targets = [t for t in owners.get(name.lower(), []) if t != table]
                if len(targets) == 1:  # several tables share that key name: ambiguous, guess nothing
                    joins.append({"from": f"{table}.{name}", "to": f"{targets[0]}.{self._pks[targets[0]][0]}"})
        return joins

    def _relationships(self) -> list[Relationship]:
        rels: list[Relationship] = []
        linked: set[frozenset] = set()
        declared_cols: set[tuple[str, str]] = set()
        for j in self._fks:
            (ft, fc), (tt, tc) = j["from"].split(".", 1), j["to"].split(".", 1)
            if ft != tt and ft in self._schema and tt in self._schema:
                linked.add(frozenset((ft, tt)))
                declared_cols.add((ft, fc.lower()))
                rels.append(Relationship(ft, fc, tt, tc, "fk"))
        # A column references one table: guesses are grouped per column and kept only when
        # exactly one target survives (small integer ids exist in many tables).
        candidates: list[tuple[str, str, str, str]] = []
        for j in self._key_name_joins() + infer_joins(self._schema):
            (ft, fc), (tt, tc) = j["from"].split(".", 1), j["to"].split(".", 1)
            if ft == tt or ft not in self._schema or tt not in self._schema:
                continue
            if frozenset((ft, tt)) in linked or (ft, fc.lower()) in declared_cols:
                continue  # a declared key already covers these tables / this column
            if self._pks.get(tt) and tc not in self._pks[tt]:
                continue  # a reference points at the target's key, not at any same-named column
            candidates.append((ft, fc, tt, tc))
        candidates = list(dict.fromkeys(candidates))
        # The data's verdict: a column whose values are not the target's keys was named misleadingly.
        confirmed = self._confirm(candidates) if self._confirm else set(candidates)
        guesses: dict[tuple[str, str], list[Relationship]] = {}
        for ft, fc, tt, tc in candidates:
            if (ft, fc, tt, tc) not in confirmed:
                continue
            found = guesses.setdefault((ft, fc.lower()), [])
            if all(r.to_table != tt for r in found):
                found.append(Relationship(ft, fc, tt, tc, "inferred"))
        for found in guesses.values():
            if len(found) == 1 and frozenset((found[0].from_table, found[0].to_table)) not in linked:
                linked.add(frozenset((found[0].from_table, found[0].to_table)))
                rels.append(found[0])
        return list(dict.fromkeys(rels))


class ProfileStore:
    """One JSON file in the database's knowledge folder."""

    def __init__(self, folder: str):
        self._path = os.path.join(folder, "profile.json")

    def load(self, fingerprint: str) -> DatabaseProfile | None:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return DatabaseProfile.from_dict(data) if data.get("fingerprint") == fingerprint else None

    def save(self, profile: DatabaseProfile) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f)
        os.replace(tmp, self._path)
