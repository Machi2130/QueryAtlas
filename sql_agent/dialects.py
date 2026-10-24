"""One call per action for every kind of database, so the MCP tools and the web routes behave the
same for SQL, MongoDB and Cassandra. SQL uses the existing functions unchanged."""

from __future__ import annotations

import json
from typing import Any

from . import cql, mongo
from .database import fetch_read
from .profile import ProfileService
from .scope import apply_scope
from .validation import QueryValidationError, validate_sql

SQL, MONGO, CASSANDRA = "sql", "mongo", "cassandra"

LANGUAGE = {
    SQL: "SQL: one SELECT (WITH / UNION allowed) in the database's dialect.",
    MONGO: ('MongoDB JSON, either a find {"collection": c, "filter": {...}, "projection": {...}, "sort": {...}, '
            '"limit": n} or an aggregate {"collection": c, "pipeline": [{"$match": ...}, {"$group": ...}, ...]} '
            "(read-only stages; $lookup / $unionWith on collections of this database)."),
    CASSANDRA: "CQL: one SELECT on one table (no joins); add ALLOW FILTERING when filtering on non-key columns.",
}


def kind(db_type: str) -> str:
    """'MongoDB' / 'mongodb' / 'mongo' -> mongo, 'Cassandra' -> cassandra, everything else -> sql."""
    label = str(db_type or "").lower()
    if label.startswith("mongo"):
        return MONGO
    return CASSANDRA if label == "cassandra" else SQL


def as_text(k: str, prepared: Any) -> str:
    """A prepared query as the text shown, audited and remembered."""
    return json.dumps(prepared, default=str) if k == MONGO else str(prepared)


def validate_read(k: str, query: Any, schema: dict, flavor: str) -> Any:
    if k == MONGO:
        if isinstance(query, str):
            try:
                query = json.loads(query)
            except json.JSONDecodeError as e:
                raise QueryValidationError(f"Not valid JSON: {e.msg} (line {e.lineno}, column {e.colno}).")
        return mongo.validate(query, schema)
    if k == CASSANDRA:
        return cql.validate(str(query), schema)
    return validate_sql(str(query), schema, flavor, "")


def scope_read(k: str, prepared: Any, scope: dict | None, flavor: str, schema: dict) -> tuple[Any, list[str]]:
    if k == MONGO:
        return mongo.apply_scope(prepared, scope)
    if k == CASSANDRA:
        return cql.apply_scope(prepared, scope, schema)
    return apply_scope(prepared, flavor, scope)


def parse_write(k: str, query: Any, schema: dict, allow_ddl: bool) -> tuple[Any, bool, list[str]]:
    """A MongoDB / CQL write, validated: (parsed, is_ddl, tables or collections it changes).
    SQL writes keep their own path (validation.parse_write) in execute_write."""
    if k == MONGO:
        if isinstance(query, str):
            try:
                query = json.loads(query)
            except json.JSONDecodeError as e:
                raise QueryValidationError(f"Not valid JSON: {e.msg} (line {e.lineno}, column {e.colno}).")
        parsed = mongo.parse_write(query, schema)
        return parsed, False, [parsed["collection"]]
    q, table, is_ddl = cql.parse_write(str(query), schema, allow_ddl)
    return q, is_ddl, [table] if table else []


def run_write(k: str, handle, parsed: Any, dry_run: bool, timeout_seconds: int) -> int:
    if k == MONGO:
        return mongo.run_write(handle, parsed, dry_run, timeout_seconds)
    if dry_run:
        raise QueryValidationError("Cassandra cannot roll back, so a dry run is not possible.")
    return cql.run_write(handle, parsed, timeout_seconds)


def open_profile(k: str, handle, *, flavor: str, schema: dict, foreign_keys: list, context: dict,
                 folder: str, keys: dict | None = None, refresh: bool = False) -> ProfileService:
    """What the agent knows about the database, built the same way for every kind: structure now,
    value hints lazily per table, references kept only when the data confirms them."""
    common = dict(flavor=flavor, schema=schema, business_context=context, folder=folder, refresh=refresh)
    if k == MONGO:
        return ProfileService.open(**common, foreign_keys=[], sampler=mongo.Sampler(handle),
                                   confirm=mongo.ReferenceChecker(handle), primary_keys={t: ["_id"] for t in schema})
    if k == CASSANDRA:
        keys = keys or {}
        return ProfileService.open(**common, foreign_keys=[], sampler=cql.Sampler(handle, keys),
                                   confirm=lambda _guesses: set(),   # CQL has no joins
                                   primary_keys={t: v["partition"] + v["clustering"] for t, v in keys.items()})
    return ProfileService.open(**common, foreign_keys=foreign_keys, engine=handle)


def run_read(k: str, handle, prepared: Any, max_rows: int, timeout_seconds: int) -> tuple[list, list]:
    if k == MONGO:
        return mongo.run_read(handle, prepared, max_rows, timeout_seconds)
    if k == CASSANDRA:
        return cql.run_read(handle, prepared, max_rows, timeout_seconds)
    return fetch_read(handle, prepared, max_rows, timeout_seconds)
