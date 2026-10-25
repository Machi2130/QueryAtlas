"""Cassandra (CQL): schema, read-query validation, tenant scope and execution.

CQL has no joins, subqueries, UNION or OR, so a SELECT reads one table under its own column names,
and the tenant filter can be added as one more AND condition.
"""

from __future__ import annotations

import itertools
import sys
import re

from .database import safe_rows
from .validation import _BLOCKED_KW, QueryValidationError, is_sensitive

DEFAULT_LIMIT = 50
_NUMERIC = {"int", "bigint", "smallint", "tinyint", "varint", "float", "double", "decimal", "counter"}
_LITERAL = re.compile(r"'(?:[^']|'')*'")
_FROM = re.compile(r'\bFROM\s+("[^"]+"|\w+)(\s*\.\s*("[^"]+"|\w+))?', re.IGNORECASE)
_TAIL = re.compile(r"\b(GROUP\s+BY|ORDER\s+BY|PER\s+PARTITION\s+LIMIT|LIMIT|ALLOW\s+FILTERING)\b", re.IGNORECASE)


def _blank(text: str) -> str:
    """String literals masked, same length, so positions found in the result apply to the text."""
    return _LITERAL.sub(lambda m: "'" + "x" * (len(m.group(0)) - 2) + "'", text)


def _name(token: str) -> str:
    return token[1:-1] if token.startswith('"') else token.lower()   # unquoted names are lower-case in CQL


def schema(session, keyspace: str) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """({table: ["column (type)"]}, {table: {"partition": [...], "clustering": [...]}}), credential columns left out."""
    columns: dict[str, list[str]] = {}
    keys: dict[str, dict[str, list[str]]] = {}
    rows = session.execute("SELECT table_name, column_name, type, kind, position FROM system_schema.columns "
                           "WHERE keyspace_name = %s", [keyspace])
    for row in sorted(rows, key=lambda r: (r.table_name, r.kind != "partition_key", r.kind != "clustering", r.position)):
        if is_sensitive(row.column_name):
            continue
        columns.setdefault(row.table_name, []).append(f"{row.column_name} ({row.type})")
        if row.kind in ("partition_key", "clustering"):
            keys.setdefault(row.table_name, {"partition": [], "clustering": []})[
                "partition" if row.kind == "partition_key" else "clustering"].append(row.column_name)
    return columns, keys


def table_of(query: str) -> str:
    m = _FROM.search(_blank(query))
    if not m:
        raise QueryValidationError("Could not find the table (FROM ...).")
    if m.group(2):
        raise QueryValidationError("Use the bare table name: this connection reads one keyspace.")
    return _name(m.group(1))


def validate(query: str, schema_dict: dict) -> str:
    """One read-only SELECT on a table of this keyspace, no credential column, with a row cap."""
    q = query.strip().rstrip(";").strip()
    blanked = _blank(q)
    if ";" in blanked:
        raise QueryValidationError("One statement per call (no ';').")
    if re.search(r"--|/\*|//", blanked):   # a comment would swallow the tenant filter added after it
        raise QueryValidationError("Comments are not allowed in a query.")
    if not re.match(r"\s*SELECT\b", blanked, re.IGNORECASE):
        raise QueryValidationError("Query must be a SELECT.")
    if re.match(r"\s*SELECT\s+(DISTINCT\s+)?JSON\b", blanked, re.IGNORECASE):
        raise QueryValidationError("SELECT JSON is not allowed: name the columns.")
    m = _BLOCKED_KW.search(blanked)
    if m:
        raise QueryValidationError(f"Blocked keyword '{m.group(0).upper()}' detected.")
    for quoted, bare in re.findall(r'"([^"]+)"|\b([A-Za-z_][A-Za-z0-9_]*)\b', blanked):
        if is_sensitive(quoted or bare):
            raise QueryValidationError(f"Access to sensitive column '{quoted or bare}' is blocked.")
    table = table_of(q)
    if table not in schema_dict:   # _name already applied CQL folding; names are exact
        raise QueryValidationError(f"Table '{table}' not in schema.")
    if not re.search(r"\bLIMIT\b", blanked, re.IGNORECASE):
        allow = re.search(r"\bALLOW\s+FILTERING\b", blanked, re.IGNORECASE)
        q = (q[:allow.start()] + f"LIMIT {DEFAULT_LIMIT} " + q[allow.start():]) if allow else f"{q} LIMIT {DEFAULT_LIMIT}"
    return q


def apply_scope(query: str, scope: dict | None, schema_dict: dict) -> tuple[str, list[str]]:
    """Add 'tenant = value' to a validated SELECT on a scoped table; (query, scoped tables)."""
    if not scope:
        return query, []
    table = table_of(query)
    if table not in scope["tables"]:   # exact: a quoted "Notes" must not slip past a lower-cased map
        return query, []
    column, value = scope["column"], str(scope["value"])
    kind = next((c.split(" ", 1)[1].strip("()").lower() for c in schema_dict.get(table, [])
                 if c.split(" ")[0].lower() == column.lower()), "text")
    if kind in _NUMERIC:
        if not re.fullmatch(r"-?\d+(\.\d+)?", value):
            raise QueryValidationError(f"Tenant value {value!r} is not a number, but {column} is {kind}.")
        literal = value
    else:
        literal = "'" + value.replace("'", "''") + "'"
    condition = f'"{column}" = {literal}'
    blanked = _blank(query)
    where = re.search(r"\bWHERE\b", blanked, re.IGNORECASE)
    if where:
        query = f"{query[:where.end()]} {condition} AND{query[where.end():]}"
    else:
        tail = _TAIL.search(blanked)
        query = (f"{query[:tail.start()]}WHERE {condition} {query[tail.start():]}" if tail
                 else f"{query} WHERE {condition}")
    if not re.search(r"\bALLOW\s+FILTERING\b", _blank(query), re.IGNORECASE):
        query += " ALLOW FILTERING"   # the tenant column is usually not part of the key
    return query, [table]


_WRITE_TABLE = re.compile(r'^\s*(?:INSERT\s+INTO|UPDATE|DELETE\b.*?\bFROM)\s+("[^"]+"|\w+)(\s*\.\s*("[^"]+"|\w+))?',
                          re.IGNORECASE | re.DOTALL)
_DDL = re.compile(r"^\s*(CREATE|ALTER|DROP|TRUNCATE)\b", re.IGNORECASE)
_NOT_TABLE_DDL = re.compile(r"^\s*\w+\s+(ROLE|USER|KEYSPACE|FUNCTION|AGGREGATE)\b", re.IGNORECASE)


def parse_write(query: str, schema_dict: dict, allow_ddl: bool) -> tuple[str, str, bool]:
    """One INSERT / UPDATE / DELETE on a table of this keyspace (table DDL only when enabled);
    returns (query, table, is_ddl). No BATCH, no credential columns, UPDATE/DELETE need a WHERE."""
    q = query.strip().rstrip(";").strip()
    blanked = _blank(q)
    if len(q) > 10_000:
        raise QueryValidationError("Statement is longer than 10000 characters; split it.")
    if ";" in blanked:
        raise QueryValidationError("One statement per call (no ';').")
    if re.match(r"\s*BEGIN\b", blanked, re.IGNORECASE):
        raise QueryValidationError("BATCH is not allowed: one statement per call.")
    for quoted, bare in re.findall(r'"([^"]+)"|\b([A-Za-z_][A-Za-z0-9_]*)\b', blanked):
        if is_sensitive(quoted or bare):
            raise QueryValidationError(f"Writing sensitive column '{quoted or bare}' is not allowed.")
    if _DDL.match(blanked):
        if not allow_ddl:
            raise QueryValidationError("DDL is disabled by an administrator in QueryAtlas Settings.")
        if _NOT_TABLE_DDL.match(blanked):
            raise QueryValidationError("Only table DDL is allowed (no roles, users, keyspaces or functions).")
        m = re.search(r'\b(?:TABLE|INDEX)\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?("[^"]+"|\w+)', blanked, re.IGNORECASE)
        return q, _name(m.group(1)) if m else "", True
    m = _WRITE_TABLE.match(blanked)
    if not m:
        raise QueryValidationError("execute_write accepts INSERT, UPDATE, DELETE"
                                   + (", CREATE, ALTER, DROP, TRUNCATE" if allow_ddl else "") + " only.")
    if m.group(2):
        raise QueryValidationError("Use the bare table name: this connection writes one keyspace.")
    table = _name(m.group(1))
    if table not in schema_dict:   # _name already applied CQL folding; names are exact
        raise QueryValidationError(f"Table '{table}' not in schema.")
    if re.match(r"\s*(UPDATE|DELETE)\b", blanked, re.IGNORECASE) and not re.search(r"\bWHERE\b", blanked, re.IGNORECASE):
        raise QueryValidationError("UPDATE/DELETE needs a WHERE clause.")
    return q, table, False


def run_write(session, query: str, timeout_seconds: int) -> int:
    """Cassandra does not report affected rows: -1 means 'applied'."""
    session.execute(query, timeout=timeout_seconds)
    return -1


class Sampler:
    """ProfileService sampler for one keyspace: one bounded read per table, the first time a
    question touches it. keys: {table: {"partition": [...], "clustering": [...]}} from schema()."""

    def __init__(self, session, keys: dict, sample: int = 2000, timeout_seconds: int = 30):
        self._session, self._keys, self._sample, self._timeout = session, keys, sample, timeout_seconds

    def profile(self, table, key_columns: set[str]) -> None:
        from .profile.sampler import fill_column
        table.sampled = True
        k = self._keys.get(table.name, {})
        table.primary_key = table.primary_key or k.get("partition", []) + k.get("clustering", [])
        for name in table.primary_key[:1]:
            if name in table.columns:
                table.columns[name].indexed = True
        quoted = '"' + table.name.replace('"', '""') + '"'   # a name from the catalog, quoted as an identifier
        try:
            rows = list(itertools.islice(self._session.execute(f"SELECT * FROM {quoted} LIMIT {self._sample}",
                                                               timeout=self._timeout), self._sample))
        except Exception as e:
            print(f"[profile] sample of {table.name} failed: {e}", file=sys.stderr)
            return
        keys = key_columns | set(table.primary_key)
        for name, col in table.columns.items():
            fill_column(col, name, [getattr(r, name) for r in rows if getattr(r, name, None) is not None], keys)


def run_read(session, query: str, max_rows: int, timeout_seconds: int) -> tuple[list, list]:
    """Pages lazily and stops at max_rows: never the whole table. Credential columns dropped by name
    (CQL cannot rename a column, so the name is reliable)."""
    data = list(itertools.islice(session.execute(query, timeout=timeout_seconds), max_rows))
    if not data:
        return [], []
    keep = [i for i, c in enumerate(data[0]._fields) if not is_sensitive(c)]
    cols = [data[0]._fields[i] for i in keep]
    return cols, safe_rows(cols, [[row[i] for i in keep] for row in data])
