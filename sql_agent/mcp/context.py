"""Shared helpers for the MCP handlers: identity and session resolution, the audit log, the
per-database notes and usage files, catalog keys, tenant-scope helpers and the accessors over
the per-connection state. Every handler module imports from here, never from each other."""

from __future__ import annotations

import json
import logging
import os
import re
import time

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

from .. import cache as redis_cache
from .. import cql, dialects, metrics, mongo, rbac
from ..config import settings
from ..connections import connection_key
from ..docstore import DocStore
from ..errors import ErrCode
from ..knowledge import KnowledgeEngine
from ..llm import LLMClient, llm_available
from ..profile import ProfileService
from ..retrieval import SchemaRetriever
from ..scope import apply_scope
from ..sqlreview import SqlReviewer
from .runtime import rt

log = logging.getLogger(__name__)

QUERY_TIMEOUT = settings.query_timeout_seconds
MAX_ROWS = settings.max_rows  # server-side cap on rows fetched per query
OTHER_TABLES_SHOWN = 25  # get_query_context lists this many more candidate tables, one line each
ALLOW_WRITES = settings.allow_writes  # .env defaults; the settings file overrides them
ALLOW_DDL = settings.allow_ddl

# The session and schema caches answer from memory or a local Redis in well under a millisecond,
# so they are read on the event loop; the notes and usage files are read and written in threads.

_docstores: dict[str, DocStore] = {}  # keyed by session: documents are per user, not per database


def get_docstore(sid: str) -> DocStore:
    if sid not in _docstores:
        _docstores[sid] = DocStore(sid)
    return _docstores[sid]


def knowledge_of(sid: str) -> KnowledgeEngine | None:
    state = rt.store.get(sid)
    return state.knowledge if state else None


def retriever_of(sid: str) -> SchemaRetriever | None:
    state = rt.store.get(sid)
    return state.retriever if state else None


def profile_service(sid: str) -> ProfileService | None:
    state = rt.store.get(sid)
    return state.profile if state else None


def scope_of(sid: str) -> dict | None:
    state = rt.store.get(sid)
    return state.scope if state else None


def generator_choice(sid: str) -> str:
    state = rt.store.get(sid)
    return state.sql_generator if state else ""


def llm_key(sid: str) -> str:
    state = rt.store.get(sid)
    return (state.llm_key if state else "") or rt.groq_api_key


def usage_of(sid: str) -> dict[str, int]:
    state = rt.store.get(sid)
    return dict(state.usage) if state else {}


def writes_enabled() -> bool:
    return bool(rt.policy.settings().get("mcp_allow_writes", ALLOW_WRITES))


def ddl_enabled() -> bool:
    return bool(rt.policy.settings().get("mcp_allow_ddl", ALLOW_DDL))


def user_can_write(user: str | None = None, local: bool | None = None) -> bool:
    return rt.policy.can(user or rt.request_user(), rbac.WRITE, rt.is_local_call() if local is None else local)


_UNKNOWN_COLUMN_RE = re.compile(
    r"no such column|invalid column name|column .* does not exist|unknown column", re.IGNORECASE
)
_UNKNOWN_TABLE_RE = re.compile(
    r"no such table|invalid object name|relation .* does not exist|table .* doesn't exist", re.IGNORECASE
)
_DENIED_RE = re.compile(r"denied|not authori[sz]ed", re.IGNORECASE)


def diagnose_db_error(error: str, sql: str, schema: dict) -> tuple[str, dict]:
    """Turn a database error into what the caller should do next, with the real schema
    facts a repair needs (not just 'try again')."""
    if _DENIED_RE.search(error):
        return ErrCode.AUTH_FAILED, {
            "next_step": "The database login lacks permission. Do not retry or work around it; tell the user."
        }
    if _UNKNOWN_COLUMN_RE.search(error):
        tables = tables_in_sql(sql, schema)
        return ErrCode.QUERY_FAILED, {
            "next_step": "Rewrite using only these columns.",
            "columns": {t: [c.split(" ")[0] for c in schema[t]] for t in tables},
        }
    if _UNKNOWN_TABLE_RE.search(error):
        return ErrCode.QUERY_FAILED, {
            "next_step": "The table exists in the cached schema but not in the database: "
            "call connect_database with refresh=true."
        }
    return ErrCode.QUERY_FAILED, {}


def resolve_sid(args: dict) -> str:
    """Return the session_id for this call. Explicit session_id wins; over HTTP
    each client gets its own (from the transport's Mcp-Session-Id header) and is
    namespaced per authenticated user so two users never share a session or cache."""
    raw = str(args.get("session_id") or "").strip()
    if not raw:
        headers = rt.request_headers()
        http_session = headers.get("mcp-session-id") if headers is not None else None
        raw = f"http-{http_session}" if http_session else rt.user_id
    # Over HTTP every session is the caller's own, whatever id is passed: no bare id may reach a
    # session opened by someone else (identities cannot contain ':', so the prefix cannot be forged).
    who = rt.request_user()
    if who != rt.user_id or not rt.is_local_call():
        return raw if raw.startswith(f"{who}:") else f"{who}:{raw}"
    return raw


def conn_key(session_id: str, host: str, port: str, database: str) -> str:
    """Stable key for one open database connection (also used as a folder name): hashed, like the web app's."""
    return connection_key(session_id, "", host, port, database)


def active(args: dict) -> str:
    """The connection key the session is currently working on, verified via the server's RSA token."""
    sid = resolve_sid(args)
    key = rt.current.get(sid, sid)
    if rt.connections.get(sid, key) is not None:
        return rt.connections.verify(sid, key)
    return key


def get_session(sid: str) -> dict | None:
    return redis_cache.get_session(sid)


def require_session(sid: str) -> dict:
    s = get_session(sid)
    if not s:
        raise ValueError("Not connected. Call connect_database first.")
    return s


def llm_for(sid: str = "") -> LLMClient:
    key = llm_key(sid)
    if not llm_available(key):
        raise ValueError(
            "No LLM API key (LLM_API_KEY / rt.groq_api_key). Use run_sql to execute SQL you wrote "
            "yourself, or pass groq_key to connect_database."
        )
    return rt.make_llm(key)


def effective_sql_generator(sid: str = "") -> str:
    """Resolve which LLM writes SQL in MCP ('server' = 1 round-trip via query_database,
    'client' = 2 round-trips via get_query_context + run_sql)."""
    chosen = generator_choice(sid)
    if chosen in ("server", "client"):
        return chosen
    mode = rt.policy.sql_generator()
    if mode in ("server", "client"):
        return mode
    key = llm_key(sid)
    return "server" if llm_available(key) else "client"


def track_usage(sid: str, event: str, count: int = 1) -> None:
    bucket = rt.store.open(sid).usage
    bucket[event] = bucket.get(event, 0) + count


def audit(
    tool: str,
    sid: str,
    sql: str,
    ok: bool,
    rows: int | None = None,
    ms: int | None = None,
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    """Append one JSON line per query/write: who, which connection, what, outcome. A local append of
    one line; the one file write left on the event loop."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user": rt.request_user(),
        "connection": sid,
        "tool": tool,
        "sql": (sql or "")[:2000],
        "ok": ok,
        "rows": rows,
        "ms": ms,
        "error": (error or "")[:500] or None,
        **(extra or {}),
    }
    try:
        with open(rt.audit_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        log.warning(f"AUDIT WRITE FAILED ({e}): {json.dumps(entry, default=str)}")


def kb_dir(sid: str) -> str:
    """This connection's private knowledge folder (per user and database)."""
    kb = (get_session(sid) or {}).get("kb") or ("db__" + sid.split("__", 1)[1])
    return os.path.join(".chroma_data", kb)


def notes_path(sid: str) -> str:
    return os.path.join(kb_dir(sid), "notes.json")


def load_notes(sid: str) -> list[dict]:
    try:
        with open(notes_path(sid), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_notes(sid: str, notes: list[dict]) -> None:
    path = notes_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=1)


def table_keys(sid: str, tables: list[str]) -> dict:
    """Primary keys, NOT NULL columns and column comments for these tables, from the
    database's own catalog. Read per table on first use and cached: a 2,000-table
    schema costs nothing until a question touches a table."""
    state = rt.store.get(sid)
    if state is None or not state.connected:
        return {}
    if state.nosql_keys is not None:  # Cassandra: partition and clustering keys, read at connect
        return {
            t: {"partition_key": k["partition"], "clustering_key": k["clustering"]}
            for t, k in state.nosql_keys.items()
            if t in tables
        }
    cache = state.table_keys
    if dialects.kind((get_session(sid) or {}).get("db_type", "sql")) == dialects.MONGO:
        mdb = state.handle  # MongoDB: _id plus the collection's indexes, read once per collection
        for t in (t for t in tables if t not in cache):
            try:
                info = mdb[t].index_information()
            except Exception as e:
                log.warning(f"index metadata unavailable for {t}: {e}")
                info = {}
            cache[t] = {
                "primary_key": ["_id"],
                "indexes": [[f for f, _ in v["key"]] for name, v in info.items() if name != "_id_"],
                "unique": [[f for f, _ in v["key"]] for name, v in info.items() if v.get("unique")],
            }
            cache[t] = {k: v for k, v in cache[t].items() if v}
        return {t: cache[t] for t in tables if cache.get(t)}
    engine = state.handle
    missing = [t for t in tables if t not in cache]
    if missing:
        insp = sa_inspect(engine)
        for t in missing:
            try:
                cols = insp.get_columns(t)
                pk = insp.get_pk_constraint(t).get("constrained_columns") or []
            except Exception as e:
                log.warning(f"catalog metadata unavailable for {t}: {e}")
                cache[t] = {}
                continue
            entry = {
                "primary_key": pk,
                "not_null": [c["name"] for c in cols if not c.get("nullable", True) and c["name"] not in pk],
                "comments": {c["name"]: c["comment"] for c in cols if c.get("comment")},
            }
            cache[t] = {k: v for k, v in entry.items() if v}
    return {t: cache[t] for t in tables if cache.get(t)}


_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([\[\]\w.`\"]+)", re.IGNORECASE)


def tables_in_sql(sql: str, schema: dict) -> list[str]:
    """Schema table names referenced by a statement (handles [a].[b], db.schema.table, backticks)."""
    lookup = {t.lower(): t for t in schema}
    found: list[str] = []
    for raw in _TABLE_REF_RE.findall(sql):
        name = raw.replace("].[", ".").replace('"."', ".").strip('[]`"')
        parts = name.split(".")
        for cand in (name, parts[-1], ".".join(parts[-2:])):
            c = cand.strip('[]`"').lower()
            if c in lookup and lookup[c] not in found:
                found.append(lookup[c])
                break
    return found


def usage_path(sid: str) -> str:
    return os.path.join(kb_dir(sid), "usage.json")


def load_usage(sid: str) -> dict:
    try:
        with open(usage_path(sid), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"tables": {}, "statements": 0}


def save_usage(sid: str, usage: dict) -> None:
    path = usage_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(usage, f)


def bump_usage(sid: str, tables: list[str], count: int = 1) -> None:
    if not tables:
        return
    usage = load_usage(sid)
    for t in tables:
        usage["tables"][t] = usage["tables"].get(t, 0) + count
    usage["statements"] = usage.get("statements", 0) + 1
    save_usage(sid, usage)


def fetch_query_history(conn, db_type: str, limit: int) -> list[tuple[str, int]]:
    """(statement, execution_count) from the engine's own statistics - what the app
    and the team actually run. Needs VIEW SERVER STATE / pg_stat_statements etc."""
    if db_type == "SQL Server":
        q = f"""SELECT TOP ({int(limit)}) st.text, qs.execution_count
                FROM sys.dm_exec_query_stats qs
                CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st
                WHERE st.text LIKE '%FROM%' ORDER BY qs.execution_count DESC"""
    elif db_type in ("MySQL", "MariaDB"):
        q = f"""SELECT DIGEST_TEXT, COUNT_STAR FROM performance_schema.events_statements_summary_by_digest
                WHERE SCHEMA_NAME = DATABASE() AND DIGEST_TEXT LIKE 'SELECT%'
                ORDER BY COUNT_STAR DESC LIMIT {int(limit)}"""
    elif db_type == "PostgreSQL":
        q = f"""SELECT query, calls FROM pg_stat_statements
                WHERE query ILIKE 'select%' ORDER BY calls DESC LIMIT {int(limit)}"""
    else:
        return []
    return [(str(r[0]), int(r[1] or 0)) for r in conn.execute(text(q)).fetchall()]


def tenant_column_candidates(schema: dict) -> list[dict]:
    """Columns shared by most tables are how a multi-client database tells clients apart."""
    counts: dict[str, int] = {}
    for cols in schema.values():
        for c in {c.split(" ")[0] for c in cols}:
            counts[c] = counts.get(c, 0) + 1
    threshold = max(3, int(0.3 * len(schema)))
    hints = ("program", "client", "tenant", "company", "org", "merchant", "bank", "store")
    out = []
    for col, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        low = col.lower()
        if n >= threshold and (low.endswith("id") or any(h in low for h in hints)) and low != "id":
            stem = re.sub(r"_?id$", "", low)
            lookups = [
                t
                for t in schema
                if stem and stem in t.lower() and ("master" in t.lower() or t.lower().startswith(stem))
            ]
            out.append({"column": col, "in_tables": n, "lookup_tables": lookups[:3]})
    return out[:5]


def reviewer_for(sid: str, flavor: str) -> SqlReviewer:
    svc = profile_service(sid)
    return SqlReviewer(flavor, svc.profile if svc else None)


def apply_scope_to(sid: str, sql: str, flavor: str) -> tuple[str, list[str]]:
    """This connection's tenant filter, enforced by rewriting (see app.apply_scope)."""
    return apply_scope(sql, flavor, scope_of(sid))


def cache_key_flavor(sid: str, flavor: str) -> str:
    """Include active tenant scope in the query-cache key so two scopes never share cached rows."""
    sc = scope_of(sid)
    return f"{flavor}|scope:{sc['column']}={sc['value']}" if sc else flavor


def handle_of(sid: str):
    """The open connection (engine, MongoDB database or Cassandra session)."""
    return rt.store.handle(sid)


def sources_of(kind: str, prepared, schema: dict) -> list[str]:
    """The tables / collections a prepared query reads, for usage counts."""
    if kind == dialects.MONGO:
        return mongo.collections_in(prepared)
    if kind == dialects.CASSANDRA:
        return [cql.table_of(prepared)]
    return tables_in_sql(prepared, schema)


def discard_connection(sid: str) -> None:
    """Forget everything held for one connection and close its database handle (MongoDB and
    Cassandra clients keep their logged-in connections open until closed)."""
    redis_cache.delete_session(sid)
    redis_cache.clear_user_query_cache(sid)
    metrics.forget(sid)
    rt.store.discard(sid)
    _docstores.pop(sid, None)
