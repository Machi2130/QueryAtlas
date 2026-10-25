
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import os
import re
import sys
import time
import urllib.parse
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.stdio import stdio_server
from mcp import types
from sqlalchemy import inspect as sa_inspect, text
from sqlglot import exp

# MCP hosts (Claude Code, Claude Desktop) launch this from an arbitrary cwd, while the
# engine uses project-relative paths (.chroma_data, audit.log, .env): work from the repo root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
# Before the imports below: cache, scope, llm, rbac and prompts read their settings at import
# time, so a later load silently ignored SCOPE_PINS, RBAC_ADMINS, LLM_MODEL... from .env.
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from . import cache as redis_cache
from . import cql, dialects, metrics, mongo
from . import storage
from .docstore import DocStore

from . import rbac
from .connections import (ConnectionInfo, ConnectionRegistry, TargetRefused, allowed_hosts, check_target,
                          connection_label, safe_owner)
from .rbac import POLICY, PermissionDenied
from .database import (_connection_failure_message, _limit_statement_time, _make_engine, close_handle, open_handle, explain_statement, fetch_foreign_keys,
                       fetch_read, fetch_sql_schema, filter_empty_tables, safe_rows, sqlserver_odbc_driver)
from .llm import LLM_MODEL, LLMClient, llm_api_key, llm_available
from .presentation import CLARIFY_MESSAGE, suggest_chart_rule
from .profile import ContextAssembler, ProfileService, names_anything, question_words
from .prompts import (CATALOG_MAX, attempt_fix, build_sql_messages, clean_query, clean_sql, improve_sql, enrich_context_joins,
                      run_schema_agent, said_not_answerable, update_memory)
from .retrieval import SchemaRetriever
from .scope import apply_scope, make_scope, pinned_scope
from .sqlreview import SqlReviewer
from .validation import QueryValidationError, parse_write, validate_sql
from .knowledge import KnowledgeEngine
from .limits import BodyLimit, SecurityHeaders, mcp_call_allowed, record_sign_in, sign_in_blocked
from .query_planner import reset_concept_cache

__version__ = "1.0.0"

# ── MCP identity — no JWT needed for local stdio ──────────────
MCP_USER_ID  = os.getenv("MCP_USER_ID", "mcp-agent")
GROQ_API_KEY = llm_api_key()   # LLM_API_KEY or GROQ_API_KEY; only query_database needs it

# Hosted-mode auth: MCP_AUTH_TOKENS="alice:<token>,bob:<token>" gives each person
# their own token (named in the audit log); MCP_AUTH_TOKEN is a single shared one.
_TOKEN_USERS: dict[str, str] = {
    t.strip(): n.strip()
    for n, _, t in (p.partition(":") for p in os.getenv("MCP_AUTH_TOKENS", "").split(",") if ":" in p)
}
if os.getenv("MCP_AUTH_TOKEN"):
    _TOKEN_USERS.setdefault(os.getenv("MCP_AUTH_TOKEN", ""), "default")

# Process-level caches keyed by CONNECTION key (session + host + port + database),
# so several databases stay open side by side and switching never reconnects.
# session_id defaults to MCP_USER_ID; pass an explicit one per call for
# multi-tenant/parallel agents. _current maps session -> its active connection.
_engines:    dict[str, Any] = {}
_mongos:     dict[str, Any] = {}
_cassandras: dict[str, Any] = {}
_retrievers: dict[str, Any] = {}
_knowledge:  dict[str, Any] = {}
_docstores:  dict[str, Any] = {}   # keyed by session (documents are per user, not per DB)
_fks:        dict[str, list] = {}  # real foreign keys per connection, as join dicts
_nosql_keys: dict[str, dict] = {}  # Cassandra partition/clustering keys per connection
_scope:      dict[str, dict] = {}  # tenant filter per connection: {column, value, label, tables}
_keys:       dict[str, dict] = {}  # per connection: table -> {primary_key, not_null, comments}, read lazily
_llm_keys:   dict[str, str] = {}   # an API key passed to connect_database: memory only, never in the session store
_sql_generators: dict[str, str] = {}  # per connection: user's chosen SQL generator ('server' or 'client')
_profiles:   dict[str, ProfileService] = {}  # per connection: what we know about the database
_current:    dict[str, str] = {}
_mcp_conns   = ConnectionRegistry(lambda: POLICY.idle_lock_seconds())   # idle lock, same settings as the web app

# In-process usage counters — keyed by session_id
_usage: dict[str, dict[str, int]] = {}


def _get_docstore(sid: str) -> DocStore:
    if sid not in _docstores:
        _docstores[sid] = DocStore(sid)
    return _docstores[sid]

DB_QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT_SECONDS", 30))
MAX_ROWS     = int(os.getenv("MCP_MAX_ROWS", 5000))   # server-side cap on rows fetched per query
OTHER_TABLES_SHOWN = 25   # get_query_context lists this many more candidate tables, one line each
ALLOW_WRITES = os.getenv("MCP_ALLOW_WRITES", "").lower() in ("1", "true", "yes")
ALLOW_DDL    = os.getenv("MCP_ALLOW_DDL", "").lower() in ("1", "true", "yes")
AUDIT_LOG    = os.getenv("MCP_AUDIT_LOG") or os.path.join(PROJECT_ROOT, "audit.log")


def _writes_enabled() -> bool:
    return bool(POLICY.settings().get("mcp_allow_writes", ALLOW_WRITES))


def _ddl_enabled() -> bool:
    return bool(POLICY.settings().get("mcp_allow_ddl", ALLOW_DDL))


def _user_can_write(user: str | None = None, local: bool | None = None) -> bool:
    return POLICY.can(user or _request_user(), rbac.WRITE, _is_local_call() if local is None else local)


class QueryAtlasMCPServer(Server):
    def create_initialization_options(self, notification_options=None, experimental_capabilities=None):
        options = notification_options or NotificationOptions(tools_changed=True)
        return super().create_initialization_options(options, experimental_capabilities)


server = QueryAtlasMCPServer("queryatlas")
_mcp_sessions: dict[int, tuple[Any, str, bool]] = {}
_policy_watch_task: asyncio.Task | None = None
_policy_watch_fingerprint = ""


def _tool_policy_fingerprint() -> str:
    settings = POLICY.settings()
    relevant = {
        "roles": settings.get("roles", {}),
        "users": settings.get("users", {}),
        "default_role": settings.get("default_role"),
        "mcp_local_role": settings.get("mcp_local_role"),
        "mcp_allow_writes": settings.get("mcp_allow_writes", ALLOW_WRITES),
        "mcp_allow_ddl": settings.get("mcp_allow_ddl", ALLOW_DDL),
    }
    return json.dumps(relevant, sort_keys=True, separators=(",", ":"))


async def _watch_tool_permissions() -> None:
    global _policy_watch_fingerprint
    _policy_watch_fingerprint = _tool_policy_fingerprint()
    while True:
        await asyncio.sleep(0.5)
        current = _tool_policy_fingerprint()
        if current == _policy_watch_fingerprint:
            continue
        _policy_watch_fingerprint = current
        for key, (session, _identity, _local) in list(_mcp_sessions.items()):
            try:
                await session.send_tool_list_changed()
            except Exception as exc:
                print(f"[queryatlas] could not notify MCP client of permission change: {exc}", file=sys.stderr)
                _mcp_sessions.pop(key, None)


def _register_mcp_session() -> None:
    global _policy_watch_task
    try:
        session = server.request_context.session
    except LookupError:
        return
    _mcp_sessions[id(session)] = (session, _request_user(), _is_local_call())
    if _policy_watch_task is None or _policy_watch_task.done():
        _policy_watch_task = asyncio.create_task(_watch_tool_permissions())


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

class ErrCode:
    NOT_CONNECTED = "NOT_CONNECTED"
    QUERY_FAILED  = "QUERY_FAILED"
    AUTH_FAILED   = "AUTH_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    UNSUPPORTED_DB = "UNSUPPORTED_DB"
    INVALID_SQL   = "INVALID_SQL"
    TIMEOUT       = "TIMEOUT"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    RATE_LIMITED  = "RATE_LIMITED"
    UNKNOWN       = "UNKNOWN"


def _ok(data: Any) -> list[types.TextContent]:
    payload = {"ok": True, "data": data}
    return [types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))]


def _err(msg: str, code: str = ErrCode.UNKNOWN, details: dict | None = None) -> list[types.TextContent]:
    payload = {"ok": False, "error": {"code": code, "message": msg, **(details or {})}}
    return [types.TextContent(type="text", text=json.dumps(payload, default=str))]


_UNKNOWN_COLUMN_RE = re.compile(
    r"no such column|invalid column name|column .* does not exist|unknown column", re.IGNORECASE)
_UNKNOWN_TABLE_RE  = re.compile(
    r"no such table|invalid object name|relation .* does not exist|table .* doesn't exist", re.IGNORECASE)
_DENIED_RE         = re.compile(r"denied|not authori[sz]ed", re.IGNORECASE)


def _diagnose_db_error(error: str, sql: str, schema: dict) -> tuple[str, dict]:
    """Turn a database error into what the caller should do next, with the real schema
    facts a repair needs (not just 'try again')."""
    if _DENIED_RE.search(error):
        return ErrCode.AUTH_FAILED, {"next_step": "The database login lacks permission. Do not retry or work around it; tell the user."}
    if _UNKNOWN_COLUMN_RE.search(error):
        tables = _tables_in_sql(sql, schema)
        return ErrCode.QUERY_FAILED, {
            "next_step": "Rewrite using only these columns.",
            "columns": {t: [c.split(" ")[0] for c in schema[t]] for t in tables},
        }
    if _UNKNOWN_TABLE_RE.search(error):
        return ErrCode.QUERY_FAILED, {"next_step": "The table exists in the cached schema but not in the database: "
                                                   "call connect_database with refresh=true."}
    return ErrCode.QUERY_FAILED, {}


def _resolve_sid(args: dict) -> str:
    """Return the session_id for this call. Explicit session_id wins; over HTTP
    each client gets its own (from the transport's Mcp-Session-Id header) and is
    namespaced per authenticated user so two users never share a session or cache."""
    raw = str(args.get("session_id") or "").strip()
    if not raw:
        try:
            request = server.request_context.request
        except LookupError:
            request = None
        headers = getattr(request, "headers", None)
        http_session = headers.get("mcp-session-id") if headers is not None else None
        raw = f"http-{http_session}" if http_session else MCP_USER_ID
    # Over HTTP every session is the caller's own, whatever id is passed: no bare id may reach a
    # session opened by someone else (identities cannot contain ':', so the prefix cannot be forged).
    who = _request_user()
    if who != MCP_USER_ID or not _is_local_call():
        return raw if raw.startswith(f"{who}:") else f"{who}:{raw}"
    return raw


def _conn_key(session_id: str, host: str, port: str, database: str) -> str:
    """Stable key for one open database connection (also used as a folder name): hashed, like the web app's."""
    from .connections import connection_key
    return connection_key(session_id, "", host, port, database)


def _active(args: dict) -> str:
    """The connection key the session is currently working on, verified via the server's RSA token."""
    sid = _resolve_sid(args)
    key = _current.get(sid, sid)
    if _mcp_conns.get(sid, key) is not None:
        return _mcp_conns.verify(sid, key)
    return key


def _get_session(sid: str = MCP_USER_ID) -> dict | None:
    return redis_cache.get_session(sid)


def _require_session(sid: str = MCP_USER_ID) -> dict:
    s = _get_session(sid)
    if not s:
        raise ValueError("Not connected. Call connect_database first.")
    return s


def _groq(sid: str = "") -> LLMClient:
    key = _llm_keys.get(sid) or GROQ_API_KEY
    if not llm_available(key):
        raise ValueError("No LLM API key (LLM_API_KEY / GROQ_API_KEY). Use run_sql to execute SQL you wrote "
                         "yourself, or pass groq_key to connect_database.")
    return LLMClient(key)


def _effective_sql_generator(sid: str = "") -> str:
    """Resolve which LLM writes SQL in MCP ('server' = 1 round-trip via query_database,
    'client' = 2 round-trips via get_query_context + run_sql)."""
    chosen = _sql_generators.get(sid, "")
    if chosen in ("server", "client"):
        return chosen
    mode = POLICY.sql_generator()
    if mode in ("server", "client"):
        return mode
    key = _llm_keys.get(sid) or GROQ_API_KEY
    return "server" if llm_available(key) else "client"


def _track_usage(sid: str, event: str, count: int = 1) -> None:
    bucket = _usage.setdefault(sid, {"queries": 0, "errors": 0, "charts": 0, "connects": 0})
    bucket[event] = bucket.get(event, 0) + count


def _request_user() -> str:
    """Who is calling: the name behind the bearer token over HTTP (from self-service
    tokens in rbac.json or static MCP_AUTH_TOKENS in .env), else MCP_USER_ID."""
    try:
        request = server.request_context.request
    except LookupError:
        request = None
    headers = getattr(request, "headers", None)
    if headers is not None:
        token = headers.get("authorization", "").removeprefix("Bearer ").strip()
        return POLICY.user_for_mcp_token(token) or _TOKEN_USERS.get(token, "unknown-token")
    return MCP_USER_ID


def _audit(tool: str, sid: str, sql: str, ok: bool, rows: int | None = None,
           ms: int | None = None, error: str | None = None, extra: dict | None = None) -> None:
    """Append one JSON line per query/write: who, which connection, what, outcome."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": _request_user(), "connection": sid,
        "tool": tool, "sql": (sql or "")[:2000], "ok": ok, "rows": rows, "ms": ms,
        "error": (error or "")[:500] or None, **(extra or {}),
    }
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        print(f"[queryatlas] AUDIT WRITE FAILED ({e}): {json.dumps(entry, default=str)}", file=sys.stderr)


# ── Glossary: human notes per database ("Account_Ref is the card number") ──

def _kb_dir(sid: str) -> str:
    """This connection's private knowledge folder (per user and database)."""
    kb = (_get_session(sid) or {}).get("kb") or ("db__" + sid.split("__", 1)[1])
    return os.path.join(".chroma_data", kb)


def _notes_path(sid: str) -> str:
    return os.path.join(_kb_dir(sid), "notes.json")


def _load_notes(sid: str) -> list[dict]:
    try:
        with open(_notes_path(sid), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save_notes(sid: str, notes: list[dict]) -> None:
    path = _notes_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=1)


def _table_keys(sid: str, tables: list[str]) -> dict:
    """Primary keys, NOT NULL columns and column comments for these tables, from the
    database's own catalog. Read per table on first use and cached: a 2,000-table
    schema costs nothing until a question touches a table."""
    if sid in _nosql_keys:   # Cassandra: partition and clustering keys, read at connect
        return {t: {"partition_key": k["partition"], "clustering_key": k["clustering"]}
                for t, k in _nosql_keys[sid].items() if t in tables}
    mdb = _mongos.get(sid)
    if mdb is not None:      # MongoDB: _id plus the collection's indexes, read once per collection
        cache = _keys.setdefault(sid, {})
        for t in (t for t in tables if t not in cache):
            try:
                info = mdb[t].index_information()
            except Exception as e:
                print(f"[queryatlas] index metadata unavailable for {t}: {e}", file=sys.stderr)
                info = {}
            cache[t] = {"primary_key": ["_id"],
                        "indexes": [[f for f, _ in v["key"]] for name, v in info.items() if name != "_id_"],
                        "unique": [[f for f, _ in v["key"]] for name, v in info.items() if v.get("unique")]}
            cache[t] = {k: v for k, v in cache[t].items() if v}
        return {t: cache[t] for t in tables if cache.get(t)}
    engine = _engines.get(sid)
    if engine is None:
        return {}
    cache = _keys.setdefault(sid, {})
    missing = [t for t in tables if t not in cache]
    if missing:
        insp = sa_inspect(engine)
        for t in missing:
            try:
                cols = insp.get_columns(t)
                pk = insp.get_pk_constraint(t).get("constrained_columns") or []
            except Exception as e:
                print(f"[queryatlas] catalog metadata unavailable for {t}: {e}", file=sys.stderr)
                cache[t] = {}
                continue
            entry = {"primary_key": pk,
                     "not_null": [c["name"] for c in cols if not c.get("nullable", True) and c["name"] not in pk],
                     "comments": {c["name"]: c["comment"] for c in cols if c.get("comment")}}
            cache[t] = {k: v for k, v in entry.items() if v}
    return {t: cache[t] for t in tables if cache.get(t)}


# ── Table popularity: which tables this database's users actually query ──

_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([\[\]\w.`\"]+)", re.IGNORECASE)


def _tables_in_sql(sql: str, schema: dict) -> list[str]:
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


def _usage_path(sid: str) -> str:
    return os.path.join(_kb_dir(sid), "usage.json")


def _load_usage(sid: str) -> dict:
    try:
        with open(_usage_path(sid), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"tables": {}, "statements": 0}


def _bump_usage(sid: str, tables: list[str], count: int = 1) -> None:
    if not tables:
        return
    usage = _load_usage(sid)
    for t in tables:
        usage["tables"][t] = usage["tables"].get(t, 0) + count
    usage["statements"] = usage.get("statements", 0) + 1
    path = _usage_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(usage, f)


def _fetch_query_history(conn, db_type: str, limit: int) -> list[tuple[str, int]]:
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


# ── Tenant scope: one database, many clients (e.g. Tenant_Id) ──

def _scope_candidates(schema: dict) -> list[dict]:
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
            lookups = [t for t in schema if stem and stem in t.lower() and ("master" in t.lower() or t.lower().startswith(stem))]
            out.append({"column": col, "in_tables": n, "lookup_tables": lookups[:3]})
    return out[:5]


def _reviewer(sid: str, flavor: str) -> SqlReviewer:
    return SqlReviewer(flavor, _profiles[sid].profile if sid in _profiles else None)


def _apply_scope(sid: str, sql: str, flavor: str) -> tuple[str, list[str]]:
    """This connection's tenant filter, enforced by rewriting (see app.apply_scope)."""
    return apply_scope(sql, flavor, _scope.get(sid))


def _cache_flavor(sid: str, flavor: str) -> str:
    """Include active tenant scope in the query-cache key so two scopes never share cached rows."""
    sc = _scope.get(sid)
    return f"{flavor}|scope:{sc['column']}={sc['value']}" if sc else flavor


# ═══════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS
#  list_tools() tells the agent what it can call.
#  inputSchema is standard JSON Schema — agents use it to
#  understand what arguments to pass.
# ═══════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    _register_mcp_session()
    gen = _effective_sql_generator(_active({}))
    server_prefers_query = (gen == "server")
    query_db_desc = (
        "PREFERRED (1 round-trip): Ask a natural language question about the connected database. "
        "The server's AI generates, reviews, repairs, and executes the SQL in a single step and returns "
        "columns, rows, the generated SQL, and usage metrics. Call this directly instead of get_query_context + run_sql."
        if server_prefers_query else
        "Ask a natural language question about the connected database (server-side NL->SQL). "
        "Returns columns, rows, the generated SQL, and usage metrics. "
        "(Note: client SQL generation mode is active — prefer get_query_context + run_sql.)"
    )
    get_ctx_desc = (
        "Uses the database's knowledge graph (built on connect, no LLM) to return ONLY what a question needs: "
        "the relevant tables with their columns, the join paths between them, business notes/metrics for those "
        "tables, and similar past question->SQL pairs. (Note: server SQL generation is active — prefer calling "
        "query_database(question) for a 1-step answer unless you need to write custom SQL yourself.)"
        if server_prefers_query else
        "Start here before writing SQL. Uses the database's knowledge graph (built on connect, no LLM) to "
        "return ONLY what a question needs: the relevant tables with their columns, the join paths between "
        "them, business notes/metrics for those tables, and similar past question->SQL pairs. Far cheaper "
        "than get_schema, which lists every table. How to use the result: write ONE read-only query in the "
        "'query_language' given, using only the tables/columns returned, then call run_sql with it and the "
        "question. If a business term (e.g. 'successful', 'revenue') has no glossary note, ask the user what it "
        "means instead of guessing. 'other_tables' names what else exists: if a part of the question needs one of "
        "those, call get_schema(tables=[...]) for it (it also returns the joins among the tables you ask for); if "
        "no table fits a part, say so instead of using a loosely related one. 'low_confidence': true means "
        "nothing matched well — ask the user which data they mean rather than guessing."
    )
    tools = [
        types.Tool(
            name="connect_database",
            description=(
                "Connect to a database, or switch to one that is already open. Must be called before any query. "
                "Supports SQLite, MySQL, MariaDB, PostgreSQL, SQL Server, MongoDB, Cassandra. "
                "Several databases can be open at once: calling this for a database that is already "
                "connected just makes it the current one (instant, no reconnect); the others stay open. "
                "Pass session_id to run multiple independent agent sessions in parallel."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "db_type":    {"type": "string", "description": "One of: SQLite, MySQL, MariaDB, PostgreSQL, SQL Server, MongoDB, Cassandra"},
                    "sql_generator": {
                        "type": "string",
                        "enum": ["auto", "server", "client"],
                        "default": "",
                        "description": (
                            "Who generates SQL for this session: 'server' (QueryAtlas's AI in 1 round-trip via query_database), "
                            "'client' (the MCP client AI writes SQL via get_query_context + run_sql), or 'auto'. "
                            "If omitted, ask the user which they prefer and pass it here or call set_sql_generator."
                        ),
                    },
                    "groq_key":   {"type": "string", "description": "Optional. Groq API key for query_database (server-side NL→SQL). Omit when the calling AI writes SQL itself and uses run_sql.", "default": ""},
                    "host":       {"type": "string", "default": "localhost",
                                   "description": "Host name or IP. MongoDB Atlas: mongodb+srv://cluster0.xxxx.mongodb.net "
                                                  "(no username or password in the address)"},
                    "port":       {"type": "string", "default": ""},
                    "username":   {"type": "string", "default": ""},
                    "password":   {"type": "string", "default": ""},
                    "database":   {"type": "string", "default": ""},
                    "keyspace":   {"type": "string", "description": "For Cassandra: keyspace name", "default": ""},
                    "db_path":    {"type": "string", "description": "For SQLite: path to .db file", "default": ""},
                    "allow_writes": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Opt this connection into data edits. Requires the server and your account to allow "
                            "writes; each INSERT/UPDATE/DELETE/MERGE still requires explicit confirmation. "
                            "DDL remains disabled unless separately enabled by the server."
                        ),
                    },
                    "session_id": {"type": "string", "description": "Optional agent session token. Omit to use the default MCP_USER_ID session."},
                    "refresh":    {"type": "boolean", "default": False, "description": "Re-read the schema of an already-open database (tables or columns changed)."},
                },
                "required": ["db_type"],
            },
        ),
        types.Tool(
            name="run_sql",
            description=(
                "Execute a read-only query you wrote yourself against the connected database, in its language: "
                "SQL (SELECT/WITH in the database's dialect), MongoDB JSON (a find or an aggregate pipeline) or "
                "CQL (one SELECT). No LLM or Groq key involved: call get_query_context(question) first; its "
                "'query_language' gives the exact format. Pass the original question too so the query is "
                "remembered for next time. Use bare table/collection names exactly as returned. Writes are rejected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql":        {"type": "string", "description": "One read-only query: SQL SELECT/WITH, MongoDB JSON, or CQL SELECT"},
                    "question":   {"type": "string", "default": "", "description": "The user's question this query answers (remembered as an example)"},
                    "page":       {"type": "integer", "default": 1,  "description": "Page number (1-based)"},
                    "page_size":  {"type": "integer", "default": 50, "description": "Rows per page (max 500)"},
                    "session_id": {"type": "string", "description": "Session token from connect_database. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="query_database",
            description=query_db_desc,
            inputSchema={
                "type": "object",
                "properties": {
                    "question":   {"type": "string", "description": "Natural language question, e.g. 'top 10 customers by revenue'"},
                    "page":       {"type": "integer", "default": 1,  "description": "Page number (1-based)"},
                    "page_size":  {"type": "integer", "default": 50, "description": "Rows per page (max 500)"},
                    "session_id": {"type": "string", "description": "Session token from connect_database. Omit to use default."},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="execute_write",
            description=(
                "Developer mode only: run ONE data-changing statement (INSERT, UPDATE, DELETE, MERGE; DDL only if "
                "the server allows it) on the current connection. Requires the connection to have been opened with "
                "allow_writes=true and the server to allow writes; the user confirms every statement in a form. "
                "UPDATE/DELETE must have a WHERE clause. Prefer dry_run=true first: it runs inside a transaction, "
                "reports the affected row count and rolls back. Never use this for SELECT (use run_sql). "
                'MongoDB: one JSON operation {"collection": c, "op": "insertOne|insertMany|updateOne|updateMany|'
                'replaceOne|deleteOne|deleteMany", "filter": {...}, "update": {"$set": {...}} | "document" | '
                '"documents" | "replacement"}; update/delete/replace need a filter on real fields. '
                "Cassandra: one CQL INSERT/UPDATE/DELETE (no BATCH, no dry run: Cassandra cannot roll back)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql":        {"type": "string", "description": "One write: SQL statement, MongoDB JSON operation, or CQL statement"},
                    "dry_run":    {"type": "boolean", "default": False, "description": "Execute inside a transaction and roll back, reporting rows affected"},
                    "reason":     {"type": "string", "default": "", "description": "Why this change is being made (recorded in the audit log)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="get_query_context",
            description=get_ctx_desc,
            inputSchema={
                "type": "object",
                "properties": {
                    "question":   {"type": "string", "description": "The user's question in plain language"},
                    "max_tables": {"type": "integer", "default": 6,
                                   "description": "Preferred context size; complete join paths may return more tables"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="get_schema",
            description=(
                "Return tables and their columns for the connected database. Without 'tables' it returns EVERY "
                "table (large); prefer get_query_context(question) and use this only to look up specific tables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tables":     {"type": "array", "items": {"type": "string"}, "description": "Only these tables (exact names)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="validate_sql",
            description=(
                "Check SQL without running it: parsed in the database's dialect, read-only, every table known. "
                "Returns the exact statement run_sql would execute (row limit and tenant filter applied), or the "
                "reason it would be rejected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql":        {"type": "string"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="explain_query",
            description=(
                "Return the database execution plan for a SQL query. "
                "Use this to identify slow queries, missing indexes, or sequential scans."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql":        {"type": "string", "description": "The SQL SELECT to explain"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="suggest_chart",
            description="Given columns and rows from a query result, suggest the best chart type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "columns": {"type": "array",  "items": {"type": "string"}},
                    "rows":    {"type": "array",  "items": {"type": "array"}},
                },
                "required": ["columns", "rows"],
            },
        ),
        types.Tool(
            name="set_scope",
            description=(
                "For a database shared by several clients/programs: fix the tenant filter for this connection, "
                "e.g. column='Tenant_Id', value='7', label='Acme Retail'. Afterwards every table that has that "
                "column is filtered to that value by the server, in any query shape, so clients' data is never mixed. connect_database "
                "and get_query_context report 'scope_candidates' (columns shared by most tables); list the values "
                "with run_sql on a lookup table (e.g. TENANT_MASTER) and ask the user which one. Empty value clears."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "column":     {"type": "string", "description": "Tenant column, e.g. Tenant_Id"},
                    "value":      {"type": "string", "description": "The client's value; empty string clears the scope"},
                    "label":      {"type": "string", "default": "", "description": "Human name for the value, e.g. 'Acme Retail'"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["column", "value"],
            },
        ),
        types.Tool(
            name="set_sql_generator",
            description=(
                "Set the user's choice for who generates SQL in MCP: "
                "'server' (QueryAtlas's own LLM generates, reviews, and executes SQL in 1 round-trip via query_database) "
                "or 'client' (the MCP client LLM — Claude, Cursor, Copilot — calls get_query_context, writes SQL itself, "
                "and executes it via run_sql), or 'auto'. Ask the user which mode they want after connecting, then call this."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode":       {"type": "string", "enum": ["server", "client", "auto"],
                                   "description": "'server' (1 round-trip via query_database), 'client' (2 round-trips via get_query_context + run_sql), or 'auto'"},
                    "groq_key":   {"type": "string", "default": "", "description": "Optional LLM API key if switching to 'server' mode"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["mode"],
            },
        ),
        types.Tool(
            name="import_query_history",
            description=(
                "Learn which tables matter from the database's own execution statistics (SQL Server plan cache, "
                "MySQL performance_schema, Postgres pg_stat_statements): counts how often each table is queried by "
                "the application and the team. Improves table ranking and 'hot_tables' in get_query_context. "
                "Needs VIEW SERVER STATE (or equivalent); reports if unavailable."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit":      {"type": "integer", "default": 300, "description": "Top-N statements to read (max 2000)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="import_queries",
            description=(
                "Import existing SQL the team already uses (paste a .sql file or several statements separated by ';'). "
                "Each SELECT becomes a remembered question->SQL example (a leading '-- comment' is used as the "
                "question) and counts toward table popularity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql_text":   {"type": "string", "description": "One or more SQL statements"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql_text"],
            },
        ),
        types.Tool(
            name="add_note",
            description=(
                "Teach the current database something a human knows and the schema doesn't, e.g. "
                "'Account_Ref is the card number' on ACCOUNT_LINK, or a database-wide rule like "
                "'amounts are in MUR'. Notes are stored once per database, shown to anyone whose question touches "
                "that table, and used to pick tables (a question mentioning 'card' will pull that table in)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "note":       {"type": "string", "description": "The fact, in one or two sentences"},
                    "table":      {"type": "string", "default": "", "description": "Table the note is about (omit for a database-wide note)"},
                    "column":     {"type": "string", "default": "", "description": "Column the note is about (requires table)"},
                    "source":     {"type": "string", "enum": ["user", "documentation", "inferred"], "default": "user",
                                   "description": "user = the user stated it; documentation = from an uploaded/approved document; "
                                                  "inferred = your own guess from names or data (shown as unverified)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["note"],
            },
        ),
        types.Tool(
            name="mark_query",
            description=(
                "Record the USER's verdict on an answer, only after they confirm or correct it - never on your own "
                "judgement. correct=true makes the question->SQL pair a verified example (shown first in "
                "similar_queries). correct=false removes the pair so it is never suggested again; pass the fixed SQL "
                "as correction to store that as verified instead. run_sql alone only stores unverified examples."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question":   {"type": "string"},
                    "sql":        {"type": "string", "description": "The SQL that was run for the question"},
                    "correct":    {"type": "boolean"},
                    "correction": {"type": "string", "default": "", "description": "Corrected SQL (when correct=false)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["question", "sql", "correct"],
            },
        ),
        types.Tool(
            name="list_notes",
            description="List the glossary notes for the current database (optionally only one table's, plus database-wide notes).",
            inputSchema={
                "type": "object",
                "properties": {
                    "table":      {"type": "string", "default": ""},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="delete_note",
            description="Delete a glossary note by id (from list_notes).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id":         {"type": "string"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["id"],
            },
        ),
        types.Tool(
            name="disconnect",
            description="Clear the current database session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="describe_capabilities",
            description=(
                "Return what databases are currently connected, available tools, "
                "supported DB types, and error codes. Call this first to bootstrap "
                "an agent session without human configuration."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="upload_document",
            description=(
                "Index an unstructured document (PDF, DOCX, TXT, MD, CSV) into the document store. "
                "Call this before search_documents. The document is chunked and stored per session. "
                "Use for policies, manuals, contracts, handbooks — any text that contains rules, "
                "thresholds, or context that cannot be queried with SQL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename":   {"type": "string", "description": "Original filename including extension, e.g. 'aml_policy.pdf'"},
                    "content_b64": {"type": "string", "description": "Base64-encoded file bytes"},
                    "url":        {"type": "string", "description": "Alternatively, a URL to fetch and index (PDF or web page). Provide either content_b64 or url, not both."},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="search_documents",
            description=(
                "Search the indexed document store for content relevant to a question. "
                "Returns the top matching text chunks with their source filename. "
                "Call this to retrieve policy rules, thresholds, or guidelines before "
                "or after querying the database — then combine both results in your answer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question":   {"type": "string", "description": "The question or topic to search for"},
                    "n_results":  {"type": "integer", "default": 4, "description": "Number of chunks to return (max 10)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="list_documents",
            description="List all documents that have been uploaded to the document store for this session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
    ]
    if not (_writes_enabled() and _user_can_write()):
        tools = [tool for tool in tools if tool.name != "execute_write"]
    return tools


# ═══════════════════════════════════════════════════════════════
#  TOOL DISPATCH
#  call_tool() routes each tool name to its implementation.
# ═══════════════════════════════════════════════════════════════

# Permission each tool needs (the settings file decides which roles have it).
TOOL_PERMISSIONS = {
    "connect_database": rbac.CONNECT, "get_schema": rbac.VIEW_TABLES, "explain_query": rbac.EXPLAIN,
    "validate_sql": rbac.EXPLAIN, "execute_write": rbac.WRITE, "add_note": rbac.NOTES, "delete_note": rbac.NOTES,
    "set_scope": rbac.SCOPE, "mark_query": rbac.APPROVE, "import_queries": rbac.APPROVE,
    "import_query_history": rbac.NOTES,
}
# Tools that use the current database: they wait behind the idle lock.
_CONNECTION_TOOLS = {"run_sql", "query_database", "execute_write", "get_query_context", "get_schema", "explain_query",
                     "validate_sql", "add_note", "list_notes", "delete_note", "mark_query", "set_scope",
                     "set_sql_generator", "import_query_history", "import_queries"}


def _is_local_call() -> bool:
    """stdio: the MCP server runs on the caller's own machine (no HTTP request)."""
    try:
        return getattr(server.request_context.request, "headers", None) is None
    except LookupError:
        return True


def _check_permission(name: str) -> list[types.TextContent] | None:
    who, local = _request_user(), _is_local_call()
    try:
        POLICY.require(who, rbac.MCP, local)
        POLICY.require(who, TOOL_PERMISSIONS.get(name, rbac.ASK), local)
    except PermissionDenied as e:
        _audit("denied", MCP_USER_ID, "", ok=False, error=f"{name}: {e}")
        return _err(str(e), ErrCode.AUTH_FAILED)
    return None


async def _ask_password(info: ConnectionInfo) -> str | None:
    """Ask the person (MCP form, client -> server; the model never sees it). None if the
    client cannot show forms or the person cancels."""
    try:
        session = server.request_context.session
        result = await session.elicit(
            message=f"{info.label} has not been used for a while. Enter its password to continue.",
            requestedSchema={"type": "object", "properties": {"password": {"type": "string", "title": "Password"}},
                             "required": ["password"]})
    except Exception as e:
        print(f"[queryatlas] password form unavailable: {e}", file=sys.stderr)
        return None
    if result.action != "accept" or not (result.content or {}).get("password"):
        return None
    return str(result.content["password"])


async def _ensure_unlocked(args: dict) -> list[types.TextContent] | None:
    """Same rule as the web app: verify the connection's RSA token in RAM, and if a
    password-opened database is idle past idle_lock_minutes, ask for the password again."""
    session_id = _resolve_sid(args)
    sid = _current.get(session_id, session_id)
    client_token = str(args.get("connection_token") or "").strip()
    info = _mcp_conns.get(session_id, sid)
    if info is None and not client_token:
        return None
    try:
        sid = _mcp_conns.verify(session_id, sid, client_token)
    except PermissionError as e:
        _audit("rsa_verify", sid, "", ok=False, error=str(e))
        return _err(str(e), ErrCode.AUTH_FAILED)
    if _mcp_conns.is_locked(session_id, sid):
        password = await _ask_password(info)
        if password is None:
            _audit("unlock", sid, "", ok=False, error="no password given")
            return _err(f"{info.label} is locked after being idle. Re-enter its password when asked "
                        "(or call connect_database again).", ErrCode.AUTH_FAILED)
        loop = asyncio.get_event_loop()
        d = info.descriptor
        try:
            # Checked again: a host name can point somewhere else since the first connect.
            address = check_target(d["db_type"], d.get("host", ""), d.get("db_path", ""), _HOSTED)
            handle = await loop.run_in_executor(None, lambda: open_handle(**d, password=password, address=address))
        except TargetRefused as refused:
            _audit("unlock", sid, "", ok=False, error=str(refused))
            return _err(str(refused), ErrCode.PERMISSION_DENIED)
        except Exception:
            _audit("unlock", sid, "", ok=False, error="wrong password")
            return _err("That password didn't work; the database stays locked.", ErrCode.AUTH_FAILED)
        close_handle(handle)             # only the proof was needed (any database type)
        _audit("unlock", sid, "", ok=True)
    _mcp_conns.touch(sid)
    return None


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        denied = _check_permission(name)
        if denied:
            return denied
        if not _is_local_call():   # hosted: one identity cannot flood the server or a customer's database
            allowed, retry_after = mcp_call_allowed(_request_user())
            if not allowed:
                _audit("rate_limited", _request_user(), "", ok=False, error=name)
                return _err(f"Too many requests. Try again in {retry_after}s.", ErrCode.RATE_LIMITED,
                            {"retry_after": retry_after})
        if name in _CONNECTION_TOOLS:
            locked = await _ensure_unlocked(arguments)
            if locked:
                return locked
        if name == "connect_database":
            return await _connect(arguments)
        if name == "query_database":
            return await _query(arguments)
        if name == "run_sql":
            return await _run_sql(arguments)
        if name == "execute_write":
            return await _execute_write(arguments)
        if name == "get_query_context":
            return await _query_context(arguments)
        if name == "get_schema":
            return await _schema(arguments)
        if name == "add_note":
            return await _add_note(arguments)
        if name == "list_notes":
            return await _list_notes(arguments)
        if name == "delete_note":
            return await _delete_note(arguments)
        if name == "mark_query":
            return await _mark_query(arguments)
        if name == "set_scope":
            return await _set_scope(arguments)
        if name == "set_sql_generator":
            return await _set_sql_generator(arguments)
        if name == "import_query_history":
            return await _import_query_history(arguments)
        if name == "import_queries":
            return await _import_queries(arguments)
        if name == "validate_sql":
            return await _validate(arguments)
        if name == "explain_query":
            return await _explain(arguments)
        if name == "suggest_chart":
            return await _chart(arguments)
        if name == "disconnect":
            return await _disconnect(arguments)
        if name == "describe_capabilities":
            return await _describe_capabilities(arguments)
        if name == "upload_document":
            return await _upload_document(arguments)
        if name == "search_documents":
            return await _search_documents(arguments)
        if name == "list_documents":
            return await _list_documents(arguments)
        return _err(f"Unknown tool: {name}", ErrCode.UNKNOWN)
    except ValueError as e:
        return _err(str(e), ErrCode.NOT_CONNECTED)
    except Exception as e:
        return _err(f"Unexpected error: {e}", ErrCode.UNKNOWN)


# ═══════════════════════════════════════════════════════════════
#  TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

_CONN_FIELDS = {
    "SQLite":    (["db_path"], ["db_path"]),
    "MongoDB":   (["host", "port", "username", "password", "database"], ["host", "database"]),
    "Cassandra": (["host", "port", "username", "password", "keyspace"], ["host", "keyspace"]),
}
_SQL_CONN_FIELDS = (["host", "port", "username", "password", "database"],
                    ["host", "username", "password", "database"])
_DEFAULT_PORTS = {"MySQL": "3306", "MariaDB": "3306", "PostgreSQL": "5432",
                  "SQL Server": "1433", "MongoDB": "27017", "Cassandra": "9042"}


_HOSTED = False   # set by _serve_http: other people's requests reach this process
# Hosted: taken only from the connection form, never from tool arguments (which the model wrote).
_FORM_ONLY_FIELDS = {"password", "groq_key", "trust_cert"}


def _env_db_password(database: str, host: str) -> str:
    """Password from the environment: MCP_DB_PASSWORD_<DATABASE> (several DBs on
    different servers), falling back to the generic MCP_DB_PASSWORD. On a hosted server only for
    a host the operator listed in MCP_ALLOWED_HOSTS: customers name any public host, and the
    server's own password must never be sent to one of theirs."""
    if _HOSTED and str(host or "").strip().lower() not in allowed_hosts():
        return ""
    key = re.sub(r"[^A-Z0-9]", "_", (database or "").upper())
    return (os.getenv(f"MCP_DB_PASSWORD_{key}") if key else "") or os.getenv("MCP_DB_PASSWORD", "")


def _missing_connection_fields(db_type: str, args: dict) -> tuple[list[str], list[str]]:
    """Return (all fields for this db_type, the required ones the caller didn't supply)."""
    fields, required = _CONN_FIELDS.get(db_type, _SQL_CONN_FIELDS)
    missing = [f for f in required if not args.get(f)]
    if "password" in missing and _env_db_password(args.get("database", ""), args.get("host", "")):
        missing.remove("password")
    return fields, missing


async def _ask_connection_fields(db_type: str, fields: list[str], required: list[str],
                                 args: dict) -> dict | None:
    """Ask the user for connection details via MCP elicitation (a form in the
    client). The answers travel client -> server only; the model never sees them.
    Returns the answers, or None if the user declined/cancelled."""
    try:
        session = server.request_context.session
    except LookupError:  # called outside an MCP request (e.g. a script)
        raise ValueError(f"Missing connection fields: {', '.join(required)}")
    props: dict[str, dict] = {}
    for f in fields:
        props[f] = {"type": "string", "title": f.replace("_", " ").title()}
        if f == "host" and db_type == "MongoDB":
            props[f]["description"] = "MongoDB Atlas: mongodb+srv://cluster0.xxxx.mongodb.net (no username or password here)"
        default = args.get(f) or {"host": "localhost", "port": _DEFAULT_PORTS.get(db_type)}.get(f)
        if default:
            props[f]["default"] = str(default)
    props["sql_generator"] = {
        "type": "string",
        "title": "Who generates SQL? (server = 1 round-trip, client = MCP AI writes SQL)",
        "enum": ["auto", "server", "client"],
        "default": args.get("sql_generator") or POLICY.sql_generator() or "auto",
    }
    if _HOSTED and db_type != "SQLite":
        props["trust_cert"] = {
            "type": "boolean",
            "title": "Trust this server's certificate",
            "description": "Only for a database with its own (self-signed) certificate. The connection is still "
                           "encrypted, but the server's identity is not checked.",
            "default": False,
        }
    if _writes_enabled() and _user_can_write():
        props["allow_writes"] = {
            "type": "boolean",
            "title": "Allow data edits on this connection",
            "description": "Each data-changing statement still requires explicit confirmation.",
            "default": False,
        }
    try:
        result = await session.elicit(
            message=f"Connect to {db_type} — enter the connection details and choose which AI generates SQL.",
            requestedSchema={"type": "object", "properties": props, "required": required},
        )
    except Exception as e:  # client cannot show forms (e.g. Claude Desktop, older clients)
        raise ValueError(f"Missing connection fields: {', '.join(required)}. "
                         f"Pass them as arguments (this client cannot show a form: {str(e)[:80]}).")
    if result.action != "accept" or not result.content:
        return None
    return {k: v for k, v in result.content.items() if v not in (None, "")}


async def _ask_sql_generator(default: str = "server") -> str | None:
    """Ask the user via MCP elicitation which AI should generate SQL ('server' vs 'client').
    Returns None silently if the MCP client cannot show forms (e.g. Claude Desktop or tests),
    in which case connect_database prompts the MCP assistant to ask the user in chat."""
    try:
        session = server.request_context.session
        result = await session.elicit(
            message=(
                "Who should generate SQL queries for this database?\n"
                "• server: QueryAtlas's AI (1 round-trip via query_database)\n"
                "• client: Your MCP assistant writes SQL (2 round-trips via get_query_context + run_sql)"
            ),
            requestedSchema={
                "type": "object",
                "properties": {
                    "sql_generator": {
                        "type": "string",
                        "title": "SQL Generator",
                        "enum": ["server", "client"],
                        "default": default if default in ("server", "client") else "server",
                    }
                },
                "required": ["sql_generator"],
            },
        )
    except Exception:
        return None
    if getattr(result, "action", None) != "accept" or not getattr(result, "content", None):
        return None
    choice = str(result.content.get("sql_generator", "")).strip().lower()
    return choice if choice in ("server", "client") else None


_opening: contextvars.ContextVar[str] = contextvars.ContextVar("sql_agent_opening", default="")


async def _connect(args: dict) -> list[types.TextContent]:
    """A connection is published step by step (handle, index, knowledge, profile, session) and
    committed when it joins _mcp_conns. If any step fails first, nothing of it may stay behind,
    above all not an open, logged-in database handle."""
    token = _opening.set("")
    try:
        return await _open_connection(args)
    finally:   # any exit before the commit point, by error result or by exception
        if sid := _opening.get():
            _discard_connection(sid)
        _opening.reset(token)


async def _open_connection(args: dict) -> list[types.TextContent]:
    sid      = _resolve_sid(args)
    db_type  = args["db_type"]
    write_requested = args.get("allow_writes") is True
    if write_requested and not _writes_enabled():
        return _err("Database writes are disabled by an administrator. Ask an administrator to enable them in Settings.",
                    ErrCode.PERMISSION_DENIED)
    if write_requested and not _user_can_write():
        return _err("Your account does not have permission to enable writes. Ask an administrator to grant sql.write.",
                    ErrCode.PERMISSION_DENIED)
    if _HOSTED:
        # Secrets and the certificate choice only through the form: its answers go client -> server,
        # while a tool argument was written by the model and has passed through the LLM provider.
        args = {k: v for k, v in args.items() if k not in _FORM_ONLY_FIELDS}
    fields, missing = _missing_connection_fields(db_type, args)
    if missing:
        answers = await _ask_connection_fields(db_type, fields, missing, args)
        if answers is None:
            return _ok({"connected": False, "cancelled": True,
                        "message": "Connection cancelled by user."})
        args = {**args, **answers}
    # After the form, so a host typed there is checked too, and before any password is resolved or used.
    try:
        address = check_target(db_type, args.get("host", "localhost"), args.get("db_path", ""), _HOSTED)
    except TargetRefused as refused:
        _audit("connect_database", sid, "", ok=False, error=str(refused))
        return _err(str(refused), ErrCode.PERMISSION_DENIED)
    trust_cert = _HOSTED and args.get("trust_cert") is True
    groq_key = args.get("groq_key") or GROQ_API_KEY   # optional: only needed for query_database
    host     = args.get("host", "localhost")
    port     = args.get("port", "")
    username = args.get("username", "")
    # Fallback lets the calling agent connect without the secret ever appearing in chat.
    password = args.get("password") or _env_db_password(args.get("database", ""), host)
    database = args.get("database", "")
    db_path  = args.get("db_path", "")

    # One key per open database; everything below is stored under it, and the
    # session just points at it. Reconnecting to an open database = switch.
    session_id = sid
    sid = _conn_key(session_id, host, port, database or db_path or args.get("keyspace", ""))
    if (sid in _engines or sid in _mongos or sid in _cassandras) and _get_session(sid) and not args.get("refresh"):
        _current[session_id] = sid
        existing_session = _get_session(sid) or {}
        if "allow_writes" in args:
            existing_session["allow_writes"] = write_requested and _writes_enabled() and _user_can_write()
            redis_cache.set_session(sid, existing_session)
        info = _mcp_conns.get(session_id, sid)
        schema = redis_cache.get_schema(sid) or {}
        gen = _effective_sql_generator(sid)
        return _ok({
            "connected":   True,
            "switched":    True,
            "session_id":  session_id,
            "connection":  sid,
            "connection_token": info.rsa_token if info else "",
            "db_flavor":   db_type,
            "table_count": len(schema),
            "write_enabled": bool(existing_session.get("allow_writes")),
            "sql_generator": gen,
            "sql_generator_mode": POLICY.sql_generator(),
            "message":     "Already open - switched to it without reconnecting.",
        })

    schema_dict: dict = {}
    db_flavor = db_type
    loop = asyncio.get_event_loop()
    if not (sid in _engines or sid in _mongos or sid in _cassandras):
        _opening.set(sid)   # new: discarded by _connect unless it reaches _mcp_conns.add below

    # ── Connect and fetch schema (I/O-bound → thread) ──────────
    def _do_connect():
        # One way to open every database: database.open_handle (also used by unlock and the web app).
        handle = open_handle(db_type, host=host, port=port, username=username, password=password,
                             database=database, db_path=db_path, keyspace=args.get("keyspace", ""),
                             secure=_HOSTED, trust_cert=trust_cert, address=address)
        if db_type == "MongoDB":
            sd = mongo.sample_schema(handle, timeout_seconds=DB_QUERY_TIMEOUT)   # nested paths, no credential fields
            close_handle(_mongos.get(sid))   # a refresh replaces, never strands, the old one
            _mongos[sid] = handle
            return sd
        if db_type == "Cassandra":
            sd, _nosql_keys[sid] = cql.schema(handle, args.get("keyspace", ""))
            close_handle(_cassandras.get(sid))
            _cassandras[sid] = handle
            return sd
        with handle.connect() as conn:
            sd = fetch_sql_schema(conn, db_type)
            _fks[sid] = fetch_foreign_keys(conn, db_type, sd)
        close_handle(_engines.get(sid))
        _engines[sid] = handle
        return sd

    try:
        schema_dict = await loop.run_in_executor(None, _do_connect)
    except Exception as exc:
        message = _connection_failure_message(db_type, exc)
        _audit("connect_database", sid, "", ok=False, error=type(exc).__name__,
               extra={"db_type": db_type})
        return _err(message, ErrCode.CONNECTION_FAILED)

    # ── Filter empty tables (SQL only) ────────────────────────
    if db_type not in ("MongoDB", "Cassandra"):
        try:
            def _filter():
                with _engines[sid].connect() as conn:
                    return filter_empty_tables(conn, schema_dict, db_flavor)
            schema_dict, _ = await loop.run_in_executor(None, _filter)
        except Exception:
            pass

    reset_concept_cache()

    # ── Build schema index ─────────────────────────────────────
    retriever = await loop.run_in_executor(None, SchemaRetriever, schema_dict)
    _retrievers[sid] = retriever

    # ── AI schema analysis (only when a Groq key is available) ─
    # Without a key the calling AI (e.g. Claude Code) reads get_schema and writes SQL for run_sql.
    # Knowledge graph + schema embeddings are built locally (no LLM) per USER and
    # DATABASE: nothing learned, noted or remembered is visible to anyone else, even
    # on a team instance (identity = token name over HTTP, MCP_USER_ID locally).
    # The Groq business summary is optional.
    kb = f"{safe_owner(_request_user())}__" + sid.split("__", 1)[1]
    ke = KnowledgeEngine(user_id=kb)
    await loop.run_in_executor(None, ke.learn_schema, schema_dict, db_flavor, bool(args.get("refresh")))
    _knowledge[sid] = ke

    context: dict = {}
    db_name = database or db_path or "default"
    cached_ctx = ke.load_sia_context(db_name)
    if cached_ctx:
        context = enrich_context_joins(cached_ctx, schema_dict)
    elif llm_available(groq_key):
        client = LLMClient(groq_key)
        ke.set_groq_client(client)
        top     = retriever.retrieve("business schema transactions orders", top_k=20)
        context = await loop.run_in_executor(None, run_schema_agent, top, db_type, client)
        context = enrich_context_joins(context, schema_dict)
        ke.save_sia_context(context, db_name)
    else:
        context = enrich_context_joins({}, schema_dict)   # heuristic joins only, no LLM
    ke.learn_context(context)
    kind = dialects.kind(db_type)   # every kind: structure now, column values lazily per table
    _profiles[sid] = await loop.run_in_executor(None, lambda: dialects.open_profile(
        kind, _handle(sid, kind), flavor=db_flavor, schema=schema_dict, foreign_keys=_fks.get(sid, []),
        context=context, folder=ke.folder, keys=_nosql_keys.get(sid), refresh=bool(args.get("refresh"))))

    # ── Save session ───────────────────────────────────────────
    memory = storage.load_memory(sid)
    redis_cache.set_session(sid, {
        "db_type":      dialects.kind(db_type),   # sql / mongo / cassandra: every tool routes on this
        "db_flavor":    db_flavor,
        "context":      context,
        "memory":       memory,
        "allow_writes": write_requested and _writes_enabled() and _user_can_write(),
        "kb":           kb,
    })
    redis_cache.set_schema(sid, schema_dict)
    _keys.pop(sid, None)
    if args.get("groq_key"):
        _llm_keys[sid] = args["groq_key"]
    req_gen = str(args.get("sql_generator", "")).strip().lower()
    if req_gen in ("server", "client"):
        _sql_generators[sid] = req_gen
    elif req_gen == "auto":
        _sql_generators.pop(sid, None)
    elif sid not in _sql_generators and POLICY.sql_generator() == "auto":
        elicited = await _ask_sql_generator(default=_effective_sql_generator(sid))
        if elicited in ("server", "client"):
            _sql_generators[sid] = elicited
    pin = pinned_scope(_request_user(), schema_dict)
    if pin:   # an administrator fixed this identity's tenant: set before any query can run
        _scope[sid] = pin
    elif sid in _scope:   # refresh: a new table with the tenant column must be scoped too
        sc = _scope[sid]
        _scope[sid] = make_scope(sc["column"], sc["value"], schema_dict, sc.get("label", "")) or sc
    if args.get("refresh"):
        redis_cache.clear_user_query_cache(sid)
    _track_usage(sid, "connects")
    _current[session_id] = sid
    _opening.set("")   # committed from here on
    info = _mcp_conns.add(session_id, ConnectionInfo(
        sid, connection_label(db_type, host, database, db_path, args.get("keyspace", "")), db_type,
        descriptor={"db_type": db_type, "host": host, "port": port, "username": username, "database": database,
                    "db_path": db_path, "keyspace": args.get("keyspace", ""),
                    "secure": _HOSTED, "trust_cert": trust_cert},
        has_password=bool(password)))
    _audit("connect_database", sid, "", ok=True, extra={
        "db_type": db_type, "tls": "local" if not _HOSTED else ("trusted-certificate" if trust_cert else "verified")})

    gen = _effective_sql_generator(sid)
    choice_needed = (sid not in _sql_generators and POLICY.sql_generator() == "auto" and not req_gen)
    workflow_hint = (
        "Server SQL generation is active (1 round-trip): call query_database(question) directly to generate, "
        "review, and execute SQL in one step."
        if gen == "server" else
        "Client SQL generation is active (2 round-trips): call get_query_context(question), write the SQL yourself, "
        "and execute it with run_sql(sql, question=...)."
    )
    hints = []
    if choice_needed:
        hints.append(
            "Ask the user which AI should generate SQL for this session: "
            "(1) 'server' — QueryAtlas's AI generates, reviews, and executes SQL in 1 round-trip via query_database, or "
            "(2) 'client' — you (the MCP client AI) write SQL via get_query_context + run_sql. "
            "Then call set_sql_generator(mode='server' or 'client')."
        )
    hints.append(
        "If scope_candidates is not empty this database holds several clients: list the values "
        "(e.g. run_sql on the lookup table), ask the user which client, then call set_scope."
    )
    return _ok({
        "connected":     True,
        "session_id":    session_id,
        "connection":    sid,
        "connection_token": info.rsa_token,
        "db_flavor":     db_flavor,
        "table_count":   len(schema_dict),
        "write_enabled": bool(_get_session(sid) and _get_session(sid).get("allow_writes")),
        "tables":        list(schema_dict.keys()),
        "business_type": context.get("business_type", "unknown"),
        "foreign_keys":  len(_fks.get(sid, [])),
        "sql_generator": gen,
        "sql_generator_mode": _sql_generators.get(sid) or POLICY.sql_generator(),
        "sql_generator_choice_needed": choice_needed,
        "sql_generator_options": {
            "server": "QueryAtlas Server AI · 1 round-trip (call query_database)",
            "client": "MCP Client AI · 2 round-trips (call get_query_context + run_sql)",
        },
        "workflow_hint": workflow_hint,
        "scope_candidates": _scope_candidates(schema_dict),
        "hint": " ".join(hints),
    })


async def _query(args: dict) -> list[types.TextContent]:
    sid       = _active(args)
    if _sql_generators.get(sid) == "client" or (sid not in _sql_generators and POLICY.sql_generator() == "client"):
        return _err(
            "SQL generator mode is set to 'client' (MCP host LLM writes SQL). "
            "Call get_query_context(question) and run_sql(sql) instead, or call set_sql_generator(mode='server').",
            ErrCode.INVALID_SQL,
        )
    question  = args["question"]
    page      = max(1, int(args.get("page", 1)))
    page_size = min(500, max(1, int(args.get("page_size", 50))))

    session = _require_session(sid)
    schema  = redis_cache.get_schema(sid) or {}
    loop    = asyncio.get_event_loop()

    db_flavor = session["db_flavor"]
    kind      = dialects.kind(session.get("db_type", "sql"))
    handle    = _handle(sid, kind)
    cache_flavor = _cache_flavor(sid, db_flavor)
    context   = session.get("context", {})
    memory    = session.get("memory", {})

    # ── Check cache ────────────────────────────────────────────
    cached = redis_cache.get_cached_query(sid, cache_flavor, question)
    if cached:
        all_rows = cached.get("rows", [])
        total    = len(all_rows)
        start    = (page - 1) * page_size
        return _ok({
            "generated_sql": cached.get("generated_sql"),
            "columns":       cached.get("columns"),
            "rows":          all_rows[start:start + page_size],
            "total_rows":    total,
            "page":          page,
            "page_size":     page_size,
            "total_pages":   -(-total // page_size),
            "from_cache":    True,
            "retried":       False,
        })

    # ── Answerable at all? Verified before? (both free: no LLM call) ──
    ke     = _knowledge.get(sid)
    notes  = _load_notes(sid)
    query  = ke.verified_sql(question) if ke else None
    client = None
    schema_str = ""
    if not query:
        client = _groq(sid)
        messages, matched, schema_str = await loop.run_in_executor(None, lambda: build_sql_messages(
            question, schema, db_flavor, context=context, memory=memory,
            retriever=_retrievers.get(sid), ke=ke, notes=notes, profile=_profiles.get(sid), picker=client,
            db_type=kind, keyspace=session.get("keyspace", ""), scoped=bool(_scope.get(sid))))
        metrics.record(sid, messages, schema)

        # ── LLM call (I/O-bound → thread) ─────────────────────
        def _llm():
            return client.chat.completions.create(
                messages=messages,
                temperature=0.0,   # SQL generation: same question, same SQL
                max_completion_tokens=512,
            )

        res   = await loop.run_in_executor(None, _llm)
        if said_not_answerable(res.choices[0].message.content):
            return _err(CLARIFY_MESSAGE, ErrCode.NEEDS_CLARIFICATION)
        query = clean_query(res.choices[0].message.content, kind)
        if kind == dialects.SQL:
            query, review_notes = await loop.run_in_executor(
                None, improve_sql, query, _reviewer(sid, db_flavor), question, client, messages)
        else:
            review_notes = []
    else:
        review_notes = []

    try:
        unscoped_query = dialects.validate_read(kind, query, schema, db_flavor)
    except QueryValidationError as ve:
        return _err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)

    # ── Execute (I/O-bound → thread) ──────────────────────────
    # The tenant filter is applied to whatever runs, including the auto-fixed retry.
    columns, rows = [], []
    language = "" if kind == dialects.SQL else dialects.LANGUAGE[kind]
    for attempt in range(2):
        try:
            prepared, _ = dialects.scope_read(kind, unscoped_query, _scope.get(sid), db_flavor, schema)
            query = dialects.as_text(kind, prepared)

            columns, rows = await asyncio.wait_for(
                loop.run_in_executor(None, dialects.run_read, kind, handle, prepared, MAX_ROWS, DB_QUERY_TIMEOUT),
                timeout=DB_QUERY_TIMEOUT,
            )
            break
        except asyncio.TimeoutError:
            return _err(f"Query timed out after {DB_QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
        except QueryValidationError as ve:   # e.g. a tenant-filtered collection joined in
            return _err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)
        except Exception as e:
            if attempt == 0 and client:
                try:
                    fixed = attempt_fix(dialects.as_text(kind, unscoped_query), str(e), schema_str, db_flavor,
                                        client, language)
                    unscoped_query = dialects.validate_read(kind, clean_query(fixed, kind), schema, db_flavor)
                except Exception:
                    return _err(str(e), ErrCode.QUERY_FAILED)
            else:
                return _err(str(e), ErrCode.QUERY_FAILED)

    # ── Learn + persist (store the query without the tenant filter, so one tenant's value never
    #    leaks into examples shown for another) ──
    if ke and rows:
        ke.learn_successful_query(question, dialects.as_text(kind, unscoped_query))
    memory = update_memory(question, context, memory)
    session["memory"] = memory
    redis_cache.set_session(sid, session)
    _track_usage(sid, "queries")
    _audit("query_database", sid, query, ok=True, rows=len(rows),
           extra={"question": question[:500], "model": getattr(client, "model", LLM_MODEL), "retried": attempt > 0})
    if attempt > 0:
        _track_usage(sid, "retries")

    total      = len(rows)
    start      = (page - 1) * page_size
    paged_rows = rows[start:start + page_size]

    result = {
        "generated_sql": query,
        "columns":       columns,
        "rows":          paged_rows,
        "total_rows":    total,
        "page":          page,
        "page_size":     page_size,
        "total_pages":   -(-total // page_size),
        "from_cache":    False,
        "retried":       attempt > 0,
        "review":        review_notes,
        "usage":         dict(_usage.get(sid, {})),
    }
    redis_cache.set_cached_query(sid, cache_flavor, question, {**result, "rows": rows})
    return _ok(result)


def _handle(sid: str, kind: str):
    """The open connection of this kind (engine, MongoDB database or Cassandra session)."""
    handle = {dialects.SQL: _engines, dialects.MONGO: _mongos, dialects.CASSANDRA: _cassandras}[kind].get(sid)
    if handle is None:   # a pymongo Database refuses truth testing: compare with None
        raise ValueError("Not connected. Call connect_database first.")
    return handle


def _sources(kind: str, prepared, schema: dict) -> list[str]:
    """The tables / collections a prepared query reads, for usage counts."""
    if kind == dialects.MONGO:
        return mongo.collections_in(prepared)
    if kind == dialects.CASSANDRA:
        return [cql.table_of(prepared)]
    return _tables_in_sql(prepared, schema)


async def _run_sql(args: dict) -> list[types.TextContent]:
    """Execute caller-written read-only query in the connection's language (SQL, MongoDB JSON, CQL).
    No LLM involved."""
    sid       = _active(args)
    raw       = args.get("sql", args.get("query", ""))
    sql       = raw if isinstance(raw, dict) else str(raw).strip().rstrip(";")
    page      = max(1, int(args.get("page", 1)))
    page_size = min(500, max(1, int(args.get("page_size", 50))))

    session   = _require_session(sid)
    kind      = dialects.kind(session.get("db_type", "sql"))
    handle    = _handle(sid, kind)
    schema    = redis_cache.get_schema(sid) or {}
    db_flavor = session["db_flavor"]

    rewrites, findings = [], []
    if kind == dialects.SQL:   # speed rewrites and checks are SQL-specific
        reviewer = _reviewer(sid, db_flavor)
        sql, rewrites = reviewer.rewrite(sql)
        findings = reviewer.review(sql, str(args.get("question", "")))
    try:
        unscoped_sql = dialects.validate_read(kind, sql, schema, db_flavor)
        prepared, scoped = dialects.scope_read(kind, unscoped_sql, _scope.get(sid), db_flavor, schema)
        sql = dialects.as_text(kind, prepared)
    except QueryValidationError as ve:
        _audit("run_sql", sid, dialects.as_text(kind, sql), ok=False, error=str(ve))
        return _err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)

    loop = asyncio.get_event_loop()

    t0 = time.monotonic()
    try:
        # +1 so we can tell "exactly MAX_ROWS" from "more than MAX_ROWS".
        columns, rows = await asyncio.wait_for(
            loop.run_in_executor(None, dialects.run_read, kind, handle, prepared, MAX_ROWS + 1, DB_QUERY_TIMEOUT),
            timeout=DB_QUERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        _audit("run_sql", sid, sql, ok=False, error=f"timeout after {DB_QUERY_TIMEOUT}s")
        return _err(f"Query timed out after {DB_QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
    except Exception as e:
        _track_usage(sid, "errors")
        _audit("run_sql", sid, sql, ok=False, error=str(e))
        code, details = (_diagnose_db_error(str(e), sql, schema) if kind == dialects.SQL
                         else (ErrCode.QUERY_FAILED, {}))
        return _err(str(e), code, details)

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    _track_usage(sid, "queries")
    _audit("run_sql", sid, sql, ok=True, rows=len(rows), ms=int((time.monotonic() - t0) * 1000))
    question = str(args.get("question", "")).strip()
    ke = _knowledge.get(sid)
    if question and rows and ke:
        # Stored as unverified (and unscoped so tenant filters never leak into RAG examples).
        # mark_query (after the user confirms) promotes it to a verified example.
        ke.learn_successful_query(question, dialects.as_text(kind, unscoped_sql))
    _bump_usage(sid, _sources(kind, prepared, schema))
    warnings = []
    if truncated:
        warnings.append(f"Result truncated at {MAX_ROWS} rows; aggregate or filter instead of reading rows.")
    if scoped:
        sc = _scope[sid]
        warnings.append(f"Tenant filter {sc['column']} = {sc['value']!r} applied to: {', '.join(scoped)}.")
    if not rows:
        warnings.append("No rows matched. Report that plainly; do not invent values.")
    warnings += [f"Rewritten for speed (same rows): {r}" for r in rewrites]
    warnings += [f"Check this query: {f.message}" for f in findings]
    total = len(rows)
    start = (page - 1) * page_size
    return _ok({
        "sql":         sql,
        "columns":     columns,
        "rows":        rows[start:start + page_size],
        "total_rows":  total,
        "truncated":   truncated,
        "max_rows":    MAX_ROWS,
        "page":        page,
        "page_size":   page_size,
        "total_pages": -(-total // page_size),
        "warnings":    warnings,
    })


# Behind parse_write: server-level commands, and statements that carry credentials
# (CREATE LOGIN / IDENTIFIED BY), which would otherwise be written to audit.log.
_FORBIDDEN_WRITE_RE = re.compile(
    r"\b(xp_\w+|sp_\w+|OPENROWSET|OPENQUERY|OPENDATASOURCE|BULK|SHUTDOWN|RECONFIGURE|BACKUP|RESTORE|DBCC|"
    r"GRANT|REVOKE|DENY|EXEC|EXECUTE|LOGIN|IDENTIFIED)\b", re.IGNORECASE)
_DDL_AUTOCOMMITS = {"MySQL", "MariaDB", "SQLite"}   # DDL commits on its own: a dry run would be real


async def _confirm_write(sid: str, sql: str, dry_run: bool) -> bool:
    """A human confirms every write through an elicitation form; the model cannot
    approve its own statement. No form available (or declined) -> not confirmed."""
    try:
        session = server.request_context.session
        result = await session.elicit(
            message=(("DRY RUN (rolled back) on " if dry_run else "EXECUTE WRITE on ") + sid
                     + ":\n\n" + sql + "\n\nType YES to confirm."),   # the whole statement; parse_write caps its size
            requestedSchema={"type": "object",
                             "properties": {"confirm": {"type": "string", "title": "Type YES to run this statement"}},
                             "required": ["confirm"]},
        )
    except Exception as e:
        print(f"[queryatlas] write confirmation unavailable: {e}", file=sys.stderr)
        return False
    return result.action == "accept" and str((result.content or {}).get("confirm", "")).strip().upper() == "YES"


async def _execute_nosql_write(sid: str, kind: str, raw, dry_run: bool, reason: str) -> list[types.TextContent]:
    """MongoDB / Cassandra writes under the SQL rules: validated, never on a tenant-filtered collection,
    shown in full and confirmed by a person, permissions re-checked after the form, time-limited, audited.
    Called after the write-enabled / role / allow_writes checks in _execute_write."""
    schema = redis_cache.get_schema(sid) or {}
    try:
        parsed, is_ddl, targets = dialects.parse_write(kind, raw, schema, _ddl_enabled())
    except QueryValidationError as ve:
        return _err(str(ve), ErrCode.INVALID_SQL)
    text_form = dialects.as_text(kind, parsed)
    if kind == dialects.CASSANDRA and dry_run:
        return _err("Cassandra cannot roll back, so a dry run is not possible.", ErrCode.INVALID_SQL)
    scope = _scope.get(sid)
    if scope:
        scoped = {t.lower() for t in scope["tables"]}
        touched = sorted(t for t in targets if t.lower() in scoped)
        if touched:
            return _err(f"Writes to {', '.join(touched)} are not allowed while the tenant filter "
                        f"{scope['column']} is active.", ErrCode.PERMISSION_DENIED)

    if not await _confirm_write(sid, text_form, dry_run):
        _audit("execute_write", sid, text_form, ok=False, error="not confirmed by user",
               extra={"dry_run": dry_run, "reason": reason})
        return _ok({"executed": False, "cancelled": True, "message": "Write not confirmed by user."})

    identity, local = _request_user(), _is_local_call()
    if (not _writes_enabled() or not POLICY.can(identity, rbac.WRITE, local)
            or not (_get_session(sid) or {}).get("allow_writes")):
        return _err("Write access changed while awaiting confirmation. Reconnect or ask an administrator to review "
                    "permissions.", ErrCode.PERMISSION_DENIED)

    t0 = time.monotonic()
    try:
        affected = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, dialects.run_write, kind, _handle(sid, kind), parsed,
                                                     dry_run, DB_QUERY_TIMEOUT),
            timeout=DB_QUERY_TIMEOUT + 5)   # the database's own limit fires first
    except asyncio.TimeoutError:
        _audit("execute_write", sid, text_form, ok=False, error=f"timeout after {DB_QUERY_TIMEOUT}s",
               extra={"dry_run": dry_run, "reason": reason})
        return _err(f"Statement timed out after {DB_QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
    except QueryValidationError as ve:
        return _err(str(ve), ErrCode.INVALID_SQL)
    except Exception as e:
        _track_usage(sid, "errors")
        _audit("execute_write", sid, text_form, ok=False, error=str(e), extra={"dry_run": dry_run, "reason": reason})
        return _err(str(e), ErrCode.QUERY_FAILED)
    _audit("execute_write", sid, text_form, ok=True, rows=affected, ms=int((time.monotonic() - t0) * 1000),
           extra={"dry_run": dry_run, "reason": reason, "ddl": is_ddl})
    if not dry_run:
        redis_cache.clear_user_query_cache(sid)   # cached reads may now be stale
    return _ok({"executed": not dry_run, "dry_run": dry_run, "rows_affected": affected, "sql": text_form})


async def _execute_write(args: dict) -> list[types.TextContent]:
    """Developer mode: one INSERT/UPDATE/DELETE (DDL if allowed), confirmed by the user, audited."""
    sid     = _active(args)
    sql     = str(args.get("sql", "")).strip().rstrip(";")
    dry_run = bool(args.get("dry_run", False))
    reason  = str(args.get("reason", ""))[:500]
    session = _require_session(sid)

    if not _writes_enabled():
        return _err("Writes are disabled by an administrator in QueryAtlas Settings.", ErrCode.PERMISSION_DENIED)
    if not _user_can_write():
        return _err("Your account no longer has sql.write permission.", ErrCode.PERMISSION_DENIED)
    if not session.get("allow_writes"):
        return _err("This connection is read-only. Reconnect with allow_writes=true (developer mode) "
                    "using your own write-capable credentials.", ErrCode.PERMISSION_DENIED)
    kind = dialects.kind(session.get("db_type", "sql"))
    if kind != dialects.SQL:
        return await _execute_nosql_write(sid, kind, args.get("sql", args.get("query", "")), dry_run, reason)
    if sid not in _engines:
        return _err("Not connected. Call connect_database first.", ErrCode.NOT_CONNECTED)
    flavor = session.get("db_flavor", "")
    try:
        tree, is_ddl = parse_write(sql, flavor, _ddl_enabled())
    except QueryValidationError as ve:
        return _err(str(ve), ErrCode.INVALID_SQL)
    if _FORBIDDEN_WRITE_RE.search(sql):
        return _err("Statement contains a forbidden keyword.", ErrCode.INVALID_SQL)
    if dry_run and is_ddl and flavor in _DDL_AUTOCOMMITS:
        return _err(f"{flavor} commits DDL immediately, so a dry run would really change the database.",
                    ErrCode.INVALID_SQL)
    scope = _scope.get(sid)
    if scope:
        scoped = {t.lower() for t in scope["tables"]}
        touched = sorted({t.name for t in tree.find_all(exp.Table) if t.name.lower() in scoped})
        if touched:
            return _err(f"Writes to {', '.join(touched)} are not allowed while the tenant filter "
                        f"{scope['column']} is active.", ErrCode.PERMISSION_DENIED)

    if not await _confirm_write(sid, sql, dry_run):
        _audit("execute_write", sid, sql, ok=False, error="not confirmed by user",
               extra={"dry_run": dry_run, "reason": reason})
        return _ok({"executed": False, "cancelled": True, "message": "Write not confirmed by user."})

    identity, local = _request_user(), _is_local_call()
    current_session = _get_session(sid) or {}
    if (not _writes_enabled() or not POLICY.can(identity, rbac.WRITE, local)
            or not current_session.get("allow_writes")):
        return _err("Write access changed while awaiting confirmation. Reconnect or ask an administrator to review permissions.",
                    ErrCode.PERMISSION_DENIED)

    loop = asyncio.get_event_loop()

    def _exec() -> int:
        with _engines[sid].connect() as conn:
            trans = conn.begin()
            # The database stops the statement at the time limit: a write the caller gave up on
            # must not commit later.
            undo = _limit_statement_time(conn, DB_QUERY_TIMEOUT)
            try:
                latest_session = _get_session(sid) or {}
                if (not _writes_enabled() or not POLICY.can(identity, rbac.WRITE, local)
                        or not latest_session.get("allow_writes")):
                    trans.rollback()
                    raise PermissionError("Write permission was revoked before execution.")
                affected = conn.execute(text(sql)).rowcount
            except Exception:
                trans.rollback()
                undo()
                raise
            try:
                if dry_run:
                    trans.rollback()
                else:
                    latest_session = _get_session(sid) or {}
                    if (not _writes_enabled() or not POLICY.can(identity, rbac.WRITE, local)
                            or not latest_session.get("allow_writes")):
                        trans.rollback()
                        raise PermissionError("Write permission was revoked before commit.")
                    trans.commit()
            finally:
                undo()
            return affected

    t0 = time.monotonic()
    try:
        affected = await asyncio.wait_for(loop.run_in_executor(None, _exec), timeout=DB_QUERY_TIMEOUT)
    except asyncio.TimeoutError:
        _audit("execute_write", sid, sql, ok=False, error=f"timeout after {DB_QUERY_TIMEOUT}s",
               extra={"dry_run": dry_run, "reason": reason})
        return _err(f"Statement timed out after {DB_QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
    except PermissionError as e:
        _audit("execute_write", sid, sql, ok=False, error=str(e), extra={"dry_run": dry_run, "reason": reason})
        return _err(str(e), ErrCode.AUTH_FAILED)
    except Exception as e:
        _track_usage(sid, "errors")
        _audit("execute_write", sid, sql, ok=False, error=str(e), extra={"dry_run": dry_run, "reason": reason})
        return _err(str(e), ErrCode.QUERY_FAILED)

    _audit("execute_write", sid, sql, ok=True, rows=affected, ms=int((time.monotonic() - t0) * 1000),
           extra={"dry_run": dry_run, "reason": reason})
    if not dry_run:
        redis_cache.clear_user_query_cache(sid)   # cached SELECT results may now be stale
    return _ok({"executed": not dry_run, "dry_run": dry_run, "rows_affected": affected, "sql": sql})


async def _query_context(args: dict) -> list[types.TextContent]:
    """Knowledge-graph lookup for the client's own LLM: only the tables, joins and
    examples a question needs, so it can write SQL without reading the whole schema."""
    sid        = _active(args)
    question   = str(args.get("question", "")).strip()
    max_tables = min(12, max(1, int(args.get("max_tables", 6))))
    session    = _require_session(sid)
    ke         = _knowledge.get(sid)
    if ke is None:
        return _err("No knowledge for this connection yet - reconnect to build it.", ErrCode.NOT_CONNECTED)

    loop   = asyncio.get_event_loop()
    schema = redis_cache.get_schema(sid) or {}
    notes  = _load_notes(sid)

    # Glossary first: a human said "this table/column means X"; then the retriever and
    # whatever the knowledge engine found. The profile joins them through the graph.
    words = question_words(question)
    glossary_tables = [n["table"] for n in notes
                       if n.get("table") and any(w in n["note"].lower() for w in words)]
    glossary_hit = bool(glossary_tables)
    retriever = _retrievers.get(sid)
    ranked = list(await loop.run_in_executor(None, retriever.retrieve, question, max(max_tables, CATALOG_MAX)) or {}) \
        if retriever else []
    ctx = await loop.run_in_executor(None, ke.get_context_for_question, question, session.get("db_type", "sql"))
    candidates = list(dict.fromkeys(glossary_tables + ranked + list(ctx.get("tables_found") or [])))
    svc = _profiles.get(sid)
    joins: list = []
    values: dict = {}
    other_tables: dict = {}
    context_selection: dict = {"status": "unavailable"}
    if svc:
        asm = ContextAssembler(svc)
        assembled = await loop.run_in_executor(None, lambda: asm.assemble(question, candidates))
        names = list(assembled.tables)
        context_selection = {
            "status": "complete" if assembled.complete else "incomplete",
            "terminals": assembled.terminals,
            "unconnected_terminals": assembled.unconnected_terminals,
            "preferred_table_limit": max_tables,
            "preferred_limit_exceeded": len(names) > max_tables,
        }
        # One line per other candidate: the caller's own model can see what else exists and ask for it
        # with get_schema(tables=[...]); a table the code ranked low is no longer invisible.
        other_tables = {t: asm.catalog_line(t, brief=True)[len(t) + 4:]
                        for t in asm.catalog_tables(question, candidates, CATALOG_MAX) if t not in names}
        other_tables = dict(list(other_tables.items())[:OTHER_TABLES_SHOWN])
        joins = [{"from": f"{j.from_table}.{j.from_column}", "to": f"{j.to_table}.{j.to_column}",
                  "source": j.source, "rows": f"many {j.from_table} per {j.to_table}"} for j in assembled.joins]
        # What columns hold: status values, numeric and date ranges. Sampled from every tenant's
        # rows, so none under a tenant scope.
        for t in ([] if _scope.get(sid) else names):
            hints = {c: cp.hint() for c, cp in svc.table(t).columns.items() if cp.hint()}
            if hints:
                values[t] = hints
    else:
        names = candidates
    tables = {t: schema[t] for t in names if t in schema}

    # Popular tables first (what this database's users actually query), so an
    # ambiguous name resolves to the one everyone uses.
    usage = _load_usage(sid).get("tables", {})
    if usage:
        tables = dict(sorted(tables.items(), key=lambda kv: -usage.get(kv[0], 0)))
    hot = [t for t, _ in sorted(usage.items(), key=lambda kv: -kv[1])[:5]]
    sc = _scope.get(sid)
    scope_info = (
        {"column": sc["column"], "value": sc["value"], "label": sc.get("label", ""),
         "rule": f"The server filters every table that has {sc['column']} to {sc['value']!r} automatically."}
        if sc else None
    )
    scope_candidates = [] if sc else _scope_candidates(schema)

    biz = session.get("context") or {}
    biz_tables = biz.get("tables") if isinstance(biz.get("tables"), dict) else {}
    glossary = [{k: v for k, v in n.items() if k in ("table", "column", "note", "source")}
                for n in notes if not n.get("table") or n["table"] in tables][:20]
    # Human-confirmed examples first; unverified ones only ran without error.
    similar = sorted(ctx.get("similar_queries") or [], key=lambda q: not q.get("verified"))
    keys = await loop.run_in_executor(None, _table_keys, sid, list(tables))
    low_confidence = not glossary_hit and not names_anything(question, schema)
    kind = dialects.kind(session.get("db_type", "sql"))
    write_one = ("Write ONE SELECT in this dialect, then call run_sql with it and the question."
                 if kind == dialects.SQL else "Write ONE query in 'query_language', then call run_sql with it and the question.")
    result = {
        "dialect":          session.get("db_flavor"),
        "query_language":   dialects.LANGUAGE[kind],
        "tables":           tables,
        "other_tables":     other_tables,   # name -> what it is + first columns; fetch with get_schema
        "keys":             keys,
        "values":           values,   # what columns hold: status values, numeric/date ranges
        "joins":            joins,
        "context_selection": context_selection,
        "glossary":         glossary,
        "scope":            scope_info,
        "scope_candidates": scope_candidates,
        "hot_tables":       hot,
        "table_notes":      {t: n for t, n in biz_tables.items() if t in tables},
        "metrics":          biz.get("metrics") or {},
        "similar_queries":  similar,
        "business_type":    biz.get("business_type", "unknown"),
        "low_confidence":   low_confidence,
        "hint": ("No table matched this question well: ask the user which data they mean, or say this "
                 "database does not appear to hold it." if low_confidence else write_one),
    }
    # Code-mode measure: what the model got versus what sending every table would have cost.
    result["context_tokens"] = metrics.record(sid, result, schema)
    return _ok(result)


async def _schema(args: dict) -> list[types.TextContent]:
    sid    = _active(args)
    schema = redis_cache.get_schema(sid)
    if not schema:
        return _err("Not connected. Call connect_database first.", ErrCode.NOT_CONNECTED)
    wanted = args.get("tables")
    if not wanted:
        return _ok({"tables": schema, "table_count": len(schema)})
    lookup = {t.lower(): t for t in schema}
    schema = {lookup[w.lower()]: schema[lookup[w.lower()]] for w in wanted if w.lower() in lookup}
    keys = await asyncio.get_event_loop().run_in_executor(None, _table_keys, sid, list(schema))
    svc = _profiles.get(sid)
    joins = [{"from": f"{r.from_table}.{r.from_column}", "to": f"{r.to_table}.{r.to_column}", "source": r.source,
              "rows": f"many {r.from_table} per {r.to_table}"}
             for r in (svc.profile.relationships if svc else []) if r.from_table in schema and r.to_table in schema]
    return _ok({"tables": schema, "keys": keys, "joins": joins, "table_count": len(schema)})


async def _add_note(args: dict) -> list[types.TextContent]:
    """Teach the database once: a note on a table, a column, or the whole database."""
    import hashlib
    sid    = _active(args)
    _require_session(sid)
    schema = redis_cache.get_schema(sid) or {}
    note   = str(args.get("note", "")).strip()
    table  = str(args.get("table", "")).strip()
    column = str(args.get("column", "")).strip()
    if len(note) < 3:
        return _err("note is required.", ErrCode.INVALID_SQL)
    if table:
        lookup = {t.lower(): t for t in schema}
        if table.lower() not in lookup:
            return _err(f"Table '{table}' not in schema.", ErrCode.INVALID_SQL)
        table = lookup[table.lower()]
        if column:
            cols = {c.split(" ")[0].lower(): c.split(" ")[0] for c in schema[table]}
            if column.lower() not in cols:
                return _err(f"Column '{column}' not in {table}.", ErrCode.INVALID_SQL)
            column = cols[column.lower()]
    elif column:
        return _err("column requires table.", ErrCode.INVALID_SQL)

    source = str(args.get("source") or "user")
    if source not in ("user", "documentation", "inferred"):
        return _err("source must be one of: user, documentation, inferred.", ErrCode.INVALID_SQL)

    notes = _load_notes(sid)
    entry = {"id": hashlib.sha1(f"{table}|{column}|{note}".encode()).hexdigest()[:10],
             "table": table, "column": column, "note": note, "source": source,
             "author": _request_user(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if any(n["id"] == entry["id"] for n in notes):
        return _ok({"added": False, "duplicate": True, "note": entry})
    notes.append(entry)
    _save_notes(sid, notes)
    _audit("add_note", sid, f"{table}.{column}: {note}" if table else note, ok=True)
    return _ok({"added": True, "note": entry, "total_notes": len(notes)})


async def _list_notes(args: dict) -> list[types.TextContent]:
    sid   = _active(args)
    _require_session(sid)
    table = str(args.get("table", "")).strip().lower()
    notes = _load_notes(sid)
    if table:
        notes = [n for n in notes if n.get("table", "").lower() == table or not n.get("table")]
    return _ok({"notes": notes, "total": len(notes)})


async def _delete_note(args: dict) -> list[types.TextContent]:
    sid   = _active(args)
    _require_session(sid)
    nid   = str(args.get("id", "")).strip()
    notes = _load_notes(sid)
    kept  = [n for n in notes if n["id"] != nid]
    if len(kept) == len(notes):
        return _err(f"No note with id '{nid}'.", ErrCode.INVALID_SQL)
    _save_notes(sid, kept)
    _audit("delete_note", sid, nid, ok=True)
    return _ok({"deleted": True, "id": nid, "total_notes": len(kept)})


async def _mark_query(args: dict) -> list[types.TextContent]:
    sid      = _active(args)
    session  = _require_session(sid)
    ke       = _knowledge.get(sid)
    if ke is None:
        return _err("No knowledge for this connection yet - reconnect to build it.", ErrCode.NOT_CONNECTED)
    question   = str(args.get("question", "")).strip()
    sql        = str(args.get("sql", "")).strip().rstrip(";")
    correct    = bool(args.get("correct"))
    correction = str(args.get("correction", "")).strip().rstrip(";")
    if not question:
        return _err("question is required.", ErrCode.INVALID_SQL)
    keep = sql if correct else correction
    if keep:   # only a valid read-only query, in this connection's language, becomes an example
        try:
            dialects.validate_read(dialects.kind(session.get("db_type", "sql")), keep,
                                   redis_cache.get_schema(sid) or {}, session["db_flavor"])
        except QueryValidationError as ve:
            return _err(f"Not stored: {ve}", ErrCode.INVALID_SQL)
    ke.learn_user_feedback(question, sql, correct, correction or None)
    _audit("mark_query", sid, keep or sql, ok=True, extra={"question": question[:500], "correct": correct})
    return _ok({"stored": bool(keep), "verified": bool(keep), "removed_bad_pair": not correct})


async def _set_scope(args: dict) -> list[types.TextContent]:
    sid    = _active(args)
    _require_session(sid)
    schema = redis_cache.get_schema(sid) or {}
    column = str(args.get("column", "")).strip()
    value  = str(args.get("value", "")).strip()
    label  = str(args.get("label", "")).strip()
    if _scope.get(sid, {}).get("pinned"):
        _audit("set_scope", sid, f"refused: pinned ({column}={value})", ok=False)
        return _err("This identity's tenant scope is fixed by the server administrator (SCOPE_PINS) and "
                    "cannot be changed or cleared.", ErrCode.AUTH_FAILED)
    if not value:
        _scope.pop(sid, None)
        redis_cache.clear_user_query_cache(sid)
        _audit("set_scope", sid, f"cleared {column}", ok=True)
        return _ok({"scope": None, "message": "Scope cleared - queries now span all clients."})
    sc = make_scope(column, value, schema, label)
    if not sc:
        return _err(f"No table has a column named '{column}'.", ErrCode.INVALID_SQL)
    _scope[sid] = sc
    redis_cache.clear_user_query_cache(sid)   # cached answers were computed under the old scope
    _audit("set_scope", sid, f"{sc['column']} = {value} ({label})", ok=True)
    return _ok({"scope": {"column": sc["column"], "value": value, "label": label, "scoped_tables": len(sc["tables"])},
                "message": f"The {len(sc['tables'])} tables that have {sc['column']} are now always filtered to "
                           f"{sc['column']} = {value}, whatever the SQL says."})


async def _set_sql_generator(args: dict) -> list[types.TextContent]:
    sid  = _active(args)
    _require_session(sid)
    mode = str(args.get("mode", "")).strip().lower()
    if mode not in ("server", "client", "auto"):
        return _err("mode must be one of: 'server', 'client', 'auto'.", ErrCode.INVALID_SQL)
    groq_key = str(args.get("groq_key", "")).strip()
    if groq_key:
        _llm_keys[sid] = groq_key
    if mode == "server" and not llm_available(_llm_keys.get(sid) or GROQ_API_KEY):
        return _err(
            "Cannot switch to 'server' SQL generator because no LLM API key is configured on the server. "
            "Pass groq_key to set_sql_generator(mode='server', groq_key='...') or use mode='client'.",
            ErrCode.AUTH_FAILED,
        )
    if mode in ("server", "client"):
        _sql_generators[sid] = mode
    else:
        _sql_generators.pop(sid, None)
    eff = _effective_sql_generator(sid)
    _audit("set_sql_generator", sid, f"mode={mode} (effective={eff})", ok=True)
    return _ok({
        "sql_generator": eff,
        "sql_generator_mode": mode,
        "workflow": (
            "1 round-trip: call query_database(question) directly — the server's AI generates, reviews, and executes SQL."
            if eff == "server" else
            "2 round-trips: call get_query_context(question), write the SQL yourself, and execute it with run_sql(sql, question=...)."
        ),
    })


async def _import_query_history(args: dict) -> list[types.TextContent]:
    sid     = _active(args)
    session = _require_session(sid)
    limit   = min(2000, max(10, int(args.get("limit", 300))))
    kind    = dialects.kind(session.get("db_type", "sql"))
    if kind == dialects.CASSANDRA:
        return _err("Cassandra keeps no per-query statistics to import. Use import_queries with your team's "
                    "CQL instead.", ErrCode.UNSUPPORTED_DB)
    schema  = redis_cache.get_schema(sid) or {}
    flavor  = session.get("db_flavor", "")
    loop    = asyncio.get_event_loop()
    handle  = _handle(sid, kind)

    def _fetch():
        if kind == dialects.MONGO:   # the database profiler's record: collection -> operations
            return [(t, n) for t, n in mongo.collection_usage(handle, limit, DB_QUERY_TIMEOUT) if t in schema]
        with handle.connect() as conn:
            return _fetch_query_history(conn, flavor, limit)
    try:
        history = await asyncio.wait_for(loop.run_in_executor(None, _fetch), timeout=DB_QUERY_TIMEOUT)
    except Exception as e:
        need = ("the database profiler on and a login allowed to read system.profile" if kind == dialects.MONGO
                else "VIEW SERVER STATE")
        return _err(f"Could not read query statistics ({flavor}); this needs {need}. {str(e)[:200]}",
                    ErrCode.QUERY_FAILED)
    if not history:
        return _ok({"imported_statements": 0, "message": f"No query statistics available for {flavor}."
                    + (" Turn on the database profiler (level 1 or 2) to record them." if kind == dialects.MONGO else "")})

    usage = _load_usage(sid)
    matched = 0
    for stmt, count in history:
        tables = [stmt] if kind == dialects.MONGO else _tables_in_sql(stmt, schema)
        if not tables:
            continue
        matched += 1
        for t in tables:
            usage["tables"][t] = usage["tables"].get(t, 0) + max(1, count)
    usage["statements"] = usage.get("statements", 0) + matched
    path = _usage_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(usage, f)
    top = sorted(usage["tables"].items(), key=lambda kv: -kv[1])[:10]
    _audit("import_query_history", sid, f"{matched} statements", ok=True, rows=matched)
    return _ok({"imported_statements": matched, "statements_read": len(history),
                "hot_tables": [{"table": t, "executions": n} for t, n in top]})


def _nosql_imports(kind: str, text_in: str) -> list[tuple[str, object]]:
    """(question, query) pairs. MongoDB: a JSON list or one JSON query per line, each with an optional
    "question". CQL: statements split on ';', '--' comment lines become the question."""
    if kind == dialects.MONGO:
        try:
            items = json.loads(text_in)
            items = items if isinstance(items, list) else [items]
        except json.JSONDecodeError:
            items = []
            for line in filter(None, (l.strip() for l in text_in.splitlines())):
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    items.append(None)   # counted as skipped by validation
        return [(str((i or {}).get("question", "")) if isinstance(i, dict) else "",
                 {k: v for k, v in i.items() if k != "question"} if isinstance(i, dict) else i) for i in items]
    pairs = []
    for chunk in filter(None, (c.strip() for c in re.split(r";\s*\n|;\s*$", text_in, flags=re.MULTILINE))):
        lines = chunk.splitlines()
        question = " ".join(l.strip()[2:].strip() for l in lines if l.strip().startswith("--"))
        pairs.append((question, "\n".join(l for l in lines if not l.strip().startswith("--")).strip()))
    return pairs


async def _import_queries(args: dict) -> list[types.TextContent]:
    sid     = _active(args)
    session = _require_session(sid)
    schema  = redis_cache.get_schema(sid) or {}
    ke      = _knowledge.get(sid)
    text_in = str(args.get("sql_text", ""))
    kind    = dialects.kind(session.get("db_type", "sql"))
    imported, skipped = 0, 0
    if kind != dialects.SQL:
        for question, query in _nosql_imports(kind, text_in):
            try:
                checked = dialects.validate_read(kind, query, schema, session["db_flavor"])
            except QueryValidationError:
                skipped += 1
                continue
            sources = _sources(kind, checked, schema)
            if ke:
                ke.learn_successful_query(question or f"query over {', '.join(sources)}",
                                          dialects.as_text(kind, checked), verified=True)
            _bump_usage(sid, sources)
            imported += 1
        _audit("import_queries", sid, f"{imported} imported, {skipped} skipped", ok=True, rows=imported)
        return _ok({"imported": imported, "skipped": skipped,
                    "hint": "Imported queries now appear as similar_queries in get_query_context and count toward hot_tables."})
    for chunk in re.split(r";\s*\n|;\s*$", text_in, flags=re.MULTILINE):
        chunk = chunk.strip()
        if not chunk:
            continue
        comments = [l[2:].strip() for l in chunk.splitlines() if l.strip().startswith("--")]
        sql = "\n".join(l for l in chunk.splitlines() if not l.strip().startswith("--")).strip()
        if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
            skipped += 1
            continue
        tables = _tables_in_sql(sql, schema)
        if not tables:
            skipped += 1
            continue
        question = " ".join(comments) or f"query over {', '.join(tables)}"
        if ke:
            ke.learn_successful_query(question, sql, verified=True)   # the team's own SQL
        _bump_usage(sid, tables)
        imported += 1
    _audit("import_queries", sid, f"{imported} imported, {skipped} skipped", ok=True, rows=imported)
    return _ok({"imported": imported, "skipped": skipped,
                "hint": "Imported SELECTs now appear as similar_queries in get_query_context and count toward hot_tables."})


async def _validate(args: dict) -> list[types.TextContent]:
    sid     = _active(args)
    session = _require_session(sid)
    schema  = redis_cache.get_schema(sid) or {}
    raw     = args.get("sql", args.get("query", ""))
    sql     = raw if isinstance(raw, dict) else str(raw).strip().rstrip(";")
    kind    = dialects.kind(session.get("db_type", "sql"))
    try:
        checked = dialects.validate_read(kind, sql, schema, session["db_flavor"])
        final, scoped = dialects.scope_read(kind, checked, _scope.get(sid), session["db_flavor"], schema)
    except QueryValidationError as ve:
        return _ok({"valid": False, "reason": str(ve)})
    result = {"valid": True, "sql_to_run": dialects.as_text(kind, final), "tables": _sources(kind, checked, schema),
              "tenant_filtered": scoped, "faster_equivalent": None, "review": []}
    if kind == dialects.SQL:   # speed rewrites and checks are SQL-specific
        reviewer = _reviewer(sid, session["db_flavor"])
        faster, rewrites = reviewer.rewrite(checked)
        result.update(faster_equivalent=faster if rewrites else None,
                      review=[{"code": f.code, "message": f.message} for f in reviewer.review(checked)])
    return _ok(result)


async def _explain(args: dict) -> list[types.TextContent]:
    sid     = _active(args)
    session = _require_session(sid)
    kind    = dialects.kind(session.get("db_type", "sql"))
    if kind == dialects.CASSANDRA:
        return _err("Cassandra has no query plans. Queries on the partition key are fast; others need ALLOW "
                    "FILTERING and read more data.", ErrCode.UNSUPPORTED_DB)
    if kind == dialects.MONGO:
        schema = redis_cache.get_schema(sid) or {}
        try:
            checked = dialects.validate_read(kind, args.get("sql", args.get("query", "")), schema, session["db_flavor"])
            prepared, _ = dialects.scope_read(kind, checked, _scope.get(sid), session["db_flavor"], schema)
            plan = await asyncio.get_event_loop().run_in_executor(None, mongo.explain, _handle(sid, kind), prepared)
        except QueryValidationError as ve:
            return _err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)
        except Exception as e:
            return _err(str(e), ErrCode.QUERY_FAILED)
        return _ok({"sql": dialects.as_text(kind, prepared), "flavor": "MongoDB", "plan": plan})
    engine  = _engines.get(sid)
    if not engine:
        return _err("No SQL engine found. Reconnect first.", ErrCode.NOT_CONNECTED)

    db_flavor = session.get("db_flavor", "")
    # EXPLAIN ANALYZE executes the statement (PostgreSQL): only validated, scoped SELECTs get here.
    try:
        sql = validate_sql(str(args["sql"]).strip().rstrip(";"), redis_cache.get_schema(sid) or {}, db_flavor, "")
        sql, _ = _apply_scope(sid, sql, db_flavor)
    except QueryValidationError as ve:
        return _err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)
    explain_sql = explain_statement(sql, db_flavor)

    def _run():
        with engine.connect() as conn:
            result = conn.execute(text(explain_sql))
            return [dict(row._mapping) for row in result]

    loop = asyncio.get_event_loop()
    try:
        plan = await loop.run_in_executor(None, _run)
        return _ok({"sql": sql, "flavor": db_flavor, "plan": plan})
    except Exception as e:
        return _err(str(e), ErrCode.QUERY_FAILED)


async def _chart(args: dict) -> list[types.TextContent]:
    """Rule-based: column types decide the chart, no LLM call."""
    _track_usage(_active(args), "charts")
    return _ok({"chart": suggest_chart_rule(args["columns"], args["rows"])})


async def _describe_capabilities(args: dict) -> list[types.TextContent]:
    sid     = _resolve_sid(args)
    current = _current.get(sid)
    session = _get_session(current) if current else None
    connected_db = None
    if session:
        connected_db = {
            "connection":    current,
            "db_flavor":     session.get("db_flavor"),
            "business_type": session.get("context", {}).get("business_type", "unknown"),
            "has_schema":    bool(redis_cache.get_schema(current)),
        }
    prefix = f"{sid}__"
    open_connections = [k for k in (*_engines, *_mongos, *_cassandras) if k.startswith(prefix)]
    gen = _effective_sql_generator(current or sid)
    return _ok({
        "server":          "queryatlas",
        "version":         __version__,
        "session_id":      sid,
        "connected":       connected_db is not None,
        "current_session": connected_db,
        "open_connections": open_connections,
        "sql_generator":   gen,
        "sql_generator_mode": POLICY.sql_generator(),
        "usage":           dict(_usage.get(sid, {})),
        "context_savings": metrics.summary(current) if current else {"questions": 0},
        "write_access": {
            "server_enabled": _writes_enabled(),
            "role_permitted": _user_can_write(),
            "connection_opted_in": bool(session and session.get("allow_writes")),
            "enabled": bool(_writes_enabled() and _user_can_write()
                            and session and session.get("allow_writes")),
            "ddl_enabled": bool(_ddl_enabled() and _user_can_write()
                                and session and session.get("allow_writes")),
        },
        "supported_databases": [
            "SQLite", "MySQL", "MariaDB", "PostgreSQL", "SQL Server", "MongoDB", "Cassandra"
        ],
        "tools": [
            {"name": "connect_database",    "requires_session": False},
            {"name": "query_database",      "requires_session": True,  "needs_llm": True, "llm_model": LLM_MODEL},
            {"name": "validate_sql",        "requires_session": True,  "databases": ["sql", "mongo", "cassandra"]},
            {"name": "run_sql",             "requires_session": True,  "databases": ["sql", "mongo", "cassandra"]},
            {"name": "get_query_context",   "requires_session": True},
            {"name": "get_schema",          "requires_session": True},
            {"name": "add_note",            "requires_session": True},
            {"name": "list_notes",          "requires_session": True},
            {"name": "delete_note",         "requires_session": True},
            {"name": "mark_query",          "requires_session": True},
            {"name": "set_scope",           "requires_session": True},
            {"name": "set_sql_generator",   "requires_session": True},
            {"name": "import_query_history", "requires_session": True, "databases": ["sql", "mongo"]},
            {"name": "import_queries",      "requires_session": True},
            {"name": "explain_query",       "requires_session": True,  "databases": ["sql", "mongo"]},
            {"name": "suggest_chart",       "requires_session": False},
            {"name": "execute_write",       "requires_session": True,  "databases": ["sql", "mongo", "cassandra"],
             "developer_mode": True,
             "enabled": _writes_enabled() and _user_can_write(),
             "connection_opted_in": bool(session and session.get("allow_writes")),
             "ddl_enabled": _ddl_enabled() and _user_can_write()},
            {"name": "disconnect",          "requires_session": True},
            {"name": "describe_capabilities", "requires_session": False},
        ],
        "error_codes": {
            ErrCode.NOT_CONNECTED:  "No active session — call connect_database first",
            ErrCode.QUERY_FAILED:   "Database returned an error executing the query",
            ErrCode.AUTH_FAILED:    "Authentication / credential error",
            ErrCode.PERMISSION_DENIED: "The current user, server settings, or connection does not allow this action",
            ErrCode.CONNECTION_FAILED: "Database connection failed; credentials are not returned in the error",
            ErrCode.UNSUPPORTED_DB: "db_type is not one of the supported databases",
            ErrCode.INVALID_SQL:    "Generated SQL failed validation",
            ErrCode.TIMEOUT:        "Query exceeded the timeout limit",
            ErrCode.NEEDS_CLARIFICATION: "The question names nothing in this database: ask the user what they mean",
            ErrCode.UNKNOWN:        "Unexpected server error",
        },
    })


async def _upload_document(args: dict) -> list[types.TextContent]:
    import base64
    sid   = _resolve_sid(args)
    store = _get_docstore(sid)
    loop  = asyncio.get_event_loop()

    url = args.get("url", "").strip()
    if url:
        try:
            chunks = await loop.run_in_executor(None, store.add_url, url, _HOSTED)
            return _ok({"indexed": True, "source": url, "chunks_indexed": chunks})
        except Exception as e:
            return _err(f"Failed to fetch URL: {e}")

    filename   = args.get("filename", "document.txt")
    content_b64 = args.get("content_b64", "")
    if not content_b64:
        return _err("Provide either content_b64 (base64 file bytes) or url.")
    try:
        data   = base64.b64decode(content_b64)
        chunks = await loop.run_in_executor(None, store.add_file, filename, data)
        return _ok({"indexed": True, "source": filename, "chunks_indexed": chunks})
    except Exception as e:
        return _err(f"Failed to index document: {e}")


async def _search_documents(args: dict) -> list[types.TextContent]:
    sid      = _resolve_sid(args)
    question = args["question"]
    n        = min(10, max(1, int(args.get("n_results", 4))))
    store    = _get_docstore(sid)
    loop     = asyncio.get_event_loop()

    if store.count() == 0:
        return _ok({
            "chunks": [],
            "hint": "No documents indexed yet. Call upload_document first.",
        })

    hits = await loop.run_in_executor(None, store.search, question, n)
    return _ok({"chunks": hits, "total_chunks_in_store": store.count()})


async def _list_documents(args: dict) -> list[types.TextContent]:
    sid   = _resolve_sid(args)
    store = _get_docstore(sid)
    loop  = asyncio.get_event_loop()
    sources = await loop.run_in_executor(None, store.list_sources)
    return _ok({"sources": sources, "total_chunks": store.count()})


async def _disconnect(args: dict) -> list[types.TextContent]:
    session_id = _resolve_sid(args)
    sid = _current.pop(session_id, session_id)   # closes only the current connection
    _mcp_conns.remove(session_id, sid)
    _discard_connection(sid)
    return _ok({"disconnected": True, "session_id": sid})


def _discard_connection(sid: str) -> None:
    """Forget everything held for one connection and close its database handle (MongoDB and
    Cassandra clients keep their logged-in connections open until closed)."""
    redis_cache.delete_session(sid)
    redis_cache.clear_user_query_cache(sid)
    metrics.forget(sid)
    handles = [_engines.pop(sid, None), _mongos.pop(sid, None), _cassandras.pop(sid, None)]
    for state in (_retrievers, _knowledge, _fks, _nosql_keys, _scope, _keys, _llm_keys, _sql_generators,
                  _profiles, _docstores, _usage):
        state.pop(sid, None)
    for handle in handles:
        close_handle(handle)


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(NotificationOptions(tools_changed=True)),
        )


_OAUTH_CODES: dict[str, dict] = {}   # one-time auth_code -> {user, redirect_uri, code_challenge, method, exp}
# Exact redirect URIs allowed besides loopback, e.g. https://claude.ai/api/mcp/auth_callback for claude.ai connectors.
_EXTRA_REDIRECT_URIS = {u.strip() for u in os.getenv("OAUTH_REDIRECT_URIS", "").split(",") if u.strip()}


def _redirect_allowed(uri: str) -> bool:
    """Registration is open to anyone, so a registered URI proves nothing: the code may only go to
    a loopback address (Claude Code, Cursor and Claude Desktop listen on localhost) or an
    administrator-listed URI. Anything else would hand a signed-in user's code to a third party."""
    parts = urllib.parse.urlsplit(uri)
    return uri in _EXTRA_REDIRECT_URIS or (
        parts.scheme in ("http", "https") and parts.hostname in ("localhost", "127.0.0.1", "::1"))


def _render_oauth_login_html(params: dict[str, str], web_signup_url: str, error: str = "") -> str:
    import html as _html
    err_html = f'<div class="err">{_html.escape(error)}</div>' if error else ""
    hidden = "\n".join(
        f'<input type="hidden" name="{_html.escape(k)}" value="{_html.escape(str(v or ""))}"/>'
        for k, v in params.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Authorize MCP · NL·DB Chat</title>
<style>
  body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0d0e14;color:#e5e7eb;font-family:system-ui,-apple-system,sans-serif}}
  .card{{background:#141620;border:1px solid #26293a;border-radius:12px;padding:28px 26px;width:min(380px,92vw)}}
  h1{{font-size:18px;margin:0 0 6px;color:#f3f4f6}}
  p{{font-size:13px;color:#9ca3af;margin:0 0 18px;line-height:1.45}}
  label{{display:block;font-size:12px;color:#9ca3af;margin:12px 0 4px}}
  input{{width:100%;box-sizing:border-box;padding:9px 11px;background:#1c1f2e;border:1px solid #2e3247;border-radius:8px;color:#f3f4f6;font-size:13.5px;outline:none}}
  input:focus{{border-color:#7c5cfc}}
  button{{width:100%;margin-top:18px;padding:10px;background:#7c5cfc;border:none;border-radius:8px;color:#fff;font-weight:600;font-size:13.5px;cursor:pointer}}
  button:hover{{background:#6b4ce6}}
  .err{{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.3);color:#f87171;padding:8px 10px;border-radius:8px;font-size:12.5px;margin-bottom:12px}}
  .foot{{margin-top:16px;font-size:12px;color:#9ca3af;text-align:center}}
  .foot a{{color:#a78bfa;text-decoration:none}}
</style></head>
<body><div class="card">
  <h1>Connect MCP to NL·DB Chat</h1>
  <p>Sign in with your account to issue a personal MCP token to your AI client.</p>
  {err_html}
  <form method="POST" action="/oauth/authorize">
    {hidden}
    <label>Username or Email</label>
    <input name="username" required autocomplete="username" autofocus placeholder="e.g. alice"/>
    <label>Password</label>
    <input name="password" type="password" required autocomplete="current-password" placeholder="Your account password"/>
    <button type="submit">Sign In &amp; Authorize MCP</button>
  </form>
  <div class="foot">Need an account first? <a href="{_html.escape(web_signup_url)}" target="_blank">Create an account on Web UI</a></div>
</div></body></html>"""


def build_http_mcp_app(manager=None):
    """Build the ASGI app serving Streamable HTTP `/mcp` + OAuth 2.1 PKCE discovery & consent."""
    import base64
    import hashlib
    import hmac
    import secrets
    import urllib.parse

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
    from starlette.middleware import Middleware
    from starlette.routing import Mount

    tokens = [t for t in _TOKEN_USERS if len(t) >= 16]
    if manager is None:
        manager = StreamableHTTPSessionManager(app=server)

    def _origin(scope) -> str:
        scheme = scope.get("scheme") or "http"
        headers = dict(scope.get("headers") or {})
        host_hdr = headers.get(b"host", b"localhost:8765").decode("latin-1")
        return f"{scheme}://{host_hdr}"

    def _web_ui_url(scope) -> str:
        scheme = scope.get("scheme") or "http"
        headers = dict(scope.get("headers") or {})
        host_only = headers.get(b"host", b"localhost").decode("latin-1").split(":")[0]
        return os.getenv("WEB_UI_URL", f"{scheme}://{host_only}:8000")

    async def guarded(scope, receive, send):
        if scope["type"] != "http":
            return
        path = scope["path"].rstrip("/") or "/"
        method = scope.get("method", "GET").upper()
        base = _origin(scope)

        # 1. OAuth 2.0 Protected Resource Metadata (RFC 9728)
        if path in ("/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"):
            await JSONResponse({
                "resource": f"{base}/mcp",
                "authorization_servers": [base],
                "bearer_methods_supported": ["header"],
            })(scope, receive, send)
            return

        # 2. OAuth 2.1 Authorization Server Metadata (RFC 8414)
        if path in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
            await JSONResponse({
                "issuer": base,
                "authorization_endpoint": f"{base}/oauth/authorize",
                "token_endpoint": f"{base}/oauth/token",
                "registration_endpoint": f"{base}/oauth/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
                "code_challenge_methods_supported": ["S256"],
            })(scope, receive, send)
            return

        # 3. Dynamic Client Registration (RFC 7591)
        if path == "/oauth/register" and method == "POST":
            req = Request(scope, receive)
            body = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
            client_id = "mcp-client-" + secrets.token_urlsafe(12)
            await JSONResponse({
                "client_id": client_id,
                "client_id_issued_at": int(time.time()),
                "client_name": body.get("client_name", "MCP Client"),
                "redirect_uris": body.get("redirect_uris", []),
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            }, status_code=201)(scope, receive, send)
            return

        # 4. Browser Sign-In & Consent Screen (/oauth/authorize)
        if path == "/oauth/authorize":
            req = Request(scope, receive)
            if method == "GET":
                qp = dict(req.query_params)
                oauth_params = {
                    k: qp.get(k, "")
                    for k in ("client_id", "redirect_uri", "state", "code_challenge", "code_challenge_method", "response_type")
                }
                await HTMLResponse(_render_oauth_login_html(oauth_params, _web_ui_url(scope)))(scope, receive, send)
                return
            if method == "POST":
                raw_body = (await req.body()).decode("utf-8", errors="replace")
                form = {k: v[0] for k, v in urllib.parse.parse_qs(raw_body, keep_blank_values=True).items()}
                oauth_params = {
                    k: form.get(k, "")
                    for k in ("client_id", "redirect_uri", "state", "code_challenge", "code_challenge_method", "response_type")
                }
                username = (form.get("username") or "").strip()
                password = form.get("password") or ""
                if not POLICY.has_account(username):
                    html_err = _render_oauth_login_html(
                        oauth_params, _web_ui_url(scope),
                        "Account not found. You must create an account first before authorizing MCP.",
                    )
                    await HTMLResponse(html_err, status_code=401)(scope, receive, send)
                    return
                if sign_in_blocked(username):   # shared with the web sign-in, across server processes
                    html_err = _render_oauth_login_html(
                        oauth_params, _web_ui_url(scope), "Too many failed sign-ins. Try again in 15 minutes."
                    )
                    await HTMLResponse(html_err, status_code=429)(scope, receive, send)
                    return
                ok = POLICY.verify_account(username, password)
                record_sign_in(username, ok)
                if not ok:
                    _audit("sign_in", username, "", ok=False, error="wrong password (MCP authorization)")
                    html_err = _render_oauth_login_html(
                        oauth_params, _web_ui_url(scope), "Incorrect username or password."
                    )
                    await HTMLResponse(html_err, status_code=401)(scope, receive, send)
                    return
                if not POLICY.can(username, rbac.MCP):
                    html_err = _render_oauth_login_html(
                        oauth_params, _web_ui_url(scope),
                        f"Your account ({POLICY.role_of(username)}) does not have 'mcp.use' permission.",
                    )
                    await HTMLResponse(html_err, status_code=403)(scope, receive, send)
                    return
                redirect_uri = oauth_params.get("redirect_uri") or ""
                if not _redirect_allowed(redirect_uri):
                    html_err = _render_oauth_login_html(
                        oauth_params, _web_ui_url(scope),
                        f"This client's redirect address ({redirect_uri or 'none'}) is not allowed. "
                        "Use Claude Code / Cursor / Claude Desktop, or ask the administrator to add it to OAUTH_REDIRECT_URIS.",
                    )
                    await HTMLResponse(html_err, status_code=400)(scope, receive, send)
                    return
                # PKCE S256 is required: it binds the code to the client that started the sign-in.
                if not oauth_params.get("code_challenge") or (oauth_params.get("code_challenge_method") or "S256").upper() != "S256":
                    html_err = _render_oauth_login_html(
                        oauth_params, _web_ui_url(scope), "The client must use PKCE with S256."
                    )
                    await HTMLResponse(html_err, status_code=400)(scope, receive, send)
                    return
                code = secrets.token_urlsafe(32)
                _OAUTH_CODES[code] = {
                    "user": username,
                    "redirect_uri": redirect_uri,
                    "code_challenge": oauth_params["code_challenge"],
                    "exp": time.time() + 300,
                }
                sep = "&" if "?" in redirect_uri else "?"
                target = f"{redirect_uri}{sep}code={urllib.parse.quote(code)}"
                if oauth_params.get("state"):
                    target += f"&state={urllib.parse.quote(oauth_params['state'])}"
                await RedirectResponse(target, status_code=302)(scope, receive, send)
                return

        # 5. Token Exchange Endpoint (/oauth/token — verifies PKCE & issues unique mcp_... token)
        if path == "/oauth/token" and method == "POST":
            req = Request(scope, receive)
            ctype = req.headers.get("content-type", "")
            if "application/json" in ctype:
                data = await req.json()
            else:
                raw_body = (await req.body()).decode("utf-8", errors="replace")
                data = {k: v[0] for k, v in urllib.parse.parse_qs(raw_body, keep_blank_values=True).items()}
            code = data.get("code", "")
            verifier = data.get("code_verifier", "")
            entry = _OAUTH_CODES.pop(code, None)
            if not entry or time.time() > entry["exp"]:
                await JSONResponse({"error": "invalid_grant", "error_description": "Invalid or expired code."}, status_code=400)(scope, receive, send)
                return
            digest = hashlib.sha256(verifier.encode("utf-8")).digest()
            computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
            if not hmac.compare_digest(computed, entry["code_challenge"]):
                await JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed."}, status_code=400)(scope, receive, send)
                return
            raw_token = POLICY.issue_mcp_token(entry["user"])
            await JSONResponse({
                "access_token": raw_token,
                "token_type": "Bearer",
                "expires_in": rbac.MCP_TOKEN_SECONDS,
            })(scope, receive, send)
            return

        if path != "/mcp":
            await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)
            return

        auth = dict(scope.get("headers") or []).get(b"authorization", b"").decode()
        bearer = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
        valid_env = any(hmac.compare_digest(auth, f"Bearer {t}") for t in tokens)
        valid_rbac = bool(bearer and POLICY.user_for_mcp_token(bearer))
        if not (valid_env or valid_rbac):
            await JSONResponse(
                {"error": "unauthorized", "auth_url": f"{base}/oauth/authorize"},
                status_code=401,
                headers={
                    "WWW-Authenticate": f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource"'
                },
            )(scope, receive, send)
            return
        await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        if hasattr(manager, "run"):
            async with manager.run():
                yield
        else:
            yield

    # MAX_REQUEST_MB: OAuth forms, MCP JSON-RPC (including document uploads) alike.
    return Starlette(routes=[Mount("/", app=guarded)], lifespan=lifespan,
                     middleware=[Middleware(BodyLimit), Middleware(SecurityHeaders)])


def _serve_http(bind: str) -> None:
    """Hosted mode: Streamable HTTP at /mcp + OAuth 2.1 browser authorization flow."""
    import uvicorn

    global _HOSTED
    _HOSTED = True
    host, _, port = bind.rpartition(":")
    print("[queryatlas] Hosted: database connections must be encrypted; hosts on private/internal networks are "
          "refused unless listed in MCP_ALLOWED_HOSTS; SQLite only inside MCP_SQLITE_DIR; passwords only "
          "through the connection form.", file=sys.stderr)
    app = build_http_mcp_app()
    print(f"[queryatlas] HTTP mode: http://{host or '0.0.0.0'}:{port or 8765}/mcp", file=sys.stderr)
    uvicorn.run(app, host=host or "0.0.0.0", port=int(port or 8765), log_level="info")


def _install_claude_code() -> None:
    """Register this server with Claude Code (user scope, stdio) using this exact interpreter and
    path. Claude Code then starts and stops it itself: no server to keep running, no ECONNREFUSED."""
    import shutil
    import subprocess

    claude = shutil.which("claude")
    if not claude:
        sys.exit("Claude Code CLI ('claude') is not on PATH. Install Claude Code first, then rerun this.")
    user = sys.argv[sys.argv.index("--user") + 1] if "--user" in sys.argv[:-1] else MCP_USER_ID
    # Replaces an earlier entry (stdio or http) of the same name; "not found" on a first install is expected.
    subprocess.run([claude, "mcp", "remove", "queryatlas", "-s", "user"], capture_output=True)
    subprocess.run([claude, "mcp", "remove", "sql-agent", "-s", "user"], capture_output=True)   # pre-rename registration
    subprocess.run([claude, "mcp", "add", "queryatlas", "-s", "user", "-e", "PYTHONUTF8=1", "-e", f"MCP_USER_ID={user}",
                    "--", sys.executable, os.path.join(PROJECT_ROOT, "mcp_server.py")], check=True)
    if not os.path.exists(os.path.join(PROJECT_ROOT, ".env")):
        print("Optional: copy .env.example to .env for database passwords and an LLM key.")
    print(f"Registered queryatlas for Claude Code as user '{user}'. Restart Claude Code, then check: claude mcp list")


def cli() -> None:
    """Entry point behind the root launcher: python mcp_server.py [--http host:port | --install [--user NAME]]."""
    # Newer Claude Code probes with 'server/discover' before initialize; the SDK answers it, but
    # logs a 31-line validation warning that reads like a startup failure in the MCP logs.
    logging.getLogger().addFilter(lambda record: "server/discover" not in record.getMessage())
    if "--install" in sys.argv:
        _install_claude_code()
    elif "--http" in sys.argv:
        i = sys.argv.index("--http")
        nxt = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
        _serve_http(nxt if nxt and not nxt.startswith("-") else "127.0.0.1:8765")
    else:
        asyncio.run(main())


if __name__ == "__main__":
    cli()
