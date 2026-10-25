"""connect_database, disconnect and the idle-lock unlock: forms for the details that must never
pass through the model, the step-by-step publish of a connection with a rollback for every earlier
exit (fix #15), and the switch to an already-open database."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import re
from dataclasses import dataclass
from typing import Any

from mcp import types

from ... import cache as redis_cache
from ... import cql, dialects, mongo, storage
from ...config import settings
from ...connections import ConnectionInfo, TargetRefused, allowed_hosts, check_target, connection_label, safe_owner
from ...database import (
    _connection_failure_message,
    close_handle,
    fetch_foreign_keys,
    fetch_sql_schema,
    filter_empty_tables,
)
from ...errors import ErrCode
from ...knowledge import KnowledgeEngine
from ...llm import llm_available
from ...prompts import enrich_context_joins
from ...query_planner import reset_concept_cache
from ...scope import make_scope, pinned_scope
from ...state import ConnectionState
from ..context import (
    QUERY_TIMEOUT,
    audit,
    conn_key,
    discard_connection,
    effective_sql_generator,
    get_session,
    resolve_sid,
    tenant_column_candidates,
    track_usage,
    user_can_write,
    writes_enabled,
)
from ..results import err, ok
from ..runtime import rt

log = logging.getLogger(__name__)


async def ask_password(info: ConnectionInfo) -> str | None:
    """Ask the person (MCP form, client -> server; the model never sees it). None if the
    client cannot show forms or the person cancels."""
    try:
        session = rt.mcp_session()
        result = await session.elicit(
            message=f"{info.label} has not been used for a while. Enter its password to continue.",
            requestedSchema={
                "type": "object",
                "properties": {"password": {"type": "string", "title": "Password"}},
                "required": ["password"],
            },
        )
    except Exception as e:
        log.warning(f"password form unavailable: {e}")
        return None
    if result.action != "accept" or not (result.content or {}).get("password"):
        return None
    return str(result.content["password"])


async def ensure_unlocked(args: dict) -> list[types.TextContent] | None:
    """Same rule as the web app: verify the connection's RSA token in RAM, and if a
    password-opened database is idle past idle_lock_minutes, ask for the password again."""
    session_id = resolve_sid(args)
    sid = rt.current.get(session_id, session_id)
    client_token = str(args.get("connection_token") or "").strip()
    info = rt.connections.get(session_id, sid)
    if info is None and not client_token:
        return None
    try:
        sid = rt.connections.verify(session_id, sid, client_token)
    except PermissionError as e:
        audit("rsa_verify", sid, "", ok=False, error=str(e))
        return err(str(e), ErrCode.AUTH_FAILED)
    if rt.connections.is_locked(session_id, sid):
        password = await rt.ask_password(info)
        if password is None:
            audit("unlock", sid, "", ok=False, error="no password given")
            return err(
                f"{info.label} is locked after being idle. Re-enter its password when asked "
                "(or call connect_database again).",
                ErrCode.AUTH_FAILED,
            )
        d = info.descriptor
        try:
            # Checked again: a host name can point somewhere else since the first connect.
            address = check_target(d["db_type"], d.get("host", ""), d.get("db_path", ""), rt.hosted)
            handle = await asyncio.to_thread(lambda: rt.open_handle(**d, password=password, address=address))
        except TargetRefused as refused:
            audit("unlock", sid, "", ok=False, error=str(refused))
            return err(str(refused), ErrCode.PERMISSION_DENIED)
        except Exception:
            audit("unlock", sid, "", ok=False, error="wrong password")
            return err("That password didn't work; the database stays locked.", ErrCode.AUTH_FAILED)
        close_handle(handle)  # only the proof was needed (any database type)
        audit("unlock", sid, "", ok=True)
    rt.connections.touch(sid)
    return None


_CONN_FIELDS = {
    "SQLite": (["db_path"], ["db_path"]),
    "MongoDB": (["host", "port", "username", "password", "database"], ["host", "database"]),
    "Cassandra": (["host", "port", "username", "password", "keyspace"], ["host", "keyspace"]),
}
_SQL_CONN_FIELDS = (["host", "port", "username", "password", "database"], ["host", "username", "password", "database"])
_DEFAULT_PORTS = {
    "MySQL": "3306",
    "MariaDB": "3306",
    "PostgreSQL": "5432",
    "SQL Server": "1433",
    "MongoDB": "27017",
    "Cassandra": "9042",
}


# Hosted: taken only from the connection form, never from tool arguments (which the model wrote).
_FORM_ONLY_FIELDS = {"password", "groq_key", "trust_cert"}


def env_db_password(database: str, host: str) -> str:
    """Password from the environment: MCP_DB_PASSWORD_<DATABASE> (several DBs on
    different servers), falling back to the generic MCP_DB_PASSWORD. On a hosted server only for
    a host the operator listed in MCP_ALLOWED_HOSTS: customers name any public host, and the
    server's own password must never be sent to one of theirs."""
    if rt.hosted and str(host or "").strip().lower() not in allowed_hosts():
        return ""
    key = re.sub(r"[^A-Z0-9]", "_", (database or "").upper())
    return settings.db_password(key)


def missing_connection_fields(db_type: str, args: dict) -> tuple[list[str], list[str]]:
    """Return (all fields for this db_type, the required ones the caller didn't supply)."""
    fields, required = _CONN_FIELDS.get(db_type, _SQL_CONN_FIELDS)
    missing = [f for f in required if not args.get(f)]
    if "password" in missing and env_db_password(args.get("database", ""), args.get("host", "")):
        missing.remove("password")
    return fields, missing


async def ask_connection_fields(db_type: str, fields: list[str], required: list[str], args: dict) -> dict | None:
    """Ask the user for connection details via MCP elicitation (a form in the
    client). The answers travel client -> server only; the model never sees them.
    Returns the answers, or None if the user declined/cancelled."""
    try:
        session = rt.mcp_session()
    except LookupError as exc:  # called outside an MCP request (e.g. a script)
        raise ValueError(f"Missing connection fields: {', '.join(required)}") from exc
    props: dict[str, dict] = {}
    for f in fields:
        props[f] = {"type": "string", "title": f.replace("_", " ").title()}
        if f == "host" and db_type == "MongoDB":
            props[f]["description"] = (
                "MongoDB Atlas: mongodb+srv://cluster0.xxxx.mongodb.net (no username or password here)"
            )
        default = args.get(f) or {"host": "localhost", "port": _DEFAULT_PORTS.get(db_type)}.get(f)
        if default:
            props[f]["default"] = str(default)
    props["sql_generator"] = {
        "type": "string",
        "title": "Who generates SQL? (server = 1 round-trip, client = MCP AI writes SQL)",
        "enum": ["auto", "server", "client"],
        "default": args.get("sql_generator") or rt.policy.sql_generator() or "auto",
    }
    if rt.hosted and db_type != "SQLite":
        props["trust_cert"] = {
            "type": "boolean",
            "title": "Trust this server's certificate",
            "description": "Only for a database with its own (self-signed) certificate. The connection is still "
            "encrypted, but the server's identity is not checked.",
            "default": False,
        }
    if writes_enabled() and user_can_write():
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
        raise ValueError(
            f"Missing connection fields: {', '.join(required)}. "
            f"Pass them as arguments (this client cannot show a form: {str(e)[:80]})."
        ) from e
    if result.action != "accept" or not result.content:
        return None
    return {k: v for k, v in result.content.items() if v not in (None, "")}


async def ask_sql_generator(default: str = "server") -> str | None:
    """Ask the user via MCP elicitation which AI should generate SQL ('server' vs 'client').
    Returns None silently if the MCP client cannot show forms (e.g. Claude Desktop or tests),
    in which case connect_database prompts the MCP assistant to ask the user in chat."""
    try:
        session = rt.mcp_session()
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
    except Exception as e:  # the client cannot show forms: connect_database asks in chat instead
        log.debug(f"generator form unavailable: {e}")
        return None
    if getattr(result, "action", None) != "accept" or not getattr(result, "content", None):
        return None
    choice = str(result.content.get("sql_generator", "")).strip().lower()
    return choice if choice in ("server", "client") else None


_opening: contextvars.ContextVar[str] = contextvars.ContextVar("sql_agent_opening", default="")


async def connect_database(args: dict) -> list[types.TextContent]:
    """A connection is published step by step (handle, index, knowledge, profile, session) and
    committed when it joins rt.connections. If any step fails first, nothing of it may stay behind,
    above all not an open, logged-in database handle."""
    token = _opening.set("")
    try:
        return await open_connection(args)
    finally:  # any exit before the commit point, by error result or by exception
        if sid := _opening.get():
            discard_connection(sid)
        _opening.reset(token)


@dataclass
class _Target:
    """A connection request after the form and the address check: what to open and how."""

    db_type: str
    host: str
    port: str
    username: str
    password: str
    database: str
    db_path: str
    keyspace: str
    address: Any
    trust_cert: bool
    own_key: str  # an API key given for this connection: memory only, never in the session store
    groq_key: str  # the key query_database will use: the connection's own or the server's
    write_requested: bool
    refresh: bool
    sql_generator: str  # '', 'auto', 'server' or 'client'

    @property
    def name(self) -> str:
        return self.database or self.db_path or self.keyspace


def _target(args: dict, address: Any) -> _Target:
    host = args.get("host", "localhost")
    own_key = args.get("groq_key") or ""
    return _Target(
        db_type=args["db_type"],
        host=host,
        port=args.get("port", ""),
        username=args.get("username", ""),
        # The environment fallback lets the calling agent connect without the secret ever appearing in chat.
        password=args.get("password") or env_db_password(args.get("database", ""), host),
        database=args.get("database", ""),
        db_path=args.get("db_path", ""),
        keyspace=args.get("keyspace", ""),
        address=address,
        trust_cert=rt.hosted and args.get("trust_cert") is True,
        own_key=own_key,
        groq_key=own_key or rt.groq_api_key,
        write_requested=args.get("allow_writes") is True,
        refresh=bool(args.get("refresh")),
        sql_generator=str(args.get("sql_generator", "")).strip().lower(),
    )


async def open_connection(args: dict) -> list[types.TextContent]:
    session_id = resolve_sid(args)
    db_type = args["db_type"]
    if args.get("allow_writes") is True:
        if not writes_enabled():
            return err(
                "Database writes are disabled by an administrator. Ask an administrator to enable them in Settings.",
                ErrCode.PERMISSION_DENIED,
            )
        if not user_can_write():
            return err(
                "Your account does not have permission to enable writes. Ask an administrator to grant sql.write.",
                ErrCode.PERMISSION_DENIED,
            )
    if rt.hosted:
        # Secrets and the certificate choice only through the form: its answers go client -> server,
        # while a tool argument was written by the model and has passed through the LLM provider.
        args = {k: v for k, v in args.items() if k not in _FORM_ONLY_FIELDS}
    fields, missing = missing_connection_fields(db_type, args)
    if missing:
        answers = await ask_connection_fields(db_type, fields, missing, args)
        if answers is None:
            return ok({"connected": False, "cancelled": True, "message": "Connection cancelled by user."})
        args = {**args, **answers}
    try:  # after the form, so a host typed there is checked too, and before any password is resolved or used
        address = check_target(db_type, args.get("host", "localhost"), args.get("db_path", ""), rt.hosted)
    except TargetRefused as refused:
        audit("connect_database", session_id, "", ok=False, error=str(refused))
        return err(str(refused), ErrCode.PERMISSION_DENIED)
    t = _target(args, address)

    # One key per open database; everything is stored under it and the session points at it.
    sid = conn_key(session_id, t.host, t.port, t.name)
    if sid in rt.store and get_session(sid) and not t.refresh:
        return _switch_to(session_id, sid, t, writes_given="allow_writes" in args)
    if sid not in rt.store:
        _opening.set(sid)  # new: discarded by connect_database unless it reaches rt.connections.add below
    try:
        schema = await asyncio.to_thread(_open_and_sample, sid, t)
    except Exception as exc:
        audit("connect_database", sid, "", ok=False, error=type(exc).__name__, extra={"db_type": db_type})
        return err(_connection_failure_message(db_type, exc), ErrCode.CONNECTION_FAILED)
    schema = await _without_empty_tables(sid, t, schema)
    reset_concept_cache()
    state = rt.store.require(sid)
    state.retriever = await asyncio.to_thread(rt.make_retriever, schema)
    ke, kb, context = await _learn(sid, state, t, schema)
    await _save_session(sid, state, t, schema, context, kb)
    choice_needed = await _choose_generator(sid, state, t)
    _apply_tenant_pin(state, schema)
    if t.refresh:
        redis_cache.clear_user_query_cache(sid)
    track_usage(sid, "connects")
    rt.current[session_id] = sid
    _opening.set("")  # committed from here on
    info = rt.connections.add(
        session_id,
        ConnectionInfo(
            sid,
            connection_label(db_type, t.host, t.database, t.db_path, t.keyspace),
            db_type,
            descriptor={
                "db_type": db_type,
                "host": t.host,
                "port": t.port,
                "username": t.username,
                "database": t.database,
                "db_path": t.db_path,
                "keyspace": t.keyspace,
                "secure": rt.hosted,
                "trust_cert": t.trust_cert,
            },
            has_password=bool(t.password),
        ),
    )
    audit(
        "connect_database",
        sid,
        "",
        ok=True,
        extra={
            "db_type": db_type,
            "tls": "local" if not rt.hosted else ("trusted-certificate" if t.trust_cert else "verified"),
        },
    )
    return ok(_connected(session_id, sid, state, schema, context, info.rsa_token, choice_needed, t.sql_generator))


def _switch_to(session_id: str, sid: str, t: _Target, writes_given: bool) -> list[types.TextContent]:
    """The database is already open: point the session at it without reconnecting."""
    rt.current[session_id] = sid
    session = get_session(sid) or {}
    if writes_given:
        session["allow_writes"] = t.write_requested and writes_enabled() and user_can_write()
        redis_cache.set_session(sid, session)
    info = rt.connections.get(session_id, sid)
    return ok(
        {
            "connected": True,
            "switched": True,
            "session_id": session_id,
            "connection": sid,
            "connection_token": info.rsa_token if info else "",
            "db_flavor": t.db_type,
            "table_count": len(redis_cache.get_schema(sid) or {}),
            "write_enabled": bool(session.get("allow_writes")),
            "sql_generator": effective_sql_generator(sid),
            "sql_generator_mode": rt.policy.sql_generator(),
            "message": "Already open - switched to it without reconnecting.",
        }
    )


def _open_and_sample(sid: str, t: _Target) -> dict:
    """Open the database and read its structure (in a thread). One way to open every database:
    database.open_handle, also used by unlock and the web app."""
    handle = rt.open_handle(
        t.db_type,
        host=t.host,
        port=t.port,
        username=t.username,
        password=t.password,
        database=t.database,
        db_path=t.db_path,
        keyspace=t.keyspace,
        secure=rt.hosted,
        trust_cert=t.trust_cert,
        address=t.address,
    )
    try:
        schema, extras = _read_structure(handle, t)
    except Exception:
        close_handle(handle)  # opened but unusable: never strand a logged-in handle
        raise
    state = rt.store.open(sid)
    state.replace_handle(handle)  # a refresh replaces, never strands, the old one
    for name, value in extras.items():
        setattr(state, name, value)
    return schema


def _read_structure(handle: Any, t: _Target) -> tuple[dict, dict]:
    """(schema, what else the state keeps for this kind of database)."""
    if t.db_type == "MongoDB":
        return mongo.sample_schema(handle, timeout_seconds=QUERY_TIMEOUT), {}  # nested paths, no credential fields
    if t.db_type == "Cassandra":
        schema, keys = cql.schema(handle, t.keyspace)
        return schema, {"nosql_keys": keys}
    with handle.connect() as conn:
        schema = fetch_sql_schema(conn, t.db_type)
        return schema, {"foreign_keys": fetch_foreign_keys(conn, t.db_type, schema)}


async def _without_empty_tables(sid: str, t: _Target, schema: dict) -> dict:
    """SQL only: tables without rows stay out of the index, since they cannot answer anything."""
    if t.db_type in ("MongoDB", "Cassandra"):
        return schema

    def _filter():
        with rt.store.handle(sid).connect() as conn:
            return filter_empty_tables(conn, schema, t.db_type)

    try:
        kept, _ = await asyncio.to_thread(_filter)
        return kept
    except Exception as e:  # counting rows is a convenience; the full schema still works
        log.warning(f"could not check for empty tables: {e}")
        return schema


async def _learn(sid: str, state: ConnectionState, t: _Target, schema: dict) -> tuple[KnowledgeEngine, str, dict]:
    """The knowledge engine per user and database (nothing learned, noted or remembered is visible to anyone
    else), the business context (cached; from the model when a key is available; heuristic joins otherwise)
    and the structure profile. Without a key the calling AI reads get_schema and writes SQL for run_sql."""
    kb = f"{safe_owner(rt.request_user())}__" + sid.split("__", 1)[1]
    ke = KnowledgeEngine(user_id=kb)
    await asyncio.to_thread(ke.learn_schema, schema, t.db_type, t.refresh)
    state.knowledge = ke
    db_name = t.database or t.db_path or "default"
    cached = ke.load_sia_context(db_name)
    if cached:
        context = enrich_context_joins(cached, schema)
    elif llm_available(t.groq_key):
        client = rt.make_llm(t.groq_key)
        ke.set_groq_client(client)
        top = state.retriever.retrieve("business schema transactions orders", top_k=20)
        context = enrich_context_joins(await asyncio.to_thread(rt.analyse_schema, top, t.db_type, client), schema)
        ke.save_sia_context(context, db_name)
    else:
        context = enrich_context_joins({}, schema)  # heuristic joins only, no LLM
    ke.learn_context(context)
    kind = dialects.kind(t.db_type)  # every kind: structure now, column values lazily per table
    state.profile = await asyncio.to_thread(
        lambda: dialects.open_profile(
            kind,
            state.handle,
            flavor=t.db_type,
            schema=schema,
            foreign_keys=state.foreign_keys,
            context=context,
            folder=ke.folder,
            keys=state.nosql_keys,
            refresh=t.refresh,
        )
    )
    return ke, kb, context


async def _save_session(sid: str, state: ConnectionState, t: _Target, schema: dict, context: dict, kb: str) -> None:
    memory = await asyncio.to_thread(storage.load_memory, sid)
    redis_cache.set_session(
        sid,
        {
            "db_type": dialects.kind(t.db_type),  # sql / mongo / cassandra: every tool routes on this
            "db_flavor": t.db_type,
            "context": context,
            "memory": memory,
            "allow_writes": t.write_requested and writes_enabled() and user_can_write(),
            "kb": kb,
        },
    )
    redis_cache.set_schema(sid, schema)
    state.table_keys.clear()


async def _choose_generator(sid: str, state: ConnectionState, t: _Target) -> bool:
    """Who writes SQL for this connection. Returns whether the client still has to ask the user."""
    if t.own_key:
        state.llm_key = t.own_key
    if t.sql_generator in ("server", "client"):
        state.sql_generator = t.sql_generator
    elif t.sql_generator == "auto":
        state.sql_generator = ""
    elif not state.sql_generator and rt.policy.sql_generator() == "auto":
        elicited = await ask_sql_generator(default=effective_sql_generator(sid))
        if elicited in ("server", "client"):
            state.sql_generator = elicited
    return not state.sql_generator and rt.policy.sql_generator() == "auto" and not t.sql_generator


def _apply_tenant_pin(state: ConnectionState, schema: dict) -> None:
    pin = pinned_scope(rt.request_user(), schema)
    if pin:  # an administrator fixed this identity's tenant: set before any query can run
        state.scope = pin
    elif state.scope:  # refresh: a new table with the tenant column must be scoped too
        sc = state.scope
        state.scope = make_scope(sc["column"], sc["value"], schema, sc.get("label", "")) or sc


def _connected(
    session_id: str,
    sid: str,
    state: ConnectionState,
    schema: dict,
    context: dict,
    token: str,
    choice_needed: bool,
    requested_generator: str,
) -> dict:
    gen = effective_sql_generator(sid)
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
    session = get_session(sid) or {}
    return {
        "connected": True,
        "session_id": session_id,
        "connection": sid,
        "connection_token": token,
        "db_flavor": session.get("db_flavor"),
        "table_count": len(schema),
        "write_enabled": bool(session.get("allow_writes")),
        "tables": list(schema.keys()),
        "business_type": context.get("business_type", "unknown"),
        "foreign_keys": len(state.foreign_keys),
        "sql_generator": gen,
        "sql_generator_mode": state.sql_generator or rt.policy.sql_generator(),
        "sql_generator_choice_needed": choice_needed,
        "sql_generator_options": {
            "server": "QueryAtlas Server AI · 1 round-trip (call query_database)",
            "client": "MCP Client AI · 2 round-trips (call get_query_context + run_sql)",
        },
        "workflow_hint": (
            "Server SQL generation is active (1 round-trip): call query_database(question) directly to generate, "
            "review, and execute SQL in one step."
            if gen == "server"
            else "Client SQL generation is active (2 round-trips): call get_query_context(question), write the SQL "
            "yourself, and execute it with run_sql(sql, question=...)."
        ),
        "scope_candidates": tenant_column_candidates(schema),
        "hint": " ".join(hints),
    }


async def disconnect(args: dict) -> list[types.TextContent]:
    session_id = resolve_sid(args)
    sid = rt.current.pop(session_id, session_id)  # closes only the current connection
    rt.connections.remove(session_id, sid)
    discard_connection(sid)
    return ok({"disconnected": True, "session_id": sid})
