"""describe_capabilities: what is connected, which tools exist, what writes are allowed, the error codes."""

from __future__ import annotations

from mcp import types

from ... import cache as redis_cache
from ... import metrics
from ...errors import ErrCode
from ...llm import LLM_MODEL
from ..context import (
    ddl_enabled,
    effective_sql_generator,
    get_session,
    resolve_sid,
    usage_of,
    user_can_write,
    writes_enabled,
)
from ..results import ok
from ..runtime import VERSION, rt


async def describe_capabilities(args: dict) -> list[types.TextContent]:
    sid = resolve_sid(args)
    current = rt.current.get(sid)
    session = get_session(current) if current else None
    connected_db = None
    if session:
        connected_db = {
            "connection": current,
            "db_flavor": session.get("db_flavor"),
            "business_type": session.get("context", {}).get("business_type", "unknown"),
            "has_schema": bool(redis_cache.get_schema(current)),
        }
    prefix = f"{sid}__"
    open_connections = [k for k in rt.store if k.startswith(prefix)]
    gen = effective_sql_generator(current or sid)
    return ok(
        {
            "server": "queryatlas",
            "version": VERSION,
            "session_id": sid,
            "connected": connected_db is not None,
            "current_session": connected_db,
            "open_connections": open_connections,
            "sql_generator": gen,
            "sql_generator_mode": rt.policy.sql_generator(),
            "usage": usage_of(sid),
            "context_savings": metrics.summary(current) if current else {"questions": 0},
            "write_access": {
                "server_enabled": writes_enabled(),
                "role_permitted": user_can_write(),
                "connection_opted_in": bool(session and session.get("allow_writes")),
                "enabled": bool(writes_enabled() and user_can_write() and session and session.get("allow_writes")),
                "ddl_enabled": bool(ddl_enabled() and user_can_write() and session and session.get("allow_writes")),
            },
            "supported_databases": ["SQLite", "MySQL", "MariaDB", "PostgreSQL", "SQL Server", "MongoDB", "Cassandra"],
            "tools": [
                {"name": "connect_database", "requires_session": False},
                {"name": "query_database", "requires_session": True, "needs_llm": True, "llm_model": LLM_MODEL},
                {"name": "validate_sql", "requires_session": True, "databases": ["sql", "mongo", "cassandra"]},
                {"name": "run_sql", "requires_session": True, "databases": ["sql", "mongo", "cassandra"]},
                {"name": "get_query_context", "requires_session": True},
                {"name": "get_schema", "requires_session": True},
                {"name": "add_note", "requires_session": True},
                {"name": "list_notes", "requires_session": True},
                {"name": "delete_note", "requires_session": True},
                {"name": "mark_query", "requires_session": True},
                {"name": "set_scope", "requires_session": True},
                {"name": "set_sql_generator", "requires_session": True},
                {"name": "import_query_history", "requires_session": True, "databases": ["sql", "mongo"]},
                {"name": "import_queries", "requires_session": True},
                {"name": "explain_query", "requires_session": True, "databases": ["sql", "mongo"]},
                {"name": "suggest_chart", "requires_session": False},
                {
                    "name": "execute_write",
                    "requires_session": True,
                    "databases": ["sql", "mongo", "cassandra"],
                    "developer_mode": True,
                    "enabled": writes_enabled() and user_can_write(),
                    "connection_opted_in": bool(session and session.get("allow_writes")),
                    "ddl_enabled": ddl_enabled() and user_can_write(),
                },
                {"name": "disconnect", "requires_session": True},
                {"name": "describe_capabilities", "requires_session": False},
            ],
            "error_codes": {
                ErrCode.NOT_CONNECTED: "No active session — call connect_database first",
                ErrCode.QUERY_FAILED: "Database returned an error executing the query",
                ErrCode.AUTH_FAILED: "Authentication / credential error",
                ErrCode.PERMISSION_DENIED: "The current user, server settings, or connection does not allow this action",
                ErrCode.CONNECTION_FAILED: "Database connection failed; credentials are not returned in the error",
                ErrCode.UNSUPPORTED_DB: "db_type is not one of the supported databases",
                ErrCode.INVALID_SQL: "Generated SQL failed validation",
                ErrCode.TIMEOUT: "Query exceeded the timeout limit",
                ErrCode.NEEDS_CLARIFICATION: "The question names nothing in this database: ask the user what they mean",
                ErrCode.UNKNOWN: "Unexpected server error",
            },
        }
    )
