"""Entry point of the MCP product at its historical import path, for the root launcher
(mcp_server.py), the console script and the local tests. The implementation lives in sql_agent.mcp."""

from __future__ import annotations

from . import cache as redis_cache  # noqa: F401  (tests inspect the session cache through this module)
from .database import _connection_failure_message, close_handle, open_handle  # noqa: F401
from .errors import ErrCode  # noqa: F401
from .mcp.context import kb_dir as _kb_dir  # noqa: F401
from .mcp.context import knowledge_of as _knowledge_of  # noqa: F401
from .mcp.context import load_notes as _load_notes  # noqa: F401
from .mcp.context import profile_service as _profile_service  # noqa: F401
from .mcp.context import resolve_sid as _resolve_sid  # noqa: F401
from .mcp.context import retriever_of as _retriever_of  # noqa: F401
from .mcp.handlers.capabilities import describe_capabilities as _describe_capabilities  # noqa: F401
from .mcp.handlers.connect import env_db_password as _env_db_password  # noqa: F401
from .mcp.handlers.query import query_database as _query  # noqa: F401
from .mcp.install import cli  # noqa: F401
from .mcp.runtime import VERSION as __version__  # noqa: F401
from .mcp.runtime import rt
from .mcp.server import call_tool, list_tools, main, server  # noqa: F401
from .mcp.server import watch_tool_permissions as _watch_tool_permissions  # noqa: F401
from .mcp.transport import build_http_mcp_app  # noqa: F401

MCP_USER_ID = rt.user_id
_store = rt.store
_current = rt.current
