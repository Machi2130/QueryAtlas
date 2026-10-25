"""Wiring: the MCP Server instance, the request-bound ports, tool listing, the permission check,
dispatch by tool name, and the stdio entry point."""

from __future__ import annotations

import asyncio
import json
import logging

from mcp import types
from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.stdio import stdio_server

from .. import rbac
from ..errors import ErrCode
from ..limits import mcp_call_allowed
from ..rbac import PermissionDenied
from . import catalog
from .context import ALLOW_DDL, ALLOW_WRITES, active, audit, effective_sql_generator, user_can_write, writes_enabled
from .handlers import capabilities, connect, documents, knowledge, query, writes
from .results import err
from .runtime import rt

log = logging.getLogger(__name__)


class QueryAtlasMCPServer(Server):
    def create_initialization_options(self, notification_options=None, experimental_capabilities=None):
        options = notification_options or NotificationOptions(tools_changed=True)
        return super().create_initialization_options(options, experimental_capabilities)


server = QueryAtlasMCPServer("queryatlas")


def _request_headers():
    try:
        request = server.request_context.request
    except LookupError:
        return None
    return getattr(request, "headers", None)


def _mcp_session():
    return server.request_context.session  # LookupError outside a request


rt.request_headers = _request_headers
rt.mcp_session = _mcp_session
rt.confirm_write = writes.confirm_write
rt.ask_password = connect.ask_password

_policy_watch_task: asyncio.Task | None = None


def tool_policy_fingerprint() -> str:
    doc = rt.policy.settings()
    relevant = {
        "roles": doc.get("roles", {}),
        "users": doc.get("users", {}),
        "default_role": doc.get("default_role"),
        "mcp_local_role": doc.get("mcp_local_role"),
        "mcp_allow_writes": doc.get("mcp_allow_writes", ALLOW_WRITES),
        "mcp_allow_ddl": doc.get("mcp_allow_ddl", ALLOW_DDL),
    }
    return json.dumps(relevant, sort_keys=True, separators=(",", ":"))


async def watch_tool_permissions() -> None:
    rt.policy_fingerprint = tool_policy_fingerprint()
    while True:
        await asyncio.sleep(0.5)
        current = tool_policy_fingerprint()
        if current == rt.policy_fingerprint:
            continue
        rt.policy_fingerprint = current
        for key, (session, _identity, _local) in list(rt.sessions.items()):
            try:
                await session.send_tool_list_changed()
            except Exception as exc:
                log.warning(f"could not notify MCP client of permission change: {exc}")
                rt.sessions.pop(key, None)


def register_mcp_session() -> None:
    global _policy_watch_task
    try:
        session = rt.mcp_session()
    except LookupError:
        return
    rt.sessions[id(session)] = (session, rt.request_user(), rt.is_local_call())
    if _policy_watch_task is None or _policy_watch_task.done():
        _policy_watch_task = asyncio.create_task(watch_tool_permissions())


def check_permission(name: str) -> list[types.TextContent] | None:
    who, local = rt.request_user(), rt.is_local_call()
    try:
        rt.policy.require(who, rbac.MCP, local)
        rt.policy.require(who, catalog.TOOL_PERMISSIONS.get(name, rbac.ASK), local)
    except PermissionDenied as e:
        audit("denied", rt.user_id, "", ok=False, error=f"{name}: {e}")
        return err(str(e), ErrCode.AUTH_FAILED)
    return None


HANDLERS = {
    "connect_database": connect.connect_database,
    "disconnect": connect.disconnect,
    "query_database": query.query_database,
    "run_sql": query.run_sql,
    "get_query_context": query.query_context,
    "get_schema": query.get_schema,
    "validate_sql": query.validate_query,
    "explain_query": query.explain_query,
    "suggest_chart": query.chart,
    "execute_write": writes.execute_write,
    "add_note": knowledge.add_note,
    "list_notes": knowledge.list_notes,
    "delete_note": knowledge.delete_note,
    "mark_query": knowledge.mark_query,
    "set_scope": knowledge.set_scope,
    "set_sql_generator": knowledge.set_sql_generator,
    "import_query_history": knowledge.import_query_history,
    "import_queries": knowledge.import_queries,
    "describe_capabilities": capabilities.describe_capabilities,
    "upload_document": documents.upload_document,
    "search_documents": documents.search_documents,
    "list_documents": documents.list_documents,
}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    register_mcp_session()
    tools = catalog.tools(effective_sql_generator(active({})) == "server")
    if not (writes_enabled() and user_can_write()):
        tools = [tool for tool in tools if tool.name != "execute_write"]
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        denied = check_permission(name)
        if denied:
            return denied
        if not rt.is_local_call():  # hosted: one identity cannot flood the server or a customer's database
            allowed, retry_after = mcp_call_allowed(rt.request_user())
            if not allowed:
                audit("rate_limited", rt.request_user(), "", ok=False, error=name)
                return err(
                    f"Too many requests. Try again in {retry_after}s.",
                    ErrCode.RATE_LIMITED,
                    {"retry_after": retry_after},
                )
        if name in catalog.CONNECTION_TOOLS:
            locked = await connect.ensure_unlocked(arguments)
            if locked:
                return locked
        handler = HANDLERS.get(name)
        if handler is None:
            return err(f"Unknown tool: {name}", ErrCode.UNKNOWN)
        return await handler(arguments)
    except ValueError as e:
        return err(str(e), ErrCode.NOT_CONNECTED)
    except Exception as e:
        return err(f"Unexpected error: {e}", ErrCode.UNKNOWN)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(NotificationOptions(tools_changed=True)),
        )
