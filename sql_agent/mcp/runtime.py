"""The MCP server's collaborators in one injectable object.

Handlers depend on `rt`, never on module globals: the policy, the connection store, the live
sessions, the hosted flag and the ports that reach outside the process (who is calling, the
elicitation forms, opening a database, the schema analyser, the retriever and the LLM client).
A test replaces one attribute on `rt`; the transport layer sets the request-bound ports once.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..connections import ConnectionInfo, ConnectionRegistry
from ..database import open_handle
from ..llm import LLMClient, llm_api_key
from ..ports import HandleOpener, LLMFactory, RetrieverFactory, SchemaAnalyser
from ..prompts import run_schema_agent
from ..rbac import POLICY, Policy
from ..retrieval import SchemaRetriever
from ..state import ConnectionStore

VERSION = "1.0.0"


def _no_request_headers() -> Mapping[str, str] | None:
    return None


def _no_session() -> Any:
    raise LookupError("not inside an MCP request")


@dataclass
class Runtime:
    policy: Policy = field(default_factory=lambda: POLICY)
    store: ConnectionStore = field(default_factory=ConnectionStore)
    current: dict[str, str] = field(default_factory=dict)  # session -> the connection it works on
    sessions: dict[int, tuple[Any, str, bool]] = field(default_factory=dict)  # live MCP sessions, for tool-list refresh
    hosted: bool = False  # other people's requests reach this process
    user_id: str = field(default_factory=lambda: settings.mcp_user_id)
    token_users: dict[str, str] = field(default_factory=lambda: settings.mcp_token_users)
    audit_log: str = field(default_factory=lambda: settings.audit_log)
    groq_api_key: str = field(default_factory=llm_api_key)  # server-side key; only query_database needs it
    policy_fingerprint: str = ""
    connections: ConnectionRegistry = field(init=False)

    # ── ports: the transport binds the request-bound ones, tests substitute any of them ──
    request_headers: Callable[[], Mapping[str, str] | None] = _no_request_headers
    mcp_session: Callable[[], Any] = _no_session  # the session that can show forms
    request_user: Callable[[], str] = field(init=False)
    is_local_call: Callable[[], bool] = field(init=False)
    confirm_write: Callable[[str, str, bool], Awaitable[bool]] = field(init=False)
    ask_password: Callable[[ConnectionInfo], Awaitable[str | None]] = field(init=False)
    open_handle: HandleOpener = open_handle
    analyse_schema: SchemaAnalyser = run_schema_agent
    make_retriever: RetrieverFactory = SchemaRetriever
    make_llm: LLMFactory = LLMClient

    def __post_init__(self) -> None:
        self.connections = ConnectionRegistry(lambda: self.policy.idle_lock_seconds())
        self.request_user = self.default_request_user
        self.is_local_call = self.default_is_local_call

    def default_request_user(self) -> str:
        """Who is calling: the name behind the bearer token over HTTP (self-service tokens in
        rbac.json or static MCP_AUTH_TOKENS), else the local MCP user."""
        headers = self.request_headers()
        if headers is not None:
            token = headers.get("authorization", "").removeprefix("Bearer ").strip()
            return self.policy.user_for_mcp_token(token) or self.token_users.get(token, "unknown-token")
        return self.user_id

    def default_is_local_call(self) -> bool:
        """stdio: the server runs on the caller's own machine (no HTTP request)."""
        return self.request_headers() is None


rt = Runtime()
