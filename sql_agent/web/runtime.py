"""The web app's collaborators in one injectable object (the MCP server has its own in mcp.runtime).

Routers and the pipeline depend on `wrt`, never on module globals: the policy, the connection store,
each person's open databases and AI settings, the limits, and the ports that reach outside the process.
A test replaces one attribute here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import settings
from ..connections import ConnectionRegistry
from ..database import open_handle
from ..docstore import DocStore
from ..llm import LLMClient, LLMSettings, list_models
from ..ports import HandleOpener, LLMFactory, RetrieverFactory, SchemaAnalyser
from ..profile import DatabaseRouter
from ..prompts import run_schema_agent
from ..rbac import POLICY, Policy
from ..retrieval import SchemaRetriever
from ..state import ConnectionStore

SESSIONS_DIR = ".chroma_data/_sessions"


@dataclass
class WebRuntime:
    policy: Policy = field(default_factory=lambda: POLICY)
    store: ConnectionStore = field(default_factory=ConnectionStore)
    router: DatabaseRouter = field(default_factory=DatabaseRouter)
    connections: ConnectionRegistry = field(init=False)  # each person's open databases; the chat asks the active one
    # Each person's own AI provider, key and model (see /api/llm). Memory only: a key is never put in the
    # session store, Redis, a file or a log, and is gone when the server stops.
    llm_settings: dict[str, LLMSettings] = field(default_factory=dict)
    docstores: dict[str, DocStore] = field(default_factory=dict)
    max_rows: int = field(default_factory=lambda: settings.max_rows)
    query_timeout: int = field(default_factory=lambda: settings.query_timeout_seconds)
    sessions_dir: str = SESSIONS_DIR

    # ── ports ──
    open_handle: HandleOpener = open_handle
    analyse_schema: SchemaAnalyser = run_schema_agent
    make_retriever: RetrieverFactory = SchemaRetriever
    make_llm: LLMFactory = LLMClient
    list_models: Callable[[str, str], list] = list_models

    def __post_init__(self) -> None:
        self.connections = ConnectionRegistry(lambda: self.policy.idle_lock_seconds())


wrt = WebRuntime()
