"""What the process holds in memory for one open database connection, and the store that owns it.

Both products (the MCP server and the web app) keep the same things per connection. They used to
keep them in fifteen parallel dicts each; this is the one place now. Ownership: the store is
mutated on the event loop only (connect, refresh, discard). Worker threads receive a state object
or a handle and read it; they never add or remove entries.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .database import close_handle

NOT_CONNECTED = "Not connected. Call connect_database first."


@dataclass
class ConnectionState:
    key: str
    handle: Any = None  # SQLAlchemy engine, pymongo Database or Cassandra session
    retriever: Any = None  # SchemaRetriever
    knowledge: Any = None  # KnowledgeEngine
    profile: Any = None  # ProfileService
    foreign_keys: list = field(default_factory=list)
    nosql_keys: dict | None = None  # Cassandra partition / clustering keys
    table_keys: dict = field(default_factory=dict)  # table -> primary key, not-null, comments (read lazily)
    scope: dict | None = None  # tenant filter: {column, value, label, tables, pinned}
    llm_key: str = ""  # key given at connect time; memory only, never persisted
    sql_generator: str = ""  # 'server' | 'client' | '' (policy default)
    vocab: Any = None  # words the database is made of, for routing between databases
    usage: dict[str, int] = field(default_factory=lambda: {"queries": 0, "errors": 0, "charts": 0, "connects": 0})

    @property
    def connected(self) -> bool:
        return self.handle is not None  # a pymongo Database refuses truth testing: compare with None

    def replace_handle(self, handle: Any) -> None:
        """A reconnect or refresh replaces the handle and never strands the old one."""
        old, self.handle = self.handle, handle
        if old is not None and old is not handle:
            close_handle(old)


class ConnectionStore:
    def __init__(self) -> None:
        self._items: dict[str, ConnectionState] = {}

    def get(self, key: str) -> ConnectionState | None:
        return self._items.get(key)

    def open(self, key: str) -> ConnectionState:
        """The state for this key, created empty if it is new."""
        state = self._items.get(key)
        if state is None:
            state = self._items[key] = ConnectionState(key)
        return state

    def require(self, key: str) -> ConnectionState:
        state = self._items.get(key)
        if state is None or not state.connected:
            raise ValueError(NOT_CONNECTED)
        return state

    def handle(self, key: str) -> Any:
        return self.require(key).handle

    def __contains__(self, key: object) -> bool:
        state = self._items.get(key)  # type: ignore[arg-type]
        return state is not None and state.connected

    def __iter__(self) -> Iterator[str]:
        return (k for k, s in self._items.items() if s.connected)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def discard(self, key: str) -> None:
        """Forget everything held for one connection and close its handle (MongoDB and Cassandra
        clients keep their logged-in connections open until closed)."""
        state = self._items.pop(key, None)
        if state is not None:
            close_handle(state.handle)
