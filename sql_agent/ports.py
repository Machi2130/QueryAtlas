"""The shapes of what the runtimes reach outside the process with. Structural, so the real database
module, the real provider client and a test double all satisfy them without inheriting anything."""

from __future__ import annotations

from typing import Any, Protocol


class ChatModel(Protocol):
    """The slice of an OpenAI-compatible client the code uses: `client.chat.completions.create(...)`."""

    model: str

    @property
    def chat(self) -> Any: ...


class LLMFactory(Protocol):
    def __call__(self, api_key: str = "", **options: Any) -> ChatModel: ...


class HandleOpener(Protocol):
    """Opens a database and returns its handle (an engine, a pymongo Database or a Cassandra session)."""

    def __call__(self, db_type: str, **details: Any) -> Any: ...


class SchemaAnalyser(Protocol):
    """Reads a shortlist of tables and returns the business context (type, table roles, metrics, joins)."""

    def __call__(self, tables: Any, db_type: str, client: ChatModel) -> dict: ...


class RetrieverFactory(Protocol):
    def __call__(self, schema: dict) -> Any: ...
