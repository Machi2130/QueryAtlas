"""Request bodies. FastAPI validates these at the HTTP boundary and documents them at /docs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConnectRequest(BaseModel):
    db_type: str
    groq_key: str = ""  # optional: becomes this person's own AI key (see /api/llm); empty = their own or the server's
    host: str = "localhost"
    port: str = ""
    username: str = ""
    password: str = ""
    database: str = ""
    keyspace: str = ""
    db_path: str = ""
    trust_cert: bool = False  # the person accepts a self-signed certificate: still encrypted, identity unchecked


class LLMRequest(BaseModel):
    provider: str = "groq"
    api_key: str = ""  # blank = keep the key already held for this provider
    base_url: str = ""  # only for provider "custom" (administrators)
    model: str = ""
    planner_model: str = ""


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    connection_id: str = ""  # the user picked the database (after a "which one?"); skips routing
    page: int = Field(default=1, ge=1, description="1-based page number")
    page_size: int = Field(default=50, ge=1, le=500)


class StreamQueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    connection_id: str = ""


class UnlockRequest(BaseModel):
    password: str


class ExplainRequest(BaseModel):
    sql: str


class FeedbackRequest(BaseModel):
    question: str
    sql: str = ""  # empty: taken from the conversation (people without view.sql never see it)
    session_id: str = ""
    is_correct: bool = True
    correction: str | None = None


class ChartRequest(BaseModel):
    columns: list[str]
    rows: list[list[Any]]


class HybridRequest(BaseModel):
    question: str
    session_id: str = "default"
    connection_id: str = ""
    url: str = ""  # optional: ingest a URL on-the-fly before answering


class AuthRequest(BaseModel):
    username: str
    password: str
