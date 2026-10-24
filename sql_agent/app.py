
from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import sys
import threading
import time

import httpx
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

from . import cache as redis_cache
from . import cql, dialects, mongo
from . import storage
from .auth import DEV_MODE, get_current_user
from .connections import (ConnectionInfo, ConnectionRegistry, TargetRefused, check_target, connection_key,
                          connection_label, safe_owner)
from . import rbac
from .rbac import POLICY, SettingsError
from .database import (SQL_TYPES, _connection_failure_message, close_handle, explain_statement, open_handle, fetch_foreign_keys,
                       fetch_read, fetch_sql_schema,
                       filter_empty_tables, safe_rows, serialize_value)
from .docstore import DocStore
from .limits import BodyLimit, SecurityHeaders, record_sign_in, record_sign_up, sign_in_blocked, sign_up_blocked
from .knowledge import KnowledgeEngine
from .llm import (LLM_MODEL, PROVIDERS, LLMClient, LLMError, LLMSettings, list_models, llm_api_key, llm_available,
                  provider_problem)
from .presentation import (CANT_FIND, FRIENDLY_ERRORS, hit_auto_limit, suggest_chart_rule, summarize_result,
                           topic_suggestions)
from .profile import DatabaseRouter, ProfileService, database_vocabulary
from .prompts import (HISTORY_TURNS, attempt_fix, build_sql_messages, clean_query, improve_sql, enrich_context_joins,
                      run_schema_agent, said_not_answerable, update_memory)
from .query_planner import reset_concept_cache
from .retrieval import SchemaRetriever, _warm_embed_model
from .scope import apply_scope, pinned_scope
from .sqlreview import SqlReviewer
from .validation import QueryValidationError, validate_sql

load_dotenv()



# ═══════════════════════════════════════════════════════════════
#  CONCURRENCY POOLS
#
#  ThreadPoolExecutor  — I/O-bound: DB queries, Redis calls, Groq HTTP
#    Threads release the GIL while waiting on sockets/network, so
#    multiple threads run concurrently even in CPython.
#
#  ProcessPoolExecutor — CPU-bound: SentenceTransformer.encode(), FAISS
#    These hold the GIL; threads cannot parallelize them.
#    Spawning separate interpreter processes bypasses the GIL entirely.
# ═══════════════════════════════════════════════════════════════

_THREAD_POOL  = ThreadPoolExecutor(max_workers=int(os.getenv("THREAD_WORKERS", 10)))
_PROCESS_POOL = ProcessPoolExecutor(max_workers=int(os.getenv("PROCESS_WORKERS", 2)))

DB_QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT_SECONDS", 30))


# ─────────────────────────────────────────────────────────────
#  LIFESPAN — startup / shutdown
#  Replaces Flask's @app.before_first_request.
#  Guarantees clean pool shutdown so in-flight tasks finish.
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEV_MODE:
        print("[sql-agent] WARNING: DEV_MODE=true: the web app accepts any bearer text as a user id. Use it on your own machine only.", file=sys.stderr)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_PROCESS_POOL, _warm_embed_model)
    yield
    _THREAD_POOL.shutdown(wait=False)
    _PROCESS_POOL.shutdown(wait=False)


app = FastAPI(title="AI Query Agent", version="2.0.0", lifespan=lifespan)
# The UI is served by this app (same origin), so no other site may call the API with a person's
# credentials unless listed here (e.g. a separately hosted front end).
_cors_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(BodyLimit)   # MAX_REQUEST_MB: no request body is held in memory past this size
app.add_middleware(SecurityHeaders)
if os.path.isdir("static"):  # web UI assets; absent when only the MCP server is deployed
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ═══════════════════════════════════════════════════════════════
#  PYDANTIC REQUEST MODELS
#  FastAPI validates these at the HTTP boundary automatically and
#  generates OpenAPI docs at /docs — no extra code needed.
# ═══════════════════════════════════════════════════════════════

class ConnectRequest(BaseModel):
    db_type:  str
    groq_key: str = ""   # optional: becomes this person's own AI key (see /api/llm); empty = their own or the server's
    host:     str = "localhost"
    port:     str = ""
    username: str = ""
    password: str = ""
    database: str = ""
    keyspace: str = ""
    db_path:  str = ""
    trust_cert: bool = False   # the person accepts a self-signed certificate: still encrypted, identity unchecked


class LLMRequest(BaseModel):
    provider:      str = "groq"
    api_key:       str = ""     # blank = keep the key already held for this provider
    base_url:      str = ""     # only for provider "custom" (administrators)
    model:         str = ""
    planner_model: str = ""


class QueryRequest(BaseModel):
    question:   str
    session_id: str = "default"
    connection_id: str = ""   # the user picked the database (after a "which one?"); skips routing
    page:       int = Field(default=1, ge=1, description="1-based page number")
    page_size:  int = Field(default=50, ge=1, le=500)


class StreamQueryRequest(BaseModel):
    question:   str
    session_id: str = "default"
    connection_id: str = ""


class UnlockRequest(BaseModel):
    password: str


class ExplainRequest(BaseModel):
    sql: str


class FeedbackRequest(BaseModel):
    question:   str
    sql:        str = ""           # empty: taken from the conversation (people without view.sql never see it)
    session_id: str = ""
    is_correct: bool = True
    correction: Optional[str] = None


class ChartRequest(BaseModel):
    columns: list[str]
    rows:    list[list[Any]]


class HybridRequest(BaseModel):
    question:   str
    session_id: str = "default"
    connection_id: str = ""
    url:        str = ""   # optional: ingest a URL on-the-fly before answering


# ═══════════════════════════════════════════════════════════════
#  ASYNC HELPERS
# ═══════════════════════════════════════════════════════════════

async def run_in_thread(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_THREAD_POOL, fn, *args)


async def run_in_process(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_PROCESS_POOL, fn, *args)


MAX_ROWS = int(os.getenv("MAX_ROWS", os.getenv("MCP_MAX_ROWS", 5000)))


async def exec_sql_async(engine, query: str) -> tuple[list, list]:
    """Execute SQL in a thread with a hard timeout and a row cap (never pulls a whole table)."""
    return await asyncio.wait_for(run_in_thread(fetch_read, engine, query, MAX_ROWS, DB_QUERY_TIMEOUT),
                                  timeout=DB_QUERY_TIMEOUT)


# ═══════════════════════════════════════════════════════════════
#  PROCESS-LEVEL CACHES
# ═══════════════════════════════════════════════════════════════

_engines:    dict[str, Any] = {}
_mongos:     dict[str, Any] = {}
_cassandras: dict[str, Any] = {}
_retrievers: dict[str, Any] = {}
_knowledge:  dict[str, Any] = {}
_docstores:  dict[str, Any] = {}
_fks:        dict[str, list] = {}   # real foreign keys per user's connection
_profiles:   dict[str, ProfileService] = {}   # what we know about each open database
_connections = ConnectionRegistry(POLICY.idle_lock_seconds)   # a user's open databases; the chat asks the active one
_nosql_keys: dict[str, dict] = {}   # Cassandra partition/clustering keys per connection
_DB_STATE    = (_engines, _mongos, _cassandras, _retrievers, _knowledge, _fks, _profiles, _nosql_keys)
_router      = DatabaseRouter()
_vocab:      dict[str, Any] = {}   # per connection: words it is made of, for routing
_CONNECTABLE = {"SQLite", "MySQL", "MariaDB", "PostgreSQL", "SQL Server", "MongoDB", "Cassandra"}

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_retriever(ck: str) -> SchemaRetriever | None:
    if ck in _retrievers:
        return _retrievers[ck]
    schema = redis_cache.get_schema(ck)
    if schema:
        r = SchemaRetriever(schema)
        _retrievers[ck] = r
        return r
    return None


def _active_key(user_id: str, client_token: str = "") -> str:
    """The connection the chat box is talking to; verified via the server's RSA token
    and refused while it is idle-locked."""
    info = _connections.active(user_id)
    if info is None:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")
    try:
        verified_key = _connections.verify(user_id, info.key, client_token)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    _require_unlocked(user_id, info)
    _connections.touch(verified_key)
    return verified_key


def _require_unlocked(user_id: str, info: ConnectionInfo) -> None:
    if _connections.is_locked(user_id, info.key):
        raise HTTPException(status_code=423, detail={
            "message": f"{info.label} has not been used for a while. Enter its password to continue.",
            "locked": {"id": info.key, "label": info.label}})


def _closed_message(bound: dict | None) -> str:
    kind = f"a {bound['db_type']} database" if bound and bound.get("db_type") else "a database"
    return f"This conversation used {kind} that is not open. Connect to it again (enter its details) to continue."


def _bound_label(user_id: str, bound: dict | None) -> str:
    """The conversation's database name while it is open (from memory); only its type otherwise."""
    info = _connections.get(user_id, bound["id"]) if bound else None
    return info.label if info else (bound or {}).get("db_type", "")


def _vocabulary(ck: str):
    if ck not in _vocab:
        _vocab[ck] = database_vocabulary(redis_cache.get_schema(ck) or {},
                                         _profiles[ck].profile if ck in _profiles else None,
                                         _knowledge[ck].notes() if ck in _knowledge else [])
    return _vocab[ck]


def _resolve_for_question(user_id: str, question: str, session_id: str, picked: str = "",
                          client_token: str = "") -> str:
    """Which database answers: the conversation's own (a conversation stays on one
    database), the one the user just picked, or for a new conversation the one the
    question is about. Close calls are put to the user. Then the idle lock applies."""
    bound = (load_sessions(user_id).get(session_id) or {}).get("connection")
    target = (bound or {}).get("id") or picked
    if target:
        if _connections.get(user_id, target) is None:
            raise HTTPException(status_code=409, detail={
                "message": _closed_message(bound), "reconnect": bound})
        _connections.activate(user_id, target)
    elif len(_connections.list(user_id)) > 1:
        active = _connections.active(user_id)
        decision = _router.route(question, {c.key: _vocabulary(c.key) for c in _connections.list(user_id)},
                                 active.key if active else None)
        if decision.key is None:
            labels = {c.key: c.label for c in _connections.list(user_id)}
            raise HTTPException(status_code=409, detail={
                "message": "Which database is this about?",
                "choose": [{"id": k, "label": labels[k]} for k, score in decision.scores if score > 0]})
        if decision.key:
            _connections.activate(user_id, decision.key)
    return _active_key(user_id, client_token)


def _close_connection(key: str) -> None:
    redis_cache.delete_session(key)
    redis_cache.clear_user_query_cache(key)
    handles = [m.get(key) for m in (_engines, _mongos, _cassandras)]
    for state in (*_DB_STATE, _vocab):
        state.pop(key, None)
    for handle in handles:   # after the state is gone, so nothing picks up a closing handle
        close_handle(handle)


# Each person's own AI provider, key and model (see /api/llm). In memory only: a key is never put in the
# session store, Redis, a file or a log, and is gone when the server stops. Without one, the server's
# default (LLM_API_KEY / LLM_BASE_URL) is used when it has one.
_llm_settings: dict[str, LLMSettings] = {}


def _llm_ready(user_id: str) -> bool:
    return user_id in _llm_settings or llm_available(llm_api_key())


def _client_from(choice: LLMSettings) -> LLMClient:
    # reasoning_effort is left out: not every model accepts it
    return LLMClient(api_key=choice.api_key, base_url=choice.url, model=choice.model,
                     planner_model=choice.planner_model, reasoning_effort="")


def _llm_client(user_id: str) -> LLMClient:
    mine = _llm_settings.get(user_id)
    if mine:
        return _client_from(mine)
    if not llm_available(llm_api_key()):
        raise HTTPException(status_code=400, detail="No AI is set up. Choose a provider and add your API key, "
                                                    "or ask your admin to set LLM_API_KEY.")
    return LLMClient(llm_api_key())


# ── Conversations: what a user asked and was answered, to reopen from the sidebar ──
# One JSON file per user next to the rest of the server-side knowledge (no customer data
# in browser storage, works without Supabase). Saved answers re-display without a query.
SESSIONS_DIR      = os.path.join(".chroma_data", "_sessions")
SESSION_ROWS      = 50     # rows kept per saved answer (the full result can be asked again)
MAX_SESSIONS      = 50
MAX_SESSION_TURNS = 100
_sessions_lock    = threading.Lock()   # ponytail: one lock for all users' files; per-user locks if it ever contends


def _sessions_path(user_id: str) -> str:
    return os.path.join(SESSIONS_DIR, safe_owner(user_id) + ".json")


def load_sessions(user_id: str) -> dict:
    try:
        with open(_sessions_path(user_id), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _write_sessions(user_id: str, sessions: dict) -> None:
    path = _sessions_path(user_id)
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sessions, f, default=str)
    os.replace(tmp, path)   # never leave a half-written file behind


def save_session_turn(user_id: str, session_id: str, turn: dict, connection: dict | None = None) -> None:
    """Append one answered question; the first question gives the conversation its title
    and its database (opening the conversation later switches back to it)."""
    with _sessions_lock:
        sessions = load_sessions(user_id)
        s = sessions.setdefault(session_id, {"title": turn["question"][:60], "messages": []})
        if connection and not s.get("connection"):
            s["connection"] = connection
        s["messages"] = (s["messages"] + [turn])[-MAX_SESSION_TURNS:]
        s["updated"] = datetime.datetime.now().isoformat(timespec="milliseconds")   # orders two answers in one second
        newest = sorted(sessions.items(), key=lambda kv: kv[1].get("updated", ""), reverse=True)[:MAX_SESSIONS]
        _write_sessions(user_id, dict(newest))


def _web_scope(user_id: str, schema: dict) -> dict | None:
    """The web app has no set_scope: a user's tenant comes only from SCOPE_PINS."""
    return pinned_scope(user_id, schema)


def _web_cache_flavor(user_id: str, schema: dict, flavor: str) -> str:
    """Include active tenant scope in the query-cache key so two scopes never share cached rows."""
    sc = _web_scope(user_id, schema)
    return f"{flavor}|scope:{sc['column']}={sc['value']}" if sc else flavor


def _prepare_nosql(query: str, db_type: str, schema: dict, user_id: str, flavor: str) -> tuple[str, str]:
    """A generated MongoDB / CQL query, validated, then the tenant filter of a pinned person added by
    the server, as for SQL. Returns (as written, as run): the first is what gets remembered."""
    validated = dialects.validate_read(db_type, query, schema, flavor)
    scoped, _ = dialects.scope_read(db_type, validated, _web_scope(user_id, schema), flavor, schema)
    return dialects.as_text(db_type, validated), dialects.as_text(db_type, scoped)


async def _run_nosql(ck: str, db_type: str, query_text: str) -> tuple[list, list]:
    """Row cap and time limit (MongoDB stops it server-side, maxTimeMS); credential fields removed."""
    handle = (_mongos if db_type == "mongo" else _cassandras)[ck]
    prepared = json.loads(query_text) if db_type == "mongo" else query_text
    return await asyncio.wait_for(run_in_thread(dialects.run_read, db_type, handle, prepared, MAX_ROWS,
                                                DB_QUERY_TIMEOUT), timeout=DB_QUERY_TIMEOUT)


def _prepare_web_sql(sql: str, schema: dict, flavor: str, question: str, user_id: str) -> str:
    """Validate, then enforce the user's tenant scope: the single gate before any web execution."""
    sql = validate_sql(sql, schema, flavor, question)
    return apply_scope(sql, flavor, _web_scope(user_id, schema))[0]


def _evt(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _reviewer(ck: str, flavor: str) -> SqlReviewer:
    """Reviews generated SQL against this database's profile (keys, indexes, sizes)."""
    return SqlReviewer(flavor, _profiles[ck].profile if ck in _profiles else None)


def _require(user_id: str, permission: str) -> None:
    """403 unless the settings file gives this person's role the permission."""
    if not POLICY.can(user_id, permission):
        raise HTTPException(status_code=403, detail={
            "message": f"Your role ({POLICY.role_of(user_id)}) does not allow this.", "permission": permission})


_TECHNICAL = ("query", "generated_sql", "sql", "tables_used", "error", "review",
              "token")   # the query as it streams in: SQL text, so also only with view.sql


def _for_viewer(user_id: str, data: dict) -> dict:
    """What this person may see of a payload: SQL, tables used and technical errors only
    with view.sql; the raw schema only with view.tables. Decided by the server, so an API
    call cannot get around the page hiding it."""
    out = dict(data)
    if not POLICY.can(user_id, rbac.VIEW_SQL):
        for k in _TECHNICAL:
            out.pop(k, None)
        if "messages" in out:
            out["messages"] = [{k: v for k, v in m.items() if k not in _TECHNICAL} for m in out["messages"]]
        if isinstance(out.get("sql_result"), dict):
            out["sql_result"] = {k: v for k, v in out["sql_result"].items() if k not in _TECHNICAL}
    if not POLICY.can(user_id, rbac.VIEW_TABLES):
        out.pop("schema", None)
    return out


# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────────────────────
#  CONNECT — SSE streaming with async generator
#
#  FastAPI SSE: StreamingResponse wraps an async generator.
#  Each blocking step (DB connect, LLM call) is awaited via
#  run_in_thread() so the event loop stays free for other requests.
#  Flask equivalent used stream_with_context() with sync generators.
# ─────────────────────────────────────────────────────────────

@app.post("/api/connect")
async def connect(req: ConnectRequest, user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.CONNECT)
    try:   # the web app is always hosted: other people choose the target
        address = check_target(req.db_type, req.host, req.db_path, hosted=True)
    except TargetRefused as refused:
        raise HTTPException(status_code=403, detail={"message": str(refused)})
    # Each database gets its own key: connecting a second one keeps the first open.
    ck = connection_key(user_id, req.db_type, req.host, req.port, req.database or req.db_path or req.keyspace)
    label = connection_label(req.db_type, req.host, req.database, req.db_path, req.keyspace)

    async def generate() -> AsyncGenerator[str, None]:
        # The connection is published step by step (engine, index, knowledge, session) and committed
        # only when it joins the person's connection list. A failure in between must not leave an
        # open, logged-in database handle behind that no list knows about.
        was_open = _connections.get(user_id, ck) is not None
        try:
            yield _evt({"phase": "init", "msg": "Initializing AI engine..."})
            if req.db_type not in _CONNECTABLE:
                raise ValueError(f"Unknown db_type: {req.db_type}")
            # Optional: without an LLM the database still connects; verified questions,
            # charts and the schema analysis cache work, new questions need one.
            if req.groq_key and user_id not in _llm_settings:   # older clients: a key typed at connect
                _llm_settings[user_id] = LLMSettings(provider="groq", api_key=req.groq_key)
            client = _llm_client(user_id) if _llm_ready(user_id) else None
            schema_dict: dict = {}
            db_flavor   = req.db_type

            yield _evt({"phase": "db", "msg": f"Connecting to {req.db_type}..."})

            # ── DB connection (I/O-bound → thread pool) ───────
            def _connect():
                nonlocal schema_dict
                handle = open_handle(req.db_type, host=req.host, port=req.port, username=req.username,
                                     password=req.password, database=req.database, db_path=req.db_path,
                                     keyspace=req.keyspace, secure=True, trust_cert=req.trust_cert,
                                     address=address)
                if req.db_type in SQL_TYPES:
                    with handle.connect() as conn:
                        schema_dict = fetch_sql_schema(conn, req.db_type)
                        _fks[ck] = fetch_foreign_keys(conn, req.db_type, schema_dict)
                    close_handle(_engines.get(ck))   # a reconnect replaces, never strands, the old one
                    _engines[ck] = handle
                elif req.db_type == "MongoDB":
                    schema_dict = mongo.sample_schema(handle, timeout_seconds=DB_QUERY_TIMEOUT)
                    close_handle(_mongos.get(ck))
                    _mongos[ck] = handle
                else:   # Cassandra
                    schema_dict, _nosql_keys[ck] = cql.schema(handle, req.keyspace)
                    close_handle(_cassandras.get(ck))
                    _cassandras[ck] = handle

            await run_in_thread(_connect)

            total_count = len(schema_dict)
            yield _evt({"phase": "db_done", "msg": f"Connected — {total_count} tables found. Checking for data..."})

            # ── Filter empty tables (I/O-bound → thread pool) ─
            if req.db_type not in ("MongoDB", "Cassandra"):
                try:
                    def _filter():
                        with _engines[ck].connect() as conn:
                            return filter_empty_tables(conn, schema_dict, db_flavor)
                    schema_dict, empty_tables = await run_in_thread(_filter)
                    yield _evt({"phase": "filter", "msg": f"{len(schema_dict)} tables with data, {len(empty_tables)} empty skipped"})
                except Exception:
                    empty_tables = []
            else:
                empty_tables = []

            reset_concept_cache()

            # ── Schema indexing (run in thread — CPU inside, but non-blocking) ──
            yield _evt({"phase": "index", "msg": f"Indexing {len(schema_dict)} tables..."})
            retriever = await run_in_thread(SchemaRetriever, schema_dict)
            _retrievers[ck] = retriever

            # ── Knowledge Graph (I/O-bound → thread pool) ─────
            yield _evt({"phase": "graph", "msg": "Loading Knowledge Graph & RAG..."})
            ke = KnowledgeEngine(user_id=ck)
            ke.set_groq_client(client)
            kg_stats = await run_in_thread(ke.learn_schema, schema_dict, db_flavor)
            _knowledge[ck] = ke
            cached_tag = " (from cache)" if kg_stats.get("from_cache") else ""
            yield _evt({"phase": "graph_done", "msg": f"Graph ready — {kg_stats.get('graph_nodes', 0)} nodes{cached_tag}"})

            # ── AI Analysis (LLM call is I/O-bound → thread pool) ─
            db_name = req.database or req.db_path or req.keyspace or "default"
            cached_context = ke.load_sia_context(db_name)
            if cached_context:
                yield _evt({"phase": "ai", "msg": "Loading cached AI analysis (0 tokens)..."})
                context = enrich_context_joins(cached_context, schema_dict)
                biz = context.get("business_type", "unknown")
                yield _evt({"phase": "ai_done", "msg": f"Loaded: {biz} (from cache)"})
            elif client:
                yield _evt({"phase": "ai", "msg": "AI analysing database structure..."})
                top_for_sia = retriever.retrieve("business schema transactions orders products", top_k=20, db_type=req.db_type)
                context = await run_in_thread(run_schema_agent, top_for_sia, req.db_type, client)
                context = enrich_context_joins(context, schema_dict)
                ke.save_sia_context(context, db_name)
                biz = context.get("business_type", "unknown")
                yield _evt({"phase": "ai_done", "msg": f"Detected: {biz} database"})
            else:
                context = enrich_context_joins({}, schema_dict)   # heuristic joins only, no LLM
                biz = "unknown"
                yield _evt({"phase": "ai_done", "msg": "No LLM configured - skipped AI analysis"})

            ke.learn_context(context)
            kind = dialects.kind(req.db_type)   # every kind: structure now, values lazily per table
            handle = {"sql": _engines, "mongo": _mongos, "cassandra": _cassandras}[kind][ck]
            _profiles[ck] = await run_in_thread(lambda: dialects.open_profile(
                kind, handle, flavor=db_flavor, schema=schema_dict, foreign_keys=_fks.get(ck, []),
                context=context, folder=ke.folder, keys=_nosql_keys.get(ck)))

            yield _evt({"phase": "session", "msg": "Setting up session..."})
            memory = storage.load_memory(user_id)
            session_data = {
                "db_type":   dialects.kind(req.db_type),   # sql / mongo / cassandra: every route branches on this
                "db_flavor": db_flavor,
                "keyspace":  req.keyspace,
                "context":   context,
                "memory":    memory,
                "conn": {
                    "db_type": req.db_type, "host": req.host, "port": req.port,
                    "username": req.username,
                    "database": req.database, "db_path": req.db_path, "keyspace": req.keyspace,
                },
            }
            redis_cache.set_session(ck, session_data)
            redis_cache.set_schema(ck, schema_dict)
            redis_cache.clear_user_query_cache(ck)

            _vocab.pop(ck, None)
            info = _connections.add(user_id, ConnectionInfo(
                ck, label, req.db_type,
                descriptor={"db_type": req.db_type, "host": req.host, "port": req.port, "username": req.username,
                            "database": req.database, "db_path": req.db_path, "keyspace": req.keyspace,
                            "secure": True, "trust_cert": req.trust_cert},
                has_password=bool(req.password)))
            yield _evt(_for_viewer(user_id, {
                "phase": "done", "success": True,
                "connection": {"id": ck, "label": label},
                "connection_token": info.rsa_token,
                "connections": _connection_list(user_id),
                "db_flavor": db_flavor,
                "tables": list(schema_dict.keys()),
                "schema": schema_dict,
                "business_type": biz,
                "knowledge_graph": ke.get_graph_stats(),
            }))

        except Exception as e:
            if not was_open and _connections.get(user_id, ck) is None:
                _close_connection(ck)
            # Never the driver's text: it can quote the address or credentials.
            print(f"[sql-agent] connect failed: {type(e).__name__}", file=sys.stderr)
            yield _evt({"error": _connection_failure_message(req.db_type, e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────
#  REFRESH SCHEMA
# ─────────────────────────────────────────────────────────────

@app.post("/api/refresh")
async def refresh_schema(user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.CONNECT)
    ck = _active_key(user_id)
    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")

    db_type   = session["db_type"]
    db_flavor = session["db_flavor"]

    try:
        if db_type == "sql":
            engine = _engines.get(ck)
            if not engine:
                raise HTTPException(status_code=400, detail="Engine not in memory — please reconnect.")
            def _fetch():
                with engine.connect() as conn:
                    sd = fetch_sql_schema(conn, db_flavor)
                    _fks[ck] = fetch_foreign_keys(conn, db_flavor, sd)
                    return sd
            schema_dict = await run_in_thread(_fetch)

        else:
            handle = (_mongos if db_type == "mongo" else _cassandras).get(ck)
            if handle is None:   # a pymongo Database refuses truth testing: compare with None
                raise HTTPException(status_code=400, detail="Connection not in memory — please reconnect.")
            if db_type == "mongo":
                schema_dict = await run_in_thread(mongo.sample_schema, handle, 200, DB_QUERY_TIMEOUT)
            else:
                schema_dict, _nosql_keys[ck] = await run_in_thread(cql.schema, handle, session.get("keyspace", ""))

        retriever = await run_in_thread(SchemaRetriever, schema_dict)
        _retrievers[ck] = retriever

        if _llm_ready(user_id):
            top     = retriever.retrieve("business schema transactions orders", top_k=20, db_type=db_type)
            context = await run_in_thread(run_schema_agent, top, db_flavor, _llm_client(user_id))
            context = enrich_context_joins(context, schema_dict)
        else:
            context = enrich_context_joins({}, schema_dict)

        session["context"] = context
        if ck in _knowledge:
            handle = {"sql": _engines, "mongo": _mongos, "cassandra": _cassandras}[db_type][ck]
            _profiles[ck] = await run_in_thread(lambda: dialects.open_profile(
                db_type, handle, flavor=db_flavor, schema=schema_dict, foreign_keys=_fks.get(ck, []),
                context=context, folder=_knowledge[ck].folder, keys=_nosql_keys.get(ck), refresh=True))
        redis_cache.set_session(ck, session)
        redis_cache.set_schema(ck, schema_dict)
        cleared = redis_cache.clear_user_query_cache(ck)

        return {
            "success":       True,
            "tables":        list(schema_dict.keys()),
            "business_type": context.get("business_type", "unknown"),
            "cache_cleared": cleared,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  QUERY — with pagination
#
#  Pagination: the DB runs the full query, but we slice rows before
#  returning them. Avoids sending millions of rows over HTTP.
#  The full result is stored in Redis cache so any page is fast.
#
#  BackgroundTasks: Supabase logging runs AFTER the response is sent.
#  The client gets the answer immediately without waiting for logging.
# ─────────────────────────────────────────────────────────────

@app.post("/api/query")
async def query(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    x_connection_token: str = Header(default=""),
):
    if not req.question:
        raise HTTPException(status_code=400, detail="Empty question.")
    _require(user_id, rbac.ASK)
    ck = _resolve_for_question(user_id, req.question, req.session_id, req.connection_id, x_connection_token)

    allowed, count, ttl = redis_cache.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({count} req). Retry in {ttl}s.",
            headers={"Retry-After": str(ttl)},
        )

    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")

    redis_cache.refresh_session(ck)

    db_type   = session["db_type"]
    db_flavor = session["db_flavor"]
    context   = session.get("context", {})
    memory    = session.get("memory", {})
    schema    = redis_cache.get_schema(ck) or {}
    cache_flavor = _web_cache_flavor(user_id, schema, db_flavor)

    # ── Cache hit — apply pagination and return immediately ───
    cached = redis_cache.get_cached_query(ck, cache_flavor, req.question)
    if cached:
        all_rows  = cached.get("rows", [])
        total     = len(all_rows)
        start     = (req.page - 1) * req.page_size
        cached.update({
            "from_cache":  True,
            "latency_ms":  0,
            "tokens":      {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "rows":        all_rows[start:start + req.page_size],
            "page":        req.page,
            "page_size":   req.page_size,
            "total_rows":  total,
            "total_pages": -(-total // req.page_size),
        })
        background_tasks.add_task(
            storage.log_query,
            user_id=user_id, question=req.question,
            generated_query=cached.get("generated_sql", ""),
            db_flavor=db_flavor, tables_used=cached.get("tables_used", []),
            row_count=total, latency_ms=0, success=True, from_cache=True,
        )
        return _for_viewer(user_id, cached)   # a cache hit is filtered like a fresh answer

    ke    = _knowledge.get(ck)
    notes = ke.notes() if ke else []

    t_start     = time.monotonic()
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    verified    = ke.verified_sql(req.question) if ke else None   # any kind: stored in its own language
    review_notes: list[str] = []

    if verified:   # a human confirmed this exact question before: no LLM call, no tokens
        generated_query, matched, schema_str = verified, {}, ""
        client = None
    else:
        client  = _llm_client(user_id)
        history = redis_cache.get_turns(ck, req.session_id, last_n=HISTORY_TURNS)
        if not history:
            history = [
                {"role": h["role"], "content": h["content"]}
                for h in storage.get_history(user_id, req.session_id, limit=HISTORY_TURNS)
            ]
        messages, matched, schema_str = await run_in_thread(lambda: build_sql_messages(
            req.question, schema, db_flavor, context=context, memory=memory,
            retriever=_get_retriever(ck), ke=ke, notes=notes, profile=_profiles.get(ck), history=history,
            db_type=db_type, keyspace=session.get("keyspace", ""), picker=client,
            scoped=_web_scope(user_id, schema) is not None))

        # LLM call is I/O-bound — run in thread pool so event loop stays free
        def _llm_call():
            return client.chat.completions.create(
                messages=messages,
                temperature=0.0,   # SQL generation: same question, same SQL
                max_completion_tokens=512,
            )

        try:
            res = await run_in_thread(_llm_call)
            if res.usage:
                token_usage = {
                    "prompt_tokens":     res.usage.prompt_tokens     or 0,
                    "completion_tokens": res.usage.completion_tokens or 0,
                    "total_tokens":      res.usage.total_tokens      or 0,
                }
            generated_query = clean_query(res.choices[0].message.content, db_type)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM error: {e}")
        if said_not_answerable(res.choices[0].message.content):
            raise HTTPException(status_code=422, detail={
                "message": CANT_FIND, "suggestions": topic_suggestions(schema, context)})
        if db_type == "sql":   # faster where provably equivalent; one repair round if it is wrong
            generated_query, review_notes = await run_in_thread(
                improve_sql, generated_query, _reviewer(ck, db_flavor), req.question, client, messages)

    unscoped_query = generated_query
    try:
        if db_type == "sql":
            unscoped_query  = validate_sql(generated_query, schema, db_flavor, req.question)
            generated_query = _prepare_web_sql(generated_query, schema, db_flavor, req.question, user_id)
        else:
            unscoped_query, generated_query = _prepare_nosql(generated_query, db_type, schema, user_id, db_flavor)
    except QueryValidationError as ve:
        if client and db_type == "sql" and ("must be a SELECT" in str(ve) or "Could not parse" in str(ve)):
            try:
                def _retry_llm():
                    return client.chat.completions.create(
                        messages=[{"role": "user", "content":
                            f"Generate ONLY a raw SQL SELECT query for: {req.question}\n"
                            f"Tables: {', '.join(matched.keys())}\n"
                            "Return NOTHING except the SQL."}],
                        temperature=0.0,
                        max_completion_tokens=512,
                    )
                retry_res = await run_in_thread(_retry_llm)
                generated_query = clean_query(retry_res.choices[0].message.content, db_type)
                unscoped_query  = validate_sql(generated_query, schema, db_flavor, req.question)
                generated_query = _prepare_web_sql(generated_query, schema, db_flavor, req.question, user_id)
            except Exception:
                raise HTTPException(status_code=400, detail=str(ve))
        else:
            raise HTTPException(status_code=400, detail=str(ve))
    except json.JSONDecodeError as je:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {je}")

    columns, rows, last_error, attempt = [], [], None, 0

    for attempt in range(2):
        try:
            if db_type == "sql":
                columns, rows = await exec_sql_async(_engines[ck], generated_query)

            else:   # MongoDB / Cassandra: the model's own, validated collection or table
                columns, rows = await _run_nosql(ck, db_type, generated_query)

            break

        except asyncio.TimeoutError:
            last_error = f"Query timed out after {DB_QUERY_TIMEOUT}s."
            break
        except Exception as e:
            last_error = str(e)
            if attempt == 0 and client:   # one repair round, in the connection's own language
                try:
                    language = "" if db_type == "sql" else dialects.LANGUAGE[db_type]
                    fixed = clean_query(attempt_fix(unscoped_query, last_error, schema_str, db_flavor, client,
                                                    language), db_type)
                    if db_type == "sql":
                        unscoped_query  = validate_sql(fixed, schema, db_flavor, req.question)
                        generated_query = _prepare_web_sql(fixed, schema, db_flavor, req.question, user_id)
                    else:
                        unscoped_query, generated_query = _prepare_nosql(fixed, db_type, schema, user_id, db_flavor)
                except Exception:
                    break
            else:
                break

    latency_ms = int((time.monotonic() - t_start) * 1000)

    if last_error and not columns:
        background_tasks.add_task(
            storage.log_query,
            user_id=user_id, question=req.question, generated_query=generated_query,
            db_flavor=db_flavor, tables_used=list(matched.keys()),
            row_count=0, latency_ms=latency_ms, success=False, error=last_error,
        )
        raise HTTPException(status_code=400, detail=last_error)

    if ke and unscoped_query and rows:
        ke.learn_successful_query(req.question, unscoped_query)

    memory = update_memory(req.question, context, memory)
    session["memory"] = memory
    redis_cache.set_session(ck, session)
    background_tasks.add_task(storage.save_memory, user_id, memory)

    redis_cache.push_turn(ck, req.session_id, "user", req.question)
    redis_cache.push_turn(ck, req.session_id, "assistant", f"Generated SQL: {generated_query[:200]}")
    background_tasks.add_task(storage.save_turn, user_id=user_id, session_id=req.session_id,
                               role="user", content=req.question)
    background_tasks.add_task(storage.save_turn, user_id=user_id, session_id=req.session_id,
                               role="assistant", content=f"Returned {len(rows)} rows.", query=generated_query)

    # ── Pagination ─────────────────────────────────────────────
    total      = len(rows)
    start      = (req.page - 1) * req.page_size
    paged_rows = rows[start:start + req.page_size]

    payload = {
        "success":       True,
        "generated_sql": generated_query,
        "review":        review_notes,
        "columns":       columns,
        "rows":          paged_rows,
        "answer":        summarize_result(columns, rows, truncated=total >= MAX_ROWS),
        "total_rows":    total,
        "page":          req.page,
        "page_size":     req.page_size,
        "total_pages":   -(-total // req.page_size),
        "business_type": context.get("business_type", ""),
        "tables_used":   list(matched.keys()),
        "retried":       attempt > 0,
        "from_cache":    False,
        "latency_ms":    latency_ms,
        "tokens":        token_usage,
    }

    # Cache the full (unpaginated) result so any page request is fast
    redis_cache.set_cached_query(ck, cache_flavor, req.question, {**payload, "rows": rows})
    background_tasks.add_task(
        storage.log_query,
        user_id=user_id, question=req.question, generated_query=generated_query,
        db_flavor=db_flavor, tables_used=list(matched.keys()),
        row_count=total, latency_ms=latency_ms, success=True, retried=attempt > 0,
    )

    return _for_viewer(user_id, payload)


# ─────────────────────────────────────────────────────────────
#  EXPLAIN — query execution plan
#
#  Use this to diagnose slow queries on large datasets:
#  - Sequential scans → missing index
#  - High cost estimate → consider partitioning or covering index
#  - Nested loops on large tables → rewrite as hash join
# ─────────────────────────────────────────────────────────────

@app.post("/api/query/explain")
async def explain_query(
    req: ExplainRequest,
    user_id: str = Depends(get_current_user),
    x_connection_token: str = Header(default=""),
):
    _require(user_id, rbac.EXPLAIN)
    ck = _active_key(user_id, x_connection_token)
    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="No active session.")

    engine    = _engines.get(ck)
    db_flavor = session.get("db_flavor", "")
    db_type   = dialects.kind(session.get("db_type", "sql"))
    if db_type == "cassandra":
        raise HTTPException(status_code=400, detail="Cassandra has no query plans.")
    if db_type == "mongo":   # the query planner's choice, no execution
        try:
            _, run = _prepare_nosql(req.sql, db_type, redis_cache.get_schema(ck) or {}, user_id, db_flavor)
            plan = await run_in_thread(mongo.explain, _mongos[ck], json.loads(run))
        except QueryValidationError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        return {"query": run, "flavor": "MongoDB", "plan": plan}

    if not engine:
        raise HTTPException(status_code=400, detail="SQL engine not found — reconnect.")
    # EXPLAIN ANALYZE executes the statement (PostgreSQL): only validated, scoped SELECTs get here.
    try:
        sql = _prepare_web_sql(req.sql.strip().rstrip(";"), redis_cache.get_schema(ck) or {},
                               db_flavor, "", user_id)
    except QueryValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    explain_sql = explain_statement(sql, db_flavor)

    def _run_explain():
        with engine.connect() as conn:
            result = conn.execute(text(explain_sql))
            return [dict(row._mapping) for row in result]

    try:
        plan = await run_in_thread(_run_explain)
        return {"query": sql, "flavor": db_flavor, "plan": plan}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  STREAM QUERY — SSE token-by-token
# ─────────────────────────────────────────────────────────────

@app.post("/api/query/stream")
async def query_stream(
    req: StreamQueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    x_connection_token: str = Header(default=""),
):
    if not req.question:
        raise HTTPException(status_code=400, detail="Empty question.")
    _require(user_id, rbac.ASK)
    ck = _resolve_for_question(user_id, req.question, req.session_id, req.connection_id, x_connection_token)

    allowed, count, ttl = redis_cache.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Retry in {ttl}s.")

    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="No active session.")

    db_type   = session["db_type"]
    db_flavor = session["db_flavor"]
    context   = session.get("context", {})
    memory    = session.get("memory", {})
    schema    = redis_cache.get_schema(ck) or {}
    ke        = _knowledge.get(ck)
    notes     = ke.notes() if ke else []
    verified = ke.verified_sql(req.question) if ke else None
    matched: dict = {}
    if not verified:
        if not _llm_ready(user_id):
            raise HTTPException(status_code=400, detail={"message": FRIENDLY_ERRORS["no_llm"]})
        client  = _llm_client(user_id)
        history = redis_cache.get_turns(ck, req.session_id, last_n=HISTORY_TURNS)
        messages, matched, schema_str = await run_in_thread(lambda: build_sql_messages(
            req.question, schema, db_flavor, context=context, memory=memory,
            retriever=_get_retriever(ck), ke=ke, notes=notes, profile=_profiles.get(ck), history=history,
            db_type=db_type, keyspace=session.get("keyspace", ""), picker=client,
            scoped=_web_scope(user_id, schema) is not None))

    def _evt_for(data: dict) -> str:
        return _evt(_for_viewer(user_id, data))

    async def generate() -> AsyncGenerator[str, None]:
        generated_query = verified or ""   # a confirmed answer: no LLM call, no tokens
        t_start = time.monotonic()
        if not verified:
            try:
                def _stream():
                    return client.chat.completions.create(
                        messages=messages,
                        temperature=0.0,   # SQL generation: same question, same SQL
                        max_completion_tokens=512,
                        stream=True,
                    )
                stream = await run_in_thread(_stream)
                # Each chunk is a blocking network read: taken on a worker thread, so a slow or
                # rate-limited provider stalls this request only, not every user on the event loop.
                it = iter(stream)
                while True:
                    chunk = await run_in_thread(next, it, None)
                    if chunk is None:
                        break
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        generated_query += token
                        yield _evt_for({"token": token})
            except Exception as e:
                yield _evt_for({"error": f"LLM stream error: {e}", "message": FRIENDLY_ERRORS["llm"]})
                return
            if said_not_answerable(generated_query):
                yield _evt_for({"error": "not answerable from this database", "message": CANT_FIND,
                            "suggestions": topic_suggestions(schema, context)})
                return

        generated_query = clean_query(generated_query, db_type)
        review_notes: list[str] = []
        if db_type == "sql" and not verified:   # faster where provably equivalent; one repair if wrong
            generated_query, review_notes = await run_in_thread(
                improve_sql, generated_query, _reviewer(ck, db_flavor), req.question, client, messages)
        as_written = generated_query

        try:
            if db_type == "sql":
                generated_query = _prepare_web_sql(generated_query, schema, db_flavor, req.question, user_id)
            else:
                as_written, generated_query = _prepare_nosql(generated_query, db_type, schema, user_id, db_flavor)
        except QueryValidationError as ve:
            yield _evt_for({"error": str(ve), "query": generated_query, "message": FRIENDLY_ERRORS["invalid"]})
            return

        columns, rows, error = [], [], None
        try:
            if db_type == "sql":
                columns, rows = await exec_sql_async(_engines[ck], generated_query)

            else:   # MongoDB / Cassandra: the model's own, validated collection or table
                columns, rows = await _run_nosql(ck, db_type, generated_query)

        except Exception as e:
            error = str(e)

        latency_ms = int((time.monotonic() - t_start) * 1000)

        if error:
            yield _evt_for({"error": error, "query": generated_query, "message": FRIENDLY_ERRORS["database"]})
            background_tasks.add_task(
                storage.log_query,
                user_id=user_id, question=req.question, generated_query=generated_query,
                db_flavor=db_flavor, tables_used=list(matched.keys()),
                row_count=0, latency_ms=latency_ms, success=False, error=error,
            )
        else:
            payload = {
                "done": True, "query": generated_query,
                "columns": columns, "rows": rows,
                "review": review_notes,
                "answer": summarize_result(columns, rows, truncated=len(rows) >= MAX_ROWS
                                           or (db_type == "sql" and hit_auto_limit(as_written, rows))),
                "verified": bool(verified),
                "connection": (_connections.active(user_id) or ConnectionInfo(ck, "", "")).label,
                "connection_id": ck,
                "connections": _connection_list(user_id),
                "latency_ms": latency_ms,
                "tables_used": list(matched.keys()),
            }
            yield _evt_for(payload)
            redis_cache.set_cached_query(ck, _web_cache_flavor(user_id, schema, db_flavor), req.question, {**payload, "from_cache": False})
            # Follow-ups ("now only March") need the previous question; the page only uses this endpoint.
            redis_cache.push_turn(ck, req.session_id, "user", req.question)
            redis_cache.push_turn(ck, req.session_id, "assistant", f"Generated SQL: {generated_query[:200]}")
            if ke and rows and not verified:
                ke.learn_successful_query(req.question, as_written)
            info = _connections.get(user_id, ck)
            # Only the hashed key and the type: no host, port, user or database name on disk.
            binding = {"id": ck, "db_type": info.db_type} if info else None
            await run_in_thread(save_session_turn, user_id, req.session_id[:64], {
                "question": req.question, "answer": payload["answer"], "sql": generated_query,
                "columns": columns, "rows": rows[:SESSION_ROWS], "total_rows": len(rows),
                "verified": payload["verified"],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, binding)
            background_tasks.add_task(
                storage.log_query,
                user_id=user_id, question=req.question, generated_query=generated_query,
                db_flavor=db_flavor, tables_used=list(matched.keys()),
                row_count=len(rows), latency_ms=latency_ms, success=True,
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────
#  HISTORY
# ──────────────────────────────── ─────────────────────────────

@app.get("/api/history")
async def get_history(
    session_id: str = "default",
    limit: int = 20,
    user_id: str = Depends(get_current_user),
):
    history = storage.get_history(user_id, session_id, limit=min(limit, 100))
    return {"history": history, "count": len(history)}


@app.get("/api/sessions")
async def list_sessions(user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.ASK)
    """The user's conversations, newest first, for the sidebar."""
    sessions = load_sessions(user_id)
    items = [{"id": sid, "title": s.get("title", ""), "updated": s.get("updated", ""), "turns": len(s.get("messages", [])),
              "connection": _bound_label(user_id, s.get("connection"))}
             for sid, s in sessions.items()]
    return {"sessions": sorted(items, key=lambda s: s["updated"], reverse=True)}


@app.get("/api/sessions/{session_id}")
async def get_session_turns(session_id: str, user_id: str = Depends(get_current_user)):
    s = load_sessions(user_id).get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    bound = s.get("connection")
    is_open = bool(bound and _connections.get(user_id, bound["id"]))
    return _for_viewer(user_id, {
        "id": session_id, "title": s.get("title", ""), "messages": s.get("messages", []),
        "connection": {**bound, "label": _bound_label(user_id, bound)} if bound else None,
        "connection_open": is_open, "closed_message": "" if is_open or not bound else _closed_message(bound)})


@app.delete("/api/sessions/{session_id}")
async def delete_session_turns(session_id: str, user_id: str = Depends(get_current_user)):
    def _delete() -> bool:
        with _sessions_lock:
            sessions = load_sessions(user_id)
            if session_id not in sessions:
                return False
            del sessions[session_id]
            _write_sessions(user_id, sessions)
            return True
    if not await run_in_thread(_delete):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    redis_cache.clear_conversation(user_id, session_id)
    return {"success": True}


@app.delete("/api/history")
async def clear_history(
    background_tasks: BackgroundTasks,
    session_id: str = "default",
    user_id: str = Depends(get_current_user),
):
    redis_cache.clear_conversation(user_id, session_id)
    # Also delete from Supabase so history doesn't reappear after Redis flush
    background_tasks.add_task(storage.delete_history, user_id, session_id)
    return {"success": True}


# ─────────────────────────────────────────────────────────────
#  CHART SUGGESTION
# ─────────────────────────────────────────────────────────────

@app.post("/api/chart")
async def suggest_chart(req: ChartRequest, user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.ASK)
    ck = _active_key(user_id)
    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="Not connected.")
    return {"chart": suggest_chart_rule(req.columns, req.rows)}


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    return {
        "auth0_domain":    os.getenv("AUTH0_DOMAIN", ""),
        "auth0_client_id": os.getenv("AUTH0_CLIENT_ID", ""),
        "auth0_audience":  os.getenv("AUTH0_AUDIENCE", ""),
        "dev_mode":        os.getenv("DEV_MODE", "false").lower() == "true",
    }


# ─────────────────────────────────────────────────────────────
#  STATUS
# ─────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status(user_id: str = Depends(get_current_user)):
    info = _connections.active(user_id)
    ck = info.key if info else ""
    session = redis_cache.get_session(ck)
    schema  = redis_cache.get_schema(ck) or {}
    return {
        "connected":     session is not None,
        "db_type":       session.get("db_type")   if session else None,
        "db_flavor":     session.get("db_flavor") if session else None,
        "business_type": session.get("context", {}).get("business_type") if session else None,
        "tables":        list(schema.keys()),
        "memory_keys":   list((session or {}).get("memory", {}).keys()),
        "redis_ok":      redis_cache.ping(),
    }


# ─────────────────────────────────────────────────────────────
#  FEEDBACK
# ─────────────────────────────────────────────────────────────

@app.post("/api/feedback")
async def feedback(req: FeedbackRequest, user_id: str = Depends(get_current_user)):
    """👍/👎. With feedback.approve the verdict is final (verified answer, or removed).
    Without it a 👍 is only a suggestion and a 👎 changes nothing: people who cannot see
    the SQL must not decide what the shared memory treats as correct."""
    _require(user_id, rbac.FEEDBACK)
    ck = _active_key(user_id)
    ke = _knowledge.get(ck)
    sql = req.sql or next((t.get("sql", "") for t in reversed(
        (load_sessions(user_id).get(req.session_id) or {}).get("messages", [])) if t.get("question") == req.question), "")
    approved = POLICY.can(user_id, rbac.APPROVE)
    if ke and sql:
        if approved:
            ke.learn_user_feedback(req.question, sql, req.is_correct, req.correction)
        elif req.is_correct:
            ke.learn_successful_query(req.question, sql)      # unverified: a suggestion for an approver
    return {"success": True, "learned": bool(ke and sql), "verified": approved and req.is_correct}


# ─────────────────────────────────────────────────────────────
#  KNOWLEDGE GRAPH STATS
# ─────────────────────────────────────────────────────────────

@app.get("/api/knowledge")
async def knowledge_stats(user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.VIEW_TABLES)
    ck = _active_key(user_id)
    ke = _knowledge.get(ck)
    if not ke:
        raise HTTPException(status_code=400, detail="No knowledge graph. Connect first.")
    return ke.get_graph_stats()


# ─────────────────────────────────────────────────────────────
#  DISCONNECT
# ─────────────────────────────────────────────────────────────

@app.post("/api/disconnect")
async def disconnect(connection_id: str = "", user_id: str = Depends(get_current_user)):
    """Close one database (the active one unless connection_id says which); the others stay open."""
    key = connection_id or _active_key(user_id)
    if key not in {c.key for c in _connections.list(user_id)}:
        raise HTTPException(status_code=404, detail="That database is not open.")
    _close_connection(key)
    now = _connections.remove(user_id, key)
    return {"success": True, "active": now.key if now else None, "connections": _connection_list(user_id)}


def _connection_list(user_id: str) -> list[dict]:
    active = _connections.active(user_id)
    return [{"id": c.key, "label": c.label, "db_type": c.db_type, "active": bool(active and c.key == active.key)}
            for c in _connections.list(user_id)]


@app.post("/api/connections/{connection_id}/unlock")
async def unlock_connection(connection_id: str, req: UnlockRequest, user_id: str = Depends(get_current_user)):
    """Re-enter the password of an idle-locked database. It is checked by really logging in
    (a fresh connection with the saved details), never compared with a stored copy."""
    allowed, _, ttl = redis_cache.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Too many attempts. Retry in {ttl}s.")
    info = _connections.get(user_id, connection_id)
    if info is None:
        raise HTTPException(status_code=404, detail="That database is not open.")
    try:
        d = info.descriptor   # checked again: a host name can point somewhere else since the first connect
        address = check_target(d["db_type"], d.get("host", ""), d.get("db_path", ""), hosted=True)
        handle = await run_in_thread(lambda: open_handle(**d, password=req.password, address=address))
    except TargetRefused as refused:
        raise HTTPException(status_code=403, detail={"message": str(refused)})
    except Exception:
        raise HTTPException(status_code=401, detail={"message": "That password didn't work."})
    close_handle(handle)                 # only the proof was needed (any database type); the open connection stays
    _connections.touch(connection_id)
    return await activate_connection(connection_id, user_id)


class AuthRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/register")
async def auth_register(body: AuthRequest, request: Request):
    """Create a new password-protected user account (scrypt + 32-byte salt) and sign in."""
    # Per client address. Behind a reverse proxy, run uvicorn with --proxy-headers so this is the
    # visitor's address, not the proxy's.
    client = request.client.host if request.client else ""
    if sign_up_blocked(client):
        raise HTTPException(status_code=429, detail={"message": "Too many new accounts from here. Try again later."})
    try:
        role = POLICY.register_account(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"message": str(e)})
    record_sign_up(client)
    user = body.username.strip()
    token = POLICY.issue_session_token(user)
    return {"user": user, "role": role, "token": token}


@app.post("/api/auth/login")
async def auth_login(body: AuthRequest):
    """Verify account username & password (scrypt) and issue a web session token (`ses_...`)."""
    user = (body.username or "").strip()
    if not POLICY.has_account(user):
        raise HTTPException(status_code=401, detail={"message": "Account not found. Please create an account first."})
    if sign_in_blocked(user):   # shared with the MCP sign-in page, across server processes
        raise HTTPException(status_code=429, detail={"message": "Too many failed sign-ins. Try again in 15 minutes."})
    ok = POLICY.verify_account(user, body.password)
    record_sign_in(user, ok)
    if not ok:
        print(f"[sql-agent] failed web sign-in for {user!r}", file=sys.stderr)
        raise HTTPException(status_code=401, detail={"message": "Incorrect username or password."})
    token = POLICY.issue_session_token(user)
    return {"user": user, "role": POLICY.role_of(user), "token": token}


@app.post("/api/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        POLICY.revoke_session_token(authorization.removeprefix("Bearer ").strip())
    return {"ok": True}


@app.get("/api/me")
async def me(user_id: str = Depends(get_current_user)):
    """Role and permissions, so the page shows only what this person may use."""
    return {"user": user_id, "role": POLICY.role_of(user_id), "permissions": sorted(POLICY.permissions(user_id))}


def _public_settings() -> dict:
    """Settings safe to return to the browser (token prefixes/dates only, never hashes or passwords)."""
    doc = POLICY.settings()
    doc.pop("accounts", None)
    doc.pop("web_sessions", None)
    doc["mcp_tokens"] = {
        who: {"prefix": v.get("prefix", ""), "created": v.get("created", "")}
        for who, v in (doc.get("mcp_tokens") or {}).items()
    }
    return doc


@app.get("/api/settings")
async def get_settings(user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.SETTINGS)
    return {"settings": _public_settings(), "permissions": rbac.ALL_PERMISSIONS}


@app.put("/api/settings")
async def put_settings(doc: dict, user_id: str = Depends(get_current_user)):
    """Replace the settings (validated: unknown permissions or roles, or removing every
    settings.manage, are refused). Takes effect on the next request; no restart."""
    _require(user_id, rbac.SETTINGS)
    cur = POLICY.settings()
    current_tokens = cur.get("mcp_tokens") or {}
    incoming_tokens = doc.get("mcp_tokens")
    if isinstance(incoming_tokens, dict):
        doc["mcp_tokens"] = {k: current_tokens[k] for k in incoming_tokens if k in current_tokens}
    else:
        doc["mcp_tokens"] = current_tokens
    doc["accounts"] = cur.get("accounts") or {}
    doc["web_sessions"] = cur.get("web_sessions") or {}
    try:
        POLICY.save(doc)
    except SettingsError as e:
        raise HTTPException(status_code=400, detail={"message": str(e)})
    return {"success": True, "settings": _public_settings()}


# ── Self-service MCP Bearer token per user (SHA-256 hashed in rbac.json; raw token shown once) ──

@app.get("/api/mcp-token")
async def get_mcp_token(user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.MCP)
    return {"user": user_id, **POLICY.mcp_token_info(user_id)}


@app.post("/api/mcp-token")
async def create_mcp_token(user_id: str = Depends(get_current_user)):
    """Issue or rotate this user's personal MCP Bearer token. Returned in full only once."""
    _require(user_id, rbac.MCP)
    raw = POLICY.issue_mcp_token(user_id)
    return {"user": user_id, "token": raw, **POLICY.mcp_token_info(user_id)}


@app.delete("/api/mcp-token")
async def delete_mcp_token(user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.MCP)
    POLICY.revoke_mcp_token(user_id)
    return {"user": user_id, **POLICY.mcp_token_info(user_id)}


# ── Each person's own AI: provider, API key, model (kept in memory only, never returned) ──

def _llm_view(user_id: str) -> dict:
    mine = _llm_settings.get(user_id)
    server_ai = llm_available(llm_api_key())
    return {"source": "yours" if mine else "server" if server_ai else "none",
            "provider": mine.provider if mine else "", "model": mine.model if mine else "",
            "planner_model": mine.planner_model if mine else "",
            "base_url": mine.base_url if mine and mine.provider == "custom" else "",
            "has_key": bool(mine and mine.api_key), "server_model": LLM_MODEL if server_ai else "",
            "providers": [{"id": k, "label": v["label"], "needs_key": v["needs_key"]} for k, v in PROVIDERS.items()],
            "can_custom": POLICY.can(user_id, rbac.SETTINGS)}


def _llm_choice(user_id: str, req: LLMRequest) -> LLMSettings:
    """The validated choice. A blank key keeps the one already held for the same provider, so the
    page never needs the key back."""
    held = _llm_settings.get(user_id)
    key = req.api_key.strip() or (held.api_key if held and held.provider == req.provider.strip() else "")
    choice = LLMSettings(provider=req.provider.strip(), api_key=key, base_url=req.base_url.strip(),
                         model=req.model.strip(), planner_model=req.planner_model.strip())
    try:
        choice.check(POLICY.can(user_id, rbac.SETTINGS))
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"message": str(e)})
    return choice


def _provider_problem(e: Exception) -> str:
    """A plain sentence for a provider failure; the provider's own text is never shown (it may echo a key)."""
    return provider_problem(getattr(e, "status", 0))


@app.get("/api/llm")
async def get_llm(user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.CONNECT)
    return _llm_view(user_id)


@app.put("/api/llm")
async def put_llm(req: LLMRequest, user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.CONNECT)
    _llm_settings[user_id] = _llm_choice(user_id, req)
    return _llm_view(user_id)


@app.delete("/api/llm")
async def delete_llm(user_id: str = Depends(get_current_user)):
    _require(user_id, rbac.CONNECT)
    _llm_settings.pop(user_id, None)
    return _llm_view(user_id)


@app.post("/api/llm/models")
async def llm_models(req: LLMRequest, user_id: str = Depends(get_current_user)):
    """The models the chosen provider offers, for the page to list. Nothing is saved."""
    _require(user_id, rbac.CONNECT)
    choice = _llm_choice(user_id, req.model_copy(update={"model": req.model or "x"}))
    try:
        return {"models": await run_in_thread(list_models, choice.url, choice.api_key)}
    except (LLMError, httpx.HTTPError) as e:
        raise HTTPException(status_code=400, detail={"message": _provider_problem(e)})


@app.post("/api/llm/test")
async def llm_test(req: LLMRequest, user_id: str = Depends(get_current_user)):
    """One tiny request with the chosen key and model, so a wrong key or model shows now, not on a question."""
    _require(user_id, rbac.CONNECT)
    choice = _llm_choice(user_id, req)
    client = _client_from(choice)
    started = time.monotonic()
    try:
        await run_in_thread(lambda: client.chat.completions.create(
            messages=[{"role": "user", "content": "Reply with the word OK."}], temperature=0, max_completion_tokens=8))
    except (LLMError, httpx.HTTPError) as e:
        raise HTTPException(status_code=400, detail={"message": _provider_problem(e)})
    return {"ok": True, "model": getattr(client, "model", choice.model), "ms": int((time.monotonic() - started) * 1000)}


@app.get("/api/connections")
async def list_connections(user_id: str = Depends(get_current_user)):
    """The user's open databases, for the sidebar."""
    return {"connections": _connection_list(user_id)}


@app.post("/api/connections/{connection_id}/activate")
async def activate_connection(connection_id: str, user_id: str = Depends(get_current_user)):
    """Point the chat box at another open database (no reconnect)."""
    info = _connections.get(user_id, connection_id)
    if info is None:
        raise HTTPException(status_code=404, detail="That database is not open.")
    _require_unlocked(user_id, info)
    _connections.activate(user_id, connection_id)
    _connections.touch(connection_id)
    session = redis_cache.get_session(info.key) or {}
    schema = redis_cache.get_schema(info.key) or {}
    return _for_viewer(user_id, {
        "success": True, "connection": {"id": info.key, "label": info.label},
        "connection_token": info.rsa_token, "db_flavor": session.get("db_flavor"),
        "tables": list(schema), "schema": schema, "connections": _connection_list(user_id),
        "business_type": (session.get("context") or {}).get("business_type", "unknown")})


# ─────────────────────────────────────────────────────────────
#  DOCSTORE HELPERS
# ─────────────────────────────────────────────────────────────

def _get_docstore(user_id: str) -> DocStore:
    if user_id not in _docstores:
        _docstores[user_id] = DocStore(user_id)
    return _docstores[user_id]


_SYNTHESIS_PROMPT = """\
You are an analytical assistant. You have been given:
1. A structured data result from a database query.
2. Relevant excerpts from documents (policies, manuals, reports, etc.).

Synthesize both into a clear, concise answer. Where numbers from the database
can be compared against rules or thresholds from the documents, do so explicitly.
Cite the SQL result and the document source. If no documents are available,
answer from the database result alone.

Keep the answer factual and auditable — show your reasoning.\
"""


def _summarise_rows(columns: list, rows: list, limit: int = 10) -> str:
    if not rows:
        return "No rows returned."
    header = " | ".join(columns)
    lines  = [header, "-" * len(header)]
    for row in rows[:limit]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more rows")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  UPLOAD DOCUMENT
# ─────────────────────────────────────────────────────────────

@app.post("/api/docs/upload")
async def upload_doc(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    data   = await file.read()
    store  = _get_docstore(user_id)
    chunks = await run_in_thread(store.add_file, file.filename, data)
    return {"success": True, "filename": file.filename, "chunks_indexed": chunks}


@app.post("/api/docs/ingest_url")
async def ingest_url(req: HybridRequest, user_id: str = Depends(get_current_user)):
    if not req.url:
        raise HTTPException(status_code=400, detail="url field is required.")
    store  = _get_docstore(user_id)
    chunks = await _add_url(store, req.url)
    return {"success": True, "url": req.url, "chunks_indexed": chunks}


async def _add_url(store: DocStore, url: str) -> int:
    """The web app is hosted: only public addresses, size-capped (see docstore.fetch_url)."""
    try:
        return await run_in_thread(store.add_url, url, True)
    except (TargetRefused, ValueError, httpx.HTTPError) as e:
        raise HTTPException(status_code=400, detail={"message": f"Could not add that address: {e}"})


@app.get("/api/docs")
async def list_docs(user_id: str = Depends(get_current_user)):
    store = _get_docstore(user_id)
    return {"sources": await run_in_thread(store.list_sources), "total_chunks": await run_in_thread(store.count)}


# ─────────────────────────────────────────────────────────────
#  HYBRID QUERY  (SQL branch + document RAG + synthesis)
# ─────────────────────────────────────────────────────────────

@app.post("/api/hybrid")
async def hybrid_query(
    req: HybridRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    x_connection_token: str = Header(default=""),
):
    if not req.question:
        raise HTTPException(status_code=400, detail="Empty question.")
    _require(user_id, rbac.ASK)
    ck = _resolve_for_question(user_id, req.question, req.session_id, req.connection_id, x_connection_token)

    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")

    db_type   = session["db_type"]
    db_flavor = session["db_flavor"]
    context   = session.get("context", {})
    memory    = session.get("memory", {})
    client    = _llm_client(user_id)
    schema    = redis_cache.get_schema(ck) or {}

    store = _get_docstore(user_id)

    # Ingest on-the-fly URL if provided
    if req.url:
        await _add_url(store, req.url)

    # ── SQL branch ────────────────────────────────────────────
    sql_result: dict = {}
    matched: dict = {}
    try:
        ke    = _knowledge.get(ck)
        notes = ke.notes() if ke else []
        generated_query = ke.verified_sql(req.question) if ke else None
        if not generated_query:
            messages, matched, _ = await run_in_thread(lambda: build_sql_messages(
                req.question, schema, db_flavor, context=context, memory=memory,
                retriever=_get_retriever(ck), ke=ke, notes=notes, profile=_profiles.get(ck),
                db_type=db_type, keyspace=session.get("keyspace", ""), picker=client,
            scoped=_web_scope(user_id, schema) is not None))

            def _sql_llm():
                return client.chat.completions.create(
                    messages=messages,
                    temperature=0.0,   # SQL generation: same question, same SQL
                    max_completion_tokens=512,
                )

            res = await run_in_thread(_sql_llm)
            if said_not_answerable(res.choices[0].message.content):
                raise QueryValidationError(CANT_FIND)   # documents may still answer it below
            generated_query = clean_query(res.choices[0].message.content, db_type)
            if db_type == "sql":
                generated_query, _ = await run_in_thread(
                    improve_sql, generated_query, _reviewer(ck, db_flavor), req.question, client, messages)

        if db_type == "sql":
            generated_query = _prepare_web_sql(generated_query, schema, db_flavor, req.question, user_id)
            columns, rows   = await exec_sql_async(_engines[ck], generated_query)
        else:
            _, generated_query = _prepare_nosql(generated_query, db_type, schema, user_id, db_flavor)
            columns, rows = await _run_nosql(ck, db_type, generated_query)

        sql_result = {
            "generated_sql": generated_query,
            "columns":       columns,
            "rows":          rows[:50],
            "row_count":     len(rows),
        }
    except Exception as e:
        sql_result = {"error": str(e), "generated_sql": "", "columns": [], "rows": []}

    # ── Document RAG branch ───────────────────────────────────
    doc_hits = await run_in_thread(store.search, req.question, 4)

    # Determine route label
    has_sql  = bool(sql_result.get("columns"))
    has_docs = bool(doc_hits)
    route    = "hybrid" if (has_sql and has_docs) else ("sql_only" if has_sql else ("doc_only" if has_docs else "no_data"))

    # ── Synthesis ─────────────────────────────────────────────
    sql_text = (
        f"SQL: {sql_result.get('generated_sql', '')}\n\nResult:\n"
        + _summarise_rows(sql_result.get("columns", []), sql_result.get("rows", []))
        if has_sql else "No database result available."
    )
    doc_text = (
        "\n\n---\n".join(
            f"[Source: {h['source']}]\n{h['text']}" for h in doc_hits
        ) if has_docs else "No document context available."
    )
    synthesis_user = (
        f"Question: {req.question}\n\n"
        f"=== Database Result ===\n{sql_text}\n\n"
        f"=== Document Context ===\n{doc_text}"
    )

    def _synth_llm():
        return client.chat.completions.create(
            messages=[
                {"role": "system", "content": _SYNTHESIS_PROMPT},
                {"role": "user",   "content": synthesis_user},
            ],
            temperature=0.2,
            max_completion_tokens=1024,
        )

    synth_res = await run_in_thread(_synth_llm)
    answer    = synth_res.choices[0].message.content.strip()

    background_tasks.add_task(
        storage.log_query,
        user_id=user_id, question=req.question,
        generated_query=sql_result.get("generated_sql", ""),
        db_flavor=db_flavor, tables_used=list(matched),
        row_count=sql_result.get("row_count", 0), latency_ms=0, success=True,
    )

    return _for_viewer(user_id, {
        "route":       route,
        "answer":      answer,
        "sql_result":  sql_result,
        "doc_context": doc_hits,
    })


if __name__ == "__main__":
    import uvicorn
    os.makedirs("static", exist_ok=True)
    uvicorn.run("sql_agent.app:app", host="0.0.0.0", port=5000, reload=False)
