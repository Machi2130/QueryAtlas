"""Request-level helpers shared by the routers: permissions and what a viewer may see, the active
connection and the idle lock, which database answers a question, each person's AI client, the tenant
scope gate that stands before any execution, and the accessors over the per-connection state."""

from __future__ import annotations

import asyncio
import json

from fastapi import HTTPException

from .. import cache as redis_cache
from .. import dialects, rbac
from ..connections import ConnectionInfo
from ..database import fetch_read
from ..docstore import DocStore
from ..knowledge import KnowledgeEngine
from ..llm import LLMSettings, llm_api_key, llm_available
from ..profile import ProfileService, database_vocabulary
from ..retrieval import SchemaRetriever
from ..scope import apply_scope, pinned_scope
from ..sqlreview import SqlReviewer
from ..validation import validate_sql
from .runtime import wrt
from .sessions import conversation_binding

CONNECTABLE = {"SQLite", "MySQL", "MariaDB", "PostgreSQL", "SQL Server", "MongoDB", "Cassandra"}

# Keys a person without view.sql never receives; "token" is the query as it streams in.
TECHNICAL = ("query", "generated_sql", "sql", "tables_used", "error", "review", "token")


# ── permissions and visibility ────────────────────────────────────────────────────────────


def require(user_id: str, permission: str) -> None:
    """403 unless the settings file gives this person's role the permission."""
    if not wrt.policy.can(user_id, permission):
        raise HTTPException(
            status_code=403,
            detail={
                "message": f"Your role ({wrt.policy.role_of(user_id)}) does not allow this.",
                "permission": permission,
            },
        )


def for_viewer(user_id: str, data: dict) -> dict:
    """What this person may see of a payload: SQL, tables used and technical errors only with view.sql;
    the raw schema only with view.tables. Decided by the server, so an API call cannot get around the
    page hiding it."""
    out = dict(data)
    if not wrt.policy.can(user_id, rbac.VIEW_SQL):
        for k in TECHNICAL:
            out.pop(k, None)
        if "messages" in out:
            out["messages"] = [{k: v for k, v in m.items() if k not in TECHNICAL} for m in out["messages"]]
        if isinstance(out.get("sql_result"), dict):
            out["sql_result"] = {k: v for k, v in out["sql_result"].items() if k not in TECHNICAL}
    if not wrt.policy.can(user_id, rbac.VIEW_TABLES):
        out.pop("schema", None)
    return out


def evt(data: dict) -> str:
    """One server-sent event."""
    return f"data: {json.dumps(data)}\n\n"


# ── a person's open databases ──────────────────────────────────────────────────────────────


def connection_list(user_id: str) -> list[dict]:
    active = wrt.connections.active(user_id)
    return [
        {"id": c.key, "label": c.label, "db_type": c.db_type, "active": bool(active and c.key == active.key)}
        for c in wrt.connections.list(user_id)
    ]


def require_unlocked(user_id: str, info: ConnectionInfo) -> None:
    if wrt.connections.is_locked(user_id, info.key):
        raise HTTPException(
            status_code=423,
            detail={
                "message": f"{info.label} has not been used for a while. Enter its password to continue.",
                "locked": {"id": info.key, "label": info.label},
            },
        )


def active_key(user_id: str, client_token: str = "") -> str:
    """The connection the chat box is talking to; verified via the server's connection token and
    refused while it is idle-locked."""
    info = wrt.connections.active(user_id)
    if info is None:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")
    try:
        verified_key = wrt.connections.verify(user_id, info.key, client_token)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    require_unlocked(user_id, info)
    wrt.connections.touch(verified_key)
    return verified_key


def closed_message(bound: dict | None) -> str:
    kind = f"a {bound['db_type']} database" if bound and bound.get("db_type") else "a database"
    return f"This conversation used {kind} that is not open. Connect to it again (enter its details) to continue."


def bound_label(user_id: str, bound: dict | None) -> str:
    """The conversation's database name while it is open (from memory); only its type otherwise."""
    info = wrt.connections.get(user_id, bound["id"]) if bound else None
    return info.label if info else (bound or {}).get("db_type", "")


def vocabulary(ck: str):
    state = wrt.store.open(ck)
    if state.vocab is None:
        state.vocab = database_vocabulary(
            redis_cache.get_schema(ck) or {},
            state.profile.profile if state.profile else None,
            state.knowledge.notes() if state.knowledge else [],
        )
    return state.vocab


def resolve_for_question(user_id: str, question: str, session_id: str, picked: str = "", client_token: str = "") -> str:
    """Which database answers: the conversation's own (a conversation stays on one database), the one
    the person just picked, or for a new conversation the one the question is about. Close calls are
    put to the person. Then the idle lock applies."""
    bound = conversation_binding(user_id, session_id)
    target = (bound or {}).get("id") or picked
    if target:
        if wrt.connections.get(user_id, target) is None:
            raise HTTPException(status_code=409, detail={"message": closed_message(bound), "reconnect": bound})
        wrt.connections.activate(user_id, target)
    elif len(wrt.connections.list(user_id)) > 1:
        active = wrt.connections.active(user_id)
        decision = wrt.router.route(
            question, {c.key: vocabulary(c.key) for c in wrt.connections.list(user_id)}, active.key if active else None
        )
        if decision.key is None:
            labels = {c.key: c.label for c in wrt.connections.list(user_id)}
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Which database is this about?",
                    "choose": [{"id": k, "label": labels[k]} for k, score in decision.scores if score > 0],
                },
            )
        if decision.key:
            wrt.connections.activate(user_id, decision.key)
    return active_key(user_id, client_token)


def close_connection(key: str) -> None:
    redis_cache.delete_session(key)
    redis_cache.clear_user_query_cache(key)
    wrt.store.discard(key)  # state goes first, then the handle closes, so nothing picks up a closing handle


# ── per-connection state ───────────────────────────────────────────────────────────────────


def knowledge_of(ck: str) -> KnowledgeEngine | None:
    state = wrt.store.get(ck)
    return state.knowledge if state else None


def profile_service(ck: str) -> ProfileService | None:
    state = wrt.store.get(ck)
    return state.profile if state else None


def retriever_for(ck: str) -> SchemaRetriever | None:
    state = wrt.store.get(ck)
    if state is not None and state.retriever is not None:
        return state.retriever
    schema = redis_cache.get_schema(ck)
    if schema:
        r = wrt.make_retriever(schema)
        wrt.store.open(ck).retriever = r
        return r
    return None


def reviewer_for(ck: str, flavor: str) -> SqlReviewer:
    """Reviews generated SQL against this database's profile (keys, indexes, sizes)."""
    svc = profile_service(ck)
    return SqlReviewer(flavor, svc.profile if svc else None)


def get_docstore(user_id: str) -> DocStore:
    if user_id not in wrt.docstores:
        wrt.docstores[user_id] = DocStore(user_id)
    return wrt.docstores[user_id]


# ── each person's AI ───────────────────────────────────────────────────────────────────────


def llm_ready(user_id: str) -> bool:
    return user_id in wrt.llm_settings or llm_available(llm_api_key())


def client_from(choice: LLMSettings):
    # reasoning_effort is left out: not every model accepts it
    return wrt.make_llm(
        api_key=choice.api_key,
        base_url=choice.url,
        model=choice.model,
        planner_model=choice.planner_model,
        reasoning_effort="",
    )


def llm_client(user_id: str):
    mine = wrt.llm_settings.get(user_id)
    if mine:
        return client_from(mine)
    if not llm_available(llm_api_key()):
        raise HTTPException(
            status_code=400,
            detail="No AI is set up. Choose a provider and add your API key, or ask your admin to set LLM_API_KEY.",
        )
    return wrt.make_llm(llm_api_key())


# ── the tenant-scope gate before any execution ─────────────────────────────────────────────


def web_scope(user_id: str, schema: dict) -> dict | None:
    """The web app has no set_scope: a person's tenant comes only from SCOPE_PINS."""
    return pinned_scope(user_id, schema)


def cache_key_flavor(user_id: str, schema: dict, flavor: str) -> str:
    """The tenant scope is part of the query-cache key, so two scopes never share cached rows."""
    sc = web_scope(user_id, schema)
    return f"{flavor}|scope:{sc['column']}={sc['value']}" if sc else flavor


def prepare_sql(sql: str, schema: dict, flavor: str, question: str, user_id: str) -> tuple[str, str]:
    """Validate, then enforce the person's tenant scope. Returns (as written, as run): the first is
    what gets remembered, so one tenant's value never leaks into another's examples."""
    validated = validate_sql(sql, schema, flavor, question)
    return validated, apply_scope(validated, flavor, web_scope(user_id, schema))[0]


def prepare_nosql(query: str, db_type: str, schema: dict, user_id: str, flavor: str) -> tuple[str, str]:
    """The same gate for a MongoDB / CQL query: validated, then the tenant filter of a pinned person
    added by the server. Returns (as written, as run)."""
    validated = dialects.validate_read(db_type, query, schema, flavor)
    scoped, _ = dialects.scope_read(db_type, validated, web_scope(user_id, schema), flavor, schema)
    return dialects.as_text(db_type, validated), dialects.as_text(db_type, scoped)


async def run_sql_read(engine, query: str) -> tuple[list, list]:
    """Execute SQL in a thread with a hard timeout and a row cap (never pulls a whole table)."""
    return await asyncio.wait_for(
        asyncio.to_thread(fetch_read, engine, query, wrt.max_rows, wrt.query_timeout), timeout=wrt.query_timeout
    )


async def run_nosql(ck: str, db_type: str, query_text: str) -> tuple[list, list]:
    """Row cap and time limit (MongoDB stops it server-side, maxTimeMS); credential fields removed."""
    handle = wrt.store.handle(ck)
    prepared = json.loads(query_text) if db_type == "mongo" else query_text
    return await asyncio.wait_for(
        asyncio.to_thread(dialects.run_read, db_type, handle, prepared, wrt.max_rows, wrt.query_timeout),
        timeout=wrt.query_timeout,
    )
