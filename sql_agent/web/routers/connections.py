"""A person's databases: connect (streamed progress), refresh the schema, status, the list, switch the
active one, unlock an idle-locked one by logging in again, disconnect."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ... import cache as redis_cache
from ... import cql, dialects, mongo, rbac, storage
from ...auth import get_current_user
from ...connections import ConnectionInfo, TargetRefused, check_target, connection_key, connection_label
from ...database import (
    SQL_TYPES,
    _connection_failure_message,
    close_handle,
    fetch_foreign_keys,
    fetch_sql_schema,
    filter_empty_tables,
)
from ...knowledge import KnowledgeEngine
from ...llm import LLMSettings
from ...prompts import enrich_context_joins
from ...query_planner import reset_concept_cache
from ..deps import (
    CONNECTABLE,
    active_key,
    close_connection,
    connection_list,
    evt,
    for_viewer,
    llm_client,
    llm_ready,
    require,
    require_unlocked,
)
from ..runtime import wrt
from ..schemas import ConnectRequest, UnlockRequest

log = logging.getLogger(__name__)
router = APIRouter()
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/api/connect")
async def connect(req: ConnectRequest, user_id: str = Depends(get_current_user)):
    """Streams the steps of opening a database as server-sent events, ending with the connection or an error."""
    require(user_id, rbac.CONNECT)
    try:  # the web app is always hosted: other people choose the target
        address = check_target(req.db_type, req.host, req.db_path, hosted=True)
    except TargetRefused as refused:
        raise HTTPException(status_code=403, detail={"message": str(refused)}) from refused
    # Each database gets its own key: connecting a second one keeps the first open.
    ck = connection_key(user_id, req.db_type, req.host, req.port, req.database or req.db_path or req.keyspace)
    label = connection_label(req.db_type, req.host, req.database, req.db_path, req.keyspace)
    return StreamingResponse(
        _connect_events(user_id, req, ck, label, address), media_type="text/event-stream", headers=SSE_HEADERS
    )


async def _connect_events(user_id: str, req: ConnectRequest, ck: str, label: str, address) -> AsyncGenerator[str, None]:
    """The connection is published step by step (handle, index, knowledge, session) and committed only when
    it joins the person's connection list. A failure in between must not leave an open, logged-in database
    handle behind that no list knows about."""
    was_open = wrt.connections.get(user_id, ck) is not None
    try:
        yield evt({"phase": "init", "msg": "Initializing AI engine..."})
        if req.db_type not in CONNECTABLE:
            raise ValueError(f"Unknown db_type: {req.db_type}")
        if req.groq_key and user_id not in wrt.llm_settings:  # older clients: a key typed at connect
            wrt.llm_settings[user_id] = LLMSettings(provider="groq", api_key=req.groq_key)
        # Optional: without an AI the database still connects; verified questions, charts and the cached
        # analysis work, new questions need one.
        client = llm_client(user_id) if llm_ready(user_id) else None

        yield evt({"phase": "db", "msg": f"Connecting to {req.db_type}..."})
        schema = await asyncio.to_thread(_open_and_sample, ck, req, address)
        yield evt({"phase": "db_done", "msg": f"Connected — {len(schema)} tables found. Checking for data..."})
        schema, empty = await _without_empty_tables(ck, req.db_type, schema)
        if empty is not None:
            yield evt({"phase": "filter", "msg": f"{len(schema)} tables with data, {len(empty)} empty skipped"})
        reset_concept_cache()

        yield evt({"phase": "index", "msg": f"Indexing {len(schema)} tables..."})
        state = wrt.store.require(ck)
        state.retriever = await asyncio.to_thread(wrt.make_retriever, schema)

        yield evt({"phase": "graph", "msg": "Loading Knowledge Graph & RAG..."})
        ke = KnowledgeEngine(user_id=ck)
        ke.set_groq_client(client)
        kg_stats = await asyncio.to_thread(ke.learn_schema, schema, req.db_type)
        state.knowledge = ke
        cached_tag = " (from cache)" if kg_stats.get("from_cache") else ""
        yield evt({"phase": "graph_done", "msg": f"Graph ready — {kg_stats.get('graph_nodes', 0)} nodes{cached_tag}"})

        db_name = req.database or req.db_path or req.keyspace or "default"
        cached = ke.load_sia_context(db_name)
        if cached:
            yield evt({"phase": "ai", "msg": "Loading cached AI analysis (0 tokens)..."})
            context = enrich_context_joins(cached, schema)
            yield evt({"phase": "ai_done", "msg": f"Loaded: {context.get('business_type', 'unknown')} (from cache)"})
        elif client:
            yield evt({"phase": "ai", "msg": "AI analysing database structure..."})
            context = await _analyse(state, req, schema, client)
            ke.save_sia_context(context, db_name)
            yield evt({"phase": "ai_done", "msg": f"Detected: {context.get('business_type', 'unknown')} database"})
        else:
            context = enrich_context_joins({}, schema)  # heuristic joins only, no LLM
            yield evt({"phase": "ai_done", "msg": "No LLM configured - skipped AI analysis"})
        ke.learn_context(context)
        state.profile = await asyncio.to_thread(
            lambda: dialects.open_profile(  # structure now, values lazily
                dialects.kind(req.db_type),
                state.handle,
                flavor=req.db_type,
                schema=schema,
                foreign_keys=state.foreign_keys,
                context=context,
                folder=ke.folder,
                keys=state.nosql_keys,
            )
        )

        yield evt({"phase": "session", "msg": "Setting up session..."})
        info = await _publish(user_id, req, ck, label, state, schema, context)
        yield evt(
            for_viewer(
                user_id,
                {
                    "phase": "done",
                    "success": True,
                    "connection": {"id": ck, "label": label},
                    "connection_token": info.rsa_token,
                    "connections": connection_list(user_id),
                    "db_flavor": req.db_type,
                    "tables": list(schema.keys()),
                    "schema": schema,
                    "business_type": context.get("business_type", "unknown"),
                    "knowledge_graph": ke.get_graph_stats(),
                },
            )
        )
    except Exception as e:
        if not was_open and wrt.connections.get(user_id, ck) is None:
            close_connection(ck)
        log.warning(f"connect failed: {type(e).__name__}")  # never the driver's text: it can quote credentials
        yield evt({"error": _connection_failure_message(req.db_type, e)})


def _open_and_sample(ck: str, req: ConnectRequest, address) -> dict:
    """Open the database and read its structure (in a thread)."""
    handle = wrt.open_handle(
        req.db_type,
        host=req.host,
        port=req.port,
        username=req.username,
        password=req.password,
        database=req.database,
        db_path=req.db_path,
        keyspace=req.keyspace,
        secure=True,
        trust_cert=req.trust_cert,
        address=address,
    )
    try:
        if req.db_type in SQL_TYPES:
            with handle.connect() as conn:
                schema = fetch_sql_schema(conn, req.db_type)
                extras = {"foreign_keys": fetch_foreign_keys(conn, req.db_type, schema)}
        elif req.db_type == "MongoDB":
            schema, extras = mongo.sample_schema(handle, timeout_seconds=wrt.query_timeout), {}
        else:  # Cassandra
            schema, keys = cql.schema(handle, req.keyspace)
            extras = {"nosql_keys": keys}
    except Exception:
        close_handle(handle)  # opened but unusable: never strand a logged-in handle
        raise
    state = wrt.store.open(ck)
    state.replace_handle(handle)  # a reconnect replaces, never strands, the old one
    for name, value in extras.items():
        setattr(state, name, value)
    return schema


async def _without_empty_tables(ck: str, db_type: str, schema: dict) -> tuple[dict, list | None]:
    """SQL only: tables without rows stay out of the index. (schema, the tables left out, or None when not checked)."""
    if db_type in ("MongoDB", "Cassandra"):
        return schema, None

    def _filter():
        with wrt.store.handle(ck).connect() as conn:
            return filter_empty_tables(conn, schema, db_type)

    try:
        return await asyncio.to_thread(_filter)
    except Exception as e:  # counting rows is a convenience; the full schema still works
        log.warning(f"could not check for empty tables: {e}")
        return schema, None


async def _analyse(state, req: ConnectRequest, schema: dict, client) -> dict:
    top = state.retriever.retrieve("business schema transactions orders products", top_k=20, db_type=req.db_type)
    return enrich_context_joins(await asyncio.to_thread(wrt.analyse_schema, top, req.db_type, client), schema)


async def _publish(
    user_id: str, req: ConnectRequest, ck: str, label: str, state, schema: dict, context: dict
) -> ConnectionInfo:
    """The commit point: the session and schema caches, then the person's connection list."""
    memory = await asyncio.to_thread(storage.load_memory, user_id)
    redis_cache.set_session(
        ck,
        {
            "db_type": dialects.kind(req.db_type),  # sql / mongo / cassandra: every route branches on this
            "db_flavor": req.db_type,
            "keyspace": req.keyspace,
            "context": context,
            "memory": memory,
            "conn": {
                "db_type": req.db_type,
                "host": req.host,
                "port": req.port,
                "username": req.username,
                "database": req.database,
                "db_path": req.db_path,
                "keyspace": req.keyspace,
            },
        },
    )
    redis_cache.set_schema(ck, schema)
    redis_cache.clear_user_query_cache(ck)
    state.vocab = None
    return wrt.connections.add(
        user_id,
        ConnectionInfo(
            ck,
            label,
            req.db_type,
            descriptor={
                "db_type": req.db_type,
                "host": req.host,
                "port": req.port,
                "username": req.username,
                "database": req.database,
                "db_path": req.db_path,
                "keyspace": req.keyspace,
                "secure": True,
                "trust_cert": req.trust_cert,
            },
            has_password=bool(req.password),
        ),
    )


@router.post("/api/refresh")
async def refresh_schema(user_id: str = Depends(get_current_user)):
    require(user_id, rbac.CONNECT)
    ck = active_key(user_id)
    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")

    db_type = session["db_type"]
    db_flavor = session["db_flavor"]

    try:
        state = wrt.store.get(ck)
        if state is None or not state.connected:
            raise HTTPException(status_code=400, detail="Connection not in memory — please reconnect.")
        handle = state.handle
        if db_type == "sql":

            def _fetch():
                with handle.connect() as conn:
                    sd = fetch_sql_schema(conn, db_flavor)
                    state.foreign_keys = fetch_foreign_keys(conn, db_flavor, sd)
                    return sd

            schema_dict = await asyncio.to_thread(_fetch)
        elif db_type == "mongo":
            schema_dict = await asyncio.to_thread(mongo.sample_schema, handle, 200, wrt.query_timeout)
        else:
            schema_dict, state.nosql_keys = await asyncio.to_thread(cql.schema, handle, session.get("keyspace", ""))

        state.retriever = retriever = await asyncio.to_thread(wrt.make_retriever, schema_dict)

        if llm_ready(user_id):
            top = retriever.retrieve("business schema transactions orders", top_k=20, db_type=db_type)
            context = await asyncio.to_thread(wrt.analyse_schema, top, db_flavor, llm_client(user_id))
            context = enrich_context_joins(context, schema_dict)
        else:
            context = enrich_context_joins({}, schema_dict)

        session["context"] = context
        if state.knowledge is not None:
            state.profile = await asyncio.to_thread(
                lambda: dialects.open_profile(
                    db_type,
                    handle,
                    flavor=db_flavor,
                    schema=schema_dict,
                    foreign_keys=state.foreign_keys,
                    context=context,
                    folder=state.knowledge.folder,
                    keys=state.nosql_keys,
                    refresh=True,
                )
            )
        redis_cache.set_session(ck, session)
        redis_cache.set_schema(ck, schema_dict)
        cleared = redis_cache.clear_user_query_cache(ck)

        return {
            "success": True,
            "tables": list(schema_dict.keys()),
            "business_type": context.get("business_type", "unknown"),
            "cache_cleared": cleared,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/status")
async def api_status(user_id: str = Depends(get_current_user)):
    info = wrt.connections.active(user_id)
    ck = info.key if info else ""
    session = redis_cache.get_session(ck)
    schema = redis_cache.get_schema(ck) or {}
    return {
        "connected": session is not None,
        "db_type": session.get("db_type") if session else None,
        "db_flavor": session.get("db_flavor") if session else None,
        "business_type": session.get("context", {}).get("business_type") if session else None,
        "tables": list(schema.keys()),
        "memory_keys": list((session or {}).get("memory", {}).keys()),
        "redis_ok": redis_cache.ping(),
    }


@router.post("/api/disconnect")
async def disconnect(connection_id: str = "", user_id: str = Depends(get_current_user)):
    """Close one database (the active one unless connection_id says which); the others stay open."""
    key = connection_id or active_key(user_id)
    if key not in {c.key for c in wrt.connections.list(user_id)}:
        raise HTTPException(status_code=404, detail="That database is not open.")
    close_connection(key)
    now = wrt.connections.remove(user_id, key)
    return {"success": True, "active": now.key if now else None, "connections": connection_list(user_id)}


@router.post("/api/connections/{connection_id}/unlock")
async def unlock_connection(connection_id: str, req: UnlockRequest, user_id: str = Depends(get_current_user)):
    """Re-enter the password of an idle-locked database. It is checked by really logging in
    (a fresh connection with the saved details), never compared with a stored copy."""
    allowed, _, ttl = redis_cache.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Too many attempts. Retry in {ttl}s.")
    info = wrt.connections.get(user_id, connection_id)
    if info is None:
        raise HTTPException(status_code=404, detail="That database is not open.")
    try:
        d = info.descriptor  # checked again: a host name can point somewhere else since the first connect
        address = check_target(d["db_type"], d.get("host", ""), d.get("db_path", ""), hosted=True)
        handle = await asyncio.to_thread(lambda: wrt.open_handle(**d, password=req.password, address=address))
    except TargetRefused as refused:
        raise HTTPException(status_code=403, detail={"message": str(refused)}) from refused
    except Exception as exc:
        raise HTTPException(status_code=401, detail={"message": "That password didn't work."}) from exc
    close_handle(handle)  # only the proof was needed (any database type); the open connection stays
    wrt.connections.touch(connection_id)
    return await activate_connection(connection_id, user_id)


@router.get("/api/connections")
async def list_connections(user_id: str = Depends(get_current_user)):
    """The user's open databases, for the sidebar."""
    return {"connections": connection_list(user_id)}


@router.post("/api/connections/{connection_id}/activate")
async def activate_connection(connection_id: str, user_id: str = Depends(get_current_user)):
    """Point the chat box at another open database (no reconnect)."""
    info = wrt.connections.get(user_id, connection_id)
    if info is None:
        raise HTTPException(status_code=404, detail="That database is not open.")
    require_unlocked(user_id, info)
    wrt.connections.activate(user_id, connection_id)
    wrt.connections.touch(connection_id)
    session = redis_cache.get_session(info.key) or {}
    schema = redis_cache.get_schema(info.key) or {}
    return for_viewer(
        user_id,
        {
            "success": True,
            "connection": {"id": info.key, "label": info.label},
            "connection_token": info.rsa_token,
            "db_flavor": session.get("db_flavor"),
            "tables": list(schema),
            "schema": schema,
            "connections": connection_list(user_id),
            "business_type": (session.get("context") or {}).get("business_type", "unknown"),
        },
    )
