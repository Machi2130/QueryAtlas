"""Questions: paged answers, token-by-token streaming, the hybrid answer over documents, the query
plan, chart choice, feedback and the knowledge-graph stats. The three question routes compose the
pipeline; the transport (JSON, server-sent events, synthesis) is what differs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from ... import cache as redis_cache
from ... import dialects, mongo, rbac, storage
from ...auth import get_current_user
from ...connections import ConnectionInfo
from ...database import explain_statement
from ...presentation import CANT_FIND, FRIENDLY_ERRORS, hit_auto_limit, suggest_chart_rule, summarize_result
from ...prompts import clean_query, update_memory
from ...validation import QueryValidationError
from .. import pipeline
from ..deps import (
    active_key,
    cache_key_flavor,
    connection_list,
    evt,
    for_viewer,
    get_docstore,
    knowledge_of,
    llm_client,
    prepare_nosql,
    prepare_sql,
    require,
    resolve_for_question,
)
from ..runtime import wrt
from ..schemas import ChartRequest, ExplainRequest, FeedbackRequest, HybridRequest, QueryRequest, StreamQueryRequest
from ..sessions import SESSION_ROWS, load_sessions, save_session_turn
from .documents import SYNTHESIS_PROMPT, add_url, summarise_rows

log = logging.getLogger(__name__)
router = APIRouter()

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _rate_limited(user_id: str, with_headers: bool) -> None:
    allowed, count, ttl = redis_cache.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({count} req). Retry in {ttl}s.",
            headers={"Retry-After": str(ttl)} if with_headers else None,
        )


def _log(
    background_tasks: BackgroundTasks,
    p: pipeline.Prepared,
    query: str,
    rows: int,
    latency_ms: int,
    success: bool,
    **extra,
) -> None:
    background_tasks.add_task(
        storage.log_query,
        user_id=p.user_id,
        question=p.question,
        generated_query=query,
        db_flavor=p.db_flavor,
        tables_used=list(p.matched.keys()),
        row_count=rows,
        latency_ms=latency_ms,
        success=success,
        **extra,
    )


# ── /api/query: a paged JSON answer ────────────────────────────────────────────────────────


def _paged(rows: list, page: int, page_size: int) -> dict:
    total = len(rows)
    start = (page - 1) * page_size
    return {
        "rows": rows[start : start + page_size],
        "page": page,
        "page_size": page_size,
        "total_rows": total,
        "total_pages": -(-total // page_size),
    }


def _cached_answer(
    cached: dict, req: QueryRequest, session: dict, background_tasks: BackgroundTasks, user_id: str
) -> dict:
    """Any page of a cached answer is served without a query, and filtered like a fresh one."""
    cached.update(
        {
            "from_cache": True,
            "latency_ms": 0,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            **_paged(cached.get("rows", []), req.page, req.page_size),
        }
    )
    background_tasks.add_task(
        storage.log_query,
        user_id=user_id,
        question=req.question,
        generated_query=cached.get("generated_sql", ""),
        db_flavor=session["db_flavor"],
        tables_used=cached.get("tables_used", []),
        row_count=cached["total_rows"],
        latency_ms=0,
        success=True,
        from_cache=True,
    )
    return for_viewer(user_id, cached)


async def _generated(p: pipeline.Prepared) -> tuple[str, list[str], dict]:
    """(query, review notes, token usage) from the model, or the verified answer without a model call."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if p.verified:
        return p.verified, [], usage
    try:
        res = await asyncio.to_thread(pipeline.completion, p)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}") from e
    if res.usage:
        usage = {
            "prompt_tokens": res.usage.prompt_tokens or 0,
            "completion_tokens": res.usage.completion_tokens or 0,
            "total_tokens": res.usage.total_tokens or 0,
        }
    try:
        query, notes = await asyncio.to_thread(pipeline.finish, p, res.choices[0].message.content)
    except pipeline.NotAnswerable as na:
        raise HTTPException(status_code=422, detail={"message": CANT_FIND, "suggestions": na.suggestions}) from na
    return query, notes, usage


async def _gated(p: pipeline.Prepared, generated: str) -> tuple[str, str]:
    """Through the validator and the tenant scope; when the model answered with prose instead of SQL, one
    bare retry asks for nothing but the query."""
    try:
        return pipeline.gate(p, generated)
    except QueryValidationError as ve:
        not_a_query = "must be a SELECT" in str(ve) or "Could not parse" in str(ve)
        if not (p.client and p.db_type == "sql" and not_a_query):
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        try:
            return pipeline.gate(p, clean_query(await asyncio.to_thread(pipeline.bare_retry, p), "sql"))
        except Exception as retry_error:
            raise HTTPException(status_code=400, detail=str(ve)) from retry_error
    except json.JSONDecodeError as je:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {je}") from je


async def _run_with_repair(
    p: pipeline.Prepared, as_written: str, to_run: str
) -> tuple[list, list, str | None, int, str, str]:
    """(columns, rows, error, attempts, as written, as run): one repair round with the database's error when
    the model wrote the query. The queries returned are the ones that produced the rows, so a repaired
    statement is what gets remembered, cached and shown, never the one that failed."""
    columns, rows, last_error, attempt = [], [], None, 0
    for attempt in range(2):
        try:
            columns, rows = await pipeline.run(p, to_run)
            return columns, rows, None, attempt, as_written, to_run
        except TimeoutError:
            return [], [], f"Query timed out after {wrt.query_timeout}s.", attempt, as_written, to_run
        except Exception as e:
            last_error = str(e)
            if attempt > 0 or not p.client:
                break
            try:
                as_written, to_run = await pipeline.repair(p, as_written, last_error)
            except Exception as repair_error:
                log.debug(f"repair round failed: {repair_error}")
                break
    return columns, rows, last_error, attempt, as_written, to_run


@router.post("/api/query")
async def query(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    x_connection_token: str = Header(default=""),
):
    """A paged JSON answer; the full result is cached so any page is fast."""
    if not req.question:
        raise HTTPException(status_code=400, detail="Empty question.")
    require(user_id, rbac.ASK)
    ck = resolve_for_question(user_id, req.question, req.session_id, req.connection_id, x_connection_token)
    _rate_limited(user_id, with_headers=True)
    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")
    redis_cache.refresh_session(ck)
    cache_flavor = cache_key_flavor(user_id, redis_cache.get_schema(ck) or {}, session["db_flavor"])
    cached = redis_cache.get_cached_query(ck, cache_flavor, req.question)
    if cached:
        return _cached_answer(cached, req, session, background_tasks, user_id)

    p = await pipeline.prepare(user_id, ck, req.question, req.session_id)
    t_start = time.monotonic()
    generated, review_notes, token_usage = await _generated(p)
    as_written, to_run = await _gated(p, generated)
    columns, rows, last_error, attempt, as_written, to_run = await _run_with_repair(p, as_written, to_run)
    latency_ms = int((time.monotonic() - t_start) * 1000)
    if last_error and not columns:
        _log(background_tasks, p, to_run, 0, latency_ms, False, error=last_error)
        raise HTTPException(status_code=400, detail=last_error)

    pipeline.remember(p, as_written, rows)
    memory = update_memory(req.question, p.context, p.memory)
    p.session["memory"] = memory
    redis_cache.set_session(ck, p.session)
    background_tasks.add_task(storage.save_memory, user_id, memory)
    redis_cache.push_turn(ck, req.session_id, "user", req.question)
    redis_cache.push_turn(ck, req.session_id, "assistant", f"Generated SQL: {to_run[:200]}")
    background_tasks.add_task(
        storage.save_turn, user_id=user_id, session_id=req.session_id, role="user", content=req.question
    )
    background_tasks.add_task(
        storage.save_turn,
        user_id=user_id,
        session_id=req.session_id,
        role="assistant",
        content=f"Returned {len(rows)} rows.",
        query=to_run,
    )
    payload = {
        "success": True,
        "generated_sql": to_run,
        "review": review_notes,
        "columns": columns,
        **_paged(rows, req.page, req.page_size),
        "answer": summarize_result(columns, rows, truncated=len(rows) >= wrt.max_rows),
        "business_type": p.context.get("business_type", ""),
        "tables_used": list(p.matched.keys()),
        "retried": attempt > 0,
        "from_cache": False,
        "latency_ms": latency_ms,
        "tokens": token_usage,
    }
    redis_cache.set_cached_query(ck, cache_flavor, req.question, {**payload, "rows": rows})
    _log(background_tasks, p, to_run, len(rows), latency_ms, True, retried=attempt > 0)
    return for_viewer(user_id, payload)


# ── /api/query/stream: the query token by token, then the answer ──────────────────────────


@router.post("/api/query/stream")
async def query_stream(
    req: StreamQueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    x_connection_token: str = Header(default=""),
):
    if not req.question:
        raise HTTPException(status_code=400, detail="Empty question.")
    require(user_id, rbac.ASK)
    ck = resolve_for_question(user_id, req.question, req.session_id, req.connection_id, x_connection_token)
    _rate_limited(user_id, with_headers=False)
    p = await pipeline.prepare(user_id, ck, req.question, req.session_id)

    def evt_for(data: dict) -> str:
        return evt(for_viewer(user_id, data))

    async def generate() -> AsyncGenerator[str, None]:
        text_so_far = p.verified or ""  # a confirmed answer: no model call, no tokens
        t_start = time.monotonic()
        if not p.verified:
            try:
                stream = await asyncio.to_thread(pipeline.completion, p, True)
                # Each chunk is a blocking network read, taken on a worker thread: a slow or rate-limited
                # provider stalls this request only, not every person on the event loop.
                it = iter(stream)
                while True:
                    chunk = await asyncio.to_thread(next, it, None)
                    if chunk is None:
                        break
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        text_so_far += token
                        yield evt_for({"token": token})
            except Exception as e:
                yield evt_for({"error": f"LLM stream error: {e}", "message": FRIENDLY_ERRORS["llm"]})
                return
        try:
            generated, review_notes = await asyncio.to_thread(pipeline.finish, p, text_so_far)
        except pipeline.NotAnswerable as na:
            yield evt_for({"error": str(na), "message": CANT_FIND, "suggestions": na.suggestions})
            return
        try:
            as_written, to_run = pipeline.gate(p, generated)
        except (QueryValidationError, json.JSONDecodeError) as ve:
            yield evt_for({"error": str(ve), "query": generated, "message": FRIENDLY_ERRORS["invalid"]})
            return

        columns, rows, error = [], [], None
        try:
            columns, rows = await pipeline.run(p, to_run)
        except Exception as e:
            error = str(e)
        latency_ms = int((time.monotonic() - t_start) * 1000)
        if error:
            yield evt_for({"error": error, "query": to_run, "message": FRIENDLY_ERRORS["database"]})
            _log(background_tasks, p, to_run, 0, latency_ms, False, error=error)
            return

        truncated = len(rows) >= wrt.max_rows or (p.db_type == "sql" and hit_auto_limit(as_written, rows))
        payload = {
            "done": True,
            "query": to_run,
            "columns": columns,
            "rows": rows,
            "review": review_notes,
            "answer": summarize_result(columns, rows, truncated=truncated),
            "verified": bool(p.verified),
            "connection": (wrt.connections.active(user_id) or ConnectionInfo(ck, "", "")).label,
            "connection_id": ck,
            "connections": connection_list(user_id),
            "latency_ms": latency_ms,
            "tables_used": list(p.matched.keys()),
        }
        yield evt_for(payload)
        redis_cache.set_cached_query(
            ck, cache_key_flavor(user_id, p.schema, p.db_flavor), req.question, {**payload, "from_cache": False}
        )
        # Follow-ups ("now only March") need the previous question; the page only uses this endpoint.
        redis_cache.push_turn(ck, req.session_id, "user", req.question)
        redis_cache.push_turn(ck, req.session_id, "assistant", f"Generated SQL: {to_run[:200]}")
        pipeline.remember(p, as_written, rows)
        info = wrt.connections.get(user_id, ck)
        binding = {"id": ck, "db_type": info.db_type} if info else None  # the hashed key and the type only
        await asyncio.to_thread(
            save_session_turn,
            user_id,
            req.session_id[:64],
            {
                "question": req.question,
                "answer": payload["answer"],
                "sql": to_run,
                "columns": columns,
                "rows": rows[:SESSION_ROWS],
                "total_rows": len(rows),
                "verified": payload["verified"],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            binding,
        )
        _log(background_tasks, p, to_run, len(rows), latency_ms, True)

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


# ── /api/hybrid: the database result and the documents, synthesised ───────────────────────


@router.post("/api/hybrid")
async def hybrid_query(
    req: HybridRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    x_connection_token: str = Header(default=""),
):
    if not req.question:
        raise HTTPException(status_code=400, detail="Empty question.")
    require(user_id, rbac.ASK)
    ck = resolve_for_question(user_id, req.question, req.session_id, req.connection_id, x_connection_token)
    client = llm_client(user_id)  # the synthesis needs a model even when the query is verified
    p = await pipeline.prepare(user_id, ck, req.question, req.session_id)
    store = get_docstore(user_id)
    if req.url:
        await add_url(store, req.url)

    sql_result: dict = {}
    try:
        if p.verified:
            generated = p.verified
        else:
            res = await asyncio.to_thread(pipeline.completion, p)
            generated, _ = await asyncio.to_thread(pipeline.finish, p, res.choices[0].message.content)
        _, to_run = pipeline.gate(p, generated)
        columns, rows = await pipeline.run(p, to_run)
        sql_result = {"generated_sql": to_run, "columns": columns, "rows": rows[:50], "row_count": len(rows)}
    except pipeline.NotAnswerable:  # documents may still answer it below
        sql_result = {"error": CANT_FIND, "generated_sql": "", "columns": [], "rows": []}
    except Exception as e:
        sql_result = {"error": str(e), "generated_sql": "", "columns": [], "rows": []}

    doc_hits = await asyncio.to_thread(store.search, req.question, 4)
    has_sql, has_docs = bool(sql_result.get("columns")), bool(doc_hits)
    route = "hybrid" if (has_sql and has_docs) else ("sql_only" if has_sql else ("doc_only" if has_docs else "no_data"))
    sql_text = (
        f"SQL: {sql_result.get('generated_sql', '')}\n\nResult:\n"
        + summarise_rows(sql_result.get("columns", []), sql_result.get("rows", []))
        if has_sql
        else "No database result available."
    )
    doc_text = (
        "\n\n---\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in doc_hits)
        if has_docs
        else "No document context available."
    )
    synthesis_user = (
        f"Question: {req.question}\n\n=== Database Result ===\n{sql_text}\n\n=== Document Context ===\n{doc_text}"
    )

    def synth():
        return client.chat.completions.create(
            messages=[{"role": "system", "content": SYNTHESIS_PROMPT}, {"role": "user", "content": synthesis_user}],
            temperature=0.2,
            max_completion_tokens=1024,
        )

    answer = (await asyncio.to_thread(synth)).choices[0].message.content.strip()
    _log(background_tasks, p, sql_result.get("generated_sql", ""), sql_result.get("row_count", 0), 0, True)
    return for_viewer(user_id, {"route": route, "answer": answer, "sql_result": sql_result, "doc_context": doc_hits})


# ── the plan, charts, feedback, knowledge ──────────────────────────────────────────────────


@router.post("/api/query/explain")
async def explain_query(
    req: ExplainRequest, user_id: str = Depends(get_current_user), x_connection_token: str = Header(default="")
):
    """Diagnose a slow query: sequential scans, high cost estimates, nested loops on large tables."""
    require(user_id, rbac.EXPLAIN)
    ck = active_key(user_id, x_connection_token)
    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="No active session.")
    db_flavor = session.get("db_flavor", "")
    db_type = dialects.kind(session.get("db_type", "sql"))
    schema = redis_cache.get_schema(ck) or {}
    if db_type == "cassandra":
        raise HTTPException(status_code=400, detail="Cassandra has no query plans.")
    if db_type == "mongo":  # the query planner's choice, no execution
        try:
            _, run = prepare_nosql(req.sql, db_type, schema, user_id, db_flavor)
            plan = await asyncio.to_thread(mongo.explain, wrt.store.handle(ck), json.loads(run))
        except QueryValidationError as ve:
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        return {"query": run, "flavor": "MongoDB", "plan": plan}
    engine = wrt.store.handle(ck) if ck in wrt.store else None
    if engine is None:
        raise HTTPException(status_code=400, detail="SQL engine not found — reconnect.")
    try:  # EXPLAIN ANALYZE executes the statement (PostgreSQL): only validated, scoped SELECTs get here
        _, sql = prepare_sql(req.sql.strip().rstrip(";"), schema, db_flavor, "", user_id)
    except QueryValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    explain_sql = explain_statement(sql, db_flavor)

    def run_explain():
        with engine.connect() as conn:
            return [dict(row._mapping) for row in conn.execute(text(explain_sql))]

    try:
        return {"query": sql, "flavor": db_flavor, "plan": await asyncio.to_thread(run_explain)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/api/chart")
async def suggest_chart(req: ChartRequest, user_id: str = Depends(get_current_user)):
    require(user_id, rbac.ASK)
    ck = active_key(user_id)
    if not redis_cache.get_session(ck):
        raise HTTPException(status_code=400, detail="Not connected.")
    return {"chart": suggest_chart_rule(req.columns, req.rows)}


@router.post("/api/feedback")
async def feedback(req: FeedbackRequest, user_id: str = Depends(get_current_user)):
    """Thumbs up or down. With feedback.approve the verdict is final (verified answer, or removed).
    Without it a thumbs-up is only a suggestion and a thumbs-down changes nothing: people who cannot
    see the SQL must not decide what the shared memory treats as correct."""
    require(user_id, rbac.FEEDBACK)
    ck = active_key(user_id)
    ke = knowledge_of(ck)
    turns = (await asyncio.to_thread(load_sessions, user_id)).get(req.session_id) or {}
    sql = req.sql or next(
        (t.get("sql", "") for t in reversed(turns.get("messages", [])) if t.get("question") == req.question), ""
    )
    approved = wrt.policy.can(user_id, rbac.APPROVE)
    if ke and sql:
        if approved:
            ke.learn_user_feedback(req.question, sql, req.is_correct, req.correction)
        elif req.is_correct:
            ke.learn_successful_query(req.question, sql)  # unverified: a suggestion for an approver
    return {"success": True, "learned": bool(ke and sql), "verified": approved and req.is_correct}


@router.get("/api/knowledge")
async def knowledge_stats(user_id: str = Depends(get_current_user)):
    require(user_id, rbac.VIEW_TABLES)
    ke = knowledge_of(active_key(user_id))
    if not ke:
        raise HTTPException(status_code=400, detail="No knowledge graph. Connect first.")
    return ke.get_graph_stats()
