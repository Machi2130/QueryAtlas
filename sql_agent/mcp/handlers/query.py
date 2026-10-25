"""Reads: query_database (the server's own model writes the query), run_sql (the caller wrote it),
get_query_context (what a question needs, nothing more), get_schema, validate, explain and chart."""

from __future__ import annotations

import asyncio
import logging
import time

from mcp import types
from sqlalchemy import text

from ... import cache as redis_cache
from ... import dialects, metrics, mongo
from ...database import explain_statement
from ...errors import ErrCode
from ...llm import LLM_MODEL
from ...presentation import CLARIFY_MESSAGE, suggest_chart_rule
from ...profile import ContextAssembler, names_anything, question_words
from ...prompts import (
    CATALOG_MAX,
    attempt_fix,
    build_sql_messages,
    clean_query,
    improve_sql,
    said_not_answerable,
    update_memory,
)
from ...validation import QueryValidationError, validate_sql
from ..context import (
    MAX_ROWS,
    OTHER_TABLES_SHOWN,
    QUERY_TIMEOUT,
    active,
    apply_scope_to,
    audit,
    bump_usage,
    cache_key_flavor,
    diagnose_db_error,
    generator_choice,
    handle_of,
    knowledge_of,
    llm_for,
    load_notes,
    load_usage,
    profile_service,
    require_session,
    retriever_of,
    reviewer_for,
    scope_of,
    sources_of,
    table_keys,
    tenant_column_candidates,
    track_usage,
    usage_of,
)
from ..results import err, ok
from ..runtime import rt

log = logging.getLogger(__name__)


def _page(rows: list, page: int, page_size: int) -> dict:
    total = len(rows)
    start = (page - 1) * page_size
    return {
        "rows": rows[start : start + page_size],
        "total_rows": total,
        "page": page,
        "page_size": page_size,
        "total_pages": -(-total // page_size),
    }


async def _generate(sid: str, session: dict, schema: dict, question: str, kind: str, ke, notes: list):
    """The server's own model writes the query: (query, review notes, client, schema text). A verified answer
    skips the model; the client stays None then."""
    db_flavor = session["db_flavor"]
    query = ke.verified_sql(question) if ke else None
    if query:
        return query, [], None, ""
    client = llm_for(sid)
    messages, _, schema_str = await asyncio.to_thread(
        lambda: build_sql_messages(
            question,
            schema,
            db_flavor,
            context=session.get("context", {}),
            memory=session.get("memory", {}),
            retriever=retriever_of(sid),
            ke=ke,
            notes=notes,
            profile=profile_service(sid),
            picker=client,
            db_type=kind,
            keyspace=session.get("keyspace", ""),
            scoped=bool(scope_of(sid)),
        )
    )
    metrics.record(sid, messages, schema)
    res = await asyncio.to_thread(
        lambda: client.chat.completions.create(messages=messages, temperature=0.0, max_completion_tokens=512)
    )  # same question, same query
    text_out = res.choices[0].message.content
    if said_not_answerable(text_out):
        return None, [], client, schema_str
    query = clean_query(text_out, kind)
    if kind == dialects.SQL:
        query, review_notes = await asyncio.to_thread(
            improve_sql, query, reviewer_for(sid, db_flavor), question, client, messages
        )
        return query, review_notes, client, schema_str
    return query, [], client, schema_str


async def _run_with_repair(
    sid: str, kind: str, handle, unscoped, schema: dict, db_flavor: str, client, schema_str: str
):
    """Run the validated query under the tenant filter; one repair round with the database's error when the
    model wrote it. Returns (columns, rows, query as run, query as written, attempts) or an error result."""
    language = "" if kind == dialects.SQL else dialects.LANGUAGE[kind]
    for attempt in range(2):
        try:
            prepared, _ = dialects.scope_read(kind, unscoped, scope_of(sid), db_flavor, schema)
            columns, rows = await asyncio.wait_for(
                asyncio.to_thread(dialects.run_read, kind, handle, prepared, MAX_ROWS, QUERY_TIMEOUT),
                timeout=QUERY_TIMEOUT,
            )
            return columns, rows, dialects.as_text(kind, prepared), dialects.as_text(kind, unscoped), attempt
        except TimeoutError:
            return err(f"Query timed out after {QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
        except QueryValidationError as ve:  # e.g. a tenant-filtered collection joined in
            return err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)
        except Exception as e:
            if attempt > 0 or not client:
                return err(str(e), ErrCode.QUERY_FAILED)
            try:
                fixed = await asyncio.to_thread(
                    attempt_fix, dialects.as_text(kind, unscoped), str(e), schema_str, db_flavor, client, language
                )
                unscoped = dialects.validate_read(kind, clean_query(fixed, kind), schema, db_flavor)
            except Exception as repair_error:
                log.debug(f"repair round failed: {repair_error}")
                return err(str(e), ErrCode.QUERY_FAILED)
    return err("Query failed.", ErrCode.QUERY_FAILED)  # unreachable: both attempts return above


async def query_database(args: dict) -> list[types.TextContent]:
    """One round-trip: the server's model writes the query, the validator and the tenant filter gate it, the
    database runs it, and a successful query is remembered as an example."""
    sid = active(args)
    if generator_choice(sid) == "client" or (not generator_choice(sid) and rt.policy.sql_generator() == "client"):
        return err(
            "SQL generator mode is set to 'client' (MCP host LLM writes SQL). Call get_query_context(question) "
            "and run_sql(sql) instead, or call set_sql_generator(mode='server').",
            ErrCode.INVALID_SQL,
        )
    question = args["question"]
    page = max(1, int(args.get("page", 1)))
    page_size = min(500, max(1, int(args.get("page_size", 50))))
    session = require_session(sid)
    schema = redis_cache.get_schema(sid) or {}
    db_flavor = session["db_flavor"]
    kind = dialects.kind(session.get("db_type", "sql"))
    cache_flavor = cache_key_flavor(sid, db_flavor)

    cached = redis_cache.get_cached_query(sid, cache_flavor, question)
    if cached:
        return ok(
            {
                "generated_sql": cached.get("generated_sql"),
                "columns": cached.get("columns"),
                **_page(cached.get("rows", []), page, page_size),
                "from_cache": True,
                "retried": False,
            }
        )

    ke = knowledge_of(sid)
    notes = await asyncio.to_thread(load_notes, sid)
    query, review_notes, client, schema_str = await _generate(sid, session, schema, question, kind, ke, notes)
    if query is None:
        return err(CLARIFY_MESSAGE, ErrCode.NEEDS_CLARIFICATION)
    try:
        unscoped = dialects.validate_read(kind, query, schema, db_flavor)
    except QueryValidationError as ve:
        return err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)
    outcome = await _run_with_repair(sid, kind, handle_of(sid), unscoped, schema, db_flavor, client, schema_str)
    if not isinstance(outcome, tuple):
        return outcome
    columns, rows, as_run, as_written, attempt = outcome

    # Remembered without the tenant filter, so one tenant's value never leaks into another's examples.
    if ke and rows:
        ke.learn_successful_query(question, as_written)
    session["memory"] = update_memory(question, session.get("context", {}), session.get("memory", {}))
    redis_cache.set_session(sid, session)
    track_usage(sid, "queries")
    if attempt > 0:
        track_usage(sid, "retries")
    audit(
        "query_database",
        sid,
        as_run,
        ok=True,
        rows=len(rows),
        extra={"question": question[:500], "model": getattr(client, "model", LLM_MODEL), "retried": attempt > 0},
    )
    result = {
        "generated_sql": as_run,
        "columns": columns,
        **_page(rows, page, page_size),
        "from_cache": False,
        "retried": attempt > 0,
        "review": review_notes,
        "usage": usage_of(sid),
    }
    redis_cache.set_cached_query(sid, cache_flavor, question, {**result, "rows": rows})
    return ok(result)


async def run_sql(args: dict) -> list[types.TextContent]:
    """Execute caller-written read-only query in the connection's language (SQL, MongoDB JSON, CQL).
    No LLM involved."""
    sid = active(args)
    raw = args.get("sql", args.get("query", ""))
    sql = raw if isinstance(raw, dict) else str(raw).strip().rstrip(";")
    page = max(1, int(args.get("page", 1)))
    page_size = min(500, max(1, int(args.get("page_size", 50))))

    session = require_session(sid)
    kind = dialects.kind(session.get("db_type", "sql"))
    handle = handle_of(sid)
    schema = redis_cache.get_schema(sid) or {}
    db_flavor = session["db_flavor"]

    rewrites, findings = [], []
    if kind == dialects.SQL:  # speed rewrites and checks are SQL-specific
        reviewer = reviewer_for(sid, db_flavor)
        sql, rewrites = reviewer.rewrite(sql)
        findings = reviewer.review(sql, str(args.get("question", "")))
    try:
        unscoped_sql = dialects.validate_read(kind, sql, schema, db_flavor)
        prepared, scoped = dialects.scope_read(kind, unscoped_sql, scope_of(sid), db_flavor, schema)
        sql = dialects.as_text(kind, prepared)
    except QueryValidationError as ve:
        audit("run_sql", sid, dialects.as_text(kind, sql), ok=False, error=str(ve))
        return err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)

    t0 = time.monotonic()
    try:
        # +1 so we can tell "exactly MAX_ROWS" from "more than MAX_ROWS".
        columns, rows = await asyncio.wait_for(
            asyncio.to_thread(dialects.run_read, kind, handle, prepared, MAX_ROWS + 1, QUERY_TIMEOUT),
            timeout=QUERY_TIMEOUT,
        )
    except TimeoutError:
        audit("run_sql", sid, sql, ok=False, error=f"timeout after {QUERY_TIMEOUT}s")
        return err(f"Query timed out after {QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
    except Exception as e:
        track_usage(sid, "errors")
        audit("run_sql", sid, sql, ok=False, error=str(e))
        code, details = diagnose_db_error(str(e), sql, schema) if kind == dialects.SQL else (ErrCode.QUERY_FAILED, {})
        return err(str(e), code, details)

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    track_usage(sid, "queries")
    audit("run_sql", sid, sql, ok=True, rows=len(rows), ms=int((time.monotonic() - t0) * 1000))
    question = str(args.get("question", "")).strip()
    ke = knowledge_of(sid)
    if question and rows and ke:
        # Stored as unverified (and unscoped so tenant filters never leak into RAG examples).
        # mark_query (after the user confirms) promotes it to a verified example.
        ke.learn_successful_query(question, dialects.as_text(kind, unscoped_sql))
    await asyncio.to_thread(bump_usage, sid, sources_of(kind, prepared, schema))
    warnings = []
    if truncated:
        warnings.append(f"Result truncated at {MAX_ROWS} rows; aggregate or filter instead of reading rows.")
    if scoped:
        sc = rt.store.require(sid).scope
        warnings.append(f"Tenant filter {sc['column']} = {sc['value']!r} applied to: {', '.join(scoped)}.")
    if not rows:
        warnings.append("No rows matched. Report that plainly; do not invent values.")
    warnings += [f"Rewritten for speed (same rows): {r}" for r in rewrites]
    warnings += [f"Check this query: {f.message}" for f in findings]
    total = len(rows)
    start = (page - 1) * page_size
    return ok(
        {
            "sql": sql,
            "columns": columns,
            "rows": rows[start : start + page_size],
            "total_rows": total,
            "truncated": truncated,
            "max_rows": MAX_ROWS,
            "page": page,
            "page_size": page_size,
            "total_pages": -(-total // page_size),
            "warnings": warnings,
        }
    )


def _assembled(
    sid: str, svc, question: str, candidates: list[str], max_tables: int
) -> tuple[list, dict, dict, list, dict]:
    """The profile joins the candidates through the graph: (table names, how complete the selection is, the other
    candidates one line each, the joins, what columns hold). Runs in a thread."""
    asm = ContextAssembler(svc)
    assembled = asm.assemble(question, candidates)
    names = list(assembled.tables)
    selection = {
        "status": "complete" if assembled.complete else "incomplete",
        "terminals": assembled.terminals,
        "unconnected_terminals": assembled.unconnected_terminals,
        "preferred_table_limit": max_tables,
        "preferred_limit_exceeded": len(names) > max_tables,
    }
    # One line per other candidate: the caller's own model can see what else exists and ask for it with
    # get_schema(tables=[...]); a table the code ranked low is no longer invisible.
    others = {
        t: asm.catalog_line(t, brief=True)[len(t) + 4 :]
        for t in asm.catalog_tables(question, candidates, CATALOG_MAX)
        if t not in names
    }
    others = dict(list(others.items())[:OTHER_TABLES_SHOWN])
    joins = [
        {
            "from": f"{j.from_table}.{j.from_column}",
            "to": f"{j.to_table}.{j.to_column}",
            "source": j.source,
            "rows": f"many {j.from_table} per {j.to_table}",
        }
        for j in assembled.joins
    ]
    # What columns hold: status values, numeric and date ranges. Sampled from every tenant's rows, so none
    # under a tenant scope.
    values: dict = {}
    for t in [] if scope_of(sid) else names:
        hints = {c: cp.hint() for c, cp in svc.table(t).columns.items() if cp.hint()}
        if hints:
            values[t] = hints
    return names, selection, others, joins, values


async def query_context(args: dict) -> list[types.TextContent]:
    """Knowledge-graph lookup for the client's own model: only the tables, joins and examples a question needs,
    so it can write the query without reading the whole schema."""
    sid = active(args)
    question = str(args.get("question", "")).strip()
    max_tables = min(12, max(1, int(args.get("max_tables", 6))))
    session = require_session(sid)
    ke = knowledge_of(sid)
    if ke is None:
        return err("No knowledge for this connection yet - reconnect to build it.", ErrCode.NOT_CONNECTED)
    schema = redis_cache.get_schema(sid) or {}
    notes = await asyncio.to_thread(load_notes, sid)

    # Glossary first: a human said "this table/column means X"; then the retriever and whatever the knowledge
    # engine found. The profile joins them through the graph.
    words = question_words(question)
    glossary_tables = [n["table"] for n in notes if n.get("table") and any(w in n["note"].lower() for w in words)]
    retriever = retriever_of(sid)
    ranked = (
        list(await asyncio.to_thread(retriever.retrieve, question, max(max_tables, CATALOG_MAX)) or {})
        if retriever
        else []
    )
    ctx = await asyncio.to_thread(ke.get_context_for_question, question, session.get("db_type", "sql"))
    candidates = list(dict.fromkeys(glossary_tables + ranked + list(ctx.get("tables_found") or [])))
    svc = profile_service(sid)
    if svc:
        names, selection, other_tables, joins, values = await asyncio.to_thread(
            _assembled, sid, svc, question, candidates, max_tables
        )
    else:
        names, selection, other_tables, joins, values = candidates, {"status": "unavailable"}, {}, [], {}
    tables = {t: schema[t] for t in names if t in schema}

    # Popular tables first (what this database's users actually query), so an ambiguous name resolves to
    # the one everyone uses.
    usage = (await asyncio.to_thread(load_usage, sid)).get("tables", {})
    if usage:
        tables = dict(sorted(tables.items(), key=lambda kv: -usage.get(kv[0], 0)))
    hot = [t for t, _ in sorted(usage.items(), key=lambda kv: -kv[1])[:5]]
    sc = scope_of(sid)
    biz = session.get("context") or {}
    biz_tables = biz.get("tables") if isinstance(biz.get("tables"), dict) else {}
    kind = dialects.kind(session.get("db_type", "sql"))
    low_confidence = not glossary_tables and not names_anything(question, schema)
    result = {
        "dialect": session.get("db_flavor"),
        "query_language": dialects.LANGUAGE[kind],
        "tables": tables,
        "other_tables": other_tables,  # name -> what it is + first columns; fetch with get_schema
        "keys": await asyncio.to_thread(table_keys, sid, list(tables)),
        "values": values,  # what columns hold: status values, numeric/date ranges
        "joins": joins,
        "context_selection": selection,
        "glossary": [
            {k: v for k, v in n.items() if k in ("table", "column", "note", "source")}
            for n in notes
            if not n.get("table") or n["table"] in tables
        ][:20],
        "scope": (
            {
                "column": sc["column"],
                "value": sc["value"],
                "label": sc.get("label", ""),
                "rule": f"The server filters every table that has {sc['column']} to {sc['value']!r} automatically.",
            }
            if sc
            else None
        ),
        "scope_candidates": [] if sc else tenant_column_candidates(schema),
        "hot_tables": hot,
        "table_notes": {t: n for t, n in biz_tables.items() if t in tables},
        "metrics": biz.get("metrics") or {},
        # Human-confirmed examples first; unverified ones only ran without error.
        "similar_queries": sorted(ctx.get("similar_queries") or [], key=lambda q: not q.get("verified")),
        "business_type": biz.get("business_type", "unknown"),
        "low_confidence": low_confidence,
        "hint": (
            "No table matched this question well: ask the user which data they mean, or say this database "
            "does not appear to hold it."
            if low_confidence
            else "Write ONE SELECT in this dialect, then call run_sql with it and the question."
            if kind == dialects.SQL
            else "Write ONE query in 'query_language', then call run_sql with it and the question."
        ),
    }
    result["context_tokens"] = metrics.record(sid, result, schema)  # what the model got versus the whole schema
    return ok(result)


async def get_schema(args: dict) -> list[types.TextContent]:
    sid = active(args)
    schema = redis_cache.get_schema(sid)
    if not schema:
        return err("Not connected. Call connect_database first.", ErrCode.NOT_CONNECTED)
    wanted = args.get("tables")
    if not wanted:
        return ok({"tables": schema, "table_count": len(schema)})
    lookup = {t.lower(): t for t in schema}
    schema = {lookup[w.lower()]: schema[lookup[w.lower()]] for w in wanted if w.lower() in lookup}
    keys = await asyncio.to_thread(table_keys, sid, list(schema))
    svc = profile_service(sid)
    joins = [
        {
            "from": f"{r.from_table}.{r.from_column}",
            "to": f"{r.to_table}.{r.to_column}",
            "source": r.source,
            "rows": f"many {r.from_table} per {r.to_table}",
        }
        for r in (svc.profile.relationships if svc else [])
        if r.from_table in schema and r.to_table in schema
    ]
    return ok({"tables": schema, "keys": keys, "joins": joins, "table_count": len(schema)})


async def validate_query(args: dict) -> list[types.TextContent]:
    sid = active(args)
    session = require_session(sid)
    schema = redis_cache.get_schema(sid) or {}
    raw = args.get("sql", args.get("query", ""))
    sql = raw if isinstance(raw, dict) else str(raw).strip().rstrip(";")
    kind = dialects.kind(session.get("db_type", "sql"))
    try:
        checked = dialects.validate_read(kind, sql, schema, session["db_flavor"])
        final, scoped = dialects.scope_read(kind, checked, scope_of(sid), session["db_flavor"], schema)
    except QueryValidationError as ve:
        return ok({"valid": False, "reason": str(ve)})
    result = {
        "valid": True,
        "sql_to_run": dialects.as_text(kind, final),
        "tables": sources_of(kind, checked, schema),
        "tenant_filtered": scoped,
        "faster_equivalent": None,
        "review": [],
    }
    if kind == dialects.SQL:  # speed rewrites and checks are SQL-specific
        reviewer = reviewer_for(sid, session["db_flavor"])
        faster, rewrites = reviewer.rewrite(checked)
        result.update(
            faster_equivalent=faster if rewrites else None,
            review=[{"code": f.code, "message": f.message} for f in reviewer.review(checked)],
        )
    return ok(result)


async def explain_query(args: dict) -> list[types.TextContent]:
    sid = active(args)
    session = require_session(sid)
    kind = dialects.kind(session.get("db_type", "sql"))
    if kind == dialects.CASSANDRA:
        return err(
            "Cassandra has no query plans. Queries on the partition key are fast; others need ALLOW "
            "FILTERING and read more data.",
            ErrCode.UNSUPPORTED_DB,
        )
    if kind == dialects.MONGO:
        schema = redis_cache.get_schema(sid) or {}
        try:
            checked = dialects.validate_read(kind, args.get("sql", args.get("query", "")), schema, session["db_flavor"])
            prepared, _ = dialects.scope_read(kind, checked, scope_of(sid), session["db_flavor"], schema)
            plan = await asyncio.to_thread(mongo.explain, handle_of(sid), prepared)
        except QueryValidationError as ve:
            return err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)
        except Exception as e:
            return err(str(e), ErrCode.QUERY_FAILED)
        return ok({"sql": dialects.as_text(kind, prepared), "flavor": "MongoDB", "plan": plan})
    engine = rt.store.handle(sid) if sid in rt.store else None
    if engine is None:
        return err("No SQL engine found. Reconnect first.", ErrCode.NOT_CONNECTED)

    db_flavor = session.get("db_flavor", "")
    # EXPLAIN ANALYZE executes the statement (PostgreSQL): only validated, scoped SELECTs get here.
    try:
        sql = validate_sql(str(args["sql"]).strip().rstrip(";"), redis_cache.get_schema(sid) or {}, db_flavor, "")
        sql, _ = apply_scope_to(sid, sql, db_flavor)
    except QueryValidationError as ve:
        return err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)
    explain_sql = explain_statement(sql, db_flavor)

    def _run():
        with engine.connect() as conn:
            result = conn.execute(text(explain_sql))
            return [dict(row._mapping) for row in result]

    try:
        plan = await asyncio.to_thread(_run)
        return ok({"sql": sql, "flavor": db_flavor, "plan": plan})
    except Exception as e:
        return err(str(e), ErrCode.QUERY_FAILED)


async def chart(args: dict) -> list[types.TextContent]:
    """Rule-based: column types decide the chart, no LLM call."""
    track_usage(active(args), "charts")
    return ok({"chart": suggest_chart_rule(args["columns"], args["rows"])})
