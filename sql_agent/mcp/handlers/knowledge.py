"""What a database learns beyond its schema: glossary notes, verified examples, the tenant scope,
the SQL-generator choice, and imports of the team's own queries and the engine's statistics."""

from __future__ import annotations

import asyncio
import json
import re
import time

from mcp import types

from ... import cache as redis_cache
from ... import dialects, mongo
from ...errors import ErrCode
from ...llm import llm_available
from ...scope import make_scope
from ...validation import QueryValidationError
from ..context import (
    QUERY_TIMEOUT,
    active,
    audit,
    bump_usage,
    effective_sql_generator,
    fetch_query_history,
    handle_of,
    knowledge_of,
    llm_key,
    load_notes,
    load_usage,
    require_session,
    save_notes,
    save_usage,
    scope_of,
    sources_of,
    tables_in_sql,
)
from ..results import err, ok
from ..runtime import rt


async def add_note(args: dict) -> list[types.TextContent]:
    """Teach the database once: a note on a table, a column, or the whole database."""
    import hashlib

    sid = active(args)
    require_session(sid)
    schema = redis_cache.get_schema(sid) or {}
    note = str(args.get("note", "")).strip()
    table = str(args.get("table", "")).strip()
    column = str(args.get("column", "")).strip()
    if len(note) < 3:
        return err("note is required.", ErrCode.INVALID_SQL)
    if table:
        lookup = {t.lower(): t for t in schema}
        if table.lower() not in lookup:
            return err(f"Table '{table}' not in schema.", ErrCode.INVALID_SQL)
        table = lookup[table.lower()]
        if column:
            cols = {c.split(" ")[0].lower(): c.split(" ")[0] for c in schema[table]}
            if column.lower() not in cols:
                return err(f"Column '{column}' not in {table}.", ErrCode.INVALID_SQL)
            column = cols[column.lower()]
    elif column:
        return err("column requires table.", ErrCode.INVALID_SQL)

    source = str(args.get("source") or "user")
    if source not in ("user", "documentation", "inferred"):
        return err("source must be one of: user, documentation, inferred.", ErrCode.INVALID_SQL)

    notes = await asyncio.to_thread(load_notes, sid)
    entry = {
        "id": hashlib.sha1(f"{table}|{column}|{note}".encode()).hexdigest()[:10],
        "table": table,
        "column": column,
        "note": note,
        "source": source,
        "author": rt.request_user(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if any(n["id"] == entry["id"] for n in notes):
        return ok({"added": False, "duplicate": True, "note": entry})
    notes.append(entry)
    await asyncio.to_thread(save_notes, sid, notes)
    audit("add_note", sid, f"{table}.{column}: {note}" if table else note, ok=True)
    return ok({"added": True, "note": entry, "total_notes": len(notes)})


async def list_notes(args: dict) -> list[types.TextContent]:
    sid = active(args)
    require_session(sid)
    table = str(args.get("table", "")).strip().lower()
    notes = await asyncio.to_thread(load_notes, sid)
    if table:
        notes = [n for n in notes if n.get("table", "").lower() == table or not n.get("table")]
    return ok({"notes": notes, "total": len(notes)})


async def delete_note(args: dict) -> list[types.TextContent]:
    sid = active(args)
    require_session(sid)
    nid = str(args.get("id", "")).strip()
    notes = await asyncio.to_thread(load_notes, sid)
    kept = [n for n in notes if n["id"] != nid]
    if len(kept) == len(notes):
        return err(f"No note with id '{nid}'.", ErrCode.INVALID_SQL)
    await asyncio.to_thread(save_notes, sid, kept)
    audit("delete_note", sid, nid, ok=True)
    return ok({"deleted": True, "id": nid, "total_notes": len(kept)})


async def mark_query(args: dict) -> list[types.TextContent]:
    sid = active(args)
    session = require_session(sid)
    ke = knowledge_of(sid)
    if ke is None:
        return err("No knowledge for this connection yet - reconnect to build it.", ErrCode.NOT_CONNECTED)
    question = str(args.get("question", "")).strip()
    sql = str(args.get("sql", "")).strip().rstrip(";")
    correct = bool(args.get("correct"))
    correction = str(args.get("correction", "")).strip().rstrip(";")
    if not question:
        return err("question is required.", ErrCode.INVALID_SQL)
    keep = sql if correct else correction
    if keep:  # only a valid read-only query, in this connection's language, becomes an example
        try:
            dialects.validate_read(
                dialects.kind(session.get("db_type", "sql")),
                keep,
                redis_cache.get_schema(sid) or {},
                session["db_flavor"],
            )
        except QueryValidationError as ve:
            return err(f"Not stored: {ve}", ErrCode.INVALID_SQL)
    ke.learn_user_feedback(question, sql, correct, correction or None)
    audit("mark_query", sid, keep or sql, ok=True, extra={"question": question[:500], "correct": correct})
    return ok({"stored": bool(keep), "verified": bool(keep), "removed_bad_pair": not correct})


async def set_scope(args: dict) -> list[types.TextContent]:
    sid = active(args)
    require_session(sid)
    schema = redis_cache.get_schema(sid) or {}
    column = str(args.get("column", "")).strip()
    value = str(args.get("value", "")).strip()
    label = str(args.get("label", "")).strip()
    if (scope_of(sid) or {}).get("pinned"):
        audit("set_scope", sid, f"refused: pinned ({column}={value})", ok=False)
        return err(
            "This identity's tenant scope is fixed by the server administrator (SCOPE_PINS) and "
            "cannot be changed or cleared.",
            ErrCode.AUTH_FAILED,
        )
    if not value:
        rt.store.require(sid).scope = None
        redis_cache.clear_user_query_cache(sid)
        audit("set_scope", sid, f"cleared {column}", ok=True)
        return ok({"scope": None, "message": "Scope cleared - queries now span all clients."})
    sc = make_scope(column, value, schema, label)
    if not sc:
        return err(f"No table has a column named '{column}'.", ErrCode.INVALID_SQL)
    rt.store.require(sid).scope = sc
    redis_cache.clear_user_query_cache(sid)  # cached answers were computed under the old scope
    audit("set_scope", sid, f"{sc['column']} = {value} ({label})", ok=True)
    return ok(
        {
            "scope": {"column": sc["column"], "value": value, "label": label, "scoped_tables": len(sc["tables"])},
            "message": f"The {len(sc['tables'])} tables that have {sc['column']} are now always filtered to "
            f"{sc['column']} = {value}, whatever the SQL says.",
        }
    )


async def set_sql_generator(args: dict) -> list[types.TextContent]:
    sid = active(args)
    require_session(sid)
    mode = str(args.get("mode", "")).strip().lower()
    if mode not in ("server", "client", "auto"):
        return err("mode must be one of: 'server', 'client', 'auto'.", ErrCode.INVALID_SQL)
    groq_key = str(args.get("groq_key", "")).strip()
    if groq_key:
        rt.store.require(sid).llm_key = groq_key
    if mode == "server" and not llm_available(llm_key(sid)):
        return err(
            "Cannot switch to 'server' SQL generator because no LLM API key is configured on the server. "
            "Pass groq_key to set_sql_generator(mode='server', groq_key='...') or use mode='client'.",
            ErrCode.AUTH_FAILED,
        )
    if mode in ("server", "client"):
        rt.store.require(sid).sql_generator = mode
    else:
        rt.store.require(sid).sql_generator = ""
    eff = effective_sql_generator(sid)
    audit("set_sql_generator", sid, f"mode={mode} (effective={eff})", ok=True)
    return ok(
        {
            "sql_generator": eff,
            "sql_generator_mode": mode,
            "workflow": (
                "1 round-trip: call query_database(question) directly — the server's AI generates, reviews, and executes SQL."
                if eff == "server"
                else "2 round-trips: call get_query_context(question), write the SQL yourself, and execute it with run_sql(sql, question=...)."
            ),
        }
    )


async def import_query_history(args: dict) -> list[types.TextContent]:
    sid = active(args)
    session = require_session(sid)
    limit = min(2000, max(10, int(args.get("limit", 300))))
    kind = dialects.kind(session.get("db_type", "sql"))
    if kind == dialects.CASSANDRA:
        return err(
            "Cassandra keeps no per-query statistics to import. Use import_queries with your team's CQL instead.",
            ErrCode.UNSUPPORTED_DB,
        )
    schema = redis_cache.get_schema(sid) or {}
    flavor = session.get("db_flavor", "")
    handle = handle_of(sid)

    def _fetch():
        if kind == dialects.MONGO:  # the database profiler's record: collection -> operations
            return [(t, n) for t, n in mongo.collection_usage(handle, limit, QUERY_TIMEOUT) if t in schema]
        with handle.connect() as conn:
            return fetch_query_history(conn, flavor, limit)

    try:
        history = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=QUERY_TIMEOUT)
    except Exception as e:
        need = (
            "the database profiler on and a login allowed to read system.profile"
            if kind == dialects.MONGO
            else "VIEW SERVER STATE"
        )
        return err(
            f"Could not read query statistics ({flavor}); this needs {need}. {str(e)[:200]}", ErrCode.QUERY_FAILED
        )
    if not history:
        return ok(
            {
                "imported_statements": 0,
                "message": f"No query statistics available for {flavor}."
                + (" Turn on the database profiler (level 1 or 2) to record them." if kind == dialects.MONGO else ""),
            }
        )

    usage = await asyncio.to_thread(load_usage, sid)
    matched = 0
    for stmt, count in history:
        tables = [stmt] if kind == dialects.MONGO else tables_in_sql(stmt, schema)
        if not tables:
            continue
        matched += 1
        for t in tables:
            usage["tables"][t] = usage["tables"].get(t, 0) + max(1, count)
    usage["statements"] = usage.get("statements", 0) + matched
    await asyncio.to_thread(save_usage, sid, usage)
    top = sorted(usage["tables"].items(), key=lambda kv: -kv[1])[:10]
    audit("import_query_history", sid, f"{matched} statements", ok=True, rows=matched)
    return ok(
        {
            "imported_statements": matched,
            "statements_read": len(history),
            "hot_tables": [{"table": t, "executions": n} for t, n in top],
        }
    )


def nosql_imports(kind: str, text_in: str) -> list[tuple[str, object]]:
    """(question, query) pairs. MongoDB: a JSON list or one JSON query per line, each with an optional
    "question". CQL: statements split on ';', '--' comment lines become the question."""
    if kind == dialects.MONGO:
        try:
            items = json.loads(text_in)
            items = items if isinstance(items, list) else [items]
        except json.JSONDecodeError:
            items = []
            for line in filter(None, (line.strip() for line in text_in.splitlines())):
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    items.append(None)  # counted as skipped by validation
        return [
            (
                str((i or {}).get("question", "")) if isinstance(i, dict) else "",
                {k: v for k, v in i.items() if k != "question"} if isinstance(i, dict) else i,
            )
            for i in items
        ]
    pairs = []
    for chunk in filter(None, (c.strip() for c in re.split(r";\s*\n|;\s*$", text_in, flags=re.MULTILINE))):
        lines = chunk.splitlines()
        question = " ".join(line.strip()[2:].strip() for line in lines if line.strip().startswith("--"))
        pairs.append((question, "\n".join(line for line in lines if not line.strip().startswith("--")).strip()))
    return pairs


async def import_queries(args: dict) -> list[types.TextContent]:
    sid = active(args)
    session = require_session(sid)
    schema = redis_cache.get_schema(sid) or {}
    ke = knowledge_of(sid)
    text_in = str(args.get("sql_text", ""))
    kind = dialects.kind(session.get("db_type", "sql"))
    imported, skipped = 0, 0
    if kind != dialects.SQL:
        for question, query in nosql_imports(kind, text_in):
            try:
                checked = dialects.validate_read(kind, query, schema, session["db_flavor"])
            except QueryValidationError:
                skipped += 1
                continue
            sources = sources_of(kind, checked, schema)
            if ke:
                ke.learn_successful_query(
                    question or f"query over {', '.join(sources)}", dialects.as_text(kind, checked), verified=True
                )
            await asyncio.to_thread(bump_usage, sid, sources)
            imported += 1
        audit("import_queries", sid, f"{imported} imported, {skipped} skipped", ok=True, rows=imported)
        return ok(
            {
                "imported": imported,
                "skipped": skipped,
                "hint": "Imported queries now appear as similar_queries in get_query_context and count toward hot_tables.",
            }
        )
    for chunk in re.split(r";\s*\n|;\s*$", text_in, flags=re.MULTILINE):
        chunk = chunk.strip()
        if not chunk:
            continue
        comments = [line[2:].strip() for line in chunk.splitlines() if line.strip().startswith("--")]
        sql = "\n".join(line for line in chunk.splitlines() if not line.strip().startswith("--")).strip()
        if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
            skipped += 1
            continue
        tables = tables_in_sql(sql, schema)
        if not tables:
            skipped += 1
            continue
        question = " ".join(comments) or f"query over {', '.join(tables)}"
        if ke:
            ke.learn_successful_query(question, sql, verified=True)  # the team's own SQL
        await asyncio.to_thread(bump_usage, sid, tables)
        imported += 1
    audit("import_queries", sid, f"{imported} imported, {skipped} skipped", ok=True, rows=imported)
    return ok(
        {
            "imported": imported,
            "skipped": skipped,
            "hint": "Imported SELECTs now appear as similar_queries in get_query_context and count toward hot_tables.",
        }
    )
