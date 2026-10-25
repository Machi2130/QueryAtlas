"""The one path a question takes, shared by /api/query, /api/query/stream and /api/hybrid.

    prepare  -> what the model needs (session, schema, knowledge, a verified answer if one exists, the prompt)
    completion / finish -> the model's text (streamed or not, the route decides), checked and cleaned
    gate     -> validated and tenant-scoped: (as written, as run)
    run      -> rows, with the row cap and time limit
    repair   -> one round with the database's error, back through the gate
    remember -> a successful, unscoped query becomes an example for next time

Each route composes these; none of them re-implements a step.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException

from .. import cache as redis_cache
from .. import dialects, storage
from ..presentation import FRIENDLY_ERRORS, topic_suggestions
from ..prompts import HISTORY_TURNS, attempt_fix, build_sql_messages, clean_query, improve_sql, said_not_answerable
from .deps import (
    knowledge_of,
    llm_client,
    llm_ready,
    prepare_nosql,
    prepare_sql,
    profile_service,
    retriever_for,
    reviewer_for,
    run_nosql,
    run_sql_read,
    web_scope,
)
from .runtime import wrt


class NotAnswerable(Exception):
    """The model said the question names nothing in this database."""

    def __init__(self, suggestions: list[str]) -> None:
        super().__init__("not answerable from this database")
        self.suggestions = suggestions


@dataclass
class Prepared:
    ck: str
    user_id: str
    question: str
    session: dict
    schema: dict
    db_type: str
    db_flavor: str
    context: dict
    memory: dict
    ke: Any
    notes: list
    verified: str | None  # a human confirmed this exact question before: no model call
    client: Any = None
    messages: list = field(default_factory=list)
    matched: dict = field(default_factory=dict)
    schema_str: str = ""


async def prepare(user_id: str, ck: str, question: str, session_id: str) -> Prepared:
    session = redis_cache.get_session(ck)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")
    schema = redis_cache.get_schema(ck) or {}
    ke = knowledge_of(ck)
    p = Prepared(
        ck=ck,
        user_id=user_id,
        question=question,
        session=session,
        schema=schema,
        db_type=session["db_type"],
        db_flavor=session["db_flavor"],
        context=session.get("context", {}),
        memory=session.get("memory", {}),
        ke=ke,
        notes=ke.notes() if ke else [],
        verified=ke.verified_sql(question) if ke else None,
    )
    if p.verified:
        return p
    if not llm_ready(user_id):
        raise HTTPException(status_code=400, detail={"message": FRIENDLY_ERRORS["no_llm"]})
    p.client = llm_client(user_id)
    history = redis_cache.get_turns(ck, session_id, last_n=HISTORY_TURNS)
    if not history:
        history = [
            {"role": h["role"], "content": h["content"]}
            for h in storage.get_history(user_id, session_id, limit=HISTORY_TURNS)
        ]
    p.messages, p.matched, p.schema_str = await asyncio.to_thread(
        lambda: build_sql_messages(
            question,
            schema,
            p.db_flavor,
            context=p.context,
            memory=p.memory,
            retriever=retriever_for(ck),
            ke=ke,
            notes=p.notes,
            profile=profile_service(ck),
            history=history,
            db_type=p.db_type,
            keyspace=session.get("keyspace", ""),
            picker=p.client,
            scoped=web_scope(user_id, schema) is not None,
        )
    )
    return p


def completion(p: Prepared, stream: bool = False):
    """The model call: blocking network I/O, so callers run it in a thread."""
    return p.client.chat.completions.create(
        messages=p.messages,
        temperature=0.0,  # same question, same query
        max_completion_tokens=512,
        **({"stream": True} if stream else {}),
    )


def bare_retry(p: Prepared) -> str:
    """When the first answer was not a query at all: ask once more for nothing but the SQL."""
    res = p.client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Generate ONLY a raw SQL SELECT query for: {p.question}\n"
                f"Tables: {', '.join(p.matched.keys())}\nReturn NOTHING except the SQL.",
            }
        ],
        temperature=0.0,
        max_completion_tokens=512,
    )
    return res.choices[0].message.content


def finish(p: Prepared, text: str) -> tuple[str, list[str]]:
    """The model's text to a query: refused if it says the question is unanswerable, cleaned, and for
    SQL reviewed (faster where provably equivalent; one repair if wrong). Calls the model, so run in
    a thread."""
    if not p.verified and said_not_answerable(text):
        raise NotAnswerable(topic_suggestions(p.schema, p.context))
    query = clean_query(text, p.db_type)
    if p.db_type == "sql" and not p.verified:
        return improve_sql(query, reviewer_for(p.ck, p.db_flavor), p.question, p.client, p.messages)
    return query, []


def gate(p: Prepared, query: str) -> tuple[str, str]:
    """(as written, as run). Raises QueryValidationError, or json.JSONDecodeError for malformed MongoDB JSON."""
    if p.db_type == "sql":
        return prepare_sql(query, p.schema, p.db_flavor, p.question, p.user_id)
    return prepare_nosql(query, p.db_type, p.schema, p.user_id, p.db_flavor)


async def run(p: Prepared, to_run: str) -> tuple[list, list]:
    if p.db_type == "sql":
        return await run_sql_read(wrt.store.handle(p.ck), to_run)
    return await run_nosql(p.ck, p.db_type, to_run)  # the model's own, validated collection or table


async def repair(p: Prepared, as_written: str, error: str) -> tuple[str, str]:
    """One repair round with the database's error, in the connection's own language."""
    language = "" if p.db_type == "sql" else dialects.LANGUAGE[p.db_type]
    fixed = await asyncio.to_thread(attempt_fix, as_written, error, p.schema_str, p.db_flavor, p.client, language)
    return gate(p, clean_query(fixed, p.db_type))


def remember(p: Prepared, as_written: str, rows: list) -> None:
    """Stored unscoped and unverified; feedback with approve promotes it."""
    if p.ke and as_written and rows and not p.verified:
        p.ke.learn_successful_query(p.question, as_written)
