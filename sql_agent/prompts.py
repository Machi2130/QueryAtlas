"""Prompts for the server-side LLM: schema analysis, SQL generation, auto-fix."""

from __future__ import annotations

import json
import os
import re

import sys

from .llm import LLM_MODEL, LLMClient
from .mongo import READ_STAGES
from .profile import ContextAssembler, ProfileService, infer_joins
from .profile.context import MAX_PICKED_TABLES, MAX_PROMPT_TABLES
from .profile.vocab import question_words
from .query_planner import build_query_plan
from .retrieval import build_schema_str, compress_schema, rank_columns

# ═══════════════════════════════════════════════════════════════
#  AGENT 1 — Schema Intelligence Agent (SIA)
# ═══════════════════════════════════════════════════════════════

SIA_PROMPT = (
    "You are a database schema analyst. Understand the schema and\n"
    "produce a structured business context JSON.\n\n"
    "Identify:\n"
    "1. Business domain (retail, banking, hotel, healthcare, logistics, etc.)\n"
    "2. For each table: its role, primary date column, primary value/amount column,\n"
    "   primary key column, and table type\n"
    "3. Core business metrics (how to compute them from columns)\n"
    "4. Common join relationships between tables\n\n"
    "Return ONLY this JSON (no markdown, no explanation):\n"
    "{\n"
    "  \"business_type\": \"<domain>\",\n"
    "  \"tables\": {\n"
    "    \"<table_name>\": {\n"
    "      \"role\": \"<what this table represents>\",\n"
    "      \"date_column\": \"<most relevant date/time column or null>\",\n"
    "      \"value_column\": \"<most relevant amount/revenue column or null>\",\n"
    "      \"id_column\": \"<primary key column>\",\n"
    "      \"type\": \"<transaction|master|lookup|log|mapping>\"\n"
    "    }\n"
    "  },\n"
    "  \"metrics\": {\n"
    "    \"revenue\":    \"<SQL expression>\",\n"
    "    \"quantity\":   \"<SQL expression or null>\",\n"
    "    \"count\":      \"<SQL expression>\",\n"
    "    \"date_trunc\": \"<SQL expression for monthly grouping>\"\n"
    "  },\n"
    "  \"joins\": [\n"
    "    { \"from\": \"<table.column>\", \"to\": \"<table.column>\", \"type\": \"INNER|LEFT\" }\n"
    "  ],\n"
    "  \"primary_transaction_table\": \"<most important table for queries>\"\n"
    "}"
)


def run_schema_agent(schema_dict: dict, db_flavor: str, client: LLMClient) -> dict:
    # 25 columns per table is enough to name a table's role; 300-column tables blew the context.
    lines = [f"{t}: {', '.join(c.split(' ')[0] for c in cols[:25])}"
             for t, cols in schema_dict.items()]
    user_msg = f"Database: {db_flavor}\n\nSchema:\n" + "\n".join(lines)
    try:
        res = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SIA_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            # 1024 truncated the JSON on schemas of ~70 tables -> unparseable reply.
            max_completion_tokens=4096,
        )
        raw = res.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "business_type": "unknown", "tables": {}, "metrics": {},
            "joins": [], "primary_transaction_table": next(iter(schema_dict), ""),
            "_error": str(e),
        }


def enrich_context_joins(context: dict, schema_dict: dict) -> dict:
    if not context.get("joins"):
        context["joins"] = infer_joins(schema_dict)
    return context

# Server-side generation lets the model say a question is out of scope: word matching
# refused 'sales by month' on an ORDERS table and embedding scores could not separate
# real questions from 'flight bookings' (measured), while the model sees the tables.
NOT_ANSWERABLE = "NOT_ANSWERABLE"


def clean_sql(llm_text: str) -> str:
    """The SQL out of a model reply: no code fences, no prose before it, one statement."""
    q = re.sub(r"^```[a-z]*\n?", "", llm_text.strip())
    q = re.sub(r"\n?```$", "", q).strip()
    # count=1: without it, a reply that already starts with SELECT lost everything up to its
    # next SELECT (any subquery or CTE), and the half-query failed to parse.
    q = re.sub(r"^.*?(?=SELECT\b|WITH\b)", "", q, count=1, flags=re.IGNORECASE | re.DOTALL)
    quoted = False
    for i, ch in enumerate(q):   # one statement: stop at the first ';' that is not inside a string literal
        if ch == "'":
            quoted = not quoted
        elif ch == ";" and not quoted:
            q = q[:i]
            break
    return q.strip().rstrip(";").strip()


def clean_query(llm_text: str, db_type: str = "sql") -> str:
    """The query out of a model reply: SQL/CQL through clean_sql; a MongoDB JSON filter only loses
    its code fences (its values may contain 'with' or ';')."""
    if db_type != "mongo":
        return clean_sql(llm_text)
    q = re.sub(r"^```[a-z]*\n?", "", llm_text.strip())
    return re.sub(r"\n?```$", "", q).strip()


def improve_sql(sql: str, reviewer, question: str, client=None, messages: list | None = None) -> tuple[str, list[str]]:
    """Safe rewrites, then (if the review finds something worth fixing and an LLM is at
    hand) ONE repair round told the exact reasons, then the rewrites again. Returns the
    SQL and notes for developers (what was rewritten, repaired, or is still worth knowing)."""
    sql, applied = reviewer.rewrite(sql)
    notes = [f"rewritten for speed: {a}" for a in applied]
    findings = [f for f in reviewer.review(sql, question) if f.fix]
    if findings and client is not None and messages:
        ask = ("This SQL has problems:\n- " + "\n- ".join(f.message for f in findings)
               + "\nReturn the corrected SQL only.")
        try:
            res = client.chat.completions.create(
                temperature=0.0, max_completion_tokens=512,
                messages=list(messages) + [{"role": "assistant", "content": sql}, {"role": "user", "content": ask}])
            fixed = clean_sql(res.choices[0].message.content)
            if fixed and not said_not_answerable(fixed):
                sql, applied = reviewer.rewrite(fixed)
                notes += [f"repaired: {f.code}" for f in findings] + [f"rewritten for speed: {a}" for a in applied]
        except Exception as e:   # a failed repair keeps the original SQL; the validator still guards it
            print(f"[review] repair round failed: {e}", file=sys.stderr)
    notes += [f"note: {f.message}" for f in reviewer.review(sql, question)]
    return sql, notes


def said_not_answerable(llm_text: str) -> bool:
    return NOT_ANSWERABLE in llm_text.upper()

# ═══════════════════════════════════════════════════════════════
#  AUTO-FIX  (one LLM retry on execution error)
# ═══════════════════════════════════════════════════════════════

def attempt_fix(bad_query: str, error: str, schema_str: str, flavor: str, client: LLMClient,
                language: str = "") -> str:
    """One repair round. language: the query format for MongoDB / Cassandra (dialects.LANGUAGE)."""
    res = client.chat.completions.create(
        messages=[{"role": "user", "content": (
            f"You are an expert {flavor} query debugger.\n\n"
            + (f"Query format: {language}\n\n" if language else "")
            + f"Schema:\n{schema_str}\n\nBroken Query:\n{bad_query}\n\nError:\n{error}\n\n"
            "Fix the query. Return ONLY the corrected query, nothing else."
        )}],
        temperature=0.0,
        max_completion_tokens=512,
    )
    fixed = res.choices[0].message.content.strip()
    fixed = re.sub(r"^```[a-z]*\n?", "", fixed)
    fixed = re.sub(r"\n?```$", "", fixed).strip()
    return fixed

# ═══════════════════════════════════════════════════════════════
#  MEMORY HELPERS
# ═══════════════════════════════════════════════════════════════

def update_memory(question: str, context: dict, memory: dict) -> dict:
    alias_map = {
        "revenue": "value_column", "sales":    "value_column",
        "income":  "value_column", "amount":   "value_column",
        "date":    "date_column",  "time":      "date_column",
        "month":   "date_column",  "quantity":  "qty_column",
        "qty":     "qty_column",   "stock":     "qty_column",
    }
    q_lower = question.lower()
    for word, meta_key in alias_map.items():
        if word in q_lower:
            for tbl, meta in context.get("tables", {}).items():
                col = meta.get(meta_key)
                if col:
                    memory[f"{word}_column"] = col
    return memory


def make_mongo_prompt(schema_str: str, valid_collections: list, context: dict) -> str:
    biz   = context.get("business_type", "")
    hints = "\n".join(
        f"  - {c}.{k.split('_')[0]} = '{context['tables'][c][k]}'"
        for c in valid_collections
        if c in context.get("tables", {})
        for k in ("date_column", "value_column")
        if context["tables"][c].get(k)
    )
    return (
        f"You are a MongoDB query expert.\n"
        + (f"Business Domain: {biz}\n" if biz else "")
        + (f"Field Hints:\n{hints}\n" if hints else "")
        + f"VALID COLLECTIONS (use only these exact names): {', '.join(valid_collections)}\n\n"
        + f"Schema:\n{schema_str}\n\n"
        'Return ONLY one JSON object, in one of two forms:\n'
        '  find:      {"collection":"<name>","filter":{},"projection":null,"sort":null,"limit":50}\n'
        '  aggregate: {"collection":"<name>","pipeline":[{"$match":{}},{"$group":{"_id":"$field","n":{"$sum":1}}}]}\n\n'
        'Rules:\n'
        '- Use find to list documents; use aggregate for counts, totals, averages, grouping, top-N or joins.\n'
        '- filter / $match: valid MongoDB filter. {} for no filter. Dotted paths reach nested fields (a.b.c).\n'
        '- Text search: {"field":{"$regex":"term","$options":"i"}}\n'
        '- sort: null or {"field":1/-1}. limit: integer default 50.\n'
        f'- Pipeline stages allowed: {", ".join(sorted(READ_STAGES))}. $lookup only from the valid collections.'
    )


def make_cassandra_prompt(schema_str: str, keyspace: str, context: dict) -> str:
    biz   = context.get("business_type", "")
    hints = "\n".join(
        f"  - {t}.{k.split('_')[0]} = '{m[k]}'"
        for t, m in context.get("tables", {}).items()
        for k in ("date_column", "value_column") if m.get(k)
    )
    return (
        f"You are a CQL expert. Keyspace: {keyspace}\n"
        + (f"Business Domain: {biz}\n" if biz else "")
        + (f"Field Hints:\n{hints}\n" if hints else "")
        + f"Schema:\n{schema_str}\n\n"
        "Rules:\n"
        "- Return ONLY raw CQL SELECT — no markdown, no explanation.\n"
        "- Use ALLOW FILTERING for non-partition-key filters.\n"
        "- Always add LIMIT (default 50).\n"
        "- No JOINs in Cassandra."
    )


def build_llm_messages(sys_prompt: str, question: str, history: list) -> list:
    messages = [{"role": "system", "content": sys_prompt}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    return messages


HISTORY_TURNS = 4          # two exchanges: enough for "now only March", not a growing transcript

_DIALECT_NOTES = {
    "MySQL":      "Use MySQL syntax. Backtick identifiers. LIMIT N.",
    "MariaDB":    "Use MariaDB/MySQL syntax. Backtick identifiers. LIMIT N.",
    "PostgreSQL": "Use PostgreSQL syntax. Double-quote identifiers if needed. LIMIT N.",
    "SQL Server": "Use T-SQL. SELECT TOP N instead of LIMIT. Square-bracket identifiers.",
    # Measured on 1.2M orders: SQLite runs a correlated EXISTS as a loop over every outer row
    # (6 s) where joining from the filtered side took 0.12 s; cost-based engines plan both alike.
    "SQLite":     "Use SQLite syntax. LIMIT N. On large tables avoid correlated EXISTS (SQLite loops over every "
                  "outer row): join from the filtered side and use COUNT(DISTINCT ...) instead.",
}

# Generic SQL rules (true for every database), phrased against the join annotations the
# profile writes ("many X rows per Y row"), not against any particular schema.
GRAIN_RULE = ("A join on the 'many' side repeats the other table's rows: never SUM or COUNT a column of the "
              "'one' side after joining its many-side table; aggregate where the amount lives, or pre-aggregate "
              "in a subquery.")
NO_INVENTED_NETTING_RULE = "Do not subtract refunds, payments or discounts unless the question asks for a net figure."
VALUE_RULE = "Bracketed values show what a column holds: use those exact values and units (a 0-100 range is a percentage)."
# How to write the fast version of the same query (generic; leans on the 'indexed' and
# '~N rows' facts the profile prints). Applies to every database.
PERFORMANCE_RULES = (
    "Exact first: EXISTS / NOT EXISTS for 'has any' / 'never' when only the main table's rows are wanted; when two "
    "many-side tables are needed, aggregate each per parent in its own subquery before joining; WHERE for rows, "
    "HAVING only for aggregates; never DISTINCT to hide join duplicates. Then fast: on large tables compare bare "
    "'indexed' columns (col >= x AND col < y, not YEAR(col), CAST(col), LOWER(col) = ...); select only the columns "
    "needed, never SELECT *; UNION ALL unless duplicates must go; ORDER BY with a tie-breaker before LIMIT/TOP.")


def usable_metrics(context: dict, tables: list[str]) -> dict[str, str]:
    """Metric formulas from the automatic connect-time analysis whose every table is in the
    prompt. They are guesses (one defined revenue as payments minus refunds, and the model
    then refused a product-revenue question it could answer), so they are offered as
    unverified suggestions, never as rules; human glossary notes are the authority."""
    present = {t.lower() for t in tables}
    out = {}
    for name, formula in (context.get("metrics") or {}).items():
        used = {m.lower() for m in re.findall(r"([A-Za-z_][\w]*)\.[A-Za-z_]", str(formula or ""))}
        if formula and used and used <= present:
            out[name] = str(formula)
    return out


def _sql_system_prompt(flavor: str, plan: dict, tables_text: str, context: dict, memory: dict,
                       tables: list[str]) -> str:
    lines = [f"You are an expert {flavor} SQL query generator.", _DIALECT_NOTES.get(flavor, "Use standard SQL. LIMIT N.")]
    if plan.get("time_range"):
        lines.append(f"Time filter hint: >= {plan['time_range']}")
    if plan.get("top_n") and plan["top_n"] != 50:
        lines.append(f"Return the top {plan['top_n']} rows.")
    metrics = usable_metrics(context, tables)
    if metrics:
        lines.append("Unverified suggestions from an automatic analysis (use only if they fit the question): "
                     + "; ".join(f"{k} = {v}" for k, v in metrics.items()))
    if memory:
        lines.append("Learned mappings: " + "; ".join(f"'{k}' means {v}" for k, v in memory.items()))
    lines += ["", tables_text, "", "Rules:",
              '- Give each output column a short plain-English alias in double quotes, e.g. AS "Total amount".',
              f"- If these tables cannot answer the question, reply exactly {NOT_ANSWERABLE}.",
              f"- {GRAIN_RULE}", f"- {NO_INVENTED_NETTING_RULE}", f"- {VALUE_RULE}", f"- {PERFORMANCE_RULES}",
              "- Lists get a LIMIT (default 50). Return ONLY raw SQL: no markdown, no explanation."]
    return "\n".join(lines)


# ── choosing the tables: a model reads a short catalog, code checks its answer ──
# Code-only matching (word overlap, embeddings) cannot tell that 'staff department' means the
# RBAC users table, or that 'billing' has no table at all; a model reading one line per table can.
# The model that plans may differ from the one that writes SQL (PLANNER_MODEL, same provider).
TABLE_PICKER = os.getenv("TABLE_PICKER", "true").lower() not in ("0", "false", "no")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "") or LLM_MODEL
CATALOG_MAX = 60          # tables described to the picker; a larger database is narrowed first
PICK_CACHE_MAX = 200

_PICK_SYSTEM = (
    "You choose which tables of a database are needed to answer a question.\n"
    'Reply with JSON only: {"tables": ["..."], "missing": ["..."]}\n'
    "- tables: every table needed, including a table that only links two others and a table that holds "
    "the names or labels the answer should show. Use the names exactly as listed. At most %d.\n"
    "- missing: parts of the question that no listed table can answer, in a few words each. Put such a "
    "part here; never add a loosely related table for it.\n"
    "- Prefer the main table over a dated or backup copy of it.")


def pick_tables(client, question: str, lines: list[str], names: list[str],
                context: str = "") -> tuple[list[str], list[str]] | None:
    """(tables, missing topics) the model chose from the catalog `lines`, names checked against
    `names`; None when the model is unavailable or answers nothing usable (the code's own choice then stands)."""
    ask = (f"Earlier in this conversation:\n{context}\n\n" if context else "") + \
          f"Question: {question}\n\nTables:\n" + "\n".join(lines)
    try:
        res = client.chat.completions.create(
            model=getattr(client, "planner_model", "") or PLANNER_MODEL, temperature=0.0, max_completion_tokens=400,
            messages=[{"role": "system", "content": _PICK_SYSTEM % MAX_PICKED_TABLES},
                      {"role": "user", "content": ask}])
        found = re.search(r"\{.*\}", res.choices[0].message.content, re.S)
        reply = json.loads(found.group(0)) if found else {}
    except Exception as e:   # unavailable, rate limited or garbled: fall back, but say why
        print(f"[pick] table picker failed: {e}", file=sys.stderr)
        return None
    by_lower = {n.lower(): n for n in names}
    chosen = list(dict.fromkeys(by_lower[t.lower()] for t in reply.get("tables", [])
                                if isinstance(t, str) and t.lower() in by_lower))
    if not chosen:
        return None
    return chosen, [m for m in reply.get("missing", []) if isinstance(m, str)][:5]


def _plan_tables(picker, svc: ProfileService, asm: ContextAssembler, question: str, candidates: list[str],
                 history: list) -> tuple[list[str] | None, list[str]]:
    """The picker's tables for this question, cached per database. (None, []) = no picker."""
    if picker is None or not TABLE_PICKER or len(svc.profile.tables) <= MAX_PROMPT_TABLES:
        return None, []   # everything fits the prompt anyway, or no model to ask
    recent = [f"{h['role']}: {h['content'][:300]}" for h in list(history)[-HISTORY_TURNS:]]
    key = (" ".join(question.lower().split()), tuple(recent))
    if key not in svc.picks:
        names = asm.catalog_tables(question, candidates, CATALOG_MAX)
        result = pick_tables(picker, question, [asm.catalog_line(n) for n in names], names, "\n".join(recent))
        if result is None:
            return None, []
        if len(svc.picks) >= PICK_CACHE_MAX:
            svc.picks.pop(next(iter(svc.picks)))
        svc.picks[key] = result
    return svc.picks[key]


def build_sql_messages(question: str, schema: dict, flavor: str, *, context: dict, memory: dict,
                       retriever=None, ke=None, notes: list[dict] = (), history: list = (),
                       profile: ProfileService | None = None, db_type: str = "sql",
                       keyspace: str = "", picker=None, scoped: bool = False) -> tuple[list, dict, str]:
    """The one prompt builder for every server-side generation path (web query, stream,
    hybrid, MCP query_database). For SQL everything comes from the connection's profile:
    tables chosen by a model reading the table catalog (`picker`) or, without one, by relevance;
    joined through the relationship graph, columns with their value hints, joins with their
    direction. Returns (messages, tables, schema text)."""
    examples = []
    if ke:
        try:
            examples = [q for q in ke.get_context_for_question(question, db_type).get("similar_queries", [])
                        if q.get("verified")][:2]
        except Exception as e:
            print(f"[prompt] knowledge lookup failed: {e}", file=sys.stderr)

    ranked = list(retriever.retrieve(question, top_k=CATALOG_MAX if db_type == "sql" else 4, db_type=db_type)) \
        if retriever else []
    missing: list[str] = []
    if db_type == "sql":
        svc = profile or ProfileService.open(flavor=flavor, schema=schema, foreign_keys=[])
        plan = build_query_plan(question, schema, [], flavor=flavor)
        asm = ContextAssembler(svc, rank_columns=rank_columns)
        words = question_words(question)
        glossary_tables = [
            n["table"] for n in notes
            if n.get("table") in schema
            and words & question_words(n.get("note", ""))
        ]
        candidates = list(dict.fromkeys(glossary_tables + ranked + list(plan["tables"])))
        picked, missing = _plan_tables(picker, svc, asm, question, candidates, history)
        ctx = asm.assemble(question, candidates, picked=picked)
        tables_text = ctx.render(svc, hints=not scoped)   # scoped: no other tenant's values in the prompt
        sys_prompt = _sql_system_prompt(flavor, plan, tables_text, context, memory, list(ctx.tables))
        if not ctx.complete:
            sys_prompt += (
                "\n\nThe selected context is incomplete: no supported relationship path connects "
                + ", ".join(ctx.unconnected_terminals)
                + ". Do not invent joins; ask for clarification or report that the schema lacks a supported path."
            )
        if missing:
            sys_prompt += ("\n\nNo table in this database covers: " + "; ".join(missing)
                           + ". Answer the rest without inventing a table for these parts; "
                           f"if nothing is left to answer, reply {NOT_ANSWERABLE}.")
        matched = {t: cols for t, cols in ctx.tables.items()}
    elif profile is not None:   # MongoDB / Cassandra with a profile: same context as SQL (hints, references)
        ctx = ContextAssembler(profile, rank_columns=rank_columns).assemble(question, ranked)
        tables_text = ctx.render(profile, hints=not scoped)   # scoped: no other tenant's values in the prompt
        matched = {t: cols for t, cols in ctx.tables.items()}
        sys_prompt = (make_mongo_prompt(tables_text, list(matched), context) if db_type == "mongo"
                      else make_cassandra_prompt(tables_text, keyspace, context))
    else:
        matched = compress_schema({t: schema[t] for t in ranked if t in schema}, question, max_cols=10)
        tables_text = build_schema_str(matched, db_type)
        sys_prompt = (make_mongo_prompt(tables_text, list(matched), context) if db_type == "mongo"
                      else make_cassandra_prompt(tables_text, keyspace, context))

    relevant = [n for n in notes if not n.get("table") or n["table"] in matched][:8]
    if relevant:
        sys_prompt += "\n\nNotes from people who know this database (follow them):\n" + "\n".join(
            f"  - {n['table'] + ('.' + n['column'] if n.get('column') else '') + ': ' if n.get('table') else ''}"
            f"{n['note'][:200]}" for n in relevant)
    if examples:
        label = {"mongo": "MongoDB", "cassandra": "CQL"}.get(db_type, "SQL")
        sys_prompt += "\n\nVerified examples:\n" + "\n".join(
            f"  Q: {q['question'][:150]}\n  {label}: {q['sql'][:400]}" for q in examples)
    return build_llm_messages(sys_prompt, question, list(history)[-HISTORY_TURNS:]), matched, tables_text
