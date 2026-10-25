"""What the agent can call: the tool definitions as data, the permission each needs and which
tools wait behind the idle lock. inputSchema is standard JSON Schema."""

from __future__ import annotations

from mcp import types

from .. import rbac


def tools(server_prefers_query: bool) -> list[types.Tool]:
    """The tool list for this session; descriptions depend on who generates SQL."""
    query_db_desc = (
        "PREFERRED (1 round-trip): Ask a natural language question about the connected database. "
        "The server's AI generates, reviews, repairs, and executes the SQL in a single step and returns "
        "columns, rows, the generated SQL, and usage metrics. Call this directly instead of get_query_context + run_sql."
        if server_prefers_query
        else "Ask a natural language question about the connected database (server-side NL->SQL). "
        "Returns columns, rows, the generated SQL, and usage metrics. "
        "(Note: client SQL generation mode is active — prefer get_query_context + run_sql.)"
    )
    get_ctx_desc = (
        "Uses the database's knowledge graph (built on connect, no LLM) to return ONLY what a question needs: "
        "the relevant tables with their columns, the join paths between them, business notes/metrics for those "
        "tables, and similar past question->SQL pairs. (Note: server SQL generation is active — prefer calling "
        "query_database(question) for a 1-step answer unless you need to write custom SQL yourself.)"
        if server_prefers_query
        else "Start here before writing SQL. Uses the database's knowledge graph (built on connect, no LLM) to "
        "return ONLY what a question needs: the relevant tables with their columns, the join paths between "
        "them, business notes/metrics for those tables, and similar past question->SQL pairs. Far cheaper "
        "than get_schema, which lists every table. How to use the result: write ONE read-only query in the "
        "'query_language' given, using only the tables/columns returned, then call run_sql with it and the "
        "question. If a business term (e.g. 'successful', 'revenue') has no glossary note, ask the user what it "
        "means instead of guessing. 'other_tables' names what else exists: if a part of the question needs one of "
        "those, call get_schema(tables=[...]) for it (it also returns the joins among the tables you ask for); if "
        "no table fits a part, say so instead of using a loosely related one. 'low_confidence': true means "
        "nothing matched well — ask the user which data they mean rather than guessing."
    )
    tools = [
        types.Tool(
            name="connect_database",
            description=(
                "Connect to a database, or switch to one that is already open. Must be called before any query. "
                "Supports SQLite, MySQL, MariaDB, PostgreSQL, SQL Server, MongoDB, Cassandra. "
                "Several databases can be open at once: calling this for a database that is already "
                "connected just makes it the current one (instant, no reconnect); the others stay open. "
                "Pass session_id to run multiple independent agent sessions in parallel."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "db_type": {
                        "type": "string",
                        "description": "One of: SQLite, MySQL, MariaDB, PostgreSQL, SQL Server, MongoDB, Cassandra",
                    },
                    "sql_generator": {
                        "type": "string",
                        "enum": ["auto", "server", "client"],
                        "default": "",
                        "description": (
                            "Who generates SQL for this session: 'server' (QueryAtlas's AI in 1 round-trip via query_database), "
                            "'client' (the MCP client AI writes SQL via get_query_context + run_sql), or 'auto'. "
                            "If omitted, ask the user which they prefer and pass it here or call set_sql_generator."
                        ),
                    },
                    "groq_key": {
                        "type": "string",
                        "description": "Optional. Groq API key for query_database (server-side NL→SQL). Omit when the calling AI writes SQL itself and uses run_sql.",
                        "default": "",
                    },
                    "host": {
                        "type": "string",
                        "default": "localhost",
                        "description": "Host name or IP. MongoDB Atlas: mongodb+srv://cluster0.xxxx.mongodb.net "
                        "(no username or password in the address)",
                    },
                    "port": {"type": "string", "default": ""},
                    "username": {"type": "string", "default": ""},
                    "password": {"type": "string", "default": ""},
                    "database": {"type": "string", "default": ""},
                    "keyspace": {"type": "string", "description": "For Cassandra: keyspace name", "default": ""},
                    "db_path": {"type": "string", "description": "For SQLite: path to .db file", "default": ""},
                    "allow_writes": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Opt this connection into data edits. Requires the server and your account to allow "
                            "writes; each INSERT/UPDATE/DELETE/MERGE still requires explicit confirmation. "
                            "DDL remains disabled unless separately enabled by the server."
                        ),
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional agent session token. Omit to use the default rt.user_id session.",
                    },
                    "refresh": {
                        "type": "boolean",
                        "default": False,
                        "description": "Re-read the schema of an already-open database (tables or columns changed).",
                    },
                },
                "required": ["db_type"],
            },
        ),
        types.Tool(
            name="run_sql",
            description=(
                "Execute a read-only query you wrote yourself against the connected database, in its language: "
                "SQL (SELECT/WITH in the database's dialect), MongoDB JSON (a find or an aggregate pipeline) or "
                "CQL (one SELECT). No LLM or Groq key involved: call get_query_context(question) first; its "
                "'query_language' gives the exact format. Pass the original question too so the query is "
                "remembered for next time. Use bare table/collection names exactly as returned. Writes are rejected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "One read-only query: SQL SELECT/WITH, MongoDB JSON, or CQL SELECT",
                    },
                    "question": {
                        "type": "string",
                        "default": "",
                        "description": "The user's question this query answers (remembered as an example)",
                    },
                    "page": {"type": "integer", "default": 1, "description": "Page number (1-based)"},
                    "page_size": {"type": "integer", "default": 50, "description": "Rows per page (max 500)"},
                    "session_id": {
                        "type": "string",
                        "description": "Session token from connect_database. Omit to use default.",
                    },
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="query_database",
            description=query_db_desc,
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question, e.g. 'top 10 customers by revenue'",
                    },
                    "page": {"type": "integer", "default": 1, "description": "Page number (1-based)"},
                    "page_size": {"type": "integer", "default": 50, "description": "Rows per page (max 500)"},
                    "session_id": {
                        "type": "string",
                        "description": "Session token from connect_database. Omit to use default.",
                    },
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="execute_write",
            description=(
                "Developer mode only: run ONE data-changing statement (INSERT, UPDATE, DELETE, MERGE; DDL only if "
                "the server allows it) on the current connection. Requires the connection to have been opened with "
                "allow_writes=true and the server to allow writes; the user confirms every statement in a form. "
                "UPDATE/DELETE must have a WHERE clause. Prefer dry_run=true first: it runs inside a transaction, "
                "reports the affected row count and rolls back. Never use this for SELECT (use run_sql). "
                'MongoDB: one JSON operation {"collection": c, "op": "insertOne|insertMany|updateOne|updateMany|'
                'replaceOne|deleteOne|deleteMany", "filter": {...}, "update": {"$set": {...}} | "document" | '
                '"documents" | "replacement"}; update/delete/replace need a filter on real fields. '
                "Cassandra: one CQL INSERT/UPDATE/DELETE (no BATCH, no dry run: Cassandra cannot roll back)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "One write: SQL statement, MongoDB JSON operation, or CQL statement",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Execute inside a transaction and roll back, reporting rows affected",
                    },
                    "reason": {
                        "type": "string",
                        "default": "",
                        "description": "Why this change is being made (recorded in the audit log)",
                    },
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="get_query_context",
            description=get_ctx_desc,
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question in plain language"},
                    "max_tables": {
                        "type": "integer",
                        "default": 6,
                        "description": "Preferred context size; complete join paths may return more tables",
                    },
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="get_schema",
            description=(
                "Return tables and their columns for the connected database. Without 'tables' it returns EVERY "
                "table (large); prefer get_query_context(question) and use this only to look up specific tables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only these tables (exact names)",
                    },
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="validate_sql",
            description=(
                "Check SQL without running it: parsed in the database's dialect, read-only, every table known. "
                "Returns the exact statement run_sql would execute (row limit and tenant filter applied), or the "
                "reason it would be rejected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="explain_query",
            description=(
                "Return the database execution plan for a SQL query. "
                "Use this to identify slow queries, missing indexes, or sequential scans."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "The SQL SELECT to explain"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="suggest_chart",
            description="Given columns and rows from a query result, suggest the best chart type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array"}},
                },
                "required": ["columns", "rows"],
            },
        ),
        types.Tool(
            name="set_scope",
            description=(
                "For a database shared by several clients/programs: fix the tenant filter for this connection, "
                "e.g. column='Tenant_Id', value='7', label='Acme Retail'. Afterwards every table that has that "
                "column is filtered to that value by the server, in any query shape, so clients' data is never mixed. connect_database "
                "and get_query_context report 'scope_candidates' (columns shared by most tables); list the values "
                "with run_sql on a lookup table (e.g. TENANT_MASTER) and ask the user which one. Empty value clears."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Tenant column, e.g. Tenant_Id"},
                    "value": {"type": "string", "description": "The client's value; empty string clears the scope"},
                    "label": {
                        "type": "string",
                        "default": "",
                        "description": "Human name for the value, e.g. 'Acme Retail'",
                    },
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["column", "value"],
            },
        ),
        types.Tool(
            name="set_sql_generator",
            description=(
                "Set the user's choice for who generates SQL in MCP: "
                "'server' (QueryAtlas's own LLM generates, reviews, and executes SQL in 1 round-trip via query_database) "
                "or 'client' (the MCP client LLM — Claude, Cursor, Copilot — calls get_query_context, writes SQL itself, "
                "and executes it via run_sql), or 'auto'. Ask the user which mode they want after connecting, then call this."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["server", "client", "auto"],
                        "description": "'server' (1 round-trip via query_database), 'client' (2 round-trips via get_query_context + run_sql), or 'auto'",
                    },
                    "groq_key": {
                        "type": "string",
                        "default": "",
                        "description": "Optional LLM API key if switching to 'server' mode",
                    },
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["mode"],
            },
        ),
        types.Tool(
            name="import_query_history",
            description=(
                "Learn which tables matter from the database's own execution statistics (SQL Server plan cache, "
                "MySQL performance_schema, Postgres pg_stat_statements): counts how often each table is queried by "
                "the application and the team. Improves table ranking and 'hot_tables' in get_query_context. "
                "Needs VIEW SERVER STATE (or equivalent); reports if unavailable."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 300, "description": "Top-N statements to read (max 2000)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="import_queries",
            description=(
                "Import existing SQL the team already uses (paste a .sql file or several statements separated by ';'). "
                "Each SELECT becomes a remembered question->SQL example (a leading '-- comment' is used as the "
                "question) and counts toward table popularity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql_text": {"type": "string", "description": "One or more SQL statements"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql_text"],
            },
        ),
        types.Tool(
            name="add_note",
            description=(
                "Teach the current database something a human knows and the schema doesn't, e.g. "
                "'Account_Ref is the card number' on ACCOUNT_LINK, or a database-wide rule like "
                "'amounts are in MUR'. Notes are stored once per database, shown to anyone whose question touches "
                "that table, and used to pick tables (a question mentioning 'card' will pull that table in)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The fact, in one or two sentences"},
                    "table": {
                        "type": "string",
                        "default": "",
                        "description": "Table the note is about (omit for a database-wide note)",
                    },
                    "column": {
                        "type": "string",
                        "default": "",
                        "description": "Column the note is about (requires table)",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["user", "documentation", "inferred"],
                        "default": "user",
                        "description": "user = the user stated it; documentation = from an uploaded/approved document; "
                        "inferred = your own guess from names or data (shown as unverified)",
                    },
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["note"],
            },
        ),
        types.Tool(
            name="mark_query",
            description=(
                "Record the USER's verdict on an answer, only after they confirm or correct it - never on your own "
                "judgement. correct=true makes the question->SQL pair a verified example (shown first in "
                "similar_queries). correct=false removes the pair so it is never suggested again; pass the fixed SQL "
                "as correction to store that as verified instead. run_sql alone only stores unverified examples."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "sql": {"type": "string", "description": "The SQL that was run for the question"},
                    "correct": {"type": "boolean"},
                    "correction": {
                        "type": "string",
                        "default": "",
                        "description": "Corrected SQL (when correct=false)",
                    },
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["question", "sql", "correct"],
            },
        ),
        types.Tool(
            name="list_notes",
            description="List the glossary notes for the current database (optionally only one table's, plus database-wide notes).",
            inputSchema={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "default": ""},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="delete_note",
            description="Delete a glossary note by id (from list_notes).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["id"],
            },
        ),
        types.Tool(
            name="disconnect",
            description="Clear the current database session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="describe_capabilities",
            description=(
                "Return what databases are currently connected, available tools, "
                "supported DB types, and error codes. Call this first to bootstrap "
                "an agent session without human configuration."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="upload_document",
            description=(
                "Index an unstructured document (PDF, DOCX, TXT, MD, CSV) into the document store. "
                "Call this before search_documents. The document is chunked and stored per session. "
                "Use for policies, manuals, contracts, handbooks — any text that contains rules, "
                "thresholds, or context that cannot be queried with SQL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Original filename including extension, e.g. 'aml_policy.pdf'",
                    },
                    "content_b64": {"type": "string", "description": "Base64-encoded file bytes"},
                    "url": {
                        "type": "string",
                        "description": "Alternatively, a URL to fetch and index (PDF or web page). Provide either content_b64 or url, not both.",
                    },
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="search_documents",
            description=(
                "Search the indexed document store for content relevant to a question. "
                "Returns the top matching text chunks with their source filename. "
                "Call this to retrieve policy rules, thresholds, or guidelines before "
                "or after querying the database — then combine both results in your answer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question or topic to search for"},
                    "n_results": {
                        "type": "integer",
                        "default": 4,
                        "description": "Number of chunks to return (max 10)",
                    },
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="list_documents",
            description="List all documents that have been uploaded to the document store for this session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
    ]
    return tools


# Permission each tool needs (the settings file decides which roles have it).
TOOL_PERMISSIONS = {
    "connect_database": rbac.CONNECT,
    "get_schema": rbac.VIEW_TABLES,
    "explain_query": rbac.EXPLAIN,
    "validate_sql": rbac.EXPLAIN,
    "execute_write": rbac.WRITE,
    "add_note": rbac.NOTES,
    "delete_note": rbac.NOTES,
    "set_scope": rbac.SCOPE,
    "mark_query": rbac.APPROVE,
    "import_queries": rbac.APPROVE,
    "import_query_history": rbac.NOTES,
}
# Tools that use the current database: they wait behind the idle lock.
CONNECTION_TOOLS = {
    "run_sql",
    "query_database",
    "execute_write",
    "get_query_context",
    "get_schema",
    "explain_query",
    "validate_sql",
    "add_note",
    "list_notes",
    "delete_note",
    "mark_query",
    "set_scope",
    "set_sql_generator",
    "import_query_history",
    "import_queries",
}
