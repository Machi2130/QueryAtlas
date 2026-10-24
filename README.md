# SQL-Agent — ask your database questions from Claude Code

![python](https://img.shields.io/badge/python-3.12-blue)
![stack](https://img.shields.io/badge/MCP%20%C2%B7%20FastAPI%20%C2%B7%20SQLAlchemy%20%C2%B7%20ChromaDB%20%C2%B7%20Docker-informational)

<p align="center">
  <img src="docs/architecture.svg" width="100%"
       alt="SQL-Agent: questions from developers (MCP) and teams (web app) pass four rings before reaching the databases — sign-in and roles, the read-only validator, the tenant filter and encrypted connections. The AI model and the knowledge layer help write each query, every request is audited, and an unsafe query is stopped at the validator.">
</p>
<p align="center">
  <img src="docs/how-it-works.svg" width="100%"
       alt="Inside SQL-Agent: a question typed in the chat becomes a glowing packet that travels through the engine. The access check confirms who is asking and their role; the router binds the chat to one of several open databases; the knowledge vault is scanned and 40 tables shrink to the 2 that matter, with a data-confirmed join, a glossary note and value hints; the query is written by SQL-Agent's AI or by Claude; the validator seals it (read-only, no credential columns, tenant filter attached, row cap and time limit); it runs through an encrypted tunnel with a database-enforced countdown; rows return as a table, a chart and a one-line summary; and once confirmed, the verified example flows back into the vault for next time. The audit tape records each step and a subtitle narrates.">
</p>

An MCP server that gives an AI assistant (Claude Code, Claude Desktop, Cursor) a
**governed, knowledge-backed connection** to databases: it learns each database once,
hands the model only the tables a question needs, enforces read-only access, and audits
every query. Supports SQL Server, MySQL/MariaDB, PostgreSQL, SQLite, MongoDB and Cassandra.

## Why this exists

Large language models are good at writing SQL and poor at three things that matter the
moment they touch a real database:

- **Context.** A production schema has hundreds of tables. A model that receives the
  whole schema spends most of its context budget on irrelevant tables and still guesses
  the joins; a model that receives nothing hallucinates table names. Neither knows that an
  opaque column such as `Account_Ref` holds the customer's card number.
- **Safety.** Nothing in the model prevents an `UPDATE`, a `SELECT INTO`, a query that
  returns fifty million rows, or — in a database shared by twenty clients — a total that
  silently mixes their data.
- **Governance.** Generic MCP database connectors are thin wrappers: one shared
  credential, no record of who ran what, no separation between users.

SQL-Agent addresses each one. A per-database **knowledge layer** (real foreign keys,
schema embeddings, a human glossary, remembered question→SQL pairs) sends the model only
what a question needs — about 700 tokens whatever the schema size, measured at 67% less
than the full schema on a 42-table database, 89% less at 120 tables and 97% less at 300 —
and removes join guessing. A **validation layer** enforces
read-only SQL, a row cap and a mandatory tenant filter, and routes any data change through
a human confirmation step. A **governance layer** provides per-user tokens, isolated
knowledge per user, and an audit line for every statement. The result is a database
connection an organisation can hand to an AI assistant without handing over the keys.

```mermaid
flowchart LR
    U[Analyst or developer<br/>in Claude Code / Cursor] -->|question| M[LLM]
    M -->|MCP tools| S[SQL-Agent server]
    S --> K[Knowledge per database<br/>foreign keys · embeddings · glossary · remembered queries]
    K -->|only the relevant tables + joins| M
    M -->|SELECT| V{Validator<br/>read-only · tenant filter · row cap}
    V -->|ok| D[(Databases<br/>SQL Server · PostgreSQL · MySQL · SQLite · MongoDB · Cassandra)]
    V -->|rejected, with the fix| M
    D -->|rows| M --> U
    S --> A[audit.log<br/>who · what · when]
```

## Setup (once per machine, ~5 minutes)

1. **Python 3.12** and **Git**.
2. Clone and install:
   ```
   git clone https://github.com/Machi2130/SQL-Agent.git
   cd SQL-Agent
   pip install -r requirements.txt
   ```
3. **SQL Server only:** install *ODBC Driver 17 for SQL Server*.
   - Windows: Microsoft's installer (search "ODBC Driver 17 for SQL Server download").
   - macOS: `brew tap microsoft/mssql-release && brew install msodbcsql17`
4. Create your own `.env`:
   ```
   cp .env.example .env
   ```
   Fill in `GROQ_API_KEY` (your own key — needed only when the server writes the SQL:
   `query_database` and the web app; any OpenAI-compatible API works via `LLM_BASE_URL` /
   `LLM_MODEL`, including a local Ollama)
   and, optionally, `MCP_DB_PASSWORD` so the database password is not entered on every
   connection (or one per database, `MCP_DB_PASSWORD_<DBNAME>`, when several servers are used).
   Everything else can stay as is. **Never commit `.env`.**
5. Register the server with Claude Code:
   ```
   python mcp_server.py --install            # optional: --user <your-name>
   ```
   This registers stdio mode with this exact Python and the absolute path (replacing any earlier
   `sql-agent` entry), so Claude Code starts the server itself — nothing to keep running.
   Restart Claude Code and check: `claude mcp list` should show `sql-agent … ✔ Connected`.
   The first start downloads an ~80 MB embedding model.

   **If it doesn't connect:**
   - `ECONNREFUSED` — Claude Code is registered for HTTP (`http://…/mcp`) but no server is running.
     Run `python mcp_server.py --install` to switch to stdio, or start `python mcp_server.py --http …` first.
   - `Connection closed` at startup — run `python mcp_server.py` in a terminal; the traceback shows the cause
     (usually a missing package: `pip install -r requirements.txt`).
   - Settings in `.env` are read from the project folder, wherever Claude Code is started.

## Use

Start `claude` in any folder and ask, for example:

> connect to SQL Server db.example.com, database SalesDB, as reporting_user, and find the customer with this mobile number

If the connection details are not supplied, Claude Code shows a form asking for
host, port, username, password and database — the password goes straight to
the server, never through the chat. On the first connect to a database the
server learns its schema (a minute or two on big schemas); later connects are instant.

How a question is answered when Claude is the client: on connect the server builds a
knowledge graph of the database (table/column embeddings + relationships, no LLM).
`get_query_context(question)` then returns just the relevant tables, their join paths,
business notes and similar past queries — about 700 tokens instead of the whole schema;
every result carries its own `context_tokens` measure, and `describe_capabilities` shows the
running `context_savings` — Claude writes the SELECT, and `run_sql` executes it and remembers the
question→SQL pair as *unverified*. When you confirm or correct an answer, `mark_query`
makes it a verified example (shown first next time) or deletes it; a wrong query is never
re-suggested just because it ran. Groq is only used by `query_database` (the web app /
clients without a strong model). If a question names nothing in the database,
`get_query_context` returns `low_confidence: true` and Claude asks instead of guessing.

Join paths come from the database's real foreign keys when it has them (read once per
connect), with name-based inference only filling gaps.

**Teach it once.** Facts the schema cannot express — "`ACCOUNT_LINK.Account_Ref` is the
card number", "amounts are in local currency" — go in as glossary notes:
`add_note(note, table?, column?)`. They're stored with your private knowledge of that
database, shown only when a question touches that table, and used to pick tables (ask
about "card" and that table comes in). `list_notes` / `delete_note` to manage them.

**One database, many clients.** If the schema shares a column across most tables
(`Tenant_Id`, `Client_Id`…), `connect_database` reports it as a *scope candidate*. Set it
once per connection — `set_scope(column="Tenant_Id", value="7", label="Acme Retail")` — and
the server rewrites every reference to those tables (joins, subqueries, CTEs) to
`(SELECT * FROM t WHERE Tenant_Id = '7')`, so no query shape can reach another client's
rows. Empty value clears it.

**It learns what matters.** `import_query_history` reads the database's own execution
statistics (which tables the application and team query most) and `import_queries` ingests
the SQL people already have; `run_sql` keeps counting. Popular tables rank first and show as
`hot_tables`, so `orders` wins over a stale `orders_archive` by default.

Tools: `connect_database`, `get_query_context`, `get_schema`, `run_sql` (SELECT/WITH
only — INSERT/UPDATE/DELETE/DDL are rejected), `execute_write` (developer mode),
`add_note` / `list_notes` / `delete_note`, `mark_query`, `query_database` (Groq writes the SQL),
`explain_query`, `suggest_chart`, `upload_document` / `search_documents`, `disconnect`.

## Safety: read-only by default, developer mode on request

- `run_sql` / `query_database` are strictly read-only (parsed per dialect: one SELECT/UNION
  statement, no data-changing node anywhere, no `SELECT INTO`, every table checked, blocked keywords), capped at `MCP_MAX_ROWS` rows (default 5 000; the result says
  `truncated: true` when there was more). For real protection use a **read-only database login**
  (`db_datareader`) for the server — the SQL guard is the second line, not the first. SQL Server:
  ```sql
  CREATE LOGIN sql_agent_ro WITH PASSWORD = '<long-random-password>';
  USE [YourDatabase];
  CREATE USER sql_agent_ro FOR LOGIN sql_agent_ro;
  ALTER ROLE db_datareader ADD MEMBER sql_agent_ro;
  ```
  Repeat the `USE … CREATE USER … ALTER ROLE` block per database the agent should read.
- Developers who need to change data use **`execute_write`**: the server must set
  writes on in the **Settings → Database writes** panel, the user's role must include
  `sql.write`, and the connection must be explicitly opened with `allow_writes=true` (using
  the developer's own write-capable credentials). Each statement is shown to the user in a
  confirmation form, `UPDATE`/`DELETE` need a `WHERE`, and DDL must be separately enabled
  in Settings. Changes to roles and write settings apply without a server restart; connected
  MCP clients are notified when their tool list should refresh. `dry_run=true` executes
  inside a transaction, reports rows affected, and rolls back — use it first.
  `MCP_ALLOW_WRITES` and `MCP_ALLOW_DDL` are first-run defaults for the settings file, not a
  replacement for the Settings UI, role permission, or per-connection opt-in.
- Every query and write is appended to `audit.log` (JSON lines: who, connection, SQL, rows,
  duration, outcome). Give each person their own token (`MCP_AUTH_TOKENS`) so "who" is a name.

## Several databases at once

Connections stay open. Connect to DB1, work, connect to DB2 — DB1 is still open;
say "switch back to DB1" and it's instant (no reconnect, no re-learning). Each
database keeps its own schema knowledge. `disconnect` closes only the current one,
and `describe_capabilities` lists what's open. Databases on the *same* SQL Server
instance can also be joined in one query with three-part names (`[OtherDB].dbo.Table`).

## Hosting it for a team (nobody installs anything, code stays on the server)

Run the server once, inside the network that can reach your databases, and let
people connect over HTTP.

```
# on the host (after setting MCP_AUTH_TOKEN and GROQ_API_KEY in .env):
python mcp_server.py --http 0.0.0.0:8000
#   or
docker build -t sql-agent . && docker run -d --name sql-agent -p 8000:8000 \
    --env-file .env -v sql-agent-knowledge:/app/.chroma_data sql-agent
```

Each colleague then runs one command — no Python, no drivers, no model download:

```
claude mcp add --transport http sql-agent http://<host>:8000/mcp --header "Authorization: Bearer <MCP_AUTH_TOKEN>"
```

**Everything is per person.** Each user's connections, learned schema knowledge,
glossary notes, remembered queries and usage counters are private to them — nothing is
shared between users, even on a team instance. Identity comes from the token, so give
every person their own entry in `MCP_AUTH_TOKENS` (a shared `MCP_AUTH_TOKEN` would make
everyone the same user). DB credentials are entered per person through the connection
form (or set `MCP_DB_PASSWORD_<DB>` on the host if everyone should use one account). Put
the server behind HTTPS (any reverse proxy) if it's reachable beyond your VPN, and remove
a person's token line to revoke access.

**How database credentials are protected when hosted** (`--http`, and the web app always):
- **Only through the form.** Passwords, LLM keys and "trust this certificate" are taken only from the
  connection form, never from tool arguments, so a password typed into the chat is not used (it would
  have passed through the LLM provider).
- **Always encrypted, certificate checked.** PostgreSQL `verify-full`, MySQL/MariaDB TLS with CA check
  (the login is refused if the server offers no TLS), SQL Server `TrustServerCertificate=no`, MongoDB and
  Cassandra TLS. A database with a self-signed certificate needs *Trust this server's certificate* ticked
  in the form (still encrypted; recorded in `audit.log`).
- **No internal networks.** A host that resolves to a private, loopback, link-local (cloud metadata) or
  reserved address is refused. List your own internal databases in `MCP_ALLOWED_HOSTS` to allow them;
  the server's `MCP_DB_PASSWORD*` are only ever used for those listed hosts. SQLite only inside `MCP_SQLITE_DIR`.
- **Never stored, never shown.** Passwords stay in memory for the open connection only (not in Redis,
  files, logs or knowledge), and connection errors are fixed sentences, never the driver's text.
- **For the customer's IT team:** create a dedicated **read-only** database login for SQL-Agent
  (see *Safety* above). Sign-ups get the `customer` role (connect and read); writes need an admin to grant
  `sql.write`, a connection opened with *Allow data edits*, and a typed YES for every statement.

## Repository layout

```
mcp_server.py            launcher: python mcp_server.py [--http host:port]
sql_agent/
  server.py              MCP tools, connections, knowledge wiring, validation gates, audit, transports
  app.py                 shared engine: schema fetch, retriever, SQL validator, Groq prompts
  knowledge.py           RAG store + relationship graph + business-context cache
  query_planner.py       intent/time-range detection for the Groq path
  docstore.py            uploaded documents (chunk, embed, hybrid search)
  cache.py               sessions and caches (Redis, in-memory fallback)
  storage.py, auth.py    persistence and auth helpers
docs/                    ARCHITECTURE.md · SECURITY.md · EXAMPLES.md
Dockerfile               hosted deployment
```

Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · threat model and guards:
[`docs/SECURITY.md`](docs/SECURITY.md) · real sessions: [`docs/EXAMPLES.md`](docs/EXAMPLES.md).

## Notes

- In local (stdio) mode knowledge is stored under `.chroma_data/` in the project folder and is not shared.
- Redis is optional; without it sessions live in memory for the life of the server process.
- `app.py` holds the shared query engine (schema retrieval, validation, Groq prompts); the
  companion web UI is not part of this repository.
