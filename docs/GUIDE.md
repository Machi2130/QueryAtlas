# QueryAtlas guide

Everything the README leaves out: how a question is answered, the tools, teaching the server about your
data, tenant scope, several databases at once, developer writes, hosting for a team, and what to do when
something does not connect.

## How a question is answered

On the first connect the server reads the database's structure and builds its knowledge: real foreign keys,
embeddings of every table and column, and a relationship graph. No model is involved and it takes a minute
or two on a large schema; later connects are instant.

When you ask something, `get_query_context(question)` returns only what that question needs: the relevant
tables, the join paths between them, glossary notes and similar past queries. That is about 700 tokens
whatever the schema size, measured at 67% less than the full schema on a 42-table database, 89% less at 120
tables and 97% less at 300. Every result carries its own `context_tokens` figure and `describe_capabilities`
shows the running `context_savings`.

The assistant writes the query, `run_sql` validates and runs it, and the question-to-query pair is
remembered as *unverified*. When you confirm or correct an answer, `mark_query` makes it a verified example
(shown first next time) or removes it; a wrong query is never re-suggested just because it ran. If a question
names nothing in the database, `get_query_context` returns `low_confidence: true` and the assistant asks
instead of guessing.

`query_database` is the one-round-trip alternative: the server's own model writes, reviews and runs the
query. It needs an API key (`GROQ_API_KEY`, or any OpenAI-compatible API through `LLM_BASE_URL` and
`LLM_MODEL`, including a local Ollama).

## The tools

| Tool | What it does |
|---|---|
| `connect_database` | Open a database, or switch to one that is already open |
| `get_query_context` | The tables, joins, notes and examples one question needs |
| `run_sql` | Run a read-only query you wrote (SQL, MongoDB JSON or CQL) |
| `query_database` | Ask in plain language; the server's model writes and runs the query |
| `validate_sql` / `explain_query` | Check a query without running it / show the database's plan |
| `get_schema` | Tables and columns, all or a named few |
| `add_note` / `list_notes` / `delete_note` | Glossary facts the schema cannot express |
| `mark_query` | Record that an answer was right or wrong |
| `set_scope` | Fix the tenant filter for a shared database |
| `import_queries` / `import_query_history` | Learn from SQL the team already has, or from the database's own statistics |
| `execute_write` | One confirmed data change, developer mode only |
| `upload_document` / `search_documents` / `list_documents` | Policies and manuals, searchable next to the data |
| `suggest_chart`, `describe_capabilities`, `disconnect` | |

## Teaching it

**Glossary notes.** Facts the schema cannot express, such as "`ACCOUNT_LINK.Account_Ref` is the card number"
or "amounts are in local currency", go in with `add_note(note, table?, column?)`. They are stored with your
private knowledge of that database, shown only when a question touches that table, and used to pick tables:
ask about "card" and that table comes in.

**What matters.** `import_query_history` reads the database's own execution statistics (which tables the
application and the team query most) and `import_queries` ingests the SQL people already have; `run_sql`
keeps counting. Popular tables rank first and appear as `hot_tables`, so `orders` wins over a stale
`orders_archive` by default.

## One database, many clients

If most tables share a column such as `Tenant_Id` or `Client_Id`, `connect_database` reports it as a
*scope candidate*. Set it once per connection:

```
set_scope(column="Tenant_Id", value="7", label="Acme Retail")
```

From then on the server rewrites every reference to those tables, in joins, subqueries and CTEs alike, to
`(SELECT * FROM t WHERE Tenant_Id = '7')`, so no query shape can reach another client's rows. MongoDB gets
the same filter as a leading `$match`, Cassandra as an added condition. An empty value clears the scope. An
administrator can pin a scope to an identity with `SCOPE_PINS`, in which case it cannot be changed.

## Several databases at once

Connections stay open. Connect to DB1, work, connect to DB2: DB1 is still open, and "switch back to DB1" is
instant with no reconnect and no re-learning. Each database keeps its own knowledge. `disconnect` closes only
the current one; `describe_capabilities` lists what is open. Databases on the same SQL Server instance can be
joined in one query with three-part names (`[OtherDB].dbo.Table`).

## Read-only by default, developer writes on request

`run_sql` and `query_database` are strictly read-only: one SELECT, parsed in the database's own dialect, no
data-changing clause anywhere, every table checked, credential columns never returned, results capped at
`MCP_MAX_ROWS` rows (5,000 by default; the result says `truncated: true` when there was more) and stopped by
the database itself at the time limit.

For real protection give the server a **read-only database login**; the validator is the second line, not
the first. SQL Server:

```sql
CREATE LOGIN queryatlas_ro WITH PASSWORD = '<long-random-password>';
USE [YourDatabase];
CREATE USER queryatlas_ro FOR LOGIN queryatlas_ro;
ALTER ROLE db_datareader ADD MEMBER queryatlas_ro;
```

Developers who need to change data use `execute_write`. Four things must all be true: writes are switched on
in **Settings → Database writes**, the person's role has `sql.write`, the connection was opened with
`allow_writes=true` using their own write-capable credentials, and the person typed YES in the confirmation
form that shows the full statement. `UPDATE` and `DELETE` need a `WHERE`; DDL is a separate switch;
`dry_run=true` runs inside a transaction, reports the affected rows and rolls back. Use it first.

Every query and write is appended to `audit.log` as one JSON line: who, which connection, what, rows,
duration, outcome.

## Hosting it for a team

Run the server once, inside the network that can reach your databases, and let people connect over HTTP:

```
python mcp_server.py --http 0.0.0.0:8000
# or
docker build -t queryatlas . && docker run -d --name queryatlas -p 8000:8000 \
    --env-file .env -v queryatlas-knowledge:/app/.chroma_data queryatlas
```

Each colleague then runs one command, with no Python, drivers or model download on their side:

```
claude mcp add --transport http queryatlas http://<host>:8000/mcp --header "Authorization: Bearer <token>"
```

People sign in through the browser page the server shows (OAuth 2.1 with PKCE) or use a personal token from
`MCP_AUTH_TOKENS`. Everything is per person: connections, learned knowledge, notes, remembered queries and
usage are private to each identity, even on a team instance. Put the server behind HTTPS if it is reachable
beyond your VPN, and remove a person's token line to revoke access.

How database credentials are protected when hosted:

- **Only through the form.** Passwords, API keys and the certificate choice are taken only from the
  connection form, never from tool arguments, so a password typed into the chat is never used.
- **Always encrypted, certificate checked.** PostgreSQL `verify-full`, MySQL and MariaDB TLS with CA check,
  SQL Server `TrustServerCertificate=no`, MongoDB and Cassandra TLS. A self-signed certificate needs *Trust
  this server's certificate* ticked in the form; it is still encrypted and the choice is audited.
- **No internal networks.** A host that resolves to a private, loopback, link-local or reserved address is
  refused unless listed in `MCP_ALLOWED_HOSTS`. SQLite files only inside `MCP_SQLITE_DIR`.
- **Never stored, never shown.** Passwords stay in memory for the open connection only, and connection errors
  are fixed sentences, never the driver's text.

The full threat model is in [SECURITY.md](SECURITY.md).

## If it does not connect

- `ECONNREFUSED`: Claude Code is registered for HTTP but no server is running. Run
  `python mcp_server.py --install` to switch to stdio, or start `python mcp_server.py --http ...` first.
- `Connection closed` at startup: run `python mcp_server.py` in a terminal; the traceback shows the cause,
  usually a missing package (`pip install -r requirements.txt`).
- Settings in `.env` are read from the project folder, wherever Claude Code is started.
- SQL Server needs *ODBC Driver 17 for SQL Server* on the machine running the server (Windows: Microsoft's
  installer; macOS: `brew tap microsoft/mssql-release && brew install msodbcsql17`).
- The first start downloads an embedding model of about 80 MB.

## Settings

`.env.example` documents every setting. The ones most people touch:

| Setting | Purpose |
|---|---|
| `GROQ_API_KEY` or `LLM_API_KEY` + `LLM_BASE_URL` + `LLM_MODEL` | The model for `query_database` and the web app |
| `MCP_DB_PASSWORD`, `MCP_DB_PASSWORD_<DB>` | A database password the local server may use, so it is not typed on every connect |
| `MCP_AUTH_TOKENS` | One token per person for the hosted server |
| `MCP_ALLOWED_HOSTS`, `MCP_SQLITE_DIR` | Internal databases and SQLite files a hosted server may reach |
| `SCOPE_PINS` | Tenant scope fixed per identity |
| `MCP_MAX_ROWS`, `DB_QUERY_TIMEOUT_SECONDS` | The row cap and the time limit |

Redis is optional; without it sessions live in memory for the life of the process. In local mode the
knowledge lives under `.chroma_data/` in the project folder.

## Repository layout

```
mcp_server.py            launcher: python mcp_server.py [--http host:port | --install]
sql_agent/
  config.py              every setting, read from the environment in one place
  state.py               what the process holds per open connection, for both products
  mcp/                   the MCP server: tool catalog, handlers by concern, runtime, transport, installer
  web/                   the web app: routers by concern, one question pipeline, runtime, create_app()
  validation.py, scope.py, sqlreview.py, dialects.py   the gates every query passes, per database kind
  profile/, knowledge.py, retrieval.py, prompts.py     what the model is told, and how little of it
  database.py, mongo.py, cql.py, connections.py        drivers, schemas, address checks, connection tokens
  cache.py, storage.py, auth.py, rbac.py, limits.py    sessions and caches, persistence, sign-in, roles, limits
static/                  the web UI
docs/                    this guide · ARCHITECTURE.md · SECURITY.md · EXAMPLES.md · decisions/
```
