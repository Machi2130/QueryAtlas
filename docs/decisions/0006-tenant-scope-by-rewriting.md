# 0006. The tenant filter is rewritten into every query by the server

**Context.** A database shared by several clients needs every query from a pinned identity limited to that
client's rows, whatever the model or the person wrote. Trusting the generated `WHERE` clause is not an
option; filtering rows after the fact leaks through aggregates.

**Decision.** `sql_agent/scope.py` rewrites every reference to a scoped table into a filtered subquery
(`(SELECT * FROM t WHERE col = 'v')`), after validation and before execution, in both products. MongoDB gets
a leading `$match` (and `$lookup`, `$unionWith` and `$facet` into a scoped collection are refused); CQL gets
`AND col = v` injected. Writes to a scoped table are refused. Examples are remembered without the filter so
one tenant's value never appears in another's prompt.

**Consequences.** The query the person sees ("as written") and the query that runs ("as run") are two
strings everywhere in the code; the second is never remembered. Any new execution path must pass through
the same gate (`prepare_sql` / `prepare_nosql` on the web, `scope_read` in MCP).
