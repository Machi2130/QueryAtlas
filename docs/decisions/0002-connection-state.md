# 0002. One state object per open connection, shared by both products

**Context.** The MCP server and the web app each kept fifteen parallel dicts keyed by connection (engine,
MongoDB client, Cassandra session, retriever, knowledge engine, foreign keys, profile, scope, keys, usage
and so on), with the close logic written twice and a database handle living in one of three slots.

**Decision.** `sql_agent/state.py` defines `ConnectionState`, one dataclass per open connection with a
single `handle` field, and `ConnectionStore`, which owns the entries and is the only path that closes a
handle. Both products hold one store on their runtime object.

**Consequences.** Per-connection memory lives in one place; a new per-connection fact is a field on the
dataclass, never a new dict. `connected` is `handle is not None`, because a pymongo `Database` refuses
truth testing. Replacing a handle closes the old one, so a reconnect never strands a logged-in connection.
The store is mutated by connect, refresh and discard; worker threads receive a state or a handle and read
it, never add or remove entries.
