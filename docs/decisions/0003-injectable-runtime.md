# 0003. Each product has an injectable runtime instead of module globals

**Context.** The policy, the connection registry, the hosted flag, the request identity and the
elicitation forms were module globals of a 2,800-line server module. Nothing could be tested without
importing the whole server and patching thirty attributes on it.

**Decision.** `sql_agent/mcp/runtime.py` (`rt`) and `sql_agent/web/runtime.py` (`wrt`) hold each product's
collaborators: the policy, the connection store, the registry of each person's open databases, the limits,
and the ports that reach outside the process (who is calling, the forms, opening a database, the schema
analyser, the retriever factory, the model client factory). Handlers and routers depend on the runtime
object, never on a module global. The transport binds the request-bound ports once; a test substitutes
one attribute.

**Consequences.** Handlers are plain async functions that take the tool arguments and return a result;
dispatch is a dict from tool name to handler. The web auth dependency asks the web runtime for the policy
rather than reaching into the app module. Replacing a port is the way to fake a database, a model or a
form in a test. The runtime is a single instance per process, which matches the one-process deployment.
