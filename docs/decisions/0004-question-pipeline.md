# 0004. One question pipeline behind the three web question routes

**Context.** `/api/query`, `/api/query/stream` and `/api/hybrid` each contained a full copy of the path a
question takes: gather the session and knowledge, build the prompt, call the model, clean and review the
text, validate it, apply the tenant scope, run it, repair once, remember it. The copies had drifted: one
used the storage history as a fallback, one did not; one learned verified answers again, one did not.

**Decision.** `sql_agent/web/pipeline.py` implements each step once (`prepare`, `completion`, `finish`,
`gate`, `run`, `repair`, `remember`). The routes compose the steps and keep only what differs: the JSON
page, the server-sent token stream, the synthesis over documents.

**Consequences.** A rule about questions (the tenant scope gate, what is remembered, when a model is
required) changes in one place. The model call itself is a synchronous function the routes run in a thread,
so the streaming route can iterate the provider's stream chunk by chunk on worker threads without blocking
the event loop for other people.
