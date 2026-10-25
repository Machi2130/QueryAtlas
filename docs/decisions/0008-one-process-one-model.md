# 0008. No process pool; the embedding model loads once per process

**Context.** The web app created a two-worker process pool at import, used only to warm the sentence
embedding model at startup. On Windows the workers re-imported the application, the model was loaded in a
worker that never served a request, and the main process loaded it again on first use.

**Decision.** The web app runs one process with one thread pool, sized by `THREAD_WORKERS` and installed
as the event loop's default executor in the lifespan. Every blocking call (database, model provider,
files) goes through `asyncio.to_thread`. The embedding model is loaded once, in a thread, before the first
request.

**Consequences.** Embedding and FAISS work hold the GIL while they run, as they did before, since that code
always ran in threads. Scaling beyond one process is a deployment decision (several uvicorn workers behind a
proxy), not something the application does to itself.
