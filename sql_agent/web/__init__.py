"""The web product: a FastAPI app built by create_app(), with routers by concern, request helpers in
deps, one question pipeline shared by the three question routes, and an injectable runtime (wrt)."""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..auth import DEV_MODE
from ..config import settings
from ..errors import configure_logging
from ..limits import BodyLimit, SecurityHeaders
from ..retrieval import _warm_embed_model
from .routers import auth, connections, documents, llm, query, sessions
from .routers import settings as settings_routes
from .runtime import wrt  # noqa: F401  (the app's runtime object, re-exported for the facade)

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """One thread pool for every blocking call (database, provider, files) and the embedding model loaded
    once, in this process, before the first request needs it."""
    configure_logging()
    if DEV_MODE:
        log.warning("DEV_MODE=true: the web app accepts any bearer text as a user id. Use it on your own machine only.")
    pool = ThreadPoolExecutor(max_workers=settings.thread_workers, thread_name_prefix="queryatlas")
    asyncio.get_running_loop().set_default_executor(pool)
    await asyncio.to_thread(_warm_embed_model)
    yield
    pool.shutdown(wait=False)


def create_app() -> FastAPI:
    app = FastAPI(title="QueryAtlas", version="2.0.0", lifespan=lifespan)
    # The UI is served by this app (same origin), so no other site may call the API with a person's
    # credentials unless listed in ALLOWED_ORIGINS (e.g. a separately hosted front end).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(BodyLimit)  # MAX_REQUEST_MB: no request body is held in memory past this size
    app.add_middleware(SecurityHeaders)
    if os.path.isdir("static"):  # web UI assets; absent when only the MCP server is deployed
        app.mount("/static", StaticFiles(directory="static"), name="static")

        @app.get("/")
        async def index():
            return FileResponse("static/index.html")

    for r in (auth, connections, query, sessions, settings_routes, llm, documents):
        app.include_router(r.router)
    return app
