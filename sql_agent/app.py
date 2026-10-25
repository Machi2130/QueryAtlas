"""Entry point of the web product at its historical import path (uvicorn sql_agent.app:app and the
local tests). The implementation lives in sql_agent.web."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from . import cache as redis_cache  # noqa: E402,F401  (tests inspect the session cache through this module)
from .auth import get_current_user  # noqa: E402,F401
from .connections import ConnectionRegistry  # noqa: E402,F401
from .web import create_app  # noqa: E402
from .web.deps import active_key as _active_key  # noqa: E402,F401
from .web.deps import knowledge_of as _knowledge_of  # noqa: E402,F401
from .web.deps import llm_client as _llm_client  # noqa: E402,F401
from .web.deps import prepare_nosql as _prepare_nosql  # noqa: E402,F401
from .web.runtime import wrt  # noqa: E402
from .web.sessions import sessions_path as _sessions_path  # noqa: E402,F401

app = create_app()
_store = wrt.store

if __name__ == "__main__":
    import uvicorn

    os.makedirs("static", exist_ok=True)
    uvicorn.run("sql_agent.app:app", host="0.0.0.0", port=5000, reload=False)
