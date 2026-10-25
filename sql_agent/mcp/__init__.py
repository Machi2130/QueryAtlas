"""The MCP product: runtime (injectable collaborators), catalog (tool definitions as data), context
(shared helpers), handlers (one module per concern), server (wiring), transport (HTTP + OAuth) and
install (Claude Code registration and the command line).

MCP hosts launch the server from an arbitrary working directory while the engine uses project-relative
paths (.chroma_data, audit.log, .env), so importing this package moves to the repository root and loads
.env before any module reads its settings.
"""

import os

from dotenv import load_dotenv

from ..config import PROJECT_ROOT

os.chdir(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
