"""The command line behind the root launcher: stdio by default, --http host:port for hosted mode,
--install [--user NAME] to register the server with Claude Code."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from ..config import PROJECT_ROOT
from ..errors import configure_logging
from .runtime import rt
from .server import main
from .transport import serve_http


def install_claude_code() -> None:
    """Register this server with Claude Code (user scope, stdio) using this exact interpreter and
    path. Claude Code then starts and stops it itself: no server to keep running, no ECONNREFUSED."""
    import shutil
    import subprocess

    claude = shutil.which("claude")
    if not claude:
        sys.exit("Claude Code CLI ('claude') is not on PATH. Install Claude Code first, then rerun this.")
    user = sys.argv[sys.argv.index("--user") + 1] if "--user" in sys.argv[:-1] else rt.user_id
    # Replaces an earlier entry (stdio or http) of the same name; "not found" on a first install is expected.
    subprocess.run([claude, "mcp", "remove", "queryatlas", "-s", "user"], capture_output=True)
    subprocess.run([claude, "mcp", "remove", "sql-agent", "-s", "user"], capture_output=True)  # pre-rename registration
    subprocess.run(
        [
            claude,
            "mcp",
            "add",
            "queryatlas",
            "-s",
            "user",
            "-e",
            "PYTHONUTF8=1",
            "-e",
            f"MCP_USER_ID={user}",
            "--",
            sys.executable,
            os.path.join(PROJECT_ROOT, "mcp_server.py"),
        ],
        check=True,
    )
    if not os.path.exists(os.path.join(PROJECT_ROOT, ".env")):
        print("Optional: copy .env.example to .env for database passwords and an LLM key.")
    print(f"Registered queryatlas for Claude Code as user '{user}'. Restart Claude Code, then check: claude mcp list")


def cli() -> None:
    """Entry point behind the root launcher: python mcp_server.py [--http host:port | --install [--user NAME]]."""
    # Newer Claude Code probes with 'server/discover' before initialize; the SDK answers it, but
    # logs a 31-line validation warning that reads like a startup failure in the MCP logs.
    configure_logging()
    logging.getLogger().addFilter(lambda record: "server/discover" not in record.getMessage())
    if "--install" in sys.argv:
        install_claude_code()
    elif "--http" in sys.argv:
        i = sys.argv.index("--http")
        nxt = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
        serve_http(nxt if nxt and not nxt.startswith("-") else "127.0.0.1:8765")
    else:
        asyncio.run(main())
