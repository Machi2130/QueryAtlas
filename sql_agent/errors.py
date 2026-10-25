"""Error codes every MCP tool result and web error carries, so clients can branch on them."""

from __future__ import annotations

import logging
import sys


class ErrCode:
    NOT_CONNECTED = "NOT_CONNECTED"
    QUERY_FAILED = "QUERY_FAILED"
    AUTH_FAILED = "AUTH_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    UNSUPPORTED_DB = "UNSUPPORTED_DB"
    INVALID_SQL = "INVALID_SQL"
    TIMEOUT = "TIMEOUT"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    RATE_LIMITED = "RATE_LIMITED"
    UNKNOWN = "UNKNOWN"


def configure_logging(level: int = logging.INFO) -> None:
    """Root logger to stderr. The MCP stdio transport owns stdout, so nothing may log there."""
    logging.basicConfig(level=level, stream=sys.stderr, format="[queryatlas] %(levelname)s %(name)s: %(message)s")
