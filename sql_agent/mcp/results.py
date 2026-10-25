"""The one shape every tool returns: {"ok": true, "data": ...} or {"ok": false, "error": {...}}."""

from __future__ import annotations

import json
from typing import Any

from mcp import types

from ..errors import ErrCode


def ok(data: Any) -> list[types.TextContent]:
    payload = {"ok": True, "data": data}
    return [types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))]


def err(msg: str, code: str = ErrCode.UNKNOWN, details: dict | None = None) -> list[types.TextContent]:
    payload = {"ok": False, "error": {"code": code, "message": msg, **(details or {})}}
    return [types.TextContent(type="text", text=json.dumps(payload, default=str))]
