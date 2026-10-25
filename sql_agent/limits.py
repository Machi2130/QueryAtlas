"""Request-size limit shared by the web app and the hosted MCP server.

One process serves every customer, so no single request may make it hold an unbounded body
in memory. The limit counts the bytes actually received: a request that declares no length
(chunked) or a false one is stopped the same way as one that declares too much.
"""

from __future__ import annotations

from . import cache
from .config import settings

MAX_REQUEST_BYTES = settings.max_request_bytes

SIGN_IN_FAILURES, SIGN_IN_WINDOW = 10, 15 * 60  # per username, web and MCP sign-in alike
SIGN_UPS_PER_HOUR = settings.sign_ups_per_hour  # per client address
MCP_CALLS_PER_MINUTE = settings.mcp_calls_per_minute  # per identity, hosted only


def sign_in_blocked(username: str) -> bool:
    return cache.recent_count(username, "signin-fail", SIGN_IN_WINDOW) >= SIGN_IN_FAILURES


def record_sign_in(username: str, ok: bool) -> None:
    if ok:
        cache.clear_count(username, "signin-fail")
    else:
        cache.check_rate_limit(username, SIGN_IN_FAILURES, SIGN_IN_WINDOW, kind="signin-fail")


def sign_up_blocked(client: str) -> bool:
    """Accounts created from this address in the last hour (refused attempts create nothing, so they don't count)."""
    return cache.recent_count(client or "unknown", "signup", 3600) >= SIGN_UPS_PER_HOUR


def record_sign_up(client: str) -> None:
    cache.check_rate_limit(client or "unknown", SIGN_UPS_PER_HOUR, 3600, kind="signup")


def mcp_call_allowed(identity: str) -> tuple[bool, int]:
    allowed, _, retry_after = cache.check_rate_limit(identity, MCP_CALLS_PER_MINUTE, 60, kind="mcp")
    return allowed, retry_after


_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),  # no page (sign-in forms included) inside another site
    (b"content-security-policy", b"frame-ancestors 'none'"),
    (b"referrer-policy", b"no-referrer"),
    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),  # browsers apply it over HTTPS only
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]


class SecurityHeaders:
    """ASGI middleware: standard browser protections on every HTTP response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def with_headers(message):
            if message["type"] == "http.response.start":
                present = {k.lower() for k, _ in message.get("headers", [])}
                message["headers"] = list(message.get("headers", [])) + [
                    h for h in _SECURITY_HEADERS if h[0] not in present
                ]
            await send(message)

        await self.app(scope, receive, with_headers)


class _TooLarge(Exception):
    pass


class BodyLimit:
    """ASGI middleware: 413 for any HTTP request body over max_bytes."""

    def __init__(self, app, max_bytes: int = MAX_REQUEST_BYTES):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await self._refuse(send)
            return
        received, started = 0, False

        async def counted_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge()
            return message

        async def tracked_send(message):
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counted_receive, tracked_send)
        except _TooLarge:
            if not started:  # once a response has begun it can only be cut off, which the server does
                await self._refuse(send)

    async def _refuse(self, send):
        body = b'{"error": "request too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})
