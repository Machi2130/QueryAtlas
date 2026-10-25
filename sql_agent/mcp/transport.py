"""Hosted mode: Streamable HTTP at /mcp behind bearer tokens, with OAuth 2.1 PKCE discovery, dynamic
client registration, a browser sign-in page and the token exchange."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import logging
import secrets
import time
import urllib.parse

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from .. import rbac
from ..config import settings
from ..limits import BodyLimit, SecurityHeaders, record_sign_in, sign_in_blocked
from .context import audit
from .runtime import rt
from .server import server

log = logging.getLogger(__name__)


_codes: dict[str, dict] = {}  # one-time auth_code -> {user, redirect_uri, code_challenge, method, exp}
# Exact redirect URIs allowed besides loopback, e.g. https://claude.ai/api/mcp/auth_callback for claude.ai connectors.
_EXTRA_REDIRECT_URIS = settings.oauth_redirect_uris


def redirect_allowed(uri: str) -> bool:
    """Registration is open to anyone, so a registered URI proves nothing: the code may only go to
    a loopback address (Claude Code, Cursor and Claude Desktop listen on localhost) or an
    administrator-listed URI. Anything else would hand a signed-in user's code to a third party."""
    parts = urllib.parse.urlsplit(uri)
    return uri in _EXTRA_REDIRECT_URIS or (
        parts.scheme in ("http", "https") and parts.hostname in ("localhost", "127.0.0.1", "::1")
    )


def render_login_page(params: dict[str, str], web_signup_url: str, error: str = "") -> str:
    import html as _html

    err_html = f'<div class="err">{_html.escape(error)}</div>' if error else ""
    hidden = "\n".join(
        f'<input type="hidden" name="{_html.escape(k)}" value="{_html.escape(str(v or ""))}"/>'
        for k, v in params.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Authorize MCP · NL·DB Chat</title>
<style>
  body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0d0e14;color:#e5e7eb;font-family:system-ui,-apple-system,sans-serif}}
  .card{{background:#141620;border:1px solid #26293a;border-radius:12px;padding:28px 26px;width:min(380px,92vw)}}
  h1{{font-size:18px;margin:0 0 6px;color:#f3f4f6}}
  p{{font-size:13px;color:#9ca3af;margin:0 0 18px;line-height:1.45}}
  label{{display:block;font-size:12px;color:#9ca3af;margin:12px 0 4px}}
  input{{width:100%;box-sizing:border-box;padding:9px 11px;background:#1c1f2e;border:1px solid #2e3247;border-radius:8px;color:#f3f4f6;font-size:13.5px;outline:none}}
  input:focus{{border-color:#7c5cfc}}
  button{{width:100%;margin-top:18px;padding:10px;background:#7c5cfc;border:none;border-radius:8px;color:#fff;font-weight:600;font-size:13.5px;cursor:pointer}}
  button:hover{{background:#6b4ce6}}
  .err{{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.3);color:#f87171;padding:8px 10px;border-radius:8px;font-size:12.5px;margin-bottom:12px}}
  .foot{{margin-top:16px;font-size:12px;color:#9ca3af;text-align:center}}
  .foot a{{color:#a78bfa;text-decoration:none}}
</style></head>
<body><div class="card">
  <h1>Connect MCP to NL·DB Chat</h1>
  <p>Sign in with your account to issue a personal MCP token to your AI client.</p>
  {err_html}
  <form method="POST" action="/oauth/authorize">
    {hidden}
    <label>Username or Email</label>
    <input name="username" required autocomplete="username" autofocus placeholder="e.g. alice"/>
    <label>Password</label>
    <input name="password" type="password" required autocomplete="current-password" placeholder="Your account password"/>
    <button type="submit">Sign In &amp; Authorize MCP</button>
  </form>
  <div class="foot">Need an account first? <a href="{_html.escape(web_signup_url)}" target="_blank">Create an account on Web UI</a></div>
</div></body></html>"""


OAUTH_PARAMS = ("client_id", "redirect_uri", "state", "code_challenge", "code_challenge_method", "response_type")
CODE_SECONDS = 300


def _origin(scope) -> str:
    scheme = scope.get("scheme") or "http"
    headers = dict(scope.get("headers") or {})
    return f"{scheme}://{headers.get(b'host', b'localhost:8765').decode('latin-1')}"


def _web_ui_url(scope) -> str:
    scheme = scope.get("scheme") or "http"
    headers = dict(scope.get("headers") or {})
    host_only = headers.get(b"host", b"localhost").decode("latin-1").split(":")[0]
    return settings.web_ui_url(f"{scheme}://{host_only}:8000")


def _resource_metadata(base: str) -> JSONResponse:
    """OAuth 2.0 Protected Resource Metadata (RFC 9728)."""
    return JSONResponse(
        {"resource": f"{base}/mcp", "authorization_servers": [base], "bearer_methods_supported": ["header"]}
    )


def _server_metadata(base: str) -> JSONResponse:
    """OAuth 2.1 Authorization Server Metadata (RFC 8414)."""
    return JSONResponse(
        {
            "issuer": base,
            "authorization_endpoint": f"{base}/oauth/authorize",
            "token_endpoint": f"{base}/oauth/token",
            "registration_endpoint": f"{base}/oauth/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
            "code_challenge_methods_supported": ["S256"],
        }
    )


async def _register(req: Request) -> JSONResponse:
    """Dynamic Client Registration (RFC 7591). Open to anyone, so a registration proves nothing: the
    redirect allow-list (redirect_allowed) is what protects a signed-in person's code."""
    body = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    return JSONResponse(
        {
            "client_id": "mcp-client-" + secrets.token_urlsafe(12),
            "client_id_issued_at": int(time.time()),
            "client_name": body.get("client_name", "MCP Client"),
            "redirect_uris": body.get("redirect_uris", []),
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
        status_code=201,
    )


async def _form(req: Request) -> dict[str, str]:
    raw = (await req.body()).decode("utf-8", errors="replace")
    return {k: v[0] for k, v in urllib.parse.parse_qs(raw, keep_blank_values=True).items()}


def _sign_in_problem(username: str, password: str, params: dict[str, str]) -> tuple[str, int] | None:
    """Why the browser sign-in may not issue a code: (message, status), or None when everything checks out."""
    if not rt.policy.has_account(username):
        return "Account not found. You must create an account first before authorizing MCP.", 401
    if sign_in_blocked(username):  # shared with the web sign-in, across server processes
        return "Too many failed sign-ins. Try again in 15 minutes.", 429
    ok = rt.policy.verify_account(username, password)
    record_sign_in(username, ok)
    if not ok:
        audit("sign_in", username, "", ok=False, error="wrong password (MCP authorization)")
        return "Incorrect username or password.", 401
    if not rt.policy.can(username, rbac.MCP):
        return f"Your account ({rt.policy.role_of(username)}) does not have 'mcp.use' permission.", 403
    redirect_uri = params.get("redirect_uri") or ""
    if not redirect_allowed(redirect_uri):
        return (
            f"This client's redirect address ({redirect_uri or 'none'}) is not allowed. Use Claude Code / "
            "Cursor / Claude Desktop, or ask the administrator to add it to OAUTH_REDIRECT_URIS.",
            400,
        )
    # PKCE S256 is required: it binds the code to the client that started the sign-in.
    if not params.get("code_challenge") or (params.get("code_challenge_method") or "S256").upper() != "S256":
        return "The client must use PKCE with S256.", 400
    return None


def _prune_codes() -> None:
    now = time.time()
    for code in [c for c, e in _codes.items() if now > e["exp"]]:
        _codes.pop(code, None)


async def _authorize(req: Request, scope, method: str):
    """The browser sign-in and consent screen: GET shows it, POST checks the account and redirects back
    to the client with a one-time code bound to its PKCE challenge."""
    if method == "GET":
        params = {k: req.query_params.get(k, "") for k in OAUTH_PARAMS}
        return HTMLResponse(render_login_page(params, _web_ui_url(scope)))
    form = await _form(req)
    params = {k: form.get(k, "") for k in OAUTH_PARAMS}
    problem = _sign_in_problem((form.get("username") or "").strip(), form.get("password") or "", params)
    if problem:
        message, status = problem
        return HTMLResponse(render_login_page(params, _web_ui_url(scope), message), status_code=status)
    _prune_codes()
    code = secrets.token_urlsafe(32)
    _codes[code] = {
        "user": (form.get("username") or "").strip(),
        "redirect_uri": params["redirect_uri"],
        "code_challenge": params["code_challenge"],
        "exp": time.time() + CODE_SECONDS,
    }
    sep = "&" if "?" in params["redirect_uri"] else "?"
    target = f"{params['redirect_uri']}{sep}code={urllib.parse.quote(code)}"
    if params.get("state"):
        target += f"&state={urllib.parse.quote(params['state'])}"
    return RedirectResponse(target, status_code=302)


async def _token_exchange(req: Request) -> JSONResponse:
    """The code comes back with the PKCE verifier; a matching pair issues a personal mcp_... token."""
    if "application/json" in req.headers.get("content-type", ""):
        data = await req.json()
    else:
        data = await _form(req)
    entry = _codes.pop(data.get("code", ""), None)
    if not entry or time.time() > entry["exp"]:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Invalid or expired code."}, status_code=400
        )
    digest = hashlib.sha256(data.get("code_verifier", "").encode("utf-8")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    if not hmac.compare_digest(computed, entry["code_challenge"]):
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "PKCE verification failed."}, status_code=400
        )
    return JSONResponse(
        {
            "access_token": rt.policy.issue_mcp_token(entry["user"]),
            "token_type": "Bearer",
            "expires_in": rbac.MCP_TOKEN_SECONDS,
        }
    )


def _bearer_accepted(scope, env_tokens: list[str]) -> bool:
    auth = dict(scope.get("headers") or []).get(b"authorization", b"").decode()
    bearer = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    valid_env = any(hmac.compare_digest(auth, f"Bearer {t}") for t in env_tokens)
    return valid_env or bool(bearer and rt.policy.user_for_mcp_token(bearer))


def _unauthorized(base: str) -> JSONResponse:
    return JSONResponse(
        {"error": "unauthorized", "auth_url": f"{base}/oauth/authorize"},
        status_code=401,
        headers={"WWW-Authenticate": f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource"'},
    )


async def _route(path: str, method: str, scope, receive, env_tokens: list[str]):
    """The response for everything but the MCP endpoint itself; None means the bearer is accepted and the
    request goes to the MCP session manager."""
    base = _origin(scope)
    if path in ("/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"):
        return _resource_metadata(base)
    if path in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
        return _server_metadata(base)
    if path == "/oauth/register" and method == "POST":
        return await _register(Request(scope, receive))
    if path == "/oauth/authorize" and method in ("GET", "POST"):
        return await _authorize(Request(scope, receive), scope, method)
    if path == "/oauth/token" and method == "POST":
        return await _token_exchange(Request(scope, receive))
    if path != "/mcp":
        return JSONResponse({"error": "not found"}, status_code=404)
    return None if _bearer_accepted(scope, env_tokens) else _unauthorized(base)


def build_http_mcp_app(manager=None):
    """The ASGI app serving Streamable HTTP at /mcp behind bearer tokens, with OAuth 2.1 PKCE discovery,
    registration, sign-in and token exchange in front of it."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.routing import Mount

    env_tokens = [t for t in rt.token_users if len(t) >= 16]
    if manager is None:
        manager = StreamableHTTPSessionManager(app=server)

    async def guarded(scope, receive, send):
        if scope["type"] != "http":
            return
        path = scope["path"].rstrip("/") or "/"
        method = scope.get("method", "GET").upper()
        response = await _route(path, method, scope, receive, env_tokens)
        if response is None:
            await manager.handle_request(scope, receive, send)
        else:
            await response(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        if hasattr(manager, "run"):
            async with manager.run():
                yield
        else:
            yield

    # MAX_REQUEST_MB: OAuth forms, MCP JSON-RPC (including document uploads) alike.
    return Starlette(
        routes=[Mount("/", app=guarded)],
        lifespan=lifespan,
        middleware=[Middleware(BodyLimit), Middleware(SecurityHeaders)],
    )


def serve_http(bind: str) -> None:
    """Hosted mode: Streamable HTTP at /mcp + OAuth 2.1 browser authorization flow."""
    import uvicorn

    rt.hosted = True
    host, _, port = bind.rpartition(":")
    log.info(
        "Hosted: database connections must be encrypted; hosts on private/internal networks are "
        "refused unless listed in MCP_ALLOWED_HOSTS; SQLite only inside MCP_SQLITE_DIR; passwords only "
        "through the connection form."
    )
    app = build_http_mcp_app()
    log.info(f"HTTP mode: http://{host or '0.0.0.0'}:{port or 8765}/mcp")
    uvicorn.run(app, host=host or "0.0.0.0", port=int(port or 8765), log_level="info")
