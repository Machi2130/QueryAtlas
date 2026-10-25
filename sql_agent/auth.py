"""Who is calling the web app: account session tokens, Auth0 JWTs when configured, and the DEV_MODE
fallback. Used as a FastAPI dependency: `user_id: str = Depends(get_current_user)`."""

import logging

import httpx
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi import status as http_status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError, JWTError

from .config import settings

load_dotenv()  # the flags below are read once, at import

_bearer_scheme = HTTPBearer()

log = logging.getLogger(__name__)

AUTH0_DOMAIN = settings.auth0_domain
AUTH0_AUDIENCE = settings.auth0_audience
DEV_MODE = settings.dev_mode

# In-process JWKS cache — refreshed on key-rotation mismatch.
_jwks_cache: dict | None = None


def _get_jwks() -> dict:
    global _jwks_cache
    if _jwks_cache is None:
        resp = httpx.get(
            f"https://{AUTH0_DOMAIN}/.well-known/jwks.json",
            timeout=10,
        )
        resp.raise_for_status()
        _jwks_cache = resp.json()
    return _jwks_cache


def _find_rsa_key(token: str) -> dict:
    """Match the JWT kid header against the JWKS."""
    jwks = _get_jwks()
    header = jwt.get_unverified_header(token)
    for key in jwks.get("keys", []):
        if key.get("kid") == header.get("kid"):
            return {k: key[k] for k in ("kty", "kid", "use", "n", "e") if k in key}
    # Key not found — JWKS may have rotated; bust the cache and retry once.
    global _jwks_cache
    _jwks_cache = None
    jwks = _get_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == header.get("kid"):
            return {k: key[k] for k in ("kty", "kid", "use", "n", "e") if k in key}
    raise ValueError("No matching public key found in JWKS — check AUTH0_DOMAIN.")


def _active_policy():
    """The web runtime's policy (the one a test may have substituted), imported lazily because the web
    package imports this module."""
    from .web.runtime import wrt

    return wrt.policy


def verify_token(token: str) -> str:
    """
    Verify a Bearer token and return the user_id:
      1. Built-in account session token (`ses_...`) or MCP token (`mcp_...`)
      2. Auth0 RS256 JWT (when configured)
      3. DEV_MODE fallback for non-account test identities
    """
    policy = _active_policy()
    if token.startswith("ses_"):
        who = policy.user_for_session_token(token)
        if not who:
            raise ValueError("Session expired or invalid. Please sign in again.")
        return who
    if token.startswith("mcp_"):
        # An MCP token is for the MCP server only (it checks it itself). Accepting it here would make a
        # 30-day AI-client token a web session that can mint fresh tokens and never effectively expires.
        raise ValueError("MCP tokens cannot be used for the web app. Sign in with your account.")
    # Never allow raw username impersonation of a password-protected account
    if policy.has_account(token):
        raise ValueError("Password sign-in is required for this account.")

    if DEV_MODE:
        return token

    if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
        raise RuntimeError("Please sign in with your account and password.")

    rsa_key = _find_rsa_key(token)
    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
    except ExpiredSignatureError as exc:
        raise ValueError("Token has expired.") from exc
    except JWTClaimsError as e:
        raise ValueError(f"Token claims invalid: {e}") from e
    except JWTError as e:
        raise ValueError(f"Token invalid: {e}") from e

    return payload["sub"]  # e.g. "auth0|abc123"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """
    FastAPI Depends() pattern — injects user_id into any route as a parameter.

    Flask pattern:   @require_auth sets g.user_id (thread-local, not testable)
    FastAPI pattern: Depends(get_current_user) passes user_id as a typed argument.
                     Testable by overriding: app.dependency_overrides[get_current_user] = lambda: "test-user"
    """
    try:
        return verify_token(credentials.credentials)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail=str(e)) from e
    except Exception as e:
        log.warning(f"token verification failed: {type(e).__name__}")
        raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail=f"Auth error: {e}") from e
