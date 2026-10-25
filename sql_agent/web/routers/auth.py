"""Accounts and sign-in: register, login, logout, who am I, and the page's configuration."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from ...auth import get_current_user
from ...config import settings
from ...limits import record_sign_in, record_sign_up, sign_in_blocked, sign_up_blocked
from ..runtime import wrt
from ..schemas import AuthRequest

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/config")
async def get_config():
    return {
        "auth0_domain": settings.auth0_domain,
        "auth0_client_id": settings.auth0_client_id,
        "auth0_audience": settings.auth0_audience,
        "dev_mode": settings.dev_mode,
    }


@router.post("/api/auth/register")
async def auth_register(body: AuthRequest, request: Request):
    """Create a new password-protected user account (scrypt + 32-byte salt) and sign in."""
    # Per client address. Behind a reverse proxy, run uvicorn with --proxy-headers so this is the
    # visitor's address, not the proxy's.
    client = request.client.host if request.client else ""
    if sign_up_blocked(client):
        raise HTTPException(status_code=429, detail={"message": "Too many new accounts from here. Try again later."})
    try:
        role = wrt.policy.register_account(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"message": str(e)}) from e
    record_sign_up(client)
    user = body.username.strip()
    token = wrt.policy.issue_session_token(user)
    return {"user": user, "role": role, "token": token}


@router.post("/api/auth/login")
async def auth_login(body: AuthRequest):
    """Verify account username & password (scrypt) and issue a web session token (`ses_...`)."""
    user = (body.username or "").strip()
    if not wrt.policy.has_account(user):
        raise HTTPException(status_code=401, detail={"message": "Account not found. Please create an account first."})
    if sign_in_blocked(user):  # shared with the MCP sign-in page, across server processes
        raise HTTPException(status_code=429, detail={"message": "Too many failed sign-ins. Try again in 15 minutes."})
    ok = wrt.policy.verify_account(user, body.password)
    record_sign_in(user, ok)
    if not ok:
        log.warning(f"failed web sign-in for {user!r}")
        raise HTTPException(status_code=401, detail={"message": "Incorrect username or password."})
    token = wrt.policy.issue_session_token(user)
    return {"user": user, "role": wrt.policy.role_of(user), "token": token}


@router.post("/api/auth/logout")
async def auth_logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        wrt.policy.revoke_session_token(authorization.removeprefix("Bearer ").strip())
    return {"ok": True}


@router.get("/api/me")
async def me(user_id: str = Depends(get_current_user)):
    """Role and permissions, so the page shows only what this person may use."""
    return {
        "user": user_id,
        "role": wrt.policy.role_of(user_id),
        "permissions": sorted(wrt.policy.permissions(user_id)),
    }
