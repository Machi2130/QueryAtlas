"""The settings file (roles, users, limits) and each person's self-service MCP token."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ... import rbac
from ...auth import get_current_user
from ...rbac import SettingsError
from ..deps import require
from ..runtime import wrt

router = APIRouter()


def public_settings() -> dict:
    """Settings safe to return to the browser (token prefixes/dates only, never hashes or passwords)."""
    doc = wrt.policy.settings()
    doc.pop("accounts", None)
    doc.pop("web_sessions", None)
    doc["mcp_tokens"] = {
        who: {"prefix": v.get("prefix", ""), "created": v.get("created", "")}
        for who, v in (doc.get("mcp_tokens") or {}).items()
    }
    return doc


@router.get("/api/settings")
async def get_settings(user_id: str = Depends(get_current_user)):
    require(user_id, rbac.SETTINGS)
    return {"settings": public_settings(), "permissions": rbac.ALL_PERMISSIONS}


@router.put("/api/settings")
async def put_settings(doc: dict, user_id: str = Depends(get_current_user)):
    """Replace the settings (validated: unknown permissions or roles, or removing every
    settings.manage, are refused). Takes effect on the next request; no restart."""
    require(user_id, rbac.SETTINGS)
    cur = wrt.policy.settings()
    current_tokens = cur.get("mcp_tokens") or {}
    incoming_tokens = doc.get("mcp_tokens")
    if isinstance(incoming_tokens, dict):
        doc["mcp_tokens"] = {k: current_tokens[k] for k in incoming_tokens if k in current_tokens}
    else:
        doc["mcp_tokens"] = current_tokens
    doc["accounts"] = cur.get("accounts") or {}
    doc["web_sessions"] = cur.get("web_sessions") or {}
    try:
        wrt.policy.save(doc)
    except SettingsError as e:
        raise HTTPException(status_code=400, detail={"message": str(e)}) from e
    return {"success": True, "settings": public_settings()}


# ── Self-service MCP Bearer token per user (SHA-256 hashed in rbac.json; raw token shown once) ──


@router.get("/api/mcp-token")
async def get_mcp_token(user_id: str = Depends(get_current_user)):
    require(user_id, rbac.MCP)
    return {"user": user_id, **wrt.policy.mcp_token_info(user_id)}


@router.post("/api/mcp-token")
async def create_mcp_token(user_id: str = Depends(get_current_user)):
    """Issue or rotate this user's personal MCP Bearer token. Returned in full only once."""
    require(user_id, rbac.MCP)
    raw = wrt.policy.issue_mcp_token(user_id)
    return {"user": user_id, "token": raw, **wrt.policy.mcp_token_info(user_id)}


@router.delete("/api/mcp-token")
async def delete_mcp_token(user_id: str = Depends(get_current_user)):
    require(user_id, rbac.MCP)
    wrt.policy.revoke_mcp_token(user_id)
    return {"user": user_id, **wrt.policy.mcp_token_info(user_id)}
