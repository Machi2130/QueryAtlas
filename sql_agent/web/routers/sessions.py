"""Conversations for the sidebar, and the model-visible history."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ... import cache as redis_cache
from ... import rbac, storage
from ...auth import get_current_user
from ..deps import bound_label, closed_message, for_viewer, require
from ..runtime import wrt
from ..sessions import delete_session, load_sessions

router = APIRouter()


@router.delete("/api/sessions/{session_id}")
async def delete_session_turns(session_id: str, user_id: str = Depends(get_current_user)):
    if not await asyncio.to_thread(delete_session, user_id, session_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    redis_cache.clear_conversation(user_id, session_id)
    return {"success": True}


@router.get("/api/history")
async def get_history(
    session_id: str = "default",
    limit: int = 20,
    user_id: str = Depends(get_current_user),
):
    history = storage.get_history(user_id, session_id, limit=min(limit, 100))
    return {"history": history, "count": len(history)}


@router.get("/api/sessions")
async def list_sessions(user_id: str = Depends(get_current_user)):
    require(user_id, rbac.ASK)
    """The user's conversations, newest first, for the sidebar."""
    sessions = await asyncio.to_thread(load_sessions, user_id)
    items = [
        {
            "id": sid,
            "title": s.get("title", ""),
            "updated": s.get("updated", ""),
            "turns": len(s.get("messages", [])),
            "connection": bound_label(user_id, s.get("connection")),
        }
        for sid, s in sessions.items()
    ]
    return {"sessions": sorted(items, key=lambda s: s["updated"], reverse=True)}


@router.get("/api/sessions/{session_id}")
async def get_session_turns(session_id: str, user_id: str = Depends(get_current_user)):
    s = (await asyncio.to_thread(load_sessions, user_id)).get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    bound = s.get("connection")
    is_open = bool(bound and wrt.connections.get(user_id, bound["id"]))
    return for_viewer(
        user_id,
        {
            "id": session_id,
            "title": s.get("title", ""),
            "messages": s.get("messages", []),
            "connection": {**bound, "label": bound_label(user_id, bound)} if bound else None,
            "connection_open": is_open,
            "closed_message": "" if is_open or not bound else closed_message(bound),
        },
    )


@router.delete("/api/history")
async def clear_history(
    background_tasks: BackgroundTasks,
    session_id: str = "default",
    user_id: str = Depends(get_current_user),
):
    redis_cache.clear_conversation(user_id, session_id)
    # Also delete from Supabase so history doesn't reappear after Redis flush
    background_tasks.add_task(storage.delete_history, user_id, session_id)
    return {"success": True}
