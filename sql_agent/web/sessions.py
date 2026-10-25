"""Conversations: what a person asked and was answered, to reopen from the sidebar.

One JSON file per person next to the rest of the server-side knowledge (no customer data in browser
storage, works without Supabase). Saved answers re-display without a query. Every function here does
file I/O: callers run them in a thread.
"""

from __future__ import annotations

import datetime
import json
import os
import threading

from ..connections import safe_owner
from .runtime import wrt

SESSION_ROWS = 50  # rows kept per saved answer (the full result can be asked again)
MAX_SESSIONS = 50
MAX_SESSION_TURNS = 100
_lock = threading.Lock()  # ponytail: one lock for all users' files; per-user locks if it ever contends


def sessions_path(user_id: str) -> str:
    return os.path.join(wrt.sessions_dir, safe_owner(user_id) + ".json")


def load_sessions(user_id: str) -> dict:
    try:
        with open(sessions_path(user_id), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _write(user_id: str, sessions: dict) -> None:
    path = sessions_path(user_id)
    os.makedirs(wrt.sessions_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sessions, f, default=str)
    os.replace(tmp, path)  # never leave a half-written file behind


def save_session_turn(user_id: str, session_id: str, turn: dict, connection: dict | None = None) -> None:
    """Append one answered question; the first question gives the conversation its title and its
    database (opening the conversation later switches back to it)."""
    with _lock:
        sessions = load_sessions(user_id)
        s = sessions.setdefault(session_id, {"title": turn["question"][:60], "messages": []})
        if connection and not s.get("connection"):
            s["connection"] = connection
        s["messages"] = (s["messages"] + [turn])[-MAX_SESSION_TURNS:]
        s["updated"] = datetime.datetime.now().isoformat(timespec="milliseconds")  # orders two answers in one second
        newest = sorted(sessions.items(), key=lambda kv: kv[1].get("updated", ""), reverse=True)[:MAX_SESSIONS]
        _write(user_id, dict(newest))


def delete_session(user_id: str, session_id: str) -> bool:
    with _lock:
        sessions = load_sessions(user_id)
        if session_id not in sessions:
            return False
        del sessions[session_id]
        _write(user_id, sessions)
        return True


def conversation_binding(user_id: str, session_id: str) -> dict | None:
    """The database a conversation is bound to ({"id", "db_type"}), if any."""
    return (load_sessions(user_id).get(session_id) or {}).get("connection")
