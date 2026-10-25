"""
storage.py — Supabase persistence layer.

Fixes:
  - No query logging / observability   → query_logs table
  - No conversation history            → conversation_history table
  - Shared memory file (agent_memory)  → per-user user_memory table
  - Cache dies on restart              → query_logs acts as cold cache fallback

All write operations are fire-and-forget (errors are swallowed) so a
Supabase outage never fails a user query.
"""

import logging

from supabase import Client, create_client

from .config import settings

log = logging.getLogger(__name__)

SUPABASE_URL = settings.supabase_url
SUPABASE_KEY = settings.supabase_key  # service-role key: never exposed client-side

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in your environment.")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def _is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


# ═══════════════════════════════════════════════════════════════
#  QUERY LOGS  — full audit trail + latency observability
# ═══════════════════════════════════════════════════════════════


def log_query(
    *,
    user_id: str,
    question: str,
    generated_query: str,
    db_flavor: str,
    tables_used: list[str],
    row_count: int,
    latency_ms: int,
    success: bool,
    error: str | None = None,
    from_cache: bool = False,
    retried: bool = False,
) -> None:
    if not _is_configured():
        return
    try:
        get_client().table("query_logs").insert(
            {
                "user_id": user_id,
                "question": question,
                "generated_query": generated_query,
                "db_flavor": db_flavor,
                "tables_used": tables_used,
                "row_count": row_count,
                "latency_ms": latency_ms,
                "success": success,
                "error": error,
                "from_cache": from_cache,
                "retried": retried,
            }
        ).execute()
    except Exception as e:  # a failed log must never break the request, but say why
        log.warning(f"query log not saved: {e}")


# ═══════════════════════════════════════════════════════════════
#  CONVERSATION HISTORY  — durable record; Redis is the hot path
# ═══════════════════════════════════════════════════════════════


def save_turn(
    *,
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    query: str | None = None,
) -> None:
    if not _is_configured():
        return
    try:
        get_client().table("conversation_history").insert(
            {
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "query": query,
            }
        ).execute()
    except Exception as e:
        log.warning(f"Supabase write failed: {e}")


def delete_history(user_id: str, session_id: str) -> None:
    if not _is_configured():
        return
    try:
        (
            get_client()
            .table("conversation_history")
            .delete()
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .execute()
        )
    except Exception as e:
        log.warning(f"Supabase write failed: {e}")


def get_history(user_id: str, session_id: str, limit: int = 20) -> list[dict]:
    if not _is_configured():
        return []
    try:
        res = (
            get_client()
            .table("conversation_history")
            .select("role, content, query, created_at")
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        log.warning(f"history unavailable: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
#  USER MEMORY  — per-user learned column mappings
# ═══════════════════════════════════════════════════════════════


def load_memory(user_id: str) -> dict:
    if not _is_configured():
        return {}
    try:
        res = get_client().table("user_memory").select("memory").eq("user_id", user_id).maybe_single().execute()
        return (res.data or {}).get("memory") or {}
    except Exception as e:
        log.warning(f"memory unavailable: {e}")
        return {}


def save_memory(user_id: str, memory: dict) -> None:
    if not _is_configured():
        return
    try:
        get_client().table("user_memory").upsert(
            {
                "user_id": user_id,
                "memory": memory,
            }
        ).execute()
    except Exception as e:
        log.warning(f"Supabase write failed: {e}")
