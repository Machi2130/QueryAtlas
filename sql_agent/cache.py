"""Sessions, the schema cache, the query cache, rate windows and recent conversation turns.

One `CacheBackend` protocol, two implementations: Redis when it is configured and reachable (so limits
and sessions hold across server processes), else a bounded in-process store. The choice is made once;
the module functions below are what the rest of the code calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Protocol

from .config import settings

log = logging.getLogger(__name__)

try:
    import redis as redis_lib
except ImportError:  # optional dependency: the in-memory backend serves without it
    redis_lib = None

SESSION_TTL = settings.session_ttl_seconds
CACHE_TTL = settings.cache_ttl_seconds
RATE_WINDOW = 60
RATE_LIMIT = settings.rate_limit_per_minute
MAX_HOT_TURNS = 12


class CacheBackend(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl: int) -> None: ...
    def expire(self, key: str, ttl: int) -> None: ...
    def delete(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def count_in_window(self, key: str, window: int) -> tuple[int, int]:
        """Count one event in a fixed window: (count so far, seconds left in the window)."""
        ...

    def count(self, key: str, window: int) -> int: ...
    def push(self, key: str, value: str, keep: int, ttl: int) -> None: ...
    def tail(self, key: str, n: int) -> list[str]: ...
    def ping(self) -> bool: ...


class RedisBackend:
    def __init__(self, client: Any) -> None:
        self._r = client

    def get(self, key: str) -> str | None:
        return self._r.get(key)

    def set(self, key: str, value: str, ttl: int) -> None:
        self._r.setex(key, ttl, value)

    def expire(self, key: str, ttl: int) -> None:
        self._r.expire(key, ttl)

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def delete_prefix(self, prefix: str) -> int:
        keys = list(self._r.scan_iter(f"{prefix}*", count=100))
        if keys:
            self._r.delete(*keys)
        return len(keys)

    def count_in_window(self, key: str, window: int) -> tuple[int, int]:
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = pipe.execute()
        if ttl == -1:  # first event of the window
            self._r.expire(key, window)
            ttl = window
        return int(count), max(0, int(ttl))

    def count(self, key: str, window: int) -> int:
        return int(self._r.get(key) or 0)

    def push(self, key: str, value: str, keep: int, ttl: int) -> None:
        self._r.rpush(key, value)
        self._r.ltrim(key, -keep, -1)
        self._r.expire(key, ttl)

    def tail(self, key: str, n: int) -> list[str]:
        return self._r.lrange(key, -n, -1)

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except (redis_lib.RedisError, OSError):  # redis_lib is set: this backend exists only when it imported
            return False


class MemoryBackend:
    """Process-local. Every store is bounded: expired keys are pruned as writes come in, and above a hard
    cap the soonest-to-expire entries go first, so a long-running process never grows without limit."""

    MAX_KEYS = 20_000
    MAX_WINDOWS = 5_000
    MAX_LISTS = 2_000
    PRUNE_EVERY = 256

    def __init__(self) -> None:
        self._values: dict[str, tuple[str, float]] = {}  # key -> (value, expires at)
        self._windows: dict[str, tuple[int, float]] = {}  # key -> (count, window start)
        self._lists: dict[str, list[str]] = {}
        self._writes = 0

    def _prune(self, force: bool = False) -> None:
        self._writes += 1
        if not force and self._writes % self.PRUNE_EVERY:
            return
        now = time.time()
        for k in [k for k, (_, exp) in self._values.items() if now > exp]:
            del self._values[k]
        if len(self._values) > self.MAX_KEYS:
            for k, _ in sorted(self._values.items(), key=lambda kv: kv[1][1])[: len(self._values) - self.MAX_KEYS]:
                del self._values[k]
        if len(self._lists) > self.MAX_LISTS:
            for k in list(self._lists)[: len(self._lists) - self.MAX_LISTS]:  # insertion order: oldest first
                del self._lists[k]

    def get(self, key: str) -> str | None:
        entry = self._values.get(key)
        if entry is None:
            return None
        value, exp = entry
        if time.time() > exp:
            del self._values[key]
            return None
        return value

    def set(self, key: str, value: str, ttl: int) -> None:
        self._values[key] = (value, time.time() + ttl)
        self._prune()

    def expire(self, key: str, ttl: int) -> None:
        entry = self._values.get(key)
        if entry is not None:
            self._values[key] = (entry[0], time.time() + ttl)

    def delete(self, key: str) -> None:
        self._values.pop(key, None)

    def delete_prefix(self, prefix: str) -> int:
        doomed = [k for k in self._values if k.startswith(prefix)]
        for k in doomed:
            del self._values[k]
        return len(doomed)

    def count_in_window(self, key: str, window: int) -> tuple[int, int]:
        now = time.time()
        count, start = self._windows.get(key, (0, now))
        if now - start > window:
            count, start = 0, now
        self._windows[key] = (count + 1, start)
        if len(self._windows) > self.MAX_WINDOWS:
            for k in [k for k, (_, s) in self._windows.items() if now - s > window]:
                del self._windows[k]
        return count + 1, max(0, int(window - (now - start)))

    def count(self, key: str, window: int) -> int:
        count, start = self._windows.get(key, (0, 0.0))
        return count if time.time() - start <= window else 0

    def clear_window(self, key: str) -> None:
        self._windows.pop(key, None)

    def push(self, key: str, value: str, keep: int, ttl: int) -> None:
        lst = self._lists.setdefault(key, [])
        lst.append(value)
        if len(lst) > keep:
            del lst[:-keep]
        self._prune()

    def tail(self, key: str, n: int) -> list[str]:
        return self._lists.get(key, [])[-n:]

    def ping(self) -> bool:
        return False  # "is Redis up": no


_backend: CacheBackend | None = None


def _redis_configured() -> bool:
    pw = settings.redis_password
    return bool(pw) and "PASTE" not in pw


def backend() -> CacheBackend:
    """Chosen once per process: Redis when configured and reachable, else memory."""
    global _backend
    if _backend is None:
        _backend = MemoryBackend()
        if _redis_configured() and redis_lib is not None:
            try:
                client = redis_lib.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    username=settings.redis_username,
                    password=settings.redis_password,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                    retry_on_timeout=True,
                )
                client.ping()
                _backend = RedisBackend(client)
                log.info("Redis connected")
            except (redis_lib.RedisError, OSError) as e:
                log.warning("Redis connection failed: %s — using in-memory fallback", e)
        else:
            log.info("Redis not configured — using in-memory fallback")
    return _backend


def ping() -> bool:
    return backend().ping()


# ── sessions and schemas ───────────────────────────────────────────────────────────────────


def get_session(user_id: str) -> dict | None:
    raw = backend().get(f"session:{user_id}")
    return json.loads(raw) if raw else None


def set_session(user_id: str, data: dict) -> None:
    backend().set(f"session:{user_id}", json.dumps(data, default=str), SESSION_TTL)


def refresh_session(user_id: str) -> None:
    backend().expire(f"session:{user_id}", SESSION_TTL)


def delete_session(user_id: str) -> None:
    backend().delete(f"session:{user_id}")


def get_schema(user_id: str) -> dict | None:
    raw = backend().get(f"schema:{user_id}")
    return json.loads(raw) if raw else None


def set_schema(user_id: str, schema: dict) -> None:
    backend().set(f"schema:{user_id}", json.dumps(schema, default=str), SESSION_TTL)


# ── the query cache ────────────────────────────────────────────────────────────────────────


def _qcache_key(user_id: str, db_flavor: str, question: str) -> str:
    h = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
    return f"qcache:{user_id}:{db_flavor}:{h}"


def get_cached_query(user_id: str, db_flavor: str, question: str) -> dict | None:
    raw = backend().get(_qcache_key(user_id, db_flavor, question))
    return json.loads(raw) if raw else None


def set_cached_query(user_id: str, db_flavor: str, question: str, result: dict) -> None:
    backend().set(_qcache_key(user_id, db_flavor, question), json.dumps(result, default=str), CACHE_TTL)


def clear_user_query_cache(user_id: str) -> int:
    return backend().delete_prefix(f"qcache:{user_id}:")


# ── rate windows ───────────────────────────────────────────────────────────────────────────


def check_rate_limit(
    user_id: str, limit: int | None = None, window: int | None = None, kind: str = "ratelimit"
) -> tuple[bool, int, int]:
    """Count one event for (kind, user_id) in a fixed window; (allowed, count, seconds left)."""
    limit, window = limit or RATE_LIMIT, window or RATE_WINDOW
    count, ttl = backend().count_in_window(f"{kind}:{user_id}", window)
    return count <= limit, count, ttl


def recent_count(user_id: str, kind: str, window: int) -> int:
    """How many (kind, user_id) events the current window holds, without adding one."""
    return backend().count(f"{kind}:{user_id}", window)


def clear_count(user_id: str, kind: str) -> None:
    b = backend()
    b.delete(f"{kind}:{user_id}")
    if isinstance(b, MemoryBackend):
        b.clear_window(f"{kind}:{user_id}")


# ── recent conversation turns (what the model sees as history) ─────────────────────────────


def push_turn(user_id: str, session_id: str, role: str, content: str) -> None:
    backend().push(
        f"conv:{user_id}:{session_id}", json.dumps({"role": role, "content": content}), MAX_HOT_TURNS, SESSION_TTL
    )


def get_turns(user_id: str, session_id: str, last_n: int = 6) -> list[dict]:
    return [json.loads(r) for r in backend().tail(f"conv:{user_id}:{session_id}", last_n)]


def clear_conversation(user_id: str, session_id: str) -> None:
    key = f"conv:{user_id}:{session_id}"
    b = backend()
    b.delete(key)
    if isinstance(b, MemoryBackend):
        b._lists.pop(key, None)
