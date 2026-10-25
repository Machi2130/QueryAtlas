"""Several open databases per web user (two SQL Servers, SQL Server + MongoDB, ...).

Each open database is a *connection key* (user + database identity); every piece of
database state in the web app (engine, schema, knowledge, profile, cache, chat history)
is stored under that key, so databases never mix. One of a user's connections is active:
the chat box asks that one. Databases are never joined to each other.

Every open connection is also bound to a unique, server-authoritative RSA-2048 OAEP
encrypted token whose private key exists only in Python process memory (never on disk,
in .env, or in logs). Every query/tool call decrypts and verifies this token on the
server before touching any engine, cache, knowledge graph, or LLM.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import ipaddress
import os
import re
import secrets
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import settings

# ── Connection tokens: opaque random ids, valid only while the server remembers them ──
# A token proves nothing by itself: the server's ledger maps it to (user, connection key), so it
# cannot be forged, replayed after revocation, or used by another user. (An earlier design encrypted
# the payload with RSA-OAEP, which cannot hold more than 190 bytes: hosted session ids broke it.)
_ACTIVE_TOKENS: dict[str, tuple[str, str]] = {}  # secret -> (user_id, conn_key)
_TOKEN_LOCK = threading.Lock()


def issue_rsa_token(user_id: str, conn_key: str) -> str:
    """Issue a unique connection token bound to (user_id, conn_key)."""
    secret = secrets.token_urlsafe(32)
    with _TOKEN_LOCK:
        _ACTIVE_TOKENS[secret] = (user_id, conn_key)
    return "r1." + secret


def verify_rsa_token(token: str, expected_user: str, expected_conn: str = "") -> str:
    """Validate a connection token against the ledger. Returns its connection key, or raises
    PermissionError. Comparison is constant-time so a token cannot be guessed byte by byte."""
    if not token or not isinstance(token, str) or not token.startswith("r1."):
        raise PermissionError("Missing or invalid connection token.")
    secret = token[3:]
    with _TOKEN_LOCK:
        match = next((v for k, v in _ACTIVE_TOKENS.items() if hmac.compare_digest(k, secret)), None)
    if match is None:
        raise PermissionError("Connection token has been revoked or expired.")
    uid, cid = match
    if uid != expected_user:
        raise PermissionError("Connection token belongs to a different user.")
    if expected_conn and cid != expected_conn:
        raise PermissionError("Connection token does not match the target database.")
    return cid


def revoke_rsa_token(token: str) -> None:
    """Forget a token on disconnect so it can never be replayed."""
    if token and token.startswith("r1."):
        with _TOKEN_LOCK:
            _ACTIVE_TOKENS.pop(token[3:], None)


@dataclass(frozen=True)
class ConnectionInfo:
    key: str
    label: str  # what the user sees, e.g. "SQL Server · SalesDB"
    db_type: str
    # How to reopen it: type, host, port, username, database, path, keyspace. Never the password.
    descriptor: dict = field(default_factory=dict, compare=False, hash=False)
    has_password: bool = False  # only password-protected connections lock when idle
    rsa_token: str = field(default="", compare=False, hash=False, repr=False)  # never in a printed object


def allowed_hosts() -> set[str]:
    return settings.allowed_hosts


_HOST_RE = re.compile(r"[a-z0-9._:\[\]\\-]{1,253}")


class TargetRefused(ValueError):
    """This server will not connect there; the message says why and what to do."""


def _not_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop an IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or (isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.ip_network("100.64.0.0/10"))
    )


MONGO_SRV = "mongodb+srv://"


def mongo_srv_name(host: str) -> str:
    """'mongodb+srv://cluster0.ab1cd.mongodb.net' (MongoDB Atlas) -> 'cluster0.ab1cd.mongodb.net';
    '' when host is a plain host name. Credentials belong in their own fields, never in the address."""
    text = str(host or "").strip()
    if not text.lower().startswith(MONGO_SRV):
        return ""
    rest = text[len(MONGO_SRV) :]
    if "@" in rest:
        raise TargetRefused("Put the username and password in their own fields, not in the address.")
    return rest.split("/", 1)[0].split("?", 1)[0].lower()


def _srv_targets(name: str) -> list[str]:
    """The servers an Atlas-style address points to (its DNS SRV record)."""
    import dns.exception
    import dns.resolver

    try:
        return [str(r.target).rstrip(".") for r in dns.resolver.resolve(f"_mongodb._tcp.{name}", "SRV")]
    except dns.exception.DNSException as exc:
        raise TargetRefused(f"'{name}' could not be found. Check the mongodb+srv address.") from exc


def check_target(db_type: str, host: str, db_path: str, hosted: bool) -> str:
    """Decide whether this server may connect to the caller's target, before any password is used.
    Shared by the MCP server and the web app. Returns the checked IP address to connect to ('' when the
    driver resolves the name itself), or raises TargetRefused.

    Hosted (other people choose the target): SQLite only inside MCP_SQLITE_DIR; a host must resolve
    only to public addresses, so a customer cannot turn the server against its own network or the
    cloud metadata address; MCP_ALLOWED_HOSTS lists the operator's own internal databases that are
    allowed anyway. Local stdio: both limits apply only when set."""
    if db_type == "SQLite":
        root = settings.sqlite_dir
        if hosted and not root:
            raise TargetRefused("SQLite files are not available on this server.")
        if any(ch in str(db_path or "") for ch in "?#"):  # a URL parser would cut the path at these
            raise TargetRefused("SQLite path contains characters that are not allowed.")
        if root:
            path, base = os.path.realpath(db_path or ""), os.path.realpath(root)
            try:
                inside = os.path.commonpath([path, base]) == base
            except ValueError:  # different drives
                inside = False
            if not inside:
                raise TargetRefused(f"SQLite files must be inside {root} (MCP_SQLITE_DIR).")
        return ""
    srv = mongo_srv_name(host) if db_type == "MongoDB" else ""
    name = srv or str(host or "localhost").strip().lower()
    # A host is a name, an IP or SQL Server's host\instance: anything else could start another
    # setting in a connection string (';Encrypt=no') or another part of a URL.
    if not _HOST_RE.fullmatch(name):
        raise TargetRefused(f"'{host}' is not a valid database host name.")
    allowed = allowed_hosts()
    if name in allowed:
        return ""
    if allowed and not hosted:
        raise TargetRefused(f"Host '{host}' is not in this server's allowed list (MCP_ALLOWED_HOSTS).")
    if not hosted:
        return ""
    if srv:  # every server the address points to must be public; the driver resolves them again itself
        for target in _srv_targets(srv):
            public_address(target)
        return ""
    return public_address(name)


def public_address(host: str) -> str:
    """The address to use for host, or TargetRefused when it resolves to any private, loopback,
    link-local (cloud metadata), reserved or unspecified address. Shared by database connections
    and URL fetching: on a hosted server, callers must not reach the server's own network."""
    try:
        addresses = sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except (socket.gaierror, UnicodeError) as exc:
        raise TargetRefused(f"Host '{host}' could not be found. Check the address.") from exc
    if not addresses or any(_not_public(a) for a in addresses):
        raise TargetRefused(
            f"Host '{host}' is on a private or internal network, which this service cannot reach. Use a public address."
        )
    return addresses[0]


_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")


def safe_owner(user_id: str) -> str:
    """A user id as a file/folder/Redis-key name that never merges two users ('a@b' and 'a_b'
    used to both become 'a_b'). An id that is already safe is kept as is, so existing folders stay
    valid; any other gets a hash of the exact id after '~', which no kept id can contain."""
    if _SAFE_ID.fullmatch(user_id):
        return user_id
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)[:64]
    return f"{cleaned}~{hashlib.sha256(user_id.encode()).hexdigest()[:16]}"


def connection_key(user_id: str, db_type: str, host: str, port: str, database: str) -> str:
    """Stable per user and database; safe as a folder name. A one-way hash, so nothing saved
    under it (conversation files, cache folders, Redis keys) names the host or database."""
    digest = hashlib.sha512(f"{db_type}\0{host}\0{port}\0{database}".encode()).hexdigest()[:32]
    return f"{safe_owner(user_id)}__{digest}"


def connection_label(db_type: str, host: str, database: str, db_path: str, keyspace: str) -> str:
    name = database or keyspace or (db_path.replace("\\", "/").rsplit("/", 1)[-1] if db_path else "")
    where = f" @ {host}" if host and db_type not in ("SQLite",) and host != "localhost" else ""
    return f"{db_type} · {name or 'default'}{where}"


class ConnectionRegistry:
    """Open connections per user, and which one is active. In-process, like the engines."""

    def __init__(self, idle_lock_seconds: float | Callable[[], float] = 1800, clock=time.monotonic):
        """idle_lock_seconds: a number, or a function read on every check (the settings
        file's idle_lock_minutes, so an admin's change applies at once)."""
        self._open: dict[str, dict[str, ConnectionInfo]] = {}
        self._active: dict[str, str] = {}
        self._last_used: dict[str, float] = {}
        self._idle = idle_lock_seconds if callable(idle_lock_seconds) else (lambda: float(idle_lock_seconds))
        self._clock = clock
        self._lock = threading.Lock()

    def add(self, user_id: str, info: ConnectionInfo) -> ConnectionInfo:
        with self._lock:
            old = self._open.get(user_id, {}).get(info.key)
            if old and old.rsa_token:
                revoke_rsa_token(old.rsa_token)
            if not info.rsa_token:
                info = dataclasses.replace(info, rsa_token=issue_rsa_token(user_id, info.key))
            self._open.setdefault(user_id, {})[info.key] = info
            self._active[user_id] = info.key
            self._last_used[info.key] = self._clock()
            return info

    def get(self, user_id: str, key: str) -> ConnectionInfo | None:
        return self._open.get(user_id, {}).get(key)

    def verify(self, user_id: str, key: str, client_token: str = "") -> str:
        """Server-authoritative RSA token verification for (user_id, key).
        Decrypts the server-vaulted RSA token (and client_token if provided) and returns
        the decrypted connection key from inside the RSA payload."""
        if client_token:
            verify_rsa_token(client_token, expected_user=user_id)
        info = self.get(user_id, key)
        if info is None or not info.rsa_token:
            raise PermissionError("No active RSA-verified connection for this user.")
        return verify_rsa_token(info.rsa_token, expected_user=user_id, expected_conn=key)

    def is_locked(self, user_id: str, key: str) -> bool:
        """Idle past the limit: the user must prove the password again (it is not stored)."""
        info, limit = self.get(user_id, key), self._idle()
        if info is None or not info.has_password or limit <= 0:
            return False
        return self._clock() - self._last_used.get(key, 0.0) > limit

    def touch(self, key: str) -> None:
        """Used now (a question answered, a password re-entered): restart its idle clock."""
        self._last_used[key] = self._clock()

    def activate(self, user_id: str, key: str) -> ConnectionInfo:
        with self._lock:
            info = self._open.get(user_id, {}).get(key)
            if info is None:
                raise KeyError(key)
            verify_rsa_token(info.rsa_token, expected_user=user_id, expected_conn=key)
            self._active[user_id] = key
            return info

    def active(self, user_id: str) -> ConnectionInfo | None:
        key = self._active.get(user_id)
        return self._open.get(user_id, {}).get(key) if key else None

    def remove(self, user_id: str, key: str) -> ConnectionInfo | None:
        """Close one; if it was active, the most recently opened other one becomes active."""
        with self._lock:
            conns = self._open.get(user_id, {})
            old = conns.pop(key, None)
            if old and old.rsa_token:
                revoke_rsa_token(old.rsa_token)
            self._last_used.pop(key, None)
            if self._active.get(user_id) == key:
                if conns:
                    self._active[user_id] = list(conns)[-1]
                else:
                    self._active.pop(user_id, None)
            return self.active(user_id)

    def list(self, user_id: str) -> list[ConnectionInfo]:
        return list(self._open.get(user_id, {}).values())
