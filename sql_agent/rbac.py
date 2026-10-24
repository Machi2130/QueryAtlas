"""Who may do what: a settings file, not code.

The code only knows the *permissions* it can check (the constants below). Which role has
which permission, which person has which role, and the idle-lock time live in a JSON
settings file an admin edits (Settings page, or the file itself). The file is re-read
when it changes, so no restart is needed.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

# ── permissions the code checks (the vocabulary of the settings file) ──
ASK            = "ask"               # ask questions, see answers / charts / tables
FEEDBACK       = "feedback"          # 👍/👎: without APPROVE it is only a suggestion
APPROVE        = "feedback.approve"  # a 👍 makes the answer verified (reused with no AI call)
VIEW_SQL       = "view.sql"          # SQL, tables used and technical errors in answers
VIEW_TABLES    = "view.tables"       # the Data tables sidebar (raw schema)
CONNECT        = "connect"           # open a database by typing its connection details
NOTES          = "notes.manage"      # glossary notes
SCOPE          = "scope.manage"      # tenant scope (MCP set_scope)
EXPLAIN        = "sql.explain"       # explain_query / validate_sql
WRITE          = "sql.write"         # execute_write (still also needs MCP_ALLOW_WRITES)
MCP            = "mcp.use"           # use the MCP tools at all
SETTINGS       = "settings.manage"   # edit this file from the Settings page

ALL_PERMISSIONS = [ASK, FEEDBACK, APPROVE, VIEW_SQL, VIEW_TABLES, CONNECT, NOTES, SCOPE, EXPLAIN, WRITE, MCP, SETTINGS]
SQL_GENERATOR_MODES = ("auto", "server", "client")

DEFAULT_SETTINGS = {
    "default_role": "user",          # anyone not listed under "users"
    "mcp_local_role": "dev",         # MCP over stdio runs on the person's own machine
    # a password-opened database unused this long asks for the password again (0 = never)
    "idle_lock_minutes": float(os.getenv("IDLE_LOCK_MINUTES", 30)),
    # who generates SQL in MCP: "auto" (server if LLM key set, else client), "server" (1 round-trip), "client" (MCP host LLM)
    "sql_generator": (os.getenv("SQL_GENERATOR_MODE", "auto").strip().lower() or "auto"),
    "mcp_allow_writes": os.getenv("MCP_ALLOW_WRITES", "").lower() in ("1", "true", "yes"),
    "mcp_allow_ddl": os.getenv("MCP_ALLOW_DDL", "").lower() in ("1", "true", "yes"),
    "roles": {
        "user":  [ASK, FEEDBACK, CONNECT],
        "admin": [ASK, FEEDBACK, APPROVE, VIEW_SQL, VIEW_TABLES, CONNECT, NOTES, SCOPE, MCP, SETTINGS],
        "dev":   [ASK, FEEDBACK, APPROVE, VIEW_SQL, VIEW_TABLES, CONNECT, NOTES, SCOPE, EXPLAIN, WRITE, MCP],
    },
    "users": {},                     # identity -> role, e.g. {"alice": "admin", "auth0|abc": "dev"}
    "mcp_tokens": {},                # identity -> {"hash": sha256_hex, "prefix": "mcp_...", "created": iso}
    "accounts": {},                  # identity -> {"salt": hex, "hash": scrypt_hex, "created": iso}
    "web_sessions": {},              # sha256(ses_token) -> {"user": identity, "created": epoch}
}


class SettingsError(ValueError):
    """The settings document is not valid (unknown permission, missing role, ...)."""


class PermissionDenied(PermissionError):
    def __init__(self, identity: str, permission: str, role: str):
        super().__init__(f"Your role ({role}) does not allow this ({permission}).")
        self.identity, self.permission, self.role = identity, permission, role


def validate_settings(doc: dict) -> dict:
    """Reject documents that would lock everyone out or name permissions the code doesn't have."""
    if not isinstance(doc.get("roles"), dict) or not doc["roles"]:
        raise SettingsError("'roles' must map role names to permission lists.")
    for role, perms in doc["roles"].items():
        unknown = set(perms) - set(ALL_PERMISSIONS)
        if unknown:
            raise SettingsError(f"Role '{role}' has unknown permissions: {sorted(unknown)}")
    for key in ("default_role", "mcp_local_role"):
        if doc.get(key) not in doc["roles"]:
            raise SettingsError(f"'{key}' must be one of the roles.")
    for who, role in (doc.get("users") or {}).items():
        if role not in doc["roles"]:
            raise SettingsError(f"User '{who}' has unknown role '{role}'.")
    if not any(SETTINGS in perms for perms in doc["roles"].values()):
        raise SettingsError("At least one role must keep 'settings.manage', or nobody could change settings again.")
    minutes = doc.get("idle_lock_minutes", 30)
    if not isinstance(minutes, (int, float)) or minutes < 0:
        raise SettingsError("'idle_lock_minutes' must be 0 or more.")
    gen = doc.get("sql_generator", "auto")
    if gen not in SQL_GENERATOR_MODES:
        raise SettingsError(f"'sql_generator' must be one of {list(SQL_GENERATOR_MODES)}.")
    doc.setdefault("sql_generator", "auto")
    for key in ("mcp_allow_writes", "mcp_allow_ddl"):
        if key in doc and not isinstance(doc[key], bool):
            raise SettingsError(f"'{key}' must be true or false.")
    doc.setdefault(
        "mcp_allow_writes",
        os.getenv("MCP_ALLOW_WRITES", "").lower() in ("1", "true", "yes"),
    )
    doc.setdefault(
        "mcp_allow_ddl",
        os.getenv("MCP_ALLOW_DDL", "").lower() in ("1", "true", "yes"),
    )
    doc.setdefault("mcp_tokens", {})
    doc.setdefault("accounts", {})
    doc.setdefault("web_sessions", {})
    return doc


def validate_password_strength(password: str) -> None:
    """Enforce strong password requirements before hashing."""
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if not any(c.isupper() for c in password):
        raise ValueError("Password must include at least one uppercase letter (A-Z).")
    if not any(c.islower() for c in password):
        raise ValueError("Password must include at least one lowercase letter (a-z).")
    if not any(c.isdigit() for c in password):
        raise ValueError("Password must include at least one number (0-9).")
    if not any(not c.isalnum() for c in password):
        raise ValueError("Password must include at least one special character (e.g. !@#$%^&*).")


def _scrypt_digest(password: str, salt: bytes) -> str:
    """Memory-hard scrypt KDF (N=16384, r=8, p=1, 64-byte derived key)."""
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        maxmem=64 * 1024 * 1024,
        dklen=64,
    )
    return dk.hex()


def _atomic(method):
    """The whole read-modify-write of the settings file under the write lock, so a concurrent
    sign-up cannot restore a token that a revoke just removed."""
    def wrapper(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)
    wrapper.__name__, wrapper.__doc__ = method.__name__, method.__doc__
    return wrapper

class Policy:
    """The settings file, cached and re-read whenever it changes on disk."""

    def __init__(self, path: str):
        self.path = path
        self._doc: dict = {}
        self._mtime: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.RLock()   # one read-modify-write of the file at a time

    def _load(self) -> dict:
        try:
            stat = os.stat(self.path)
            signature = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            self.save(copy.deepcopy(DEFAULT_SETTINGS))
            stat = os.stat(self.path)
            signature = (stat.st_mtime_ns, stat.st_size)
        if signature != self._mtime:
            with self._lock, open(self.path, encoding="utf-8") as f:
                self._doc = validate_settings(json.load(f))
                stat = os.stat(self.path)
                self._mtime = (stat.st_mtime_ns, stat.st_size)
        return self._doc

    def settings(self) -> dict:
        return copy.deepcopy(self._load())

    def save(self, doc: dict) -> None:
        validate_settings(doc)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        # The write is inside the lock: two concurrent read-modify-write cycles (a sign-up and a
        # token revoke) would otherwise interleave, and the later write would restore the revoked token.
        with self._write_lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            os.replace(tmp, self.path)
            # Two saves can land on the same file timestamp (Windows): never rely on it to
            # notice our own change.
            with self._lock:
                stat = os.stat(self.path)
                self._doc, self._mtime = copy.deepcopy(doc), (stat.st_mtime_ns, stat.st_size)

    def update(self, change) -> dict:
        """Read-modify-write as one step: `change(doc)` edits the current document under the
        write lock, so no other save can slip between the read and the write."""
        with self._write_lock:
            doc = self.settings()
            change(doc)
            self.save(doc)
            return doc

    def role_of(self, identity: str, mcp_local: bool = False) -> str:
        doc = self._load()
        if identity in BOOTSTRAP_ADMINS:   # .env: whoever installs it can always reach Settings
            return next(r for r, perms in doc["roles"].items() if SETTINGS in perms)
        if identity in (doc.get("users") or {}):
            return doc["users"][identity]
        return doc["mcp_local_role"] if mcp_local else doc["default_role"]

    def permissions(self, identity: str, mcp_local: bool = False) -> set[str]:
        return set(self._load()["roles"][self.role_of(identity, mcp_local)])

    def can(self, identity: str, permission: str, mcp_local: bool = False) -> bool:
        return permission in self.permissions(identity, mcp_local)

    def require(self, identity: str, permission: str, mcp_local: bool = False) -> None:
        if not self.can(identity, permission, mcp_local):
            raise PermissionDenied(identity, permission, self.role_of(identity, mcp_local))

    def idle_lock_seconds(self) -> float:
        return float(self._load().get("idle_lock_minutes", 30)) * 60

    def sql_generator(self) -> str:
        return str(self._load().get("sql_generator", "auto"))

    @_atomic
    def issue_mcp_token(self, identity: str) -> str:
        """Generate a unique personal MCP Bearer token for `identity`, storing only its
        SHA-256 hash in the settings file. Returns the raw token once."""
        raw = "mcp_" + secrets.token_urlsafe(32)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        doc = self.settings()
        tokens = doc.setdefault("mcp_tokens", {})
        tokens[identity] = {
            "hash": digest,
            "prefix": raw[:10] + "…",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.save(doc)
        return raw

    @_atomic
    def revoke_mcp_token(self, identity: str) -> bool:
        doc = self.settings()
        removed = bool((doc.get("mcp_tokens") or {}).pop(identity, None))
        if removed:
            self.save(doc)
        return removed

    def mcp_token_info(self, identity: str) -> dict:
        entry = (self._load().get("mcp_tokens") or {}).get(identity)
        if not entry:
            return {"has_token": False, "prefix": "", "created": ""}
        return {"has_token": True, "prefix": entry.get("prefix", ""), "created": entry.get("created", "")}

    def user_for_mcp_token(self, raw_token: str) -> str | None:
        """Resolve a raw Bearer token to its owner identity by comparing SHA-256 digests."""
        if not raw_token:
            return None
        digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        for who, entry in (self._load().get("mcp_tokens") or {}).items():
            stored = (entry or {}).get("hash", "")
            if stored and hmac.compare_digest(digest, stored):
                return who if _age_seconds(entry.get("created", "")) <= MCP_TOKEN_SECONDS else None
        return None

    # ── User Accounts (scrypt + 32-byte salt) & Web Session Tokens ──

    def has_account(self, identity: str) -> bool:
        return identity in (self._load().get("accounts") or {})

    @_atomic
    def register_account(self, identity: str, password: str, by_operator: bool = False) -> str:
        """Create a new password-protected account and return its role.

        Self sign-up (web page, SaaS customers) gets SIGNUP_ROLE: connect their own databases and read,
        through the UI and MCP, but never write until an administrator grants sql.write. Anyone who can
        reach the server can sign up. Administrators come only from RBAC_ADMINS, and those names
        are created by `by_operator` (python -m sql_agent.rbac create-account, run on the server), so
        nobody can sign up first under an administrator's name."""
        identity = (identity or "").strip()
        if len(identity) < 3 or len(identity) > 64 or any(c.isspace() or c == ":" for c in identity):
            raise ValueError("Username/email must be 3–64 characters with no spaces or colons.")
        if identity.lower() in {n.lower() for n in _reserved_identities()}:
            raise ValueError("This username is reserved. Choose another one.")
        if not by_operator and identity.lower() in {a.lower() for a in BOOTSTRAP_ADMINS}:
            raise ValueError("This username is reserved. Choose another one.")
        validate_password_strength(password)
        doc = self.settings()
        accounts = doc.setdefault("accounts", {})
        if identity in accounts:
            raise ValueError("An account with this username/email already exists.")
        salt = secrets.token_bytes(32)
        accounts[identity] = {
            "salt": salt.hex(),
            "hash": _scrypt_digest(password, salt),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        users = doc.setdefault("users", {})
        if not by_operator or identity not in users:
            # Even a role an admin pre-assigned to this name is not granted: the admin confirms it is that person.
            doc["roles"].setdefault(SIGNUP_ROLE, list(SIGNUP_PERMISSIONS))
            users[identity] = SIGNUP_ROLE
        self.save(doc)
        return self.role_of(identity)

    def verify_account(self, identity: str, password: str) -> bool:
        """Constant-time verification of an existing account's password."""
        identity = (identity or "").strip()
        entry = (self._load().get("accounts") or {}).get(identity)
        if not entry or not password:
            return False
        try:
            salt = bytes.fromhex(entry["salt"])
            computed = _scrypt_digest(password, salt)
        except Exception:
            return False
        return hmac.compare_digest(computed, entry.get("hash", ""))

    @_atomic
    def issue_session_token(self, identity: str) -> str:
        """Mint an opaque web session token (`ses_...`), storing only its SHA-256 hash."""
        raw = "ses_" + secrets.token_urlsafe(32)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        doc = self.settings()
        sessions = doc.setdefault("web_sessions", {})
        now = int(time.time())
        for k in [k for k, v in sessions.items() if now - int((v or {}).get("created", 0)) > WEB_SESSION_SECONDS]:
            sessions.pop(k, None)
        sessions[digest] = {"user": identity, "created": now}
        self.save(doc)
        return raw

    def user_for_session_token(self, raw_token: str) -> str | None:
        if not raw_token or not raw_token.startswith("ses_"):
            return None
        digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        entry = (self._load().get("web_sessions") or {}).get(digest)
        if not entry or time.time() - int(entry.get("created", 0)) > WEB_SESSION_SECONDS:
            return None   # expired sessions stop working at once, not when someone next signs in
        return entry.get("user")

    @_atomic
    def revoke_session_token(self, raw_token: str) -> None:
        if not raw_token or not raw_token.startswith("ses_"):
            return
        digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        doc = self.settings()
        if (doc.get("web_sessions") or {}).pop(digest, None) is not None:
            self.save(doc)


# People who always get a settings-managing role, whatever the file says: the first admin
# of a fresh install, and the way back in if someone locks themselves out.
WEB_SESSION_SECONDS = 12 * 3600       # web sign-in lasts 12 hours
MCP_TOKEN_SECONDS = 30 * 86400        # MCP tokens 30 days (the OAuth expires_in); revocable any time


def _age_seconds(created: str) -> float:
    """Seconds since an issue time saved as %Y-%m-%dT%H:%M:%S (local); unreadable = too old (fail closed)."""
    try:
        return time.time() - time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return float("inf")


SIGNUP_ROLE = "customer"   # self-signed-up accounts: their own databases, read-only
SIGNUP_PERMISSIONS = [ASK, FEEDBACK, APPROVE, VIEW_SQL, VIEW_TABLES, CONNECT, NOTES, SCOPE, EXPLAIN, MCP]
BOOTSTRAP_ADMINS = {a.strip() for a in os.getenv("RBAC_ADMINS", "").split(",") if a.strip()}


def _reserved_identities() -> set[str]:
    """Identities that already belong to someone without an account: .env token holders (the
    shared MCP_AUTH_TOKEN is 'default'), the local MCP user and the unknown-token fallback. An
    account with the same name would become that identity and see its connections and knowledge."""
    token_names = {p.partition(":")[0].strip() for p in os.getenv("MCP_AUTH_TOKENS", "").split(",") if ":" in p}
    return token_names | {os.getenv("MCP_USER_ID", "mcp-agent"), "default", "unknown-token"}

POLICY = Policy(os.getenv("RBAC_FILE", os.path.join("config", "rbac.json")))


def _create_account_cli(argv: list[str]) -> None:
    """python -m sql_agent.rbac create-account NAME: the only way to create an RBAC_ADMINS account."""
    import getpass

    if len(argv) != 2 or argv[0] != "create-account":
        raise SystemExit("Usage: python -m sql_agent.rbac create-account NAME")
    name = argv[1]
    password = getpass.getpass(f"Password for {name}: ")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match.")
    print(f"Created {name} with role '{POLICY.register_account(name, password, by_operator=True)}'.")


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    # The same files the server uses, from any folder; .env first, then the settings it holds
    # (the module-level values above were read before it was loaded).
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(".env")
    BOOTSTRAP_ADMINS = {a.strip() for a in os.getenv("RBAC_ADMINS", "").split(",") if a.strip()}
    POLICY = Policy(os.getenv("RBAC_FILE", os.path.join("config", "rbac.json")))
    _create_account_cli(sys.argv[1:])
