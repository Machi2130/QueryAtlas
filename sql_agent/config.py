"""Every setting QueryAtlas reads from the environment, in one place.

Values are read when accessed, not when this module is imported, so a test or an operator can
change a variable at run time and the next read sees it. Each property documents one variable.
"""

from __future__ import annotations

import os

_TRUE = ("1", "true", "yes")
_FALSE = ("0", "false", "no")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _text(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _on(name: str) -> bool:
    """Off unless explicitly switched on (writes, DDL, dev mode)."""
    return os.getenv(name, "").lower() in _TRUE


def _not_off(name: str) -> bool:
    """On unless explicitly switched off (value hints, table picker)."""
    return os.getenv(name, "true").lower() not in _FALSE


def _csv(name: str) -> list[str]:
    return [p.strip() for p in os.getenv(name, "").split(",") if p.strip()]


class Settings:
    # ── identity, policy and hosted-mode auth ─────────────────────────────────
    @property
    def mcp_user_id(self) -> str:
        return _text("MCP_USER_ID", "mcp-agent")

    @property
    def mcp_token_users(self) -> dict[str, str]:
        """token -> user name: MCP_AUTH_TOKENS="alice:<token>,bob:<token>" plus the shared MCP_AUTH_TOKEN."""
        users = {t.strip(): n.strip() for n, _, t in (p.partition(":") for p in _csv("MCP_AUTH_TOKENS")) if t.strip()}
        if _text("MCP_AUTH_TOKEN"):
            users.setdefault(_text("MCP_AUTH_TOKEN"), "default")
        return users

    @property
    def rbac_file(self) -> str:
        return _text("RBAC_FILE", os.path.join("config", "rbac.json"))

    @property
    def bootstrap_admins(self) -> set[str]:
        return set(_csv("RBAC_ADMINS"))

    @property
    def idle_lock_minutes(self) -> float:
        return _float("IDLE_LOCK_MINUTES", 30)

    @property
    def sql_generator_mode(self) -> str:
        return _text("SQL_GENERATOR_MODE", "auto").strip().lower() or "auto"

    @property
    def allow_writes(self) -> bool:
        return _on("MCP_ALLOW_WRITES")

    @property
    def allow_ddl(self) -> bool:
        return _on("MCP_ALLOW_DDL")

    @property
    def scope_pins(self) -> str:
        return _text("SCOPE_PINS")

    # ── web auth ───────────────────────────────────────────────────────────────
    @property
    def dev_mode(self) -> bool:
        return _text("DEV_MODE", "false").lower() == "true"

    @property
    def auth0_domain(self) -> str:
        return _text("AUTH0_DOMAIN")

    @property
    def auth0_audience(self) -> str:
        return _text("AUTH0_AUDIENCE")

    @property
    def auth0_client_id(self) -> str:
        return _text("AUTH0_CLIENT_ID")

    @property
    def allowed_origins(self) -> list[str]:
        return _csv("ALLOWED_ORIGINS")

    @property
    def oauth_redirect_uris(self) -> set[str]:
        return set(_csv("OAUTH_REDIRECT_URIS"))

    def web_ui_url(self, default: str) -> str:
        return _text("WEB_UI_URL", default)

    # ── databases ──────────────────────────────────────────────────────────────
    @property
    def query_timeout_seconds(self) -> int:
        return _int("DB_QUERY_TIMEOUT_SECONDS", 30)

    @property
    def max_rows(self) -> int:
        return int(_text("MAX_ROWS") or _int("MCP_MAX_ROWS", 5000))

    @property
    def allowed_hosts(self) -> set[str]:
        """Hosts a hosted server may reach even on a private network (MCP_ALLOWED_HOSTS)."""
        return {h.lower() for h in _csv("MCP_ALLOWED_HOSTS")}

    @property
    def sqlite_dir(self) -> str:
        return _text("MCP_SQLITE_DIR")

    @property
    def sqlserver_odbc_driver(self) -> str:
        return _text("SQLSERVER_ODBC_DRIVER").strip()

    def db_password(self, key: str) -> str:
        """MCP_DB_PASSWORD_<KEY> for one database, else the shared MCP_DB_PASSWORD."""
        return (_text(f"MCP_DB_PASSWORD_{key}") if key else "") or _text("MCP_DB_PASSWORD")

    @property
    def audit_log(self) -> str:
        return _text("MCP_AUDIT_LOG") or os.path.join(PROJECT_ROOT, "audit.log")

    @property
    def large_table_rows(self) -> int:
        return _int("LARGE_TABLE_ROWS", 100_000)

    @property
    def sqlite_rewrite_selectivity(self) -> float:
        return _float("SQLITE_REWRITE_SELECTIVITY", 0.15)

    # ── LLM ────────────────────────────────────────────────────────────────────
    @property
    def llm_base_url(self) -> str:
        return _text("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")

    @property
    def llm_model(self) -> str:
        return _text("LLM_MODEL", "openai/gpt-oss-120b")

    @property
    def llm_reasoning_effort(self) -> str:
        # a Groq/OpenAI reasoning-model field; only defaulted for the default model
        return _text("LLM_REASONING_EFFORT", "" if _text("LLM_MODEL") else "low")

    @property
    def llm_api_key(self) -> str:
        return _text("LLM_API_KEY") or _text("GROQ_API_KEY")

    @property
    def planner_model(self) -> str:
        return _text("PLANNER_MODEL") or self.llm_model

    @property
    def table_picker(self) -> bool:
        return _not_off("TABLE_PICKER")

    @property
    def prompt_value_hints(self) -> bool:
        return _not_off("PROMPT_VALUE_HINTS")

    # ── cache, limits and pools ────────────────────────────────────────────────
    @property
    def redis_host(self) -> str:
        return _text("REDIS_HOST", "localhost")

    @property
    def redis_port(self) -> int:
        return _int("REDIS_PORT", 6379)

    @property
    def redis_username(self) -> str:
        return _text("REDIS_USERNAME", "default")

    @property
    def redis_password(self) -> str:
        return _text("REDIS_PASSWORD")

    @property
    def session_ttl_seconds(self) -> int:
        return _int("SESSION_TTL_SECONDS", 3600)

    @property
    def cache_ttl_seconds(self) -> int:
        return _int("CACHE_TTL_SECONDS", 86400)

    @property
    def rate_limit_per_minute(self) -> int:
        return _int("RATE_LIMIT_PER_MINUTE", 30)

    @property
    def mcp_calls_per_minute(self) -> int:
        return _int("MCP_RATE_LIMIT_PER_MINUTE", 60)

    @property
    def sign_ups_per_hour(self) -> int:
        return _int("SIGN_UPS_PER_HOUR", 5)

    @property
    def max_request_bytes(self) -> int:
        return int(_float("MAX_REQUEST_MB", 25) * 1024 * 1024)

    @property
    def thread_workers(self) -> int:
        return _int("THREAD_WORKERS", 10)

    @property
    def process_workers(self) -> int:
        return _int("PROCESS_WORKERS", 2)

    # ── optional external stores ───────────────────────────────────────────────
    @property
    def supabase_url(self) -> str:
        return _text("SUPABASE_URL")

    @property
    def supabase_key(self) -> str:
        return _text("SUPABASE_KEY")  # service-role key: never exposed client-side


settings = Settings()
