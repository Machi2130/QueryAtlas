"""Developer-mode writes: one statement, shown in full and confirmed by a person through a form the
model never sees, permissions re-checked after the form and again inside the transaction, time-limited
by the database itself, never on a tenant-filtered table, audited."""

from __future__ import annotations

import asyncio
import logging
import re
import time

from mcp import types
from sqlalchemy import text
from sqlglot import exp

from ... import cache as redis_cache
from ... import dialects, rbac
from ...database import _limit_statement_time
from ...errors import ErrCode
from ...validation import QueryValidationError, parse_write
from ..context import (
    QUERY_TIMEOUT,
    active,
    audit,
    ddl_enabled,
    get_session,
    handle_of,
    require_session,
    scope_of,
    track_usage,
    user_can_write,
    writes_enabled,
)
from ..results import err, ok
from ..runtime import rt

log = logging.getLogger(__name__)


# Behind parse_write: server-level commands, and statements that carry credentials
# (CREATE LOGIN / IDENTIFIED BY), which would otherwise be written to audit.log.
_FORBIDDEN_WRITE_RE = re.compile(
    r"\b(xp_\w+|sp_\w+|OPENROWSET|OPENQUERY|OPENDATASOURCE|BULK|SHUTDOWN|RECONFIGURE|BACKUP|RESTORE|DBCC|"
    r"GRANT|REVOKE|DENY|EXEC|EXECUTE|LOGIN|IDENTIFIED)\b",
    re.IGNORECASE,
)
_DDL_AUTOCOMMITS = {"MySQL", "MariaDB", "SQLite"}  # DDL commits on its own: a dry run would be real


async def confirm_write(sid: str, sql: str, dry_run: bool) -> bool:
    """A human confirms every write through an elicitation form; the model cannot
    approve its own statement. No form available (or declined) -> not confirmed."""
    try:
        session = rt.mcp_session()
        result = await session.elicit(
            message=(
                ("DRY RUN (rolled back) on " if dry_run else "EXECUTE WRITE on ")
                + sid
                + ":\n\n"
                + sql
                + "\n\nType YES to confirm."
            ),  # the whole statement; parse_write caps its size
            requestedSchema={
                "type": "object",
                "properties": {"confirm": {"type": "string", "title": "Type YES to run this statement"}},
                "required": ["confirm"],
            },
        )
    except Exception as e:
        log.warning(f"write confirmation unavailable: {e}")
        return False
    return result.action == "accept" and str((result.content or {}).get("confirm", "")).strip().upper() == "YES"


async def execute_nosql_write(sid: str, kind: str, raw, dry_run: bool, reason: str) -> list[types.TextContent]:
    """MongoDB / Cassandra writes under the SQL rules: validated, never on a tenant-filtered collection,
    shown in full and confirmed by a person, permissions re-checked after the form, time-limited, audited.
    Called after the write-enabled / role / allow_writes checks in execute_write."""
    schema = redis_cache.get_schema(sid) or {}
    try:
        parsed, is_ddl, targets = dialects.parse_write(kind, raw, schema, ddl_enabled())
    except QueryValidationError as ve:
        return err(str(ve), ErrCode.INVALID_SQL)
    text_form = dialects.as_text(kind, parsed)
    if kind == dialects.CASSANDRA and dry_run:
        return err("Cassandra cannot roll back, so a dry run is not possible.", ErrCode.INVALID_SQL)
    scope = scope_of(sid)
    if scope:
        scoped = {t.lower() for t in scope["tables"]}
        touched = sorted(t for t in targets if t.lower() in scoped)
        if touched:
            return err(
                f"Writes to {', '.join(touched)} are not allowed while the tenant filter {scope['column']} is active.",
                ErrCode.PERMISSION_DENIED,
            )

    if not await rt.confirm_write(sid, text_form, dry_run):
        audit(
            "execute_write",
            sid,
            text_form,
            ok=False,
            error="not confirmed by user",
            extra={"dry_run": dry_run, "reason": reason},
        )
        return ok({"executed": False, "cancelled": True, "message": "Write not confirmed by user."})

    identity, local = rt.request_user(), rt.is_local_call()
    if (
        not writes_enabled()
        or not rt.policy.can(identity, rbac.WRITE, local)
        or not (get_session(sid) or {}).get("allow_writes")
    ):
        return err(
            "Write access changed while awaiting confirmation. Reconnect or ask an administrator to review "
            "permissions.",
            ErrCode.PERMISSION_DENIED,
        )

    t0 = time.monotonic()
    try:
        affected = await asyncio.wait_for(
            asyncio.to_thread(dialects.run_write, kind, handle_of(sid), parsed, dry_run, QUERY_TIMEOUT),
            timeout=QUERY_TIMEOUT + 5,
        )  # the database's own limit fires first
    except TimeoutError:
        audit(
            "execute_write",
            sid,
            text_form,
            ok=False,
            error=f"timeout after {QUERY_TIMEOUT}s",
            extra={"dry_run": dry_run, "reason": reason},
        )
        return err(f"Statement timed out after {QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
    except QueryValidationError as ve:
        return err(str(ve), ErrCode.INVALID_SQL)
    except Exception as e:
        track_usage(sid, "errors")
        audit("execute_write", sid, text_form, ok=False, error=str(e), extra={"dry_run": dry_run, "reason": reason})
        return err(str(e), ErrCode.QUERY_FAILED)
    audit(
        "execute_write",
        sid,
        text_form,
        ok=True,
        rows=affected,
        ms=int((time.monotonic() - t0) * 1000),
        extra={"dry_run": dry_run, "reason": reason, "ddl": is_ddl},
    )
    if not dry_run:
        redis_cache.clear_user_query_cache(sid)  # cached reads may now be stale
    return ok({"executed": not dry_run, "dry_run": dry_run, "rows_affected": affected, "sql": text_form})


def _write_still_allowed(sid: str, identity: str, local: bool) -> bool:
    """Checked after the form and again inside the transaction: nothing may change between the person's
    YES and the commit."""
    return bool(
        writes_enabled() and rt.policy.can(identity, rbac.WRITE, local) and (get_session(sid) or {}).get("allow_writes")
    )


def _refused_sql_write(sid: str, sql: str, flavor: str, dry_run: bool):
    """Why a SQL write may not run at all: an error result, or None with the parsed statement."""
    try:
        tree, is_ddl = parse_write(sql, flavor, ddl_enabled())
    except QueryValidationError as ve:
        return err(str(ve), ErrCode.INVALID_SQL), None
    if _FORBIDDEN_WRITE_RE.search(sql):
        return err("Statement contains a forbidden keyword.", ErrCode.INVALID_SQL), None
    if dry_run and is_ddl and flavor in _DDL_AUTOCOMMITS:
        return err(
            f"{flavor} commits DDL immediately, so a dry run would really change the database.", ErrCode.INVALID_SQL
        ), None
    scope = scope_of(sid)
    if scope:
        scoped = {t.lower() for t in scope["tables"]}
        touched = sorted({t.name for t in tree.find_all(exp.Table) if t.name.lower() in scoped})
        if touched:
            return err(
                f"Writes to {', '.join(touched)} are not allowed while the tenant filter {scope['column']} is active.",
                ErrCode.PERMISSION_DENIED,
            ), None
    return None, is_ddl


def _run_write(sid: str, sql: str, dry_run: bool, identity: str, local: bool) -> int:
    """One statement in one transaction (in a thread). The database stops it at the time limit, so a write
    the caller gave up on cannot commit later; a dry run always rolls back."""
    with rt.store.handle(sid).connect() as conn:
        trans = conn.begin()
        undo = _limit_statement_time(conn, QUERY_TIMEOUT)
        try:
            if not _write_still_allowed(sid, identity, local):
                raise PermissionError("Write permission was revoked before execution.")
            affected = conn.execute(text(sql)).rowcount
            if dry_run:
                trans.rollback()
            elif not _write_still_allowed(sid, identity, local):
                raise PermissionError("Write permission was revoked before commit.")
            else:
                trans.commit()
        except BaseException:
            trans.rollback()
            raise
        finally:
            undo()
        return affected


async def execute_write(args: dict) -> list[types.TextContent]:
    """Developer mode: one INSERT/UPDATE/DELETE (DDL if allowed), confirmed by the user, audited."""
    sid = active(args)
    sql = str(args.get("sql", "")).strip().rstrip(";")
    dry_run = bool(args.get("dry_run", False))
    reason = str(args.get("reason", ""))[:500]
    session = require_session(sid)
    if not writes_enabled():
        return err("Writes are disabled by an administrator in QueryAtlas Settings.", ErrCode.PERMISSION_DENIED)
    if not user_can_write():
        return err("Your account no longer has sql.write permission.", ErrCode.PERMISSION_DENIED)
    if not session.get("allow_writes"):
        return err(
            "This connection is read-only. Reconnect with allow_writes=true (developer mode) "
            "using your own write-capable credentials.",
            ErrCode.PERMISSION_DENIED,
        )
    kind = dialects.kind(session.get("db_type", "sql"))
    if kind != dialects.SQL:
        return await execute_nosql_write(sid, kind, args.get("sql", args.get("query", "")), dry_run, reason)
    if sid not in rt.store:
        return err("Not connected. Call connect_database first.", ErrCode.NOT_CONNECTED)
    refused, _ = _refused_sql_write(sid, sql, session.get("db_flavor", ""), dry_run)
    if refused:
        return refused

    log_extra = {"dry_run": dry_run, "reason": reason}
    if not await rt.confirm_write(sid, sql, dry_run):
        audit("execute_write", sid, sql, ok=False, error="not confirmed by user", extra=log_extra)
        return ok({"executed": False, "cancelled": True, "message": "Write not confirmed by user."})
    identity, local = rt.request_user(), rt.is_local_call()
    if not _write_still_allowed(sid, identity, local):
        return err(
            "Write access changed while awaiting confirmation. Reconnect or ask an administrator to "
            "review permissions.",
            ErrCode.PERMISSION_DENIED,
        )

    t0 = time.monotonic()
    try:
        affected = await asyncio.wait_for(
            asyncio.to_thread(_run_write, sid, sql, dry_run, identity, local), timeout=QUERY_TIMEOUT
        )
    except TimeoutError:
        audit("execute_write", sid, sql, ok=False, error=f"timeout after {QUERY_TIMEOUT}s", extra=log_extra)
        return err(f"Statement timed out after {QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
    except PermissionError as e:
        audit("execute_write", sid, sql, ok=False, error=str(e), extra=log_extra)
        return err(str(e), ErrCode.AUTH_FAILED)
    except Exception as e:
        track_usage(sid, "errors")
        audit("execute_write", sid, sql, ok=False, error=str(e), extra=log_extra)
        return err(str(e), ErrCode.QUERY_FAILED)
    audit("execute_write", sid, sql, ok=True, rows=affected, ms=int((time.monotonic() - t0) * 1000), extra=log_extra)
    if not dry_run:
        redis_cache.clear_user_query_cache(sid)  # cached SELECT results may now be stale
    return ok({"executed": not dry_run, "dry_run": dry_run, "rows_affected": affected, "sql": sql})
