"""Tenant scope: administrator pins and enforcement by rewriting every scoped table."""

from __future__ import annotations

import os

from sqlglot import exp

from .validation import SQLGLOT_DIALECT, parse_select

# ═══════════════════════════════════════════════════════════════
#  TENANT SCOPE — enforced by rewriting, shared by MCP and the web app
# ═══════════════════════════════════════════════════════════════

def _parse_scope_pins(raw: str) -> dict[str, tuple[str, str]]:
    """SCOPE_PINS="alice:Tenant_Id=7,bob:Tenant_Id=9" -> {identity: (column, value)}.
    Identity is the MCP token name (HTTP) / MCP_USER_ID (stdio) or the web app's user id."""
    pins: dict[str, tuple[str, str]] = {}
    for item in filter(None, (p.strip() for p in raw.split(","))):
        who, _, rule = item.partition(":")
        column, _, value = rule.partition("=")
        if not (who.strip() and column.strip() and value.strip()):
            raise ValueError(f"SCOPE_PINS entry '{item}' must look like identity:Column=value")
        pins[who.strip()] = (column.strip(), value.strip())
    return pins


SCOPE_PINS = _parse_scope_pins(os.getenv("SCOPE_PINS", ""))


def make_scope(column: str, value: str, schema: dict, label: str = "", pinned: bool = False) -> dict | None:
    """Scope over every table that has the column (exact column name taken from the schema)."""
    tables = [t for t, cols in schema.items() if any(c.split(" ")[0].lower() == column.lower() for c in cols)]
    if not tables:
        return None
    exact = next(c.split(" ")[0] for c in schema[tables[0]] if c.split(" ")[0].lower() == column.lower())
    return {"column": exact, "value": value, "label": label, "tables": tables, "pinned": pinned}


def pinned_scope(identity: str, schema: dict) -> dict | None:
    """The scope an administrator fixed for this identity; the LLM can neither choose nor clear it."""
    pin = SCOPE_PINS.get(identity)
    return make_scope(pin[0], pin[1], schema, label="pinned", pinned=True) if pin else None


def apply_scope(sql: str, flavor: str, scope: dict | None) -> tuple[str, list[str]]:
    """Enforce the tenant filter instead of trusting the model to write it: every reference
    to a scoped table, at any depth (joins, subqueries, CTE bodies), becomes
    (SELECT * FROM t WHERE col = 'value') under the same alias. Returns (sql, scoped tables)."""
    if not scope:
        return sql, []
    scoped = {t.lower() for t in scope["tables"]}
    touched: set[str] = set()

    def _filter(node: exp.Expression) -> exp.Expression:
        if not (isinstance(node, exp.Table) and node.name.lower() in scoped):
            return node
        touched.add(node.name)
        alias = node.args["alias"].this.copy() if node.args.get("alias") else node.this.copy()
        bare = node.copy()
        bare.set("alias", None)
        tenant = exp.EQ(this=exp.column(scope["column"], quoted=True), expression=exp.Literal.string(scope["value"]))
        return exp.select("*").from_(bare).where(tenant).subquery(alias)

    tree = parse_select(sql, flavor).transform(_filter)
    return (tree.sql(dialect=SQLGLOT_DIALECT.get(flavor)) if touched else sql), sorted(touched)
