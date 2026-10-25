"""Reviewing generated SQL before it runs: faster where provably equivalent, flagged where wrong.

Works on the parsed query (sqlglot) with the database profile in hand, so every rule is
generic: it reasons about joins, keys, indexes and table sizes, never about one schema.

* rewrite(): only transformations that return the same rows (a function on a date
  column turned into a half-open range, so an index on the column can be used).
* review(): findings the model fixes in one repair round, each phrased as the reason
  (fan-out, LEFT JOIN turned inner, top-N without an order, unknown columns, and
  performance problems that only matter on large tables).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .config import settings
from .profile.models import ColumnKind, DatabaseProfile
from .validation import SQLGLOT_DIALECT

LARGE_TABLE_ROWS = settings.large_table_rows  # below this, speed findings are noise
# Measured on 1.2M orders (SQLite): a range keeping ~2% of rows was 1.2x faster through the
# index, one keeping ~27% was 4x SLOWER than the plain scan. SQLite takes an available index
# whatever the cost, so there a rewrite must be selective. SQL Server / PostgreSQL / MySQL
# choose by cost: an index-friendly predicate only adds options, so they always get it.
SQLITE_REWRITE_SELECTIVITY = settings.sqlite_rewrite_selectivity
# Ranking questions. Not "at least one" / "at most 3": a live repair once turned a correct
# count into a top-50 list because "least" matched.
_RANKING = re.compile(
    r"\b(top \d+|top|highest|lowest|largest|smallest|best|worst|biggest)\b|\bthe (most|least)\b", re.IGNORECASE
)
_DATE_TYPE = re.compile(r"date|time", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    code: str  # fan_out | left_join_filter | top_without_order | unknown_column | select_star | ...
    message: str  # the reason, as the model is told in a repair round
    fix: bool = True  # False: worth knowing (shown to developers), not worth an extra LLM call


class SqlReviewer:
    def __init__(self, flavor: str, profile: DatabaseProfile | None = None, rewrite_up_to: float | None = None):
        """rewrite_up_to: largest share of a table a rewritten range may keep (default: any
        for cost-based engines, SQLITE_REWRITE_SELECTIVITY for SQLite)."""
        self._dialect = SQLGLOT_DIALECT.get(flavor)
        self._profile = profile
        self._rewrite_up_to = (
            rewrite_up_to if rewrite_up_to is not None else (SQLITE_REWRITE_SELECTIVITY if flavor == "SQLite" else 1.0)
        )
        self._tables = {t.lower(): t for t in (profile.tables if profile else {})}

    # ── profile lookups ────────────────────────────────────────
    def _table(self, name: str):
        key = self._tables.get(name.lower())
        return self._profile.tables[key] if key else None

    def _column(self, table: str, column: str):
        t = self._table(table)
        if t is None:
            return None
        return next((c for n, c in t.columns.items() if n.lower() == column.lower()), None)

    def _many_per(self, one: str, many: str) -> bool:
        """`many` has many rows per `one` row (a relationship many.x -> one.key)."""
        return bool(self._profile) and any(
            r.from_table.lower() == many.lower() and r.to_table.lower() == one.lower()
            for r in self._profile.relationships
        )

    def _large(self, table: str) -> bool:
        t = self._table(table)
        return bool(t and t.row_estimate and t.row_estimate >= LARGE_TABLE_ROWS)

    def _is_date(self, table: str | None, column: str) -> bool:
        col = self._column(table, column) if table else None
        return bool(col and (col.kind is ColumnKind.DATE or _DATE_TYPE.search(col.type)))

    @staticmethod
    def _scope(select: exp.Select) -> dict[str, str]:
        """alias (lower) -> table name, for the tables this SELECT reads directly."""
        out = {}
        sources = [select.args.get("from")] + list(select.args.get("joins") or [])
        for src in sources:
            node = src.this if src is not None else None
            if isinstance(node, exp.Table):
                out[(node.alias_or_name or node.name).lower()] = node.name
        return out

    def _resolve(self, col: exp.Column, scope: dict[str, str]) -> str | None:
        if col.table:
            return scope.get(col.table.lower())
        owners = [t for t in scope.values() if self._column(t, col.name)]
        return owners[0] if len(owners) == 1 else None

    @staticmethod
    def _own(node: exp.Expression, select: exp.Select) -> bool:
        """The node belongs to this SELECT, not to a subquery inside it."""
        return node.find_ancestor(exp.Select) is select

    # ── safe rewrites ──────────────────────────────────────────
    def rewrite(self, sql: str) -> tuple[str, list[str]]:
        try:
            tree = sqlglot.parse_one(sql, read=self._dialect)
        except sqlglot.errors.SqlglotError:
            return sql, []
        applied: list[str] = []
        for select in list(tree.find_all(exp.Select)):
            scope, where = self._scope(select), select.args.get("where")
            if where is not None:
                for eq in list(where.find_all(exp.EQ)):
                    if not self._own(eq, select):
                        continue
                    rng = self._date_range(eq, scope)
                    if rng and self._worth_rewriting(rng, scope):
                        col, lo, hi = rng
                        before = eq.sql(self._dialect)
                        eq.replace(
                            exp.Paren(
                                this=exp.and_(
                                    exp.GTE(this=col.copy(), expression=exp.Literal.string(lo)),
                                    exp.LT(this=col.copy(), expression=exp.Literal.string(hi)),
                                )
                            )
                        )
                        applied.append(f"{before} -> {col.sql(self._dialect)} range {lo}..{hi}")
        if self._dialect == "tsql":
            for cast in list(tree.find_all(exp.Cast)):
                if type(cast) is exp.Cast and cast.find_ancestor(exp.Where, exp.Join) is not None:
                    to_type = cast.args.get("to")
                    if to_type and any(
                        t in to_type.sql().upper() for t in ("INT", "DECIMAL", "NUMERIC", "FLOAT", "DATE")
                    ):
                        before = cast.sql(self._dialect)
                        cast.replace(exp.TryCast(this=cast.this.copy(), to=to_type.copy()))
                        applied.append(f"{before} -> TRY_CAST")
        return (tree.sql(dialect=self._dialect) if applied else sql), applied

    def _selectivity(self, table: str | None, column: str, lo: str, hi: str) -> float | None:
        """Share of the table a [lo, hi) date range keeps, from the profiled min/max dates."""
        cp = self._column(table, column) if table else None
        if cp is None or not cp.low or not cp.high:
            return None
        try:
            first, last = dt.date.fromisoformat(cp.low[:10]), dt.date.fromisoformat(cp.high[:10])
            a, b = max(dt.date.fromisoformat(lo), first), min(dt.date.fromisoformat(hi), last + dt.timedelta(days=1))
        except ValueError:
            return None
        span = (last - first).days + 1
        return max(0, (b - a).days) / span if span > 0 else None

    def _worth_rewriting(self, rng, scope) -> bool:
        if self._rewrite_up_to >= 1.0:
            return True
        col, lo, hi = rng
        share = self._selectivity(self._resolve(col, scope), col.name, lo, hi)
        return share is not None and share <= self._rewrite_up_to

    @staticmethod
    def _bare(node):
        """The column under type conversions sqlglot adds (CAST, TsOrDsToTimestamp, ...)."""
        while isinstance(node, (exp.Cast, exp.TsOrDsToTimestamp, exp.TsOrDsToDate, exp.TimeStrToTime)):
            node = node.this
        return node if isinstance(node, exp.Column) else None

    def _date_range(self, eq: exp.EQ, scope: dict[str, str]) -> tuple[exp.Column, str, str] | None:
        """FUNC(date_col) = literal for a year or a month -> (col, start, end) of a half-open range."""
        fn, lit = (eq.this, eq.expression) if isinstance(eq.expression, exp.Literal) else (eq.expression, eq.this)
        if not isinstance(lit, exp.Literal):
            return None
        col, period = None, None
        if isinstance(fn, exp.Year) and self._bare(fn.this):
            col, period = self._bare(fn.this), "year"
        elif isinstance(fn, exp.Extract) and self._bare(fn.expression) and fn.this.name.upper() == "YEAR":
            col, period = self._bare(fn.expression), "year"
        elif isinstance(fn, exp.TimeToStr) and self._bare(fn.this):
            fmt = fn.args.get("format")
            period = {"%Y": "year", "%Y-%m": "month"}.get(fmt.name if fmt is not None else "")
            col = self._bare(fn.this)
        elif isinstance(fn, exp.Substring) and isinstance(fn.this, exp.Column):
            start, length = fn.args.get("start"), fn.args.get("length")
            if start is not None and length is not None and start.name == "1":
                period = {"4": "year", "7": "month"}.get(length.name)
                col = fn.this
        if col is None or period is None or not self._is_date(self._resolve(col, scope), col.name):
            return None
        value = lit.name
        m = re.fullmatch(r"(\d{4})" if period == "year" else r"(\d{4})-(\d{2})", value)
        if not m:
            return None
        y = int(m.group(1))
        if period == "year":
            return col, f"{y}-01-01", f"{y + 1}-01-01"
        mo = int(m.group(2))
        if not 1 <= mo <= 12:
            return None
        ny, nm = (y + 1, 1) if mo == 12 else (y, mo + 1)
        return col, f"{y}-{mo:02d}-01", f"{ny}-{nm:02d}-01"

    # ── findings ───────────────────────────────────────────────
    def review(self, sql: str, question: str = "") -> list[Finding]:
        try:
            tree = sqlglot.parse_one(sql, read=self._dialect)
        except sqlglot.errors.SqlglotError:
            return []  # the validator reports unparseable SQL
        findings: list[Finding] = []
        for select in tree.find_all(exp.Select):
            scope = self._scope(select)
            findings += self._fan_out(select, scope)
            findings += self._left_join_filter(select, scope)
            findings += self._unused_left_joins(select, scope)
            findings += self._unknown_columns(select, scope)
            findings += self._speed(select, scope, outermost=select is tree or select.parent is tree)
        if isinstance(tree, exp.Select) and _RANKING.search(question or "") and not tree.args.get("order"):
            findings.append(
                Finding(
                    "top_without_order",
                    "The question asks for a ranking but the query has no "
                    "ORDER BY, so the rows returned are arbitrary. Add ORDER BY on the ranked measure "
                    "(with a tie-breaker) before LIMIT/TOP.",
                )
            )
        return list(dict.fromkeys(findings))

    def _fan_out(self, select: exp.Select, scope: dict[str, str]) -> list[Finding]:
        out = []
        for agg in select.find_all(exp.Sum, exp.Avg, exp.Count):
            if not self._own(agg, select) or isinstance(agg.this, exp.Distinct):
                continue
            cols = [c for c in agg.find_all(exp.Column)]
            owners = {self._resolve(c, scope) for c in cols} - {None}
            for one in owners:
                many = [t for t in scope.values() if t != one and self._many_per(one, t)]
                if many:
                    out.append(
                        Finding(
                            "fan_out",
                            f"{agg.sql(self._dialect)} totals a column of {one} after joining "
                            f"{', '.join(many)}, which has many rows per {one} row, so each {one} value is "
                            f"counted once per {many[0]} row. Aggregate {many[0]} in a subquery first, or "
                            f"take the {one} figure without joining {many[0]}.",
                        )
                    )
                    continue
                # Two many-side tables of the same parent: each row of one repeats per row of the other.
                for parent in scope.values():
                    if not self._many_per(parent, one):
                        continue
                    sibling = [t for t in scope.values() if t not in (one, parent) and self._many_per(parent, t)]
                    if sibling:
                        out.append(
                            Finding(
                                "fan_out",
                                f"{agg.sql(self._dialect)} totals {one} rows, but {sibling[0]} is "
                                f"joined too and both have many rows per {parent} row, so each {one} row "
                                f"repeats once per {sibling[0]} row of the same {parent}. Aggregate {one} and "
                                f"{sibling[0]} per {parent} in separate subqueries, then join those.",
                            )
                        )
                        break
        return out

    def _left_join_filter(self, select: exp.Select, scope: dict[str, str]) -> list[Finding]:
        where = select.args.get("where")
        if where is None:
            return []
        out = []
        for join in select.args.get("joins") or []:
            if (join.args.get("side") or "").upper() != "LEFT" or not isinstance(join.this, exp.Table):
                continue
            alias = (join.this.alias_or_name or join.this.name).lower()
            for col in where.find_all(exp.Column):
                if (col.table or "").lower() == alias and self._own(col, select) and not col.find_ancestor(exp.Is):
                    out.append(
                        Finding(
                            "left_join_filter",
                            f"WHERE filters on {col.sql(self._dialect)} from the LEFT "
                            f"JOINed {join.this.name}: rows without a match have NULL there and are dropped, "
                            "so this behaves like an inner join. Move that condition into the JOIN's ON if "
                            "unmatched rows must stay.",
                        )
                    )
                    break
        return out

    def _unused_left_joins(self, select: exp.Select, scope: dict[str, str]) -> list[Finding]:
        if any(isinstance(e, exp.Star) or isinstance(e.this, exp.Star) for e in select.expressions):
            return []
        out = []
        for join in select.args.get("joins") or []:
            if (join.args.get("side") or "").upper() != "LEFT" or not isinstance(join.this, exp.Table):
                continue
            alias = (join.this.alias_or_name or join.this.name).lower()
            table = join.this.name
            used = any(
                self._own(col, select)
                and col.find_ancestor(exp.Join) is not join
                and ((col.table or "").lower() == alias or (not col.table and self._resolve(col, scope) == table))
                for col in select.find_all(exp.Column)
            )
            if not used:
                out.append(
                    Finding(
                        "unused_left_join",
                        f"LEFT JOIN {table} is never used outside its ON clause, "
                        "so it contributes no columns or filters and can only duplicate rows. Remove it.",
                    )
                )
        return out

    def _unknown_columns(self, select: exp.Select, scope: dict[str, str]) -> list[Finding]:
        out = []
        for col in select.find_all(exp.Column):
            if not col.table or not self._own(col, select):
                continue
            table = scope.get(col.table.lower())
            t = self._table(table) if table else None
            if t is not None and self._column(table, col.name) is None:
                names = ", ".join(list(t.columns)[:25])
                out.append(Finding("unknown_column", f"{table} has no column {col.name}. Its columns are: {names}."))
        return out

    def _speed(self, select: exp.Select, scope: dict[str, str], outermost: bool) -> list[Finding]:
        big = [t for t in scope.values() if self._large(t)]
        if not big:
            return []  # small tables: any plan is fast, findings would be noise
        out = []
        if outermost and any(isinstance(e, exp.Star) for e in select.expressions):
            out.append(
                Finding(
                    "select_star",
                    f"SELECT * reads every column of {', '.join(big)} (large); select only "
                    "the columns the question needs.",
                )
            )
        if select.args.get("distinct") and any(self._many_per(a, b) for a in scope.values() for b in scope.values()):
            out.append(
                Finding(
                    "distinct_join",
                    "DISTINCT over a join to a many-side table: check the join is not "
                    "creating the duplicates DISTINCT removes (EXISTS or aggregation may say it more directly). "
                    "Not necessarily slower: measured faster than EXISTS on SQLite.",
                    fix=False,
                )
            )
        where = select.args.get("where")
        for cmp in where.find_all(exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE) if where is not None else []:
            for side in (cmp.this, cmp.expression):
                if isinstance(side, exp.Func) and not isinstance(side, exp.Column):
                    if isinstance(cmp, exp.EQ) and self._date_range(cmp, scope):
                        continue  # a date period: rewrite() decides, using selectivity
                    col = side.find(exp.Column)
                    table = self._resolve(col, scope) if col is not None else None
                    cp = self._column(table, col.name) if table else None
                    if cp is not None and cp.indexed and self._large(table):
                        out.append(
                            Finding(
                                "non_sargable",
                                f"{side.sql(self._dialect)} wraps the indexed column "
                                f"{table}.{col.name} in a function, so its index cannot be used on a large "
                                "table; compare the bare column with a range or value instead.",
                            )
                        )
        return out
