"""Value profiling from a bounded sample: what a column's values look like.

Tells the model that discount_pct holding 10 means 10%, which status word means 'done',
and which dates exist, without any row ever leaving the server. Generic: decided from
values and column roles, never from one database's names.
"""

from __future__ import annotations

import datetime
import decimal
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlglot import exp

from ..config import settings
from ..database import estimate_rows
from ..presentation import fmt_value
from ..validation import SQLGLOT_DIALECT, is_sensitive
from .models import ColumnKind, TableProfile

log = logging.getLogger(__name__)

PROMPT_VALUE_HINTS = settings.prompt_value_hints

# Contact and identity columns are never listed; names pass only when few-valued
# (region names yes, customer names no: those have hundreds of values).
_NEVER_LIST = re.compile(
    r"(^|_)(id|no|num|email|phone|mobile|addr|address|card|account|acct|pan|aadhaar|ssn)$", re.IGNORECASE
)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


# Rows of one column read to judge a join. Real data is skewed (on TBO two members own most emails and
# transactions) and holds references that are not in the target, so the verdict weighs each value by its rows.
READ_ROWS = 2000
MAX_LOOKUP = 200  # distinct values looked up in the target, most frequent first


def _norm(v) -> str:
    return str(v).strip().lower()


class JoinChecker:
    """The data's verdict on a guessed join: are this column's values the target's keys?
    Reads a bounded run of values (no DISTINCT, so no full scan) and looks the most frequent ones up in
    the target column (normally its key, so an index seek). A join holds when most ROWS carry a value
    found in the target and a fair share of the distinct values are found too: rows, because a few
    busy members and a tail of orphan references must not hide a real join; distinct values, so that one
    default value present in both tables cannot make a join."""

    def __init__(
        self,
        engine,
        flavor: str,
        min_share: float = 0.8,
        min_distinct_share: float = 0.5,
        workers: int = 8,
        min_values: int = 1,
    ):
        """min_values: fewest distinct values a column must show to be judged (a flag column with
        three values is 'contained' in everything). Name-guessed joins have name evidence and use 1;
        joins looked for in the data alone use 10."""
        self._engine, self._dialect = engine, SQLGLOT_DIALECT.get(flavor)
        self._top = "TOP" if flavor == "SQL Server" else ""
        self._min_share, self._min_distinct_share, self._min_values = min_share, min_distinct_share, min_values
        self._workers = 1 if flavor == "SQLite" else workers  # remote round trips dominate; SQLite is local
        self.rejected: list[str] = []  # "from -> to: why" for every guess the data did not confirm

    def __call__(self, candidates: list[tuple[str, str, str, str]]) -> set:
        """Checks every guess, one column per task (its values read once), in parallel."""
        by_column: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for ft, fc, tt, tc in candidates:
            by_column.setdefault((ft, fc), []).append((tt, tc))
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            results = [r for rs in pool.map(lambda kv: self._check_column(kv[0], kv[1]), by_column.items()) for r in rs]
        self.rejected = [f"{a}.{b} -> {c}.{d}: {why}" for (a, b, c, d), ok, why in results if not ok]
        return {cand for cand, ok, _ in results if ok}

    def _check_column(self, source: tuple[str, str], targets: list[tuple[str, str]]) -> list[tuple]:
        """[(candidate, confirmed?, why)] for every target of one source column."""
        with self._engine.connect() as conn:
            values, why = self._read(conn, *source)
            counts = Counter(values)
            out = []
            for t in targets:
                cand = (*source, *t)
                if why:
                    out.append((cand, False, why))
                elif len(counts) < self._min_values:
                    out.append((cand, False, f"only {len(counts)} distinct values to judge by"))  # too little to go on
                else:
                    out.append((cand, *self._contained(conn, counts, *t)))
            return out

    def unique(self, table: str, column: str, rows: int = 200, needed: int = 10) -> bool:
        """Does the column hold (nearly) a different value in every row of a sample? Only such a
        column can be what other tables reference."""
        col = self._q(column)
        try:
            with self._engine.connect() as conn:
                got = [
                    r[0]
                    for r in conn.execute(
                        text(self._select(col, f"FROM {self._t(table)} WHERE {col} IS NOT NULL", rows))
                    ).fetchall()
                ]
        except SQLAlchemyError as e:
            log.warning(f"could not read {table}.{column}: {self._why(e)}")
            return False
        if not (len(got) >= needed and len(set(got)) >= 0.95 * len(got)):
            return False
        # Identifier strings are compact tokens (no whitespace, <= 120 chars), never free-text prose.
        return all(" " not in str(v).strip() and len(str(v)) <= 120 for v in got[:30])

    def _read(self, conn, table: str, column: str) -> tuple[list, str]:
        """(the column's values, why it could not be read). Errors are reported, never mistaken for 'not a join'."""
        col = self._q(column)
        try:
            rows = conn.execute(
                text(self._select(col, f"FROM {self._t(table)} WHERE {col} IS NOT NULL", READ_ROWS))
            ).fetchall()
        except SQLAlchemyError as e:
            conn.rollback()
            return [], f"could not read the column: {self._why(e)}"
        return [r[0] for r in rows], ""

    @staticmethod
    def _why(e: SQLAlchemyError) -> str:
        """The kind of failure and its SQLSTATE, never the message: it quotes the statement and the values."""
        code = (getattr(getattr(e, "orig", None), "args", None) or ("",))[0]
        return f"database error {type(e).__name__} {code}".strip()

    def _q(self, name: str) -> str:
        return exp.to_identifier(name, quoted=True).sql(dialect=self._dialect)

    def _t(self, name: str) -> str:
        return exp.to_table(name).sql(dialect=self._dialect, identify=True)

    def _select(self, cols: str, rest: str, n: int) -> str:
        return f"SELECT TOP {n} {cols} {rest}" if self._top else f"SELECT {cols} {rest} LIMIT {n}"

    def _contained(self, conn, counts: Counter, to_table: str, to_col: str) -> tuple[bool, str]:
        """(confirmed?, why) for the most frequent values of the source column against the target column."""
        tc = self._q(to_col)
        values = [v for v, _ in counts.most_common(MAX_LOOKUP)]
        try:
            params = {f"v{i}": v for i, v in enumerate(values)}
            found = {
                _norm(r[0])
                for r in conn.execute(
                    text(
                        f"SELECT DISTINCT {tc} FROM {self._t(to_table)} WHERE {tc} IN "
                        f"({', '.join(':' + p for p in params)})"
                    ),
                    params,
                ).fetchall()
            }
        except SQLAlchemyError as e:  # e.g. text values that cannot convert to the key's type
            conn.rollback()
            return False, self._why(e)
        hit = [v for v in values if _norm(v) in found]
        row_share = sum(counts[v] for v in hit) / max(1, sum(counts[v] for v in values))
        distinct_share = len(hit) / max(1, len(values))
        why = f"{len(hit)} of {len(values)} distinct values found, {row_share:.0%} of rows"
        return row_share >= self._min_share and distinct_share >= self._min_distinct_share, why


class ValueSampler:
    def __init__(
        self, engine, flavor: str, sample_rows: int = 2000, max_values: int = 10, enabled: bool = PROMPT_VALUE_HINTS
    ):
        self.engine = engine
        self._engine, self._flavor = engine, flavor
        self._sample_rows, self._max_values = sample_rows, max_values
        self._enabled = enabled and engine is not None

    def _sample_sql(self, table: str) -> str:
        quoted = exp.to_table(table).sql(dialect=SQLGLOT_DIALECT.get(self._flavor), identify=True)
        if self._flavor == "SQL Server":
            return f"SELECT TOP {self._sample_rows} * FROM {quoted}"
        return f"SELECT * FROM {quoted} LIMIT {self._sample_rows}"

    def profile(self, table: TableProfile, key_columns: set[str]) -> None:
        """Fill kind / values / range for every column, and the primary key. One bounded
        query per table, run the first time a question touches it."""
        table.sampled = True
        if not self._enabled:
            return
        inspector = sa_inspect(self._engine)
        if not table.primary_key:  # normally read in bulk when the profile was built
            try:
                table.primary_key = inspector.get_pk_constraint(table.name).get("constrained_columns") or []
            except Exception as e:
                log.warning(f"primary key of {table.name} unavailable: {e}")
        self._indexes_and_size(table, inspector)
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(self._sample_sql(table.name)))
                names, rows = list(result.keys()), result.fetchall()
        except Exception as e:
            log.warning(f"sample of {table.name} failed: {e}")
            return
        keys = key_columns | set(table.primary_key)
        for i, name in enumerate(names):
            col = table.columns.get(name)
            if col is not None:
                fill_column(col, name, [r[i] for r in rows if r[i] is not None], keys, self._max_values)

    def _indexes_and_size(self, table: TableProfile, inspector) -> None:
        """Which columns lead an index (a bare predicate on them can seek) and roughly how
        many rows the table has: the facts a fast query is written from."""
        leading = set(table.primary_key[:1])
        try:
            for ix in inspector.get_indexes(table.name):
                if ix.get("column_names"):
                    leading.add(ix["column_names"][0])
            for uq in inspector.get_unique_constraints(table.name):
                if uq.get("column_names"):
                    leading.add(uq["column_names"][0])
        except Exception as e:
            log.warning(f"indexes of {table.name} unavailable: {e}")
        for name in leading:
            if name in table.columns:
                table.columns[name].indexed = True
        with self._engine.connect() as conn:
            table.row_estimate = estimate_rows(conn, self._flavor, table.name)


def fill_column(col, name: str, vals: list, keys: set, max_values: int = 10) -> None:
    """One column's hint from sampled values, the same for every database: keys and contact-like
    columns are never listed, credential columns never read. name may be a dotted MongoDB path."""
    leaf = name.rsplit(".", 1)[-1]
    if name in keys or _NEVER_LIST.search(leaf):
        col.kind = ColumnKind.KEY
    elif any(is_sensitive(part) for part in name.split(".")):
        col.kind = ColumnKind.TEXT
    elif not vals:
        col.kind = ColumnKind.TEXT
    elif all(isinstance(v, (int, float, decimal.Decimal)) and not isinstance(v, bool) for v in vals):
        col.kind, col.low, col.high = ColumnKind.NUMBER, fmt_value(float(min(vals))), fmt_value(float(max(vals)))
    elif all(isinstance(v, (str, datetime.date)) and _ISO_DATE.match(str(v)) for v in vals):
        col.kind, col.low, col.high = ColumnKind.DATE, str(min(vals))[:10], str(max(vals))[:10]
    else:
        distinct = sorted({str(v) for v in vals if not isinstance(v, (dict, list))})
        if distinct and len(distinct) <= max_values:
            col.kind, col.values = ColumnKind.CATEGORY, [v[:30] for v in distinct]
        else:
            col.kind = ColumnKind.TEXT
