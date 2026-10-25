"""Read-only SQL validation, parsed per dialect with sqlglot."""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

AUTO_LIMIT = 100  # validate_sql adds this when the SQL has no LIMIT/TOP

# ═══════════════════════════════════════════════════════════════
#  AGENT 3 — Validator
# ═══════════════════════════════════════════════════════════════

# Credentials and identity documents, matched on words of the column name so any naming
# style is caught (Web_Password, MPIN, POSPIN, Passport_Number), not only exact names.
_SENSITIVE_PARTS = ("password", "passwd", "secret", "api_key", "apikey", "private_key", "privatekey")
_SENSITIVE_WORDS = {"pwd", "token", "hash", "salt", "pin", "otp", "cvv", "cvc", "passport", "maiden"}


def is_sensitive(column: str) -> bool:
    """Never shown to the model, never selectable: the database keeps it, we just never read it."""
    name = column.lower()
    words = re.split(r"[^a-z0-9]+", name)
    return (
        any(p in name for p in _SENSITIVE_PARTS)
        or any(w in _SENSITIVE_WORDS for w in words)
        or any(w.endswith("pin") and len(w) <= 6 for w in words)
    )  # mpin, pospin, atmpin


_BLOCKED_KW = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE|MERGE|EXEC|EXECUTE|"
    r"xp_\w+|sp_\w+|OPENROWSET|OPENQUERY|OPENDATASOURCE|BULK|DBCC|BACKUP|RESTORE|"
    r"SHUTDOWN|RECONFIGURE|WAITFOR|GRANT|REVOKE|DENY)\b",
    re.IGNORECASE,
)
# Functions a SELECT can call that read server files, write sequences, sleep or take locks.
# A read-only login is the real guard; this keeps them out of the agent's hands regardless.
_BLOCKED_FN = re.compile(
    r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|pg_stat_file|pg_sleep\w*|lo_import|lo_export|dblink\w*|"
    r"nextval|setval|set_config|pg_terminate_backend|pg_cancel_backend|pg_reload_conf|"
    r"load_file|sleep|benchmark|get_lock|load_extension|readfile|writefile|"
    r"query_to_xml\w*|table_to_xml\w*|schema_to_xml\w*|database_to_xml\w*|cursor_to_xml\w*|"
    r"row_to_json|to_json|to_jsonb|json_agg|jsonb_agg|json_object_agg|jsonb_object_agg|"
    r"xmlforest|xmlagg|hstore|for_json|json_object|json_arrayagg|json_objectagg)\s*\(",
    re.IGNORECASE,
)
_COMMENT = re.compile(r"/\*.*?\*/|--[^\n]*", re.DOTALL)  # /*!50000 ... */ is live SQL on MySQL
_COMMENT_MYSQL = re.compile(r"/\*.*?\*/|--[^\n]*|#[^\n]*", re.DOTALL)  # '#' comments exist only on MySQL
MAX_LIMIT = 5000  # an explicit LIMIT above this is clamped: a cap, not a suggestion


def strip_comments(sql: str, flavor: str = "") -> str:
    """Comments removed, outside string literals. Refuses a comment that MySQL would execute
    (/*! ... */): a UNION hidden there would run while every scanner skipped it. '#' starts a
    comment on MySQL only; elsewhere it is an operator (PostgreSQL's #> JSON path)."""
    comment = _COMMENT_MYSQL if flavor in ("MySQL", "MariaDB") else _COMMENT
    out, i, n = [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            m = _STRING_LITERAL.match(sql, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
        m = comment.match(sql, i)
        if m:
            if m.group(0).startswith("/*!"):
                raise QueryValidationError("MySQL conditional comments (/*! ... */) are not allowed.")
            out.append(" ")
            i = m.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# String literals are data, not statements: blanked before the keyword scan so
# WHERE status = 'delete' is not mistaken for a DELETE.
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")

SQLGLOT_DIALECT = {
    "SQLite": "sqlite",
    "MySQL": "mysql",
    "MariaDB": "mysql",
    "PostgreSQL": "postgres",
    "SQL Server": "tsql",
}
_DEFAULT_SCHEMAS = {"", "dbo", "public", "main"}  # SQL Server, PostgreSQL, SQLite
# Statement kinds that change data or the database, at any depth (CTE, subquery).
_WRITE_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Command,
)


def parse_select(query: str, flavor: str) -> exp.Expression:
    """Parse exactly one read-only statement in the database's dialect, or raise."""
    try:
        trees = [t for t in sqlglot.parse(query, read=SQLGLOT_DIALECT.get(flavor)) if t is not None]
    except sqlglot.errors.SqlglotError as e:
        raise QueryValidationError(f"Could not parse the SQL as {flavor}: {str(e).splitlines()[0][:200]}") from e
    if len(trees) != 1:
        raise QueryValidationError("One statement per call.")
    tree = trees[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise QueryValidationError("Query must be a SELECT (optionally with WITH / UNION).")
    if tree.find(*_WRITE_NODES):
        raise QueryValidationError("Data-changing statements are not allowed (read-only).")
    if tree.find(exp.Into):
        raise QueryValidationError("SELECT ... INTO is not allowed (read-only).")
    return tree


class QueryValidationError(Exception):
    pass


def _expand_stars(tree: exp.Expression, columns_of: dict[str, list[str]]) -> bool:
    """Replace * and t.* over real tables with their readable columns, so a sensitive column
    (left out of the schema) can never come back through a star, a UNION or a renaming alias
    list. Stars over subqueries/CTEs stay: their own SELECTs are expanded the same way.
    Returns True when anything changed."""
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    changed = False
    for select in list(tree.find_all(exp.Select)):
        if not any(
            isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star))
            for e in select.expressions
        ):
            continue
        sources = ([select.args["from"].this] if select.args.get("from") else []) + [
            j.this for j in select.args.get("joins") or []
        ]
        resolved = []  # (qualifier identifier or None, real table name or None)
        for src in sources:
            real = isinstance(src, exp.Table) and src.name.lower() in columns_of and src.name.lower() not in ctes
            alias = src.args.get("alias")
            qual = alias.this if alias and alias.this else (src.this if real else None)
            resolved.append((qual, src.name.lower() if real else None))

        def _columns(qual, table):
            if table is None:
                return [exp.Column(this=exp.Star(), table=qual.copy())] if qual else [exp.Star()]
            if not columns_of[table]:
                raise QueryValidationError(f"Table '{table}' has no readable columns.")
            return [exp.Column(this=exp.to_identifier(c, quoted=True), table=qual.copy()) for c in columns_of[table]]

        if not any(table for _, table in resolved):
            continue  # star over subqueries/CTEs only: already safe
        if any(qual is None for qual, _ in resolved):
            raise QueryValidationError("Give every subquery in FROM an alias, or list the columns instead of *.")
        expanded = []
        for e in select.expressions:
            if isinstance(e, exp.Star):
                for qual, table in resolved:
                    expanded += _columns(qual, table)
            elif isinstance(e, exp.Column) and isinstance(e.this, exp.Star):
                match = [(q, t) for q, t in resolved if q.name.lower() == e.table.lower()]
                if not match:
                    raise QueryValidationError(f"'{e.sql()}' does not name a table in this SELECT's FROM.")
                expanded += _columns(*match[0])
            else:
                expanded.append(e)
        select.set("expressions", expanded)
        changed = True
    return changed


def _check_sensitive(tree: exp.Expression, schema_dict: dict) -> None:
    """Credential columns stay unreadable by name, by field access ((u).password) and through
    a whole-row reference (SELECT u / row_to_json(u) in PostgreSQL)."""
    for node in tree.find_all(exp.Column, exp.Dot):
        for ident in [node.this] if isinstance(node, exp.Column) else node.find_all(exp.Identifier):
            if isinstance(ident, exp.Identifier) and is_sensitive(ident.name):
                raise QueryValidationError(f"Access to sensitive column '{ident.name}' is blocked.")
    columns_of = {t.lower(): cols for t, cols in schema_dict.items()}
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    row_names, known_columns = set(), set()  # whole-row names; columns of the tables this query reads
    for tbl in tree.find_all(exp.Table):
        if tbl.name.lower() in ctes:
            continue
        row_names.add(tbl.name.lower())
        if tbl.alias:
            row_names.add(tbl.alias.lower())
        known_columns |= {c.split(" ")[0].lower() for c in columns_of.get(tbl.name.lower(), [])}
    for col in tree.find_all(exp.Column):
        if isinstance(col.this, exp.Star):
            if col.table.lower() in row_names and not isinstance(col.parent, exp.Select):
                raise QueryValidationError(f"'{col.sql()}' outside the SELECT list is not allowed; name the columns.")
            # 'u.*' inside a subquery whose FROM does not define u (LATERAL / CROSS APPLY reaching an
            # outer table): the star expander only sees the subquery's own sources, so it would stay.
            select = col.parent if isinstance(col.parent, exp.Select) else None
            if select is not None and col.table.lower() in row_names:
                own = {
                    (s.alias or s.name).lower()
                    for s in ([select.args["from"].this] if select.args.get("from") else [])
                    + [j.this for j in select.args.get("joins") or []]
                }
                if col.table.lower() not in own:
                    raise QueryValidationError(
                        f"'{col.sql()}' refers to a table outside this subquery; name the columns."
                    )
        elif not col.table and col.name.lower() in row_names and col.name.lower() not in known_columns:
            raise QueryValidationError(f"Whole-row reference '{col.name}' is not allowed; name the columns.")


def validate_sql(query: str, schema_dict: dict, flavor: str, question: str) -> str:
    q = strip_comments(query, flavor).strip()
    blanked = _STRING_LITERAL.sub("''", q)
    # One statement: a ';' outside a standard '...' literal is refused even where the parser
    # would see one statement, in case a driver reads quotes differently (backslash escapes).
    if ";" in blanked.rstrip().rstrip(";"):
        raise QueryValidationError("One statement per call (no ';').")
    # Keyword scan stays as a second line behind the parser (e.g. xp_/OPENROWSET calls).
    m = _BLOCKED_KW.search(blanked)
    if m:
        # The user's question wording never licenses a write (asking to "update me on sales"
        # must not let an UPDATE statement through).
        raise QueryValidationError(f"Blocked keyword '{m.group(0).upper()}' detected.")
    m = _BLOCKED_FN.search(blanked)
    if m:
        raise QueryValidationError(f"Function '{m.group(1)}' is blocked (side effects or server access).")
    tree = parse_select(q, flavor)
    # Every table reference at any depth (comma joins, subqueries, CTE bodies) must be a
    # real table or a CTE defined in this statement. Only the default schema may be named:
    # otherdb.orders / OtherDB.dbo.Orders would read outside the learned database.
    cte_names = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    shadow = cte_names & {t.lower() for t in schema_dict}
    if shadow:
        raise QueryValidationError(f"A CTE may not use a table's name ({', '.join(sorted(shadow))}); rename it.")
    known = {t.lower() for t in schema_dict} | cte_names
    for tbl in tree.find_all(exp.Table):
        if tbl.catalog or tbl.db.lower() not in _DEFAULT_SCHEMAS:
            raise QueryValidationError(
                f"Use bare table names: '{tbl.sql()}' points outside this database's default schema."
            )
        if tbl.name.lower() not in known:
            raise QueryValidationError(f"Table '{tbl.name or tbl.sql()}' not in schema.")
    _check_sensitive(tree, schema_dict)
    columns_of = {t.lower(): [c.split(" ")[0] for c in cols] for t, cols in schema_dict.items()}
    # The SQL is regenerated from the tree only when it changed, so other queries run as written.
    rewritten = _expand_stars(tree, columns_of)
    # The cap goes on the outermost statement (from the parse tree): a TOP/LIMIT inside a CTE
    # or subquery is not a cap, and SELECT DISTINCT / WITH must not get 'SELECT TOP n DISTINCT'.
    if tree.args.get("with") and tree.args["with"].args.get("recursive"):
        raise QueryValidationError("Recursive CTEs are not allowed: they can run without bound.")
    limit = tree.args.get("limit")
    if not (limit or tree.args.get("fetch")):
        tree = tree.limit(AUTO_LIMIT)
        rewritten = True
    elif limit is not None:
        n = limit.expression
        if isinstance(n, exp.Literal) and n.is_int and int(n.name) > MAX_LIMIT:
            tree = tree.limit(MAX_LIMIT)  # the row cap is a cap, not a default
            rewritten = True
    if rewritten:
        q = tree.sql(dialect=SQLGLOT_DIALECT.get(flavor))
        # sqlglot drops a CTE's column list (WITH t(a, b) AS ...) when it regenerates SQLite; put it back,
        # or the query fails with 'no such column a'.
        for cte in tree.find_all(exp.CTE):
            cols = cte.args.get("alias").args.get("columns") if cte.args.get("alias") else None
            if cols and f"{cte.alias}(" not in q.split(" AS ")[0]:
                q = q.replace(f"WITH {cte.alias} AS", f"WITH {cte.alias}({', '.join(c.name for c in cols)}) AS", 1)
                q = q.replace(f", {cte.alias} AS", f", {cte.alias}({', '.join(c.name for c in cols)}) AS", 1)
    return q.rstrip(";").strip()


WRITE_LIMIT = 10_000  # longer statements are refused: the confirmation form must show the whole thing
_DML_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Merge)
_DDL_NODES = (exp.Create, exp.Drop, exp.Alter, exp.TruncateTable)


def parse_write(sql: str, flavor: str, allow_ddl: bool) -> tuple[exp.Expression, bool]:
    """Parse exactly one data-changing statement, in the database's dialect, or raise.
    Returns (tree, is_ddl). The parse tree decides, not the first word: a statement the parser
    cannot read (a T-SQL batch 'UPDATE ... DROP TABLE x' with no ';') is refused, and only a WHERE
    of the statement itself counts (not one in a comment, a string or a subquery)."""
    if len(sql) > WRITE_LIMIT:
        raise QueryValidationError(f"Statement is longer than {WRITE_LIMIT} characters; split it.")
    if ";" in _STRING_LITERAL.sub("''", sql).rstrip().rstrip(";"):
        raise QueryValidationError("One statement per call (no ';').")
    try:
        trees = [t for t in sqlglot.parse(sql, read=SQLGLOT_DIALECT.get(flavor)) if t is not None]
    except sqlglot.errors.SqlglotError as exc:
        raise QueryValidationError(f"Could not read this as one {flavor} statement; nothing was run.") from exc
    if len(trees) != 1:
        raise QueryValidationError("One statement per call.")
    tree = trees[0]
    is_ddl = isinstance(tree, _DDL_NODES)
    if not (isinstance(tree, _DML_NODES) or is_ddl):
        raise QueryValidationError(
            "execute_write accepts INSERT, UPDATE, DELETE, MERGE"
            + (", CREATE, ALTER, DROP, TRUNCATE" if allow_ddl else "")
            + " only. Use run_sql for SELECT."
        )
    if is_ddl and not allow_ddl:
        raise QueryValidationError("DDL is disabled by an administrator in QueryAtlas Settings.")
    if isinstance(tree, (exp.Update, exp.Delete)):
        where = tree.args.get("where")
        # A condition with no column in it (WHERE 1=1, WHERE TRUE) matches every row.
        if where is None or not where.find(exp.Column):
            raise QueryValidationError("UPDATE/DELETE needs a WHERE clause on the table's columns.")
    return tree, is_ddl
