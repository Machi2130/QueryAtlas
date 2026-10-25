"""Engines, schema and foreign-key introspection, row serialization."""

from __future__ import annotations

import datetime
import decimal
import json
import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from .config import settings
from .connections import MONGO_SRV, mongo_srv_name
from .validation import is_sensitive

log = logging.getLogger(__name__)

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None

# ═══════════════════════════════════════════════════════════════
#  CONNECTION POOL — production SQLAlchemy config
# ═══════════════════════════════════════════════════════════════


def _make_engine(url, connect_args: dict | None = None):
    """
    pool_size=5       — persistent connections per worker process
    max_overflow=10   — extra connections during traffic spikes
    pool_timeout=30   — wait before QueuePool raises timeout error
    pool_recycle=1800 — recycle after 30 min; prevents MySQL 'gone away'
    pool_pre_ping=True — sends SELECT 1 before checkout; detects stale connections
    """
    return create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        connect_args=connect_args or {},
    )


def _require_mysql_tls(engine) -> None:
    """pymysql switches to TLS only if the server offers it, and sends the password in clear text
    if the server asks for mysql_clear_password. Refuse to log in unless TLS is on."""
    from pymysql.connections import Connection
    from pymysql.constants import CLIENT
    from pymysql.err import OperationalError

    class _TLSOnlyConnection(Connection):
        def _request_authentication(self):
            if not self.server_capabilities & CLIENT.SSL:
                raise OperationalError(2026, "TLS required: the server does not offer TLS, credentials were not sent.")
            super()._request_authentication()

    event.listen(engine, "do_connect", lambda _dialect, _rec, cargs, cparams: _TLSOnlyConnection(*cargs, **cparams))


def _explicit_sqlite_transactions(engine) -> None:
    """Python's sqlite3 opens a transaction on its own only before a statement that starts with
    INSERT/UPDATE/DELETE/REPLACE: 'WITH ... DELETE' or DDL ran outside one, so a rollback (a dry
    run) did not undo it. SQLAlchemy's recipe: the driver stops guessing and BEGIN is explicit."""

    @event.listens_for(engine, "connect")
    def _no_driver_transactions(dbapi_connection, _record):
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _begin(conn):
        conn.exec_driver_sql("BEGIN")


def _connection_failure_message(db_type: str, error: Exception) -> str:
    """What to tell the person, never the driver's text (it can quote the address or credentials)."""
    detail = str(error).lower()
    if any(term in detail for term in ("tls", "ssl", "encryption", "certificate", "pre-login handshake")):
        return (
            f"Could not establish an encrypted {db_type} connection. Ask your database administrator "
            "to check server TLS support, the certificate, and the installed client driver. If the server "
            "uses its own (self-signed) certificate, tick 'Trust this server's certificate'."
        )
    if any(
        term in detail
        for term in (
            "login failed",
            "authentication failed",
            "password authentication failed",
            "access denied",
            "18456",
        )
    ):
        return (
            f"The {db_type} server rejected the login. Verify the username, password, authentication mode, "
            "and database access."
        )
    if any(
        term in detail
        for term in ("driver not found", "data source name not found", "no module named", "not installed")
    ):
        return f"The required {db_type} driver is unavailable. Install or configure the database client driver."
    return (
        f"Could not connect to {db_type}. Check the host, port, database name, network access, "
        "and database credentials."
    )


# ═══════════════════════════════════════════════════════════════
#  SERIALIZATION
# ═══════════════════════════════════════════════════════════════


def serialize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if ObjectId and isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    if isinstance(v, set):
        return list(v)
    return v


def safe_rows(columns: list, raw_rows: list) -> list:
    return [[serialize_value(cell) for cell in row] for row in raw_rows]


def _limit_statement_time(conn, seconds: int) -> Callable[[], None]:
    """Make the database itself stop the statement after `seconds`: the caller's asyncio
    timeout only stops waiting, the query would keep running on the server. Returns the undo
    for settings that outlive this checkout (pooled MySQL session, pyodbc connection)."""
    dialect, ms = conn.dialect.name, int(seconds * 1000)
    if dialect == "sqlite":
        # SQLite has no statement timeout; its progress handler is called every N VM steps and can
        # abort the statement. Cleared in the undo so pooled connections do not keep the deadline.
        raw, deadline = conn.connection.dbapi_connection, time.monotonic() + seconds
        raw.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        return lambda: raw.set_progress_handler(None, 0)
    if dialect == "postgresql":
        conn.exec_driver_sql(f"SET LOCAL statement_timeout = {ms}")  # ends with the transaction
    elif dialect in ("mysql", "mariadb"):
        for var, value in (("MAX_EXECUTION_TIME", ms), ("max_statement_time", seconds)):  # MySQL, MariaDB
            try:
                conn.exec_driver_sql(f"SET SESSION {var} = {value}")
                return lambda: conn.exec_driver_sql(f"SET SESSION {var} = 0")
            except DBAPIError:  # unknown variable on the other server family
                continue
    elif conn.dialect.driver == "pyodbc":
        raw = conn.connection.dbapi_connection
        raw.timeout = seconds
        return lambda: setattr(raw, "timeout", 0)
    return lambda: None


def fetch_read(engine, sql: str, max_rows: int, timeout_seconds: int) -> tuple[list, list]:
    """Run one validated read with a row cap (never pulls a whole table) and a database-side time limit."""
    with engine.connect() as conn:
        undo = _limit_statement_time(conn, timeout_seconds)
        try:
            result = conn.execute(text(sql))
            cols = list(result.keys())
            return cols, safe_rows(cols, result.fetchmany(max_rows))
        finally:
            undo()


# ═══════════════════════════════════════════════════════════════
#  SCHEMA FETCHING
# ═══════════════════════════════════════════════════════════════


def fetch_sql_schema(conn, flavor: str) -> dict:
    schema: dict[str, list] = {}
    if flavor == "SQLite":
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")).fetchall()
        for (tbl,) in tables:
            cols = conn.execute(text(f'PRAGMA table_info("{tbl}")')).fetchall()
            schema[tbl] = [f"{c[1]} ({c[2]})" for c in cols]
    elif flavor in ("MySQL", "MariaDB"):
        rows = conn.execute(
            text(
                "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() ORDER BY TABLE_NAME, ORDINAL_POSITION"
            )
        ).fetchall()
        for tbl, col, dtype in rows:
            schema.setdefault(tbl, []).append(f"{col} ({dtype})")
    elif flavor == "PostgreSQL":
        rows = conn.execute(
            text(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema='public' ORDER BY table_name, ordinal_position"
            )
        ).fetchall()
        for tbl, col, dtype in rows:
            schema.setdefault(tbl, []).append(f"{col} ({dtype})")
    elif flavor == "SQL Server":
        rows = conn.execute(
            text(
                "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                "ORDER BY TABLE_NAME, ORDINAL_POSITION"
            )
        ).fetchall()
        for tbl, col, dtype in rows:
            schema.setdefault(tbl, []).append(f"{col} ({dtype})")
    # Credential/identity columns are left out of everything built from the schema
    # (prompts, profile, retrieval); the database itself is untouched.
    return {t: [c for c in cols if not is_sensitive(c.split(" ")[0])] for t, cols in schema.items()}


def fetch_foreign_keys(conn, db_type: str, schema_dict: dict) -> list[dict]:
    """Real FK constraints as join dicts {from: 'a.col', to: 'b.col', source: 'fk'}.
    One metadata query per connect; bare table names to match schema_dict."""
    rows: list = []
    try:
        if db_type == "SQLite":
            for t in schema_dict:
                for r in conn.execute(text(f'PRAGMA foreign_key_list("{t}")')).fetchall():
                    rows.append((t, r[3], r[2], r[4]))
        elif db_type == "SQL Server":
            rows = conn.execute(
                text("""
                SELECT CASE WHEN sp.name = 'dbo' THEN tp.name ELSE sp.name + '.' + tp.name END,
                       cp.name,
                       CASE WHEN sr.name = 'dbo' THEN tr.name ELSE sr.name + '.' + tr.name END,
                       cr.name
                FROM sys.foreign_key_columns fkc
                JOIN sys.tables  tp ON tp.object_id = fkc.parent_object_id
                JOIN sys.schemas sp ON sp.schema_id = tp.schema_id
                JOIN sys.columns cp ON cp.object_id = tp.object_id AND cp.column_id = fkc.parent_column_id
                JOIN sys.tables  tr ON tr.object_id = fkc.referenced_object_id
                JOIN sys.schemas sr ON sr.schema_id = tr.schema_id
                JOIN sys.columns cr ON cr.object_id = tr.object_id AND cr.column_id = fkc.referenced_column_id
            """)
            ).fetchall()
        elif db_type in ("MySQL", "MariaDB"):
            rows = conn.execute(
                text("""
                SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL
            """)
            ).fetchall()
        elif db_type == "PostgreSQL":
            rows = conn.execute(
                text("""
                SELECT kcu.table_name, kcu.column_name, ccu.table_name, ccu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = current_schema()
            """)
            ).fetchall()
    except Exception as e:
        log.warning(f"foreign key extraction failed ({db_type}): {e}")
        return []
    known = {t.lower(): t for t in schema_dict}
    fks: list[dict] = []
    for ft, fc, tt, tc in rows:
        a, b = known.get(str(ft).lower()), known.get(str(tt).lower())
        if a and b:
            fks.append({"from": f"{a}.{fc}", "to": f"{b}.{tc}", "type": "INNER", "source": "fk"})
    return fks


SQL_TYPES = {"SQLite", "MySQL", "MariaDB", "PostgreSQL", "SQL Server"}
_DEFAULT_PORTS = {
    "MySQL": 3306,
    "MariaDB": 3306,
    "PostgreSQL": 5432,
    "SQL Server": 1433,
    "MongoDB": 27017,
    "Cassandra": 9042,
}


def sqlserver_odbc_driver() -> str:
    """Prefer Driver 18 when installed, while retaining compatibility with Driver 17."""
    configured = settings.sqlserver_odbc_driver
    if configured:
        return configured
    try:
        import pyodbc

        installed = set(pyodbc.drivers())
    except ImportError:
        installed = set()
    for driver in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"):
        if driver in installed:
            return driver
    return "ODBC Driver 18 for SQL Server"


def sqlserver_connection_url(
    host: str, port: str | int, username: str, password: str, database: str, trust_server_certificate: bool = True
) -> str:
    """Build an SQLAlchemy URL with ODBC-escaped values, including special password chars.
    Always encrypted; the certificate is checked unless trust_server_certificate."""
    import urllib.parse

    def odbc_value(value: str | int) -> str:  # in {} with '}' doubled: a ';' or '}' cannot end the attribute
        return "{" + str(value).replace("}", "}}") + "}"

    if not re.fullmatch(r"\d{1,5}", str(port)):  # it goes into an ODBC string: ';Encrypt=no' must not ride along
        raise ValueError("Port must be a number.")
    driver = "{" + sqlserver_odbc_driver().replace("}", "}}") + "}"
    connection = (
        f"DRIVER={driver};SERVER={host},{port};DATABASE={odbc_value(database)};"
        f"UID={odbc_value(username)};PWD={odbc_value(password)};"
        f"Encrypt=yes;TrustServerCertificate={'yes' if trust_server_certificate else 'no'};Connection Timeout=30;"
    )
    return "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(connection)


def _tls_context(trust_cert: bool):
    """Verified TLS (public CA bundle, host name checked), or encrypted-only for a self-signed
    certificate the person chose to trust."""
    import ssl

    import certifi

    ctx = ssl.create_default_context(cafile=certifi.where())
    if trust_cert:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def open_handle(
    db_type: str,
    *,
    host: str = "",
    port: str = "",
    username: str = "",
    password: str = "",
    database: str = "",
    db_path: str = "",
    keyspace: str = "",
    secure: bool = False,
    trust_cert: bool = False,
    address: str = "",
):
    """Open and prove a connection (a round trip, so a wrong password fails here).
    Returns an SQLAlchemy engine, a pymongo Database or a Cassandra session.

    Credentials are passed as separate fields, never pasted into a URL, so no character in a
    username or host can move the password into another part (or into an error message).
    secure (hosted): encryption is mandatory and the certificate is verified, unless trust_cert
    (a self-signed certificate the person accepted); credentials are never sent unencrypted.
    address: the IP that check_target approved; PostgreSQL connects to it while still verifying
    the certificate against host, so a DNS answer cannot change after the check."""
    import certifi

    p = port or _DEFAULT_PORTS.get(db_type, "")
    if db_type == "SQLite":
        if any(ch in str(db_path) for ch in "?#"):
            raise ValueError("SQLite path contains characters that are not allowed.")
        # The resolved path, so what is opened is exactly what check_target examined.
        engine = _make_engine(URL.create("sqlite", database=os.path.realpath(db_path) if db_path else db_path))
        _explicit_sqlite_transactions(engine)
    elif db_type == "SQL Server":
        engine = _make_engine(
            sqlserver_connection_url(
                host, p, username, password, database, trust_server_certificate=not secure or trust_cert
            )
        )
    elif db_type in ("MySQL", "MariaDB"):
        url = URL.create(
            "mysql+pymysql",
            username=username or None,
            password=password or None,
            host=host,
            port=int(p),
            database=database or None,
        )
        tls = (
            {"ssl": {"verify_mode": False, "check_hostname": False}}
            if trust_cert
            else {"ssl": {"ca": certifi.where(), "verify_mode": True, "check_hostname": True}}
        )
        engine = _make_engine(url, tls if secure else None)
        if secure:
            _require_mysql_tls(engine)
    elif db_type == "PostgreSQL":
        url = URL.create(
            "postgresql+psycopg2",
            username=username or None,
            password=password or None,
            host=host,
            port=int(p),
            database=database or None,
        )
        args: dict = {}
        if secure:
            # sslrootcert only when verifying: with a CA file present, 'require' would verify too
            args = {"sslmode": "require"} if trust_cert else {"sslmode": "verify-full", "sslrootcert": certifi.where()}
            if address:
                args["hostaddr"] = address
        engine = _make_engine(url, args)
    elif db_type == "MongoDB":
        from pymongo import MongoClient

        tls = (
            (
                {"tls": True, "tlsAllowInvalidCertificates": True}
                if trust_cert
                else {"tls": True, "tlsCAFile": certifi.where()}
            )
            if secure
            else {}
        )
        auth = {"username": username, "password": password, "authSource": "admin"} if username else {}
        srv = mongo_srv_name(host)  # MongoDB Atlas: mongodb+srv://cluster0.xxxx.mongodb.net, no port
        target = {"host": f"{MONGO_SRV}{srv}/"} if srv else {"host": host, "port": int(p)}
        client = MongoClient(**target, serverSelectionTimeoutMS=5000, **auth, **tls)
        client.server_info()
        return client[database]
    elif db_type == "Cassandra":
        from cassandra.auth import PlainTextAuthProvider
        from cassandra.cluster import Cluster

        auth = PlainTextAuthProvider(username, password) if username else None
        tls = {"ssl_context": _tls_context(trust_cert), "ssl_options": {"server_hostname": host}} if secure else {}
        return Cluster([host], port=int(p), auth_provider=auth, **tls).connect(keyspace)
    else:
        raise ValueError(f"Unknown db_type: {db_type}")
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def close_handle(handle) -> None:
    """Really close what open_handle returned. Dropping the reference is not enough: MongoDB and
    Cassandra clients run background threads that keep their logged-in connections open."""
    if handle is None:
        return
    if hasattr(handle, "dispose"):  # SQLAlchemy engine: closes the pooled connections
        handle.dispose()
    elif hasattr(handle, "client"):  # pymongo Database
        handle.client.close()
    elif hasattr(handle, "cluster"):  # Cassandra session
        handle.cluster.shutdown()


def estimate_rows(conn, db_type: str, table: str) -> int | None:
    """Approximate row count from the catalog's statistics: no COUNT(*) scan of a big table.
    SQLite keeps no statistics; MAX(rowid) is an index lookup and close enough."""
    queries = {
        "SQL Server": (
            "SELECT SUM(p.rows) FROM sys.partitions p JOIN sys.tables t ON t.object_id = p.object_id "
            "WHERE t.name = :t AND p.index_id IN (0, 1)"
        ),
        "PostgreSQL": "SELECT reltuples::bigint FROM pg_class WHERE relname = :t AND relkind = 'r'",
        "MySQL": "SELECT table_rows FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = :t",
        "MariaDB": "SELECT table_rows FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = :t",
    }
    try:
        if db_type == "SQLite":
            row = conn.execute(text(f'SELECT MAX(rowid) FROM "{table}"')).fetchone()
        elif db_type in queries:
            row = conn.execute(text(queries[db_type]), {"t": table}).fetchone()
        else:
            return None
    except Exception as e:
        log.warning(f"row estimate for {table} unavailable: {e}")
        return None
    return int(row[0]) if row and row[0] is not None and int(row[0]) >= 0 else None


def fetch_primary_keys(conn, db_type: str, schema_dict: dict) -> dict[str, list[str]]:
    """Declared primary keys {table: [columns]}: one catalog query (SQLite: one PRAGMA per
    table, which is local and cheap). Tables without a primary key are left out."""
    rows: list = []
    try:
        if db_type == "SQLite":
            for t in schema_dict:
                rows += [(t, r[1], r[5]) for r in conn.execute(text(f'PRAGMA table_info("{t}")')).fetchall() if r[5]]
        elif db_type in ("SQL Server", "MySQL", "MariaDB", "PostgreSQL"):
            scope = {
                "SQL Server": "",
                "PostgreSQL": "AND tc.table_schema = current_schema()",
                "MySQL": "AND tc.table_schema = DATABASE()",
                "MariaDB": "AND tc.table_schema = DATABASE()",
            }[db_type]
            rows = conn.execute(
                text(f"""
                SELECT kcu.table_name, kcu.column_name, kcu.ordinal_position
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON kcu.constraint_name = tc.constraint_name AND kcu.table_name = tc.table_name
                 AND kcu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY' {scope}""")
            ).fetchall()
    except Exception as e:
        log.warning(f"primary key extraction failed ({db_type}): {e}")
        return {}
    known = {t.lower(): t for t in schema_dict}
    pks: dict[str, list[tuple[int, str]]] = {}
    for table, column, position in rows:
        if str(table).lower() in known:
            pks.setdefault(known[str(table).lower()], []).append((int(position or 0), str(column)))
    return {t: [c for _, c in sorted(cols)] for t, cols in pks.items()}


def _quote_table(tbl: str, flavor: str) -> str:
    if flavor == "SQL Server":
        return f"[{tbl}]"
    if flavor in ("MySQL", "MariaDB"):
        return f"`{tbl}`"
    return f'"{tbl}"'  # PostgreSQL, SQLite, others


def filter_empty_tables(conn, schema_dict: dict, flavor: str) -> tuple[dict, list]:
    active = {}
    empty = []
    for tbl in schema_dict:
        try:
            if flavor == "SQL Server":
                row = conn.execute(
                    text(
                        "SELECT SUM(p.rows) FROM sys.partitions p "
                        "JOIN sys.tables t ON p.object_id = t.object_id "
                        "WHERE t.name = :tbl AND p.index_id IN (0,1)"
                    ),
                    {"tbl": tbl},
                ).fetchone()
            else:
                quoted = _quote_table(tbl, flavor)
                row = conn.execute(text(f"SELECT COUNT(*) FROM {quoted}")).fetchone()
            count = int(row[0] or 0) if row else 0
            if count > 0:
                active[tbl] = schema_dict[tbl]
            else:
                empty.append(tbl)
        except SQLAlchemyError as e:  # cannot count it: keep the table
            log.debug(f"row count of {tbl} unavailable: {e}")
            active[tbl] = schema_dict[tbl]
    return active, empty


def explain_statement(sql: str, flavor: str) -> str:
    """The plan query for an already-validated SELECT."""
    return {
        "PostgreSQL": f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}",
        "MySQL": f"EXPLAIN {sql}",
        "MariaDB": f"EXPLAIN {sql}",
        "SQLite": f"EXPLAIN QUERY PLAN {sql}",
        "SQL Server": f"SET SHOWPLAN_TEXT ON; {sql}",
    }.get(flavor, f"EXPLAIN {sql}")
