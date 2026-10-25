# 0001. Every environment variable is read through one settings object

**Context.** Settings were read with `os.getenv` in thirteen modules, parsed slightly differently each
time (`"1"/"true"/"yes"` here, `"true"` only there), and some were read at import time, so a `.env` loaded
later was silently ignored.

**Decision.** `sql_agent/config.py` exposes a `Settings` object with one property per variable. Properties
read the environment when accessed, not when the module is imported, so a test or an operator can change a
variable at run time and the next read sees it. No other module reads `os.environ`.

**Consequences.** New settings are added as a property with its default and its parsing in one place. The
cost is one attribute lookup per read, which is negligible. Settings that must be computed once (pool
sizes, the audit log path) are copied into module constants at import by the modules that own them.
