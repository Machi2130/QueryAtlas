"""MongoDB: schema sampling, read-query validation, tenant scope, execution and explain.

A query is JSON, one of:
  find:      {"collection": c, "filter": {...}, "projection": {...}, "sort": {...}, "skip": n, "limit": n}
  aggregate: {"collection": c, "pipeline": [{"$match": ...}, {"$group": ...}, ...]}

Same rules as SQL: read-only, credential fields excluded, a row cap, a server-side time limit,
and the tenant filter applied by the server.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .database import serialize_value
from .validation import QueryValidationError, is_sensitive

log = logging.getLogger(__name__)

SAMPLE_DOCS = 200  # documents sampled per collection to learn its fields
DEFAULT_LIMIT = 50

_BLOCKED_OPERATORS = {"$where", "$function", "$accumulator", "$objectToArray", "$getField", "$setField"}
_WHOLE_DOCUMENT = {"$$ROOT", "$$CURRENT"}
READ_STAGES = {
    "$match",
    "$project",
    "$addFields",
    "$set",
    "$unset",
    "$group",
    "$sort",
    "$limit",
    "$skip",
    "$count",
    "$unwind",
    "$lookup",
    "$facet",
    "$bucket",
    "$bucketAuto",
    "$sortByCount",
    "$replaceRoot",
    "$replaceWith",
    "$unionWith",
}
_FIND_KEYS = {"collection", "filter", "projection", "sort", "skip", "limit"}


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__.lower()  # datetime, objectid, decimal128 ...


def _sensitive_path(path: str) -> bool:
    return any(is_sensitive(part) for part in path.split("."))


def _paths(doc: dict, prefix: str = ""):
    """(dotted path, type) for every field, nested objects and arrays of objects included."""
    for key, value in doc.items():
        path = f"{prefix}{key}"
        yield path, _type_name(value)
        if isinstance(value, dict):
            yield from _paths(value, path + ".")
        elif isinstance(value, list):
            for item in value[:20]:
                if isinstance(item, dict):
                    yield from _paths(item, path + ".")


def sample_schema(db, sample: int = SAMPLE_DOCS, timeout_seconds: int = 30) -> dict[str, list[str]]:
    """{collection: ["path (type)", ...]} from a random sample of each collection, credential fields left
    out. Same "name (type)" strings as SQL, so retrieval, notes and the profile work unchanged."""
    schema: dict[str, list[str]] = {}
    for name in sorted(db.list_collection_names()):
        if name.startswith("system."):
            continue
        types: dict[str, set[str]] = {}
        for doc in db[name].aggregate([{"$sample": {"size": sample}}], maxTimeMS=timeout_seconds * 1000):
            for path, kind in _paths(doc):
                types.setdefault(path, set()).add(kind)
        fields = [f"{p} ({'|'.join(sorted(t))})" for p, t in types.items() if not _sensitive_path(p)]
        schema[name] = fields or ["(empty)"]
    return schema


# ── Validation ────────────────────────────────────────────────


def _check(node: Any) -> None:
    """Walk filters, projections, sorts and pipeline stages at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            key = str(key)
            if key.startswith("$"):
                if key in _BLOCKED_OPERATORS:
                    raise QueryValidationError(f"'{key}' is not allowed.")
            elif _sensitive_path(key):
                raise QueryValidationError(f"Access to sensitive field '{key}' is blocked.")
            _check(value)
    elif isinstance(node, list):
        for item in node:
            _check(item)
    elif isinstance(node, str) and node.startswith("$"):
        if node.startswith("$$"):
            if node.split(".")[0] in _WHOLE_DOCUMENT:
                raise QueryValidationError(f"'{node}' is not allowed; name the fields.")
        elif _sensitive_path(node[1:]):
            raise QueryValidationError(f"Access to sensitive field '{node[1:]}' is blocked.")


def _check_pipeline(pipeline: Any, schema: dict) -> None:
    if not isinstance(pipeline, list):
        raise QueryValidationError("'pipeline' must be a list of stages.")
    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            raise QueryValidationError('Each pipeline stage must be an object with one key, e.g. {"$match": {...}}.')
        name, body = next(iter(stage.items()))
        if name not in READ_STAGES:
            raise QueryValidationError(f"Stage '{name}' is not allowed (read-only).")
        if name == "$lookup":
            if body.get("from") not in schema:
                raise QueryValidationError(f"$lookup from '{body.get('from')}': not a collection of this database.")
            for key in ("localField", "foreignField"):  # field names without a '$'
                if _sensitive_path(str(body.get(key, ""))):
                    raise QueryValidationError(f"Access to sensitive field '{body[key]}' is blocked.")
            if "pipeline" in body:
                _check_pipeline(body["pipeline"], schema)
        elif name == "$unionWith":
            coll = body if isinstance(body, str) else body.get("coll")
            if coll not in schema:
                raise QueryValidationError(f"$unionWith '{coll}': not a collection of this database.")
            if isinstance(body, dict) and "pipeline" in body:
                _check_pipeline(body["pipeline"], schema)
        elif name == "$facet":
            for sub in body.values():
                _check_pipeline(sub, schema)
        _check(body)


def validate(query: Any, schema: dict) -> dict:
    """One read-only find or aggregate on a collection of this database; returns it normalised."""
    if not isinstance(query, dict):
        raise QueryValidationError("A MongoDB query must be a JSON object.")
    coll = query.get("collection")
    if coll not in schema:
        raise QueryValidationError(f"Collection '{coll}' not in schema.")
    if "pipeline" in query:
        extra = set(query) - {"collection", "pipeline"}
        if extra:
            raise QueryValidationError(f"An aggregate takes only 'collection' and 'pipeline' (not {sorted(extra)}).")
        _check_pipeline(query["pipeline"], schema)
        return {"collection": coll, "pipeline": query["pipeline"]}
    extra = set(query) - _FIND_KEYS
    if extra:
        raise QueryValidationError(f"Unknown keys {sorted(extra)}; a find takes {sorted(_FIND_KEYS)}.")
    out = {
        "collection": coll,
        "filter": query.get("filter") or {},
        "projection": query.get("projection") or None,
        "sort": query.get("sort") or None,
        "skip": int(query.get("skip") or 0),
        "limit": int(query.get("limit") or DEFAULT_LIMIT),
    }
    for part in ("filter", "projection", "sort"):
        if out[part] is not None and not isinstance(out[part], dict):
            raise QueryValidationError(f"'{part}' must be a JSON object.")
        _check(out[part] or {})
    return out


def collections_in(query: dict) -> list[str]:
    """Every collection a validated query reads."""
    found = [query["collection"]]

    def walk(pipeline):
        for stage in pipeline or []:
            name, body = next(iter(stage.items()))
            if name == "$lookup":
                found.append(body["from"])
                walk(body.get("pipeline"))
            elif name == "$unionWith":
                found.append(body if isinstance(body, str) else body["coll"])
                walk(body.get("pipeline") if isinstance(body, dict) else None)
            elif name == "$facet":
                for sub in body.values():
                    walk(sub)

    walk(query.get("pipeline"))
    return list(dict.fromkeys(found))


# ── Tenant scope ──────────────────────────────────────────────


def _tenant_match(scope: dict) -> dict:
    """Values are typed in MongoDB: '7' and 7 differ, so a numeric value matches either."""
    value = scope["value"]
    try:
        number = int(value) if str(value).lstrip("-").isdigit() else float(value)
        return {scope["column"]: {"$in": [value, number]}}
    except ValueError:
        return {scope["column"]: value}


def apply_scope(query: dict, scope: dict | None) -> tuple[dict, list[str]]:
    """The tenant filter added by the server to a validated query; (query, scoped collections)."""
    if not scope:
        return query, []
    scoped = {t.lower() for t in scope["tables"]}

    def walk(pipeline, found):  # collections read by $lookup / $unionWith at any depth, the main one included
        for stage in pipeline or []:
            name, body = next(iter(stage.items()))
            if name == "$lookup":
                found.append(body["from"])
                walk(body.get("pipeline"), found)
            elif name == "$unionWith":
                found.append(body if isinstance(body, str) else body["coll"])
                walk(body.get("pipeline") if isinstance(body, dict) else None, found)
            elif name == "$facet":
                for sub in body.values():
                    walk(sub, found)
        return found

    joined = sorted({c for c in walk(query.get("pipeline"), []) if c.lower() in scoped})
    if joined:
        raise QueryValidationError(
            f"Reading {', '.join(joined)} through $lookup/$unionWith is not allowed "
            f"while the tenant filter {scope['column']} is active; query it directly."
        )
    if query["collection"].lower() not in scoped:
        return query, []
    match = _tenant_match(scope)
    if "pipeline" in query:
        return {**query, "pipeline": [{"$match": match}, *query["pipeline"]]}, [query["collection"]]
    return {**query, "filter": {"$and": [match, query["filter"]]} if query["filter"] else match}, [query["collection"]]


# ── Execution ────────────────────────────────────────────────


def _strip_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_sensitive(v) for k, v in value.items() if not is_sensitive(str(k))}
    if isinstance(value, list):
        return [_strip_sensitive(v) for v in value]
    return value


def run_read(db, query: dict, max_rows: int, timeout_seconds: int) -> tuple[list, list]:
    """Run a validated (and scoped) query with a row cap and a server-side time limit (maxTimeMS)."""
    coll, ms = db[query["collection"]], timeout_seconds * 1000
    if "pipeline" in query:
        docs = list(coll.aggregate([*query["pipeline"], {"$limit": max_rows}], maxTimeMS=ms))
        keep_id = True  # a $group's key is its _id
    else:
        cursor = coll.find(query["filter"], query["projection"], max_time_ms=ms)
        if query.get("sort"):
            cursor = cursor.sort(list(query["sort"].items()))
        docs = list(cursor.skip(query.get("skip", 0)).limit(min(query["limit"], max_rows)))
        keep_id = bool((query.get("projection") or {}).get("_id"))
    docs = [_strip_sensitive(d) for d in docs]
    columns: list[str] = []
    for d in docs:
        columns += [k for k in d if k not in columns and (keep_id or k != "_id")]
    return columns, [[serialize_value(d.get(c)) for c in columns] for d in docs]


# ── Writes (developer mode; confirmed by a person, like SQL) ──

WRITE_OPS = {
    "insertOne": "document",
    "insertMany": "documents",
    "updateOne": "update",
    "updateMany": "update",
    "replaceOne": "replacement",
    "deleteOne": None,
    "deleteMany": None,
}
_UPDATE_OPERATORS = {
    "$set",
    "$unset",
    "$inc",
    "$mul",
    "$rename",
    "$min",
    "$max",
    "$currentDate",
    "$push",
    "$pull",
    "$addToSet",
    "$pop",
    "$setOnInsert",
}
MAX_INSERT = 1000


def _plain_fields(node: Any) -> bool:
    """Does a filter name at least one field, at any depth ($and / $or included)? A filter of only
    expressions ({"$expr": {"$eq": [1, 1]}}) can match every document."""
    if isinstance(node, list):
        return any(_plain_fields(item) for item in node)
    return isinstance(node, dict) and any(not str(k).startswith("$") or _plain_fields(v) for k, v in node.items())


def _check_document(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise QueryValidationError("Each document must be a JSON object.")
    if any(str(k).startswith("$") for k in doc):
        raise QueryValidationError("Documents cannot contain operators ($...).")
    _check(doc)


def parse_write(query: Any, schema: dict) -> dict:
    """One data-changing operation on a collection of this database:
    {"collection": c, "op": "updateMany", "filter": {...}, "update": {"$set": {...}}} and the like."""
    if not isinstance(query, dict):
        raise QueryValidationError("A MongoDB write must be a JSON object.")
    coll, op = query.get("collection"), query.get("op")
    if coll not in schema:
        raise QueryValidationError(f"Collection '{coll}' not in schema.")
    if op not in WRITE_OPS:
        raise QueryValidationError(f"op must be one of {sorted(WRITE_OPS)}.")
    body_key = WRITE_OPS[op]
    allowed = {"collection", "op", "filter"} | ({body_key} if body_key else set())
    if op.startswith("insert"):
        allowed.discard("filter")
    extra = set(query) - allowed
    if extra:
        raise QueryValidationError(f"Unknown keys {sorted(extra)} for {op} (upsert is not allowed).")
    out = {"collection": coll, "op": op}
    if not op.startswith("insert"):
        flt = query.get("filter")
        if not flt or not _plain_fields(flt):
            raise QueryValidationError(f"{op} needs a filter on the documents' fields; an empty filter matches all.")
        _check(flt)
        out["filter"] = flt
    if op == "insertOne":
        _check_document(query.get("document"))
        out["document"] = query["document"]
    elif op == "insertMany":
        docs = query.get("documents")
        if not isinstance(docs, list) or not 0 < len(docs) <= MAX_INSERT:
            raise QueryValidationError(f"documents must be a list of 1 to {MAX_INSERT} objects.")
        for d in docs:
            _check_document(d)
        out["documents"] = docs
    elif op == "replaceOne":
        _check_document(query.get("replacement"))
        out["replacement"] = query["replacement"]
    elif body_key == "update":
        update = query.get("update")
        if not isinstance(update, dict) or not update or set(update) - _UPDATE_OPERATORS:
            raise QueryValidationError(f"update must use update operators only: {sorted(_UPDATE_OPERATORS)}.")
        _check(update)
        out["update"] = update
    return out


def run_write(db, parsed: dict, dry_run: bool, timeout_seconds: int) -> int:
    """Run a parsed write within the time limit; a dry run happens inside a transaction that is always
    aborted (needs a replica set, as on Atlas). Returns documents inserted / modified / deleted."""
    import pymongo
    from pymongo.errors import OperationFailure

    coll = db[parsed["collection"]]

    def apply(session=None) -> int:
        op = parsed["op"]
        if op == "insertOne":
            coll.insert_one(parsed["document"], session=session)
            return 1
        if op == "insertMany":
            return len(coll.insert_many(parsed["documents"], session=session).inserted_ids)
        if op == "replaceOne":
            return coll.replace_one(parsed["filter"], parsed["replacement"], session=session).modified_count
        if op in ("updateOne", "updateMany"):
            fn = coll.update_one if op == "updateOne" else coll.update_many
            return fn(parsed["filter"], parsed["update"], session=session).modified_count
        fn = coll.delete_one if op == "deleteOne" else coll.delete_many
        return fn(parsed["filter"], session=session).deleted_count

    with pymongo.timeout(timeout_seconds):  # the server stops it too, like a statement timeout
        if not dry_run:
            return apply()
        with db.client.start_session() as session:
            session.start_transaction()
            try:
                return apply(session)
            except OperationFailure as e:
                if "replica set" in str(e) or e.code == 20:
                    raise QueryValidationError(
                        "A dry run needs transactions (a replica set, as on Atlas); this server has none."
                    ) from e
                raise
            finally:
                if session.in_transaction:
                    session.abort_transaction()


# ── Knowledge: value hints and references (same facts the SQL profile has) ──


def _values_at(doc: Any, path: list[str]) -> list:
    """Every value at a dotted path, through arrays."""
    if not path:
        return [v for item in doc for v in _values_at(item, [])] if isinstance(doc, list) else [doc]
    if isinstance(doc, list):
        return [v for item in doc for v in _values_at(item, path)]
    if isinstance(doc, dict) and path[0] in doc:
        return _values_at(doc[path[0]], path[1:])
    return []


class Sampler:
    """ProfileService sampler for one MongoDB database: one bounded sample per collection, taken the
    first time a question touches it. The tenant-scope rule (no sampled values in prompts while a
    scope is active) is applied where hints are rendered, as for SQL."""

    def __init__(self, db, sample: int = 2000, timeout_seconds: int = 30):
        self._db, self._sample, self._ms = db, sample, timeout_seconds * 1000

    def profile(self, table, key_columns: set[str]) -> None:
        from .profile.sampler import fill_column

        table.sampled = True
        table.primary_key = table.primary_key or ["_id"]
        coll = self._db[table.name]
        try:
            for info in coll.index_information().values():
                lead = info["key"][0][0]
                if lead in table.columns:
                    table.columns[lead].indexed = True
            table.row_estimate = coll.estimated_document_count(maxTimeMS=self._ms)
            docs = list(coll.aggregate([{"$sample": {"size": self._sample}}], maxTimeMS=self._ms))
        except Exception as e:
            log.warning(f"sample of {table.name} failed: {e}")
            return
        keys = key_columns | set(table.primary_key)
        for name, col in table.columns.items():
            vals = [v for d in docs for v in _values_at(d, name.split(".")) if v is not None]
            fill_column(col, name, vals, keys)


class ReferenceChecker:
    """ProfileBuilder 'confirm': a guessed reference (orders.user_id -> users._id) is kept only when
    most sampled values of the field exist in the target, as JoinChecker does for SQL."""

    def __init__(self, db, min_share: float = 0.8, sample: int = 200, timeout_seconds: int = 10):
        self._db, self._min, self._sample, self._ms = db, min_share, sample, timeout_seconds * 1000

    def __call__(self, candidates: list[tuple[str, str, str, str]]) -> set:
        confirmed = set()
        for ft, fc, tt, tc in candidates:
            try:
                docs = self._db[ft].aggregate(
                    [{"$sample": {"size": self._sample}}, {"$project": {fc: 1}}], maxTimeMS=self._ms
                )
                values = list(
                    {
                        repr(v): v
                        for d in docs
                        for v in _values_at(d, fc.split("."))
                        if v is not None and not isinstance(v, (dict, list))
                    }.values()
                )
                if values:
                    found = self._db[tt].distinct(tc, {tc: {"$in": values}}, maxTimeMS=self._ms)
                    if len(found) / len(values) >= self._min:
                        confirmed.add((ft, fc, tt, tc))
            except Exception as e:
                log.warning(f"reference {ft}.{fc} -> {tt}.{tc} not checked: {e}")
        return confirmed


def collection_usage(db, limit: int, timeout_seconds: int) -> list[tuple[str, int]]:
    """(collection, recorded operations) from the database profiler (system.profile), most used first."""
    prefix = db.name + "."
    rows = db["system.profile"].aggregate(
        [
            {"$match": {"op": {"$in": ["query", "command", "getmore"]}, "ns": {"$regex": "^" + re.escape(prefix)}}},
            {"$group": {"_id": "$ns", "n": {"$sum": 1}}},
            {"$sort": {"n": -1}},
            {"$limit": limit},
        ],
        maxTimeMS=timeout_seconds * 1000,
    )
    return [(r["_id"][len(prefix) :], int(r["n"])) for r in rows]


def explain(db, query: dict) -> dict:
    """The query planner's choice (no execution): winning plan stages and the indexes used."""
    if "pipeline" in query:
        command = {"aggregate": query["collection"], "pipeline": query["pipeline"], "cursor": {}}
    else:
        command = {"find": query["collection"], "filter": query["filter"]}
        for key in ("projection", "sort"):
            if query.get(key):
                command[key] = query[key]
        command["limit"] = query["limit"]
    plan = db.command({"explain": command, "verbosity": "queryPlanner"})
    stages, indexes = [], []

    def walk(node):
        if isinstance(node, dict):
            if "stage" in node:
                stages.append(node["stage"])
            if node.get("indexName"):
                indexes.append(node["indexName"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(plan)
    return {
        "stages": list(dict.fromkeys(stages)),
        "indexes": list(dict.fromkeys(indexes)),
        "collection_scan": "COLLSCAN" in stages,
    }
