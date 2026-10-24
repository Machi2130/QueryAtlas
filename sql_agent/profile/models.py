"""What the agent knows about one database: tables, columns, relationships.

Built once per connect from the database's own catalog (see builder.py), enriched lazily
with value profiles (sampler.py), and the same for every database: nothing here knows
about any particular schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property

from .graph import SchemaGraph


class ColumnKind(str, Enum):
    KEY = "key"             # identifier or join column: joined on, never listed or summed
    CATEGORY = "category"   # few distinct values: the values are listed
    NUMBER = "number"
    DATE = "date"
    TEXT = "text"           # free text or too many values to list
    UNKNOWN = "unknown"     # not sampled yet


class TableRole(str, Enum):
    """Position in the relationship graph, derived from foreign-key direction."""
    LOOKUP = "lookup"           # referenced by others, references nothing (customers, products)
    HEADER = "header"           # references others and is referenced (orders)
    DETAIL = "detail"           # references others, nothing references it (order lines, payments)
    STANDALONE = "standalone"   # no relationships


@dataclass
class ColumnProfile:
    name: str
    type: str = ""
    kind: ColumnKind = ColumnKind.UNKNOWN
    values: list[str] = field(default_factory=list)   # CATEGORY only
    low: str = ""                                      # NUMBER / DATE range
    high: str = ""
    indexed: bool = False                              # leads an index: filter / join / sort on it bare

    def hint(self) -> str:
        """Short description of the values for a prompt, e.g. 'Delivered|Cancelled' or '0–25'."""
        if self.kind is ColumnKind.CATEGORY:
            return "|".join(self.values)
        if self.kind is ColumnKind.NUMBER and self.low:
            return f"{self.low}–{self.high}"
        if self.kind is ColumnKind.DATE and self.low:
            return f"{self.low}…{self.high}"
        return ""


@dataclass(frozen=True)
class Relationship:
    """from_table.from_column references to_table.to_column: many 'from' rows per 'to' row."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    source: str = "fk"          # "fk" = declared constraint, "inferred" = guessed from names, "learned" = found in the data

    @property
    def weight(self) -> float:
        """Path cost: a declared key is preferred over a guessed shortcut."""
        return 1.0 if self.source == "fk" else 3.0

    def other(self, table: str) -> str:
        return self.to_table if table == self.from_table else self.from_table

    def describe(self) -> str:
        guessed = {"inferred": ", guessed from names", "learned": ", found in the data"}.get(self.source, "")
        return (f"{self.from_table}.{self.from_column} = {self.to_table}.{self.to_column}"
                f"  (many {self.from_table} rows per {self.to_table} row{guessed})")


@dataclass
class TableProfile:
    name: str
    columns: dict[str, ColumnProfile]
    role: TableRole = TableRole.STANDALONE
    primary_key: list[str] = field(default_factory=list)
    description: str = ""       # business meaning (from the connect-time analysis); unverified
    sampled: bool = False
    row_estimate: int | None = None   # from catalog statistics; tells the model what is expensive


@dataclass
class DatabaseProfile:
    flavor: str
    fingerprint: str            # changes when tables or columns change: the profile is rebuilt
    tables: dict[str, TableProfile]
    relationships: list[Relationship]

    @cached_property
    def graph(self) -> SchemaGraph:
        return SchemaGraph(self.relationships)

    def to_dict(self) -> dict:
        return {"flavor": self.flavor, "fingerprint": self.fingerprint,
                "tables": {n: asdict(t) for n, t in self.tables.items()},
                "relationships": [asdict(r) for r in self.relationships]}

    @classmethod
    def from_dict(cls, d: dict) -> "DatabaseProfile":
        tables = {}
        for name, t in d["tables"].items():
            cols = {c: ColumnProfile(**{**cp, "kind": ColumnKind(cp["kind"])}) for c, cp in t["columns"].items()}
            tables[name] = TableProfile(**{**t, "columns": cols, "role": TableRole(t["role"])})
        return cls(d["flavor"], d["fingerprint"], tables, [Relationship(**r) for r in d["relationships"]])
