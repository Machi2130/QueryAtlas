"""Join selection as a graph problem.

Tables are nodes and relationships are weighted, undirected edges in an adjacency list.
Choosing the joins for a question = connecting the tables it needs with the cheapest set
of edges: the Steiner tree problem (NP-hard), solved with the classic shortest-path
heuristic (Takahashi–Matsuyama): grow a tree from the most relevant table and attach each
further table by its shortest path to the tree. Shortest paths use Dijkstra with a binary
heap, seeded from every tree node at once (multi-source).

Cost: k terminals x O((V + E) log V). A 2,000-table schema with a few thousand keys runs
in milliseconds, and the path length is unbounded, so 'refunds by store city' finds
RETURNS -> ORDERS -> STORES without any per-database hop limit.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Relationship


class SchemaGraph:
    def __init__(self, relationships: list[Relationship]):
        self._adj: dict[str, list[tuple[str, Relationship]]] = defaultdict(list)
        for r in relationships:
            self._adj[r.from_table].append((r.to_table, r))
            self._adj[r.to_table].append((r.from_table, r))

    def neighbours(self, table: str) -> list[str]:
        return sorted({t for t, _ in self._adj.get(table, [])})

    def degree(self, table: str) -> int:
        return len(self._adj.get(table, []))

    def _dijkstra(self, sources: set[str]) -> tuple[dict[str, float], dict[str, tuple[str, Relationship]]]:
        """Distances from the nearest source, and the edge each node was reached by."""
        dist: dict[str, float] = {s: 0.0 for s in sources}
        prev: dict[str, tuple[str, Relationship]] = {}
        heap = [(0.0, s) for s in sorted(sources)]  # sorted: deterministic tie-breaking
        heapq.heapify(heap)
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, float("inf")):
                continue  # stale entry
            for nxt, rel in self._adj.get(node, []):
                nd = d + rel.weight
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt], prev[nxt] = nd, (node, rel)
                    heapq.heappush(heap, (nd, nxt))
        return dist, prev

    def shortest_path(self, src: str, dst: str) -> list[Relationship] | None:
        _, prev = self._dijkstra({src})
        if dst != src and dst not in prev:
            return None
        path, node = [], dst
        while node != src:
            node, rel = prev[node]
            path.append(rel)
        return path[::-1]

    def distances_from(self, source: str) -> dict[str, float]:
        """Relationship-weighted distance from one table to every reachable table."""
        return self._dijkstra({source})[0]

    def connect(self, terminals: list[str], max_tables: int | None = None) -> tuple[list[str], list[Relationship]]:
        """Tables and joins connecting as many terminals as possible (given most relevant
        first). The tree is grown from each terminal in turn and the best kept: most
        terminals covered, then fewest tables, then the more relevant root. Growing only
        from the first one fails when it has no relationships at all. By default, do not
        truncate a valid path; `max_tables` is an optional explicit constraint."""
        terminals = list(dict.fromkeys(terminals))
        best: tuple | None = None
        for rank, root in enumerate(terminals):
            tables, edges = self._grow([root] + [t for t in terminals if t != root], max_tables)
            score = (sum(t in tables for t in terminals), -len(tables), -rank)
            if best is None or score > best[0]:
                best = (score, tables, edges)
        return (best[1], best[2]) if best else ([], [])

    def _grow(self, terminals: list[str], max_tables: int | None) -> tuple[list[str], list[Relationship]]:
        """Takahashi–Matsuyama: attach each terminal, in order, by its shortest path to the
        tree. Unreachable terminals, or ones that would pass max_tables, are skipped."""
        tree: list[str] = [terminals[0]]
        edges: list[Relationship] = []
        for term in terminals[1:]:
            if term in tree:
                continue
            dist, prev = self._dijkstra(set(tree))
            if term not in dist:
                continue  # no relationship path: leave it out
            path_nodes, path_edges, node = [], [], term
            while node not in tree:
                path_nodes.append(node)
                node, rel = prev[node]
                path_edges.append(rel)
            if max_tables is not None and len(tree) + len(path_nodes) > max_tables:
                continue
            tree += path_nodes[::-1]
            edges += path_edges[::-1]
        return tree, edges
