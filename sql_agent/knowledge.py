"""
knowledge.py — ChromaDB RAG + NetworkX Knowledge Graph.

Provides:
  1. ChromaDB RAG — stores DDL, documentation, and question/SQL pairs as embeddings
  2. NetworkX Knowledge Graph — explicit table relationships for join traversal
  3. Query history learning — successful queries improve future accuracy

No Vanna dependency — uses ChromaDB directly for vector storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from typing import Any

import chromadb
import networkx as nx


class RAGStore:
    """
    Lightweight RAG using ChromaDB directly.

    Three collections per user:
      - ddl: CREATE TABLE statements
      - docs: business context, table roles, metrics
      - queries: successful question→SQL pairs
    """

    def __init__(self, user_id: str, persist_dir: str = ".chroma_data"):
        path = os.path.join(persist_dir, user_id)
        self._client = chromadb.PersistentClient(path=path)
        self._ddl = self._client.get_or_create_collection(
            name="ddl",
            metadata={"hnsw:space": "cosine"},
        )
        self._docs = self._client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
        )
        self._queries = self._client.get_or_create_collection(
            name="queries",
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _hash_id(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()   # full digest: no two questions share an id

    def add_ddl(self, table_name: str, ddl: str) -> None:
        doc_id = self._hash_id(f"ddl:{table_name}")
        self._ddl.upsert(
            ids=[doc_id],
            documents=[ddl],
            metadatas=[{"table": table_name}],
        )

    def add_ddl_many(self, items: list[tuple[str, str]], batch: int = 64) -> int:
        """Index many tables with one embedding call per batch. One call per table reloaded the
        embedding model every time: 154 s for 523 tables. Returns how many tables were indexed."""
        done = 0
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            try:
                self._ddl.upsert(ids=[self._hash_id(f"ddl:{t}") for t, _ in chunk],
                                 documents=[d for _, d in chunk],
                                 metadatas=[{"table": t} for t, _ in chunk])
                done += len(chunk)
            except Exception as e:   # the rest of the schema is still indexed; say which part was not
                print(f"[knowledge] could not index {len(chunk)} tables ({chunk[0][0]} ...): {e}", file=sys.stderr)
        return done

    def add_documentation(self, doc: str) -> None:
        doc_id = self._hash_id(f"doc:{doc[:100]}")
        self._docs.upsert(
            ids=[doc_id],
            documents=[doc],
        )

    def add_query_pair(self, question: str, sql: str, verified: bool = False) -> None:
        """verified = a human confirmed this SQL answers the question (or the team wrote it).
        An unverified run never overwrites a verified example for the same question."""
        doc_id = self._hash_id(f"q:{question[:100]}")
        if not verified:
            existing = self._queries.get(ids=[doc_id])["metadatas"]
            if existing and existing[0].get("verified"):
                return
        self._queries.upsert(
            ids=[doc_id],
            documents=[question],
            metadatas=[{"sql": sql, "verified": verified}],
        )

    def delete_query_pair(self, question: str) -> bool:
        doc_id = self._hash_id(f"q:{question[:100]}")
        if not self._queries.get(ids=[doc_id])["ids"]:
            return False
        self._queries.delete(ids=[doc_id])
        return True

    def verified_query_pairs(self) -> list[dict]:
        got = self._queries.get(where={"verified": True})
        return [{"question": d, "sql": m.get("sql", "")} for d, m in zip(got["documents"], got["metadatas"])]

    def ddl_for(self, tables: list[str]) -> list[str]:
        if not tables:
            return []
        return self._ddl.get(ids=[self._hash_id(f"ddl:{t}") for t in tables])["documents"]

    def search_ddl(self, question: str, n_results: int = 5) -> list[str]:
        if self._ddl.count() == 0:
            return []
        results = self._ddl.query(
            query_texts=[question],
            n_results=min(n_results, self._ddl.count()),
        )
        return results["documents"][0] if results["documents"] else []

    def search_docs(self, question: str, n_results: int = 3) -> list[str]:
        if self._docs.count() == 0:
            return []
        results = self._docs.query(
            query_texts=[question],
            n_results=min(n_results, self._docs.count()),
        )
        return results["documents"][0] if results["documents"] else []

    def search_similar_queries(self, question: str, n_results: int = 3) -> list[dict]:
        if self._queries.count() == 0:
            return []
        results = self._queries.query(
            query_texts=[question],
            n_results=min(n_results, self._queries.count()),
        )
        pairs = []
        if results["documents"] and results["metadatas"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                pairs.append({"question": doc, "sql": meta.get("sql", ""),
                              "verified": bool(meta.get("verified"))})
        return pairs

    def stats(self) -> dict:
        return {
            "ddl_count": self._ddl.count(),
            "docs_count": self._docs.count(),
            "queries_count": self._queries.count(),
        }


class KnowledgeEngine:
    """
    Combines ChromaDB RAG + NetworkX Knowledge Graph.

    Usage:
        engine = KnowledgeEngine(user_id="user123")
        engine.learn_schema(schema_dict, db_flavor)
        engine.learn_context(sia_context)

        result = engine.get_context_for_question(question)
    """

    def __init__(self, user_id: str, persist_dir: str = ".chroma_data"):
        self.user_id = user_id
        self.graph = nx.DiGraph()
        self._schema_dict: dict = {}
        self._persist_dir = persist_dir
        self.rag = RAGStore(user_id, persist_dir)

    def set_groq_client(self, client) -> None:
        pass

    @property
    def folder(self) -> str:
        """This database's knowledge folder: embeddings, notes, profile."""
        return os.path.join(self._persist_dir, self.user_id)

    def notes(self) -> list[dict]:
        """Glossary notes for this database (written by the MCP add_note tool)."""
        try:
            with open(os.path.join(self.folder, "notes.json"), encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def verified_sql(self, question: str) -> str | None:
        """SQL a human confirmed for exactly this question: reuse it, no LLM call."""
        got = self.rag._queries.get(ids=[self.rag._hash_id(f"q:{question[:100]}")])
        meta = got["metadatas"][0] if got["metadatas"] else {}
        return meta.get("sql") if meta.get("verified") else None

    def _sia_cache_path(self, db_name: str) -> str:
        # db_name may be a SQLite file path (C:/x/y.sqlite): use its basename, filename-safe.
        name = os.path.basename(db_name.replace("\\", "/")) or db_name
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return os.path.join(self._persist_dir, self.user_id, f"sia_cache_{safe}.json")

    def save_sia_context(self, context: dict, db_name: str) -> None:
        if context.get("_error"):
            return  # a failed analysis must not become the cached truth
        path = self._sia_cache_path(db_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(context, f)

    def load_sia_context(self, db_name: str) -> dict | None:
        path = self._sia_cache_path(db_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                ctx = json.load(f)
            if not ctx.get("_error"):
                return ctx
        return None

    # ═══════════════════════════════════════════════════════
    #  KNOWLEDGE GRAPH — build from schema
    # ═══════════════════════════════════════════════════════

    def _build_graph(self, schema_dict: dict) -> None:
        self.graph.clear()

        for table, columns in schema_dict.items():
            self.graph.add_node(table, node_type="table")

            for col_def in columns:
                col_name = col_def.split(" ")[0]
                col_type = ""
                type_match = re.search(r"\((.+?)\)", col_def)
                if type_match:
                    col_type = type_match.group(1)

                col_id = f"{table}.{col_name}"
                self.graph.add_node(col_id, node_type="column", data_type=col_type)
                self.graph.add_edge(table, col_id, relation="has_column")

        from .profile.inference import infer_joins   # one inference for the whole app
        for j in infer_joins(schema_dict):
            self.graph.add_edge(j["from"], j["to"], relation="foreign_key")
            (ft, fc), (tt, tc) = j["from"].split(".", 1), j["to"].split(".", 1)
            self.graph.add_edge(ft, tt, relation="joins_to", via_from=fc, via_to=tc)
            self.graph.add_edge(tt, ft, relation="joins_to", via_from=tc, via_to=fc)

    def set_relationships(self, relationships: list) -> None:
        """Replace name-guessed graph edges with the profile's data-confirmed relationships."""
        drop = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("relation") in ("foreign_key", "joins_to")]
        self.graph.remove_edges_from(drop)
        for r in relationships:
            self.graph.add_edge(f"{r.from_table}.{r.from_column}", f"{r.to_table}.{r.to_column}", relation="foreign_key")
            self.graph.add_edge(r.from_table, r.to_table, relation="joins_to", via_from=r.from_column, via_to=r.to_column)
            self.graph.add_edge(r.to_table, r.from_table, relation="joins_to", via_from=r.to_column, via_to=r.from_column)

    def find_join_path(self, table_a: str, table_b: str) -> list[dict] | None:
        table_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "table"
        ]
        if table_a not in table_nodes or table_b not in table_nodes:
            return None

        try:
            path = nx.shortest_path(self.graph, table_a, table_b)
        except nx.NetworkXNoPath:
            return None

        joins = []
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1]) or {}
            if edge_data.get("relation") == "joins_to":
                joins.append({
                    "from": f"{path[i]}.{edge_data.get('via_from', 'id')}",
                    "to": f"{path[i+1]}.{edge_data.get('via_to', 'id')}",
                    "type": "LEFT",
                })
        return joins if joins else None

    def find_all_join_paths(self, tables: list[str]) -> list[dict]:
        if len(tables) < 2:
            return []

        all_joins = []
        seen = set()
        for i, t1 in enumerate(tables):
            for t2 in tables[i + 1:]:
                key = tuple(sorted([t1, t2]))
                if key in seen:
                    continue
                seen.add(key)
                path = self.find_join_path(t1, t2)
                if path:
                    all_joins.extend(path)
        return all_joins

    def get_related_tables(self, table: str, depth: int = 1) -> list[str]:
        related = []
        for neighbor in self.graph.neighbors(table):
            edge = self.graph.get_edge_data(table, neighbor) or {}
            if edge.get("relation") == "joins_to":
                related.append(neighbor)
                if depth > 1:
                    related.extend(self.get_related_tables(neighbor, depth - 1))
        return list(set(related))

    # ═══════════════════════════════════════════════════════
    #  RAG — train on schema and context
    # ═══════════════════════════════════════════════════════

    def learn_schema(self, schema_dict: dict, db_flavor: str, refresh: bool = False) -> dict:
        self._schema_dict = schema_dict
        self._build_graph(schema_dict)

        existing = self.rag.stats()
        if existing["ddl_count"] >= len(schema_dict) and not refresh:
            return {
                "tables_trained": existing["ddl_count"],
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
                "from_cache": True,
            }

        trained_tables = self.rag.add_ddl_many(
            [(table, self._schema_to_ddl(table, columns, db_flavor)) for table, columns in schema_dict.items()])

        return {
            "tables_trained": trained_tables,
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "from_cache": False,
        }

    def learn_context(self, sia_context: dict) -> None:
        existing = self.rag.stats()
        if existing["docs_count"] > 0:
            return

        biz = sia_context.get("business_type", "")
        if biz and biz != "unknown":
            self.rag.add_documentation(f"This is a {biz} database.")

        for table, meta in sia_context.get("tables", {}).items():
            role = meta.get("role", "")
            if role:
                self.rag.add_documentation(
                    f"Table '{table}': {role}. "
                    f"Date column: {meta.get('date_column', 'none')}. "
                    f"Value column: {meta.get('value_column', 'none')}."
                )

        metrics = sia_context.get("metrics", {})
        if metrics:
            doc = "Business metrics:\n" + "\n".join(
                f"- {k}: {v}" for k, v in metrics.items() if v
            )
            self.rag.add_documentation(doc)

        joins = sia_context.get("joins", [])
        if joins:
            doc = "Table relationships:\n" + "\n".join(
                f"- {j['from']} joins to {j['to']} ({j['type']} JOIN)"
                for j in joins
            )
            self.rag.add_documentation(doc)

    def learn_successful_query(self, question: str, sql: str, verified: bool = False) -> None:
        try:
            self.rag.add_query_pair(question, sql, verified=verified)
        except Exception as e:
            import sys
            print(f"[knowledge] add_query_pair failed: {e}", file=sys.stderr)

    def learn_user_feedback(self, question: str, sql: str, is_correct: bool, correction: str | None = None) -> None:
        """A human judged the answer: right -> verified example; wrong -> the bad pair is
        dropped so it is never retrieved again, and the correction (if any) is verified."""
        if is_correct:
            self.learn_successful_query(question, sql, verified=True)
            return
        self.rag.delete_query_pair(question)
        if correction:
            self.learn_successful_query(question, correction, verified=True)

    # ═══════════════════════════════════════════════════════
    #  QUERY — get full context for a question
    # ═══════════════════════════════════════════════════════

    def get_context_for_question(self, question: str, db_type: str = "sql") -> dict:
        """
        Retrieval only — no LLM call. Returns:
          - Relevant DDL from ChromaDB
          - Similar past question→SQL pairs
          - Knowledge Graph join paths
        """
        similar_queries = []
        relevant_ddl = []
        relevant_docs = []

        # Retrieval failures degrade the answer silently, so at least say so on stderr.
        import sys
        try:
            similar_queries = self.rag.search_similar_queries(question, n_results=3)
        except Exception as e:
            print(f"[knowledge] search_similar_queries failed: {e}", file=sys.stderr)

        try:
            relevant_ddl = self.rag.search_ddl(question, n_results=5)
        except Exception as e:
            print(f"[knowledge] search_ddl failed: {e}", file=sys.stderr)

        try:
            relevant_docs = self.rag.search_docs(question, n_results=3)
        except Exception as e:
            print(f"[knowledge] search_docs failed: {e}", file=sys.stderr)

        tables_mentioned = self._extract_tables_from_context(
            relevant_ddl, similar_queries, question
        )
        join_paths = self.find_all_join_paths(tables_mentioned)

        related_tables = []
        for t in tables_mentioned:
            related_tables.extend(self.get_related_tables(t, depth=1))
        all_tables = list(set(tables_mentioned + related_tables))

        relevant_schema = {
            t: self._schema_dict[t]
            for t in all_tables
            if t in self._schema_dict
        }

        return {
            "relevant_schema": relevant_schema,
            "join_paths": join_paths,
            "similar_queries": similar_queries[:3],
            "relevant_ddl": relevant_ddl[:5],
            "relevant_docs": relevant_docs[:3],
            "tables_found": all_tables,
        }

    # ═══════════════════════════════════════════════════════
    #  GRAPH STATS
    # ═══════════════════════════════════════════════════════

    def get_graph_stats(self) -> dict:
        table_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "table"
        ]
        col_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "column"
        ]
        fk_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relation") == "foreign_key"
        ]
        rag_stats = self.rag.stats()
        return {
            "tables": len(table_nodes),
            "columns": len(col_nodes),
            "foreign_keys": len(fk_edges),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "rag_ddl": rag_stats["ddl_count"],
            "rag_docs": rag_stats["docs_count"],
            "rag_queries_learned": rag_stats["queries_count"],
        }

    def get_table_relationships(self, table: str) -> list[dict]:
        relationships = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "joins_to":
                if u == table or v == table:
                    relationships.append({
                        "from_table": u,
                        "to_table": v,
                        "from_column": data.get("via_from"),
                        "to_column": data.get("via_to"),
                    })
        return relationships

    # ═══════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _schema_to_ddl(table: str, columns: list, flavor: str) -> str:
        type_map = {
            "int": "INTEGER", "bigint": "BIGINT", "smallint": "SMALLINT",
            "varchar": "VARCHAR(255)", "text": "TEXT", "char": "CHAR(50)",
            "decimal": "DECIMAL(18,2)", "numeric": "NUMERIC(18,2)",
            "float": "FLOAT", "double": "DOUBLE", "real": "REAL",
            "datetime": "DATETIME", "timestamp": "TIMESTAMP",
            "date": "DATE", "time": "TIME",
            "boolean": "BOOLEAN", "bit": "BIT",
            "uuid": "UUID", "json": "JSON", "jsonb": "JSONB",
        }
        col_lines = []
        for col_def in columns:
            col_name = col_def.split(" ")[0]
            type_match = re.search(r"\((.+?)\)", col_def)
            raw_type = type_match.group(1).lower() if type_match else "text"
            base_type = raw_type.split("(")[0].strip()
            sql_type = type_map.get(base_type, raw_type.upper())
            col_lines.append(f"  {col_name} {sql_type}")

        return f"CREATE TABLE {table} (\n" + ",\n".join(col_lines) + "\n);"

    def _extract_tables_from_context(
        self, ddl_list: list, query_list: list, question: str
    ) -> list[str]:
        all_tables = set(self._schema_dict.keys())
        found = set()

        for ddl in ddl_list:
            match = re.search(r"CREATE TABLE\s+(\w+)", ddl, re.IGNORECASE)
            if match and match.group(1) in all_tables:
                found.add(match.group(1))

        for item in query_list:
            sql = item.get("sql", "") if isinstance(item, dict) else str(item)
            for tbl in all_tables:
                if re.search(rf"\b{re.escape(tbl)}\b", sql, re.IGNORECASE):
                    found.add(tbl)

        q_lower = question.lower()
        for tbl in all_tables:
            tbl_lower = tbl.lower().replace("_", " ")
            if tbl_lower in q_lower or tbl.lower() in q_lower:
                found.add(tbl)

        return list(found)
