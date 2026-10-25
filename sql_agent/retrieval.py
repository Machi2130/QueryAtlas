"""Embeddings and schema retrieval: which tables and columns a question is about."""

from __future__ import annotations

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


class _OnnxEmbedder:
    """all-MiniLM-L6-v2 through chromadb's bundled ONNX runtime: the same
    384-dim L2-normalized vectors sentence-transformers produced, without the
    ~750 MB torch/transformers stack. Model file (~80 MB) downloads once."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        return np.asarray(self._ef(list(texts)), dtype=np.float32)


_EMBED_MODEL: _OnnxEmbedder | None = None


def _warm_embed_model() -> None:
    global _EMBED_MODEL
    _EMBED_MODEL = _OnnxEmbedder()


def _get_embed_model() -> _OnnxEmbedder:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = _OnnxEmbedder()
    return _EMBED_MODEL


# ═══════════════════════════════════════════════════════════════
#  EMBEDDING MODEL  (lazy-loaded once per process)
# ═══════════════════════════════════════════════════════════════


def _encode_texts(texts: list[str]) -> np.ndarray:
    """CPU-bound; callers run it in a worker thread (the model holds the GIL while it encodes)."""
    return _get_embed_model().encode(texts, normalize_embeddings=True).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  SCHEMA RETRIEVER  (50% semantic + 30% BM25 + 20% column signal)
# ═══════════════════════════════════════════════════════════════


class SchemaRetriever:
    COLUMN_SIGNALS = {
        "sales": ["total_amount", "sale_amount", "revenue", "net_amount"],
        "revenue": ["total_amount", "price", "amount", "net_sales"],
        "amount": ["total_amount", "price", "amt", "amount"],
        "stock": ["qty_on_hand", "stock_qty", "quantity", "available_qty"],
        "inventory": ["qty_on_hand", "stock_qty", "quantity", "reorder_level"],
        "customer": ["customer_id", "cust_id", "email", "phone"],
        "product": ["sku", "product_id", "barcode", "category"],
        "order": ["order_id", "order_date", "total", "status"],
        "date": ["order_date", "created_at", "updated_at", "invoice_date"],
        "tax": ["cgst", "sgst", "igst", "tax_amount"],
        "gst": ["cgst", "sgst", "igst", "hsn_code"],
        "purchase": ["po_id", "vendor_id", "purchase_date", "grn_id"],
        "employee": ["emp_id", "designation", "department", "salary"],
        "store": ["store_id", "branch_id", "store_name", "location"],
    }

    def __init__(self, schema_dict: dict):
        self.schema_dict = schema_dict
        self.table_names = list(schema_dict.keys())
        self._build_docs()
        self._build_faiss_index()
        self._build_bm25()

    def _build_docs(self):
        self.docs = []
        for tbl, cols in self.schema_dict.items():
            clean_cols = [c.split(" ")[0] for c in cols]
            readable = tbl.replace("_", " ")
            self.docs.append(f"{readable} {readable} {tbl} " + " ".join(clean_cols))

    def _build_faiss_index(self):
        model = _get_embed_model()
        embs = model.encode(self.docs, normalize_embeddings=True).astype(np.float32)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def _build_bm25(self):
        self.bm25 = BM25Okapi([doc.lower().split() for doc in self.docs])

    def _column_score(self, question: str, table_name: str) -> float:
        ql = question.lower()
        cols = [c.split(" ")[0].lower() for c in self.schema_dict[table_name]]
        score = sum(1.0 for kw, sigs in self.COLUMN_SIGNALS.items() if kw in ql and any(s in cols for s in sigs))
        return min(score, 3.0) / 3.0

    def retrieve(self, question: str, top_k: int = 5, db_type: str = "sql") -> dict:
        model = _get_embed_model()
        q_emb = model.encode([question], normalize_embeddings=True).astype(np.float32)
        dists, idxs = self.index.search(q_emb, len(self.table_names))
        sem_scores = {self.table_names[i]: float(dists[0][j]) for j, i in enumerate(idxs[0])}

        bm25_raw = self.bm25.get_scores(question.lower().split())
        bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_scores = {self.table_names[i]: float(bm25_raw[i]) / bm25_max for i in range(len(self.table_names))}

        hybrid = {
            tbl: (
                0.50 * sem_scores.get(tbl, 0.0)
                + 0.30 * bm25_scores.get(tbl, 0.0)
                + 0.20 * self._column_score(question, tbl)
            )
            for tbl in self.table_names
        }
        ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)

        selected = [t for t, s in ranked[:top_k] if s > 0.05] or [t for t, _ in ranked[:5]]
        return {t: self.schema_dict[t] for t in selected}


# ═══════════════════════════════════════════════════════════════
#  COLUMN COMPRESSION
# ═══════════════════════════════════════════════════════════════

_NOISE_COLS = {
    "created_by",
    "updated_by",
    "deleted_by",
    "created_at",
    "updated_at",
    "deleted_at",
    "is_deleted",
    "is_active",
    "row_version",
    "last_modified_by",
    "modified_at",
}


def compress_schema(matched: dict, question: str, max_cols: int = 12) -> dict:
    model = _get_embed_model()
    q_emb = model.encode([question.lower()], normalize_embeddings=True).astype(np.float32)
    result = {}
    for tbl, cols in matched.items():
        clean = [c for c in cols if c.split(" ")[0].lower() not in _NOISE_COLS]
        if len(clean) <= max_cols:
            result[tbl] = clean
            continue
        col_names = [c.split(" ")[0].replace("_", " ") for c in clean]
        col_embs = model.encode(col_names, normalize_embeddings=True).astype(np.float32)
        sims = (col_embs @ q_emb.T).flatten()
        result[tbl] = [c for _, c in sorted(zip(sims, clean, strict=False), reverse=True)][:max_cols]
    return result


def build_schema_str(matched: dict, db_type: str) -> str:
    label = "Collection" if db_type == "mongo" else "Table"
    field = "Fields" if db_type == "mongo" else "Columns"
    return "\n".join(f"{label}: {name}\n  {field}: {', '.join(cols)}" for name, cols in matched.items())


def rank_columns(candidates: dict[str, list[str]], question: str, max_cols: int) -> dict[str, list[str]]:
    """Per table, the max_cols column names most similar to the question (embeddings), kept
    in table order. Unlike compress_schema nothing is dropped as 'noise': created_at may be
    the only date a question about sign-ups can use."""
    model = _get_embed_model()
    q_emb = model.encode([question.lower()], normalize_embeddings=True).astype(np.float32)
    out: dict[str, list[str]] = {}
    for table, cols in candidates.items():
        if len(cols) <= max_cols:
            out[table] = list(cols)
            continue
        embs = model.encode([c.replace("_", " ") for c in cols], normalize_embeddings=True).astype(np.float32)
        keep = set(np.argsort(-(embs @ q_emb.T).flatten())[:max_cols].tolist())
        out[table] = [c for i, c in enumerate(cols) if i in keep]
    return out
