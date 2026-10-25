"""The per-person document store: upload a file, ingest a URL, list what is indexed; plus the
synthesis prompt the hybrid answer uses."""

from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ...auth import get_current_user
from ...connections import TargetRefused
from ...docstore import DocStore
from ..deps import get_docstore
from ..schemas import HybridRequest

router = APIRouter()


SYNTHESIS_PROMPT = """\
You are an analytical assistant. You have been given:
1. A structured data result from a database query.
2. Relevant excerpts from documents (policies, manuals, reports, etc.).

Synthesize both into a clear, concise answer. Where numbers from the database
can be compared against rules or thresholds from the documents, do so explicitly.
Cite the SQL result and the document source. If no documents are available,
answer from the database result alone.

Keep the answer factual and auditable — show your reasoning.\
"""


def summarise_rows(columns: list, rows: list, limit: int = 10) -> str:
    if not rows:
        return "No rows returned."
    header = " | ".join(columns)
    lines = [header, "-" * len(header)]
    for row in rows[:limit]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more rows")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  UPLOAD DOCUMENT
# ─────────────────────────────────────────────────────────────


@router.post("/api/docs/upload")
async def upload_doc(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    data = await file.read()
    store = get_docstore(user_id)
    chunks = await asyncio.to_thread(store.add_file, file.filename, data)
    return {"success": True, "filename": file.filename, "chunks_indexed": chunks}


@router.post("/api/docs/ingest_url")
async def ingest_url(req: HybridRequest, user_id: str = Depends(get_current_user)):
    if not req.url:
        raise HTTPException(status_code=400, detail="url field is required.")
    store = get_docstore(user_id)
    chunks = await add_url(store, req.url)
    return {"success": True, "url": req.url, "chunks_indexed": chunks}


async def add_url(store: DocStore, url: str) -> int:
    """The web app is hosted: only public addresses, size-capped (see docstore.fetch_url)."""
    try:
        return await asyncio.to_thread(store.add_url, url, True)
    except (TargetRefused, ValueError, httpx.HTTPError) as e:
        raise HTTPException(status_code=400, detail={"message": f"Could not add that address: {e}"}) from e


@router.get("/api/docs")
async def list_docs(user_id: str = Depends(get_current_user)):
    store = get_docstore(user_id)
    return {
        "sources": await asyncio.to_thread(store.list_sources),
        "total_chunks": await asyncio.to_thread(store.count),
    }
