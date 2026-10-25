"""The per-user document store: index a file or a URL, search it, list what is in it."""

from __future__ import annotations

import asyncio

from mcp import types

from ..context import get_docstore, resolve_sid
from ..results import err, ok
from ..runtime import rt


async def upload_document(args: dict) -> list[types.TextContent]:
    import base64

    sid = resolve_sid(args)
    store = get_docstore(sid)

    url = args.get("url", "").strip()
    if url:
        try:
            chunks = await asyncio.to_thread(store.add_url, url, rt.hosted)
            return ok({"indexed": True, "source": url, "chunks_indexed": chunks})
        except Exception as e:
            return err(f"Failed to fetch URL: {e}")

    filename = args.get("filename", "document.txt")
    content_b64 = args.get("content_b64", "")
    if not content_b64:
        return err("Provide either content_b64 (base64 file bytes) or url.")
    try:
        data = base64.b64decode(content_b64)
        chunks = await asyncio.to_thread(store.add_file, filename, data)
        return ok({"indexed": True, "source": filename, "chunks_indexed": chunks})
    except Exception as e:
        return err(f"Failed to index document: {e}")


async def search_documents(args: dict) -> list[types.TextContent]:
    sid = resolve_sid(args)
    question = args["question"]
    n = min(10, max(1, int(args.get("n_results", 4))))
    store = get_docstore(sid)

    if store.count() == 0:
        return ok(
            {
                "chunks": [],
                "hint": "No documents indexed yet. Call upload_document first.",
            }
        )

    hits = await asyncio.to_thread(store.search, question, n)
    return ok({"chunks": hits, "total_chunks_in_store": store.count()})


async def list_documents(args: dict) -> list[types.TextContent]:
    sid = resolve_sid(args)
    store = get_docstore(sid)
    sources = await asyncio.to_thread(store.list_sources)
    return ok({"sources": sources, "total_chunks": store.count()})
