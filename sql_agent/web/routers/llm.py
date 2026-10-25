"""Each person's own AI: provider, API key and model, kept in memory only and never returned."""

from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException

from ... import rbac
from ...auth import get_current_user
from ...llm import LLM_MODEL, PROVIDERS, LLMError, LLMSettings, llm_api_key, llm_available, provider_problem
from ..deps import client_from, require
from ..runtime import wrt
from ..schemas import LLMRequest

router = APIRouter()


def llm_view(user_id: str) -> dict:
    mine = wrt.llm_settings.get(user_id)
    server_ai = llm_available(llm_api_key())
    return {
        "source": "yours" if mine else "server" if server_ai else "none",
        "provider": mine.provider if mine else "",
        "model": mine.model if mine else "",
        "planner_model": mine.planner_model if mine else "",
        "base_url": mine.base_url if mine and mine.provider == "custom" else "",
        "has_key": bool(mine and mine.api_key),
        "server_model": LLM_MODEL if server_ai else "",
        "providers": [{"id": k, "label": v["label"], "needs_key": v["needs_key"]} for k, v in PROVIDERS.items()],
        "can_custom": wrt.policy.can(user_id, rbac.SETTINGS),
    }


def llm_choice(user_id: str, req: LLMRequest) -> LLMSettings:
    """The validated choice. A blank key keeps the one already held for the same provider, so the
    page never needs the key back."""
    held = wrt.llm_settings.get(user_id)
    key = req.api_key.strip() or (held.api_key if held and held.provider == req.provider.strip() else "")
    choice = LLMSettings(
        provider=req.provider.strip(),
        api_key=key,
        base_url=req.base_url.strip(),
        model=req.model.strip(),
        planner_model=req.planner_model.strip(),
    )
    try:
        choice.check(wrt.policy.can(user_id, rbac.SETTINGS))
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"message": str(e)}) from e
    return choice


def provider_message(e: Exception) -> str:
    """A plain sentence for a provider failure; the provider's own text is never shown (it may echo a key)."""
    return provider_problem(getattr(e, "status", 0))


@router.get("/api/llm")
async def get_llm(user_id: str = Depends(get_current_user)):
    require(user_id, rbac.CONNECT)
    return llm_view(user_id)


@router.put("/api/llm")
async def put_llm(req: LLMRequest, user_id: str = Depends(get_current_user)):
    require(user_id, rbac.CONNECT)
    wrt.llm_settings[user_id] = llm_choice(user_id, req)
    return llm_view(user_id)


@router.delete("/api/llm")
async def delete_llm(user_id: str = Depends(get_current_user)):
    require(user_id, rbac.CONNECT)
    wrt.llm_settings.pop(user_id, None)
    return llm_view(user_id)


@router.post("/api/llm/models")
async def llm_models(req: LLMRequest, user_id: str = Depends(get_current_user)):
    """The models the chosen provider offers, for the page to list. Nothing is saved."""
    require(user_id, rbac.CONNECT)
    choice = llm_choice(user_id, req.model_copy(update={"model": req.model or "x"}))
    try:
        return {"models": await asyncio.to_thread(wrt.list_models, choice.url, choice.api_key)}
    except (LLMError, httpx.HTTPError) as e:
        raise HTTPException(status_code=400, detail={"message": provider_message(e)}) from e


@router.post("/api/llm/test")
async def llm_test(req: LLMRequest, user_id: str = Depends(get_current_user)):
    """One tiny request with the chosen key and model, so a wrong key or model shows now, not on a question."""
    require(user_id, rbac.CONNECT)
    choice = llm_choice(user_id, req)
    client = client_from(choice)
    started = time.monotonic()
    try:
        await asyncio.to_thread(
            lambda: client.chat.completions.create(
                messages=[{"role": "user", "content": "Reply with the word OK."}],
                temperature=0,
                max_completion_tokens=8,
            )
        )
    except (LLMError, httpx.HTTPError) as e:
        raise HTTPException(status_code=400, detail={"message": provider_message(e)}) from e
    return {"ok": True, "model": getattr(client, "model", choice.model), "ms": int((time.monotonic() - started) * 1000)}
