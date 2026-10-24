"""LLM client for any OpenAI-compatible chat API (Groq by default)."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx

# ═══════════════════════════════════════════════════════════════
#  LLM CLIENT — any OpenAI-compatible chat API
#  Defaults reproduce the original Groq setup. Point LLM_BASE_URL at OpenAI,
#  Ollama (http://localhost:11434/v1), vLLM, LM Studio or a fine-tuned model to
#  switch provider; nothing else changes.
# ═══════════════════════════════════════════════════════════════

_GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
LLM_BASE_URL         = os.getenv("LLM_BASE_URL", _GROQ_BASE_URL).rstrip("/")
LLM_MODEL            = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# reasoning_effort is a Groq/OpenAI reasoning-model field; other models reject or ignore it.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "" if os.getenv("LLM_MODEL") else "low")


RETRY_STATUSES = {429, 503}     # rate limited / overloaded: worth waiting for
RETRY_DELAYS   = (2.0, 4.0)     # seconds, when the provider gives no Retry-After
MAX_RETRY_WAIT = 20.0           # never hold a user's request longer than this per retry


PROVIDER_PROBLEMS = {401: "The provider rejected the API key.", 403: "The provider rejected the API key.",
                     404: "The provider does not know this model.",
                     429: "The provider says this key is out of quota or going too fast."}


def provider_problem(status: int) -> str:
    """A plain sentence for a provider failure; the provider's own text is never shown."""
    return PROVIDER_PROBLEMS.get(status, "The provider could not be reached or returned an error.")


class LLMError(RuntimeError):
    """The LLM endpoint answered with an error (status and body, never the key)."""
    status: int = 0


def llm_api_key(explicit: str = "") -> str:
    return explicit or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")


def llm_available(api_key: str) -> bool:
    """Hosted APIs need a key; a self-hosted endpoint (Ollama, vLLM) may not."""
    return bool(api_key) or LLM_BASE_URL != _GROQ_BASE_URL


# Providers a person can pick without anyone editing code. The addresses are fixed, so a user
# cannot make the server call an address of their choosing; "custom" needs the settings permission.
PROVIDERS = {
    "groq":       {"label": "Groq",                   "base_url": "https://api.groq.com/openai/v1",  "needs_key": True},
    "openai":     {"label": "OpenAI",                 "base_url": "https://api.openai.com/v1",       "needs_key": True},
    "openrouter": {"label": "OpenRouter",             "base_url": "https://openrouter.ai/api/v1",    "needs_key": True},
    "ollama":     {"label": "Ollama (on the server)", "base_url": "http://localhost:11434/v1",       "needs_key": False},
}


@dataclass
class LLMSettings:
    """One person's AI choice. Kept in memory only: never written to disk, a cache, Redis or a log."""
    provider: str = "groq"
    api_key: str = field(default="", repr=False)
    base_url: str = ""          # used only when provider == "custom"
    model: str = ""             # empty = the server's default model
    planner_model: str = ""     # optional: a different model for choosing tables

    @property
    def url(self) -> str:
        return self.base_url if self.provider == "custom" else PROVIDERS[self.provider]["base_url"]

    def check(self, allow_custom: bool) -> None:
        """ValueError, in plain words, when this choice cannot be used."""
        if self.provider == "custom":
            if not allow_custom:
                raise ValueError("Only an administrator can use a custom address.")
            if not self.base_url.startswith(("http://", "https://")):
                raise ValueError("The address must start with http:// or https://")
        elif self.provider not in PROVIDERS:
            raise ValueError("Unknown provider.")
        elif PROVIDERS[self.provider]["needs_key"] and not self.api_key:
            raise ValueError(f"{PROVIDERS[self.provider]['label']} needs an API key.")
        if not self.model and self.provider != "groq":   # only Groq has the server's default model
            raise ValueError("Choose a model.")


def list_models(url: str, api_key: str = "", timeout: float = 15.0) -> list[str]:
    """The model names a provider offers (OpenAI-compatible GET /models). The reply body is never
    quoted in an error: a provider may echo part of the key."""
    r = httpx.get(f"{url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                  timeout=timeout)
    if r.status_code >= 400:
        err = LLMError(f"The provider answered {r.status_code}")
        err.status = r.status_code
        raise err
    return sorted({m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")})


class LLMClient:
    """The slice of the OpenAI/Groq SDK interface this code uses:
    client.chat.completions.create(...) -> .choices[0].message.content and .usage,
    or with stream=True an iterator of chunks with .choices[0].delta.content.
    `model` / `planner_model` are this client's defaults (a person's own choice); call sites
    do not name a model. `reasoning_effort=""` leaves the field out (not every model accepts it)."""

    def __init__(self, api_key: str = "", base_url: str = "", timeout: float = 60.0, model: str = "",
                 planner_model: str = "", reasoning_effort: str | None = None):
        self._url     = f"{(base_url or LLM_BASE_URL).rstrip('/')}/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._secret  = api_key
        self._timeout = timeout
        self._effort  = LLM_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
        self.model, self.planner_model = model or LLM_MODEL, planner_model
        self.chat     = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, messages: list, model: str = "", temperature: float = 0.1,
                max_completion_tokens: int = 512, stream: bool = False, **_ignored):
        body = {"model": model or self.model, "messages": messages, "temperature": temperature,
                "max_completion_tokens": max_completion_tokens, "stream": stream}
        if self._effort:
            body["reasoning_effort"] = self._effort
        if stream:
            return self._stream(body)
        for attempt in range(len(RETRY_DELAYS) + 1):
            r = httpx.post(self._url, json=body, headers=self._headers, timeout=self._timeout)
            if r.status_code not in RETRY_STATUSES or attempt == len(RETRY_DELAYS):
                break
            self._wait(r, attempt)
        if r.status_code >= 400:
            raise self._error(r.status_code, r.text)
        data = r.json()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=c["message"].get("content") or ""))
                     for c in data["choices"]],
            usage=SimpleNamespace(**data["usage"]) if data.get("usage") else None,
        )

    @staticmethod
    def _wait(r: httpx.Response, attempt: int) -> None:
        """Rate-limited or overloaded: wait as long as the provider asks, else back off."""
        try:
            delay = min(float(r.headers.get("retry-after", "")), MAX_RETRY_WAIT)
        except ValueError:
            delay = RETRY_DELAYS[attempt]
        print(f"[llm] {r.status_code}, retrying in {delay:.0f}s", file=sys.stderr)
        time.sleep(delay)

    def _error(self, status: int, body: str) -> LLMError:
        # The provider's reply stays on the server: it can echo the key (encoded or partly masked) or the
        # request. Every caller passes str(err) on to a client, so the message is a fixed sentence.
        if self._secret:
            body = body.replace(self._secret, "<API_KEY>")
        print(f"[llm] {status}: {body[:300]}", file=sys.stderr)
        err = LLMError(f"LLM {status}: {provider_problem(status)}")
        err.status = status
        return err

    def _stream(self, body: dict):
        # Retries happen only before the first token: a stream half-delivered is not restarted.
        for attempt in range(len(RETRY_DELAYS) + 1):
            with httpx.stream("POST", self._url, json=body, headers=self._headers, timeout=self._timeout) as r:
                if r.status_code in RETRY_STATUSES and attempt < len(RETRY_DELAYS):
                    r.read()
                    self._wait(r, attempt)
                    continue
                if r.status_code >= 400:
                    raise self._error(r.status_code, r.read().decode(errors="replace"))
                yield from self._chunks(r)
                return

    @staticmethod
    def _chunks(r: httpx.Response):
        for line in r.iter_lines():
            payload = line[5:].strip() if line.startswith("data:") else ""
            if payload == "[DONE]":
                return
            if payload:
                choices = json.loads(payload).get("choices") or [{}]
                content = (choices[0].get("delta") or {}).get("content")
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])
