"""Groq client (OpenAI-compatible chat completions) returning schema-checked JSON.

Groq's free tier allows ~8,000 tokens per minute, so calls are paced through a
60-second sliding window (GROQ_TPM) and retried on 429 using the retry-after header.
"""

import asyncio
import json
import time
from collections import deque
from typing import Any

import httpx
from fastapi import HTTPException

from . import cache
from .config import get_env, require_key

CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS_URL = "https://api.groq.com/openai/v1/models"
DEFAULT_MODEL = "openai/gpt-oss-120b"
TIMEOUT = httpx.Timeout(120.0)
MAX_RETRIES = 4

# Entries are [timestamp, tokens]; tokens start as an estimate and are corrected after the call.
_window: deque[list[float]] = deque()
_lock = asyncio.Lock()


def get_model() -> str:
    return get_env("GROQ_MODEL") or DEFAULT_MODEL


async def _reserve(estimate: int) -> list[float]:
    limit = int(get_env("GROQ_TPM") or 8000)
    async with _lock:
        while True:
            now = time.monotonic()
            while _window and now - _window[0][0] >= 60:
                _window.popleft()
            used = sum(tokens for _, tokens in _window)
            if not _window or used + estimate <= limit:
                break
            await asyncio.sleep(60 - (now - _window[0][0]) + 0.5)
        entry = [now, float(estimate)]
        _window.append(entry)
        return entry


def _groq_error(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return err.get("message") or json.dumps(err)
    except ValueError:
        return resp.text[:500] or f"HTTP {resp.status_code}"


async def chat_json(
    system: str,
    user: str,
    schema_name: str,
    schema: dict[str, Any],
    *,
    max_tokens: int = 2000,
    effort: str = "medium",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (parsed JSON, meta) where meta = {model, cached, tokens}."""
    model = get_model()
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        "temperature": 0.2,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": effort,
        "include_reasoning": False,
    }
    hit = cache.get("groq", body)
    if hit is not None:
        return hit, {"model": model, "cached": True, "tokens": 0}

    headers = {"Authorization": f"Bearer {require_key('GROQ_API_KEY')}"}
    # Rough estimate (~3.5 chars per token) plus half the output allowance; replies rarely use it all.
    entry = await _reserve(int((len(system) + len(user)) / 3.5) + max_tokens // 2)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await client.post(CHAT_URL, json=body, headers=headers)
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"Could not reach Groq: {exc}") from exc
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                wait = float(resp.headers.get("retry-after") or 10)
                await asyncio.sleep(min(max(wait, 1.0), 60.0))
                continue
            break

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Groq: {_groq_error(resp)}")

    data = resp.json()
    usage = data.get("usage") or {}
    entry[1] = float(usage.get("total_tokens") or entry[1])
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise HTTPException(status_code=502, detail=f"Groq reply was cut off at {max_tokens} tokens.")
    try:
        parsed = json.loads(choice["message"]["content"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Groq returned invalid JSON.") from exc

    cache.put("groq", body, parsed)
    return parsed, {"model": model, "cached": False, "tokens": usage.get("total_tokens", 0)}


async def status() -> dict[str, Any]:
    """Checks the key by listing models. Free: does not use tokens."""
    headers = {"Authorization": f"Bearer {require_key('GROQ_API_KEY')}"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        try:
            resp = await client.get(MODELS_URL, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Could not reach Groq: {exc}") from exc
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Groq: {_groq_error(resp)}")
    models = sorted(m["id"] for m in resp.json().get("data", []))
    model = get_model()
    return {"model": model, "model_available": model in models, "models": models}
