"""SerpApi client: Google Patents search, patent details and account info."""

from typing import Any

import httpx
from fastapi import HTTPException

from . import cache
from .config import require_key

SEARCH_URL = "https://serpapi.com/search.json"
ACCOUNT_URL = "https://serpapi.com/account.json"
TIMEOUT = httpx.Timeout(60.0)


def get_api_key() -> str:
    return require_key("SERPAPI_API_KEY")


async def call_serpapi(url: str, params: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            resp = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Could not reach SerpApi: {exc}") from exc

    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail=f"SerpApi returned non-JSON (HTTP {resp.status_code}).")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=data.get("error") or f"SerpApi error (HTTP {resp.status_code}).",
        )
    return data


async def cached_call(params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """search.json with the disk cache. Returns (data, from_cache); a cache hit costs no credit."""
    key = dict(sorted(params.items()))
    hit = cache.get("serpapi", key)
    if hit is not None:
        return hit, True
    data = await call_serpapi(SEARCH_URL, {**params, "api_key": get_api_key()})
    # Don't cache failures (SerpApi can answer 200 with an "error" field).
    if not data.get("error"):
        cache.put("serpapi", key, data)
    return data, False


async def patents_search(params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    return await cached_call({"engine": "google_patents", **params})


async def patent_details(patent_id: str) -> tuple[dict[str, Any], bool]:
    """patent_id as returned by search, e.g. "patent/US11734097B1/en"."""
    return await cached_call({"engine": "google_patents_details", "patent_id": patent_id})


def normalize_result(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": r.get("position"),
        "patent_id": r.get("patent_id"),
        "publication_number": r.get("publication_number"),
        "title": r.get("title"),
        "snippet": r.get("snippet"),
        "priority_date": r.get("priority_date"),
        "filing_date": r.get("filing_date"),
        "grant_date": r.get("grant_date"),
        "publication_date": r.get("publication_date"),
        "inventor": r.get("inventor"),
        "assignee": r.get("assignee"),
        "pdf": r.get("pdf"),
        "thumbnail": r.get("thumbnail"),
        "patent_link": r.get("patent_link"),
        "country_status": r.get("country_status"),
        "language": r.get("language"),
    }
