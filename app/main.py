"""FastAPI proxy for testing SerpApi's Google Patents engine, plus the LLM-assisted idea search.

API keys stay server-side (loaded from .env); the browser only talks to /api/*.
"""

import base64
import binascii
import os
import secrets
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import ENV_FILE, STATIC_DIR, get_env
from .idea_search import router as idea_router
from .serpapi import ACCOUNT_URL, SEARCH_URL, call_serpapi, get_api_key, normalize_result

app = FastAPI(title="Patent Search Tester")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(idea_router)


def _password_ok(header: str, password: str) -> bool:
    if not header.startswith("Basic "):
        return False
    try:
        _, _, given = base64.b64decode(header[6:]).decode().partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    return secrets.compare_digest(given.encode(), password.encode())


@app.middleware("http")
async def require_password(request: Request, call_next):
    # When APP_PASSWORD is set (e.g. on Vercel), every page and API call needs it; any username works.
    # Unset locally, so the app stays open on 127.0.0.1.
    password = get_env("APP_PASSWORD")
    if password and not _password_ok(request.headers.get("authorization", ""), password):
        return PlainTextResponse(
            "Password required.", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Patent search"'}
        )
    return await call_next(request)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/idea", include_in_schema=False)
async def idea_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "idea.html")


@app.get("/how-it-works", include_in_schema=False)
async def how_it_works_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "how-it-works.html")


@app.get("/api/account")
async def account() -> dict[str, Any]:
    """Key validity + remaining quota. Does not consume a search."""
    data = await call_serpapi(ACCOUNT_URL, {"api_key": get_api_key()})
    return {
        "plan_name": data.get("plan_name"),
        "searches_per_month": data.get("searches_per_month"),
        "this_month_usage": data.get("this_month_usage"),
        "plan_searches_left": data.get("plan_searches_left"),
        "total_searches_left": data.get("total_searches_left"),
        "account_rate_limit_per_hour": data.get("account_rate_limit_per_hour"),
    }


@app.get("/api/search")
async def search(
    q: str | None = Query(None, description="Keywords, e.g. (Coffee) OR (Tea);(A47J)"),
    page: int = Query(1, ge=1),
    num: int = Query(10, ge=10, le=100),
    sort: str | None = Query(None, pattern="^(new|old)$"),
    before: str | None = Query(None, description="e.g. priority:20231231"),
    after: str | None = Query(None, description="e.g. publication:20200101"),
    inventor: str | None = None,
    assignee: str | None = None,
    country: str | None = Query(None, description="Comma-separated, e.g. US,WO"),
    language: str | None = Query(None, description="Comma-separated, e.g. ENGLISH,GERMAN"),
    status: str | None = Query(None, pattern="^(GRANT|APPLICATION)$"),
    type: str | None = Query(None, pattern="^(PATENT|DESIGN)$"),
    litigation: str | None = Query(None, pattern="^(YES|NO)$"),
    dups: str | None = Query(None, pattern="^(language)$"),
    scholar: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "engine": "google_patents",
        "api_key": get_api_key(),
        "page": page,
        "num": num,
    }
    optional = {
        "q": q, "sort": sort, "before": before, "after": after,
        "inventor": inventor, "assignee": assignee, "country": country,
        "language": language, "status": status, "type": type,
        "litigation": litigation, "dups": dups,
    }
    params.update({k: v.strip() for k, v in optional.items() if v and v.strip()})
    if scholar:
        params["scholar"] = "true"

    data = await call_serpapi(SEARCH_URL, params)

    meta = data.get("search_metadata", {})
    info = data.get("search_information", {})
    return {
        "summary": {
            "total_results": info.get("total_results"),
            "page": page,
            "num": num,
            "search_id": meta.get("id"),
            "status": meta.get("status"),
            "total_time_taken": meta.get("total_time_taken"),
            # SerpApi returns 200 + "error" for things like "no results".
            "error": data.get("error"),
        },
        "results": [normalize_result(r) for r in data.get("organic_results", [])],
        "raw": data,
    }


if __name__ == "__main__":
    import uvicorn

    load_dotenv(ENV_FILE)
    # Default avoids 8000, which other local Docker services use.
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=True)
