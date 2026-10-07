# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small FastAPI app with two self-contained web pages: a keyword tester for SerpApi's `google_patents` engine and the USPTO Open Data Portal (`/`, source switch), and an LLM-assisted "idea search" (`/idea`) that turns a plain-language invention description into Google Patents and USPTO searches and compares the hits feature by feature using Groq. Google Patents and USPTO results are deliberately kept separate (separate endpoints, separate tabs; never merged). The backend is a thin proxy so the API keys never reach the browser.

## Commands (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main                   # UI at http://127.0.0.1:8765 and /idea, Swagger at /docs
# or: uvicorn app.main:app --port 8765
```

Don't use port 8000: Docker services on this machine use it (plus 5100, 8101, 6380, 5433, 9000–9001). The port comes from `PORT` in `.env` (default 8765) when run via `python -m app.main`; plain `uvicorn` ignores it, so pass `--port` explicitly.

`--reload` (also on by default in `python -m app.main`) is unreliable on this machine: after a code change the reloader can hang while an orphaned `multiprocessing` worker keeps port 8765 and serves the old code. If responses look stale, stop every python process listening on 8765 and start without `--reload`.

There are no tests, linter, or build step configured.

Deployment: Vercel, from the GitHub repo `abdelhakimbenkrama/patent_doric` (branch `main`). Vercel auto-detects the FastAPI app at `app/main.py`; `vercel.json` sets `maxDuration: 300` for it. Keys are Vercel environment variables, not `.env`.

## Configuration

- `.env` at the repo root (template: `.env.example`): `SERPAPI_API_KEY`, `GROQ_API_KEY`, `USPTO_API_KEY` (free ODP key from data.uspto.gov), `PORT`, optional `GROQ_MODEL` (default `openai/gpt-oss-120b`) and `GROQ_TPM` (default 8000, the Groq free-tier tokens-per-minute).
- `APP_PASSWORD` (optional): when set, a middleware in `app/main.py` requires HTTP Basic auth (any username) on every route, including `/static` and `/api`. Unset locally; set on Vercel.
- `CACHE_DIR` (optional): defaults to `cache/` locally and `/tmp/patent-cache` when `VERCEL` is set (Vercel's filesystem is read-only except `/tmp`). Cache writes are best-effort and never fail a request.
- `app/config.py` `get_env()` re-reads `.env` on **every call** (`override=True`), so key changes take effect without restarting. `require_key()` treats the placeholder `your_key_here` as unset.
- The SerpApi account is on the **Free plan (250 searches/month)**. An idea search costs ~14 credits; be sparing when testing.

## Architecture

- `app/main.py`: FastAPI app, the keyword-search endpoints `/api/account` and `/api/search` (both unchanged from the original tester, uncached), the USPTO tester endpoints `/api/uspto/status` and `/api/uspto/search`, page routes `/` and `/idea`, and it includes the idea-search router.
- `app/serpapi.py`: `call_serpapi()` turns network failures and non-JSON replies into 502s and passes SerpApi's non-200 status and `error` message through as an `HTTPException`. `cached_call()` wraps it with the disk cache (cache key excludes the API key; responses with an `error` field are not cached). `normalize_result()` is the fixed whitelist of result fields. `account.json` costs no credit; `search.json` (both `google_patents` and `google_patents_details`) costs one unless cached.
- `app/llm.py`: Groq chat completions with `response_format: json_schema` and `strict: true`. All schema fields must be required and every object needs `additionalProperties: false`. Calls are paced through an in-process 60 s sliding window (`GROQ_TPM`), retried on 429 using `retry-after`, and cached by full request body. `reasoning_effort` is set per call; `include_reasoning: false`.
- `app/uspto.py`: USPTO Open Data Portal client (`https://api.uspto.gov`, `X-API-KEY` header, free). `POST /api/v1/patent/applications/search` takes an OpenSearch query-string `q` plus `filters`/`rangeFilters`/`sort`/`fields`/`pagination`; it only covers bibliographic data (title, CPC, applicant, dates), never abstract or claims, and a bare word matches every field (including addresses). A 404 means "no matching records" and is turned into an empty result. `expand_query()` maps `TI=`/`CPC=`/`APP=`/`INV=` to `applicationMetaData.*` field names. `full_text()` gets abstract + first claim from `associated-documents` → per-application grant XML (else pre-grant publication XML) under `/api/v1/datasets/products/files/`, parsed with ElementTree. Searches are cached for idea search only (`uspto` namespace); full text in `uspto-text`. The keyword tester's `/api/uspto/search` is uncached. Field names and response shapes came from the ODP OpenAPI spec (mirrored in github.com/patent-dev/uspto-odp, which also has real recorded responses); data.uspto.gov itself blocks scripted access.
- `app/cache.py`: JSON files in `cache/<namespace>/<sha256>.json`, never expire.
- `app/idea_search.py`: the three stateless steps (the UI passes each step's output to the next):
  - `analyze`: one LLM call that produces features (ids `F1..Fn` are assigned server-side) and 4 queries.
  - `candidates`: runs the queries in parallel. A query with fewer than `MIN_RESULTS` results is retried once via `_relax()`. Results are merged by publication number and screened by the LLM in chunks of 25.
  - `compare`: up to 5 patents per call. It fetches details, takes the first claim from `claims_translated` (English) when present and skips header lines like "CLAIMS". When the abstract is missing it falls back to the search snippet.
  - `_plain()` converts typographic hyphens and quotes from LLM output to ASCII, because Google Patents doesn't match them.
  - `analyze` also returns 3 `uspto_queries` (title-only, `TI=(…) AND TI=(…)`). `/api/idea/uspto/candidates` runs them (utility applications only, filter `before`/`status` mapped to ODP fields, `country`/`language` ignored), relaxes a query with fewer than `MIN_RESULTS` hits by dropping its last top-level AND group, merges by application number and screens on title + CPC. `/api/idea/uspto/compare` takes `application_number`s and compares on the USPTO full text. Screening and comparison share `_screen()` / `_compare_llm()` with the Google path, whose prompts must stay byte-identical so existing Groq cache entries keep hitting.
- Prompt notes, learned from testing: Google Patents full-text matching is very noisy, so queries need `AB=`/`TI=`/`CL=` field prefixes, quoted phrases, short truncated terms and 2 groups. LLM-guessed CPC codes were usually wrong, so they are only displayed, never searched.
- `static/how-it-works.html` (served at `/how-it-works`) is the user-facing explanation of the approach; it quotes prompt rules, score bands, costs and test results, so update it when those change.
- `static/index.html` and `static/idea.html` are self-contained pages (inline CSS + vanilla JS, no build). `idea.html` runs the Google and USPTO candidate searches in parallel, then compares each source in batches of 3 and re-renders after each batch; results are in two tabs (`SRC` in the script holds the per-source URLs, payloads and texts). The `static/` dir is also mounted at `/static`.
