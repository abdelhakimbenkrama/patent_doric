# Patent Search Tester (SerpApi · Google Patents + Groq)

A small FastAPI app with three web pages:

- **Keyword search** (`/`) tests SerpApi's `google_patents` engine directly.
- **Idea search** (`/idea`): describe an invention in plain language. An LLM (Groq) turns the description into features and Google Patents searches, then compares the closest patents with the idea feature by feature.
- **How it works** (`/how-it-works`): a user-facing explanation of the idea-search approach, costs, tips and limits.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Put your keys in `.env` (template: `.env.example`):

```
SERPAPI_API_KEY=your_real_key
GROQ_API_KEY=your_groq_key
PORT=8765
# optional
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TPM=8000
APP_PASSWORD=          # if set, every page and API call asks for this password
```

## Run

```powershell
python -m app.main                            # uses PORT from .env (default 8765)
# or: uvicorn app.main:app --port 8765
```

Then open http://127.0.0.1:8765 (keyword search) or http://127.0.0.1:8765/idea (idea search). Port 8000 is avoided because local Docker services use it.

## Deploy to Vercel

The repo deploys to Vercel as-is: Vercel detects the FastAPI app in `app/main.py`, installs `requirements.txt`, and `vercel.json` allows requests up to 300 seconds (the Hobby plan maximum; the slowest step takes about 2 minutes).

1. In Vercel, **Add New → Project** and import this GitHub repository. Keep the detected settings.
2. Under **Environment Variables**, add `SERPAPI_API_KEY`, `GROQ_API_KEY` and `APP_PASSWORD` (optionally `GROQ_MODEL`, `GROQ_TPM`).
3. Deploy. Opening the site shows the browser's login prompt: any username, and the `APP_PASSWORD` as password. Without `APP_PASSWORD`, anyone with the URL could spend your SerpApi credits and Groq tokens.

Every push to `main` redeploys. Differences from running locally:

- **Cache is not persistent.** Vercel only allows writing to `/tmp`, which is per server instance and wiped regularly, so repeated searches usually cost credits again. Persistent caching would need a store such as Upstash Redis.
- **Rate limiting is per instance.** If several people run searches at once, Groq may answer 429; the app waits and retries, so it only slows down.

## Idea search

1. **Analyze** (no SerpApi credits, 1 LLM call): the idea becomes a title, 3–7 key features, suggested CPC classes and 4 Google Patents queries. You can edit the features and queries.
2. **Find similar patents**:
   - The 4 queries run through SerpApi (1 credit each). A query that returns fewer than 5 results is retried once in a looser form (1 more credit).
   - Results are merged, and the LLM screens titles and snippets to pick the top N (default 10).
   - For those, SerpApi's `google_patents_details` fetches the abstract and first claim (1 credit each). The LLM compares 3 patents per call, marking each feature present, partial or absent with a quoted piece of evidence.
3. **Results**: a similarity score per patent, a feature matrix, which of your features no compared patent discloses, other candidates (each can be compared for about 1 more credit), and JSON export.

**Cost per idea:** about 14 SerpApi credits and 12–15k Groq tokens. It takes 2–4 minutes, because Groq's free tier allows about 8,000 tokens per minute and the app paces its calls to stay under that.

**Cache:** every SerpApi and Groq response is stored in `cache/`, so re-running the same idea or query costs nothing. Delete `cache/` to clear it.

**Limits:** this is a screening tool, not a legal novelty opinion. It only reads abstracts and first claims. It cannot see applications filed in the last ~18 months (not yet published) or non-patent prior art. In tests it found closely related patents but sometimes missed the best-known original patent in a field.

## Endpoints

| Endpoint | Purpose | SerpApi credits |
|---|---|---|
| `GET /api/account` | Checks the SerpApi key and shows plan and searches left | 0 |
| `GET /api/search?q=...` | Proxies a Google Patents search and returns `{summary, results, raw}` | 1 |
| `GET /api/llm/status` | Checks the Groq key and lists available models | 0 |
| `POST /api/idea/analyze` | `{idea}` → features, CPC codes, queries | 0 |
| `POST /api/idea/candidates` | `{idea, features, queries, filters, num, keep}` → merged, screened candidates | 1 per query (+1 per loosened retry) |
| `POST /api/idea/compare` | `{idea, features, patents[≤5]}` → per-patent similarity and feature evidence | 1 per patent |
| `GET /docs` | Interactive Swagger UI | — |

All credit costs drop to 0 when the response is already in `cache/`.

Keyword search parameters: `q`, `page`, `num` (10–100), `sort` (`new`/`old`), `before`/`after` (e.g. `priority:20200101`), `inventor`, `assignee`, `country` (e.g. `US,WO`), `language`, `status` (`GRANT`/`APPLICATION`), `type` (`PATENT`/`DESIGN`), `litigation` (`YES`/`NO`), `dups` (`language`), `scholar` (bool).

Example:

```powershell
curl "http://127.0.0.1:8765/api/search?q=coffee+machine&country=US&status=GRANT"
```
