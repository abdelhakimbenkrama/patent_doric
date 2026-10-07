"""Idea search: plain-language invention idea -> similar patents, compared feature by feature.

Three stateless steps (the UI passes each step's output to the next):
  1. /api/idea/analyze     LLM splits the idea into features and writes Google Patents and USPTO queries.
  2. /api/idea/candidates  Runs the queries through SerpApi, merges them, LLM screens -> top N.
  3. /api/idea/compare     Fetches abstract + first claim for a few patents, LLM compares per feature.

Steps 2 and 3 also exist for the USPTO Open Data Portal (/api/idea/uspto/...). Its results are kept
separate from Google Patents: ODP only searches titles and bibliographic data, and the text for
the comparison comes from the USPTO's own grant / publication XML.
"""

import asyncio
import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from . import llm, serpapi, uspto

router = APIRouter(prefix="/api")

SCREEN_CHUNK = 25          # candidates per screening call
COMPARE_MAX = 5            # patents per compare call (UI sends 3)
ABSTRACT_CHARS = 1200
CLAIM_CHARS = 1500
SNIPPET_CHARS = 300
MIN_RESULTS = 5            # a query returning fewer is retried once in a looser form


# ---------- request models ----------

class Feature(BaseModel):
    id: str = Field(max_length=10)
    text: str = Field(min_length=1, max_length=400)
    terms: list[str] = Field(default_factory=list, max_length=12)


class Filters(BaseModel):
    country: str | None = Field(None, description="Comma-separated, e.g. US,EP,WO")
    before: str | None = Field(None, pattern=r"^(priority|filing|publication):\d{8}$")
    after: str | None = Field(None, pattern=r"^(priority|filing|publication):\d{8}$")
    status: str | None = Field(None, pattern="^(GRANT|APPLICATION)$")
    language: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def blank_to_none(cls, v: Any) -> Any:
        return v.strip() or None if isinstance(v, str) else v


class AnalyzeIn(BaseModel):
    idea: str = Field(min_length=20, max_length=4000)


class CandidatesIn(BaseModel):
    idea: str = Field(min_length=20, max_length=4000)
    features: list[Feature] = Field(min_length=1, max_length=12)
    queries: list[str] = Field(min_length=1, max_length=6)
    filters: Filters = Field(default_factory=Filters)
    num: int = Field(20, ge=10, le=100)
    keep: int = Field(10, ge=1, le=30)


class ComparePatent(BaseModel):
    patent_id: str
    publication_number: str | None = None
    title: str | None = None
    snippet: str | None = None


class CompareIn(BaseModel):
    idea: str = Field(min_length=20, max_length=4000)
    features: list[Feature] = Field(min_length=1, max_length=12)
    patents: list[ComparePatent] = Field(min_length=1, max_length=COMPARE_MAX)


class UsptoComparePatent(BaseModel):
    application_number: str = Field(pattern=r"^[\d/,]{6,16}$")
    publication_number: str | None = None
    title: str | None = None


class UsptoCompareIn(BaseModel):
    idea: str = Field(min_length=20, max_length=4000)
    features: list[Feature] = Field(min_length=1, max_length=12)
    patents: list[UsptoComparePatent] = Field(min_length=1, max_length=COMPARE_MAX)


# ---------- JSON schemas for Groq strict mode (every field required, no extras) ----------

def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


STR = {"type": "string"}
STR_LIST = {"type": "array", "items": STR}

ANALYZE_SCHEMA = _obj({
    "title": STR,
    "summary": STR,
    "features": {"type": "array", "items": _obj({"text": STR, "terms": STR_LIST})},
    "cpc_codes": STR_LIST,
    "queries": {"type": "array", "items": _obj({"q": STR, "purpose": STR})},
    "uspto_queries": {"type": "array", "items": _obj({"q": STR, "purpose": STR})},
})

SCREEN_SCHEMA = _obj({
    "scores": {"type": "array", "items": _obj({"ref": STR, "score": {"type": "integer"}})},
})

COMPARE_SCHEMA = _obj({
    "patents": {"type": "array", "items": _obj({
        "ref": STR,
        "similarity": {"type": "integer"},
        "explanation": STR,
        "features": {"type": "array", "items": _obj({
            "id": STR,
            "status": {"type": "string", "enum": ["present", "partial", "absent"]},
            "evidence": STR,
        })},
    })},
})


# ---------- prompts ----------

ANALYZE_SYSTEM = """You are an experienced patent search professional. A user describes an invention idea in plain language; you prepare a prior-art search on Google Patents.

Do the following, always writing in English even if the idea is in another language:
1. title: a short name for the idea (max 8 words). summary: 1-2 sentences restating the idea in technical terms.
2. features: 3-7 essential technical features, i.e. the elements a patent claim for this idea would need. Only use features the user stated or that the idea strictly requires; never add optional extras of your own. Each has "text" (one short sentence) and "terms": 3-6 search terms, meaning synonyms and the wording patent drafters use (e.g. "water-soluble" for "dissolves"). Terms are plain words or short phrases, no boolean syntax.
3. cpc_codes: 0-3 CPC classification codes most likely to contain similar patents, at group level (format like "A61B5/00" or "B60L53/16"). Only list codes you are confident about for this specific technology; an empty list is better than a wrong code.
4. queries: exactly 4 Google Patents queries, each with a one-line "purpose".
   Syntax: concept groups in parentheses joined with ";" (every group must match); OR between alternatives inside a group; "*" for truncation (dissolv*). Every multi-word phrase MUST be in double quotes, e.g. AB=("heat exchanger" OR radiator); unquoted words would all be required.
   A group can be limited to one field: TI=(...) title, AB=(...) abstract, CL=(...) claims. A group without a prefix matches anywhere in the full text, which is very noisy, so put a field prefix on every group.
   Vocabulary matters most. Patents use formal, generic wording, not product names or marketing words, and older patents use older terms. Write terms the way a patent drafter would and mix generic and specific wording, e.g. "motor vehicle" rather than "car"; "fastening member" rather than "screw"; "wireless transceiver" rather than "Bluetooth chip"; "energy storage unit" rather than "battery pack". Never use brand names, slogans or words that describe what is absent (no-, non-, -less, -free, without); describe the structure or function that is present instead.
   Keep terms short: prefer single words with truncation (rotat*, magnet*, adhes*) and phrases of at most 2 words; long phrases almost never match. Broad is good: Google ranks by relevance and only the top results are used, so a query matching hundreds or thousands of patents is fine, while one matching fewer than 20 is too narrow.
   - Every query has a group naming the product or system in generic terms.
   - Use 2 groups per query; at most one query may have 3. Each group has 4-8 alternatives.
   - Query 1: the product plus the most distinctive feature, both AB=.
   - Query 2: the product plus a second distinctive feature, both AB=.
   - Query 3: the product plus the most distinctive feature, both CL=.
   - Query 4: the product in TI= plus the most distinctive feature in AB=.
   Do not put CPC codes in queries.
5. uspto_queries: exactly 3 searches for the USPTO database, which can only search patent TITLES (no abstract or claims), so they must be broader than the Google Patents queries.
   Syntax: TI=(...) groups joined with AND; OR between alternatives inside a group; "*" for truncation; multi-word phrases in double quotes.
   - Each query has exactly 2 groups: TI=(the product or system in generic terms) AND TI=(one distinctive feature).
   - Query 1 uses the most distinctive feature, query 2 a second feature, query 3 a third feature or the main function.
   - Titles are short and generic (e.g. "BEVERAGE CAPSULE", "ELECTRIC VEHICLE CHARGING SYSTEM"), so use 5-10 alternatives per group: single truncated words (capsul*, cartridg*, pod*) and at most 2-word phrases. Avoid rare words that seldom appear in titles."""

SCREEN_SYSTEM = """You screen patent search results for similarity to an invention idea. For every candidate, give a score from 0 to 10 based only on its title and snippet:
9-10 = describes essentially the same invention
6-8 = same problem and several of the key features
3-5 = same field, few of the features
0-2 = unrelated
Return one entry per candidate, using its ref exactly as given."""

COMPARE_SYSTEM = """You are a patent analyst comparing existing patents with a new invention idea, feature by feature. Judge only from the abstract and claim text provided.

For each patent:
- features: one entry for every feature id of the idea. status is "present" only if the patent text clearly discloses that feature, "partial" if it discloses something similar, broader or narrower, "absent" if it is not mentioned. evidence is a short quote (max 25 words) from the patent text supporting the status, or "" when absent.
- similarity: 0-100 for how close the patent as a whole is to the idea. 90+ = the same invention; 70-89 = most key features; 40-69 = related approach; below 40 = same field only.
- explanation: 1-2 sentences naming the main overlap and the main difference.
Return one entry per patent, using its ref exactly as given."""


_ASCII = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
                        "“": '"', "”": '"', "‘": "'", "’": "'", " ": " "})


def _plain(text: str) -> str:
    """LLMs emit typographic hyphens/quotes that Google Patents doesn't treat as - and "."""
    return " ".join(text.translate(_ASCII).split())


def _features_block(features: list[Feature]) -> str:
    return "\n".join(f"{f.id}: {f.text}" for f in features)


def _clip(text: str | None, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _llm_meta(metas: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": metas[0]["model"] if metas else llm.get_model(),
        "calls": len(metas),
        "cached_calls": sum(m["cached"] for m in metas),
        "tokens": sum(m["tokens"] for m in metas),
    }


# ---------- endpoints ----------

@router.get("/llm/status")
async def llm_status() -> dict[str, Any]:
    """Checks the Groq key by listing models. Uses no tokens."""
    return await llm.status()


@router.post("/idea/analyze")
async def analyze(body: AnalyzeIn) -> dict[str, Any]:
    """Idea -> features, CPC codes, 4 Google Patents and 3 USPTO queries. One LLM call, no SerpApi credits."""
    data, meta = await llm.chat_json(
        ANALYZE_SYSTEM, f"Invention idea:\n{body.idea.strip()}", "idea_analysis", ANALYZE_SCHEMA,
        max_tokens=3000, effort="medium",
    )
    features = [
        {"id": f"F{i}", "text": _plain(f["text"]), "terms": [_plain(t) for t in f["terms"] if t.strip()]}
        for i, f in enumerate(data["features"], 1)
    ]
    return {
        "title": _plain(data["title"]),
        "summary": _plain(data["summary"]),
        "features": features,
        "cpc_codes": [_plain(c) for c in data["cpc_codes"] if c.strip()],
        "queries": [{"q": _plain(q["q"]), "purpose": _plain(q["purpose"])} for q in data["queries"] if q["q"].strip()],
        "uspto_queries": [
            {"q": _plain(q["q"]), "purpose": _plain(q["purpose"])} for q in data["uspto_queries"] if q["q"].strip()
        ],
        "llm": _llm_meta([meta]),
    }


async def _run_query(q: str, filters: dict[str, str], num: int) -> dict[str, Any]:
    try:
        data, cached = await serpapi.patents_search({"q": q, "num": num, **filters})
    except HTTPException as exc:
        return {"q": q, "error": str(exc.detail), "cached": False, "total_results": 0, "returned": 0, "results": []}
    results = [
        serpapi.normalize_result(r) for r in data.get("organic_results", [])
        if not str(r.get("patent_id", "")).startswith("scholar/")
    ]
    return {
        "q": q,
        "error": data.get("error"),  # e.g. "Google Patents hasn't returned any results for this query."
        "cached": cached,
        "total_results": (data.get("search_information") or {}).get("total_results"),
        "returned": len(results),
        "results": results,
    }


def _relax(q: str) -> str | None:
    """Looser version of a query that found too little: drop the last group, or with two groups,
    search the second one in the full text instead of one field."""
    groups = [g.strip() for g in q.split(";") if g.strip()]
    if len(groups) >= 3:
        return ";".join(groups[:-1])
    if len(groups) == 2:
        second = re.sub(r"^(TI|AB|CL)=", "", groups[1])
        if second != groups[1]:
            return f"{groups[0]};{second}"
    return None


async def _screen(idea: str, features: list[Feature], cands: list[dict[str, Any]], system: str,
                  describe) -> list[dict[str, Any]]:
    """Scores every candidate 0-10 in place (c["screen_score"]); describe(c) is its one-line text."""
    metas: list[dict[str, Any]] = []
    header = f"Invention idea:\n{idea.strip()}\n\nKey features:\n{_features_block(features)}\n\nCandidates:\n"
    for i in range(0, len(cands), SCREEN_CHUNK):
        chunk = cands[i:i + SCREEN_CHUNK]
        lines = "\n".join(f"{c['ref']} | {describe(c)}" for c in chunk)
        data, meta = await llm.chat_json(system, header + lines, "screening", SCREEN_SCHEMA, max_tokens=2000, effort="low")
        metas.append(meta)
        scores = {s["ref"]: s["score"] for s in data["scores"]}
        for c in chunk:
            s = scores.get(c["ref"])
            c["screen_score"] = max(0, min(10, s)) if isinstance(s, int) else None
    return metas


@router.post("/idea/candidates")
async def candidates(body: CandidatesIn) -> dict[str, Any]:
    """Runs the queries (1 credit each unless cached), merges, and screens titles/snippets with the LLM."""
    filters = {k: v for k, v in body.filters.model_dump().items() if v}
    queries = [_plain(q) for q in body.queries if q.strip()]
    runs = await asyncio.gather(*(_run_query(q, filters, body.num) for q in queries))

    retry = [(i, _relax(run["q"])) for i, run in enumerate(runs) if run["returned"] < MIN_RESULTS]
    retry = [(i, rq) for i, rq in retry if rq]
    relaxed = await asyncio.gather(*(_run_query(rq, filters, body.num) for _, rq in retry))
    for (i, _), rr in zip(retry, relaxed):
        runs[i]["relaxed"] = {k: v for k, v in rr.items() if k != "results"}
        runs[i]["results"] = runs[i]["results"] + rr["results"]

    merged: dict[str, dict[str, Any]] = {}
    for qi, run in enumerate(runs):
        for r in run["results"]:
            key = r["publication_number"] or r["patent_id"]
            if not key:
                continue
            c = merged.setdefault(key, {**r, "found_by": [], "best_position": r["position"] or 999})
            if qi not in c["found_by"]:
                c["found_by"].append(qi)
            c["best_position"] = min(c["best_position"], r["position"] or 999)
    cands = list(merged.values())
    for i, c in enumerate(cands, 1):
        c["ref"] = f"C{i}"

    metas = await _screen(body.idea, body.features, cands, SCREEN_SYSTEM,
                          lambda c: f"{c['title'] or ''} | {_clip(c['snippet'], SNIPPET_CHARS)}")
    cands.sort(key=lambda c: (-(c.get("screen_score") if c.get("screen_score") is not None else -1),
                              -len(c["found_by"]), c["best_position"]))
    return {
        "queries": [{k: v for k, v in run.items() if k != "results"} for run in runs],
        "candidates": cands,
        "selected": [c["ref"] for c in cands[:body.keep]],
        "serpapi_calls": sum(1 for run in [*runs, *relaxed] if not run["cached"] and not run["error"]),
        "llm": _llm_meta(metas),
    }


def _first_claim(details: dict[str, Any]) -> str:
    # Non-English patents carry a machine translation in claims_translated.
    for claim in details.get("claims_translated") or details.get("claims") or []:
        text = " ".join(str(claim).split())
        if len(text) > 40:  # skip header lines such as "CLAIMS" / "CONCLUSIONS"
            return text
    return ""


async def _fetch_details(p: ComparePatent) -> dict[str, Any]:
    try:
        d, cached = await serpapi.patent_details(p.patent_id)
    except HTTPException as exc:
        return {"cached": False, "error": str(exc.detail), "abstract": "", "first_claim": ""}
    return {
        "cached": cached,
        "error": d.get("error"),
        "abstract": _clip(d.get("abstract"), ABSTRACT_CHARS),
        "first_claim": _clip(_first_claim(d), CLAIM_CHARS),
    }


@router.post("/idea/compare")
async def compare(body: CompareIn) -> dict[str, Any]:
    """Fetches details (1 credit each unless cached) and compares each patent with the idea's features."""
    details = await asyncio.gather(*(_fetch_details(p) for p in body.patents))

    verdicts, meta = await _compare_llm(body.idea, body.features, [
        # Fall back to the search snippet when the details have no abstract.
        (p.publication_number or p.patent_id, p.title, d["abstract"] or _clip(p.snippet, ABSTRACT_CHARS), d["first_claim"])
        for p, d in zip(body.patents, details)
    ])
    out = [
        {"patent_id": p.patent_id, "abstract": d["abstract"], "first_claim": d["first_claim"],
         "details_cached": d["cached"], "details_error": d["error"], **v}
        for p, d, v in zip(body.patents, details, verdicts)
    ]
    return {
        "results": out,
        "serpapi_calls": sum(1 for d in details if not d["cached"] and not d["error"]),
        "llm": _llm_meta([meta]),
    }


async def _compare_llm(idea: str, features: list[Feature],
                       patents: list[tuple[str, str | None, str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """patents: (number, title, abstract, first claim). Returns one {similarity, explanation, features} per patent."""
    blocks = [
        f"P{i} ({number}): {title or ''}\nAbstract: {abstract or '(not available)'}\n"
        f"First claim: {claim or '(not available)'}"
        for i, (number, title, abstract, claim) in enumerate(patents, 1)
    ]
    user = (
        f"Invention idea:\n{idea.strip()}\n\nFeatures:\n{_features_block(features)}\n\n"
        f"Patents:\n\n" + "\n\n".join(blocks)
    )
    data, meta = await llm.chat_json(
        COMPARE_SYSTEM, user, "comparison", COMPARE_SCHEMA, max_tokens=3500, effort="medium",
    )

    by_ref = {r["ref"]: r for r in data["patents"]}
    out = []
    for i in range(1, len(patents) + 1):
        r = by_ref.get(f"P{i}", {})
        given = {f["id"]: f for f in r.get("features", [])}
        out.append({
            "similarity": max(0, min(100, r["similarity"])) if isinstance(r.get("similarity"), int) else None,
            "explanation": r.get("explanation", ""),
            "features": [
                {"id": f.id, "status": given.get(f.id, {}).get("status", "absent"),
                 "evidence": given.get(f.id, {}).get("evidence", "")}
                for f in features
            ],
        })
    return out, meta


# ---------- USPTO Open Data Portal (kept separate from the Google Patents results) ----------

USPTO_FIELDS = [
    "applicationNumberText", "grantDocumentMetaData", "pgpubDocumentMetaData",
    *(f"applicationMetaData.{f}" for f in (
        "inventionTitle", "patentNumber", "earliestPublicationNumber", "earliestPublicationDate", "filingDate",
        "effectiveFilingDate", "grantDate", "firstApplicantName", "firstInventorName", "cpcClassificationBag",
        "applicationStatusDescriptionText", "applicationTypeLabelName",
    )),
]
# Google Patents date types mapped to the closest ODP field.
USPTO_DATE_FIELDS = {
    "priority": "applicationMetaData.effectiveFilingDate",
    "filing": "applicationMetaData.filingDate",
    "publication": "applicationMetaData.earliestPublicationDate",
}
GRANTED = 'applicationMetaData.publicationCategoryBag:"Granted/Issued"'
USPTO_SCREEN_SYSTEM = SCREEN_SYSTEM.replace("title and snippet", "title and CPC classes")


class UsptoCandidatesIn(BaseModel):
    idea: str = Field(min_length=20, max_length=4000)
    features: list[Feature] = Field(min_length=1, max_length=12)
    queries: list[str] = Field(min_length=1, max_length=6)
    filters: Filters = Field(default_factory=Filters)  # country and language don't apply (US only)
    num: int = Field(25, ge=10, le=100)
    keep: int = Field(5, ge=1, le=30)


def _uspto_payload(q: str, filters: Filters, num: int) -> dict[str, Any]:
    """Utility applications only: designs have no abstract and a single ornamental claim."""
    q = f"({uspto.expand_query(q)})"
    if filters.status == "GRANT":
        q += f" AND {GRANTED}"
    elif filters.status == "APPLICATION":
        q += f" AND NOT {GRANTED}"
    payload: dict[str, Any] = {
        "q": q,
        "filters": [{"name": "applicationMetaData.applicationTypeCode", "value": ["UTL"]}],
        "fields": USPTO_FIELDS,
        "pagination": {"offset": 0, "limit": num},
    }
    if filters.before:
        kind, _, d = filters.before.partition(":")
        payload["rangeFilters"] = [{"field": USPTO_DATE_FIELDS[kind], "valueFrom": "1900-01-01",
                                    "valueTo": f"{d[:4]}-{d[4:6]}-{d[6:]}"}]
    return payload


async def _run_uspto_query(q: str, filters: Filters, num: int) -> dict[str, Any]:
    try:
        data, cached = await uspto.search(_uspto_payload(q, filters, num))
    except HTTPException as exc:
        return {"q": q, "error": str(exc.detail), "cached": False, "total_results": 0, "returned": 0, "results": []}
    results = [uspto.normalize_result(r) for r in data.get("patentFileWrapperDataBag") or []]
    for i, r in enumerate(results, 1):
        r["position"] = i
    return {
        "q": q,
        "error": None if results else "No matching USPTO applications.",
        "cached": cached,
        "total_results": data.get("count"),
        "returned": len(results),
        "results": results,
    }


def _split_and(q: str) -> list[str]:
    """Top-level AND groups (ANDs inside parentheses or quotes stay put)."""
    groups, depth, quoted, start = [], 0, False, 0
    for i, ch in enumerate(q):
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch in "()":
            depth += 1 if ch == "(" else -1
        elif not quoted and depth == 0 and q.startswith(" AND ", i):
            groups.append(q[start:i])
            start = i + 5
    groups.append(q[start:])
    return [g.strip() for g in groups if g.strip()]


@router.post("/idea/uspto/candidates")
async def uspto_candidates(body: UsptoCandidatesIn) -> dict[str, Any]:
    """Runs title queries on the USPTO ODP (free), merges by application number, screens titles with the LLM."""
    queries = [_plain(q) for q in body.queries if q.strip()]
    runs = await asyncio.gather(*(_run_uspto_query(q, body.filters, body.num) for q in queries))

    # Too few hits: drop the last AND group once (titles are short, two groups can be too strict).
    retry = [(i, " AND ".join(_split_and(run["q"])[:-1])) for i, run in enumerate(runs)
             if run["returned"] < MIN_RESULTS and len(_split_and(run["q"])) >= 2]
    relaxed = await asyncio.gather(*(_run_uspto_query(rq, body.filters, body.num) for _, rq in retry))
    for (i, _), rr in zip(retry, relaxed):
        runs[i]["relaxed"] = {k: v for k, v in rr.items() if k != "results"}
        runs[i]["results"] = runs[i]["results"] + rr["results"]

    merged: dict[str, dict[str, Any]] = {}
    for qi, run in enumerate(runs):
        for r in run["results"]:
            c = merged.setdefault(r["application_number"], {**r, "found_by": [], "best_position": r["position"]})
            if qi not in c["found_by"]:
                c["found_by"].append(qi)
            c["best_position"] = min(c["best_position"], r["position"])
    cands = list(merged.values())
    for i, c in enumerate(cands, 1):
        c["ref"] = f"U{i}"

    metas = await _screen(body.idea, body.features, cands, USPTO_SCREEN_SYSTEM,
                          lambda c: f"{c['title'] or ''} | CPC: {', '.join(c['cpc'][:4]) or '-'}")
    # Prefer patents whose full text can be fetched for the comparison.
    cands.sort(key=lambda c: (-(c.get("screen_score") if c.get("screen_score") is not None else -1),
                              not c["has_full_text"], -len(c["found_by"]), c["best_position"]))
    return {
        "queries": [{k: v for k, v in run.items() if k != "results"} for run in runs],
        "candidates": cands,
        "selected": [c["ref"] for c in cands[:body.keep]],
        "uspto_calls": sum(1 for run in [*runs, *relaxed] if not run["cached"]),
        "llm": _llm_meta(metas),
    }


async def _fetch_uspto_text(p: UsptoComparePatent) -> dict[str, Any]:
    try:
        t, cached = await uspto.full_text(p.application_number)
    except HTTPException as exc:
        return {"cached": False, "error": str(exc.detail), "abstract": "", "first_claim": "", "source": None}
    return {
        "cached": cached,
        "error": None if t["source"] else "No full-text XML available for this application.",
        "abstract": _clip(t["abstract"], ABSTRACT_CHARS),
        "first_claim": _clip(t["first_claim"], CLAIM_CHARS),
        "source": t["source"],
    }


@router.post("/idea/uspto/compare")
async def uspto_compare(body: UsptoCompareIn) -> dict[str, Any]:
    """Fetches abstract + first claim from the USPTO grant/publication XML (free) and compares per feature."""
    texts = await asyncio.gather(*(_fetch_uspto_text(p) for p in body.patents))
    verdicts, meta = await _compare_llm(body.idea, body.features, [
        (p.publication_number or p.application_number, p.title, t["abstract"], t["first_claim"])
        for p, t in zip(body.patents, texts)
    ])
    out = [
        {"application_number": p.application_number, "abstract": t["abstract"], "first_claim": t["first_claim"],
         "text_source": t["source"], "details_cached": t["cached"], "details_error": t["error"], **v}
        for p, t, v in zip(body.patents, texts, verdicts)
    ]
    return {"results": out, "llm": _llm_meta([meta])}
