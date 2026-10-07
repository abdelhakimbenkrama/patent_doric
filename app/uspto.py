"""USPTO Open Data Portal (ODP) client: patent application search and full text.

ODP is free with an API key (X-API-KEY header) and covers US applications filed since 2001.
Search only sees bibliographic data (title, CPC, applicant, inventor, dates), not abstracts or
claims; the text comes from the grant or pre-grant publication XML that each application links
to under associated-documents.
"""

import asyncio
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from fastapi import HTTPException

from . import cache
from .config import require_key

BASE_URL = "https://api.uspto.gov"
SEARCH_PATH = "/api/v1/patent/applications/search"
FILES_PREFIX = f"{BASE_URL}/api/v1/datasets/products/files/"
TIMEOUT = httpx.Timeout(60.0)
MAX_RETRIES = 2

# Short prefixes accepted in queries (Google Patents-like), expanded to ODP field names.
FIELD_ALIASES = {
    "TI": "applicationMetaData.inventionTitle",
    "CPC": "applicationMetaData.cpcClassificationBag",
    "APP": "applicationMetaData.firstApplicantName",
    "INV": "applicationMetaData.inventorBag.inventorNameText",
}
_ALIAS_RE = re.compile(r"\b(" + "|".join(FIELD_ALIASES) + r")=")

_limit = asyncio.Semaphore(4)  # keep parallel requests well under ODP's rate limits


def get_api_key() -> str:
    return require_key("USPTO_API_KEY")


def expand_query(q: str) -> str:
    """TI=(a OR b) AND TI=(c) -> applicationMetaData.inventionTitle:(a OR b) AND ..."""
    return _ALIAS_RE.sub(lambda m: FIELD_ALIASES[m.group(1)] + ":", q)


def _odp_error(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300] or f"HTTP {resp.status_code}"
    return data.get("errorDetails") or data.get("error") or data.get("message") or f"HTTP {resp.status_code}"


async def _request(method: str, url: str, **kwargs: Any) -> httpx.Response:
    headers = {"X-API-KEY": get_api_key(), "Accept": "application/json"}
    async with _limit, httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await client.request(method, url, headers=headers, **kwargs)
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"Could not reach USPTO ODP: {exc}") from exc
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                await asyncio.sleep(min(max(float(resp.headers.get("retry-after") or 5), 1.0), 30.0))
                continue
            # File downloads may redirect to storage; follow without sending the key along.
            if resp.is_redirect and "location" in resp.headers:
                try:
                    resp = await client.get(resp.headers["location"], follow_redirects=True)
                except httpx.HTTPError as exc:
                    raise HTTPException(status_code=502, detail=f"Could not download from USPTO: {exc}") from exc
            return resp
    return resp


async def search(payload: dict[str, Any], *, use_cache: bool = True) -> tuple[dict[str, Any], bool]:
    """POST search. Returns (data, from_cache). "No matching records" (404) becomes an empty result."""
    if use_cache:
        hit = cache.get("uspto", payload)
        if hit is not None:
            return hit, True
    resp = await _request("POST", BASE_URL + SEARCH_PATH, json=payload)
    if resp.status_code == 404:
        return {"count": 0, "patentFileWrapperDataBag": []}, False
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"USPTO ODP: {_odp_error(resp)}")
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="USPTO ODP returned non-JSON.")
    if use_cache:
        cache.put("uspto", payload, data)
    return data, False


async def status() -> dict[str, Any]:
    """Checks the key with a one-result search. ODP is free, so this costs nothing."""
    data, _ = await search({"q": "applicationMetaData.applicationTypeCode:UTL", "pagination": {"offset": 0, "limit": 1},
                            "fields": ["applicationNumberText"]}, use_cache=False)
    return {"ok": True, "total_applications": data.get("count")}


def _cpc(code: str) -> str:
    return re.sub(r"\s+", "", code)  # ODP pads codes: "G06N   3/08" -> "G06N3/08"


def _xml_url(r: dict[str, Any]) -> str | None:
    for key in ("grantDocumentMetaData", "pgpubDocumentMetaData"):
        url = (r.get(key) or {}).get("fileLocationURI")
        if url and url.startswith(FILES_PREFIX):
            return url
    return None


def normalize_result(r: dict[str, Any]) -> dict[str, Any]:
    m = r.get("applicationMetaData") or {}
    app_no = r.get("applicationNumberText")
    patent_no = m.get("patentNumber")
    return {
        "application_number": app_no,
        # The granted patent when there is one, else the first pre-grant publication.
        "publication_number": f"US{patent_no}" if patent_no else m.get("earliestPublicationNumber"),
        "patent_number": patent_no,
        "earliest_publication_number": m.get("earliestPublicationNumber"),
        "title": " ".join((m.get("inventionTitle") or "").split()) or None,
        "type": m.get("applicationTypeLabelName"),
        "status": m.get("applicationStatusDescriptionText"),
        "filing_date": m.get("filingDate"),
        "effective_filing_date": m.get("effectiveFilingDate"),
        "publication_date": m.get("earliestPublicationDate"),
        "grant_date": m.get("grantDate"),
        "applicant": m.get("firstApplicantName"),
        "inventor": m.get("firstInventorName"),
        "cpc": [_cpc(c) for c in m.get("cpcClassificationBag") or []],
        "has_full_text": _xml_url(r) is not None,
        "link": f"https://patentcenter.uspto.gov/applications/{app_no}" if app_no else None,
    }


# ---------- full text (abstract + first claim) ----------

def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_full_text(xml: bytes) -> dict[str, str]:
    """Abstract and first claim from a USPTO grant or application XML (red book format)."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {"abstract": "", "first_claim": ""}
    return {"abstract": _text(root.find(".//abstract")), "first_claim": _text(root.find(".//claims/claim"))}


async def full_text(application_number: str) -> tuple[dict[str, Any], bool]:
    """Returns ({abstract, first_claim, source}, from_cache). source is "grant", "application" or None."""
    app_no = re.sub(r"\D", "", application_number)
    if not app_no:
        raise HTTPException(status_code=400, detail="Invalid application number.")
    hit = cache.get("uspto-text", app_no)
    if hit is not None:
        return hit, True

    resp = await _request("GET", f"{BASE_URL}/api/v1/patent/applications/{app_no}/associated-documents")
    if resp.status_code == 404:
        out = {"abstract": "", "first_claim": "", "source": None}
        cache.put("uspto-text", app_no, out)
        return out, False
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"USPTO ODP: {_odp_error(resp)}")
    bag = (resp.json().get("patentFileWrapperDataBag") or [{}])[0]

    out = {"abstract": "", "first_claim": "", "source": None}
    failed = False
    for key, source in (("grantDocumentMetaData", "grant"), ("pgpubDocumentMetaData", "application")):
        url = (bag.get(key) or {}).get("fileLocationURI")
        if not url or not url.startswith(FILES_PREFIX):
            continue
        xml = await _request("GET", url)
        if xml.status_code != 200:
            failed = True
            continue
        parsed = parse_full_text(xml.content)
        if parsed["abstract"] or parsed["first_claim"]:
            out = {**parsed, "source": source}
            break
    if out["source"] or not failed:  # don't remember a failed download as "no text"
        cache.put("uspto-text", app_no, out)
    return out, False
