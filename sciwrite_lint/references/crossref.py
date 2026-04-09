"""CrossRef API client — async httpx.

CrossRef is the gold standard for DOI metadata. Provides authoritative
title, authors, year, venue, and retraction status.

Uses httpx.AsyncClient (shared with all other API clients), eliminating
the per-call ~100-200 MB overhead that habanero/requests caused via
separate SSL contexts and connection pools per thread.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from loguru import logger

from sciwrite_lint._network import clean_and_validate_doi
from sciwrite_lint.models import Citation

_CROSSREF_BASE = "https://api.crossref.org/works"

# Fields needed by _parse_crossref — avoids downloading full records
# (reference lists alone can be 5-20 MB per paper).
_CROSSREF_SELECT = (
    "DOI,title,author,published-print,published-online,issued,"
    "container-title,is-referenced-by-count,type,abstract,update-to"
)


def _crossref_year(item: dict[str, Any]) -> int | None:
    """Extract publication year from a raw CrossRef work item."""
    for date_field in ("published-print", "published-online", "issued"):
        parts = item.get(date_field, {}).get("date-parts", [[]])
        if parts and parts[0] and parts[0][0]:
            return parts[0][0]
    return None


def _normalize_crossref_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert raw CrossRef item to the normalized format _best_match expects."""
    titles = item.get("title", [])
    authors = []
    for a in item.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        if family:
            authors.append(f"{given} {family}".strip() if given else family)
    containers = item.get("container-title", [])
    return {
        "title": titles[0] if titles else "",
        "authors": authors,
        "year": _crossref_year(item),
        "venue": containers[0] if containers else "",
    }


def _best_title_match(
    query_title: str,
    items: list[dict[str, Any]],
    query_authors: list[str] | tuple[str, ...] | None = None,
    query_year: int | None = None,
    query_venue: str = "",
) -> dict[str, Any] | None:
    """Pick the CrossRef result whose title best matches the query.

    Delegates scoring to ``_best_match`` (single implementation of
    title + author + year + venue scoring). Returns the raw CrossRef item.
    """
    from sciwrite_lint.api import _best_match

    # Build normalized candidates, keeping a map back to the raw items
    normalized = []
    raw_by_idx: dict[int, dict[str, Any]] = {}
    for i, item in enumerate(items):
        norm = _normalize_crossref_item(item)
        norm["_idx"] = i
        normalized.append(norm)
        raw_by_idx[i] = item

    best = _best_match(
        query_title,
        normalized,
        query_authors=query_authors,
        query_year=query_year,
        query_venue=query_venue,
    )
    if best is None:
        return None
    return raw_by_idx[best["_idx"]]


async def crossref_lookup(
    citation: Citation,
    polite_email: str = "",
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any] | None:
    """Look up a citation via CrossRef. Returns normalized dict or None.

    Async — uses httpx. Caller should pass a shared client for connection
    reuse. Rate limiting is the caller's responsibility.
    """
    _client = client or httpx.AsyncClient(timeout=15.0, follow_redirects=True)

    headers = (
        {"User-Agent": f"sciwrite-lint/1.0 (mailto:{polite_email})"}
        if polite_email
        else {}
    )

    try:
        return await _do_crossref_lookup(citation, _client, headers, polite_email)
    except Exception as e:
        logger.debug("CrossRef lookup failed for {}: {}", citation.key, e)
        return None
    finally:
        if client is None:
            await _client.aclose()


async def _do_crossref_lookup(
    citation: Citation,
    client: httpx.AsyncClient,
    headers: dict[str, str],
    polite_email: str = "",
) -> dict[str, Any] | None:
    """Core lookup logic with retries handled by retry_on_transient."""
    from sciwrite_lint.rate_limiter import retry_on_transient

    # mailto query param ensures polite pool (higher rate limits)
    base_params: dict[str, str] = {}
    if polite_email:
        base_params["mailto"] = polite_email

    # DOI lookup first (most reliable)
    # Note: /works/{doi} does not support select= (returns 400)
    clean_doi = clean_and_validate_doi(citation.doi) if citation.doi else None
    if clean_doi:
        resp = await retry_on_transient(
            lambda: client.get(
                f"{_CROSSREF_BASE}/{clean_doi}",
                params=base_params,
                headers=headers,
            ),
            label=f"CrossRef DOI {citation.key}",
        )
        if resp.status_code == 200:
            msg = resp.json().get("message")
            if msg:
                return _parse_crossref(msg)

    # Title search (DOI not available or failed)
    if citation.title:
        from sciwrite_lint.api import _first_surname

        surname = _first_surname(citation.authors)
        query = f"{surname} {citation.title}" if surname else citation.title
        resp = await retry_on_transient(
            lambda query=query: client.get(
                _CROSSREF_BASE,
                params={
                    **base_params,
                    "query": query,
                    "rows": 10,
                    "select": _CROSSREF_SELECT,
                },
                headers=headers,
            ),
            label=f"CrossRef title {citation.key}",
        )
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            if items:
                best = _best_title_match(
                    citation.title,
                    items,
                    query_authors=citation.authors,
                    query_year=int(citation.year) if citation.year.isdigit() else None,
                    query_venue=citation.venue,
                )
                if best is not None:
                    return _parse_crossref(best)

    return None


async def check_retraction(
    doi: str,
    polite_email: str = "",
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Check if a DOI points to a retracted paper via CrossRef metadata."""
    if not doi:
        return False

    _client = client or httpx.AsyncClient(timeout=15.0, follow_redirects=True)

    headers = (
        {"User-Agent": f"sciwrite-lint/1.0 (mailto:{polite_email})"}
        if polite_email
        else {}
    )

    try:
        # /works/{doi} does not support select= (returns 400)
        resp = await _client.get(
            f"{_CROSSREF_BASE}/{doi}",
            headers=headers,
        )
        if resp.status_code != 200:
            return False
        msg = resp.json().get("message", {})
        updates = msg.get("update-to", [])
        for update in updates:
            if update.get("type") == "retraction":
                return True
            label = update.get("label", "").lower()
            if "retract" in label:
                return True
        return False
    except Exception as e:
        logger.debug("Retraction check failed for DOI {}: {}", doi, e)
        return False
    finally:
        if client is None:
            await _client.aclose()


def _parse_crossref(work: dict) -> dict[str, Any]:
    """Parse CrossRef work into normalized dict."""
    # Authors
    authors = []
    for a in work.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        if family:
            name = f"{given} {family}".strip() if given else family
            authors.append(name)

    # Title
    titles = work.get("title", [])
    title = titles[0] if titles else ""

    # Year — prefer print date, then online
    year = None
    for date_field in ("published-print", "published-online", "issued"):
        parts = work.get(date_field, {}).get("date-parts", [[]])
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break

    # Venue
    containers = work.get("container-title", [])
    venue = containers[0] if containers else ""

    # DOI
    doi = work.get("DOI", "")

    # Abstract
    abstract = work.get("abstract", "")
    if abstract:
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()

    # Retraction
    retracted = False
    for update in work.get("update-to", []):
        if "retract" in (update.get("type", "") + update.get("label", "")).lower():
            retracted = True
            break

    return {
        "source": "crossref",
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "citation_count": work.get("is-referenced-by-count", 0),
        "abstract": abstract or None,
        "retracted": retracted,
    }
