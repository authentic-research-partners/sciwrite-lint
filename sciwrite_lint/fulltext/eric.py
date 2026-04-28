"""ERIC (eric.ed.gov) — US Department of Education research database.

ERIC indexes ~2.1M education-research documents. Two ID families:

- ``ED*`` IDs (~660k): ERIC Documents — ERIC hosts the PDF at a
  deterministic URL (``files.eric.ed.gov/fulltext/{ID}.pdf``). Freely
  downloadable, this is the half we can serve.
- ``EJ*`` IDs: journal articles — ERIC indexes only metadata; the PDF
  lives on a paywalled publisher site. Skipped via the ``id:ED*`` filter.

Title tokens go to the default text field (ERIC's Solr rejects phrase
queries on that field — "field was indexed without position data" — so
the value is escaped per-char but not wrapped in quotes). Optional bib
author filter uses the ``author`` Solr field in phrase form; Solr-
reserved chars are backslash-escaped so non-English names and titles
pass through unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
from loguru import logger

from sciwrite_lint._network import ssrf_safe_client
from sciwrite_lint.fulltext._common import (
    ERIC_PDF_BASE,
    ERIC_SEARCH_API,
    AcquisitionResult,
    _MAX_JSON_BYTES,
    _eric_limiter,
    _escape_solr_query,
    _polite_user_agent,
)
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._search import (
    SearchCandidate,
    _coerce_year,
    _first_string,
    _string_list,
)
from sciwrite_lint.fulltext._validation import BibEvidence, _extract_surname
from sciwrite_lint.rate_limiter import retry_on_transient

_ERIC_ED_ID_RE = re.compile(r"^ED\d{4,7}$")


def _parse_eric_candidates(payload: dict) -> list[SearchCandidate]:
    """Parse an ERIC search response into candidates.

    Only ED* IDs qualify; EJ* entries point at paywalled publisher PDFs
    and are skipped by the ``id:ED*`` query filter. The ID format is
    validated against a strict pattern before we construct the
    deterministic PDF URL, to prevent any path-injection risk from a
    hypothetically malformed response.
    """
    response = payload.get("response") or {}
    docs = response.get("docs") or []
    candidates: list[SearchCandidate] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("id")
        if not (isinstance(doc_id, str) and _ERIC_ED_ID_RE.match(doc_id)):
            continue
        candidates.append(
            SearchCandidate(
                url=f"{ERIC_PDF_BASE}/{doc_id}.pdf",
                title=_first_string(doc.get("title")),
                authors=_string_list(doc.get("author")),
                year=_coerce_year(doc.get("publicationdateyear")),
            )
        )
    return candidates


def _build_query(title: str, authors: list[str] | None) -> str:
    """Build an ERIC Solr query.

    - Title tokens: escaped but *not* wrapped in quotes (ERIC's default
      text field lacks position data, so phrase queries error out).
    - Author: when a bib surname is provided, filter on the ``author``
      field in phrase form.
    - ``id:ED*``: always appended so only downloadable ED documents come
      back (EJ* = paywalled publisher).
    """
    q_parts: list[str] = []
    escaped_title = _escape_solr_query(title.strip())
    if escaped_title:
        q_parts.append(escaped_title)
    if authors:
        surname = _extract_surname(authors[0]).strip()
        if surname:
            q_parts.append(f'author:"{_escape_solr_query(surname)}"')
    q_parts.append("id:ED*")
    return " AND ".join(q_parts)


async def lookup_eric_by_title(
    title: str,
    polite_email: str = "",
    *,
    authors: list[str] | None = None,
    year: int | None = None,  # noqa: ARG001  # carried on candidates
) -> list[SearchCandidate]:
    """Search ERIC for ED* documents. Return up to 5 candidates."""
    if not title or not title.strip():
        return []
    query = _build_query(title, authors)
    try:
        async with _eric_limiter:
            async with ssrf_safe_client(timeout=15.0) as client:
                resp = await retry_on_transient(
                    lambda: client.get(
                        ERIC_SEARCH_API,
                        params={
                            "search": query,
                            "rows": "5",
                            "format": "json",
                            "fields": "id,title,author,publicationdateyear",
                        },
                        headers={
                            "User-Agent": _polite_user_agent(polite_email),
                            "Accept": "application/json",
                        },
                    ),
                    label="ERIC search",
                )
    except httpx.HTTPError as e:
        logger.debug("ERIC search failed for {!r}: {}", title[:60], e)
        return []

    if resp.status_code != 200:
        logger.debug("ERIC search returned status {}", resp.status_code)
        return []
    if len(resp.content) > _MAX_JSON_BYTES:
        logger.debug("ERIC response too large: {} bytes", len(resp.content))
        return []
    try:
        payload = resp.json()
    except ValueError as e:
        logger.debug("ERIC: could not parse JSON ({})", e)
        return []
    return _parse_eric_candidates(payload)


async def download_eric_pdf(
    pdf_url: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download a PDF from an ERIC ``files.eric.ed.gov/fulltext/`` URL."""
    if not pdf_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_eric.pdf"
    return await _download_pdf(pdf_url, dest, references_dir, "eric", evidence=evidence)
