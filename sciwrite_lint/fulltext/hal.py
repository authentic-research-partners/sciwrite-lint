"""HAL (archives-ouvertes.fr) — French national open archive, JSON API.

HAL is a general-purpose open archive for French research (CS, math,
physics, HSS) indexing ~4.5M documents. It exposes a Solr JSON API. A
single search returns ``fileMain_s`` as a direct PDF URL on the HAL
domain, together with ``openAccess_bool`` which we use to skip closed
entries. This is the cleanest title-search source of all — one request,
one field, one URL, no HTML parsing.

Author and year filtering are supported server-side via the Solr fields
``authFullName_s`` and ``producedDateY_i``; when the caller supplies a
bib surname, we add it to the query so less-relevant hits never reach
the ranker. Candidates still carry title/authors/year so the ranker can
corroborate the server-side filter with its own signals.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger

from sciwrite_lint._network import ssrf_safe_client
from sciwrite_lint.fulltext._common import (
    HAL_SEARCH_API,
    AcquisitionResult,
    _MAX_JSON_BYTES,
    _escape_solr_query,
    _hal_limiter,
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


def _parse_hal_candidates(payload: dict) -> list[SearchCandidate]:
    """Parse a HAL Solr response into a list of candidates.

    Filters to OA-flagged entries only — non-OA entries' ``fileMain_s``,
    when present, points at landing pages or embargoed full text rather
    than at a downloadable PDF. ``title_s`` and ``authFullName_s`` come
    back as arrays in Solr's response; the first element of each is the
    canonical value.
    """
    response = payload.get("response") or {}
    docs = response.get("docs") or []
    candidates: list[SearchCandidate] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if not doc.get("openAccess_bool"):
            continue
        url = doc.get("fileMain_s")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        candidates.append(
            SearchCandidate(
                url=url.strip(),
                title=_first_string(doc.get("title_s")),
                authors=_string_list(doc.get("authFullName_s")),
                year=_coerce_year(doc.get("producedDateY_i")),
            )
        )
    return candidates


def _build_query(title: str, authors: list[str] | None) -> str:
    """Combine a title with an optional first-author surname filter.

    The title goes to the default text field — Solr's analyzer tokenises
    and matches against all indexed title subfields. When bib authors are
    provided the first author's surname is added as an
    ``authFullName_s:"..."`` clause so non-matching authors are dropped
    server-side.

    Both values are run through :func:`_escape_solr_query` so diacritics,
    CJK scripts, and other non-ASCII characters pass through unchanged
    while Solr-reserved chars get backslash-escaped.
    """
    escaped_title = _escape_solr_query(title.strip())
    q_parts = [escaped_title] if escaped_title else []
    if authors:
        surname = _extract_surname(authors[0]).strip()
        if surname:
            q_parts.append(f'authFullName_s:"{_escape_solr_query(surname)}"')
    return " AND ".join(q_parts) if q_parts else ""


async def lookup_hal_by_title(
    title: str,
    polite_email: str = "",
    *,
    authors: list[str] | None = None,
    year: int | None = None,  # noqa: ARG001  # carried on candidates; ranker uses it
) -> list[SearchCandidate]:
    """Search HAL by title (+ optional author surname). Return candidates.

    :param title: Title tokens drive the main text query.
    :param polite_email: Contact email for the ``mailto:`` User-Agent.
    :param authors: Bib author names; when non-empty, the first surname
        is added as an ``authFullName_s`` server-side filter.
    :param year: Reserved for future server-side year filters; currently
        unused here — the ranker uses ``candidate.year`` itself.
    :returns: Up to 5 candidates, newest-relevant first per HAL's own
        scoring. Empty list on transport/parse errors.
    """
    if not title or not title.strip():
        return []
    query = _build_query(title, authors)
    if not query:
        return []

    try:
        async with _hal_limiter:
            async with ssrf_safe_client(timeout=15.0) as client:
                resp = await retry_on_transient(
                    lambda: client.get(
                        HAL_SEARCH_API,
                        params={
                            "q": query,
                            "rows": "5",
                            "wt": "json",
                            "fl": (
                                "openAccess_bool,fileMain_s,title_s,"
                                "authFullName_s,producedDateY_i"
                            ),
                        },
                        headers={
                            "User-Agent": _polite_user_agent(polite_email),
                            "Accept": "application/json",
                        },
                    ),
                    label="HAL search",
                )
    except httpx.HTTPError as e:
        logger.debug("HAL search failed for {!r}: {}", title[:60], e)
        return []

    if resp.status_code != 200:
        logger.debug("HAL search returned status {}", resp.status_code)
        return []
    if len(resp.content) > _MAX_JSON_BYTES:
        logger.debug("HAL response too large: {} bytes", len(resp.content))
        return []
    try:
        payload = resp.json()
    except ValueError as e:
        logger.debug("HAL: could not parse JSON ({})", e)
        return []
    return _parse_hal_candidates(payload)


async def download_hal_pdf(
    pdf_url: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download a PDF from a HAL ``fileMain_s`` URL."""
    if not pdf_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_hal.pdf"
    return await _download_pdf(pdf_url, dest, references_dir, "hal", evidence=evidence)
