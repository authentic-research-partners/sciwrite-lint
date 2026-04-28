"""NBER Working Papers — title search via JSON API.

NBER (nber.org) indexes US-economics working papers. The canonical QJE/AER
version is typically paywalled; the NBER WP precursor is openly downloadable
at a deterministic URL (``/system/files/working_papers/wNNNNN/wNNNNN.pdf``).
NBER's public site search is driven by a JSON API at
``/api/v1/search?q=...`` which returns results with ``type`` and ``url``
fields. We filter to ``type == "working_paper"`` and extract the WP number
from the ``url`` (``/papers/wNNNNN``).

Author filtering is best-effort — the API is undocumented publicly so we
append the bib surname to the ``q`` parameter rather than rely on a
dedicated field filter. The ranker corroborates author matches client-side
from per-candidate ``authors`` metadata when the API returns it.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
from loguru import logger

from sciwrite_lint._network import ssrf_safe_client
from sciwrite_lint.fulltext._common import (
    NBER_SEARCH_API,
    AcquisitionResult,
    _MAX_JSON_BYTES,
    _nber_limiter,
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

_NBER_WP_URL_RE = re.compile(r"^/papers/w(\d+)$")
_NBER_WP_ID_RE = re.compile(r"^\d{1,7}$")


def _parse_nber_candidates(payload: dict) -> list[SearchCandidate]:
    """Parse an NBER search response into working-paper candidates.

    The API returns mixed entry types (working_paper, chapter, person).
    Only ``working_paper`` entries whose ``url`` matches ``/papers/wNNNNN``
    qualify; the numeric ID drives the deterministic PDF URL on the
    nber.org CDN. Title and authors are taken from the entry when present;
    the ranker treats their absence as neutral.
    """
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    candidates: list[SearchCandidate] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "working_paper":
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        match = _NBER_WP_URL_RE.match(url.strip())
        if not match:
            continue
        wp_id = match.group(1)
        if not _NBER_WP_ID_RE.fullmatch(wp_id):
            continue
        candidates.append(
            SearchCandidate(
                url=(
                    f"https://www.nber.org/system/files/working_papers/"
                    f"w{wp_id}/w{wp_id}.pdf"
                ),
                title=_first_string(entry.get("title")),
                authors=_string_list(entry.get("authors")),
                year=_coerce_year(entry.get("year")),
            )
        )
    return candidates


def _build_query(title: str, authors: list[str] | None) -> str:
    """Combine title with an optional first-author surname for the ``q``
    parameter. NBER's Drupal-backed search weighs all query tokens so an
    extra surname biases relevance toward the bib's author."""
    tokens = [title.strip()] if title.strip() else []
    if authors:
        surname = _extract_surname(authors[0]).strip()
        if surname:
            tokens.append(surname)
    return " ".join(tokens)


async def lookup_nber_by_title(
    title: str,
    polite_email: str = "",
    *,
    authors: list[str] | None = None,
    year: int | None = None,  # noqa: ARG001  # carried on candidates
) -> list[SearchCandidate]:
    """Search NBER working papers. Return up to 20 ranked candidates."""
    if not title or not title.strip():
        return []
    query = _build_query(title, authors)
    try:
        async with _nber_limiter:
            async with ssrf_safe_client(timeout=15.0) as client:
                resp = await retry_on_transient(
                    lambda: client.get(
                        NBER_SEARCH_API,
                        params={"q": query, "perPage": "20"},
                        headers={
                            "User-Agent": _polite_user_agent(polite_email),
                            "Accept": "application/json",
                        },
                    ),
                    label="NBER search",
                )
    except httpx.HTTPError as e:
        logger.debug("NBER search failed for {!r}: {}", title[:60], e)
        return []

    if resp.status_code != 200:
        logger.debug("NBER search returned status {}", resp.status_code)
        return []
    if len(resp.content) > _MAX_JSON_BYTES:
        logger.debug(
            "NBER response too large: {} bytes (cap {})",
            len(resp.content),
            _MAX_JSON_BYTES,
        )
        return []
    try:
        payload = resp.json()
    except ValueError as e:
        logger.debug("NBER: could not parse JSON ({})", e)
        return []
    return _parse_nber_candidates(payload)


async def download_nber_pdf(
    pdf_url: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download a PDF from an NBER working-paper URL."""
    if not pdf_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_nber.pdf"
    return await _download_pdf(pdf_url, dest, references_dir, "nber", evidence=evidence)
