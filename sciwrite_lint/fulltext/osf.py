"""OSF Preprints (osf.io) — cross-discipline preprint server, JSON-API.

OSF hosts ~190k preprints spanning SocArXiv, PsyArXiv, ChemRxiv,
EarthArXiv, EngrXiv, MetaArXiv and a handful of other discipline-specific
servers. It's broad but thin — many refs are already indexed by
OpenAlex/Semantic Scholar, so OSF is checked last (priority 12, just
above CORE).

Two hops:
  1. ``/v2/preprints/?filter[title]=...``: returns preprints with
     ``attributes.title``, ``attributes.date_published``, and
     ``relationships.primary_file.links.related`` (the file API URL).
  2. GET that file API URL: returns ``links.download`` — a URL on osf.io
     that 302-redirects to the actual bytes.

OSF isn't Solr-backed; its JSON-API handles its own query syntax. Author
filtering server-side isn't cleanly supported (``filter[contributors]``
on the preprints endpoint is not documented to work reliably), so the
ranker discriminates on title + year using the metadata we do return.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger

from sciwrite_lint._network import ssrf_safe_client
from sciwrite_lint.fulltext._common import (
    OSF_PREPRINTS_API,
    AcquisitionResult,
    _MAX_JSON_BYTES,
    _osf_limiter,
    _polite_user_agent,
)
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._search import (
    SearchCandidate,
    _coerce_year,
)
from sciwrite_lint.fulltext._validation import BibEvidence
from sciwrite_lint.rate_limiter import retry_on_transient

# Number of top preprints to resolve via the file-detail hop. More gives
# the ranker more alternatives; each one costs an extra HTTP request.
_OSF_MAX_CANDIDATES = 3


def _parse_osf_preprints(payload: dict) -> list[tuple[str, str, int | None]]:
    """Parse hop-1 OSF preprints response.

    Returns a list of ``(file_api_url, title, year)`` for each preprint
    that has a resolvable ``primary_file`` link. Order is preserved from
    OSF's own relevance ranking.
    """
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    out: list[tuple[str, str, int | None]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rels = item.get("relationships") or {}
        primary_file = rels.get("primary_file") or {}
        related = (primary_file.get("links") or {}).get("related") or {}
        file_api_url = related.get("href")
        if not (
            isinstance(file_api_url, str)
            and file_api_url.startswith(("http://", "https://"))
        ):
            continue

        attrs = item.get("attributes") or {}
        title = attrs.get("title") or ""
        if not isinstance(title, str):
            title = ""
        year = _coerce_year(_extract_year_from_date(attrs.get("date_published")))
        out.append((file_api_url.strip(), title, year))
    return out


def _extract_year_from_date(date_field: object) -> str | None:
    """Pull the year out of an ISO-ish date string (``YYYY-MM-DD`` or
    ``YYYY``). Any other shape returns ``None``."""
    if not isinstance(date_field, str):
        return None
    stripped = date_field.strip()
    if len(stripped) >= 4 and stripped[:4].isdigit():
        return stripped[:4]
    return None


def _parse_osf_download_url(payload: dict) -> str | None:
    """Return the file's download URL from a hop-2 file detail response."""
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    if data.get("type") != "files":
        return None
    download = (data.get("links") or {}).get("download")
    if isinstance(download, str) and download.startswith(("http://", "https://")):
        return download.strip()
    return None


async def _fetch_file_detail(file_api_url: str, user_agent: str) -> dict | None:
    """Fetch the hop-2 file detail JSON. Returns parsed payload or None."""
    try:
        async with _osf_limiter:
            async with ssrf_safe_client(timeout=15.0) as client:
                resp = await retry_on_transient(
                    lambda: client.get(
                        file_api_url,
                        headers={
                            "User-Agent": user_agent,
                            "Accept": "application/vnd.api+json",
                        },
                    ),
                    label="OSF file detail",
                )
    except httpx.HTTPError as e:
        logger.debug("OSF file detail failed: {}", e)
        return None
    if resp.status_code != 200:
        logger.debug("OSF file detail returned status {}", resp.status_code)
        return None
    if len(resp.content) > _MAX_JSON_BYTES:
        logger.debug("OSF file detail too large: {} bytes", len(resp.content))
        return None
    try:
        return resp.json()
    except ValueError as e:
        logger.debug("OSF file detail: could not parse JSON ({})", e)
        return None


async def lookup_osf_by_title(
    title: str,
    polite_email: str = "",
    *,
    authors: list[str] | None = None,  # noqa: ARG001  # no server-side filter
    year: int | None = None,  # noqa: ARG001  # carried on candidates
) -> list[SearchCandidate]:
    """Search OSF Preprints by title. Return up to 3 ranked candidates.

    The first hop uses OSF's ``filter[title]`` for server-side narrowing.
    The second hop resolves the primary-file link for up to
    :data:`_OSF_MAX_CANDIDATES` preprints so the ranker has alternatives
    when the top OSF hit is the wrong paper.
    """
    if not title or not title.strip():
        return []
    user_agent = _polite_user_agent(polite_email)

    try:
        async with _osf_limiter:
            async with ssrf_safe_client(timeout=15.0) as client:
                resp = await retry_on_transient(
                    lambda: client.get(
                        OSF_PREPRINTS_API,
                        params={
                            "filter[title]": title,
                            "page[size]": "5",
                        },
                        headers={
                            "User-Agent": user_agent,
                            "Accept": "application/vnd.api+json",
                        },
                    ),
                    label="OSF search",
                )
    except httpx.HTTPError as e:
        logger.debug("OSF search failed for {!r}: {}", title[:60], e)
        return []
    if resp.status_code != 200:
        logger.debug("OSF search returned status {}", resp.status_code)
        return []
    if len(resp.content) > _MAX_JSON_BYTES:
        logger.debug("OSF response too large: {} bytes", len(resp.content))
        return []
    try:
        payload = resp.json()
    except ValueError as e:
        logger.debug("OSF: could not parse JSON ({})", e)
        return []

    preprints = _parse_osf_preprints(payload)
    candidates: list[SearchCandidate] = []
    for file_api_url, preprint_title, preprint_year in preprints[:_OSF_MAX_CANDIDATES]:
        detail = await _fetch_file_detail(file_api_url, user_agent)
        if detail is None:
            continue
        download_url = _parse_osf_download_url(detail)
        if not download_url:
            continue
        candidates.append(
            SearchCandidate(
                url=download_url,
                title=preprint_title,
                authors=[],
                year=preprint_year,
            )
        )
    return candidates


async def download_osf_pdf(
    pdf_url: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download a PDF from an OSF download URL."""
    if not pdf_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_osf.pdf"
    return await _download_pdf(pdf_url, dest, references_dir, "osf", evidence=evidence)
