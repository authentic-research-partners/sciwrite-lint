"""NASA ADS (adsabs.harvard.edu) — astronomy database, JSON API, key required.

NASA ADS indexes ~17M astronomy/astrophysics/physics bibliographic
records. The search API requires an OAuth bearer token; 5,000
requests per day per token. Token: https://ui.adsabs.harvard.edu/user/
settings/token, stored at ``~/.sciwrite-lint/nasa_ads_api_key``.

Each search result carries an ``esources`` list identifying the kinds
of links ADS knows about for the paper (``EPRINT_PDF`` for arXiv,
``PUB_PDF`` for publisher, ``AUTHOR_PDF`` for author-hosted, etc.).
The actual PDF is served via the ``link_gateway/{bibcode}/{type}``
redirector. We prefer ``EPRINT_PDF`` (= arXiv, already OA) and fall
back to ``PUB_PDF`` / ``AUTHOR_PDF`` for papers that aren't on arXiv.

ADS supports server-side author filtering via the ``author:`` field in
Solr phrase form, and year filtering via ``year:``. When the bib
provides authors/year we include them in the query so less-relevant hits
never reach the ranker. Candidates still carry per-hit title/authors/year
so the ranker can corroborate the server-side filter with its own signals.

When no key is configured, ADS is skipped with a one-time INFO log
(not per-request) so batch runs don't produce noise.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import httpx
from loguru import logger

from sciwrite_lint._network import ssrf_safe_client
from sciwrite_lint.fulltext._common import (
    NASA_ADS_LINK_GATEWAY,
    NASA_ADS_SEARCH_API,
    AcquisitionResult,
    _MAX_JSON_BYTES,
    _escape_solr_query,
    _nasa_ads_limiter,
    _polite_user_agent,
    _read_nasa_ads_key,
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

# Order of preference for ADS esources — EPRINT first (arXiv, most
# reliable OA), then publisher/author copies.
_ADS_PDF_SOURCES_IN_ORDER = ("EPRINT_PDF", "PUB_PDF", "AUTHOR_PDF")

# Process-level flag to avoid spamming the "no NASA ADS key set" warning.
_nasa_ads_warning_emitted = False


def _parse_ads_candidates(payload: dict) -> list[SearchCandidate]:
    """Parse an ADS search response into link-gateway candidates.

    Each doc contributes at most one candidate: the first available
    ``esources`` entry matching :data:`_ADS_PDF_SOURCES_IN_ORDER` picks
    the esource (arXiv first, then publisher, then author-hosted). Docs
    without any acceptable esource are skipped entirely.
    """
    response = payload.get("response") or {}
    docs = response.get("docs") or []
    candidates: list[SearchCandidate] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        bibcode = doc.get("bibcode")
        if not isinstance(bibcode, str) or not bibcode:
            continue
        esources = doc.get("esources") or []
        if not isinstance(esources, list):
            continue
        esource = _pick_esource(esources)
        if esource is None:
            continue
        candidates.append(
            SearchCandidate(
                url=f"{NASA_ADS_LINK_GATEWAY}/{quote(bibcode, safe='')}/{esource}",
                title=_first_string(doc.get("title")),
                authors=_string_list(doc.get("author")),
                year=_coerce_year(doc.get("year")),
            )
        )
    return candidates


def _pick_esource(esources: list) -> str | None:
    """Return the first preferred esource present on the doc, else None."""
    for source in _ADS_PDF_SOURCES_IN_ORDER:
        if source in esources:
            return source
    return None


def _build_query(title: str, authors: list[str] | None, year: int | None) -> str:
    """Combine title, author, and year into an ADS Solr query.

    ``title:"..."`` is a phrase query against the title field (ADS
    supports phrase queries there). ``author:"..."`` filters by first
    bib surname. ``year:NNNN`` narrows by exact publication year.
    """
    q_parts = [f'title:"{_escape_solr_query(title.strip())}"']
    if authors:
        surname = _extract_surname(authors[0]).strip()
        if surname:
            q_parts.append(f'author:"{_escape_solr_query(surname)}"')
    if year is not None:
        q_parts.append(f"year:{year}")
    return " ".join(q_parts)


async def lookup_nasa_ads_by_title(
    title: str,
    polite_email: str = "",
    api_key: str | None = None,
    *,
    authors: list[str] | None = None,
    year: int | None = None,
) -> list[SearchCandidate]:
    """Search ADS by title (+ optional author/year). Return candidates.

    Requires an API key. Without one the source is skipped (a one-time
    INFO log is emitted so the user knows why). Returned URLs are public
    link-gateway endpoints which 302-redirect to the actual PDF host
    (arXiv, publisher, or author page). The SSRF-safe client follows
    redirects but blocks non-global IPs.
    """
    global _nasa_ads_warning_emitted
    if not title or not title.strip():
        return []

    key = api_key if api_key is not None else _read_nasa_ads_key()
    if not key:
        if not _nasa_ads_warning_emitted:
            logger.info(
                "NASA ADS API key not set — astronomy source will be skipped. "
                "Configure with: sciwrite-lint config set-key nasa-ads <TOKEN> "
                "(token: https://ui.adsabs.harvard.edu/user/settings/token)"
            )
            _nasa_ads_warning_emitted = True
        return []

    query = _build_query(title, authors, year)
    try:
        async with _nasa_ads_limiter:
            async with ssrf_safe_client(timeout=15.0) as client:
                resp = await retry_on_transient(
                    lambda: client.get(
                        NASA_ADS_SEARCH_API,
                        params={
                            "q": query,
                            "rows": "5",
                            "fl": "bibcode,title,author,year,esources,property",
                        },
                        headers={
                            "Authorization": f"Bearer {key}",
                            "User-Agent": _polite_user_agent(polite_email),
                            "Accept": "application/json",
                        },
                    ),
                    label="NASA ADS search",
                )
    except httpx.HTTPError as e:
        logger.debug("NASA ADS search failed for {!r}: {}", title[:60], e)
        return []

    if resp.status_code == 401:
        logger.warning(
            "NASA ADS returned 401 — check API key at ~/.sciwrite-lint/nasa_ads_api_key"
        )
        return []
    if resp.status_code == 429:
        logger.warning("NASA ADS daily quota (5000/day) exhausted — skipping")
        return []
    if resp.status_code != 200:
        logger.debug("NASA ADS returned status {}", resp.status_code)
        return []
    if len(resp.content) > _MAX_JSON_BYTES:
        logger.debug("NASA ADS response too large: {} bytes", len(resp.content))
        return []
    try:
        payload = resp.json()
    except ValueError as e:
        logger.debug("NASA ADS: could not parse JSON ({})", e)
        return []
    return _parse_ads_candidates(payload)


async def download_nasa_ads_pdf(
    link_gateway_url: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download a PDF from a NASA ADS link_gateway URL."""
    if not link_gateway_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_nasa_ads.pdf"
    return await _download_pdf(
        link_gateway_url,
        dest,
        references_dir,
        "nasa_ads",
        evidence=evidence,
    )
