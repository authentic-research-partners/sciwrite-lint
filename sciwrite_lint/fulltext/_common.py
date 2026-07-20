"""Shared constants, models, validators, key readers, and rate limiters."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from sciwrite_lint.rate_limiter import MonotonicRateLimiter

# Apache Solr "Standard Query Parser" reserved characters. Per the Solr
# Reference Guide (section: "The Standard Query Parser") these must be
# escaped with a backslash when used as literals in a query, otherwise
# Solr interprets them as operators (``+`` and ``-`` for required/
# prohibited terms, ``:`` for field delimiter, etc.) or rejects the
# query outright.
_SOLR_RESERVED_CHARS = frozenset('+-&|!(){}[]^"~*?:/\\')

CORE_BASE = "https://api.core.ac.uk/v3"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
PMC_ID_CONVERTER = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
NBER_SEARCH_API = "https://www.nber.org/api/v1/search"
IDEAS_SEARCH = "https://ideas.repec.org/cgi-bin/htsearch2"
HAL_SEARCH_API = "https://api.archives-ouvertes.fr/search/"
ERIC_SEARCH_API = "https://api.ies.ed.gov/eric/"
ERIC_PDF_BASE = "https://files.eric.ed.gov/fulltext"
OSF_PREPRINTS_API = "https://api.osf.io/v2/preprints/"
NASA_ADS_SEARCH_API = "https://api.adsabs.harvard.edu/v1/search/query"
NASA_ADS_LINK_GATEWAY = "https://ui.adsabs.harvard.edu/link_gateway"

_MIN_PDF_SIZE = 5000  # bytes — reject anything smaller
_MAX_PDF_SIZE = 100 * 1024 * 1024  # 100 MB — reject downloads larger than this
_MAX_HTML_BYTES = 10 * 1024 * 1024  # 10 MB — reject search pages larger than this
_MAX_JSON_BYTES = 2 * 1024 * 1024  # 2 MB — search-result JSON payloads


_DEFINITIVE_FAILURE_PHRASES: tuple[str, ...] = (
    "not found",
    "not available",
    "no match",
    "no pdf url",
    "no download url",
    "title mismatch",
    "too large",
    "unparseable",
)


def _classify_acquisition_failure(
    failed_sources: list[str],
) -> Literal["exhausted", "transient"]:
    """Classify a download failure for caching purposes.

    Returns ``"exhausted"`` iff at least one source was attempted and
    *every* attempt failed with a definitive reason (see
    ``_DEFINITIVE_FAILURE_PHRASES``). Returns ``"transient"`` when
    ``failed_sources`` is empty (no sources were tried — next run may
    have fresh metadata) or any entry indicates a retryable condition
    (network error, timeout, 5xx). The fetch pipeline caches and skips
    exhausted failures until the configured TTL expires; transient
    failures are always retried on the next run.
    """
    if not failed_sources:
        return "transient"
    for entry in failed_sources:
        entry_lc = entry.lower()
        if not any(phrase in entry_lc for phrase in _DEFINITIVE_FAILURE_PHRASES):
            return "transient"
    return "exhausted"


def _is_valid_pdf(data: bytes) -> bool:
    """Check if bytes start with PDF magic header and meet min/max size."""
    return _MIN_PDF_SIZE <= len(data) <= _MAX_PDF_SIZE and data[:5] == b"%PDF-"


def _polite_user_agent(polite_email: str) -> str:
    """User-Agent string for OA-source HTTP requests.

    When an email is configured, we include it as a ``mailto:`` hint so
    source operators can contact us about abusive or malformed traffic
    (the community norm for anonymous API access). Without an email we
    fall back to the plain product token — still identifies us as
    sciwrite-lint but gives operators no contact path.
    """
    if polite_email:
        return f"sciwrite-lint/1.0 (mailto:{polite_email})"
    return "sciwrite-lint/1.0"


def _escape_solr_query(s: str) -> str:
    """Backslash-escape Solr reserved characters in ``s``.

    Only characters in :data:`_SOLR_RESERVED_CHARS` are escaped; every
    other code point — Unicode letters (diacritics, CJK, Cyrillic, …),
    digits, whitespace, apostrophes, hyphens embedded in words — passes
    through unchanged. This is the correct escape strategy for any
    Solr-backed endpoint (HAL, ERIC, NASA ADS): it preserves the full
    title including non-English scripts, while neutralising any character
    that would otherwise be interpreted as a Solr operator.
    """
    if not s:
        return ""
    return "".join(f"\\{c}" if c in _SOLR_RESERVED_CHARS else c for c in s)


def _read_key_file(name: str) -> str | None:
    """Read an API key from ~/.sciwrite-lint/{name} if it exists."""
    key_path = Path.home() / ".sciwrite-lint" / name
    if key_path.exists():
        mode = key_path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            logger.warning(
                "API key file {} is readable by other users (mode {:o}). "
                "Run: chmod 600 {}",
                key_path,
                stat.S_IMODE(mode),
                key_path,
            )
        text = key_path.read_text().strip()
        return text if text else None
    return None


def _read_core_key() -> str | None:
    """Read CORE API key from ~/.sciwrite-lint/core_api_key."""
    return _read_key_file("core_api_key")


def _read_ncbi_key() -> str | None:
    """Read NCBI API key from ~/.sciwrite-lint/ncbi_api_key.

    With an NCBI API key, PMC rate limit increases from 3 to 10 req/s.
    Get one at: https://www.ncbi.nlm.nih.gov/account/settings/
    """
    return _read_key_file("ncbi_api_key")


def _read_s2_key() -> str | None:
    """Read Semantic Scholar API key from ~/.sciwrite-lint/s2_api_key.

    With an S2 API key, rate limit increases from 1 to 100 req/s.
    Request one at: https://www.semanticscholar.org/product/api#api-key
    """
    return _read_key_file("s2_api_key")


def _read_nasa_ads_key() -> str | None:
    """Read NASA ADS API token from ~/.sciwrite-lint/nasa_ads_api_key.

    NASA ADS requires an OAuth bearer token for all API calls. The free
    tier is capped at 5,000 requests per day per token. Without a token
    the NASA ADS source is skipped entirely (an INFO-level log records
    this once per process). Request one at:
    https://ui.adsabs.harvard.edu/user/settings/token
    """
    return _read_key_file("nasa_ads_api_key")


# Rate limiters: per-API, not per-download.
# PMC rate depends on whether NCBI API key is present (~/.sciwrite-lint/ncbi_api_key).
_core_limiter = MonotonicRateLimiter(1, 1.0)
_unpaywall_limiter = MonotonicRateLimiter(1, 0.1)
_pmc_limiter = MonotonicRateLimiter(
    10 if _read_ncbi_key() else 3,
    1.0,
)
_europepmc_limiter = MonotonicRateLimiter(10, 1.0)
# Title-search sources — no published rate limits on any of these; values
# below are conservative choices informed by either written policy
# ("be respectful"), community norms, or (for OSF) the throttle config in
# upstream source code.
#
# - NBER: no published limit; 1 req/s is polite for a Drupal site search.
# - IDEAS: RePEc "be respectful" policy; 1 req/s matches community norms.
# - HAL (archives-ouvertes.fr): no hard limit; docs reserve the right to
#   block abusive clients. 2 req/s is the community-recommended ceiling.
# - ERIC (IES Dept of Education): no rate limit stated; 2 req/s polite.
# - OSF: 1000/hour anonymous cap (`root-anon-throttle`, see CenterForOpen
#   Science/osf.io defaults.py). We target 5/min = 300/hour, well under
#   the cap, with `burst: 10/second` allowing short bursts if adjacent
#   requests happen to clump together.
# - NASA ADS: 5000/day cap per token (cannot be expressed as req/s
#   directly). 1 req/s prevents burst abuse; users must monitor the
#   daily quota themselves if linting many papers in a single day.
_nber_limiter = MonotonicRateLimiter(1, 1.0)
_ideas_limiter = MonotonicRateLimiter(1, 1.0)
_hal_limiter = MonotonicRateLimiter(2, 1.0)
_eric_limiter = MonotonicRateLimiter(2, 1.0)
_osf_limiter = MonotonicRateLimiter(5, 60.0)
_nasa_ads_limiter = MonotonicRateLimiter(1, 1.0)


class AcquisitionResult(BaseModel):
    """Result of a full-text acquisition attempt.

    ``status`` classifies the outcome for caching. ``"found"`` means a
    PDF was acquired and validated. ``"exhausted"`` means every source
    was tried and returned a *definitive* negative (not found / no match
    / title mismatch / too large) — safe to cache and skip on future
    runs within the TTL window. ``"transient"`` means the attempt was
    inconclusive (network error, timeout, or no sources to try yet) and
    must be retried on the next run. Only ``acquire_fulltext`` sets
    this field authoritatively; the default ``"transient"`` is the safe
    choice for intermediate ``_download_pdf`` results that never reach
    the fetch-pipeline cache layer.
    """

    found: bool
    source: str = ""
    local_path: str | None = None
    url: str | None = None
    abstract: str | None = None
    reason: str = ""
    is_oa: bool = False
    oa_url: str | None = None
    status: Literal["found", "exhausted", "transient"] = "transient"
    # Title-similarity score the match validator computed for the accepted
    # PDF. ``None`` when the validator was skipped (no evidence supplied) or
    # accepted without comparing titles (e.g. a DOI-only bib).
    title_sim: float | None = None
