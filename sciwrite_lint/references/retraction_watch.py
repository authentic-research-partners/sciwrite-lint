"""Retraction Watch database: download, cache, and lookup.

Downloads the Retraction Watch CSV from CrossRef Labs, caches it locally,
and provides DOI-keyed lookup for retraction status. The database covers
retractions, expressions of concern, corrections, and reinstatements.

Requires polite_email in config (used as mailto= parameter for download).
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from sciwrite_lint.config import LintConfig

_RW_URL = "https://api.labs.crossref.org/data/retractionwatch"
_CACHE_DIR = Path.home() / ".sciwrite-lint"
_CACHE_FILE = _CACHE_DIR / "retraction-watch.csv"
_MAX_CSV_SIZE = 200 * 1024 * 1024  # 200 MB — expected ~50 MB, cap for safety

# Module-level cache: parsed once per process
_cached_db: dict[str, RWEntry] | None = None
_cached_at: float = 0.0


@dataclass(frozen=True, slots=True)
class RWEntry:
    """A single entry from the Retraction Watch database."""

    doi: str
    nature: str  # "Retraction", "Expression of Concern", "Correction", "Reinstatement"
    reason: str
    date: str  # RetractionDate from CSV
    title: str


def _normalize_doi(doi: str) -> str:
    """Normalize a DOI for consistent lookup."""
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi.strip()


def _parse_csv(csv_text: str) -> dict[str, RWEntry]:
    """Parse Retraction Watch CSV into a DOI-keyed dict.

    When multiple entries exist for the same DOI (e.g. retraction then
    reinstatement), the latest entry wins — reinstatement clears retraction.
    """
    db: dict[str, RWEntry] = {}
    reader = csv.DictReader(io.StringIO(csv_text))

    for row in reader:
        raw_doi = row.get("OriginalPaperDOI", "").strip()
        if not raw_doi or raw_doi.lower() in ("", "unavailable", "n/a"):
            continue

        doi = _normalize_doi(raw_doi)
        if not doi:
            continue

        nature = row.get("RetractionNature", "").strip()
        if not nature:
            continue

        entry = RWEntry(
            doi=doi,
            nature=nature,
            reason=row.get("Reason", "").strip(),
            date=row.get("RetractionDate", "").strip(),
            title=row.get("Title", "").strip(),
        )

        # If DOI already in db, keep the entry with the latest date.
        # This handles retraction→reinstatement correctly.
        existing = db.get(doi)
        if existing is None or entry.date >= existing.date:
            db[doi] = entry

    logger.info(f"Retraction Watch database: {len(db)} entries loaded")
    return db


def _cache_age_hours() -> float:
    """Return age of cached CSV in hours, or inf if not cached."""
    if not _CACHE_FILE.exists():
        return float("inf")
    mtime = _CACHE_FILE.stat().st_mtime
    return (time.time() - mtime) / 3600


async def _download_csv(polite_email: str) -> str:
    """Download Retraction Watch CSV from CrossRef Labs."""
    import httpx

    from sciwrite_lint._network import stream_with_limit
    from sciwrite_lint.rate_limiter import retry_on_transient

    url = f"{_RW_URL}?mailto={polite_email}"
    logger.info(f"Downloading Retraction Watch database from {_RW_URL}")

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        resp = await retry_on_transient(
            lambda: stream_with_limit(client, url, _MAX_CSV_SIZE),
            label="Retraction Watch CSV",
        )
        resp.raise_for_status()

    csv_text = resp.text
    logger.info(f"Retraction Watch CSV downloaded: {len(csv_text)} bytes")

    # Cache to disk
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(csv_text, encoding="utf-8")
    logger.info(f"Cached Retraction Watch CSV to {_CACHE_FILE}")

    return csv_text


async def ensure_rw_database(config: LintConfig) -> dict[str, RWEntry]:
    """Load the Retraction Watch database, downloading if stale.

    Downloads the CSV if the cached file is older than config.rw_cache_hours
    (default 24). Parses into an in-memory dict keyed by normalized DOI.
    Cached in a module-level variable for reuse within the same process.

    Raises:
        RuntimeError: If polite_email is not configured.
        httpx.HTTPStatusError: If download fails.
    """
    global _cached_db, _cached_at

    # Return module-level cache if still valid
    cache_hours = config.rw_cache_hours
    if _cached_db is not None and (time.time() - _cached_at) / 3600 < cache_hours:
        return _cached_db

    if not config.polite_email:
        logger.warning(
            "Retraction Watch check skipped: polite_email not set in "
            ".sciwrite-lint.toml [api] section"
        )
        return {}

    # Check disk cache age
    if _cache_age_hours() < cache_hours:
        logger.debug("Loading Retraction Watch database from disk cache")
        csv_text = _CACHE_FILE.read_text(encoding="utf-8")
    else:
        csv_text = await _download_csv(config.polite_email)

    _cached_db = _parse_csv(csv_text)
    _cached_at = time.time()
    return _cached_db


def lookup_doi(db: dict[str, RWEntry], doi: str) -> RWEntry | None:
    """Look up a DOI in the Retraction Watch database."""
    if not doi:
        return None
    return db.get(_normalize_doi(doi))


def clear_cache() -> None:
    """Clear module-level cache (for testing)."""
    global _cached_db, _cached_at
    _cached_db = None
    _cached_at = 0.0
