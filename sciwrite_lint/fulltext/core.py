"""CORE (institutional repositories) search + PDF download."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from sciwrite_lint._network import clean_and_validate_doi as _clean_and_validate_doi
from sciwrite_lint.fulltext._common import (
    CORE_BASE,
    AcquisitionResult,
    _core_limiter,
    _read_core_key,
)
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._validation import BibEvidence
from sciwrite_lint.rate_limiter import rate_limited_get


async def search_core(
    doi: str,
    interval: float = 1.0,
) -> dict[str, Any] | None:
    """Look up a paper by DOI via CORE API."""
    if not doi:
        return None

    clean_doi = _clean_and_validate_doi(doi)
    if not clean_doi:
        return None
    core_key = _read_core_key()
    headers: dict[str, str] = {}
    if core_key:
        headers["Authorization"] = f"Bearer {core_key}"

    try:
        resp = await rate_limited_get(
            _core_limiter,
            f"{CORE_BASE}/search/works/",
            params={"q": f'doi:"{clean_doi}"', "limit": "1"},
            headers=headers,
            timeout=15.0,
            label="CORE",
            service="core",
        )
        if resp.status_code in (401, 403):
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        work = results[0]
        for candidate in results:
            if candidate.get("abstract"):
                work = candidate
                break

        abstract = work.get("abstract")
        if not abstract:
            return None

        authors = [
            a.get("name", "") for a in (work.get("authors") or []) if a.get("name")
        ]

        return {
            "abstract": abstract,
            "download_url": work.get("downloadUrl"),
            "authors": authors,
            "year": work.get("yearPublished"),
            "core_id": work.get("id"),
            "source": "CORE",
        }
    except httpx.HTTPError as e:
        logger.debug("CORE API lookup failed for DOI {}: {}", doi, e)
        return None


async def download_core_pdf(
    download_url: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download full text from CORE repository URL."""
    if not download_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_core.pdf"
    return await _download_pdf(
        download_url, dest, references_dir, "core", evidence=evidence
    )
