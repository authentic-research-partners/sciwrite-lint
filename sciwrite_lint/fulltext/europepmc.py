"""Europe PMC lookup + PDF download."""

from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger

from sciwrite_lint._network import clean_and_validate_doi as _clean_and_validate_doi
from sciwrite_lint.fulltext._common import (
    EUROPEPMC_BASE,
    AcquisitionResult,
    _europepmc_limiter,
)
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._validation import BibEvidence
from sciwrite_lint.rate_limiter import rate_limited_get


async def lookup_europepmc(
    doi: str,
    pmcid: str | None = None,
) -> str | None:
    """Look up paper on Europe PMC. Returns PDF URL or None."""
    if pmcid:
        return (
            f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        )

    if not doi:
        return None
    clean_doi = _clean_and_validate_doi(doi)
    if not clean_doi:
        return None

    try:
        resp = await rate_limited_get(
            _europepmc_limiter,
            f"{EUROPEPMC_BASE}/search",
            params={
                "query": f"DOI:{clean_doi}",
                "format": "json",
                "resultType": "core",
            },
            label="Europe PMC",
            service="europepmc",
        )
        if resp.status_code != 200:
            return None
        results = (resp.json().get("resultList") or {}).get("result") or []
        for result in results:
            for url_info in result.get("fullTextUrlList", {}).get("fullTextUrl", []):
                if url_info.get("documentStyle") == "pdf":
                    return url_info.get("url")
            epmc_pmcid = result.get("pmcid")
            if epmc_pmcid:
                return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={epmc_pmcid}&blobtype=pdf"
    except httpx.HTTPError as e:
        logger.debug("Europe PMC lookup failed for DOI {}: {}", doi, e)
    return None


async def download_europepmc_pdf(
    pdf_url: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download PDF from Europe PMC."""
    if not pdf_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_europepmc.pdf"
    return await _download_pdf(
        pdf_url, dest, references_dir, "europepmc", evidence=evidence
    )
