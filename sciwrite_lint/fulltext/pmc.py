"""PubMed Central (PMC) lookup + PDF download."""

from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger

from sciwrite_lint._network import clean_and_validate_doi as _clean_and_validate_doi
from sciwrite_lint.fulltext._common import (
    PMC_ID_CONVERTER,
    AcquisitionResult,
    _pmc_limiter,
    _read_ncbi_key,
)
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._validation import BibEvidence
from sciwrite_lint.rate_limiter import rate_limited_get


async def lookup_pmc(doi: str) -> str | None:
    """Convert DOI to PMCID via NCBI ID Converter API. Returns PMCID or None."""
    if not doi:
        return None
    clean_doi = _clean_and_validate_doi(doi)
    if not clean_doi:
        return None
    params: dict[str, str] = {
        "ids": clean_doi,
        "format": "json",
        "tool": "sciwrite-lint",
    }
    ncbi_key = _read_ncbi_key()
    if ncbi_key:
        params["api_key"] = ncbi_key
    try:
        resp = await rate_limited_get(
            _pmc_limiter,
            PMC_ID_CONVERTER,
            params=params,
            timeout=5.0,
            label="PMC ID converter",
            service="pmc",
        )
        if resp.status_code != 200:
            return None
        records = resp.json().get("records") or []
        for rec in records:
            pmcid = rec.get("pmcid")
            if pmcid:
                return pmcid
    except httpx.HTTPError as e:
        logger.debug("PMC ID converter failed for DOI {}: {}", doi, e)
    return None


async def download_pmc_pdf(
    pmcid: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download PDF from PubMed Central."""
    if not pmcid:
        return AcquisitionResult(found=False)
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    dest = references_dir / f"{key}_pmc.pdf"
    return await _download_pdf(url, dest, references_dir, "pmc", evidence=evidence)
