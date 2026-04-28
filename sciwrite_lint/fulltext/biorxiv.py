"""bioRxiv / medRxiv direct PDF download."""

from __future__ import annotations

from pathlib import Path

from sciwrite_lint._network import clean_and_validate_doi as _clean_and_validate_doi
from sciwrite_lint.fulltext._common import AcquisitionResult
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._validation import BibEvidence


async def download_biorxiv_pdf(
    doi: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download from bioRxiv/medRxiv if DOI matches 10.1101/ prefix."""
    if not doi:
        return AcquisitionResult(found=False)
    clean_doi = _clean_and_validate_doi(doi)
    if not clean_doi or not clean_doi.startswith("10.1101/"):
        return AcquisitionResult(found=False)

    url = f"https://www.biorxiv.org/content/{clean_doi}v1.full.pdf"
    dest = references_dir / f"{key}_biorxiv.pdf"
    result = await _download_pdf(
        url, dest, references_dir, "biorxiv", evidence=evidence
    )
    if result.found:
        return result

    url = f"https://www.medrxiv.org/content/{clean_doi}v1.full.pdf"
    dest = references_dir / f"{key}_medrxiv.pdf"
    return await _download_pdf(url, dest, references_dir, "medrxiv", evidence=evidence)
