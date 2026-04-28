"""Semantic Scholar openAccessPdf download."""

from __future__ import annotations

from pathlib import Path

from sciwrite_lint.fulltext._common import AcquisitionResult
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._validation import BibEvidence


async def download_s2_pdf(
    s2_pdf_url: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download PDF from Semantic Scholar openAccessPdf URL."""
    if not s2_pdf_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_s2.pdf"
    return await _download_pdf(
        s2_pdf_url,
        dest,
        references_dir,
        "semantic_scholar",
        evidence=evidence,
    )
