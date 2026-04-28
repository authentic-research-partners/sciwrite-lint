"""arXiv direct PDF download."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sciwrite_lint._network import is_valid_arxiv_id
from sciwrite_lint.fulltext._common import AcquisitionResult
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._validation import BibEvidence


async def download_arxiv_pdf(
    arxiv_id: str,
    key: str,
    references_dir: Path,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download PDF from arXiv."""
    if not arxiv_id:
        return AcquisitionResult(found=False)
    clean_id = arxiv_id.strip()
    if not is_valid_arxiv_id(clean_id):
        logger.debug("Invalid arXiv ID format, skipping: {}", clean_id[:40])
        return AcquisitionResult(found=False)
    url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    dest = references_dir / f"{key}_arxiv.pdf"
    return await _download_pdf(url, dest, references_dir, "arxiv", evidence=evidence)
