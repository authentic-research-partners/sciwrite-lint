"""Local PDF matching: use user-provided PDFs instead of downloading.

Users drop PDFs into local_pdfs_dir (e.g. paywalled content). Filenames
are fuzzy-matched against reference titles from the .bib. Matched PDFs
are copied into the paper workspace and treated identically to downloaded
PDFs — GROBID parsing, metadata checks, everything.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from loguru import logger

from sciwrite_lint.pdf.pdf_download import _title_similarity


# Threshold for filename↔title match. Higher than download title check
# (0.65) because filenames are user-controlled and should be close.
_MATCH_THRESHOLD = 0.80


def _normalize_filename(filename: str) -> str:
    """Normalize a PDF filename to a comparable string.

    Strips extension, replaces separators with spaces, lowercases,
    removes non-alphanumeric (except spaces).
    """
    name = Path(filename).stem
    # Replace common separators with spaces
    name = re.sub(r"[_\-\.]+", " ", name)
    # Remove non-alphanumeric except spaces
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    # Collapse whitespace
    return re.sub(r"\s+", " ", name).strip()


def match_local_pdfs(
    local_pdfs_dir: Path,
    titles: dict[str, str],
) -> tuple[dict[str, Path], list[Path]]:
    """Match local PDF files against reference titles.

    Args:
        local_pdfs_dir: Directory containing user-provided PDFs.
        titles: Mapping of citation key → reference title.

    Returns:
        (matched, unmatched) where:
        - matched: {citation_key: pdf_path} for successful matches
        - unmatched: list of PDF paths that didn't match any title
    """
    if not local_pdfs_dir.is_dir():
        return {}, []

    pdf_files = sorted(local_pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        return {}, []

    matched: dict[str, Path] = {}
    used_files: set[Path] = set()

    # For each PDF, find the best matching title
    for pdf_path in pdf_files:
        normalized = _normalize_filename(pdf_path.name)
        if not normalized:
            continue

        best_key = ""
        best_score = 0.0

        for key, title in titles.items():
            if key in matched:
                continue  # already matched to another file
            if not title:
                continue
            score = _title_similarity(normalized, title)
            if score > best_score:
                best_score = score
                best_key = key

        if best_score >= _MATCH_THRESHOLD and best_key:
            matched[best_key] = pdf_path
            used_files.add(pdf_path)
            logger.info(
                "Local PDF match: {} → {} (score={:.2f})",
                pdf_path.name,
                best_key,
                best_score,
            )
        else:
            if best_key:
                logger.debug(
                    "Local PDF no match: {} best={} (score={:.2f} < {:.2f})",
                    pdf_path.name,
                    best_key,
                    best_score,
                    _MATCH_THRESHOLD,
                )

    unmatched = [p for p in pdf_files if p not in used_files]
    return matched, unmatched


def copy_local_pdf(
    pdf_path: Path,
    key: str,
    references_dir: Path,
) -> str:
    """Copy a matched local PDF into the paper workspace.

    Returns the relative path (e.g. "smith2020_local.pdf") for storage
    in metadata access.local_file.
    """
    dest_name = f"{key}_local.pdf"
    dest = references_dir / dest_name
    references_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, dest)
    logger.info("Copied local PDF: {} → {}", pdf_path.name, dest_name)
    return dest_name
