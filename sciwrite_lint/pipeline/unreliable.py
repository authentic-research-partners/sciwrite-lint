"""Stage 6: aggregate reliability signals into reference-unreliable findings.

Uses claim results (deep path) when available, otherwise metadata only.
Bibliography checks are passed in from Stage 4.6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sciwrite_lint.models import Finding


def _stage_unreliable(
    tex_path: Path,
    references_dir: Path,
    claim_results: list[dict],
    bib_checks: list[Any] | None = None,
) -> list[Finding]:
    """Aggregate reliability signals into reference-unreliable findings.

    Uses claim results (deep path) when available, otherwise metadata only.
    Bibliography checks are passed from the pipeline (Stage 4.6).
    """
    from sciwrite_lint.checks.reference_unreliable import (
        claims_to_unreliable_findings,
        metadata_to_unreliable_findings,
    )
    from sciwrite_lint.references.metadata import load_all_metadata

    all_meta = load_all_metadata(references_dir)
    if not all_meta:
        return []

    if claim_results:
        return claims_to_unreliable_findings(
            claim_results,
            tex_path,
            metadata_map=all_meta,
            bib_checks=bib_checks,
        )
    return metadata_to_unreliable_findings(
        all_meta,
        tex_path,
        bib_checks=bib_checks,
    )
