"""Stages 4.6 + 5: bibliography verification and claim verification.

Bibliography verification (4.6) validates GROBID-extracted bib entries of
parsed formal references against external APIs — no GPU. Claim
verification (5) runs ``run_claim_verification`` (sem-001) and converts
the raw results into ``claim-support`` / ``cite-purpose`` findings.

``_collect_parse_hashes`` is the cache-invalidation key for the bib_checks
table: when a reference's parsed markdown changes, its cached bib result
is rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Finding


def _collect_parse_hashes(references_dir: Path) -> dict[str, str]:
    """Collect MD5 hashes of parsed markdown files for cache invalidation."""
    import hashlib

    parsed_dir = references_dir / "parsed"
    if not parsed_dir.exists():
        return {}
    hashes: dict[str, str] = {}
    for md_path in parsed_dir.glob("*.md"):
        content = md_path.read_bytes()
        hashes[md_path.stem] = hashlib.md5(content, usedforsecurity=False).hexdigest()
    return hashes


async def _stage_bib_verify(
    config: LintConfig,
    references_dir: Path,
    fresh: bool = False,
) -> list[Any]:
    """Verify bibliography entries of parsed formal references via APIs.

    Runs after parse (Stage 4) and concurrently with claim verification
    (Stage 5). Uses structured GROBID data stored in workspace.db at
    parse time. No GPU needed — API calls only.

    Results are cached in workspace.db (bib_checks table), invalidated
    when a reference's parsed markdown changes (hash mismatch).
    """
    from sciwrite_lint.references.workspace_db import (
        get_db,
        load_bib_checks as load_bib_checks_db,
        save_bib_checks,
    )
    from sciwrite_lint.references.chain import RefBibCheck

    with get_db(references_dir) as conn:
        # Compute parse hashes for cache invalidation
        parse_hashes = _collect_parse_hashes(references_dir)

        if not fresh:
            cached = load_bib_checks_db(conn, parse_hashes=parse_hashes)
            if cached:
                logger.info("Bibliography verification: {} cached results", len(cached))
                return [RefBibCheck(**c) for c in cached]

        from sciwrite_lint.references.chain import run_bib_verification

        results = await run_bib_verification(references_dir, config)

        # Cache in workspace.db with parse hashes
        if results:
            save_bib_checks(
                conn,
                [r.model_dump() for r in results],
                parse_hashes=parse_hashes,
            )

        return results


async def _stage_claims(
    paper_name: str,
    tex_path: Path,
    config: LintConfig,
    references_dir: Path,
    bib_path: Path | None = None,
    rerun: bool = False,
) -> tuple[list[Finding], list[dict]]:
    """Run claim verification. Returns (findings, raw_results)."""
    from sciwrite_lint.eval_claims import run_claim_verification

    results = await run_claim_verification(
        paper_name,
        tex_path,
        references_dir,
        config=config,
        bib_path=bib_path,
        model=config.llm_model or "",
        rerun=rerun,
    )

    return _claims_to_findings(results, tex_path), results


def _claims_to_findings(results: list[dict], tex_path: Path) -> list[Finding]:
    """Convert claim verification results to findings. Delegates to check modules."""
    from sciwrite_lint.checks.claim_support import claims_to_findings
    from sciwrite_lint.checks.cite_purpose import cite_purposes_to_findings

    findings = claims_to_findings(results, tex_path)
    findings.extend(cite_purposes_to_findings(results, tex_path))
    return findings
