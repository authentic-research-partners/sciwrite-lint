"""Stage 4a: GROBID parse for downloaded reference PDFs.

The parse itself happens in ``reference_store.parse_all_missing``; this
stage wraps it and — unless the orchestrator asks to skip — runs the
embedding subprocess afterwards so each parsed paper has vectors in
``workspace.db`` before claim verification needs them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig

from sciwrite_lint.pipeline.embeddings import _run_embeddings_for_paper


async def _stage_parse(
    config: LintConfig,
    references_dir: Path,
    parse_sem: asyncio.Semaphore | None = None,
    skip_embeddings: bool = False,
    tex_path: Path | None = None,
) -> tuple[int, int]:
    """Parse unparsed PDFs via GROBID + build embeddings. Returns (parsed_count, cached_count).

    Args:
        skip_embeddings: If True, skip the embedding subprocess. Used by
            ``run_papers_staged()`` which runs embedding in a single batch
            subprocess across all papers (see ``_batch_embed``).
        tex_path: Path to .tex file for claim text extraction (optional).
    """
    from sciwrite_lint.references.reference_store import parse_all_missing

    results = await parse_all_missing(references_dir, sem=parse_sem)

    cached = sum(1 for v in results.values() if v == "cached")
    parsed = sum(1 for v in results.values() if v == "parsed")
    failed = sum(1 for v in results.values() if v == "failed")

    if not skip_embeddings:
        _run_embeddings_for_paper(results, references_dir, config, tex_path=tex_path)

    parts = []
    if parsed:
        parts.append(f"{parsed} new")
    if cached:
        parts.append(f"{cached} cached")
    if failed:
        parts.append(f"{failed} failed")
    if parts:
        logger.info("Parse: {}", ", ".join(parts))

    return parsed, cached
