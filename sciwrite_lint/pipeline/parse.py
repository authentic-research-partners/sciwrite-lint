"""Stage 4a: GROBID parse for downloaded reference PDFs.

The parse itself happens in ``reference_store.parse_all_missing``; this
stage wraps it and — unless the orchestrator asks to skip — runs the
embedding subprocess afterwards so each parsed paper has vectors in
``workspace.db`` before claim verification needs them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from sciwrite_lint.config import LintConfig

from sciwrite_lint.pipeline.embeddings import _run_embeddings_for_paper

if TYPE_CHECKING:
    from sciwrite_lint.models import Finding


async def _stage_parse(
    config: LintConfig,
    references_dir: Path,
    parse_sem: asyncio.Semaphore | None = None,
    skip_embeddings: bool = False,
    tex_path: Path | None = None,
) -> tuple[int, int, list["Finding"], bool]:
    """Parse unparsed PDFs via GROBID + build embeddings.

    When ``skip_embeddings=False`` we pre-warm the embedder during
    parse: text vLLM is stopped before GROBID parse begins, the
    embedder subprocess starts and loads the model in parallel (~3-5 s),
    and after parse finishes we hand the keys off to the already-warm
    subprocess. This overlaps embedder model load with GROBID work
    instead of paying it serially after parse.

    On exit, the stage decides whether to restart text vLLM based on
    whether the *next* stage (cited vision) will need to stop it again
    immediately. The predicate (``_cited_needs_inference``) requires
    parsed ``.md`` files to be on disk, so it can only run after parse
    completes. When True, the stage skips its restart and signals the
    caller via the fourth return-tuple element.

    Returns ``(parsed_count, cached_count, findings, text_vllm_stopped)``:

    * ``findings`` — up to two system-bucket findings:

      - ``build_finding`` (rule_id ``llm-unavailable`` etc.) when
        ``persist_manuscript_citations`` surfaced a non-fatal warning
        while building the embedding context.
      - ``parse-failed`` summary when at least one cited PDF ended in
        ``"error"`` or ``"failed"`` status (per-key reasons in log).

      Both route to the report's ``system_issues`` bucket via
      ``split_findings`` and never affect manuscript scores.

    * ``text_vllm_stopped`` — True if the stage left text vLLM stopped
      (deferred restart because cited vision will swap immediately).
      The caller is then responsible for the eventual restart, either
      via ``_swap_to_text_vllm`` after cited vision or as a direct
      ``_restart_vllm_after_embedding`` if cited vision ends up
      skipping for some reason.

    Args:
        skip_embeddings: If True, skip the embedding subprocess. Used by
            ``run_papers_staged()`` which runs embedding in a single batch
            subprocess across all papers (see ``_batch_embed``). With
            this set, parse runs without the vLLM swap; the staged
            orchestrator manages swap timing itself.
        tex_path: Path to .tex file for claim text extraction (optional).
    """
    from sciwrite_lint.checks._diagnostics import parse_failed_finding
    from sciwrite_lint.pipeline.runners import EmbedderWarmer
    from sciwrite_lint.pipeline.swap import (
        _cited_needs_inference,
        _needs_embedding_swap,
        _restart_vllm_after_embedding,
        _stop_vllm_for_embedding,
    )
    from sciwrite_lint.references.reference_store import parse_all_missing

    # Pre-warm: stop text vLLM and launch the embedder subprocess so
    # its model load (~3-5 s) overlaps with GROBID parse.
    warmer: EmbedderWarmer | None = None
    if not skip_embeddings and _needs_embedding_swap(config):
        _stop_vllm_for_embedding(config)
        warmer = EmbedderWarmer(references_dir, config)
        warmer.start()

    findings: list[Finding] = []
    text_vllm_stopped = False
    try:
        results = await parse_all_missing(references_dir, sem=parse_sem)

        cached = sum(1 for v in results.values() if v == "cached")
        parsed = sum(1 for v in results.values() if v == "parsed")
        failed = sum(1 for v in results.values() if v == "failed")
        failed_keys = sorted(k for k, v in results.items() if v in ("error", "failed"))
        if failed_keys:
            findings.append(parse_failed_finding(failed_keys))

        if not skip_embeddings:
            build_finding = _run_embeddings_for_paper(
                results, references_dir, config, tex_path=tex_path, warmer=warmer
            )
            if build_finding is not None:
                findings.append(build_finding)
    except BaseException:
        # Parse / embed failed — the warmer is no longer useful. Kill
        # the subprocess so VRAM is released before the failure
        # propagates. ``cancel`` is idempotent and never raises.
        if warmer is not None:
            warmer.cancel()
        raise
    finally:
        # Restart text vLLM unless the next stage (cited vision swap)
        # will stop it again immediately — in that case
        # ``_swap_to_vision_vllm`` would waste a ~37 s text-vLLM warm-up.
        # ``_cited_needs_inference`` is computed here, AFTER parse, so
        # the just-parsed ``.md`` files are on disk for the predicate
        # to read; pre-parse the predicate would always return False on
        # a fresh workspace. Caller takes over the restart via the
        # ``text_vllm_stopped`` return flag.
        if warmer is not None:
            cited_will_swap = (
                not skip_embeddings
                and config.vision_backend == "vllm"
                and _cited_needs_inference(references_dir, fresh=False)
            )
            if cited_will_swap:
                text_vllm_stopped = True
                logger.info(
                    "Skipping text vLLM restart after embed — "
                    "cited vision swap will start vision next"
                )
            else:
                _restart_vllm_after_embedding(config)

    parts = []
    if parsed:
        parts.append(f"{parsed} new")
    if cached:
        parts.append(f"{cached} cached")
    if failed:
        parts.append(f"{failed} failed")
    if parts:
        logger.info("Parse: {}", ", ".join(parts))

    return parsed, cached, findings, text_vllm_stopped
