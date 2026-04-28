"""Batch-staged orchestrator for multi-paper runs.

Complement to :func:`run_full_check` in ``orchestration.py``. Where the
single-paper orchestrator runs stages sequentially for one paper,
``run_papers_staged`` runs each stage across **all** papers before
moving on:

- GPU stages (vision, embedding, cited vision) spawn a single
  subprocess that processes every paper — the model loads once.
- Non-GPU stages (vLLM checks, verify, fetch, GROBID, ref_internal,
  claims) run up to ``concurrency`` papers concurrently.

Used by ``sciwrite-lint check`` when 2+ papers are in scope, by the
real-world eval, and by calibration.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.config import LintConfig, PaperConfig
from sciwrite_lint.models import Citation, Finding
from sciwrite_lint.references.workspace_db import (
    get_db,
    init_pipeline_stages,
)

from sciwrite_lint.pipeline.checks import run_llm_checks_batched, run_text_checks
from sciwrite_lint.pipeline.claims import _stage_bib_verify, _stage_claims
from sciwrite_lint.pipeline.embeddings import _extract_claim_texts
from sciwrite_lint.pipeline.fetch import _stage_fetch
from sciwrite_lint.pipeline.orchestration import _backup_workspace
from sciwrite_lint.pipeline.pdf_context import citations_from_pdf_context
from sciwrite_lint.pipeline.ref_internal import _stage_ref_internal
from sciwrite_lint.pipeline.runners import (
    _batch_cited_vision,
    _batch_embed,
    _batch_vision,
)
from sciwrite_lint.pipeline.swap import (
    _cited_needs_inference,
    _manuscript_needs_inference,
    _needs_embedding_swap,
    _restart_vllm_after_embedding,
    _stop_vllm_for_embedding,
    _swap_to_text_vllm,
    _swap_to_vision_vllm,
)
from sciwrite_lint.pipeline.tracking import (
    _stage_failure_guard,
    _stage_tracking,
    _track,
)
from sciwrite_lint.pipeline.unreliable import _stage_unreliable
from sciwrite_lint.pipeline.verify import _stage_verify


@dataclass
class StagedPaperResult:
    """Result from ``run_papers_staged`` for one paper.

    Contains all pipeline outputs needed for scoring: findings from all
    stages (text, LLM, verify, claims, unreliable), raw claim verification
    results, and bibliography checks. Used by the real-world eval and
    calibration eval to compute SciLint Scores.
    """

    paper_name: str
    findings: list[Finding]
    claim_results: list[dict]
    bib_checks: list[Any]
    error: str | None = None


@dataclass
class _PaperCtx:
    """Per-paper mutable state carried across stages in ``run_papers_staged``.

    Each paper gets one context object at setup. Stages mutate it
    (appending findings, storing claim results, etc.) as they complete.
    """

    name: str
    tex_path: Path
    pc: PaperConfig
    config: LintConfig
    refs_dir: Path
    citations: list[Citation]
    check_findings: list[Finding]
    verify_findings: list[Finding]
    claim_findings: list[Finding]
    claim_results: list[dict]
    bib_checks: list[Any]
    ref_figure_descs: dict[str, str]
    run: Any  # RunStats from usage.py
    error: str | None = None


async def run_papers_staged(
    papers: list[tuple[str, Path, PaperConfig, LintConfig]],
    fresh: bool = False,
    concurrency: int = 0,
) -> list[StagedPaperResult]:
    """Run multiple papers through the pipeline with batch-by-stage orchestration.

    GPU stages (vision, embedding, cited vision) run in a single subprocess
    per batch — the model loads once and processes all papers sequentially.
    Non-GPU stages (vLLM checks, verify, fetch, GROBID, ref_internal, claims)
    run up to ``concurrency`` papers concurrently (0 = unlimited).

    Args:
        papers: List of (paper_name, tex_path, PaperConfig, LintConfig) tuples.
        fresh: Start from scratch for all papers.
        concurrency: Max papers in non-GPU concurrent stages. 0 = no limit.

    Returns:
        List of StagedPaperResult, one per paper.
    """
    from sciwrite_lint.usage import end_run, save_run_async, set_current, start_run

    if not papers:
        return []

    if concurrency < 0:
        raise ValueError(f"concurrency must be >= 0, got {concurrency}")

    if concurrency > 0 and concurrency >= len(papers):
        logger.info(
            "concurrency={} >= {} papers — all papers will run concurrently",
            concurrency,
            len(papers),
        )

    # Soft warning for high concurrency. On the single-GPU + single-vLLM-
    # server setup we ship with, validated baseline is up to 4 concurrent
    # papers. Above 4, the live monitor (which polls each workspace.db
    # plus the shared usage.db) saturates and can fail to surface stage
    # progress, vLLM queue depth grows, API rate limiters bite, and GROBID
    # memory pressure can stall parses. Larger numbers may still work on
    # bigger hardware but are not part of the validated baseline.
    _CONCURRENCY_TESTED_MAX = 4
    if concurrency > _CONCURRENCY_TESTED_MAX:
        logger.warning(
            "concurrency={} exceeds tested maximum ({}). Above this, the "
            "live monitor may fail to surface progress, vLLM queue depth "
            "grows, and API rate limits / GROBID memory may degrade "
            "throughput. The setting is allowed but not part of the "
            "validated baseline.",
            concurrency,
            _CONCURRENCY_TESTED_MAX,
        )

    # ------------------------------------------------------------------
    # Setup: workspace, citations, usage tracking for each paper
    # ------------------------------------------------------------------
    ctxs: list[_PaperCtx] = []
    for paper_name, tex_path, pc, config in papers:
        ws = config.paper_workspace(paper_name)
        if not fresh:
            ok, reason = ws.check_source(tex_path, bib_path=pc.bib)
            if not ok:
                raise RuntimeError(
                    f"[{paper_name}] Source type changed ({reason}). "
                    "Run with --fresh to start over."
                )
        if fresh:
            _backup_workspace(ws)
        ws.ensure_dirs()
        refs_dir = ws.root
        config.current_paper = paper_name

        # Register in usage.db + init stages BEFORE doing work, so the
        # monitor can show "setup running" immediately.
        run = start_run(
            paper=paper_name,
            model=config.llm_model or "",
            workspace_root=str(refs_dir),
        )
        with get_db(refs_dir) as conn:
            init_pipeline_stages(conn)
        _track(refs_dir, "setup", "running")

        # Now do the actual setup work (save source, extract citations)
        ws.save_source(tex_path, bib_path=pc.bib)

        if config.is_pdf:
            citations = citations_from_pdf_context(config)
        else:
            from sciwrite_lint.footnote_urls import ingest_footnote_sources
            from sciwrite_lint.references.citations import (
                check_local_sources,
                extract_bibitems,
                filter_to_cited,
            )

            citations = extract_bibitems(tex_path, "auto", bib_path=pc.bib)
            n_bib = len(citations)
            aux_path = tex_path.with_suffix(".aux")
            citations = filter_to_cited(
                citations,
                tex_path,
                aux_path=aux_path if aux_path.exists() else None,
            )
            if n_bib > len(citations):
                logger.info(
                    "[{}] {}/{} bib entries are cited — "
                    "skipping verify+fetch for {} uncited",
                    paper_name,
                    len(citations),
                    n_bib,
                    n_bib - len(citations),
                )
            check_local_sources(citations, refs_dir)

            # Footnote-URL sources: each \footnote{\url{URL}} whose URL
            # appears in a local_web_dir capture's Source: header becomes
            # a synthetic T1 Citation, pre-registered to short-circuit
            # verify + fetch.
            footnote_citations = ingest_footnote_sources(
                tex_path,
                config.effective_local_web_dir(paper_name),
                refs_dir,
                source_paper=tex_path.stem,
            )
            citations.extend(footnote_citations)

        logger.info("[{}] {} citations extracted", paper_name, len(citations))
        run.citations = len(citations)
        _track(refs_dir, "setup", "done", f"{len(citations)} citations")

        ctxs.append(
            _PaperCtx(
                name=paper_name,
                tex_path=tex_path,
                pc=pc,
                config=config,
                refs_dir=refs_dir,
                citations=citations,
                check_findings=[],
                verify_findings=[],
                claim_findings=[],
                claim_results=[],
                bib_checks=[],
                ref_figure_descs={},
                run=run,
            )
        )

    t0 = time.monotonic()

    # Semaphore for non-GPU concurrent stages (0 = unlimited)
    paper_sem: asyncio.Semaphore | None = None
    if concurrency > 0:
        paper_sem = asyncio.Semaphore(concurrency)

    def _active() -> list[_PaperCtx]:
        """Papers that haven't failed yet."""
        return [ctx for ctx in ctxs if ctx.error is None]

    # Use first paper's config for vision backend (all papers share config)
    vision_backend_is_vllm = ctxs[0].config.vision_backend == "vllm"
    vllm_vision_up = False

    try:
        # ------------------------------------------------------------------
        # Stage 0.5: BATCH VISION — one subprocess, all papers
        # ------------------------------------------------------------------
        logger.info("=== Stage 0.5: Batch vision ({} papers) ===", len(ctxs))

        # Swap to vision vLLM only if at least one paper has actual VL
        # inference work (uncached raster OR any TikZ figures). Cached
        # reruns skip the swap. When no swap happens the batch subprocess
        # reads from cache and never calls vLLM.
        if vision_backend_is_vllm:
            any_needs = False
            for ctx in ctxs:
                ws_obj = ctx.config.paper_workspace(ctx.name)
                if _manuscript_needs_inference(
                    ctx.tex_path, ctx.refs_dir, ws_obj, fresh=fresh
                ):
                    any_needs = True
                    break
            if any_needs:
                await _swap_to_vision_vllm(ctxs[0].config)
                vllm_vision_up = True
            else:
                logger.info(
                    "Vision: all manuscript figures cached — skipping vision vLLM swap"
                )

        vision_manifest = [
            {
                "paper_name": ctx.name,
                "tex_path": str(ctx.tex_path),
                "config_path": str(ctx.config.config_path)
                if ctx.config.config_path
                else None,
                "fresh": fresh,
                "backend": ctx.config.vision_backend,
                "device": ctx.config.vision_device,
            }
            for ctx in ctxs
        ]
        with _stage_tracking([ctx.refs_dir for ctx in ctxs], "vision"):
            _batch_vision(vision_manifest)

        if vllm_vision_up:
            await _swap_to_text_vllm(ctxs[0].config)
            vllm_vision_up = False

        # ------------------------------------------------------------------
        # Stages 1+2: TEXT + LLM + VERIFY — all papers concurrent
        # ------------------------------------------------------------------
        logger.info("=== Stages 1+2: Checks + verify ({} papers) ===", len(ctxs))

        async def _checks_verify(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                with _stage_tracking(
                    ctx.refs_dir, ["text_checks", "llm_checks", "verify"]
                ):
                    loop = asyncio.get_event_loop()
                    text_task = loop.run_in_executor(
                        None, run_text_checks, ctx.tex_path, ctx.config
                    )
                    llm_task = run_llm_checks_batched(ctx.tex_path, ctx.config)
                    verify_task = _stage_verify(
                        ctx.citations, ctx.config, ctx.refs_dir, fresh=fresh
                    )
                    text_findings, llm_findings, verify_findings = await asyncio.gather(
                        text_task,
                        llm_task,
                        verify_task,
                    )
                    ctx.check_findings = list(text_findings) + list(llm_findings)
                    ctx.verify_findings = list(verify_findings)
            except Exception as e:
                logger.error("[{}] Checks+verify failed: {}", ctx.name, e)
                ctx.error = str(e)
            finally:
                if paper_sem:
                    paper_sem.release()

        await asyncio.gather(*[_checks_verify(ctx) for ctx in _active()])
        t_checks = time.monotonic() - t0
        logger.info("Checks + verify: {:.1f}s", t_checks)

        # ------------------------------------------------------------------
        # Stage 3: FETCH — all papers concurrent
        # ------------------------------------------------------------------
        logger.info("=== Stage 3: Fetch ({} papers) ===", len(ctxs))

        async def _fetch_paper(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                with _stage_tracking(ctx.refs_dir, "fetch") as st:
                    fetched = await _stage_fetch(
                        ctx.citations, ctx.config, ctx.refs_dir, embed_inline=False
                    )
                    st.detail = f"{fetched} downloaded"
            except Exception as e:
                logger.error("[{}] Fetch failed: {}", ctx.name, e)
                ctx.error = str(e)
            finally:
                if paper_sem:
                    paper_sem.release()

        await asyncio.gather(*[_fetch_paper(ctx) for ctx in _active()])

        # ------------------------------------------------------------------
        # Stage 4a: GROBID PARSE — concurrent with shared memory monitor
        # ------------------------------------------------------------------
        logger.info("=== Stage 4a: GROBID parse ({} papers) ===", len(ctxs))
        from sciwrite_lint.pdf.grobid import (
            MAX_PARSE_CONCURRENCY,
            monitor_container_memory,
        )

        parse_sem = asyncio.Semaphore(MAX_PARSE_CONCURRENCY)
        mem_monitor = asyncio.create_task(monitor_container_memory(parse_sem))

        # Store parse results for each paper (needed to collect embedding keys)
        parse_results_map: dict[str, dict[str, str]] = {}

        async def _parse_paper(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                _track(ctx.refs_dir, "parse", "running")
                from sciwrite_lint.references.reference_store import parse_all_missing

                pr = await parse_all_missing(ctx.refs_dir, sem=parse_sem)
                parse_results_map[ctx.name] = pr
                # Keep stage "running" — batch embedding runs after all papers
                # finish parsing, and its timing must be attributed to this stage.
                # Stage is marked "done" after `_batch_embed` completes below.
            except Exception as e:
                logger.error("[{}] Parse failed: {}", ctx.name, e)
                ctx.error = str(e)
                _track(ctx.refs_dir, "parse", "failed", str(e))
            finally:
                if paper_sem:
                    paper_sem.release()

        try:
            await asyncio.gather(*[_parse_paper(ctx) for ctx in _active()])
        finally:
            mem_monitor.cancel()

        # ------------------------------------------------------------------
        # Stage 4b: BATCH EMBEDDING — one subprocess, all papers
        # ------------------------------------------------------------------
        logger.info("=== Stage 4b: Batch embedding ({} papers) ===", len(ctxs))
        from sciwrite_lint.references.embedding_store import has_embeddings
        from sciwrite_lint.references.metadata import load_all_metadata
        from sciwrite_lint.references.reference_store import _get_embedding_config

        model_name, _, _ = _get_embedding_config()

        embed_manifest: list[dict[str, Any]] = []
        for ctx in _active():
            keys_to_embed: list[str] = []

            # Scan ALL parsed markdown files — catches refs parsed during
            # fetch (eager parse_and_embed) as well as _stage_parse.
            parsed_dir = ctx.refs_dir / "parsed"
            if parsed_dir.exists():
                for md_file in sorted(parsed_dir.glob("*.md")):
                    key = md_file.stem
                    if not has_embeddings(key, ctx.refs_dir, model_name=model_name):
                        keys_to_embed.append(key)

            # Web summary keys
            all_meta = load_all_metadata(ctx.refs_dir)
            for key, meta in all_meta.items():
                local_file = meta.access.get("local_file") or ""
                if not local_file.endswith("_web.md"):
                    continue
                if has_embeddings(key, ctx.refs_dir, model_name=model_name):
                    continue
                web_path = ctx.refs_dir / local_file
                if web_path.exists():
                    keys_to_embed.append(key)

            # Extract claim texts for query vector pre-computation.
            # The embedding subprocess encodes these alongside document
            # chunks so Stage 5 never loads the model in the parent process.
            claim_texts = _extract_claim_texts(ctx.config, ctx.tex_path)

            if keys_to_embed or claim_texts:
                embed_manifest.append(
                    {
                        "keys": keys_to_embed,
                        "references_dir": str(ctx.refs_dir),
                        "claim_texts": claim_texts,
                    }
                )

        batch_embed_swap = bool(embed_manifest) and _needs_embedding_swap(
            ctxs[0].config
        )
        if batch_embed_swap:
            _stop_vllm_for_embedding(ctxs[0].config)

        with _stage_failure_guard([ctx.refs_dir for ctx in _active()], "parse"):
            embed_diag = _batch_embed(embed_manifest)

        if batch_embed_swap:
            _restart_vllm_after_embedding(ctxs[0].config)

        # Mark parse stage done per paper, now that batch embedding has run.
        # _batch_embed swallows subprocess failures (crash, timeout, partial)
        # and returns a diagnostic string — we verify actual success by
        # re-checking has_embeddings() against the DB (authoritative source of
        # truth) and propagate the subprocess stderr into ctx.error so the
        # user sees the real failure reason (CUDA OOM, etc.) inline rather
        # than a cryptic "no embeddings" crash during claim verification.
        keys_requested: dict[str, list[str]] = {
            entry["references_dir"]: entry["keys"] for entry in embed_manifest
        }
        for ctx in _active():
            pr = parse_results_map.get(ctx.name, {})
            n_parsed = sum(1 for v in pr.values() if v == "parsed")
            n_cached = sum(1 for v in pr.values() if v == "cached")
            requested = keys_requested.get(str(ctx.refs_dir), [])
            n_embedded = sum(
                1
                for key in requested
                if has_embeddings(key, ctx.refs_dir, model_name=model_name)
            )
            if n_embedded < len(requested):
                missing = len(requested) - n_embedded
                _track(
                    ctx.refs_dir,
                    "parse",
                    "failed",
                    f"{n_parsed} new, {n_cached} cached, "
                    f"only {n_embedded}/{len(requested)} embedded",
                )
                if embed_diag:
                    ctx.error = (
                        f"batch embed: {missing}/{len(requested)} keys "
                        f"missing | {embed_diag}"
                    )
                else:
                    ctx.error = (
                        f"batch embed: {missing}/{len(requested)} keys "
                        "missing (subprocess exited cleanly — likely silent "
                        "per-key skip, check subprocess logs)"
                    )
                logger.error(
                    "[{}] Batch embed incomplete: {} of {} keys missing | {}",
                    ctx.name,
                    missing,
                    len(requested),
                    embed_diag or "subprocess exited cleanly",
                )
            else:
                _track(
                    ctx.refs_dir,
                    "parse",
                    "done",
                    f"{n_parsed} new, {n_cached} cached, {n_embedded} embedded",
                )

        # ------------------------------------------------------------------
        # Stage 4.2: BATCH CITED VISION — one subprocess, all papers
        # ------------------------------------------------------------------
        logger.info("=== Stage 4.2: Batch cited vision ({} papers) ===", len(ctxs))
        active_ctxs = _active()
        for ctx in active_ctxs:
            _track(ctx.refs_dir, "cited_vision", "running")

        # Swap to vision vLLM only if at least one active paper has
        # uncached cited figures. If all refs are cached (or no papers
        # have any cited PDFs), the subprocess just reads from cache —
        # no HTTP call to vLLM, no need for the swap.
        if vision_backend_is_vllm:
            any_cited_needs = any(
                _cited_needs_inference(ctx.refs_dir, fresh=fresh) for ctx in active_ctxs
            )
            if any_cited_needs:
                await _swap_to_vision_vllm(ctxs[0].config)
                vllm_vision_up = True
            else:
                logger.info(
                    "Cited vision: all figures cached — skipping vision vLLM swap"
                )

        cited_vis_manifest = [
            {
                "references_dir": str(ctx.refs_dir),
                "fresh": False,
                "backend": ctx.config.vision_backend,
            }
            for ctx in active_ctxs
        ]
        # Count total images across all papers for dynamic timeout.
        # Image extraction is lightweight (no GPU) — just PDF page scanning.
        from sciwrite_lint.vision.image_extraction import extract_images_from_pdf

        total_images = 0
        for ctx in active_ctxs:
            parsed_dir = ctx.refs_dir / "parsed"
            if not parsed_dir.exists():
                continue
            for md_file in sorted(parsed_dir.glob("*.md")):
                key = md_file.stem
                candidates = sorted(ctx.refs_dir.glob(f"{key}*.pdf"))
                if candidates:
                    total_images += len(
                        extract_images_from_pdf(
                            candidates[0],
                            ctx.refs_dir / "parsed" / "ref_figures" / key,
                        )
                    )

        # ~5s per image (conservative, GPU batch=16 or vLLM concurrent),
        # 60s for model load.
        cited_vis_timeout = max(120, 60 + total_images * 5)
        logger.info(
            "Cited vision: {} images across {} papers, timeout={}s",
            total_images,
            len(active_ctxs),
            cited_vis_timeout,
        )
        with _stage_failure_guard(
            [ctx.refs_dir for ctx in active_ctxs], "cited_vision"
        ):
            _batch_cited_vision(cited_vis_manifest, timeout=cited_vis_timeout)

        # Read cited vision results from DB for each paper
        from sciwrite_lint.vision.cache import format_descriptions_from_db
        from sciwrite_lint.vision.image_extraction import collect_cited_images

        for ctx in active_ctxs:
            parsed_dir = ctx.refs_dir / "parsed"
            if not parsed_dir.exists():
                _track(ctx.refs_dir, "cited_vision", "done")
                continue
            keys = [f.stem for f in sorted(parsed_dir.glob("*.md"))]
            all_imgs, ranges = collect_cited_images(keys, ctx.refs_dir)
            descriptions: dict[str, str] = {}
            for key, (start, end) in ranges.items():
                desc = format_descriptions_from_db(
                    all_imgs[start:end], ctx.refs_dir, source=key
                )
                if desc:
                    descriptions[key] = desc
            ctx.ref_figure_descs = descriptions
            _track(ctx.refs_dir, "cited_vision", "done")

        if vllm_vision_up:
            await _swap_to_text_vllm(ctxs[0].config)
            vllm_vision_up = False

        # ------------------------------------------------------------------
        # Stage 4.5: REF INTERNAL — all papers concurrent (vLLM)
        # ------------------------------------------------------------------
        logger.info("=== Stage 4.5: Ref internal ({} papers) ===", len(ctxs))

        async def _ref_internal_paper(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                with _stage_tracking(ctx.refs_dir, "ref_internal"):
                    await _stage_ref_internal(
                        ctx.refs_dir,
                        ctx.config,
                        fresh=fresh,
                        ref_figure_descs=ctx.ref_figure_descs,
                    )
            except Exception as e:
                logger.error("[{}] Ref internal failed: {}", ctx.name, e)
                ctx.error = str(e)
            finally:
                if paper_sem:
                    paper_sem.release()

        await asyncio.gather(*[_ref_internal_paper(ctx) for ctx in _active()])

        # ------------------------------------------------------------------
        # Stage 5: CLAIMS + BIB VERIFY — all papers concurrent (vLLM + network)
        # ------------------------------------------------------------------
        logger.info("=== Stage 5: Claims + bib verify ({} papers) ===", len(ctxs))

        async def _claims_bib_paper(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                with _stage_tracking(ctx.refs_dir, ["bib_verify", "claims"]):
                    bib, (c_findings, c_results) = await asyncio.gather(
                        _stage_bib_verify(ctx.config, ctx.refs_dir, fresh=fresh),
                        _stage_claims(
                            ctx.name,
                            ctx.tex_path,
                            ctx.config,
                            ctx.refs_dir,
                            bib_path=ctx.pc.bib,
                            rerun=fresh,
                        ),
                    )
                    ctx.bib_checks = bib
                    ctx.claim_findings = list(c_findings)
                    ctx.claim_results = list(c_results)
            except Exception as e:
                logger.error("[{}] Claims+bib failed: {}", ctx.name, e)
                ctx.error = str(e)
            finally:
                if paper_sem:
                    paper_sem.release()

        await asyncio.gather(*[_claims_bib_paper(ctx) for ctx in _active()])

        # ------------------------------------------------------------------
        # Stage 6: UNRELIABLE — all papers
        # ------------------------------------------------------------------
        active_for_unreliable = _active()
        logger.info(
            "=== Stage 6: Unreliable ({} papers) ===", len(active_for_unreliable)
        )
        for ctx in active_for_unreliable:
            try:
                with _stage_tracking(ctx.refs_dir, "unreliable"):
                    unreliable = _stage_unreliable(
                        ctx.tex_path,
                        ctx.refs_dir,
                        ctx.claim_results,
                        bib_checks=ctx.bib_checks,
                    )
            except Exception as e:
                logger.error("[{}] Unreliable failed: {}", ctx.name, e)
                ctx.error = str(e)
                continue

            # Merge all findings for this paper
            ctx.check_findings.extend(ctx.verify_findings)
            ctx.check_findings.extend(ctx.claim_findings)
            ctx.check_findings.extend(unreliable)

    finally:
        # Save usage tracking for all papers.
        # Note: each ctx.run was created by start_run() which wrote a
        # preliminary row to usage.db. We save stats directly from ctx.run
        # (not via end_run()) because the global _current only holds the
        # last paper's RunStats — the batch creates multiple runs.
        total_elapsed = time.monotonic() - t0
        for ctx in ctxs:
            ctx.run.total_elapsed_s = total_elapsed
            await save_run_async(ctx.run)
            ctx.config.last_run_stats = ctx.run
        # Clear the global so end_run() doesn't return stale data
        end_run()

    # Build results
    results: list[StagedPaperResult] = []
    for ctx in ctxs:
        results.append(
            StagedPaperResult(
                paper_name=ctx.name,
                findings=ctx.check_findings,
                claim_results=ctx.claim_results,
                bib_checks=ctx.bib_checks,
                error=ctx.error,
            )
        )

    failed = [ctx for ctx in ctxs if ctx.error]
    if failed:
        logger.warning(
            "Batch pipeline: {}/{} papers failed: {}",
            len(failed),
            len(ctxs),
            ", ".join(f"{ctx.name} ({ctx.error})" for ctx in failed),
        )
    logger.info(
        "Batch pipeline complete: {}/{} papers succeeded in {:.1f}s",
        len(ctxs) - len(failed),
        len(ctxs),
        time.monotonic() - t0,
    )
    return results
