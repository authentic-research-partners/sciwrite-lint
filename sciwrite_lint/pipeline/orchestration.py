"""Single-paper orchestrator and preflight.

:func:`run_full_check` is the sequential per-paper pipeline used by
``sciwrite-lint check --paper <name>``. The batch-staged variant for
multi-paper runs lives in ``staged.py`` (shares :func:`_backup_workspace`
from here).

:func:`preflight` verifies that vLLM, GROBID, network, and API
credentials are configured before any paper starts.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from sciwrite_lint.config import LintConfig, PaperConfig, PaperWorkspace
from sciwrite_lint.models import Finding
from sciwrite_lint.references.workspace_db import (
    get_db,
    init_pipeline_stages,
)

from sciwrite_lint.pipeline.checks import run_llm_checks_batched, run_text_checks
from sciwrite_lint.pipeline.claims import _stage_bib_verify, _stage_claims
from sciwrite_lint.pipeline.fetch import _stage_fetch
from sciwrite_lint.pipeline.parse import _stage_parse
from sciwrite_lint.pipeline.pdf_context import citations_from_pdf_context
from sciwrite_lint.pipeline.ref_internal import _stage_ref_internal
from sciwrite_lint.pipeline.swap import (
    _cited_needs_inference,
    _manuscript_needs_inference,
    _swap_to_text_vllm,
    _swap_to_vision_vllm,
)
from sciwrite_lint.pipeline.tracking import _stage_tracking, _track
from sciwrite_lint.pipeline.unreliable import _stage_unreliable
from sciwrite_lint.pipeline.verify import _stage_verify
from sciwrite_lint.pipeline.vision import _stage_cited_vision, _stage_vision


async def preflight(config: LintConfig) -> list[str]:
    """Check vLLM, GROBID, network, and API configuration. Return list of errors."""
    from sciwrite_lint.api_keys import check_api_config

    errors: list[str] = check_api_config(config, needs_email=True)

    from sciwrite_lint.pdf.grobid import is_grobid_running

    if not await is_grobid_running():
        errors.append("GROBID not running. Start with: sciwrite-lint containers start")

    try:
        from sciwrite_lint.llm_utils import get_model_config

        model_cfg = get_model_config(config)
        served_name = model_cfg["model"]

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{config.llm_endpoint}/models", timeout=5.0)
            if resp.status_code != 200:
                errors.append(f"vLLM not responding at {config.llm_endpoint}")
            else:
                model_ids = [m["id"] for m in resp.json().get("data", [])]
                if served_name not in model_ids:
                    errors.append(
                        f"vLLM model mismatch: config wants '{served_name}' "
                        f"but server has {model_ids}. "
                        f"Either change [llm] model in .sciwrite-lint.toml "
                        f"or restart vLLM with the right model."
                    )
    except httpx.HTTPError as e:
        errors.append(
            f"vLLM not responding at {config.llm_endpoint}: {e}. "
            "Start with: sciwrite-lint containers start"
        )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.openalex.org/works?per_page=1", timeout=10.0
            )
            if resp.status_code != 200:
                errors.append("Network: OpenAlex API unreachable")
    except httpx.HTTPError:
        errors.append("Network: cannot reach api.openalex.org")

    return errors


def _format_usage_summary(run: Any) -> str:
    """One-line summary of external tool calls for terminal output."""
    parts: list[str] = []

    # APIs
    api_parts: list[str] = []
    for name, svc in [
        ("OpenAlex", run.openalex),
        ("S2", run.semantic_scholar),
        ("CrossRef", run.crossref),
    ]:
        if svc.calls:
            api_parts.append(f"{svc.calls} {name}")
    if api_parts:
        parts.append(f"APIs: {', '.join(api_parts)}")

    # GROBID
    if run.grobid.calls:
        parts.append(f"GROBID: {run.grobid.calls} parsed")

    # Fetch
    if run.fetch.calls:
        parts.append(f"Fetch: {run.fetch.calls} downloads")

    # vLLM
    if run.vllm.calls:
        tokens = run.vllm.extra.get("prompt_tokens", 0) + run.vllm.extra.get(
            "completion_tokens", 0
        )
        token_str = f", {tokens:,} tokens" if tokens else ""
        err_str = f", {run.vllm.errors} errors" if run.vllm.errors else ""
        parts.append(f"vLLM: {run.vllm.calls} calls{token_str}{err_str}")

    return " | ".join(parts) if parts else "No external calls (all cached)"


def _backup_workspace(ws: PaperWorkspace) -> Path | None:
    """Back up a paper workspace as a zip archive before --fresh wipe.

    Creates references/{paper}_backup_YYYYMMDD_HHMMSS.zip from the workspace,
    then removes the original directory. Zip-first ensures the backup exists
    even if removal is interrupted.

    Returns the zip path, or None if nothing to back up.
    """
    if not ws.root.exists():
        return None
    import shutil
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = ws.root.parent / f"{ws.root.name}_backup_{stamp}"
    zip_path = Path(
        shutil.make_archive(str(archive_base), "zip", ws.root.parent, ws.root.name)
    )
    shutil.rmtree(ws.root)
    logger.info(f"Backed up {ws.root.name} → {zip_path.name}")
    return zip_path


async def run_full_check(
    paper_name: str,
    tex_path: Path,
    pc: PaperConfig,
    config: LintConfig,
    fresh: bool = False,
) -> list[Finding]:
    """Run the complete check pipeline for one paper.

    Stages 1 (text+LLM) and 2 (verify) run concurrently via asyncio.
    Stages 3-5 run sequentially after.

    **Not concurrency-safe.** GPU subprocess stages (vision, embedding,
    cited vision) assume exclusive VRAM access. Do not call this function
    concurrently for multiple papers — use ``run_papers_staged()`` instead,
    which coordinates GPU stages across papers.

    Args:
        fresh: Start from scratch — back up existing workspace, then
               re-verify, re-fetch, re-parse, and re-run claims.

    Supports both LaTeX (.tex) and PDF input. For PDF, call
    build_pdf_context() before this function.
    """
    from sciwrite_lint.usage import end_run, save_run_async, start_run

    # Per-paper workspace: references/{paper_name}/
    ws = config.paper_workspace(paper_name)

    # Check source compatibility (tex→pdf switch requires --fresh)
    if not fresh:
        ok, reason = ws.check_source(tex_path, bib_path=pc.bib)
        if not ok:
            raise RuntimeError(
                f"Source type changed ({reason}). Run with --fresh to start over."
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
            citations, tex_path, aux_path=aux_path if aux_path.exists() else None
        )
        if n_bib > len(citations):
            logger.info(
                f"{len(citations)}/{n_bib} bib entries are cited — "
                f"skipping verify+fetch for {n_bib - len(citations)} uncited"
            )
        check_local_sources(citations, refs_dir)

        # Synthesize citations for \footnote{\url{URL}} claims whose URLs
        # are archived as .md captures in local_web_dir (Source: header).
        # These flow through the same verify/fetch/parse/claims pipeline
        # as cited refs — pre-registered with api_match="manual" + T1 so
        # verify skips (nothing to check) and fetch skips (local already).
        footnote_citations = ingest_footnote_sources(
            tex_path,
            config.effective_local_web_dir(paper_name),
            refs_dir,
            source_paper=tex_path.stem,
        )
        citations.extend(footnote_citations)

    logger.info(f"{len(citations)} citations extracted")
    run.citations = len(citations)
    _track(refs_dir, "setup", "done", f"{len(citations)} citations")

    t0 = time.monotonic()
    all_findings: list[Finding] = []

    try:
        # Stage 0.5: Vision — describe figures for full-paper consistency checks.
        # Runs before LLM checks so descriptions are in the cache when
        # _build_system_prompt loads them.
        #
        # When vision_backend == "vllm": swap to vision vLLM only when
        # there is actual inference work; swap back immediately after so
        # text vLLM is available for Stages 1+2. Stage 4.2 does its own
        # swap if cited figures need inference.
        # When vision_backend == "transformers": VL model runs in subprocess,
        # coexists with text vLLM via WSL2 memory overcommit.
        vision_backend_is_vllm = config.vision_backend == "vllm"

        # Stage 0.5: Manuscript vision. Swap to vision vLLM only if the
        # backend is "vllm" AND there is actual VL inference work to do
        # (uncached raster images, or any TikZ figures whose cache state
        # cannot be known until rendered). Cached reruns skip the swap.
        # When no swap happens we keep config.vision_backend unchanged —
        # the subprocess will read from cache and never call the (absent)
        # vLLM endpoint.
        vllm_vision_up = False
        if vision_backend_is_vllm and _manuscript_needs_inference(
            tex_path, refs_dir, ws, fresh=fresh
        ):
            await _swap_to_vision_vllm(config)
            vllm_vision_up = True
        elif vision_backend_is_vllm:
            logger.info(
                "Vision: manuscript figures all cached — skipping vision vLLM swap"
            )

        with _stage_tracking(refs_dir, "vision"):
            _stage_vision(tex_path, config, paper_name, fresh=fresh)

        if vllm_vision_up:
            await _swap_to_text_vllm(config)
            vllm_vision_up = False

        # Stage 1 + 2: concurrent (text checks in thread, LLM + verify async)
        with _stage_tracking(refs_dir, ["text_checks", "llm_checks", "verify"]):
            loop = asyncio.get_event_loop()
            text_task = loop.run_in_executor(None, run_text_checks, tex_path, config)
            llm_task = run_llm_checks_batched(tex_path, config)
            verify_task = _stage_verify(citations, config, refs_dir, fresh=fresh)
            text_findings, llm_findings, verify_findings = await asyncio.gather(
                text_task,
                llm_task,
                verify_task,
            )
            check_findings = text_findings + llm_findings

        t_checks = time.monotonic() - t0
        run.stage_rules_s = t_checks
        run.stage_verify_s = t_checks  # concurrent, same wall time
        logger.info(f"Checks + verify: {t_checks:.1f}s")

        # Note: reference-accuracy findings are produced by _stage_verify
        # via _classify_verify_issue routing mismatch issues. The registered
        # reference-accuracy check exists for standalone use (sciwrite-lint check).

        # Stage 3: fetch + Stage 4: parse (monitor GROBID memory)
        from sciwrite_lint.pdf.grobid import monitor_container_memory

        parse_sem = asyncio.Semaphore(config.grobid_max_concurrency)
        mem_monitor = asyncio.create_task(
            monitor_container_memory(
                parse_sem, max_concurrency=config.grobid_max_concurrency
            )
        )
        try:
            t1 = time.monotonic()
            with _stage_tracking(refs_dir, "fetch") as st:
                fetched = await _stage_fetch(citations, config, refs_dir)
                st.detail = f"{fetched} downloaded"
            run.stage_fetch_s = time.monotonic() - t1

            t2 = time.monotonic()
            with _stage_tracking(refs_dir, "parse") as st:
                # ``_stage_parse`` decides for itself whether to restart
                # text vLLM after embed: if ``_cited_needs_inference``
                # (computed AFTER parse, when parsed .md files exist)
                # returns True, it skips the restart and reports
                # ``text_vllm_stopped=True``. Pre-parse the predicate
                # always returns False on fresh workspaces, so the
                # decision must live inside the stage.
                parsed, cached, parse_findings, text_vllm_stopped = await _stage_parse(
                    config,
                    refs_dir,
                    parse_sem=parse_sem,
                    tex_path=tex_path,
                )
                all_findings.extend(parse_findings)
                st.detail = f"{parsed} new, {cached} cached"
            run.stage_parse_s = time.monotonic() - t2
        finally:
            mem_monitor.cancel()

        if fetched or parsed:
            logger.info(f"Fetch + parse: {run.stage_fetch_s + run.stage_parse_s:.1f}s")

        # Stage 4.2: Vision on cited papers (VL model, sequential).
        # Swap to vision vLLM only if the backend is "vllm" AND there is
        # actual cited inference work to do. If all cited figures are
        # cached (or there are no cited PDFs yet), skip the swap — the
        # subprocess will read from cache and never call vLLM.
        if vision_backend_is_vllm and _cited_needs_inference(refs_dir, fresh=fresh):
            await _swap_to_vision_vllm(config)
            vllm_vision_up = True

        with _stage_tracking(refs_dir, "cited_vision"):
            ref_figure_descs, vision_findings = _stage_cited_vision(
                refs_dir,
                fresh=False,
                backend=config.vision_backend,
                per_image_timeout_s=config.vision_subprocess_per_image_timeout_s,
            )

        if vllm_vision_up:
            await _swap_to_text_vllm(config)
            vllm_vision_up = False
        elif text_vllm_stopped:
            # ``_stage_parse`` left text vLLM stopped because its
            # post-parse ``_cited_needs_inference`` returned True. But
            # the cited-vision swap just decided not to fire — state
            # changed between the two calls (rare; e.g. the figure
            # cache got populated by a concurrent process between
            # parse end and now). Restart text vLLM here so the next
            # stage (ref_internal) finds it running.
            from sciwrite_lint.pipeline.swap import _restart_vllm_after_embedding

            _restart_vllm_after_embedding(config)

        # Stage 4.5: ref internal checks (vLLM, thinking=low/medium)
        t3 = time.monotonic()
        with _stage_tracking(refs_dir, "ref_internal"):
            ref_internal_results = await _stage_ref_internal(
                refs_dir, config, fresh=fresh, ref_figure_descs=ref_figure_descs
            )

        # Stage 4.6 + 5: bib verify (network) + claims (vLLM) — concurrent
        with _stage_tracking(refs_dir, ["bib_verify", "claims"]):
            bib_checks, (claim_findings, claim_results) = await asyncio.gather(
                _stage_bib_verify(config, refs_dir, fresh=fresh),
                _stage_claims(
                    paper_name,
                    tex_path,
                    config,
                    refs_dir,
                    bib_path=pc.bib,
                    rerun=fresh,
                ),
            )
        run.stage_claims_s = time.monotonic() - t3
        if claim_findings or ref_internal_results:
            logger.info(f"Claims + ref checks: {run.stage_claims_s:.1f}s")

        # Stage 6: reference-unreliable (aggregates signals from verify + claims)
        with _stage_tracking(refs_dir, "unreliable"):
            unreliable_findings = _stage_unreliable(
                tex_path,
                refs_dir,
                claim_results,
                bib_checks=bib_checks,
            )

        # Merge all findings
        all_findings = (
            list(check_findings)
            + verify_findings
            + claim_findings
            + unreliable_findings
            + vision_findings
        )

    finally:
        # Always save usage — even on crash, partial data is useful
        run.total_elapsed_s = time.monotonic() - t0
        logger.info(f"Total: {run.total_elapsed_s:.1f}s")
        logger.info(f"{_format_usage_summary(run)}")
        stats = end_run()
        if stats:
            await save_run_async(stats)
            config.last_run_stats = stats

    return all_findings
