"""CLI handlers for 'check' and 'checks' commands."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Finding
from sciwrite_lint.report import format_findings


def _resolve_helpers(
    args: argparse.Namespace, config: LintConfig
) -> list[tuple[str, Path]]:
    """Import and call _resolve_input_files from __main__."""
    from sciwrite_lint.__main__ import _resolve_input_files

    return _resolve_input_files(args, config)


async def run_llm_checks_batched(
    checks: list[tuple],
    tex_path: Path,
    config: LintConfig,
) -> list[Finding]:
    """Run all LLM checks batched by thinking mode.

    All LLM checks must implement the build_queries/process_results protocol:
    - build_queries(tex_path, config) -> list of (system, user, schema, name) tuples
    - process_results(results) -> list[Finding]

    Each check can set a ``thinking`` attribute ("off", "low", "medium", "high")
    to control chain-of-thought reasoning. Queries are grouped by thinking mode
    so each group runs as a single batch call.
    """
    from sciwrite_lint.llm_utils import llm_query_batch

    # Phase 1: collect queries, grouped by (thinking, max_tokens)
    # max_tokens comes from build_queries._max_tokens (set by full-paper
    # checks based on paper size); None means use model default.
    BatchKey = tuple[str, int | None]
    queries_by_batch: dict[BatchKey, list[tuple[str, str, dict, str]]] = {}
    check_slices: list[tuple[Any, Any, BatchKey, int, int]] = []
    build_failures: list[Finding] = []

    for meta, fn in checks:
        build = getattr(fn, "build_queries", None)
        if build is None:
            raise RuntimeError(f"LLM check {meta.id} missing build_queries")
        try:
            queries = build(tex_path, config)
            thinking = getattr(fn, "thinking", "off")
            mt: int | None = getattr(build, "_max_tokens", None)
            batch_key: BatchKey = (thinking, mt)
            if batch_key not in queries_by_batch:
                queries_by_batch[batch_key] = []
            start = len(queries_by_batch[batch_key])
            queries_by_batch[batch_key].extend(queries)
            check_slices.append((meta, fn, batch_key, start, len(queries)))
        except Exception as e:
            logger.error(f"Check {meta.id} build_queries failed: {e}")
            build_failures.append(
                Finding(
                    level="info",
                    rule_id=meta.id,
                    message=f"Check {meta.id} could not run (internal error)",
                    context=f"{type(e).__name__}: {e!s}"[:200],
                )
            )

    # Phase 2: one batch call per (thinking, max_tokens) group
    results_by_batch: dict[BatchKey, list[dict | None]] = {}
    for (mode, mt), batch_queries in queries_by_batch.items():
        if batch_queries:
            try:
                results_by_batch[(mode, mt)] = await llm_query_batch(
                    batch_queries,
                    config=config,
                    thinking=mode,
                    max_tokens=mt,
                )
            except Exception as e:
                logger.error(f"LLM batch query failed (thinking={mode}): {e}")
                results_by_batch[(mode, mt)] = [None] * len(batch_queries)

    # Phase 3: distribute results to each check
    findings: list[Finding] = []
    for meta, fn, batch_key, start, count in check_slices:
        try:
            if count > 0:
                process = getattr(fn, "process_results")
                mode_results = results_by_batch.get(batch_key, [])
                check_results = mode_results[start : start + count]
                check_findings = process(check_results)
            else:
                check_findings = []

            for f in check_findings:
                override = config.effective_severity(meta.id, meta.severity)
                if override != f.level:
                    f.level = override  # type: ignore[assignment]
            findings.extend(check_findings)
        except Exception as e:
            logger.error(f"Check {meta.id} failed: {e}")
            findings.append(
                Finding(
                    level="info",
                    rule_id=meta.id,
                    message=f"Check {meta.id} could not run (internal error)",
                    context=f"{type(e).__name__}: {e!s}"[:200],
                )
            )

    return build_failures + findings


def run_check(args: argparse.Namespace) -> int:
    """Run the full check pipeline: rules + verify + fetch + parse + claims.

    Single-paper case dispatches to ``run_full_check`` (sequential pipeline).
    Multi-paper case (2+ papers) dispatches to ``run_papers_staged`` for the
    batch-staged pipeline — GPU stages run once with all papers, vLLM and
    network stages run concurrently up to ``--concurrency``.
    """
    from sciwrite_lint.pipeline import preflight, run_full_check

    from sciwrite_lint.__main__ import _load_config, _resolve_paper

    config = _load_config(args)
    fmt = args.format or config.output_format
    fresh = getattr(args, "fresh", False)
    concurrency = getattr(args, "concurrency", 2)

    # Explicit file — no paper config, run text + LLM rules only
    if hasattr(args, "file") and args.file:
        p = Path(args.file)
        if p.suffix.lower() == ".pdf":
            return run_check_pdf(p, config, fmt)
        return run_check_quick(args, config, fmt)

    # Resolve to paper configs (need bib_path for claims)
    paper_filter = getattr(args, "paper", None)
    paper_configs: list[tuple[str, Path, Any]] = []
    if paper_filter:
        pc = _resolve_paper(config, paper_filter)
        if not pc:
            return 2
        paper_configs.append((pc.name, pc.file_path, pc))
    elif config.papers:
        for pc in config.papers:
            paper_configs.append((pc.name, pc.file_path, pc))
    else:
        logger.error("No papers registered. Use --paper or add [[papers]] to config.")
        return 2

    # Filter out missing files up front so the run/batch path sees only
    # papers that actually exist.
    runnable: list[tuple[str, Path, Any]] = []
    all_ok = True
    for name, tex_path, pc in paper_configs:
        if not tex_path.exists():
            logger.error(f"Error: {tex_path} not found")
            all_ok = False
            continue
        runnable.append((name, tex_path, pc))

    if not runnable:
        return 1

    # Preflight
    errors = asyncio.run(preflight(config))
    if errors:
        logger.error("Prerequisites not met:")
        for e in errors:
            logger.error(f"  ✗ {e}")
        return 2

    # Multi-paper: use batch-staged pipeline (single GPU model load per
    # stage, concurrent vLLM/network across papers).
    if len(runnable) > 1:
        return _run_check_batch(runnable, config, fmt, fresh, concurrency, all_ok)

    # Single paper: keep the existing sequential path
    name, tex_path, pc = runnable[0]
    print(f"\n{'=' * 60}")
    print(f"Checking {name} ({tex_path.name})")
    print(f"{'=' * 60}")

    # PDF files: build ManuscriptContext via GROBID, then run full pipeline
    if tex_path.suffix.lower() == ".pdf":
        from sciwrite_lint.pipeline import build_pdf_context

        asyncio.run(build_pdf_context(tex_path, config))

    findings = asyncio.run(run_full_check(name, tex_path, pc, config, fresh=fresh))

    label = f"{name} ({tex_path.name})" if name != tex_path.stem else tex_path.name
    print()
    format_findings(findings, label, fmt=fmt, color=config.color)
    _print_integrity_summary(name, findings, config)

    if any(f.level == "error" for f in findings):
        all_ok = False

    return 0 if all_ok else 1


def _run_check_batch(
    runnable: list[tuple[str, Path, Any]],
    config: LintConfig,
    fmt: str,
    fresh: bool,
    concurrency: int,
    all_ok: bool,
) -> int:
    """Run multiple papers via the batch-staged pipeline."""
    from sciwrite_lint.pipeline import build_pdf_context, run_papers_staged

    print(f"\n{'=' * 60}")
    print(
        f"Checking {len(runnable)} papers via batch-staged pipeline "
        f"(concurrency={concurrency})"
    )
    print(f"{'=' * 60}")

    # PDFs need ManuscriptContext built upfront (GROBID parse) before staged
    # pipeline runs — same pattern as eval_real_world/runner.py.
    async def _build_pdf_contexts() -> None:
        for _name, tex_path, _pc in runnable:
            if tex_path.suffix.lower() == ".pdf":
                await build_pdf_context(tex_path, config)

    asyncio.run(_build_pdf_contexts())

    # run_papers_staged expects (name, tex_path, paper_config, lint_config) tuples
    staged_input = [(name, tex_path, pc, config) for name, tex_path, pc in runnable]

    try:
        results = asyncio.run(
            run_papers_staged(staged_input, fresh=fresh, concurrency=concurrency)
        )
    except Exception as e:
        logger.error(f"Batch-staged pipeline failed: {e}")
        return 1

    # Format output for each paper, in original order
    result_map = {r.paper_name: r for r in results}
    for name, tex_path, _pc in runnable:
        sr = result_map.get(name)
        print(f"\n{'=' * 60}")
        print(f"{name} ({tex_path.name})")
        print(f"{'=' * 60}")
        if not sr or sr.error:
            err = sr.error if sr else "missing from batch results"
            logger.error(f"[{name}] Pipeline failed: {err}")
            all_ok = False
            continue

        label = f"{name} ({tex_path.name})" if name != tex_path.stem else tex_path.name
        format_findings(sr.findings, label, fmt=fmt, color=config.color)
        _print_integrity_summary(name, sr.findings, config)

        if any(f.level == "error" for f in sr.findings):
            all_ok = False

    return 0 if all_ok else 1


def _print_integrity_summary(
    paper_name: str,
    findings: list[Finding],
    config: LintConfig,
) -> None:
    """Compute integrity, save report, print summary."""
    import json

    from sciwrite_lint.scoring.scilint_score import (
        _aggregate_ref_claims,
        _score_reference,
        compute_integrity,
    )

    output_dir = config.results_dir

    # Load claims, ref internal scores, and metadata from workspace.db
    ws = config.paper_workspace(paper_name)
    claims: list[dict] = []
    ref_internal_scores: dict[str, float] | None = None
    metadata_map = None

    if ws.root.exists():
        from sciwrite_lint.references.metadata import load_all_metadata
        from sciwrite_lint.references.workspace_db import (
            get_db,
            load_all_ref_internal_scores,
            load_claim_results,
        )

        with get_db(ws.root) as conn:
            claims = load_claim_results(conn)
            scores = load_all_ref_internal_scores(conn)
            if scores:
                ref_internal_scores = scores

        metadata_map = load_all_metadata(ws.root)

    findings_dicts = [f.model_dump() for f in findings]
    result = compute_integrity(
        findings_dicts,
        claims,
        metadata_map=metadata_map,
        ref_internal_scores=ref_internal_scores,
    )

    # Build per-reference detail for report (matches target format)
    by_ref = _aggregate_ref_claims(claims)
    ref_details: list[dict] = []
    for key, ref_claims in sorted(by_ref.items()):
        rs = _score_reference(ref_claims)
        reliability_score = result.reference_reliability.get(key)

        # Build reliability breakdown
        reliability: dict[str, Any] = {}
        if metadata_map and key in metadata_map:
            meta = metadata_map[key]
            reliability["score"] = (
                round(reliability_score, 4) if reliability_score is not None else None
            )
            reliability["tier"] = meta.access.get("tier", "")
            reliability["retracted"] = bool(meta.canonical.get("retracted"))
            reliability["metadata_mismatches"] = meta.mismatches
            consistency = ref_internal_scores.get(key) if ref_internal_scores else None
            reliability["consistency"] = (
                round(consistency, 4) if consistency is not None else None
            )
        elif reliability_score is not None:
            reliability["score"] = round(reliability_score, 4)

        # Compute per-ref SciLint Score (reliability only, contribution=1.0)
        ref_scilint = (
            round(reliability_score, 4) if reliability_score is not None else None
        )

        # Build signals list
        signals: list[str] = []
        if metadata_map and key in metadata_map:
            meta = metadata_map[key]
            if meta.access.get("tier") == "T3":
                signals.append("not found in APIs")
            if meta.canonical.get("retracted"):
                signals.append("RETRACTED")
            signals.extend(meta.mismatches)
        if rs.verdict == "NOT_SUPPORTED":
            signals.append("claim not supported")
        elif rs.verdict == "PARTIALLY_SUPPORTS":
            signals.append("claim partially supported")

        detail: dict[str, Any] = {
            "key": key,
            "scilint_score": ref_scilint,
            "verdict": rs.verdict,
            "purpose": {
                "role": rs.purpose,
                "weight": round(rs.weight, 2),
            },
            "reliability": reliability if reliability else None,
            "signals": signals,
        }
        ref_details.append(detail)

    # Save report
    report = {
        "paper": paper_name,
        "scilint_score": round(
            result.internal_consistency * result.referencing_quality, 4
        ),
        "internal_consistency": round(result.internal_consistency, 4),
        "referencing_quality": round(result.referencing_quality, 4),
        "contribution": None,
        "total_findings": len(findings_dicts),
        "errors": sum(1 for f in findings_dicts if f.get("level") == "error"),
        "warnings": sum(1 for f in findings_dicts if f.get("level") == "warning"),
        "references": ref_details,
        "findings": findings_dicts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"check_{paper_name}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"Check report saved to {report_path}")

    # Print summary
    scilint = result.internal_consistency * result.referencing_quality
    print(f"\n  SciLint Score: {scilint:.2f}")
    print(f"    Internal Consistency:  {result.internal_consistency:.2f}")
    print(f"    Referencing Quality:   {result.referencing_quality:.2f}")
    print("  (Run 'sciwrite-lint contributions' to add contribution axes)")


def run_check_pdf(pdf_path: Path, config: LintConfig, fmt: str) -> int:
    """Run checks on a PDF file. Requires GROBID."""
    from sciwrite_lint.pipeline import (
        build_pdf_context,
        run_llm_checks_batched as pipeline_llm_batched,
        run_text_checks,
    )

    if not pdf_path.exists():
        logger.error(f"Error: {pdf_path} not found")
        return 2

    async def _do_check() -> list[Finding]:
        await build_pdf_context(pdf_path, config)
        findings = run_text_checks(pdf_path, config)
        findings.extend(await pipeline_llm_batched(pdf_path, config))
        return findings

    findings = asyncio.run(_do_check())

    label = pdf_path.name
    print()
    format_findings(findings, label, fmt=fmt, color=config.color)

    return 0 if not any(f.level == "error" for f in findings) else 1


def run_check_quick(
    args: argparse.Namespace,
    config: LintConfig,
    fmt: str,
) -> int:
    """Quick mode: text + LLM rules only, no verify/fetch/parse/claims."""
    from sciwrite_lint.pipeline import (
        run_llm_checks_batched as pipeline_llm_batched,
        run_text_checks,
    )

    papers = _resolve_helpers(args, config)
    if not papers:
        return 2

    all_ok = True
    for name, tex_path in papers:
        if not tex_path.exists():
            logger.error(f"Error: {tex_path} not found")
            all_ok = False
            continue

        findings = run_text_checks(tex_path, config)
        findings.extend(asyncio.run(pipeline_llm_batched(tex_path, config)))

        label = f"{name} ({tex_path.name})" if name != tex_path.stem else tex_path.name
        format_findings(findings, label, fmt=fmt, color=config.color)

        if any(f.level == "error" for f in findings):
            all_ok = False

    return 0 if all_ok else 1


def run_checks_list(args: argparse.Namespace) -> int:
    """List all registered checks."""
    from sciwrite_lint.checks.registry import ensure_checks_loaded, list_checks

    ensure_checks_loaded()
    all_checks = list_checks()

    for meta in all_checks:
        print(f"  {meta.id:<32} [{meta.category}] {meta.description}")

    print(f"\n  {len(all_checks)} checks registered.")
    return 0
