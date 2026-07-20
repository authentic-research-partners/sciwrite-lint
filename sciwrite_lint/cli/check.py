"""CLI handlers for 'check' and 'checks' commands."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.checks._diagnostics import split_findings
from sciwrite_lint.config import LintConfig
from sciwrite_lint.exceptions import SciWriteLintError
from sciwrite_lint.models import Finding

from sciwrite_lint.report import format_findings


def _resolve_helpers(
    args: argparse.Namespace, config: LintConfig
) -> list[tuple[str, Path]]:
    """Import and call _resolve_input_files from __main__."""
    from sciwrite_lint.cli._common import _resolve_input_files

    return _resolve_input_files(args, config)


def _emit_paper_results(
    name: str,
    label: str,
    findings: list[Finding],
    config: LintConfig,
    fmt: str,
) -> bool:
    """Render + save results for one paper. Returns True if any error.

    Splits findings into manuscript / system buckets via the canonical
    helper, prints the manuscript panel, then a separate system-issues
    panel (terminal mode only — JSON-stdout consumers see only the
    manuscript findings; the on-disk report carries both fields). The
    integrity summary, JSON report, and error gate all see manuscript
    findings only — system issues describe the linter's state, not the
    manuscript.

    Used by both single-paper and batch CLI flows so the split / render
    / count contract is enforced in one place.
    """
    manuscript, system_issues = split_findings(findings)
    format_findings(manuscript, label, fmt=fmt, color=config.color)
    if system_issues and fmt != "json":
        format_findings(
            system_issues,
            f"{label} — system issues",
            fmt=fmt,
            color=config.color,
        )
    _print_integrity_summary(name, manuscript, config, system_issues=system_issues)
    return any(f.level == "error" for f in manuscript)


def run_check(args: argparse.Namespace) -> int:
    """Run the full check pipeline: rules + verify + fetch + parse + claims.

    Single-paper case dispatches to ``run_full_check`` (sequential pipeline).
    Multi-paper case (2+ papers) dispatches to ``run_papers_staged`` for the
    batch-staged pipeline — GPU stages run once with all papers, vLLM and
    network stages run concurrently up to ``--concurrency``.
    """
    from sciwrite_lint.pipeline import preflight, run_full_check

    from sciwrite_lint.cli._common import _load_config, _resolve_paper

    config = _load_config(args)
    fmt = args.format or config.output_format
    fresh = getattr(args, "fresh", False)
    concurrency = getattr(args, "concurrency", 2)

    # CLI flag overrides config file
    vision_backend = getattr(args, "vision_backend", None)
    if vision_backend:
        config.vision_backend = vision_backend

    # --checks filter: disable every registered check not in the allow-list.
    # Validated against the registry so a typo fails loudly rather than
    # silently matching nothing.
    only_checks_raw = getattr(args, "checks", None)
    if only_checks_raw:
        from sciwrite_lint.checks.registry import list_checks

        wanted = {c.strip() for c in only_checks_raw.split(",") if c.strip()}
        known = {m.id for m in list_checks()}
        unknown = wanted - known
        if unknown:
            logger.error(
                "Unknown check IDs in --checks: {}. Run `sciwrite-lint checks` "
                "to see available IDs.",
                ", ".join(sorted(unknown)),
            )
            return 2
        config.disabled_rules = config.disabled_rules | (known - wanted)

    # Explicit file — no paper config, run text + LLM rules only
    if hasattr(args, "file") and args.file:
        from sciwrite_lint.cli._common import (
            MANUSCRIPT_SUFFIXES,
            unsupported_manuscript_error,
        )

        p = Path(args.file)
        if p.suffix.lower() not in MANUSCRIPT_SUFFIXES:
            logger.error(unsupported_manuscript_error(p))
            return 2
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
    from sciwrite_lint.cli._common import (
        MANUSCRIPT_SUFFIXES,
        unsupported_manuscript_error,
    )

    runnable: list[tuple[str, Path, Any]] = []
    all_ok = True
    for name, tex_path, pc in paper_configs:
        if not tex_path.exists():
            logger.error(f"Error: {tex_path} not found")
            all_ok = False
            continue
        if tex_path.suffix.lower() not in MANUSCRIPT_SUFFIXES:
            logger.error(f"{name}: {unsupported_manuscript_error(tex_path)}")
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

    # Non-LaTeX inputs build their ManuscriptContext up front so the full
    # pipeline (citation extraction, prose/LLM checks) sees the right
    # source: PDF via GROBID, markdown via pandoc.
    suffix = tex_path.suffix.lower()
    if suffix == ".pdf":
        from sciwrite_lint.pipeline import build_pdf_context

        asyncio.run(build_pdf_context(tex_path, config))
    elif suffix == ".md":
        from sciwrite_lint.pipeline import build_markdown_context

        build_markdown_context(tex_path, config)

    try:
        findings = asyncio.run(run_full_check(name, tex_path, pc, config, fresh=fresh))
    except SciWriteLintError as e:
        logger.error(f"Tool error: {e}")
        return 2

    label = f"{name} ({tex_path.name})" if name != tex_path.stem else tex_path.name
    print()
    if _emit_paper_results(name, label, findings, config, fmt):
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
    from sciwrite_lint.pipeline import (
        build_markdown_context,
        build_pdf_context,
        run_papers_staged,
    )

    print(f"\n{'=' * 60}")
    print(
        f"Checking {len(runnable)} papers via batch-staged pipeline "
        f"(concurrency={concurrency})"
    )
    print(f"{'=' * 60}")

    # Non-LaTeX inputs need their ManuscriptContext built upfront (PDF via
    # GROBID, markdown via pandoc) before the staged pipeline runs — same
    # pattern as eval_real_world/runner.py. Each build seeds the per-path
    # context cache the staged stages read from.
    async def _build_contexts() -> None:
        for _name, tex_path, _pc in runnable:
            suffix = tex_path.suffix.lower()
            if suffix == ".pdf":
                await build_pdf_context(tex_path, config)
            elif suffix == ".md":
                build_markdown_context(tex_path, config)

    asyncio.run(_build_contexts())

    # run_papers_staged expects (name, tex_path, paper_config, lint_config) tuples
    staged_input = [(name, tex_path, pc, config) for name, tex_path, pc in runnable]

    try:
        results = asyncio.run(
            run_papers_staged(staged_input, fresh=fresh, concurrency=concurrency)
        )
    except SciWriteLintError as e:
        logger.error(f"Tool error: {e}")
        return 2
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
        if _emit_paper_results(name, label, sr.findings, config, fmt):
            all_ok = False

    return 0 if all_ok else 1


def _print_integrity_summary(
    paper_name: str,
    findings: list[Finding],
    config: LintConfig,
    system_issues: list[Finding] | None = None,
) -> None:
    """Compute integrity, save report, print summary.

    ``findings`` must be manuscript-only — system issues (linter could
    not run a check) are passed via ``system_issues`` and routed to a
    separate JSON field so they don't contaminate scores or counts.
    """
    import json

    from sciwrite_lint.scoring.scilint_score import (
        _aggregate_ref_claims,
        _score_reference,
        compute_integrity,
    )

    system_issues = system_issues or []

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
    system_issue_dicts = [f.model_dump() for f in system_issues]
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
            # For T2 refs, the pipeline records *why* no PDF was acquired
            # (e.g. "image-only PDF — GROBID cannot parse (from nber);
            # HAL: no match; …"). Surface this in the report so humans
            # and agents can see which sources were tried and what each
            # one said, without reading the debug log.
            acquisition_reason = meta.access.get("acquisition_reason", "")
            if acquisition_reason:
                reliability["acquisition_reason"] = acquisition_reason
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

    # Save report. ``findings`` and ``total_findings``/``errors``/
    # ``warnings`` describe the manuscript only; ``system_issues`` and
    # ``total_system_issues`` describe linter operational state (LLM
    # gave up, vision subprocess timed out). The split lets a reviewer
    # see manuscript quality without operational noise inflating it.
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
        "total_system_issues": len(system_issue_dicts),
        "references": ref_details,
        "findings": findings_dicts,
        "system_issues": system_issue_dicts,
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

    # Claim coverage — show how many cites were verified vs. skipped so
    # the reviewer can tell at a glance which slice of the bibliography
    # actually got LLM-checked. SKIPPED rows are persisted but produce no
    # manuscript findings; their counts belong with the score banner, not
    # the findings panel.
    if claims:
        verdicts: dict[str, int] = {}
        for c in claims:
            v = c.get("verdict") or "UNKNOWN"
            verdicts[v] = verdicts.get(v, 0) + 1
        n_skipped = verdicts.get("SKIPPED", 0)
        n_verified = sum(
            verdicts.get(v, 0)
            for v in ("SUPPORTS", "NOT_SUPPORTED", "PARTIALLY_SUPPORTS")
        )
        n_undet = verdicts.get("CANNOT_DETERMINE", 0)
        if n_skipped or n_undet:
            print(
                f"  Claim coverage: {n_verified} verified, "
                f"{n_undet} undetermined, {n_skipped} skipped (no local source)"
            )

    # System issues report on the linter, not the manuscript — show
    # them as a separate one-liner so they don't dilute the score
    # banner. The full list is in the JSON report under system_issues.
    if system_issue_dicts:
        n = len(system_issue_dicts)
        print(
            f"\n  Note: {n} system issue{'s' if n != 1 else ''} "
            f"(linter could not run all checks — see "
            f"check_{paper_name}.json:system_issues)"
        )


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

    try:
        findings = asyncio.run(_do_check())
    except SciWriteLintError as e:
        logger.error(f"Tool error: {e}")
        return 2

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
        build_markdown_context,
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

        # Reset per paper so a prior markdown manuscript does not leave
        # ``config.is_markdown`` True for a following .tex in the same run.
        config.manuscript_context = None
        # Markdown manuscripts: parse into a ManuscriptContext and seed
        # the cache so context-consuming checks receive the markdown parse
        # (cache hit) instead of re-parsing the .md as LaTeX. Without this
        # the file falls through to the LaTeX path and mis-parses.
        if tex_path.suffix.lower() == ".md":
            build_markdown_context(tex_path, config)

        findings = run_text_checks(tex_path, config)
        try:
            findings.extend(asyncio.run(pipeline_llm_batched(tex_path, config)))
        except SciWriteLintError as e:
            logger.error(f"Tool error: {e}")
            return 2

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
