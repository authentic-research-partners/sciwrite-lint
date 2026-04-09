"""CLI handlers for 'contributions' command — SciLint Score (integrity × contribution)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.config import LintConfig, PaperConfig
from sciwrite_lint.models import Finding


def extract_claims_from_context(
    ctx: Any,  # ManuscriptContext
    file_path: Path,
) -> list[dict[str, Any]]:
    """Extract claim dicts from a ManuscriptContext (PDF or .tex)."""
    if ctx.source_type == "latex":
        from sciwrite_lint.eval_claims import extract_claim_contexts

        return [
            {"key": cc.key, "context": cc.context, "line": cc.line}
            for cc in extract_claim_contexts(file_path)
        ]
    # PDF: convert InlineCitation objects to claim dicts
    return [
        {
            "key": ic.key,
            "context": ic.context,
            "line": ic.line or 0,
            "source_section": ic.section,
        }
        for ic in ctx.inline_citations
        if ic.context
    ]


async def score_standalone_async(
    file_path: Path,
    config: LintConfig,
    *,
    contribution: bool = True,
    model: str = "",
) -> tuple[Any, list[Finding], dict[str, float] | None, dict[str, str] | None]:
    """Async core: parse, check, extract claims, compute contribution axes.

    Designed to be called from asyncio.run() (single paper) or from a batch
    loop that scores multiple papers within one event loop.
    """
    from sciwrite_lint.manuscript_store import ManuscriptContext
    from sciwrite_lint.pipeline import (
        build_pdf_context,
        run_llm_checks_batched,
        run_text_checks,
    )

    suffix = file_path.suffix.lower()

    # 1. Parse manuscript and run text checks
    if suffix == ".pdf":
        await build_pdf_context(file_path, config)
        ctx = config.manuscript_context
    else:
        ctx = ManuscriptContext.from_latex(file_path, config)
    findings = run_text_checks(file_path, config)
    findings.extend(await run_llm_checks_batched(file_path, config))

    # 2. Extract claims
    claim_dicts = extract_claims_from_context(ctx, file_path)

    # 3. Contribution axes
    c_scores: dict[str, float] | None = None
    c_reasoning: dict[str, str] | None = None
    if contribution:
        ns = argparse.Namespace(model=model)
        c_scores, c_reasoning = await compute_contribution_axes_from_ctx(
            ctx, claim_dicts, config, ns
        )
    return ctx, findings, c_scores, c_reasoning


def build_scilint_result(
    file_path: Path,
    findings: list[Finding],
    contribution_scores: dict[str, float] | None,
    contribution_reasoning: dict[str, str] | None,
    output_dir: Path | None = None,
) -> Any:
    """Compute SciLint Score from findings + contribution and optionally save."""
    from sciwrite_lint.scoring.scilint_score import (
        SciLintScoreResult,
        compute_scilint_score,
    )

    paper_name = file_path.stem
    findings_dicts = [f.model_dump() for f in findings]
    result: SciLintScoreResult = compute_scilint_score(
        paper_name,
        claim_results=[],
        findings=findings_dicts,
        contribution_scores=contribution_scores,
        contribution_reasoning=contribution_reasoning,
    )
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"scilint_{paper_name}.json"
        out_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info(f"SciLint Score saved to {out_path}")
    return result


def score_standalone_file(
    file_path: Path,
    config: LintConfig,
    *,
    contribution: bool = True,
    model: str = "",
    output_dir: Path | None = None,
) -> Any:
    """Score a standalone PDF/tex file. Single-paper entry point (uses asyncio.run).

    For batch scoring (multiple papers), use score_standalone_async directly
    within a single event loop to avoid repeated asyncio.run() calls.
    """
    _ctx, findings, contribution_scores, contribution_reasoning = asyncio.run(
        score_standalone_async(
            file_path, config, contribution=contribution, model=model
        )
    )

    return build_scilint_result(
        file_path, findings, contribution_scores, contribution_reasoning, output_dir
    )


def run_contributions(args: argparse.Namespace) -> int:
    """Compute SciLint Score from claim verification results."""
    from sciwrite_lint.__main__ import _load_config, _resolve_paper

    config = _load_config(args)

    # File path mode — standalone ranking without verify-claims
    if hasattr(args, "file") and args.file:
        return run_contributions_file(Path(args.file), config, args)

    if not getattr(args, "paper", None):
        logger.error("Either a file path or --paper is required.")
        return 2

    from sciwrite_lint.scoring.scilint_score import (
        run_contributions as scilint_run_contributions,
    )

    pc = _resolve_paper(config, args.paper)
    if not pc:
        return 2

    output_dir = Path(args.output_dir) if args.output_dir else config.results_dir

    ws = config.paper_workspace(pc.name)
    refs_dir = ws.root

    # Load claims from workspace.db
    from sciwrite_lint.references.workspace_db import get_db, load_claim_results

    if not refs_dir.exists():
        logger.error(
            f"No workspace found for paper '{pc.name}'. "
            f"Run 'sciwrite-lint check --paper {pc.name}' first."
        )
        return 1

    with get_db(refs_dir) as conn:
        claims = load_claim_results(conn)

    if not claims:
        logger.error(
            f"No claim results for paper '{pc.name}'. "
            f"Run 'sciwrite-lint check --paper {pc.name}' first."
        )
        return 1

    findings_path = Path(args.findings) if getattr(args, "findings", None) else None

    # Contribution axes — always computed for score command
    contribution_scores, contribution_reasoning = asyncio.run(
        compute_contribution_axes(pc, claims, config, args)
    )

    result = scilint_run_contributions(
        pc.name,
        claims,
        findings_path=findings_path,
        references_dir=refs_dir,
        contribution_scores=contribution_scores,
        contribution_reasoning=contribution_reasoning,
        output_dir=output_dir,
    )

    if getattr(args, "format", "terminal") == "json":
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    ir = result.integrity_result
    print(f"\n  SciLint Score for '{pc.name}':")
    print(f"    SciLint Score:    {result.scilint_score:.4f}")
    print(f"    Internal Consistency: {ir.internal_consistency:.4f}")
    print(f"    Referencing Quality:  {ir.referencing_quality:.4f}")
    if ir.reference_reliability:
        print(
            f"      Reference reliability: {len(ir.reference_reliability)} refs scored"
        )
    print(f"    Contribution:     {result.contribution.overall:.4f}")
    c = result.contribution
    if c.empirical_content > 0:
        print(f"      Empirical:      {c.empirical_content:.4f}")
        print(f"      Progressive:    {c.progressiveness:.4f}")
        print(f"      Unification:    {c.unification:.4f}")
        print(f"      Problem-solv:   {c.problem_solving:.4f}")
        print(f"      Test severity:  {c.test_severity:.4f}")
    print(f"    Claims verified:  {result.total_claims}")
    print(f"    Refs scored:      {result.total_refs_scored}")

    if result.ref_scores:
        print("\n  Per-reference breakdown:")
        for rs in sorted(result.ref_scores, key=lambda r: r.weighted_score):
            print(
                f"    {rs.key:30s}  {rs.verdict:20s}  "
                f"w={rs.weight:.1f}  score={rs.verification_score:.2f}  "
                f"({rs.purpose})"
            )

    # Summary counts — quick overview of actionable issues
    _print_issue_summary(claims)

    return 0


def _print_issue_summary(
    claims: list[dict[str, Any]],
) -> None:
    """Print a one-line issue summary below the score breakdown."""
    from sciwrite_lint.checks.cite_purpose import PURPOSE_WEIGHTS, UNSPECIFIED_THRESHOLD

    active = [r for r in claims if not r.get("dismissed")]

    not_supported = sum(1 for r in active if r.get("verdict") == "NOT_SUPPORTED")
    partial = sum(1 for r in active if r.get("verdict") == "PARTIALLY_SUPPORTS")
    weak_cite = sum(
        1
        for r in active
        if PURPOSE_WEIGHTS.get(
            r.get("cite_purpose") or r.get("citation_purpose", ""), 1.0
        )
        <= UNSPECIFIED_THRESHOLD
    )

    parts: list[str] = []
    if not_supported:
        parts.append(f"{not_supported} unsupported")
    if partial:
        parts.append(f"{partial} partial")
    if weak_cite:
        parts.append(f"{weak_cite} weak citation{'s' if weak_cite != 1 else ''}")

    if parts:
        print(f"\n  Issues: {' | '.join(parts)}")


def run_contributions_file(
    file_path: Path, config: LintConfig, args: argparse.Namespace
) -> int:
    """Score a standalone file (PDF or .tex) without prior verify-claims."""
    if not file_path.exists():
        logger.error(f"Error: {file_path} not found")
        return 2

    suffix = file_path.suffix.lower()
    if suffix not in (".pdf", ".tex"):
        logger.error(f"Unsupported file type: {suffix}. Use .pdf or .tex")
        return 2

    output_dir = Path(args.output_dir) if args.output_dir else file_path.parent
    result = score_standalone_file(
        file_path,
        config,
        contribution=True,
        model=getattr(args, "model", ""),
        output_dir=output_dir,
    )

    if getattr(args, "format", "terminal") == "json":
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    ir = result.integrity_result
    print(f"\n  SciLint Score for '{file_path.stem}' (standalone file):")
    print(f"    SciLint Score:    {result.scilint_score:.4f}")
    print(f"    Internal Consistency: {ir.internal_consistency:.4f}")
    print(f"    Internal Consistency: {ir.internal_consistency:.4f}")
    print(f"    Contribution:     {result.contribution.overall:.4f}")
    if result.contribution.empirical_content > 0:
        c = result.contribution
        print(f"      Empirical:      {c.empirical_content:.4f}")
        print(f"      Progressive:    {c.progressiveness:.4f}")
        print(f"      Unification:    {c.unification:.4f}")
        print(f"      Problem-solv:   {c.problem_solving:.4f}")
        print(f"      Test severity:  {c.test_severity:.4f}")

    return 0


async def compute_contribution_axes_from_ctx(
    ctx: Any,  # ManuscriptContext
    claim_dicts: list[dict[str, Any]],
    config: LintConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, str]]:
    """Contribution axes from a ManuscriptContext (no PaperConfig needed)."""
    from sciwrite_lint.claims import classify_claims_batch
    from sciwrite_lint.scoring.contribution import compute_all_contribution_axes

    model = getattr(args, "model", "") or ""

    # Enrich claims with paper context
    abstract = ctx.abstract or ""
    methods_sections = ctx.get_section_by_title(
        "method",
        "approach",
        "experiment",
        "setup",
        "implementation",
    )
    methods_text = "\n\n".join(s.clean_text for s in methods_sections)
    results_sections = ctx.get_section_by_title(
        "result",
        "evaluation",
        "finding",
        "ablation",
        "analysis",
    )
    results_text = "\n\n".join(s.clean_text for s in results_sections)

    paper_context = f"ABSTRACT: {abstract[:1000]}"
    if methods_text:
        paper_context += f"\n\nMETHODS SUMMARY: {methods_text[:1500]}"
    if results_text:
        paper_context += f"\n\nRESULTS SUMMARY: {results_text[:1500]}"

    enriched_claims = []
    for c in claim_dicts:
        enriched = dict(c)
        original_ctx = enriched.get("context", "")
        enriched["context"] = f"{original_ctx}\n\n{paper_context}"
        enriched_claims.append(enriched)

    logger.info("Classifying claims for contribution scoring...")
    classifications = await classify_claims_batch(enriched_claims, config, model)

    # Laudan sections
    intro_sections = ctx.get_section_by_title(
        "introduction",
        "intro",
        "overview",
        "background",
    )
    limit_sections = ctx.get_section_by_title(
        "limitation",
        "discussion",
        "conclusion",
        "threat",
        "future work",
        "shortcoming",
    )
    intro_text = "\n\n".join(s.clean_text for s in intro_sections)
    limitations_text = "\n\n".join(s.clean_text for s in limit_sections)

    if len(intro_text) < 200 and abstract:
        intro_text = f"{abstract}\n\n{intro_text}"

    return await compute_all_contribution_axes(
        claim_dicts, classifications, intro_text, limitations_text, config, model
    )


async def compute_contribution_axes(
    pc: PaperConfig,
    claims: list[dict[str, Any]],
    config: LintConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, str]]:
    """Run claim taxonomy + contribution axes for a paper."""
    from sciwrite_lint.claims import classify_claims_batch
    from sciwrite_lint.manuscript_store import ManuscriptContext
    from sciwrite_lint.scoring.contribution import compute_all_contribution_axes

    model = getattr(args, "model", "") or ""

    # Build manuscript context for section selection
    ctx = ManuscriptContext.from_latex(pc.file_path, config)

    # Enrich claim context with abstract + methods for better taxonomy
    abstract = ctx.abstract or ""
    methods_sections = ctx.get_section_by_title(
        "method",
        "approach",
        "experiment",
        "setup",
        "implementation",
    )
    methods_text = "\n\n".join(s.clean_text for s in methods_sections)
    results_sections = ctx.get_section_by_title(
        "result",
        "evaluation",
        "finding",
        "ablation",
        "analysis",
    )
    results_text = "\n\n".join(s.clean_text for s in results_sections)

    # Add paper context to each claim for better classification
    paper_context = f"ABSTRACT: {abstract[:1000]}"
    if methods_text:
        paper_context += f"\n\nMETHODS SUMMARY: {methods_text[:1500]}"
    if results_text:
        paper_context += f"\n\nRESULTS SUMMARY: {results_text[:1500]}"

    enriched_claims = []
    for c in claims:
        enriched = dict(c)
        original_ctx = enriched.get("context", "")
        enriched["context"] = f"{original_ctx}\n\n{paper_context}"
        enriched_claims.append(enriched)

    # Classify claims with enriched context
    logger.info("Classifying claims for contribution scoring...")
    classifications = await classify_claims_batch(enriched_claims, config, model)

    # Extract sections for Laudan (intro + limitations)
    intro_sections = ctx.get_section_by_title(
        "introduction",
        "intro",
        "overview",
        "background",
    )
    limit_sections = ctx.get_section_by_title(
        "limitation",
        "discussion",
        "conclusion",
        "threat",
        "future work",
        "shortcoming",
    )
    intro_text = "\n\n".join(s.clean_text for s in intro_sections)
    limitations_text = "\n\n".join(s.clean_text for s in limit_sections)

    # Prepend abstract to intro if intro is thin
    if len(intro_text) < 200 and abstract:
        intro_text = f"{abstract}\n\n{intro_text}"

    return await compute_all_contribution_axes(
        claims, classifications, intro_text, limitations_text, config, model
    )
