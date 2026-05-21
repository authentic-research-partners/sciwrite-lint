"""Synthetic evaluation for SciLint Score claim taxonomy and contribution axes.

Unlike the check-level eval (P/R/F1 on findings), this evaluates:
- Claim taxonomy: per-dimension classification accuracy against ground truth
- Contribution axes: score-range validation on papers with known properties
- Problem-solving (Laudan): LLM scoring accuracy on controlled inputs

Requires vLLM for claim taxonomy and Laudan axis. Other axes are
deterministic (graph computation + aggregation from taxonomy).

Usage:
    python -m evals eval-scilint-score                   # all axes
    python -m evals eval-scilint-score --axes taxonomy   # taxonomy only
    python -m evals eval-scilint-score --axes laudan     # Laudan only
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from loguru import logger

from sciwrite_lint.claims import ClaimClassification, classify_claims_batch
from sciwrite_lint.config import LintConfig
from sciwrite_lint.scoring.contribution import compute_problem_solving_score

from evals.scilint_score_types import (
    TAXONOMY_DIMS,
    LaudanCase,
    LaudanCaseResult,
    LaudanMetrics,
    SciLintScoreEvalResult,
    TaxonomyCase,
    TaxonomyCaseResult,
    TaxonomyMetrics,
)
from evals.scilint_score_cases import (
    generate_laudan_cases,
    generate_taxonomy_cases,
)

# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


async def _run_taxonomy_eval(
    cases: list[TaxonomyCase],
    config: LintConfig,
    model_name: str = "",
) -> tuple[TaxonomyMetrics, list[TaxonomyCaseResult]]:
    """Run taxonomy classification on all cases and compute accuracy."""
    # Convert cases to the format classify_claims_batch expects
    claims = [
        {"key": c.key, "claim_text": c.claim_text, "context": c.context} for c in cases
    ]

    classifications = await classify_claims_batch(claims, config, model_name)

    # Build a map from key to classification for matching
    classified_by_idx: dict[int, ClaimClassification | None] = {}
    for i, case in enumerate(cases):
        # classify_claims_batch drops failed classifications, so we
        # need to match by key+claim_text
        classified_by_idx[i] = None

    # Re-match: classifications are in order of successful results
    cls_idx = 0
    for i, case in enumerate(cases):
        if cls_idx < len(classifications):
            c = classifications[cls_idx]
            if c.key == case.key:
                classified_by_idx[i] = c
                cls_idx += 1

    metrics = TaxonomyMetrics()
    for dim in TAXONOMY_DIMS:
        metrics.per_dim[dim] = {"correct": 0, "total": 0}

    results: list[TaxonomyCaseResult] = []

    for i, case in enumerate(cases):
        cls = classified_by_idx.get(i)
        cr = TaxonomyCaseResult(name=case.name, expected=case.expected)

        if cls is None:
            cr.failed = True
            metrics.cases_failed += 1
            results.append(cr)
            metrics.cases_run += 1
            continue

        for dim in TAXONOMY_DIMS:
            predicted = getattr(cls, dim, "")
            expected = case.expected.get(dim, "")
            match = predicted == expected
            cr.correct[dim] = match
            cr.predicted[dim] = predicted
            metrics.per_dim[dim]["total"] += 1
            if match:
                metrics.per_dim[dim]["correct"] += 1

        results.append(cr)
        metrics.cases_run += 1

    return metrics, results


async def _run_laudan_eval(
    cases: list[LaudanCase],
    config: LintConfig,
    model_name: str = "",
) -> tuple[LaudanMetrics, list[LaudanCaseResult]]:
    """Run Laudan problem-solving scoring on all cases."""
    metrics = LaudanMetrics()
    results: list[LaudanCaseResult] = []
    total_score = 0.0

    for case in cases:
        score, reasoning = await compute_problem_solving_score(
            case.intro_text,
            case.limitations_text,
            config,
            model_name,
        )

        in_range = case.expected_min <= score <= case.expected_max
        results.append(
            LaudanCaseResult(
                name=case.name,
                score=score,
                expected_min=case.expected_min,
                expected_max=case.expected_max,
                in_range=in_range,
                reasoning=reasoning,
            )
        )

        metrics.cases_run += 1
        if in_range:
            metrics.in_range += 1
        total_score += score

    if metrics.cases_run > 0:
        metrics.mean_score = total_score / metrics.cases_run

    return metrics, results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

ValidAxis = Literal["taxonomy", "laudan", "all"]


async def run_scilint_score_eval_async(
    axes: list[str] | None = None,
    output_dir: Path | None = None,
    model_name: str = "",
) -> SciLintScoreEvalResult:
    """Run SciLint Score evaluation.

    Args:
        axes: Which axes to evaluate. None or ["all"] = everything.
            Options: "taxonomy", "laudan".
        output_dir: Where to save results. None = evals/results/.
        model_name: vLLM model preset name.

    Returns:
        SciLintScoreEvalResult with per-case results and metrics.
    """
    config = LintConfig()
    result = SciLintScoreEvalResult()

    axes_set = set(axes) if axes else set()
    run_all = not axes_set or "all" in axes_set
    run_taxonomy = run_all or "taxonomy" in axes_set
    run_laudan = run_all or "laudan" in axes_set

    if run_taxonomy:
        tax_cases = generate_taxonomy_cases()
        logger.info("Running taxonomy eval ({} cases)...", len(tax_cases))
        tax_metrics, tax_results = await _run_taxonomy_eval(
            tax_cases, config, model_name
        )
        result.taxonomy = tax_metrics
        result.taxonomy_cases = tax_results

    if run_laudan:
        lau_cases = generate_laudan_cases()
        logger.info("Running Laudan eval ({} cases)...", len(lau_cases))
        lau_metrics, lau_results = await _run_laudan_eval(lau_cases, config, model_name)
        result.laudan = lau_metrics
        result.laudan_cases = lau_results

    # Save
    out = output_dir or config.results_dir
    result.save(out / "scilint_score_eval.json")

    return result


def run_scilint_score_eval(
    axes: list[str] | None = None,
    output_dir: Path | None = None,
    model_name: str = "",
) -> SciLintScoreEvalResult:
    """Sync wrapper for run_scilint_score_eval_async."""
    return asyncio.run(run_scilint_score_eval_async(axes, output_dir, model_name))


def print_scilint_score_summary(result: SciLintScoreEvalResult) -> None:
    """Print human-readable summary of SciLint Score eval results."""
    print("\n" + "=" * 64)
    print("  SCILINT SCORE EVALUATION RESULTS")
    print("=" * 64)

    if result.taxonomy.cases_run > 0:
        tm = result.taxonomy
        print(f"\n  Claim Taxonomy ({tm.cases_run} cases, {tm.cases_failed} failed)")
        print(f"  Overall accuracy: {tm.overall_accuracy:.1%}")
        print(f"\n  {'Dimension':<20} {'Accuracy':>8} {'Correct':>8} {'Total':>6}")
        print("  " + "-" * 44)
        for dim in TAXONOMY_DIMS:
            acc = tm.dim_accuracy(dim)
            d = tm.per_dim[dim]
            print(f"  {dim:<20} {acc:>7.1%} {d['correct']:>8} {d['total']:>6}")

        # Show mismatches
        mismatches = [
            c
            for c in result.taxonomy_cases
            if not c.failed and not all(c.correct.values())
        ]
        if mismatches:
            print(f"\n  Mismatches ({len(mismatches)} cases):")
            for c in mismatches:
                wrong_dims = [dim_name for dim_name, ok in c.correct.items() if not ok]
                for dim_name in wrong_dims:
                    print(
                        f"    {c.name}: {dim_name} "
                        f"predicted={c.predicted[dim_name]} "
                        f"expected={c.expected[dim_name]}"
                    )

    if result.laudan.cases_run > 0:
        lm = result.laudan
        print(f"\n  Problem-Solving / Laudan ({lm.cases_run} cases)")
        print(f"  Range accuracy: {lm.range_accuracy:.1%}")
        print(f"  Mean score:     {lm.mean_score:.4f}")
        print(f"\n  {'Case':<30} {'Score':>6} {'Range':>12} {'OK':>4}")
        print("  " + "-" * 54)
        for lc in result.laudan_cases:
            ok = "yes" if lc.in_range else "NO"
            print(
                f"  {lc.name:<30} {lc.score:>5.2f} "
                f"[{lc.expected_min:.1f}-{lc.expected_max:.1f}] "
                f"{ok:>4}"
            )

    print()
