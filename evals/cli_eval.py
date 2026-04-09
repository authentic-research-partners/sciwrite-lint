"""CLI handlers for user-facing eval commands."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sciwrite_lint.config import LintConfig


def run_eval_synthetic(args: argparse.Namespace) -> int:
    """Run synthetic detection evaluation."""
    from evals.synthetic import print_summary, run_synthetic_eval

    checks = args.checks.split(",") if args.checks else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    result = run_synthetic_eval(checks=checks, output_dir=output_dir)
    print_summary(result)

    total_fn = sum(m.fn for m in result.metrics.values())
    return 1 if total_fn > 0 else 0


def run_eval_scilint_score(args: argparse.Namespace) -> int:
    """Run SciLint Score taxonomy and contribution axes evaluation."""
    from evals.scilint_score_eval import (
        print_scilint_score_summary,
        run_scilint_score_eval,
    )

    axes = args.axes.split(",") if args.axes else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    model = getattr(args, "model", "") or ""
    result = run_scilint_score_eval(axes=axes, output_dir=output_dir, model_name=model)
    print_scilint_score_summary(result)

    # Exit 1 if taxonomy accuracy < 60% or Laudan range accuracy < 50%
    tax_ok = (
        result.taxonomy.overall_accuracy >= 0.6 if result.taxonomy.cases_run else True
    )
    lau_ok = result.laudan.range_accuracy >= 0.5 if result.laudan.cases_run else True
    return 0 if (tax_ok and lau_ok) else 1


def run_eval_calibration(args: argparse.Namespace) -> int:
    """Run calibration eval against ground-truth papers."""
    from evals.calibration import (
        CALIBRATION_DIR,
        print_calibration_report,
        run_calibration,
        save_calibration_json,
    )

    from sciwrite_lint.__main__ import _load_config

    config = _load_config(args)
    papers = args.papers.split(",") if args.papers else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    result = run_calibration(
        calibration_dir=CALIBRATION_DIR,
        config=config,
        papers=papers,
        rerun=args.rerun,
        model=getattr(args, "model", ""),
        output_dir=output_dir,
        concurrency=getattr(args, "concurrency", 2),
    )

    print_calibration_report(result)

    out_path = (output_dir or CALIBRATION_DIR / "results") / "calibration.json"
    save_calibration_json(result, out_path)

    # Also save to results/ with timestamped history
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    save_calibration_json(result, results_dir / "calibration.json")

    history_dir = results_dir / "calibration_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(history_dir.glob("iter_*.json"))
    next_idx = len(existing)
    passed = result.total_passed
    total = result.total_constraints
    history_path = history_dir / f"iter_{next_idx:03d}_{passed}of{total}.json"
    save_calibration_json(result, history_path)

    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2))

    return 0 if result.total_passed == result.total_constraints else 1


def run_eval_real_world(args: argparse.Namespace) -> int:
    """Dispatch real-world evaluation subcommands."""
    from sciwrite_lint.__main__ import _load_config

    action = args.rw_action

    if action == "corpus":
        from eval_real_world.runner import run_corpus

        cats = args.categories.split(",") if args.categories else None
        srcs = args.sources.split(",") if getattr(args, "sources", None) else None
        config = _load_config(args) if getattr(args, "config", None) else LintConfig()
        run_corpus(
            Path(args.workspace),
            n=args.n,
            categories=cats,
            seed=args.seed,
            sources=srcs,
            config=config,
        )
        return 0

    if action == "fpr":
        from eval_real_world.runner import run_fpr

        config = _load_config(args) if getattr(args, "config", None) else LintConfig()
        output = Path(args.output) if args.output else None
        run_fpr(
            Path(args.workspace),
            output_dir=output,
            config=config,
            max_papers=args.max_papers,
            max_findings_per_paper=args.max_findings,
        )
        return 0

    if action == "inject":
        from eval_real_world.runner import run_inject

        config = _load_config(args) if getattr(args, "config", None) else LintConfig()
        output = Path(args.output) if args.output else None
        run_inject(
            Path(args.workspace),
            output_dir=output,
            config=config,
            max_papers=args.max_papers,
            seed=args.seed,
            llm=getattr(args, "llm", False),
        )
        return 0

    if action == "pipeline":
        from eval_real_world.runner import run_full_pipeline

        config = _load_config(args) if getattr(args, "config", None) else LintConfig()
        output = Path(args.output) if getattr(args, "output", None) else None
        run_full_pipeline(
            Path(args.workspace),
            output_dir=output,
            config=config,
            max_papers=getattr(args, "max_papers", None),
            contribution=not getattr(args, "no_contribution", False),
            model=getattr(args, "model", ""),
            concurrency=getattr(args, "concurrency", 2),
            judge=getattr(args, "judge", False),
            max_judge_findings=getattr(args, "max_judge_findings", 20),
        )
        return 0

    if action == "matching":
        from eval_real_world.matching import run_matching_eval

        output = Path(args.output) if getattr(args, "output", None) else None
        asyncio.run(
            run_matching_eval(
                Path(args.workspace),
                max_papers=getattr(args, "max_papers", None),
                seed=getattr(args, "seed", 42),
                output_dir=output,
            )
        )
        return 0

    if action == "report":
        from eval_real_world.runner import run_report

        run_report(Path(args.results_dir))
        return 0

    return 0
