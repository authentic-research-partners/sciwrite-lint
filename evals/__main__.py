"""CLI for eval commands: python -m evals <command>

User-facing commands:
    python -m evals eval-synthetic [--checks ...]
    python -m evals eval-scilint-score [--axes ...]
    python -m evals eval-calibration [--papers ...]
    python -m evals eval-real-world corpus|fpr|inject|report [...]
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from evals.cli_eval import (
        run_eval_calibration,
        run_eval_real_world,
        run_eval_scilint_score,
        run_eval_synthetic,
    )

    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Evaluation commands for sciwrite-lint.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- eval-synthetic ---
    p_synth = sub.add_parser(
        "eval-synthetic", help="Run synthetic detection evaluation (P/R/F1 per check)"
    )
    p_synth.add_argument(
        "--checks",
        help="Comma-separated check IDs (default: all)",
    )
    p_synth.add_argument("--output-dir", help="Output directory (default: results)")
    p_synth.set_defaults(func=run_eval_synthetic)

    # --- eval-scilint-score ---
    p_score = sub.add_parser(
        "eval-scilint-score",
        help="Evaluate SciLint Score claim taxonomy and contribution axes",
    )
    p_score.add_argument("--axes", help="Comma-separated axes: taxonomy, laudan")
    p_score.add_argument("--output-dir", help="Output directory")
    p_score.add_argument("--model", default="", help="vLLM model preset")
    p_score.set_defaults(func=run_eval_scilint_score)

    # --- eval-calibration ---
    p_cal = sub.add_parser(
        "eval-calibration",
        help="Run SciLint Score calibration against ground-truth papers",
    )
    p_cal.add_argument(
        "--fresh",
        action="store_true",
        help="Re-run pipeline from scratch (ignore all caches)",
    )
    p_cal.add_argument("--papers", help="Comma-separated short names")
    p_cal.add_argument("--output-dir", default=None, help="Output directory")
    p_cal.add_argument("--model", default="", help="vLLM model preset")
    p_cal.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max papers in concurrent stages (vLLM, network); GPU stages batched. "
        "Default 2, validated up to 4",
    )
    p_cal.add_argument("--json", action="store_true", help="Output JSON")
    p_cal.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_cal.set_defaults(func=run_eval_calibration)

    # --- eval-real-world ---
    p_rw = sub.add_parser(
        "eval-real-world", help="Real-world evaluation on arXiv papers"
    )
    rw_sub = p_rw.add_subparsers(dest="rw_action")

    p_rw_corpus = rw_sub.add_parser("corpus", help="Download papers")
    p_rw_corpus.add_argument("--workspace", default="real_world_corpus")
    p_rw_corpus.add_argument("-n", type=int, default=100)
    p_rw_corpus.add_argument("--categories", help="Comma-separated arXiv categories")
    p_rw_corpus.add_argument("--sources", help="Comma-separated: arxiv,biorxiv")
    p_rw_corpus.add_argument("--seed", type=int, default=42)
    p_rw_corpus.set_defaults(func=run_eval_real_world)

    p_rw_fpr = rw_sub.add_parser("fpr", help="FPR eval (Sonnet-judged)")
    p_rw_fpr.add_argument("--workspace", default="real_world_corpus")
    p_rw_fpr.add_argument("--output", help="Output directory")
    p_rw_fpr.add_argument("--max-papers", type=int)
    p_rw_fpr.add_argument("--max-findings", type=int, default=20)
    p_rw_fpr.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_rw_fpr.set_defaults(func=run_eval_real_world)

    p_rw_inject = rw_sub.add_parser("inject", help="Inject errors, measure detection")
    p_rw_inject.add_argument("--workspace", default="real_world_corpus")
    p_rw_inject.add_argument("--output", help="Output directory")
    p_rw_inject.add_argument("--max-papers", type=int)
    p_rw_inject.add_argument("--seed", type=int, default=42)
    p_rw_inject.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_rw_inject.add_argument("--llm", action="store_true", help="Include LLM rules")
    p_rw_inject.set_defaults(func=run_eval_real_world)

    p_rw_pipeline = rw_sub.add_parser(
        "pipeline", help="Full pipeline (SciLint Score) on corpus papers"
    )
    p_rw_pipeline.add_argument("--workspace", default="real_world_corpus")
    p_rw_pipeline.add_argument("--max-papers", type=int)
    p_rw_pipeline.add_argument("--output", help="Output directory")
    p_rw_pipeline.add_argument(
        "--no-contribution",
        action="store_true",
        help="Skip contribution axes (faster, integrity only)",
    )
    p_rw_pipeline.add_argument("--model", default="", help="vLLM model override")
    p_rw_pipeline.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max papers in concurrent stages (vLLM, network); GPU stages batched. "
        "Default 2, validated up to 4",
    )
    p_rw_pipeline.add_argument(
        "--judge", action="store_true", help="Sonnet judges findings as TP/FP"
    )
    p_rw_pipeline.add_argument(
        "--max-judge-findings",
        type=int,
        default=20,
        help="Max findings to judge per paper (default: 20)",
    )
    p_rw_pipeline.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_rw_pipeline.set_defaults(func=run_eval_real_world)

    p_rw_report = rw_sub.add_parser("report", help="Show aggregated results")
    p_rw_report.add_argument("--results-dir", default="real_world_results")
    p_rw_report.set_defaults(func=run_eval_real_world)


    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "eval-real-world" and not getattr(args, "rw_action", None):
        p_rw.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
