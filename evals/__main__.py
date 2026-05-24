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


def _register_optional_extensions(
    sub: argparse._SubParsersAction,  # type: ignore[type-arg]
    rw_sub: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> None:
    """Load CLI extensions discovered under ``evals``.

    An underscore-prefixed subpackage is a CLI extension when its
    ``__init__.py`` defines ``register_cli(sub, rw_sub)``. Discovery walks
    ``evals.__path__`` via ``pkgutil``, so no extension package names
    appear in this source — adding or removing one is a filesystem
    change.
    """
    import importlib
    import pkgutil

    import evals as _pkg

    for _finder, name, ispkg in pkgutil.iter_modules(_pkg.__path__):
        if not ispkg or not name.startswith("_"):
            continue
        ext = importlib.import_module(f"{_pkg.__name__}.{name}")
        register = getattr(ext, "register_cli", None)
        if register is not None:
            register(sub, rw_sub)


def main(argv: list[str] | None = None) -> int:
    from evals.cli_eval import (
        run_eval_calibration,
        run_eval_scilint_score,
        run_eval_synthetic,
        run_rw_corpus,
        run_rw_fetch,
        run_rw_fpr,
        run_rw_inject,
        run_rw_pipeline,
        run_rw_report,
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
    p_rw_corpus.set_defaults(func=run_rw_corpus)

    p_rw_fpr = rw_sub.add_parser("fpr", help="FPR eval (Sonnet-judged)")
    p_rw_fpr.add_argument("--workspace", default="real_world_corpus")
    p_rw_fpr.add_argument("--output", help="Output directory")
    p_rw_fpr.add_argument("--max-papers", type=int)
    p_rw_fpr.add_argument("--max-findings", type=int, default=20)
    p_rw_fpr.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_rw_fpr.set_defaults(func=run_rw_fpr)

    p_rw_inject = rw_sub.add_parser("inject", help="Inject errors, measure detection")
    p_rw_inject.add_argument("--workspace", default="real_world_corpus")
    p_rw_inject.add_argument("--output", help="Output directory")
    p_rw_inject.add_argument("--max-papers", type=int)
    p_rw_inject.add_argument("--seed", type=int, default=42)
    p_rw_inject.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_rw_inject.add_argument("--llm", action="store_true", help="Include LLM rules")
    p_rw_inject.set_defaults(func=run_rw_inject)

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
    p_rw_pipeline.set_defaults(func=run_rw_pipeline)

    p_rw_report = rw_sub.add_parser("report", help="Show aggregated results")
    p_rw_report.add_argument("--results-dir", default="real_world_results")
    p_rw_report.set_defaults(func=run_rw_report)

    p_rw_fetch = rw_sub.add_parser(
        "fetch",
        help=(
            "OA fetch-only eval: run the 14-source waterfall on a curated "
            "set of references; no GROBID, no LLM. Smoke-tests the "
            "fulltext/ adapters + ranker + validator against live endpoints."
        ),
    )
    p_rw_fetch.add_argument(
        "--output",
        help="Directory to write oa_fetch_eval.json into (default: no file output)",
    )
    p_rw_fetch.add_argument(
        "--download",
        help=(
            "Keep downloaded PDFs in this directory. Without --download, "
            "downloads go to a temporary directory and are discarded."
        ),
    )
    p_rw_fetch.add_argument(
        "--email",
        default="",
        help="Polite-contact email for Unpaywall + User-Agent.",
    )
    p_rw_fetch.set_defaults(func=run_rw_fetch)

    # Load optional underscore-prefixed CLI extension subpackages (if any).
    _register_optional_extensions(sub, rw_sub)

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
