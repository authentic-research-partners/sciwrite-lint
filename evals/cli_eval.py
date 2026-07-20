"""CLI handlers for user-facing eval commands."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger


def run_eval_synthetic(args: argparse.Namespace) -> int:
    """Run synthetic detection evaluation."""
    from evals.synthetic import print_summary, run_synthetic_eval

    checks = args.checks.split(",") if args.checks else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    result = run_synthetic_eval(checks=checks, output_dir=output_dir)
    print_summary(result)

    total_fn = sum(m.fn for m in result.metrics.values())
    return 1 if total_fn > 0 else 0


def run_eval_synth_corpus(args: argparse.Namespace) -> int:
    """Render synthetic scenarios and report cross-format check coverage.

    Default: PDF mode — render each scenario via ``pdflatex`` and report the
    deterministic rules that survive a GROBID round-trip (rule-level, since
    PDF loses symbols). ``--llm``: render the LLM scenarios to tex/md and
    report which LLM-engine rules fired (recall-level, since LLM output is
    non-deterministic). ``--out DIR`` keeps the materialized corpus.
    """
    import tempfile

    out_dir = Path(args.out) if getattr(args, "out", None) else None
    runner = _run_synth_llm if getattr(args, "llm", False) else _run_synth_pdf
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Materializing synthetic corpus under {out_dir}")
        return runner(out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        return runner(Path(tmp))


def _run_synth_pdf(dest: Path) -> int:
    """PDF coverage: render to PDF, lint via GROBID, flag regressions."""
    import shutil

    from sciwrite_lint.pdf.grobid import is_grobid_running

    from evals.synthetic_corpus import pdf_coverage_report

    if shutil.which("pdflatex") is None:
        logger.error(
            "synth-corpus needs pdflatex to render PDFs. Install a LaTeX "
            "distribution (e.g. `apt install texlive-latex-base "
            "texlive-latex-extra`)."
        )
        return 1
    if not asyncio.run(is_grobid_running()):
        logger.error(
            "synth-corpus needs GROBID to parse the rendered PDFs. Start it "
            "with: sciwrite-lint containers start"
        )
        return 1

    report = pdf_coverage_report(dest)
    print("\nPDF rule-level coverage (GROBID round-trip):\n")
    regressions = 0
    for cov in report:
        line = f"  {cov.name:22} detected={cov.detected_rules or '[]'}"
        if cov.regressions:
            line += f"  REGRESSION (missed {cov.regressions})"
            regressions += len(cov.regressions)
        if cov.known_gaps:
            line += f"  known-gap (PDF-lossy: {cov.known_gaps})"
        if cov.unexpected:
            line += f"  unexpected={cov.unexpected}"
        print(line)
    print()

    if regressions:
        logger.error(f"{regressions} PDF coverage regression(s)")
        return 1
    logger.info("PDF coverage: no regressions")
    return 0


def _run_synth_llm(dest: Path) -> int:
    """LLM coverage: render to tex/md, run LLM checks, report rule recall."""
    from sciwrite_lint.config import LintConfig
    from sciwrite_lint.vllm.vllm_server import _check_api_health

    from evals.synthetic_corpus import llm_coverage_report

    if asyncio.run(_check_api_health(LintConfig().llm_endpoint)) is None:
        logger.error(
            "synth-corpus --llm needs the text vLLM. Start it with: "
            "sciwrite-lint containers start"
        )
        return 1

    from evals.synthetic_corpus import LLM_RULES_UNDER_TEST

    report = asyncio.run(llm_coverage_report(dest))
    scope = ", ".join(sorted(LLM_RULES_UNDER_TEST))
    print(f"\nLLM-check rule recall (same prose, each format; scope: {scope}):\n")
    expected_total = hit_total = 0
    for cov in report:
        expected_total += len(cov.expected_rules)
        hit_total += len(cov.hit)
        line = f"  {cov.name:18} {cov.fmt:3} detected={cov.detected_rules or '[]'}"
        if cov.missed:
            line += f"  MISSED={cov.missed}"
        if cov.unexpected:
            line += f"  unexpected={cov.unexpected}"
        print(line)
    print()

    if expected_total == 0:
        logger.info("LLM coverage: no expected rules to assert")
        return 0
    recall = hit_total / expected_total
    logger.info(f"LLM coverage recall: {hit_total}/{expected_total} ({recall:.0%})")
    # A single miss is within LLM non-determinism; total recall of zero means
    # the LLM checks are not firing at all (e.g. vLLM died mid-run).
    if hit_total == 0:
        logger.error("LLM coverage: no expected rule fired — checks not running")
        return 1
    return 0


def run_synth_corpus(args: argparse.Namespace) -> int:
    """Materialize the synthetic manuscript corpus to a directory.

    No services for ``tex``/``md`` (pure file writes); ``pdf`` needs
    ``pdflatex``. Writes a ``MANIFEST.md`` describing each scenario and the
    checks it is built to trigger, so the corpus is self-documenting.
    """
    from evals.synthetic_corpus import materialize_corpus, render_manifest

    formats = tuple(f.strip() for f in args.format.split(",") if f.strip())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        entries = materialize_corpus(out, formats)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    manifest = render_manifest(entries)
    (out / "MANIFEST.md").write_text(manifest, encoding="utf-8")
    print(manifest)
    logger.info(
        f"Wrote {len(entries)} scenarios ({', '.join(formats)}) to {out} "
        "— see MANIFEST.md"
    )
    return 0


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
        fresh=getattr(args, "fresh", False),
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


def run_rw_corpus(args: argparse.Namespace) -> int:
    """Handler for ``eval-real-world corpus``."""
    from eval_real_world.runner import run_corpus
    from sciwrite_lint.__main__ import _load_config

    cats = args.categories.split(",") if args.categories else None
    srcs = args.sources.split(",") if getattr(args, "sources", None) else None
    config = _load_config(args)
    run_corpus(
        Path(args.workspace),
        n=args.n,
        categories=cats,
        seed=args.seed,
        sources=srcs,
        config=config,
    )
    return 0


def run_rw_fpr(args: argparse.Namespace) -> int:
    """Handler for ``eval-real-world fpr``."""
    from eval_real_world.runner import run_fpr
    from sciwrite_lint.__main__ import _load_config

    config = _load_config(args)
    output = Path(args.output) if args.output else None
    run_fpr(
        Path(args.workspace),
        output_dir=output,
        config=config,
        max_papers=args.max_papers,
        max_findings_per_paper=args.max_findings,
    )
    return 0


def run_rw_inject(args: argparse.Namespace) -> int:
    """Handler for ``eval-real-world inject``."""
    from eval_real_world.runner import run_inject
    from sciwrite_lint.__main__ import _load_config

    config = _load_config(args)
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


def run_rw_pipeline(args: argparse.Namespace) -> int:
    """Handler for ``eval-real-world pipeline``."""
    from eval_real_world.runner import run_full_pipeline
    from sciwrite_lint.__main__ import _load_config
    from sciwrite_lint.cli.config import check_api_config

    config = _load_config(args)
    # Pipeline runs the full fetch stage (OA PDF acquisition), which
    # requires polite_email for Unpaywall and Retraction Watch. Fail
    # fast here rather than letting every paper succeed through verify
    # and then error out mid-pipeline at fetch.
    api_errors = check_api_config(config, needs_email=True)
    if api_errors:
        for e in api_errors:
            logger.error(f"  ✗ {e}")
        return 2
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


def run_rw_report(args: argparse.Namespace) -> int:
    """Handler for ``eval-real-world report``."""
    from eval_real_world.runner import run_report

    run_report(Path(args.results_dir))
    return 0


def run_rw_fetch(args: argparse.Namespace) -> int:
    """Handler for ``eval-real-world fetch``."""
    from eval_real_world.fetch import run_fetch_eval

    output = Path(args.output) if getattr(args, "output", None) else None
    download = Path(args.download) if getattr(args, "download", None) else None
    asyncio.run(
        run_fetch_eval(
            output_dir=output,
            download_to=download,
            email=getattr(args, "email", ""),
        )
    )
    return 0
