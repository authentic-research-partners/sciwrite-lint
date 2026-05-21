"""Synthetic evaluation framework for sciwrite-lint checks.

Generates LaTeX documents with known issues (and clean variants),
runs the linter checks, and computes precision/recall/F1 per check.

This evaluates the *linter's detection performance*, not paper quality.
Each synthetic case has explicit ground truth — no LLM judgment needed
for scoring (though LLM checks themselves still require vLLM).

Usage:
    from evals.synthetic import run_synthetic_eval
    results = run_synthetic_eval(checks=["dangling-cite", "dangling-ref"])
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import CitationMetadata, Finding

from evals.synthetic_types import (
    CaseResult,
    CheckMetrics,
    EvalResult,
    ExpectedFinding,
    SyntheticCase,
)


# ---------------------------------------------------------------------------
# Runner: execute checks against synthetic cases
# ---------------------------------------------------------------------------


def _run_check_on_case(
    case: SyntheticCase,
    config: LintConfig,
) -> list[Finding]:
    """Run a single check on a synthetic case (LaTeX or PDF)."""
    if case.grobid_result is not None:
        return _run_check_on_pdf(
            case.grobid_result, case.check_id, config, case.metadata
        )
    return _run_check_on_tex(
        case.tex_content,
        case.check_id,
        config,
        case.metadata,
        figure_descriptions=case.figure_descriptions,
    )


def _run_check_on_pdf(
    grobid_result: Any,
    check_id: str,
    config: LintConfig,
    metadata: dict[str, Any] | None = None,
) -> list[Finding]:
    """Run a single check in PDF mode using a GrobidResult."""
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_check
    from sciwrite_lint.manuscript_store import (
        ManuscriptContext,
        clear_cache,
        set_manuscript_context,
    )

    # Pipeline-stage checks (not in registry) — route directly
    _PIPELINE_STAGE_IDS = {"claim-support", "cite-purpose", "reference-unreliable"}
    if check_id in _PIPELINE_STAGE_IDS:
        if metadata:
            return _run_database_check(check_id, metadata, "")
        raise ValueError(f"Check {check_id} requires metadata (pipeline-stage check)")

    ensure_checks_loaded()
    entry = get_check(check_id)
    if entry is None:
        raise ValueError(f"Check not found: {check_id}")

    meta, check_fn = entry

    # Build ManuscriptContext from GrobidResult
    pdf_path = Path("/tmp/synthetic_eval.pdf")
    ctx = ManuscriptContext.from_grobid(pdf_path, grobid_result)
    set_manuscript_context(pdf_path, ctx)
    config._manuscript_context = ctx  # type: ignore[attr-defined]

    try:
        if meta.category == "local-llm":
            return _run_llm_check(check_fn, "", config, pdf_path=pdf_path)
        return check_fn(pdf_path, config)
    finally:
        clear_cache()
        if hasattr(config, "_manuscript_context"):
            del config._manuscript_context  # type: ignore[attr-defined]


def _run_check_on_tex(
    tex_content: str,
    check_id: str,
    config: LintConfig,
    metadata: dict[str, Any] | None = None,
    figure_descriptions: str = "",
) -> list[Finding]:
    """Run a single check on synthetic LaTeX content."""
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_check

    # Pipeline-stage checks (not in registry) — route directly
    _PIPELINE_STAGE_IDS = {"claim-support", "cite-purpose", "reference-unreliable"}
    if check_id in _PIPELINE_STAGE_IDS:
        if metadata:
            return _run_database_check(check_id, metadata, tex_content)
        raise ValueError(f"Check {check_id} requires metadata (pipeline-stage check)")

    ensure_checks_loaded()
    entry = get_check(check_id)
    if entry is None:
        raise ValueError(f"Check not found: {check_id}")

    meta, check_fn = entry

    # Checks that use pre-computed metadata (reference-exists, reference-accuracy)
    if metadata and meta.category == "reference-db":
        return _run_database_check(check_id, metadata, tex_content)

    if meta.category == "local-llm":
        return _run_llm_check(
            check_fn, tex_content, config, figure_descriptions=figure_descriptions
        )

    # Manuscript engine: write to temp file and run
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tex", delete=False, encoding="utf-8"
    ) as f:
        f.write(tex_content)
        tex_path = Path(f.name)

    try:
        return check_fn(tex_path, config)
    finally:
        tex_path.unlink(missing_ok=True)


def _run_database_check(
    check_id: str,
    metadata: dict[str, Any],
    tex_content: str = "",
) -> list[Finding]:
    """Run a reference-db-engine check with synthetic metadata."""
    if check_id == "claim-support":
        return _run_claim_support_check(metadata, tex_content)
    if check_id == "cite-purpose":
        return _run_cite_purpose_check(metadata, tex_content)

    # Convert raw dicts to CitationMetadata objects
    all_metadata: dict[str, CitationMetadata] = {}
    for key, data in metadata.items():
        if key.startswith("_"):
            continue  # skip internal keys like _claim_results
        all_metadata[key] = CitationMetadata(
            key=data.get("key", key),
            api_match=data.get("api_match", ""),
            canonical=data.get("canonical", {}),
            bibitem=data.get("bibitem", {}),
            issues=data.get("issues", []),
        )

    if check_id == "reference-exists":
        from sciwrite_lint.checks.reference_exists import (
            check_reference_exists_from_metadata,
        )

        return check_reference_exists_from_metadata(all_metadata)

    if check_id == "reference-accuracy":
        from sciwrite_lint.checks.reference_accuracy import (
            check_reference_accuracy_from_metadata,
        )

        return check_reference_accuracy_from_metadata(all_metadata)

    if check_id == "retracted-cite":
        from sciwrite_lint.checks.retracted_cite import (
            check_retraction_from_metadata,
        )

        return check_retraction_from_metadata(all_metadata)

    raise ValueError(f"Unknown database check: {check_id}")


def _run_claim_support_check(
    metadata: dict[str, Any],
    tex_content: str,
) -> list[Finding]:
    """Run claim-support check using pre-computed or live claim results.

    Two modes:
    - Pre-computed: ``_claim_results`` in metadata → finding conversion only.
    - Live vLLM: ``_live_context`` in metadata → runs ``verify_claim_vllm()``
      with real LLM, then converts result to findings.  Used for evals that
      test the full verification pipeline (e.g., context narrowing).
    """
    from sciwrite_lint.checks.claim_support import claims_to_findings

    live = metadata.get("_live_context")
    if live:
        import asyncio

        from sciwrite_lint.eval_claims import ClaimContext, Section, verify_claim_vllm

        claim = ClaimContext(
            key=live["key"],
            context=live["context"],
            line=live.get("line", 1),
        )
        sections = [
            Section(title=s["title"], text=s["text"], index=i)
            for i, s in enumerate(live["sections"])
        ]
        verdict = asyncio.run(verify_claim_vllm(claim, sections))
        verdict["key"] = claim.key
        verdict["line"] = claim.line
        return claims_to_findings([verdict], Path("synthetic.tex"))

    claim_results: list[dict[str, Any]] = metadata.get("_claim_results", [])
    if not claim_results:
        return []

    # Create a fake tex_path for the finding's file field
    return claims_to_findings(claim_results, Path("synthetic.tex"))


def _run_cite_purpose_check(
    metadata: dict[str, Any],
    tex_content: str,
) -> list[Finding]:
    """Run cite-purpose check using pre-computed purpose results."""
    from sciwrite_lint.checks.cite_purpose import cite_purposes_to_findings

    purpose_results: list[dict[str, Any]] = metadata.get("_purpose_results", [])
    if not purpose_results:
        return []

    return cite_purposes_to_findings(purpose_results, Path("synthetic.tex"))


# Cache LLM results by content hash — avoids re-running vLLM when
# multiple eval cases share the same paper body.  APC caches the KV
# prefix inside vLLM; this cache avoids the Python-side overhead of
# rebuilding ManuscriptContext and dispatching the batch.
_llm_result_cache: dict[str, list[Finding]] = {}


def _run_llm_check(
    check_fn: Callable,
    tex_content: str,
    config: LintConfig,
    pdf_path: Path | None = None,
    figure_descriptions: str = "",
) -> list[Finding]:
    """Run an LLM-engine check via the async batch runner."""
    import asyncio
    import hashlib

    if pdf_path is not None:
        # PDF mode — ManuscriptContext already in cache
        from sciwrite_lint.pipeline import run_llm_checks_batched

        findings = asyncio.run(run_llm_checks_batched(pdf_path, config))
        target_id = check_fn.check_meta.id  # type: ignore[attr-defined]
        return [f for f in findings if f.rule_id == target_id]

    # Check cache — same paper body + figure descriptions → reuse all LLM findings
    combined = tex_content + "\0" + figure_descriptions
    content_hash = hashlib.md5(combined.encode()).hexdigest()[:16]
    if content_hash in _llm_result_cache:
        target_id = check_fn.check_meta.id  # type: ignore[attr-defined]
        return [f for f in _llm_result_cache[content_hash] if f.rule_id == target_id]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tex", delete=False, encoding="utf-8"
    ) as f:
        f.write(tex_content)
        tex_path = Path(f.name)

    try:
        from sciwrite_lint.pipeline import run_llm_checks_batched

        # run_llm_checks_batched runs ALL llm checks; we filter after
        findings = asyncio.run(run_llm_checks_batched(tex_path, config))
        _llm_result_cache[content_hash] = findings
        target_id = check_fn.check_meta.id  # type: ignore[attr-defined]
        return [f for f in findings if f.rule_id == target_id]
    finally:
        tex_path.unlink(missing_ok=True)


def _match_findings(
    findings: list[Finding],
    expected: list[ExpectedFinding],
    check_id: str,
) -> tuple[int, int, int, list[bool]]:
    """Match actual findings to expected ground truth.

    Returns (tp, fp, fn, matched) where matched is parallel to expected.
    """
    # Findings for this check only
    check_findings = [f for f in findings if f.rule_id == check_id]

    matched = []
    used_findings: set[int] = set()

    for exp in expected:
        found = False
        for i, f in enumerate(check_findings):
            if i in used_findings:
                continue
            # Match by context substring in finding context or message
            if exp.context and (
                exp.context in (f.context or "") or exp.context in (f.message or "")
            ):
                found = True
                used_findings.add(i)
                break
        # Next pass: if no context-specific match, any unmatched finding counts
        if not found and exp.context == "":
            for i, f in enumerate(check_findings):
                if i not in used_findings:
                    found = True
                    used_findings.add(i)
                    break
        matched.append(found)

    tp = sum(matched)
    fn = len(expected) - tp
    # FP = findings from this check that weren't matched to any expected
    fp = len(check_findings) - len(used_findings)

    return tp, fp, fn, matched


# ---------------------------------------------------------------------------
# Concurrent LLM execution: run all unique papers in one event loop
# ---------------------------------------------------------------------------


def _setup_figure_workspace(
    figure_descriptions: str, config: LintConfig, tmp_dir: Path
) -> str:
    """Create a temp workspace with vision_cache entries for figure eval.

    Returns the paper name to set on config.current_paper.
    """
    from sciwrite_lint.references.workspace_db import get_db, save_vision_entry

    paper_name = "_synth_eval"
    ws_root = tmp_dir / paper_name
    parsed = ws_root / "parsed"
    parsed.mkdir(parents=True, exist_ok=True)

    # Store the full figure description block as a single entry
    with get_db(ws_root) as conn:
        save_vision_entry(
            conn,
            "_all_figures",
            image_hash="synthetic",
            description=figure_descriptions,
        )

    config.references_dir = tmp_dir
    config.current_paper = paper_name
    return paper_name


def _warm_llm_cache(cases: list[SyntheticCase], config: LintConfig) -> None:
    """Run all unique LLM papers concurrently in one ``asyncio.run()``.

    Collects unique tex_content from LLM cases, writes temp files, fires
    ``run_llm_checks_batched`` for all papers concurrently, and caches
    results.  After this, every ``_run_llm_check`` call hits the cache.

    For cases with ``figure_descriptions``, creates a temp workspace with
    vision_cache entries so ``_load_figure_descriptions`` picks them up.
    """
    import asyncio
    import hashlib

    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_check

    ensure_checks_loaded()

    # Collect unique papers that need LLM runs
    # Key includes figure_descriptions hash so different figure combos get separate runs
    unique_papers: dict[str, tuple[str, str]] = {}  # hash → (tex_content, fig_descs)
    for case in cases:
        if not case.tex_content:
            continue
        entry = get_check(case.check_id)
        if entry is None:
            continue
        meta, _ = entry
        if meta.category != "local-llm":
            continue
        combined = case.tex_content + "\0" + case.figure_descriptions
        content_hash = hashlib.md5(combined.encode()).hexdigest()[:16]
        if content_hash not in unique_papers:
            unique_papers[content_hash] = (case.tex_content, case.figure_descriptions)

    if not unique_papers:
        return

    logger.info(
        "Running {} unique LLM papers concurrently",
        len(unique_papers),
    )

    async def _run_all() -> None:
        from sciwrite_lint.pipeline import run_llm_checks_batched

        saved_refs_dir = config.references_dir
        saved_paper = config.current_paper

        tex_paths: list[tuple[str, Path]] = []
        for content_hash, (tex_content, _fig) in unique_papers.items():
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".tex", delete=False, encoding="utf-8"
            ) as f:
                f.write(tex_content)
                tex_paths.append((content_hash, Path(f.name)))

        # Split papers by whether they need per-paper config-state setup.
        # Figure-workspace setup mutates config.current_paper + references_dir,
        # so those cases must stay sequential. Plain text cases share the
        # same neutral config and can fan out concurrently via asyncio.gather,
        # which is what unlocks vLLM's continuous-batching throughput for
        # per-sentence checks like prose-quality (each case contributes
        # ~1 query; 33 cases × 1 = 33 in-flight queries instead of 1).
        simple_jobs: list[tuple[str, Path]] = []
        complex_jobs: list[tuple[str, Path, str]] = []
        for content_hash, tex_path in tex_paths:
            _, fig_descs = unique_papers[content_hash]
            if fig_descs:
                complex_jobs.append((content_hash, tex_path, fig_descs))
            else:
                simple_jobs.append((content_hash, tex_path))

        async def _run_one_simple(
            content_hash: str, tex_path: Path
        ) -> tuple[str, list]:
            try:
                return (
                    content_hash,
                    await run_llm_checks_batched(tex_path, config),
                )
            except Exception as e:
                logger.error("LLM batch failed for {}: {}", content_hash, e)
                return (content_hash, [])

        if simple_jobs:
            config.current_paper = ""
            results = await asyncio.gather(
                *(_run_one_simple(ch, tp) for ch, tp in simple_jobs)
            )
            for content_hash, findings in results:
                _llm_result_cache[content_hash] = findings
            for _, tex_path in simple_jobs:
                tex_path.unlink(missing_ok=True)

        # Figure-workspace cases still run sequentially because each mutates
        # config.current_paper / references_dir to a different temp workspace.
        import shutil

        for content_hash, tex_path, fig_descs in complex_jobs:
            tmp_dir = Path(tempfile.mkdtemp())
            try:
                _setup_figure_workspace(fig_descs, config, tmp_dir)
                result = await run_llm_checks_batched(tex_path, config)
                _llm_result_cache[content_hash] = result
            except Exception as e:
                logger.error("LLM batch failed for {}: {}", content_hash, e)
                _llm_result_cache[content_hash] = []
            finally:
                tex_path.unlink(missing_ok=True)
                shutil.rmtree(tmp_dir, ignore_errors=True)

        config.references_dir = saved_refs_dir
        config.current_paper = saved_paper

    asyncio.run(_run_all())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_synthetic_eval(
    checks: list[str] | None = None,
    output_dir: Path | None = None,
) -> EvalResult:
    """Run synthetic evaluation for specified checks.

    Args:
        checks: Check IDs to evaluate. None = text + database checks.
        output_dir: Where to save results. None = evals/results/.

    Returns:
        EvalResult with per-case results and aggregate metrics.
    """
    from evals.synthetic_generators import generate_cases

    cases = generate_cases(checks)
    config = LintConfig()
    result = EvalResult()
    _llm_result_cache.clear()

    # Clear full-paper body cache so fresh papers are processed
    from sciwrite_lint.checks.full_paper_consistency import _body_cache

    _body_cache.clear()

    # Batch all unique LLM papers concurrently, then score from cache
    _warm_llm_cache(cases, config)

    for case in cases:
        logger.info("Running case: {}", case.name)

        t0 = time.monotonic()
        try:
            findings = _run_check_on_case(case, config)
        except Exception as e:
            logger.error("Case {} failed: {}", case.name, e)
            continue
        elapsed = time.monotonic() - t0

        tp, fp, fn, matched = _match_findings(findings, case.expected, case.check_id)

        cr = CaseResult(
            name=case.name,
            check_id=case.check_id,
            tp=tp,
            fp=fp,
            fn=fn,
            findings=[f.model_dump() for f in findings],
            expected=[e.model_dump() for e in case.expected],
            matched=matched,
            elapsed_s=round(elapsed, 2),
        )
        result.cases.append(cr)

        # Accumulate per-check metrics
        if case.check_id not in result.metrics:
            result.metrics[case.check_id] = CheckMetrics(check_id=case.check_id)
        m = result.metrics[case.check_id]
        m.tp += tp
        m.fp += fp
        m.fn += fn
        m.cases_run += 1

    # Save results
    out = output_dir or config.results_dir
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "synthetic.json")

    return result


def print_summary(result: EvalResult) -> None:
    """Print human-readable summary of synthetic eval results."""
    print("\n" + "=" * 64)
    print("  SYNTHETIC EVALUATION RESULTS")
    print("=" * 64)

    total_tp = sum(m.tp for m in result.metrics.values())
    total_fp = sum(m.fp for m in result.metrics.values())
    total_fn = sum(m.fn for m in result.metrics.values())
    total_cases = sum(m.cases_run for m in result.metrics.values())
    total_time = sum(c.elapsed_s for c in result.cases)
    print(
        f"\n  Cases: {total_cases}   TP: {total_tp}   FP: {total_fp}   FN: {total_fn}"
        f"   Time: {total_time:.1f}s"
    )

    print(
        f"\n  {'Check':<30} {'Cases':>5} {'TP':>4} {'FP':>4} {'FN':>4}"
        f" {'Prec':>7} {'Rec':>7} {'F1':>7} {'Time':>7}"
    )
    print("  " + "-" * 70)

    for check_id, m in sorted(result.metrics.items()):
        check_time = sum(c.elapsed_s for c in result.cases if c.check_id == check_id)
        print(
            f"  {check_id:<30} {m.cases_run:>5} {m.tp:>4} {m.fp:>4} {m.fn:>4}"
            f" {m.precision:>6.1%} {m.recall:>6.1%} {m.f1:>6.1%}"
            f" {check_time:>6.1f}s"
        )

    # Show failed cases
    failures = [c for c in result.cases if c.fn > 0 or c.fp > 0]
    if failures:
        print(f"\n  Issues ({len(failures)} cases):")
        for c in failures:
            parts = []
            if c.fn > 0:
                parts.append(f"{c.fn} missed")
            if c.fp > 0:
                parts.append(f"{c.fp} false positive")
            print(f"    {c.name}: {', '.join(parts)}")

    print()
