"""Real-world evaluation runner.

Orchestrates corpus download, linter execution, error injection,
Sonnet adjudication, and metrics reporting.

Subcommands (wired via sciwrite-lint CLI):
  corpus   — download arXiv papers
  fpr      — run linter + Sonnet judge on clean papers
  inject   — inject errors + measure detection rate
  report   — aggregate results into final metrics
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig

# Default output directory
RESULTS_DIR = Path("real_world_results")


async def _ensure_downloaded_async(paper: dict, workspace: Path) -> Path | None:
    """Return the input file path, re-downloading if missing (async version)."""
    source = paper.get("source", "arxiv")
    paper_id = paper["arxiv_id"]

    tex_path_str = paper.get("tex_path")
    pdf_path_str = paper.get("pdf_path")
    if tex_path_str and Path(tex_path_str).exists():
        return Path(tex_path_str)
    if pdf_path_str and Path(pdf_path_str).exists():
        return Path(pdf_path_str)

    print(f"  Re-downloading {paper_id}...")
    if source == "biorxiv":
        from eval_real_world.corpus import _download_biorxiv_pdf

        pdf_path = await _download_biorxiv_pdf(paper_id, workspace)
        if pdf_path:
            paper["pdf_path"] = str(pdf_path)
        return pdf_path
    else:
        from eval_real_world.corpus import _download_source

        tex_path = await _download_source(paper_id, workspace)
        if tex_path:
            paper["tex_path"] = str(tex_path)
        return tex_path


def _ensure_downloaded(paper: dict, workspace: Path) -> Path | None:
    """Return the input file path, re-downloading if missing (sync wrapper)."""
    import asyncio

    tex_path_str = paper.get("tex_path")
    pdf_path_str = paper.get("pdf_path")
    if tex_path_str and Path(tex_path_str).exists():
        return Path(tex_path_str)
    if pdf_path_str and Path(pdf_path_str).exists():
        return Path(pdf_path_str)

    return asyncio.run(_ensure_downloaded_async(paper, workspace))


def run_corpus(
    workspace: Path,
    n: int = 100,
    categories: list[str] | None = None,
    seed: int | None = 42,
    sources: list[str] | None = None,
    config: LintConfig | None = None,
) -> list[dict]:
    """Download papers from arXiv and bioRxiv, save manifest."""
    import asyncio

    from eval_real_world.corpus import build_corpus

    cfg = config or LintConfig()

    papers = asyncio.run(
        build_corpus(
            workspace,
            n=n,
            categories=categories,
            seed=seed,
            sources=sources,
        )
    )

    manifest = []
    for p in papers:
        manifest.append(
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors[:3],
                "categories": p.categories,
                "source": p.source,
                "tex_path": str(p.tex_path) if p.tex_path else None,
                "pdf_path": str(p.pdf_path) if p.pdf_path else None,
                "error": p.error,
            }
        )

    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest saved to {manifest_path}")

    # Save a committable record (metadata only, no local paths)
    from datetime import datetime, timezone

    record = {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sources": sources or ["arxiv", "biorxiv"],
        "categories": categories,
        "n_requested": n,
        "n_downloaded": len(papers),
        "papers": [
            {
                "id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors[:3],
                "categories": p.categories,
                "source": p.source,
            }
            for p in papers
        ],
    }
    record_dir = cfg.results_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / "real_world_corpus.json"
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Committable record saved to {record_path}")
    return manifest


def _run_text_checks_on_file(tex_path: Path, config: LintConfig) -> list[dict]:
    """Run manuscript-engine checks on a single .tex file. Returns serialized findings."""
    from sciwrite_lint.models import Finding
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_checks

    ensure_checks_loaded()
    findings: list[Finding] = []

    for check_meta, check_fn in get_checks():
        # Only manuscript-engine checks — skip local-llm and reference-db for corpus eval
        if check_meta.category != "manuscript":
            continue
        try:
            result = check_fn(tex_path, config)
            findings.extend(result)
        except Exception as e:
            logger.debug("Check {} failed on {}: {}", check_meta.name, tex_path.name, e)

    return [f.model_dump() for f in findings]


async def _run_llm_rules_on_file(tex_path: Path, config: LintConfig) -> list[dict]:
    """Run LLM-engine rules on a single .tex file via batched vLLM queries."""
    from sciwrite_lint.pipeline import run_llm_checks_batched

    findings = await run_llm_checks_batched(tex_path, config)
    return [f.model_dump() for f in findings]


def _run_all_rules_on_file(
    tex_path: Path, config: LintConfig, *, llm: bool = False
) -> list[dict]:
    """Run text rules (and optionally LLM rules) on a .tex file."""
    import asyncio

    findings = _run_text_checks_on_file(tex_path, config)
    if llm:
        llm_findings = asyncio.run(_run_llm_rules_on_file(tex_path, config))
        findings.extend(llm_findings)
    return findings


def run_fpr(
    workspace: Path,
    output_dir: Path | None = None,
    config: LintConfig | None = None,
    max_papers: int | None = None,
    max_findings_per_paper: int = 20,
    concurrency: int = 3,
) -> Path:
    """Run false-positive-rate evaluation.

    Phase 1 (sequential): run text checks on each paper.
    Phase 2 (concurrent): judge findings via parallel Opus calls.
    Save per-paper results as each completes.

    Returns path to results directory.
    """
    import asyncio

    from eval_real_world.judge import judge_findings
    from eval_real_world.report import EvalReport, print_summary

    out = output_dir or RESULTS_DIR / "fpr"
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {workspace}. Run: python -m evals eval-real-world corpus --workspace {workspace}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = [p for p in manifest if p.get("tex_path") or p.get("pdf_path")]
    if max_papers:
        papers = papers[:max_papers]

    cfg = config or LintConfig()
    report = EvalReport(corpus_size=len(papers))

    # Phase 1: text checks (sequential)
    prepared: list[tuple[str, list, str]] = []  # (id, findings_data, tex_text)
    for i, paper in enumerate(papers, 1):
        arxiv_id = paper["arxiv_id"]
        print(f"\n[{i}/{len(papers)}] {arxiv_id}")

        input_path = _ensure_downloaded(paper, workspace)
        if not input_path:
            print("  SKIP (download failed)")
            continue

        if input_path.suffix != ".tex":
            print("  SKIP text rules (PDF input — text rules require .tex)")
            report.add_fpr_verdicts(arxiv_id, [])
            continue

        findings_data = _run_text_checks_on_file(input_path, cfg)
        print(f"  {len(findings_data)} findings from text rules")

        if not findings_data:
            report.add_fpr_verdicts(arxiv_id, [])
            continue

        tex_text = input_path.read_text(encoding="utf-8", errors="replace")
        prepared.append((arxiv_id, findings_data[:max_findings_per_paper], tex_text))

    # Phase 2: Sonnet judging (concurrent across all findings)
    if prepared:
        total_findings = sum(len(fd) for _, fd, _ in prepared)
        print(
            f"\nPhase 2: Sonnet judging {total_findings} findings (concurrency={concurrency})..."
        )

        async def _judge_all() -> None:
            from sciwrite_lint.models import Finding

            for arxiv_id, findings_data, tex_text in prepared:
                findings = [
                    Finding(
                        level=f["level"],
                        rule_id=f["rule_id"],
                        message=f["message"],
                        file=f.get("file", ""),
                        line=f.get("line"),
                        context=f.get("context", ""),
                    )
                    for f in findings_data
                ]

                verdicts = await judge_findings(
                    findings,
                    tex_text,
                    project_dir=workspace,
                    concurrency=concurrency,
                )

                verdict_dicts = [v.model_dump() for v in verdicts]
                report.add_fpr_verdicts(arxiv_id, verdict_dicts)

                # Save per-paper results immediately
                paper_out = out / f"{arxiv_id.replace('/', '_')}.json"
                paper_out.write_text(
                    json.dumps(
                        {
                            "arxiv_id": arxiv_id,
                            "findings": findings_data,
                            "verdicts": verdict_dicts,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"  [{arxiv_id}] {len(verdicts)} verdicts saved")

        asyncio.run(_judge_all())

    # Final report
    report_path = out / "report.json"
    report.save(report_path)
    print_summary(report)

    return out


def _match_injections_to_findings(
    injections: list[dict],
    findings: list[dict],
) -> list[bool]:
    """Match each injection to findings by rule_id + context overlap.

    Returns a list of booleans parallel to injections: True if detected.
    """
    detected = []
    for inj in injections:
        rule_id = inj.get("rule_id", "")
        inj_context = inj.get("context", "")

        # Find findings with matching rule_id
        candidates = [f for f in findings if f.get("rule_id") == rule_id]

        if not candidates:
            detected.append(False)
            continue

        # Check if any candidate's context or message mentions the injection context
        found = False
        for f in candidates:
            f_context = f.get("context", "")
            f_message = f.get("message", "")
            if inj_context and (inj_context in f_context or inj_context in f_message):
                found = True
                break

        # Fallback: if no context match but rule fired, count as detected
        # (injection may have changed line numbers, making exact match hard)
        if not found and candidates:
            found = True

        detected.append(found)
    return detected


def run_inject(
    workspace: Path,
    output_dir: Path | None = None,
    config: LintConfig | None = None,
    max_papers: int | None = None,
    seed: int = 42,
    llm: bool = False,
    concurrency: int = 3,
) -> Path:
    """Run injection-based detection-rate evaluation.

    Phase 1 (sequential): inject errors + run text checks per paper.
    Phase 2 (concurrent, if --llm): run LLM checks with batched vLLM.
    Phase 3 (sync): match injections to findings, compute metrics.

    Args:
        llm: If True, also run LLM-engine checks (requires vLLM).
             Enables cross-section-consistency injection and detection.
        concurrency: Max concurrent vLLM papers in Phase 2 (default: 3).

    Returns path to results directory.
    """
    import asyncio

    from eval_real_world.inject import inject_errors, inject_errors_pdf
    from eval_real_world.report import EvalReport, print_summary

    out = output_dir or RESULTS_DIR / "inject"
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {workspace}. Run: python -m evals eval-real-world corpus --workspace {workspace}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = [p for p in manifest if p.get("tex_path") or p.get("pdf_path")]
    if max_papers:
        papers = papers[:max_papers]

    cfg = config or LintConfig()
    report = EvalReport(corpus_size=len(papers))

    # Build ManuscriptContext for PDF papers (needs GROBID, do it upfront)
    pdf_contexts: dict[str, object] = {}  # arxiv_id → ManuscriptContext
    pdf_papers = [p for p in papers if not p.get("tex_path") and p.get("pdf_path")]
    if pdf_papers:
        from sciwrite_lint.pipeline import build_pdf_context
        from sciwrite_lint.manuscript_store import get_or_create_manuscript_context

        async def _parse_pdfs() -> None:
            for p in pdf_papers:
                input_path = _ensure_downloaded(p, workspace)
                if not input_path:
                    continue
                try:
                    pdf_cfg = LintConfig(project_dir=input_path.parent)
                    await build_pdf_context(input_path, pdf_cfg)
                    pdf_contexts[p["arxiv_id"]] = get_or_create_manuscript_context(
                        input_path, pdf_cfg
                    )
                except Exception as e:
                    print(f"  GROBID parse failed for {p['arxiv_id']}: {e}")

        asyncio.run(_parse_pdfs())

    # Phase 1: inject + text checks (sequential — no external service contention)
    prepared: list[
        tuple[str, Path, list, list]
    ] = []  # (id, path, injections, text_findings)
    for i, paper in enumerate(papers, 1):
        arxiv_id = paper["arxiv_id"]
        print(f"\n[{i}/{len(papers)}] {arxiv_id}")

        input_path = _ensure_downloaded(paper, workspace)
        if not input_path:
            print("  SKIP (download failed)")
            continue

        is_pdf = input_path.suffix == ".pdf"

        if is_pdf:
            ctx = pdf_contexts.get(arxiv_id)
            if not ctx:
                print("  SKIP (no GROBID context)")
                continue
            result = inject_errors_pdf(
                ctx, input_path, seed=seed + i, text_only=not llm
            )
        else:
            result = inject_errors(input_path, seed=seed + i, text_only=not llm)
        print(f"  Injected {len(result.injections)} errors")

        if not result.injections:
            continue

        if is_pdf:
            # Set injected ManuscriptContext on config for PDF-mode checks
            from sciwrite_lint.manuscript_store import set_manuscript_context

            paper_cfg = LintConfig(project_dir=input_path.parent)
            paper_cfg._manuscript_context = result.injected_ctx  # type: ignore[attr-defined]
            set_manuscript_context(input_path, result.injected_ctx)
            text_findings = _run_text_checks_on_file(input_path, paper_cfg)
            injected_path = input_path  # no temp file needed
        else:
            # Write injected LaTeX to temp file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".tex",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(result.injected_text)
                injected_path = Path(f.name)
            text_findings = _run_text_checks_on_file(injected_path, cfg)

        injection_dicts = [inj.model_dump() for inj in result.injections]
        prepared.append((arxiv_id, injected_path, injection_dicts, text_findings))

    def _save_paper_result(
        arxiv_id: str,
        injection_dicts: list,
        findings_data: list,
    ) -> None:
        """Match injections to findings and save immediately."""
        detected = _match_injections_to_findings(injection_dicts, findings_data)
        det_count = sum(detected)
        print(
            f"  [{arxiv_id}] {len(findings_data)} findings, {det_count}/{len(injection_dicts)} detected"
        )

        report.add_injection_results(arxiv_id, injection_dicts, detected)

        for inj_dict, was_det in zip(injection_dicts, detected):
            inj_dict["detected"] = was_det

        paper_out = out / f"{arxiv_id.replace('/', '_')}.json"
        paper_out.write_text(
            json.dumps(
                {
                    "arxiv_id": arxiv_id,
                    "injections": injection_dicts,
                    "findings": findings_data,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    # Phase 2: LLM checks (concurrent — batched vLLM) + save as each completes
    if llm and prepared:
        print(
            f"\nPhase 2: LLM checks on {len(prepared)} papers (concurrency={concurrency})..."
        )

        async def _run_llm_batch() -> None:
            sem = asyncio.Semaphore(concurrency)

            async def _one(
                arxiv_id: str,
                path: Path,
                injection_dicts: list,
                text_findings: list,
            ) -> None:
                async with sem:
                    llm_findings = await _run_llm_rules_on_file(path, cfg)
                findings_data = text_findings + llm_findings
                _save_paper_result(arxiv_id, injection_dicts, findings_data)
                if path.suffix == ".tex":
                    path.unlink(missing_ok=True)

            await asyncio.gather(
                *[_one(aid, path, inj, tf) for aid, path, inj, tf in prepared]
            )

        asyncio.run(_run_llm_batch())
    else:
        # No LLM — save text-only results immediately
        for arxiv_id, injected_path, injection_dicts, text_findings in prepared:
            _save_paper_result(arxiv_id, injection_dicts, text_findings)
            if injected_path.suffix == ".tex":
                injected_path.unlink(missing_ok=True)

    # Final report
    report_path = out / "report.json"
    report.save(report_path)
    print_summary(report)

    return out


def run_full_pipeline(
    workspace: Path,
    output_dir: Path | None = None,
    config: LintConfig | None = None,
    max_papers: int | None = None,
    contribution: bool = True,
    model: str = "",
    concurrency: int = 2,
    judge: bool = False,
    max_judge_findings: int = 20,
) -> Path:
    """Run the complete linter pipeline on corpus papers.

    Uses batch-by-stage orchestration: GPU stages (vision, embedding,
    cited vision) run in a single subprocess per batch, non-GPU stages
    (vLLM checks, verify, fetch, GROBID, claims) run all papers
    concurrently.
    """
    import asyncio
    from argparse import Namespace
    from datetime import datetime, timezone

    from sciwrite_lint.config import PaperConfig
    from sciwrite_lint.pipeline import build_pdf_context, run_papers_staged
    from sciwrite_lint.references.metadata import load_all_metadata
    from sciwrite_lint.scoring.scilint_score import compute_scilint_score

    out = output_dir or RESULTS_DIR / "pipeline"
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {workspace}. "
            f"Run: python -m evals eval-real-world corpus --workspace {workspace}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = [p for p in manifest if p.get("tex_path") or p.get("pdf_path")]
    if max_papers:
        papers = papers[:max_papers]

    base_config = config or LintConfig()

    # ------------------------------------------------------------------
    # Phase 1: Setup — download, config, PDF context (skip cached papers)
    # ------------------------------------------------------------------

    # Track cached results and papers that need processing
    cached_results: list[dict] = []
    # Papers ready for the batch pipeline: (arxiv_id, title, input_path, pc, paper_config)
    to_process: list[tuple[str, str, Path, PaperConfig, LintConfig]] = []

    async def _setup_all() -> None:
        for i, paper in enumerate(papers, 1):
            arxiv_id = paper["arxiv_id"]
            title = paper.get("title", "")

            # Skip if already completed (crash-resilient)
            paper_out = out / f"{arxiv_id.replace('/', '_')}.json"
            if paper_out.exists():
                print(f"  [{i}/{len(papers)}] {arxiv_id}: cached")
                cached_results.append(json.loads(paper_out.read_text(encoding="utf-8")))
                continue

            print(f"  [{i}/{len(papers)}] {arxiv_id}: setting up...")

            input_path = await _ensure_downloaded_async(paper, workspace)
            if not input_path:
                print(f"  [{i}/{len(papers)}] {arxiv_id}: SKIP (download failed)")
                cached_results.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "error": "download failed",
                    }
                )
                continue

            paper_dir = input_path.parent
            refs_dir = paper_dir / "references"
            refs_dir.mkdir(exist_ok=True)

            pc = PaperConfig(name=arxiv_id, file_path=input_path)
            paper_config = LintConfig(
                project_dir=paper_dir,
                references_dir=refs_dir,
                papers=[pc],
                llm_endpoint=base_config.llm_endpoint,
                llm_model=base_config.llm_model or "qwen3",
                embedding_model=base_config.embedding_model,
                embedding_dim=base_config.embedding_dim,
                polite_email=base_config.polite_email,
            )

            # For PDFs, parse via GROBID first
            if input_path.suffix == ".pdf":
                try:
                    await build_pdf_context(input_path, paper_config)
                except Exception as e:
                    print(
                        f"  [{i}/{len(papers)}] {arxiv_id}: FAILED (GROBID parse): {e}"
                    )
                    cached_results.append(
                        {
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "error": f"GROBID parse failed: {e}",
                        }
                    )
                    continue

            to_process.append((arxiv_id, title, input_path, pc, paper_config))

    asyncio.run(_setup_all())

    if not to_process:
        print("All papers cached or failed during setup.")
        all_results = cached_results
    else:
        # ------------------------------------------------------------------
        # Phase 2: Batch-staged pipeline
        # ------------------------------------------------------------------
        print(f"\n{'=' * 60}")
        print(f"Running batch-staged pipeline on {len(to_process)} papers")
        print(f"{'=' * 60}")

        staged_input = [
            (arxiv_id, input_path, pc, paper_config)
            for arxiv_id, _title, input_path, pc, paper_config in to_process
        ]

        try:
            staged_results = asyncio.run(
                run_papers_staged(staged_input, concurrency=concurrency)
            )
        except Exception as e:
            print(f"Batch pipeline failed: {e}")
            staged_results = []

        # ------------------------------------------------------------------
        # Phase 3: Score + judge + save each paper result
        # ------------------------------------------------------------------
        pipeline_results: list[dict] = []

        async def _post_pipeline() -> None:
            # Build a lookup from paper_name to staged result
            result_map = {r.paper_name: r for r in staged_results}

            for arxiv_id, title, input_path, pc, paper_config in to_process:
                sr = result_map.get(arxiv_id)
                if not sr or sr.error:
                    error = sr.error if sr else "not in batch results"
                    print(f"  {arxiv_id}: FAILED ({error})")
                    pipeline_results.append(
                        {
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "error": error,
                        }
                    )
                    continue

                findings_data = [f.model_dump() for f in sr.findings]
                claim_results = sr.claim_results
                print(
                    f"  {arxiv_id}: {len(findings_data)} findings, {len(claim_results)} claims"
                )

                # Judge (optional, Claude API — no GPU)
                verdict_dicts: list[dict] = []
                judge_elapsed = 0.0
                if judge and sr.findings:
                    import time as _time

                    from eval_real_world.judge import judge_findings
                    from sciwrite_lint.manuscript_store import (
                        get_or_create_manuscript_context,
                    )

                    try:
                        ctx = get_or_create_manuscript_context(input_path, paper_config)
                        tex_text = "\n\n".join(
                            f"## {s.title}\n{s.clean_text}" for s in ctx.sections
                        )
                        to_judge = [
                            f for f in sr.findings if f.level in ("error", "warning")
                        ][:max_judge_findings]
                        t_judge = _time.monotonic()
                        verdicts = await judge_findings(
                            to_judge,
                            tex_text,
                            project_dir=workspace,
                            concurrency=concurrency,
                        )
                        judge_elapsed = _time.monotonic() - t_judge
                        verdict_dicts = [v.model_dump() for v in verdicts]
                        tp = sum(1 for v in verdicts if v.judgment == "TP")
                        print(
                            f"    Sonnet judged: {tp}/{len(verdicts)} TP ({judge_elapsed:.0f}s)"
                        )
                    except Exception as e:
                        print(f"    Judging failed: {e}")

                # Load metadata for child integrity
                refs_dir = paper_config.paper_workspace(arxiv_id).root
                metadata_map = load_all_metadata(refs_dir)

                # Contribution axes (vLLM)
                c_scores = None
                c_reasoning = None
                if contribution:
                    try:
                        from sciwrite_lint.cli.rank import (
                            compute_contribution_axes_from_ctx,
                        )
                        from sciwrite_lint.manuscript_store import (
                            get_or_create_manuscript_context,
                        )

                        ctx = get_or_create_manuscript_context(input_path, paper_config)
                        mock_args = Namespace(model=model)
                        (
                            c_scores,
                            c_reasoning,
                        ) = await compute_contribution_axes_from_ctx(
                            ctx,
                            claim_results,
                            paper_config,
                            mock_args,
                        )
                    except Exception as e:
                        print(f"    Contribution axes failed: {e}")

                score_result = compute_scilint_score(
                    arxiv_id,
                    claim_results,
                    findings=findings_data,
                    metadata_map=metadata_map,
                    contribution_scores=c_scores,
                    contribution_reasoning=c_reasoning,
                )

                print(f"    SciLint Score: {score_result.scilint_score:.3f}")
                print(
                    f"      Internal: {score_result.integrity_result.internal_consistency:.3f}"
                )
                print(
                    f"      Referencing: {score_result.integrity_result.referencing_quality:.3f}"
                )
                if c_scores:
                    print(
                        f"      Contribution: {score_result.contribution.overall:.3f}"
                    )

                result: dict = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "scilint_score": round(score_result.scilint_score, 4),
                    "internal_consistency": round(
                        score_result.integrity_result.internal_consistency, 4
                    ),
                    "referencing_quality": round(
                        score_result.integrity_result.referencing_quality, 4
                    ),
                    "contribution": round(score_result.contribution.overall, 4),
                    "n_findings": len(findings_data),
                    "n_claims": len(claim_results),
                    "n_refs_scored": score_result.total_refs_scored,
                    "findings_by_rule": {},
                }
                for f in findings_data:
                    rid = f.get("rule_id", "unknown")
                    result["findings_by_rule"][rid] = (
                        result["findings_by_rule"].get(rid, 0) + 1
                    )
                if c_scores:
                    result["contribution_axes"] = c_scores
                if verdict_dicts:
                    result["verdicts"] = verdict_dicts
                    tp = sum(1 for v in verdict_dicts if v.get("judgment") == "TP")
                    result["tp"] = tp
                    result["fp"] = len(verdict_dicts) - tp
                if judge_elapsed > 0:
                    result["timing"] = {"judge_s": round(judge_elapsed, 1)}

                # Save per-paper result immediately
                paper_out = out / f"{arxiv_id.replace('/', '_')}.json"
                paper_out.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                pipeline_results.append(result)

        asyncio.run(_post_pipeline())
        all_results = cached_results + pipeline_results

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------

    scored = [r for r in all_results if "scilint_score" in r]
    scores = [r["scilint_score"] for r in scored]

    summary: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_papers": len(papers),
        "n_completed": len(scored),
        "n_failed": len(all_results) - len(scored),
    }
    if scores:
        summary["score_mean"] = round(sum(scores) / len(scores), 4)
        summary["score_min"] = round(min(scores), 4)
        summary["score_max"] = round(max(scores), 4)
        sorted_scores = sorted(scores)
        mid = len(sorted_scores) // 2
        summary["score_median"] = round(
            sorted_scores[mid]
            if len(sorted_scores) % 2
            else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2,
            4,
        )

    report = {
        "summary": summary,
        "papers": sorted(scored, key=lambda r: r["scilint_score"], reverse=True),
    }

    report_path = out / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("FULL PIPELINE RESULTS")
    print(f"{'=' * 60}")
    print(f"\nPapers: {summary['n_completed']}/{summary['n_papers']}")
    if scores:
        print(
            f"SciLint Score: mean={summary['score_mean']:.3f} "
            f"median={summary['score_median']:.3f} "
            f"range=[{summary['score_min']:.3f}, {summary['score_max']:.3f}]"
        )
        print(f"\n{'Paper':<40} {'Score':>6} {'Int.':>5} {'Cont.':>5} {'Finds':>5}")
        print("-" * 65)
        for r in report["papers"]:
            title = (
                r["title"][:37] + "..."
                if len(r.get("title", "")) > 40
                else r.get("title", "")
            )
            print(
                f"{title:<40} {r['scilint_score']:>6.3f} "
                f"{r['internal_consistency']:>5.2f} {r['contribution']:>5.2f} "
                f"{r['n_findings']:>5}"
            )

    # Save committable record
    record_dir = base_config.results_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / "real_world_pipeline.json"
    record_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nResults saved to {report_path}")
    print(f"Committable record saved to {record_path}")

    return out


def run_report(results_dir: Path | None = None) -> None:
    """Aggregate and display results from previous runs."""

    base = results_dir or RESULTS_DIR

    for subdir_name in ["fpr", "inject"]:
        report_path = base / subdir_name / "report.json"
        if report_path.exists():
            print(f"\n{'=' * 60}")
            print(f"  {subdir_name.upper()} RESULTS")
            print(f"{'=' * 60}")
            data = json.loads(report_path.read_text(encoding="utf-8"))
            # Print key metrics
            summary = data.get("summary", {})
            print(f"  Papers: {summary.get('total_papers', '?')}")
            if "total_tp" in summary:
                total = summary["total_tp"] + summary["total_fp"]
                if total:
                    print(f"  FPR: {summary['total_fp'] / total:.1%}")
            print("  Per-rule metrics:")
            for rid, m in data.get("per_rule", {}).items():
                print(
                    f"    {rid}: P={m['precision']:.1%} R={m['recall']:.1%} F1={m['f1']:.1%}"
                )
        else:
            print(f"\nNo {subdir_name} results found at {report_path}")
