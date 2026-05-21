"""SciLint Score calibration eval: score ground-truth papers, check ordinal constraints.

20 papers with known quality (Nobel landmarks, retracted fraud, incremental,
null results, etc.) define the calibration set. Ordinal constraints in
MANIFEST.md specify expected ranking relationships. The eval scores all
papers and reports which constraints pass/fail — the feedback loop for
iterating on scoring weights, prompts, and taxonomy.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.checks._diagnostics import split_findings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Calibration spec (paper list, download URLs, ordinal constraints).
CALIBRATION_MANIFEST = (
    Path(__file__).resolve().parent / "calibration_manifest.md"
).resolve()

# Runtime data directory: PDFs download here on first run, workspaces and
# results are generated here.  Entirely gitignored — no tracked files.
CALIBRATION_DIR = (Path(__file__).resolve().parent / "calibration_data").resolve()

NAME_TO_FILE: dict[str, str] = {
    "Graphene": "novoselov2004_graphene.pdf",
    "Transformer": "vaswani2017_attention.pdf",
    "LIGO": "ligo2016_gravitational_waves.pdf",
    "ResNet": "he2015_resnet.pdf",
    "CRISPR": "jinek2013_crispr_human.pdf",
    "AlphaFold": "jumper2021_alphafold.pdf",
    "Shoukat": "shoukat2024_hallucinated_refs.pdf",
    "Baughman": "baughman2016_fabricated_retracted.pdf",
    "LaCour": "lacour2014_fabricated_retracted.pdf",
    "Macchiarini": "macchiarini2014_fabricated_retracted.pdf",
    "LK-99": "lk99_2023.pdf",
    "Reinhart-Rogoff": "reinhart2010_debt.pdf",
    "Camerer": "camerer2018_replication.pdf",
    "Ioannidis": "ioannidis2005_false_findings.pdf",
    "Ritchie": "ritchie2012_failing_future.pdf",
    "RECOVERY": "recovery2020_dexamethasone.pdf",
    "Wu-survey": "wu2021_gnn_survey.pdf",
    "BERT-finetune": "sun2019_bert_finetune.pdf",
    "Wilczek": "wilczek2012_time_crystals.pdf",
    "Ceballos": "ceballos2015_sixth_extinction.pdf",
}

# Axis name normalization: MANIFEST syntax → dict key in contribution scores
AXIS_MAP: dict[str, str] = {
    "test-severity": "test_severity",
    "test_severity": "test_severity",
    "progressiveness": "progressiveness",
    "unification": "unification",
    "empirical": "empirical_content",
    "empirical_content": "empirical_content",
    "problem-solving": "problem_solving",
    "problem_solving": "problem_solving",
    "contribution": "overall",
    "integrity": "integrity",
}

# Papers that are "experimental" (have empirical content) — for "any experimental paper"
EXPERIMENTAL_PAPERS: list[str] = [
    "Graphene",
    "Transformer",
    "LIGO",
    "ResNet",
    "CRISPR",
    "AlphaFold",
    "RECOVERY",
    "Camerer",
    "Ritchie",
    "Ceballos",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class OrdinalConstraint(BaseModel):
    """One ordinal constraint parsed from MANIFEST.md."""

    left: str
    op: str  # ">" | "<" | "≈"
    right: str
    left_axis: str | None = None
    right_axis: str | None = None
    raw: str = ""


class ConstraintResult(BaseModel):
    """Result of evaluating one constraint."""

    constraint: OrdinalConstraint
    passed: bool
    left_value: float
    right_value: float


class CalibrationResult(BaseModel):
    """Full calibration run result."""

    scores: dict[str, dict[str, Any]]
    constraints: list[ConstraintResult]
    total_passed: int
    total_constraints: int
    skipped_papers: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        failures = [
            {
                "raw": cr.constraint.raw,
                "left": cr.constraint.left,
                "op": cr.constraint.op,
                "right": cr.constraint.right,
                "left_value": round(cr.left_value, 4),
                "right_value": round(cr.right_value, 4),
            }
            for cr in self.constraints
            if not cr.passed
        ]
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_papers": len(self.scores),
            "constraints_passed": self.total_passed,
            "constraints_total": self.total_constraints,
            "pass_rate": (
                round(self.total_passed / self.total_constraints, 4)
                if self.total_constraints
                else 0.0
            ),
            "papers": self.scores,
            "failures": failures,
            "skipped_papers": self.skipped_papers,
        }


# ---------------------------------------------------------------------------
# Download URLs (parsed from MANIFEST.md curl commands)
# ---------------------------------------------------------------------------

_CURL_RE = re.compile(r'curl\s+-sL\s+-o\s+(\S+)\s+"([^"]+)"')


def parse_download_urls(manifest_path: Path) -> dict[str, str]:
    """Parse filename → URL mapping from curl commands in MANIFEST.md."""
    urls: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        m = _CURL_RE.search(line)
        if m:
            urls[m.group(1)] = m.group(2)
    return urls


def download_missing_pdfs(
    calibration_dir: Path,
    manifest_path: Path,
    filenames: list[str],
) -> list[str]:
    """Download missing PDFs from URLs in MANIFEST.md. Returns list of failures."""
    import subprocess

    urls = parse_download_urls(manifest_path)
    failures: list[str] = []

    for filename in filenames:
        pdf_path = calibration_dir / filename
        if pdf_path.exists():
            continue

        url = urls.get(filename)
        if not url:
            logger.warning(f"No download URL for {filename}")
            failures.append(filename)
            continue

        logger.info(f"Downloading {filename}...")
        result = subprocess.run(
            ["curl", "-sL", "-o", str(pdf_path), url],
            timeout=120,
            capture_output=True,
        )
        if result.returncode != 0 or not pdf_path.exists():
            logger.error(f"Failed to download {filename}")
            failures.append(filename)
            continue

        # Validate it's actually a PDF
        with open(pdf_path, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            logger.error(f"{filename}: not a PDF (got HTML redirect?)")
            pdf_path.unlink()
            failures.append(filename)

    return failures


# ---------------------------------------------------------------------------
# Constraint parsing
# ---------------------------------------------------------------------------

# Matches: Name [axis] OP Name [axis]
# Examples: "LIGO > LK-99", "Wu-survey unification > Transformer unification"
_CONSTRAINT_RE = re.compile(
    r"^([\w-]+)"  # left name
    r"(?:\s+([\w-]+))?"  # optional left axis
    r"\s*([><\u2248])\s*"  # operator (>, <, ≈)
    r"([\w-]+)"  # right name
    r"(?:\s+([\w-]+))?"  # optional right axis
    r"$"
)


def parse_constraints(manifest_path: Path) -> list[OrdinalConstraint]:
    """Parse ordinal constraints from the ``` block in MANIFEST.md."""
    text = manifest_path.read_text(encoding="utf-8")

    # Extract content between first pair of ``` fences after "ordinal constraints"
    in_block = False
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip() == "```" and not in_block:
            in_block = True
            continue
        if line.strip() == "```" and in_block:
            break
        if in_block:
            lines.append(line)

    constraints: list[OrdinalConstraint] = []
    for raw_line in lines:
        raw = raw_line.strip()
        # Strip comments
        if "#" in raw:
            raw = raw[: raw.index("#")].strip()
        # Strip parenthetical explanations
        raw = re.sub(r"\(.*?\)", "", raw).strip()
        if not raw:
            continue

        # Special case: "any experimental paper"
        if "any experimental paper" in raw_line.lower():
            # Wu-survey test-severity < any experimental paper
            # → expand to individual constraints
            parts = raw_line.strip().split("<")[0].strip()
            parts = re.sub(r"\(.*?\)", "", parts).strip()
            tokens = parts.split()
            if len(tokens) >= 2:
                left_name = tokens[0]
                left_axis = AXIS_MAP.get(tokens[1], tokens[1])
                for exp_paper in EXPERIMENTAL_PAPERS:
                    constraints.append(
                        OrdinalConstraint(
                            left=left_name,
                            op="<",
                            right=exp_paper,
                            left_axis=left_axis,
                            right_axis=left_axis,
                            raw=raw_line.strip(),
                        )
                    )
            continue

        m = _CONSTRAINT_RE.match(raw)
        if not m:
            logger.warning(f"Unparseable constraint: {raw_line.strip()}")
            continue

        left, left_axis, op, right, right_axis = m.groups()

        # Normalize axis names
        if left_axis:
            left_axis = AXIS_MAP.get(left_axis, left_axis)
        if right_axis:
            right_axis = AXIS_MAP.get(right_axis, right_axis)

        constraints.append(
            OrdinalConstraint(
                left=left,
                op=op,
                right=right,
                left_axis=left_axis,
                right_axis=right_axis,
                raw=raw_line.strip(),
            )
        )

    return constraints


# ---------------------------------------------------------------------------
# Constraint evaluation
# ---------------------------------------------------------------------------


def _get_value(
    scores: dict[str, dict[str, Any]], name: str, axis: str | None
) -> float | None:
    """Extract a score value for a paper, optionally for a specific axis."""
    paper = scores.get(name)
    if paper is None:
        return None
    if axis is None:
        return float(paper.get("scilint_score", 0.0))
    if axis == "integrity":
        int_data = paper.get("integrity", {})
        return float(
            int_data.get("internal_consistency", 0.0)
            * int_data.get("referencing_quality", 0.0)
        )
    return float(paper.get("contribution", {}).get(axis, 0.0))


def _compute_quartiles(
    scores: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Assign each paper to a quartile (1=top, 4=bottom) by scilint_score."""
    sorted_names = sorted(
        scores.keys(), key=lambda n: scores[n].get("scilint_score", 0.0), reverse=True
    )
    n = len(sorted_names)
    quartiles: dict[str, int] = {}
    for i, name in enumerate(sorted_names):
        quartiles[name] = (i * 4) // n + 1
    return quartiles


def evaluate_constraints(
    constraints: list[OrdinalConstraint],
    scores: dict[str, dict[str, Any]],
) -> list[ConstraintResult]:
    """Evaluate all ordinal constraints against computed scores."""
    quartiles = _compute_quartiles(scores)
    results: list[ConstraintResult] = []

    for c in constraints:
        left_val = _get_value(scores, c.left, c.left_axis)
        right_val = _get_value(scores, c.right, c.right_axis)

        if left_val is None or right_val is None:
            # Paper not scored — skip constraint
            continue

        if c.op == ">":
            passed = left_val > right_val
        elif c.op == "<":
            passed = left_val < right_val
        elif c.op == "\u2248":  # ≈
            passed = quartiles.get(c.left, 0) == quartiles.get(c.right, 0)
        else:
            passed = False

        results.append(
            ConstraintResult(
                constraint=c,
                passed=passed,
                left_value=left_val,
                right_value=right_val,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------


def run_calibration(
    calibration_dir: Path,
    config: Any,  # LintConfig
    *,
    papers: list[str] | None = None,
    fresh: bool = False,
    model: str = "",
    output_dir: Path | None = None,
    concurrency: int = 4,
) -> CalibrationResult:
    """Score calibration papers and evaluate ordinal constraints.

    Uses a single asyncio.run() for all papers to avoid event loop
    teardown issues with httpx async clients between papers.

    Args:
        calibration_dir: Directory containing PDFs and MANIFEST.md.
        config: LintConfig for GROBID/vLLM settings.
        papers: Optional subset of paper short names to score.
        fresh: Re-run the full pipeline from scratch (ignore all caches).
        model: vLLM model preset.
        output_dir: Where to save scilint_*.json files (default: calibration_dir).
    """
    out_dir = output_dir or (calibration_dir / "results")
    constraints = parse_constraints(CALIBRATION_MANIFEST)
    logger.info(f"Parsed {len(constraints)} ordinal constraints")

    # Determine which papers to score
    names_to_score = papers if papers else list(NAME_TO_FILE.keys())

    to_score: list[tuple[str, Path]] = []
    skipped: list[str] = []

    for name in names_to_score:
        filename = NAME_TO_FILE.get(name)
        if not filename:
            logger.warning(f"Unknown paper name: {name}")
            skipped.append(name)
            continue

        pdf_path = calibration_dir / filename
        if not pdf_path.exists():
            # Auto-download from MANIFEST.md URLs
            fails = download_missing_pdfs(
                calibration_dir, CALIBRATION_MANIFEST, [filename]
            )
            if fails or not pdf_path.exists():
                skipped.append(name)
                continue

        to_score.append((name, pdf_path))

    # Score all papers
    all_scores: dict[str, dict[str, Any]] = {}
    if to_score:
        all_scores = asyncio.run(
            _score_batch_async(
                to_score,
                config,
                model=model,
                output_dir=out_dir,
                concurrency=concurrency,
                fresh=fresh,
            )
        )

    # Evaluate constraints
    constraint_results = evaluate_constraints(constraints, all_scores)
    total_passed = sum(1 for cr in constraint_results if cr.passed)

    return CalibrationResult(
        scores=all_scores,
        constraints=constraint_results,
        total_passed=total_passed,
        total_constraints=len(constraint_results),
        skipped_papers=skipped,
    )


async def _score_batch_async(
    papers: list[tuple[str, Path]],
    config: Any,
    *,
    model: str = "",
    output_dir: Path | None = None,
    concurrency: int = 4,
    fresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """Score multiple papers with batch-staged pipeline + concurrent vLLM.

    Phase 1 (sequential): GROBID parse (build_pdf_context) for all PDFs.
    Phase 2 (batch-staged): Full pipeline via ``run_papers_staged()`` —
        GPU stages batched, non-GPU stages concurrent across papers.
    Phase 3 (concurrent): Contribution axes scoring via vLLM.
    Phase 4 (sync): Compute final scores + save.

    """
    import argparse as _argparse

    from sciwrite_lint.cli.rank import (
        compute_contribution_axes_from_ctx,
        extract_claims_from_context,
    )
    from sciwrite_lint.config import LintConfig, PaperConfig
    from sciwrite_lint.manuscript_store import (
        get_or_create_manuscript_context,
    )
    from sciwrite_lint.pipeline import build_pdf_context, run_papers_staged
    from sciwrite_lint.references.metadata import load_all_metadata
    from sciwrite_lint.references.workspace_db import get_db, update_pipeline_stage
    from sciwrite_lint.scoring.scilint_score import compute_scilint_score

    # Phase 1: GROBID parse + per-paper config setup
    staged_input: list[tuple[str, Path, PaperConfig, Any]] = []
    paper_configs: dict[str, Any] = {}  # name → LintConfig

    for name, pdf_path in papers:
        logger.info(f"[{name}] Phase 1: parsing {pdf_path.name}...")

        # Per-paper config — workspaces go under {calibration_dir}/references/
        # (matches pipeline eval pattern; keeps the eval's .gitignore effective)
        cal_dir = pdf_path.parent
        pc = PaperConfig(name=name, file_path=pdf_path)
        paper_config = LintConfig(
            project_dir=cal_dir,
            references_dir=cal_dir / "references",
            papers=[pc],
            llm_endpoint=config.llm_endpoint,
            llm_model=config.llm_model or "qwen3",
            embedding_model=config.embedding_model,
            embedding_dim=config.embedding_dim,
            polite_email=config.polite_email,
        )

        if pdf_path.suffix.lower() == ".pdf":
            await build_pdf_context(pdf_path, paper_config)

        staged_input.append((name, pdf_path, pc, paper_config))
        paper_configs[name] = paper_config

    # Phase 2: Full pipeline via batch-staged orchestration
    logger.info("Phase 2: batch-staged pipeline ({} papers)", len(staged_input))
    staged_results = await run_papers_staged(
        staged_input, concurrency=concurrency, fresh=fresh
    )

    # Build lookup from name to result
    result_map = {r.paper_name: r for r in staged_results}

    # Phase 3: Contribution axes (concurrent vLLM)
    results: dict[str, dict[str, Any]] = {}
    sem = asyncio.Semaphore(concurrency)

    async def _score_and_save(name: str, pdf_path: Path) -> None:
        sr = result_map.get(name)
        if not sr or sr.error:
            logger.error(f"[{name}] Pipeline failed: {sr.error if sr else 'missing'}")
            return

        paper_config = paper_configs[name]
        # Strip system issues so they don't pollute the calibration
        # score. They are still emitted via the per-paper check_*.json
        # report — calibration only cares about manuscript quality.
        manuscript_findings, _system_issues = split_findings(sr.findings)
        findings_data = [f.model_dump() for f in manuscript_findings]
        claim_results = sr.claim_results

        # Contribution axes
        refs_dir = paper_config.paper_workspace(name).root
        async with sem:
            logger.info(f"[{name}] Phase 3: contribution axes...")
            try:
                with get_db(refs_dir) as conn:
                    update_pipeline_stage(conn, "contributions", "running")
            except sqlite3.Error as e:
                logger.debug(
                    "pipeline_stage update (running) failed for {}: {}",
                    name,
                    e,
                )
            ctx = get_or_create_manuscript_context(pdf_path, paper_config)
            claim_dicts = extract_claims_from_context(ctx, pdf_path)
            ns = _argparse.Namespace(model=model)
            try:
                c_scores, c_reasoning = await compute_contribution_axes_from_ctx(
                    ctx, claim_dicts, paper_config, ns
                )
                with get_db(refs_dir) as conn:
                    update_pipeline_stage(conn, "contributions", "done")
            except Exception as e:
                with get_db(refs_dir) as conn:
                    update_pipeline_stage(conn, "contributions", "failed", str(e)[:200])
                raise

        # Load metadata for child integrity
        metadata_map = load_all_metadata(refs_dir)

        # Compute score with real pipeline data
        score_result = compute_scilint_score(
            name,
            claim_results,
            findings=findings_data,
            metadata_map=metadata_map,
            contribution_scores=c_scores,
            contribution_reasoning=c_reasoning,
        )

        # Save
        result_dict = score_result.to_dict()
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"scilint_{pdf_path.stem}.json"
            out_path.write_text(
                json.dumps(result_dict, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        results[name] = result_dict
        logger.info(f"[{name}] Done: score={score_result.scilint_score:.3f}")

    tasks = [_score_and_save(name, pdf_path) for name, pdf_path in papers]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Log any failures
    for name, _pdf_path in papers:
        if name not in results:
            logger.error(f"[{name}] Scoring failed — check logs above")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_calibration_report(result: CalibrationResult) -> None:
    """Print formatted terminal report."""
    n = len(result.scores)
    print(f"\n{'=' * 72}")
    print(f"  SCILINT SCORE CALIBRATION ({n} papers)")
    print(f"{'=' * 72}")

    # Sort papers by descending scilint_score
    sorted_papers = sorted(
        result.scores.items(),
        key=lambda kv: kv[1].get("scilint_score", 0.0),
        reverse=True,
    )

    print(
        f"\n  {'#':>3}  {'Paper':<30s}  {'Score':>5}  "
        f"{'Int.':>5}  {'Emp.':>5}  {'Prog.':>5}  "
        f"{'Unif.':>5}  {'Prob.':>5}  {'Sev.':>5}"
    )
    print(f"  {'─' * 68}")

    for i, (name, data) in enumerate(sorted_papers, 1):
        score = data.get("scilint_score", 0.0)
        int_data = data.get("integrity", {})
        integrity = int_data.get("internal_consistency", 0.0) * int_data.get(
            "referencing_quality", 0.0
        )
        c = data.get("contribution", {})
        print(
            f"  {i:3d}  {name:<30s}  {score:5.3f}  "
            f"{integrity:5.3f}  {c.get('empirical_content', 0):5.3f}  "
            f"{c.get('progressiveness', 0):5.3f}  "
            f"{c.get('unification', 0):5.3f}  "
            f"{c.get('problem_solving', 0):5.3f}  "
            f"{c.get('test_severity', 0):5.3f}"
        )

    # Constraint summary
    print(f"\n  Constraints: {result.total_passed}/{result.total_constraints} passed")

    failures = [cr for cr in result.constraints if not cr.passed]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for cr in failures:
            c = cr.constraint
            axis_info = ""
            if c.left_axis:
                axis_info = f" [{c.left_axis}]"
            print(
                f"    x {c.left}{axis_info} {c.op} {c.right}{axis_info}: "
                f"{cr.left_value:.3f} vs {cr.right_value:.3f}"
            )
    else:
        print("\n  All constraints passed!")

    if result.skipped_papers:
        print(
            f"\n  Skipped ({len(result.skipped_papers)}): "
            f"{', '.join(result.skipped_papers)}"
        )

    print()


def save_calibration_json(result: CalibrationResult, output_path: Path) -> None:
    """Save structured JSON for tracking progress across iterations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"Calibration results saved to {output_path}")
