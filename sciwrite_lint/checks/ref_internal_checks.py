"""Run internal consistency checks on T1 cited papers.

After GROBID parsing, each T1 reference has markdown at
``references/{paper}/parsed/{key}.md``. This module builds a
ManuscriptContext from each, runs cross-section-consistency and
structure-promises checks via a single batched vLLM call, and
returns per-reference internal scores for the scoring formula.

Not a registered ``@check`` — called directly by the pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Finding

# Cache format version — bump when prompts or schemas change.
_CACHE_VERSION = "4"


class RefContributionScores(BaseModel):
    """Contribution scores for a single cited paper (5 axes)."""

    empirical_content: float = 0.0
    progressiveness: float = 0.5
    unification: float = 0.0
    problem_solving: float = 0.5
    test_severity: float = 0.0
    overall: float = 0.0
    reasoning: dict[str, str] = Field(default_factory=dict)


class RefInternalResult(BaseModel):
    """Internal consistency + contribution results for a single cited paper."""

    key: str
    internal_score: float  # [0, 1]
    contribution_score: float = 1.0  # [0, 1] — overall contribution
    contribution: RefContributionScores | None = None
    findings: list[dict[str, Any]] = Field(default_factory=list)
    sections_found: int = 0
    checks_run: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Query construction — uses prompts/schemas from check modules directly
# to avoid the mutable _build_queries._state pattern.
# ---------------------------------------------------------------------------

# Section pairs for cross-section-consistency (mirrors the check module)
_SECTION_PAIRS = [
    (
        ["abstract"],
        ["result", "finding", "experiment", "evaluation"],
        "Abstract vs Results",
    ),
    (["abstract"], ["conclusion", "discussion"], "Abstract vs Conclusion"),
    (
        ["introduction", "intro"],
        ["conclusion", "discussion"],
        "Introduction vs Conclusion",
    ),
    (
        ["method", "methodology", "approach"],
        ["result", "finding", "experiment", "evaluation"],
        "Methods vs Results",
    ),
]


def _build_consistency_queries(
    ctx: Any,  # ManuscriptContext
) -> list[tuple[str, str, dict, str]]:
    """Build cross-section-consistency queries from a ManuscriptContext."""
    from sciwrite_lint.checks.cross_section_consistency import (
        _CONSISTENCY_SCHEMA,
        _CONSISTENCY_SYSTEM,
    )

    queries: list[tuple[str, str, dict, str]] = []
    for a_titles, b_titles, _pair_desc in _SECTION_PAIRS:
        if a_titles == ["abstract"]:
            if not ctx.abstract:
                continue
            a_text = ctx.abstract[:3000]
        else:
            a_sections = ctx.get_section_by_title(*a_titles)
            if not a_sections:
                continue
            a_text = "\n\n".join(s.clean_text for s in a_sections)[:3000]

        b_sections = ctx.get_section_by_title(*b_titles)
        if not b_sections:
            continue
        b_text = "\n\n".join(s.clean_text for s in b_sections)[:3000]

        a_label = "Abstract" if a_titles == ["abstract"] else a_titles[0].title()
        b_label = b_sections[0].title or b_titles[0].title()

        from sciwrite_lint.prompt_safety import wrap_untrusted

        user_prompt = (
            f"## PASSAGE A (from: {a_label})\n\n"
            f"{wrap_untrusted(a_text, 'source_section')}\n\n"
            f"## PASSAGE B (from: {b_label})\n\n"
            f"{wrap_untrusted(b_text, 'source_section')}\n"
        )
        queries.append(
            (_CONSISTENCY_SYSTEM, user_prompt, _CONSISTENCY_SCHEMA, "Consistency")
        )
    return queries


def _build_promises_queries(
    ctx: Any,  # ManuscriptContext
) -> list[tuple[str, str, dict, str]]:
    """Build structure-promises queries from a ManuscriptContext."""
    from sciwrite_lint.checks.structure_promises import _CONTRIBUTION_SCHEMA, _SYSTEM

    intro_sections = ctx.get_section_by_title("introduction", "intro")
    if not intro_sections:
        return []
    conclusion_sections = ctx.get_section_by_title(
        "conclusion", "conclusions", "discussion", "summary"
    )
    intro_text = "\n\n".join(s.clean_text for s in intro_sections)
    conclusion_text = (
        "\n\n".join(s.clean_text for s in conclusion_sections)
        if conclusion_sections
        else "(No conclusion section found.)"
    )
    from sciwrite_lint.prompt_safety import wrap_untrusted

    user_prompt = (
        f"## INTRODUCTION\n\n{wrap_untrusted(intro_text[:4000], 'source_section')}\n\n"
        f"## CONCLUSION\n\n{wrap_untrusted(conclusion_text[:4000], 'source_section')}\n"
    )
    return [(_SYSTEM, user_prompt, _CONTRIBUTION_SCHEMA, "ContribCount")]


def _build_full_paper_queries(
    ctx: Any,  # ManuscriptContext
    config: LintConfig,
    figure_descriptions: str = "",
) -> list[tuple[str, str, dict, str]]:
    """Build full-paper consistency queries for a cited paper.

    Uses the same system prompt (full paper body) and per-check questions
    as the manuscript checks, but on GROBID-parsed cited paper text.
    Returns queries for all full-paper checks (mechanical + figure).
    """
    from sciwrite_lint.checks.full_paper_consistency import (
        _CHECK_DEFS,
        _ISSUE_SCHEMA,
        _REFERENCES_HEADINGS,
        _SYSTEM_TEMPLATE,
        _estimate_tokens,
    )

    # Build paper body from the cited paper's ManuscriptContext
    parts: list[str] = []
    if ctx.abstract:
        parts.append(f"## Abstract\n\n{ctx.abstract}")
    for sec in ctx.sections:
        title_lower = sec.title.lower().strip()
        if title_lower in _REFERENCES_HEADINGS:
            continue
        text = sec.clean_text  # cited papers are always markdown (from GROBID)
        if not text.strip():
            continue
        depth_marker = "#" * (sec.depth + 2)
        parts.append(f"{depth_marker} {sec.title}\n\n{text}")

    body = "\n\n".join(parts)
    figure_section = figure_descriptions or "Not available."
    body_tokens = _estimate_tokens(body) + _estimate_tokens(figure_section)

    # Size check against max_model_len
    from sciwrite_lint.checks.full_paper_consistency import _get_max_model_len

    max_model_len = _get_max_model_len(config)
    if body_tokens > max_model_len - 3500:  # overhead + min output
        logger.debug(
            "Ref paper body ~{}K tokens, skipping full-paper checks",
            body_tokens // 1000,
        )
        return []

    system = _SYSTEM_TEMPLATE.format(
        paper_body=body,
        figure_section=figure_section,
    )

    queries: list[tuple[str, str, dict, str]] = []
    has_figures = figure_section != "Not available."
    for _check_id, _desc, question, _thinking, needs_figs in _CHECK_DEFS:
        # Skip figure checks when no figure descriptions available
        if needs_figs and not has_figures:
            continue
        queries.append((system, question, _ISSUE_SCHEMA, "FullPaperIssue"))
    return queries


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------


def _process_consistency_results(
    results: list[dict[str, Any] | None],
    pair_descs: list[str],
    ref_key: str,
    md_name: str,
) -> list[Finding]:
    """Convert cross-section-consistency LLM results to findings."""
    findings: list[Finding] = []
    seen_keys: set[str] = set()
    for pair_desc, result in zip(pair_descs, results):
        if not result:
            continue
        for item in result.get("contradictions", []):
            if not item.get("is_genuine", False):
                continue
            ctype = item.get("type", "inconsistency")
            a_says = item.get("section_a_says", "?")
            b_says = item.get("section_b_says", "?")
            dedup_key = f"{ctype}:{a_says}:{b_says}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            findings.append(
                Finding(
                    level="warning",
                    rule_id="cross-section-consistency",
                    message=(
                        f"[{ref_key}] {pair_desc} — {ctype}: "
                        f'one says "{a_says}", '
                        f'other says "{b_says}". '
                        f"{item.get('explanation', '')}"
                    ),
                    file=md_name,
                )
            )
    return findings


def _process_full_paper_results(
    results: list[dict[str, Any] | None],
    ref_key: str,
    md_name: str,
    has_figures: bool = False,
) -> list[Finding]:
    """Convert full-paper consistency LLM results to findings."""
    from sciwrite_lint.checks.full_paper_consistency import _CHECK_DEFS

    # Only iterate over checks that were actually queried
    queried = [
        (cid, desc, q, th)
        for cid, desc, q, th, needs_figs in _CHECK_DEFS
        if not needs_figs or has_figures
    ]

    findings: list[Finding] = []
    for (_check_id, _desc, _question, _thinking), result in zip(queried, results):
        if not result:
            continue
        for item in result.get("issues", []):
            if not item.get("is_genuine", False):
                continue
            findings.append(
                Finding(
                    level="warning",
                    rule_id=_check_id,
                    message=f"[{ref_key}] {item.get('description', '')}",
                    file=md_name,
                    context=item.get("evidence", ""),
                )
            )
    return findings


def _compute_ref_score(findings: list[dict[str, Any]]) -> float:
    """Score a cited paper based on internal check findings.

    Unlike ``compute_internal_score`` (which only penalizes errors), this
    penalizes both warnings and errors because for a cited paper, internal
    contradictions (warnings) are meaningful unreliability signals.

    Penalty: error = -0.10, warning = -0.05 per finding.
    """
    if not findings:
        return 1.0
    penalty = 0.0
    for f in findings:
        level = f.get("level", "info")
        if level == "error":
            penalty += 0.10
        elif level == "warning":
            penalty += 0.05
    return max(0.0, 1.0 - penalty)


def _process_promises_results(
    results: list[dict[str, Any] | None],
    ref_key: str,
    md_name: str,
) -> list[Finding]:
    """Convert structure-promises LLM results to findings."""
    findings: list[Finding] = []
    result = results[0] if results else None
    if result and result.get("mismatch"):
        findings.append(
            Finding(
                level="warning",
                rule_id="structure-promises",
                message=(
                    f"[{ref_key}] Claims {result.get('claimed_count', '?')} "
                    f"contributions but delivers {result.get('listed_count', '?')}. "
                    f"{result.get('explanation', '')}"
                ),
                file=md_name,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _md_hash(md_path: Path) -> str:
    """SHA-256 of the markdown content for cache invalidation."""
    return hashlib.sha256(md_path.read_bytes()).hexdigest()[:16]


def _load_cache(
    conn: "sqlite3.Connection", ref_key: str, md_path: Path
) -> RefInternalResult | None:
    """Load cached result from workspace.db if valid (hash + version match)."""
    from sciwrite_lint.references.workspace_db import load_ref_internal_cache

    data = load_ref_internal_cache(
        conn,
        ref_key,
        expected_version=_CACHE_VERSION,
        expected_md_hash=_md_hash(md_path),
    )
    if not data:
        return None
    contrib_raw = json.loads(data["contribution_json"])
    contrib = RefContributionScores(**contrib_raw) if contrib_raw else None
    return RefInternalResult(
        key=ref_key,
        internal_score=data["internal_score"],
        contribution_score=data["contribution_score"],
        contribution=contrib,
        findings=json.loads(data["findings_json"]),
        sections_found=data["sections_found"],
        checks_run=json.loads(data["checks_run_json"]),
    )


def _save_cache(
    conn: "sqlite3.Connection", ref_key: str, md_path: Path, result: RefInternalResult
) -> None:
    """Persist result in workspace.db."""
    from sciwrite_lint.references.workspace_db import save_ref_internal_cache

    save_ref_internal_cache(
        conn,
        ref_key,
        md_hash=_md_hash(md_path),
        cache_version=_CACHE_VERSION,
        internal_score=result.internal_score,
        contribution_score=result.contribution_score,
        contribution_json=json.dumps(
            result.contribution.model_dump() if result.contribution else None,
            ensure_ascii=False,
        ),
        findings_json=json.dumps(result.findings, ensure_ascii=False),
        sections_found=result.sections_found,
        checks_run_json=json.dumps(result.checks_run, ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# Contribution scoring on cited papers
# ---------------------------------------------------------------------------


async def _compute_ref_contributions(
    ref_contexts: dict[str, Any],  # key → ManuscriptContext
    config: LintConfig,
) -> dict[str, RefContributionScores]:
    """Compute contribution scores for each cited paper.

    Extracts inline citations from each paper's markdown, classifies
    claims via batched LLM, and runs all 5 contribution axes. Returns
    per-ref contribution scores.
    """
    from sciwrite_lint.claims import classify_claims_batch
    from sciwrite_lint.scoring.chain import extract_citations_from_markdown
    from sciwrite_lint.scoring.contribution import compute_all_contribution_axes

    # Phase 1: extract inline citations from each ref, build claim dicts
    all_claims: list[dict[str, Any]] = []
    # Track (start_idx, count, ref_key) for distributing classifications back
    claim_slices: list[tuple[int, int, str]] = []

    for ref_key, ctx in ref_contexts.items():
        md_text = "\n\n".join(s.clean_text for s in ctx.sections)
        citations = extract_citations_from_markdown(md_text)

        # Build claim dicts from inline citations
        abstract = ctx.abstract or ""
        methods_sections = ctx.get_section_by_title(
            "method", "approach", "experiment", "setup", "implementation"
        )
        methods_text = "\n\n".join(s.clean_text for s in methods_sections)[:1500]
        results_sections = ctx.get_section_by_title(
            "result", "evaluation", "finding", "ablation", "analysis"
        )
        results_text = "\n\n".join(s.clean_text for s in results_sections)[:1500]

        from sciwrite_lint.prompt_safety import wrap_untrusted

        paper_context = f"ABSTRACT: {wrap_untrusted(abstract[:1000], 'source_section')}"
        if methods_text:
            paper_context += (
                f"\n\nMETHODS SUMMARY: {wrap_untrusted(methods_text, 'source_section')}"
            )
        if results_text:
            paper_context += (
                f"\n\nRESULTS SUMMARY: {wrap_untrusted(results_text, 'source_section')}"
            )

        start_idx = len(all_claims)
        for ec in citations:
            all_claims.append(
                {
                    "key": str(ec.index),
                    "context": f"{wrap_untrusted(ec.context, 'citation_context')}"
                    f"\n\n{paper_context}",
                    "line": 0,
                    "_ref_key": ref_key,
                }
            )
        claim_slices.append((start_idx, len(all_claims) - start_idx, ref_key))

    if not all_claims:
        return {}

    # Phase 2: batch classify all claims across all refs (single LLM batch)
    logger.info(
        "Contribution scoring: classifying {} claims across {} refs",
        len(all_claims),
        len(ref_contexts),
    )
    all_classifications = await classify_claims_batch(all_claims, config)

    # Phase 3: compute per-ref contribution axes (concurrent — each ref's
    # only LLM call is Laudan problem-solving: thinking=off, ~38 tokens)
    from sciwrite_lint.scoring.scilint_score import compute_contribution

    async def _score_one(
        start_idx: int, count: int, ref_key: str
    ) -> tuple[str, RefContributionScores]:
        ctx = ref_contexts[ref_key]
        ref_claims = all_claims[start_idx : start_idx + count]
        ref_classifications = all_classifications[start_idx : start_idx + count]

        intro_sections = ctx.get_section_by_title(
            "introduction", "intro", "overview", "background"
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

        if len(intro_text) < 200 and ctx.abstract:
            intro_text = f"{ctx.abstract}\n\n{intro_text}"

        scores, reasoning = await compute_all_contribution_axes(
            ref_claims,
            ref_classifications,
            intro_text,
            limitations_text,
            config,
        )

        contrib = compute_contribution(scores, reasoning)
        return ref_key, RefContributionScores(
            empirical_content=contrib.empirical_content,
            progressiveness=contrib.progressiveness,
            unification=contrib.unification,
            problem_solving=contrib.problem_solving,
            test_severity=contrib.test_severity,
            overall=contrib.overall,
            reasoning=contrib.reasoning,
        )

    pairs = await asyncio.gather(*[_score_one(s, c, k) for s, c, k in claim_slices])
    return dict(pairs)


# ---------------------------------------------------------------------------
# Vision: figure descriptions for cited papers
# ---------------------------------------------------------------------------


def _describe_cited_figures(
    ref_keys: list[str],
    references_dir: Path,
    fresh: bool = False,
) -> dict[str, str]:
    """Extract and describe figures from cited paper PDFs.

    Finds local PDFs for each ref_key, extracts raster images, runs VL
    inference (batched across all papers), caches in workspace.db.

    Returns {ref_key: formatted_figure_descriptions}.
    """
    from sciwrite_lint.vision.cache import format_descriptions_from_db
    from sciwrite_lint.vision.image_extraction import (
        ExtractedImage,
        extract_images_from_pdf,
    )

    # Collect images from all cited PDFs
    all_images: list[ExtractedImage] = []
    ref_image_ranges: dict[str, tuple[int, int]] = {}  # key → (start, end) index

    output_dir = references_dir / "parsed" / "ref_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in ref_keys:
        # Find the PDF for this key (may have suffix like _core, _arxiv)
        candidates = sorted(references_dir.glob(f"{key}*.pdf"))
        if not candidates:
            continue
        pdf_path = candidates[0]

        ref_output = output_dir / key
        start = len(all_images)
        images = extract_images_from_pdf(pdf_path, ref_output)
        all_images.extend(images)
        if images:
            ref_image_ranges[key] = (start, len(all_images))

    if not all_images:
        return {}

    # Run VL inference on all images at once (batched)
    from sciwrite_lint.vision.describe import describe_figures

    describe_figures(
        all_images,
        references_dir=references_dir,
        fresh=fresh,
    )

    # Build per-ref description strings
    result: dict[str, str] = {}
    for key, (start, end) in ref_image_ranges.items():
        ref_images = all_images[start:end]
        desc = format_descriptions_from_db(ref_images, references_dir)
        if desc:
            result[key] = desc

    if result:
        logger.info(
            "Described figures for {}/{} cited papers",
            len(result),
            len(ref_keys),
        )
    return result


def _describe_cited_figures_vl(references_dir_str: str, fresh: bool = False) -> None:
    """Subprocess entry point: run VL inference on cited paper images.

    Extracts images, runs VL model, caches results in workspace.db.
    Called by _stage_cited_vision() in a subprocess for CUDA isolation.
    """
    from pathlib import Path

    references_dir = Path(references_dir_str)
    parsed_dir = references_dir / "parsed"
    keys = [f.stem for f in sorted(parsed_dir.glob("*.md"))]
    if not keys:
        return

    from sciwrite_lint.vision.describe import describe_figures
    from sciwrite_lint.vision.image_extraction import (
        ExtractedImage,
        extract_images_from_pdf,
    )

    all_images: list[ExtractedImage] = []
    output_dir = references_dir / "parsed" / "ref_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        candidates = sorted(references_dir.glob(f"{key}*.pdf"))
        if not candidates:
            continue
        images = extract_images_from_pdf(candidates[0], output_dir / key)
        all_images.extend(images)

    if all_images:
        describe_figures(all_images, references_dir=references_dir, fresh=fresh)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_ref_internal_checks(
    references_dir: Path,
    config: LintConfig,
    *,
    keys: list[str] | None = None,
    fresh: bool = False,
    contribution: bool = False,
    ref_figure_descs: dict[str, str] | None = None,
) -> dict[str, RefInternalResult]:
    """Run consistency checks (and optionally contribution scoring) on cited papers.

    Consistency checks (cross-section-consistency, structure-promises)
    run by default as part of the standard pipeline. Contribution scoring
    (5 axes from philosophy of science) runs automatically on cited papers.

    Args:
        references_dir: Paper workspace root (references/{paper}/).
        config: Lint configuration.
        keys: Optional subset of keys to check. None = all available.
        fresh: Ignore cached results.
        contribution: Also compute 5 contribution axes on cited papers.
        ref_figure_descs: Pre-computed figure descriptions per ref key
            (from ``_describe_cited_figures``). Computed in a separate
            pipeline stage to avoid GPU contention with vLLM.

    Returns:
        Mapping of ref key -> RefInternalResult.
    """
    from sciwrite_lint.llm_utils import llm_query_batch
    from sciwrite_lint.manuscript_store import ManuscriptContext
    from sciwrite_lint.references.workspace_db import get_db

    parsed_dir = references_dir / "parsed"
    if not parsed_dir.exists():
        return {}

    md_files = sorted(parsed_dir.glob("*.md"))
    if not md_files:
        return {}

    # Filter to requested keys
    if keys:
        key_set = set(keys)
        md_files = [f for f in md_files if f.stem in key_set]

    with get_db(references_dir) as conn:
        # Check cached results first
        results: dict[str, RefInternalResult] = {}
        to_check: list[tuple[str, Path]] = []  # (key, md_path)

        for md_path in md_files:
            ref_key = md_path.stem
            if not fresh:
                cached = _load_cache(conn, ref_key, md_path)
                if cached:
                    results[ref_key] = cached
                    continue
            to_check.append((ref_key, md_path))

        if not to_check:
            if results:
                logger.debug("Ref internal checks: {} cached, 0 new", len(results))
            return results

        # Build ManuscriptContext for each ref and collect LLM queries
        # Track: (ref_key, check_name, query_count) for result distribution
        all_queries: list[tuple[str, str, dict, str]] = []
        query_map: list[tuple[str, str, int]] = []  # (key, check, count)
        ref_contexts: dict[str, Any] = {}  # key → ManuscriptContext (for contribution)

        for ref_key, md_path in to_check:
            ctx = ManuscriptContext.from_markdown(md_path, ref_key=ref_key)

            if len(ctx.sections) < 2:
                logger.debug(
                    "Ref {}: {} sections, skipping", ref_key, len(ctx.sections)
                )
                results[ref_key] = RefInternalResult(
                    key=ref_key,
                    internal_score=1.0,  # no data, not penalized
                    sections_found=len(ctx.sections),
                )
                _save_cache(conn, ref_key, md_path, results[ref_key])
                continue

            total_chars = sum(len(s.clean_text) for s in ctx.sections)
            if ctx.abstract:
                total_chars += len(ctx.abstract)
            if total_chars > config.max_document_chars:
                logger.debug(
                    "Ref {}: ~{} pages ({} chars), skipping LLM checks",
                    ref_key,
                    total_chars // 3500,
                    total_chars,
                )
                results[ref_key] = RefInternalResult(
                    key=ref_key,
                    internal_score=1.0,  # not penalized
                    sections_found=len(ctx.sections),
                )
                _save_cache(conn, ref_key, md_path, results[ref_key])
                continue

            ref_contexts[ref_key] = ctx

            # Cross-section-consistency queries
            csc_queries = _build_consistency_queries(ctx)
            if csc_queries:
                query_map.append(
                    (ref_key, "cross-section-consistency", len(csc_queries))
                )
                all_queries.extend(csc_queries)

            # Structure-promises queries
            sp_queries = _build_promises_queries(ctx)
            if sp_queries:
                query_map.append((ref_key, "structure-promises", len(sp_queries)))
                all_queries.extend(sp_queries)

            # Full-paper consistency queries (separate batch — needs thinking=medium)
            fig_desc = ref_figure_descs.get(ref_key, "") if ref_figure_descs else ""
            fp_queries = _build_full_paper_queries(
                ctx, config, figure_descriptions=fig_desc
            )
            if fp_queries:
                query_map.append((ref_key, "full-paper", len(fp_queries)))
                all_queries.extend(fp_queries)

            # Store section count for result
            query_map.append((ref_key, "_sections", len(ctx.sections)))

        if not all_queries:
            logger.debug("Ref internal checks: no LLM queries to run")
            return results

        # Split into two batches by thinking mode:
        # - pairwise checks (cross-section, promises): thinking=low
        # - full-paper checks: thinking=medium
        pairwise_queries: list[tuple[str, str, dict, str]] = []
        fullpaper_queries: list[tuple[str, str, dict, str]] = []
        # Track which query_map entries are pairwise vs full-paper
        pairwise_map: list[tuple[str, str, int]] = []
        fullpaper_map: list[tuple[str, str, int]] = []

        idx = 0
        for ref_key, check_name, count in query_map:
            if check_name == "_sections":
                continue
            batch = all_queries[idx : idx + count]
            idx += count
            if check_name == "full-paper":
                fullpaper_queries.extend(batch)
                fullpaper_map.append((ref_key, check_name, count))
            else:
                pairwise_queries.extend(batch)
                pairwise_map.append((ref_key, check_name, count))

        logger.info(
            "Ref internal checks: {} pairwise + {} full-paper queries across {} refs",
            len(pairwise_queries),
            len(fullpaper_queries),
            len(to_check),
        )

        # Run both batches concurrently (different thinking modes)
        async def _empty() -> list[dict | None]:
            return []

        pairwise_coro = (
            llm_query_batch(pairwise_queries, config=config, thinking="low")
            if pairwise_queries
            else _empty()
        )
        fullpaper_coro = (
            llm_query_batch(fullpaper_queries, config=config, thinking="medium")
            if fullpaper_queries
            else _empty()
        )

        pairwise_raw, fullpaper_raw = await asyncio.gather(
            pairwise_coro, fullpaper_coro
        )

        # Distribute results back to each (ref_key, check) pair
        per_ref_findings: dict[str, list[Finding]] = {}
        per_ref_checks: dict[str, list[str]] = {}
        per_ref_sections: dict[str, int] = {}

        # Extract section counts from query_map
        for ref_key, check_name, count in query_map:
            if check_name == "_sections":
                per_ref_sections[ref_key] = count

        # Process pairwise results
        idx = 0
        for ref_key, check_name, count in pairwise_map:
            raw_batch = pairwise_raw[idx : idx + count]
            idx += count
            per_ref_findings.setdefault(ref_key, [])
            per_ref_checks.setdefault(ref_key, [])

            if check_name == "cross-section-consistency":
                md_path = parsed_dir / f"{ref_key}.md"
                ctx = ManuscriptContext.from_markdown(md_path, ref_key=ref_key)
                pair_descs = []
                for a_titles, b_titles, pair_desc in _SECTION_PAIRS:
                    if a_titles == ["abstract"]:
                        if not ctx.abstract:
                            continue
                    else:
                        a_sections = ctx.get_section_by_title(*a_titles)
                        if not a_sections:
                            continue
                    b_sections = ctx.get_section_by_title(*b_titles)
                    if not b_sections:
                        continue
                    pair_descs.append(pair_desc)

                findings = _process_consistency_results(
                    raw_batch, pair_descs, ref_key, f"{ref_key}.md"
                )
                per_ref_findings[ref_key].extend(findings)
                per_ref_checks[ref_key].append("cross-section-consistency")

            elif check_name == "structure-promises":
                findings = _process_promises_results(
                    raw_batch, ref_key, f"{ref_key}.md"
                )
                per_ref_findings[ref_key].extend(findings)
                per_ref_checks[ref_key].append("structure-promises")

        # Process full-paper results
        idx = 0
        for ref_key, check_name, count in fullpaper_map:
            fp_batch = fullpaper_raw[idx : idx + count]
            idx += count
            per_ref_findings.setdefault(ref_key, [])
            per_ref_checks.setdefault(ref_key, [])
            ref_has_figs = bool(ref_figure_descs and ref_figure_descs.get(ref_key))
            findings = _process_full_paper_results(
                fp_batch, ref_key, f"{ref_key}.md", has_figures=ref_has_figs
            )
            per_ref_findings[ref_key].extend(findings)
            per_ref_checks[ref_key].append("full-paper-consistency")

        # Contribution scoring on cited papers (axes 1-5)
        ref_contributions: dict[str, RefContributionScores] = {}
        if contribution and ref_contexts:
            ref_contributions = await _compute_ref_contributions(ref_contexts, config)

        # Compute internal scores and cache
        for ref_key, md_path in to_check:
            if ref_key in results:
                continue  # already handled (< 2 sections)
            findings = per_ref_findings.get(ref_key, [])
            findings_dicts = [f.model_dump() for f in findings]
            score = _compute_ref_score(findings_dicts)
            contrib = ref_contributions.get(ref_key)
            checks = per_ref_checks.get(ref_key, [])
            if contrib:
                checks.append("contribution")

            result = RefInternalResult(
                key=ref_key,
                internal_score=score,
                contribution_score=contrib.overall if contrib else 1.0,
                contribution=contrib,
                findings=findings_dicts,
                sections_found=per_ref_sections.get(ref_key, 0),
                checks_run=checks,
            )
            results[ref_key] = result
            _save_cache(conn, ref_key, md_path, result)

    checked = len(to_check) - sum(
        1 for k, _ in to_check if results.get(k) and not results[k].checks_run
    )
    logger.info(
        "Ref internal checks: {} checked, {} cached, {} total",
        checked,
        len(results) - len(to_check),
        len(results),
    )
    return results
