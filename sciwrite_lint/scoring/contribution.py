"""Contribution axes for SciLint Score scoring.

Five axes, each returning a score in [0, 1]:

1. Empirical content (Popper) — from claim taxonomy
2. Progressiveness (Lakatos) — from claim taxonomy
3. Unification (Kitcher) — citation graph computation, no LLM
4. Problem-solving (Laudan) — LLM prompt on intro + limitations
5. Test severity (Mayo) — from claim taxonomy

Axes 1, 2, 5 are computed from ClaimClassification (see claims.py).
Axis 3 is pure graph computation on the citation co-occurrence matrix.
Axis 4 requires one LLM call per paper.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from loguru import logger

from sciwrite_lint.claims import (
    ClaimClassification,
    empirical_content_score,
    progressiveness_score,
    severity_score,
)
from sciwrite_lint.config import LintConfig
from sciwrite_lint.llm_utils import llm_query
from sciwrite_lint.schemas import LaudanProblemSolving, vllm_schema_unbounded


# ---------------------------------------------------------------------------
# Axis 3: Unification (Kitcher) — graph computation
# ---------------------------------------------------------------------------


def _build_co_citation_graph(
    claim_results: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Build co-citation adjacency: refs that appear in the same claim context.

    Two references are co-cited if they appear in claim contexts from the
    same section or within a small text window. We approximate this by
    grouping claims by their source section (if available) or by line
    proximity.
    """
    # Group refs by section or line bucket (50-line windows)
    buckets: dict[str, set[str]] = defaultdict(set)
    for c in claim_results:
        key = c.get("key", "")
        if not key:
            continue
        section = c.get("source_section", "")
        line = c.get("line", 0)
        bucket_id = section if section else f"lines_{line // 50}"
        buckets[bucket_id].add(key)

    # Build adjacency from co-occurrence in buckets
    adjacency: dict[str, set[str]] = defaultdict(set)
    for refs in buckets.values():
        refs_list = list(refs)
        for i, r1 in enumerate(refs_list):
            for r2 in refs_list[i + 1 :]:
                adjacency[r1].add(r2)
                adjacency[r2].add(r1)

    return dict(adjacency)


def _count_clusters(adjacency: dict[str, set[str]], all_refs: set[str]) -> int:
    """Count connected components in the co-citation graph via BFS."""
    visited: set[str] = set()
    clusters = 0

    for ref in all_refs:
        if ref in visited:
            continue
        clusters += 1
        queue = [ref]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

    return clusters


def _count_bridges(adjacency: dict[str, set[str]], all_refs: set[str]) -> int:
    """Count refs that bridge multiple clusters (appear in multiple components
    when removed)."""
    bridges = 0
    for ref in all_refs:
        if ref not in adjacency or not adjacency[ref]:
            continue
        # Check if removing this ref increases the number of components
        reduced: dict[str, set[str]] = {}
        for r in all_refs:
            if r == ref:
                continue
            reduced[r] = adjacency.get(r, set()) - {ref}

        remaining = all_refs - {ref}
        original_clusters = _count_clusters(adjacency, all_refs)
        reduced_clusters = _count_clusters(reduced, remaining)
        if reduced_clusters > original_clusters:
            bridges += 1

    return bridges


def compute_unification_score(claim_results: list[dict[str, Any]]) -> float:
    """Kitcher unification: how many distinct citation clusters does the paper bridge?

    Score = 1 - (1 / num_clusters) if clusters > 1, else 0.
    A paper that bridges many distinct citation clusters scores higher.
    Bonus for having bridge nodes (refs connecting different clusters).
    """
    all_refs = {c.get("key", "") for c in claim_results if c.get("key")}
    if len(all_refs) < 2:
        return 0.0

    adjacency = _build_co_citation_graph(claim_results)
    num_clusters = _count_clusters(adjacency, all_refs)

    if num_clusters <= 1:
        return 0.0  # everything is one cluster — no unification

    # Base score: more clusters bridged = higher score
    # Formula: 1 - 1/clusters (asymptotes to 1.0 as clusters increase)
    base = 1.0 - (1.0 / num_clusters)

    # Bonus for bridge nodes (cap at 0.2)
    bridges = _count_bridges(adjacency, all_refs)
    bridge_bonus = min(0.2, bridges * 0.05)

    return min(1.0, base + bridge_bonus)


# ---------------------------------------------------------------------------
# Axis 4: Problem-solving (Laudan) — LLM prompt
# ---------------------------------------------------------------------------

LAUDAN_SYSTEM = """\
You are a critical philosophy-of-science evaluator assessing a paper's \
problem-solving effectiveness (Larry Laudan's framework). Be skeptical.

Given the introduction and limitations/discussion sections:

1. Count PROBLEMS CLAIMED SOLVED — only count problems with concrete evidence, \
not aspirational statements. "We address X" without results is not solved.

2. Count ACKNOWLEDGED LIMITATIONS — explicit admissions of weaknesses.

3. Count UNACKNOWLEDGED LIMITATIONS — be thorough. Always check for these \
common omissions that authors frequently overlook:
- Generalizability: tested on limited domains/datasets/languages?
- Scalability: will the approach work at larger scale?
- Reproducibility: are all details provided to reproduce?
- Fairness/bias: any social impact considerations missing?
- Comparison fairness: are baselines truly comparable?
- Statistical rigor: single seed? No confidence intervals?

If the limitations section is EMPTY or MISSING, count at least 2 \
unacknowledged limitations (missing self-assessment is itself a red flag).

If the paper makes grand claims ("solve AGI", "outperform in all domains") \
without proportional evidence, count each unsupported grand claim as an \
unacknowledged limitation.

Compute: score = num_problems_solved / (num_problems_solved + num_unacknowledged)
If num_problems_solved is 0, score is 0.

Respond with ONLY a valid JSON object:
{
  "num_problems_solved": <integer>,
  "num_acknowledged": <integer>,
  "num_unacknowledged": <integer>,
  "score": <float 0-1>
}"""

LAUDAN_USER_TEMPLATE = """\
INTRODUCTION:
<section>
{intro_text}
</section>

LIMITATIONS / DISCUSSION:
<section>
{limitations_text}
</section>"""

LAUDAN_SCHEMA = vllm_schema_unbounded(LaudanProblemSolving)


async def compute_problem_solving_score(
    intro_text: str,
    limitations_text: str,
    config: LintConfig | None = None,
    model_name: str = "",
) -> tuple[float, str]:
    """Laudan problem-solving: LLM evaluates solved vs unacknowledged problems.

    Returns (score, reasoning). Score 0-1. Returns (0.0, error_msg) on failure.
    """
    if not intro_text.strip():
        return 0.5, "No introduction text available (neutral default)"

    # If no limitations section found, that's itself a limitation
    if not limitations_text.strip():
        limitations_text = "(No limitations section found in the paper.)"

    user = LAUDAN_USER_TEMPLATE.format(
        intro_text=intro_text[:4000],
        limitations_text=limitations_text[:4000],
    )

    result = await llm_query(
        system=LAUDAN_SYSTEM,
        user=user,
        schema=LAUDAN_SCHEMA,
        schema_name="LaudanProblemSolving",
        config=config,
        model_name=model_name,
        thinking="off",
    )

    if not result:
        logger.warning("Laudan problem-solving LLM call failed")
        return 0.0, "LLM call failed"

    score = result.get("score", 0.0)
    score = max(0.0, min(1.0, float(score)))
    n_solved = result.get("num_problems_solved", 0)
    n_ack = result.get("num_acknowledged", 0)
    n_unack = result.get("num_unacknowledged", 0)
    reasoning = f"{n_solved} solved, {n_ack} acknowledged, {n_unack} unacknowledged"

    return score, reasoning


# ---------------------------------------------------------------------------
# Full contribution computation
# ---------------------------------------------------------------------------


async def compute_all_contribution_axes(
    claim_results: list[dict[str, Any]],
    classifications: list[ClaimClassification],
    intro_text: str = "",
    limitations_text: str = "",
    config: LintConfig | None = None,
    model_name: str = "",
) -> tuple[dict[str, float], dict[str, str]]:
    """Compute all five contribution axes.

    Args:
        claim_results: Claim verification results (for unification graph).
        classifications: Claim taxonomy classifications (for Popper/Lakatos/Mayo).
        intro_text: Introduction section text (for Laudan).
        limitations_text: Limitations/discussion section text (for Laudan).
        config: Lint configuration.
        model_name: vLLM model preset.

    Returns:
        Tuple of (axis_scores, axis_reasoning) dicts.
    """
    scores: dict[str, float] = {}
    reasoning: dict[str, str] = {}

    # Axes 1, 2, 5: from claim taxonomy (no LLM calls needed)
    if classifications:
        scores["empirical_content"] = empirical_content_score(classifications)
        reasoning["empirical_content"] = (
            f"{sum(1 for c in classifications if c.testability == 'falsifiable')}"
            f"/{len(classifications)} claims falsifiable"
        )

        scores["progressiveness"] = progressiveness_score(classifications)
        predictions = sum(1 for c in classifications if c.type == "prediction")
        explanations = sum(1 for c in classifications if c.type == "explanation")
        reasoning["progressiveness"] = (
            f"{predictions} predictions, {explanations} explanations"
        )

        scores["test_severity"] = severity_score(classifications)
        severe = sum(1 for c in classifications if c.support == "severe_test")
        reasoning["test_severity"] = (
            f"{severe}/{len(classifications)} claims with severe tests"
        )
    else:
        scores["empirical_content"] = 0.0
        scores["progressiveness"] = 0.5
        scores["test_severity"] = 0.0
        reasoning["empirical_content"] = "No claim classifications available"
        reasoning["progressiveness"] = "No claim classifications available"
        reasoning["test_severity"] = "No claim classifications available"

    # Axis 3: Unification (graph, no LLM)
    scores["unification"] = compute_unification_score(claim_results)
    all_refs = {c.get("key", "") for c in claim_results if c.get("key")}
    reasoning["unification"] = f"{len(all_refs)} references in citation graph"

    # Axis 4: Problem-solving (LLM)
    if intro_text:
        ps_score, ps_reasoning = await compute_problem_solving_score(
            intro_text, limitations_text, config, model_name
        )
        scores["problem_solving"] = ps_score
        reasoning["problem_solving"] = ps_reasoning
    else:
        scores["problem_solving"] = 0.5
        reasoning["problem_solving"] = (
            "No introduction text available (neutral default)"
        )

    return scores, reasoning
