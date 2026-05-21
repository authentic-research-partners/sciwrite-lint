"""SciLint Score: integrity × contribution scoring for scientific manuscripts.

Two independent components multiplied into a final score:

    S(p) = integrity(p) × contribution(p)

**Integrity** (recursive, depth-N):
    integrity(p) = α × I(p) + (1-α) × mean(w_i × V(p, r_i) × integrity(r_i))

Where:
- I(p): internal consistency score (fraction of non-error findings)
- V(p, r_i): existence × accuracy × claim_support for reference r_i
- w_i: citation purpose weight
- integrity(r_i): child integrity of cited paper
- α: balance factor (default 0.3 — evidence chain > internal consistency)

**Child integrity modes** (for computing integrity(r_i)):
- Lightweight (default): uses metadata signals (tier, retraction, mismatches)
  — fast, no LLM/GROBID needed, runs after ``verify`` stage.
- Ref-internal (automatic in pipeline): LLM consistency checks + contribution
  scoring on cited papers.

**Contribution** (five axes, each 0–1):
- Empirical content (Popper): fraction of falsifiable claims
- Progressiveness (Lakatos): novel predictions vs accommodations
- Unification (Kitcher): distinct citation clusters bridged
- Problem-solving (Laudan): problems solved vs unacknowledged limitations
- Test severity (Mayo): ablations, baselines, alternatives addressed

Citation purpose weights are defined in ``sciwrite_lint.checks.cite_purpose``
(single source of truth) and imported here.

Since the citation graph is a DAG, a single forward pass suffices —
no iterative PageRank convergence is needed.
"""

from __future__ import annotations

import json
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.checks.cite_purpose import PURPOSE_WEIGHTS
from sciwrite_lint.models import CitationMetadata
from sciwrite_lint.references.metadata import compute_tier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Verdict → verification score [0, 1]
VERDICT_SCORES: dict[str, float] = {
    "SUPPORTS": 1.0,
    "PARTIALLY_SUPPORTS": 0.5,
    "CANNOT_DETERMINE": 0.25,
    "NOT_SUPPORTED": 0.0,
}

# Default α for integrity: internal vs evidence chain balance.
# Lower α = evidence chain matters more than internal consistency.

# Contribution axis weights (equal by default).
CONTRIBUTION_WEIGHTS: dict[str, float] = {
    "empirical_content": 0.2,
    "progressiveness": 0.2,
    "unification": 0.2,
    "problem_solving": 0.2,
    "test_severity": 0.2,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RefScore(BaseModel):
    """Verification score for a single reference."""

    key: str
    purpose: str
    weight: float
    verdict: str
    verification_score: float
    weighted_score: float
    claim_count: int
    supports_count: int
    partial_count: int
    not_supported_count: int


class ContributionScores(BaseModel):
    """Five-axis contribution assessment."""

    empirical_content: float = 0.0  # Popper: falsifiability
    progressiveness: float = 0.0  # Lakatos: novel predictions
    unification: float = 0.0  # Kitcher: citation cluster bridging
    problem_solving: float = 0.0  # Laudan: solved vs unacknowledged
    test_severity: float = 0.0  # Mayo: ablations, baselines, alternatives
    overall: float = 0.0  # weighted mean of all axes

    # Per-axis reasoning from LLM (optional, for explainability)
    reasoning: dict[str, str] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "empirical_content": round(self.empirical_content, 4),
            "progressiveness": round(self.progressiveness, 4),
            "unification": round(self.unification, 4),
            "problem_solving": round(self.problem_solving, 4),
            "test_severity": round(self.test_severity, 4),
            "overall": round(self.overall, 4),
        }
        if self.reasoning:
            d["reasoning"] = self.reasoning
        return d


class IntegrityResult(BaseModel):
    """Integrity scoring result with child integrity detail."""

    internal_consistency: float  # manuscript's own consistency
    referencing_quality: float  # weighted mean of weight × verdict × reliability
    reference_reliability: dict[str, float] = Field(default_factory=dict)
    integrity_source: str = "default"  # "ref_internal", "metadata", "default"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "internal_consistency": round(self.internal_consistency, 4),
            "referencing_quality": round(self.referencing_quality, 4),
            "integrity_source": self.integrity_source,
        }
        if self.reference_reliability:
            d["reference_reliability"] = {
                k: round(v, 4) for k, v in self.reference_reliability.items()
            }
        return d


class SciLintScoreResult(BaseModel):
    """Complete SciLint Score result for a paper."""

    paper: str
    scilint_score: float
    integrity_result: IntegrityResult
    contribution: ContributionScores
    total_claims: int
    total_refs_scored: int
    ref_scores: list[RefScore] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "paper": self.paper,
            "scilint_score": round(self.scilint_score, 4),
            "integrity": self.integrity_result.to_dict(),
            "contribution": self.contribution.to_dict(),
            "total_claims": self.total_claims,
            "total_refs_scored": self.total_refs_scored,
            "ref_scores": [
                {
                    "key": rs.key,
                    "purpose": rs.purpose,
                    "weight": rs.weight,
                    "verdict": rs.verdict,
                    "verification_score": round(rs.verification_score, 4),
                    "weighted_score": round(rs.weighted_score, 4),
                    "claim_count": rs.claim_count,
                    "supports_count": rs.supports_count,
                    "partial_count": rs.partial_count,
                    "not_supported_count": rs.not_supported_count,
                }
                for rs in self.ref_scores
            ],
        }


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_internal_score(findings: list[dict[str, Any]]) -> float:
    """Compute internal manuscript quality score from check findings.

    Returns a score in [0, 1] where 1.0 means no errors.
    Based on the ratio of non-error findings to total findings.
    If no findings at all, returns 1.0 (clean manuscript).
    """
    if not findings:
        return 1.0

    errors = sum(1 for f in findings if f.get("level") == "error")
    total = len(findings)
    return max(0.0, 1.0 - (errors / total))


def _aggregate_ref_claims(
    claims: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group claim results by reference key."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for c in claims:
        key = c.get("key", "")
        if key:
            by_key.setdefault(key, []).append(c)
    return by_key


def _score_reference(claims: list[dict[str, Any]]) -> RefScore:
    """Compute the verification score for a single reference.

    Aggregates all claims citing this reference. The overall verdict
    is the *worst* verdict across all claims (conservative). The
    purpose is taken from the most common purpose across claims.
    SKIPPED rows (cite extracted but never verified) are excluded from
    scoring — they carry no LLM judgment and would otherwise dilute the
    average. If every claim for the ref is SKIPPED, the returned
    RefScore has ``verdict='SKIPPED'`` and ``weight=0.0`` so the ref
    contributes nothing to the weighted referencing-quality average.
    """
    key = claims[0].get("key", "")

    # Exclude SKIPPED rows from scoring — they have no LLM verdict.
    scored_claims = [c for c in claims if c.get("verdict") != "SKIPPED"]

    if not scored_claims:
        return RefScore(
            key=key,
            purpose="context",
            weight=0.0,
            verdict="SKIPPED",
            verification_score=0.0,
            weighted_score=0.0,
            claim_count=len(claims),
            supports_count=0,
            partial_count=0,
            not_supported_count=0,
        )

    # Count verdicts
    supports = sum(1 for c in scored_claims if c.get("verdict") == "SUPPORTS")
    partial = sum(1 for c in scored_claims if c.get("verdict") == "PARTIALLY_SUPPORTS")
    not_supported = sum(1 for c in scored_claims if c.get("verdict") == "NOT_SUPPORTED")

    # Determine dominant purpose (most frequent, breaking ties by weight)
    purpose_counts: dict[str, int] = {}
    for c in scored_claims:
        p = c.get("citation_purpose", "evidence")
        purpose_counts[p] = purpose_counts.get(p, 0) + 1
    dominant_purpose = max(
        purpose_counts,
        key=lambda p: (purpose_counts[p], PURPOSE_WEIGHTS.get(p, 0.2)),
    )

    # Verification score: weighted average of all claim verdicts for this ref
    total_score = 0.0
    for c in scored_claims:
        if not c.get("dismissed"):
            v = c.get("verdict", "CANNOT_DETERMINE")
            total_score += VERDICT_SCORES.get(v, 0.25)
    active_claims = [c for c in scored_claims if not c.get("dismissed")]
    avg_score = total_score / len(active_claims) if active_claims else 0.25

    # Overall verdict (worst non-dismissed)
    priority = ["NOT_SUPPORTED", "CANNOT_DETERMINE", "PARTIALLY_SUPPORTS", "SUPPORTS"]
    overall_verdict = "SUPPORTS"
    for c in active_claims:
        v = c.get("verdict", "CANNOT_DETERMINE")
        if v in priority and priority.index(v) < priority.index(overall_verdict):
            overall_verdict = v

    weight = PURPOSE_WEIGHTS.get(dominant_purpose, 0.2)

    return RefScore(
        key=key,
        purpose=dominant_purpose,
        weight=weight,
        verdict=overall_verdict,
        verification_score=avg_score,
        weighted_score=weight * avg_score,
        claim_count=len(scored_claims),
        supports_count=supports,
        partial_count=partial,
        not_supported_count=not_supported,
    )


def compute_integrity(
    findings: list[dict[str, Any]],
    claim_results: list[dict[str, Any]],
    metadata_map: dict[str, CitationMetadata] | None = None,
    ref_internal_scores: dict[str, float] | None = None,
) -> IntegrityResult:
    """Compute internal consistency and referencing quality.

    These are two independent components of SciLint Score:
    - internal_consistency: manuscript's own consistency
    - referencing_quality: weighted mean of weight × verdict × reliability

    Child reliability sources (in priority order):
    1. ref_internal_scores: LLM consistency checks on cited papers
    2. metadata_map: metadata-based signals (tier, retraction, mismatches)
    3. default: reliability = verdict score

    Returns:
        IntegrityResult with both components.
    """
    internal = compute_internal_score(findings)

    # Group claims by reference and score each
    by_ref = _aggregate_ref_claims(claim_results)
    ref_scores: list[RefScore] = []
    for _key, claims in sorted(by_ref.items()):
        ref_scores.append(_score_reference(claims))

    if not ref_scores:
        return IntegrityResult(
            internal_consistency=internal,
            referencing_quality=1.0,  # no refs → no negative signal
        )

    # Build child integrity map:
    # Blend ref_internal (content-based consistency) with metadata (credibility signals)
    # ref_internal is always available (consistency checks run in default pipeline)
    child_integrity: dict[str, float] = {}
    integrity_source = "default"

    if ref_internal_scores and metadata_map:
        metadata_scores = _compute_metadata_integrity(metadata_map)
        all_keys = set(ref_internal_scores) | set(metadata_scores)
        for key in all_keys:
            ri = ref_internal_scores.get(key)
            ms = metadata_scores.get(key)
            if ri is not None and ms is not None:
                child_integrity[key] = 0.6 * ri + 0.4 * ms
            elif ri is not None:
                child_integrity[key] = ri
            else:
                child_integrity[key] = ms  # type: ignore[assignment]
        integrity_source = "ref_internal"
    elif ref_internal_scores:
        child_integrity = dict(ref_internal_scores)
        integrity_source = "ref_internal"
    elif metadata_map:
        child_integrity = _compute_metadata_integrity(metadata_map)
        integrity_source = "metadata"

    # Compute weighted reference score with recursive integrity
    weighted_sum = 0.0
    weight_sum = 0.0
    reference_reliability: dict[str, float] = {}

    for rs in ref_scores:
        # V(p, r_i) = verification_score from claim verdicts
        v_score = rs.verification_score
        # integrity(r_i): from chain/metadata/ref_internal, or V_self at leaf
        child_score = child_integrity.get(rs.key, v_score)
        combined = rs.weight * v_score * child_score
        weighted_sum += combined
        weight_sum += rs.weight
        if rs.key in child_integrity:
            reference_reliability[rs.key] = child_score

    referencing_quality = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    return IntegrityResult(
        internal_consistency=internal,
        referencing_quality=referencing_quality,
        reference_reliability=reference_reliability,
        integrity_source=integrity_source,
    )


# ---------------------------------------------------------------------------
# Lightweight child integrity from metadata signals
# ---------------------------------------------------------------------------

# Base scores by verification tier
TIER_BASE_SCORES: dict[str, float] = {
    "T1": 0.9,
    "T2": 0.7,
    "T3": 0.3,
}

MISMATCH_PENALTY = 0.1  # per accuracy mismatch (title, author, year, venue)
API_MISMATCH_PENALTY = 0.1  # api_match == "mismatch"
NON_FORMAL_PENALTY = 0.2  # non-formal document (news, guide, etc.)


def _compute_metadata_integrity(
    metadata_map: dict[str, CitationMetadata],
) -> dict[str, float]:
    """Compute child integrity scores from citation metadata signals.

    Uses tier, api_match, retraction status, and mismatches
    to produce a [0, 1] integrity estimate per reference.
    """
    return {key: _score_metadata(meta) for key, meta in metadata_map.items()}


def _score_metadata(meta: CitationMetadata) -> float:
    """Compute a single reference's lightweight integrity from its metadata."""
    # Retracted papers get zero integrity
    if meta.canonical.get("retracted"):
        # Expression of Concern: severe penalty but not terminal
        rs = meta.canonical.get("retraction_status")
        if rs and rs.get("nature") == "Expression of Concern":
            return 0.3
        return 0.0

    # Base score from tier
    tier = meta.access.get("tier") or compute_tier(meta)
    score = TIER_BASE_SCORES.get(tier, TIER_BASE_SCORES["T3"])

    # Penalty for API-level mismatch (title didn't match well)
    if meta.api_match == "mismatch":
        score -= API_MISMATCH_PENALTY

    # Penalty for accuracy mismatches (title, author, year, venue)
    score -= MISMATCH_PENALTY * len(meta.mismatches)

    # Penalty for non-formal documents (news, guides, etc.)
    if meta.access.get("is_formal") is False:
        score -= NON_FORMAL_PENALTY

    return max(0.0, min(1.0, score))


def compute_contribution(
    axis_scores: dict[str, float],
    axis_reasoning: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
) -> ContributionScores:
    """Compute overall contribution from individual axis scores.

    Args:
        axis_scores: Dict with keys matching ContributionScores fields.
            Expected keys: empirical_content, progressiveness, unification,
            problem_solving, test_severity. Missing keys default to 0.
        axis_reasoning: Optional per-axis reasoning strings.
        weights: Optional custom weights per axis.

    Returns:
        ContributionScores with per-axis and overall scores.
    """
    w = weights or CONTRIBUTION_WEIGHTS
    axes = [
        "empirical_content",
        "progressiveness",
        "unification",
        "problem_solving",
        "test_severity",
    ]

    scores = {a: max(0.0, min(1.0, axis_scores.get(a, 0.0))) for a in axes}
    total_weight = sum(w.get(a, 0.2) for a in axes)
    weighted = (
        sum(w.get(a, 0.2) * scores[a] for a in axes) / total_weight
        if total_weight > 0
        else 0.0
    )

    # Bold-claims penalty: papers with high Popper/Lakatos but zero
    # problem-solving get dampened. A null-result paper (low Lakatos)
    # with zero Laudan is fine — it's not claiming much. But a paper
    # that makes bold predictions (high Lakatos) without solving any
    # problem or acknowledging limitations is suspect.
    ps = scores.get("problem_solving", 0.0)
    lakatos = scores.get("progressiveness", 0.0)
    if ps < 0.1 and lakatos > 0.5:
        # Bold claims + zero self-awareness → dampen by how bold the claims are
        penalty = 0.5 + 0.5 * (1.0 - lakatos)  # lakatos=1.0→0.5x, lakatos=0.5→0.75x
        overall = weighted * penalty
    else:
        overall = weighted

    return ContributionScores(
        empirical_content=scores["empirical_content"],
        progressiveness=scores["progressiveness"],
        unification=scores["unification"],
        problem_solving=scores["problem_solving"],
        test_severity=scores["test_severity"],
        overall=overall,
        reasoning=axis_reasoning or {},
    )


def compute_scilint_score(
    paper_name: str,
    claim_results: list[dict[str, Any]],
    findings: list[dict[str, Any]] | None = None,
    metadata_map: dict[str, CitationMetadata] | None = None,
    ref_internal_scores: dict[str, float] | None = None,
    contribution_scores: dict[str, float] | None = None,
    contribution_reasoning: dict[str, str] | None = None,
) -> SciLintScoreResult:
    """Compute SciLint Score for a paper.

    SciLint Score = internal_consistency × referencing_quality × contribution

    Args:
        paper_name: Name of the paper.
        claim_results: Results from run_claim_verification().
        findings: Optional check findings for internal score.
        metadata_map: Optional metadata for child reliability.
        ref_internal_scores: Optional per-ref internal scores from
            consistency checks on cited papers.
        contribution_scores: Optional dict of axis name → score (0-1).
            If None, contribution defaults to 1.0.
        contribution_reasoning: Optional per-axis reasoning.

    Returns:
        SciLintScoreResult with overall score and detailed breakdown.
    """
    integrity_result = compute_integrity(
        findings or [],
        claim_results,
        metadata_map,
        ref_internal_scores=ref_internal_scores,
    )

    # Group claims by reference for ref_scores
    by_ref = _aggregate_ref_claims(claim_results)
    ref_scores: list[RefScore] = []
    for _key, claims in sorted(by_ref.items()):
        ref_scores.append(_score_reference(claims))

    if contribution_scores:
        contribution = compute_contribution(contribution_scores, contribution_reasoning)
    else:
        # No contribution data → default to 1.0 (integrity only)
        contribution = ContributionScores(overall=1.0)

    scilint_score = (
        integrity_result.internal_consistency
        * integrity_result.referencing_quality
        * contribution.overall
    )

    return SciLintScoreResult(
        paper=paper_name,
        scilint_score=scilint_score,
        integrity_result=integrity_result,
        contribution=contribution,
        total_claims=len(claim_results),
        total_refs_scored=len(ref_scores),
        ref_scores=ref_scores,
    )


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def run_contributions(
    paper_name: str,
    claims: list[dict[str, Any]],
    findings_path: Path | None = None,
    references_dir: Path | None = None,
    contribution_scores: dict[str, float] | None = None,
    contribution_reasoning: dict[str, str] | None = None,
    output_dir: Path | None = None,
) -> SciLintScoreResult:
    """Compute SciLint Score from claims + findings + metadata.

    SciLint Score = internal_consistency × referencing_quality × contribution

    Args:
        paper_name: Paper name.
        claims: Claim result dicts (from workspace.db or pipeline).
        findings_path: Optional path to check findings JSON.
        references_dir: Optional per-paper workspace root for child reliability.
        contribution_scores: Optional pre-computed contribution axis scores.
        contribution_reasoning: Optional per-axis reasoning.
        output_dir: Where to save results.

    Returns:
        SciLintScoreResult.
    """
    findings: list[dict[str, Any]] = []
    if findings_path and findings_path.exists():
        findings = json.loads(findings_path.read_text(encoding="utf-8"))

    # Load metadata for child integrity
    metadata_map: dict[str, CitationMetadata] | None = None
    if references_dir and references_dir.exists():
        from sciwrite_lint.references.metadata import load_all_metadata

        metadata_map = load_all_metadata(references_dir)

    # Load per-ref internal scores from workspace.db
    ref_internal_scores: dict[str, float] | None = None
    if references_dir and references_dir.exists():
        from sciwrite_lint.references.workspace_db import (
            get_db,
            load_all_ref_internal_scores,
        )

        with get_db(references_dir) as conn:
            scores = load_all_ref_internal_scores(conn)
            if scores:
                ref_internal_scores = scores

    result = compute_scilint_score(
        paper_name,
        claims,
        findings=findings,
        metadata_map=metadata_map,
        ref_internal_scores=ref_internal_scores,
        contribution_scores=contribution_scores,
        contribution_reasoning=contribution_reasoning,
    )

    if not output_dir:
        raise ValueError("output_dir is required")

    # Save
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scilint_{paper_name}.json"
    out_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"SciLint Score saved to {out_path}")

    return result
