"""Claim taxonomy classifier for SciLint Score contribution scoring.

Classifies each extracted claim along five dimensions:
- type: prediction | explanation | reproduction | synthesis
- specificity: quantified | directional | vague
- testability: falsifiable | unfalsifiable | tautological
- support: severe_test | weak_test | no_test | post_hoc
- scope: cross_domain | within_domain | within_paper

Uses vLLM for classification. One LLM call per claim (batched).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.llm_utils import (
    MEDIUM_PROMPT_CONCURRENCY,
    llm_query,
    llm_query_batch,
)
from sciwrite_lint.schemas import (
    ClaimClassification as ClaimClassificationSchema,
    vllm_schema_unbounded,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class ClaimClassification(BaseModel):
    """Taxonomy classification for a single claim."""

    claim_text: str
    key: str  # citation key

    type: Literal["prediction", "explanation", "reproduction", "synthesis"]
    specificity: Literal["quantified", "directional", "vague"]
    testability: Literal["falsifiable", "unfalsifiable", "tautological"]
    support: Literal["severe_test", "weak_test", "no_test", "post_hoc"]
    scope: Literal["cross_domain", "within_domain", "within_paper"]

    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "key": self.key,
            "type": self.type,
            "specificity": self.specificity,
            "testability": self.testability,
            "support": self.support,
            "scope": self.scope,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """\
You are a philosophy-of-science claim classifier. You will receive a CLAIM \
from a paper, plus SURROUNDING CONTEXT that may include information about \
the paper's methodology and results. Classify the claim along five dimensions.

**Type** — what role does THIS claim play in THIS paper?
- prediction: a NOVEL result or contribution from the authors — empirical OR \
theoretical. Any new finding, derivation, or proof counts. Examples: \
"Our drug reduces tumor volume by 40%." / "We prove the error rate exceeds \
50% under these conditions." / "The model achieves state-of-the-art on the \
benchmark."
- explanation: accounts for WHY something happens using EXISTING knowledge, \
without novel contribution. Example: "This effect may stem from increased \
membrane permeability" (known mechanism applied to explain a result).
- reproduction: explicitly replicates a PREVIOUSLY PUBLISHED result by other \
authors. Must reference the original and confirm it. Example: "We replicate \
the findings of Garcia et al. on dataset X."
- synthesis: combines ideas from multiple sources into a new framework. \
Example: "We integrate game theory and epidemiology into a unified model."

**Specificity** — how precisely is the claim stated?
- quantified: contains specific numbers, thresholds, or measurable quantities. \
Example: "reduces latency by 37ms" or "AUC of 0.92."
- directional: states a direction without exact values. Example: "significantly \
improves over the baseline" or "correlates positively with."
- vague: no direction or magnitude — a general assertion. Example: "plays an \
important role" or "is becoming increasingly relevant."

**Testability** (Popper) — could this claim be proven wrong?
- falsifiable: a concrete experiment could disprove it. Example: "mortality \
rate drops below 5% with treatment X" is falsifiable — run the trial.
- unfalsifiable: no observation could disprove it, often due to open-ended \
timeframes or unmeasurable scope. Example: "consciousness will ultimately be \
understood through neuroscience" — no failure point exists.
- tautological: true by definition with no empirical content. Example: "all \
bachelors are unmarried."

**Support** (Mayo — test severity) — determine from SURROUNDING CONTEXT:
- severe_test: the claim is backed by rigorous evidence. Indicators include:
  * Randomized controlled trials (RCTs), pre-registered studies, large sample sizes
  * Standard benchmarks or blind evaluations (competition-style assessments)
  * Multiple comparisons against strong alternatives, ablation studies
  * Statistical tests with confidence intervals, replicated runs
  * Formal proof, mathematical derivation, rigorous logical argument
- weak_test: some evidence but lacking rigor — few baselines, no ablations, \
single run, missing controls. Also applies when claims are extraordinary \
(overturning established scientific consensus) but the evidence is not \
proportionally strong — e.g., no peer review, no independent replication, \
no controls for confounders.
- no_test: no evidence described in context for this claim.
- post_hoc: the claim explains a result that was observed first, then \
rationalized. Signals: "surprisingly", "we speculate this is because", \
"one possible explanation is."

**Scope** (Kitcher — unification):
- cross_domain: bridges distinct fields (e.g., physics and biology).
- within_domain: extends knowledge within one field.
- within_paper: relevant only to this paper's specific setup.

Respond with ONLY a valid JSON object."""

CLASSIFY_USER_TEMPLATE = """\
PAPER CONTEXT (read this FIRST — it determines the support classification):
<paper_context>
{context}
</paper_context>

CLAIM TO CLASSIFY:
<claim>
{claim_text}
</claim>
CITATION KEY: {key}"""

CLASSIFY_SCHEMA = vllm_schema_unbounded(ClaimClassificationSchema)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Valid values for each dimension (used for validation)
_VALID: dict[str, set[str]] = {
    "type": {"prediction", "explanation", "reproduction", "synthesis"},
    "specificity": {"quantified", "directional", "vague"},
    "testability": {"falsifiable", "unfalsifiable", "tautological"},
    "support": {"severe_test", "weak_test", "no_test", "post_hoc"},
    "scope": {"cross_domain", "within_domain", "within_paper"},
}


def _validate_classification(result: dict[str, Any]) -> bool:
    """Check that all required fields have valid enum values."""
    for dim, valid_vals in _VALID.items():
        if result.get(dim) not in valid_vals:
            return False
    return True


async def classify_claim(
    claim_text: str,
    key: str,
    context: str = "",
    config: LintConfig | None = None,
    model_name: str = "",
) -> ClaimClassification | None:
    """Classify a single claim using vLLM.

    Returns None if LLM call fails or returns invalid output.
    """
    user = CLASSIFY_USER_TEMPLATE.format(
        claim_text=claim_text, key=key, context=context
    )

    result = await llm_query(
        system=CLASSIFY_SYSTEM,
        user=user,
        schema=CLASSIFY_SCHEMA,
        schema_name="ClaimClassification",
        config=config,
        model_name=model_name,
        thinking="medium",
    )

    if not result or not _validate_classification(result):
        logger.warning(f"Failed to classify claim for {key}: {result}")
        return None

    return ClaimClassification(
        claim_text=claim_text,
        key=key,
        type=result["type"],
        specificity=result["specificity"],
        testability=result["testability"],
        support=result["support"],
        scope=result["scope"],
        reasoning=result.get("reasoning", ""),
    )


async def classify_claims_batch(
    claims: list[dict[str, Any]],
    config: LintConfig | None = None,
    model_name: str = "",
) -> list[ClaimClassification]:
    """Classify a batch of claims using vLLM.

    Args:
        claims: List of dicts with 'key', 'context' (claim text), and
            optional 'line' fields — the format from verify-claims output.
        config: Lint configuration.
        model_name: vLLM model preset name.

    Returns:
        List of successfully classified claims.
    """
    if not claims:
        return []

    queries: list[tuple[str, str, dict, str]] = []
    for c in claims:
        # claim_text: the specific claim sentence to classify
        # context: surrounding paper context (may include methods/results)
        claim_text = c.get("claim_text", "") or c.get("context", "")
        context = c.get("context", claim_text)
        user = CLASSIFY_USER_TEMPLATE.format(
            claim_text=claim_text,
            key=c.get("key", ""),
            context=context,
        )
        queries.append((CLASSIFY_SYSTEM, user, CLASSIFY_SCHEMA, "ClaimClassification"))

    results = await llm_query_batch(
        queries,
        config=config,
        model_name=model_name,
        thinking="medium",
        concurrency=MEDIUM_PROMPT_CONCURRENCY,
        size_class="medium",
    )

    classified: list[ClaimClassification] = []
    for c, result in zip(claims, results):
        if result and _validate_classification(result):
            classified.append(
                ClaimClassification(
                    claim_text=c.get("claim_text", "") or c.get("context", ""),
                    key=c.get("key", ""),
                    type=result["type"],
                    specificity=result["specificity"],
                    testability=result["testability"],
                    support=result["support"],
                    scope=result["scope"],
                    reasoning=result.get("reasoning", ""),
                )
            )
        else:
            logger.warning(f"Failed to classify claim for {c.get('key', '?')}")

    logger.info(f"Classified {len(classified)}/{len(claims)} claims")
    return classified


# ---------------------------------------------------------------------------
# Aggregation helpers for contribution scoring
# ---------------------------------------------------------------------------


def empirical_content_score(classifications: list[ClaimClassification]) -> float:
    """Popper: fraction of falsifiable claims, weighted by specificity.

    quantified falsifiable → 1.0
    directional falsifiable → 0.7
    vague falsifiable → 0.3
    unfalsifiable/tautological → 0.0
    """
    if not classifications:
        return 0.0

    specificity_weight = {"quantified": 1.0, "directional": 0.7, "vague": 0.3}
    total = 0.0
    for c in classifications:
        if c.testability == "falsifiable":
            total += specificity_weight.get(c.specificity, 0.3)

    return total / len(classifications)


def progressiveness_score(classifications: list[ClaimClassification]) -> float:
    """Lakatos: novel contributions / (novel + accommodations).

    Predictions count fully. Syntheses count at 0.7 (novel framework).
    Explanations of NOVEL results (cross_domain scope) count at 0.5.
    Pure accommodations (within_paper explanations) count against.
    Reproductions are neutral (not counted).
    """
    novel = 0.0
    accommodations = 0.0
    for c in classifications:
        if c.type == "prediction":
            novel += 1.0
        elif c.type == "synthesis":
            novel += 0.7
        elif c.type == "explanation":
            if c.scope == "cross_domain":
                novel += 0.5  # novel cross-domain insight
            else:
                accommodations += 1.0

    denominator = novel + accommodations
    if denominator == 0:
        return 0.5

    return novel / denominator


def severity_score(classifications: list[ClaimClassification]) -> float:
    """Mayo: fraction of claims with severe tests.

    severe_test → 1.0, weak_test → 0.5, no_test → 0.0, post_hoc → 0.0
    """
    if not classifications:
        return 0.0

    support_weight = {
        "severe_test": 1.0,
        "weak_test": 0.5,
        "no_test": 0.0,
        "post_hoc": 0.0,
    }
    total = sum(support_weight.get(c.support, 0.0) for c in classifications)
    return total / len(classifications)
