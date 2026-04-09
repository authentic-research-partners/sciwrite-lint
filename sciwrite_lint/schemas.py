"""Pydantic models for vLLM structured output schemas.

Every string field has a generous ``max_length`` (chars) set at ~2x natural
output length. This tells vLLM's constrained decoder how much space each
field gets, so the JSON structure completes within ``max_tokens``. The limits
are ceilings — the model writes naturally below them.

Conservative token math: at worst-case 3 chars/token, the total constrained
response for the largest schema (ClaimVerdict) is ~192 tokens, leaving 3904
of 4096 max_tokens for thinking.

To get the JSON schema dict for vLLM::

    from sciwrite_lint.schemas import ClaimVerdict, vllm_schema
    schema = vllm_schema(ClaimVerdict)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Citation purpose categories — single source of truth
# ---------------------------------------------------------------------------

# Canonical ordered dict: purpose → (short description, verification question).
# Every consumer (prompts, schemas, weights, verify questions) derives from this.
CITATION_PURPOSES: OrderedDict[str, tuple[str, str]] = OrderedDict(
    [
        (
            "evidence",
            (
                "The sentence states a specific, quantitative or empirical finding and cites this source as proof. Remove the citation and the claim loses its support",
                "Does this source contain data, findings, or arguments that support the specific factual claim being made?",
            ),
        ),
        (
            "example",
            (
                "The cited work is a case or instance being discussed to illustrate a broader point. The sentence talks ABOUT the cited work as a story or case — it does not credit authorship. E.g. 'the graphene paper was a curiosity before it was a Nobel Prize'",
                "Is this the work being cited as an illustrative example? The source does not need to support the broader claim — it only needs to be the example described.",
            ),
        ),
        (
            "attribution",
            (
                "The sentence names WHO created a term, concept, or idea — the focus is on crediting the person, not explaining what the concept means. E.g. 'X was introduced by [cite]'",
                "Is this source where the term, concept, or idea being attributed was originally introduced?",
            ),
        ),
        (
            "tool",
            (
                "Names a software tool, library, system, or dataset used in this work",
                "Does this source describe or document the tool, system, or dataset being referenced?",
            ),
        ),
        (
            "method",
            (
                "The sentence describes a methodology, algorithm, or approach adopted from this source. The focus is on HOW something is done, not on what a concept means",
                "Does this source describe the methodology or approach being referenced?",
            ),
        ),
        (
            "definition",
            (
                "The sentence explains WHAT a concept means, citing this source as the authority for the definition. Look for 'as defined by', 'the notion of X as', or explicit criteria/thresholds from the source",
                "Does this source contain the definition being cited?",
            ),
        ),
        (
            "contrast",
            (
                "Presents a finding or approach that this paper disagrees with, compares against, or improves upon",
                "Does this source present findings, methods, or claims that the citing paper disagrees with, improves upon, or tests against?",
            ),
        ),
        (
            "context",
            (
                "The citation decorates a general statement — remove it and the sentence still makes the same point. Typical: 'X has been applied to A [cite], B [cite]' or 'there has been growing interest in [cite]'. No specific finding from this source is used",
                "Is this source relevant to the topic being discussed?",
            ),
        ),
    ]
)

# Tuple of valid purpose names (for Literal type construction and validation).
CITATION_PURPOSE_NAMES: tuple[str, ...] = tuple(CITATION_PURPOSES.keys())

# Pre-built dicts for common access patterns.
PURPOSE_DESCRIPTIONS: dict[str, str] = {k: v[0] for k, v in CITATION_PURPOSES.items()}
VERIFY_QUESTIONS: dict[str, str] = {k: v[1] for k, v in CITATION_PURPOSES.items()}

# Type alias — Literal can't be built dynamically, so we spell it out once here.
CitationPurposeLiteral = Literal[
    "evidence",
    "example",
    "attribution",
    "tool",
    "method",
    "definition",
    "contrast",
    "context",
]


def vllm_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate a JSON schema dict for vLLM's constrained decoder.

    Inlines ``$ref`` references (vLLM doesn't resolve ``$defs``) and strips
    Pydantic metadata (``title``, ``description``) to keep the schema compact.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].rsplit("/", 1)[-1]
                return _resolve(defs[ref_name])
            return {k: _resolve(v) for k, v in obj.items() if k != "title"}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj

    return _resolve(raw)


# ---------------------------------------------------------------------------
# Claim verification (eval_claims.py)
# ---------------------------------------------------------------------------


class CitationClassify(BaseModel):
    """Citation purpose classification (sentence-level)."""

    purpose: CitationPurposeLiteral
    reasoning: str = Field(max_length=400)


class ClaimVerdict(BaseModel):
    """Claim-vs-source verification result."""

    verdict: Literal[
        "SUPPORTS",
        "PARTIALLY_SUPPORTS",
        "NOT_SUPPORTED",
        "CANNOT_DETERMINE",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    relevant_quote: str = Field(max_length=500)
    explanation: str = Field(max_length=500)


class NarrowContext(BaseModel):
    """Extracted sentence(s) for a citation from a paragraph."""

    sentences: str = Field(max_length=400)


# ---------------------------------------------------------------------------
# Cross-section consistency (checks/cross_section_consistency.py)
# ---------------------------------------------------------------------------


class Contradiction(BaseModel):
    """A single contradiction between two sections."""

    type: str = Field(max_length=100)
    section_a_says: str = Field(max_length=400)
    section_b_says: str = Field(max_length=400)
    explanation: str = Field(max_length=400)
    is_genuine: bool = Field(
        description="true ONLY if the two passages actively disagree"
    )


class ConsistencyResult(BaseModel):
    """Cross-section consistency check result."""

    contradictions: list[Contradiction]


# ---------------------------------------------------------------------------
# Structure promises (checks/structure_promises.py)
# ---------------------------------------------------------------------------


class ContribCount(BaseModel):
    """Contribution count mismatch check result."""

    claimed_count: int
    listed_count: int
    mismatch: bool
    explanation: str = Field(max_length=400)


# ---------------------------------------------------------------------------
# Claim taxonomy (claims.py)
# ---------------------------------------------------------------------------


class ClaimClassification(BaseModel):
    """5-dimension claim taxonomy classification."""

    type: Literal["prediction", "explanation", "reproduction", "synthesis"]
    specificity: Literal["quantified", "directional", "vague"]
    testability: Literal["falsifiable", "unfalsifiable", "tautological"]
    support: Literal["severe_test", "weak_test", "no_test", "post_hoc"]
    scope: Literal["cross_domain", "within_domain", "within_paper"]


# ---------------------------------------------------------------------------
# Laudan problem-solving (scoring/contribution.py)
# ---------------------------------------------------------------------------


class LaudanProblemSolving(BaseModel):
    """Laudan problem-solving effectiveness score."""

    num_problems_solved: int = Field(ge=0)
    num_acknowledged: int = Field(ge=0)
    num_unacknowledged: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Venue matching (references/matching.py)
# ---------------------------------------------------------------------------


class VenueMatch(BaseModel):
    """Venue name confirmation result."""

    same_venue: bool


# ---------------------------------------------------------------------------
# Full-paper consistency (checks/full_paper_consistency.py)
# ---------------------------------------------------------------------------


class FullPaperIssue(BaseModel):
    """A single issue found by a full-paper consistency check."""

    description: str = Field(max_length=300)
    evidence: str = Field(max_length=300)
    location: str = Field(max_length=100)
    is_genuine: bool = Field(description="true ONLY for clear, unambiguous issues")


class FullPaperIssueList(BaseModel):
    """Result from a full-paper consistency check."""

    issues: list[FullPaperIssue]
