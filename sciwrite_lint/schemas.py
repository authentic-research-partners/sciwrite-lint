"""Pydantic models for vLLM structured output schemas.

Every string field has a generous ``max_length`` (chars) set at ~2x natural
output length. **These bounds enforce post-decode validation only** — they
do NOT reach the wire schema sent to vLLM, because ``maxLength`` /
``maxItems`` constraints trigger xgrammar's slow path on this stack
(0/30 success at concurrency=60 with a single bounded field). Bounds are
enforced via a four-layer architecture: prompt word-count guidance +
``max_tokens`` sized to fit + parser truncation (``truncate_to_model``)
+ Pydantic ``model_validate`` as a defense-in-depth check.

Conservative token math (3 chars/token worst case): the widest single-item
schema is ``ClaimVerdict`` at ~300 tokens; list-valued schemas dominate the
budget at their ``max_length`` cap (``FullPaperIssueList``: 5 × ~220 ≈ 1100
tokens; ``ConsistencyResult``: 4 × ~400 ≈ 1600 tokens). These response
sizes, combined with the active thinking preset's budget, determine the
total ``max_tokens`` sent to vLLM — see ``llm_utils.py::llm_query`` for
how the two are combined.

To build a wire schema for vLLM::

    from sciwrite_lint.schemas import ClaimVerdict, vllm_schema_unbounded
    schema = vllm_schema_unbounded(ClaimVerdict)  # strips bounds for the wire

The ``vllm_schema()`` (without ``_unbounded``) helper exists for the
schema-generator unit tests; do NOT use it for live ``response_format``
payloads — the bounded form ships ``maxLength`` / ``maxItems`` on the
wire and triggers xgrammar's slow path.
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

    !!! Important !!! For vLLM call sites where Pydantic ``Field(max_length=...)``
    or list-length bounds exist on the model, prefer
    ``vllm_schema_unbounded`` — sending ``maxLength`` / ``maxItems`` over
    the wire on this stack triggers xgrammar's slow path and collapses
    throughput.
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


# Constraint keys stripped by ``vllm_schema_unbounded``. These trigger
# xgrammar's slow path on the vLLM/xgrammar version we run — bench
# confirmed a single ``maxLength`` collapses 30/30 success to 0/30 with
# every request hitting the client-side timeout. The Pydantic model
# still carries these constraints for **post-decode validation**; they
# just don't go on the wire.
_VLLM_BANNED_SCHEMA_KEYS = frozenset(
    {
        "maxLength",
        "minLength",
        "maxItems",
        "minItems",
        "pattern",
        "multipleOf",
        "maximum",
        "minimum",
        "exclusiveMaximum",
        "exclusiveMinimum",
    }
)


def _strip_vllm_banned_keys(obj: Any) -> Any:
    """Recursively strip xgrammar-slow constraint keys from a JSON
    schema dict tree. Returns a new structure (does not mutate input)."""
    if isinstance(obj, dict):
        return {
            k: _strip_vllm_banned_keys(v)
            for k, v in obj.items()
            if k not in _VLLM_BANNED_SCHEMA_KEYS
        }
    if isinstance(obj, list):
        return [_strip_vllm_banned_keys(v) for v in obj]
    return obj


def vllm_schema_unbounded(model: type[BaseModel]) -> dict[str, Any]:
    """Generate a JSON schema dict for vLLM, with length / count
    constraints stripped.

    Use this **instead of** ``vllm_schema`` when the Pydantic model
    carries ``Field(max_length=...)`` or list-length bounds — those
    constraints trigger xgrammar's slow path on this stack (a single
    ``maxLength`` field is enough to collapse throughput from 30/30 to
    0/30, every request hits the client-side timeout). The Pydantic
    model still validates post-decode; only the wire schema sent to
    vLLM is stripped.
    """
    return _strip_vllm_banned_keys(vllm_schema(model))


# ---------------------------------------------------------------------------
# Defensive truncation for LLM-output dicts
# ---------------------------------------------------------------------------


def _max_length_from_metadata(metadata: list[Any]) -> int | None:
    """Pull a ``max_length`` constraint out of a Pydantic field's
    metadata list. Used by ``truncate_to_model`` to apply the same
    cap that ``vllm_schema_unbounded`` strips from the wire."""
    from annotated_types import MaxLen

    for m in metadata:
        if isinstance(m, MaxLen):
            return int(m.max_length)
    return None


def _coerce_to_bounds(value: Any, annotation: Any, cap: int | None) -> Any:
    """Truncate one value against its declared annotation + max_length.

    Strings: clip to ``cap`` chars. Lists: clip to ``cap`` items, then
    recurse into each item if it's a nested ``BaseModel`` (so item
    fields get their own caps applied). Other types pass through.
    """
    from typing import get_args, get_origin

    if isinstance(value, str):
        return value[:cap] if cap is not None else value
    if isinstance(value, list):
        truncated = value[:cap] if cap is not None else value
        if get_origin(annotation) is list:
            args = get_args(annotation)
            if args:
                inner = args[0]
                if isinstance(inner, type) and issubclass(inner, BaseModel):
                    return [
                        truncate_to_model(inner, item)
                        if isinstance(item, dict)
                        else item
                        for item in truncated
                    ]
        return truncated
    return value


_CHARS_PER_WORD = 8  # English including spaces & punctuation


def pydantic_max(model: type[BaseModel], field: str) -> int | None:
    """Read the ``max_length`` constraint from a Pydantic field.

    For string fields, returns the character cap. For list fields,
    returns the item cap (Pydantic emits this as ``maxItems`` in JSON
    schema). Returns ``None`` if the field doesn't exist on the
    model or doesn't carry a ``max_length``.

    Used by prompt builders to keep "under ~N words" / "at most N
    items" guidance in sync with Pydantic's ``max_length`` — this is
    the prompt-side layer of the schema bounds architecture (see the
    module docstring above).
    """
    if field not in model.model_fields:
        return None
    return _max_length_from_metadata(model.model_fields[field].metadata)


def chars_to_word_hint(chars: int | None) -> int:
    """Convert a character cap to a natural-language word target.

    Rounds up to the nearest 5 so the prompt reads cleanly ("under
    ~50 words" not "under ~37 words"). Returns 0 if input is None,
    so callers can use this on optional max_length lookups without
    branching.

    Math: ~8 chars/word for English with spaces and punctuation.
    """
    if chars is None or chars <= 0:
        return 0
    words = chars // _CHARS_PER_WORD
    return ((words + 4) // 5) * 5


def truncate_to_model(model: type[BaseModel], data: Any) -> Any:
    """Truncate an LLM-output dict to match a Pydantic model's bounds.

    The wire schema is unbounded (``vllm_schema_unbounded``), so the
    LLM can emit slightly more than the prompt asked for — extra list
    items or longer strings. Calling this helper before
    ``model.model_validate(data)`` (or before iterating dict fields
    directly) ensures consumers see bounded data and never raise
    ``ValidationError`` on a minor over-run.

    Walks the model's fields recursively: top-level lists clipped to
    their ``max_length`` count, top-level strings clipped to their
    ``max_length``, and nested submodels in lists get the same
    treatment for their fields. Returns a new structure (does not
    mutate input).
    """
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k not in model.model_fields:
            out[k] = v
            continue
        finfo = model.model_fields[k]
        out[k] = _coerce_to_bounds(
            v, finfo.annotation, _max_length_from_metadata(finfo.metadata)
        )
    return out


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

    contradictions: list[Contradiction] = Field(max_length=4)


# ---------------------------------------------------------------------------
# Internal-consistency pairs (checks/internal_consistency_pairs.py)
# ---------------------------------------------------------------------------


class PairContradiction(BaseModel):
    """A contradiction between an anchor sentence and one of its neighbors.

    ``neighbor_index`` is 1-based and references the numbered neighbor list
    in the user prompt — so process_results can map the finding back to the
    specific (anchor, neighbor) pair without re-running retrieval. The
    upper bound is set comfortably above ``_MAX_NEIGHBORS`` in
    ``checks/internal_consistency_pairs.py`` (currently 12) so a future
    bump of that ceiling does not require a schema change. Out-of-range
    values are dropped in ``_process_results``.
    """

    neighbor_index: int = Field(ge=1, le=32)
    type: str = Field(max_length=100)
    anchor_says: str = Field(max_length=400)
    neighbor_says: str = Field(max_length=400)
    explanation: str = Field(max_length=400)
    is_genuine: bool = Field(
        description="true ONLY if the anchor and that neighbor actively disagree"
    )


class PairConsistencyResult(BaseModel):
    """Per-anchor result: contradictions found across this anchor's neighbors.

    Capped at 4 entries to bound the LLM's output budget — when more
    than 4 contradictions exist for an anchor the prompt instructs the
    model to return the most clear-cut.
    """

    contradictions: list[PairContradiction] = Field(max_length=4)


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

    issues: list[FullPaperIssue] = Field(max_length=5)


# ---------------------------------------------------------------------------
# Prose quality — grammar + semantic word-choice (checks/prose_quality.py)
# ---------------------------------------------------------------------------


class ProseIssue(BaseModel):
    """One grammar or semantic-word-choice issue in a single sentence.

    ``confidence`` maps to ``Finding.level`` — ``low`` findings are
    dropped; ``medium`` becomes ``info``; ``high`` becomes ``warning``.
    """

    type: Literal["grammar", "semantic"]
    span: str = Field(max_length=300)
    issue: str = Field(max_length=300)
    suggestion: str = Field(max_length=300)
    confidence: Literal["low", "medium", "high"]


class ProseIssueList(BaseModel):
    """Per-sentence result from the prose-quality check."""

    issues: list[ProseIssue] = Field(max_length=3)
