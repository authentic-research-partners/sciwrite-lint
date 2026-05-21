"""Check: cite-purpose — citation has an identifiable argumentative role.

For each reference in a paper, determines whether the citation serves a
legitimate argumentative function. Citations that serve none — typically
background padding — are flagged.

Classification is done per-claim during claim verification (eval_claims.py,
_classify_citation_vllm). This module owns the weights, threshold, and
finding conversion logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sciwrite_lint.models import CheckMeta, Finding
from sciwrite_lint.schemas import CITATION_PURPOSE_NAMES

# ---------------------------------------------------------------------------
# Citation purpose weights — single source of truth
# ---------------------------------------------------------------------------

# Every purpose category has a weight. Used by:
# - cite-purpose check: weight <= UNSPECIFIED_THRESHOLD → flag
# - SciLint Score: w_i in the integrity equation
PURPOSE_WEIGHTS: dict[str, float] = {
    "evidence": 1.0,  # Popper: supports falsifiable claim
    "contrast": 0.9,  # Mayo: tested against, improves upon
    "method": 0.8,  # Laudan: methodology provenance
    "definition": 0.7,  # establishes terminology
    "example": 0.6,  # illustrative instance of a broader point
    "attribution": 0.5,  # Laudan: establishes origin
    "tool": 0.4,  # tool/dataset provenance
    "context": 0.2,  # background padding — no specific claim
}

assert set(PURPOSE_WEIGHTS) == set(CITATION_PURPOSE_NAMES), (
    f"PURPOSE_WEIGHTS keys {set(PURPOSE_WEIGHTS)} != "
    f"CITATION_PURPOSE_NAMES {set(CITATION_PURPOSE_NAMES)}"
)

# Citations at or below this weight are flagged as lacking argumentative role
UNSPECIFIED_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Finding conversion
# ---------------------------------------------------------------------------


def _is_tool_error(reasoning: str) -> bool:
    """Check if the reasoning indicates a tool/LLM failure, not a real classification."""
    if not reasoning:
        return False
    r = reasoning.lower()
    return r.startswith("parse error") or r.startswith("error:")


def cite_purposes_to_findings(
    results: list[dict[str, Any]],
    tex_path: Path,
) -> list[Finding]:
    """Flag citations with no identifiable argumentative function.

    Accepts results from the claim verification pipeline (citation_purpose
    field). Uses PURPOSE_WEIGHTS to determine if the citation is unspecified.
    """
    findings: list[Finding] = []
    for r in results:
        if r.get("dismissed"):
            continue
        # SKIPPED rows never reached the verifier (no local source / filtered
        # out), so cite_purpose was never classified — don't fabricate a
        # "no argumentative function" finding from missing data.
        if r.get("verdict") == "SKIPPED":
            continue

        # Prefer independent classification; use eval_claims field if absent
        purpose = r.get("cite_purpose") or r.get("citation_purpose", "context")
        reasoning = r.get("cite_reasoning", "")
        weight = PURPOSE_WEIGHTS.get(purpose, 0.2)

        # Tool errors should not produce false findings — report as info
        if weight <= UNSPECIFIED_THRESHOLD and _is_tool_error(reasoning):
            findings.append(
                Finding(
                    level="info",
                    rule_id="cite-purpose",
                    message=(
                        f"{r.get('key', '?')}: Could not classify citation "
                        f"purpose (LLM error)"
                    ),
                    file=tex_path.name,
                    line=r.get("line"),
                    context=reasoning[:200],
                )
            )
            continue

        if weight <= UNSPECIFIED_THRESHOLD:
            # Prefer LLM reasoning over raw paragraph text
            ctx = reasoning or r.get("context", "")
            findings.append(
                Finding(
                    level="warning",
                    rule_id="cite-purpose",
                    message=(
                        f"{r.get('key', '?')}: Citation provides background "
                        f"context but does not support a specific claim, "
                        f"method, problem, or contrast"
                    ),
                    file=tex_path.name,
                    line=r.get("line"),
                    context=ctx[:200],
                )
            )
    return findings


# Metadata for `sciwrite-lint checks` listing.
# This check runs as a pipeline stage (not from the check registry).
CITE_PURPOSE_META = CheckMeta(
    id="cite-purpose",
    severity="warning",
    category="local-llm",
    description="Citation has no identifiable argumentative role.",
)
