"""Check: claim-support — cited paper supports the claim made about it.

Runs during ``sciwrite-lint verify-claims``. For each \\cite{key} with full
text available (T1), extracts the claim context, reads the cited source,
and uses an LLM to verify whether the source supports the claim.

This check requires vLLM + GROBID-parsed PDFs. The actual verification
happens in the pipeline's claims stage; this module registers the check
and provides the finding conversion logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sciwrite_lint.models import CheckMeta, Finding


def _is_tool_error(explanation: str) -> bool:
    """Check if the explanation indicates a tool/LLM failure."""
    if not explanation:
        return False
    e = explanation.lower()
    return e.startswith("parse error") or "connection" in e or "timeout" in e


def claims_to_findings(
    results: list[dict[str, Any]],
    tex_path: Path,
) -> list[Finding]:
    """Convert claim verification results to findings.

    Verdicts:
    - NOT_SUPPORTED → error
    - PARTIALLY_SUPPORTS → warning
    - SUPPORTS → no finding
    - CANNOT_DETERMINE → info if tool error, otherwise no finding
    """
    findings: list[Finding] = []
    for r in results:
        if r.get("dismissed"):
            continue
        # Example citations illustrate a point — the source doesn't need to
        # "support" the broader claim, so skip claim-support findings.
        purpose = r.get("cite_purpose") or r.get("citation_purpose", "")
        if purpose == "example":
            continue
        verdict = r.get("verdict", "CANNOT_DETERMINE")
        explanation = r.get("explanation", "")
        if verdict == "NOT_SUPPORTED":
            findings.append(
                Finding(
                    level="error",
                    rule_id="claim-support",
                    message=f"{r.get('key', '?')}: Claim not supported by cited source",
                    file=tex_path.name,
                    line=r.get("line"),
                    context=explanation,
                )
            )
        elif verdict == "PARTIALLY_SUPPORTS":
            findings.append(
                Finding(
                    level="warning",
                    rule_id="claim-support",
                    message=f"{r.get('key', '?')}: Claim only partially supported",
                    file=tex_path.name,
                    line=r.get("line"),
                    context=explanation,
                )
            )
        elif verdict == "CANNOT_DETERMINE" and _is_tool_error(explanation):
            findings.append(
                Finding(
                    level="info",
                    rule_id="claim-support",
                    message=(
                        f"{r.get('key', '?')}: Could not verify claim (LLM error)"
                    ),
                    file=tex_path.name,
                    line=r.get("line"),
                    context=explanation[:200],
                )
            )
    return findings


# Metadata for `sciwrite-lint checks` listing.
# This check runs as a pipeline stage (not from the check registry).
CLAIM_SUPPORT_META = CheckMeta(
    id="claim-support",
    severity="warning",
    category="local-llm",
    description="Cited paper does not support the claim attached to it.",
)
