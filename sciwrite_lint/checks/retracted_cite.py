"""Check: retracted-cite — flags references in the Retraction Watch database.

Uses the Retraction Watch database (downloaded CSV from CrossRef Labs) to
detect retracted papers, expressions of concern, and corrections among
the manuscript's references. The CSV is cached locally and refreshed daily.

Runs during ``sciwrite-lint verify``. No additional API calls beyond the
initial CSV download — uses DOI from stored metadata for lookup.
"""

from __future__ import annotations

from pathlib import Path

from sciwrite_lint.checks.registry import check
from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import CitationMetadata, Finding


def check_retraction_from_metadata(
    all_metadata: dict[str, CitationMetadata],
) -> list[Finding]:
    """Generate findings from retraction_status stored in metadata.

    Called after the RW enrichment pass has run and retraction_status
    has been written into canonical dicts.
    """
    findings: list[Finding] = []

    for key, meta in sorted(all_metadata.items()):
        status = meta.canonical.get("retraction_status")
        if not status:
            continue

        nature = status.get("nature", "")
        reason = status.get("reason", "")
        reason_suffix = f" — {reason}" if reason else ""
        doi = meta.canonical.get("doi", "")
        ctx = f"DOI: {doi}" if doi else ""

        if nature == "Retraction":
            findings.append(
                Finding(
                    level="error",
                    rule_id="retracted-cite",
                    message=f"{key}: RETRACTED{reason_suffix}",
                    context=ctx,
                )
            )
        elif nature == "Expression of Concern":
            findings.append(
                Finding(
                    level="warning",
                    rule_id="retracted-cite",
                    message=(
                        f"{key}: EXPRESSION OF CONCERN — this paper has an "
                        f"editorial expression of concern{reason_suffix}"
                    ),
                    context=ctx,
                )
            )
        elif nature == "Correction":
            findings.append(
                Finding(
                    level="info",
                    rule_id="retracted-cite",
                    message=f"{key}: has a published correction{reason_suffix}",
                    context=ctx,
                )
            )

    return findings


@check(
    id="retracted-cite",
    category="reference-db",
    severity="error",
    description="Reference appears in the Retraction Watch database (retracted, expression of concern, or correction).",
)
def check_retracted_cite(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Load stored metadata and check for retraction status.

    No API calls — uses retraction_status in references/metadata/{key}.json
    populated by the RW enrichment pass during verify.
    """
    from sciwrite_lint.references.workspace_db import get_db, query_retracted_refs

    refs_dir = config.effective_references_dir()

    if not refs_dir.exists():
        return []

    with get_db(refs_dir) as conn:
        retracted = query_retracted_refs(conn)
    return check_retraction_from_metadata(retracted)
