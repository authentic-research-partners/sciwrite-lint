"""Check: reference-exists — citation found in academic APIs or URL alive.

Runs during `sciwrite-lint verify`. Checks stored metadata to determine
whether each reference was found in CrossRef, OpenAlex, or Semantic Scholar
(academic) or has a live URL (web resources).

Uses metadata already persisted in references/metadata/{key}.json — the
actual API calls happen in the verify pipeline stage, not here.
"""

from __future__ import annotations

from pathlib import Path

from sciwrite_lint.checks.registry import check
from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import CitationMetadata, Finding


def _searched_ids_context(meta: CitationMetadata) -> str:
    """Build a human-readable summary of identifiers available for lookup."""
    bib = meta.bibitem or {}
    ids: list[str] = []
    if bib.get("doi"):
        ids.append(f"doi={bib['doi']}")
    if bib.get("arxiv_id"):
        ids.append(f"arxiv={bib['arxiv_id']}")
    if bib.get("pmid"):
        ids.append(f"pmid={bib['pmid']}")
    if bib.get("isbn"):
        ids.append(f"isbn={bib['isbn']}")
    if bib.get("lccn"):
        ids.append(f"lccn={bib['lccn']}")
    if bib.get("url"):
        ids.append(f"url={bib['url']}")
    return f"Searched with: {', '.join(ids)}" if ids else "No identifiers in bib entry"


def check_reference_exists_from_metadata(
    all_metadata: dict[str, CitationMetadata],
) -> list[Finding]:
    """Check that all references were found in APIs.

    Flags:
    - not_found: ERROR — reference not in any API
    - dead URL: ERROR — web resource URL returned HTTP 404/410 (server
      explicitly confirmed the resource does not exist)
    - blocked by ...: WARNING — unverifiable URL (server refused us,
      server error, TLS failure, timeout, connection error, decoding
      error, oversized response); URL may still be valid, user must
      verify manually
    - content extraction failed: WARNING — URL alive but content not extracted
    """
    findings: list[Finding] = []

    for key, meta in sorted(all_metadata.items()):
        # Skip manually overridden references
        if meta.manual_override and meta.manual_override.get("skip_exists"):
            continue

        if meta.api_match == "not_found":
            findings.append(
                Finding(
                    level="error",
                    rule_id="reference-exists",
                    message=f"{key}: Not found in CrossRef, OpenAlex, Semantic Scholar, Open Library, or Library of Congress",
                    context=_searched_ids_context(meta),
                )
            )
            continue

        # Check web resource issues stored in the issues list
        for issue in meta.issues:
            issue_lower = issue.lower()
            if "dead url" in issue_lower:
                findings.append(
                    Finding(
                        level="error",
                        rule_id="reference-exists",
                        message=f"{key}: {issue}",
                        context=f"Source: {meta.api_source}" if meta.api_source else "",
                    )
                )
            elif "blocked by " in issue_lower:
                findings.append(
                    Finding(
                        level="warning",
                        rule_id="reference-exists",
                        message=f"{key}: {issue}",
                        context=f"Source: {meta.api_source}" if meta.api_source else "",
                    )
                )
            elif "content extraction failed" in issue_lower:
                findings.append(
                    Finding(
                        level="warning",
                        rule_id="reference-exists",
                        message=f"{key}: {issue}",
                        context=f"Source: {meta.api_source}" if meta.api_source else "",
                    )
                )

    return findings


@check(
    id="reference-exists",
    category="reference-db",
    severity="error",
    description="Reference not found in CrossRef, OpenAlex, Semantic Scholar, Open Library, or Library of Congress.",
)
def check_reference_exists(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Load stored metadata and check existence status.

    No API calls — uses references/metadata/{key}.json persisted by verify.
    """
    from sciwrite_lint.references.workspace_db import get_db, query_refs_by_match

    refs_dir = config.effective_references_dir()

    if not refs_dir.exists():
        return []

    # Only load refs that could produce findings (not_found / web_dead /
    # web_blocked / web_verified-with-extract-issue).
    with get_db(refs_dir) as conn:
        candidates = query_refs_by_match(conn, "not_found")
        candidates.update(query_refs_by_match(conn, "web_dead"))
        candidates.update(query_refs_by_match(conn, "web_blocked"))
        # Web-verified refs may have content extraction issues
        candidates.update(query_refs_by_match(conn, "web_verified"))
    if not candidates:
        return []

    return check_reference_exists_from_metadata(candidates)
