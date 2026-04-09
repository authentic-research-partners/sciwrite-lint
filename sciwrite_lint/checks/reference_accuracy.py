"""Check: reference-accuracy — manuscript metadata vs canonical API records.

Runs after reference-exists confirms a reference is real. Compares title,
authors, year, and venue in the bibitem against stored canonical data from
CrossRef / OpenAlex / Semantic Scholar.

Uses metadata already persisted in references/metadata/{key}.json — no new
API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sciwrite_lint.checks.registry import check
from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import CitationMetadata, Finding


# ---------------------------------------------------------------------------
# Field-level comparison helpers
# ---------------------------------------------------------------------------


def _check_title(
    key: str, bibitem: dict, canonical: dict, api_source: str = ""
) -> Finding | None:
    """Title: fuzzy match, ERROR if below threshold (possible fabricated title)."""
    from sciwrite_lint.references.matching import TITLE_THRESHOLD, title_similarity

    tex_title = bibitem.get("title", "")
    api_title = canonical.get("title", "")
    if not tex_title or not api_title:
        return None

    sim = title_similarity(tex_title, api_title)
    if sim >= TITLE_THRESHOLD:
        return None

    return Finding(
        level="error",
        rule_id="reference-accuracy",
        message=(
            f"{key}: Title mismatch (similarity={sim:.2f}): "
            f"tex='{tex_title[:80]}', canonical='{api_title[:80]}'"
        ),
        context=f"Source: {api_source}" if api_source else "",
    )


def _check_authors(
    key: str, bibitem: dict, canonical: dict, api_source: str = ""
) -> Finding | None:
    """Authors: set comparison on normalized last names."""
    from sciwrite_lint.references.matching import AUTHOR_THRESHOLD, author_similarity

    tex_authors = bibitem.get("authors", [])
    api_authors = canonical.get("authors", [])
    if not tex_authors or not api_authors:
        return None

    # Handle case where tex_authors is a single joined string
    if len(tex_authors) == 1 and "," in tex_authors[0]:
        tex_authors = [a.strip() for a in tex_authors[0].split(",") if a.strip()]

    sim = author_similarity(tex_authors, api_authors)
    if sim >= AUTHOR_THRESHOLD:
        return None

    tex_str = ", ".join(str(a) for a in tex_authors[:3])
    api_str = ", ".join(str(a) for a in api_authors[:3])
    suffix = "..." if len(api_authors) > 3 else ""
    return Finding(
        level="warning",
        rule_id="reference-accuracy",
        message=(
            f"{key}: Author mismatch (similarity={sim:.2f}): "
            f"tex='{tex_str}', canonical='{api_str}{suffix}'"
        ),
        context=f"Source: {api_source}" if api_source else "",
    )


def _check_year(
    key: str, bibitem: dict, canonical: dict, api_source: str = ""
) -> Finding | None:
    """Year: exact match, flag if off by >1 year."""
    tex_year = bibitem.get("year", "")
    api_year = canonical.get("year")
    if not tex_year or not api_year:
        return None

    try:
        ty = int(tex_year)
        ay = int(api_year)
    except (ValueError, TypeError):
        return None

    diff = abs(ty - ay)
    if diff <= 1:
        return None

    return Finding(
        level="warning",
        rule_id="reference-accuracy",
        message=f"{key}: Year mismatch: tex={ty}, canonical={ay} (off by {diff})",
        context=f"Source: {api_source}" if api_source else "",
    )


def _check_venue(
    key: str, bibitem: dict, canonical: dict, api_source: str = ""
) -> tuple[Finding | None, tuple[str, str] | None]:
    """Venue: fuzzy match on normalized name.

    Returns (finding_or_none, venue_pair_needing_llm_confirmation_or_none).
    When fuzzy score is below threshold, returns the finding AND the venue
    pair so the caller can optionally confirm via vLLM before emitting.
    """
    from sciwrite_lint.references.matching import VENUE_THRESHOLD, venue_similarity

    tex_venue = bibitem.get("venue", "")
    api_venue = canonical.get("venue", "")
    if not tex_venue or not api_venue:
        return None, None

    sim = venue_similarity(tex_venue, api_venue)
    if sim >= VENUE_THRESHOLD:
        return None, None

    finding = Finding(
        level="warning",
        rule_id="reference-accuracy",
        message=(
            f"{key}: Venue mismatch (similarity={sim:.2f}): "
            f"tex='{tex_venue[:60]}', canonical='{api_venue[:60]}'"
        ),
        context=f"Source: {api_source}" if api_source else "",
    )
    return finding, (tex_venue, api_venue)


# ---------------------------------------------------------------------------
# Main check: run on all stored metadata
# ---------------------------------------------------------------------------


def check_reference_accuracy_from_metadata(
    all_metadata: dict[str, CitationMetadata],
) -> list[Finding]:
    """Compare bibitem vs canonical for all verified references.

    Only checks references where api_match is "verified" or "mismatch"
    (i.e., reference-exists already passed).

    Synchronous version — venue mismatches use fuzzy matching only.
    For vLLM-confirmed venue matching, use the async variant.
    """
    findings: list[Finding] = []

    for key, meta in sorted(all_metadata.items()):
        # Skip unverified, not-found, and web resources
        if meta.api_match not in ("verified", "mismatch"):
            continue

        # Skip manually overridden references
        if meta.manual_override and meta.manual_override.get("skip_accuracy"):
            continue

        canonical = meta.canonical
        bibitem = meta.bibitem
        if not canonical or not bibitem:
            continue

        src = meta.api_source
        for checker in (_check_title, _check_authors, _check_year):
            finding = checker(key, bibitem, canonical, api_source=src)
            if finding:
                findings.append(finding)

        venue_finding, _venue_pair = _check_venue(
            key, bibitem, canonical, api_source=src
        )
        if venue_finding:
            findings.append(venue_finding)

    return findings


async def check_reference_accuracy_from_metadata_async(
    all_metadata: dict[str, CitationMetadata],
    config: Any | None = None,
) -> list[Finding]:
    """Compare bibitem vs canonical, with vLLM venue confirmation.

    Same as sync version, but venue mismatches below the fuzzy threshold
    are sent to vLLM for confirmation before flagging. If vLLM says the
    venues match, the finding is suppressed.
    """
    from sciwrite_lint.references.matching import venue_match_llm

    findings: list[Finding] = []
    venue_candidates: list[tuple[Finding, str, str]] = []

    for key, meta in sorted(all_metadata.items()):
        if meta.api_match not in ("verified", "mismatch"):
            continue
        if meta.manual_override and meta.manual_override.get("skip_accuracy"):
            continue

        canonical = meta.canonical
        bibitem = meta.bibitem
        if not canonical or not bibitem:
            continue

        src = meta.api_source
        for checker in (_check_title, _check_authors, _check_year):
            finding = checker(key, bibitem, canonical, api_source=src)
            if finding:
                findings.append(finding)

        venue_finding, venue_pair = _check_venue(
            key, bibitem, canonical, api_source=src
        )
        if venue_finding and venue_pair:
            venue_candidates.append((venue_finding, venue_pair[0], venue_pair[1]))

    # Confirm venue mismatches via vLLM (concurrent, bounded)
    if venue_candidates:
        import asyncio

        sem = asyncio.Semaphore(50)

        async def _confirm(finding: Finding, tex_v: str, api_v: str) -> Finding | None:
            async with sem:
                same = await venue_match_llm(tex_v, api_v, config=config)
            if same is True:
                return None  # vLLM says same venue — suppress
            return finding

        results = await asyncio.gather(
            *[_confirm(f, tv, av) for f, tv, av in venue_candidates]
        )
        for r in results:
            if r is not None:
                findings.append(r)

    return findings


@check(
    id="reference-accuracy",
    category="reference-db",
    severity="warning",
    description="Manuscript metadata vs canonical API records (title, authors, year, venue).",
)
def check_reference_accuracy(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Load stored metadata and compare bibitem fields against canonical records.

    No API calls — uses references/metadata/{key}.json persisted by verify.
    """
    from sciwrite_lint.references.workspace_db import get_db, query_refs_by_match

    refs_dir = config.effective_references_dir()

    if not refs_dir.exists():
        return []

    # Only load refs that have been verified (the only ones with accuracy data)
    with get_db(refs_dir) as conn:
        candidates = query_refs_by_match(conn, "verified")
        candidates.update(query_refs_by_match(conn, "mismatch"))
    if not candidates:
        return []

    return check_reference_accuracy_from_metadata(candidates)
