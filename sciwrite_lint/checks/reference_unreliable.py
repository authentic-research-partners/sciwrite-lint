"""Check: reference-unreliable — per-reference aggregate reliability.

Aggregates signals from whatever checks have run (metadata, claim verdicts,
chain verification) into a single per-reference reliability score. Fires a
warning when convergent evidence says a reference is unreliable.

Two entry points:
- metadata_to_unreliable_findings: fast path (after verify), metadata only
- claims_to_unreliable_findings: deep path (after verify-claims), all signals
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sciwrite_lint.models import CheckMeta, CitationMetadata, Finding

if TYPE_CHECKING:
    from sciwrite_lint.scoring.chain import RefBibCheck

# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

UNRELIABLE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Signal descriptions (for human-readable context)
# ---------------------------------------------------------------------------


def _describe_metadata(meta: CitationMetadata) -> list[str]:
    """Build a list of human-readable signal descriptions from metadata."""
    from sciwrite_lint.references.metadata import compute_tier

    parts: list[str] = []
    if meta.canonical.get("retracted"):
        parts.append("retracted")
        return parts  # retraction is definitive

    tier = meta.access.get("tier") or compute_tier(meta)
    if tier == "T3":
        parts.append("not found in APIs")
    if meta.api_match == "mismatch":
        parts.append("API metadata mismatch")
    if meta.mismatches:
        parts.append(f"{len(meta.mismatches)} field mismatch(es)")
    return parts


def _describe_claims(claims: list[dict[str, Any]]) -> list[str]:
    """Build signal descriptions from claim verification results."""
    parts: list[str] = []
    active = [c for c in claims if not c.get("dismissed")]
    not_sup = sum(1 for c in active if c.get("verdict") == "NOT_SUPPORTED")
    partial = sum(1 for c in active if c.get("verdict") == "PARTIALLY_SUPPORTS")
    if not_sup:
        parts.append(f"{not_sup} claim(s) not supported")
    if partial:
        parts.append(f"{partial} claim(s) partially supported")
    return parts


# ---------------------------------------------------------------------------
# Scoring (reuses scilint_score constants)
# ---------------------------------------------------------------------------


def _score_from_metadata(meta: CitationMetadata) -> float:
    """Compute reliability score from metadata signals only."""
    from sciwrite_lint.scoring.scilint_score import _score_metadata

    return _score_metadata(meta)


def _score_from_claims(claims: list[dict[str, Any]]) -> float:
    """Compute verification score from claim verdicts."""
    from sciwrite_lint.scoring.scilint_score import VERDICT_SCORES

    active = [c for c in claims if not c.get("dismissed")]
    if not active:
        return 1.0  # no claims to verify — no negative signal
    total = sum(
        VERDICT_SCORES.get(c.get("verdict", "CANNOT_DETERMINE"), 0.25) for c in active
    )
    return total / len(active)


def _describe_bib_checks(bib_checks: list[RefBibCheck], key: str) -> list[str]:
    """Build signal descriptions from standalone bibliography checks."""
    parts: list[str] = []
    for rb in bib_checks:
        if rb.key == key and rb.total_entries > 0:
            if rb.not_found > 0:
                parts.append(
                    f"{rb.not_found}/{rb.total_entries} bibliography entries "
                    f"not found in APIs ({rb.hallucination_rate:.0%})"
                )
            if rb.retracted > 0:
                parts.append(f"{rb.retracted} retracted entries in bibliography")
            if rb.metadata_mismatches > 0:
                parts.append(
                    f"{rb.metadata_mismatches} bibliography metadata mismatches "
                    f"({rb.mismatch_rate:.0%} of found)"
                )
    return parts


def _score_from_bib_checks(bib_checks: list[RefBibCheck], key: str) -> float:
    """Compute score from standalone bibliography checks."""
    for rb in bib_checks:
        if rb.key == key and rb.total_entries > 0:
            existence_score = rb.found / rb.total_entries
            mismatch_penalty = min(rb.metadata_mismatches * 0.05, 0.3)
            retraction_penalty = min(rb.retracted * 0.15, 0.3)
            return max(existence_score - mismatch_penalty - retraction_penalty, 0.0)
    return 1.0


# ---------------------------------------------------------------------------
# Finding generators
# ---------------------------------------------------------------------------


def _make_finding(
    key: str,
    reliability: float,
    signals: list[str],
    tex_path: Path,
    line: int | None = None,
) -> Finding:
    return Finding(
        level="warning",
        rule_id="reference-unreliable",
        message=f"{key}: Reference has low reliability ({reliability:.2f})",
        file=tex_path.name,
        line=line,
        context=", ".join(signals) if signals else "",
    )


def metadata_to_unreliable_findings(
    metadata_map: dict[str, CitationMetadata],
    tex_path: Path,
    bib_checks: list[RefBibCheck] | None = None,
) -> list[Finding]:
    """Fast path: flag unreliable references from metadata signals only.

    Call after ``verify`` when metadata is available but claim verification
    has not run. Bibliography checks from the pipeline are included when
    available.
    """
    findings: list[Finding] = []
    for key, meta in metadata_map.items():
        meta_score = _score_from_metadata(meta)
        bib_score = _score_from_bib_checks(bib_checks, key) if bib_checks else 1.0
        reliability = meta_score * bib_score
        if reliability < UNRELIABLE_THRESHOLD:
            signals = _describe_metadata(meta)
            if bib_checks:
                signals.extend(_describe_bib_checks(bib_checks, key))
            findings.append(_make_finding(key, reliability, signals, tex_path))
    return findings


def claims_to_unreliable_findings(
    claim_results: list[dict[str, Any]],
    tex_path: Path,
    metadata_map: dict[str, CitationMetadata] | None = None,
    bib_checks: list[RefBibCheck] | None = None,
) -> list[Finding]:
    """Deep path: flag unreliable references using all available signals.

    Call after ``verify-claims``. Combines metadata, claim verdicts, and
    bibliography checks (existence + metadata + retraction from pipeline).
    """
    # Group claims by reference key
    by_key: dict[str, list[dict[str, Any]]] = {}
    for c in claim_results:
        k = c.get("key", "")
        if k:
            by_key.setdefault(k, []).append(c)

    findings: list[Finding] = []
    seen_keys = set(by_key.keys())

    for key, claims in by_key.items():
        # Start with metadata if available
        if metadata_map and key in metadata_map:
            meta_score = _score_from_metadata(metadata_map[key])
        else:
            meta_score = 1.0  # no metadata → no negative signal

        claim_score = _score_from_claims(claims)

        # Bibliography checks (from pipeline Stage 4.6)
        bib_score = _score_from_bib_checks(bib_checks, key) if bib_checks else 1.0

        reliability = meta_score * claim_score * bib_score

        if reliability < UNRELIABLE_THRESHOLD:
            signals: list[str] = []
            if metadata_map and key in metadata_map:
                signals.extend(_describe_metadata(metadata_map[key]))
            signals.extend(_describe_claims(claims))
            if bib_checks:
                signals.extend(_describe_bib_checks(bib_checks, key))

            # Use line from first claim for location
            line = claims[0].get("line") if claims else None
            findings.append(_make_finding(key, reliability, signals, tex_path, line))

    # Also check metadata-only keys not in claims (e.g., T3 refs with no T1 source)
    if metadata_map:
        for key, meta in metadata_map.items():
            if key in seen_keys:
                continue
            meta_score = _score_from_metadata(meta)
            bib_score = _score_from_bib_checks(bib_checks, key) if bib_checks else 1.0
            reliability = meta_score * bib_score
            if reliability < UNRELIABLE_THRESHOLD:
                signals = _describe_metadata(meta)
                if bib_checks:
                    signals.extend(_describe_bib_checks(bib_checks, key))
                findings.append(_make_finding(key, reliability, signals, tex_path))

    return findings


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Metadata for `sciwrite-lint checks` listing.
# This check runs as a pipeline stage (not from the check registry).
REFERENCE_UNRELIABLE_META = CheckMeta(
    id="reference-unreliable",
    severity="warning",
    category="reference-db",
    description="Reference has low aggregate reliability across multiple signals.",
)
