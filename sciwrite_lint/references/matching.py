"""Fuzzy comparison for citation verification.

Compares parsed bibitem data against API-returned canonical data using
calibrated similarity thresholds inspired by bibtex-updater's algorithms.
"""

from __future__ import annotations

import re
from typing import Any

from anyascii import anyascii
from rapidfuzz import fuzz as _rf_fuzz

from sciwrite_lint.models import Citation
from sciwrite_lint.schemas import VenueMatch, vllm_schema_unbounded

# Thresholds
TITLE_THRESHOLD = 0.80  # rapidfuzz ratio
AUTHOR_THRESHOLD = 0.60  # combined Jaccard + sequence
YEAR_TOLERANCE = 1  # ±1 year


def _fuzzy_ratio(a: str, b: str) -> float:
    return _rf_fuzz.ratio(a, b) / 100.0


def _normalize(text: str) -> str:
    """Normalize text for comparison: transliterate, lowercase, strip LaTeX, collapse whitespace."""
    text = anyascii(text)  # Cyrillic→Latin, Unicode hyphens→ASCII, etc.
    text = text.lower()
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)  # \cmd{arg} -> arg
    text = re.sub(r"[{}~\\]", " ", text)
    text = re.sub(r"['\"`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_similarity(tex_title: str, api_title: str) -> float:
    """Fuzzy title similarity score (0.0 to 1.0).

    Handles subtitle variations: books often have subtitles in one source
    but not the other (e.g. "Mind in Society" vs "Mind in Society: The
    Development of Higher Psychological Processes"). Tries all combinations
    of full title and main title (before colon) and returns the best score.
    """
    a = _normalize(tex_title)
    b = _normalize(api_title)
    if not a or not b:
        return 0.0
    a_variants = [a]
    if ":" in a:
        a_variants.append(a.split(":")[0].strip())
    b_variants = [b]
    if ":" in b:
        b_variants.append(b.split(":")[0].strip())
    return max(_fuzzy_ratio(av, bv) for av in a_variants for bv in b_variants)


def author_similarity(tex_authors: list[str], api_authors: list[str]) -> float:
    """Pairwise fuzzy author similarity with name variant generation.

    Returns 0.0 to 1.0. Handles name order, initials, transliteration,
    middle name dropping, and East Asian name conventions.

    Delegates to ``_author_overlap`` in api.py (single implementation
    shared with the candidate selection engine).
    """
    from sciwrite_lint.api import _author_overlap

    return _author_overlap(tex_authors, api_authors)


def _normalize_venue(venue: str) -> str:
    """Normalize venue name: lowercase, strip boilerplate prefixes/suffixes."""
    v = _normalize(venue)
    # Strip common prefixes
    for prefix in ("proceedings of the ", "proceedings of ", "proc. ", "in "):
        if v.startswith(prefix):
            v = v[len(prefix) :]
    # Strip trailing year patterns like ", 2020" or " 2020"
    v = re.sub(r"[,\s]+\d{4}$", "", v)
    return v


def _fuzzy_partial_ratio(a: str, b: str) -> float:
    return _rf_fuzz.partial_ratio(a, b) / 100.0


def venue_similarity(tex_venue: str, api_venue: str) -> float:
    """Fuzzy venue similarity score (0.0 to 1.0).

    Uses partial_ratio for better handling of abbreviations and contained
    strings (e.g. "NIPS" inside "NeurIPS", "Softw." vs "Software").
    No hardcoded alias table — relies on fuzzy matching only.
    Returns 1.0 if either venue is empty (can't compare).
    """
    a = _normalize_venue(tex_venue)
    b = _normalize_venue(api_venue)
    if not a or not b:
        return 1.0  # can't compare, not a mismatch
    if a == b:
        return 1.0
    return _fuzzy_partial_ratio(a, b)


VENUE_THRESHOLD = (
    0.65  # venue partial_ratio (slightly above 0.60 to avoid false positives)
)


_VENUE_CONFIRM_SCHEMA = vllm_schema_unbounded(VenueMatch)


async def venue_match_llm(
    tex_venue: str,
    api_venue: str,
    config: Any | None = None,
) -> bool | None:
    """Ask vLLM whether two venue names refer to the same publication venue.

    Returns True (same), False (different), or None (vLLM unavailable).
    Only called when fuzzy matching is ambiguous (score below threshold).
    """
    from sciwrite_lint.llm_utils import llm_query

    result = await llm_query(
        system=(
            "You compare academic venue names. Reply with JSON only. "
            "same_venue=true if both names refer to the same journal or conference, "
            "even if one is an abbreviation or acronym of the other."
        ),
        user=f'Venue A: "{tex_venue}"\nVenue B: "{api_venue}"\nAre these the same venue?',
        schema=_VENUE_CONFIRM_SCHEMA,
        schema_name="venue_match",
        config=config,
        max_tokens=32,
        thinking="off",
    )
    if result is None:
        return None
    return result.get("same_venue")


def year_match(tex_year: str, api_year: Any) -> bool:
    """Check if years match within tolerance."""
    if not tex_year or not api_year:
        return True  # can't compare, not a mismatch
    try:
        ty = int(tex_year)
        ay = int(api_year)
        return abs(ty - ay) <= YEAR_TOLERANCE
    except (ValueError, TypeError):
        return True


def compare_citation_detailed(
    citation: Citation,
    api_data: dict[str, Any],
) -> list[str]:
    """Compare parsed citation against API data with fuzzy matching.

    Returns list of issue strings.
    """
    if api_data.get("error"):
        return [f"API error: {api_data['error']}"]

    issues = []

    # Year check
    api_year = api_data.get("year")
    if citation.year and api_year and not year_match(citation.year, api_year):
        issues.append(f"Year mismatch: tex={citation.year}, API={api_year}")

    # Title check
    api_title = api_data.get("title", "")
    if api_title and citation.title:
        sim = title_similarity(citation.title, api_title)
        if sim < TITLE_THRESHOLD:
            issues.append(
                f"Title mismatch (similarity={sim:.2f}): "
                f"tex='{citation.title[:60]}', API='{api_title[:60]}'"
            )

    # Author check
    api_authors = api_data.get("authors", [])
    if api_authors and citation.authors:
        sim = author_similarity(citation.authors, api_authors)
        if sim < AUTHOR_THRESHOLD:
            tex_str = ", ".join(citation.authors[:3])
            api_str = ", ".join(str(a) for a in api_authors[:3])
            issues.append(
                f"Author mismatch (similarity={sim:.2f}): "
                f"tex='{tex_str}', API='{api_str}'"
            )

    # Venue check
    api_venue = api_data.get("venue", "")
    if api_venue and citation.venue:
        sim = venue_similarity(citation.venue, api_venue)
        if sim < VENUE_THRESHOLD:
            issues.append(
                f"Venue mismatch (similarity={sim:.2f}): "
                f"tex='{citation.venue[:60]}', API='{api_venue[:60]}'"
            )

    # Wrong identifier in bib entry (ID lookup returned different paper)
    id_mismatch = api_data.get("id_mismatch")
    if id_mismatch:
        wrong_doi = id_mismatch.get("bib_doi", "")
        wrong_arxiv = id_mismatch.get("bib_arxiv_id", "")
        wrong_pmid = id_mismatch.get("bib_pmid", "")
        wrong_id = wrong_doi or wrong_arxiv or wrong_pmid
        issues.append(
            f"Wrong identifier in bib entry: '{wrong_id}' resolves to "
            f"'{id_mismatch.get('wrong_title', '')[:60]}', "
            f"not the paper found by title search"
        )

    # No identifier in bib entry
    has_any_id = (
        citation.doi
        or citation.isbn
        or citation.pmid
        or citation.arxiv_id
        or citation.pmc_id
        or citation.lccn
    )
    if not has_any_id:
        if api_data.get("doi"):
            issues.append(
                f"No identifier in bib entry; DOI available: {api_data['doi']}"
            )
        else:
            issues.append(
                "No identifier in bib entry (no DOI, ISBN, PMID, or arXiv ID)"
            )

    # Open access PDF available but no local file
    if api_data.get("pdf_url") and citation.local_status == "none":
        issues.append(f"Open access PDF available: {api_data['pdf_url']}")

    # Retraction check
    if api_data.get("retracted"):
        issues.append("RETRACTED: This paper has been retracted")

    return issues
