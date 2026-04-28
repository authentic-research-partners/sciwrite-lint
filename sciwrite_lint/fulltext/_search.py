"""Candidate ranking for title-search OA sources.

Each title-search adapter in this package (``nber``, ``ideas``, ``hal``,
``eric``, ``nasa_ads``, ``osf``) returns a list of :class:`SearchCandidate`
— ``(url, title, authors, year)`` tuples with whatever metadata the
source exposes. :func:`rank_candidates` picks the best candidate against
a :class:`BibEvidence`, or returns ``None`` if nothing passes.

This runs *before* download. A separate post-download gate lives in
:mod:`sciwrite_lint.fulltext._validation`. Both layers use the same
signals (title similarity, surname overlap, year), but the ranker saves
the bandwidth of downloading a PDF we would otherwise reject after the
fact.

Design for adapter authors
--------------------------

- Populate as many :class:`SearchCandidate` fields as the source cheaply
  exposes. More fields give the ranker more to work with; missing fields
  are handled gracefully (treated as "no signal," not "negative signal").
- A candidate with no ``title`` and no ``authors`` — e.g. an IDEAS
  landing URL before the per-landing HTML is fetched — gets a neutral
  baseline score that lets it through. The downstream validator is the
  safety net in that case.
- When the source supports server-side author filtering (HAL, NASA ADS),
  using it *and* returning candidates for ranking is defence in depth —
  the filter narrows to likely-matches, the ranker confirms.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sciwrite_lint.fulltext._validation import (
    BibEvidence,
    _extract_surname,
    _normalise_for_match,
)

# Below this title similarity, the candidate is a confident mismatch and
# we reject outright — no point downloading. Matches the lowest rung of
# the post-download validator's decision table (see _validation._decide)
# so the two layers agree on what counts as "probably unrelated."
_TITLE_MISMATCH_THRESHOLD = 0.4

# Minimum combined score to accept a candidate. Tuned so:
# - A candidate with *no* metadata (URL only) scores 0.5 — exactly at
#   threshold, lets adapters like IDEAS through for the validator to
#   adjudicate.
# - A candidate with only a weak title signal (sim below threshold) or
#   an author list that doesn't overlap with the bib is hard-rejected
#   and never reaches the download step.
_MIN_CANDIDATE_SCORE = 0.5


class SearchCandidate(BaseModel):
    """One result from a title-search source.

    Only ``url`` is required. Other fields are populated on a best-effort
    basis; the ranker treats missing fields as "no signal," not as
    "negative signal." Adapters should populate everything the source
    exposes without extra fetches — a richer candidate means better
    ranking.
    """

    url: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None


def _first_string(field: object) -> str:
    """Return the first non-empty string from a scalar-or-list field.

    Solr and similar document stores often return multi-valued fields as
    lists even when only one value is present, and sometimes as bare
    scalars when the schema is inconsistent. Adapters feed the raw value
    through this helper to get a canonical ``str`` out (empty when the
    field is missing, None, or contains no string entries).
    """
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        for item in field:
            if isinstance(item, str) and item:
                return item
    return ""


def _string_list(field: object) -> list[str]:
    """Return every string entry from a list field, or ``[field]`` if the
    field is a bare string. Non-strings and missing fields yield ``[]``.
    """
    if isinstance(field, list):
        return [item for item in field if isinstance(item, str)]
    if isinstance(field, str):
        return [field]
    return []


def _coerce_year(field: object) -> int | None:
    """Best-effort year extraction: accepts int, str-of-int, or a list
    whose first valid entry is one of those."""
    if isinstance(field, int):
        return field
    if isinstance(field, str):
        try:
            return int(field.strip())
        except ValueError:
            return None
    if isinstance(field, list):
        for item in field:
            result = _coerce_year(item)
            if result is not None:
                return result
    return None


def rank_candidates(
    candidates: list[SearchCandidate],
    evidence: BibEvidence,
) -> SearchCandidate | None:
    """Pick the best candidate matching ``evidence``, or ``None``.

    Scores each candidate on title similarity + surname overlap + year
    agreement, then returns the highest-scoring one if it is at or above
    :data:`_MIN_CANDIDATE_SCORE`. Returns ``None`` when the list is empty
    or no candidate is a plausible match.

    Ties break toward the first candidate in the input list, preserving
    the source's own ranking when scores are equal.
    """
    if not candidates:
        return None
    best_candidate: SearchCandidate | None = None
    best_score: float = -1.0
    for candidate in candidates:
        score = _score(candidate, evidence)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    if best_candidate is None or best_score < _MIN_CANDIDATE_SCORE:
        return None
    return best_candidate


def _score(candidate: SearchCandidate, evidence: BibEvidence) -> float:
    """Ranking score: ``0.0`` means hard reject, otherwise higher is better.

    Two hard-reject conditions short-circuit the positive signals:

    1. Both sides have a title and similarity is below
       :data:`_TITLE_MISMATCH_THRESHOLD` — the candidate is confidently
       about something else.
    2. Both sides have authors and no bib surname appears in the
       candidate's author list — the candidate is confidently by someone
       else.

    Otherwise the score is the sum of a title component, an author bonus,
    and a year bonus. Signals that can't be computed (either side lacks
    the field) contribute a neutral default rather than penalising, so a
    source that doesn't expose per-candidate metadata isn't unfairly
    ranked out.

    The score is *not* capped — a perfect match with matching authors and
    year scores higher than a perfect match alone, so the extra signals
    break ties between otherwise-equivalent candidates.
    """
    title_sim = _compute_title_sim(candidate, evidence)
    if title_sim is not None and title_sim < _TITLE_MISMATCH_THRESHOLD:
        return 0.0

    # Surname overlap computed once, driving both the hard reject and
    # the author bonus. ``None`` means "can't compute" (either side has
    # no author list) — distinct from "computed and zero" (hard reject).
    author_overlap: int | None = None
    if evidence.authors and candidate.authors:
        author_overlap = _surname_overlap(evidence.authors, candidate.authors)
        if author_overlap == 0:
            return 0.0

    title_score = 0.5 if title_sim is None else 0.2 + 0.8 * title_sim
    author_bonus = 0.2 if author_overlap else 0.0
    year_bonus = (
        0.05 if evidence.year is not None and candidate.year == evidence.year else 0.0
    )
    return title_score + author_bonus + year_bonus


def _compute_title_sim(
    candidate: SearchCandidate, evidence: BibEvidence
) -> float | None:
    """Title similarity, or ``None`` when either side lacks a title."""
    if not (candidate.title and evidence.title):
        return None
    # Import locally to avoid circular imports via pdf.pdf_download.
    from sciwrite_lint.pdf.pdf_download import _title_similarity

    return _title_similarity(evidence.title, candidate.title)


def _surname_overlap(bib_authors: list[str], candidate_authors: list[str]) -> int:
    """Count of bib surnames that appear in the candidate's author list.

    Uses the same NFKD-folded lowercase normalisation as the downstream
    validator so "Müller" / "Mueller" and "van Fraassen" variants match
    consistently across the two layers.
    """
    if not bib_authors or not candidate_authors:
        return 0
    bib_surnames = {_normalise_for_match(_extract_surname(a)) for a in bib_authors}
    bib_surnames.discard("")
    if not bib_surnames:
        return 0
    candidate_surnames = {
        _normalise_for_match(_extract_surname(a)) for a in candidate_authors
    }
    candidate_surnames.discard("")
    return len(bib_surnames & candidate_surnames)
