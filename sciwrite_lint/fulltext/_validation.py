"""Multi-signal PDF-match validation for fetched OA content.

Runs after every direct PDF download to decide whether the bytes at
``pdf_path`` actually represent the reference identified by ``evidence``.
There is no silent-accept path: if no positive signal fires the PDF is
rejected and the caller moves to the next source in the waterfall.

Hard rejects (bypass all positive signals):

- ERIC "DOCUMENT RESUME" bibliographic-record templates
- CORE.ac.uk metadata landing pages
- Article-type entries whose extractable body is shorter than 1.5k chars

Positive signals (need at least one per the ``_decide`` table):

- ``doi_match``  — bib DOI appears verbatim in first two pages
- ``title_sim`` — GROBID header title vs bib title (``_title_similarity``)
- ``surname_match`` — count of bib-author surnames found in first two pages
- ``year_match`` — bib year appears in first two pages

Entry-type carve-out for books / techreports / manuals / inbook /
inproceedings / incollection: skip the body-length hard reject, and when
the bib has no author list (corporate authors like NASEM / NRC) accept on
``title_sim >= 0.75`` alone.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


_CORPORATE_ENTRY_TYPES = frozenset(
    {"book", "techreport", "manual", "inbook", "inproceedings", "incollection"}
)

# Minimum extractable body text for article-type entries. PubMed abstract
# landing pages and ERIC templates sit well under this (16 KB of PDF
# chrome may yield <500 chars of actual text).
_ARTICLE_MIN_BODY_CHARS = 1500

# First-page text window we inspect for template patterns. Kept small
# because legitimate papers never mention ERIC / CORE boilerplate in the
# abstract, so a narrow window avoids false positives.
_TEMPLATE_WINDOW_CHARS = 3000

# First-two-pages text window for surname / DOI / year checks.
_EVIDENCE_WINDOW_CHARS = 8000

# Compiled once — hot path.
_ERIC_TEMPLATE_RE = re.compile(r"\bDOCUMENT\s+RESUME\b", re.IGNORECASE)
_CORE_TEMPLATE_RE = re.compile(
    r"View\s+metadata,\s+citation\s+and\s+similar\s+papers\s+at\s+core\.ac\.uk",
    re.IGNORECASE,
)
_DOI_NORMALIZE_RE = re.compile(r"\s+")


class BibEvidence(BaseModel):
    """Signals a bib entry provides for validating a fetched PDF.

    Replaces the loose ``expected_title`` / ``expected_authors`` kwargs
    that the fetch stack used before multi-signal validation existed.
    """

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    doi: str = ""
    year: int | None = None
    entry_type: str = "article"


class ValidationResult(BaseModel):
    """Outcome of :func:`validate_pdf_match`.

    ``signals`` holds per-check diagnostic values; callers may surface
    them in logs or debug output but must not branch on their shape.
    """

    accepted: bool
    confidence: float = 0.0
    signals: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


async def validate_pdf_match(pdf_path: Path, evidence: BibEvidence) -> ValidationResult:
    """Decide whether ``pdf_path`` matches the bib entry in ``evidence``.

    Runs hard-reject template / length checks first, then collects
    positive signals (title, DOI, authors, year) and applies the decision
    table in :func:`_decide`.

    Empty-evidence contract: when ``evidence`` carries no positive fields
    (no title, authors, DOI, or year — the public API case where only an
    identifier was supplied), the caller has told us they have nothing to
    match against. We still run the template-pattern hard reject (ERIC /
    CORE landing pages are always bugs) but skip the body-length floor
    and the positive-signal gate. Internal pipeline callers always
    populate evidence from the bib and so never hit this path.
    """
    text = _extract_first_pages_text(pdf_path)

    template_hit, template_name = _match_template_patterns(text)
    if template_hit:
        return ValidationResult(
            accepted=False,
            confidence=0.0,
            signals={"template": template_name},
            reason=f"template page ({template_name})",
        )

    if _is_empty_evidence(evidence):
        return ValidationResult(
            accepted=True,
            confidence=0.0,
            signals={"body_chars": len(text)},
            reason="no evidence supplied",
        )

    is_corporate_form = evidence.entry_type in _CORPORATE_ENTRY_TYPES
    body_len = len(text)
    if not is_corporate_form and body_len < _ARTICLE_MIN_BODY_CHARS:
        return ValidationResult(
            accepted=False,
            confidence=0.0,
            signals={"body_chars": body_len},
            reason=(
                f"body too short ({body_len} chars < "
                f"{_ARTICLE_MIN_BODY_CHARS}) for {evidence.entry_type}"
            ),
        )

    grobid_title = await _extract_header_title(pdf_path)
    title_sim = _compute_title_sim(evidence.title, grobid_title)
    surname_match = _count_surname_matches(text, evidence.authors)
    doi_match = _doi_in_text(text, evidence.doi)
    year_match = _year_in_text(text, evidence.year)

    signals = {
        "grobid_title": grobid_title,
        "title_sim": round(title_sim, 3),
        "surname_match": surname_match,
        "doi_match": doi_match,
        "year_match": year_match,
        "body_chars": body_len,
        "entry_type": evidence.entry_type,
    }

    accepted, reason, confidence = _decide(
        evidence=evidence,
        grobid_title=grobid_title,
        title_sim=title_sim,
        surname_match=surname_match,
        doi_match=doi_match,
        year_match=year_match,
    )

    if not accepted:
        logger.warning(
            "PDF validation rejected {}: {} (signals={})",
            pdf_path.name,
            reason,
            signals,
        )

    return ValidationResult(
        accepted=accepted,
        confidence=confidence,
        signals=signals,
        reason=reason,
    )


def _decide(
    *,
    evidence: BibEvidence,
    grobid_title: str,
    title_sim: float,
    surname_match: int,
    doi_match: bool,
    year_match: bool,
) -> tuple[bool, str, float]:
    """Apply the acceptance table. Returns (accepted, reason, confidence)."""
    if doi_match:
        return True, "doi match", 1.0

    has_authors = bool(evidence.authors)
    is_corporate_form = evidence.entry_type in _CORPORATE_ENTRY_TYPES

    if title_sim >= 0.85:
        if not has_authors:
            return True, f"strong title match, no authors ({title_sim:.2f})", 0.85
        if surname_match >= 1:
            return True, f"strong title+surname match ({title_sim:.2f})", 0.9
        # Bib has authors but none appeared in PDF evidence window — a
        # high title similarity alone is unsafe for short generic titles
        # (e.g. "Deep Learning"). Fall through to the weaker rules; they
        # all require surname_match and will reject this case cleanly.

    if is_corporate_form and not has_authors:
        if title_sim >= 0.75:
            return True, f"corporate-author title match ({title_sim:.2f})", 0.75
        return (
            False,
            (
                f"corporate-author title too weak ({title_sim:.2f} < 0.75); "
                f"grobid_title={grobid_title[:50]!r}"
            ),
            title_sim,
        )

    if title_sim >= 0.65 and surname_match >= 1:
        return (
            True,
            f"title+surname match (sim={title_sim:.2f}, surnames={surname_match})",
            0.8,
        )

    if title_sim >= 0.40 and surname_match >= 1 and year_match:
        return (
            True,
            (
                f"title+surname+year match (sim={title_sim:.2f}, "
                f"surnames={surname_match})"
            ),
            0.65,
        )

    if not grobid_title and surname_match >= 1 and year_match:
        return (
            True,
            f"grobid-no-title, surname+year match (surnames={surname_match})",
            0.55,
        )

    return (
        False,
        (
            f"no positive signal (title_sim={title_sim:.2f}, "
            f"surnames={surname_match}, doi={doi_match}, year={year_match}, "
            f"grobid_title={grobid_title[:50]!r})"
        ),
        title_sim,
    )


def _is_empty_evidence(evidence: BibEvidence) -> bool:
    """True when the caller provided no matchable fields (title / authors /
    DOI / year). ``entry_type`` alone does not count as evidence — it's a
    default, not a claim about the reference."""
    return (
        not evidence.title
        and not evidence.authors
        and not evidence.doi
        and evidence.year is None
    )


def _compute_title_sim(bib_title: str, grobid_title: str) -> float:
    """Fuzzy title similarity; returns 0.0 if either side is empty."""
    if not bib_title or not grobid_title:
        return 0.0
    from sciwrite_lint.pdf.pdf_download import _title_similarity

    return _title_similarity(bib_title, grobid_title)


async def _extract_header_title(pdf_path: Path) -> str:
    """Return the GROBID header title or an empty string on failure.

    Catches ``GrobidUnparseableError`` here so the validator can fold the
    no-title path into its decision logic (the ``grobid-no-title`` rule).
    """
    from sciwrite_lint.pdf.grobid import (
        GrobidUnparseableError,
        extract_title_from_header,
    )

    try:
        title = await extract_title_from_header(pdf_path)
    except GrobidUnparseableError as e:
        logger.debug("GROBID header extraction failed for {}: {}", pdf_path.name, e)
        return ""
    return title or ""


def _extract_first_pages_text(pdf_path: Path, n_pages: int = 2) -> str:
    """Extract text from the first ``n_pages`` of a PDF via pypdf.

    Returns an empty string on any pypdf failure — the caller treats
    empty text as "no positive evidence available" which fails the
    article-length hard reject (intentional — unparseable PDFs are
    almost always not what we asked for).
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError, ValueError) as e:
        logger.debug("pypdf could not open {}: {}", pdf_path.name, e)
        return ""

    chunks: list[str] = []
    total = 0
    for page in reader.pages[:n_pages]:
        try:
            chunk = page.extract_text() or ""
        except (PdfReadError, ValueError, AttributeError, KeyError) as e:
            logger.debug("pypdf text-extract failed on {}: {}", pdf_path.name, e)
            continue
        chunks.append(chunk)
        total += len(chunk)
        if total >= _EVIDENCE_WINDOW_CHARS:
            break
    return "\n".join(chunks)[:_EVIDENCE_WINDOW_CHARS]


def _match_template_patterns(text: str) -> tuple[bool, str]:
    """Return (matched, pattern_name) if ``text`` looks like a known template.

    Only the first :data:`_TEMPLATE_WINDOW_CHARS` of ``text`` are inspected —
    the templates we catch put their boilerplate at the top of the document,
    so a narrow window avoids a legitimate paper quoting the phrase deep in
    its body from triggering a false reject.
    """
    window = text[:_TEMPLATE_WINDOW_CHARS]
    if _ERIC_TEMPLATE_RE.search(window):
        return True, "eric_document_resume"
    if _CORE_TEMPLATE_RE.search(window):
        return True, "core_ac_uk_landing"
    return False, ""


def _count_surname_matches(text: str, authors: list[str]) -> int:
    """Count bib author surnames that appear in ``text`` (token-boundary).

    Diacritics are folded via NFKD normalisation on both sides so "van
    Fraassen" in the bib matches "van Fraassen" / "Van Fraassen" / "van
    fraassen" in the PDF. Each author contributes at most one match —
    duplicate hits for the same surname are not double-counted.
    """
    if not authors:
        return 0
    normalised_text = _normalise_for_match(text)
    hits = 0
    seen: set[str] = set()
    for author in authors:
        surname = _extract_surname(author)
        if not surname:
            continue
        key = _normalise_for_match(surname)
        if not key or key in seen:
            continue
        seen.add(key)
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", normalised_text):
            hits += 1
    return hits


def _extract_surname(author: str) -> str:
    """Pull the surname out of a bib-style author string.

    Handles "Last, First" (comma-separated) and "First Middle Last" forms,
    plus the "van / de / von" particle convention — for "van Fraassen, Bas"
    or "Bas van Fraassen" we return "van Fraassen".
    """
    cleaned = author.strip()
    if not cleaned:
        return ""
    if "," in cleaned:
        last, _ = cleaned.split(",", 1)
        return last.strip()
    tokens = cleaned.split()
    if not tokens:
        return ""
    for i, tok in enumerate(tokens):
        if tok.lower() in {"van", "de", "von", "der", "den", "du", "da", "la", "le"}:
            return " ".join(tokens[i:])
    return tokens[-1]


def _normalise_for_match(s: str) -> str:
    """Lowercase, NFKD-fold, drop combining marks, collapse whitespace."""
    folded = unicodedata.normalize("NFKD", s)
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.lower())


def _doi_in_text(text: str, doi: str) -> bool:
    """Return True if ``doi`` appears in ``text`` after light normalisation."""
    if not doi:
        return False
    from sciwrite_lint._network import clean_and_validate_doi

    clean = clean_and_validate_doi(doi)
    if not clean:
        return False
    needle = _DOI_NORMALIZE_RE.sub("", clean).lower()
    haystack = _DOI_NORMALIZE_RE.sub("", text).lower()
    return needle in haystack


def _year_in_text(text: str, year: int | None) -> bool:
    """Return True if ``year`` appears as a 4-digit token in ``text``."""
    if year is None:
        return False
    return bool(re.search(rf"(?<!\d){year}(?!\d)", text))
