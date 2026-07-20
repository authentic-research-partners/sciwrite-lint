"""Citation parsing, orphan detection, and local source checking."""

from __future__ import annotations

import re
from pathlib import Path

import bibtexparser
from loguru import logger

from sciwrite_lint.models import Citation, Finding
from sciwrite_lint.tex_parser import (
    extract_bibliography,
    find_all_cite_keys,
    find_bare_cite_keys,
    find_unverified_cite_keys,
)


# ---------------------------------------------------------------------------
# Citation extraction (bibitem or .bib)
# ---------------------------------------------------------------------------


class CitationSource:
    """Describes how citations were extracted (for reporting)."""

    def __init__(
        self, method: str, path: str, count: int, warnings: list[str] | None = None
    ):
        self.method = method  # "bibtex", "bibitem_simple", "bibitem_natbib"
        self.path = path  # file that was parsed
        self.count = count  # number of citations found
        self.warnings = warnings or []

    def __str__(self) -> str:
        return f"{self.method} ({self.count} citations from {self.path})"


# Module-level last extraction info (set by extract_bibitems)
_last_source: CitationSource | None = None


def get_last_citation_source() -> CitationSource | None:
    """Return info about the last extraction (for UI/CLI reporting)."""
    return _last_source


def extract_bibitems(
    tex_path: Path,
    bib_format: str = "auto",
    bib_path: Path | None = None,
) -> list[Citation]:
    r"""Extract citations from a paper.

    Supports two modes:
    - \bibitem inline in .tex (simple or natbib)
    - .bib file referenced via \bibliography{}

    Args:
        tex_path: Path to .tex file.
        bib_format: "simple", "natbib", or "auto" (detect).
        bib_path: Explicit path to .bib file. If None, auto-detected from
                  \bibliography{} command in .tex.

    Raises:
        ValueError: If both .bib file and \bibitem entries are found.
    """
    global _last_source
    text = tex_path.read_text(encoding="utf-8")

    # Detect both sources
    resolved_bib = bib_path or _find_bib_file(tex_path, text)
    has_bib_file = resolved_bib is not None and resolved_bib.exists()
    bib_text = extract_bibliography(text)
    has_bibitems = bool(bib_text and "\\bibitem" in bib_text)

    # Conflict: both .bib and \bibitem present
    if has_bib_file and has_bibitems:
        assert resolved_bib is not None  # narrowing for mypy
        raise ValueError(
            f"Ambiguous bibliography in {tex_path.name}: "
            f"found both .bib file ({resolved_bib.name}) and "
            f"\\bibitem entries. Use one or the other."
        )

    # No citations found at all
    if not has_bib_file and not has_bibitems:
        _last_source = CitationSource(
            "none", str(tex_path.name), 0, ["No bibliography found"]
        )
        return []

    # .bib file mode
    if has_bib_file:
        assert resolved_bib is not None  # narrowing for mypy
        citations = _extract_from_bib(resolved_bib, tex_path.stem)
        _last_source = CitationSource(
            "bibtex",
            str(resolved_bib.name),
            len(citations),
        )
        return citations

    # \bibitem mode
    if bib_format == "auto":
        bib_format = "natbib" if "\\newblock" in bib_text else "simple"

    pattern = re.compile(
        r"\\bibitem(?:\[([^\]]*)\])?\{([^}]+)\}\s*(.*?)(?=\\bibitem|$)",
        re.DOTALL,
    )
    citations = []
    for match in pattern.finditer(bib_text):
        label, key, body = match.group(1), match.group(2), match.group(3).strip()
        if bib_format == "natbib":
            c = _parse_natbib(key, body, label or "")
        else:
            c = _parse_simple(key, body)
        c.source_paper = tex_path.stem
        c.bib_format = bib_format
        citations.append(c)

    _last_source = CitationSource(
        f"bibitem_{bib_format}",
        str(tex_path.name),
        len(citations),
    )
    return citations


def _find_bib_file(tex_path: Path, text: str) -> Path | None:
    r"""Find .bib file from \bibliography{name} command in .tex."""
    match = re.search(r"\\bibliography\{([^}]+)\}", text)
    if not match:
        return None
    bib_name = match.group(1).strip()
    if not bib_name.endswith(".bib"):
        bib_name += ".bib"
    bib_path = tex_path.parent / bib_name
    return bib_path if bib_path.exists() else None


def parse_bib_file(bib_path: Path, source_paper: str = "") -> list[Citation]:
    """Parse a standalone ``.bib`` file into Citation objects.

    Public entry for callers that already hold a resolved ``.bib`` path
    (e.g. a markdown manuscript's sibling bibliography) and only need the
    bibliography entries — no ``\\cite`` / ``\\bibitem`` discovery.
    """
    return _extract_from_bib(bib_path, source_paper)


def _extract_from_bib(bib_path: Path, source_paper: str) -> list[Citation]:
    """Parse a .bib file into Citation objects using bibtexparser."""
    text = bib_path.read_text(encoding="utf-8")
    db = bibtexparser.loads(text)

    citations = []
    for entry in db.entries:
        key = entry.get("ID", "")
        if not key:
            continue

        # Authors: bibtexparser gives "First Last and First Last"
        author_str = entry.get("author", "")
        authors = _parse_author_string(author_str) if author_str else []

        # Title: strip braces used for capitalization preservation
        title = entry.get("title", "")
        title = re.sub(r"[{}]", "", title).strip()

        # Venue: journal, booktitle, or howpublished
        venue = (
            entry.get("journal", "")
            or entry.get("booktitle", "")
            or entry.get("howpublished", "")
        )

        # DOI and URL
        doi = entry.get("doi", "")
        url = entry.get("url", "")
        # Extract URL from howpublished if url field is empty
        if not url:
            hp = entry.get("howpublished", "")
            url_match = re.search(r"\\url\{([^}]+)\}", hp)
            if url_match:
                url = url_match.group(1)
        # Extract DOI from howpublished URL if present
        if not doi and url:
            doi_match = re.search(r"(10\.\d{4,9}/[^\s,}]+)", url)
            if doi_match:
                doi = doi_match.group(1).rstrip(".")
        if not doi:
            hp = entry.get("howpublished", "")
            doi_match = re.search(r"(10\.\d{4,9}/[^\s,}]+)", hp)
            if doi_match:
                doi = doi_match.group(1).rstrip(".")

        # Build raw text for display
        raw_parts = []
        if author_str:
            raw_parts.append(author_str)
        if title:
            raw_parts.append(title)
        if venue:
            raw_parts.append(venue)
        raw_text = ". ".join(raw_parts)

        # arXiv ID from eprint field — the NNNN.NNNNN format is
        # arXiv-specific, no other preprint server uses it.
        arxiv_id = ""
        eprint = entry.get("eprint", "")
        if eprint:
            arxiv_match = re.match(r"(\d{4}\.\d{4,5})", eprint)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)

        # ISBN — standard BibTeX field for books/chapters
        isbn = entry.get("isbn", "")

        # LCCN — Library of Congress Control Number
        lccn = entry.get("lccn", "")

        c = Citation(
            key=key,
            raw_text=raw_text,
            authors=authors,
            title=title,
            year=entry.get("year", ""),
            venue=venue,
            doi=doi,
            url=url,
            arxiv_id=arxiv_id,
            isbn=isbn,
            lccn=lccn,
            source_paper=source_paper,
            bib_format="bibtex",
            entry_type=entry.get("ENTRYTYPE", ""),
        )

        # Sweep non-standard fields for identifiers not yet found
        _sweep_identifiers(c, entry)

        citations.append(c)

    return citations


# Patterns for identifier extraction from non-standard fields.
# Strict length constraints to minimize false positives.
_ARXIV_RE = re.compile(r"arXiv[:\s]+(\d{4}\.\d{4,5})", re.IGNORECASE)
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s,}]+)")
_PMID_RE = re.compile(r"PMID[:\s]+(\d{6,9})")
_PMC_RE = re.compile(r"(PMC\d{6,8})\b")
_ISBN_RE = re.compile(r"(?:ISBN[:\s-]*)?(97[89]\d{10}|\d{9}[\dXx])", re.IGNORECASE)
_LCCN_RE = re.compile(r"LCCN[:\s]+(\d{8,12})", re.IGNORECASE)

# Standard BibTeX fields — skip these in the sweep (already parsed above)
_STANDARD_ID_FIELDS = {
    "doi",
    "eprint",
    "url",
    "eprinttype",
    "archiveprefix",
    "isbn",
    "lccn",
}


def _sweep_identifiers(citation: Citation, entry: dict[str, str]) -> None:
    """Scan non-standard bib fields for structured identifiers.

    Only fills in identifiers that are still missing after standard-field
    parsing. Logs every regex-extracted identifier so we can trace the source.
    """
    # Collect text from non-standard fields only
    sweep_parts: list[tuple[str, str]] = []
    for field_name, value in entry.items():
        if field_name.lower() in _STANDARD_ID_FIELDS:
            continue
        if field_name in ("ID", "ENTRYTYPE"):
            continue
        if value and isinstance(value, str):
            sweep_parts.append((field_name, value))

    if not sweep_parts:
        return

    for field_name, value in sweep_parts:
        # DOI
        if not citation.doi:
            doi_match = _DOI_RE.search(value)
            if doi_match:
                citation.doi = doi_match.group(1).rstrip(".")
                logger.debug(
                    "Extracted DOI from '{}' field of '{}': {}",
                    field_name,
                    citation.key,
                    citation.doi,
                )

        # arXiv
        if not citation.arxiv_id:
            arxiv_match = _ARXIV_RE.search(value)
            if arxiv_match:
                citation.arxiv_id = arxiv_match.group(1)
                logger.debug(
                    "Extracted arXiv ID from '{}' field of '{}': {}",
                    field_name,
                    citation.key,
                    citation.arxiv_id,
                )

        # PMID
        if not citation.pmid:
            pmid_match = _PMID_RE.search(value)
            if pmid_match:
                citation.pmid = pmid_match.group(1)
                logger.debug(
                    "Extracted PMID from '{}' field of '{}': {}",
                    field_name,
                    citation.key,
                    citation.pmid,
                )

        # PMC
        if not citation.pmc_id:
            pmc_match = _PMC_RE.search(value)
            if pmc_match:
                citation.pmc_id = pmc_match.group(1)
                logger.debug(
                    "Extracted PMC ID from '{}' field of '{}': {}",
                    field_name,
                    citation.key,
                    citation.pmc_id,
                )

        # ISBN
        if not citation.isbn:
            isbn_match = _ISBN_RE.search(value)
            if isbn_match:
                citation.isbn = isbn_match.group(1)
                logger.debug(
                    "Extracted ISBN from '{}' field of '{}': {}",
                    field_name,
                    citation.key,
                    citation.isbn,
                )

        # LCCN
        if not citation.lccn:
            lccn_match = _LCCN_RE.search(value)
            if lccn_match:
                citation.lccn = lccn_match.group(1)
                logger.debug(
                    "Extracted LCCN from '{}' field of '{}': {}",
                    field_name,
                    citation.key,
                    citation.lccn,
                )


def _parse_simple(key: str, body: str) -> Citation:
    r"""Parse a simple-style bibitem (no \newblock)."""
    c = Citation(key=key, raw_text=body)
    _extract_doi_url(c, body)
    _extract_year(c, body)

    # Split by newlines BEFORE cleaning (to preserve line structure)
    raw_lines = [ln.strip() for ln in body.split("\n") if ln.strip()]

    if raw_lines:
        c.authors = _parse_author_string(_clean_text(raw_lines[0]))

    # Title: second line (between author and venue)
    if len(raw_lines) > 1:
        title_text = raw_lines[1]
        # Remove \emph{} blocks (those are venue names)
        title_text = re.sub(r"\\emph\{[^}]+\}", "", title_text)
        title_text = _clean_text(title_text).rstrip(".,")
        if title_text:
            c.title = title_text

    # Venue: first \emph{} in body
    emph_matches = re.findall(r"\\emph\{([^}]+)\}", body)
    if emph_matches:
        c.venue = emph_matches[0]

    return c


def _parse_natbib(key: str, body: str, label: str) -> Citation:
    r"""Parse a natbib-style bibitem with \newblock separators."""
    c = Citation(key=key, raw_text=body)
    _extract_doi_url(c, body)
    _extract_year(c, body)

    # Extract year from natbib label if present: [Author(Year)]
    label_year = re.search(r"\((\d{4})\)", label)
    if label_year and not c.year:
        c.year = label_year.group(1)

    # Split by \newblock
    blocks = re.split(r"\\newblock\s*", body)

    # Block 0: author line (e.g., "Bloom, B.~S. (1984).")
    if blocks:
        author_block = blocks[0].strip()
        # Remove year in parens
        author_clean = re.sub(r"\(\d{4}\)\.", "", author_block).strip()
        c.authors = _parse_author_string(author_clean)

    # Block 1: typically the title
    if len(blocks) > 1:
        title_block = blocks[1].strip()
        # If title is in \emph{}, it's a book title
        emph_match = re.match(r"\\emph\{(.+?)\}", title_block, re.DOTALL)
        if emph_match:
            c.title = _clean_text(emph_match.group(1)).rstrip(".")
        else:
            c.title = _clean_text(title_block).rstrip(".")

    # Block 2+: venue
    if len(blocks) > 2:
        venue_block = blocks[2].strip()
        emph_match = re.search(r"\\emph\{([^}]+)\}", venue_block)
        if emph_match:
            c.venue = emph_match.group(1)

    return c


def _extract_doi_url(c: Citation, body: str) -> None:
    """Extract DOI and URL from bibitem body text."""
    url_match = re.search(r"\\url\{([^}]+)\}", body)
    if url_match:
        c.url = url_match.group(1)

    doi_match = re.search(r"(10\.\d{4,9}/[^\s,}]+)", body)
    if doi_match:
        c.doi = doi_match.group(1).rstrip(".")
    elif c.url:
        doi_in_url = re.search(r"(10\.\d{4,9}/[^\s,}]+)", c.url)
        if doi_in_url:
            c.doi = doi_in_url.group(1).rstrip(".")


def _extract_year(c: Citation, body: str) -> None:
    """Extract publication year from bibitem body."""
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", body)
    if year_match:
        c.year = year_match.group(1)


def _parse_author_string(text: str) -> list[str]:
    """Parse an author string into individual author names."""
    text = _clean_text(text).rstrip(".")
    if not text:
        return []
    parts = re.split(r"\s+and\s+|\s*&\s*|\s*,\s+and\s+", text)
    authors = []
    for part in parts:
        part = part.strip().rstrip(",")
        if part and len(part) > 1:
            authors.append(part)
    return authors


def _clean_text(text: str) -> str:
    """Remove LaTeX markup from text."""
    text = re.sub(r"\\emph\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\texttt\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\url\{[^}]*\}", "", text)
    text = re.sub(r"\{([^}]*)\}", r"\1", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\\newblock\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Web resource detection
# ---------------------------------------------------------------------------


def is_web_resource(citation: Citation) -> bool:
    """Detect if a citation is a web resource rather than an academic paper."""
    return citation.entry_type == "misc" and bool(citation.url) and not citation.doi


# ---------------------------------------------------------------------------
# Orphan detection
# ---------------------------------------------------------------------------


def find_orphans(
    citations: list[Citation],
    tex_path: Path,
    aux_path: Path | None = None,
) -> tuple[set[str], set[str]]:
    """Find cite keys without bibitems and bibitems never cited.

    Returns (cited_but_no_bib, bib_but_not_cited).
    """
    text = tex_path.read_text(encoding="utf-8")
    cite_keys = {k for _, k in find_all_cite_keys(text)}
    bib_keys = {c.key for c in citations}

    if aux_path and aux_path.exists():
        aux_cite, aux_bib = parse_aux_citations(aux_path)
        cite_keys |= aux_cite

    return cite_keys - bib_keys, bib_keys - cite_keys


_NOCITE_PATTERN = re.compile(r"\\nocite\{([^}]+)\}")


def cited_keys(tex_path: Path, aux_path: Path | None = None) -> set[str]:
    r"""Return the set of bib keys referenced from the manuscript.

    Includes keys from \cite* commands, \nocite{key} (and \nocite{\*}
    sentinel), and — when present — the \citation records in the .aux
    that BibTeX writes at compile time.

    Strips comments first so commented-out lines don't leak keys.
    The .aux union covers the "\nocite{*} means every bib entry"
    case: BibTeX expands the sentinel into concrete \citation{key}
    lines that parse_aux_citations picks up.
    """
    from sciwrite_lint.tex_parser import strip_comments

    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    keys = {k for _, k in find_all_cite_keys(text)}
    for m in _NOCITE_PATTERN.finditer(text):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    if aux_path and aux_path.exists():
        aux_cite, _ = parse_aux_citations(aux_path)
        keys |= aux_cite
    return keys


def filter_to_cited(
    citations: list[Citation],
    tex_path: Path,
    aux_path: Path | None = None,
) -> list[Citation]:
    r"""Drop bib entries that are never \cite'd in the paper.

    Uncited entries don't belong in API verification or full-text
    acquisition — they have no claim to verify and no text to fetch.
    The dangling-cite check already reports key-level mismatches
    directly from the bib, so filtering here doesn't hide anything.
    """
    keys = cited_keys(tex_path, aux_path)
    return [c for c in citations if c.key in keys]


def parse_aux_citations(aux_path: Path) -> tuple[set[str], set[str]]:
    """Parse .aux file for citation and bibcite keys."""
    text = aux_path.read_text(encoding="utf-8")
    cite_keys = set()
    bib_keys = set()

    for match in re.finditer(r"\\citation\{([^}]+)\}", text):
        for key in match.group(1).split(","):
            cite_keys.add(key.strip())

    for match in re.finditer(r"\\bibcite\{([^}]+)\}", text):
        bib_keys.add(match.group(1).strip())

    return cite_keys, bib_keys


# ---------------------------------------------------------------------------
# Local source checking
# ---------------------------------------------------------------------------


def _is_valid_local_pdf(path: Path) -> bool:
    """Check that a local file is actually a PDF (magic header + minimum size)."""
    from sciwrite_lint.fulltext import _is_valid_pdf

    try:
        data = path.read_bytes()
    except OSError:
        return False
    return _is_valid_pdf(data)


def check_local_sources(
    citations: list[Citation],
    refs_dir: Path,
) -> None:
    """Check which citations have local PDFs or markdown summaries.

    Match rule: the filename's leading alphanumeric token (everything up
    to the first ``_``, ``-``, ``.``, or whitespace) must equal the
    citekey, case-insensitive. So ``smith2020.pdf``,
    ``smith2020_v2.pdf``, and ``Smith2020_paper.pdf`` all match key
    ``smith2020``, while ``smith2020rm.pdf`` does NOT match
    ``smith2020opening`` — its leading token is a different citekey.

    The historical ``'2' → '_2'`` insertion (``smith2020`` →
    ``smith_2020.pdf``) is preserved via a full-stem equality check:
    the filename stem must equal the citekey with an underscore
    inserted before the first ``2``. Extended variants like
    ``smith_2020_v2.pdf`` are not accepted under this convention —
    use the leading-token form instead.

    There is no author-only or author+year fuzzy match: that path
    silently misassigned PDFs across references that share an author
    and year (e.g. several ``smith2020*`` keys all picking up the
    first matching file from ``iterdir()``). Files that don't follow
    the citekey-leading-token convention are not considered local
    sources here; users can drop them into ``local_pdfs/`` instead,
    which has a fuzzy title matcher with a similarity threshold.
    """
    if not refs_dir.exists():
        return

    from sciwrite_lint.local_sources import leading_token

    suffixes = (".pdf", ".md")
    by_token: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for f in refs_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in suffixes:
            continue
        by_stem.setdefault(f.stem.lower(), f)
        token = leading_token(f.name).lower()
        # Last-write wins on collision, but a real workspace has at
        # most one academic source per citekey so collisions imply a
        # stray file the user should clean up.
        if token:
            by_token.setdefault(token, f)

    for c in citations:
        key_lower = c.key.lower()
        underscore_variant = key_lower.replace("2", "_2", 1)
        match = by_token.get(key_lower) or by_stem.get(underscore_variant)

        if match is None:
            c.local_status = "none"
            continue

        suffix = match.suffix.lower()
        if suffix == ".pdf" and not _is_valid_local_pdf(match):
            logger.warning(
                "{}: local file {} is not a valid PDF, skipping",
                c.key,
                match.name,
            )
            c.local_status = "none"
            continue
        c.local_status = "pdf" if suffix == ".pdf" else "md"
        c.local_path = str(match)


# ---------------------------------------------------------------------------
# Shared citation consistency
# ---------------------------------------------------------------------------


def check_shared_citations(
    all_citations: dict[str, list[Citation]],
) -> list[Finding]:
    """Check citations shared across papers for consistency."""
    by_key: dict[str, list[Citation]] = {}
    for paper_name, citations in all_citations.items():
        for c in citations:
            by_key.setdefault(c.key, []).append(c)

    findings = []
    for key, cites in by_key.items():
        if len(cites) < 2:
            continue
        years = {c.year for c in cites if c.year}
        if len(years) > 1:
            findings.append(
                Finding(
                    level="error",
                    rule_id="ref-006",
                    message=f"Citation '{key}' has different years across papers: {years}",
                    file="cross",
                )
            )
        titles = {c.title for c in cites if c.title}
        if len(titles) > 1:
            findings.append(
                Finding(
                    level="warning",
                    rule_id="ref-006",
                    message=f"Citation '{key}' has different titles across papers: {titles}",
                    file="cross",
                )
            )

    return findings


# ---------------------------------------------------------------------------
# Tier enforcement
# ---------------------------------------------------------------------------


def check_tiers(
    citations: list[Citation],
    tex_path: Path,
    metadata_dir: Path,
) -> list[Finding]:
    r"""Check tier enforcement: bare \cite on T2 = warning, T3 cite = error."""
    from sciwrite_lint.references.metadata import load_all_metadata

    findings: list[Finding] = []
    all_meta = load_all_metadata(metadata_dir)
    if not all_meta:
        return findings

    text = tex_path.read_text(encoding="utf-8")
    bare_keys = find_bare_cite_keys(text)
    unverified_keys = find_unverified_cite_keys(text)

    for c in citations:
        meta = all_meta.get(c.key)
        if not meta:
            continue

        tier = meta.access.get("tier", "")

        if tier == "T3" and c.key in (bare_keys | unverified_keys):
            reason = "dead URL" if meta.api_match == "web_dead" else "not found in APIs"
            findings.append(
                Finding(
                    level="error",
                    rule_id="ref-008",
                    message=(
                        f"'{c.key}' is T3 ({reason}). Cannot cite — remove or replace."
                    ),
                    file=str(tex_path.name),
                )
            )
        elif tier == "T2" and c.key in bare_keys:
            if meta.api_match == "web_blocked":
                message = (
                    f"'{c.key}' is T2 (blocked by site — we could not verify "
                    f"automatically). Verify the URL manually and use "
                    f"\\citeunverified{{{c.key}}}."
                )
            else:
                message = (
                    f"'{c.key}' is T2 (no full text). Use "
                    f"\\citeunverified{{{c.key}}} or obtain full text."
                )
            findings.append(
                Finding(
                    level="warning",
                    rule_id="ref-009",
                    message=message,
                    file=str(tex_path.name),
                )
            )
        elif tier == "T1" and c.key in unverified_keys:
            findings.append(
                Finding(
                    level="info",
                    rule_id="ref-009",
                    message=(
                        f"'{c.key}' is T1 (full text available). "
                        f"Can upgrade \\citeunverified to \\cite."
                    ),
                    file=str(tex_path.name),
                )
            )

    return findings
