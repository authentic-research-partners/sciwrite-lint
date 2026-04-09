"""Bibliography existence verification for cited references.

After GROBID parses each reference PDF and stores its bibliography entries
in workspace.db, this module batch-verifies those entries against
OpenAlex and Semantic Scholar to detect hallucinated references,
retracted papers, and metadata mismatches.

This runs as part of the default pipeline (no special flags needed).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loguru import logger

from sciwrite_lint.config import LintConfig


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RefBibCheck(BaseModel):
    """Result of checking a reference's own bibliography for existence and metadata."""

    key: str
    total_entries: int
    found: int
    not_found: int
    retracted: int = 0
    metadata_mismatches: int = 0
    mismatch_details: list[str] = Field(default_factory=list)

    @property
    def hallucination_rate(self) -> float:
        return self.not_found / self.total_entries if self.total_entries else 0.0

    @property
    def mismatch_rate(self) -> float:
        return self.metadata_mismatches / self.found if self.found else 0.0


# ---------------------------------------------------------------------------
# Bibliography parsing helpers
# ---------------------------------------------------------------------------

# Pattern for references section entries like "1. Author (2020). Title..."
_REF_ENTRY_RE = re.compile(
    r"^(\d+)\.\s+(.+?)(?:\.\s+|\n)",
    re.MULTILINE,
)

_TITLE_RE = re.compile(
    r"^\d+\.\s+"  # "1. "
    r"(?:[A-Z][^.]*?\.\s+)*"  # author block: "Author, A. et al. "
    r"(?:\(\d{4}\)\.\s*)?"  # optional year: "(2020). "
    r"(.+?)(?:\.|$)",  # title: everything up to next period
    re.MULTILINE,
)

_DOI_RE = re.compile(r"(?:doi:\s*|doi\.org/)(10\.\d{4,}/[^\s,;]+)", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"(?:arXiv:\s*|arxiv\.org/abs/)(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE
)
_PMID_RE = re.compile(r"(?:PMID:\s*|pubmed/)(\d{6,9})", re.IGNORECASE)
_BRACKET_CITE_RE = re.compile(r"\[(\d+(?:[,;\s]+\d+)*)\]")


class ExtractedCitation(BaseModel):
    """A citation extracted from GROBID-parsed markdown."""

    index: int
    context: str
    ref_entry: str = ""


def extract_citations_from_markdown(text: str) -> list[ExtractedCitation]:
    """Extract inline bracket citations from GROBID-parsed markdown.

    Returns citations with their surrounding paragraph context.
    Only extracts from body text (stops at References/Bibliography heading).
    """
    bib_start = _find_bibliography_start(text)
    body = text[:bib_start] if bib_start else text
    bib_text = text[bib_start:] if bib_start else ""
    bib_entries = _parse_bib_entries(bib_text)

    results: list[ExtractedCitation] = []
    seen: set[tuple[int, str]] = set()

    for match in _BRACKET_CITE_RE.finditer(body):
        indices_str = match.group(1)
        start = max(0, body.rfind("\n\n", 0, match.start()))
        end = body.find("\n\n", match.end())
        if end == -1:
            end = len(body)
        context = body[start:end].strip()

        for idx_str in re.split(r"[,;]\s*", indices_str):
            idx_str = idx_str.strip()
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
            dedup_key = (idx, context[:100])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            results.append(
                ExtractedCitation(
                    index=idx,
                    context=context,
                    ref_entry=bib_entries.get(idx, ""),
                )
            )

    return results


class _BibEntry(BaseModel):
    """A parsed bibliography entry with extractable IDs and metadata."""

    doi: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    venue: str = ""

    @property
    def has_id(self) -> bool:
        return bool(self.doi or self.arxiv_id or self.pmid)

    @property
    def has_metadata(self) -> bool:
        """Whether this entry has structured metadata for comparison."""
        return bool(self.title and (self.authors or self.year))


def _prioritize_entries(entries: list[_BibEntry], limit: int) -> list[_BibEntry]:
    """Sort entries so formal-looking ones come first, then cap at *limit*.

    Entries with structured IDs (DOI/arXiv/PMID) are most likely to be formal
    academic papers and most useful for API verification. Entries with a title
    but no ID can still be title-searched. Entries with neither are least
    useful. Within each tier the original order is preserved (stable sort).
    """
    entries.sort(
        key=lambda e: (e.has_id, bool(e.title)),
        reverse=True,
    )
    return entries[:limit]


def _find_bibliography_start(text: str) -> int | None:
    """Find where the references/bibliography section starts."""
    pattern = re.compile(
        r"(?m)^#{1,3}\s+(?:References|Bibliography|Works Cited)\s*$",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.start() if m else None


def _parse_bib_entries(bib_text: str) -> dict[int, str]:
    """Parse numbered bibliography entries."""
    entries: dict[int, str] = {}
    for match in _REF_ENTRY_RE.finditer(bib_text):
        idx = int(match.group(1))
        # Grab the full entry — up to next numbered entry or end
        start = match.start()
        next_match = _REF_ENTRY_RE.search(bib_text, match.end())
        end = next_match.start() if next_match else len(bib_text)
        entries[idx] = bib_text[start:end].strip()
    return entries


def _extract_bib_entries(bib_entries: dict[int, str]) -> list[_BibEntry]:
    """Extract IDs and titles from GROBID bibliography entries.

    Extracts DOIs, arXiv IDs, and PMIDs for batch verification.
    Titles are used for entries without any structured ID.
    """
    results: list[_BibEntry] = []
    for _idx, entry in sorted(bib_entries.items()):
        doi = ""
        arxiv_id = ""
        pmid = ""
        title = ""

        doi_match = _DOI_RE.search(entry)
        if doi_match:
            doi = doi_match.group(1).rstrip(".")

        arxiv_match = _ARXIV_RE.search(entry)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)

        pmid_match = _PMID_RE.search(entry)
        if pmid_match:
            pmid = pmid_match.group(1)

        title_match = _TITLE_RE.match(entry)
        if title_match:
            t = title_match.group(1).strip()
            if 10 < len(t) < 300:
                title = t

        if doi or arxiv_id or pmid or title:
            results.append(
                _BibEntry(doi=doi, arxiv_id=arxiv_id, pmid=pmid, title=title)
            )
    return results


# ---------------------------------------------------------------------------
# Metadata comparison helpers
# ---------------------------------------------------------------------------


def _oa_work_to_api_data(work: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAlex work record to a common metadata dict."""
    authors: list[str] = []
    for authorship in work.get("authorships") or []:
        name = (authorship.get("author") or {}).get("display_name", "")
        if name:
            authors.append(name)
    venue = ""
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    venue = source.get("display_name") or ""
    return {
        "title": work.get("title") or "",
        "authors": authors,
        "year": str(work.get("publication_year") or ""),
        "venue": venue,
        "retracted": bool(work.get("is_retracted", False)),
    }


def _s2_paper_to_api_data(paper: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Semantic Scholar paper record to a common metadata dict."""
    authors = [a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")]
    return {
        "title": paper.get("title") or "",
        "authors": authors,
        "year": str(paper.get("year") or ""),
        "venue": paper.get("venue") or "",
        "retracted": bool(
            paper.get("isRetracted", False)
            or paper.get("publicationTypes", None) == ["Retracted"]
        ),
    }


def _compare_bib_entry_metadata(
    entry: _BibEntry, api_data: dict[str, Any]
) -> list[str]:
    """Compare a bibliography entry's metadata against API data.

    Returns list of mismatch description strings (empty if all match).
    """
    from sciwrite_lint.references.matching import (
        AUTHOR_THRESHOLD,
        TITLE_THRESHOLD,
        VENUE_THRESHOLD,
        author_similarity,
        title_similarity,
        venue_similarity,
        year_match,
    )

    issues: list[str] = []

    api_title = api_data.get("title", "")
    if entry.title and api_title:
        sim = title_similarity(entry.title, api_title)
        if sim < TITLE_THRESHOLD:
            issues.append(
                f"Title mismatch (sim={sim:.2f}): "
                f"bib='{entry.title[:60]}', API='{api_title[:60]}'"
            )

    api_authors: list[str] = api_data.get("authors", [])
    if entry.authors and api_authors:
        sim = author_similarity(entry.authors, api_authors)
        if sim < AUTHOR_THRESHOLD:
            bib_str = ", ".join(entry.authors[:3])
            api_str = ", ".join(api_authors[:3])
            issues.append(
                f"Author mismatch (sim={sim:.2f}): bib='{bib_str}', API='{api_str}'"
            )

    api_year = api_data.get("year", "")
    if entry.year and api_year and not year_match(entry.year, api_year):
        issues.append(f"Year mismatch: bib={entry.year}, API={api_year}")

    api_venue = api_data.get("venue", "")
    if entry.venue and api_venue:
        sim = venue_similarity(entry.venue, api_venue)
        if sim < VENUE_THRESHOLD:
            issues.append(
                f"Venue mismatch (sim={sim:.2f}): "
                f"bib='{entry.venue[:60]}', API='{api_venue[:60]}'"
            )

    return issues


# ---------------------------------------------------------------------------
# BibVerifier — batch bibliography verification
# ---------------------------------------------------------------------------


class BibVerifier:
    """Collects bibliography entries from multiple references, then
    batch-verifies all of them in minimal API calls.

    Usage::

        verifier = BibVerifier(config)
        verifier.add_reference("smith2020", ref_text_smith)
        verifier.add_reference("jones2021", ref_text_jones)
        await verifier.verify_all()
        result = verifier.get_result("smith2020")  # RefBibCheck

    Two-pass strategy across ALL collected entries:
    1. Batch DOI lookup — all DOIs from all references in one OpenAlex
       request (up to 200 per request, paginated if needed).
    2. Parallel title search — remaining entries without DOIs.

    10 refs × 30 entries = 300 total. If 60% have DOIs: 2 batch requests
    + ~120 title searches = ~122 requests instead of 300.
    """

    def __init__(self, config: LintConfig | None = None) -> None:
        self._config = config or LintConfig()
        # {ref_key: list of entries}
        self._entries: dict[str, list[_BibEntry]] = {}
        # {ref_key: RefBibCheck} — populated after verify_all()
        self._results: dict[str, RefBibCheck] = {}

    # Maximum bibliography entries to verify per cited paper.
    # Books and monographs can have hundreds of references;
    # verifying all would be slow and provide diminishing signal.
    MAX_BIB_ENTRIES = 100

    def add_reference(self, ref_key: str, ref_text: str) -> None:
        """Extract and store bibliography entries from a parsed reference.

        Regex-based extraction from markdown text. Prefer
        ``add_reference_from_registry`` when structured GROBID data is
        available in workspace.db.
        """
        bib_start = _find_bibliography_start(ref_text)
        bib_text = ref_text[bib_start:] if bib_start else ""
        raw_entries = _parse_bib_entries(bib_text)
        entries = _extract_bib_entries(raw_entries)
        if len(entries) > self.MAX_BIB_ENTRIES:
            logger.info(
                "Capping bibliography verification for {} ({} entries > {} limit)",
                ref_key,
                len(entries),
                self.MAX_BIB_ENTRIES,
            )
            entries = _prioritize_entries(entries, self.MAX_BIB_ENTRIES)
        self._entries[ref_key] = entries

    def add_reference_from_registry(
        self,
        ref_key: str,
        conn: "sqlite3.Connection",
    ) -> bool:
        """Load bibliography entries from workspace.db instead of parsing text.

        Returns True if entries were found, False if registry has no data
        for this parent_key (caller should use ``add_reference`` text path).
        """
        from sciwrite_lint.references.workspace_db import load_bibliography_entries

        rows = load_bibliography_entries(conn, parent_key=ref_key, depth=1)
        if not rows:
            return False

        entries = [
            _BibEntry(
                doi=r.get("doi") or "",
                arxiv_id=r.get("arxiv_id") or "",
                pmid=r.get("pmid") or "",
                title=r.get("title") or "",
                authors=r.get("authors") or [],
                year=r.get("year") or "",
                venue=r.get("venue") or "",
            )
            for r in rows
        ]
        if len(entries) > self.MAX_BIB_ENTRIES:
            logger.info(
                "Capping bibliography verification for {} ({} entries > {} limit)",
                ref_key,
                len(entries),
                self.MAX_BIB_ENTRIES,
            )
            entries = _prioritize_entries(entries, self.MAX_BIB_ENTRIES)
        self._entries[ref_key] = entries
        logger.debug(
            "{}: loaded {} bibliography entries from registry",
            ref_key,
            len(entries),
        )
        return True

    def get_result(self, ref_key: str) -> RefBibCheck:
        """Get verification result for a reference. Call after verify_all()."""
        return self._results.get(
            ref_key, RefBibCheck(key=ref_key, total_entries=0, found=0, not_found=0)
        )

    async def verify_all(self) -> None:
        """Batch-verify all collected bibliography entries.

        Three-pass strategy across ALL collected entries:
        1. OpenAlex batch DOI (one request per 200 DOIs)
        2. S2 batch for arXiv + PMID IDs (one request per 500 IDs)
        3. Parallel title search for entries with no ID

        When entries have structured metadata (from GROBID via registry),
        also compares title/authors/year/venue against API responses.
        """
        import asyncio

        import httpx

        from sciwrite_lint.rate_limiter import MonotonicRateLimiter

        # Partition entries by best available ID
        all_doi: list[tuple[str, _BibEntry]] = []
        all_s2_id: list[tuple[str, _BibEntry]] = []  # arXiv + PMID
        all_title_only: list[tuple[str, _BibEntry]] = []

        for ref_key, entries in self._entries.items():
            for e in entries:
                if e.doi:
                    all_doi.append((ref_key, e))
                elif e.arxiv_id or e.pmid:
                    all_s2_id.append((ref_key, e))
                elif e.title:
                    all_title_only.append((ref_key, e))

        # Track found entries per ref
        ref_found: dict[str, int] = {}
        # Track metadata mismatches: {ref_key: list of detail strings}
        ref_mismatches: dict[str, list[str]] = {}
        # Track retracted entries per ref
        ref_retracted: dict[str, int] = {}

        def _mark_found(rk: str) -> None:
            ref_found[rk] = ref_found.get(rk, 0) + 1

        def _check_metadata(
            rk: str, entry: _BibEntry, api_data: dict[str, Any]
        ) -> None:
            """Compare entry metadata against API response, record mismatches."""
            if api_data.get("retracted"):
                ref_retracted[rk] = ref_retracted.get(rk, 0) + 1
                ref_mismatches.setdefault(rk, []).append(
                    f"RETRACTED: '{entry.title[:60]}'" if entry.title else "RETRACTED"
                )
            if not entry.has_metadata:
                return
            issues = _compare_bib_entry_metadata(entry, api_data)
            if issues:
                ref_mismatches.setdefault(rk, []).extend(issues)

        _OA_SELECT = (
            "id,doi,title,authorships,publication_year,primary_location,is_retracted"
        )

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            # Pass 1: OpenAlex batch DOI (one request per 200)
            # Index API results by DOI for metadata comparison
            doi_api_data: dict[str, dict[str, Any]] = {}
            oa_params: dict[str, str | int] = {"select": _OA_SELECT}
            if self._config.polite_email:
                oa_params["mailto"] = self._config.polite_email

            for i in range(0, len(all_doi), 200):
                batch = all_doi[i : i + 200]
                doi_filter = "|".join(f"https://doi.org/{e.doi}" for _, e in batch)
                try:
                    resp = await client.get(
                        "https://api.openalex.org/works",
                        params={
                            **oa_params,
                            "filter": f"doi:{doi_filter}",
                            "per_page": 200,
                        },
                    )
                    if resp.status_code == 200:
                        for work in resp.json().get("results", []):
                            doi = (
                                (work.get("doi") or "")
                                .replace("https://doi.org/", "")
                                .lower()
                            )
                            if doi:
                                doi_api_data[doi] = _oa_work_to_api_data(work)
                except httpx.HTTPError as e:
                    logger.debug("OpenAlex DOI batch lookup failed: {}", e)

            for ref_key, entry in all_doi:
                api_data = doi_api_data.get(entry.doi.lower())
                if api_data:
                    _mark_found(ref_key)
                    _check_metadata(ref_key, entry, api_data)
                elif entry.arxiv_id or entry.pmid:
                    all_s2_id.append((ref_key, entry))
                elif entry.title:
                    all_title_only.append((ref_key, entry))

            # Pass 2: S2 batch for arXiv + PMID (one request per 500)
            if all_s2_id:
                s2_ids: list[tuple[str, str]] = []
                s2_entries: list[tuple[str, _BibEntry]] = []
                for ref_key, entry in all_s2_id:
                    if entry.arxiv_id:
                        s2_ids.append((ref_key, f"ARXIV:{entry.arxiv_id}"))
                        s2_entries.append((ref_key, entry))
                    elif entry.pmid:
                        s2_ids.append((ref_key, f"PMID:{entry.pmid}"))
                        s2_entries.append((ref_key, entry))

                for i in range(0, len(s2_ids), 500):
                    s2_batch = s2_ids[i : i + 500]
                    s2_entry_batch = s2_entries[i : i + 500]
                    ids = [sid for _, sid in s2_batch]
                    try:
                        resp = await client.post(
                            "https://api.semanticscholar.org/graph/v1/paper/batch",
                            json={"ids": ids},
                            params={
                                "fields": "externalIds,title,authors,year,venue,isRetracted",
                            },
                        )
                        if resp.status_code == 200:
                            papers = resp.json()
                            for paper, (rk, _), (_, entry) in zip(
                                papers, s2_batch, s2_entry_batch
                            ):
                                if paper:  # S2 returns null for not-found
                                    _mark_found(rk)
                                    api_data = _s2_paper_to_api_data(paper)
                                    _check_metadata(rk, entry, api_data)
                                elif rk not in [r for r, _ in all_title_only]:
                                    orig = next(
                                        (
                                            e
                                            for r, e in all_s2_id
                                            if r == rk and e.title
                                        ),
                                        None,
                                    )
                                    if orig:
                                        all_title_only.append((rk, orig))
                    except httpx.HTTPError as e:
                        logger.debug("S2 batch lookup failed: {}", e)

            # Pass 3: parallel title search for entries with no ID
            title_results: list[tuple[str, bool, _BibEntry]] = []
            if all_title_only:
                limiter = MonotonicRateLimiter(10, 0.15)

                async def _check_title(
                    rk: str,
                    entry: _BibEntry,
                ) -> tuple[str, bool, _BibEntry]:
                    async with limiter:
                        pass
                    query = re.sub(r"[^\w\s-]", "", entry.title).strip()
                    if not query:
                        return rk, False, entry
                    try:
                        resp = await client.get(
                            "https://api.openalex.org/works",
                            params={
                                **oa_params,
                                "filter": f"title.search:{query}",
                                "per_page": 1,
                            },
                        )
                        if resp.status_code == 200:
                            results = resp.json().get("results", [])
                            if results:
                                api_data = _oa_work_to_api_data(results[0])
                                _check_metadata(rk, entry, api_data)
                                return rk, True, entry
                    except httpx.HTTPError as e:
                        logger.debug("OpenAlex title search failed for {}: {}", rk, e)
                    return rk, False, entry

                title_results = await asyncio.gather(
                    *[_check_title(rk, e) for rk, e in all_title_only]
                )

        for rk, found, _entry in title_results:
            if found:
                _mark_found(rk)

        # Build RefBibCheck per reference
        for ref_key, entries in self._entries.items():
            total = len(entries)
            n_found = ref_found.get(ref_key, 0)
            details = ref_mismatches.get(ref_key, [])
            n_retracted = ref_retracted.get(ref_key, 0)
            # Don't count retraction strings as metadata mismatches
            n_mismatches = len(details) - n_retracted
            self._results[ref_key] = RefBibCheck(
                key=ref_key,
                total_entries=total,
                found=n_found,
                not_found=total - n_found,
                retracted=n_retracted,
                metadata_mismatches=n_mismatches,
                mismatch_details=details,
            )


# ---------------------------------------------------------------------------
# Standalone bibliography verification (runs in default pipeline)
# ---------------------------------------------------------------------------


async def run_bib_verification(
    references_dir: Path,
    config: LintConfig | None = None,
) -> list[RefBibCheck]:
    """Verify bibliography entries for all parsed formal references.

    Loads structured bibliography data from workspace.db (stored at parse
    time), verifies existence + metadata + retraction via APIs, and returns
    per-reference results.

    This runs as part of the default pipeline (no ``--chain`` needed).
    """
    config = config or LintConfig()

    from sciwrite_lint.references.metadata import load_all_metadata
    from sciwrite_lint.references.workspace_db import get_db, is_formal_cached_db

    all_meta = load_all_metadata(references_dir)

    verifier = BibVerifier(config)
    refs_loaded = 0

    with get_db(references_dir) as conn:
        for key, meta in all_meta.items():
            # Only formal PDFs have reliable bibliography entries
            local_file = meta.access.get("local_file") or ""
            if local_file.endswith(".pdf") and not is_formal_cached_db(conn, key):
                continue
            # Skip entries without local files (T3 references)
            if not local_file:
                continue
            # Try structured registry data
            if verifier.add_reference_from_registry(key, conn):
                refs_loaded += 1

    if not refs_loaded:
        return []

    logger.info(
        "Bibliography verification: {} references with stored entries",
        refs_loaded,
    )
    await verifier.verify_all()

    results: list[RefBibCheck] = []
    for key in verifier._entries:
        bib_check = verifier.get_result(key)
        if bib_check.total_entries > 0:
            results.append(bib_check)
            parts = [
                f"{bib_check.found}/{bib_check.total_entries} found"
                f" ({bib_check.hallucination_rate:.0%} hallucination)",
            ]
            if bib_check.retracted:
                parts.append(f"{bib_check.retracted} retracted")
            if bib_check.metadata_mismatches:
                parts.append(f"{bib_check.metadata_mismatches} metadata mismatches")
            logger.info("  {}: bibliography {}", key, ", ".join(parts))

    return results


# Keep single-reference convenience function for backward compatibility
async def verify_ref_bibliography(
    ref_key: str,
    ref_text: str,
    config: LintConfig | None = None,
) -> RefBibCheck:
    """Check how many of a reference's own citations exist in OpenAlex.

    Convenience wrapper around BibVerifier for single-reference use.
    For multiple references, use BibVerifier directly to batch across refs.
    """
    verifier = BibVerifier(config)
    verifier.add_reference(ref_key, ref_text)
    await verifier.verify_all()
    return verifier.get_result(ref_key)
