"""Persistent per-citation metadata store.

Stores CitationMetadata in workspace.db (citation_metadata table).
All functions accept ``references_dir`` — the per-paper workspace root
(e.g. ``references/my-paper/``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from sciwrite_lint.models import Citation, CitationMetadata


def load_metadata(key: str, references_dir: Path) -> CitationMetadata | None:
    """Load metadata for a citation key from workspace.db."""
    from sciwrite_lint.references.workspace_db import (
        get_db,
        load_citation_metadata,
    )

    with get_db(references_dir) as conn:
        return load_citation_metadata(conn, key)


def load_all_metadata(references_dir: Path) -> dict[str, CitationMetadata]:
    """Load all metadata from workspace.db."""
    from sciwrite_lint.references.workspace_db import (
        get_db,
        load_all_citation_metadata,
    )

    with get_db(references_dir) as conn:
        return load_all_citation_metadata(conn)


def save_metadata(meta: CitationMetadata, references_dir: Path) -> None:
    """Save metadata to workspace.db."""
    from sciwrite_lint.references.workspace_db import (
        get_db,
        save_citation_metadata,
    )

    with get_db(references_dir) as conn:
        save_citation_metadata(conn, meta)


def compute_tier(meta: CitationMetadata) -> str:
    """Compute access tier from metadata.

    Academic citations:
      T1: verified + full text locally (can check claims)
      T2: verified (confirmed real, no full text to check claims against)
      T3: not found in APIs (cannot confirm existence)

    Web resources:
      T1: URL alive + content downloaded locally
      T2: URL alive but no local content, OR blocked by site (unverifiable)
      T3: Dead URL

    ``web_blocked`` → T2 rather than T3: a WAF 403 is evidence the site is
    operational (just refusing us), so the URL is probably still valid. The
    user gets a WARN from ``reference-exists`` telling them to verify
    manually; downstream claim-verification skips it the same as a T2
    without local content.
    """
    # Manual override takes precedence
    if meta.manual_override and meta.manual_override.get("tier"):
        return meta.manual_override["tier"]

    has_local = bool(meta.access.get("local_file"))

    # Web resource path
    if meta.api_match == "web_dead":
        return "T3"
    if meta.api_match == "web_verified":
        return "T1" if has_local else "T2"
    if meta.api_match == "web_blocked":
        return "T2"

    # Academic path
    is_verified = meta.api_match in ("verified", "mismatch")

    if not is_verified:
        return "T3"
    if has_local:
        return "T1"
    return "T2"


def build_metadata_from_citation(
    citation: Citation,
    api_data: dict[str, Any] | None = None,
    references_dir: Path | None = None,
) -> CitationMetadata:
    """Build a CitationMetadata from a Citation and optional API data."""
    meta = CitationMetadata(key=citation.key)

    # Bibitem data (what our .tex says)
    meta.bibitem = {
        "title": citation.title,
        "authors": citation.authors,
        "year": citation.year,
        "venue": citation.venue,
        "doi": citation.doi,
        "url": citation.url,
        "arxiv_id": citation.arxiv_id,
        "pmid": citation.pmid,
        "pmc_id": citation.pmc_id,
        "isbn": citation.isbn,
        "lccn": citation.lccn,
        "entry_type": citation.entry_type,
        "source_papers": [citation.source_paper],
    }

    # API data (what the world says)
    if api_data and api_data.get("source") == "web":
        meta.api_source = "web"
        meta.verified_date = str(date.today())
        meta.canonical = {
            "title": api_data.get("title") or citation.title,
            "url": api_data.get("url"),
            "status_code": api_data.get("status_code"),
            "content_type": api_data.get("content_type"),
            "resource_type": "web",
        }
        meta.api_match = citation.api_match or "web_verified"
    elif api_data and not api_data.get("error"):
        meta.api_source = api_data.get("source", "")
        meta.verified_date = str(date.today())
        meta.canonical = {
            "title": api_data.get("title", ""),
            "authors": api_data.get("authors", []),
            "year": api_data.get("year"),
            "venue": api_data.get("venue", ""),
            "doi": api_data.get("doi", ""),
            "url": api_data.get("url"),
            "abstract": api_data.get("abstract"),
            "arxiv_id": api_data.get("arxiv_id"),
            "pmid": api_data.get("pmid"),
            "pmcid": api_data.get("pmcid"),
            "isbn": api_data.get("isbn"),
            "lccn": api_data.get("lccn"),
            "s2_pdf_url": api_data.get("s2_pdf_url"),
            "citation_count": api_data.get("citation_count", 0),
            "retracted": api_data.get("retracted", False),
        }
        meta.api_match = citation.api_match or "verified"
    else:
        meta.api_match = citation.api_match or "not_found"

    # Access status
    local_file = None
    if citation.local_path and references_dir:
        try:
            local_file = str(Path(citation.local_path).relative_to(references_dir))
        except ValueError:
            local_file = Path(citation.local_path).name
    elif citation.local_path:
        local_file = Path(citation.local_path).name

    meta.access = {
        "tier": "",
        "local_file": local_file,
        "local_type": _classify_local_file(local_file, references_dir),
        "oa_url": api_data.get("pdf_url") if api_data else None,
        "oa_source": None,
    }

    meta.mismatches = [i for i in citation.issues if "mismatch" in i.lower()]
    meta.issues = citation.issues

    # Compute tier
    meta.access["tier"] = compute_tier(meta)

    return meta


def _classify_local_file(
    local_file: str | None,
    references_dir: Path | None = None,
) -> str:
    """Classify a local reference file by type.

    Returns: "pdf", "summary", "web_page", or "none"
    """
    if not local_file:
        return "none"

    path = Path(local_file)
    if not path.is_absolute() and references_dir:
        path = references_dir / local_file

    if path.suffix == ".pdf":
        return "pdf"

    if path.suffix == ".md":
        if "_web" in path.stem:
            return "web_page"
        if path.exists():
            try:
                head = path.read_text(encoding="utf-8")[:500]
                if "Source: http" in head or "source: http" in head:
                    return "web_page"
            except OSError:
                pass
        return "summary"

    return "none"


def enrich_retraction_status(
    meta: CitationMetadata,
    rw_db: dict[str, Any],
) -> bool:
    """Enrich metadata with Retraction Watch database lookup.

    Looks up the reference DOI in the RW database. If found, sets
    canonical["retracted"] and canonical["retraction_status"] with
    nature, reason, date, and source.

    Args:
        meta: Citation metadata to enrich.
        rw_db: Retraction Watch database (DOI-keyed dict of RWEntry).

    Returns:
        True if metadata was changed (caller should re-save).
    """
    from sciwrite_lint.references.retraction_watch import RWEntry, lookup_doi

    doi = meta.canonical.get("doi", "")
    if not doi:
        return False

    entry: RWEntry | None = lookup_doi(rw_db, doi)
    if entry is None:
        return False

    # Reinstatement clears retraction
    if entry.nature == "Reinstatement":
        changed = meta.canonical.get("retracted", False)
        meta.canonical["retracted"] = False
        meta.canonical.pop("retraction_status", None)
        return changed

    # Retraction or Expression of Concern → set status
    new_status = {
        "nature": entry.nature,
        "reason": entry.reason,
        "date": entry.date,
        "source": "retraction_watch",
    }

    old_status = meta.canonical.get("retraction_status")
    if old_status == new_status:
        return False

    meta.canonical["retraction_status"] = new_status
    if entry.nature in ("Retraction", "Expression of Concern"):
        meta.canonical["retracted"] = True

    return True


def merge_source_paper(meta: CitationMetadata, source_paper: str) -> None:
    """Add a source paper to an existing metadata record."""
    papers = meta.bibitem.get("source_papers", [])
    if source_paper not in papers:
        papers.append(source_paper)
        meta.bibitem["source_papers"] = papers
