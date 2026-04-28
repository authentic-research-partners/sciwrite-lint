"""Reference registry CRUD — cross-depth dedup by structured IDs."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from loguru import logger

_RESULT_COLS = ("ref_key", "workspace_path", "depth", "parent_key")
_ID_COLS = ("doi", "arxiv_id", "pmid", "pmcid", "isbn", "lccn")


def register_reference(
    conn: sqlite3.Connection,
    *,
    ref_key: str,
    workspace_path: str,
    depth: int = 0,
    parent_key: str = "",
    doi: str | None = None,
    arxiv_id: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    isbn: str | None = None,
    lccn: str | None = None,
    title: str | None = None,
    authors: list[str] | None = None,
    year: str | None = None,
    venue: str | None = None,
) -> None:
    """Register a reference in the workspace DB for cross-depth dedup.

    Upserts: if (ref_key, depth, parent_key) exists, updates all fields.
    """
    authors_json = json.dumps(authors) if authors else None
    conn.execute(
        """INSERT INTO ref_registry
           (ref_key, doi, arxiv_id, pmid, pmcid, isbn, lccn,
            title, authors_json, year, venue,
            depth, parent_key, workspace_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (ref_key, depth, parent_key) DO UPDATE SET
             doi = excluded.doi,
             arxiv_id = excluded.arxiv_id,
             pmid = excluded.pmid,
             pmcid = excluded.pmcid,
             isbn = excluded.isbn,
             lccn = excluded.lccn,
             title = excluded.title,
             authors_json = excluded.authors_json,
             year = excluded.year,
             venue = excluded.venue,
             workspace_path = excluded.workspace_path
        """,
        (
            ref_key,
            doi or None,
            arxiv_id or None,
            pmid or None,
            pmcid or None,
            isbn or None,
            lccn or None,
            title or None,
            authors_json,
            year or None,
            venue or None,
            depth,
            parent_key,
            workspace_path,
        ),
    )
    conn.commit()


def lookup_reference(
    conn: sqlite3.Connection,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    isbn: str | None = None,
    lccn: str | None = None,
    title: str | None = None,
    authors: list[str] | None = None,
    year: str | None = None,
    venue: str | None = None,
) -> dict[str, Any] | None:
    """Look up an existing reference by exact structured ID match.

    Checks in priority order: DOI → arXiv → PMID → PMCID → ISBN → LCCN.

    **Fully deterministic.** A hit requires at least one structured ID to
    match exactly.  When a candidate is found, ALL other available data is
    cross-checked (other IDs, title, authors, year, venue) — if any field
    present on both sides disagrees, the candidate is rejected.

    All text comparisons use deterministic normalization (lowercase, strip
    LaTeX, collapse whitespace) — no fuzzy scoring, no thresholds.

    If no structured IDs are available (or none match), returns None.
    Worst case: duplicate download. Better than merging two different
    papers and suppressing real findings.

    Returns dict with ref_key, workspace_path, depth, parent_key on match, or None.
    """
    query_ids = {
        "doi": doi,
        "arxiv_id": arxiv_id,
        "pmid": pmid,
        "pmcid": pmcid,
        "isbn": isbn,
        "lccn": lccn,
    }

    for col, val in query_ids.items():
        if not val:
            continue
        row = conn.execute(
            "SELECT ref_key, workspace_path, depth, parent_key, "  # noqa: S608
            f"doi, arxiv_id, pmid, pmcid, isbn, lccn, "
            f"title, authors_json, year, venue "
            f"FROM ref_registry WHERE {col} = ? LIMIT 1",
            (val,),
        ).fetchone()
        if row and _all_data_agrees(row, query_ids, title, authors, year, venue):
            return _row_to_result(row)

    return None


def _row_to_result(row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert a ref_registry full row to a result dict."""
    return dict(zip(_RESULT_COLS, row[:4]))


def _all_data_agrees(
    row: tuple[Any, ...],
    query_ids: dict[str, str | None],
    title: str | None,
    authors: list[str] | None,
    year: str | None = None,
    venue: str | None = None,
) -> bool:
    """Check that ALL available data agrees with the stored row.

    Row layout: ref_key(0), workspace_path(1), depth(2), parent_key(3),
                doi(4), arxiv_id(5), pmid(6), pmcid(7), isbn(8), lccn(9),
                title(10), authors_json(11), year(12), venue(13).

    Input data can be unreliable (LLM hallucinations, GROBID misextraction,
    bib typos), so we cross-check every field present on both sides.
    Cache hit only when all available data agrees. All text comparisons
    use deterministic normalization — no fuzzy scoring, no thresholds.
    """
    from sciwrite_lint.references.matching import _normalize

    stored_ids = dict(zip(_ID_COLS, row[4:10]))
    for col, query_val in query_ids.items():
        if not query_val:
            continue
        stored_val = stored_ids.get(col)
        if stored_val and stored_val != query_val:
            logger.debug(
                "Dedup conflict on {}: query={!r} vs stored={!r} for ref_key={!r}",
                col,
                query_val,
                stored_val,
                row[0],
            )
            return False

    if title:
        stored_title = row[10]
        if stored_title:
            if _normalize(title) != _normalize(stored_title):
                logger.debug(
                    "Dedup conflict on title for ref_key={!r}",
                    row[0],
                )
                return False

    if authors:
        stored_authors_json = row[11]
        if stored_authors_json:
            stored_authors = json.loads(stored_authors_json)
            q_norm = sorted(_normalize(a) for a in authors)
            s_norm = sorted(_normalize(a) for a in stored_authors)
            if q_norm != s_norm:
                logger.debug(
                    "Dedup conflict on authors for ref_key={!r}",
                    row[0],
                )
                return False

    if year:
        stored_year = row[12]
        if stored_year and stored_year != year:
            logger.debug(
                "Dedup conflict on year: query={!r} vs stored={!r} for ref_key={!r}",
                year,
                stored_year,
                row[0],
            )
            return False

    if venue:
        stored_venue = row[13]
        if stored_venue:
            if _normalize(venue) != _normalize(stored_venue):
                logger.debug(
                    "Dedup conflict on venue for ref_key={!r}",
                    row[0],
                )
                return False

    return True


def load_bibliography_entries(
    conn: sqlite3.Connection,
    parent_key: str,
    depth: int = 1,
) -> list[dict[str, Any]]:
    """Load bibliography entries registered under a parent reference.

    Returns all ref_registry rows where parent_key and depth match,
    each as a dict with structured metadata fields.
    """
    rows = conn.execute(
        "SELECT ref_key, doi, arxiv_id, pmid, pmcid, isbn, lccn, "
        "title, authors_json, year, venue "
        "FROM ref_registry WHERE parent_key = ? AND depth = ? "
        "ORDER BY ref_key",
        (parent_key, depth),
    ).fetchall()

    cols = (
        "ref_key",
        "doi",
        "arxiv_id",
        "pmid",
        "pmcid",
        "isbn",
        "lccn",
        "title",
        "authors_json",
        "year",
        "venue",
    )
    results: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(zip(cols, row))
        aj = entry.pop("authors_json")
        entry["authors"] = json.loads(aj) if aj else []
        results.append(entry)
    return results
