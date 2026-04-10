"""Per-paper workspace SQLite database.

Single DB per paper workspace (``references/{paper}/parsed/workspace.db``)
that holds all per-paper persistent data: embeddings, chunk metadata,
reference registry, and future tables.

Migrates automatically from the legacy ``embeddings.db`` name on first open.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from sciwrite_lint.models import CitationMetadata

DB_NAME = "workspace.db"


def db_path(references_dir: Path) -> Path:
    """Return the workspace DB file path for a paper's references dir."""
    return references_dir / "parsed" / DB_NAME


# ---------------------------------------------------------------------------
# Embedding tables (managed by embedding_store.py)
# ---------------------------------------------------------------------------

_EMBEDDING_SCHEMA = """\
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_key TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    section_title TEXT NOT NULL DEFAULT '',
    granularity TEXT NOT NULL DEFAULT 'paragraph',
    start_char INTEGER NOT NULL DEFAULT 0,
    text_len INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_key ON chunks(ref_key);

CREATE TABLE IF NOT EXISTS embed_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ref_status (
    ref_key TEXT PRIMARY KEY,
    expected_chunks INTEGER NOT NULL,
    stored_chunks INTEGER NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Reference registry (cross-depth dedup)
# ---------------------------------------------------------------------------

_REGISTRY_SCHEMA = """\
CREATE TABLE IF NOT EXISTS ref_registry (
    ref_key TEXT NOT NULL,
    doi TEXT,
    arxiv_id TEXT,
    pmid TEXT,
    pmcid TEXT,
    isbn TEXT,
    lccn TEXT,
    title TEXT,
    authors_json TEXT,
    year TEXT,
    venue TEXT,
    depth INTEGER NOT NULL DEFAULT 0,
    parent_key TEXT,
    workspace_path TEXT NOT NULL,
    PRIMARY KEY (ref_key, depth, parent_key)
);
CREATE INDEX IF NOT EXISTS idx_reg_doi ON ref_registry(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_reg_arxiv ON ref_registry(arxiv_id) WHERE arxiv_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_reg_pmid ON ref_registry(pmid) WHERE pmid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_reg_pmcid ON ref_registry(pmcid) WHERE pmcid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_reg_isbn ON ref_registry(isbn) WHERE isbn IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_reg_lccn ON ref_registry(lccn) WHERE lccn IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_reg_parent ON ref_registry(parent_key, depth)
    WHERE parent_key IS NOT NULL;
"""


# ---------------------------------------------------------------------------
# Bibliography verification results
# ---------------------------------------------------------------------------

_BIB_CHECKS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS bib_checks (
    ref_key TEXT PRIMARY KEY,
    parse_hash TEXT NOT NULL DEFAULT '',
    total_entries INTEGER NOT NULL,
    found INTEGER NOT NULL,
    not_found INTEGER NOT NULL,
    retracted INTEGER NOT NULL DEFAULT 0,
    metadata_mismatches INTEGER NOT NULL DEFAULT 0,
    mismatch_details_json TEXT NOT NULL DEFAULT '[]'
);
"""


# ---------------------------------------------------------------------------
# Parse cache (migrated from per-key .meta.json files)
# ---------------------------------------------------------------------------

_PARSE_CACHE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS parse_cache (
    ref_key TEXT PRIMARY KEY,
    pdf_hash TEXT NOT NULL,
    parse_date TEXT NOT NULL DEFAULT '',
    parser TEXT NOT NULL DEFAULT 'grobid',
    sections_count INTEGER NOT NULL DEFAULT 0,
    char_count INTEGER NOT NULL DEFAULT 0,
    is_formal INTEGER NOT NULL DEFAULT 1,
    has_embeddings INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT NOT NULL DEFAULT '',
    chunks_count INTEGER NOT NULL DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Ref internal cache (migrated from per-key .internal.json files)
# ---------------------------------------------------------------------------

_REF_INTERNAL_CACHE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS ref_internal_cache (
    ref_key TEXT PRIMARY KEY,
    md_hash TEXT NOT NULL,
    cache_version TEXT NOT NULL DEFAULT '1',
    internal_score REAL NOT NULL DEFAULT 0.0,
    contribution_score REAL,
    contribution_json TEXT NOT NULL DEFAULT '{}',
    findings_json TEXT NOT NULL DEFAULT '[]',
    sections_found INTEGER NOT NULL DEFAULT 0,
    checks_run_json TEXT NOT NULL DEFAULT '[]'
);
"""


# ---------------------------------------------------------------------------
# Claim results (migrated from claims_{paper}.json files)
# ---------------------------------------------------------------------------

_CLAIM_RESULTS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS claim_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_key TEXT NOT NULL,
    claim_text TEXT NOT NULL DEFAULT '',
    line INTEGER,
    context TEXT NOT NULL DEFAULT '',
    verdict TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    relevant_quote TEXT NOT NULL DEFAULT '',
    explanation TEXT NOT NULL DEFAULT '',
    backend TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    ref_type TEXT NOT NULL DEFAULT '',
    cite_purpose TEXT NOT NULL DEFAULT '',
    dismissed INTEGER NOT NULL DEFAULT 0,
    reviewer_comment TEXT NOT NULL DEFAULT '',
    dismissed_date TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_claims_key ON claim_results(ref_key);
CREATE INDEX IF NOT EXISTS idx_claims_verdict ON claim_results(verdict);
"""


# ---------------------------------------------------------------------------
# Vision cache (figure descriptions from Qwen3-VL-2B)
# ---------------------------------------------------------------------------

_VISION_CACHE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS vision_cache (
    image_key TEXT NOT NULL,
    source TEXT NOT NULL,
    image_hash TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    caption TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    figure_type TEXT NOT NULL DEFAULT '',
    readability_issues_json TEXT NOT NULL DEFAULT '[]',
    cache_version TEXT NOT NULL DEFAULT '3',
    PRIMARY KEY (image_key, source)
);
"""


# ---------------------------------------------------------------------------
# Citation metadata (migrated from per-key JSON files)
# ---------------------------------------------------------------------------

_CITATION_METADATA_SCHEMA = """\
CREATE TABLE IF NOT EXISTS citation_metadata (
    ref_key TEXT PRIMARY KEY,
    verified_date TEXT NOT NULL DEFAULT '',
    api_source TEXT NOT NULL DEFAULT '',
    api_match TEXT NOT NULL DEFAULT '',
    canonical_json TEXT NOT NULL DEFAULT '{}',
    bibitem_json TEXT NOT NULL DEFAULT '{}',
    access_json TEXT NOT NULL DEFAULT '{}',
    mismatches_json TEXT NOT NULL DEFAULT '[]',
    issues_json TEXT NOT NULL DEFAULT '[]',
    manual_override_json TEXT NOT NULL DEFAULT '{}'
);
"""

_QUERY_VECTORS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS query_vectors (
    text_hash TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    vector BLOB NOT NULL
);
"""


_PIPELINE_STAGE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS pipeline_stage (
    stage TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    start_time REAL,
    end_time REAL,
    detail TEXT NOT NULL DEFAULT ''
);
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def open_db(references_dir: Path) -> sqlite3.Connection:
    """Open (or create) the workspace DB with sqlite-vec loaded.

    Creates all schema tables if they don't exist.

    **Do not call directly** — use ``get_db()`` context manager instead,
    which guarantees the connection is closed on exit.
    """
    import sqlite_vec

    db_file = db_path(references_dir)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file), timeout=10.0)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_EMBEDDING_SCHEMA)
    conn.executescript(_REGISTRY_SCHEMA)
    conn.executescript(_BIB_CHECKS_SCHEMA)
    conn.executescript(_PARSE_CACHE_SCHEMA)
    conn.executescript(_REF_INTERNAL_CACHE_SCHEMA)
    conn.executescript(_CLAIM_RESULTS_SCHEMA)
    conn.executescript(_CITATION_METADATA_SCHEMA)
    conn.executescript(_VISION_CACHE_SCHEMA)
    _migrate_vision_cache(conn)
    conn.executescript(_QUERY_VECTORS_SCHEMA)
    conn.executescript(_PIPELINE_STAGE_SCHEMA)

    return conn


@contextmanager
def get_db(references_dir: Path) -> Iterator[sqlite3.Connection]:
    """Context manager for workspace DB connections.

    This is the **only** way to obtain a DB connection outside this module.
    Guarantees cleanup on exit (including exceptions and early returns).

    Usage::

        with get_db(references_dir) as conn:
            save_parse_cache(conn, ...)
    """
    conn = open_db(references_dir)
    try:
        yield conn
    finally:
        conn.close()


def serialize_f32(vec: list[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Reference registry operations
# ---------------------------------------------------------------------------


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


_RESULT_COLS = ("ref_key", "workspace_path", "depth", "parent_key")
_ID_COLS = ("doi", "arxiv_id", "pmid", "pmcid", "isbn", "lccn")


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

    # Cross-check structured IDs
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

    # Cross-check title (normalized exact match)
    if title:
        stored_title = row[10]
        if stored_title:
            if _normalize(title) != _normalize(stored_title):
                logger.debug(
                    "Dedup conflict on title for ref_key={!r}",
                    row[0],
                )
                return False

    # Cross-check authors (normalized, sorted exact match)
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

    # Cross-check year (exact string match)
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

    # Cross-check venue (normalized exact match)
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
        # Deserialize authors
        aj = entry.pop("authors_json")
        entry["authors"] = json.loads(aj) if aj else []
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Bibliography verification results
# ---------------------------------------------------------------------------


def save_bib_checks(
    conn: sqlite3.Connection,
    checks: list[dict[str, Any]],
    parse_hashes: dict[str, str] | None = None,
) -> None:
    """Save bibliography verification results to workspace.db.

    Each check is a RefBibCheck dict. parse_hashes maps ref_key to the
    parsed markdown hash — used for cache invalidation on re-parse.
    """
    hashes = parse_hashes or {}
    for c in checks:
        conn.execute(
            """INSERT INTO bib_checks
               (ref_key, parse_hash, total_entries, found, not_found, retracted,
                metadata_mismatches, mismatch_details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (ref_key) DO UPDATE SET
                 parse_hash = excluded.parse_hash,
                 total_entries = excluded.total_entries,
                 found = excluded.found,
                 not_found = excluded.not_found,
                 retracted = excluded.retracted,
                 metadata_mismatches = excluded.metadata_mismatches,
                 mismatch_details_json = excluded.mismatch_details_json
            """,
            (
                c["key"],
                hashes.get(c["key"], ""),
                c["total_entries"],
                c["found"],
                c["not_found"],
                c.get("retracted", 0),
                c.get("metadata_mismatches", 0),
                json.dumps(c.get("mismatch_details", [])),
            ),
        )
    conn.commit()


def load_bib_checks(
    conn: sqlite3.Connection,
    parse_hashes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Load bibliography verification results from workspace.db.

    If parse_hashes is provided, only returns results where the stored
    hash matches the current parse hash (invalidates stale entries).
    """
    rows = conn.execute(
        "SELECT ref_key, parse_hash, total_entries, found, not_found, retracted, "
        "metadata_mismatches, mismatch_details_json FROM bib_checks"
    ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        # Invalidate if parse hash changed
        if parse_hashes:
            current_hash = parse_hashes.get(row[0])
            if current_hash and row[1] and current_hash != row[1]:
                continue  # stale — re-parsed since last bib check

        results.append(
            {
                "key": row[0],
                "total_entries": row[2],
                "found": row[3],
                "not_found": row[4],
                "retracted": row[5],
                "metadata_mismatches": row[6],
                "mismatch_details": json.loads(row[7]),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Citation metadata CRUD
# ---------------------------------------------------------------------------


def save_citation_metadata(
    conn: sqlite3.Connection,
    meta: CitationMetadata,
) -> None:
    """Upsert a CitationMetadata record into workspace.db."""
    conn.execute(
        """INSERT INTO citation_metadata
           (ref_key, verified_date, api_source, api_match,
            canonical_json, bibitem_json, access_json,
            mismatches_json, issues_json, manual_override_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (ref_key) DO UPDATE SET
             verified_date = excluded.verified_date,
             api_source = excluded.api_source,
             api_match = excluded.api_match,
             canonical_json = excluded.canonical_json,
             bibitem_json = excluded.bibitem_json,
             access_json = excluded.access_json,
             mismatches_json = excluded.mismatches_json,
             issues_json = excluded.issues_json,
             manual_override_json = excluded.manual_override_json
        """,
        (
            meta.key,
            meta.verified_date,
            meta.api_source,
            meta.api_match,
            json.dumps(meta.canonical, ensure_ascii=False),
            json.dumps(meta.bibitem, ensure_ascii=False),
            json.dumps(meta.access, ensure_ascii=False),
            json.dumps(meta.mismatches, ensure_ascii=False),
            json.dumps(meta.issues, ensure_ascii=False),
            json.dumps(meta.manual_override, ensure_ascii=False),
        ),
    )
    conn.commit()


def load_citation_metadata(
    conn: sqlite3.Connection,
    ref_key: str,
) -> "CitationMetadata | None":
    """Load a single CitationMetadata by key. Returns None if not found."""
    row = conn.execute(
        "SELECT ref_key, verified_date, api_source, api_match, "
        "canonical_json, bibitem_json, access_json, "
        "mismatches_json, issues_json, manual_override_json "
        "FROM citation_metadata WHERE ref_key = ?",
        (ref_key,),
    ).fetchone()
    if not row:
        return None
    return _row_to_citation_metadata(row)


def load_all_citation_metadata(
    conn: sqlite3.Connection,
) -> "dict[str, CitationMetadata]":
    """Load all CitationMetadata records. Returns {key: CitationMetadata}."""
    rows = conn.execute(
        "SELECT ref_key, verified_date, api_source, api_match, "
        "canonical_json, bibitem_json, access_json, "
        "mismatches_json, issues_json, manual_override_json "
        "FROM citation_metadata"
    ).fetchall()
    result: dict[str, CitationMetadata] = {}
    for row in rows:
        meta = _row_to_citation_metadata(row)
        result[meta.key] = meta
    return result


def delete_citation_metadata(conn: sqlite3.Connection, ref_key: str) -> None:
    """Delete a single citation metadata record (used by --fresh)."""
    conn.execute("DELETE FROM citation_metadata WHERE ref_key = ?", (ref_key,))
    conn.commit()


def _row_to_citation_metadata(row: tuple[Any, ...]) -> CitationMetadata:
    """Convert a DB row to CitationMetadata."""
    from sciwrite_lint.models import CitationMetadata

    return CitationMetadata(
        key=row[0],
        verified_date=row[1],
        api_source=row[2],
        api_match=row[3],
        canonical=json.loads(row[4]),
        bibitem=json.loads(row[5]),
        access=json.loads(row[6]),
        mismatches=json.loads(row[7]),
        issues=json.loads(row[8]),
        manual_override=json.loads(row[9]),
    )


# ---------------------------------------------------------------------------
# Targeted citation metadata queries
# ---------------------------------------------------------------------------


def query_verified_metadata(
    conn: sqlite3.Connection,
) -> dict[str, CitationMetadata]:
    """Load only already-verified metadata (verified/mismatch/not_found/web_*)."""
    rows = conn.execute(
        "SELECT ref_key, verified_date, api_source, api_match, "
        "canonical_json, bibitem_json, access_json, "
        "mismatches_json, issues_json, manual_override_json "
        "FROM citation_metadata "
        "WHERE api_match IN ('verified', 'mismatch', 'not_found', "
        "'web_verified', 'web_dead')"
    ).fetchall()
    return {row[0]: _row_to_citation_metadata(row) for row in rows}


def query_refs_with_local_pdfs(
    conn: sqlite3.Connection,
) -> dict[str, tuple[str, str]]:
    """Return {ref_key: (local_file, entry_type)} for refs with local PDFs.

    Uses json_extract on access_json to filter server-side.
    """
    rows = conn.execute(
        "SELECT ref_key, "
        "json_extract(access_json, '$.local_file'), "
        "COALESCE(json_extract(bibitem_json, '$.entry_type'), '') "
        "FROM citation_metadata "
        "WHERE json_extract(access_json, '$.local_file') LIKE '%.pdf'"
    ).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def query_refs_by_match(
    conn: sqlite3.Connection,
    api_match: str,
) -> dict[str, CitationMetadata]:
    """Load metadata for refs with a specific api_match value."""
    rows = conn.execute(
        "SELECT ref_key, verified_date, api_source, api_match, "
        "canonical_json, bibitem_json, access_json, "
        "mismatches_json, issues_json, manual_override_json "
        "FROM citation_metadata WHERE api_match = ?",
        (api_match,),
    ).fetchall()
    return {row[0]: _row_to_citation_metadata(row) for row in rows}


def query_retracted_refs(
    conn: sqlite3.Connection,
) -> dict[str, CitationMetadata]:
    """Load only refs that have a retraction_status in canonical."""
    rows = conn.execute(
        "SELECT ref_key, verified_date, api_source, api_match, "
        "canonical_json, bibitem_json, access_json, "
        "mismatches_json, issues_json, manual_override_json "
        "FROM citation_metadata "
        "WHERE json_extract(canonical_json, '$.retraction_status') IS NOT NULL"
    ).fetchall()
    return {row[0]: _row_to_citation_metadata(row) for row in rows}


def query_refs_with_mismatches(
    conn: sqlite3.Connection,
) -> dict[str, CitationMetadata]:
    """Load only refs that have non-empty mismatches."""
    rows = conn.execute(
        "SELECT ref_key, verified_date, api_source, api_match, "
        "canonical_json, bibitem_json, access_json, "
        "mismatches_json, issues_json, manual_override_json "
        "FROM citation_metadata WHERE mismatches_json != '[]'"
    ).fetchall()
    return {row[0]: _row_to_citation_metadata(row) for row in rows}


# ---------------------------------------------------------------------------
# Parse cache CRUD
# ---------------------------------------------------------------------------


def save_parse_cache(
    conn: sqlite3.Connection,
    ref_key: str,
    *,
    pdf_hash: str,
    parse_date: str = "",
    parser: str = "grobid",
    sections_count: int = 0,
    char_count: int = 0,
    is_formal: bool = False,
    has_embeddings: bool = False,
    embedding_model: str = "",
    chunks_count: int = 0,
) -> None:
    """Upsert a parse cache record."""
    conn.execute(
        """INSERT INTO parse_cache
           (ref_key, pdf_hash, parse_date, parser, sections_count, char_count,
            is_formal, has_embeddings, embedding_model, chunks_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (ref_key) DO UPDATE SET
             pdf_hash = excluded.pdf_hash,
             parse_date = excluded.parse_date,
             parser = excluded.parser,
             sections_count = excluded.sections_count,
             char_count = excluded.char_count,
             is_formal = excluded.is_formal,
             has_embeddings = excluded.has_embeddings,
             embedding_model = excluded.embedding_model,
             chunks_count = excluded.chunks_count
        """,
        (
            ref_key,
            pdf_hash,
            parse_date,
            parser,
            sections_count,
            char_count,
            int(is_formal),
            int(has_embeddings),
            embedding_model,
            chunks_count,
        ),
    )
    conn.commit()


def load_parse_cache(
    conn: sqlite3.Connection,
    ref_key: str,
) -> dict[str, Any] | None:
    """Load a single parse cache record. Returns dict or None."""
    row = conn.execute(
        "SELECT ref_key, pdf_hash, parse_date, parser, sections_count, char_count, "
        "is_formal, has_embeddings, embedding_model, chunks_count "
        "FROM parse_cache WHERE ref_key = ?",
        (ref_key,),
    ).fetchone()
    if not row:
        return None
    return {
        "ref_key": row[0],
        "pdf_hash": row[1],
        "parse_date": row[2],
        "parser": row[3],
        "sections_count": row[4],
        "char_count": row[5],
        "is_formal": bool(row[6]),
        "has_embeddings": bool(row[7]),
        "embedding_model": row[8],
        "chunks_count": row[9],
    }


def load_all_parse_cache(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    """Load all parse cache records. Returns {ref_key: dict}."""
    rows = conn.execute(
        "SELECT ref_key, pdf_hash, parse_date, parser, sections_count, char_count, "
        "is_formal, has_embeddings, embedding_model, chunks_count "
        "FROM parse_cache"
    ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[row[0]] = {
            "pdf_hash": row[1],
            "parse_date": row[2],
            "parser": row[3],
            "sections_count": row[4],
            "char_count": row[5],
            "is_formal": bool(row[6]),
            "has_embeddings": bool(row[7]),
            "embedding_model": row[8],
            "chunks_count": row[9],
        }
    return result


def update_parse_cache_embeddings(
    conn: sqlite3.Connection,
    ref_key: str,
    *,
    has_embeddings: bool,
    embedding_model: str,
    chunks_count: int,
) -> None:
    """Update embedding fields on an existing parse cache record."""
    conn.execute(
        "UPDATE parse_cache SET has_embeddings = ?, embedding_model = ?, "
        "chunks_count = ? WHERE ref_key = ?",
        (int(has_embeddings), embedding_model, chunks_count, ref_key),
    )
    conn.commit()


def is_formal_cached_db(
    conn: sqlite3.Connection,
    ref_key: str,
) -> bool:
    """Check if a parsed reference is a formal academic document via DB."""
    row = conn.execute(
        "SELECT is_formal FROM parse_cache WHERE ref_key = ?",
        (ref_key,),
    ).fetchone()
    if not row:
        return False
    return bool(row[0])


# ---------------------------------------------------------------------------
# Ref internal cache CRUD
# ---------------------------------------------------------------------------


def save_ref_internal_cache(
    conn: sqlite3.Connection,
    ref_key: str,
    *,
    md_hash: str,
    cache_version: str,
    internal_score: float,
    contribution_score: float | None = None,
    contribution_json: str = "{}",
    findings_json: str = "[]",
    sections_found: int = 0,
    checks_run_json: str = "[]",
) -> None:
    """Upsert a ref internal cache record."""
    conn.execute(
        """INSERT INTO ref_internal_cache
           (ref_key, md_hash, cache_version, internal_score, contribution_score,
            contribution_json, findings_json, sections_found, checks_run_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (ref_key) DO UPDATE SET
             md_hash = excluded.md_hash,
             cache_version = excluded.cache_version,
             internal_score = excluded.internal_score,
             contribution_score = excluded.contribution_score,
             contribution_json = excluded.contribution_json,
             findings_json = excluded.findings_json,
             sections_found = excluded.sections_found,
             checks_run_json = excluded.checks_run_json
        """,
        (
            ref_key,
            md_hash,
            cache_version,
            internal_score,
            contribution_score,
            contribution_json,
            findings_json,
            sections_found,
            checks_run_json,
        ),
    )
    conn.commit()


def load_ref_internal_cache(
    conn: sqlite3.Connection,
    ref_key: str,
    *,
    expected_version: str,
    expected_md_hash: str,
) -> dict[str, Any] | None:
    """Load a ref internal cache record if version and hash match."""
    row = conn.execute(
        "SELECT ref_key, md_hash, cache_version, internal_score, contribution_score, "
        "contribution_json, findings_json, sections_found, checks_run_json "
        "FROM ref_internal_cache WHERE ref_key = ?",
        (ref_key,),
    ).fetchone()
    if not row:
        return None
    if row[2] != expected_version:
        return None
    if row[1] != expected_md_hash:
        return None
    return {
        "ref_key": row[0],
        "md_hash": row[1],
        "cache_version": row[2],
        "internal_score": row[3],
        "contribution_score": row[4] if row[4] is not None else 1.0,
        "contribution_json": row[5],
        "findings_json": row[6],
        "sections_found": row[7],
        "checks_run_json": row[8],
    }


def load_all_ref_internal_scores(
    conn: sqlite3.Connection,
) -> dict[str, float]:
    """Load all ref internal scores as {ref_key: combined_score}.

    Combined score = internal_score × contribution_score.
    """
    rows = conn.execute(
        "SELECT ref_key, internal_score, contribution_score FROM ref_internal_cache"
    ).fetchall()
    result: dict[str, float] = {}
    for row in rows:
        i_score = row[1]
        c_score = row[2] if row[2] is not None else 1.0
        result[row[0]] = i_score * c_score
    return result


# ---------------------------------------------------------------------------
# Claim results CRUD
# ---------------------------------------------------------------------------


def save_claim_results(
    conn: sqlite3.Connection,
    results: list[dict[str, Any]],
) -> None:
    """Replace all claim results with new data.

    Deletes existing rows first (full replacement per run).
    """
    conn.execute("DELETE FROM claim_results")
    for r in results:
        conn.execute(
            """INSERT INTO claim_results
               (ref_key, claim_text, line, context, verdict, confidence,
                relevant_quote, explanation, backend, model, ref_type,
                cite_purpose, dismissed, reviewer_comment, dismissed_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.get("key", ""),
                r.get("claim_text", ""),
                r.get("line"),
                r.get("context", ""),
                r.get("verdict", ""),
                r.get("confidence", 0.0),
                r.get("relevant_quote", ""),
                r.get("explanation", ""),
                r.get("backend", ""),
                r.get("model", ""),
                r.get("ref_type", ""),
                r.get("cite_purpose", r.get("citation_purpose", "")),
                int(r.get("dismissed", False)),
                r.get("reviewer_comment", ""),
                r.get("dismissed_date", ""),
            ),
        )
    conn.commit()


def load_claim_results(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Load all claim results as a list of dicts."""
    rows = conn.execute(
        "SELECT id, ref_key, claim_text, line, context, verdict, confidence, "
        "relevant_quote, explanation, backend, model, ref_type, cite_purpose, "
        "dismissed, reviewer_comment, dismissed_date "
        "FROM claim_results ORDER BY id"
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        d: dict[str, Any] = {
            "id": row[0],
            "key": row[1],
            "claim_text": row[2],
            "line": row[3],
            "context": row[4],
            "verdict": row[5],
            "confidence": row[6],
            "relevant_quote": row[7],
            "explanation": row[8],
            "backend": row[9],
            "model": row[10],
            "ref_type": row[11],
            "cite_purpose": row[12],
        }
        if row[13]:
            d["dismissed"] = True
            d["reviewer_comment"] = row[14]
            d["dismissed_date"] = row[15]
        results.append(d)
    return results


def dismiss_claim(
    conn: sqlite3.Connection,
    claim_id: int,
    *,
    reason: str,
    date_str: str,
) -> bool:
    """Mark a claim as dismissed. Returns True if found."""
    cur = conn.execute(
        "UPDATE claim_results SET dismissed = 1, reviewer_comment = ?, "
        "dismissed_date = ? WHERE id = ?",
        (reason, date_str, claim_id),
    )
    conn.commit()
    return cur.rowcount > 0


def clear_claim_dismissal(
    conn: sqlite3.Connection,
    claim_id: int,
) -> bool:
    """Clear dismissal on a claim. Returns True if found."""
    cur = conn.execute(
        "UPDATE claim_results SET dismissed = 0, reviewer_comment = '', "
        "dismissed_date = '' WHERE id = ?",
        (claim_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def find_claim(
    conn: sqlite3.Connection,
    ref_key: str,
    line: int,
) -> dict[str, Any] | None:
    """Find a claim by ref_key and line number."""
    row = conn.execute(
        "SELECT id, ref_key, claim_text, line, context, verdict, confidence, "
        "relevant_quote, explanation, backend, model, ref_type, cite_purpose, "
        "dismissed, reviewer_comment, dismissed_date "
        "FROM claim_results WHERE ref_key = ? AND line = ? LIMIT 1",
        (ref_key, line),
    ).fetchone()
    if not row:
        return None
    d: dict[str, Any] = {
        "id": row[0],
        "key": row[1],
        "claim_text": row[2],
        "line": row[3],
        "context": row[4],
        "verdict": row[5],
        "confidence": row[6],
        "relevant_quote": row[7],
        "explanation": row[8],
        "backend": row[9],
        "model": row[10],
        "ref_type": row[11],
        "cite_purpose": row[12],
    }
    if row[13]:
        d["dismissed"] = True
        d["reviewer_comment"] = row[14]
        d["dismissed_date"] = row[15]
    return d


def list_claims_for_key(
    conn: sqlite3.Connection,
    ref_key: str,
) -> list[dict[str, Any]]:
    """List all claims for a given ref_key."""
    rows = conn.execute(
        "SELECT id, line, verdict, context FROM claim_results WHERE ref_key = ?",
        (ref_key,),
    ).fetchall()
    return [
        {"id": row[0], "line": row[1], "verdict": row[2], "context": row[3]}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Vision cache operations
# ---------------------------------------------------------------------------

_VISION_CACHE_VERSION = "3"


def _migrate_vision_cache(conn: sqlite3.Connection) -> None:
    """Migrate vision_cache to the current schema.

    Old schema had ``image_key TEXT PRIMARY KEY`` without a ``source``
    column.  Since vision cache is fully rebuildable (``--fresh``), we
    drop and recreate when the old schema is detected.

    Also adds structured output columns if missing from older DBs:
    - ``figure_type`` (v2 onwards)
    - ``readability_issues_json`` (v3 onwards — structured list; the
      legacy ``readability_issues`` TEXT column from v2, if present, is
      left alone because v2 rows are ignored by cache_version filter).
    """
    # Check if source column exists
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(vision_cache)").fetchall()
    }
    if "source" not in cols:
        # Old schema — drop and recreate with composite PK
        conn.execute("DROP TABLE vision_cache")
        conn.executescript(_VISION_CACHE_SCHEMA)
        logger.debug("Migrated vision_cache to composite PK (image_key, source)")
        return

    # Add structured output columns if missing.
    for col, default in [
        ("figure_type", "''"),
        ("readability_issues_json", "'[]'"),
    ]:
        try:
            conn.execute(
                f"ALTER TABLE vision_cache ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists


def save_vision_entry(
    conn: sqlite3.Connection,
    image_key: str,
    *,
    image_hash: str,
    source: str = "manuscript",
    label: str = "",
    caption: str = "",
    description: str = "",
    figure_type: str = "",
    readability_issues: list[str] | None = None,
) -> None:
    """Upsert a single vision cache entry.

    ``source`` discriminates manuscript figures (``"manuscript"``) from
    cited reference figures (the ref_key, e.g. ``"tanaka2017"``).
    Together with ``image_key``, it forms the composite primary key.

    ``readability_issues`` is a list of short issue strings (empty list
    means no issues). Serialized as JSON into the
    ``readability_issues_json`` column on write.
    """
    issues_json = json.dumps(readability_issues or [])
    conn.execute(
        """INSERT INTO vision_cache
           (image_key, source, image_hash, label, caption, description,
            figure_type, readability_issues_json, cache_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (image_key, source) DO UPDATE SET
             image_hash = excluded.image_hash,
             label = excluded.label,
             caption = excluded.caption,
             description = excluded.description,
             figure_type = excluded.figure_type,
             readability_issues_json = excluded.readability_issues_json,
             cache_version = excluded.cache_version
        """,
        (
            image_key,
            source,
            image_hash,
            label,
            caption,
            description,
            figure_type,
            issues_json,
            _VISION_CACHE_VERSION,
        ),
    )
    conn.commit()


def load_vision_entry(
    conn: sqlite3.Connection,
    image_key: str,
    *,
    expected_hash: str,
    source: str = "manuscript",
) -> dict[str, Any] | None:
    """Load a vision cache entry if hash matches. Returns None on miss.

    Looks up by composite PK ``(image_key, source)``. The returned dict
    uses the logical key ``readability_issues`` (a ``list[str]``), with
    JSON deserialization from the storage column
    ``readability_issues_json``.
    """
    row = conn.execute(
        "SELECT image_hash, label, caption, description, "
        "figure_type, readability_issues_json, cache_version "
        "FROM vision_cache WHERE image_key = ? AND source = ?",
        (image_key, source),
    ).fetchone()
    if not row:
        return None
    if row[6] != _VISION_CACHE_VERSION:
        return None
    if row[0] != expected_hash:
        return None
    return {
        "image_hash": row[0],
        "label": row[1],
        "caption": row[2],
        "description": row[3],
        "figure_type": row[4],
        "readability_issues": json.loads(row[5]),
    }


def load_all_vision_entries(
    conn: sqlite3.Connection,
    source: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load vision cache entries. Returns {image_key: {...}}.

    When ``source`` is provided, only entries matching that source are
    returned.  When None, returns all entries (use with care). The
    returned dicts use the logical key ``readability_issues`` (a
    ``list[str]``), deserialized from the storage column
    ``readability_issues_json``.
    """
    query = (
        "SELECT image_key, image_hash, label, caption, description, "
        "figure_type, readability_issues_json "
        "FROM vision_cache WHERE cache_version = ?"
    )
    params: list[str] = [_VISION_CACHE_VERSION]
    if source is not None:
        query += " AND source = ?"
        params.append(source)
    rows = conn.execute(query, params).fetchall()
    return {
        row[0]: {
            "image_hash": row[1],
            "label": row[2],
            "caption": row[3],
            "description": row[4],
            "figure_type": row[5],
            "readability_issues": json.loads(row[6]),
        }
        for row in rows
    }


def clear_vision_cache(conn: sqlite3.Connection) -> None:
    """Delete all vision cache entries (for --fresh)."""
    conn.execute("DELETE FROM vision_cache")
    conn.commit()


# ---------------------------------------------------------------------------
# Pipeline stage tracking
# ---------------------------------------------------------------------------

# Canonical stage names and their display order.
PIPELINE_STAGES: list[str] = [
    "setup",  # Stage 0: workspace + citations + GROBID (PDF)
    "vision",  # Stage 0.5: manuscript figure descriptions
    "text_checks",  # Stage 1a: regex-based text rules
    "llm_checks",  # Stage 1b: vLLM batch checks
    "verify",  # Stage 2: API verification
    "fetch",  # Stage 3: full-text acquisition
    "parse",  # Stage 4: GROBID parse + embeddings
    "cited_vision",  # Stage 4.2: VL on cited paper figures
    "ref_internal",  # Stage 4.5: ref internal consistency (vLLM)
    "bib_verify",  # Stage 4.6: bibliography verification
    "claims",  # Stage 5: claim verification (vLLM)
    "unreliable",  # Stage 6: reference-unreliable aggregation
    "contributions",  # Stage 7: contribution axes scoring (vLLM)
]


def init_pipeline_stages(conn: sqlite3.Connection) -> None:
    """Reset all stages to 'pending' at the start of a run."""
    conn.execute("DELETE FROM pipeline_stage")
    for stage in PIPELINE_STAGES:
        conn.execute(
            "INSERT INTO pipeline_stage (stage, status) VALUES (?, 'pending')",
            (stage,),
        )
    conn.commit()


def update_pipeline_stage(
    conn: sqlite3.Connection,
    stage: str,
    status: str,
    detail: str = "",
) -> None:
    """Update a stage's status. status: 'running', 'done', 'failed', 'skipped'."""
    import time

    if status == "running":
        conn.execute(
            "UPDATE pipeline_stage SET status = ?, start_time = ?, detail = ? "
            "WHERE stage = ?",
            (status, time.time(), detail, stage),
        )
    elif status in ("done", "failed", "skipped"):
        conn.execute(
            "UPDATE pipeline_stage SET status = ?, end_time = ?, detail = ? "
            "WHERE stage = ?",
            (status, time.time(), detail, stage),
        )
    else:
        conn.execute(
            "UPDATE pipeline_stage SET status = ?, detail = ? WHERE stage = ?",
            (status, detail, stage),
        )
    conn.commit()


def load_pipeline_stages(
    conn: sqlite3.Connection,
) -> list[dict[str, str | float | None]]:
    """Load all pipeline stages in order. Returns list of dicts."""
    rows = conn.execute(
        "SELECT stage, status, start_time, end_time, detail "
        "FROM pipeline_stage ORDER BY rowid"
    ).fetchall()
    return [
        {
            "stage": r[0],
            "status": r[1],
            "start_time": r[2],
            "end_time": r[3],
            "detail": r[4],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Query vector cache (pre-computed claim query embeddings)
# ---------------------------------------------------------------------------


def save_query_vector(
    conn: sqlite3.Connection,
    text_hash: str,
    model: str,
    vector: bytes,
) -> None:
    """Store a pre-computed query embedding vector."""
    conn.execute(
        "INSERT OR REPLACE INTO query_vectors (text_hash, model, vector) VALUES (?, ?, ?)",
        (text_hash, model, vector),
    )
    conn.commit()


def load_query_vector(
    conn: sqlite3.Connection,
    text_hash: str,
    model: str,
) -> bytes | None:
    """Load a pre-computed query vector. Returns None on miss or model mismatch."""
    row = conn.execute(
        "SELECT vector FROM query_vectors WHERE text_hash = ? AND model = ?",
        (text_hash, model),
    ).fetchone()
    return row[0] if row else None
