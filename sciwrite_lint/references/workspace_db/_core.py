"""Connection, schema, and bootstrap for the per-paper workspace DB.

All ``_*_SCHEMA`` constants are the source of truth for workspace.db's
shape. ``open_db()`` applies them on every open (CREATE IF NOT EXISTS).
If a table's shape changes, users run ``--fresh`` to rebuild.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

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
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(vision_cache)").fetchall()
    }
    if "source" not in cols:
        conn.execute("DROP TABLE vision_cache")
        conn.executescript(_VISION_CACHE_SCHEMA)
        logger.debug("Migrated vision_cache to composite PK (image_key, source)")
        return

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
