"""Manuscript inline-citations CRUD.

Persists ``ManuscriptContext.inline_citations`` so the embedding subprocess
can read claim contexts directly from workspace.db instead of receiving them
across the process boundary as JSON. One row per inline citation occurrence
(the same key may appear multiple times). ``line`` is NULL for PDF-derived
citations.
"""

from __future__ import annotations

import hashlib
import sqlite3

from pydantic import BaseModel

from sciwrite_lint.references.workspace_db.query_vectors import load_query_vector


class ManuscriptCitation(BaseModel):
    """One inline citation occurrence, as persisted."""

    ref_key: str
    line: int | None
    section: str
    context: str


def replace_manuscript_citations(
    conn: sqlite3.Connection,
    source_type: str,
    source_hash: str,
    citations: list[ManuscriptCitation],
) -> None:
    """Replace all rows for the manuscript with the given citations.

    Truncate-and-insert: the table reflects the most recently built
    ManuscriptContext for the workspace. ``source_type`` is one of
    ``latex`` / ``pdf`` / ``markdown``. ``source_hash`` records the source
    file's hash for traceability (empty string is acceptable when the
    caller has no hash to record).
    """
    conn.execute("DELETE FROM manuscript_citations")
    if citations:
        conn.executemany(
            "INSERT INTO manuscript_citations "
            "(ref_key, line, section, context, source_type, source_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (c.ref_key, c.line, c.section, c.context, source_type, source_hash)
                for c in citations
            ],
        )
    conn.commit()


def load_unique_contexts(conn: sqlite3.Connection) -> list[str]:
    """Return distinct non-empty citation contexts.

    Used by the embedding subprocess to compute query vectors for claim
    retrieval. Order is stable (sorted) so encoding is deterministic.
    """
    rows = conn.execute(
        "SELECT DISTINCT context FROM manuscript_citations "
        "WHERE context != '' ORDER BY context"
    ).fetchall()
    return [row[0] for row in rows]


def find_unembedded_contexts(conn: sqlite3.Connection, model_name: str) -> list[str]:
    """Return persisted contexts that lack a query vector for ``model_name``.

    Single source of truth for the "what still needs encoding?" check —
    used both by the parent (to decide whether spawning the embedding
    subprocess is worth it) and by the subprocess itself (to encode only
    the missing subset, while the model is loaded).
    """
    return [
        c
        for c in load_unique_contexts(conn)
        if load_query_vector(conn, hashlib.sha256(c.encode()).hexdigest(), model_name)
        is None
    ]


def count_manuscript_citations(conn: sqlite3.Connection) -> int:
    """Return the number of rows in ``manuscript_citations``."""
    row = conn.execute("SELECT COUNT(*) FROM manuscript_citations").fetchone()
    return int(row[0]) if row else 0
