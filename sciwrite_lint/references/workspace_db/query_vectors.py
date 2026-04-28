"""Query vector cache CRUD (pre-computed claim query embeddings)."""

from __future__ import annotations

import sqlite3


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
