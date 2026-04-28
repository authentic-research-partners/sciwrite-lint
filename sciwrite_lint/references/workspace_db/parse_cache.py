"""Parse cache CRUD (replaces per-key .meta.json files)."""

from __future__ import annotations

import sqlite3
from typing import Any


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
