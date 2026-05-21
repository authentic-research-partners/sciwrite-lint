"""SQLite-vec embedding storage for claim verification.

Stores chunk text + embeddings in the per-paper workspace DB
(``references/{paper}/parsed/workspace.db``) using sqlite-vec for KNN
retrieval. Each key's chunks are encoded and persisted in a single
bulk transaction — see ``store_embeddings``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from loguru import logger
from pydantic import BaseModel

from sciwrite_lint.references.workspace_db import (
    db_path as _db_path,
    get_db,
    serialize_f32 as _serialize_f32,
)


class ChunkHit(BaseModel):
    """One KNN hit from the embedding store.

    Used by the verify-claim retrieval helpers in
    ``sciwrite_lint/references/reference_store.py`` and consumed by the
    escalation ladder in ``sciwrite_lint/eval_claims.py``. Carries the
    chunk text the LLM will see plus the locator information
    (``section_title``, ``start_char``) needed to identify the chunk in
    the source after the run.
    """

    text: str
    section_title: str
    granularity: str  # "paragraph" | "sentence"
    start_char: int
    text_len: int
    distance: float
    score: float


_ENCODE_BATCH_SIZE = 32  # uniform batch; ST sorts by length internally


def _encode_adaptive(model: object, texts: list[str]) -> np.ndarray:
    """Encode all texts in one ``model.encode`` call.

    Sentence-transformers sorts inputs by length internally and batches
    the sorted sequence, so a single uniform batch size keeps padding
    overhead minimal — short texts batch with short, long with long.
    Under SDPA + bf16 a batch of 32 max-length chunks costs ~3 GB peak,
    well inside the 20 GB GPU ceiling.
    """
    from sciwrite_lint.references._embed_timing import time_phase

    with time_phase("gpu_encode"):
        return model.encode(  # type: ignore[attr-defined]
            texts,
            normalize_embeddings=True,
            batch_size=_ENCODE_BATCH_SIZE,
        )


def _ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Create the vec0 virtual table if it doesn't exist, or recreate on dimension change."""
    # Check if table exists
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
    ).fetchone()

    if row:
        # Check dimension matches
        stored_dim = conn.execute(
            "SELECT value FROM embed_meta WHERE key='dimension'"
        ).fetchone()
        if stored_dim and int(stored_dim[0]) == dim:
            return
        # Dimension changed — drop and recreate
        logger.info("Embedding dimension changed to {}, rebuilding vec table", dim)
        conn.execute("DROP TABLE chunk_embeddings")

    conn.execute(
        f"CREATE VIRTUAL TABLE chunk_embeddings USING vec0("
        f"embedding float[{dim}] distance_metric=cosine)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO embed_meta (key, value) VALUES ('dimension', ?)",
        (str(dim),),
    )
    conn.commit()


def has_embeddings(
    ref_key: str, references_dir: Path, model_name: str | None = None
) -> bool:
    """Check if a reference has complete, current embeddings.

    Returns False if:
    - No embeddings exist for this key
    - Embedding was interrupted (complete=0) — cleans up broken data
    - Embedding model changed (stored model != current model)

    A fresh workspace DB without the ``ref_status`` / ``embed_meta`` tables
    is treated as "no embeddings yet" — that is the only swallowed
    sqlite error. Any other failure (corrupt DB, sqlite-vec extension
    load failure, locked DB) is allowed to propagate so the caller
    surfaces the real fault instead of silently re-embedding forever.
    """
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return False

    with get_db(references_dir) as conn:
        # Check completion status
        try:
            row = conn.execute(
                "SELECT complete FROM ref_status WHERE ref_key = ?", (ref_key,)
            ).fetchone()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return False
            raise
        if row is None:
            return False
        if row[0] != 1:
            # Incomplete — clean up broken data
            logger.debug("Cleaning up incomplete embeddings for {}", ref_key)
            _delete_ref_data(conn, ref_key)
            conn.commit()
            return False

        # Check model compatibility
        if model_name is not None:
            try:
                stored = conn.execute(
                    "SELECT value FROM embed_meta WHERE key='model'"
                ).fetchone()
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    return False
                raise
            if stored and stored[0] != model_name:
                return False

        return True


def _delete_ref_data(conn: sqlite3.Connection, ref_key: str) -> int:
    """Delete all data for a reference (chunks, vectors, status). Returns count."""
    existing_ids = conn.execute(
        "SELECT id FROM chunks WHERE ref_key = ?", (ref_key,)
    ).fetchall()
    for (chunk_id,) in existing_ids:
        conn.execute("DELETE FROM chunk_embeddings WHERE rowid = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE ref_key = ?", (ref_key,))
    conn.execute("DELETE FROM ref_status WHERE ref_key = ?", (ref_key,))
    return len(existing_ids)


def store_embeddings(
    ref_key: str,
    chunks: list[dict],
    references_dir: Path,
    model_name: str,
    embedding_dim: int,
) -> int:
    """Encode all chunks for a key, then bulk-insert in a single transaction.

    Each chunk dict must have: text, section_title, granularity, start_char.

    Strategy: encode the full chunk list in one ``_encode_adaptive`` call,
    then write rows + embeddings via two ``executemany`` calls inside a
    single transaction (one fsync per key). Partial-key progress isn't
    tracked — resume granularity is per-key via the ``ref_status.complete``
    flag.

    Returns the number of chunks stored.
    """
    from sciwrite_lint.references._embed_timing import time_phase
    from sciwrite_lint.references.reference_store import _get_embedding_model

    with get_db(references_dir) as conn:
        _ensure_vec_table(conn, embedding_dim)

        # Clear existing data for this key (including incomplete runs).
        _delete_ref_data(conn, ref_key)

        # Mark as in-progress (complete=0) with expected count. Committed
        # immediately so a crash mid-encode leaves a tombstone for the next
        # ``has_embeddings`` call to clean up.
        conn.execute(
            "INSERT INTO ref_status (ref_key, expected_chunks, stored_chunks, complete) "
            "VALUES (?, ?, 0, 0)",
            (ref_key, len(chunks)),
        )
        conn.commit()

        if not chunks:
            conn.execute(
                "UPDATE ref_status SET complete = 1 WHERE ref_key = ?", (ref_key,)
            )
            conn.commit()
            return 0

        # Encode every chunk for this key in one pass. ``_encode_adaptive``
        # records the ``tokenize_lengths`` and ``gpu_encode`` phases.
        model = _get_embedding_model()
        texts = [c["text"] for c in chunks]
        vectors = _encode_adaptive(model, texts)

        with time_phase("db_store"):
            # Pre-assign rowids so chunks and chunk_embeddings can be loaded
            # via independent ``executemany`` calls without round-tripping
            # through ``cursor.lastrowid``. SQLite is single-writer and the
            # embedder subprocess is the only writer to this DB, so MAX(id)
            # is stable across the read-then-insert window.
            max_id_row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM chunks"
            ).fetchone()
            max_id = int(max_id_row[0])

            chunk_rows = [
                (
                    max_id + i + 1,
                    ref_key,
                    i,
                    chunk["text"],
                    chunk["section_title"],
                    chunk["granularity"],
                    chunk["start_char"],
                    len(chunk["text"]),
                )
                for i, chunk in enumerate(chunks)
            ]
            embedding_rows = [
                (max_id + i + 1, _serialize_f32(vec.tolist()))
                for i, vec in enumerate(vectors)
            ]

            # Single transaction wrapping both bulk inserts plus the
            # completion update — one fsync for the whole key.
            try:
                conn.execute("BEGIN")
                conn.executemany(
                    "INSERT INTO chunks "
                    "(id, ref_key, chunk_index, text, section_title, "
                    "granularity, start_char, text_len) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    chunk_rows,
                )
                conn.executemany(
                    "INSERT INTO chunk_embeddings (rowid, embedding) VALUES (?, ?)",
                    embedding_rows,
                )
                conn.execute(
                    "UPDATE ref_status SET stored_chunks = ?, complete = 1 "
                    "WHERE ref_key = ?",
                    (len(chunks), ref_key),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO embed_meta (key, value) VALUES ('model', ?)",
                    (model_name,),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return len(chunks)


def retrieve_similar(
    query_text: str,
    ref_key: str,
    references_dir: Path,
    top_k: int = 15,
    granularity: str | None = None,
) -> list[ChunkHit]:
    """Find chunks most similar to *query_text* via sqlite-vec KNN.

    Distance is cosine distance (0 = identical, 2 = opposite); score is
    cosine similarity (``1 - distance``). Returns an empty list if the
    DB is missing, the embedding model has changed since the chunks were
    indexed, or no pre-computed query vector is available.

    When *granularity* is given (``"sentence"`` or ``"paragraph"``), the
    rowid pre-filter additionally scopes the KNN MATCH to that
    granularity, so the verify-claim ladder retrieves the top-N at one
    level without sentence and paragraph chunks competing for the same
    ranking slots.
    """
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return []

    with get_db(references_dir) as conn:
        # Check model compatibility
        stored_model = conn.execute(
            "SELECT value FROM embed_meta WHERE key='model'"
        ).fetchone()
        from sciwrite_lint.references.reference_store import _get_embedding_config

        current_model, _, _ = _get_embedding_config()
        if stored_model and stored_model[0] != current_model:
            logger.warning(
                "Embedding model changed ({} → {}), re-embedding needed",
                stored_model[0],
                current_model,
            )
            return []

        # Load pre-computed query vector from DB (encoded in Stage 4b subprocess).
        # Never load the embedding model in the parent process — it competes
        # with vLLM for VRAM during Stage 5 (claim verification).
        import hashlib

        from sciwrite_lint.references.workspace_db import load_query_vector

        text_hash = hashlib.sha256(query_text.encode()).hexdigest()
        query_blob = load_query_vector(conn, text_hash, current_model)
        if query_blob is None:
            logger.warning(
                "No pre-computed query vector (hash {}) — "
                "run pipeline again to pre-compute",
                text_hash[:12],
            )
            return []

        # KNN search scoped to target ref via rowid pre-filter.
        # sqlite-vec supports `rowid IN (...)` during MATCH, so we restrict
        # the search to only the target ref's chunks — no global scan needed.
        # When granularity is given, the inner SELECT also filters by
        # granularity so the KNN sees only chunks of that level.
        if granularity is not None and granularity not in ("sentence", "paragraph"):
            raise ValueError(
                f"granularity must be 'sentence' or 'paragraph', got {granularity!r}"
            )

        inner_where = "ref_key = ?"
        inner_params: tuple[object, ...] = (ref_key,)
        if granularity is not None:
            inner_where += " AND granularity = ?"
            inner_params = (ref_key, granularity)

        rows = conn.execute(
            f"""
            SELECT c.text, c.section_title, c.granularity, c.start_char,
                   c.text_len, ce.distance
            FROM chunk_embeddings ce
            INNER JOIN chunks c ON c.id = ce.rowid
            WHERE ce.embedding MATCH ?
                AND k = ?
                AND ce.rowid IN (SELECT id FROM chunks WHERE {inner_where})
            ORDER BY ce.distance
            """,
            (query_blob, top_k, *inner_params),
        ).fetchall()

    return [
        ChunkHit(
            text=row[0],
            section_title=row[1],
            granularity=row[2],
            start_char=row[3],
            text_len=row[4],
            distance=row[5],
            score=1.0 - row[5],
        )
        for row in rows
    ]


def delete_embeddings(ref_key: str, references_dir: Path) -> int:
    """Delete all embeddings for a reference. Returns count deleted."""
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return 0

    with get_db(references_dir) as conn:
        count = _delete_ref_data(conn, ref_key)
        conn.commit()
    return count


def clear_all_embeddings(references_dir: Path) -> int:
    """Delete the entire embeddings DB. Returns number of chunks removed.

    Used by ``--fresh`` to force re-embedding of all references.
    """
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return 0

    with get_db(references_dir) as conn:
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM ref_status")
        conn.execute("DELETE FROM embed_meta")
        # Drop vec table — will be recreated with correct dim on next store
        try:
            conn.execute("DROP TABLE IF EXISTS chunk_embeddings")
        except Exception as e:
            logger.debug(f"chunk_embeddings drop skipped ({type(e).__name__}: {e})")
        conn.commit()
        conn.execute("VACUUM")
    logger.info("Cleared {} embeddings from {}", count, db_file)
    return count


def get_stored_model(references_dir: Path) -> str | None:
    """Return the embedding model name stored in the DB, or None."""
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return None
    try:
        with get_db(references_dir) as conn:
            row = conn.execute(
                "SELECT value FROM embed_meta WHERE key='model'"
            ).fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        logger.debug("get_stored_model: DB read failed ({}: {})", type(e).__name__, e)
        return None
