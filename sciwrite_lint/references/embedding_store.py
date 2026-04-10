"""SQLite-vec embedding storage for claim verification.

Stores chunk text + embeddings in the per-paper workspace DB
(``references/{paper}/parsed/workspace.db``) using sqlite-vec for KNN
retrieval.

OOM-safe: encodes and inserts in small batches (EMBED_BATCH_SIZE),
never holding all vectors in RAM at once.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from loguru import logger

from sciwrite_lint.references.workspace_db import (
    db_path as _db_path,
    open_db as _open_db,
    serialize_f32 as _serialize_f32,
)

EMBED_BATCH_SIZE = 32  # chunks per encode + insert batch

# Adaptive encode batch sizing based on token length.
# Eager attention memory scales as O(batch * seq_len^2). With 20 GB VRAM
# on RTX 4000 Ada and vLLM occupying ~18.6 GB (paged out on WSL2):
#   ≤512 tokens, batch=32: ~2.5 GB peak — fits easily
#   ≤2048 tokens, batch=8: ~5.5 GB peak — safe
#   >2048 tokens, batch=4:  ~11 GB peak — safe on WSL2 (worst real: 3810 tokens)
_ENCODE_TIERS: list[tuple[int, int]] = [
    (512, 32),  # short chunks: full batch
    (2048, 8),  # medium chunks: reduced batch
]
_ENCODE_BATCH_LONG = 4  # anything above the last tier threshold


def _encode_adaptive(model: object, texts: list[str]) -> np.ndarray:
    """Encode texts with batch size adapted to token length.

    Short texts (≤512 tokens) are encoded in large batches for throughput.
    Long texts get smaller batches to stay within GPU memory limits.
    Results are reassembled in the original order.
    """
    import numpy as np

    tokenizer = model.tokenizer  # type: ignore[attr-defined]
    # Tokenize to get lengths (fast, CPU-only)
    token_counts = [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]

    # Group indices by tier
    groups: dict[int, list[int]] = {}
    for idx, tok_len in enumerate(token_counts):
        batch_size = _ENCODE_BATCH_LONG
        for threshold, bs in _ENCODE_TIERS:
            if tok_len <= threshold:
                batch_size = bs
                break
        groups.setdefault(batch_size, []).append(idx)

    # Encode each group with its batch size, collect results
    results: dict[int, np.ndarray] = {}
    for batch_size, indices in groups.items():
        group_texts = [texts[i] for i in indices]
        vecs = model.encode(  # type: ignore[attr-defined]
            group_texts, normalize_embeddings=True, batch_size=batch_size
        )
        for i, idx in enumerate(indices):
            results[idx] = vecs[i]

    # Reassemble in original order
    return np.stack([results[i] for i in range(len(texts))])


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
    """
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return False
    try:
        conn = _open_db(references_dir)

        # Check completion status
        row = conn.execute(
            "SELECT complete FROM ref_status WHERE ref_key = ?", (ref_key,)
        ).fetchone()
        if row is None:
            conn.close()
            return False
        if row[0] != 1:
            # Incomplete — clean up broken data
            logger.debug("Cleaning up incomplete embeddings for {}", ref_key)
            _delete_ref_data(conn, ref_key)
            conn.commit()
            conn.close()
            return False

        # Check model compatibility
        if model_name is not None:
            stored = conn.execute(
                "SELECT value FROM embed_meta WHERE key='model'"
            ).fetchone()
            if stored and stored[0] != model_name:
                conn.close()
                return False

        conn.close()
        return True
    except Exception:
        return False


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
    """Store chunk text + embeddings in batches (OOM-safe).

    Each chunk dict must have: text, section_title, granularity, start_char.
    Embeddings are computed internally in batches of EMBED_BATCH_SIZE.

    Returns the number of chunks stored.
    """
    from sciwrite_lint.references.reference_store import _get_embedding_model

    conn = _open_db(references_dir)
    _ensure_vec_table(conn, embedding_dim)

    # Clear existing data for this key (including incomplete runs)
    _delete_ref_data(conn, ref_key)

    # Mark as in-progress (complete=0) with expected count
    conn.execute(
        "INSERT INTO ref_status (ref_key, expected_chunks, stored_chunks, complete) "
        "VALUES (?, ?, 0, 0)",
        (ref_key, len(chunks)),
    )
    conn.commit()

    model = _get_embedding_model()
    total_stored = 0

    # Process in batches — encode + insert, then free vectors
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Encode with adaptive batch size: group by token length to avoid
        # OOM on long chunks (eager attention is O(batch * seq_len^2)).
        vectors = _encode_adaptive(model, texts)

        # Insert chunk metadata + vectors
        for i, (chunk, vec) in enumerate(zip(batch, vectors)):
            cursor = conn.execute(
                "INSERT INTO chunks (ref_key, chunk_index, text, section_title, "
                "granularity, start_char, text_len) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ref_key,
                    batch_start + i,
                    chunk["text"],
                    chunk["section_title"],
                    chunk["granularity"],
                    chunk["start_char"],
                    len(chunk["text"]),
                ),
            )
            chunk_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO chunk_embeddings (rowid, embedding) VALUES (?, ?)",
                (chunk_id, _serialize_f32(vec.tolist())),
            )

        conn.commit()
        total_stored += len(batch)

        # Update progress
        conn.execute(
            "UPDATE ref_status SET stored_chunks = ? WHERE ref_key = ?",
            (total_stored, ref_key),
        )
        conn.commit()

        # vectors go out of scope here — GC can reclaim

    # Mark complete
    conn.execute("UPDATE ref_status SET complete = 1 WHERE ref_key = ?", (ref_key,))
    conn.execute(
        "INSERT OR REPLACE INTO embed_meta (key, value) VALUES ('model', ?)",
        (model_name,),
    )
    conn.commit()
    conn.close()

    return total_stored


def retrieve_similar(
    query_text: str,
    ref_key: str,
    references_dir: Path,
    top_k: int = 15,
) -> list[dict]:
    """Find chunks most similar to query via sqlite-vec KNN.

    Returns list of dicts with: text, section_title, granularity, distance, score.
    Distance is cosine distance (0 = identical, 2 = opposite).
    Score is cosine similarity (1 - distance).
    """
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return []

    conn = _open_db(references_dir)

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
        conn.close()
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
        conn.close()
        return []

    # KNN search scoped to target ref via rowid pre-filter.
    # sqlite-vec supports `rowid IN (...)` during MATCH, so we restrict
    # the search to only the target ref's chunks — no global scan needed.
    rows = conn.execute(
        """
        SELECT c.text, c.section_title, c.granularity, c.start_char,
               c.text_len, ce.distance
        FROM chunk_embeddings ce
        INNER JOIN chunks c ON c.id = ce.rowid
        WHERE ce.embedding MATCH ?
            AND k = ?
            AND ce.rowid IN (SELECT id FROM chunks WHERE ref_key = ?)
        ORDER BY ce.distance
        """,
        (query_blob, top_k, ref_key),
    ).fetchall()

    conn.close()

    return [
        {
            "text": row[0],
            "section_title": row[1],
            "granularity": row[2],
            "start_char": row[3],
            "text_len": row[4],
            "distance": row[5],
            "score": 1.0 - row[5],
        }
        for row in rows
    ]


def delete_embeddings(ref_key: str, references_dir: Path) -> int:
    """Delete all embeddings for a reference. Returns count deleted."""
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return 0

    conn = _open_db(references_dir)
    count = _delete_ref_data(conn, ref_key)
    conn.commit()
    conn.close()
    return count


def clear_all_embeddings(references_dir: Path) -> int:
    """Delete the entire embeddings DB. Returns number of chunks removed.

    Used by ``--fresh`` to force re-embedding of all references.
    """
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return 0

    conn = _open_db(references_dir)
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
    conn.close()
    logger.info("Cleared {} embeddings from {}", count, db_file)
    return count


def get_stored_model(references_dir: Path) -> str | None:
    """Return the embedding model name stored in the DB, or None."""
    db_file = _db_path(references_dir)
    if not db_file.exists():
        return None
    try:
        conn = _open_db(references_dir)
        row = conn.execute("SELECT value FROM embed_meta WHERE key='model'").fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None
