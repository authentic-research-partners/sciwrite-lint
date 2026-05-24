"""Manuscript LLM check cache CRUD.

The cache stores one row per `(prompt_hash, check_id)` pair, where
`prompt_hash` is the SHA-256 of the canonical LLM input
(system + user + schema_name + model + cache_version). The model and
cache_version are also stored as columns so cleanup queries can target
them; they are redundant with the hash on the lookup path.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any


def lookup_many(
    conn: sqlite3.Connection,
    prompt_hashes: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Bulk-lookup cached results.

    Returns a dict ``{(prompt_hash, check_id): result_json}`` containing
    only the entries that were hits. Misses are simply absent.
    """
    if not prompt_hashes:
        return {}
    out: dict[tuple[str, str], str] = {}
    # SQLite's parameter limit caps at ~999; batch in groups of 200
    # (prompt_hash, check_id) pairs → 400 params, well under the limit
    # and avoids a single overly wide IN-clause.
    for i in range(0, len(prompt_hashes), 200):
        chunk = prompt_hashes[i : i + 200]
        placeholders = ",".join(["(?, ?)"] * len(chunk))
        flat: list[str] = []
        for ph, cid in chunk:
            flat.append(ph)
            flat.append(cid)
        rows = conn.execute(
            "SELECT prompt_hash, check_id, result_json "
            f"FROM manuscript_check_cache WHERE (prompt_hash, check_id) IN ({placeholders})",
            flat,
        ).fetchall()
        for row in rows:
            out[(row[0], row[1])] = row[2]
    return out


def save_many(
    conn: sqlite3.Connection,
    entries: list[dict[str, Any]],
) -> None:
    """Insert or replace cached results.

    Each entry must carry: prompt_hash, check_id, model, cache_version,
    result_json. ``created_at`` is stamped here.
    """
    if not entries:
        return
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        """INSERT OR REPLACE INTO manuscript_check_cache
           (prompt_hash, check_id, model, cache_version, result_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                e["prompt_hash"],
                e["check_id"],
                e["model"],
                e["cache_version"],
                e["result_json"],
                now,
            )
            for e in entries
        ],
    )
    conn.commit()


def delete_by_model(conn: sqlite3.Connection, model: str) -> int:
    """Drop all cached rows for a given model. Returns rows deleted."""
    cur = conn.execute(
        "DELETE FROM manuscript_check_cache WHERE model = ?",
        (model,),
    )
    conn.commit()
    return int(cur.rowcount)


def delete_by_cache_version(conn: sqlite3.Connection, cache_version: str) -> int:
    """Drop all cached rows whose cache_version != the given current value."""
    cur = conn.execute(
        "DELETE FROM manuscript_check_cache WHERE cache_version != ?",
        (cache_version,),
    )
    conn.commit()
    return int(cur.rowcount)


def count_rows(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM manuscript_check_cache").fetchone()
    return int(row[0]) if row else 0
