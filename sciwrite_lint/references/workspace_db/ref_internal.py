"""Ref internal cache CRUD (replaces per-key .internal.json files)."""

from __future__ import annotations

import sqlite3
from typing import Any


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
