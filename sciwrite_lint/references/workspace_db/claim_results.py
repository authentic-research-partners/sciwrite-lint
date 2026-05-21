"""Claim results CRUD (replaces claims_{paper}.json files)."""

from __future__ import annotations

import sqlite3
from typing import Any

# load_claim_results and find_claim share the same projection + row→dict
# shape; defining it once avoids drift when columns change.
_SELECT_FULL = (
    "SELECT id, ref_key, claim_text, line, context, verdict, confidence, "
    "relevant_quote, explanation, backend, model, ref_type, cite_purpose, "
    "skip_reason, dismissed, reviewer_comment, dismissed_date, "
    "resolved_at, evidence_locator "
    "FROM claim_results"
)


def _row_to_claim_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    """Build the canonical claim dict from a _SELECT_FULL row."""
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
        "skip_reason": row[13],
        "resolved_at": row[17],
        "evidence_locator": row[18],
    }
    if row[14]:
        d["dismissed"] = True
        d["reviewer_comment"] = row[15]
        d["dismissed_date"] = row[16]
    return d


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
                cite_purpose, skip_reason, dismissed, reviewer_comment,
                dismissed_date, resolved_at, evidence_locator)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                r.get("skip_reason", ""),
                int(r.get("dismissed", False)),
                r.get("reviewer_comment", ""),
                r.get("dismissed_date", ""),
                r.get("resolved_at", ""),
                r.get("evidence_locator", ""),
            ),
        )
    conn.commit()


def count_by_verdict(conn: sqlite3.Connection) -> dict[str, int]:
    """Return verdict → count for the current run's claim_results.

    Used by the integrity summary to show verified / SKIPPED /
    CANNOT_DETERMINE counts without loading every row.
    """
    rows = conn.execute(
        "SELECT verdict, COUNT(*) FROM claim_results GROUP BY verdict"
    ).fetchall()
    return {row[0]: int(row[1]) for row in rows}


def load_claim_results(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Load all claim results as a list of dicts."""
    rows = conn.execute(f"{_SELECT_FULL} ORDER BY id").fetchall()
    return [_row_to_claim_dict(row) for row in rows]


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
        f"{_SELECT_FULL} WHERE ref_key = ? AND line = ? LIMIT 1",
        (ref_key, line),
    ).fetchone()
    return _row_to_claim_dict(row) if row else None


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
