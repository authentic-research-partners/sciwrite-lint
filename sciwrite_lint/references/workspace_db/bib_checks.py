"""Bibliography verification result CRUD."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


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
