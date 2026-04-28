"""Citation metadata CRUD and targeted queries."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sciwrite_lint.models import CitationMetadata

# All CRUD and query functions project the same columns; defining the
# SELECT once avoids drift if the schema changes.
_SELECT_ALL = (
    "SELECT ref_key, verified_date, api_source, api_match, "
    "canonical_json, bibitem_json, access_json, "
    "mismatches_json, issues_json, manual_override_json "
    "FROM citation_metadata"
)


def save_citation_metadata(
    conn: sqlite3.Connection,
    meta: "CitationMetadata",
) -> None:
    """Upsert a CitationMetadata record into workspace.db."""
    conn.execute(
        """INSERT INTO citation_metadata
           (ref_key, verified_date, api_source, api_match,
            canonical_json, bibitem_json, access_json,
            mismatches_json, issues_json, manual_override_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (ref_key) DO UPDATE SET
             verified_date = excluded.verified_date,
             api_source = excluded.api_source,
             api_match = excluded.api_match,
             canonical_json = excluded.canonical_json,
             bibitem_json = excluded.bibitem_json,
             access_json = excluded.access_json,
             mismatches_json = excluded.mismatches_json,
             issues_json = excluded.issues_json,
             manual_override_json = excluded.manual_override_json
        """,
        (
            meta.key,
            meta.verified_date,
            meta.api_source,
            meta.api_match,
            json.dumps(meta.canonical, ensure_ascii=False),
            json.dumps(meta.bibitem, ensure_ascii=False),
            json.dumps(meta.access, ensure_ascii=False),
            json.dumps(meta.mismatches, ensure_ascii=False),
            json.dumps(meta.issues, ensure_ascii=False),
            json.dumps(meta.manual_override, ensure_ascii=False),
        ),
    )
    conn.commit()


def load_citation_metadata(
    conn: sqlite3.Connection,
    ref_key: str,
) -> "CitationMetadata | None":
    """Load a single CitationMetadata by key. Returns None if not found."""
    row = conn.execute(
        f"{_SELECT_ALL} WHERE ref_key = ?",
        (ref_key,),
    ).fetchone()
    if not row:
        return None
    return _row_to_citation_metadata(row)


def load_all_citation_metadata(
    conn: sqlite3.Connection,
) -> "dict[str, CitationMetadata]":
    """Load all CitationMetadata records. Returns {key: CitationMetadata}."""
    rows = conn.execute(_SELECT_ALL).fetchall()
    return {row[0]: _row_to_citation_metadata(row) for row in rows}


def delete_citation_metadata(conn: sqlite3.Connection, ref_key: str) -> None:
    """Delete a single citation metadata record (used by --fresh)."""
    conn.execute("DELETE FROM citation_metadata WHERE ref_key = ?", (ref_key,))
    conn.commit()


def _row_to_citation_metadata(row: tuple[Any, ...]) -> "CitationMetadata":
    """Convert a DB row to CitationMetadata."""
    from sciwrite_lint.models import CitationMetadata

    return CitationMetadata(
        key=row[0],
        verified_date=row[1],
        api_source=row[2],
        api_match=row[3],
        canonical=json.loads(row[4]),
        bibitem=json.loads(row[5]),
        access=json.loads(row[6]),
        mismatches=json.loads(row[7]),
        issues=json.loads(row[8]),
        manual_override=json.loads(row[9]),
    )


# ---------------------------------------------------------------------------
# Targeted citation metadata queries
# ---------------------------------------------------------------------------


def query_verified_metadata(
    conn: sqlite3.Connection,
) -> "dict[str, CitationMetadata]":
    """Load metadata that should short-circuit re-verification.

    Includes ``verified``/``mismatch``/``not_found`` (academic API
    outcomes), ``web_verified``/``web_dead``/``web_blocked`` (URL
    verification outcomes), and ``manual`` (user override via
    ``sciwrite-lint override`` and synthetic footnote-URL citations
    from :mod:`sciwrite_lint.footnote_urls`). All of these represent
    a terminal state — re-running verify against them would either
    repeat the same API call or, for ``manual``, override a
    deliberate user decision.
    """
    rows = conn.execute(
        f"{_SELECT_ALL} WHERE api_match IN "
        "('verified', 'mismatch', 'not_found', "
        "'web_verified', 'web_dead', 'web_blocked', 'manual')"
    ).fetchall()
    return {row[0]: _row_to_citation_metadata(row) for row in rows}


def query_refs_with_local_pdfs(
    conn: sqlite3.Connection,
) -> dict[str, tuple[str, str]]:
    """Return {ref_key: (local_file, entry_type)} for refs with local PDFs.

    Uses json_extract on access_json to filter server-side.
    """
    rows = conn.execute(
        "SELECT ref_key, "
        "json_extract(access_json, '$.local_file'), "
        "COALESCE(json_extract(bibitem_json, '$.entry_type'), '') "
        "FROM citation_metadata "
        "WHERE json_extract(access_json, '$.local_file') LIKE '%.pdf'"
    ).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def query_refs_by_match(
    conn: sqlite3.Connection,
    api_match: str,
) -> "dict[str, CitationMetadata]":
    """Load metadata for refs with a specific api_match value."""
    rows = conn.execute(
        f"{_SELECT_ALL} WHERE api_match = ?",
        (api_match,),
    ).fetchall()
    return {row[0]: _row_to_citation_metadata(row) for row in rows}


def query_retracted_refs(
    conn: sqlite3.Connection,
) -> "dict[str, CitationMetadata]":
    """Load only refs that have a retraction_status in canonical."""
    rows = conn.execute(
        f"{_SELECT_ALL} "
        "WHERE json_extract(canonical_json, '$.retraction_status') IS NOT NULL"
    ).fetchall()
    return {row[0]: _row_to_citation_metadata(row) for row in rows}


def query_refs_with_mismatches(
    conn: sqlite3.Connection,
) -> "dict[str, CitationMetadata]":
    """Load only refs that have non-empty mismatches."""
    rows = conn.execute(f"{_SELECT_ALL} WHERE mismatches_json != '[]'").fetchall()
    return {row[0]: _row_to_citation_metadata(row) for row in rows}
