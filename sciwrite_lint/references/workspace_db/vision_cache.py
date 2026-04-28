"""Vision cache CRUD (figure descriptions from Qwen3-VL).

Composite PK ``(image_key, source)`` lets the same ``image_key``
(e.g. ``fig-1``) appear once per source — manuscript (``"manuscript"``)
and each cited reference (keyed by ``ref_key``).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

_VISION_CACHE_VERSION = "3"


def save_vision_entry(
    conn: sqlite3.Connection,
    image_key: str,
    *,
    image_hash: str,
    source: str = "manuscript",
    label: str = "",
    caption: str = "",
    description: str = "",
    figure_type: str = "",
    readability_issues: list[str] | None = None,
) -> None:
    """Upsert a single vision cache entry.

    ``source`` discriminates manuscript figures (``"manuscript"``) from
    cited reference figures (the ref_key, e.g. ``"tanaka2017"``).
    Together with ``image_key``, it forms the composite primary key.

    ``readability_issues`` is a list of short issue strings (empty list
    means no issues). Serialized as JSON into the
    ``readability_issues_json`` column on write.
    """
    issues_json = json.dumps(readability_issues or [])
    conn.execute(
        """INSERT INTO vision_cache
           (image_key, source, image_hash, label, caption, description,
            figure_type, readability_issues_json, cache_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (image_key, source) DO UPDATE SET
             image_hash = excluded.image_hash,
             label = excluded.label,
             caption = excluded.caption,
             description = excluded.description,
             figure_type = excluded.figure_type,
             readability_issues_json = excluded.readability_issues_json,
             cache_version = excluded.cache_version
        """,
        (
            image_key,
            source,
            image_hash,
            label,
            caption,
            description,
            figure_type,
            issues_json,
            _VISION_CACHE_VERSION,
        ),
    )
    conn.commit()


def load_vision_entry(
    conn: sqlite3.Connection,
    image_key: str,
    *,
    expected_hash: str,
    source: str = "manuscript",
) -> dict[str, Any] | None:
    """Load a vision cache entry if hash matches. Returns None on miss.

    Looks up by composite PK ``(image_key, source)``. The returned dict
    uses the logical key ``readability_issues`` (a ``list[str]``), with
    JSON deserialization from the storage column
    ``readability_issues_json``.
    """
    row = conn.execute(
        "SELECT image_hash, label, caption, description, "
        "figure_type, readability_issues_json, cache_version "
        "FROM vision_cache WHERE image_key = ? AND source = ?",
        (image_key, source),
    ).fetchone()
    if not row:
        return None
    if row[6] != _VISION_CACHE_VERSION:
        return None
    if row[0] != expected_hash:
        return None
    return {
        "image_hash": row[0],
        "label": row[1],
        "caption": row[2],
        "description": row[3],
        "figure_type": row[4],
        "readability_issues": json.loads(row[5]),
    }


def load_all_vision_entries(
    conn: sqlite3.Connection,
    source: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load vision cache entries. Returns {image_key: {...}}.

    When ``source`` is provided, only entries matching that source are
    returned.  When None, returns all entries (use with care). The
    returned dicts use the logical key ``readability_issues`` (a
    ``list[str]``), deserialized from the storage column
    ``readability_issues_json``.
    """
    query = (
        "SELECT image_key, image_hash, label, caption, description, "
        "figure_type, readability_issues_json "
        "FROM vision_cache WHERE cache_version = ?"
    )
    params: list[str] = [_VISION_CACHE_VERSION]
    if source is not None:
        query += " AND source = ?"
        params.append(source)
    rows = conn.execute(query, params).fetchall()
    return {
        row[0]: {
            "image_hash": row[1],
            "label": row[2],
            "caption": row[3],
            "description": row[4],
            "figure_type": row[5],
            "readability_issues": json.loads(row[6]),
        }
        for row in rows
    }


def clear_vision_cache(conn: sqlite3.Connection) -> None:
    """Delete all vision cache entries (for --fresh)."""
    conn.execute("DELETE FROM vision_cache")
    conn.commit()
