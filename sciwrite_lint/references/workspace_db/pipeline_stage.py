"""Pipeline stage tracking for the monitor."""

from __future__ import annotations

import sqlite3

# Canonical stage names and their display order.
PIPELINE_STAGES: list[str] = [
    "setup",  # Stage 0: workspace + citations + GROBID (PDF)
    "vision",  # Stage 0.5: manuscript figure descriptions
    "text_checks",  # Stage 1a: regex-based text rules
    "llm_checks",  # Stage 1b: vLLM batch checks
    "verify",  # Stage 2: API verification
    "fetch",  # Stage 3: full-text acquisition
    "parse",  # Stage 4: GROBID parse + embeddings
    "cited_vision",  # Stage 4.2: VL on cited paper figures
    "ref_internal",  # Stage 4.5: ref internal consistency (vLLM)
    "bib_verify",  # Stage 4.6: bibliography verification
    "claims",  # Stage 5: claim verification (vLLM)
    "unreliable",  # Stage 6: reference-unreliable aggregation
    "contributions",  # Stage 7: contribution axes scoring (vLLM)
]


def init_pipeline_stages(conn: sqlite3.Connection) -> None:
    """Reset all stages to 'pending' at the start of a run."""
    conn.execute("DELETE FROM pipeline_stage")
    for stage in PIPELINE_STAGES:
        conn.execute(
            "INSERT INTO pipeline_stage (stage, status) VALUES (?, 'pending')",
            (stage,),
        )
    conn.commit()


def update_pipeline_stage(
    conn: sqlite3.Connection,
    stage: str,
    status: str,
    detail: str = "",
) -> None:
    """Update a stage's status. status: 'running', 'done', 'failed', 'skipped'."""
    import time

    if status == "running":
        conn.execute(
            "UPDATE pipeline_stage SET status = ?, start_time = ?, detail = ? "
            "WHERE stage = ?",
            (status, time.time(), detail, stage),
        )
    elif status in ("done", "failed", "skipped"):
        conn.execute(
            "UPDATE pipeline_stage SET status = ?, end_time = ?, detail = ? "
            "WHERE stage = ?",
            (status, time.time(), detail, stage),
        )
    else:
        conn.execute(
            "UPDATE pipeline_stage SET status = ?, detail = ? WHERE stage = ?",
            (status, detail, stage),
        )
    conn.commit()


def load_pipeline_stages(
    conn: sqlite3.Connection,
) -> list[dict[str, str | float | None]]:
    """Load all pipeline stages in order. Returns list of dicts."""
    rows = conn.execute(
        "SELECT stage, status, start_time, end_time, detail "
        "FROM pipeline_stage ORDER BY rowid"
    ).fetchall()
    return [
        {
            "stage": r[0],
            "status": r[1],
            "start_time": r[2],
            "end_time": r[3],
            "detail": r[4],
        }
        for r in rows
    ]
