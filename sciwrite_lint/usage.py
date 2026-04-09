"""Usage tracking for sciwrite-lint pipeline runs.

Records LLM calls, API calls, GROBID parsing, and fetch operations.
Stats accumulate in memory during a run and persist to SQLite.

All DB access goes through ``get_usage_db()`` (sync reads) or
``save_run_async()`` (async writes via aiosqlite).

DB: ~/.sciwrite-lint/usage.db (global, WAL mode, concurrent reads safe).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr


# ---------------------------------------------------------------------------
# Run stats (accumulated during a pipeline run)
# ---------------------------------------------------------------------------


class ServiceStats(BaseModel):
    """Stats for one service (vLLM, CrossRef, etc.)."""

    calls: int = 0
    errors: int = 0
    elapsed_s: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)

    def record(self, elapsed: float, error: bool = False, **kw: Any) -> None:
        self.calls += 1
        self.elapsed_s += elapsed
        if error:
            self.errors += 1
        for k, v in kw.items():
            if isinstance(v, (int, float)):
                self.extra[k] = self.extra.get(k, 0) + v
            else:
                self.extra[k] = v


class RunStats(BaseModel):
    """All stats for a single pipeline run."""

    paper: str = ""
    timestamp: str = ""
    total_elapsed_s: float = 0.0
    citations: int = 0
    model: str = ""
    workspace_root: str = ""  # absolute path to paper workspace (for monitor)
    _preliminary_row_id: int = PrivateAttr(
        default=0
    )  # set by start_run(), used by save_run_async()

    # Per-service stats
    vllm: ServiceStats = Field(default_factory=ServiceStats)
    crossref: ServiceStats = Field(default_factory=ServiceStats)
    openalex: ServiceStats = Field(default_factory=ServiceStats)
    semantic_scholar: ServiceStats = Field(default_factory=ServiceStats)
    core: ServiceStats = Field(default_factory=ServiceStats)
    unpaywall: ServiceStats = Field(default_factory=ServiceStats)
    pmc: ServiceStats = Field(default_factory=ServiceStats)
    europepmc: ServiceStats = Field(default_factory=ServiceStats)
    openlibrary: ServiceStats = Field(default_factory=ServiceStats)
    loc: ServiceStats = Field(default_factory=ServiceStats)
    grobid: ServiceStats = Field(default_factory=ServiceStats)
    fetch: ServiceStats = Field(default_factory=ServiceStats)

    # Stage timings
    stage_rules_s: float = 0.0
    stage_verify_s: float = 0.0
    stage_fetch_s: float = 0.0
    stage_parse_s: float = 0.0
    stage_claims_s: float = 0.0

    def to_dict(self) -> dict:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Active-run context — async-safe via ContextVar
# ---------------------------------------------------------------------------
#
# Why ContextVar instead of a module global: ``run_papers_staged()`` runs
# multiple papers concurrently via ``asyncio.gather``. With a plain global,
# the LAST paper's stats would overwrite earlier ones during setup, and all
# concurrent ``tracked()`` calls would attribute work to that one paper.
# ContextVar is copied per asyncio Task, so each per-paper task can bind its
# own ``ctx.run`` via ``set_current(ctx.run)`` and stays isolated from
# sibling tasks. The single-paper path (``run_full_check``) is unchanged
# because it sets and reads the var within a single task.

_current: ContextVar[RunStats | None] = ContextVar(
    "sciwrite_lint_current_run", default=None
)


def start_run(paper: str = "", model: str = "", workspace_root: str = "") -> RunStats:
    """Start tracking a new pipeline run.

    Writes a preliminary row to usage.db immediately so the monitor can
    find workspace_root while the run is still active. The row is updated
    with full stats by save_run_async() when the run ends.

    Sets the current-run ContextVar to the new RunStats. In multi-paper
    batch mode, callers should additionally call ``set_current(ctx.run)``
    inside each per-paper task wrapper so concurrent tracking lands in
    the right paper's stats — see ``run_papers_staged`` in pipeline.py.
    """
    run = RunStats(
        paper=paper,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=model,
        workspace_root=workspace_root,
    )
    if workspace_root:
        import os

        try:
            with get_usage_db() as conn:
                cur = conn.execute(
                    "INSERT INTO runs (paper, timestamp, workspace_root, pid) "
                    "VALUES (?, ?, ?, ?)",
                    (run.paper, run.timestamp, workspace_root, os.getpid()),
                )
                conn.commit()
                run._preliminary_row_id = cur.lastrowid or 0
        except Exception as e:
            logger.debug(f"usage run insert skipped ({type(e).__name__}: {e})")
    _current.set(run)
    return run


def set_current(run: RunStats | None) -> None:
    """Bind the current-run ContextVar to ``run`` in the current task.

    Use this from inside a per-paper task wrapper in batch mode so all
    ``tracked()`` calls within that task (and its child coroutines) write
    to the correct paper's stats. The change is local to the calling
    task's context — sibling tasks are unaffected.
    """
    _current.set(run)


def current() -> RunStats | None:
    """Get current run stats, or None if no run active."""
    return _current.get()


def end_run() -> RunStats | None:
    """Finalize the current run. Returns stats (caller saves)."""
    stats = _current.get()
    _current.set(None)
    return stats


# ---------------------------------------------------------------------------
# Convenience: timed context manager
# ---------------------------------------------------------------------------


class _TrackContext:
    """Yielded by ``tracked()`` — lets callers mark errors from inside the block."""

    __slots__ = ("error",)

    def __init__(self) -> None:
        self.error: bool = False


@asynccontextmanager
async def tracked(service: str, **extra: Any) -> AsyncIterator[_TrackContext]:
    """Async context manager to time and record a service call.

    Usage::

        async with tracked("crossref") as t:
            resp = await do_request()
            if resp.status_code >= 400:
                t.error = True

    Automatically marks ``error=True`` on unhandled exceptions.
    """
    ctx = _TrackContext()
    start = time.monotonic()
    try:
        yield ctx
    except Exception as exc:
        ctx.error = True
        extra["error_type"] = type(exc).__name__
        raise
    finally:
        elapsed = time.monotonic() - start
        run = _current.get()
        if run is not None:
            svc = getattr(run, service, None)
            if svc is not None:
                svc.record(elapsed, error=ctx.error, **extra)


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

USAGE_DB = str(Path.home() / ".sciwrite-lint" / "usage.db")

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    model TEXT DEFAULT '',
    citations INTEGER DEFAULT 0,
    total_elapsed_s REAL DEFAULT 0,

    -- Stage timings
    stage_rules_s REAL DEFAULT 0,
    stage_verify_s REAL DEFAULT 0,
    stage_fetch_s REAL DEFAULT 0,
    stage_parse_s REAL DEFAULT 0,
    stage_claims_s REAL DEFAULT 0,

    -- Per-service stats (JSON blobs)
    vllm TEXT DEFAULT '{}',
    crossref TEXT DEFAULT '{}',
    openalex TEXT DEFAULT '{}',
    semantic_scholar TEXT DEFAULT '{}',
    core TEXT DEFAULT '{}',
    unpaywall TEXT DEFAULT '{}',
    pmc TEXT DEFAULT '{}',
    europepmc TEXT DEFAULT '{}',
    openlibrary TEXT DEFAULT '{}',
    loc TEXT DEFAULT '{}',
    grobid TEXT DEFAULT '{}',
    fetch TEXT DEFAULT '{}'
);
"""

_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN pmc TEXT DEFAULT '{}'",
    "ALTER TABLE runs ADD COLUMN europepmc TEXT DEFAULT '{}'",
    "ALTER TABLE runs ADD COLUMN openlibrary TEXT DEFAULT '{}'",
    "ALTER TABLE runs ADD COLUMN loc TEXT DEFAULT '{}'",
    "ALTER TABLE runs ADD COLUMN workspace_root TEXT DEFAULT ''",
    "ALTER TABLE runs ADD COLUMN pid INTEGER DEFAULT 0",
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns missing from older schema versions."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    for stmt in _MIGRATIONS:
        col = stmt.split("ADD COLUMN ")[1].split()[0]
        if col not in existing:
            conn.execute(stmt)


_INSERT = """\
INSERT INTO runs (
    paper, timestamp, model, citations, total_elapsed_s, workspace_root,
    stage_rules_s, stage_verify_s, stage_fetch_s, stage_parse_s, stage_claims_s,
    vllm, crossref, openalex, semantic_scholar, core, unpaywall,
    pmc, europepmc, openlibrary, loc, grobid, fetch
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE = """\
UPDATE runs SET
    paper=?, timestamp=?, model=?, citations=?, total_elapsed_s=?, workspace_root=?,
    stage_rules_s=?, stage_verify_s=?, stage_fetch_s=?, stage_parse_s=?, stage_claims_s=?,
    vllm=?, crossref=?, openalex=?, semantic_scholar=?, core=?, unpaywall=?,
    pmc=?, europepmc=?, openlibrary=?, loc=?, grobid=?, fetch=?
WHERE id=?
"""


def _run_to_params(stats: RunStats) -> tuple:
    return (
        stats.paper,
        stats.timestamp,
        stats.model,
        stats.citations,
        stats.total_elapsed_s,
        stats.workspace_root,
        stats.stage_rules_s,
        stats.stage_verify_s,
        stats.stage_fetch_s,
        stats.stage_parse_s,
        stats.stage_claims_s,
        json.dumps(stats.vllm.model_dump()),
        json.dumps(stats.crossref.model_dump()),
        json.dumps(stats.openalex.model_dump()),
        json.dumps(stats.semantic_scholar.model_dump()),
        json.dumps(stats.core.model_dump()),
        json.dumps(stats.unpaywall.model_dump()),
        json.dumps(stats.pmc.model_dump()),
        json.dumps(stats.europepmc.model_dump()),
        json.dumps(stats.openlibrary.model_dump()),
        json.dumps(stats.loc.model_dump()),
        json.dumps(stats.grobid.model_dump()),
        json.dumps(stats.fetch.model_dump()),
    )


# ---------------------------------------------------------------------------
# Service column names (single source of truth for serialization + parsing)
# ---------------------------------------------------------------------------

_SERVICE_FIELDS = (
    "vllm",
    "crossref",
    "openalex",
    "semantic_scholar",
    "core",
    "unpaywall",
    "pmc",
    "europepmc",
    "openlibrary",
    "loc",
    "grobid",
    "fetch",
)


# ---------------------------------------------------------------------------
# DB connection helpers
# ---------------------------------------------------------------------------


def _init_conn(conn: sqlite3.Connection) -> None:
    """Configure pragmas, create schema, run migrations."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    _migrate(conn)


@contextmanager
def get_usage_db(db_path: str | Path = USAGE_DB) -> Iterator[sqlite3.Connection]:
    """Context manager for usage DB connections.

    This is the **only** way to obtain a usage DB connection.
    Guarantees cleanup on exit (including exceptions and early returns).
    """
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        _init_conn(conn)
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Async write (pipeline saves)
# ---------------------------------------------------------------------------


async def save_run_async(stats: RunStats, db_path: str | Path = USAGE_DB) -> int:
    """Save run stats to SQLite via aiosqlite. Returns the row ID."""
    import aiosqlite

    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(p)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.executescript(_SCHEMA)
        # Migrate older DBs
        rows = await db.execute("PRAGMA table_info(runs)")
        existing = {row[1] async for row in rows}
        for stmt in _MIGRATIONS:
            col = stmt.split("ADD COLUMN ")[1].split()[0]
            if col not in existing:
                await db.execute(stmt)
        row_id = stats._preliminary_row_id
        if row_id:
            # Update the preliminary row written by start_run()
            await db.execute(_UPDATE, (*_run_to_params(stats), row_id))
            await db.commit()
            return row_id
        cur = await db.execute(_INSERT, _run_to_params(stats))
        await db.commit()
        return cur.lastrowid or 0


# ---------------------------------------------------------------------------
# Queries (sync — for Streamlit UI and CLI monitor)
# ---------------------------------------------------------------------------


def _parse_service_cols(d: dict) -> dict:
    for svc in _SERVICE_FIELDS:
        try:
            d[svc] = json.loads(d[svc])
        except (json.JSONDecodeError, TypeError, KeyError):
            d[svc] = {}
    return d


def load_runs(
    db_path: str | Path = USAGE_DB,
    paper: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Load run stats from SQLite. Returns list of dicts."""
    p = Path(db_path)
    if not p.exists():
        return []

    with get_usage_db(db_path) as conn:
        if paper:
            rows = conn.execute(
                "SELECT * FROM runs WHERE paper = ? ORDER BY timestamp DESC LIMIT ?",
                (paper, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_parse_service_cols(dict(r)) for r in rows]


def get_workspace_root(
    paper: str,
    db_path: str | Path = USAGE_DB,
) -> str | None:
    """Get workspace_root for the most recent run of a paper."""
    p = Path(db_path)
    if not p.exists():
        return None
    with get_usage_db(db_path) as conn:
        row = conn.execute(
            "SELECT workspace_root FROM runs WHERE paper = ? AND workspace_root != '' "
            "ORDER BY timestamp DESC LIMIT 1",
            (paper,),
        ).fetchone()
        return row["workspace_root"] if row else None


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` currently exists.

    Uses ``os.kill(pid, 0)`` which sends no signal but performs the
    normal permission + existence checks. On Linux/WSL2:
      - success → process exists and we can signal it
      - PermissionError → process exists but belongs to another user
      - ProcessLookupError → process does not exist
      - any other OSError → treat as unknown, assume alive
    """
    import os

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True


def find_active_db_runs(
    db_path: str | Path = USAGE_DB,
) -> list[dict[str, str]]:
    """Find runs with in-progress pipeline stages by checking workspace.db.

    Scans recent usage.db rows for workspace_root entries, then checks each
    workspace.db for pipeline stages in "running" status. Returns a list of
    dicts with paper name and workspace_root for display in the monitor.
    Works for CLI runs, eval runs, and batch-staged runs alike.
    """
    import sqlite3

    p = Path(db_path)
    if not p.exists():
        return []

    results: list[dict[str, str]] = []

    try:
        with get_usage_db(db_path) as conn:
            rows = conn.execute(
                "SELECT paper, workspace_root, timestamp, pid FROM runs "
                "WHERE workspace_root != '' ORDER BY timestamp DESC LIMIT 50",
            ).fetchall()
    except Exception:
        return []

    seen: set[str] = set()
    for row in rows:
        paper = row["paper"]
        ws_root = row["workspace_root"]
        pid = row["pid"] or 0
        if paper in seen or not ws_root:
            continue
        seen.add(paper)

        # Skip runs whose owning process is gone. Any live run writes
        # its PID in start_run(), so a row without a live PID cannot
        # be making progress — even if the stage table looks mid-flight.
        if not _pid_alive(pid):
            continue

        # Check if workspace.db has running stages
        ws_db = Path(ws_root) / "parsed" / "workspace.db"
        if not ws_db.exists():
            continue
        try:
            ws_conn = sqlite3.connect(str(ws_db), timeout=1)
            ws_conn.row_factory = sqlite3.Row
            # A run is "active" if any stage is running, OR the pipeline is
            # mid-flight: at least one stage has reached a terminal state
            # (done/failed/skipped) while others are still pending. The
            # second condition keeps the panel visible during brief gaps
            # between stages when no stage is currently marked "running".
            counts = ws_conn.execute(
                "SELECT "
                "SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS n_running, "
                "SUM(CASE WHEN status IN ('done','failed','skipped') THEN 1 ELSE 0 END) AS n_terminal, "
                "SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS n_pending "
                "FROM pipeline_stage"
            ).fetchone()
            ws_conn.close()
            if counts is None:
                continue
            n_running = counts["n_running"] or 0
            n_terminal = counts["n_terminal"] or 0
            n_pending = counts["n_pending"] or 0
            is_active = n_running > 0 or (n_terminal > 0 and n_pending > 0)
            if is_active:
                results.append({"paper": paper, "workspace_root": ws_root})
        except Exception as e:
            logger.debug(
                f"active-run check skipped for {paper} ({type(e).__name__}: {e})"
            )
            continue

    return results


def get_summary(
    db_path: str | Path = USAGE_DB,
    paper: str | None = None,
) -> dict:
    """Aggregate stats across all runs."""
    p = Path(db_path)
    if not p.exists():
        return {}

    with get_usage_db(db_path) as conn:
        where = "WHERE paper = ?" if paper else ""
        params = (paper,) if paper else ()
        row = conn.execute(
            f"""\
            SELECT
                COUNT(*) as runs,
                SUM(citations) as total_citations,
                SUM(total_elapsed_s) as total_elapsed_s,
                AVG(total_elapsed_s) as avg_elapsed_s,
                MIN(total_elapsed_s) as min_elapsed_s,
                MAX(total_elapsed_s) as max_elapsed_s
            FROM runs {where}
            """,
            params,
        ).fetchone()
        if not row or row[0] == 0:
            return {}
        return {
            "runs": row[0],
            "total_citations": row[1],
            "total_elapsed_s": round(row[2], 1),
            "avg_elapsed_s": round(row[3], 1),
            "min_elapsed_s": round(row[4], 1),
            "max_elapsed_s": round(row[5], 1),
        }
