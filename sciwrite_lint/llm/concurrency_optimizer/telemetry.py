"""Persistent telemetry for the dynamic concurrency controller.

Stores a wide rolling-history table in a dedicated SQLite file (WAL
mode for concurrent readers) so other tools (``vllm monitor``,
post-hoc analysis notebooks) can render any view of controller
behaviour without reading the controller's in-memory state.

DB location resolution (most-specific first):

1. ``db_path`` argument passed explicitly to ``write_sample`` /
   ``read_recent`` / ``cleanup_partition``.
2. ``CONCURRENCY_OPTIMIZER_DB`` environment variable — useful for
   tests / CI / dev overrides without touching code.
3. Generic default ``~/.cache/concurrency_optimizer/telemetry.db`` —
   a dedicated file separate from any host-project state. Concurrency
   telemetry is short-lived rolling-history data; keeping it in its
   own file means the host project's primary state DB is unaffected
   by schema changes here, and the package can be copied out without
   touching host paths.

Each row is a per-tick snapshot:

- **Partition / identification** — ``endpoint``, ``size_class``,
  ``service`` (``"text"`` or ``"vision"``), ``model_served_name``.
- **Controller decision** — ``current_cap``, ``local_in_flight``,
  ``reason`` (init / hold / grow_kv_low / shrink_queue / shrink_kv_high
  / override).
- **vLLM /metrics gauges** — KV cache utilisation, requests
  running / waiting / swapped, preemptions.
- **Prefix cache** — hit and query counters.
- **Token counters (cumulative)** — prompt and generation totals.
  Analysis layer takes deltas between consecutive rows to derive
  throughput.
- **Latency histograms** — sum and count for e2e latency, time-to-
  first-token, inter-token latency. Average = sum / count, deltas
  give windowed averages.
- **Finish-reason counters** — stop / length / abort / error counts.
- **Host snapshot** — VRAM used/total, GPU compute %, host RAM
  used/total. Best-effort via nvidia-smi + psutil — zeros when
  unavailable.

Writer policy:

- ``write_sample`` does only INSERT. No per-tick rolling delete.
- ``cleanup_partition`` is called once at controller startup to trim
  each ``(endpoint, size_class)`` partition down to the last
  ``KEEP_LAST_N`` (default 1000) rows.
- All writes are best-effort — SQLite errors are logged at debug and
  swallowed so telemetry never crashes the controller.

Reader: :func:`read_recent` returns the most recent samples,
optionally filtered by endpoint, size class, or service. Each
``(endpoint, size_class)`` partition is independent so text and vision
controllers can be queried in isolation.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from loguru import logger
from pydantic import BaseModel, Field

_ENV_VAR = "CONCURRENCY_OPTIMIZER_DB"
_GENERIC_DEFAULT = Path.home() / ".cache" / "concurrency_optimizer" / "telemetry.db"
# Resolved at module-import time. Host projects that want to share an
# existing SQLite file should set ``CONCURRENCY_OPTIMIZER_DB`` BEFORE
# importing this module (e.g., in their package's ``__init__.py``).
DEFAULT_DB_PATH = Path(os.environ.get(_ENV_VAR, str(_GENERIC_DEFAULT)))
KEEP_LAST_N = 1000

Service = Literal["text", "vision"]

# Initial-version schema — only columns and indexes that existed in the
# very first persisted controller_samples table. Newer columns are
# added via _migrate_schema below so older databases upgrade in place.
# (executescript runs all CREATEs in one go; an index that referenced a
# new column would fail before migrate ran.)
_SCHEMA_INITIAL = """\
CREATE TABLE IF NOT EXISTS controller_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    endpoint TEXT NOT NULL,
    size_class TEXT NOT NULL,
    current_cap INTEGER NOT NULL,
    local_in_flight INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL,
    requests_running INTEGER NOT NULL DEFAULT 0,
    requests_waiting INTEGER NOT NULL DEFAULT 0,
    kv_cache_pct REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_controller_samples_partition_ts
    ON controller_samples (endpoint, size_class, ts);
"""

# Indexes that depend on columns added by _migrate_schema. Created
# AFTER migration so the referenced columns exist.
_POST_MIGRATE_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_controller_samples_service_ts
    ON controller_samples (service, ts);
"""


class TelemetryRow(BaseModel):
    """One per-tick snapshot. All fields except partition + decision
    have safe defaults so partial data still produces a valid row."""

    ts: float = Field(description="epoch seconds")

    # Partition / identification
    endpoint: str
    size_class: str
    service: Service
    model_served_name: str = ""

    # Controller decision
    current_cap: int
    local_in_flight: int
    reason: str

    # vLLM /metrics gauges
    requests_running: int = 0
    requests_waiting: int = 0
    requests_swapped: int = 0
    kv_cache_pct: float = 0.0
    num_preemptions: float = 0.0

    # Prefix cache
    prefix_cache_hits: float = 0.0
    prefix_cache_queries: float = 0.0

    # Token counters (cumulative)
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0

    # Latency histograms (sum / count, both cumulative)
    e2e_latency_sum: float = 0.0
    e2e_latency_count: float = 0.0
    ttft_sum: float = 0.0
    ttft_count: float = 0.0
    itl_sum: float = 0.0
    itl_count: float = 0.0

    # Finish-reason counters (cumulative)
    finish_stop: float = 0.0
    finish_length: float = 0.0
    finish_abort: float = 0.0
    finish_error: float = 0.0

    # Host snapshot (best-effort — zeros when nvidia-smi/psutil missing)
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    gpu_util_pct: float = 0.0
    host_ram_used_mb: float = 0.0
    host_ram_total_mb: float = 0.0


_INSERT_SQL = """\
INSERT INTO controller_samples (
    ts,
    endpoint, size_class, service, model_served_name,
    current_cap, local_in_flight, reason,
    requests_running, requests_waiting, requests_swapped,
    kv_cache_pct, num_preemptions,
    prefix_cache_hits, prefix_cache_queries,
    prompt_tokens_total, generation_tokens_total,
    e2e_latency_sum, e2e_latency_count,
    ttft_sum, ttft_count,
    itl_sum, itl_count,
    finish_stop, finish_length, finish_abort, finish_error,
    vram_used_mb, vram_total_mb, gpu_util_pct,
    host_ram_used_mb, host_ram_total_mb
) VALUES (
    ?,
    ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?,
    ?, ?,
    ?, ?,
    ?, ?,
    ?, ?,
    ?, ?,
    ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?
)
"""

# Column order for SELECT — must match TelemetryRow field order so
# read_recent can construct rows positionally.
_SELECT_COLS = (
    "ts, endpoint, size_class, service, model_served_name, "
    "current_cap, local_in_flight, reason, "
    "requests_running, requests_waiting, requests_swapped, "
    "kv_cache_pct, num_preemptions, "
    "prefix_cache_hits, prefix_cache_queries, "
    "prompt_tokens_total, generation_tokens_total, "
    "e2e_latency_sum, e2e_latency_count, "
    "ttft_sum, ttft_count, "
    "itl_sum, itl_count, "
    "finish_stop, finish_length, finish_abort, finish_error, "
    "vram_used_mb, vram_total_mb, gpu_util_pct, "
    "host_ram_used_mb, host_ram_total_mb"
)
_SELECT_FIELD_NAMES = [c.strip() for c in _SELECT_COLS.split(",")]


# Columns added after the initial controller_samples schema. ALTER
# TABLE migrations on open so existing usage.db files pick up new
# columns (with defaults) without losing past samples.
_LATER_COLUMNS: list[tuple[str, str]] = [
    ("service", "TEXT NOT NULL DEFAULT 'text'"),
    ("model_served_name", "TEXT NOT NULL DEFAULT ''"),
    ("requests_swapped", "INTEGER NOT NULL DEFAULT 0"),
    ("num_preemptions", "REAL NOT NULL DEFAULT 0"),
    ("prefix_cache_hits", "REAL NOT NULL DEFAULT 0"),
    ("prefix_cache_queries", "REAL NOT NULL DEFAULT 0"),
    ("prompt_tokens_total", "REAL NOT NULL DEFAULT 0"),
    ("generation_tokens_total", "REAL NOT NULL DEFAULT 0"),
    ("e2e_latency_sum", "REAL NOT NULL DEFAULT 0"),
    ("e2e_latency_count", "REAL NOT NULL DEFAULT 0"),
    ("ttft_sum", "REAL NOT NULL DEFAULT 0"),
    ("ttft_count", "REAL NOT NULL DEFAULT 0"),
    ("itl_sum", "REAL NOT NULL DEFAULT 0"),
    ("itl_count", "REAL NOT NULL DEFAULT 0"),
    ("finish_stop", "REAL NOT NULL DEFAULT 0"),
    ("finish_length", "REAL NOT NULL DEFAULT 0"),
    ("finish_abort", "REAL NOT NULL DEFAULT 0"),
    ("finish_error", "REAL NOT NULL DEFAULT 0"),
    ("vram_used_mb", "REAL NOT NULL DEFAULT 0"),
    ("vram_total_mb", "REAL NOT NULL DEFAULT 0"),
    ("gpu_util_pct", "REAL NOT NULL DEFAULT 0"),
    ("host_ram_used_mb", "REAL NOT NULL DEFAULT 0"),
    ("host_ram_total_mb", "REAL NOT NULL DEFAULT 0"),
]


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns missing from older ``controller_samples`` tables."""
    cur = conn.execute("PRAGMA table_info(controller_samples)")
    existing = {row[1] for row in cur.fetchall()}
    for col_name, col_def in _LATER_COLUMNS:
        if col_name in existing:
            continue
        try:
            conn.execute(
                f"ALTER TABLE controller_samples ADD COLUMN {col_name} {col_def}"
            )
        except sqlite3.OperationalError as e:
            logger.debug(f"telemetry migration ({col_name}): {type(e).__name__}: {e}")


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=10.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA_INITIAL)
        _migrate_schema(conn)
        conn.executescript(_POST_MIGRATE_INDEXES)
        yield conn
    finally:
        conn.close()


def write_sample(row: TelemetryRow, db_path: Path | None = None) -> None:
    """Append one sample. Best-effort — SQLite errors are swallowed."""
    if db_path is None:
        from . import telemetry as _self

        db_path = _self.DEFAULT_DB_PATH
    try:
        with _connect(db_path) as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    row.ts,
                    row.endpoint,
                    row.size_class,
                    row.service,
                    row.model_served_name,
                    row.current_cap,
                    row.local_in_flight,
                    row.reason,
                    row.requests_running,
                    row.requests_waiting,
                    row.requests_swapped,
                    row.kv_cache_pct,
                    row.num_preemptions,
                    row.prefix_cache_hits,
                    row.prefix_cache_queries,
                    row.prompt_tokens_total,
                    row.generation_tokens_total,
                    row.e2e_latency_sum,
                    row.e2e_latency_count,
                    row.ttft_sum,
                    row.ttft_count,
                    row.itl_sum,
                    row.itl_count,
                    row.finish_stop,
                    row.finish_length,
                    row.finish_abort,
                    row.finish_error,
                    row.vram_used_mb,
                    row.vram_total_mb,
                    row.gpu_util_pct,
                    row.host_ram_used_mb,
                    row.host_ram_total_mb,
                ),
            )
    except sqlite3.Error as e:
        logger.debug(f"telemetry write failed: {type(e).__name__}: {e}")


def cleanup_partition(
    endpoint: str,
    size_class: str,
    keep_last_n: int = KEEP_LAST_N,
    db_path: Path | None = None,
) -> int:
    """Trim a single ``(endpoint, size_class)`` partition to the most
    recent ``keep_last_n`` rows. Returns the number of rows deleted.
    """
    if db_path is None:
        from . import telemetry as _self

        db_path = _self.DEFAULT_DB_PATH
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                "DELETE FROM controller_samples WHERE id IN ("
                "  SELECT id FROM controller_samples"
                "  WHERE endpoint = ? AND size_class = ?"
                "  ORDER BY ts DESC"
                "  LIMIT -1 OFFSET ?"
                ")",
                (endpoint, size_class, keep_last_n),
            )
            return cur.rowcount or 0
    except sqlite3.Error as e:
        logger.debug(f"telemetry cleanup failed: {type(e).__name__}: {e}")
        return 0


def read_recent(
    endpoint: str | None = None,
    size_class: str | None = None,
    service: Service | None = None,
    limit: int = 100,
    db_path: Path | None = None,
) -> list[TelemetryRow]:
    """Return the most recent samples, optionally filtered.

    Result is ordered newest first.
    """
    if db_path is None:
        from . import telemetry as _self

        db_path = _self.DEFAULT_DB_PATH
    if not db_path.exists():
        return []
    where: list[str] = []
    args: list[object] = []
    if endpoint is not None:
        where.append("endpoint = ?")
        args.append(endpoint)
    if size_class is not None:
        where.append("size_class = ?")
        args.append(size_class)
    if service is not None:
        where.append("service = ?")
        args.append(service)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    args.append(limit)
    sql = (
        f"SELECT {_SELECT_COLS} FROM controller_samples "
        f"{where_sql} ORDER BY ts DESC LIMIT ?"
    )
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(sql, args)
            return [
                TelemetryRow(**dict(zip(_SELECT_FIELD_NAMES, row)))
                for row in cur.fetchall()
            ]
    except sqlite3.Error as e:
        logger.debug(f"telemetry read failed: {type(e).__name__}: {e}")
        return []


def list_active_streams(db_path: Path | None = None) -> list[tuple[str, str]]:
    """Return all distinct ``(endpoint, size_class)`` partitions present."""
    if db_path is None:
        from . import telemetry as _self

        db_path = _self.DEFAULT_DB_PATH
    if not db_path.exists():
        return []
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                "SELECT DISTINCT endpoint, size_class FROM controller_samples "
                "ORDER BY endpoint, size_class"
            )
            return [(r[0], r[1]) for r in cur.fetchall()]
    except sqlite3.Error as e:
        logger.debug(f"telemetry list failed: {type(e).__name__}: {e}")
        return []
