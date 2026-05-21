"""Phase-level timing collector for the embedder subprocess.

Used to answer "where does the embed stage spend its time" — split into
CPU-side prep (read, split, chunk, tokenize), GPU-side encode, and DB-side
store. Designed to live inside one ``_embed_keys`` subprocess invocation
and emit a single summary log line at the end.

Module-level state is fine here: each invocation runs in its own
subprocess (see ``pipeline/runners.py::_run_embed_subprocess``), so the
totals never leak across pipeline runs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from loguru import logger

_PHASE_TOTALS: dict[str, float] = {}


@contextmanager
def time_phase(phase: str) -> Iterator[None]:
    """Accumulate wall time spent inside the ``with`` block under ``phase``.

    Multiple entries with the same name accumulate. Cheap (~1µs per call);
    safe to leave on in production.
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        _PHASE_TOTALS[phase] = _PHASE_TOTALS.get(phase, 0.0) + (time.monotonic() - t0)


def reset() -> None:
    """Clear accumulated totals (call before a fresh run)."""
    _PHASE_TOTALS.clear()


def log_summary(n_keys: int) -> None:
    """Emit one INFO line with the recorded breakdown.

    Phases not seen in this run are skipped. Percentages are computed
    against the sum of recorded phases — they describe the breakdown of
    *instrumented* time, not the full subprocess wall (which also
    includes Python startup, model load, and final cleanup).
    """
    if not _PHASE_TOTALS:
        return
    total = sum(_PHASE_TOTALS.values())
    parts: list[str] = []
    for phase, dt in sorted(_PHASE_TOTALS.items(), key=lambda kv: -kv[1]):
        pct = 100 * dt / total if total > 0 else 0
        parts.append(f"{phase}={dt:.1f}s ({pct:.0f}%)")
    logger.info(
        "Embed timing breakdown ({} keys, {:.1f}s instrumented): {}",
        n_keys,
        total,
        "  ".join(parts),
    )
