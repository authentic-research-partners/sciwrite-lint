"""Stage status tracking for the pipeline monitor.

Writes ``running``/``done``/``failed`` entries to each paper's
``workspace.db`` so ``sciwrite-lint containers monitor`` can show live stage
progress.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from sciwrite_lint.references.workspace_db import (
    get_db,
    update_pipeline_stage,
)


def _track(refs_dir: Path, stage: str, status: str, detail: str = "") -> None:
    """Write a stage status update to workspace.db."""
    try:
        with get_db(refs_dir) as conn:
            update_pipeline_stage(conn, stage, status, detail)
    except sqlite3.Error as e:
        logger.debug(f"pipeline stage tracking failed ({type(e).__name__}: {e})")


class _StageStatus:
    """Mutable detail holder yielded by :func:`_stage_tracking`.

    Assign ``status.detail`` inside the ``with`` block to set the detail
    string recorded when the stage is marked ``done``.
    """

    __slots__ = ("detail",)

    def __init__(self) -> None:
        self.detail: str = ""


@contextmanager
def _stage_tracking(
    refs_dirs: Path | list[Path],
    stages: str | list[str],
) -> Iterator[_StageStatus]:
    """Mark one or more pipeline stages ``running`` → ``done`` / ``failed``.

    On enter, every (refs_dir, stage) pair is marked ``running``. On clean
    exit, every pair is marked ``done`` with the detail set on the yielded
    ``_StageStatus``. On exception, every pair is marked ``failed`` with a
    truncated error message and the exception is re-raised so the caller
    can still set ``ctx.error`` or propagate.

    Use for simple stages where ``running`` and ``done`` are tracked in the
    same scope. For stages where ``done`` is deferred to a later step (e.g.
    multi-paper parse + batch embed), track manually with :func:`_track`.
    """
    dirs = [refs_dirs] if isinstance(refs_dirs, Path) else list(refs_dirs)
    stage_list = [stages] if isinstance(stages, str) else list(stages)

    for d in dirs:
        for s in stage_list:
            _track(d, s, "running")

    status = _StageStatus()
    try:
        yield status
    except Exception as e:
        msg = str(e)[:200]
        for d in dirs:
            for s in stage_list:
                _track(d, s, "failed", msg)
        raise
    else:
        for d in dirs:
            for s in stage_list:
                _track(d, s, "done", status.detail)


@contextmanager
def _stage_failure_guard(
    refs_dirs: Path | list[Path],
    stages: str | list[str],
) -> Iterator[None]:
    """Mark stages ``failed`` if the block raises, but never marks running/done.

    Use for stages where ``running`` and ``done`` transitions are managed
    per-item outside this scope (e.g. multi-paper parse + batch embed, or
    batch cited-vision where different ctxs have different done details).
    This guard only guarantees that an uncaught batch-level failure does
    not leave stages stuck in ``running`` in the monitor DB.
    """
    dirs = [refs_dirs] if isinstance(refs_dirs, Path) else list(refs_dirs)
    stage_list = [stages] if isinstance(stages, str) else list(stages)
    try:
        yield
    except Exception as e:
        msg = str(e)[:200]
        for d in dirs:
            for s in stage_list:
                _track(d, s, "failed", msg)
        raise
