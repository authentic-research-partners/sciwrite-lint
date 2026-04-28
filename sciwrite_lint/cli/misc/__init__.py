"""CLI handlers for misc commands (init, parse, override, dismiss-claim, grobid, vllm, services).

This module is a compatibility surface: implementations live in per-command
submodules, but every symbol external callers (``__main__.py``) and tests
import is re-exported here.
"""

from __future__ import annotations

from sciwrite_lint.cli.misc._monitor import (
    _STAGE_LABELS,
    _STAGE_RESOURCES,
    _STAGE_SHORT,
    _build_batch_stages_panel,
    _build_grobid_panel,
    _build_monitor_table,
    _build_stages_panel,
    _build_stages_panel_from_path,
    _build_stages_panel_impl,
    _fetch_vllm_metrics,
    _fetch_vllm_metrics_full,
    _load_stages,
    _print_container_logs,
    _run_containers_monitor,
    _run_vllm_monitor,
)
from sciwrite_lint.cli.misc.containers import run_containers
from sciwrite_lint.cli.misc.dismiss_claim import run_dismiss_claim
from sciwrite_lint.cli.misc.grobid import run_grobid
from sciwrite_lint.cli.misc.init import run_init
from sciwrite_lint.cli.misc.override import run_override
from sciwrite_lint.cli.misc.parse import run_parse
from sciwrite_lint.cli.misc.vision import run_vision
from sciwrite_lint.cli.misc.vllm import run_vllm

__all__ = [
    # Public CLI handlers
    "run_containers",
    "run_dismiss_claim",
    "run_grobid",
    "run_init",
    "run_override",
    "run_parse",
    "run_vision",
    "run_vllm",
    # Re-exported for tests and internal callers
    "_STAGE_LABELS",
    "_STAGE_RESOURCES",
    "_STAGE_SHORT",
    "_build_batch_stages_panel",
    "_build_grobid_panel",
    "_build_monitor_table",
    "_build_stages_panel",
    "_build_stages_panel_from_path",
    "_build_stages_panel_impl",
    "_fetch_vllm_metrics",
    "_fetch_vllm_metrics_full",
    "_load_stages",
    "_print_container_logs",
    "_run_containers_monitor",
    "_run_vllm_monitor",
]
