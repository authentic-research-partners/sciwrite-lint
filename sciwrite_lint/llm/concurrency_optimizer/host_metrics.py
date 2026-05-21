"""Best-effort host-level metric snapshot for controller telemetry.

Wraps ``nvidia-smi`` (GPU VRAM + utilization) and ``psutil`` (host RAM)
behind a small ``HostSnapshot`` dataclass with safe defaults. The
telemetry writer treats every field as optional — when nvidia-smi is
unavailable or the subprocess errors, the row records ``0`` and the
analysis layer can filter on that. Never raises.

``psutil`` is a hard dependency (controller default-on path), so the
RAM probe only guards against runtime OS errors, not a missing import.
``nvidia-smi`` is an OS subprocess and may legitimately be absent
(non-NVIDIA hosts), so those probes still handle ``FileNotFoundError``.

Subprocess shells out — these calls are cheap (~10–30 ms) so we do
them per controller tick in ``asyncio.to_thread``. If that ever
becomes a hot-path concern we can throttle to every Nth tick.
"""

from __future__ import annotations

import subprocess

import psutil
from loguru import logger
from pydantic import BaseModel


class HostSnapshot(BaseModel):
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    gpu_util_pct: float = 0.0
    host_ram_used_mb: float = 0.0
    host_ram_total_mb: float = 0.0


def _gpu_memory_mb() -> tuple[float, float] | None:
    """Return ``(used_mb, total_mb)`` from nvidia-smi, or None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"nvidia-smi memory probe failed: {type(e).__name__}: {e}")
        return None
    if result.returncode != 0:
        return None
    try:
        used_str, total_str = result.stdout.strip().split(",")
        return float(used_str.strip()), float(total_str.strip())
    except (ValueError, IndexError):
        return None


def _gpu_utilization_pct() -> float | None:
    """Return GPU compute utilization (0-100), or None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"nvidia-smi utilization probe failed: {type(e).__name__}: {e}")
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _host_ram_mb() -> tuple[float, float] | None:
    """Return ``(used_mb, total_mb)`` from psutil, or None on probe error."""
    try:
        m = psutil.virtual_memory()
    except OSError as e:
        logger.debug(f"psutil RAM probe failed: {type(e).__name__}: {e}")
        return None
    return m.used / 1e6, m.total / 1e6


def gather_host_snapshot() -> HostSnapshot:
    """Build a ``HostSnapshot`` from whatever sources are available.

    Each probe is independent — failure of one does not affect the
    others. Returns a fully-populated row with zeros for missing fields.
    """
    snap = HostSnapshot()
    gpu_mem = _gpu_memory_mb()
    if gpu_mem is not None:
        snap.vram_used_mb, snap.vram_total_mb = gpu_mem
    gpu_util = _gpu_utilization_pct()
    if gpu_util is not None:
        snap.gpu_util_pct = gpu_util
    ram = _host_ram_mb()
    if ram is not None:
        snap.host_ram_used_mb, snap.host_ram_total_mb = ram
    return snap
