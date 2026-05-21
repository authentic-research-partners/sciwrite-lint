"""Process-wide controller registry — share one controller per partition.

Without this, every ``concurrency_slot`` call instantiates its own
``DynamicConcurrencyController``. Concurrent callers on the same vLLM
endpoint and ``size_class`` then end up with independent semaphores all
reading the same ``/metrics`` and competing on shared KV. The telemetry
shape is "two rows per tick at half the local in-flight" — when a
single client's ``local_in_flight < requests_running``, that's the
duplicate-controller smell.

The registry stores at most one ``DynamicConcurrencyController`` per
``(endpoint, size_class)`` key. Acquire calls increment a refcount and
hand back the existing slot factory; the controller stops only when
the last holder releases. Initial-cap parameters from the *first*
caller win — subsequent callers reuse whatever's already running.
That's a deliberate trade: the alternative (reject mismatched
parameters) makes the API noisy for what is, in practice, a config
constant per ``size_class``.

Scope: this is an in-process dict. With a single ``sciwrite-lint`` CLI
invocation that's the entire shared-state surface. Multi-process
coordination across separate sciwrite-lint runs against one vLLM is
not a concern today and would be solved differently (server-side
coordinator, not a SQLite shared file).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from loguru import logger
from pydantic import BaseModel, ConfigDict, NonNegativeInt

from .compute_cap import SizeClass
from .controller import DynamicConcurrencyController
from .decide import ControllerParams


class _Entry(BaseModel):
    # ``DynamicConcurrencyController`` isn't a pydantic model and we
    # don't want field validation rebuilding it; this just holds the
    # reference plus the refcount for shared ownership.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    controller: DynamicConcurrencyController
    refcount: NonNegativeInt = 0
    # First caller's label, used in the cap-timeline log on final release.
    label: str = "controller"


_REGISTRY: dict[tuple[str, SizeClass], _Entry] = {}
# ``asyncio.Lock`` binds to the first event loop that uses it. Production
# is one loop per CLI invocation so a module-level lock would work, but
# pytest creates a fresh loop per test and an old lock raises
# ``different event loop``. Recreate the lock when the running loop
# changes so tests don't trip on stale locks.
_LOCK: asyncio.Lock | None = None
_LOCK_LOOP: asyncio.AbstractEventLoop | None = None


def _get_lock() -> asyncio.Lock:
    global _LOCK, _LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _LOCK is None or _LOCK_LOOP is not loop:
        _LOCK = asyncio.Lock()
        _LOCK_LOOP = loop
    return _LOCK


def _key(endpoint: str, size_class: SizeClass) -> tuple[str, SizeClass]:
    return (endpoint, size_class)


async def _acquire(
    *,
    endpoint: str,
    size_class: SizeClass,
    params: ControllerParams,
    label: str,
) -> _Entry:
    """Get-or-create the controller for this key, incrementing refcount."""
    key = _key(endpoint, size_class)
    async with _get_lock():
        entry = _REGISTRY.get(key)
        if entry is None:
            ctrl = await DynamicConcurrencyController.from_endpoint(
                endpoint=endpoint,
                size_class=size_class,
                params=params,
            )
            await ctrl.start()
            entry = _Entry(controller=ctrl, label=label)
            _REGISTRY[key] = entry
            logger.debug(
                "registry: created controller endpoint={} size_class={} "
                "label={} initial_cap={}",
                endpoint,
                size_class,
                label,
                ctrl.current_cap,
            )
        else:
            logger.debug(
                "registry: reused controller endpoint={} size_class={} "
                "new_label={} (existing label={} refcount={})",
                endpoint,
                size_class,
                label,
                entry.label,
                entry.refcount,
            )
        entry.refcount += 1
        return entry


async def _release(
    *,
    endpoint: str,
    size_class: SizeClass,
    label: str,
) -> None:
    """Decrement refcount; stop the controller and emit timeline when it
    drops to 0."""
    key = _key(endpoint, size_class)
    async with _get_lock():
        entry = _REGISTRY.get(key)
        if entry is None:
            logger.warning(
                "registry: release for missing key endpoint={} size_class={} "
                "label={} — refcount accounting bug?",
                endpoint,
                size_class,
                label,
            )
            return
        entry.refcount -= 1
        if entry.refcount > 0:
            return
        ctrl = entry.controller
        del _REGISTRY[key]

    # Stop outside the lock — ``ctrl.stop()`` awaits the polling task's
    # cancellation and we don't want that blocking new acquires.
    await ctrl.stop()
    for t, cap, reason in ctrl.cap_history:
        logger.info(
            "{} cap timeline: t={:.1f}s cap={} reason={}",
            entry.label,
            t,
            cap,
            reason,
        )


@asynccontextmanager
async def shared_controller(
    *,
    endpoint: str,
    size_class: SizeClass,
    params: ControllerParams,
    label: str,
) -> AsyncIterator[DynamicConcurrencyController]:
    """Async context manager yielding the shared controller for a partition.

    Concurrent callers with the same ``(endpoint, size_class)`` get the
    same controller instance and share its semaphore. The controller is
    started on first acquire and stopped on last release.
    """
    entry = await _acquire(
        endpoint=endpoint,
        size_class=size_class,
        params=params,
        label=label,
    )
    try:
        yield entry.controller
    finally:
        await _release(endpoint=endpoint, size_class=size_class, label=label)


async def reset_for_tests() -> None:
    """Drop all registry entries — for test isolation only.

    Stops any still-running controllers. Production code should rely on
    refcount-driven cleanup; this exists so tests don't leak controllers
    across runs.
    """
    async with _get_lock():
        entries = list(_REGISTRY.values())
        _REGISTRY.clear()
    for entry in entries:
        await entry.controller.stop()


def active_keys() -> list[tuple[str, SizeClass]]:
    """Snapshot of currently-registered keys — for diagnostics / tests."""
    return list(_REGISTRY.keys())
