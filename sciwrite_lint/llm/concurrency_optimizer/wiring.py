"""Single integration point for the dynamic concurrency controller.

Each call site (e.g. vision describe, text ``llm_query_batch``,
claim verification) needs the same skeleton:

- pick between ``DynamicConcurrencyController`` (when dynamic is on)
  and a static ``asyncio.Semaphore``,
- wrap the chosen slot-issuer in a uniform ``async with slot()``
  interface so the request body doesn't branch,
- on shutdown, stop the controller (if any) and log the final cap
  timeline.

Inlining that at every call site would duplicate ~25 lines of glue and
make test injection awkward (mock at three places to swap behaviour).
:func:`concurrency_slot` collapses it to one async context manager.

Portability. This module takes only **primitive parameters** — no
``LintConfig`` dependency. To wire it into a different project that
talks to a vLLM endpoint, copy the ``concurrency_optimizer/`` package
and supply your own ``use_dynamic`` flag and band overrides; nothing
in here references the host project's config schema.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncContextManager, AsyncIterator, Callable

from .compute_cap import SizeClass
from .decide import ControllerParams

SlotFactory = Callable[[], AsyncContextManager[None]]


@asynccontextmanager
async def concurrency_slot(
    *,
    use_dynamic: bool,
    endpoint: str,
    size_class: SizeClass,
    static_cap: int,
    label: str = "controller",
    params: ControllerParams | None = None,
) -> AsyncIterator[SlotFactory]:
    """Yield a slot-issuer factory, picking between dynamic controller
    and static semaphore based on ``use_dynamic``.

    Parameters
    ----------
    use_dynamic
        ``True`` to spin up a ``DynamicConcurrencyController``; ``False``
        for a plain ``asyncio.Semaphore`` (back-compat / simple path).
    endpoint
        vLLM ``/v1`` URL the controller should observe.
    size_class
        Hint for the controller's initial cap calculation.
    static_cap
        Cap used by the static-semaphore path AND as the controller's
        ``upper_bound`` (so client-side and server-side admission
        ceilings stay in sync).
    label
        Free-form tag used in the cap-timeline log line. Helps the
        operator distinguish multiple controllers (``vision`` / ``text`` /
        ``claim-verify``) in mixed logs.
    params
        Optional ``ControllerParams`` for tuning bands and hysteresis.
        Defaults to ``ControllerParams()`` (single-tenant tuning,
        70 % grow target, 80 % shrink ceiling). ``upper_bound`` is
        always overridden with ``static_cap`` to keep the dynamic
        ceiling aligned with the static-semaphore cap.

    Yields
    ------
    A zero-arg callable that returns a fresh ``async with`` slot::

        async with concurrency_slot(
            use_dynamic=True,
            endpoint="http://localhost:5001/v1",
            size_class="heavy",
            static_cap=64,
        ) as slot:
            async def _do_one(payload):
                async with slot():
                    return await client.post(..., json=payload)
            await asyncio.gather(*[_do_one(p) for p in payloads])

    On exit:

    - dynamic path: stops the controller and emits one log line per
      cap-history entry, prefixed with ``label``;
    - static path: nothing to clean up.
    """
    if use_dynamic:
        eff_params = (params or ControllerParams()).model_copy(
            update={"upper_bound": max(1, static_cap)}
        )
        # Share one controller per ``(endpoint, size_class)`` across
        # concurrent callers (gathered ``llm_query_batch`` calls, paper
        # parallelism). See ``registry.py`` for the duplicate-controller
        # bug this prevents.
        from .registry import shared_controller

        async with shared_controller(
            endpoint=endpoint,
            size_class=size_class,
            params=eff_params,
            label=label,
        ) as ctrl:
            yield ctrl.slot
    else:
        sem = asyncio.Semaphore(max(1, static_cap))

        @asynccontextmanager
        async def _static_slot() -> AsyncIterator[None]:
            async with sem:
                yield

        yield _static_slot
