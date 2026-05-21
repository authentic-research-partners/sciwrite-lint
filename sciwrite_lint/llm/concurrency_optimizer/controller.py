"""Asyncio dynamic concurrency controller.

Pulls together:

- ``ResizableSemaphore`` — like ``asyncio.Semaphore`` but its cap can be
  changed at runtime; growing it wakes pending waiters.
- ``DynamicConcurrencyController`` — owns a ``ResizableSemaphore`` plus a
  background task that polls vLLM ``/metrics`` every ``poll_interval_s``
  seconds, calls ``decide()``, and applies the new cap.

Use as an async context manager:

    async with await DynamicConcurrencyController.from_endpoint(
        endpoint="http://localhost:5001/v1",
        size_class="heavy",
    ) as ctrl:
        async def _bounded(payload):
            async with ctrl.slot():
                return await client.post(..., json=payload)
        results = await asyncio.gather(*[_bounded(p) for p in payloads])

Lifecycle: ``__aenter__`` starts the polling task; ``__aexit__`` cancels
it cleanly. The semaphore's slot accounting is independent of the polling
task — if the task crashes, in-flight calls keep working at the
last-applied cap.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from types import TracebackType
from typing import AsyncIterator, Self

from loguru import logger

from .compute_cap import SIZE_CLASS_PROFILES, SizeClass, compute_cap
from .decide import (
    ControllerParams,
    ControllerState,
    Sample,
    decide,
)
from .host_metrics import gather_host_snapshot
from .metrics_probe import probe_kv_pool
from .telemetry import (
    Service,
    TelemetryRow,
    cleanup_partition,
    read_recent,
    write_sample,
)

# Warm-start window: how recent the prior telemetry must be to seed the
# initial cap. Older than this and we'd risk drifting from changed
# hardware / model / workload composition; better to fall back to the
# math-based scenario default and let decide() rediscover.
_WARM_START_MAX_AGE_S = 24 * 60 * 60


def _fetch_served_model_name(endpoint: str) -> str:
    """Best-effort: read the served model id from ``/v1/models``.

    Returns ``""`` on any failure — telemetry treats it as optional.
    """
    import json
    import urllib.parse
    import urllib.request

    if urllib.parse.urlparse(endpoint).scheme not in ("http", "https"):
        return ""
    try:
        with urllib.request.urlopen(  # nosec B310
            f"{endpoint}/models", timeout=3
        ) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("data", [])
        if models:
            return str(models[0].get("id", ""))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"served-model probe failed: {type(e).__name__}: {e}")
    return ""


class ResizableSemaphore:
    """Asyncio semaphore whose cap can be resized at runtime.

    ``asyncio.Semaphore`` doesn't support resize, so this is a small
    re-implementation: explicit waiter queue, non-reentrant, FIFO.
    """

    def __init__(self, initial_cap: int) -> None:
        if initial_cap < 1:
            raise ValueError(f"initial_cap must be >= 1, got {initial_cap}")
        self._cap = initial_cap
        self._in_flight = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiters(self) -> int:
        return len(self._waiters)

    async def acquire(self) -> None:
        """Block until a slot is free; reserve it on return."""
        if self._in_flight < self._cap and not self._waiters:
            self._in_flight += 1
            return
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._waiters.append(fut)
        try:
            await fut
        except BaseException:
            # Cancelled or errored before the slot was granted. If the
            # future is still pending, just drop it from the queue.
            # If it was already completed (slot reserved by _wake_waiters),
            # release the unused slot so we don't leak capacity.
            if fut.done() and not fut.cancelled():
                self._in_flight -= 1
                self._wake_waiters()
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self) -> None:
        """Free one slot. Wakes the next waiter if any."""
        if self._in_flight <= 0:
            raise RuntimeError("release() called more times than acquire()")
        self._in_flight -= 1
        self._wake_waiters()

    def resize(self, new_cap: int) -> None:
        """Change the cap. Growing wakes waiters; shrinking is lazy
        (in-flight requests keep their slots and drain naturally)."""
        if new_cap < 1:
            raise ValueError(f"new_cap must be >= 1, got {new_cap}")
        old = self._cap
        self._cap = new_cap
        if new_cap > old:
            self._wake_waiters()

    def _wake_waiters(self) -> None:
        while self._waiters and self._in_flight < self._cap:
            fut = self._waiters.popleft()
            if not fut.done():
                self._in_flight += 1
                fut.set_result(None)

    async def __aenter__(self) -> ResizableSemaphore:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


class DynamicConcurrencyController:
    """Long-lived concurrency controller backed by ``ResizableSemaphore``.

    Construct via :meth:`from_endpoint` so the initial cap is computed
    from a live ``/metrics`` probe + ``compute_cap``.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        size_class: SizeClass,
        initial_cap: int,
        params: ControllerParams,
        poll_interval_s: float = 2.0,
    ) -> None:
        self.endpoint = endpoint
        self.size_class = size_class
        self.params = params
        self.poll_interval_s = poll_interval_s
        self._sem = ResizableSemaphore(initial_cap)
        self._state = ControllerState(current_cap=initial_cap)
        self._task: asyncio.Task[None] | None = None
        self._cap_history: list[tuple[float, int, str]] = []  # (t, cap, reason)
        self._sample_history: list[Sample] = []

    @classmethod
    async def from_endpoint(
        cls,
        *,
        endpoint: str,
        size_class: SizeClass,
        params: ControllerParams | None = None,
        poll_interval_s: float = 2.0,
        override: int | None = None,
    ) -> Self:
        import time as _time

        pool = await asyncio.to_thread(probe_kv_pool, endpoint)
        eff_params = params or ControllerParams(override=override)

        # Math-based scenario default — always valid, conservative.
        math_cap = compute_cap(
            kv_tokens=pool.kv_tokens,
            size_class=size_class,
            override=eff_params.override,
            upper_bound=eff_params.upper_bound,
            lower_bound=eff_params.lower_bound,
        )

        # Floor every shrink at half the math-based starting cap. The
        # math_cap is the conservative scenario default that should
        # always be safe; dropping below half of it means we've
        # abandoned the scenario, which is more pessimistic than any
        # single-paper signal warrants. Default of 1 in ControllerParams
        # is a no-op safety net for tests / standalone use.
        eff_params = eff_params.model_copy(
            update={"effective_min_cap": max(1, math_cap // 2)}
        )

        # Warm start from telemetry: if a recent run on this same
        # ``(endpoint, size_class)`` partition observed a stable cap,
        # use that as the initial value instead of math_cap. Saves the
        # ~15-30 s of "rediscovery" ramp on every fresh run. Skipped
        # when ``override`` is explicitly set (operator wants a fixed
        # value) or when the recent observation is implausibly low —
        # the math floor protects against a previous run that crashed
        # to ``lower_bound`` from anchoring this run there.
        initial_cap = math_cap
        warm_start_age_s: float | None = None
        if eff_params.override is None:
            recent = await asyncio.to_thread(read_recent, endpoint, size_class, None, 1)
            if recent:
                age_s = _time.time() - recent[0].ts
                last_cap = recent[0].current_cap
                # Trust if observation is recent AND not absurdly below
                # the math floor (which would indicate a crash-shrunk
                # tail rather than a real workload signal).
                if age_s < _WARM_START_MAX_AGE_S and last_cap >= max(1, math_cap // 2):
                    candidate = max(eff_params.lower_bound, last_cap)
                    candidate = min(eff_params.upper_bound, candidate)
                    initial_cap = candidate
                    warm_start_age_s = age_s

        if warm_start_age_s is not None:
            logger.info(
                "Controller warm-start: endpoint={} class={} kv_tokens={:,} "
                "initial_cap={} (from telemetry, age={:.0f}s; math says {})",
                endpoint,
                size_class,
                pool.kv_tokens,
                initial_cap,
                warm_start_age_s,
                math_cap,
            )
        else:
            logger.info(
                "Controller starting: endpoint={} class={} kv_tokens={:,} "
                "initial_cap={} typical_tokens={}",
                endpoint,
                size_class,
                pool.kv_tokens,
                initial_cap,
                SIZE_CLASS_PROFILES[size_class].typical_prompt_tokens,
            )

        return cls(
            endpoint=endpoint,
            size_class=size_class,
            initial_cap=initial_cap,
            params=eff_params,
            poll_interval_s=poll_interval_s,
        )

    @property
    def current_cap(self) -> int:
        return self._sem.cap

    @property
    def cap_history(self) -> list[tuple[float, int, str]]:
        return list(self._cap_history)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._sem:
            yield

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def _run(self) -> None:
        import time as _time

        from .vllm_metrics import fetch_metrics

        loop = asyncio.get_event_loop()
        start_t = loop.time()
        self._cap_history.append((0.0, self._state.current_cap, "init"))

        # Trim this partition's persistent telemetry to KEEP_LAST_N
        # before we start writing — keeps the global usage.db bounded
        # without paying for a delete on every tick.
        await asyncio.to_thread(cleanup_partition, self.endpoint, self.size_class)

        # Service tag separates text (small/medium/heavy) from vision
        # in the telemetry — most downstream queries want one or the
        # other. ``model_served_name`` adds another layer of granularity.
        service: Service = "vision" if self.size_class == "vision" else "text"
        model_name = await asyncio.to_thread(_fetch_served_model_name, self.endpoint)

        # Persist the init sample so monitor tools can show the starting
        # cap even before the first decide tick.
        await asyncio.to_thread(
            write_sample,
            TelemetryRow(
                ts=_time.time(),
                endpoint=self.endpoint,
                size_class=self.size_class,
                service=service,
                model_served_name=model_name,
                current_cap=self._sem.cap,
                local_in_flight=self._sem.in_flight,
                reason="init",
            ),
        )

        # Track cumulative preemption counter across ticks; the delta
        # is what decide() consumes. ``None`` until the first reading
        # so the very first tick reports a delta of 0 (we have no prior
        # baseline to subtract).
        prev_preemptions: float | None = None

        while True:
            try:
                await asyncio.sleep(self.poll_interval_s)
                metrics = await asyncio.to_thread(fetch_metrics, self.endpoint)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(
                    "controller metrics fetch failed: {}: {}",
                    type(e).__name__,
                    e,
                )
                continue
            if not metrics:
                # /metrics unreachable — skip this tick. cleanup will
                # log; we don't want to crash the controller over a
                # transient blip.
                continue

            preempt_now = float(metrics.get("num_preemptions", 0))
            if prev_preemptions is None:
                preempt_delta = 0
            else:
                preempt_delta = max(0, int(preempt_now - prev_preemptions))
            prev_preemptions = preempt_now

            sample = Sample(
                kv_cache_pct=float(metrics.get("kv_cache_pct", 0.0)),
                requests_running=int(metrics.get("requests_running", 0)),
                requests_waiting=int(metrics.get("requests_waiting", 0)),
                local_in_flight=self._sem.in_flight,
                preemption_delta=preempt_delta,
            )
            self._sample_history.append(sample)
            decision = decide(self._state, sample, self.params)
            self._state = decision.state
            t = loop.time() - start_t
            if decision.changed:
                logger.info(
                    "[{}] cap {} -> {} ({}) at t={:.1f}s kv={:.1%} run={} wait={}",
                    self.size_class,
                    self._sem.cap,
                    decision.next_cap,
                    decision.reason,
                    t,
                    sample.kv_cache_pct,
                    sample.requests_running,
                    sample.requests_waiting,
                )
                self._sem.resize(decision.next_cap)
                self._cap_history.append((t, decision.next_cap, decision.reason))

            host = await asyncio.to_thread(gather_host_snapshot)

            # Persist every tick (not just changes) so the monitor can
            # show live load even during long hold periods. All vLLM
            # /metrics fields are recorded as cumulative values; the
            # analysis layer takes deltas to derive throughput / rates.
            row = TelemetryRow(
                ts=_time.time(),
                endpoint=self.endpoint,
                size_class=self.size_class,
                service=service,
                model_served_name=model_name,
                current_cap=self._sem.cap,
                local_in_flight=sample.local_in_flight,
                reason=decision.reason,
                requests_running=sample.requests_running,
                requests_waiting=sample.requests_waiting,
                requests_swapped=int(metrics.get("requests_swapped", 0)),
                kv_cache_pct=sample.kv_cache_pct,
                num_preemptions=float(metrics.get("num_preemptions", 0)),
                prefix_cache_hits=float(metrics.get("prefix_cache_hits", 0)),
                prefix_cache_queries=float(metrics.get("prefix_cache_queries", 0)),
                prompt_tokens_total=float(metrics.get("prompt_tokens_total", 0)),
                generation_tokens_total=float(
                    metrics.get("generation_tokens_total", 0)
                ),
                e2e_latency_sum=float(metrics.get("e2e_latency_sum", 0)),
                e2e_latency_count=float(metrics.get("e2e_latency_count", 0)),
                ttft_sum=float(metrics.get("ttft_sum", 0)),
                ttft_count=float(metrics.get("ttft_count", 0)),
                itl_sum=float(metrics.get("itl_sum", 0)),
                itl_count=float(metrics.get("itl_count", 0)),
                finish_stop=float(metrics.get("req_success_stop", 0)),
                finish_length=float(metrics.get("req_success_length", 0)),
                finish_abort=float(metrics.get("req_success_abort", 0)),
                finish_error=float(metrics.get("req_success_error", 0)),
                vram_used_mb=host.vram_used_mb,
                vram_total_mb=host.vram_total_mb,
                gpu_util_pct=host.gpu_util_pct,
                host_ram_used_mb=host.host_ram_used_mb,
                host_ram_total_mb=host.host_ram_total_mb,
            )
            await asyncio.to_thread(write_sample, row)
