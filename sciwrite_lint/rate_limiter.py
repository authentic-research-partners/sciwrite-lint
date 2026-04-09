"""Async rate limiter and retry helpers for HTTP APIs.

- ``MonotonicRateLimiter``: Token-bucket rate limiter using
  ``time.monotonic()`` — works across multiple ``asyncio.run()`` calls.
- ``retry_on_transient()``: Retry an async callable on 429/5xx responses
  with exponential backoff (powered by tenacity).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)


class MonotonicRateLimiter:
    """Token-bucket rate limiter using ``time.monotonic()``.

    Args:
        max_rate: Maximum number of tokens (requests) in the bucket.
        time_period: Period in seconds over which tokens refill.

    Usage::

        limiter = MonotonicRateLimiter(2, 3)  # 2 requests per 3 seconds
        async with limiter:
            await do_request()
    """

    def __init__(self, max_rate: float, time_period: float = 1.0) -> None:
        self.max_rate = max_rate
        self.time_period = time_period
        self._tokens = max_rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self.max_rate / self.time_period)
        self._tokens = min(self.max_rate, self._tokens + new_tokens)
        self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) * (self.time_period / self.max_rate)
            await asyncio.sleep(wait)

    def pause(self, seconds: float) -> None:
        """Drain all tokens and delay refill by *seconds*.

        Called when a 429 Retry-After header is received. All tasks
        waiting on ``acquire()`` will block until the pause expires
        and tokens refill naturally.
        """
        self._tokens = 0.0
        self._last_refill = time.monotonic() + seconds

    async def __aenter__(self) -> MonotonicRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_MAX_RETRY_AFTER = 60.0  # cap Retry-After pause to prevent pipeline stalls


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Extract delay from Retry-After header (seconds or HTTP-date)."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    # HTTP-date format (RFC 7231)
    from email.utils import parsedate_to_datetime

    try:
        from datetime import datetime, timezone

        target = parsedate_to_datetime(value)
        delay = (target - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delay)
    except (ValueError, TypeError):
        return None


def _is_transient(response: httpx.Response) -> bool:
    """Return True if the response status code is transient (should retry)."""
    return response.status_code in _TRANSIENT_CODES


async def retry_on_transient(
    request_fn: object,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    label: str = "",
) -> httpx.Response:
    """Call an async function that returns an httpx.Response, retrying on transient errors.

    Retries on 429/5xx status codes and network errors (timeouts, connection
    failures) with exponential backoff. Powered by tenacity.

    Args:
        request_fn: Async callable returning httpx.Response.
        max_retries: Maximum number of attempts (default: 3).
        base_delay: Base delay in seconds for exponential backoff (default: 2.0).
        label: Label for log messages.

    Returns:
        The httpx.Response from the first successful (non-transient) attempt,
        or the last response after all retries are exhausted.
    """

    def _return_last_response(retry_state: object) -> httpx.Response:
        """On exhausted retries, return the last response instead of raising."""
        outcome = retry_state.outcome  # type: ignore[attr-defined]
        if outcome and not outcome.failed:
            return outcome.result()  # type: ignore[return-value]
        raise outcome.exception()  # type: ignore[misc]

    @retry(
        retry=(
            retry_if_result(_is_transient)
            | retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
        ),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=30),
        stop=stop_after_attempt(max_retries),
        reraise=True,
        retry_error_callback=_return_last_response,
        before_sleep=lambda rs: logger.warning(
            "{} attempt {}/{} ({}), retrying in {:.0f}s",
            label or "HTTP request",
            rs.attempt_number,
            max_retries,
            (
                f"status {rs.outcome.result().status_code}"
                if rs.outcome and not rs.outcome.failed
                else str(rs.outcome.exception() if rs.outcome else "unknown")
            ),
            rs.next_action.sleep if rs.next_action else 0,  # type: ignore[union-attr]
        ),
    )
    async def _do_request() -> httpx.Response:
        return await request_fn()  # type: ignore[operator]

    return await _do_request()


async def rate_limited_get(
    limiter: MonotonicRateLimiter,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 10.0,
    label: str = "",
    client: httpx.AsyncClient | None = None,
    service: str = "",
) -> httpx.Response:
    """Acquire rate limiter, make GET request with transient-error retry.

    If *client* is provided, uses it directly. Otherwise creates a
    temporary AsyncClient per request. On 429 with Retry-After header,
    pauses the limiter so all queued tasks wait.

    If *service* is set (e.g. ``"core"``), records elapsed time and
    error status to the active usage run via ``usage.tracked()``.
    """
    from contextlib import nullcontext

    from sciwrite_lint.usage import tracked

    track_ctx: Any = tracked(service) if service else nullcontext()

    async with track_ctx as t:
        async with limiter:
            if client is not None:
                resp = await retry_on_transient(
                    lambda: client.get(url, params=params or {}, headers=headers or {}),
                    label=label,
                )
            else:
                async with httpx.AsyncClient(
                    timeout=timeout, follow_redirects=True
                ) as tmp_client:
                    resp = await retry_on_transient(
                        lambda: tmp_client.get(
                            url, params=params or {}, headers=headers or {}
                        ),
                        label=label,
                    )
            if resp.status_code == 429:
                delay = _parse_retry_after(resp)
                if delay is not None and delay > 0:
                    delay = min(delay, _MAX_RETRY_AFTER)
                    logger.warning(
                        "{}: 429 with Retry-After {:.0f}s, pausing limiter",
                        label or "HTTP request",
                        delay,
                    )
                    limiter.pause(delay)
            if resp.status_code >= 400 and t is not None:
                t.error = True
            return resp
