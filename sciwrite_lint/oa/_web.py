"""Workspace-free web-content fetch: URL → extracted markdown in memory.

Public API. Returns a :class:`WebResult` with ``.markdown`` populated; the
caller writes to disk if they want persistence. The internal orchestrator
:func:`sciwrite_lint.web.fetch_web_content` is a thin wrapper that adds
the workspace-aware file write on top of this.
"""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from sciwrite_lint.oa._models import FetchConfig, WebResult
from sciwrite_lint.web import (
    BROWSER_HEADERS,
    _extract_redirect_url,
    _extract_title,
    _fetch_once,
    _format_status_reason,
    _html_to_markdown,
    _www_variant,
    is_dead_status,
)


async def fetch_web(
    url: str,
    *,
    config: FetchConfig | None = None,
    client: httpx.AsyncClient | None = None,
) -> WebResult:
    """Fetch ``url``, extract content as markdown, return it in memory.

    Follows JS/meta-redirects. Retries once with the www-toggled hostname
    variant only when the failure might be hostname-specific — a 404
    response, or a DNS/connection error that didn't reach a live server.
    Refusals (401/403/429/451), server errors (5xx), TLS failures,
    timeouts, and decoding errors affect both hostname variants equally,
    so retrying is skipped. Does not touch the filesystem.

    :param client: Optional pre-built :class:`httpx.AsyncClient`. When
        provided, its transport/base-url/etc. are used verbatim (useful for
        tests with :class:`httpx.MockTransport`). The caller is responsible
        for closing it. When omitted, a temporary client is created using
        ``config.timeout`` and ``config.user_agent`` and closed on return.

    Returns a :class:`WebResult` with:
    - ``url_alive`` — True if a 2xx/3xx response was produced.
    - ``blocked`` — True when we could not verify but the URL may still be
      valid (4xx refusals, 5xx, TLS/timeout/connection/decoding errors).
      False only for explicit HTTP 404/410 or unfixable URL issues.
    - ``markdown`` — extracted markdown (None if extraction failed).
    - ``title`` — extracted ``<title>`` (may be None).
    - ``final_url`` — URL that actually produced the content (after
      www-variant retries and JS/meta-redirect chasing).
    - ``error`` — short reason phrase (composed into downstream issue
      strings as ``"Blocked by {error}: ..."`` or ``"Dead URL ({error}): ..."``).
    """
    cfg = config or FetchConfig()

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            timeout=cfg.timeout,
            follow_redirects=True,
            headers={"User-Agent": cfg.user_agent, **BROWSER_HEADERS},
        )
    assert client is not None
    try:
        resp, fetch_failure = await _fetch_once(url, client, cfg.user_agent)

        if fetch_failure is not None and resp is None:
            await asyncio.sleep(1)
            resp, fetch_failure = await _fetch_once(url, client, cfg.user_agent)

        # Retry www-variant only when the failure is hostname-level:
        # - Response 404 (path may exist on www-variant)
        # - DNS/connection failure flagged hostname_level=True
        # WAF refusals, server errors, TLS, decoding, timeouts affect both
        # variants equally; retrying wastes a request.
        should_try_alt = (resp is not None and resp.status_code == 404) or (
            fetch_failure is not None and fetch_failure.hostname_level
        )
        if should_try_alt:
            alt = _www_variant(url)
            if alt:
                alt_resp, alt_failure = await _fetch_once(alt, client, cfg.user_agent)
                if (
                    alt_failure is None
                    and alt_resp is not None
                    and alt_resp.status_code < 400
                ):
                    resp, fetch_failure = alt_resp, alt_failure
                    url = alt

        if fetch_failure is not None:
            return WebResult(
                url_alive=False,
                error=fetch_failure.reason,
                blocked=fetch_failure.blocked,
                final_url=url,
            )

        assert resp is not None
        if resp.status_code >= 400:
            dead = is_dead_status(resp.status_code)
            if dead:
                error_msg = f"HTTP {resp.status_code}"
            else:
                error_msg = _format_status_reason(resp.status_code)
            return WebResult(
                url_alive=False,
                status_code=resp.status_code,
                error=error_msg,
                final_url=url,
                blocked=not dead,
            )

        content_type = resp.headers.get("content-type", "")
        html = resp.text
        title = _extract_title(html)
        markdown = _html_to_markdown(html, url)

        if not markdown or len(markdown.strip()) < 100:
            redirect_url = _extract_redirect_url(html, url)
            if redirect_url and redirect_url != url:
                logger.info("Following redirect from {} to {}", url, redirect_url)
                redirect_resp, redirect_err = await _fetch_once(
                    redirect_url, client, cfg.user_agent
                )
                if redirect_err is None and redirect_resp is not None:
                    html = redirect_resp.text
                    url = redirect_url
                    title = _extract_title(html) or title
                    markdown = _html_to_markdown(html, url)

        if not markdown or len(markdown.strip()) < 100:
            error = (
                "Content extraction failed"
                if not markdown
                else f"Extracted content too short ({len(markdown.strip())} chars)"
            )
            return WebResult(
                url_alive=True,
                status_code=resp.status_code,
                content_type=content_type,
                title=title,
                error=error,
                final_url=url,
            )

        return WebResult(
            url_alive=True,
            status_code=resp.status_code,
            content_type=content_type,
            markdown=markdown,
            title=title,
            final_url=url,
        )
    finally:
        if own_client:
            await client.aclose()
