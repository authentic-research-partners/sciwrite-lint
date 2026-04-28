"""Web resource verification: URL liveness check + content download as markdown.

For citations that are blog posts, GitHub repos, or other web resources rather
than academic papers. These skip academic APIs and instead verify the URL is
alive and download content for LLM claim-checking.

Public-API surface note: the underscore-prefixed helpers ``_fetch_once``,
``_extract_title``, ``_html_to_markdown``, ``_www_variant``,
``_extract_redirect_url``, ``_format_status_reason``, plus the public
``is_dead_status`` and ``BROWSER_HEADERS``, are imported by the public
OA-acquisition API at ``sciwrite_lint.oa._web``. Treat them as part of the
effective public contract — do not rename or remove them without updating
that consumer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic import BaseModel

from sciwrite_lint._network import is_hostname_safe

# Defaults (overridable via LintConfig)
_DEFAULT_TIMEOUT = 15.0
# Browser-like UA. Scientific citations routinely point behind WAFs
# (Cloudflare, Akamai, Imperva) that 403 anything self-identifying as a
# script. A realistic UA lets us distinguish "this URL is dead" from
# "this URL refused us" — see BROWSER_HEADERS for the companion signals.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)
_MAX_HTML_BYTES = 10 * 1024 * 1024  # 10 MB — reject pages larger than this

# Header set real browsers send alongside User-Agent. WAFs fingerprint the
# combination, not just the UA; a UA string without these headers still
# reads as "non-browser" to most bot-protection stacks. Content-Encoding
# is handled by httpx's default "Accept-Encoding: gzip, deflate" — we do
# not duplicate it here to avoid double-decoding edge cases.
BROWSER_HEADERS: dict[str, str] = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# HTTP status codes where the server positively confirmed the URL is gone.
# This is the ONLY evidence we accept for classifying a URL as dead:
# 404 Not Found and 410 Gone are explicit "not here" responses from the
# server itself. Every other non-2xx/3xx outcome is ambiguous —
# server refusals (401/403/429/451), server errors (5xx), TLS failures,
# timeouts, connection errors, and decoding failures all mean "we could
# not verify", not "URL is gone". Per ``.claude/rules/error-handling.md``:
# "API failures are not empty results."
_DEAD_STATUS_CODES: frozenset[int] = frozenset({404, 410})

# HTTP 4xx codes where HEAD→GET escalation may succeed. Some WAFs refuse
# HEAD from non-browser clients but accept GET with the same headers; 405
# explicitly signals Method Not Allowed.
_HEAD_RETRY_STATUSES: frozenset[int] = frozenset({401, 403, 405, 429, 451})


def is_dead_status(status_code: int) -> bool:
    """True only for status codes that positively confirm the URL is gone.

    Dead means: the server itself explicitly said the resource does not
    exist (404) or has been removed (410). Everything else, including
    4xx refusals and 5xx server errors, is unverifiable — not dead.
    """
    return status_code in _DEAD_STATUS_CODES


def _format_status_reason(status_code: int) -> str:
    """Short human-readable phrase describing a non-alive HTTP status.

    Used as the ``reason`` field of blocked WebResults, composed into the
    user-facing issue string ``"Blocked by {reason}: ..."``. Keeps the
    "site" wording for 4xx refusals (our earlier convention) and uses
    "server error" for 5xx so the user sees whether the problem is the
    server refusing us or the server malfunctioning.
    """
    if status_code in _HEAD_RETRY_STATUSES:
        return f"site (HTTP {status_code})"
    if 500 <= status_code < 600:
        return f"server error (HTTP {status_code})"
    return f"HTTP {status_code}"


class _FetchFailure(NamedTuple):
    """Classified failure from a single fetch attempt.

    Carries three pieces of information:

    - ``reason``: short phrase describing what went wrong (e.g.,
      ``"timeout (ConnectTimeout)"``, ``"TLS error (...)"``,
      ``"connection error (...)"``). Composed into downstream issue
      strings.
    - ``blocked``: ``True`` if the URL may still be valid and we simply
      could not verify (network/TLS/timeout/protocol errors, oversized
      response). ``False`` only for unfixable-as-written cases (invalid
      URL, unsupported scheme) — those are genuinely dead.
    - ``hostname_level``: ``True`` if trying a different hostname
      (www-variant) might succeed where this attempt failed (DNS /
      connection failures). ``False`` for failures that affect both
      variants equally (TLS, decoding, protocol, size limits).
    """

    reason: str
    blocked: bool
    hostname_level: bool


def _classify_http_exception(exc: httpx.HTTPError) -> _FetchFailure:
    """Classify an httpx exception into reason/blocked/hostname_level.

    Order matters: some httpx exceptions inherit from multiple base
    classes (e.g. ``ConnectTimeout`` is both ``TimeoutException`` and
    ``ConnectError``), so the checks below are ordered from most- to
    least-specific.
    """
    if isinstance(exc, httpx.TimeoutException):
        return _FetchFailure(
            reason=f"timeout ({type(exc).__name__})",
            blocked=True,
            hostname_level=False,
        )
    if isinstance(exc, httpx.ConnectError):
        msg = str(exc)
        upper = msg.upper()
        if "SSL" in upper or "CERTIFICATE" in upper or "TLS" in upper:
            return _FetchFailure(
                reason=f"TLS error ({msg})",
                blocked=True,
                hostname_level=False,
            )
        return _FetchFailure(
            reason=f"connection error ({msg})",
            blocked=True,
            hostname_level=True,
        )
    if isinstance(exc, (httpx.ReadError, httpx.WriteError, httpx.CloseError)):
        return _FetchFailure(
            reason=f"connection error ({exc})",
            blocked=True,
            hostname_level=False,
        )
    if isinstance(exc, httpx.ProtocolError):
        return _FetchFailure(
            reason=f"protocol error ({exc})",
            blocked=True,
            hostname_level=False,
        )
    if isinstance(exc, httpx.DecodingError):
        return _FetchFailure(
            reason=f"decoding error ({exc})",
            blocked=True,
            hostname_level=False,
        )
    if isinstance(exc, httpx.TooManyRedirects):
        return _FetchFailure(
            reason="too many redirects",
            blocked=True,
            hostname_level=False,
        )
    if isinstance(exc, httpx.InvalidURL):
        # The citation URL cannot be parsed — no amount of retrying helps.
        return _FetchFailure(
            reason=f"invalid URL ({exc})",
            blocked=False,
            hostname_level=False,
        )
    if isinstance(exc, httpx.UnsupportedProtocol):
        return _FetchFailure(
            reason=f"unsupported scheme ({exc})",
            blocked=False,
            hostname_level=False,
        )
    # Unknown error class: default to blocked (we did not verify, so we
    # cannot honestly claim the URL is dead).
    return _FetchFailure(
        reason=f"{type(exc).__name__}: {exc}",
        blocked=True,
        hostname_level=False,
    )


def _request_headers(user_agent: str) -> dict[str, str]:
    """Build the full browser-like header set for a single request."""
    return {"User-Agent": user_agent, **BROWSER_HEADERS}


def _www_variant(url: str) -> str | None:
    """Return the www-toggled variant of a URL, or None if not applicable."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host.startswith("www."):
        alt_host = host[4:]
    else:
        alt_host = f"www.{host}"
    # Rebuild netloc preserving port if present
    if parsed.port:
        alt_netloc = f"{alt_host}:{parsed.port}"
    else:
        alt_netloc = alt_host
    return urlunparse(parsed._replace(netloc=alt_netloc))


class WebResult(BaseModel):
    """Result of a web resource verification attempt.

    Three-valued outcome for failed attempts:
    - ``url_alive=True``: server served content (2xx/3xx).
    - ``url_alive=False, blocked=True``: we could not verify the URL
      (4xx refusals, 5xx, TLS/timeout/connection/decoding errors). The
      URL may still be valid. Downstream maps this to ``api_match="web_blocked"``.
    - ``url_alive=False, blocked=False``: server positively confirmed the
      URL is gone (HTTP 404/410) or URL form is unfixable.

    ``hostname_level`` is True when a www-variant retry might succeed —
    404 responses or DNS/unreachable connection failures. Set by the
    fetch layer so retry logic can make typed decisions instead of
    parsing the error string.
    """

    url_alive: bool
    status_code: int = 0
    content_type: str = ""
    local_path: str | None = None  # path relative to references/
    title: str | None = None  # extracted page title
    error: str | None = None
    resolved_url: str | None = (
        None  # URL that actually responded (may differ from input after www alternation)
    )
    blocked: bool = False
    hostname_level: bool = False


async def check_url(
    url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> WebResult:
    """Check if a URL is alive via HEAD request.

    HEAD→GET escalates on 405 or on WAF-style refusals (401/403/429/451)
    where some servers accept GET but refuse HEAD.

    Alternates between www and non-www variants only when the failure
    might be hostname-specific (404 response, or a DNS/connection error
    in ``_check_url_once``). Retries are skipped for failures that affect
    both hostname variants equally (4xx refusals, 5xx, TLS, timeouts).

    Classifies refusals and infrastructure failures as ``blocked=True``;
    only HTTP 404/410 produce ``blocked=False`` (genuinely dead).
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=_request_headers(user_agent),
        )
    assert client is not None
    try:
        result = await _check_url_once(url, client, user_agent)

        # One retry on transient connection-level failures. Typed check
        # instead of string parsing: ``hostname_level`` + no HTTP status
        # means the request never reached a live server (DNS or connect
        # failure) — those are the cases worth a short-sleep retry.
        # TLS / decoding / protocol errors are deterministic, no retry.
        if not result.url_alive and result.status_code == 0 and result.hostname_level:
            import asyncio as _asyncio

            await _asyncio.sleep(1)
            result = await _check_url_once(url, client, user_agent)

        if result.url_alive:
            result.resolved_url = url
            return result

        # Try www-variant only when the failure is hostname-level (404
        # from the server, or a connection error that didn't reach a
        # live server). Blocked statuses (WAF refusals, 5xx, TLS) apply
        # equally to both variants — retrying burns a request with no
        # informational gain.
        if _should_retry_alt_hostname(result):
            alt = _www_variant(url)
            if alt:
                alt_result = await _check_url_once(alt, client, user_agent)
                if alt_result.url_alive:
                    alt_result.resolved_url = alt
                    return alt_result
        return result
    finally:
        if own_client:
            await client.aclose()


def _should_retry_alt_hostname(result: WebResult) -> bool:
    """True if the WebResult failure might be resolved by a www-variant retry.

    Single typed check: the fetch layer classified this attempt as
    hostname-level (DNS/unreachable connection error, or an explicit
    404 response). No string parsing — the flag is authoritative.
    """
    return result.hostname_level


async def _check_url_once(
    url: str,
    client: httpx.AsyncClient,
    user_agent: str,
) -> WebResult:
    """Single-attempt URL liveness check.

    Tries HEAD first (cheap); escalates to GET on 405 or on 4xx WAF
    refusals — some WAFs refuse HEAD from non-browser clients but
    accept GET with the same headers.
    """
    headers = _request_headers(user_agent)
    try:
        resp = await client.head(url, headers=headers)
        if resp.status_code in _HEAD_RETRY_STATUSES:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as e:
        failure = _classify_http_exception(e)
        return WebResult(
            url_alive=False,
            error=failure.reason,
            blocked=failure.blocked,
            hostname_level=failure.hostname_level,
        )

    alive = 200 <= resp.status_code < 400
    content_type = resp.headers.get("content-type", "")
    blocked = not alive and not is_dead_status(resp.status_code)
    error: str | None = None
    if not alive:
        error = (
            _format_status_reason(resp.status_code)
            if blocked
            else f"HTTP {resp.status_code}"
        )
    # An explicit 404 may be path-on-this-hostname-specific; the path
    # might exist on the www-variant. 410 is explicit "gone" — no retry.
    # Other statuses affect both variants equally.
    hostname_level = resp.status_code == 404
    return WebResult(
        url_alive=alive,
        status_code=resp.status_code,
        content_type=content_type,
        blocked=blocked,
        error=error,
        hostname_level=hostname_level,
    )


async def fetch_web_content(
    url: str,
    key: str,
    references_dir: Path,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> WebResult:
    """Fetch a web page, extract content as markdown, save to references/.

    Thin workspace-aware wrapper around :func:`sciwrite_lint.oa.fetch_web`.
    Writes ``{key}_web.md`` into ``references_dir`` with a source header
    when extraction succeeds; returns a :class:`WebResult` preserving the
    internal-caller contract (``local_path`` + ``resolved_url``).
    """
    from sciwrite_lint.oa import FetchConfig, fetch_web

    public = await fetch_web(
        url,
        config=FetchConfig(timeout=timeout, user_agent=user_agent),
        client=client,
    )

    final_url = public.final_url or url

    if not public.url_alive:
        return WebResult(
            url_alive=False,
            status_code=public.status_code or 0,
            content_type=public.content_type or "",
            error=public.error,
            resolved_url=final_url,
            blocked=public.blocked,
        )

    if public.markdown is None:
        return WebResult(
            url_alive=True,
            status_code=public.status_code or 0,
            content_type=public.content_type or "",
            title=public.title,
            error=public.error,
            resolved_url=final_url,
        )

    references_dir.mkdir(parents=True, exist_ok=True)
    dest = references_dir / f"{key}_web.md"
    header = f"# {public.title or key}\n\nSource: {final_url}\n\n---\n\n"
    dest.write_text(header + public.markdown, encoding="utf-8")
    local_path = str(dest.relative_to(references_dir))

    return WebResult(
        url_alive=True,
        status_code=public.status_code or 0,
        content_type=public.content_type or "",
        local_path=local_path,
        title=public.title,
        resolved_url=final_url,
    )


async def _fetch_once(
    url: str,
    client: httpx.AsyncClient,
    user_agent: str,
    max_bytes: int = _MAX_HTML_BYTES,
) -> tuple[httpx.Response | None, _FetchFailure | None]:
    """Single GET attempt with size enforcement.

    Uses streaming with a gzip-safe size guard: skips Content-Length
    pre-check when Content-Encoding is present (compressed wire size
    != decompressed body size), and always enforces max_bytes on the
    decompressed data stream.

    On exception or refused download, returns a classified
    :class:`_FetchFailure` so the caller can distinguish genuinely
    unreachable URLs from unverifiable ones without re-parsing error
    strings.
    """
    try:
        async with client.stream(
            "GET", url, headers=_request_headers(user_agent)
        ) as resp:
            # Only trust Content-Length when no Content-Encoding —
            # otherwise the header reports compressed wire size,
            # not the decompressed size we'll actually read.
            if not resp.headers.get("content-encoding"):
                cl = resp.headers.get("content-length")
                if cl and int(cl) > max_bytes:
                    return None, _FetchFailure(
                        reason=f"oversized response ({cl} bytes)",
                        blocked=True,
                        hostname_level=False,
                    )

            # Byte-count guard on decompressed data — always enforced
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    return None, _FetchFailure(
                        reason=(
                            f"oversized response (exceeded {max_bytes} bytes "
                            f"during download)"
                        ),
                        blocked=True,
                        hostname_level=False,
                    )
                chunks.append(chunk)

        body = b"".join(chunks)
        # Strip Content-Encoding / Content-Length from the synthetic response.
        # aiter_bytes() already decompressed; leaving these intact makes
        # httpx try to decompress .content a second time and raise
        # DecodingError on servers that gzip real binary content.
        synthetic_headers = httpx.Headers(
            [
                (name, value)
                for name, value in resp.headers.raw
                if name.lower() not in (b"content-encoding", b"content-length")
            ]
        )
        full_resp = httpx.Response(
            status_code=resp.status_code,
            headers=synthetic_headers,
            content=body,
            request=resp.request,
        )
        return full_resp, None
    except httpx.HTTPError as e:
        return None, _classify_http_exception(e)


def _extract_title(html: str) -> str | None:
    """Extract <title> from HTML."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        title = match.group(1).strip()
        title = re.sub(r"\s+", " ", title)
        return title if title else None
    return None


def _extract_redirect_url(html: str, source_url: str) -> str | None:
    """Extract redirect target from HTML meta-refresh or JS patterns.

    Detects:
      - <meta http-equiv="refresh" content="0;url=...">
      - window.location = "..."
      - window.location.href = "..."
      - window.location.replace("...")
      - document.location = "..."
      - document.location.href = "..."

    Returns None if multiple conflicting redirect targets are found.
    """
    targets: set[str] = set()

    # Meta refresh
    match = re.search(
        r'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+content=["\']?\d+;\s*url=([^"\'>]+)',
        html,
        re.IGNORECASE,
    )
    if match:
        targets.add(_resolve_url(match.group(1).strip(), source_url))

    # JS location patterns
    for pattern in (
        r'(?:window|document)\.location(?:\.href)?\s*=\s*["\']([^"\']+)["\']',
        r'(?:window|document)\.location\.replace\(\s*["\']([^"\']+)["\']\s*\)',
    ):
        for m in re.finditer(pattern, html):
            targets.add(_resolve_url(m.group(1).strip(), source_url))

    # Filter out unsafe redirect targets
    safe_targets = {t for t in targets if _is_safe_redirect(t)}
    if safe_targets != targets:
        from loguru import logger

        blocked = targets - safe_targets
        logger.warning("Blocked unsafe redirect target(s): {}", blocked)

    if len(safe_targets) == 1:
        return safe_targets.pop()

    if len(safe_targets) > 1:
        from loguru import logger

        logger.warning(
            "Multiple conflicting redirect targets found, ignoring: {}",
            safe_targets,
        )

    return None


def _is_safe_redirect(url: str) -> bool:
    """Check that a redirect URL is safe to follow.

    Rejects non-HTTPS/HTTP schemes and any hostname that resolves to a
    non-public IP. Delegates IP validation to ``_network.is_hostname_safe``.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    return is_hostname_safe(host)


def _resolve_url(target: str, source_url: str) -> str:
    """Resolve a possibly-relative redirect URL against the source URL."""
    from urllib.parse import urljoin

    return urljoin(source_url, target)


def _html_to_markdown(html: str, url: str) -> str | None:
    """Convert HTML to clean markdown using trafilatura."""
    import trafilatura

    result = trafilatura.extract(html, url=url, include_links=True)
    return result or None
