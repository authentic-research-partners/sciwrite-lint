"""Web resource verification: URL liveness check + content download as markdown.

For citations that are blog posts, GitHub repos, or other web resources rather
than academic papers. These skip academic APIs and instead verify the URL is
alive and download content for LLM claim-checking.
"""

from __future__ import annotations

import re
from pydantic import BaseModel
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx

from sciwrite_lint._network import is_hostname_safe

# Defaults (overridable via LintConfig)
_DEFAULT_TIMEOUT = 15.0
_DEFAULT_USER_AGENT = "sciwrite-lint/0.1 (citation-verification)"
_MAX_HTML_BYTES = 10 * 1024 * 1024  # 10 MB — reject pages larger than this


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
    """Result of a web resource verification attempt."""

    url_alive: bool
    status_code: int = 0
    content_type: str = ""
    local_path: str | None = None  # path relative to references/
    title: str | None = None  # extracted page title
    error: str | None = None
    resolved_url: str | None = (
        None  # URL that actually responded (may differ from input after www alternation)
    )


async def check_url(
    url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> WebResult:
    """Check if a URL is alive via HEAD request, retry with GET on 405.

    Alternates between www and non-www variants (adds www. if missing,
    removes it if present) to handle sites where only one variant resolves.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )
    assert client is not None
    try:
        result = await _check_url_once(url, client, user_agent)

        # Retry once on transient connection errors
        if not result.url_alive and result.error and result.status_code == 0:
            import asyncio as _asyncio

            await _asyncio.sleep(1)
            result = await _check_url_once(url, client, user_agent)

        if result.url_alive:
            result.resolved_url = url
        else:
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


async def _check_url_once(
    url: str,
    client: httpx.AsyncClient,
    user_agent: str,
) -> WebResult:
    """Single-attempt URL liveness check (HEAD, retry with GET on 405)."""
    try:
        resp = await client.head(url, headers={"User-Agent": user_agent})
        if resp.status_code == 405:
            resp = await client.get(url, headers={"User-Agent": user_agent})
    except httpx.HTTPError as e:
        return WebResult(url_alive=False, error=str(e))

    alive = 200 <= resp.status_code < 400
    content_type = resp.headers.get("content-type", "")
    return WebResult(
        url_alive=alive,
        status_code=resp.status_code,
        content_type=content_type,
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

    Returns WebResult with local_path set if content was saved successfully.
    Alternates between www and non-www variants if the original URL fails.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )
    assert client is not None
    try:
        resp, fetch_error = await _fetch_once(url, client, user_agent)

        # Retry once on transient errors (decompression, connection reset)
        if fetch_error is not None and resp is None:
            import asyncio as _asyncio

            await _asyncio.sleep(1)
            resp, fetch_error = await _fetch_once(url, client, user_agent)

        if fetch_error is not None or (resp is not None and resp.status_code >= 400):
            alt = _www_variant(url)
            if alt:
                alt_resp, alt_error = await _fetch_once(alt, client, user_agent)
                if (
                    alt_error is None
                    and alt_resp is not None
                    and alt_resp.status_code < 400
                ):
                    resp, fetch_error = alt_resp, alt_error
                    url = alt  # use working URL for saved metadata

        if fetch_error is not None:
            return WebResult(url_alive=False, error=fetch_error, resolved_url=url)

        assert resp is not None  # mypy: if fetch_error is None, resp is set
        if resp.status_code >= 400:
            return WebResult(
                url_alive=False,
                status_code=resp.status_code,
                error=f"HTTP {resp.status_code}",
                resolved_url=url,
            )

        content_type = resp.headers.get("content-type", "")
        html = resp.text

        # Extract page title
        title = _extract_title(html)

        # Convert to markdown
        markdown = _html_to_markdown(html, url)
        if not markdown or len(markdown.strip()) < 100:
            # Check for JS/meta redirect before giving up
            redirect_url = _extract_redirect_url(html, url)
            if redirect_url and redirect_url != url:
                from loguru import logger

                logger.info("Following redirect from {} to {}", key, redirect_url)
                redirect_resp, redirect_err = await _fetch_once(
                    redirect_url, client, user_agent
                )
                if redirect_err is None and redirect_resp is not None:
                    html = redirect_resp.text
                    url = redirect_url
                    title = _extract_title(html) or title
                    markdown = _html_to_markdown(html, url)

        if not markdown or len(markdown.strip()) < 100:
            return WebResult(
                url_alive=True,
                status_code=resp.status_code,
                content_type=content_type,
                title=title,
                error="Content extraction failed"
                if not markdown
                else f"Extracted content too short ({len(markdown.strip())} chars)",
                resolved_url=url,
            )

        # Save to references/
        references_dir.mkdir(parents=True, exist_ok=True)
        dest = references_dir / f"{key}_web.md"
        header = f"# {title or key}\n\nSource: {url}\n\n---\n\n"
        dest.write_text(header + markdown, encoding="utf-8")
        local_path = str(dest.relative_to(references_dir))

        return WebResult(
            url_alive=True,
            status_code=resp.status_code,
            content_type=content_type,
            local_path=local_path,
            title=title,
            resolved_url=url,
        )
    finally:
        if own_client:
            await client.aclose()


async def _fetch_once(
    url: str,
    client: httpx.AsyncClient,
    user_agent: str,
    max_bytes: int = _MAX_HTML_BYTES,
) -> tuple[httpx.Response | None, str | None]:
    """Single GET attempt with size enforcement.

    Uses streaming with a gzip-safe size guard: skips Content-Length
    pre-check when Content-Encoding is present (compressed wire size
    != decompressed body size), and always enforces max_bytes on the
    decompressed data stream.
    """
    try:
        async with client.stream(
            "GET", url, headers={"User-Agent": user_agent}
        ) as resp:
            # Only trust Content-Length when no Content-Encoding —
            # otherwise the header reports compressed wire size,
            # not the decompressed size we'll actually read.
            if not resp.headers.get("content-encoding"):
                cl = resp.headers.get("content-length")
                if cl and int(cl) > max_bytes:
                    return None, f"Response too large ({cl} bytes)"

            # Byte-count guard on decompressed data — always enforced
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    return None, f"Response exceeded {max_bytes} bytes during download"
                chunks.append(chunk)

        body = b"".join(chunks)
        full_resp = httpx.Response(
            status_code=resp.status_code,
            headers=resp.headers,
            content=body,
            request=resp.request,
        )
        return full_resp, None
    except httpx.HTTPError as e:
        return None, str(e)


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
