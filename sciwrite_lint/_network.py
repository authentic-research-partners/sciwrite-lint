"""Shared network security utilities.

Provides:
- SSRF-safe httpx transport that validates resolved IPs via ``ipaddress.is_global``
- Identifier format validators (DOI, arXiv, PMCID, ISBN)
"""

from __future__ import annotations

import ipaddress
import re
import socket

import httpx
from loguru import logger

# ---------------------------------------------------------------------------
# Identifier validators
# ---------------------------------------------------------------------------

DOI_RE = re.compile(r"^10\.\d{4,}/\S+$")
ARXIV_RE = re.compile(r"^(\d{4}\.\d{4,})(v\d+)?$")
PMCID_RE = re.compile(r"^PMC\d+$")
PMID_RE = re.compile(r"^\d{1,9}$")
ISBN_RE = re.compile(r"^[\d\-]{10,17}$")
LCCN_RE = re.compile(r"^[\da-z\-]{1,12}$", re.IGNORECASE)


def clean_and_validate_doi(doi: str) -> str | None:
    """Strip URL prefixes and validate DOI format. Returns cleaned DOI or None."""
    clean = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/").strip()
    if not DOI_RE.match(clean):
        logger.debug("Invalid DOI format, skipping: {}", clean[:80])
        return None
    return clean


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """Check if string matches arXiv ID format (YYMM.NNNNN)."""
    return bool(ARXIV_RE.match(arxiv_id.strip()))


def is_valid_pmid(pmid: str) -> bool:
    """Check if string looks like a PubMed ID (1-9 digits)."""
    return bool(PMID_RE.match(pmid.strip()))


def is_valid_pmcid(pmcid: str) -> bool:
    """Check if string matches PMC ID format (PMC followed by digits)."""
    return bool(PMCID_RE.match(pmcid.strip()))


def is_valid_isbn(isbn: str) -> bool:
    """Check if string looks like an ISBN (digits and hyphens, 10-17 chars)."""
    return bool(ISBN_RE.match(isbn.strip()))


def is_valid_lccn(lccn: str) -> bool:
    """Check if string looks like a Library of Congress Control Number."""
    return bool(LCCN_RE.match(lccn.strip()))


# ---------------------------------------------------------------------------
# SSRF-safe IP validation
# ---------------------------------------------------------------------------


def is_ip_safe(ip_str: str) -> bool:
    """Return True if an IP address is globally routable (not private/reserved)."""
    try:
        return ipaddress.ip_address(ip_str).is_global
    except ValueError:
        return False


def is_hostname_safe(hostname: str) -> bool:
    """Resolve hostname and check that ALL returned IPs are globally routable.

    Returns False if the hostname is unresolvable or any IP is non-global.
    """
    # Literal IP — check directly
    try:
        addr = ipaddress.ip_address(hostname.strip("[]"))
        return addr.is_global
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    if not infos:
        return False

    return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)


# ---------------------------------------------------------------------------
# SSRF-safe httpx transport
# ---------------------------------------------------------------------------


class SSRFSafeTransport(httpx.AsyncHTTPTransport):
    """httpx transport wrapper that blocks requests to non-global IPs.

    Validates the resolved IP address of the target hostname before
    connecting. This protects against SSRF via HTTP 3xx redirects —
    httpx resolves the redirect target through this transport, so
    redirects to private/reserved IPs are blocked.

    Usage::

        transport = SSRFSafeTransport()
        async with httpx.AsyncClient(transport=transport, ...) as client:
            resp = await client.get(url)
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        hostname = request.url.host or ""
        if not hostname:
            raise httpx.RequestError("Empty hostname", request=request)

        if not is_hostname_safe(hostname):
            raise httpx.RequestError(
                f"SSRF blocked: {hostname} resolves to non-public IP",
                request=request,
            )

        return await super().handle_async_request(request)


class ResponseTooLarge(Exception):
    """Raised when a streaming download exceeds the byte limit."""


async def stream_with_limit(
    client: httpx.AsyncClient,
    url: str,
    max_bytes: int,
    **kwargs: object,
) -> httpx.Response:
    """Stream a GET request with gzip-safe size enforcement.

    Skips the Content-Length pre-check when Content-Encoding is present
    because the header reports compressed wire size, not the decompressed
    size we actually read.  Always enforces *max_bytes* on the
    decompressed data stream via ``aiter_bytes()``.

    Returns a synthetic ``httpx.Response`` with the full decompressed body
    populated.  Raises ``ResponseTooLarge`` if the limit is exceeded.

    Content-Encoding (and the now-stale Content-Length) is stripped from the
    synthetic response's headers. ``aiter_bytes`` already decompressed the
    body; leaving the header intact would cause httpx to try to decompress a
    second time when the caller reads ``.content``, which fails with
    ``DecodingError: incorrect header check`` on servers that gzip real
    binary content (observed on ``mpra.ub.uni-muenchen.de`` serving PDFs).
    """
    async with client.stream("GET", url, **kwargs) as resp:  # type: ignore[arg-type]
        # Non-success responses are small — read eagerly for diagnostics
        if resp.status_code >= 400:
            await resp.aread()
            return resp

        # Only trust Content-Length when no Content-Encoding
        if not resp.headers.get("content-encoding"):
            cl = resp.headers.get("content-length")
            if cl and int(cl) > max_bytes:
                raise ResponseTooLarge(
                    f"Response too large ({cl} bytes, limit {max_bytes})"
                )

        # Byte-count guard on decompressed data — always enforced
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ResponseTooLarge(
                    f"Response exceeded {max_bytes} bytes during download"
                )
            chunks.append(chunk)

    body = b"".join(chunks)
    synthetic_headers = httpx.Headers(
        [
            (name, value)
            for name, value in resp.headers.raw
            if name.lower() not in (b"content-encoding", b"content-length")
        ]
    )
    return httpx.Response(
        status_code=resp.status_code,
        headers=synthetic_headers,
        content=body,
        request=resp.request,
    )


def ssrf_safe_client(**kwargs: object) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with SSRF protection.

    All keyword arguments are forwarded to ``httpx.AsyncClient``.
    ``follow_redirects=True`` is set by default.
    """
    kwargs.setdefault("follow_redirects", True)  # type: ignore[arg-type]
    kwargs.setdefault("transport", SSRFSafeTransport())  # type: ignore[arg-type]
    return httpx.AsyncClient(**kwargs)  # type: ignore[arg-type]
