"""MHTML (MIME HTML archive) → markdown conversion.

Browsers save single-file web archives with the ``.mhtml`` (Chromium, Edge,
Firefox) or ``.mht`` (legacy Windows/IE) extensions. Both formats are
multipart MIME containers whose first ``text/html`` part is the rendered
page — useful for JavaScript-heavy sites where a headless ``requests.get``
would only see the pre-hydration shell.

This module parses one of those archives, pulls out the root HTML part,
and runs it through trafilatura to produce markdown with the same header
shape that ``sciwrite_lint.web.fetch_web_content`` writes for ordinary
HTTP captures — so downstream classification (``_classify_local_file``)
returns ``"web_page"`` without any format-specific branching.

Errors are loud on purpose: if the file is not valid MHTML, or has no
HTML part, or trafilatura extracts nothing usable, the caller gets a
:class:`MHTMLParseError` with enough context to decide what to do.
Silent degradation is banned here (see ``.claude/rules/error-handling.md``).
"""

from __future__ import annotations

from email import message_from_bytes, policy
from email.message import EmailMessage
from pathlib import Path

from loguru import logger


class MHTMLParseError(RuntimeError):
    """Raised when an MHTML archive cannot be turned into markdown."""


def _decode_html_part(part: EmailMessage) -> str:
    """Return the decoded HTML body of a MIME part as a ``str``.

    ``EmailMessage.get_content()`` handles both the transfer encoding
    (``quoted-printable``, ``base64``, ``7bit``) and the charset
    declared in the ``Content-Type`` header. When the part is binary and
    lacks a charset, falls back to a best-effort ``utf-8`` decode on the
    raw payload — MHTML archives from real browsers always carry a
    charset, so this path is a defensive one for malformed inputs.
    """
    content = part.get_content()
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        return content.decode("utf-8", errors="replace")
    raise MHTMLParseError(
        f"Unexpected content type {type(content).__name__} in HTML part"
    )


def _find_root_html_part(msg: EmailMessage) -> EmailMessage:
    """Return the root ``text/html`` part of an MHTML archive.

    Browsers put the rendered page first, followed by resource parts
    (stylesheets, images, fonts) referenced via ``cid:`` URLs. We pick
    the first ``text/html`` part we see — that is the document the user
    was looking at when they hit Save.

    Raises :class:`MHTMLParseError` when no ``text/html`` part exists.
    """
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            return part  # type: ignore[return-value]
    raise MHTMLParseError("No text/html part found in MHTML archive")


def read_mhtml_source_url(path: Path) -> str:
    """Return the source URL declared in an ``.mhtml`` / ``.mht`` archive.

    Reads only the envelope and html-part headers (no trafilatura
    conversion), so this is cheap enough to call for every file in a
    drop folder when building a URL → file index. Returns an empty
    string when the file is missing, unparseable, or carries no
    ``Content-Location`` / ``Snapshot-Content-Location`` header.
    """
    try:
        raw = path.read_bytes()
        msg = message_from_bytes(raw, policy=policy.default)
    except OSError:
        return ""
    except Exception:
        return ""
    if not isinstance(msg, EmailMessage):
        return ""
    for header in ("Snapshot-Content-Location", "Content-Location"):
        value = msg.get(header)
        if value:
            return str(value).strip()
    try:
        html_part = _find_root_html_part(msg)
    except MHTMLParseError:
        return ""
    value = html_part.get("Content-Location")
    return str(value).strip() if value else ""


def _extract_source_url(msg: EmailMessage, html_part: EmailMessage) -> str:
    """Return the best available source URL for an MHTML archive.

    Resolution order:
      1. Envelope ``Snapshot-Content-Location`` (Chromium/Edge convention).
      2. Envelope ``Content-Location``.
      3. HTML part's own ``Content-Location``.
      4. Empty string (the conversion still proceeds, header omits the URL).
    """
    for header in ("Snapshot-Content-Location", "Content-Location"):
        value = msg.get(header)
        if value:
            return str(value).strip()
    value = html_part.get("Content-Location")
    if value:
        return str(value).strip()
    return ""


def _extract_title(html: str) -> str:
    """Extract ``<title>`` text from an HTML document, or return ``""``."""
    import re

    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title


def mhtml_to_markdown(path: Path) -> tuple[str, str]:
    """Convert an MHTML archive to a markdown document.

    Parses the archive, extracts the first ``text/html`` part, runs it
    through trafilatura with the same options used by
    :func:`sciwrite_lint.web._html_to_markdown`, and prepends a header
    matching the shape written by
    :func:`sciwrite_lint.web.fetch_web_content`::

        # {title}

        Source: {url}

        ---

        {markdown body}

    Args:
        path: Path to a ``.mhtml`` or ``.mht`` file.

    Returns:
        ``(markdown, source_url)`` — ``source_url`` is empty when the
        archive carries no ``Content-Location`` headers.

    Raises:
        MHTMLParseError: if the file cannot be parsed as MIME, has no
            ``text/html`` part, or produces no usable markdown after
            trafilatura extraction.
    """
    import trafilatura

    try:
        raw = path.read_bytes()
    except OSError as e:
        raise MHTMLParseError(f"Cannot read MHTML file {path}: {e}") from e

    try:
        msg = message_from_bytes(raw, policy=policy.default)
    except Exception as e:
        raise MHTMLParseError(f"Failed to parse MHTML envelope from {path}: {e}") from e

    if not isinstance(msg, EmailMessage):
        raise MHTMLParseError(
            f"MHTML envelope from {path} did not parse as EmailMessage "
            f"(got {type(msg).__name__})"
        )

    html_part = _find_root_html_part(msg)
    html = _decode_html_part(html_part)
    source_url = _extract_source_url(msg, html_part)
    title = _extract_title(html) or path.stem

    body = trafilatura.extract(html, url=source_url or None, include_links=True)
    if not body:
        raise MHTMLParseError(
            f"trafilatura produced no markdown from {path} — the HTML part "
            f"may be empty, non-article, or below trafilatura's extraction threshold"
        )

    source_line = f"Source: {source_url}" if source_url else "Source: (unknown)"
    header = f"# {title}\n\n{source_line}\n\n---\n\n"
    logger.info(
        "Converted MHTML: {} → markdown ({} chars, source={})",
        path.name,
        len(body),
        source_url or "(unknown)",
    )
    return header + body, source_url
