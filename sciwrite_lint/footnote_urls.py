r"""Footnote-URL source discovery and synthesis.

A paper can reference external material in two ways:

1. A formal citation, ``\cite{key}``, pointing at a ``.bib`` entry.
2. An inline footnote URL, ``\footnote{\url{https://example.com/page}}``,
   used for informational web pages that do not belong in the formal
   bibliography (about pages, program descriptions, FAQs).

For (1) sciwrite-lint has always linked the citation to an archived file
via the ``.bib`` citekey. For (2) there is no citekey — only a URL. This
module provides the missing link: it extracts every ``\footnote{\url{…}}``
from the manuscript, resolves each URL to an archived ``.md`` capture in
``local_web_dir`` using a simple ``Source:`` header convention, and
synthesizes ``Citation`` objects that flow through the same verify/fetch/
parse/claim-verify pipeline as cited references. No new paths, no code
duplication.

URL-declaration convention for archived captures
-------------------------------------------------
For ``.md`` files in ``local_web_dir`` (or ``Sources/full_text_web/``
when auto-detected), within the first 20 lines, a line of the form::

    Source: https://example.com/page

*or*::

    Source URL: https://example.com/page

declares the canonical URL the file documents. The line is
case-insensitive, tolerates wrapping in an HTML comment (``<!-- -->``),
and ignores optional leading whitespace. Hand-written captures add
this header manually.

For ``.mhtml`` / ``.mht`` files (browser "Save As → Webpage, Single
File" archives), the URL is recovered from the MIME envelope's
``Snapshot-Content-Location`` or ``Content-Location`` header — no user
action needed. When a match hits, :func:`copy_web_source` converts the
archive to markdown (via :func:`sciwrite_lint.mhtml.mhtml_to_markdown`)
at ingest time, so downstream readers only ever see markdown.

URL matching is normalised: lowercase scheme + host, strip trailing
slash from the path, strip ``#fragment``, strip well-known tracking
query parameters (``utm_*``, ``fbclid``, ``gclid``). The normalised
URL from the header is matched against the normalised URL inside
each ``\url{...}`` of every ``\footnote{...}``. A single footnote may
contain more than one ``\url{}``; each is matched independently.

Synthetic citation shape
------------------------
Each matched footnote URL becomes a ``Citation`` with:

* ``key`` — deterministic ``fn_<sha256(normalized_url)[:10]>``. The
  prefix marks it as synthetic so downstream code (or humans reading
  logs) can tell at a glance that the entry was not in the ``.bib``.
* ``bib_format`` — ``"footnote"`` (new value). Used by tests and
  potential future guards.
* ``entry_type`` — ``"misc"``.
* ``url`` — the normalized URL.
* ``title`` — recovered from the archived file's H1 heading or, failing
  that, its stem.
* ``local_path`` / ``local_status`` — set to the copied workspace file,
  so the fetch stage's existing T1-with-local-file short-circuit applies.

A ``CitationMetadata`` record is written to ``workspace.db`` at ingest
time with ``api_match="manual"`` and ``api_source="footnote"``, which
makes :func:`_stage_verify` skip the API waterfall (there is nothing to
verify — the evidence is already local). The record carries
``access.local_file_src_hash`` so re-runs skip re-copying unchanged
captures; overwriting a capture under the same filename with different
bytes changes the hash and triggers a fresh copy on the next run.

Claim wire-up
-------------
:func:`sciwrite_lint.eval_claims.extract_claim_contexts` additionally
walks every ``\footnote{...}`` that contains at least one matched URL
and emits a ``ClaimContext`` with the synthetic key, so the existing
claim-verification machinery sees the footnote-backed sentence as a
first-class claim with a known source.

Scope
-----
Handles LaTeX (``\footnote{\url{}}`` via :func:`extract_footnote_urls`)
and markdown (``[^id]`` / ``^[…]`` footnotes carrying a ``<url>`` /
``[text](url)`` link, via :func:`extract_footnote_urls_markdown`);
:func:`ingest_footnote_sources` dispatches on the file suffix and shares
the matching + synthesis. PDF input (GROBID) is out of scope — the TEI
``<note place="foot">`` path works differently.

The ``Claim wire-up`` above (footnote host sentence → ``ClaimContext``)
is implemented for both: LaTeX via :func:`extract_claim_contexts`,
markdown via :func:`extract_footnote_claims_markdown` (wired into the
claims stage in ``eval_claims``).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode

import rfc3986
import rfc3986.exceptions
from loguru import logger

from sciwrite_lint.models import Citation

# ---------------------------------------------------------------------------
# Public constants (also used by callers / tests)
# ---------------------------------------------------------------------------

#: Prefix marking a synthetic footnote-URL citation key.
FOOTNOTE_KEY_PREFIX = "fn_"

#: Value written to ``Citation.bib_format`` for synthetic footnote citations.
BIB_FORMAT_FOOTNOTE = "footnote"

#: How many leading lines of a ``.md`` file are scanned for the Source header.
MAX_HEADER_LINES = 20

#: Workspace-relative filename pattern for an ingested footnote source.
FOOTNOTE_WORKSPACE_SUFFIX = "_local_web.md"

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Tracking params stripped during URL normalization. Keeping the list
# small and conservative — anything not here is preserved, so semantic
# query params (?id=123, ?article=456) are not dropped.
_TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
    }
)

# Matches a `Source:` or `Source URL:` header line, case-insensitive,
# tolerant of a leading HTML-comment opener and leading whitespace.
# The URL is captured in group 1 and runs to end-of-line (stripped by
# the caller).
_SOURCE_HEADER_RE = re.compile(
    r"""
    ^\s*                            # leading whitespace
    (?:<!--\s*)?                    # optional HTML-comment opener
    Source(?:\s*URL)?               # "Source" or "Source URL"
    \s*:\s*                         # colon, optional spaces
    (https?://\S+)                  # the URL (group 1)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches the outer `\footnote{...}` block. Greedy inside, but
# constrained by a matching-brace counter (see :func:`_iter_footnotes`)
# rather than a simple regex, so nested ``{...}`` inside the body don't
# break extraction.
_FOOTNOTE_OPENING_RE = re.compile(r"\\footnote\{")

# Matches `\url{URL}` anywhere in text.
_URL_COMMAND_RE = re.compile(r"\\url\{([^}]+)\}")

# Matches an H1 Markdown heading at the start of a line.
_MD_TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)

# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Return a canonical form of *url* for equality comparison.

    Uses :func:`rfc3986.uri_reference` + ``.normalize()`` for the
    RFC-correct part of canonicalization (scheme and host lowercased,
    percent-encoding normalized, path dot-segments like ``/a/../b``
    resolved to ``/b``), then applies project-specific canonicalization
    on top:

    * strip default ports (``:80`` on http, ``:443`` on https),
    * strip a trailing slash on non-empty paths (root-only ``/`` is
      preserved — some servers treat ``https://host`` and
      ``https://host/`` as different),
    * drop the URL fragment (``#...``),
    * drop tracking query parameters in :data:`_TRACKING_PARAMS`,
    * preserve the order of non-tracking query parameters (this
      deliberately differs from ``w3lib.canonicalize_url``, which
      sorts — order can matter for some endpoints).

    Returns the original input (stripped) when ``rfc3986`` can't parse
    it — a malformed URL still needs to compare equal to itself.
    """
    raw = url.strip()
    if not raw:
        return ""
    try:
        ref = rfc3986.uri_reference(raw).normalize()
    except rfc3986.exceptions.RFC3986Exception as e:
        logger.debug(
            "URL normalize failed ({}: {}); using raw {}", type(e).__name__, e, raw
        )
        return raw

    scheme = ref.scheme or ""
    # rfc3986 returns port as a string (RFC spec treats it as textual).
    if ref.port and (
        (scheme == "http" and ref.port == "80")
        or (scheme == "https" and ref.port == "443")
    ):
        authority = ref.host or ""
        if ref.userinfo:
            authority = f"{ref.userinfo}@{authority}"
        ref = ref.copy_with(authority=authority)

    path = ref.path or ""
    if path.endswith("/") and len(path) > 1:
        ref = ref.copy_with(path=path[:-1])

    ref = ref.copy_with(fragment=None)

    if ref.query:
        pairs = [
            (k, v)
            for k, v in parse_qsl(ref.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS
        ]
        ref = ref.copy_with(query=urlencode(pairs) if pairs else None)

    return ref.unsplit()


# ---------------------------------------------------------------------------
# Source-header reading
# ---------------------------------------------------------------------------


def read_source_url(md_path: Path, max_lines: int = MAX_HEADER_LINES) -> str | None:
    """Return the ``Source:`` URL declared in the first *max_lines* of *md_path*.

    Returns ``None`` when no header is present, when the file cannot be
    read, or when the captured URL fails :func:`normalize_url`. The
    returned URL is already normalized.
    """
    try:
        with md_path.open("r", encoding="utf-8", errors="replace") as fh:
            lines: list[str] = []
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                lines.append(line)
    except OSError as exc:
        logger.debug(f"Footnote ingest: cannot read {md_path}: {exc}")
        return None

    for line in lines:
        match = _SOURCE_HEADER_RE.match(line)
        if match:
            url = match.group(1).rstrip(">").rstrip(".,;)")
            norm = normalize_url(url)
            return norm or None
    return None


#: File suffixes scanned when building the URL → path index. ``.md``
#: declares its URL in a ``Source:`` / ``Source URL:`` header; ``.mhtml``
#: and ``.mht`` declare it in the MIME envelope (``Snapshot-Content-
#: Location`` / ``Content-Location``), read via :func:`read_mhtml_source_url`.
SCANNED_SUFFIXES: frozenset[str] = frozenset({".md", ".mhtml", ".mht"})


def build_web_url_map(web_dir: Path) -> dict[str, Path]:
    """Scan *web_dir* for captures and index them by declared Source URL.

    Accepts ``.md`` (``Source:`` header), ``.mhtml``, and ``.mht`` (URL
    recovered from the MIME envelope). Returns a dict mapping normalized
    URL → path; :func:`ingest_footnote_sources` hands the path to
    :func:`sciwrite_lint.local_sources.copy_web_source`, which converts
    ``.mhtml`` to markdown at copy time.

    If two files declare the same URL, the first-scanned wins and a
    debug message is emitted. Scan order is alphabetical for
    determinism; MHTML and markdown share the ordering.
    """
    if not web_dir.is_dir():
        return {}

    url_map: dict[str, Path] = {}
    scanned = 0
    skipped = 0

    for entry in sorted(web_dir.iterdir()):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in SCANNED_SUFFIXES:
            continue
        scanned += 1
        if suffix == ".md":
            url = read_source_url(entry)
        else:
            from sciwrite_lint.mhtml import read_mhtml_source_url

            raw_url = read_mhtml_source_url(entry)
            url = normalize_url(raw_url) if raw_url else None
        if not url:
            skipped += 1
            continue
        if url in url_map:
            logger.debug(
                f"Footnote ingest: duplicate Source URL {url} — "
                f"keeping {url_map[url].name}, ignoring {entry.name}"
            )
            continue
        url_map[url] = entry

    if scanned and not url_map:
        logger.debug(
            f"Footnote ingest: scanned {scanned} capture file(s) in {web_dir}, "
            f"none carry a recoverable source URL"
        )
    elif skipped:
        logger.debug(
            f"Footnote ingest: {len(url_map)} of {scanned} capture file(s) in "
            f"{web_dir} carry a source URL ({skipped} without)"
        )

    return url_map


# ---------------------------------------------------------------------------
# Footnote / URL extraction from .tex
# ---------------------------------------------------------------------------


def _iter_footnotes(tex: str) -> list[tuple[int, int, str]]:
    r"""Yield ``(start, end, body)`` for every ``\footnote{...}`` in *tex*.

    Uses a brace-matching scan rather than a fixed regex so nested
    ``{...}`` inside the footnote body (common) don't truncate the
    extracted text. Malformed footnotes (unbalanced braces) are
    skipped with a debug log.
    """
    results: list[tuple[int, int, str]] = []
    for opening in _FOOTNOTE_OPENING_RE.finditer(tex):
        brace_start = opening.end() - 1  # points at the `{`
        depth = 0
        i = brace_start
        while i < len(tex):
            ch = tex[i]
            if ch == "\\" and i + 1 < len(tex):
                # skip escaped char (\{, \}, etc.) so it doesn't shift depth
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    body = tex[brace_start + 1 : i]
                    results.append((opening.start(), i + 1, body))
                    break
            i += 1
        else:
            logger.debug(
                f"Footnote ingest: unbalanced braces in footnote "
                f"starting at offset {opening.start()}; skipping"
            )
    return results


def extract_footnote_urls(tex_path: Path) -> list[tuple[int, str]]:
    r"""Return ``(line_number, normalized_url)`` for every URL inside a footnote.

    One footnote can carry multiple ``\url{...}`` commands — each is
    reported independently, so a single ``\footnote{See \url{a} and
    \url{b}}`` yields two entries. Duplicate URLs are **not** deduped
    here; the caller handles that.
    """
    try:
        tex = tex_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug(f"Footnote extract: cannot read {tex_path}: {exc}")
        return []

    results: list[tuple[int, str]] = []
    for start, _end, body in _iter_footnotes(tex):
        line_no = tex.count("\n", 0, start) + 1
        for url_match in _URL_COMMAND_RE.finditer(body):
            raw_url = url_match.group(1).strip()
            norm = normalize_url(raw_url)
            if norm:
                results.append((line_no, norm))
    return results


def _note_link_urls(node: object) -> list[str]:
    r"""Collect ``Link`` targets that appear inside ``Note`` nodes under *node*.

    The markdown analogue of "URLs inside a footnote" — pandoc renders both
    ``[^id]`` reference footnotes and ``^[inline]`` footnotes as ``Note``
    nodes, and ``<url>`` autolinks / ``[text](url)`` links inside them become
    ``Link`` nodes (the markdown counterpart of LaTeX ``\url{}``).
    """
    urls: list[str] = []

    def _links(n: object) -> None:
        if isinstance(n, dict):
            if n.get("t") == "Link":
                c = n.get("c")
                if (
                    isinstance(c, list)
                    and len(c) > 2
                    and isinstance(c[2], list)
                    and c[2]
                ):
                    urls.append(c[2][0])
            for v in n.values():
                _links(v)
        elif isinstance(n, list):
            for v in n:
                _links(v)

    def _notes(n: object) -> None:
        if isinstance(n, dict):
            if n.get("t") == "Note":
                _links(n.get("c"))
            else:
                for v in n.values():
                    _notes(v)
        elif isinstance(n, list):
            for v in n:
                _notes(v)

    _notes(node)
    return urls


def extract_footnote_urls_markdown(md_path: Path) -> list[tuple[int, str]]:
    r"""Return ``(line, normalized_url)`` for every URL inside a markdown footnote.

    The markdown analogue of :func:`extract_footnote_urls`. ``line`` is ``0``
    (markdown footnote line numbers aren't tracked; the field is used only for
    the caller's log dedup). Returns the same shape as the LaTeX extractor so
    ``ingest_footnote_sources`` treats both sources uniformly.
    """
    import json

    import pypandoc

    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug(f"Footnote extract: cannot read {md_path}: {exc}")
        return []

    ast = json.loads(pypandoc.convert_text(text, to="json", format="markdown"))
    results: list[tuple[int, str]] = []
    for raw_url in _note_link_urls(ast):
        norm = normalize_url(raw_url)
        if norm:
            results.append((0, norm))
    return results


def extract_footnote_claims_markdown(md_text: str) -> list[tuple[str, str]]:
    r"""Return ``(synthetic_key, host_context)`` for each markdown footnote URL.

    The markdown analogue of the LaTeX footnote-claim wire-up: a footnote
    that carries a URL backs a claim made in the sentence that *bears* the
    footnote marker, not in the footnote body. For each block holding a
    ``Note`` with a URL, the host context is the block's prose with the
    footnote body excluded (``_inline_text`` renders ``Note`` as empty), and
    the key is :func:`synthesize_footnote_key` of the normalized URL — the
    same key :func:`ingest_footnote_sources` assigns the synthesized
    citation, so the claim binds to its source.
    """
    import json

    import pypandoc

    from sciwrite_lint.markdown_cites import _inline_text

    ast = json.loads(pypandoc.convert_text(md_text, to="json", format="markdown"))
    results: list[tuple[str, str]] = []

    def _walk(blocks: object) -> None:
        if not isinstance(blocks, list):
            return
        for block in blocks:
            if not isinstance(block, dict):
                continue
            t = block.get("t")
            c = block.get("c")
            if t in ("Para", "Plain"):
                urls = _note_link_urls(c)
                if not urls:
                    continue
                host = _inline_text(c).strip()
                for raw_url in urls:
                    norm = normalize_url(raw_url)
                    if norm and host:
                        results.append((synthesize_footnote_key(norm), host))
            elif t == "BlockQuote":
                _walk(c)
            elif t == "Div" and isinstance(c, list) and len(c) > 1:
                _walk(c[1])
            elif t == "BulletList" and isinstance(c, list):
                for item in c:
                    _walk(item)
            elif t == "OrderedList" and isinstance(c, list) and len(c) > 1:
                for item in c[1]:
                    _walk(item)

    _walk(ast.get("blocks", []))
    return results


# ---------------------------------------------------------------------------
# Key synthesis + title recovery
# ---------------------------------------------------------------------------


def synthesize_footnote_key(url: str) -> str:
    """Return a deterministic, collision-resistant citation key for *url*.

    The key format is ``fn_<sha256(normalized_url)[:10]>``. The hash is
    taken over the normalized URL (so trivially-equivalent URLs collide
    on the same key), truncated to 10 hex characters — 40 bits, which
    keeps the key short while giving ~1-in-10¹² collision odds for the
    reference counts any single paper could plausibly carry.

    The output is stable across runs, so caching / re-ingest keyed on
    this string works reliably.
    """
    norm = normalize_url(url)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"{FOOTNOTE_KEY_PREFIX}{digest[:10]}"


def _title_from_md(md_path: Path) -> str:
    """Extract the first H1 heading from *md_path*, or fall back to the stem."""
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return md_path.stem
    match = _MD_TITLE_RE.search(text)
    if match:
        return match.group(1).strip()
    return md_path.stem


# ---------------------------------------------------------------------------
# Ingest: match URLs, copy files, synthesize citations, register metadata
# ---------------------------------------------------------------------------


def ingest_footnote_sources(
    tex_path: Path,
    web_dir: Path,
    references_dir: Path,
    source_paper: str,
) -> list[Citation]:
    r"""Extract footnote URLs, match each to an archived ``.md``, and
    synthesize :class:`Citation` objects ready for the claim pipeline.

    Handles LaTeX ``\footnote{\url{URL}}`` and markdown footnotes
    (``[^id]`` / ``^[…]`` with a ``<url>`` or ``[text](url)`` link) — the
    source type is taken from ``tex_path``'s suffix. For every such URL
    whose normalized form is declared in a ``local_web_dir`` capture's
    ``Source:`` header, this function:

    1. synthesizes a stable key via :func:`synthesize_footnote_key`,
    2. copies the capture into the paper workspace as
       ``{key}_local_web.md`` (skipped when the source file's SHA-256
       matches the hash already recorded in ``workspace.db``),
    3. writes a :class:`CitationMetadata` record with ``api_match =
       "manual"``, ``tier = "T1"``, and ``access.source = "footnote"``
       so :func:`_stage_verify` and :func:`_stage_fetch` short-circuit,
    4. returns a :class:`Citation` with ``bib_format="footnote"`` and
       ``local_status="md"``.

    Footnote URLs without a matching archived capture are logged at
    INFO (so the user can see what's missing) and skipped; they do
    not produce ``Citation`` objects.

    Returns an empty list — without side effects — when *web_dir* does
    not exist, contains no captures, or the tex has no footnote URLs.
    """
    from sciwrite_lint.local_sources import copy_web_source, file_sha256
    from sciwrite_lint.models import CitationMetadata
    from sciwrite_lint.references.metadata import save_metadata
    from sciwrite_lint.references.workspace_db import (
        get_db,
        query_verified_metadata,
    )

    if not web_dir.is_dir():
        return []

    url_map = build_web_url_map(web_dir)
    if not url_map:
        return []

    footnote_hits = (
        extract_footnote_urls_markdown(tex_path)
        if tex_path.suffix.lower() == ".md"
        else extract_footnote_urls(tex_path)
    )
    if not footnote_hits:
        return []

    # Dedup by normalized URL (keep the earliest line number for logging)
    first_line: dict[str, int] = {}
    for line_no, norm_url in footnote_hits:
        first_line.setdefault(norm_url, line_no)

    with get_db(references_dir) as conn:
        existing_meta = query_verified_metadata(conn)

    citations: list[Citation] = []
    unmatched: list[str] = []

    for norm_url, _line_no in first_line.items():
        src_path = url_map.get(norm_url)
        if src_path is None:
            unmatched.append(norm_url)
            continue

        key = synthesize_footnote_key(norm_url)
        src_hash = file_sha256(src_path)
        workspace_file = f"{key}{FOOTNOTE_WORKSPACE_SUFFIX}"
        workspace_path = references_dir / workspace_file

        existing = existing_meta.get(key)
        unchanged = (
            existing is not None
            and workspace_path.exists()
            and existing.access.get("local_file_src_hash") == src_hash
        )
        if not unchanged:
            copy_web_source(src_path, key, references_dir)

        title = _title_from_md(workspace_path) if workspace_path.exists() else ""

        citations.append(
            Citation(
                key=key,
                raw_text=f"(footnote URL: {norm_url})",
                title=title,
                url=norm_url,
                entry_type="misc",
                source_paper=source_paper,
                bib_format=BIB_FORMAT_FOOTNOTE,
                local_status="md",
                local_path=workspace_file,
                api_match="manual",
                tier="T1",
            )
        )

        meta = CitationMetadata(key=key)
        meta.bibitem = {
            "title": title,
            "url": norm_url,
            "entry_type": "misc",
            "bib_format": BIB_FORMAT_FOOTNOTE,
        }
        meta.canonical = {
            "url": norm_url,
            "title": title,
        }
        access: dict[str, Any] = {
            "tier": "T1",
            "local_file": workspace_file,
            "local_file_src_hash": src_hash,
            "source": "footnote",
            "is_formal": False,
        }
        meta.access = access
        meta.api_match = "manual"
        meta.api_source = "footnote"
        save_metadata(meta, references_dir)

    if citations:
        logger.info(
            f"Footnote URLs: matched {len(citations)} to archived sources "
            f"in {web_dir.name}"
        )
    if unmatched:
        logger.info(
            f"Footnote URLs: {len(unmatched)} without archived source "
            f"(drop a capture with a `Source: <url>` header into {web_dir}) — "
            f"claims backed only by these URLs will be unverified"
        )
        for url in unmatched[:5]:
            logger.debug(f"  no match: {url}")

    return citations


__all__ = [
    "BIB_FORMAT_FOOTNOTE",
    "FOOTNOTE_KEY_PREFIX",
    "FOOTNOTE_WORKSPACE_SUFFIX",
    "MAX_HEADER_LINES",
    "build_web_url_map",
    "extract_footnote_urls",
    "ingest_footnote_sources",
    "normalize_url",
    "read_source_url",
    "synthesize_footnote_key",
]
