"""Local reference sources: user-provided PDFs, markdown, and MHTML archives.

Users drop reference material into one of two directories so the tool can
skip the OA waterfall for items the user already has:

- **Academic** (``local_academic_dir``, auto-detects ``Sources/full_text/``):
  accepts ``.pdf`` (primary) and ``.md`` summaries. Files ingested from
  this directory carry ``local_type`` ``"pdf"`` or ``"summary"``.
- **Web** (``local_web_dir``, auto-detects ``Sources/full_text_web/``):
  accepts ``.md`` (hand-written or from a prior capture) and ``.mhtml`` /
  ``.mht`` archives saved from a browser for JavaScript-heavy pages.
  Everything ingested from this directory lands in the workspace as
  ``{key}_local_web.md`` and is classified ``"web_page"``.

Filenames are fuzzy-matched against reference titles from the ``.bib``
in two passes: citekey-prefix (authoritative — the user named the file
after its citekey) then fuzzy title similarity. Matched files are copied
into the paper workspace and treated identically to downloaded ones —
GROBID parsing for PDFs, direct read for markdown, MHTML converted to
markdown at copy time via :mod:`sciwrite_lint.mhtml`.

Separation of academic vs. web sources is intentional: the directory
itself is the trust-level signal, not filename or header sniffing. A
claim supported by a peer-reviewed PDF is treated as stronger evidence
than one supported by a blog capture, so we do not want those streams
to commingle at ingest time.

Re-ingest is **hash-aware**: the SHA-256 of each matched source file is
recorded on ``meta.access["local_file_src_hash"]`` at ingest time.
:func:`ingest_local_sources` skips keys whose source hash matches the
stored value, so repeated ``sciwrite-lint check`` runs don't re-copy
unchanged files — but overwriting a drop-folder file with different
bytes (a corrected PDF, a re-captured MHTML, a refreshed markdown
summary) changes the hash and triggers a fresh copy / MHTML re-parse /
downstream GROBID + embedding recompute on the next run. Filename
alone is not sufficient: the user can overwrite a workspace file while
keeping the name, and we detect it.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Callable, Literal

from loguru import logger
from pydantic import BaseModel

from sciwrite_lint.pdf.pdf_download import _title_similarity


# Accepted filename suffixes per directory kind. Files outside the set
# are ignored (logged at DEBUG) — e.g. a stray ``.mhtml`` in the
# academic directory will not be miscategorised as a PDF.
ACADEMIC_SUFFIXES: frozenset[str] = frozenset({".pdf", ".md"})
WEB_SUFFIXES: frozenset[str] = frozenset({".md", ".mhtml", ".mht"})

# Threshold for filename ↔ title fuzzy match. Higher than the download-
# time title gate (0.65) because filenames are user-controlled and
# should be close. Citekey-prefix matches bypass this threshold.
_MATCH_THRESHOLD = 0.80


def _normalize_filename(filename: str) -> str:
    """Normalize a filename to a comparable string.

    Strips extension, replaces separators with spaces, lowercases,
    removes non-alphanumeric (except spaces).
    """
    name = Path(filename).stem
    name = re.sub(r"[_\-\.]+", " ", name)
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def _leading_token(filename: str) -> str:
    """Return the first alphanumeric token of a filename stem.

    Used for citekey-prefix lookup: ``dewey1910_How_We_Think.pdf`` →
    ``"dewey1910"``. Returns ``""`` when the stem has no alphanumeric
    characters.
    """
    stem = Path(filename).stem
    match = re.match(r"[A-Za-z0-9]+", stem)
    return match.group(0) if match else ""


def _iter_candidate_files(
    directory: Path,
    accept_suffixes: frozenset[str],
) -> list[Path]:
    """Collect files from ``directory`` whose suffix is in the accept set.

    Non-matching suffixes are logged at DEBUG so the user can see (with
    log level DEBUG) why a file they dropped was skipped. Files are
    returned in sorted order for deterministic matching.
    """
    if not directory.is_dir():
        return []
    kept: list[Path] = []
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        if suffix in accept_suffixes:
            kept.append(f)
        else:
            logger.debug(
                "Local source skipped (suffix {} not in {}): {}",
                suffix or "(none)",
                sorted(accept_suffixes),
                f,
            )
    return kept


def match_local_sources(
    directory: Path,
    titles: dict[str, str],
    accept_suffixes: frozenset[str],
) -> tuple[dict[str, Path], list[Path]]:
    """Match files in ``directory`` to bib citekeys by filename.

    Two passes:

    1. **Citekey-prefix match.** Any filename whose leading alphanumeric
       token equals a known citation key (case-insensitive) is an
       authoritative match — the citekey is a BibTeX label, so when it
       appears as the filename prefix the user has already told us which
       reference this file is.
    2. **Fuzzy title match.** Remaining files go through
       :func:`_title_similarity` against the bib titles; matches at or
       above :data:`_MATCH_THRESHOLD` are accepted.

    Args:
        directory: Directory to scan. Missing directories return empty.
        titles: Mapping of citation key → reference title.
        accept_suffixes: Filename suffixes to consider; files with other
            suffixes are ignored. Use :data:`ACADEMIC_SUFFIXES` for the
            academic directory and :data:`WEB_SUFFIXES` for the web
            directory.

    Returns:
        ``(matched, unmatched)`` where ``matched`` is
        ``{citation_key: source_path}`` and ``unmatched`` lists source
        paths that did not match any title.
    """
    candidates = _iter_candidate_files(directory, accept_suffixes)
    if not candidates:
        return {}, []

    matched: dict[str, Path] = {}
    used_files: set[Path] = set()

    keys_by_lower = {k.lower(): k for k in titles}
    for src in candidates:
        token = _leading_token(src.name).lower()
        if not token:
            continue
        key = keys_by_lower.get(token)
        if key is None or key in matched:
            continue
        matched[key] = src
        used_files.add(src)
        logger.info(
            "Local source match: {} → {} (citekey prefix)",
            src.name,
            key,
        )

    for src in candidates:
        if src in used_files:
            continue
        normalized = _normalize_filename(src.name)
        if not normalized:
            continue

        best_key = ""
        best_score = 0.0
        for key, title in titles.items():
            if key in matched or not title:
                continue
            score = _title_similarity(normalized, title)
            if score > best_score:
                best_score = score
                best_key = key

        if best_score >= _MATCH_THRESHOLD and best_key:
            matched[best_key] = src
            used_files.add(src)
            logger.info(
                "Local source match: {} → {} (score={:.2f})",
                src.name,
                best_key,
                best_score,
            )
        elif best_key:
            logger.debug(
                "Local source no match: {} best={} (score={:.2f} < {:.2f})",
                src.name,
                best_key,
                best_score,
                _MATCH_THRESHOLD,
            )

    unmatched = [p for p in candidates if p not in used_files]
    return matched, unmatched


def copy_academic_source(
    src_path: Path,
    key: str,
    references_dir: Path,
) -> str:
    """Copy a matched academic source (PDF or MD summary) into the workspace.

    Destination filename:

    - ``.pdf`` → ``{key}_local.pdf`` (classified ``local_type="pdf"``)
    - ``.md``  → ``{key}_local.md``  (classified ``local_type="summary"``
      unless the file itself carries a ``Source: http`` header, in which
      case :func:`sciwrite_lint.references.metadata._classify_local_file`
      will mark it ``"web_page"`` — the directory is the primary signal,
      but header-based promotion still happens for correctness).

    Returns:
        The workspace-relative filename (e.g. ``"smith2020_local.pdf"``)
        suitable for storage in ``meta.access["local_file"]``.

    Raises:
        ValueError: if the source file has a suffix not in
            :data:`ACADEMIC_SUFFIXES`.
    """
    suffix = src_path.suffix.lower()
    if suffix not in ACADEMIC_SUFFIXES:
        raise ValueError(
            f"Cannot copy {src_path} as academic source: suffix {suffix!r} "
            f"not in {sorted(ACADEMIC_SUFFIXES)}"
        )
    dest_name = f"{key}_local{suffix}"
    dest = references_dir / dest_name
    references_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    logger.info("Copied academic source: {} → {}", src_path.name, dest_name)
    return dest_name


def copy_web_source(
    src_path: Path,
    key: str,
    references_dir: Path,
) -> str:
    """Copy or convert a matched web source into the workspace.

    Destination is always ``{key}_local_web.md``:

    - ``.md`` → copied verbatim (any existing ``Source:`` header is
      preserved).
    - ``.mhtml`` / ``.mht`` → parsed via :mod:`sciwrite_lint.mhtml` and
      written as markdown with the same header shape that
      :func:`sciwrite_lint.web.fetch_web_content` uses for live fetches.

    The ``_web`` infix in the filename is the primary signal picked up
    by :func:`sciwrite_lint.references.metadata._classify_local_file`
    to classify the reference as ``local_type="web_page"``.

    Returns:
        The workspace-relative filename (e.g.
        ``"smith2020_local_web.md"``).

    Raises:
        ValueError: if the source file has a suffix not in
            :data:`WEB_SUFFIXES`.
        sciwrite_lint.mhtml.MHTMLParseError: if an MHTML archive is
            malformed or produces no extractable text.
    """
    suffix = src_path.suffix.lower()
    if suffix not in WEB_SUFFIXES:
        raise ValueError(
            f"Cannot copy {src_path} as web source: suffix {suffix!r} "
            f"not in {sorted(WEB_SUFFIXES)}"
        )
    dest_name = f"{key}_local_web.md"
    dest = references_dir / dest_name
    references_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".md":
        shutil.copy2(src_path, dest)
        logger.info("Copied web source: {} → {}", src_path.name, dest_name)
        return dest_name

    # .mhtml / .mht — convert at ingest time so downstream readers only
    # ever see markdown. No runtime MHTML parsing in the claim-verify path.
    from sciwrite_lint.mhtml import mhtml_to_markdown

    markdown, source_url = mhtml_to_markdown(src_path)
    dest.write_text(markdown, encoding="utf-8")
    logger.info(
        "Converted web source: {} → {} (source={})",
        src_path.name,
        dest_name,
        source_url or "(unknown)",
    )
    return dest_name


# ---------------------------------------------------------------------------
# Hash-aware re-ingest
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """Return the hex-encoded SHA-256 of ``path``.

    Streamed in 64 KB chunks so large PDFs don't load into memory. Used
    by :func:`ingest_local_sources` to detect when a drop-folder source
    has been replaced with different bytes under the same filename.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class IngestOutcome(BaseModel):
    """Per-key result from :func:`ingest_local_sources`.

    Only returned for keys whose source was actually (re-)ingested.
    Keys whose recorded ``local_file_src_hash`` matches the current
    source on disk are absent from the result dict — callers should
    leave their metadata untouched for those.
    """

    local_file: str  # workspace-relative filename ({key}_local.pdf etc.)
    src_hash: str  # hex SHA-256 of the source file as read at ingest time
    kind: Literal["academic", "web"]


def ingest_local_sources(
    keys_titles_hashes: dict[str, tuple[str, str]],
    academic_dir: Path,
    web_dir: Path,
    references_dir: Path,
) -> dict[str, IngestOutcome]:
    """Match both drop folders and (re-)ingest any changed sources.

    This is the single entry point callers (``pipeline/fetch.py``,
    ``cli/fetch.py``) use for local-source ingestion. It handles
    matching precedence (academic beats web when a key is present in
    both folders) and hash-based re-ingest detection in one place so
    the two call sites stay identical.

    Workflow per key:

    1. Match academic dir first. If a key matches there, web is
       skipped for that key — PDF trumps web capture.
    2. For each match, compute the source's SHA-256 and compare to the
       hash the caller recorded at the previous ingest.
    3. If the hash differs (or no hash was recorded), copy/convert into
       the workspace and return an :class:`IngestOutcome`. If the hash
       matches, the source is unchanged — skip (absent from result).

    Args:
        keys_titles_hashes: ``{citekey: (bib_title, previous_src_hash)}``
            for every citation. ``previous_src_hash`` is empty when no
            previous ingest recorded one (first-time ingest will
            always re-copy). Callers normally read this from
            ``meta.access.get("local_file_src_hash", "")``.
        academic_dir: Drop folder for PDFs + MD summaries.
        web_dir: Drop folder for MD + MHTML web captures.
        references_dir: The paper workspace — ingested files land here.

    Returns:
        ``{citekey: IngestOutcome}`` for keys whose source was actually
        (re-)copied or (re-)converted. Callers should update
        ``meta.access["local_file"]`` and ``meta.access["local_file_src_hash"]``
        from the outcome and recompute tier.
    """
    if not keys_titles_hashes:
        return {}

    titles = {k: title for k, (title, _h) in keys_titles_hashes.items()}
    prior_hashes = {k: h for k, (_t, h) in keys_titles_hashes.items()}

    academic_matched, _ = match_local_sources(academic_dir, titles, ACADEMIC_SUFFIXES)
    web_titles = {k: v for k, v in titles.items() if k not in academic_matched}
    web_matched, _ = match_local_sources(web_dir, web_titles, WEB_SUFFIXES)

    outcomes: dict[str, IngestOutcome] = {}

    def _ingest_one(
        key: str,
        src_path: Path,
        kind: Literal["academic", "web"],
        copy_fn: Callable[[Path, str, Path], str],
    ) -> None:
        new_hash = file_sha256(src_path)
        if prior_hashes.get(key) == new_hash:
            logger.debug(
                "Local source unchanged for {} (hash {}...): skip re-ingest",
                key,
                new_hash[:8],
            )
            return
        local_file = copy_fn(src_path, key, references_dir)
        outcomes[key] = IngestOutcome(
            local_file=local_file,
            src_hash=new_hash,
            kind=kind,
        )

    for key, src in academic_matched.items():
        _ingest_one(key, src, "academic", copy_academic_source)
    for key, src in web_matched.items():
        _ingest_one(key, src, "web", copy_web_source)

    return outcomes
