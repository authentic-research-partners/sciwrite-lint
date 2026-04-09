"""Persistent reference store: parse once, reuse everywhere.

Parses PDFs via GROBID, stores the markdown in
``references/parsed/{key}.md`` with a metadata sidecar. Subsequent reads
are instant file reads — no re-parsing.

Optionally computes Snowflake Arctic Embed-S embeddings for each chunk
and stores them in ``references/parsed/embeddings.db`` (sqlite-vec).
This enables fast semantic search for claim verification (see
``retrieve_relevant_sections``).

Design rationale:
- GROBID parsing is 5-15 s per PDF — deterministic output, cache it.
- Embeddings are ~500 ms per reference on CPU — no GPU contention.
- Hash-based invalidation: re-parse only if PDF content changes.
- Fallback: if cache missing/stale, parse on demand (requires GROBID).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from sciwrite_lint.pdf.grobid import GrobidReference, GrobidResult

# Re-export Section for convenience (canonical definition in eval_claims)
from sciwrite_lint.eval_claims import Section

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

PARSED_DIR_NAME = "parsed"

# Entry types that are book-like (used for logging, not routing).
_BOOK_ENTRY_TYPES = {"book", "inbook", "incollection", "manual", "proceedings"}


def _pdf_page_count(pdf_path: Path) -> int | None:
    """Return the number of pages in a PDF, or None if unreadable."""
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return None


_FORMAL_MIN_REFERENCES = 10


@dataclass
class ParseMeta:
    """Metadata sidecar for a cached parse."""

    pdf_hash: str
    parse_date: str
    parser: str  # "grobid"
    sections_count: int
    char_count: int
    is_formal: bool = False
    has_embeddings: bool = False
    embedding_model: str = ""
    chunks_count: int = 0


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    """SHA-256 of file contents (fast enough for PDFs up to ~50 MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Parsed cache paths
# ---------------------------------------------------------------------------


def _parsed_dir(references_dir: Path) -> Path:
    return references_dir / PARSED_DIR_NAME


def _parsed_md_path(references_dir: Path, key: str) -> Path:
    return _parsed_dir(references_dir) / f"{key}.md"


def _parsed_meta_path(references_dir: Path, key: str) -> Path:
    return _parsed_dir(references_dir) / f"{key}.meta.json"


def _embeddings_path(references_dir: Path, key: str) -> Path:
    return _parsed_dir(references_dir) / f"{key}.embeddings.npz"


# ---------------------------------------------------------------------------
# Core: parse and store
# ---------------------------------------------------------------------------


async def parse_and_store(
    key: str,
    pdf_path: Path,
    references_dir: Path,
    force: bool = False,
    entry_type: str = "",
) -> str | None:
    """Parse a PDF via GROBID and store the result in references/parsed/.

    Returns the parsed markdown text, or None on failure.
    Skips parsing if a valid cache exists (unless *force*).

    All PDFs (articles and books) are parsed via GROBID, which produces
    usable section structure even for 55-page books.
    """
    from datetime import datetime

    if not pdf_path.exists() or pdf_path.suffix != ".pdf":
        return None

    parsed_dir = _parsed_dir(references_dir)
    md_path = _parsed_md_path(references_dir, key)

    # Check cache validity via DB
    from sciwrite_lint.references.workspace_db import (
        get_db,
        load_parse_cache,
        save_parse_cache,
    )

    with get_db(references_dir) as conn:
        if not force and md_path.exists():
            cached = load_parse_cache(conn, key)
            if cached and cached["pdf_hash"] == _file_hash(pdf_path):
                return md_path.read_text(encoding="utf-8")

        grobid_result = await _try_grobid(pdf_path)
        text = _grobid_result_to_markdown(grobid_result)
        formal = _is_formal_document(grobid_result)

        # Store
        parsed_dir.mkdir(parents=True, exist_ok=True)

        md_path.write_text(text, encoding="utf-8")

        from sciwrite_lint.eval_claims import split_sections

        sections = split_sections(text)

        save_parse_cache(
            conn,
            key,
            pdf_hash=_file_hash(pdf_path),
            parse_date=datetime.now().isoformat(timespec="seconds"),
            parser="grobid",
            sections_count=len(sections),
            char_count=len(text),
            is_formal=formal,
        )

    if not formal:
        logger.info(
            "{}: non-formal document (title={}, authors={}, refs={})",
            key,
            bool(grobid_result.title),
            len(grobid_result.authors),
            len(grobid_result.references),
        )

    # Persist structured bibliography entries in workspace.db for chain
    # metadata verification. Only for formal documents — non-formal PDFs
    # (webpages, news) have unreliable GROBID bibliographies.
    # Note: the parsed markdown + embeddings are stored regardless and
    # remain available for claim verification at depth 0→1.
    if formal and grobid_result.references:
        _register_grobid_bibliography(key, grobid_result.references, references_dir)

    return text


def _register_grobid_bibliography(
    parent_key: str,
    references: list[GrobidReference],
    references_dir: Path,
) -> None:
    """Register GROBID-parsed bibliography entries in workspace.db.

    Each entry is stored at depth=1 with parent_key pointing to the
    reference whose bibliography was parsed.
    """
    from sciwrite_lint.references.workspace_db import get_db, register_reference

    with get_db(references_dir) as conn:
        for ref in references:
            register_reference(
                conn,
                ref_key=f"bib_{ref.index}",
                workspace_path=".",
                depth=1,
                parent_key=parent_key,
                doi=ref.doi or None,
                arxiv_id=ref.arxiv_id or None,
                pmid=ref.pmid or None,
                isbn=ref.isbn or None,
                lccn=ref.lccn or None,
                title=ref.title or None,
                authors=ref.authors if ref.authors else None,
                year=ref.year or None,
                venue=ref.venue or None,
            )


async def _try_grobid(pdf_path: Path) -> "GrobidResult":
    """Parse PDF via GROBID. Raises RuntimeError if GROBID is unavailable."""
    from sciwrite_lint.pdf.grobid import is_grobid_running, process_pdf

    if not await is_grobid_running():
        raise RuntimeError(
            "GROBID is required for PDF parsing.\n"
            "  Start with: sciwrite-lint containers start"
        )
    result = await process_pdf(pdf_path)
    text = _grobid_result_to_markdown(result)
    if not text or len(text) <= 100:
        raise RuntimeError(
            f"GROBID returned empty/too-short output for {pdf_path.name}"
        )
    return result


def _grobid_result_to_markdown(result: "GrobidResult") -> str:
    """Convert a GrobidResult to markdown with ## headings."""
    lines: list[str] = []
    if result.abstract:
        lines.append("## Abstract")
        lines.append("")
        lines.append(result.abstract)
        lines.append("")
    for sec in result.sections:
        prefix = "#" * (sec.level + 2)
        lines.append(f"{prefix} {sec.title}")
        lines.append("")
        lines.append(sec.text)
        lines.append("")
    return "\n".join(lines)


def _is_formal_document(result: "GrobidResult") -> bool:
    """Classify a GROBID parse result as formal academic document.

    Formal = has title + at least 1 author + at least 10 references.
    Non-formal (news articles, guides, short essays) lack structured
    metadata and are used as text only — no reference extraction or
    depth-2 chain verification.
    """
    return (
        bool(result.title)
        and len(result.authors) >= 1
        and len(result.references) >= _FORMAL_MIN_REFERENCES
    )


def is_formal_cached(key: str, references_dir: Path) -> bool:
    """Check if a parsed reference is a formal academic document.

    Reads the ``is_formal`` flag from the parse_cache table in workspace.db.
    Returns False if no record exists.

    Convenience wrapper — use ``is_formal_cached_db(conn, key)`` directly
    when you already have a connection open.
    """
    from sciwrite_lint.references.workspace_db import get_db, is_formal_cached_db

    with get_db(references_dir) as conn:
        return is_formal_cached_db(conn, key)


# ---------------------------------------------------------------------------
# Read from cache (with lazy parse on miss)
# ---------------------------------------------------------------------------


async def read_cached_reference(
    key: str,
    ref_path: Path,
    references_dir: Path,
) -> str | None:
    """Read a reference, using the persistent cache when available.

    For PDFs: check parsed cache first, parse on demand if missing.
    For .md files: read directly (no caching needed).
    """
    if ref_path.suffix == ".md":
        return ref_path.read_text(encoding="utf-8") if ref_path.exists() else None

    if ref_path.suffix == ".pdf":
        if not ref_path.exists():
            return None

        # Try cache first via DB
        md_path = _parsed_md_path(references_dir, key)
        if md_path.exists():
            from sciwrite_lint.references.workspace_db import get_db, load_parse_cache

            with get_db(references_dir) as conn:
                cached = load_parse_cache(conn, key)
                if cached and cached["pdf_hash"] == _file_hash(ref_path):
                    return md_path.read_text(encoding="utf-8")

        # Cache miss — parse on demand
        return await parse_and_store(key, ref_path, references_dir)

    return None


# ---------------------------------------------------------------------------
# Batch parse: all T1 references
# ---------------------------------------------------------------------------


async def parse_all_missing(
    references_dir: Path,
    force: bool = False,
    sem: "asyncio.Semaphore | None" = None,
) -> dict[str, str]:
    """Parse all PDFs with local files that don't have a cached parse.

    Returns {key: status} where status is "cached", "parsed", or "failed".
    Sends up to 4 concurrent requests to GROBID (multi-threaded Java server).

    Includes all tiers with local PDFs (T0, T1, T2) — claim verification
    runs against any ref with a local file, so all need parsed markdown
    and embeddings.

    Args:
        sem: Semaphore for concurrency control. If None, creates a local one.
            Pass the pipeline's shared semaphore to enable memory-based throttling.
    """
    from sciwrite_lint.references.workspace_db import (
        get_db,
        load_all_parse_cache,
        query_refs_with_local_pdfs,
    )

    with get_db(references_dir) as conn:
        pdf_refs = query_refs_with_local_pdfs(conn)
        all_parse_cache = load_all_parse_cache(conn) if not force else {}

    results: dict[str, str] = {}
    to_parse: list[tuple[str, Path, str]] = []

    for key, (local_file, entry_type) in pdf_refs.items():
        pdf_path = references_dir / local_file
        if not pdf_path.exists():
            results[key] = "missing_pdf"
            continue

        md_path = _parsed_md_path(references_dir, key)

        # Already cached and valid?
        if not force and md_path.exists():
            cached = all_parse_cache.get(key)
            if cached and cached["pdf_hash"] == _file_hash(pdf_path):
                results[key] = "cached"
                continue

        to_parse.append((key, pdf_path, entry_type))

    if to_parse:
        from sciwrite_lint.pdf.grobid import MAX_PARSE_CONCURRENCY

        parse_sem = sem or asyncio.Semaphore(MAX_PARSE_CONCURRENCY)

        async def _parse_one(key: str, pdf_path: Path, entry_type: str) -> None:
            async with parse_sem:
                try:
                    text = await parse_and_store(
                        key,
                        pdf_path,
                        references_dir,
                        force=force,
                        entry_type=entry_type,
                    )
                except RuntimeError as exc:
                    logger.error("Failed to parse {}: {}", key, exc)
                    results[key] = "error"
                    return
            results[key] = "parsed" if text else "failed"

        await asyncio.gather(*[_parse_one(k, p, et) for k, p, et in to_parse])

    return results


# ---------------------------------------------------------------------------
# Embeddings: chunk + embed + store
# ---------------------------------------------------------------------------

# Embedding defaults (overridden by [embeddings] in .sciwrite-lint.toml)
EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-m-v2.0"
EMBEDDING_DIM = 768

# Maximum chars per embedding chunk. Arctic Embed M v2.0 has an 8192-token
# context window (~32k chars). This cap prevents storing absurdly large
# chunks from badly parsed PDFs. Set to 30k to stay within the model's
# window with margin for non-English text.
MAX_CHUNK_CHARS = 30_000


def _get_embedding_config() -> tuple[str, int, str]:
    """Return (model_name, dimension, device) from config, or defaults."""
    try:
        from sciwrite_lint.config import load_config

        cfg = load_config()
        return cfg.embedding_model, cfg.embedding_dim, cfg.embedding_device
    except Exception as e:
        logger.debug("Could not load embedding config, using defaults: {}", e)
        return EMBEDDING_MODEL, EMBEDDING_DIM, "auto"


@dataclass
class Chunk:
    """A text chunk with its position in the original document."""

    text: str
    start_char: int
    section_title: str
    granularity: str  # "paragraph" or "sentence"


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _chunk_text(
    text: str,
    section_title: str = "",
    paragraph_only: bool = False,
    paragraph_window: int = 3,
    sentence_window: int = 3,
) -> list[Chunk]:
    """Split text into overlapping chunks at paragraph and sentence level.

    Two granularities, both using sliding windows:
    - **Paragraph**: 3-paragraph window, stride 1. Each chunk is 1-3 real
      paragraphs as the author wrote them. Captures thematic ideas.
    - **Sentence**: 3-sentence window, stride 1, within each paragraph.
      Captures precise facts (numbers, definitions).

    Args:
        paragraph_only: skip sentence-level splitting entirely.
        paragraph_window: paragraphs per paragraph-level chunk (default 3).
        sentence_window: sentences per sentence-level chunk (default 3).
    """
    chunks: list[Chunk] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Paragraph-level chunks: sliding window of real paragraphs
    for i in range(len(paragraphs)):
        window = paragraphs[i : i + paragraph_window]
        window_text = "\n\n".join(window)
        if len(window_text) > MAX_CHUNK_CHARS:
            window_text = window_text[:MAX_CHUNK_CHARS]

        # Find position of first paragraph in window
        start = text.find(paragraphs[i])
        if start == -1:
            start = 0

        chunks.append(
            Chunk(
                text=window_text,
                start_char=start,
                section_title=section_title,
                granularity="paragraph",
            )
        )
        if i + paragraph_window >= len(paragraphs):
            break

    if paragraph_only:
        return chunks

    # Sentence-level chunks: sliding window within each paragraph
    for para in paragraphs:
        para_start = text.find(para)
        if para_start == -1:
            para_start = 0
        sentences = _SENTENCE_RE.split(para)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 30]

        if not sentences:
            continue

        for i in range(len(sentences)):
            window = sentences[i : i + sentence_window]
            window_text = " ".join(window)
            if len(window_text) > MAX_CHUNK_CHARS:
                window_text = window_text[:MAX_CHUNK_CHARS]
            chunks.append(
                Chunk(
                    text=window_text,
                    start_char=para_start,
                    section_title=section_title,
                    granularity="sentence",
                )
            )
            if i + sentence_window >= len(sentences):
                break

    return chunks


_embedding_model = None
_embedding_model_name: str = ""


def _resolve_embedding_device(device_cfg: str) -> str:
    """Resolve device string: "auto" picks CUDA on WSL2, else CPU.

    On WSL2, CUDA memory overcommit lets the embedding model (~1.2 GB)
    share VRAM with vLLM transparently — idle KV-cache pages swap to
    system RAM while embedding runs. On native Linux, cudaMalloc is
    physical with no overcommit, so GPU embedding is not supported
    in auto mode (use device="cuda" to force it).
    """
    from sciwrite_lint.config import is_wsl2

    if device_cfg == "auto":
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
            if is_wsl2():
                return "cuda"
            # Native Linux: no VRAM overcommit, default to CPU.
            # Users can force GPU via [embeddings] device = "cuda".
            return "cpu"
        except ImportError:
            return "cpu"
    return device_cfg


def _get_embedding_model():
    """Lazy singleton for embedding model.

    Device is selected via config: "auto" (default) uses CUDA when available.
    See _resolve_embedding_device() for GPU memory sharing details.

    Reads model name from config. If the config model differs from the
    loaded singleton, reloads (handles config changes between runs).
    """
    global _embedding_model, _embedding_model_name

    model_name, _, device_cfg = _get_embedding_config()

    if _embedding_model is not None and _embedding_model_name == model_name:
        return _embedding_model

    device = _resolve_embedding_device(device_cfg)

    from sentence_transformers import SentenceTransformer

    # Arctic Embed's custom attention code requires xformers for "sdpa" mode.
    # Force "eager" attention to avoid that dependency on GPU. Also disable
    # memory_efficient_attention (xformers assertion guard).
    _embedding_model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        config_kwargs={
            "use_memory_efficient_attention": False,
            "attn_implementation": "eager",
        },
    )
    _embedding_model_name = model_name
    logger.info("Embedding model loaded on {} ({})", device, model_name)
    return _embedding_model


def release_embedding_model() -> None:
    """Free the embedding model and release GPU memory.

    Must be called after embedding completes and before vLLM inference
    resumes, to return VRAM to the vLLM process. Without this, the
    embedding model stays in GPU memory until garbage collected, which
    may overlap with vLLM's claim verification stage.
    """
    global _embedding_model, _embedding_model_name
    if _embedding_model is None:
        return

    device = str(_embedding_model.device)
    del _embedding_model
    _embedding_model = None
    _embedding_model_name = ""

    if "cuda" in device:
        try:
            import torch

            torch.cuda.empty_cache()
            logger.debug("GPU memory released after embedding")
        except ImportError:
            pass


def compute_and_store_embeddings(
    key: str,
    text: str,
    references_dir: Path,
) -> int:
    """Chunk text, compute embeddings, store in sqlite-vec DB.

    OOM-safe: encodes and inserts in batches of 32, never holding all
    vectors in RAM at once.

    Uses 3-sentence sliding window for sentence-level chunks (proven
    by eval to achieve 100% Recall@5 on both books and articles with
    fewer chunks than single-sentence splitting).

    Returns chunk count.
    """
    from sciwrite_lint.eval_claims import split_sections
    from sciwrite_lint.references.embedding_store import store_embeddings

    sections = split_sections(text)
    all_chunks: list[Chunk] = []
    for sec in sections:
        all_chunks.extend(_chunk_text(sec.text, section_title=sec.title))

    if not all_chunks:
        return 0

    model_name, dim, _ = _get_embedding_config()

    chunk_dicts = [
        {
            "text": c.text,
            "section_title": c.section_title,
            "granularity": c.granularity,
            "start_char": c.start_char,
        }
        for c in all_chunks
    ]

    count = store_embeddings(key, chunk_dicts, references_dir, model_name, dim)

    # Update parse cache in DB
    from sciwrite_lint.references.workspace_db import (
        get_db,
        update_parse_cache_embeddings,
    )

    with get_db(references_dir) as conn:
        update_parse_cache_embeddings(
            conn,
            key,
            has_embeddings=True,
            embedding_model=model_name,
            chunks_count=count,
        )

    return count


# ---------------------------------------------------------------------------
# Semantic retrieval: find relevant sections for a claim
# ---------------------------------------------------------------------------


def retrieve_relevant_sections(
    claim_text: str,
    key: str,
    references_dir: Path,
    all_sections: list[Section],
    top_k: int = 5,
    min_score_ratio: float = 0.95,
) -> list[Section] | None:
    """Find sections most relevant to a claim using sqlite-vec KNN.

    Returns a filtered list of sections, or None if embeddings unavailable
    (caller should use full scan instead).

    Strategy:
    1. KNN query against sqlite-vec for this reference's chunks.
    2. Apply relative scoring cutoff (within min_score_ratio of best).
    3. Map chunk hits back to sections (expand ±1 neighbor for context).
    4. Return deduplicated sections ordered by original position.

    Returns None if embeddings are not available (caller uses full scan).
    """
    from sciwrite_lint.references.embedding_store import (
        has_embeddings,
        retrieve_similar,
    )

    model_name, _, _ = _get_embedding_config()

    # has_embeddings checks: exists + complete + model match.
    # If False (missing, incomplete, or model changed), try to (re-)embed.
    if not has_embeddings(key, references_dir, model_name=model_name):
        md_path = _parsed_md_path(references_dir, key)
        if not md_path.exists():
            return None
        try:
            text = md_path.read_text(encoding="utf-8")
            compute_and_store_embeddings(key, text, references_dir)
        except Exception as e:
            logger.debug("Embedding failed for {}: {}", key, e)
            return None

    # KNN search via sqlite-vec
    hits = retrieve_similar(claim_text, key, references_dir, top_k=top_k * 3)
    if not hits:
        logger.debug(
            "retrieve_similar returned no hits for {} (model mismatch or no chunks?)",
            key,
        )
        return None

    # Apply relative scoring cutoff
    best_score = hits[0]["score"]
    cutoff = best_score * min_score_ratio
    hits = [h for h in hits if h["score"] >= cutoff]

    # Map chunk hits to sections (deduplicate, include ±1 neighbors)
    seen_titles: set[str] = set()
    matched_sections: list[Section] = []

    for hit in hits:
        title = hit["section_title"]
        if title in seen_titles:
            continue
        seen_titles.add(title)

        for i, sec in enumerate(all_sections):
            if sec.title == title:
                if i > 0 and all_sections[i - 1].title not in seen_titles:
                    matched_sections.append(all_sections[i - 1])
                    seen_titles.add(all_sections[i - 1].title)
                matched_sections.append(sec)
                if (
                    i + 1 < len(all_sections)
                    and all_sections[i + 1].title not in seen_titles
                ):
                    matched_sections.append(all_sections[i + 1])
                    seen_titles.add(all_sections[i + 1].title)
                break

        if len(matched_sections) >= top_k:
            break

    matched_sections.sort(key=lambda s: s.index)

    if not matched_sections:
        # KNN returned hits but none mapped to sections in all_sections.
        # This indicates a section title mismatch between embedding store
        # and the current parsed sections — treat as unavailable.
        logger.debug(
            "Embedding hits for {} didn't map to any sections (title mismatch?)",
            key,
        )
        return None

    if len(matched_sections) >= len(all_sections) * 0.7:
        # Filtering wasn't selective — claim is broadly relevant to the
        # document. Return all sections so the caller uses a full scan.
        logger.debug(
            "Embedding filter for {} matched {}/{} sections, using full scan",
            key,
            len(matched_sections),
            len(all_sections),
        )
        return all_sections

    return matched_sections


# ---------------------------------------------------------------------------
# Convenience: parse + embed in one call
# ---------------------------------------------------------------------------


async def parse_and_embed(
    key: str,
    pdf_path: Path,
    references_dir: Path,
    force: bool = False,
    embed: bool = True,
) -> tuple[str | None, int]:
    """Parse a PDF and optionally compute embeddings.

    Returns (markdown_text, chunk_count). chunk_count is 0 if embed=False
    or if sentence-transformers is not installed.
    """
    text = await parse_and_store(key, pdf_path, references_dir, force=force)
    if not text:
        return None, 0

    chunk_count = 0
    if embed:
        try:
            chunk_count = compute_and_store_embeddings(key, text, references_dir)
        except ImportError:
            pass  # sentence-transformers not installed — skip embeddings

    return text, chunk_count
