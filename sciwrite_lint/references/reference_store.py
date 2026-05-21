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
- On-demand parse: if cache is missing or stale, parse afresh (requires GROBID).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from sciwrite_lint.pdf.grobid import GrobidReference, GrobidResult
    from sciwrite_lint.references.embedding_store import ChunkHit

# Re-export Section for convenience (canonical definition in eval_claims)
from sciwrite_lint.eval_claims import Section

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

PARSED_DIR_NAME = "parsed"

# Entry types that are book-like (used for logging, not routing).
_BOOK_ENTRY_TYPES = {"book", "inbook", "incollection", "manual", "proceedings"}


def _pdf_page_count(pdf_path: Path) -> int | None:
    """Return the number of pages in a PDF, or None if unreadable.

    pdfplumber wraps pdfminer's parsing errors plus its own; the bare
    ``Exception`` catch is intentional — pdfminer doesn't expose a stable
    public exception hierarchy, and a malformed PDF can surface as many
    things. The failure is logged at DEBUG so it's still discoverable.
    """
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        logger.debug(
            "PDF page count failed for {}: {}: {}", pdf_path.name, type(e).__name__, e
        )
        return None


_FORMAL_MIN_REFERENCES = 10


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
                except Exception as exc:  # noqa: BLE001
                    # Last-resort catch so one pathological PDF doesn't kill
                    # the whole batch. ``process_pdf`` converts httpx
                    # timeouts/transport errors into RuntimeError after
                    # retries, so this branch should be rare — but if any
                    # other library raises (sqlite, pdfplumber, etc.), we
                    # log loudly per-key and continue. The error is recorded
                    # in ``results[key] = "error"``; callers can surface it.
                    logger.error(
                        "Unexpected error parsing {} ({}): {}",
                        key,
                        type(exc).__name__,
                        exc,
                    )
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

# Default VRAM budget when CUDA isn't queryable. Used as a final
# resort by ``resolve_embed_vram_budget_gb``; on CUDA the budget is
# computed dynamically from currently-free VRAM (see that function's
# docstring).
_EMBED_VRAM_BUDGET_DEFAULT_GB = 12.0


def expected_embed_vram_budget_gb() -> float:
    """Budget the embedder *will* see when it runs (post-swap reality).

    Used by display surfaces (monitor's embedder panel) where the
    operator wants to know "is the configured embedder going to fit on
    this GPU when it runs", not "would it fit if it tried to coexist
    with whatever else is currently loaded".

    The swap pattern (``pipeline/swap.py``) guarantees the embedder
    runs alone — text vLLM and vision vLLM are stopped before its
    subprocess starts. So the right denominator for a *predicted*
    budget is total VRAM, not currently-free VRAM (which fluctuates
    with whichever container is currently resident).

    Resolution order:

    1. ``SCIWRITE_EMBED_VRAM_BUDGET_GB`` env var (override)
    2. 85 % of CUDA ``total_memory`` (the 15 % reserve covers CUDA
       runtime context + PyTorch caching-allocator drift)
    3. ``_EMBED_VRAM_BUDGET_DEFAULT_GB`` constant

    For the load-time check (where the embedder *is* currently
    loading and current free reflects what it actually has), use
    ``resolve_embed_vram_budget_gb`` instead.
    """
    import os

    override = os.environ.get("SCIWRITE_EMBED_VRAM_BUDGET_GB")
    if override:
        return float(override)

    try:
        import torch

        if torch.cuda.is_available():
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return 0.85 * total_bytes / (1024**3)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"VRAM total-memory query failed ({type(e).__name__}: {e})")
    return _EMBED_VRAM_BUDGET_DEFAULT_GB


def resolve_embed_vram_budget_gb() -> float:
    """Resolve the soft VRAM budget for the embedder load-time check.

    Resolution order:

    1. ``SCIWRITE_EMBED_VRAM_BUDGET_GB`` env var (for tests / overrides)
    2. 85 % of currently-free CUDA memory at call time
    3. ``_EMBED_VRAM_BUDGET_DEFAULT_GB`` constant (≈ 75 % of the 16 GB
       hardware minimum documented in ``docs/services.md``)

    Querying free VRAM at the moment the embedder loads gives the right
    answer regardless of hardware: the swap pattern guarantees text
    vLLM is already stopped by the time ``_get_embedding_model`` runs,
    so ``mem_get_info()`` reflects exactly what the embedder can claim.
    The 15 % reserve absorbs PyTorch caching-allocator drift, kernel
    scratch, and the small slice of free VRAM consumed by the
    just-loaded model weights themselves.

    On the 16 GB-minimum hardware the project supports, free VRAM at
    embedder-load time is typically ~14-15 GB → budget ~12-13 GB,
    which fits Arctic Embed M v2.0 (~9 GB peak at batch=32) but
    correctly warns for L v2.0 (~24 GB peak).

    Returns the budget in GiB.
    """
    import os

    override = os.environ.get("SCIWRITE_EMBED_VRAM_BUDGET_GB")
    if override:
        return float(override)

    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _total = torch.cuda.mem_get_info()
            return 0.85 * free_bytes / (1024**3)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"VRAM budget auto-detect failed ({type(e).__name__}: {e})")
    return _EMBED_VRAM_BUDGET_DEFAULT_GB


def _estimate_embedder_peak_vram_gb(
    hidden_size: int,
    num_layers: int,
    max_seq_len: int,
    batch_size: int,
) -> float:
    """Estimate worst-case peak VRAM (bf16) for the embedder forward pass.

    Lower-bound activation memory is
    ``batch × max_seq × hidden × num_layers × 2 bytes``. Empirically on
    Arctic Embed M v2.0 the actual peak runs ~2× that — accounting for
    sentence-transformers internal padding, attention scratch, weight
    storage, and PyTorch caching-allocator overhead. Sweep data point:
    batch=32 seq=6002 hidden=768 layers=12 predicted 3.4 GB; measured
    peak was 6.6 GB. The 2× factor is a conservative envelope.

    Returns the estimate in GiB. Used by ``_get_embedding_model`` for a
    load-time budget warning and by the monitor's embedder panel to
    surface the same number to the operator.
    """
    overhead = 2.0
    bf16_bytes = 2
    return (
        overhead * batch_size * max_seq_len * hidden_size * num_layers * bf16_bytes
    ) / (1024**3)


def _get_embedding_config() -> tuple[str, int, str]:
    """Return (model_name, dimension, device) from config, or defaults."""
    try:
        from sciwrite_lint.config import load_config

        cfg = load_config()
        return cfg.embedding_model, cfg.embedding_dim, cfg.embedding_device
    except Exception as e:
        logger.debug("Could not load embedding config, using defaults: {}", e)
        return EMBEDDING_MODEL, EMBEDDING_DIM, "auto"


class Chunk(BaseModel):
    """A text chunk with its position in the original document."""

    text: str
    start_char: int
    section_title: str
    granularity: str  # "paragraph" or "sentence"


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Largest a single "paragraph" can be before we treat the source's
# paragraph delimiters as unreliable and skip the L2 layer for that
# section. Auto-fetched web archives and OCR'd PDFs sometimes produce
# one giant ``\n\n``-free blob; synthesizing fake paragraph chunks
# would lie about structure that isn't there.
_MAX_PARAGRAPH_CHARS = 10_000


def _centered_window(items: list[str], center: int, half: int) -> list[str]:
    """Symmetric window around *center*: ``[center-half, center+half]``,
    clipped at sequence ends. With half=1 this is the matched item plus
    one neighbor on each side (the ±1 windowing the chunking strategy
    promises). Edge clipping means first/last items get an asymmetric
    window naturally (no synthetic padding)."""
    start = max(0, center - half)
    end = min(len(items), center + half + 1)
    return items[start:end]


def _chunk_text(
    text: str,
    section_title: str = "",
    paragraph_only: bool = False,
    paragraph_half_window: int = 1,
    sentence_half_window: int = 1,
) -> list[Chunk]:
    """Split *text* into overlapping symmetric chunks at paragraph and
    sentence granularity.

    Two granularities, both centered ±half-window sliding windows:
    - **Paragraph**: ±1 paragraph window (3 paragraphs total when in the
      interior; 2 at edges). Captures thematic context around each
      author paragraph.
    - **Sentence**: ±1 sentence window within each paragraph (3 sentences
      total when in the interior). Captures precise facts (numbers,
      definitions) with their immediately surrounding context.

    Args:
        paragraph_only: skip sentence-level splitting entirely.
        paragraph_half_window: half-width of paragraph window (default 1
            ⇒ ±1 ⇒ 3-paragraph window).
        sentence_half_window: half-width of sentence window (default 1
            ⇒ ±1 ⇒ 3-sentence window).
    """
    chunks: list[Chunk] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Detect unreliable paragraph structure: auto-fetched web archives
    # and OCR'd PDFs sometimes lose ``\n\n`` separators entirely, so
    # ``text.split("\n\n")`` returns one giant blob. Emitting a single
    # 30-KB "paragraph" chunk would lie about the source's structure
    # and defeat the L2 layer of the verify-claim ladder (L2 ends up
    # the same size as L3). Skip paragraph chunks for this section
    # when the largest paragraph exceeds ``_MAX_PARAGRAPH_CHARS``;
    # ``retrieve_top_chunks`` will return zero hits at the paragraph
    # level for this section, and the ladder escalates L1 → L3
    # naturally via ``if not units: continue``.
    has_reliable_paragraphs = paragraphs and all(
        len(p) <= _MAX_PARAGRAPH_CHARS for p in paragraphs
    )

    if has_reliable_paragraphs:
        for i, center_para in enumerate(paragraphs):
            window = _centered_window(paragraphs, i, paragraph_half_window)
            window_text = "\n\n".join(window)
            if len(window_text) > MAX_CHUNK_CHARS:
                window_text = window_text[:MAX_CHUNK_CHARS]

            # Locator points at the matched paragraph (the center), so
            # the chunk's evidence_locator can re-find it in the source.
            start = text.find(center_para)
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

    if paragraph_only:
        return chunks

    for para in paragraphs:
        para_start = text.find(para)
        if para_start == -1:
            para_start = 0
        sentences = _SENTENCE_RE.split(para)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 30]

        if not sentences:
            continue

        for i, center_sent in enumerate(sentences):
            window = _centered_window(sentences, i, sentence_half_window)
            window_text = " ".join(window)
            if len(window_text) > MAX_CHUNK_CHARS:
                window_text = window_text[:MAX_CHUNK_CHARS]
            # Locator points at the matched sentence in the source (not
            # the parent paragraph) so evidence_locator can identify the
            # specific sentence within the chunk. Search starts at
            # ``para_start`` to avoid matching an identical sentence
            # elsewhere in the document; falls back to the paragraph's
            # start if the sentence-as-stripped doesn't match verbatim.
            sent_start = text.find(center_sent, para_start)
            if sent_start == -1:
                sent_start = para_start
            chunks.append(
                Chunk(
                    text=window_text,
                    start_char=sent_start,
                    section_title=section_title,
                    granularity="sentence",
                )
            )

    return chunks


_embedding_model = None
_embedding_model_name: str = ""


def _resolve_embedding_device(device_cfg: str) -> str:
    """Resolve device string: "auto" picks CUDA when available.

    On WSL2, CUDA memory overcommit lets the embedding model (~1.2 GB)
    share VRAM with vLLM transparently. On native Linux, the pipeline
    stops vLLM before embedding to free VRAM, so GPU is safe in both cases.
    """
    if device_cfg == "auto":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device_cfg


def _embedder_config_kwargs(model_name: str) -> dict[str, object]:
    """Return ``config_kwargs`` to load ``model_name`` under SDPA attention.

    SDPA (PyTorch's native scaled_dot_product_attention dispatcher) is
    O(N) in sequence length and works for any modern HuggingFace
    transformer encoder, so it is the universal default. Some model
    families need extra config flags to actually honor that choice — this
    helper encodes the per-family quirks so swapping ``EMBEDDING_MODEL``
    does not require remembering them.

    Quirks:

    * **Alibaba GTE family** (incl. ``Snowflake/snowflake-arctic-embed-m``,
      ``Snowflake/snowflake-arctic-embed-l``, ``Alibaba-NLP/gte-*``):
      the custom modeling code's ``use_memory_efficient_attention`` flag
      defaults to True, which silently overrides ``attn_implementation``
      back to its xformers path. Without xformers installed this raises
      ``AssertionError: please install xformers``. We pin the flag False
      so SDPA actually takes effect.
    """
    kwargs: dict[str, object] = {"attn_implementation": "sdpa"}
    if model_name.startswith(("Snowflake/", "Alibaba-NLP/")):
        kwargs["use_memory_efficient_attention"] = False
    return kwargs


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

    _embedding_model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        config_kwargs=_embedder_config_kwargs(model_name),
    )
    # Cast weights to bf16 on CUDA. Ada-Lovelace / Ampere class GPUs
    # have bf16 tensor cores; matmul throughput rises ~3.5x over fp32
    # cuda cores and peak VRAM halves. bf16 embeddings match fp32 to
    # cosine ≥ 0.9998 — well within retrieval noise. bf16 is preferred
    # over fp16 because its exponent range matches fp32, avoiding NaN
    # risk in attention scores. CPU stays fp32: most x86 CPUs have no
    # native bf16 path and would run slower. Cast happens post-load to
    # sidestep the ``torch_dtype`` kwarg path, which conflicts with
    # Arctic Embed's custom ``__init__``.
    if device == "cuda":
        _embedding_model = _embedding_model.bfloat16()
        _warn_if_over_vram_budget(_embedding_model, model_name)
    _embedding_model_name = model_name
    logger.info("Embedding model loaded on {} ({})", device, model_name)
    return _embedding_model


def _warn_if_over_vram_budget(model: object, model_name: str) -> None:
    """Log a warning if the loaded model's worst-case peak exceeds the budget.

    Soft guard — does not raise. Computed at load time so a switch to a
    larger model (e.g. ``Snowflake/snowflake-arctic-embed-l-v2.0``) is
    surfaced before the first ``model.encode`` call rather than as a
    surprise OOM mid-pipeline. Budget is auto-resolved from current
    free VRAM by ``resolve_embed_vram_budget_gb``.
    """
    from sciwrite_lint.references.embedding_store import _ENCODE_BATCH_SIZE

    cfg = model[0].auto_model.config  # type: ignore[index,attr-defined]
    max_seq = (
        getattr(model, "max_seq_length", None)
        or getattr(cfg, "max_position_embeddings", None)
        or 8192
    )
    peak_gb = _estimate_embedder_peak_vram_gb(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_hidden_layers,
        max_seq_len=int(max_seq),
        batch_size=_ENCODE_BATCH_SIZE,
    )
    budget_gb = resolve_embed_vram_budget_gb()
    if peak_gb > budget_gb:
        logger.warning(
            "Embedder peak VRAM estimate {:.1f} GB exceeds budget {:.1f} GB "
            "(model={}, batch={}, max_seq={}, hidden={}, layers={}). "
            "Consider lowering _ENCODE_BATCH_SIZE in embedding_store.py "
            "or switching to a smaller embedder.",
            peak_gb,
            budget_gb,
            model_name,
            _ENCODE_BATCH_SIZE,
            int(max_seq),
            cfg.hidden_size,
            cfg.num_hidden_layers,
        )
    else:
        logger.info(
            "Embedder peak VRAM estimate {:.1f} GB (within {:.1f} GB budget)",
            peak_gb,
            budget_gb,
        )


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
        import torch

        torch.cuda.empty_cache()
        logger.debug("GPU memory released after embedding")


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
    from sciwrite_lint.references._embed_timing import time_phase
    from sciwrite_lint.references.embedding_store import store_embeddings

    with time_phase("split_sections"):
        sections = split_sections(text)
    all_chunks: list[Chunk] = []
    sections_with_paragraph_chunks: set[str] = set()
    with time_phase("chunk_text"):
        for sec in sections:
            sec_chunks = _chunk_text(sec.text, section_title=sec.title)
            all_chunks.extend(sec_chunks)
            if any(c.granularity == "paragraph" for c in sec_chunks):
                sections_with_paragraph_chunks.add(sec.title)

    if not all_chunks:
        return 0

    # L2 coverage diagnostic: ratio of sections that emitted paragraph
    # chunks to sections we tried to chunk. < 100% means some sections
    # tripped the ``_MAX_PARAGRAPH_CHARS`` reliability gate (oversized
    # paragraph → unreliable structure → L2 honestly skipped). One INFO
    # line per ref so the user can spot bad apples (web archives where
    # trafilatura emits one ``<p>`` per section, OCR'd PDFs that lost
    # paragraph structure) without per-section log spam.
    sections_attempted = sum(1 for s in sections if s.text.strip())
    if sections_attempted:
        l2_pct = 100 * len(sections_with_paragraph_chunks) / sections_attempted
        logger.info(
            "Chunked {}: {}/{} sections L2-covered ({:.0f}%)",
            key,
            len(sections_with_paragraph_chunks),
            sections_attempted,
            l2_pct,
        )

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

    with time_phase("update_parse_cache"):
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

    Used at the section level (L3) of the verify-claim ladder. Returns
    a filtered list of sections, or ``None`` if embeddings are
    unavailable for *key* — the ladder cannot run without embeddings,
    so the caller surfaces the claim as ``CANNOT_DETERMINE`` rather
    than falling back to an unfiltered scan.

    Strategy:
    1. KNN query against sqlite-vec for this reference's chunks.
    2. Apply relative scoring cutoff (within min_score_ratio of best).
    3. Map chunk hits back to sections (expand ±1 neighbor for context).
    4. Return deduplicated sections ordered by original position.

    When the filter would select ≥70% of the document's sections,
    returns ``all_sections`` instead (the claim is broadly relevant —
    no filter is selective enough to be useful).
    """
    from sciwrite_lint.references.embedding_store import (
        has_embeddings,
        retrieve_similar,
    )

    model_name, _, _ = _get_embedding_config()

    # If document chunk embeddings are missing, return None immediately.
    # Do NOT try to embed here — that would load the embedding model in
    # the parent process, competing with vLLM for VRAM. Embeddings are
    # pre-computed in Stage 4b subprocess.
    if not has_embeddings(key, references_dir, model_name=model_name):
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
    best_score = hits[0].score
    cutoff = best_score * min_score_ratio
    hits = [h for h in hits if h.score >= cutoff]

    # Map chunk hits to sections (deduplicate, include ±1 neighbors)
    seen_titles: set[str] = set()
    matched_sections: list[Section] = []

    for hit in hits:
        title = hit.section_title
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


def retrieve_top_chunks(
    claim_text: str,
    key: str,
    references_dir: Path,
    granularity: str,
    top_n: int = 3,
) -> list["ChunkHit"] | None:
    """Find top-N chunks of a specific granularity for a claim.

    Used by the verify-claim escalation ladder
    (``sciwrite_lint/eval_claims.py``) to ship matched sentence ±1 or
    paragraph ±1 chunks directly to the LLM instead of the parent
    section. The chunk windows themselves already encode neighbor
    context (see ``_chunk_text``: ``paragraph_half_window=1``,
    ``sentence_half_window=1`` ⇒ ±1 symmetric).

    The KNN is scoped to *granularity* via a rowid pre-filter, so the
    returned hits are the genuine top-N for that level — they are not
    competing with the other granularity's chunks for ranking slots
    (mixed-granularity KNN lets sentence chunks out-rank paragraph
    chunks because shorter, denser chunks score higher on focused
    queries; per-level scoping fixes that).

    Returns:
      - ``None`` only when embeddings are unavailable for *key* — the
        ladder cannot run, caller bails with CANNOT_DETERMINE.
      - ``[]`` when embeddings exist but no chunks of this granularity
        do (e.g. the chunker's reliability gate skipped paragraph
        chunks for every section). Ladder escalates to the next level.
      - non-empty list otherwise, capped at *top_n*.
    """
    if granularity not in ("sentence", "paragraph"):
        raise ValueError(
            f"granularity must be 'sentence' or 'paragraph', got {granularity!r}"
        )

    from sciwrite_lint.references.embedding_store import (
        has_embeddings,
        retrieve_similar,
    )

    model_name, _, _ = _get_embedding_config()
    if not has_embeddings(key, references_dir, model_name=model_name):
        return None

    return retrieve_similar(
        claim_text, key, references_dir, top_k=top_n, granularity=granularity
    )


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

    Returns (markdown_text, chunk_count). chunk_count is 0 if embed=False.
    """
    text = await parse_and_store(key, pdf_path, references_dir, force=force)
    if not text:
        return None, 0

    chunk_count = 0
    if embed:
        chunk_count = compute_and_store_embeddings(key, text, references_dir)

    return text, chunk_count
