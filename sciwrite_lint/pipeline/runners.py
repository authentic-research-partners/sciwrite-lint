"""Subprocess entry-points and batch subprocess launchers.

Every GPU-heavy stage runs in a subprocess so its CUDA context is fully
released when done — otherwise the context persists in the parent and
steals VRAM from vLLM's KV cache. The batch launchers spawn **one**
subprocess for all papers in a stage (single model load), while the
single-paper launchers are kept for :func:`run_full_check`.

The ``_*_entry`` functions are the subprocess targets referenced from
``python -c "from sciwrite_lint.pipeline import _..._entry; _..._entry(...)"``
strings, so they must remain importable from ``sciwrite_lint.pipeline``
(re-exported via ``__init__.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.config import LintConfig

# Embedding subprocess timeout is data-dependent: short docs finish in under
# a minute, but book-length parsed references (e.g. a 278-page Nuffield
# review at ~500 KB of markdown) chunk and encode for several minutes. We
# scale the timeout from the total bytes going into the subprocess, with a
# conservative throughput floor so the budget holds even when the GPU is
# contended. The ceiling is sized to cover up to a ~1000-page book at
# roughly 4 KB/page of parsed markdown (~4 MB) — anything beyond that is
# almost certainly a wedged subprocess and should be killed.
_EMBED_TIMEOUT_BASE_S = 120  # model load + fixed overhead
_EMBED_BYTES_PER_SECOND = 1024  # conservative effective throughput floor
_EMBED_TIMEOUT_MIN_S = 300  # prior hardcoded value — lower bound
_EMBED_TIMEOUT_CEILING_S = 4500  # hard cap: 75 min (fits ~1000-page book)


def _resolve_embed_text_paths(keys: list[str], references_dir: Path) -> dict[str, Path]:
    """Resolve each key's source .md file for embedding.

    Resolution order per key:
      1. ``parsed/{key}.md`` — GROBID output for downloaded/local PDFs.
      2. ``{references_dir}/{local_file}`` where ``local_file`` comes from
         ``CitationMetadata.access["local_file"]`` — covers OA web summaries
         (``_web.md``), local web-page drops (``_local_web.md``), and any
         future variant the ingest layer adds. Metadata is the authoritative
         source; the subprocess must not reconstruct filenames by convention.

    Keys with no resolvable source file are absent from the result. Metadata
    is loaded once per call.
    """
    paths: dict[str, Path] = {}
    need_meta: list[str] = []
    for key in keys:
        md = references_dir / "parsed" / f"{key}.md"
        if md.exists():
            paths[key] = md
        else:
            need_meta.append(key)
    if need_meta:
        from sciwrite_lint.references.metadata import load_all_metadata

        all_meta = load_all_metadata(references_dir)
        for key in need_meta:
            meta = all_meta.get(key)
            if meta is None:
                continue
            local_file = meta.access.get("local_file") or ""
            if not local_file:
                continue
            p = references_dir / local_file
            if p.exists():
                paths[key] = p
    return paths


def _iter_embed_text_paths(keys: list[str], references_dir: Path) -> list[Path]:
    """Resolve the .md files the embedding subprocess will read for ``keys``."""
    return list(_resolve_embed_text_paths(keys, references_dir).values())


def _compute_embed_timeout(text_paths: list[Path]) -> int:
    """Scale embedding-subprocess timeout by total input bytes.

    Returns a clamped timeout in seconds: ``max(MIN, min(CEILING, BASE +
    total_bytes / RATE))``. Unreadable files are skipped silently — the
    subprocess will surface the real failure.
    """
    total_bytes = 0
    for p in text_paths:
        try:
            total_bytes += p.stat().st_size
        except OSError:
            continue
    scaled = _EMBED_TIMEOUT_BASE_S + total_bytes // _EMBED_BYTES_PER_SECOND
    return max(_EMBED_TIMEOUT_MIN_S, min(_EMBED_TIMEOUT_CEILING_S, scaled))


def _run_embeddings_subprocess(
    keys: list[str],
    references_dir: Path,
    config: LintConfig,
    claim_texts: list[str] | None = None,
) -> str:
    """Run embedding computation in a subprocess for CUDA isolation.

    The embedding model brings batch data to VRAM; subprocess isolation
    ensures all CUDA allocations are released when embedding finishes.
    Also pre-computes claim query vectors if ``claim_texts`` is provided.

    Returns:
        Empty string on success. On failure (non-zero exit, timeout, or
        crash), returns a human-readable diagnostic (subprocess stderr
        tail, timeout notice, or exception message) for inclusion in the
        RuntimeError raised by ``_verify_embeddings_or_raise``.
    """
    import json as _json
    import subprocess
    import sys

    # Pass keys as comma-separated arg, references_dir, claim texts as JSON
    keys_str = ",".join(keys)
    claims_json = _json.dumps(claim_texts or [])
    cmd = [
        sys.executable,
        "-c",
        "import sys; "
        "from sciwrite_lint.pipeline import _embed_keys; "
        f"_embed_keys({keys_str!r}, {str(references_dir)!r}, {claims_json!r})",
    ]
    timeout = _compute_embed_timeout(_iter_embed_text_paths(keys, references_dir))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(config.project_dir) if config.project_dir else None,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.warning("Embedding subprocess failed: {}", stderr)
            return f"non-zero exit {result.returncode}: {stderr}"
    except subprocess.TimeoutExpired:
        logger.warning("Embedding subprocess timed out ({}s)", timeout)
        return f"subprocess timed out after {timeout}s"
    except Exception as e:
        logger.warning("Embedding subprocess error: {}", e)
        return f"{type(e).__name__}: {e}"
    return ""


def _embed_keys(
    keys_csv: str, references_dir_str: str, claim_texts_json: str = "[]"
) -> None:
    """Subprocess entry point: compute embeddings for given keys + claim queries."""
    import json as _json

    from sciwrite_lint.references.reference_store import (
        compute_and_store_embeddings,
        release_embedding_model,
    )

    references_dir = Path(references_dir_str)
    keys = keys_csv.split(",") if keys_csv else []

    paths = _resolve_embed_text_paths(keys, references_dir)
    for key in keys:
        text_path = paths.get(key)
        if text_path is None:
            continue
        try:
            text = text_path.read_text(encoding="utf-8")
            compute_and_store_embeddings(key, text, references_dir)
        except ImportError:
            break  # sentence-transformers not installed
        except Exception as e:
            logger.debug(f"embedding skipped for {key} ({type(e).__name__}: {e})")
            continue

    # Pre-compute claim query vectors (model already loaded above)
    claim_texts = _json.loads(claim_texts_json)
    if claim_texts:
        _encode_claim_queries(claim_texts, references_dir)

    release_embedding_model()


def _batch_embed_entry(manifest_path_str: str) -> None:
    """Subprocess entry point: embed keys for multiple papers in one process.

    Loads the embedding model once and iterates over papers. Each paper's
    keys are embedded and stored in its own workspace. Also pre-computes
    claim query vectors so Stage 5 never loads the model in the parent.

    Manifest JSON: list of {"keys": [...], "references_dir": "...",
    "claim_texts": [...]}.
    """
    import json

    from sciwrite_lint.references.reference_store import (
        compute_and_store_embeddings,
        release_embedding_model,
    )

    manifest = json.loads(Path(manifest_path_str).read_text(encoding="utf-8"))

    for entry in manifest:
        references_dir = Path(entry["references_dir"])
        keys = entry.get("keys", [])
        paths = _resolve_embed_text_paths(keys, references_dir)
        for key in keys:
            text_path = paths.get(key)
            if text_path is None:
                continue
            try:
                text = text_path.read_text(encoding="utf-8")
                compute_and_store_embeddings(key, text, references_dir)
            except ImportError:
                release_embedding_model()
                return
            except Exception as e:
                logger.debug(f"embedding skipped for {key} ({type(e).__name__}: {e})")
                continue

        # Pre-compute claim query vectors (model already loaded above)
        claim_texts = entry.get("claim_texts", [])
        if claim_texts:
            _encode_claim_queries(claim_texts, references_dir)

    release_embedding_model()


def _encode_claim_queries(claim_texts: list[str], references_dir: Path) -> None:
    """Encode claim query texts and store vectors in workspace.db.

    Called from the embedding subprocess (Stage 4b) while the model is
    already loaded. The vectors are used by ``retrieve_similar()`` in
    Stage 5 so it never needs to load the model in the parent process.
    """
    import hashlib

    from sciwrite_lint.references.reference_store import (
        _get_embedding_config,
        _get_embedding_model,
    )
    from sciwrite_lint.references.workspace_db import (
        get_db,
        save_query_vector,
        serialize_f32,
    )

    model_name, _, _ = _get_embedding_config()
    model = _get_embedding_model()
    vecs = model.encode(claim_texts, normalize_embeddings=True)

    with get_db(references_dir) as conn:
        for text, vec in zip(claim_texts, vecs):
            h = hashlib.sha256(text.encode()).hexdigest()
            save_query_vector(conn, h, model_name, serialize_f32(vec.tolist()))

    logger.info("Pre-computed {} claim query vectors", len(claim_texts))


def _batch_cited_vision_entry(manifest_path_str: str) -> None:
    """Subprocess entry point: run VL inference on cited papers for multiple papers.

    Loads the VL model once and iterates over papers. Each paper's cited
    PDFs are processed and results cached in workspace.db. Called by
    ``_batch_cited_vision()`` via ``subprocess.run``.

    Manifest JSON: list of {"references_dir": "...", "fresh": bool}.
    """
    import json

    manifest = json.loads(Path(manifest_path_str).read_text(encoding="utf-8"))

    for entry in manifest:
        references_dir = Path(entry["references_dir"])
        fresh = entry.get("fresh", False)
        backend = entry.get("backend", "transformers")

        parsed_dir = references_dir / "parsed"
        if not parsed_dir.exists():
            continue

        keys = [f.stem for f in sorted(parsed_dir.glob("*.md"))]
        if not keys:
            continue

        from sciwrite_lint.vision.describe import describe_figures_by_source
        from sciwrite_lint.vision.image_extraction import collect_cited_images

        all_images, ref_image_ranges = collect_cited_images(keys, references_dir)
        if all_images:
            describe_figures_by_source(
                all_images,
                ref_image_ranges,
                references_dir,
                fresh=fresh,
                backend=backend,
            )


# ---------------------------------------------------------------------------
# Batch launchers — spawn ONE subprocess for all papers in a stage
# ---------------------------------------------------------------------------


def _batch_vision(
    papers: list[dict[str, Any]],
    timeout: int = 600,
) -> None:
    """Run vision for all papers in one subprocess (single model load).

    Args:
        papers: List of dicts with keys: paper_name, tex_path, config_path, fresh.
        timeout: Subprocess timeout in seconds.
    """
    import json
    import subprocess
    import sys
    import tempfile

    if not papers:
        return

    manifest = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(papers, manifest)
        manifest.close()
        cmd = [
            sys.executable,
            "-m",
            "sciwrite_lint.vision.pipeline",
            "--batch",
            manifest.name,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.error(
                "Batch vision subprocess failed (exit {}): {}\n"
                "Manuscript figures will be missing from full-paper LLM "
                "checks for all papers in this batch — checks will run with "
                "reduced visual context.",
                result.returncode,
                stderr,
            )
    except subprocess.TimeoutExpired:
        logger.error(
            "Batch vision subprocess timed out ({}s) — manuscript figures "
            "missing for batch",
            timeout,
        )
    except Exception as e:
        logger.error("Batch vision subprocess error: {}: {}", type(e).__name__, e)
    finally:
        Path(manifest.name).unlink(missing_ok=True)


def _batch_embed(
    papers: list[dict[str, Any]],
    timeout: int | None = None,
) -> str:
    """Run embedding for all papers in one subprocess (single model load).

    Args:
        papers: List of dicts with keys: keys (list[str]), references_dir (str).
        timeout: Subprocess timeout in seconds. When ``None`` (default),
            scales with total input size via ``_compute_embed_timeout`` so
            book-length references don't trip the floor.

    Returns:
        Empty string on success. On failure, returns a human-readable
        diagnostic (subprocess stderr tail, timeout notice, or exception
        message) for inclusion in any per-paper error surfaced by the
        post-embed has_embeddings() re-check in ``run_papers_staged``.
    """
    import json
    import subprocess
    import sys
    import tempfile

    if not papers:
        return ""

    # Filter out papers with no keys to embed
    papers = [p for p in papers if p.get("keys")]
    if not papers:
        return ""

    if timeout is None:
        all_paths: list[Path] = []
        for p in papers:
            all_paths.extend(
                _iter_embed_text_paths(p.get("keys", []), Path(p["references_dir"]))
            )
        timeout = _compute_embed_timeout(all_paths)

    manifest = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(papers, manifest)
        manifest.close()
        cmd = [
            sys.executable,
            "-c",
            "from sciwrite_lint.pipeline import _batch_embed_entry; "
            f"_batch_embed_entry({manifest.name!r})",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Batch embedding subprocess timed out ({}s)", timeout)
            return f"subprocess timed out after {timeout}s"
        except Exception as e:
            logger.warning("Batch embedding subprocess error: {}", e)
            return f"{type(e).__name__}: {e}"
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.warning("Batch embedding subprocess failed: {}", stderr)
            return f"non-zero exit {result.returncode}: {stderr}"
        return ""
    finally:
        Path(manifest.name).unlink(missing_ok=True)


def _batch_cited_vision(
    papers: list[dict[str, Any]],
    timeout: int = 600,
) -> None:
    """Run cited-paper vision for all papers in one subprocess (single model load).

    Args:
        papers: List of dicts with keys: references_dir (str), fresh (bool).
        timeout: Subprocess timeout in seconds.
    """
    import json
    import subprocess
    import sys
    import tempfile

    if not papers:
        return

    manifest = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(papers, manifest)
        manifest.close()
        cmd = [
            sys.executable,
            "-c",
            "from sciwrite_lint.pipeline import _batch_cited_vision_entry; "
            f"_batch_cited_vision_entry({manifest.name!r})",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.error(
                "Batch cited vision subprocess failed (exit {}): {}\n"
                "Cited paper figures will be missing for all papers in this "
                "batch — ref-internal checks will run with reduced context.",
                result.returncode,
                stderr,
            )
    except subprocess.TimeoutExpired:
        logger.error(
            "Batch cited vision subprocess timed out ({}s) — cited figures "
            "missing for batch",
            timeout,
        )
    except Exception as e:
        logger.error("Batch cited vision subprocess error: {}: {}", type(e).__name__, e)
    finally:
        Path(manifest.name).unlink(missing_ok=True)
