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

import subprocess
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
) -> str:
    """Run embedding computation in a subprocess for CUDA isolation.

    The embedding model brings batch data to VRAM; subprocess isolation
    ensures all CUDA allocations are released when embedding finishes.
    Claim query vectors are also pre-computed in the same subprocess —
    contexts come from ``workspace.db.manuscript_citations``, so the
    parent does not marshal claim text across the process boundary.

    Returns:
        Empty string on success. On failure (non-zero exit, timeout, or
        crash), returns a human-readable diagnostic (subprocess stderr
        tail, timeout notice, or exception message) for inclusion in the
        RuntimeError raised by ``_verify_embeddings_or_raise``.
    """
    import subprocess
    import sys

    keys_str = ",".join(keys)
    cmd = [
        sys.executable,
        "-c",
        "from sciwrite_lint.pipeline import _embed_keys; "
        f"_embed_keys({keys_str!r}, {str(references_dir)!r})",
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


def _embed_keys(keys_csv: str, references_dir_str: str) -> None:
    """Subprocess entry point: compute embeddings for given keys + claim queries.

    Reference texts come from ``parsed/{key}.md`` (resolved via
    ``_resolve_embed_text_paths``). Claim contexts come from the
    ``manuscript_citations`` table — the parent must have populated it
    via ``persist_inline_citations`` before invoking the subprocess.
    """
    from sciwrite_lint.references.reference_store import (
        compute_and_store_embeddings,
        release_embedding_model,
    )
    from sciwrite_lint.references._embed_timing import (
        log_summary as _log_timing_summary,
        reset as _reset_timing,
        time_phase,
    )

    _reset_timing()

    references_dir = Path(references_dir_str)
    keys = keys_csv.split(",") if keys_csv else []

    paths = _resolve_embed_text_paths(keys, references_dir)
    keys_processed = 0
    for key in keys:
        text_path = paths.get(key)
        if text_path is None:
            continue
        try:
            with time_phase("read_text"):
                text = text_path.read_text(encoding="utf-8")
            compute_and_store_embeddings(key, text, references_dir)
            keys_processed += 1
        except Exception as e:
            logger.debug(f"embedding skipped for {key} ({type(e).__name__}: {e})")
            continue

    _encode_missing_claim_queries(references_dir)

    _log_timing_summary(keys_processed)
    release_embedding_model()


def _embed_keys_via_stdin() -> None:
    """Subprocess entry point: pre-warm embedder, then receive keys via stdin.

    Loads the embedder model into VRAM immediately, then blocks reading
    one JSON line from stdin: ``{"keys_csv": "a,b,c", "references_dir": "..."}``.
    Once received, delegates to ``_embed_keys`` with the same semantics
    as the direct one-shot entry point.

    Used by ``EmbedderWarmer`` to overlap the embedder's model-load
    latency (~3-5 s) with GROBID parse work — the parent kicks off this
    subprocess at parse start and submits the keys list when parse
    finishes.

    Logs to stderr; the parent inherits both pipes and surfaces stderr
    in the failure path of ``submit_and_wait``.
    """
    import json
    import sys as _sys

    from sciwrite_lint.references.reference_store import _get_embedding_model

    # Eagerly load model into VRAM. The bf16 cast + budget check inside
    # _get_embedding_model fires here, BEFORE the parent has handed us
    # any work — so any model-related warning/error surfaces during the
    # parse stage rather than at embed time.
    _get_embedding_model()

    line = _sys.stdin.readline()
    if not line:
        logger.warning(
            "embed pre-warm subprocess: stdin closed without payload; "
            "exiting without doing work"
        )
        return
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as e:
        logger.error(f"embed pre-warm subprocess: bad stdin JSON: {e}")
        raise
    _embed_keys(payload["keys_csv"], payload["references_dir"])


class EmbedderWarmer:
    """Long-running embedder subprocess that loads its model in parallel.

    Lifecycle:

    1. ``start()`` — launches the subprocess via ``_embed_keys_via_stdin``.
       The subprocess loads the embedder model into VRAM (~3-5 s) and
       then blocks on stdin. The parent returns immediately so its work
       (typically the GROBID parse stage) overlaps with model load.
    2. ``submit_and_wait(keys)`` — sends the keys CSV + references_dir
       JSON line over stdin, then blocks until the subprocess completes
       embedding. Returns the same diagnostic string contract as
       ``_run_embeddings_subprocess``: ``""`` on success, a
       human-readable error message otherwise.
    3. ``cancel()`` — kills the subprocess without sending work; called
       from the failure path so VRAM is released even if parse raises.

    Caller is responsible for vLLM swap orchestration. The warmer assumes
    text vLLM has already been stopped (peak VRAM during model load is
    ~3 GB; without the swap it would brush the VRAM ceiling).
    """

    def __init__(self, references_dir: Path, config: LintConfig) -> None:
        self.references_dir = references_dir
        self.config = config
        self.proc: "subprocess.Popen[str] | None" = None

    def start(self) -> None:
        """Launch the subprocess; it loads the model and blocks on stdin."""
        import sys

        cmd = [
            sys.executable,
            "-c",
            "from sciwrite_lint.pipeline import _embed_keys_via_stdin; "
            "_embed_keys_via_stdin()",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.config.project_dir) if self.config.project_dir else None,
        )
        logger.info(f"Embedder pre-warm subprocess started (pid={self.proc.pid})")

    def submit_and_wait(self, keys: list[str]) -> str:
        """Send keys to the warming subprocess and wait for completion.

        Returns ``""`` on success or a diagnostic on failure (subprocess
        non-zero exit, timeout, or pre-submit pipe error).
        """
        import json

        if self.proc is None:
            return "EmbedderWarmer.start() was not called"

        keys_csv = ",".join(keys)
        timeout = _compute_embed_timeout(
            _iter_embed_text_paths(keys, self.references_dir)
        )
        payload = json.dumps(
            {
                "keys_csv": keys_csv,
                "references_dir": str(self.references_dir),
            }
        )

        if self.proc.stdin is None:
            return "EmbedderWarmer subprocess has no stdin"
        try:
            self.proc.stdin.write(payload + "\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except (BrokenPipeError, OSError) as e:
            stderr_tail = ""
            if self.proc.stderr is not None:
                stderr_tail = self.proc.stderr.read()[-1500:]
            return f"failed to send keys to subprocess ({e}); stderr: {stderr_tail}"

        try:
            _stdout, stderr = self.proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return f"subprocess timed out after {timeout}s"
        if self.proc.returncode != 0:
            tail = stderr.strip()[-1500:] if stderr else ""
            logger.warning("Embedding subprocess failed: {}", tail)
            return f"non-zero exit {self.proc.returncode}: {tail}"
        return ""

    def cancel(self) -> None:
        """Kill the subprocess without sending work; idempotent."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired as e:
                logger.debug(f"Embedding subprocess wait timed out after kill: {e}")
            except OSError as e:
                logger.debug(f"Embedding subprocess wait raised OSError: {e}")


def _batch_embed_entry(manifest_path_str: str) -> None:
    """Subprocess entry point: embed keys for multiple papers in one process.

    Loads the embedding model once and iterates over papers. Each paper's
    keys are embedded and stored in its own workspace. Claim query
    vectors are pre-computed by reading each paper's
    ``manuscript_citations`` table — the parent must have populated it
    via ``persist_inline_citations`` before this subprocess runs.

    Manifest JSON: list of {"keys": [...], "references_dir": "..."}.
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
            except Exception as e:
                logger.debug(f"embedding skipped for {key} ({type(e).__name__}: {e})")
                continue

        _encode_missing_claim_queries(references_dir)

    release_embedding_model()


def _encode_missing_claim_queries(references_dir: Path) -> None:
    """Encode any persisted citation contexts that lack a query vector.

    Reads the missing-context set from ``manuscript_citations`` (via
    ``find_unembedded_contexts``) and encodes only that subset. Called
    from the embedding subprocess while the model is already loaded.
    """
    import hashlib

    from sciwrite_lint.references.reference_store import (
        _get_embedding_config,
        _get_embedding_model,
    )
    from sciwrite_lint.references.workspace_db import (
        find_unembedded_contexts,
        get_db,
        save_query_vector,
        serialize_f32,
    )

    model_name, _, _ = _get_embedding_config()
    with get_db(references_dir) as conn:
        missing = find_unembedded_contexts(conn, model_name)
    if not missing:
        return

    model = _get_embedding_model()
    vecs = model.encode(missing, normalize_embeddings=True)

    with get_db(references_dir) as conn:
        for text, vec in zip(missing, vecs):
            h = hashlib.sha256(text.encode()).hexdigest()
            save_query_vector(conn, h, model_name, serialize_f32(vec.tolist()))

    logger.info("Pre-computed {} claim query vectors", len(missing))


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
