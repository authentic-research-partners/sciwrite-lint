"""Vision stages — manuscript figures (Stage 0.5) and cited paper figures (Stage 4.2).

Both stages always run in a subprocess:
- transformers backend: CUDA context isolation (the context persists in
  the parent after del + gc + empty_cache and steals VRAM from vLLM)
- vllm backend: clean asyncio context (``_describe_images_vllm`` calls
  ``asyncio.run`` which crashes when invoked from an already-running loop)

Results are written to ``workspace.db`` (vision_cache table); the parent
reads cached descriptions back from DB.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig


def _stage_vision(
    tex_path: Path,
    config: LintConfig,
    paper_name: str,
    fresh: bool = False,
) -> None:
    """Extract and describe manuscript figures via VL model.

    Backend is selected by ``config.vision_backend``:
    - ``"transformers"``: Qwen3-VL-2B in subprocess (default)
    - ``"vllm"``: Qwen3-VL-8B-FP8 via vLLM container on port 5002

    Populates the vision cache so that full-paper consistency checks
    (Stage 2) can include figure descriptions in the shared prefix.

    Runs in a subprocess so the CUDA context (~500 MB) is fully released
    when the VL model finishes. Without subprocess isolation, the CUDA
    runtime context persists in the parent process even after del model +
    gc.collect() + empty_cache(), stealing VRAM from vLLM's KV cache and
    causing timeouts on LLM queries.

    Results are written to workspace.db (vision_cache table) by the
    subprocess; the parent reads them from DB — no return value transfer.
    """
    import subprocess
    import sys

    # Build a command that runs the vision pipeline in isolation.
    # Uses the same Python interpreter and config.
    cmd = [
        sys.executable,
        "-m",
        "sciwrite_lint.vision.pipeline",
        str(tex_path),
        "--paper",
        paper_name,
    ]
    if fresh:
        cmd.append("--fresh")
    if config.config_path:
        cmd.extend(["--config", str(config.config_path)])
    cmd.extend(["--backend", config.vision_backend])
    cmd.extend(["--device", config.vision_device])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(config.project_dir) if config.project_dir else None,
        )
        if result.returncode == 0:
            logger.info("Vision: figure descriptions ready for full-paper checks")
        else:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.error(
                "Vision subprocess exited with code {}: {}\n"
                "Manuscript figures missing — full-paper checks will run "
                "with reduced visual context.",
                result.returncode,
                stderr,
            )
    except subprocess.TimeoutExpired:
        logger.error(
            "Vision subprocess timed out (300s) — manuscript figures missing, "
            "checks continue with reduced visual context"
        )
    except Exception as e:
        # Vision is best-effort — don't block the pipeline
        logger.error(
            "Vision pipeline failed ({}: {}) — checks continue without figures",
            type(e).__name__,
            e,
        )


def _stage_cited_vision(
    references_dir: Path,
    fresh: bool = False,
    backend: str = "transformers",
) -> dict[str, str]:
    """Stage 4.2: Describe figures from cited paper PDFs.

    Always runs in a subprocess:
    - transformers: CUDA context isolation (same reason as _stage_vision)
    - vllm: clean asyncio context (_describe_images_vllm calls asyncio.run,
      which crashes when invoked from inside an already-running event loop)

    ``backend`` is forwarded to the subprocess; the caller must have
    ensured the corresponding service is reachable (vision vLLM on :5002
    for ``"vllm"``) before invocation. This function does not swap
    containers — swap decisions live in the orchestrator.

    Results are cached in workspace.db by the subprocess; the parent
    reads cached descriptions from DB afterwards.

    Returns {ref_key: figure_descriptions_str} for injection into
    ref_internal consistency queries.
    """
    import subprocess
    import sys

    from sciwrite_lint.vision.cache import format_descriptions_from_db
    from sciwrite_lint.vision.image_extraction import collect_cited_images

    parsed_dir = references_dir / "parsed"
    if not parsed_dir.exists():
        return {}

    keys = [f.stem for f in sorted(parsed_dir.glob("*.md"))]
    if not keys:
        return {}

    all_images, ref_image_ranges = collect_cited_images(keys, references_dir)
    if not all_images:
        return {}

    # Dynamic timeout: ~5s per image (conservative, GPU batch=16 or vLLM
    # concurrent), 60s for model load.
    timeout = max(120, 60 + len(all_images) * 5)

    cmd = [
        sys.executable,
        "-c",
        "from sciwrite_lint.checks.ref_internal_checks import "
        "_describe_cited_figures_vl; "
        f"_describe_cited_figures_vl({str(references_dir)!r}, {fresh!r}, {backend!r})",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.error(
                "Cited vision subprocess failed (exit {}): {}\n"
                "Cited figures missing — ref-internal checks will run with "
                "reduced visual context.",
                result.returncode,
                stderr,
            )
    except subprocess.TimeoutExpired:
        logger.error(
            "Cited vision subprocess timed out ({}s, {} images) — figures missing",
            timeout,
            len(all_images),
        )
    except Exception as e:
        logger.error("Cited vision subprocess error: {}: {}", type(e).__name__, e)

    # Read cached descriptions from DB (written by subprocess or in-process)
    descriptions: dict[str, str] = {}
    for key, (start, end) in ref_image_ranges.items():
        ref_images = all_images[start:end]
        desc = format_descriptions_from_db(ref_images, references_dir, source=key)
        if desc:
            descriptions[key] = desc

    if descriptions:
        logger.info(
            "Cited paper figures: {} papers, {} images described",
            len(descriptions),
            len(all_images),
        )

    return descriptions
