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

from sciwrite_lint.checks._diagnostics import vision_incomplete_finding
from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Finding


def _parse_vision_stats(stdout: str) -> dict[str, int] | None:
    """Extract retry/soft-fail counters from the cited-vision subprocess.

    The subprocess emits a single ``VISION_RUN_STATS={...json...}`` line
    on stdout after ``asyncio.gather`` completes. Returns ``None`` if no
    marker line is present (older subprocess code, or non-vllm backend
    that doesn't emit it). Defensive: malformed JSON or missing keys
    produce ``None`` rather than crashing the parent.
    """
    import json

    if not stdout:
        return None
    marker = "VISION_RUN_STATS="
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            try:
                payload = json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                return None
            if isinstance(payload, dict):
                return payload
            return None
    return None


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
    per_image_timeout_s: float = 10.0,
) -> tuple[dict[str, str], list[Finding]]:
    """Stage 4.2: Describe figures from cited paper PDFs.

    Always runs in a subprocess:
    - transformers: CUDA context isolation (same reason as _stage_vision)
    - vllm: clean asyncio context (_describe_images_vllm calls asyncio.run,
      which crashes when invoked from inside an already-running event loop)

    ``backend`` is forwarded to the subprocess; the caller must have
    ensured the corresponding service is reachable (vision vLLM on :5002
    for ``"vllm"``) before invocation. This function does not swap
    containers — swap decisions live in the orchestrator.

    ``per_image_timeout_s`` sets the per-image wall-clock budget for the
    subprocess. Total budget is ``60 + N * per_image_timeout_s``, floored
    at 120s to cover model load.

    Results are cached in workspace.db by the subprocess; the parent
    reads cached descriptions from DB afterwards.

    Returns ``(descriptions, findings)``:
    - ``descriptions``: ``{ref_key: figure_descriptions_str}`` for
      injection into ref_internal consistency queries.
    - ``findings``: zero or one ``vision-incomplete`` Finding listing
      cited papers that ended up without descriptions, so the gap is
      visible in the JSON report rather than only in the log.
    """
    import subprocess
    import sys

    from sciwrite_lint.vision.cache import (
        format_descriptions_from_db,
        split_cached_and_new,
    )
    from sciwrite_lint.vision.image_extraction import collect_cited_images

    parsed_dir = references_dir / "parsed"
    if not parsed_dir.exists():
        return {}, []

    keys = [f.stem for f in sorted(parsed_dir.glob("*.md"))]
    if not keys:
        return {}, []

    all_images, ref_image_ranges = collect_cited_images(keys, references_dir)
    if not all_images:
        return {}, []

    # Snapshot cache state *before* the subprocess so the post-run log
    # line can distinguish cache reads from real VL inference. Without
    # this split, observed wall-clock-per-image is unattributable —
    # 7s/image could mean 7s of actual VL work or 0.5s per fresh image
    # diluted by cache hits. ``--fresh`` re-runs everything, so the
    # subprocess will treat every image as new regardless of cache.
    if fresh:
        new_inference_count = len(all_images)
    else:
        new_inference_count = sum(
            len(split_cached_and_new(all_images[start:end], references_dir, source=key))
            for key, (start, end) in ref_image_ranges.items()
        )
    cached_count = len(all_images) - new_inference_count

    # Dynamic timeout: per_image_timeout_s per image (default 10s, sized
    # for vLLM-vision sharing the GPU with text vLLM), plus 60s baseline
    # for model load / container handshake. Floor at 120s for tiny batches.
    timeout = max(120, int(60 + len(all_images) * per_image_timeout_s))

    cmd = [
        sys.executable,
        "-c",
        "from sciwrite_lint.checks.ref_internal_checks import "
        "_describe_cited_figures_vl; "
        f"_describe_cited_figures_vl({str(references_dir)!r}, {fresh!r}, {backend!r})",
    ]
    run_stats: dict[str, int] | None = None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        run_stats = _parse_vision_stats(result.stdout)
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

    # Read cached descriptions from DB (written by subprocess or in-process).
    # When the subprocess crashed mid-run, partial output may exist —
    # surface which papers got descriptions and which were missed so the
    # operator can audit coverage rather than silently running ref-internal
    # checks with reduced visual context.
    descriptions: dict[str, str] = {}
    missing: list[str] = []
    for key, (start, end) in ref_image_ranges.items():
        ref_images = all_images[start:end]
        desc = format_descriptions_from_db(ref_images, references_dir, source=key)
        if desc:
            descriptions[key] = desc
        else:
            missing.append(key)

    if descriptions:
        logger.info(
            "Cited paper figures: {}/{} papers described, {} images total "
            "({} cache hits, {} new inference)",
            len(descriptions),
            len(ref_image_ranges),
            len(all_images),
            cached_count,
            new_inference_count,
        )
        if run_stats is not None and new_inference_count > 0:
            # Attribute slow runs: a high ``retried`` count points at vLLM
            # saturation backpressure, ``soft_failed`` at requests that
            # gave up entirely (never reach the cache and so re-run on
            # the next pipeline invocation).
            logger.info(
                "Cited vision retries: {}/{} succeeded ({} on retry), {} soft-failed",
                run_stats.get("succeeded", 0),
                run_stats.get("total_images", new_inference_count),
                run_stats.get("retried", 0),
                run_stats.get("soft_failed", 0),
            )
    findings: list[Finding] = []
    if missing:
        # Log at WARNING and emit a finding so the gap is visible in the
        # JSON report — covers both subprocess crash and silent per-image
        # VL failures.
        logger.warning(
            "Cited paper figures: {} ref(s) missing descriptions "
            "(ref-internal checks will run with reduced visual context): {}",
            len(missing),
            ", ".join(missing),
        )
        findings.append(vision_incomplete_finding(missing))

    return descriptions, findings
