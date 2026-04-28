"""vLLM container swap + GPU probes used by GPU-bound stages.

Vision and embedding stages share one GPU with text vLLM. On native Linux
(no CUDA overcommit) we stop text vLLM before GPU embedding and restart
after. For the vision backend we swap text↔vision vLLM containers, but
only when there is actual inference work to do — cached reruns skip the
swap entirely.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig, PaperWorkspace


def _needs_embedding_swap(config: LintConfig) -> bool:
    """Check if we need to stop text vLLM for GPU embedding.

    On native Linux (no CUDA overcommit), embedding can't coexist with
    vLLM on the GPU. We stop vLLM first, embed on GPU, then restart.
    On WSL2, overcommit handles coexistence — no swap needed.
    """
    from sciwrite_lint.config import is_wsl2

    if is_wsl2():
        return False
    if config.embedding_device == "cpu":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _is_vllm_responding(endpoint: str) -> bool:
    """Sync check if vLLM API is responding (for use in sync contexts)."""
    try:
        import httpx

        resp = httpx.get(f"{endpoint}/models", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def _stop_vllm_for_embedding(config: LintConfig) -> None:
    """Stop text vLLM to free GPU for embedding on native Linux.

    Only stops if text vLLM is actually running.
    """
    from sciwrite_lint.vllm.vllm_server import stop_container

    if not _is_vllm_responding(config.llm_endpoint):
        return
    logger.info("Stopping text vLLM for GPU embedding (native Linux, no overcommit)...")
    stop_container(config, model=config.llm_model)


def _restart_vllm_after_embedding(config: LintConfig) -> None:
    """Restart text vLLM after embedding completes."""
    import time as _time

    from sciwrite_lint.vllm.vllm_server import start_container

    logger.info("Restarting text vLLM after embedding...")
    ret = start_container(config, model=config.llm_model)
    if ret != 0:
        raise RuntimeError(
            "Failed to restart text vLLM after embedding. "
            "Check: sciwrite-lint containers start"
        )

    # Sync poll — we're in a sync context (called from _run_embeddings_for_paper)
    import httpx

    deadline = _time.monotonic() + 300
    while _time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{config.llm_endpoint}/models", timeout=5.0)
            if resp.status_code == 200:
                logger.info("Text vLLM ready")
                return
        except httpx.HTTPError:
            pass
        _time.sleep(5)
    raise RuntimeError(
        f"Text vLLM did not become ready within 300s at {config.llm_endpoint}. "
        "Check logs: sciwrite-lint vllm logs"
    )


def _manuscript_needs_inference(
    tex_path: Path,
    refs_dir: Path,
    ws: PaperWorkspace,
    fresh: bool = False,
) -> bool:
    """True if the manuscript has any figure that will require VL inference.

    Mirrors exactly what ``_stage_vision`` processes: raster
    ``\\includegraphics`` figures via ``extract_images_from_latex`` plus
    TikZ/vector figures rendered from the compiled PDF via
    ``render_tikz_figures``. Both go through ``split_cached_and_new`` so
    a rerun with an unchanged manuscript skips the vision-vLLM swap.

    Returns False when every figure already has a matching vision_cache
    entry (or there are no figures, or TikZ figures exist but the .tex
    hasn't been compiled yet — nothing to render). ``fresh=True`` always
    returns True because the subprocess will clear the cache.
    """
    from sciwrite_lint.vision.cache import split_cached_and_new
    from sciwrite_lint.vision.image_extraction import (
        _find_compiled_pdf,
        extract_images_from_latex,
        extract_images_from_pdf,
        render_tikz_figures,
    )

    if tex_path.suffix.lower() == ".pdf":
        out_dir = ws.parsed / "extracted_images"
        probe_imgs = extract_images_from_pdf(tex_path, out_dir)
        if not probe_imgs:
            return False
        if fresh:
            return True
        return bool(split_cached_and_new(probe_imgs, refs_dir))

    ws_tex = ws.source / tex_path.name
    effective = ws_tex if ws_tex.is_file() else tex_path
    probe_imgs = extract_images_from_latex(effective)

    pdf_path = _find_compiled_pdf(tex_path)
    if pdf_path is not None:
        rendered = render_tikz_figures(
            effective, ws.parsed / "rendered_figures", pdf_path=pdf_path
        )
        probe_imgs.extend(rendered)

    if not probe_imgs:
        return False
    if fresh:
        return True
    return bool(split_cached_and_new(probe_imgs, refs_dir))


def _cited_needs_inference(refs_dir: Path, fresh: bool = False) -> bool:
    """True if any cited-paper PDF has an uncached (or ``fresh``) figure.

    Mirrors ``_stage_cited_vision``'s work discovery AND caching scheme:
    for each parsed ref key, find ``{key}*.pdf`` in ``refs_dir``, extract
    images, and check the vision_cache under ``source=<ref_key>`` — cited
    figures are stored per-ref by ``describe_figures_by_source``, not
    under ``"manuscript"``. Returns False once every cited figure has a
    matching cache entry so reruns skip the vision-vLLM swap.
    """
    from sciwrite_lint.vision.cache import split_cached_and_new
    from sciwrite_lint.vision.image_extraction import collect_cited_images

    parsed_dir = refs_dir / "parsed"
    if not parsed_dir.exists():
        return False
    keys = [f.stem for f in sorted(parsed_dir.glob("*.md"))]
    if not keys:
        return False
    all_images, ref_image_ranges = collect_cited_images(keys, refs_dir)
    if not all_images:
        return False
    if fresh:
        return True
    for key, (start, end) in ref_image_ranges.items():
        if split_cached_and_new(all_images[start:end], refs_dir, source=key):
            return True
    return False


async def _swap_to_vision_vllm(config: LintConfig) -> None:
    """Stop text vLLM, start vision vLLM, wait for API ready.

    Called before vision stages when ``config.vision_backend == "vllm"``.
    Skips if the vision API is already responding (user pre-started it).
    Skips if no container runtime is available (test environments).
    """
    from sciwrite_lint.vllm.vllm_server import (
        MODELS,
        _check_api_health,
        start_container,
        stop_container,
        wait_for_ready,
    )

    vision_port = MODELS["qwen3-vl"]["port"]
    vision_endpoint = f"http://localhost:{vision_port}/v1"

    # Already running? Skip the swap.
    if await _check_api_health(vision_endpoint):
        logger.info("Vision vLLM already running on port {}", vision_port)
        return

    # Text vLLM not running? Nothing to swap from.
    if not await _check_api_health(config.llm_endpoint):
        logger.debug("Text vLLM not running — skipping vision swap")
        return

    # Stop text vLLM to free GPU
    logger.info("Stopping text vLLM to free GPU for vision...")
    stop_container(config, model=config.llm_model)

    # Start vision vLLM
    logger.info("Starting vision vLLM (qwen3-vl)...")
    ret = start_container(config, model="qwen3-vl")
    if ret != 0:
        raise RuntimeError(
            "Failed to start vision vLLM container. "
            "Check: sciwrite-lint vllm start --model qwen3-vl"
        )

    # Wait for API
    logger.info("Waiting for vision vLLM API on port {}...", vision_port)
    ready = await wait_for_ready(vision_endpoint, timeout=300)
    if not ready:
        raise RuntimeError(
            f"Vision vLLM did not become ready within 300s on port {vision_port}. "
            "Check logs: sciwrite-lint vllm logs --model qwen3-vl"
        )
    logger.info("Vision vLLM ready")


async def _swap_to_text_vllm(config: LintConfig) -> None:
    """Stop vision vLLM, start text vLLM, wait for API ready.

    Called after vision stages to restore text vLLM for Stages 1+2.
    Skips if text API is already responding.
    Skips if no container runtime is available (test environments).
    """
    from sciwrite_lint.vllm.vllm_server import (
        MODELS,
        _check_api_health,
        start_container,
        stop_container,
        wait_for_ready,
    )

    text_endpoint = config.llm_endpoint

    # Always stop vision vLLM if it's running — free VRAM for text stages
    vision_port = MODELS["qwen3-vl"]["port"]
    if await _check_api_health(f"http://localhost:{vision_port}/v1"):
        logger.info("Stopping vision vLLM to free GPU...")
        stop_container(config, model="qwen3-vl")

    # Text already running? Done.
    if await _check_api_health(text_endpoint):
        logger.info("Text vLLM already running")
        return

    # Start text vLLM
    logger.info("Starting text vLLM ({})...", config.llm_model)
    ret = start_container(config, model=config.llm_model)
    if ret != 0:
        raise RuntimeError(
            "Failed to start text vLLM container. Check: sciwrite-lint containers start"
        )

    # Wait for API
    logger.info("Waiting for text vLLM API...")
    ready = await wait_for_ready(text_endpoint, timeout=300)
    if not ready:
        raise RuntimeError(
            f"Text vLLM did not become ready within 300s at {text_endpoint}. "
            "Check logs: sciwrite-lint vllm logs"
        )
    logger.info("Text vLLM ready")
