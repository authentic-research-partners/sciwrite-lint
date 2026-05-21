"""vLLM container swap + GPU probes used by GPU-bound stages.

Vision and embedding stages share one GPU with text vLLM. We stop text vLLM
before GPU embedding and restart after — on both native Linux (where the
alternative is a hard CUDA OOM) and WSL2 (where the alternative is a silent
spill to system RAM that runs 30–100× slower). For the vision backend we
swap text↔vision vLLM containers, but only when there is actual inference
work to do — cached reruns skip the swap entirely.

GPU exclusivity invariant. Only one heavy GPU process runs at a time:
text vLLM, vision vLLM, or an in-process model (embedder /
transformers vision). The single entry point ``claim_gpu_exclusive``
enforces this — every stage that touches the GPU calls it before
starting work. The function stops every competitor BEFORE deciding
whether to start the target, so it is correct regardless of what the
caller pre-started (including the foot-gun case of
``containers start --vision`` having both up at once).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from loguru import logger

from sciwrite_lint.config import LintConfig, PaperWorkspace

GPUTarget = Literal["text", "vision", "none"]


def _needs_embedding_swap(config: LintConfig) -> bool:
    """Check if we need to stop text vLLM for GPU embedding.

    Embedder + vLLM together overflow VRAM on a 20 GB consumer GPU. On
    native Linux that fails outright (no CUDA overcommit). On WSL2 the
    overflow silently spills to shared GPU memory (system RAM mapped
    over PCIe), which works but runs 30–100× slower — KV cache becomes
    underused and gen tok/s collapses by an order of magnitude. Either
    way the right move is to stop text vLLM before GPU embedding.
    """
    if config.embedding_device == "cpu":
        return False
    return torch.cuda.is_available()


def _is_vllm_responding(endpoint: str) -> bool:
    """Sync check if vLLM API is responding (for use in sync contexts)."""
    try:
        import httpx

        resp = httpx.get(f"{endpoint}/models", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def _stop_vllm_for_embedding(config: LintConfig) -> None:
    """Stop both vLLM containers to free GPU for embedding.

    Embedder is an in-process sentence-transformers model that loads
    onto the GPU. Either vLLM container competing for VRAM at that
    moment causes a hard CUDA OOM on native Linux, or a silent spill
    to system RAM on WSL2 (30–100× slower; see
    ``_needs_embedding_swap``).

    We stop BOTH containers (not just text) so the foot-gun of having
    vision pre-started by the user is also handled — see
    ``claim_gpu_exclusive`` for the same invariant.
    """
    from sciwrite_lint.vllm.vllm_server import MODELS, stop_container

    vision_port = MODELS["qwen3-vl"]["port"]
    vision_endpoint = f"http://localhost:{vision_port}/v1"

    if _is_vllm_responding(config.llm_endpoint):
        logger.info("Stopping text vLLM for GPU embedding...")
        stop_container(config, model=config.llm_model)
    if _is_vllm_responding(vision_endpoint):
        logger.info("Stopping vision vLLM for GPU embedding...")
        stop_container(config, model="qwen3-vl")


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


async def claim_gpu_exclusive(target: GPUTarget, config: LintConfig) -> None:
    """Single defensive entry point — ensure only ``target`` is running.

    Heavy GPU users tracked:
    - text vLLM container (port 5001)
    - vision vLLM container (port 5002)
    - in-process models (embedder, transformers vision) — caller is
      responsible for unloading these before calling this with
      ``target="none"``; this function only stops the vLLM containers
      it knows about.

    Behavior:
    - ``target="text"``: stop vision if up, ensure text is up.
    - ``target="vision"``: stop text if up, ensure vision is up.
    - ``target="none"``: stop both vLLMs (e.g. before GPU embedding
      or transformers vision inference).

    The "stop competitors" pass runs **before** the "is target already
    up?" check, so the function is correct even when both containers
    are running unsupervised (the foot-gun case of
    ``containers start --vision``). Idempotent — safe to call
    repeatedly.

    Skips silently if no container runtime is available (test
    environments).
    """
    from sciwrite_lint.vllm.vllm_server import (
        MODELS,
        _check_api_health,
        start_container,
        stop_container,
        wait_for_ready,
    )

    text_endpoint = config.llm_endpoint
    vision_port = MODELS["qwen3-vl"]["port"]
    vision_endpoint = f"http://localhost:{vision_port}/v1"

    text_up = await _check_api_health(text_endpoint)
    vision_up = await _check_api_health(vision_endpoint)

    # Stop every competitor BEFORE deciding to start anything. Doing
    # this unconditionally (rather than "if target is not already up")
    # is what closes the foot-gun: we never leave a non-target container
    # running just because the target is also already up.
    if target != "text" and text_up:
        logger.info("Stopping text vLLM (claiming GPU for {})", target)
        stop_container(config, model=config.llm_model)
    if target != "vision" and vision_up:
        logger.info("Stopping vision vLLM (claiming GPU for {})", target)
        stop_container(config, model="qwen3-vl")

    if target == "none":
        return

    # Start target if not already up
    if target == "text":
        if text_up:
            logger.info("Text vLLM already running")
            return
        logger.info("Starting text vLLM ({})...", config.llm_model)
        ret = start_container(config, model=config.llm_model)
        if ret != 0:
            raise RuntimeError(
                "Failed to start text vLLM container. "
                "Check: sciwrite-lint containers start"
            )
        ready = await wait_for_ready(text_endpoint, timeout=300)
        if not ready:
            raise RuntimeError(
                f"Text vLLM did not become ready within 300s at "
                f"{text_endpoint}. Check logs: sciwrite-lint vllm logs"
            )
        logger.info("Text vLLM ready")
        return

    # target == "vision"
    if vision_up:
        logger.info("Vision vLLM already running on port {}", vision_port)
        return
    logger.info("Starting vision vLLM (qwen3-vl)...")
    ret = start_container(config, model="qwen3-vl")
    if ret != 0:
        raise RuntimeError(
            "Failed to start vision vLLM container. "
            "Check: sciwrite-lint vllm start --model qwen3-vl"
        )
    ready = await wait_for_ready(vision_endpoint, timeout=300)
    if not ready:
        raise RuntimeError(
            f"Vision vLLM did not become ready within 300s on port "
            f"{vision_port}. Check logs: sciwrite-lint vllm logs --model qwen3-vl"
        )
    logger.info("Vision vLLM ready")


async def _swap_to_vision_vllm(config: LintConfig) -> None:
    """Claim exclusive GPU access for the vision vLLM (stops competitors first)."""
    await claim_gpu_exclusive("vision", config)


async def _swap_to_text_vllm(config: LintConfig) -> None:
    """Claim exclusive GPU access for the text vLLM (stops competitors first)."""
    await claim_gpu_exclusive("text", config)
