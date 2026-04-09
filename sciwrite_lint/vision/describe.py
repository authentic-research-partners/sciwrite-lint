"""Describe figure images using Qwen3-VL-2B-Instruct.

Loads the vision-language model, runs batched inference on extracted images,
and returns structured text descriptions suitable for injection into the
full-paper consistency check system prompt.

On WSL2, CUDA memory overcommit lets the VL model (~4 GB float16) share
VRAM with vLLM transparently — idle KV-cache pages swap to system RAM.
No container stop needed; same pattern as the embedding model.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from sciwrite_lint.vision.image_extraction import ExtractedImage

_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
_MAX_NEW_TOKENS = 512  # Structured description, not free-form
_MAX_IMAGE_DIM = 1024  # Resize longest side (benchmarked: no accuracy gain above this)
_BATCH_SIZE = 16  # Benchmarked sweet spot: 2.8s/img, 6.8 GB VRAM


def _resolve_vision_device(device_cfg: str) -> str:
    """Resolve device string: "auto" picks CUDA on WSL2, else CPU.

    On WSL2, CUDA memory overcommit lets the VL model (~4 GB float16) share
    VRAM with vLLM — idle KV-cache pages swap to system RAM while VL
    inference runs.  On native Linux, cudaMalloc is physical with no
    overcommit, so GPU vision is not safe in auto mode (use "cuda" to force).
    """
    from sciwrite_lint.config import is_wsl2

    if device_cfg == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        if is_wsl2():
            return "cuda"
        return "cpu"
    return device_cfg


_DESCRIBE_ITEMS = """\
1. Figure type (bar chart, line plot, scatter plot, table, diagram, photo, etc.)
2. All axis labels and units
3. All data series, categories, or groups
4. Key numeric values you can read (peaks, intersections, notable data points)
5. All text labels, legends, and annotations within the figure
6. Overall trend or pattern shown
7. Readability issues: overlapping labels, obscured text, truncated axes, \
illegible values. For anything you cannot read clearly, say so explicitly \
(e.g., "y-axis label partially obscured, cannot determine units")\
"""

_DESCRIBE_PROMPT_BASE = f"""\
You are analyzing a figure from a scientific paper. Describe precisely:
{_DESCRIBE_ITEMS}

Be factual and precise. Report only what is visible in the image. \
Never guess values you cannot read — say they are unreadable.\
"""

_DESCRIBE_PROMPT_WITH_CAPTION = (
    """\
You are analyzing a figure from a scientific paper.

The figure's caption reads: "{caption}"

Describe precisely what the figure ACTUALLY shows:
"""
    + _DESCRIBE_ITEMS
    + """

Be factual and precise. Report only what is visible in the image. \
Never guess values you cannot read — say they are unreadable. \
If the caption misrepresents the content, describe what you see, not what \
the caption claims.\
"""
)


def _build_prompt(caption: str) -> str:
    """Build the VL prompt, including the caption if available."""
    if caption:
        return _DESCRIBE_PROMPT_WITH_CAPTION.format(caption=caption)
    return _DESCRIBE_PROMPT_BASE


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(device: str) -> tuple[Any, Any]:
    """Load Qwen3-VL-2B model and processor."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.float32 if device == "cpu" else torch.float16

    logger.info(
        "Loading vision model: {} (device={}, dtype={})", _MODEL_NAME, device, dtype
    )
    processor = AutoProcessor.from_pretrained(_MODEL_NAME)
    # Left-padding required for correct batched generation with decoder-only models
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        _MODEL_NAME,
        device_map=device,
        torch_dtype=dtype,
    )
    logger.info("Vision model loaded")
    return model, processor


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def _resize_image(image: Any, max_dim: int = _MAX_IMAGE_DIM) -> Any:
    """Resize image so longest side is at most max_dim. Preserves aspect ratio."""
    from PIL import Image

    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    ratio = max_dim / max(w, h)
    return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------


def _describe_images_batch(
    image_paths: list[Path],
    prompts: list[str],
    model: Any,
    processor: Any,
    device: str,
) -> list[str]:
    """Run VL model on a batch of images, return descriptions.

    Each image gets its own prompt (may include the figure's caption).
    """
    from PIL import Image

    images = [_resize_image(Image.open(p).convert("RGB")) for p in image_paths]

    # Build per-image messages and text inputs
    all_texts: list[str] = []
    for p, prompt in zip(image_paths, prompts):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(p)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_texts.append(text_input)

    inputs = processor(
        text=all_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)

    # Decode each sequence, skipping input tokens
    input_len = inputs["input_ids"].shape[1]
    descriptions: list[str] = []
    for i in range(len(image_paths)):
        generated = output_ids[i][input_len:]
        desc = processor.decode(generated, skip_special_tokens=True).strip()
        descriptions.append(desc)

    return descriptions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def describe_figures(
    extracted_images: list[ExtractedImage],
    references_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = _BATCH_SIZE,
    fresh: bool = False,
) -> str:
    """Describe all extracted figures and return formatted text for the LLM.

    Uses workspace.db ``vision_cache`` table so only new or changed images
    trigger VL inference.

    Args:
        extracted_images: Images from ``extract_images_from_latex`` or
            ``extract_images_from_pdf``.
        references_dir: Paper workspace root (``references/{paper}/``) for
            DB caching via ``get_db()``.  If None, no caching.
        device: ``"auto"`` (CUDA on WSL2, CPU elsewhere), ``"cpu"``, or ``"cuda"``.
        batch_size: Images per batch for VL inference.
        fresh: Ignore cache and re-describe all images.

    Returns:
        Formatted string ready for injection into the full-paper consistency
        check system prompt (the ``figure_descriptions`` parameter).
    """
    from sciwrite_lint.vision.cache import (
        clear_cache,
        format_descriptions_from_db,
        split_cached_and_new,
        update_cache,
    )

    if not extracted_images:
        return ""

    # Clear cache if --fresh
    if fresh and references_dir:
        clear_cache(references_dir)

    # Determine which images need inference
    if references_dir and not fresh:
        new_images = split_cached_and_new(extracted_images, references_dir)
    else:
        new_images = list(extracted_images)

    if not new_images:
        logger.info(
            "All {} figure(s) cached, skipping VL inference", len(extracted_images)
        )
        return format_descriptions_from_db(extracted_images, references_dir)  # type: ignore[arg-type]

    logger.info(
        "{}/{} figure(s) need VL inference",
        len(new_images),
        len(extracted_images),
    )

    # Resolve device (same pattern as embedding model: CUDA on WSL2, CPU elsewhere)
    device = _resolve_vision_device(device)

    logger.info("Running VL inference on {} (batch_size={})", device, batch_size)

    model, processor = _load_model(device)

    try:
        # Process in batches — each image gets a prompt with its caption
        all_descriptions: list[str] = []
        paths = [img.path for img in new_images]
        prompts = [_build_prompt(img.caption) for img in new_images]
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]
            descs = _describe_images_batch(
                batch_paths, batch_prompts, model, processor, device
            )
            all_descriptions.extend(descs)
    finally:
        # Free GPU memory immediately — gc.collect() breaks circular refs
        # before empty_cache() so CUDA can reclaim all allocations.
        del model
        del processor
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save to workspace.db
    if references_dir:
        update_cache(new_images, all_descriptions, references_dir)

    # Format all descriptions (cached + newly inferred)
    if references_dir:
        result = format_descriptions_from_db(extracted_images, references_dir)
    else:
        # No DB — format from what we just inferred
        parts: list[str] = []
        for img, desc in zip(new_images, all_descriptions):
            header = "Figure"
            if img.label:
                header += f" ({img.label})"
            if img.caption:
                header += f' — Caption: "{img.caption}"'
            parts.append(f"{header}\nVisual content: {desc}")
        result = "\n\n".join(parts)

    logger.info(
        "Generated {} figure descriptions (~{} chars)",
        len(extracted_images),
        len(result),
    )
    return result
