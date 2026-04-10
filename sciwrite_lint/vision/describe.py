"""Describe figure images using a vision-language model.

Two backends:
- **transformers** (default): Qwen3-VL-2B-Instruct loaded in-process via
  transformers. On WSL2, CUDA memory overcommit lets the VL model (~4 GB
  float16) share VRAM with vLLM — no container stop needed.
- **vllm**: Qwen3-VL-8B-Instruct-FP8 served via a dedicated vLLM container
  on port 5002. Higher accuracy (+15% on real-world caption mismatches),
  but requires GPU time-sharing with text vLLM.

"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from sciwrite_lint.llm_utils import _VLLM_RETRIES
from sciwrite_lint.vision.cache import VisionResult
from sciwrite_lint.vision.image_extraction import ExtractedImage

_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
_MAX_NEW_TOKENS_TRANSFORMERS = (
    # 2B free text: budget ~750 words at 1.3 tok/word; prompt caps at
    # 500 words, leaving ~330 tokens of headroom. VRAM-safe at batch=16.
    1024
)
_MAX_NEW_TOKENS_VLLM = (
    # 8B JSON: safety net above the per-field maxLength bounds in
    # VisionResult (figure_type=80, description=4000, readability_issues=600
    # chars ≈ 1170 tokens worst case); 2048 leaves headroom for JSON chrome.
    2048
)
_MAX_IMAGE_DIM = 1024
_BATCH_SIZE = 16


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

# --- Free-text prompts (transformers 2B backend) ---

_DESCRIBE_PROMPT_BASE = f"""\
You are analyzing a figure from a scientific paper. Describe precisely:
{_DESCRIBE_ITEMS}

Be factual and precise. Report only what is visible in the image. \
Never guess values you cannot read — say they are unreadable. \
Stay under 500 words — focus on what matters for verifying the paper's claims.\
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
the caption claims. \
Stay under 500 words — focus on what matters for verifying the paper's claims.\
"""
)

# --- Structured JSON prompts (vLLM 8B backend) ---
# Uses json_schema + strict (see ``_vision_response_format``) so the
# constrained decoder enforces the per-field bounds on ``VisionResult``.
#
# The guidance below mirrors the decoder-enforced caps:
# - ``description``: 4000 chars ≈ 1000 words max → prompt says "under
#   500 words, most important details first" (same length/priority
#   pattern used in cross_section_consistency).
# - ``readability_issues``: list with maxItems=5 and per-item maxLength=150
#   → prompt says "empty list if none, otherwise at most 5 short items,
#   most important first" (same count/priority pattern used in
#   FullPaperIssueList and ConsistencyResult).

_JSON_PROMPT_BASE = """\
You are analyzing a figure from a scientific paper. Your description will \
be used to verify that the paper's caption and text accurately describe \
this figure.

Respond with a JSON object:
{
  "figure_type": "bar chart | line plot | scatter plot | table | diagram | photo | other",
  "description": "Detailed description: axes with units, all data series/curves \
with their labels, key numeric values you can read, trends and patterns. \
Include enough detail to compare against what the paper claims about this figure.",
  "readability_issues": ["short note about one thing you cannot read clearly", "..."]
}

Be factual. Report only what is visible. Never guess unreadable values. \
Keep ``description`` under 500 words — lead with the most important details \
(axes, key values, trends) first. For ``readability_issues``, return an \
empty list ``[]`` if there are no issues; otherwise list at most 5 short \
items (most important first), each a single concrete problem like \
"y-axis label partially obscured" or "legend cut off on the right".\
"""

_JSON_PROMPT_WITH_CAPTION = """\
You are analyzing a figure from a scientific paper. Your description will \
be used to verify that the paper's caption and text accurately describe \
this figure.

The figure's caption reads: "{caption}"

Respond with a JSON object:
{{
  "figure_type": "bar chart | line plot | scatter plot | table | diagram | photo | other",
  "description": "Detailed description: axes with units, all data series/curves \
with their labels, key numeric values you can read, trends and patterns. \
Include enough detail to compare against what the paper claims about this figure.",
  "readability_issues": ["short note about one thing you cannot read clearly", "..."]
}}

Be factual. Report only what is visible. Never guess unreadable values. \
If the caption misrepresents the content, describe what you see, not what \
the caption claims. \
Keep ``description`` under 500 words — lead with the most important details \
(axes, key values, trends) first. For ``readability_issues``, return an \
empty list ``[]`` if there are no issues; otherwise list at most 5 short \
items (most important first), each a single concrete problem like \
"y-axis label partially obscured" or "legend cut off on the right".\
"""


def _build_prompt(caption: str, *, json_mode: bool = False) -> str:
    """Build the VL prompt, including the caption if available.

    Args:
        caption: Figure caption text (empty string if none).
        json_mode: If True, use the structured JSON prompt (vLLM backend).
            If False, use the free-text prompt (transformers backend).
    """
    if json_mode:
        if caption:
            return _JSON_PROMPT_WITH_CAPTION.format(caption=caption)
        return _JSON_PROMPT_BASE
    if caption:
        return _DESCRIBE_PROMPT_WITH_CAPTION.format(caption=caption)
    return _DESCRIBE_PROMPT_BASE


# ---------------------------------------------------------------------------
# JSON response parsing (vLLM backend)
# ---------------------------------------------------------------------------


def _parse_json_response(raw: str) -> VisionResult:
    """Parse structured JSON from the vLLM vision model.

    Expected format: {"figure_type": "...", "description": "...",
    "readability_issues": "..."}.  If parsing fails, the raw text is stored
    as the description field — the model may occasionally produce valid but
    unexpected JSON structures.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Vision model returned invalid JSON, storing as free text")
        return VisionResult(figure_type="", description=raw, readability_issues=[])

    return VisionResult(
        figure_type=data.get("figure_type", ""),
        description=data.get("description", ""),
        readability_issues=data.get("readability_issues", []),
    )


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
        output_ids = model.generate(
            **inputs, max_new_tokens=_MAX_NEW_TOKENS_TRANSFORMERS
        )

    # Decode each sequence, skipping input tokens
    input_len = inputs["input_ids"].shape[1]
    descriptions: list[str] = []
    for i in range(len(image_paths)):
        generated = output_ids[i][input_len:]
        desc = processor.decode(generated, skip_special_tokens=True).strip()
        descriptions.append(desc)

    return descriptions


# ---------------------------------------------------------------------------
# vLLM backend inference
# ---------------------------------------------------------------------------

_VLLM_VISION_PORT = 5002
_VLLM_VISION_MODEL = "qwen3-vl-8b-fp8"


_VLLM_VISION_CONCURRENCY = 64  # concurrent requests to vision vLLM


def _vision_response_format() -> dict[str, Any]:
    """Build the vLLM ``response_format`` that enforces ``VisionResult``.

    Uses ``json_schema`` + ``strict=True`` so the constrained decoder
    respects per-field ``max_length`` from the Pydantic model. Computed
    lazily to avoid paying the schema-generation cost when the vLLM
    backend isn't used.
    """
    from sciwrite_lint.schemas import vllm_schema

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "VisionResult",
            "schema": vllm_schema(VisionResult),
            "strict": True,
        },
    }


def _describe_images_vllm(
    image_paths: list[Path],
    prompts: list[str],
) -> list[str]:
    """Describe images via vLLM vision container on port 5002.

    Sends concurrent requests (up to _VLLM_VISION_CONCURRENCY) to maximize
    GPU utilization. vLLM batches concurrent requests internally.

    Per request: ~2000 tokens (image) + ~300 (prompt) + ~1300 (output) ≈ 3600.
    max_model_len=8192 per request. vLLM manages KV cache across all concurrent
    requests.
    """
    import asyncio

    async def _run() -> list[str]:
        import httpx
        from io import BytesIO

        from PIL import Image

        endpoint = f"http://localhost:{_VLLM_VISION_PORT}/v1"
        sem = asyncio.Semaphore(_VLLM_VISION_CONCURRENCY)
        results: list[str | None] = [None] * len(image_paths)
        response_format = _vision_response_format()

        async def _describe_one(idx: int, path: Path, prompt: str) -> None:
            img = _resize_image(Image.open(path).convert("RGB"))
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            suffix = path.suffix.lower()
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }.get(suffix, "image/png")

            payload = {
                "model": _VLLM_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": _MAX_NEW_TOKENS_VLLM,
                "temperature": 0.1,
                "response_format": response_format,
            }

            # Retry on empty or invalid JSON — same pattern as
            # eval_claims.py and llm_query. Constrained decoding
            # normally guarantees valid JSON, but transient server-side
            # glitches have been observed. On final failure we leave
            # results[idx] as None so the caller can soft-fail the
            # figure (empty VisionResult) without blocking the pipeline.
            async with sem:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    for _attempt in range(_VLLM_RETRIES + 1):
                        resp = await client.post(
                            f"{endpoint}/chat/completions", json=payload
                        )
                        resp.raise_for_status()
                        content = resp.json()["choices"][0]["message"]["content"] or ""
                        content = content.strip()
                        if content:
                            try:
                                json.loads(content)
                            except json.JSONDecodeError:
                                pass
                            else:
                                results[idx] = content
                                return
                        if _attempt < _VLLM_RETRIES:
                            delay = 0.5 * (_attempt + 1)
                            logger.warning(
                                "Vision model returned bad response for "
                                "{} (attempt {}/{}, retrying in {:.1f}s)",
                                path.name,
                                _attempt + 1,
                                _VLLM_RETRIES + 1,
                                delay,
                            )
                            await asyncio.sleep(delay)
                    logger.warning(
                        "Vision model returned bad response for {} after "
                        "{} attempts, leaving empty",
                        path.name,
                        _VLLM_RETRIES + 1,
                    )

        await asyncio.gather(
            *[
                _describe_one(i, p, pr)
                for i, (p, pr) in enumerate(zip(image_paths, prompts))
            ]
        )
        return [r or "" for r in results]

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def describe_figures(
    extracted_images: list[ExtractedImage],
    references_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = _BATCH_SIZE,
    fresh: bool = False,
    backend: str = "transformers",
    source: str = "manuscript",
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
            Only used for the transformers backend.
        batch_size: Images per batch for VL inference (transformers backend only).
        fresh: Ignore cache and re-describe all images.
        backend: ``"transformers"`` (2B in-process) or ``"vllm"`` (8B FP8 container).
        source: ``"manuscript"`` for the paper's own figures, or a ref_key
            (e.g. ``"tanaka2017"``) for cited paper figures.

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
        new_images = split_cached_and_new(
            extracted_images, references_dir, source=source
        )
    else:
        new_images = list(extracted_images)

    if not new_images:
        logger.info(
            "All {} figure(s) cached, skipping VL inference", len(extracted_images)
        )
        return format_descriptions_from_db(
            extracted_images,
            references_dir,  # type: ignore[arg-type]
            source=source,
        )

    logger.info(
        "{}/{} figure(s) need VL inference",
        len(new_images),
        len(extracted_images),
    )

    paths = [img.path for img in new_images]
    use_json = backend == "vllm"
    prompts = [_build_prompt(img.caption, json_mode=use_json) for img in new_images]

    if backend == "vllm":
        logger.info(
            "Running VL inference via vLLM ({}, {} images)",
            _VLLM_VISION_MODEL,
            len(new_images),
        )
        raw_outputs = _describe_images_vllm(paths, prompts)
        all_results = [_parse_json_response(raw) for raw in raw_outputs]
    else:
        # Transformers backend: load model in-process (free text only)
        device = _resolve_vision_device(device)
        logger.info("Running VL inference on {} (batch_size={})", device, batch_size)

        model, processor = _load_model(device)

        try:
            text_outputs: list[str] = []
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]
                batch_prompts = prompts[i : i + batch_size]
                descs = _describe_images_batch(
                    batch_paths, batch_prompts, model, processor, device
                )
                text_outputs.extend(descs)
        finally:
            # Free GPU memory immediately — gc.collect() breaks circular refs
            # before empty_cache() so CUDA can reclaim all allocations.
            del model
            del processor
            import gc

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        all_results = [
            VisionResult(figure_type="", description=desc, readability_issues=[])
            for desc in text_outputs
        ]

    # Save to workspace.db
    if references_dir:
        update_cache(new_images, all_results, references_dir, source=source)

    # Format all descriptions (cached + newly inferred)
    if references_dir:
        result = format_descriptions_from_db(
            extracted_images, references_dir, source=source
        )
    else:
        # No DB — format from what we just inferred
        from sciwrite_lint.vision.cache import _format_entry

        parts: list[str] = []
        for img, vr in zip(new_images, all_results):
            entry = {
                "description": vr.description,
                "figure_type": vr.figure_type,
                "readability_issues": vr.readability_issues,
            }
            parts.append(_format_entry(img.label, img.caption, entry))
        result = "\n\n".join(parts)

    logger.info(
        "Generated {} figure descriptions (~{} chars)",
        len(extracted_images),
        len(result),
    )
    return result


def describe_figures_by_source(
    all_images: list[ExtractedImage],
    ref_image_ranges: dict[str, tuple[int, int]],
    references_dir: Path,
    fresh: bool = False,
    backend: str = "transformers",
) -> None:
    """Run VL inference per-ref, tagging each with its source key.

    Clears the vision cache once (when ``fresh=True``), then calls
    ``describe_figures`` per-ref with ``fresh=False`` so subsequent refs
    don't destroy earlier results.
    """
    from sciwrite_lint.vision.cache import clear_cache

    if fresh:
        clear_cache(references_dir)

    for key, (start, end) in ref_image_ranges.items():
        describe_figures(
            all_images[start:end],
            references_dir=references_dir,
            fresh=False,
            backend=backend,
            source=key,
        )
