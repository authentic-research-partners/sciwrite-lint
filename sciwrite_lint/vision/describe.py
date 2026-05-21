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
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sciwrite_lint.config import LintConfig
from loguru import logger

from sciwrite_lint.llm_utils import _VLLM_RETRIES
from sciwrite_lint.schemas import vllm_schema_unbounded
from sciwrite_lint.vision.cache import (
    VisionResult,
    _DESCRIPTION_MAX,
    _FIGURE_TYPE_MAX,
    _ISSUE_MAX,
    _MAX_ISSUES,
)
from sciwrite_lint.vision.image_extraction import ExtractedImage

_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
_MAX_NEW_TOKENS_TRANSFORMERS = (
    # 2B free text: budget ~750 words at 1.3 tok/word; prompt caps at
    # 500 words, leaving ~330 tokens of headroom. VRAM-safe at batch=16.
    1024
)

# Token budget for the 8B vLLM JSON output, sized for a fully-filled
# response (every field at its Pydantic cap). ~4 chars/token is a
# rough English approximation; ``_JSON_OVERHEAD_TOKENS`` covers the
# JSON chrome (braces, field names, quotes, commas).
#
# Tight-by-design: this budget intentionally has no slack for runaway
# generation. If the model writes a description that exceeds the cap,
# the token budget cuts off mid-string before issues can be emitted,
# the JSON truncates, and the parser falls back to free-text storage.
# That's how runaway is detected — see cache.py module comment for
# the layered protection model.
_CHARS_PER_TOKEN = 4
_JSON_OVERHEAD_TOKENS = 100
_MAX_NEW_TOKENS_VLLM = (
    (_FIGURE_TYPE_MAX + _DESCRIPTION_MAX + _MAX_ISSUES * _ISSUE_MAX) // _CHARS_PER_TOKEN
) + _JSON_OVERHEAD_TOKENS


def _resolve_max_image_dim() -> int:
    """Resolve the longest-edge resize target for vision inputs.

    Read from ``SCIWRITE_VISION_MAX_IMAGE_DIM`` env var when set, else
    1024. Used to sweep resolution without redeploying — figure-
    description tasks don't need 1024 px to identify figure type / axes /
    trends, so smaller values cut ViT prefill time roughly proportionally
    (each 32 × 32 effective patch becomes one vision token; halving the
    edge quarters the token count).
    """
    import os as _os

    raw = _os.environ.get("SCIWRITE_VISION_MAX_IMAGE_DIM", "")
    if raw:
        try:
            v = int(raw)
            if v >= 64:
                return v
        except ValueError:
            pass
    return 1024


_MAX_IMAGE_DIM = _resolve_max_image_dim()
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
# constrained decoder validates structure. ``readability_issues`` is
# an unbounded array of string in the wire schema (per-item maxLength
# / array maxItems are intentionally absent — those constraints
# trigger xgrammar's slow path). The prompt does the heavy lifting on
# length and active-look-for-issues guidance.

# Prompt soft-cap defaults — same constants as the Pydantic /
# parser hard caps (see ``cache.py``). One source of truth: the
# prompt asks for the same numbers Pydantic enforces. The model
# rarely emits at the cap so the prompt doesn't need extra slack;
# rare over-runs are silently truncated by ``_parse_json_response``.
_DEFAULT_DESCRIPTION_MAX_CHARS = _DESCRIPTION_MAX
_DEFAULT_ISSUE_MAX_CHARS = _ISSUE_MAX
_DEFAULT_MAX_ISSUES = _MAX_ISSUES

# Categories presented to the model as independent checks. Each is a
# yes/no assessment for THIS figure — no prevalence framing ("most
# figures have ..."), no quota suggestion. The model decides per
# category whether the problem actually exists in this image, then
# aggregates the positive findings into the array. This phrasing is
# meant to produce a graded distribution (0/1/2/3 issues common, 5
# only for genuinely poor figures), avoiding the bimodal 0-vs-max
# pattern that prevalence framing produces.
_READABILITY_CATEGORIES = """\
- Axis tick numbers (too small or unreadable)
- Axis labels (cut off, missing units, or incomplete)
- Legend (missing, cut off, or overlapping the data)
- Data labels and in-figure annotations (illegible or low-contrast)
- Color contrast (data series colors too similar to distinguish)
- Panel labels a/b/c (missing or unclear)
- Resolution (pixelation, blur, or artifacts that obscure detail)\
"""

_JSON_PROMPT_BASE = """\
You are analyzing a figure from a scientific paper. Your description will \
be used to verify that the paper's caption and text accurately describe \
this figure.

Respond with a JSON object:
{{
  "figure_type": "bar chart | line plot | scatter plot | table | diagram | photo | other",
  "description": "Detailed description: axes with units, all data series/curves \
with their labels, key numeric values you can read, trends and patterns. \
Include enough detail to compare against what the paper claims about this figure.",
  "readability_issues": ["short concrete problem", "..."]
}}

Be factual. Report only what is visible. Never guess unreadable values. \
Keep ``description`` under {description_max_words} words — lead with the \
most important details (axes, key values, trends) first.

For ``readability_issues``, evaluate each of these categories \
independently — does THIS figure have a problem in this category that \
genuinely impairs the reader's ability to interpret what's shown?
{readability_categories}

For each category, judge yes or no. Include only the categories where \
your honest answer is yes — minor cosmetic imperfections don't count, \
only problems that materially impair interpretation. The output array \
is the issues you found; it can be empty ``[]`` if every category is \
fine. List most important first, up to {max_issues} items, **each a \
brief phrase under {issue_max_words} words** — write single short \
notes like "y-axis tick labels too small" or "legend cut off on \
right", not full sentences.\
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
  "readability_issues": ["short concrete problem", "..."]
}}

Be factual. Report only what is visible. Never guess unreadable values. \
If the caption misrepresents the content, describe what you see, not what \
the caption claims. Keep ``description`` under {description_max_words} \
words — lead with the most important details (axes, key values, \
trends) first.

For ``readability_issues``, evaluate each of these categories \
independently — does THIS figure have a problem in this category that \
genuinely impairs the reader's ability to interpret what's shown?
{readability_categories}

For each category, judge yes or no. Include only the categories where \
your honest answer is yes — minor cosmetic imperfections don't count, \
only problems that materially impair interpretation. The output array \
is the issues you found; it can be empty ``[]`` if every category is \
fine. List most important first, up to {max_issues} items, **each a \
brief phrase under {issue_max_words} words** — write single short \
notes like "y-axis tick labels too small" or "legend cut off on \
right", not full sentences.\
"""


def _build_prompt(
    caption: str,
    *,
    json_mode: bool = False,
    description_max_chars: int = _DEFAULT_DESCRIPTION_MAX_CHARS,
    issue_max_chars: int = _DEFAULT_ISSUE_MAX_CHARS,
    max_issues: int = _DEFAULT_MAX_ISSUES,
) -> str:
    """Build the VL prompt, including the caption if available.

    Args:
        caption: Figure caption text (empty string if none).
        json_mode: If True, use the structured JSON prompt (vLLM backend).
            If False, use the free-text prompt (transformers backend).
        description_max_chars: Soft cap on ``description`` length, in
            characters. Defaults to ``_DESCRIPTION_MAX`` (the same cap
            Pydantic and the parser enforce). Override only if you
            need a tighter description for a specific call site.
        issue_max_chars: Soft cap on each ``readability_issues`` item.
            Defaults to ``_ISSUE_MAX``.
        max_issues: Cap on number of issues. Defaults to ``_MAX_ISSUES``.
    """
    if json_mode:
        # Word-based phrasing in the prompt is more natural — LLMs are
        # better calibrated to "500 words" than to "4000 characters".
        # Internally we track chars (the unit Pydantic / parser
        # truncate to); the prompt converts to approximate words at
        # ~8 chars/word for English including spaces and punctuation.
        description_max_words = description_max_chars // 8
        issue_max_words = max(issue_max_chars // 8, 1)
        kwargs = {
            "description_max_words": description_max_words,
            "issue_max_words": issue_max_words,
            "max_issues": max_issues,
            "readability_categories": _READABILITY_CATEGORIES,
        }
        if caption:
            return _JSON_PROMPT_WITH_CAPTION.format(caption=caption, **kwargs)
        return _JSON_PROMPT_BASE.format(**kwargs)
    if caption:
        return _DESCRIBE_PROMPT_WITH_CAPTION.format(caption=caption)
    return _DESCRIBE_PROMPT_BASE


# ---------------------------------------------------------------------------
# JSON response parsing (vLLM backend)
# ---------------------------------------------------------------------------


def _parse_json_response(raw: str) -> VisionResult:
    """Parse structured JSON from the vLLM vision model.

    Expected format: ``figure_type``, ``description``, and
    ``readability_issues`` (an array of strings — schema enforced by
    ``_vision_response_format``).

    The wire schema has no length / count constraints (see
    ``_vision_response_format`` for why). To prevent
    ``VisionResult.ValidationError`` on minor over-runs, this parser
    truncates each field to the same caps the Pydantic model enforces
    before construction:
    - ``readability_issues`` clamped to the first ``_MAX_ISSUES`` items
      (skipping empty/non-string entries), each item clamped to
      ``_ISSUE_MAX`` chars
    - ``description`` clamped to ``_DESCRIPTION_MAX`` chars
    - ``figure_type`` clamped to ``_FIGURE_TYPE_MAX`` chars

    If JSON parsing fails the raw text is stored as the description
    (truncated to ``_DESCRIPTION_MAX``) — the model may occasionally
    produce valid but unexpected JSON structures.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Vision model returned invalid JSON, storing as free text")
        return VisionResult(
            figure_type="",
            description=raw[:_DESCRIPTION_MAX],
            readability_issues=[],
        )

    raw_issues = data.get("readability_issues") or []
    issues: list[str] = []
    if isinstance(raw_issues, list):
        for item in raw_issues:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if not cleaned:
                continue
            issues.append(cleaned[:_ISSUE_MAX])
            if len(issues) >= _MAX_ISSUES:
                break

    description = data.get("description") or ""
    if not isinstance(description, str):
        description = ""
    figure_type = data.get("figure_type") or ""
    if not isinstance(figure_type, str):
        figure_type = ""

    return VisionResult(
        figure_type=figure_type[:_FIGURE_TYPE_MAX],
        description=description[:_DESCRIPTION_MAX],
        readability_issues=issues,
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


def _vision_response_format() -> dict[str, Any]:
    """Build the vLLM ``response_format`` for figure description.

    Schema is auto-derived from the ``VisionResult`` Pydantic model
    via ``vllm_schema_unbounded`` — single source of truth. The
    helper strips ``maxLength`` / ``maxItems`` / ``pattern`` / etc.
    from the JSON schema before sending; if those reach the wire on
    this vLLM / xgrammar stack a single one collapses 30/30 success
    to 0/30 timeout. Pydantic's bounds still validate post-decode (in
    ``_parse_json_response``); they just don't go on the wire.

    !!! DANGER — DO NOT switch to ``vllm_schema(VisionResult)`` !!!
    The unsuffixed helper would faithfully translate Pydantic's
    ``Field(max_length=...)`` constraints into JSON Schema
    ``maxLength`` and re-trigger the trap. The ``_unbounded`` suffix
    is load-bearing. ``TestVisionWireSchemaUnbounded`` in
    ``tests/test_schema_bounds.py`` regression-guards this at CI time.

    Set ``SCIWRITE_VISION_NO_SCHEMA=1`` to disable schema enforcement
    entirely (for benchmarking only — production code expects valid
    JSON output).
    """
    schema = vllm_schema_unbounded(VisionResult)
    # Pydantic only marks fields without a default as ``required``; our
    # ``VisionResult`` fields all have Python-side defaults (for the
    # transformers backend and test fixtures) but on the wire we want
    # vLLM to enforce that all three are emitted. Add the ``required``
    # list explicitly. The class docstring also rides along as
    # ``"description"`` — strip it (vLLM doesn't need it and the wire
    # payload stays compact).
    schema.pop("description", None)
    schema["required"] = list(schema["properties"].keys())
    schema["additionalProperties"] = False

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "VisionResult",
            "schema": schema,
            "strict": True,
        },
    }


_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _encode_image_for_vllm(path: Path) -> tuple[str, str]:
    """Resize, PNG-encode, and base64 a single image. CPU-only, sync.

    Hoisted out of the async hot loop so PIL doesn't block the event
    loop thread: with PIL inline, only one task at a time can advance
    past preprocessing, starving vLLM down to ~5 concurrent requests
    regardless of the semaphore cap. Called via ``asyncio.to_thread``
    so the default executor's thread pool can encode in parallel while
    network requests are in flight.
    """
    from io import BytesIO

    from PIL import Image

    img = _resize_image(Image.open(path).convert("RGB"))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    return b64, mime


def _describe_images_vllm(
    image_paths: list[Path],
    prompts: list[str],
    config: LintConfig | None = None,
) -> list[str]:
    """Describe images via vLLM vision container on port 5002.

    Concurrency is capped at ``config.vision_max_concurrency`` so vLLM's
    multimodal KV cache can't be saturated by a large image batch — the
    same backpressure principle as ``llm_query_batch``. Without it,
    firing hundreds of image requests at once could cause one request to
    wait past the per-request timeout and crash the subprocess with
    ``httpx.ReadTimeout``.

    Per request: ~2000 tokens (image) + ~300 (prompt) + ~1300 (output)
    ≈ 3600. max_model_len=8192. vLLM manages KV cache across all
    concurrent requests.

    Image PIL+PNG encoding runs upfront via ``asyncio.to_thread`` so the
    default executor's thread pool parallelizes preprocessing across
    cores while the async network phase keeps vLLM saturated. Without
    this, sync PIL inside ``_describe_one`` blocks the event loop one
    image at a time — vLLM would only see ~5 concurrent requests even
    with cap=64, because the loop can't dispatch faster than PIL.

    Per-image failures (network or content) leave ``results[idx] =
    None`` so the caller can soft-fail one figure without aborting the
    entire batch — propagating any exception out of ``asyncio.gather``
    would cancel all in-flight peers and lose their work.
    """
    import asyncio

    if config is None:
        from sciwrite_lint.config import load_config

        config = load_config()
    cap = max(1, config.vision_max_concurrency)
    # Never push past vLLM's admission ceiling — see effective_max_concurrency.
    from sciwrite_lint.vllm.vllm_server import effective_max_concurrency

    cap = effective_max_concurrency(config, cap, model="qwen3-vl", label="vision")
    request_timeout = config.vision_request_timeout

    async def _run() -> list[str]:
        import httpx

        from sciwrite_lint.llm.concurrency_optimizer import concurrency_slot

        endpoint = f"http://localhost:{_VLLM_VISION_PORT}/v1"
        results: list[str | None] = [None] * len(image_paths)
        # Schema-constrained decode is ~3× slower than free-form on
        # this hardware (each token goes through xgrammar on CPU,
        # serialized across all in-flight requests — observed as GPU
        # oscillation between 0 and 100 % during decode). Set
        # ``SCIWRITE_VISION_NO_SCHEMA=1`` to test the unconstrained
        # path without redeploying.
        import os as _os

        if _os.environ.get("SCIWRITE_VISION_NO_SCHEMA") == "1":
            response_format = None
        else:
            response_format = _vision_response_format()

        encoded: list[tuple[str, str]] = await asyncio.gather(
            *[asyncio.to_thread(_encode_image_for_vllm, p) for p in image_paths]
        )

        # ``config`` is guaranteed non-None by line ~411 (the cap calc
        # would have crashed otherwise). Capture it locally so the
        # closure inside concurrency_slot's ``async with`` sees the
        # narrowed type, not the broader ``LintConfig | None``.
        assert config is not None
        cfg = config

        # Per-run counters surfaced to the parent process via the
        # ``VISION_RUN_STATS=`` stdout marker below — lets
        # ``_stage_cited_vision`` log the retry/soft-fail picture
        # alongside cache-hit counts so a slow run is diagnosable from
        # the log alone (no need to re-grep WARN lines).
        stats = {
            "total_images": len(image_paths),
            "succeeded": 0,
            "retried": 0,
            "soft_failed": 0,
        }

        # Same transient-error category as llm_utils._TRANSIENT_NET_ERRS,
        # at the raw-httpx layer (this path doesn't go through openai SDK).
        # ``httpx.RequestError`` is the parent of every request-time
        # error httpx raises — timeouts, network-level read/write/connect
        # errors, protocol errors. ``HTTPStatusError`` is a sibling, not
        # a child, so 5xx / 429 from vLLM (saturation backpressure) is
        # still handled by the dedicated branch below.
        transient = (httpx.RequestError,)

        async def _describe_one(
            idx: int, path: Path, b64: str, mime: str, prompt: str
        ) -> None:
            payload = {
                "model": _VLLM_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": _MAX_NEW_TOKENS_VLLM,
                "temperature": 0.1,
            }
            if response_format is not None:
                payload["response_format"] = response_format

            # Unified retry budget covers both content errors
            # (empty/invalid JSON) and transient HTTP/network errors
            # (read timeout, 5xx, 429, dropped connection). On terminal
            # failure leave ``results[idx] = None`` so the caller
            # soft-fails one figure rather than crashing the whole
            # asyncio.gather (which would lose all in-flight peers).
            async with _slot():
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    for _attempt in range(_VLLM_RETRIES + 1):
                        try:
                            resp = await client.post(
                                f"{endpoint}/chat/completions", json=payload
                            )
                            resp.raise_for_status()
                            content = (
                                resp.json()["choices"][0]["message"]["content"] or ""
                            )
                        except transient as net_err:
                            if _attempt < _VLLM_RETRIES:
                                delay = 2.0 * (_attempt + 1)
                                logger.warning(
                                    "Vision transient error for {} "
                                    "(attempt {}/{}): {}: {} — retrying in {:.1f}s",
                                    path.name,
                                    _attempt + 1,
                                    _VLLM_RETRIES + 1,
                                    type(net_err).__name__,
                                    net_err,
                                    delay,
                                )
                                await asyncio.sleep(delay)
                                continue
                            logger.warning(
                                "Vision transient error for {} after "
                                "{} attempts ({}: {}), leaving empty",
                                path.name,
                                _VLLM_RETRIES + 1,
                                type(net_err).__name__,
                                net_err,
                            )
                            stats["soft_failed"] += 1
                            return
                        except httpx.HTTPStatusError as status_err:
                            # raise_for_status raised — classify by code:
                            # 5xx / 429 are vLLM saturation backpressure,
                            # retry. Other 4xx are programmer errors,
                            # bail with empty result.
                            code = status_err.response.status_code
                            if code >= 500 or code == 429:
                                if _attempt < _VLLM_RETRIES:
                                    delay = 2.0 * (_attempt + 1)
                                    logger.warning(
                                        "Vision vLLM HTTP {} for {} "
                                        "(attempt {}/{}) — retrying in {:.1f}s",
                                        code,
                                        path.name,
                                        _attempt + 1,
                                        _VLLM_RETRIES + 1,
                                        delay,
                                    )
                                    await asyncio.sleep(delay)
                                    continue
                                logger.warning(
                                    "Vision vLLM HTTP {} for {} after {} "
                                    "attempts, leaving empty",
                                    code,
                                    path.name,
                                    _VLLM_RETRIES + 1,
                                )
                                stats["soft_failed"] += 1
                                return
                            logger.warning(
                                "Vision vLLM HTTP {} for {}: {} — leaving empty",
                                code,
                                path.name,
                                status_err,
                            )
                            stats["soft_failed"] += 1
                            return

                        content = content.strip()
                        if content:
                            # In SCIWRITE_VISION_NO_SCHEMA test mode the
                            # output is free text — accept anything
                            # non-empty. Production path keeps strict
                            # JSON validation so downstream parsers
                            # (``_parse_json_response``) get well-formed
                            # input.
                            if response_format is None:
                                results[idx] = content
                                stats["succeeded"] += 1
                                return
                            try:
                                json.loads(content)
                            except json.JSONDecodeError:
                                pass
                            else:
                                results[idx] = content
                                stats["succeeded"] += 1
                                if _attempt > 0:
                                    stats["retried"] += 1
                                    logger.info(
                                        "Vision call for {} recovered on attempt {}/{}",
                                        path.name,
                                        _attempt + 1,
                                        _VLLM_RETRIES + 1,
                                    )
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
                    stats["soft_failed"] += 1

        from sciwrite_lint.llm.concurrency_optimizer import ControllerParams

        ctrl_params = ControllerParams(
            target_kv_lo=cfg.concurrency_target_kv_lo,
            target_kv_grow=cfg.concurrency_target_kv_grow,
            target_kv_hi=cfg.concurrency_target_kv_hi,
        )
        async with concurrency_slot(
            use_dynamic=cfg.use_dynamic_concurrency,
            endpoint=endpoint,
            size_class="vision",
            static_cap=cap,
            label="vision",
            params=ctrl_params,
        ) as _slot:
            await asyncio.gather(
                *[
                    _describe_one(i, p, b64, mime, pr)
                    for i, (p, (b64, mime), pr) in enumerate(
                        zip(image_paths, encoded, prompts)
                    )
                ]
            )
        # Marker line for the parent process (``_stage_cited_vision``)
        # to parse from captured stdout — keeps retry/soft-fail counts
        # visible in the post-run summary log without an IPC channel.
        # Plain ``print`` deliberately: subprocess.run captures stdout
        # but loguru routes elsewhere.
        print(f"VISION_RUN_STATS={json.dumps(stats)}")
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
    config: LintConfig | None = None,
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
        raw_outputs = _describe_images_vllm(paths, prompts, config=config)
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
    config: LintConfig | None = None,
) -> None:
    """Run VL inference across ALL refs in one batch, tagged by source.

    Per-ref serial batching produced tiny per-call batches (1-26 images)
    that left vLLM at ~5 concurrent requests / 15% GPU util — the
    semaphore cap never engaged because each ref's batch was smaller
    than the cap. Cross-ref batching collapses all refs into a single
    inference run; cache writes still tag each image with its
    originating ref_key so downstream lookups by source still work.
    """
    from collections import defaultdict

    from sciwrite_lint.vision.cache import (
        clear_cache,
        split_cached_and_new,
        update_cache,
    )

    if fresh:
        clear_cache(references_dir)

    if not all_images or not ref_image_ranges:
        return

    images_by_source: dict[str, list[ExtractedImage]] = defaultdict(list)
    for key, (start, end) in ref_image_ranges.items():
        for img in all_images[start:end]:
            images_by_source[key].append(img)

    new_with_source: list[tuple[ExtractedImage, str]] = []
    total = 0
    for source, imgs in images_by_source.items():
        total += len(imgs)
        if fresh:
            new_imgs = imgs
        else:
            new_imgs = split_cached_and_new(imgs, references_dir, source=source)
        for img in new_imgs:
            new_with_source.append((img, source))

    if not new_with_source:
        logger.info("All {} cited-paper figure(s) cached", total)
        return

    logger.info(
        "{}/{} cited-paper figure(s) need VL inference (across {} refs)",
        len(new_with_source),
        total,
        len(images_by_source),
    )

    use_json = backend == "vllm"
    paths = [img.path for img, _ in new_with_source]
    prompts = [
        _build_prompt(img.caption, json_mode=use_json) for img, _ in new_with_source
    ]

    if backend == "vllm":
        logger.info(
            "Running VL inference via vLLM ({}, {} images cross-ref)",
            _VLLM_VISION_MODEL,
            len(new_with_source),
        )
        raw_outputs = _describe_images_vllm(paths, prompts, config=config)
        all_results = [_parse_json_response(raw) for raw in raw_outputs]
    else:
        device = _resolve_vision_device("auto")
        logger.info(
            "Running VL inference on {} (cross-ref batched, batch_size={})",
            device,
            _BATCH_SIZE,
        )
        model, processor = _load_model(device)
        try:
            text_outputs: list[str] = []
            for i in range(0, len(paths), _BATCH_SIZE):
                batch_paths = paths[i : i + _BATCH_SIZE]
                batch_prompts = prompts[i : i + _BATCH_SIZE]
                descs = _describe_images_batch(
                    batch_paths, batch_prompts, model, processor, device
                )
                text_outputs.extend(descs)
        finally:
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

    by_source: dict[str, list[tuple[ExtractedImage, VisionResult]]] = defaultdict(list)
    for (img, source), result in zip(new_with_source, all_results):
        by_source[source].append((img, result))

    for source, items in by_source.items():
        imgs = [i for i, _ in items]
        ress = [r for _, r in items]
        update_cache(imgs, ress, references_dir, source=source)

    logger.info(
        "Cached {} VL descriptions across {} refs",
        len(new_with_source),
        len(by_source),
    )
