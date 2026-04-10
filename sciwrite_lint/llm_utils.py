"""Shared LLM utilities for sciwrite-lint rules and eval pipelines.

Provides model configuration, a single-query async helper for rules that
use the local vLLM server, and a permissive JSON extractor (``extract_json``)
retained for eval pipelines. Production ``llm_query`` calls use vLLM's
constrained decoding (``response_format=json_schema`` + ``strict=True``)
and parse the result with ``json.loads`` directly — no regex fallback
needed, since the decoder guarantees a valid JSON object that matches the
Pydantic schema bounds.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger

from sciwrite_lint.config import LintConfig

# ---------------------------------------------------------------------------
# vLLM model presets
# ---------------------------------------------------------------------------

VLLM_MODELS: dict[str, dict[str, Any]] = {
    "qwen3": {
        "model": "qwen3-8b-fp8",  # matches --served-model-name
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2048,
    },
    "gemma3": {
        "model": "gemma3-12b-fp8",  # matches --served-model-name
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 2048,
    },
}

VLLM_DEFAULT_MODEL = "qwen3"


def get_model_config(config: LintConfig | None = None, model_name: str = "") -> dict:
    """Resolve vLLM model configuration from config or explicit name."""
    config = config or LintConfig()
    key = model_name or config.llm_model or VLLM_DEFAULT_MODEL
    return VLLM_MODELS.get(key, VLLM_MODELS[VLLM_DEFAULT_MODEL])


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json(text: str | None) -> dict | None:
    """Parse JSON from LLM output, handling <think> tags, code fences, etc."""
    if text is None:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# Number of retries for "bad response" from vLLM — applies to both empty
# content (``content=None``, typically ``finish_reason=length`` on the
# thinking budget) and invalid JSON (rare with constrained decoding, but
# transient server-side glitches and truncated reads have been observed
# in production). 2 retries gives ~1.5s of exponential backoff, which
# fixes the transient case without wasting much if the cause is a real
# misconfiguration.
_VLLM_RETRIES = 2


async def retry_on_empty(
    create_call: Any,
    label: str,
    retries: int = _VLLM_RETRIES,
) -> Any:
    """Retry a vLLM completion call when the model returns empty content.

    Retries the same call with a short delay. Handles intermittent vLLM
    issues (empty content only — does not validate JSON).

    Args:
        create_call: Async callable (no args) returning a completion.
        label: Human-readable label for log messages.
        retries: Number of retry attempts after the first empty response.

    Returns:
        The completion object (with possibly None content after all retries).
    """
    for attempt in range(1 + retries):
        completion = await create_call()
        content = completion.choices[0].message.content
        if content is not None:
            return completion

        finish = getattr(completion.choices[0], "finish_reason", "unknown")
        comp_tokens = 0
        if hasattr(completion, "usage") and completion.usage:
            comp_tokens = getattr(completion.usage, "completion_tokens", 0) or 0

        # Transient retry with delay
        if attempt < retries:
            delay = 0.5 * (attempt + 1)
            logger.warning(
                "vLLM returned empty response for {} (attempt {}/{}, "
                "finish_reason={}, comp_tokens={}), retrying in {:.1f}s",
                label,
                attempt + 1,
                1 + retries,
                finish,
                comp_tokens,
                delay,
            )
            await asyncio.sleep(delay)
        else:
            logger.warning(
                "vLLM returned empty response for {} after {} attempts "
                "(finish_reason={}, comp_tokens={}), giving up",
                label,
                1 + retries,
                finish,
                comp_tokens,
            )
    return completion


# ---------------------------------------------------------------------------
# Async query helpers
# ---------------------------------------------------------------------------


# Thinking presets: effort (soft guidance) + budget (hard token cap).
# Must be paired — high effort with low budget causes truncated reasoning.
THINKING_PRESETS: dict[str, dict[str, Any]] = {
    "off": {"effort": None, "budget": 0},  # no thinking, fastest
    "low": {"effort": "low", "budget": 200},  # quick sanity check
    "medium": {"effort": "medium", "budget": 1024},  # moderate reasoning
    "high": {"effort": "high", "budget": 4096},  # deep analysis
}


async def llm_query(
    system: str,
    user: str,
    schema: dict,
    schema_name: str,
    config: LintConfig | None = None,
    model_name: str = "",
    max_tokens: int | None = None,
    client: Any | None = None,
    thinking: str = "off",
) -> dict | None:
    """Send a JSON query to vLLM.

    *thinking* controls chain-of-thought reasoning:

    - ``"off"`` (default): disables thinking entirely. Plain JSON prompting
      with direct ``json.loads`` parsing. Fastest (4-12 q/s on Qwen3).

    - ``"low"``: brief reasoning (200 token budget, low effort). Good for
      simple judgment calls. Uses structured output for JSON safety.

    - ``"medium"``: moderate reasoning (1024 tokens). For nuanced comparisons.

    - ``"high"``: deep analysis (4096 tokens). For claim-source verification
      and complex multi-step reasoning.

    Effort is a soft guidance signal — the model may think less.
    Budget is a hard cap — thinking is abruptly truncated at the limit.
    Both must be paired: high effort + low budget = truncated reasoning.

    JSON structure and field bounds are enforced by vLLM's constrained
    decoder via ``response_format=json_schema`` with ``strict=True`` —
    the returned ``raw`` content is always a valid JSON object matching
    *schema*. Parsed with ``json.loads`` directly.

    A single unified retry loop handles both failure modes that have
    been observed in production: empty content (``content=None``,
    usually ``finish_reason=length`` on the thinking budget) and
    invalid JSON (rare server-side glitch). Both retry up to
    ``_VLLM_RETRIES`` times with short exponential backoff. Each vLLM
    call is recorded exactly once in usage stats — tokens on success,
    tokens + error on failure, never both.

    Requires vLLM started with ``--reasoning-parser qwen3`` for Qwen3
    (added automatically by ``sciwrite-lint vllm start --model qwen3``).

    If *client* is provided, uses it (caller manages lifecycle).
    Otherwise creates and closes its own AsyncOpenAI client.
    """
    from openai import AsyncOpenAI

    config = config or LintConfig()
    model_cfg = get_model_config(config, model_name)
    own_client = client is None
    if own_client:
        client = AsyncOpenAI(
            base_url=config.llm_endpoint,
            api_key="dummy",
            timeout=config.llm_timeout,
        )
    assert client is not None  # narrowing for mypy

    preset = THINKING_PRESETS.get(thinking, THINKING_PRESETS["off"])

    try:
        kwargs: dict[str, Any] = {
            "model": model_cfg["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": model_cfg["temperature"],
            "top_p": model_cfg["top_p"],
            "max_tokens": max_tokens or model_cfg["max_tokens"],
        }

        # Always use structured output — guarantees valid JSON via
        # constrained decoding, with Pydantic maxLength on all fields.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }

        if preset["effort"] is not None:
            kwargs["extra_body"] = {
                "thinking": {"budget": preset["budget"]},
            }
            kwargs["reasoning_effort"] = preset["effort"]
        else:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }

        from sciwrite_lint.usage import current as _usage_current

        def _record_call(
            completion: Any,
            *,
            error: bool = False,
            error_type: str | None = None,
        ) -> None:
            """Record one vLLM call in usage stats.

            Always increments ``calls``; adds token counts when the
            completion exposes usage metadata; sets ``error_type`` and
            bumps ``errors`` when *error* is True. Called exactly once
            per vLLM request — never twice.
            """
            run = _usage_current()
            if run is None:
                return
            record_kwargs: dict[str, Any] = {}
            try:
                u = completion.usage
                record_kwargs["prompt_tokens"] = getattr(u, "prompt_tokens", 0) or 0
                record_kwargs["completion_tokens"] = (
                    getattr(u, "completion_tokens", 0) or 0
                )
            except (AttributeError, TypeError) as e:
                logger.debug("Could not extract vLLM usage stats: {}", e)
            if error_type is not None:
                record_kwargs["error_type"] = error_type
            run.vllm.record(0.0, error=error, **record_kwargs)

        # Unified retry loop: a "bad response" from vLLM is either empty
        # content (content=None, usually finish_reason=length on the
        # thinking budget) or content that isn't valid JSON (rare —
        # constrained decoding should prevent it, but we've seen
        # transient server-side glitches in production). Both cases get
        # the same treatment: short backoff and retry up to _VLLM_RETRIES
        # times.
        last_err_msg = ""
        for attempt in range(_VLLM_RETRIES + 1):
            completion = await client.chat.completions.create(**kwargs)
            raw = completion.choices[0].message.content

            if raw is None:
                finish = getattr(completion.choices[0], "finish_reason", "unknown")
                comp_tokens = 0
                if hasattr(completion, "usage") and completion.usage:
                    comp_tokens = getattr(completion.usage, "completion_tokens", 0) or 0
                _record_call(completion, error=True, error_type="EmptyContent")
                last_err_msg = (
                    f"empty content (finish_reason={finish}, comp_tokens={comp_tokens})"
                )
            else:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    _record_call(completion, error=True, error_type="JSONDecodeError")
                    last_err_msg = f"invalid JSON: {e} — raw prefix: {raw[:200]!r}"
                else:
                    _record_call(completion)
                    return parsed

            if attempt < _VLLM_RETRIES:
                delay = 0.5 * (attempt + 1)
                logger.warning(
                    "vLLM bad response for {} (attempt {}/{}, retrying in {:.1f}s): {}",
                    schema_name,
                    attempt + 1,
                    _VLLM_RETRIES + 1,
                    delay,
                    last_err_msg,
                )
                await asyncio.sleep(delay)

        logger.warning(
            "vLLM bad response for {} after {} attempts, giving up: {}",
            schema_name,
            _VLLM_RETRIES + 1,
            last_err_msg,
        )
        return None
    except Exception as e:
        logger.debug("LLM query failed: {}", e)
        from sciwrite_lint.usage import current as _usage_current

        run = _usage_current()
        if run:
            run.vllm.record(0.0, error=True, error_type=type(e).__name__)
        return None
    finally:
        if own_client:
            await client.close()


async def llm_query_batch(
    queries: list[tuple[str, str, dict, str]],
    config: LintConfig | None = None,
    model_name: str = "",
    max_tokens: int | None = None,
    thinking: str = "off",
) -> list[dict | None]:
    """Run multiple LLM queries in parallel, sharing one client.

    Each query is a tuple of (system_prompt, user_prompt, schema, schema_name).
    Returns results in the same order as input queries.
    """
    from openai import AsyncOpenAI

    config = config or LintConfig()
    client = AsyncOpenAI(
        base_url=config.llm_endpoint,
        api_key="dummy",
        timeout=config.llm_timeout,
    )

    try:
        tasks = [
            llm_query(
                sys,
                usr,
                sch,
                name,
                config,
                model_name,
                max_tokens,
                client,
                thinking=thinking,
            )
            for sys, usr, sch, name in queries
        ]
        return await asyncio.gather(*tasks)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Two-phase rule protocol for batched execution
# ---------------------------------------------------------------------------


def batchable(
    build_fn,
    process_fn,
):
    """Decorator that attaches build_queries and process_results to a rule function.

    The async batch runner (``_run_llm_rules_batched``) collects queries
    from all batchable rules, makes one async vLLM call, then dispatches
    results via ``process_results``. The rule body is never called in
    production — it exists only to satisfy the ``@rule`` decorator
    signature and should raise ``RuntimeError``.

    Usage::

        @check(id="cross-section-consistency", ..., category="local-llm")
        def check_cross_section(tex_path, config):
            raise RuntimeError("LLM checks must run via the async batch runner")
    """

    def decorator(fn):
        fn.build_queries = build_fn
        fn.process_results = process_fn
        return fn

    return decorator
