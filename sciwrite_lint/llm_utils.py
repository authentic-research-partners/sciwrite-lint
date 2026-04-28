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
from typing import Any, Literal, overload

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


# Thinking presets: effort (soft guidance) + budget (soft token budget).
# Budget is a hint propagated through the chat template — Qwen3 can and
# does overrun it on complex inputs. ``llm_query`` absorbs this by
# (a) sizing ``max_tokens`` as ``response_reserve + budget`` so the answer
# has room even at the declared budget, and (b) stepping thinking down one
# preset on ``finish_reason=length`` so genuine overruns recover
# deterministically instead of burning retries on the same prompt.
THINKING_PRESETS: dict[str, dict[str, Any]] = {
    "off": {"effort": None, "budget": 0},  # no thinking, fastest
    "low": {"effort": "low", "budget": 200},  # quick sanity check
    "medium": {"effort": "medium", "budget": 1024},  # moderate reasoning
    "high": {"effort": "high", "budget": 4096},  # deep analysis
}

# Ordered from most to least thinking. On ``finish_reason=length``,
# ``llm_query`` steps one level down this ladder before retrying. "off"
# is the floor — length failures there fall through to same-prompt
# backoff (a true empty response with thinking disabled is a transient
# glitch, not a budget problem).
_THINKING_LADDER: tuple[str, ...] = ("high", "medium", "low", "off")


def _step_down_thinking(current: str) -> str | None:
    """Return the next lower thinking mode, or None if already at the floor."""
    try:
        idx = _THINKING_LADDER.index(current)
    except ValueError:
        return None
    if idx + 1 < len(_THINKING_LADDER):
        return _THINKING_LADDER[idx + 1]
    return None


# Overloads: n=1 (default or explicit Literal[1]) returns dict|None so
# existing callers retain their types. Any other int returns a list of
# n parsed samples (one per vLLM choice). Overlap is intentional — the
# Literal[1] case is a proper subtype of int and mypy picks it first.
@overload
async def llm_query(  # type: ignore[overload-overlap]
    system: str,
    user: str,
    schema: dict,
    schema_name: str,
    config: LintConfig | None = ...,
    model_name: str = ...,
    max_tokens: int | None = ...,
    client: Any | None = ...,
    thinking: str = ...,
    temperature: float | None = ...,
    n: Literal[1] = 1,
) -> dict | None: ...


@overload
async def llm_query(
    system: str,
    user: str,
    schema: dict,
    schema_name: str,
    config: LintConfig | None = ...,
    model_name: str = ...,
    max_tokens: int | None = ...,
    client: Any | None = ...,
    thinking: str = ...,
    temperature: float | None = ...,
    *,
    n: int,
) -> list[dict | None]: ...


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
    temperature: float | None = None,
    n: int = 1,
) -> dict | None | list[dict | None]:
    """Send a JSON query to vLLM.

    *thinking* controls chain-of-thought reasoning:

    - ``"off"`` (default): disables thinking entirely. Plain JSON prompting
      with direct ``json.loads`` parsing. Fastest (4-12 q/s on Qwen3).

    - ``"low"``: brief reasoning (200 token budget, low effort). Good for
      simple judgment calls. Uses structured output for JSON safety.

    - ``"medium"``: moderate reasoning (1024 tokens). For nuanced comparisons.

    - ``"high"``: deep analysis (4096 tokens). For claim-source verification
      and complex multi-step reasoning.

    Effort and budget are both soft signals propagated through the chat
    template — the model can over- or undershoot. Length-truncation
    recovery (see below) is the backstop for real overruns.

    *max_tokens* is the **response budget** (JSON output size), not the
    combined cap. ``llm_query`` adds the active thinking preset's budget
    on top before dispatching to vLLM so the total ``max_tokens`` sent to
    the API covers both the thinking phase and the response phase. If
    omitted, falls back to ``model_cfg["max_tokens"]``.

    JSON structure and field bounds are enforced by vLLM's constrained
    decoder via ``response_format=json_schema`` with ``strict=True`` —
    the returned ``raw`` content is always a valid JSON object matching
    *schema*. Parsed with ``json.loads`` directly.

    Retry behavior. Up to ``_VLLM_RETRIES`` retries on bad responses,
    with two branches:

    - ``finish_reason=length`` with empty content → thinking overran the
      ``max_tokens`` cap. Step thinking down one preset
      (``high``→``medium``→``low``→``off``) and retry with a short fixed
      delay. At the ``"off"`` floor there's nothing to step down, so
      the call falls through to the same-prompt backoff.
    - Any other empty content or invalid JSON → transient glitch. Retry
      the same call after exponential backoff (0.5s, 1.0s).

    Each vLLM call is recorded exactly once in usage stats — tokens on
    success, tokens + error on failure, never both.

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

    try:
        from sciwrite_lint.usage import current as _usage_current

        def _build_kwargs(mode: str) -> dict[str, Any]:
            """Build chat-completion kwargs for the given thinking mode.

            ``max_tokens`` sent to vLLM is the caller's response reserve
            (or model default) plus the active preset's thinking budget —
            the total cap must cover both phases, and callers pass the
            response portion only.
            """
            preset = THINKING_PRESETS.get(mode, THINKING_PRESETS["off"])
            base_max = max_tokens or model_cfg["max_tokens"]
            effective_max_tokens = base_max + preset["budget"]
            effective_temperature = (
                temperature if temperature is not None else model_cfg["temperature"]
            )
            kw: dict[str, Any] = {
                "model": model_cfg["model"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": effective_temperature,
                "top_p": model_cfg["top_p"],
                "max_tokens": effective_max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                },
            }
            if n > 1:
                # Request n samples in a single completion. vLLM does one
                # prefill and n parallel decodes with the shared KV cache,
                # so cost is ~1× prefill + n× decode rather than n× full
                # calls — this is the right tool for self-consistency
                # voting (see checks.prose_quality).
                kw["n"] = n
            if preset["effort"] is not None:
                kw["extra_body"] = {"thinking": {"budget": preset["budget"]}}
                kw["reasoning_effort"] = preset["effort"]
            else:
                kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            return kw

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

        # Unified retry loop. Two failure branches under one budget:
        #
        #  - finish_reason=length with empty content → thinking consumed
        #    max_tokens before the JSON phase. Retrying the same prompt
        #    is deterministic failure, so step thinking down one preset
        #    and retry with a short fixed delay. At the "off" floor
        #    there's nothing to step down — fall through to generic
        #    backoff (treats it as a transient glitch).
        #  - Any other empty content or invalid JSON → transient
        #    glitch. Retry the same call with exponential backoff.
        #
        # Multi-sample (n > 1) semantics: retry only when ALL n choices
        # fail. Partial success is returned as a list with None entries
        # for the failed choices — self-consistency voting consumes the
        # survivors and treats Nones as abstentions.
        current_thinking = thinking
        last_err_msg = ""

        def _parse_choices(
            completion: Any,
        ) -> tuple[list[dict | None], bool, int]:
            """Parse each choice's content. Returns (parsed, any_length, valid_count)."""
            parsed_list: list[dict | None] = []
            any_length_truncation = False
            valid_count = 0
            for choice in completion.choices:
                raw = choice.message.content
                finish = getattr(choice, "finish_reason", "unknown")
                if raw is None:
                    parsed_list.append(None)
                    if finish == "length":
                        any_length_truncation = True
                    continue
                try:
                    parsed_list.append(json.loads(raw))
                    valid_count += 1
                except json.JSONDecodeError:
                    parsed_list.append(None)
            return parsed_list, any_length_truncation, valid_count

        for attempt in range(_VLLM_RETRIES + 1):
            kwargs = _build_kwargs(current_thinking)
            completion = await client.chat.completions.create(**kwargs)
            parsed_list, any_length, valid_count = _parse_choices(completion)

            if valid_count > 0:
                # At least one good choice — success. For n=1 this is
                # the same as before. For n>1, failed choices stay as
                # None in the returned list.
                _record_call(completion)
                if n == 1:
                    return parsed_list[0]
                return parsed_list

            # All choices failed. Build the error message and decide
            # whether to step down thinking (length truncation) or
            # plain backoff-retry.
            comp_tokens = 0
            if hasattr(completion, "usage") and completion.usage:
                comp_tokens = getattr(completion.usage, "completion_tokens", 0) or 0
            err_type = "EmptyContent" if any_length else "JSONDecodeError"
            _record_call(completion, error=True, error_type=err_type)
            last_err_msg = (
                f"all {n} choices failed (any_length={any_length}, "
                f"comp_tokens={comp_tokens}, thinking={current_thinking})"
            )

            stepped_down: str | None = None
            if any_length:
                stepped_down = _step_down_thinking(current_thinking)

            if attempt < _VLLM_RETRIES:
                if stepped_down is not None:
                    logger.warning(
                        "vLLM length truncation for {} (attempt {}/{}, "
                        "thinking={}) — retrying with thinking={}",
                        schema_name,
                        attempt + 1,
                        _VLLM_RETRIES + 1,
                        current_thinking,
                        stepped_down,
                    )
                    current_thinking = stepped_down
                    await asyncio.sleep(0.2)
                else:
                    delay = 0.5 * (attempt + 1)
                    logger.warning(
                        "vLLM bad response for {} (attempt {}/{}, "
                        "retrying in {:.1f}s): {}",
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
        if n == 1:
            return None
        return [None] * n
    except Exception as e:
        import httpx
        from openai import APIConnectionError, APITimeoutError

        from sciwrite_lint.exceptions import LLMConnectionError
        from sciwrite_lint.usage import current as _usage_current

        run = _usage_current()
        if run:
            run.vllm.record(0.0, error=True, error_type=type(e).__name__)
        if isinstance(e, (APIConnectionError, APITimeoutError, httpx.ConnectError)):
            raise LLMConnectionError(
                f"vLLM at {config.llm_endpoint} became unreachable mid-request: "
                f"{type(e).__name__}: {e}\n"
                "Check server: sciwrite-lint containers status"
            ) from e
        logger.debug("LLM query failed: {}", e)
        if n == 1:
            return None
        return [None] * n
    finally:
        if own_client:
            await client.close()


async def llm_query_batch(
    queries: list[tuple[str, str, dict, str]],
    config: LintConfig | None = None,
    model_name: str = "",
    max_tokens: int | None = None,
    thinking: str = "off",
    temperature: float | None = None,
    n: int = 1,
) -> list[dict | None]:
    """Run multiple LLM queries in parallel, sharing one client.

    Each query is a tuple of (system_prompt, user_prompt, schema, schema_name).
    Returns results in the same order as input queries.

    ``temperature`` applies to every query in the batch. ``None`` (default)
    falls back to the model config's temperature (see ``VLLM_MODELS``).
    The batch runner groups queries by ``(thinking, temperature, n)`` so
    checks with different sampling regimes don't collide.

    ``n`` applies to every query in the batch: each query's n samples are
    generated from a single vLLM completion (one prefill, n decodes), and
    the output list is flattened so query i contributes results
    ``[i*n : (i+1)*n]``. The callers downstream slice by n to vote.
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
                temperature=temperature,
                n=n,
            )
            for sys, usr, sch, name in queries
        ]
        raw_results = await asyncio.gather(*tasks)
        # For n=1, each element is dict|None → append as-is. For n>1,
        # each element is list[dict|None] of length n → extend. Output
        # is always a flat list[dict|None] of length len(queries) * n,
        # so downstream can slice uniformly.
        flat: list[dict | None] = []
        for r in raw_results:
            if isinstance(r, list):
                if len(r) == n:
                    flat.extend(r)
                else:
                    # llm_query normalises length, but be defensive.
                    flat.extend(list(r) + [None] * (n - len(r)))
            else:
                if n == 1:
                    flat.append(r)
                else:
                    # Single dict/None returned despite n>1 — pad.
                    flat.extend([r] + [None] * (n - 1))
        return flat
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
