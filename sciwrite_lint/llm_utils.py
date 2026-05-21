"""Shared LLM utilities for sciwrite-lint rules and eval pipelines.

Provides model configuration, a single-query async helper for rules that
use the local vLLM server, and a permissive JSON extractor (``extract_json``)
retained for eval pipelines. Production ``llm_query`` calls use vLLM's
constrained decoding (``response_format=json_schema`` + ``strict=True``)
and parse the result with ``json.loads`` directly — no regex extraction
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
from sciwrite_lint.vllm.vllm_server import MODELS as _SERVED_MODELS

# ---------------------------------------------------------------------------
# vLLM model presets
# ---------------------------------------------------------------------------
#
# Sampling parameters per model. Served names (``model``) come from the
# canonical registry in ``vllm.vllm_server.MODELS``, so a model rename
# in one place propagates here automatically.
#
# ``max_tokens`` notes (qwen3 / gemma3): response reserve. ``llm_query``
# sends ``max_tokens + active_thinking_budget`` to vLLM, so the actual
# per-call cap is 4096 + 1024 = 5120 at thinking=medium. The
# ``FullPaperIssueList`` / ``ConsistencyResult`` worst-case responses are
# ~1100-1700 tokens — well within the response half — and the thinking
# phase gets a generous 4096-token slack (Qwen3's chain-of-thought
# routinely overshoots its declared 1024-token budget on heavy 30K-token
# full-paper prompts, which used to fire the length-truncation retry
# ladder repeatedly).
_SAMPLING: dict[str, dict[str, Any]] = {
    "qwen3": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096},
    "gemma3": {"temperature": 0.3, "top_p": 0.95, "max_tokens": 4096},
}

VLLM_MODELS: dict[str, dict[str, Any]] = {
    key: {"model": _SERVED_MODELS[key]["served_name"], **sampling}
    for key, sampling in _SAMPLING.items()
}

VLLM_DEFAULT_MODEL = "qwen3"


# ---------------------------------------------------------------------------
# Per-caller concurrency size classes for ``llm_query_batch(concurrency=…)``
# ---------------------------------------------------------------------------
#
# The global default ``config.llm_max_concurrency`` is sized for the
# heaviest caller in the codebase: ``ref_internal_checks`` full-paper
# queries with ~30K-token prompts. Lighter callers can pass an explicit
# ``concurrency=`` higher because KV-cache pressure scales with
# ``concurrency × per-request KV tokens``, not request count alone.
#
# These constants name the size classes so call sites read intent ("this
# is a small-prompt batch") rather than a bare number, and so a single
# retune updates every caller in the same class. The two thresholds
# reflect whether the caller runs concurrently with the heavy batch
# (which dominates KV pressure in ``ref_internal_checks``):
#
# - SMALL_PROMPT — short prompts, manuscript LLM checks. Runs in
#   isolation (rules stage, no concurrent batch), so the cap is set
#   above the throughput plateau to reduce wait-time variance.
# - MEDIUM_PROMPT — medium prompts, claim taxonomy and ref-internal
#   pairwise. Set at the throughput plateau, not above, because this
#   call site runs concurrently with the heavy fullpaper batch and the
#   combined KV budget needs headroom.
# - HEAVY_PROMPT — long prompts (ref-internal full-paper). Inherits
#   ``config.llm_max_concurrency`` from TOML; pass ``concurrency=None``
#   at the call site.
SMALL_PROMPT_CONCURRENCY = 100
MEDIUM_PROMPT_CONCURRENCY = 50


def get_model_config(config: LintConfig | None = None, model_name: str = "") -> dict:
    """Resolve vLLM model configuration from config or explicit name."""
    config = config or LintConfig()
    key = model_name or config.llm_model or VLLM_DEFAULT_MODEL
    return VLLM_MODELS.get(key, VLLM_MODELS[VLLM_DEFAULT_MODEL])


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json(text: str | None) -> dict[str, Any] | None:
    """Parse JSON from LLM output, handling thinking tags, code fences, etc.

    Strips both ``<think>...</think>`` (vLLM convention) and
    ``<thinking>...</thinking>`` (Claude CLI convention).
    """
    if text is None:
        return None
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL)
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
        finish = getattr(completion.choices[0], "finish_reason", "unknown")
        # "Usable" requires both: content channel emitted something AND
        # the model didn't get cut off at max_tokens. Without the
        # finish-reason check, vLLM's constrained-JSON decoder emitting
        # *partial* JSON before max_tokens hits would short-circuit
        # this retry and return a partial completion that the caller's
        # outer JSON-parse loop then re-retries at a different layer.
        # Same conflation as the one fixed in ``_parse_choices``;
        # decouple it here too so all helpers agree on what "usable"
        # means.
        if content is not None and finish != "length":
            return completion

        comp_tokens = 0
        if hasattr(completion, "usage") and completion.usage:
            comp_tokens = getattr(completion.usage, "completion_tokens", 0) or 0

        # Transient retry with delay
        if attempt < retries:
            delay = 0.5 * (attempt + 1)
            logger.warning(
                "vLLM returned bad response for {} (attempt {}/{}, "
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
                "vLLM returned bad response for {} after {} attempts "
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

    # Transient network/server errors (server stall, mid-request timeout,
    # dropped connection, vLLM saturation 5xx/429) flow through one path:
    # retry up to _VLLM_RETRIES times in the loop below, then raise
    # LLMConnectionError. The outer ``except Exception`` only catches
    # non-transient unexpected errors — see below.
    #
    # Catch list rationale:
    # - ``APIConnectionError`` (openai-SDK parent of APITimeoutError) and
    #   ``httpx.TimeoutException`` / ``httpx.ConnectError`` match the
    #   project pattern in ``rate_limiter._fetch_with_retry`` and
    #   ``web._classify_http_exception``.
    # - ``InternalServerError`` (5xx) and ``RateLimitError`` (429) are
    #   how vLLM signals queue / scheduler / admission backpressure.
    #   Without retrying these, a saturated vLLM silently demotes
    #   responses to None — a counter-only error that never surfaces in
    #   logs (the source of the "144 errors / 7 warnings" discrepancy
    #   in the v10 saturation run).
    # 4xx errors (BadRequestError, NotFoundError, AuthenticationError,
    # …) are programmer errors and intentionally NOT retried.
    import httpx as _httpx
    from openai import (
        APIConnectionError as _APIConnectionError,
        InternalServerError as _InternalServerError,
        RateLimitError as _RateLimitError,
    )

    from sciwrite_lint.exceptions import LLMConnectionError

    _TRANSIENT_NET_ERRS: tuple[type[BaseException], ...] = (
        _APIConnectionError,
        _InternalServerError,
        _RateLimitError,
        _httpx.TimeoutException,
        _httpx.ConnectError,
    )

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
                # Mark length-truncation regardless of whether ``raw`` is
                # None or partial. vLLM with constrained-JSON decoding
                # often emits a *partial* JSON before hitting max_tokens
                # — raw is non-empty, ``json.loads`` fails, but the
                # underlying problem is still "no room for the answer."
                # Treating those as plain JSONDecodeError used to skip
                # the thinking-ladder step-down and waste a retry on the
                # same prompt at the same thinking budget (observed in
                # FullPaperIssue give-ups: medium → backoff at medium
                # → step to low → length again → give up at low).
                if finish == "length":
                    any_length_truncation = True
                if raw is None:
                    parsed_list.append(None)
                    continue
                try:
                    parsed_list.append(json.loads(raw))
                    valid_count += 1
                except json.JSONDecodeError:
                    parsed_list.append(None)
            return parsed_list, any_length_truncation, valid_count

        # Network-level transient errors share the same retry budget as
        # content errors. On the final attempt we convert directly to
        # LLMConnectionError — no re-raise into the outer except, no
        # double-counting in usage stats, no second isinstance check.
        for attempt in range(_VLLM_RETRIES + 1):
            kwargs = _build_kwargs(current_thinking)
            try:
                completion = await client.chat.completions.create(**kwargs)
            except _TRANSIENT_NET_ERRS as net_err:
                run = _usage_current()
                if run:
                    run.vllm.record(0.0, error=True, error_type=type(net_err).__name__)
                if attempt < _VLLM_RETRIES:
                    delay = 2.0 * (attempt + 1)
                    logger.warning(
                        "vLLM transient network error for {} (attempt {}/{}, "
                        "thinking={}): {}: {} — retrying in {:.1f}s",
                        schema_name,
                        attempt + 1,
                        _VLLM_RETRIES + 1,
                        current_thinking,
                        type(net_err).__name__,
                        net_err,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise LLMConnectionError(
                    f"vLLM at {config.llm_endpoint} became unreachable mid-request: "
                    f"{type(net_err).__name__}: {net_err}\n"
                    "Check server: sciwrite-lint containers status"
                ) from net_err
            parsed_list, any_length, valid_count = _parse_choices(completion)

            if valid_count > 0:
                # At least one good choice — success. For n=1 this is
                # the same as before. For n>1, failed choices stay as
                # None in the returned list.
                _record_call(completion)
                if attempt > 0:
                    # Auditable record that the retry harness actually
                    # produced a usable answer — without this, the log
                    # only shows failures and the operator can't tell
                    # which retried calls recovered vs. were dropped.
                    logger.info(
                        "vLLM call for {} recovered on attempt {}/{} "
                        "(thinking={}, valid_choices={}/{})",
                        schema_name,
                        attempt + 1,
                        _VLLM_RETRIES + 1,
                        current_thinking,
                        valid_count,
                        n,
                    )
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
    except LLMConnectionError:
        # Already shaped above; let it propagate to the caller.
        raise
    except Exception as e:
        # Non-transient unexpected error — record once, log loudly, and
        # swallow (callers tolerate a None / [None]*n result). Logged at
        # WARNING (not DEBUG) so any error that increments
        # ``run.vllm.errors`` is also visible in the log stream — the
        # invariant is "every counted error is also a logged event." 4xx
        # programmer errors and unexpected exceptions land here.
        from sciwrite_lint.usage import current as _usage_current

        run = _usage_current()
        if run:
            run.vllm.record(0.0, error=True, error_type=type(e).__name__)
        logger.warning(
            "vLLM call for {} failed with non-transient error {}: {}",
            schema_name,
            type(e).__name__,
            e,
        )
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
    concurrency: int | None = None,
    size_class: str = "heavy",
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

    Concurrency. In-flight HTTP requests are capped to keep vLLM's
    queue near-empty so the openai client's blind read timeout
    (``config.llm_timeout``) only spans prefill + decode (both
    bounded) and not queue wait. Two paths:

    - Static (default): ``asyncio.Semaphore`` at
      ``concurrency`` if provided, else ``config.llm_max_concurrency``.
    - Dynamic (``config.use_dynamic_concurrency``): a
      ``DynamicConcurrencyController`` resizes the cap from observed
      ``/metrics`` so submission rate matches vLLM's admission rate
      (``requests_waiting`` stays under tolerance). See
      ``sciwrite_lint/llm/concurrency_optimizer/`` for the implementation.

    Per-caller hint. The default is sized for the heaviest caller
    (``ref_internal_checks`` full-paper queries at ~30K-token prompts).
    Lighter callers (small-prompt LLM checks, claim taxonomy) can pass
    ``concurrency=`` higher to amortize wall-time, since KV-cache
    pressure scales with ``concurrency × per-request KV tokens`` —
    not request count alone.

    The cap is per-batch, not process-global. The ``max(1, …)`` clamp
    prevents a misconfigured 0 / negative value from deadlocking.
    """
    from openai import AsyncOpenAI

    config = config or LintConfig()
    client = AsyncOpenAI(
        base_url=config.llm_endpoint,
        api_key="dummy",
        timeout=config.llm_timeout,
    )

    # Cap concurrent in-flight requests so vLLM's queue never sees the
    # full batch at once. Per-caller ``concurrency`` argument wins over
    # ``config.llm_max_concurrency`` so callers with small prompts can
    # opt into higher concurrency (the global default is sized for the
    # heaviest caller).
    effective_cap = (
        concurrency if concurrency is not None else config.llm_max_concurrency
    )
    effective_cap = max(1, effective_cap)

    # Never push past vLLM's admission ceiling — see effective_max_concurrency.
    from sciwrite_lint.vllm.vllm_server import effective_max_concurrency

    effective_cap = effective_max_concurrency(
        config, effective_cap, label=f"text-{size_class}"
    )

    from sciwrite_lint.llm.concurrency_optimizer import (
        ControllerParams,
        concurrency_slot,
    )

    ctrl_params = ControllerParams(
        target_kv_lo=config.concurrency_target_kv_lo,
        target_kv_grow=config.concurrency_target_kv_grow,
        target_kv_hi=config.concurrency_target_kv_hi,
    )

    # ``size_class`` keys the dynamic-controller registry (one shared
    # controller per partition). The default "heavy" matches the original
    # full-paper caller; lighter callers pass "medium" / "small" so they
    # share a controller with other lighter callers and don't fight over
    # the same KV-pressure model as the 30K-token batch.
    from sciwrite_lint.llm.concurrency_optimizer import SizeClass as _SizeClass

    typed_size_class: _SizeClass = size_class  # type: ignore[assignment]

    async with concurrency_slot(
        use_dynamic=config.use_dynamic_concurrency,
        endpoint=config.llm_endpoint,
        size_class=typed_size_class,
        static_cap=effective_cap,
        label=f"text-{size_class}",
        params=ctrl_params,
    ) as _slot:

        async def _bounded(
            sys: str, usr: str, sch: dict, name: str
        ) -> dict | None | list[dict | None]:
            async with _slot():
                return await llm_query(
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

        try:
            tasks = [_bounded(sys, usr, sch, name) for sys, usr, sch, name in queries]
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
