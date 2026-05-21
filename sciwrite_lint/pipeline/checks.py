"""Stage 1: manuscript + local-LLM checks (the registered check engine).

``run_text_checks`` runs every non-LLM, non-reference-db check.
``run_llm_checks_batched`` dispatches the ``local-llm`` checks through
the batched vLLM runner so one query can cover multiple checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.checks._diagnostics import internal_error_finding
from sciwrite_lint.config import LintConfig
from sciwrite_lint.exceptions import LLMConnectionError
from sciwrite_lint.models import Finding


def run_text_checks(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Run all manuscript-engine checks (CPU-bound, no I/O). Returns findings."""
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_checks

    ensure_checks_loaded()
    findings: list[Finding] = []

    for meta, fn in get_checks(config=config):
        if meta.category in ("reference-db", "local-llm"):
            continue
        try:
            check_findings = fn(tex_path, config)
            for f in check_findings:
                override = config.effective_severity(meta.id, meta.severity)
                if override != f.level:
                    f.level = override  # type: ignore[assignment]
            findings.extend(check_findings)
        except Exception as e:
            logger.warning(f"Check {meta.id} skipped: {e}")
            findings.append(internal_error_finding(meta.id, e))

    return findings


async def run_llm_checks_batched_impl(
    checks: list[tuple],
    tex_path: Path,
    config: LintConfig,
) -> list[Finding]:
    """Run all LLM checks batched by (thinking, temperature, n_samples).

    All LLM checks must implement the build_queries/process_results protocol:
    - ``build_queries(tex_path, config)`` returns either a bare
      ``list[(system, user, schema, name)]`` (stateless check) or a
      ``(queries, state)`` 2-tuple where ``queries`` is the list above
      and ``state`` is any object the check needs to keep between
      ``build_queries`` and ``process_results``.
    - ``process_results(results, state=...)`` returns ``list[Finding]``.
      ``state`` is keyword-only so stateless checks can keep accepting
      one positional argument.

    State is plumbed per call so concurrent papers do not race. The
    earlier protocol stored state on a ``_build_queries._state``
    function attribute — a single shared slot that two paper-runs would
    clobber when batched.

    Each check can set ``thinking`` ("off", "low", "medium", "high") to
    control chain-of-thought reasoning, ``temperature`` (float, or None
    for the model default) to control sampling, and ``n_samples`` (int,
    default 1) to request multiple samples per query for self-consistency
    voting. Queries are grouped by the ``(thinking, temperature,
    n_samples)`` triple so each group runs as a single batch call with
    the correct sampling regime.

    When n_samples > 1, each query contributes ``n_samples`` consecutive
    entries to its check's result slice — process_results is responsible
    for slicing in groups of n_samples and voting.
    """
    from sciwrite_lint.llm_utils import SMALL_PROMPT_CONCURRENCY, llm_query_batch

    # Phase 1: collect queries, grouped by (thinking, temperature, n). max_tokens
    # is a fixed per-model constant (see VLLM_MODELS); output length is
    # bounded by Pydantic schema constraints, not per-query overrides.
    BatchKey = tuple[str, float | None, int]
    queries_by_key: dict[BatchKey, list[tuple[str, str, dict, str]]] = {}
    # Slice entry: (meta, fn, key, start_query_idx, query_count, n_samples, state).
    # The result slice for this check is
    #   key_results[start_query_idx * n : (start_query_idx + query_count) * n]
    check_slices: list[tuple[Any, Any, BatchKey, int, int, int, Any]] = []
    build_failures: list[Finding] = []

    for meta, fn in checks:
        build = getattr(fn, "build_queries", None)
        if build is None:
            raise RuntimeError(f"LLM check {meta.id} missing build_queries")
        try:
            build_result = build(tex_path, config)
            # ``build`` returns either ``queries`` or ``(queries, state)``.
            # The state-bearing form lets each paper carry per-call
            # context (line numbers, sentence lookups, etc.) into
            # ``process_results`` without sharing a function attribute.
            if (
                isinstance(build_result, tuple)
                and len(build_result) == 2
                and isinstance(build_result[0], list)
            ):
                queries, state = build_result
            else:
                queries = build_result
                state = None
            thinking = getattr(fn, "thinking", "off")
            temperature = getattr(fn, "temperature", None)
            n_samples = int(getattr(fn, "n_samples", 1))
            key: BatchKey = (thinking, temperature, n_samples)
            if key not in queries_by_key:
                queries_by_key[key] = []
            start = len(queries_by_key[key])
            queries_by_key[key].extend(queries)
            check_slices.append((meta, fn, key, start, len(queries), n_samples, state))
        except Exception as e:
            logger.error(f"Check {meta.id} build_queries failed: {e}")
            build_failures.append(internal_error_finding(meta.id, e))

    # Phase 2: one batch call per (thinking, temperature, n) group
    results_by_key: dict[BatchKey, list[dict | None]] = {}
    for key, batch_queries in queries_by_key.items():
        thinking, temperature, n_samples = key
        if batch_queries:
            try:
                results_by_key[key] = await llm_query_batch(
                    batch_queries,
                    config=config,
                    thinking=thinking,
                    temperature=temperature,
                    n=n_samples,
                    concurrency=SMALL_PROMPT_CONCURRENCY,
                    size_class="small",
                )
            except LLMConnectionError:
                raise
            except Exception as e:
                logger.error(
                    f"LLM batch query failed (thinking={thinking}, "
                    f"temperature={temperature}, n={n_samples}): {e}"
                )
                results_by_key[key] = [None] * (len(batch_queries) * n_samples)

    # Phase 3: distribute results to each check. Slice bounds scale by
    # n_samples — each query contributed n consecutive result entries.
    findings: list[Finding] = []
    for meta, fn, key, start, count, n_samples, state in check_slices:
        try:
            if count > 0:
                process = getattr(fn, "process_results")
                key_results = results_by_key.get(key, [])
                result_start = start * n_samples
                result_end = (start + count) * n_samples
                check_results = key_results[result_start:result_end]
                # ``state`` is keyword-only on ``process_results`` so
                # stateless checks can keep their one-positional-arg
                # signature. Pass it whenever build_queries supplied a
                # non-None state — checks that opt into the new protocol
                # accept ``state=`` and the rest never see it.
                if state is not None:
                    check_findings = process(check_results, state=state)
                else:
                    check_findings = process(check_results)
            else:
                check_findings = []

            for f in check_findings:
                override = config.effective_severity(meta.id, meta.severity)
                if override != f.level:
                    f.level = override  # type: ignore[assignment]
            findings.extend(check_findings)
        except Exception as e:
            logger.error(f"Check {meta.id} failed: {e}")
            findings.append(internal_error_finding(meta.id, e))

    return build_failures + findings


async def run_llm_checks_batched(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Run all local-llm-engine checks via batched vLLM queries. Returns findings."""
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_checks

    ensure_checks_loaded()
    llm_checks = [
        (meta, fn)
        for meta, fn in get_checks(config=config)
        if meta.category == "local-llm"
    ]
    if not llm_checks:
        return []
    return await run_llm_checks_batched_impl(llm_checks, tex_path, config)
