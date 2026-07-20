"""Stage 1: manuscript + local-LLM checks (the registered check engine).

``run_text_checks`` runs every non-LLM, non-reference-db check.
``run_llm_checks_batched`` dispatches the ``local-llm`` checks through
the batched vLLM runner so one query can cover multiple checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.checks._diagnostics import (
    internal_error_finding,
    source_unsupported_finding,
)
from sciwrite_lint.config import LintConfig
from sciwrite_lint.exceptions import LLMConnectionError
from sciwrite_lint.models import Finding


def run_text_checks(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Run all manuscript-engine checks (CPU-bound, no I/O). Returns findings.

    For a markdown manuscript, checks that declare
    ``supports_markdown=False`` (they parse LaTeX citation/cross-reference/
    figure syntax) are skipped and collected into a single
    ``source-unsupported`` system issue so the coverage gap is explicit.
    """
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_checks

    ensure_checks_loaded()
    findings: list[Finding] = []
    skipped_for_markdown: list[str] = []

    for meta, fn in get_checks(config=config):
        if meta.category in ("reference-db", "local-llm"):
            continue
        # The supports_markdown gate is enforced here because every check
        # that opts out is category="manuscript" (the only category this
        # runner handles) — they parse LaTeX syntax deterministically.
        # tests/test_markdown_manuscript.py guards that invariant so a
        # future local-llm opt-out can't slip the gate unnoticed.
        if config.is_markdown and not meta.supports_markdown:
            skipped_for_markdown.append(meta.id)
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

    if skipped_for_markdown:
        findings.append(source_unsupported_finding("markdown", skipped_for_markdown))

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

    Cache interaction: results route through ``manuscript_check_cache``
    when ``n_samples == 1`` and ``temperature`` is ``None`` or ``0.0``.
    A check with no explicit ``.temperature`` is still cache-eligible
    even though vLLM samples at the model default — the cache stamps
    the first valid sample as canonical. Pin ``temperature=0.0`` for
    bit-exact reuse (see ``prose_quality.py``).

    Manuscript-LLM checks currently default to ``n_samples=1``; the
    voting branch is reserved infrastructure kept on the measured
    speed-vs-precision trade-off (decode-bound at N>1).
    """
    from sciwrite_lint.llm.manuscript_cache import (
        save_misses as _cache_save_misses,
        split_cached_and_misses as _cache_split,
    )
    from sciwrite_lint.llm_utils import SMALL_PROMPT_CONCURRENCY, llm_query_batch

    # Phase 1: collect queries, grouped by (thinking, temperature, n). max_tokens
    # is a fixed per-model constant (see VLLM_MODELS); output length is
    # bounded by Pydantic schema constraints, not per-query overrides.
    BatchKey = tuple[str, float | None, int]
    queries_by_key: dict[BatchKey, list[tuple[str, str, dict, str]]] = {}
    # Per-query check_id aligned with queries_by_key — needed to key the
    # manuscript-check cache and to distinguish two checks that might
    # accidentally emit the same prompt.
    check_ids_by_key: dict[BatchKey, list[str]] = {}
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
                check_ids_by_key[key] = []
            start = len(queries_by_key[key])
            queries_by_key[key].extend(queries)
            check_ids_by_key[key].extend([meta.id] * len(queries))
            check_slices.append((meta, fn, key, start, len(queries), n_samples, state))
        except Exception as e:
            logger.error(f"Check {meta.id} build_queries failed: {e}")
            build_failures.append(internal_error_finding(meta.id, e))

    # Phase 2: one batch call per (thinking, temperature, n) group. Two
    # modes drive whether the manuscript-check cache is consulted:
    #   - single-answer mode (n_samples == 1 and temperature is None|0):
    #     the check wants one answer per prompt. Route through the cache —
    #     only queries whose canonical prompt hash misses reach vLLM, fresh
    #     results are written back. Note: temperature=None still samples at
    #     the model default at the vLLM layer; the cache stamps the first
    #     valid sample as canonical. Pin temperature=0.0 for bit-exact reuse.
    #   - voting mode (n_samples > 1 or explicit temperature > 0): the check
    #     needs N fresh independent samples each run to aggregate (self-
    #     consistency vote). Bypass the cache on both read and write —
    #     storing one sample would defeat the vote.
    results_by_key: dict[BatchKey, list[dict | None]] = {}
    cache_hits = 0
    cache_misses = 0
    for key, batch_queries in queries_by_key.items():
        thinking, temperature, n_samples = key
        if not batch_queries:
            continue
        single_answer = n_samples == 1 and (temperature is None or temperature == 0.0)
        if single_answer:
            (
                merged,
                miss_queries,
                miss_indices,
                miss_check_ids,
                miss_prompt_hashes,
            ) = _cache_split(batch_queries, check_ids_by_key[key], config)
            cache_hits += sum(1 for r in merged if r is not None)
            cache_misses += len(miss_queries)
        else:
            merged = [None] * len(batch_queries)
            miss_queries = list(batch_queries)
            miss_indices = list(range(len(batch_queries)))
            miss_check_ids = []
            miss_prompt_hashes = []

        if miss_queries:
            try:
                fresh = await llm_query_batch(
                    miss_queries,
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
                fresh = [None] * (len(miss_queries) * n_samples)
            # Single-answer mode (n_samples == 1) is the only path that
            # caches: fresh entries align 1:1 with miss_queries. Voting
            # mode (n_samples > 1) skips the cache, so the merge below
            # stays correct.
            if single_answer:
                for pos, val in zip(miss_indices, fresh, strict=True):
                    merged[pos] = val
                _cache_save_misses(config, miss_prompt_hashes, miss_check_ids, fresh)
                results_by_key[key] = merged
            else:
                # Voting mode: results scaled by n_samples (each query
                # produces n consecutive entries). Re-emit the flattened
                # list as-is for process_results to slice and vote.
                results_by_key[key] = fresh
        else:
            results_by_key[key] = merged

    if cache_hits or cache_misses:
        total = cache_hits + cache_misses
        pct = (100.0 * cache_hits / total) if total else 0.0
        logger.info(
            "Manuscript-check cache: {}/{} hits ({:.0f}%)",
            cache_hits,
            total,
            pct,
        )

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
