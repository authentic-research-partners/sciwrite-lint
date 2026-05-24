"""Per-prompt cache for manuscript-LLM checks.

Wraps the batched LLM runner so each `(system, user, schema_name, model,
cache_version)` tuple is memoized in `workspace.db.manuscript_check_cache`.
A clean rerun against unchanged prose hits every cached row; editing one
sentence misses exactly that sentence's row (and any sibling row whose
paragraph context shifted).

## When to bump MANUSCRIPT_CHECK_CACHE_VERSION

The constant below is the global version stamp on every cached
manuscript-check row. Bump it (e.g. ``"1"`` → ``"2"``) in any release
that changes:

  * a check's ``_SYSTEM_*`` / ``_USER_*`` prompt template wording;
  * the Pydantic ``response_format`` schema for a check (field names,
    types, or schema name);
  * the way a result is interpreted by ``process_results`` (e.g. a new
    "confidence" interpretation that would yield different findings from
    the same LLM output);
  * thinking-budget defaults or other sampling-parameter defaults that
    aren't passed per-batch.

If unsure: bump it. A spurious bump only forces one extra full rerun;
a missed bump leaves users on stale findings indefinitely.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.llm_utils import VLLM_DEFAULT_MODEL
from sciwrite_lint.references.workspace_db import (
    get_db,
    lookup_manuscript_check_cache,
    save_manuscript_check_cache,
)

# Bump on any change to manuscript-check prompts or schemas (see module
# docstring for the exact triggers).
MANUSCRIPT_CHECK_CACHE_VERSION = "1"


def manuscript_prompt_hash(
    system: str,
    user: str,
    schema_name: str,
    model: str,
    cache_version: str,
) -> str:
    """SHA-256 of the canonical input tuple.

    Joins with NUL so substring boundaries are unambiguous (no real
    prompt contains NUL). No normalization — a one-byte edit must miss.
    """
    canonical = "\x00".join((system, user, schema_name, model, cache_version))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_model(config: LintConfig) -> str:
    return f"vllm:{config.llm_model or VLLM_DEFAULT_MODEL}"


def split_cached_and_misses(
    queries: list[tuple[str, str, dict, str]],
    check_ids: list[str],
    config: LintConfig,
    *,
    cache_version: str = MANUSCRIPT_CHECK_CACHE_VERSION,
) -> tuple[
    list[dict | None],
    list[tuple[str, str, dict, str]],
    list[int],
    list[str],
    list[str],
]:
    """Filter queries through the cache.

    Returns a 5-tuple:

    * ``results``: pre-populated list aligned with ``queries``;
      cached entries hold the decoded dict, miss positions hold ``None``.
    * ``miss_queries``: the subset of ``queries`` that did not hit cache.
    * ``miss_indices``: positions in the original ``queries`` list of
      each miss, in order. Used to merge LLM results back in.
    * ``miss_check_ids``: ``check_ids`` aligned with ``miss_queries``.
    * ``miss_prompt_hashes``: prompt hashes aligned with ``miss_queries``,
      so the caller can save back without recomputing.
    """
    assert len(queries) == len(check_ids), (
        f"queries ({len(queries)}) and check_ids ({len(check_ids)}) length mismatch"
    )
    model = _resolve_model(config)
    references_dir = config.effective_references_dir()

    # Compute hashes up front, then bulk-lookup.
    prompt_hashes = [
        manuscript_prompt_hash(q[0], q[1], q[3], model, cache_version) for q in queries
    ]
    lookup_keys = list(zip(prompt_hashes, check_ids, strict=True))

    with get_db(references_dir) as conn:
        cached = lookup_manuscript_check_cache(conn, lookup_keys)

    results: list[dict | None] = [None] * len(queries)
    miss_queries: list[tuple[str, str, dict, str]] = []
    miss_indices: list[int] = []
    miss_check_ids: list[str] = []
    miss_prompt_hashes: list[str] = []

    for i, (ph, cid) in enumerate(lookup_keys):
        raw = cached.get((ph, cid))
        if raw is not None:
            try:
                results[i] = json.loads(raw)
                continue
            except json.JSONDecodeError as e:
                # Cache row is corrupt — re-verify instead of trusting
                # garbage. Log so the next debug session can find it.
                logger.warning(
                    "manuscript-check cache row corrupt (check={}, hash={}): {}",
                    cid,
                    ph[:12],
                    e,
                )
        miss_queries.append(queries[i])
        miss_indices.append(i)
        miss_check_ids.append(cid)
        miss_prompt_hashes.append(ph)

    return results, miss_queries, miss_indices, miss_check_ids, miss_prompt_hashes


def save_misses(
    config: LintConfig,
    miss_prompt_hashes: list[str],
    miss_check_ids: list[str],
    miss_results: list[dict | None],
    *,
    cache_version: str = MANUSCRIPT_CHECK_CACHE_VERSION,
) -> int:
    """Persist deterministic LLM results that just came back fresh.

    Skips any miss whose result is ``None`` (LLM call failed / returned
    nothing). Returns the number of rows written.
    """
    model = _resolve_model(config)
    references_dir = config.effective_references_dir()
    entries: list[dict[str, Any]] = []
    for ph, cid, res in zip(
        miss_prompt_hashes, miss_check_ids, miss_results, strict=True
    ):
        if res is None:
            continue
        entries.append(
            {
                "prompt_hash": ph,
                "check_id": cid,
                "model": model,
                "cache_version": cache_version,
                "result_json": json.dumps(res, ensure_ascii=False),
            }
        )
    if not entries:
        return 0
    with get_db(references_dir) as conn:
        save_manuscript_check_cache(conn, entries)
    return len(entries)
