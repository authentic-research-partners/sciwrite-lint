"""vLLM Prometheus metrics fetching — thin re-export.

The canonical implementation lives in
``sciwrite_lint.llm.concurrency_optimizer.vllm_metrics`` so the
concurrency-optimizer package can be lifted out of this codebase as a
single directory. This module re-exports ``fetch_metrics`` for
existing callers (CLI monitor, dev scripts) that import from the older
``sciwrite_lint.vllm.metrics`` path; it also keeps the
monitor-specific ``fetch_metrics_summary`` formatter that is not part
of the portable subset.
"""

from __future__ import annotations

from sciwrite_lint.llm.concurrency_optimizer.vllm_metrics import fetch_metrics

__all__ = ["fetch_metrics", "fetch_metrics_summary"]


def fetch_metrics_summary(endpoint: str) -> str | None:
    """One-line summary string for ``containers status``.

    Returns e.g. ``"max_seq: 20,000, KV cache: 0.3% used, 7145 blocks"``
    or ``None`` on failure.
    """
    m = fetch_metrics(endpoint)
    if not m:
        return None

    info: dict[str, str] = {}
    if "max_seq" in m:
        info["max_seq"] = f"{m['max_seq']:,}"
    if "kv_cache_pct" in m:
        pct = m["kv_cache_pct"]
        assert isinstance(pct, float)
        info["KV cache"] = f"{pct:.1%} used"
    if "num_gpu_blocks" in m:
        info["KV blocks"] = str(m["num_gpu_blocks"])
    if "gpu_util_cap" in m:
        info["GPU util cap"] = str(m["gpu_util_cap"])

    return ", ".join(f"{k}: {v}" for k, v in info.items()) if info else None
