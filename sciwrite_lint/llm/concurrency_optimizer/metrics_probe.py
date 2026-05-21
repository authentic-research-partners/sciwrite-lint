"""Probe a running vLLM container for its KV cache pool.

Thin typed wrapper around the bundled ``vllm_metrics.fetch_metrics`` —
that module owns the Prometheus regex parsing; this one converts its
``dict`` output into a typed ``KVPool`` and raises loudly when the
required labels are missing.

Why delegate the regex work: a vLLM upgrade that renames a Prometheus
key only breaks one place in this package.
"""

from __future__ import annotations

from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, PositiveInt


class KVPool(BaseModel):
    """KV cache pool snapshot from a running vLLM endpoint."""

    kv_tokens: PositiveInt
    num_gpu_blocks: PositiveInt
    block_size: PositiveInt
    max_model_len: NonNegativeInt = 0
    kv_cache_pct: NonNegativeFloat = 0.0
    requests_running: NonNegativeInt = 0
    requests_waiting: NonNegativeInt = 0


def probe_kv_pool(endpoint: str) -> KVPool:
    """Fetch the KV cache pool snapshot from a running vLLM endpoint.

    *endpoint* is the OpenAI-compatible base, e.g.
    ``http://localhost:5001/v1``.

    Raises
    ------
    RuntimeError
        If ``/metrics`` is unreachable, or if it does not expose
        ``num_gpu_blocks`` / ``block_size`` (the ``cache_config:info``
        gauge added in vLLM 0.5+). The error message names the start
        command so the operator does not have to look it up.
    """
    from .vllm_metrics import fetch_metrics

    metrics = fetch_metrics(endpoint)
    if not metrics:
        raise RuntimeError(
            f"vLLM /metrics unreachable at {endpoint}. "
            "Start the container with: sciwrite-lint containers start"
        )
    if "num_gpu_blocks" not in metrics or "block_size" not in metrics:
        raise RuntimeError(
            f"vLLM at {endpoint} did not expose num_gpu_blocks / block_size "
            "via /metrics — the cache_config:info gauge is required. "
            "Check vLLM version with: sciwrite-lint containers status"
        )

    num_gpu_blocks = int(metrics["num_gpu_blocks"])
    block_size = int(metrics["block_size"])
    kv_tokens = num_gpu_blocks * block_size

    return KVPool(
        kv_tokens=kv_tokens,
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        max_model_len=int(metrics.get("max_seq", 0)),
        kv_cache_pct=float(metrics.get("kv_cache_pct", 0.0)),
        requests_running=int(metrics.get("requests_running", 0)),
        requests_waiting=int(metrics.get("requests_waiting", 0)),
    )
