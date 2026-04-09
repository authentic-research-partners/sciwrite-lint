"""vLLM Prometheus metrics fetching.

Parses the ``/metrics`` endpoint exposed by vLLM and returns a flat dict
of key–value pairs.  Used by the CLI monitor (``sciwrite-lint vllm monitor``)
and the Streamlit UI dashboard.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from loguru import logger


def _require_http_url(url: str) -> str:
    """Validate *url* uses http/https; raise ValueError otherwise.

    Guards ``urlopen`` against file:// and other non-network schemes that
    could be injected via a misconfigured endpoint.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(f"Refusing non-http(s) URL scheme: {scheme!r}")
    return url


# Patterns: (regex, result-key).  Each regex must have one capture group.
# Value pattern handles scientific notation (e.g. 1.572691e+06) from Prometheus.
_NUM = r"[\d.]+(?:[eE][+-]?\d+)?"
_GAUGE_PATTERNS: list[tuple[str, str]] = [
    (rf"vllm:kv_cache_usage_perc\{{[^}}]*\}}\s+({_NUM})", "kv_cache_pct"),
    (rf"vllm:num_requests_running\{{[^}}]*\}}\s+({_NUM})", "requests_running"),
    (rf"vllm:num_requests_waiting\{{[^}}]*\}}\s+({_NUM})", "requests_waiting"),
    (rf"vllm:num_requests_swapped\{{[^}}]*\}}\s+({_NUM})", "requests_swapped"),
    (rf"vllm:prompt_tokens_total\{{[^}}]*\}}\s+({_NUM})", "prompt_tokens_total"),
    (
        rf"vllm:generation_tokens_total\{{[^}}]*\}}\s+({_NUM})",
        "generation_tokens_total",
    ),
    # Prefix cache (KV cache hit tracking)
    (rf"vllm:prefix_cache_hits_total\{{[^}}]*\}}\s+({_NUM})", "prefix_cache_hits"),
    (
        rf"vllm:prefix_cache_queries_total\{{[^}}]*\}}\s+({_NUM})",
        "prefix_cache_queries",
    ),
    # Preemptions (KV evictions under memory pressure)
    (rf"vllm:num_preemptions_total\{{[^}}]*\}}\s+({_NUM})", "num_preemptions"),
    # Latency histograms (sum/count → averages)
    (
        rf"vllm:e2e_request_latency_seconds_sum\{{[^}}]*\}}\s+({_NUM})",
        "e2e_latency_sum",
    ),
    (
        rf"vllm:e2e_request_latency_seconds_count\{{[^}}]*\}}\s+({_NUM})",
        "e2e_latency_count",
    ),
    (rf"vllm:time_to_first_token_seconds_sum\{{[^}}]*\}}\s+({_NUM})", "ttft_sum"),
    (rf"vllm:time_to_first_token_seconds_count\{{[^}}]*\}}\s+({_NUM})", "ttft_count"),
    (rf"vllm:inter_token_latency_seconds_sum\{{[^}}]*\}}\s+({_NUM})", "itl_sum"),
    (rf"vllm:inter_token_latency_seconds_count\{{[^}}]*\}}\s+({_NUM})", "itl_count"),
]

_REQUEST_REASONS = ("stop", "length", "abort", "error")


def fetch_metrics(endpoint: str) -> dict[str, float | int | str]:
    """Fetch all vLLM metrics from Prometheus endpoint and ``/v1/models``.

    *endpoint* is the OpenAI-compatible base, e.g. ``http://localhost:5001/v1``.

    Returns a dict with keys like ``kv_cache_pct``, ``requests_running``,
    ``prefix_cache_hits``, ``e2e_latency_sum``, ``req_success_stop``, etc.
    Returns an empty dict if the server is unreachable.
    """
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    result: dict[str, float | int | str] = {}

    try:
        url = _require_http_url(f"{base}/metrics")
        # Scheme is validated by _require_http_url() above.
        with urllib.request.urlopen(url, timeout=3) as resp:  # nosec B310
            text = resp.read().decode()
    except Exception as e:
        logger.debug(f"vLLM metrics fetch failed ({type(e).__name__}: {e})")
        return result

    # Gauge and counter metrics
    for pattern, key in _GAUGE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            result[key] = float(m.group(1))

    # Request outcomes by finished_reason
    for reason in _REQUEST_REASONS:
        m = re.search(
            rf'vllm:request_success_total\{{[^}}]*finished_reason="{reason}"[^}}]*\}}\s+([\d.]+)',
            text,
        )
        if m:
            result[f"req_success_{reason}"] = float(m.group(1))

    # Label-embedded config values
    m = re.search(r'num_gpu_blocks="(\d+)"', text)
    if m:
        result["num_gpu_blocks"] = int(m.group(1))
    m = re.search(r'block_size="(\d+)"', text)
    if m:
        result["block_size"] = int(m.group(1))
    m = re.search(r'gpu_memory_utilization="([\d.]+)"', text)
    if m:
        result["gpu_util_cap"] = float(m.group(1))

    # max_model_len from models API
    try:
        url = _require_http_url(f"{endpoint}/models")
        # Scheme is validated by _require_http_url() above.
        with urllib.request.urlopen(url, timeout=3) as resp:  # nosec B310
            data = json.loads(resp.read().decode())
            for model in data.get("data", []):
                if "max_model_len" in model:
                    result["max_seq"] = model["max_model_len"]
                    break
    except Exception as e:
        logger.debug(f"vLLM /models fetch failed ({type(e).__name__}: {e})")

    return result


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
