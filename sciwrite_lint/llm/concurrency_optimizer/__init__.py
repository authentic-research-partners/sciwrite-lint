"""Dynamic concurrency optimizer for vLLM call sites.

Sizes the application-side ``Semaphore`` against vLLM's KV cache pool
and observed queue depth so the server queue stays empty during a
batch. With queue depth ~= 0 the OpenAI client's blind read timeout
(``config.llm_timeout``) only spans prefill+decode (both bounded),
instead of queue+prefill+decode.
"""

from __future__ import annotations

from .compute_cap import (
    SIZE_CLASS_PROFILES,
    SizeClass,
    SizeClassProfile,
    compute_cap,
)
from .controller import DynamicConcurrencyController, ResizableSemaphore
from .decide import ControllerParams, ControllerState, Decision, Sample, decide
from .host_metrics import HostSnapshot, gather_host_snapshot
from .metrics_probe import KVPool, probe_kv_pool
from .registry import active_keys, reset_for_tests, shared_controller
from .telemetry import (
    Service,
    TelemetryRow,
    cleanup_partition,
    list_active_streams,
    read_recent,
    write_sample,
)
from .wiring import SlotFactory, concurrency_slot

__all__ = [
    "SIZE_CLASS_PROFILES",
    "ControllerParams",
    "ControllerState",
    "Decision",
    "DynamicConcurrencyController",
    "HostSnapshot",
    "KVPool",
    "ResizableSemaphore",
    "Sample",
    "Service",
    "SizeClass",
    "SizeClassProfile",
    "SlotFactory",
    "TelemetryRow",
    "active_keys",
    "cleanup_partition",
    "compute_cap",
    "concurrency_slot",
    "decide",
    "gather_host_snapshot",
    "list_active_streams",
    "probe_kv_pool",
    "read_recent",
    "reset_for_tests",
    "shared_controller",
    "write_sample",
]
