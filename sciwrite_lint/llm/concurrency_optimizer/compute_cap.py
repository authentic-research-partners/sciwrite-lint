"""Pure function: KV pool size + size class -> Semaphore cap.

The math:

    cap = floor(kv_tokens * safety_factor / typical_prompt_tokens)
    cap = clip(cap, lower_bound, upper_bound)

``typical_prompt_tokens`` and ``safety_factor`` are calibration constants
measured against observed production behavior (see ``README.md``). They
are not magic numbers; if the model or prompt mix changes, re-measure.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt

SizeClass = Literal["small", "medium", "heavy", "vision"]


class SizeClassProfile(BaseModel):
    """Calibration constants for one size class."""

    typical_prompt_tokens: PositiveInt
    safety_factor: float = Field(gt=0.0, le=1.0)


# Calibration anchors (per-class typical-prompt sizes and safety factor):
# the ``typical_prompt_tokens`` numbers below are sized for the heaviest
# caller in each class. The ``safety_factor`` (0.6–0.7) buffers per-class
# capacity against vLLM's preemption zone — the controller resizes inside
# this envelope based on observed ``kv_cache_usage`` and queue depth.
SIZE_CLASS_PROFILES: dict[SizeClass, SizeClassProfile] = {
    "small": SizeClassProfile(typical_prompt_tokens=1_000, safety_factor=0.7),
    "medium": SizeClassProfile(typical_prompt_tokens=5_000, safety_factor=0.7),
    "heavy": SizeClassProfile(typical_prompt_tokens=30_000, safety_factor=0.7),
    "vision": SizeClassProfile(typical_prompt_tokens=3_500, safety_factor=0.6),
}


def compute_cap(
    *,
    kv_tokens: int,
    size_class: SizeClass,
    override: int | None = None,
    upper_bound: int = 100,
    lower_bound: int = 1,
) -> int:
    """Return the application-side Semaphore cap for one size class.

    Parameters
    ----------
    kv_tokens
        Total KV cache tokens reported by vLLM at startup
        (``num_gpu_blocks * block_size``).
    size_class
        Caller's hint about prompt size class.
    override
        Operator escape hatch. If non-None, returned unchanged so a TOML
        knob always wins over the optimizer.
    upper_bound, lower_bound
        Hard clamps on the returned value.

    Raises
    ------
    ValueError
        If ``kv_tokens <= 0`` (when no override is provided), or if the
        bounds are inconsistent.
    """
    if override is not None:
        return override
    if kv_tokens <= 0:
        raise ValueError(f"kv_tokens must be positive, got {kv_tokens}")
    if lower_bound < 1:
        raise ValueError(f"lower_bound must be >= 1, got {lower_bound}")
    if upper_bound < lower_bound:
        raise ValueError(
            f"upper_bound ({upper_bound}) must be >= lower_bound ({lower_bound})"
        )

    profile = SIZE_CLASS_PROFILES[size_class]
    raw = int(kv_tokens * profile.safety_factor / profile.typical_prompt_tokens)
    return max(lower_bound, min(upper_bound, raw))
