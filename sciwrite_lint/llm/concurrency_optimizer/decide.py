"""Pure decision function for the dynamic concurrency controller.

Stateless brain: ``decide(state, sample, params) -> Decision`` consumes
one fresh ``/metrics`` sample plus the rolling state and returns the
next cap to apply (and the updated state).

Separated from ``controller.py`` so the logic is testable without any
asyncio machinery — feed it scripted samples, assert on the decisions.

Signal precedence (most authoritative first):

    1. ``override`` is set                                      -> return override.
    2. preemption window sum > threshold (vLLM evicted)         -> shrink hard, no hysteresis.
    3. queue exceeds tolerance AND (KV > hi OR preemption_sum>0) -> shrink.
    4. KV utilization above hi                                  -> shrink (saturation risk).
    5. KV below lo AND we're saturating cap                     -> grow (real demand exists).
    6. Otherwise                                                -> hold.

Why queue alone is not a shrink signal: vLLM's ``requests_waiting``
gauge is the *server-side admission queue*. When the client sends
more than ``--max-num-seqs``, vLLM by design admits the excess into
this queue and drains it as the running batch frees slots. KV stays
healthy, no preemptions, throughput is fine — that is normal server
saturation, not backpressure. Treating it as "shrink now" causes a
chained ladder all the way to single digits while the server is
still processing its full batch. So queue depth must be paired with
a real saturation symptom (KV crossing the high band, or preemptions)
to qualify as backpressure.

Preemptions are vLLM's ground truth that we over-admitted last window:
when KV pressure forces it to evict a running sequence, that work is
discarded and recomputed later. Unlike ``waiting`` (which is a normal
admission queue and clears within ms) preemptions cost real seconds.
A non-zero ``preemption_delta`` between two consecutive ticks therefore
bypasses the consecutive-signal hysteresis and triggers an immediate
hard shrink (``preemption_shrink_factor``, default 0.5x), then locks
out grows for ``preemption_cooldown_ticks`` so we don't re-overshoot
the same equilibrium we just escaped.

Single-tenant tuning. This codebase is the only client of vLLM, so
/metrics readings reflect *our* load only. Two consequences:

- We do not grow the cap unless we're actually saturating it
  (``local_in_flight >= cap × min_utilization_for_grow``). If our
  workers aren't even asking for more slots, growing would be ritual.
- We can be lazy about hysteresis. The defaults are looser than a
  multi-tenant controller would need: 5 consecutive signals (not 2),
  queue tolerance 8 (not 3), and a [0.60, 0.80] dead band. Small
  oscillation is fine; perfect stability is not the goal.

Asymmetric thresholds. The grow trigger is 60% (``target_kv_lo``) but
the predictive grow aims at 70% (``target_kv_grow``). The shrink
ceiling is 80% (``target_kv_hi``). So the controller grows when below
60%, predictively jumps toward 70% utilization, then holds steady
across [60%, 80%], and only shrinks when KV crosses 80%. The 10-point
gap above the grow aim avoids flapping at equilibrium and keeps a
safe margin below vLLM's preemption zone (~85–90%).

Predictive grow. With single-tenancy we know /metrics reflects our own
load, so per-request KV ≈ ``kv_pct / requests_running``. The cap that
would equilibrate at the target is ``(target_kv_grow / kv_pct) ×
requests_running`` — a one-shot prediction, not a slow ramp.

Assumption underlying the prediction: per-request KV is roughly
constant within one workload (i.e. doubling the cap roughly doubles
KV utilization). This is true when each request has unique context
(cache-cold vision images, or text with paper-specific prompts). It
breaks down with synthetic workloads sharing a fixed prefix — vLLM's
prefix cache deduplicates and per-request KV shrinks with concurrency.
That regime would cause the controller to overshoot, but overshoot is
not critical: vLLM queues the excess, the controller observes
``requests_waiting > queue_tolerance`` on the next sample, and shrinks.
The system self-corrects within a few ticks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveInt


class Sample(BaseModel):
    """One ``/metrics`` snapshot fed to the controller, plus our own
    semaphore's in-flight count.

    Including ``local_in_flight`` is what makes the single-tenant
    assumption pay off: we know whether the cap is binding from our
    side, so we don't grow it speculatively when our own workers aren't
    saturating it.
    """

    kv_cache_pct: NonNegativeFloat
    requests_running: NonNegativeInt
    requests_waiting: NonNegativeInt
    local_in_flight: NonNegativeInt = 0
    # Delta in vLLM's cumulative preemption counter since the previous
    # tick. >0 means vLLM evicted a running sequence under KV pressure
    # — wasted compute, the strongest "shrink now" signal we can read.
    # Caller is responsible for tracking the cumulative value and
    # passing the delta; ``decide`` is stateless w.r.t. the counter.
    preemption_delta: NonNegativeInt = 0


class ControllerParams(BaseModel):
    """Tuning knobs for the decision function — single-tenant defaults."""

    target_kv_lo: float = Field(default=0.60, ge=0.0, le=1.0)
    target_kv_hi: float = Field(default=0.80, ge=0.0, le=1.0)
    ewma_alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    consecutive_signals: PositiveInt = 5
    max_step_factor: float = Field(default=0.15, gt=0.0, le=1.0)
    lower_bound: PositiveInt = 1
    upper_bound: PositiveInt = 100
    override: int | None = None
    # vLLM normally has a small standing queue (admitted-but-not-running
    # sequences). Default 8 is loose so transient bookkeeping doesn't
    # trigger a shrink — only sustained backpressure counts.
    queue_tolerance: NonNegativeInt = 8
    # Minimum fraction of the current cap that must be in-flight from
    # our own side before we consider growing it. Without this, the
    # controller will keep growing the cap during low-traffic phases
    # because /metrics shows low KV — even though our workers aren't
    # asking for more slots. Set to 0 only in multi-tenant deployments.
    min_utilization_for_grow: float = Field(default=0.8, ge=0.0, le=1.0)
    # Target KV utilization for the predictive grow path. When the grow
    # path fires, the next cap is set to ``(target_kv_grow / observed_kv)
    # × requests_running`` — the cap that would equilibrate at this
    # utilization given the observed per-request KV. Keep below
    # ``target_kv_hi`` so we land in-band rather than at the shrink
    # threshold; the gap (default 10 points: 70% grow aim, 80% shrink)
    # is the dead band that lets the controller hold steady around
    # equilibrium without flapping.
    target_kv_grow: float = Field(default=0.7, ge=0.0, le=1.0)
    # When ``preemption_delta > 0``, multiply current cap by this factor
    # (default halves it) and bypass hysteresis. Aggressive on purpose:
    # preemptions waste real compute, so we'd rather under-admit briefly
    # than continue to evict.
    preemption_shrink_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    # After a preemption-triggered shrink, lock out grow decisions for
    # this many subsequent ticks. Prevents the predictive grow from
    # immediately re-overshooting the same equilibrium that caused the
    # preemption. Holds + queue/kv-high shrinks still fire normally.
    preemption_cooldown_ticks: NonNegativeInt = 5
    # The cap that triggered a preemption is, by definition, unsafe.
    # We learn this and clamp future grows to ``preempt_safe_factor ×
    # cap_at_preempt`` (default 0.85 — 15 % below the value we know
    # broke). Prevents the grow path from rediscovering the same
    # over-admission ceiling tick after tick.
    preempt_safe_factor: float = Field(default=0.85, gt=0.0, le=1.0)
    # Practical floor applied to every shrink. Even under sustained
    # pressure, caps below this trim "one request at a time" — that's
    # churn, not pressure relief. The user would rather have requests
    # queue than have the cap collapse to single digits. Set above
    # ``lower_bound`` to take effect; below it has no effect.
    # Defaults to 1 (effectively no floor); the controller wires this
    # to ``math_cap // 2`` at startup — half of the math-based starting
    # cap for the current ``size_class``. Rationale: ``math_cap`` is
    # the conservative scenario default that *should* always be safe;
    # going below half of it means we've abandoned the scenario, which
    # is more pessimistic than any single-paper signal warrants.
    effective_min_cap: PositiveInt = 1
    # Number of recent ticks whose preemption deltas we sum to decide
    # if there's a sustained preemption signal. A single tick may miss
    # a preemption (poll race) or catch a transient one-off; summing a
    # short window smooths both. Default 5 ticks ≈ 10 s at 2 s polling.
    preemption_window_ticks: PositiveInt = 5
    # Threshold on the windowed sum. A windowed sum strictly greater
    # than this triggers a preempt shrink. Default 0 means "any
    # preemption in the window counts" — preemptions cost real compute
    # so we react fast. Raise to 2 or 3 if calibration shows vLLM emits
    # bookkeeping preemptions that aren't real overload.
    preemption_threshold: NonNegativeInt = 0
    # ``max_safe_cap`` learned from a preemption is meant as "this is
    # where vLLM ran out of room *under that workload*". For workloads
    # with bursty arrival (vision: many huge image prompts queued
    # together, then steady decode) the early burst can pin
    # ``max_safe_cap`` permanently at a value far below sustainable
    # steady-state — observed in cited-vision runs where 10 startup
    # preempts left the controller capped at 17 even with KV at 20 %
    # and GPU at 17 %. To self-correct, ``max_safe_cap`` is incremented
    # by this many slots per tick when conditions are healthy
    # (no preempts in window, past the cooldown, KV below
    # ``target_kv_lo``, current cap fully saturating). Default 1 = one
    # slot per polling interval (~2 s) so it takes ~minutes to creep
    # back to ``upper_bound``; tune higher for impatient probing or 0
    # to disable the decay (strict-ratchet behavior — cap never
    # recovers once tightened).
    max_safe_cap_decay_per_tick: NonNegativeInt = 1


class ControllerState(BaseModel):
    """Rolling state — produced by ``decide``, fed back on next call."""

    current_cap: PositiveInt
    smoothed_kv: NonNegativeFloat = 0.0
    smoothed_waiting: NonNegativeFloat = 0.0
    consecutive_above: NonNegativeInt = 0
    consecutive_below: NonNegativeInt = 0
    consecutive_queue: NonNegativeInt = 0
    samples_seen: NonNegativeInt = 0
    # Decremented each tick after a preemption shrink; while >0, grow
    # decisions are suppressed (queue/kv-high shrinks still fire).
    preemption_cooldown: NonNegativeInt = 0
    # Rolling window of recent ``preemption_delta`` values, oldest
    # first. Bounded to ``params.preemption_window_ticks`` by ``decide``.
    preemption_window: list[int] = Field(default_factory=list)
    # Learned upper ceiling. Set when a preemption shrink fires; the
    # grow path will not propose a cap above this. 0 means "no ceiling
    # learned yet" — grow respects only ``upper_bound``. Once set, this
    # only ever decreases (each new preemption tightens the ceiling).
    max_safe_cap: NonNegativeInt = 0


class Decision(BaseModel):
    """Output of one ``decide()`` call."""

    next_cap: PositiveInt
    state: ControllerState
    reason: str
    changed: bool


def _ewma(prev: float, new: float, alpha: float) -> float:
    """One-step exponential weighted moving average."""
    return alpha * new + (1.0 - alpha) * prev


def _update_smoothed(
    state: ControllerState, sample: Sample, params: ControllerParams
) -> tuple[float, float]:
    """Return next ``(smoothed_kv, smoothed_waiting)`` for one sample.

    On the very first sample (``samples_seen == 0``) we *prime* the
    EWMA with the raw value, otherwise the smoothed series would lag
    toward 0 and yield a spurious "below_lo" reading even when the
    system is already in-band. Used by every branch in ``decide()``
    that returns a new ``ControllerState``.
    """
    if state.samples_seen == 0:
        return sample.kv_cache_pct, float(sample.requests_waiting)
    return (
        _ewma(state.smoothed_kv, sample.kv_cache_pct, params.ewma_alpha),
        _ewma(
            state.smoothed_waiting,
            float(sample.requests_waiting),
            params.ewma_alpha,
        ),
    )


def _shrink(current: int, params: ControllerParams) -> int:
    target = int(current * (1.0 - params.max_step_factor))
    # Floor the result at ``effective_min_cap`` so chained shrinks
    # don't ladder down to single digits where each decrement trims
    # one request at a time and accomplishes nothing useful.
    return max(params.lower_bound, params.effective_min_cap, target)


def _grow(state: ControllerState, sample: Sample, params: ControllerParams) -> int:
    """Predict the cap that equilibrates KV at ``target_kv_grow``.

    Math: at this moment we have ``R`` sequences running at ``K`` KV
    utilization, so per-request KV is ``K/R``. The cap that would put
    KV at ``T = params.target_kv_grow`` is ``(T/K) × R``.

    Returns ``current+1`` when the signal is too weak to project
    (no running requests, or KV essentially zero — early in the run).

    Overshoot is intentionally tolerated. If the linearity assumption
    breaks (e.g. prefix-cache deduplication), the predicted cap may be
    too high; vLLM will queue the excess, the next decide tick observes
    ``requests_waiting > queue_tolerance``, and the shrink path catches
    the overshoot. Better to discover the real ceiling fast than to
    crawl up by 15%/tick.
    """
    if sample.requests_running > 0 and state.smoothed_kv > 0.02:
        predicted = int(
            (params.target_kv_grow / state.smoothed_kv) * sample.requests_running
        )
    else:
        predicted = state.current_cap + 1
    progressive = max(predicted, state.current_cap + 1)
    # Apply the learned safe ceiling if one has been recorded by a
    # prior preempt shrink. The kv-target prediction is correct *in
    # expectation* but ignores the empirical fact that we previously
    # exceeded vLLM's actual KV ceiling at a known cap. Cap the grow
    # so we don't keep rediscovering that ceiling.
    if state.max_safe_cap > 0:
        progressive = min(progressive, state.max_safe_cap)
    progressive = max(progressive, state.current_cap)
    return min(params.upper_bound, progressive)


def decide(
    state: ControllerState, sample: Sample, params: ControllerParams
) -> Decision:
    """Return the next cap and updated state given one fresh sample.

    Pure function. ``state`` is not mutated; the new state is returned
    inside the ``Decision``.
    """
    if params.override is not None:
        new_state = state.model_copy(
            update={
                "current_cap": params.override,
                "consecutive_above": 0,
                "consecutive_below": 0,
                "consecutive_queue": 0,
                "preemption_cooldown": 0,
                "preemption_window": [],
            }
        )
        return Decision(
            next_cap=params.override,
            state=new_state,
            reason="override",
            changed=state.current_cap != params.override,
        )

    # Maintain rolling preemption window — the windowed sum is what
    # decide() actually consults, which smooths over poll-vs-vLLM-tick
    # races and one-off bookkeeping evictions.
    preemption_window = (state.preemption_window + [sample.preemption_delta])[
        -params.preemption_window_ticks :
    ]
    preemption_sum = sum(preemption_window)

    if preemption_sum > params.preemption_threshold:
        # Floor: half the running load (don't drop below what's clearly
        # sustainable client-side) AND the configured ``effective_min_cap``
        # AND the absolute lower_bound. Aggressive enough to clear the
        # over-admission, conservative enough not to collapse to 1.
        target = max(
            params.lower_bound,
            params.effective_min_cap,
            sample.requests_running // 2,
            int(state.current_cap * params.preemption_shrink_factor),
        )
        # Learn the ceiling: the cap that just caused preemptions is
        # unsafe by ``preempt_safe_factor``. If we've seen preemptions
        # before, take the tighter of the new and previously-learned
        # ceilings so the bound only ratchets down.
        learned = int(state.current_cap * params.preempt_safe_factor)
        if state.max_safe_cap > 0:
            learned = min(learned, state.max_safe_cap)
        smoothed_kv, smoothed_waiting = _update_smoothed(state, sample, params)
        new_state = ControllerState(
            current_cap=target,
            smoothed_kv=smoothed_kv,
            smoothed_waiting=smoothed_waiting,
            consecutive_above=0,
            consecutive_below=0,
            consecutive_queue=0,
            samples_seen=state.samples_seen + 1,
            preemption_cooldown=params.preemption_cooldown_ticks,
            # Reset window: the entries that triggered this shrink are
            # no longer informative once we've acted on them. Carrying
            # them forward would keep re-triggering shrink on the next
            # tick even when no new preemptions have occurred.
            preemption_window=[],
            max_safe_cap=learned,
        )
        return Decision(
            next_cap=target,
            state=new_state,
            reason="shrink_preempt",
            changed=target != state.current_cap,
        )

    smoothed_kv, smoothed_waiting = _update_smoothed(state, sample, params)

    # ``requests_waiting`` is vLLM's *internal* admission queue, not a
    # backpressure signal on its own. When the client sends more than
    # the server's ``--max-num-seqs`` ceiling, vLLM by design admits
    # the excess into a waiting queue and drains it as the running
    # batch frees slots — KV stays healthy, no preemptions, throughput
    # is fine. Treating that as "shrink now" causes a chained ladder
    # all the way to single digits while the server is still happily
    # processing 48 sequences. Only treat queue depth as a shrink
    # signal when paired with a real saturation symptom: KV crossing
    # the high band, OR a non-empty preemption window. Pure queue
    # without those is normal server-side scheduling, not overload.
    raw_queue = sample.requests_waiting > params.queue_tolerance
    queue_pressure = raw_queue and (
        smoothed_kv > params.target_kv_hi or preemption_sum > 0
    )
    consecutive_queue = state.consecutive_queue + 1 if queue_pressure else 0

    above_hi = smoothed_kv > params.target_kv_hi
    saturating_cap = (
        sample.local_in_flight >= state.current_cap * params.min_utilization_for_grow
    )
    below_lo = (
        smoothed_kv < params.target_kv_lo and not queue_pressure and saturating_cap
    )
    consecutive_above = state.consecutive_above + 1 if above_hi else 0
    consecutive_below = state.consecutive_below + 1 if below_lo else 0

    next_cap = state.current_cap
    reason = "hold"

    if consecutive_queue >= params.consecutive_signals:
        next_cap = _shrink(state.current_cap, params)
        reason = "shrink_queue"
    elif consecutive_above >= params.consecutive_signals:
        next_cap = _shrink(state.current_cap, params)
        reason = "shrink_kv_high"
    elif consecutive_below >= params.consecutive_signals:
        if state.preemption_cooldown > 0:
            reason = "hold_post_preempt"
        else:
            next_cap = _grow(state, sample, params)
            reason = "grow_kv_low"

    if next_cap != state.current_cap:
        consecutive_above = 0
        consecutive_below = 0
        consecutive_queue = 0

    # Decay ``max_safe_cap`` when conditions clearly indicate the old
    # ceiling is stale. The learned-ceiling logic is correct for
    # steady-state preemptions (vLLM truly cannot fit more at that cap)
    # but wrong for transient bursts that preempt during initial fill
    # then settle into a healthy regime — observed in vision workloads,
    # where startup preempts pinned cap at 17 while KV stayed at 20 %
    # and GPU at 17 %. Decay lets the controller probe upward; a real
    # preempt re-establishes the floor via the ``shrink_preempt``
    # branch above, so the cost of a bad probe is one shrink cycle.
    new_max_safe_cap = state.max_safe_cap
    can_decay = (
        state.max_safe_cap > 0
        and params.max_safe_cap_decay_per_tick > 0
        and state.preemption_cooldown == 0
        and preemption_sum == 0
        and below_lo
        and saturating_cap
        and state.max_safe_cap < params.upper_bound
    )
    if can_decay:
        new_max_safe_cap = min(
            state.max_safe_cap + params.max_safe_cap_decay_per_tick,
            params.upper_bound,
        )

    new_state = ControllerState(
        current_cap=next_cap,
        smoothed_kv=smoothed_kv,
        smoothed_waiting=smoothed_waiting,
        consecutive_above=consecutive_above,
        consecutive_below=consecutive_below,
        consecutive_queue=consecutive_queue,
        samples_seen=state.samples_seen + 1,
        preemption_cooldown=max(0, state.preemption_cooldown - 1),
        preemption_window=preemption_window,
        max_safe_cap=new_max_safe_cap,
    )

    return Decision(
        next_cap=next_cap,
        state=new_state,
        reason=reason,
        changed=next_cap != state.current_cap,
    )
