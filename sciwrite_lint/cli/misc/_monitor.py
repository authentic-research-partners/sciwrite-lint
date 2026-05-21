"""Live-refresh terminal monitor for GROBID + vLLM.

Shared by ``sciwrite-lint containers monitor`` and ``sciwrite-lint vllm monitor``.
Contains the rich Table/Panel builders, stage-label dictionaries, small
formatting helpers, and the main event loop.
"""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from rich.panel import Panel
    from rich.table import Table

from sciwrite_lint.config import LintConfig


def _print_container_logs(runtime: str, name: str, tail: int = 15) -> None:
    """Print last N lines of a container's logs.

    Uses ``--since 1h`` instead of ``--tail`` because podman scans the entire
    log to compute tail, which takes seconds on chatty containers like GROBID.
    """
    result = subprocess.run(
        [runtime, "logs", "--since", "1h", name],
        capture_output=True,
        text=True,
    )
    lines: list[str] = []
    if result.stdout:
        lines.extend(result.stdout.splitlines())
    if result.stderr:
        lines.extend(result.stderr.splitlines())
    for line in lines[-tail:]:
        print(line)


def _fetch_vllm_metrics(endpoint: str) -> str | None:
    """One-line summary for ``containers status``."""
    from sciwrite_lint.vllm.metrics import fetch_metrics_summary

    return fetch_metrics_summary(endpoint)


def _read_vllm_server_args(runtime: str, container: str) -> dict[str, str]:
    """Read scheduling-relevant launch args from a running vLLM container.

    Returns a dict with ``max_num_batched_tokens`` and ``max_num_seqs`` (and
    any future scheduler knobs we want to surface). vLLM does not expose
    these through ``/metrics`` (only ``cache_config_info`` is published),
    so we have to inspect the container's command-line. Args don't change
    while the container is running, so the caller can cache the result.
    Best-effort — any failure returns an empty dict so the monitor still
    renders.
    """
    try:
        import json as _json

        result = subprocess.run(
            [runtime, "inspect", container],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}
        data = _json.loads(result.stdout)
        if not data:
            return {}
        cmd: list[str] = data[0].get("Config", {}).get("Cmd", []) or []
    except (OSError, subprocess.TimeoutExpired, _json.JSONDecodeError, IndexError) as e:
        logger.debug(
            "container inspect failed for {}: {}: {}", container, type(e).__name__, e
        )
        return {}

    args: dict[str, str] = {}
    keys = {"--max-num-batched-tokens", "--max-num-seqs"}
    for i, tok in enumerate(cmd):
        if tok in keys and i + 1 < len(cmd):
            args[tok.lstrip("-").replace("-", "_")] = cmd[i + 1]
    return args


def _fetch_vllm_metrics_full(endpoint: str) -> dict[str, float | int | str]:
    """Full metrics dict for the live monitor."""
    from sciwrite_lint.vllm.metrics import fetch_metrics

    return fetch_metrics(endpoint)


def _fmt_duration(seconds: float | int) -> str:
    """Format seconds as ``Ns`` or ``Nm Ns`` when >= 60."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60}s"


def _bar(pct: float, width: int = 30) -> str:
    """Render a compact bar like ``[████████░░░░░░░░]  25%``."""
    filled = int(pct * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1%}"


def _bar_color(pct: float) -> str:
    """Return a rich color name based on utilization percentage."""
    if pct < 0.6:
        return "green"
    if pct < 0.85:
        return "yellow"
    return "red"


def _read_recent_controller_samples(endpoint: str, limit: int = 10) -> list:
    """Read the last ``limit`` controller telemetry rows for this endpoint.

    Returns an empty list if the controller never ran or the table is
    empty. Best-effort — never raises so monitor stays robust.
    """
    try:
        from sciwrite_lint.llm.concurrency_optimizer import read_recent

        return read_recent(endpoint=endpoint, limit=limit)
    except (sqlite3.Error, OSError) as e:
        logger.debug(
            "controller-telemetry read failed for {}: {}: {}",
            endpoint,
            type(e).__name__,
            e,
        )
        return []


def _count_embed_keys_in_cmdline(cmd: str) -> int:
    """Count how many ref_keys are passed to a running embedder subprocess.

    The embedder is invoked as ``python -c "... _embed_keys('a,b,c', '/path')"``.
    The first argument is a quoted, comma-separated CSV of citation keys.
    We have to find the closing quote of that arg rather than splitting on
    the first comma — the keys themselves are comma-separated, so naive
    splitting truncates the count to 1.

    Returns 0 if the cmdline doesn't contain the entry-point name or the
    first arg can't be parsed as a quoted string.
    """
    if "_embed_keys(" not in cmd:
        return 0
    try:
        after = cmd.split("_embed_keys(", 1)[1]
        quote = after[:1]
        if quote not in ("'", '"'):
            return 0
        keys_str = after[1:].split(quote, 1)[0]
        return len([k for k in keys_str.split(",") if k])
    except (IndexError, ValueError) as e:
        logger.debug("embed-key count parse failed ({}: {})", type(e).__name__, e)
        return 0


def _detect_embedder_processes() -> list[tuple[int, float, int]]:
    """Find sciwrite-lint embedder subprocesses currently running on this host.

    Returns a list of ``(pid, age_seconds, n_keys)`` for each match. The
    embedder is invoked via ``python -c "from sciwrite_lint.pipeline import
    _embed_keys; ..."`` (or ``_batch_embed_entry`` in the staged pipeline);
    we scan ``/proc/<pid>/cmdline`` for those entry-point names. ``n_keys``
    is parsed from the comma-separated keys argument when available, else 0.
    Best-effort — Linux only (no /proc on macOS), never raises.
    """
    import time as _time

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []

    out: list[tuple[int, float, int]] = []
    now = _time.time()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmd_path = entry / "cmdline"
        try:
            cmd = (
                cmd_path.read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", errors="replace")
            )
        except OSError:
            continue
        if "_embed_keys(" not in cmd and "_batch_embed_entry(" not in cmd:
            continue
        try:
            stat = (entry / "stat").read_text()
            # field 22 is starttime in clock ticks since boot
            after_comm = stat.rsplit(") ", 1)[-1].split()
            starttime_ticks = int(after_comm[19])
            import os as _os

            hz = _os.sysconf("SC_CLK_TCK") or 100
            with (Path("/proc/uptime")).open() as f:
                uptime_s = float(f.read().split()[0])
            boot = now - uptime_s
            age_s = now - (boot + starttime_ticks / hz)
        except (OSError, ValueError, IndexError):
            age_s = 0.0
        n_keys = _count_embed_keys_in_cmdline(cmd)
        out.append((int(entry.name), max(0.0, age_s), n_keys))
    return out


_embedder_model_info_cache: tuple[str, int, int, int] | None = None


def _resolve_embedder_model_info() -> tuple[str, int, int, int] | None:
    """Resolve (model_name, hidden_size, num_layers, max_seq_len) for display.

    Reads the configured embedding model name and probes its HuggingFace
    config (config.json only — no weights download). Cached per monitor
    process since the model name doesn't change at runtime. Returns
    ``None`` if probing fails (no cache, network unreachable, weird
    custom config), in which case the panel just omits the budget line.
    """
    global _embedder_model_info_cache
    if _embedder_model_info_cache is not None:
        return _embedder_model_info_cache
    try:
        from transformers import AutoConfig

        from sciwrite_lint.references.reference_store import _get_embedding_config

        model_name, _, _ = _get_embedding_config()
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        max_seq = getattr(cfg, "max_position_embeddings", None) or 8192
        _embedder_model_info_cache = (
            model_name,
            int(cfg.hidden_size),
            int(cfg.num_hidden_layers),
            int(max_seq),
        )
    except (OSError, AttributeError, ValueError) as e:
        # HF AutoConfig.from_pretrained surfaces OSError (download/IO),
        # AttributeError (missing config field), or ValueError (parse).
        logger.debug("embedder model info probe failed ({}: {})", type(e).__name__, e)
        return None
    return _embedder_model_info_cache


def _embedder_panel_title(running: bool) -> str:
    """Panel title with the configured embedder model name.

    Reads the model name from the project config (cheap, always
    available). Falls back to a generic title if config can't be read.
    """
    try:
        from sciwrite_lint.references.reference_store import _get_embedding_config

        model_name, _, _ = _get_embedding_config()
        short = model_name.split("/")[-1]
    except (OSError, FileNotFoundError, ValueError, AttributeError) as e:
        # Config file may be missing or unparseable; fall back to the
        # generic title. WARN-by-DEBUG so monitor stays robust.
        logger.debug(
            "embedder panel title — config probe failed ({}: {})",
            type(e).__name__,
            e,
        )
        return "Embedder (sentence-transformers subprocess)" if running else "Embedder"
    suffix = " (sentence-transformers subprocess)" if running else ""
    return f"Embedder · {short}{suffix}"


def _build_embedder_panel(
    infos: list[tuple[int, float, int]],
) -> "Panel":
    """Build the Embedder status panel — host-side subprocess view.

    The embedder runs as a Python subprocess (not a container), so it's
    invisible to ``podman ps``. This panel makes it visible by scanning
    ``/proc`` for the entry-point match and showing per-process age and
    key count when active. The configured model name appears in the
    title regardless of run state. The estimated-peak-VRAM / budget
    line is shown when the HF AutoConfig probe succeeds.
    """
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    model_info = _resolve_embedder_model_info()
    budget_line: Text | None = None
    if model_info is not None:
        from sciwrite_lint.references.embedding_store import _ENCODE_BATCH_SIZE
        from sciwrite_lint.references.reference_store import (
            _estimate_embedder_peak_vram_gb,
            expected_embed_vram_budget_gb,
        )

        _, hidden, layers, max_seq = model_info
        peak_gb = _estimate_embedder_peak_vram_gb(
            hidden_size=hidden,
            num_layers=layers,
            max_seq_len=max_seq,
            batch_size=_ENCODE_BATCH_SIZE,
        )
        # Use total VRAM × 0.85 (post-swap budget) so the displayed
        # number reflects "what the embedder will have when its
        # subprocess actually runs", not "what's free right now while
        # text vLLM is hogging VRAM". The swap pattern guarantees the
        # embedder always runs alone — the displayed budget should
        # match that reality.
        budget_gb = expected_embed_vram_budget_gb()
        budget_line = Text()
        budget_line.append(f"batch={_ENCODE_BATCH_SIZE}", style="dim")
        budget_line.append("  ·  ", style="dim")
        peak_style = "yellow" if peak_gb > budget_gb else "dim"
        budget_line.append(f"est. peak {peak_gb:.1f} GiB", style=peak_style)
        budget_line.append(f" / budget {budget_gb:.1f} GiB", style="dim")

    if not infos:
        msg = Text()
        msg.append("idle", style="dim")
        msg.append("  —  ", style="dim")
        msg.append("starts as a subprocess during parse / ref_internal", style="dim")
        title = _embedder_panel_title(running=False)
        if budget_line is None:
            return Panel(msg, title=title, border_style="dim", padding=(0, 2))
        grid = Table.grid(padding=(0, 2))
        grid.add_column()
        grid.add_row(msg)
        grid.add_row(budget_line)
        return Panel(grid, title=title, border_style="dim", padding=(0, 2))

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", min_width=12)
    grid.add_column()

    for pid, age_s, n_keys in infos:
        line = Text()
        line.append(f"running for {_fmt_duration(age_s)}", style="green")
        if n_keys:
            line.append(f"  ·  {n_keys} keys", style="dim")
        line.append(f"  ·  pid {pid}", style="dim")
        grid.add_row("Status", line)
    grid.add_row(
        "",
        Text(
            "owns GPU during embed; text vLLM is stopped to free VRAM",
            style="dim italic",
        ),
    )
    if budget_line is not None:
        grid.add_row("", budget_line)

    return Panel(
        grid,
        title=_embedder_panel_title(running=True),
        border_style="green",
        padding=(0, 2),
    )


def _build_monitor_table(
    *,
    model_name: str,
    endpoint: str,
    metrics: dict[str, float | int | str],
    vram: tuple[int, int] | None,
    gpu_util_pct: int | None,
    ram_used_bytes: int | None,
    ram_limit_bytes: int | None,
    config_ram_limit: str,
    prompt_rate: float,
    gen_rate: float,
    prompt_tok: float,
    gen_tok: float,
    preemption_rate: float,
    server_args: dict[str, str] | None = None,
) -> "Table":
    """Build a rich Table with all monitor sections."""
    from rich.table import Table
    from rich.text import Text

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", min_width=12)  # label
    grid.add_column()  # value

    # --- Server ---
    max_seq = metrics.get("max_seq")
    max_seq_str = f"  max_seq: {max_seq:,}" if max_seq else ""
    grid.add_row("Model", f"{model_name}{max_seq_str}")

    # vLLM scheduler knobs that don't show up in /metrics. We read them
    # from the container's launch command (cached by the caller). They
    # govern how many requests vLLM admits and how much per-step compute
    # budget is split among prefill chunks + decode tokens — the levers
    # the operator is most likely tuning when monitoring saturation.
    # ``max_model_len`` is intentionally omitted: it's already shown as
    # ``max_seq`` next to the Model row (vLLM publishes it in /metrics
    # as ``cache_config_info``-derived ``max_seq``, same value).
    if server_args:
        knobs = []
        if "max_num_seqs" in server_args:
            knobs.append(f"max-num-seqs: {server_args['max_num_seqs']}")
        if "max_num_batched_tokens" in server_args:
            knobs.append(f"max-batched-tokens: {server_args['max_num_batched_tokens']}")
        if knobs:
            grid.add_row("Scheduler", "  ".join(knobs))
    if vram:
        vram_used, vram_total = vram
        vram_used_gb = vram_used / (1024**3)
        vram_total_gb = vram_total / (1024**3)
        vram_pct = vram_used / vram_total
        # High VRAM usage is expected (vLLM pre-allocates) — always use calm color
        vram_color = "cyan"
        vram_text = Text()
        vram_text.append(_bar(vram_pct), style=vram_color)
        vram_text.append(f"  {vram_used_gb:.1f}GB / {vram_total_gb:.1f}GB", style="dim")
        grid.add_row("VRAM", vram_text)

        # Spillover-risk warning when free VRAM is tight. Windows
        # ``\GPU Adapter Memory`` counters can't see NVIDIA's CUDA
        # allocations (they bypass WDDM accounting), so we can't directly
        # measure shared-memory usage from inside WSL2. The reliable
        # leading indicator is dedicated VRAM free: when it drops below
        # ~500 MB, the next consumer (embedder, second container) will
        # either OOM (native Linux) or silently spill to system RAM via
        # PCIe (WSL2). Surface it before that happens.
        vram_free = vram_total - vram_used
        free_mb = vram_free / (1024**2)
        if free_mb < 500:
            from sciwrite_lint.config import is_wsl2

            warn_text = Text()
            if free_mb < 200:
                warn_text.append(f"only {free_mb:.0f} MB free", style="red bold")
            else:
                warn_text.append(f"{free_mb:.0f} MB free", style="yellow bold")
            # Same trigger, different failure mode: WSL2 spills the
            # overflow to shared sysmem (slow but succeeds); native
            # Linux raises CUDA OOM (the new consumer crashes).
            if is_wsl2():
                hint = "  — new GPU work may spill to shared sysmem (slow)"
            else:
                hint = "  — new GPU work may hit CUDA OOM"
            warn_text.append(hint, style="dim")
            grid.add_row("VRAM pressure", warn_text)
    if gpu_util_pct is not None:
        gpu_pct = gpu_util_pct / 100
        # High GPU load is good (working), not a problem — use green/dim
        color = "green" if gpu_pct > 0.1 else "dim"
        gpu_text = Text()
        gpu_text.append(_bar(gpu_pct, width=20), style=color)
        grid.add_row("GPU load", gpu_text)
    if ram_used_bytes is not None:
        used_gb = ram_used_bytes / (1024**3)
        if ram_limit_bytes:
            limit_gb = ram_limit_bytes / (1024**3)
            pct = ram_used_bytes / ram_limit_bytes
            color = _bar_color(pct)
            ram_text = Text()
            ram_text.append(_bar(pct), style=color)
            ram_text.append(f"  {used_gb:.1f}GB / {limit_gb:.1f}GB", style="dim")
            grid.add_row("RAM", ram_text)
        else:
            ram_text = Text()
            ram_text.append(f"{used_gb:.1f}GB", style="bold")
            ram_text.append(
                f"  (no limit! config says {config_ram_limit}"
                " — run: sciwrite-lint containers restart --recreate)",
                style="yellow",
            )
            grid.add_row("RAM", ram_text)

    grid.add_row("", "")  # spacer

    # --- Requests ---
    running = int(metrics.get("requests_running", 0))
    waiting = int(metrics.get("requests_waiting", 0))
    swapped = int(metrics.get("requests_swapped", 0))
    req_text = Text()
    req_text.append(f"running: {running}", style="bold" if running else "dim")
    req_text.append(f"   queued: {waiting}", style="red bold" if waiting else "dim")
    req_text.append(f"   swapped: {swapped}", style="red bold" if swapped else "dim")
    grid.add_row("Requests", req_text)

    # Request outcomes
    stops = int(metrics.get("req_success_stop", 0))
    lengths = int(metrics.get("req_success_length", 0))
    aborts = int(metrics.get("req_success_abort", 0))
    errors = int(metrics.get("req_success_error", 0))
    total_reqs = stops + lengths + aborts + errors
    if total_reqs > 0:
        hist = Text()
        hist.append(f"{stops} ok", style="green")
        if lengths:
            hist.append(f"  {lengths} truncated", style="yellow")
        if aborts:
            hist.append(f"  {aborts} aborted", style="yellow")
        if errors:
            hist.append(f"  {errors} errors", style="red")
        hist.append(f"  ({total_reqs} total)", style="dim")
        grid.add_row("History", hist)

    grid.add_row("", "")  # spacer

    # --- KV cache with bar ---
    kv_pct = metrics.get("kv_cache_pct")
    blocks = metrics.get("num_gpu_blocks")
    if kv_pct is not None:
        assert isinstance(kv_pct, float)
        block_size = metrics.get("block_size")
        if blocks and block_size:
            tokens = int(blocks) * int(block_size)
            blocks_str = f"  ({tokens:,} tokens, {int(blocks)} blocks)"
        elif blocks:
            blocks_str = f"  ({int(blocks)} GPU blocks)"
        else:
            blocks_str = ""
        # High KV cache usage is normal under load (vLLM pre-allocates).
        # Only warn at very high levels where preemptions become likely.
        if kv_pct < 0.85:
            kv_color = "cyan"
        elif kv_pct < 0.95:
            kv_color = "yellow"
        else:
            kv_color = "red"
        bar_text = Text()
        bar_text.append(_bar(kv_pct), style=kv_color)
        bar_text.append(blocks_str, style="dim")
        grid.add_row("KV cache", bar_text)

    # Prefix cache hit rate with bar
    cache_hits = float(metrics.get("prefix_cache_hits", 0.0))
    cache_queries = float(metrics.get("prefix_cache_queries", 0.0))
    if cache_queries > 0:
        hit_rate = cache_hits / cache_queries
        # For hit rate, green is high (good), red is low
        hit_color = "green" if hit_rate > 0.3 else "yellow" if hit_rate > 0.1 else "dim"
        hit_text = Text()
        hit_text.append(_bar(hit_rate), style=hit_color)
        hit_text.append(
            f"  ({int(cache_hits):,}/{int(cache_queries):,} tokens)", style="dim"
        )
        grid.add_row("Cache hit", hit_text)
    else:
        grid.add_row("Cache hit", Text("no queries yet", style="dim"))

    # Preemptions
    cur_preemptions = int(metrics.get("num_preemptions", 0))
    if cur_preemptions > 0 or preemption_rate > 0:
        evict_text = Text(
            f"{cur_preemptions:,} preemptions  ({preemption_rate:.1f}/s)",
            style="red bold",
        )
    else:
        evict_text = Text("0 preemptions", style="green")
    grid.add_row("Evictions", evict_text)

    grid.add_row("", "")  # spacer

    # --- Latency ---
    e2e_count = float(metrics.get("e2e_latency_count", 0))
    if e2e_count > 0:
        e2e_sum = float(metrics.get("e2e_latency_sum", 0))
        ttft_sum = float(metrics.get("ttft_sum", 0))
        ttft_count = float(metrics.get("ttft_count", 0))
        itl_sum = float(metrics.get("itl_sum", 0))
        itl_count = float(metrics.get("itl_count", 0))
        e2e_avg = e2e_sum / e2e_count
        ttft_avg = ttft_sum / ttft_count if ttft_count > 0 else 0
        itl_avg = itl_sum / itl_count if itl_count > 0 else 0

        lat = Text()
        lat.append(f"e2e: {e2e_avg:.2f}s", style="bold")
        lat.append(f"   TTFT: {ttft_avg:.3f}s")
        lat.append(f"   ITL: {itl_avg * 1000:.1f}ms")
        lat.append(f"  (avg over {int(e2e_count):,} reqs)", style="dim")
        grid.add_row("Latency", lat)
    else:
        grid.add_row("Latency", Text("no completed requests yet", style="dim"))

    # --- Throughput ---
    tp = Text()
    tp.append(f"prompt: {prompt_rate:,.0f} tok/s", style="bold")
    tp.append(f"   gen: {gen_rate:,.0f} tok/s")
    tp.append(f"   total: {int(prompt_tok + gen_tok):,} tokens served", style="dim")
    grid.add_row("Throughput", tp)

    # --- Concurrency controller ---
    # When ``use_dynamic_concurrency`` is on, every controller tick
    # writes to ``controller_samples`` in usage.db. Show the most recent
    # state plus a tiny cap-trend so the operator sees what the
    # controller is doing without having to grep logs.
    ctrl_rows = _read_recent_controller_samples(endpoint, limit=10)
    if ctrl_rows:
        import time as _time

        latest = ctrl_rows[0]
        age_s = _time.time() - latest.ts
        # The polling task writes every ~2s; older than ~10s means the
        # controller has stopped (refcount→0 at end of stage), not just
        # idle. Show a stopped-state line so stale ``awaiting`` /
        # ``cap`` values don't read as live.
        if age_s > 10:
            mins = age_s / 60.0
            age_str = f"{age_s:.0f}s ago" if age_s < 90 else f"{mins:.1f}m ago"
            ctrl_text = Text()
            ctrl_text.append("no active controller", style="dim")
            ctrl_text.append(f"  (last row {age_str})", style="dim")
            grid.add_row("Controller", ctrl_text)
        else:
            # Cap trend: oldest -> newest, deduped to changes only.
            seen_caps: list[int] = []
            for row in reversed(ctrl_rows):  # oldest first
                if not seen_caps or seen_caps[-1] != row.current_cap:
                    seen_caps.append(row.current_cap)
            trend = " → ".join(str(c) for c in seen_caps[-5:])

            reason_color = (
                "yellow"
                if latest.reason.startswith("shrink")
                else "green"
                if latest.reason.startswith("grow")
                else "dim"
            )
            ctrl_text = Text()
            ctrl_text.append(f"cap: {latest.current_cap}", style="bold cyan")
            ctrl_text.append(
                f"   awaiting: {latest.local_in_flight}",
                style="dim" if latest.local_in_flight else "dim",
            )
            ctrl_text.append(f"   {latest.reason}", style=reason_color)
            if len(seen_caps) > 1:
                ctrl_text.append(f"   trend: {trend}", style="dim")
            grid.add_row("Controller", ctrl_text)
            legend = Text(
                "cap = max we'll send · awaiting = sent, response not back "
                "· running/queued = vLLM's view (above)",
                style="dim italic",
            )
            grid.add_row("", legend)

    # --- Health assessment ---
    grid.add_row("", "")
    warnings: list[str] = []
    if kv_pct is not None and isinstance(kv_pct, float) and kv_pct > 0.9:
        warnings.append(f"KV cache at {kv_pct:.0%} — risk of preemptions")
    if waiting > 0:
        warnings.append(f"{waiting} requests waiting — server is at capacity")
    if preemption_rate > 0.5:
        warnings.append(
            f"Preemptions at {preemption_rate:.1f}/s"
            " — reduce concurrency or max_model_len"
        )
    if lengths >= 5 and total_reqs >= 10 and lengths / total_reqs > 0.1:
        pct = lengths / total_reqs * 100
        warnings.append(
            f"{lengths} of {total_reqs} ({pct:.0f}%) responses cut short"
            " — model hit output token limit before finishing"
        )
    if warnings:
        warn_text = Text()
        for i, w in enumerate(warnings):
            if i > 0:
                warn_text.append("\n")
            warn_text.append(f"  {w}", style="yellow bold")
        grid.add_row(Text("⚠ Warning", style="yellow bold"), warn_text)
    else:
        grid.add_row("", Text("✓ Load looks healthy", style="green"))

    return grid


def _build_grobid_panel(
    *,
    grobid_up: bool,
    ram_used_bytes: int | None,
    ram_limit_bytes: int | None,
    config_limit: str,
    cpu_cores: float | None = None,
) -> "Panel":
    """Build the GROBID status panel."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    if not grobid_up:
        msg = Text()
        msg.append("not running", style="red")
        msg.append("  —  ", style="dim")
        msg.append("sciwrite-lint grobid start", style="bold")
        return Panel(msg, title="GROBID", border_style="red", padding=(0, 2))

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", min_width=12)
    grid.add_column()

    grid.add_row("Status", Text("running", style="green"))

    # Activity indicator instead of a raw CPU bar — for ~80% of a
    # bench / pipeline run GROBID is genuinely idle (parse is the
    # only stage that drives it), so a CPU bar reads as "stuck at
    # zero" and looks like a broken metric. The CPU sample is still
    # the underlying signal, just rendered semantically: any
    # in-flight HTTP request pushes cores well above 0.1, so the
    # threshold cleanly distinguishes "parsing" from "idle".
    if cpu_cores is not None:
        if cpu_cores > 0.1:
            activity = Text()
            activity.append("parsing", style="green")
            activity.append(f"  ({cpu_cores:.1f} cores)", style="dim")
        else:
            activity = Text("idle", style="dim")
        grid.add_row("Activity", activity)

    if ram_used_bytes is not None:
        used_gb = ram_used_bytes / (1024**3)
        if ram_limit_bytes:
            limit_gb = ram_limit_bytes / (1024**3)
            pct = ram_used_bytes / ram_limit_bytes
            color = _bar_color(pct)
            ram_text = Text()
            ram_text.append(_bar(pct), style=color)
            ram_text.append(f"  {used_gb:.1f}GB / {limit_gb:.1f}GB", style="dim")
            grid.add_row("RAM", ram_text)
        else:
            ram_text = Text()
            ram_text.append(f"{used_gb:.1f}GB", style="bold")
            ram_text.append(
                f"  (no limit! config says {config_limit}"
                " — run: sciwrite-lint containers restart --recreate)",
                style="yellow",
            )
            grid.add_row("RAM", ram_text)

    return Panel(
        grid,
        title="GROBID (PDF parsing) — localhost:8070",
        border_style="green",
        padding=(0, 2),
    )


# Stage display names + resource tags (shorter for terminal).
# Resource suffixes show what hardware each stage uses when running.
_STAGE_LABELS: dict[str, str] = {
    "setup": "Setup",
    "vision": "Vision",
    "text_checks": "Text rules",
    "llm_checks": "LLM checks",
    "verify": "API verify",
    "fetch": "Fetch PDFs",
    "parse": "Parse+embed",
    "cited_vision": "Cited figs",
    "ref_internal": "Ref internal",
    "bib_verify": "Bib verify",
    "claims": "Claims",
    "unreliable": "Unreliable",
    "contributions": "Contributions",
}
_STAGE_RESOURCES: dict[str, str] = {
    "setup": "grobid",
    "vision": "gpu",
    "text_checks": "cpu",
    "llm_checks": "vllm",
    "verify": "net",
    "fetch": "net",
    "parse": "gpu",
    "cited_vision": "gpu",
    "ref_internal": "vllm",
    "bib_verify": "net",
    "claims": "vllm",
    "unreliable": "cpu",
    "contributions": "vllm",
}
# Short column headers for the batch table (max ~5 chars each).
_STAGE_SHORT: dict[str, str] = {
    "setup": "Setup",
    "vision": "Vis",
    "text_checks": "Text",
    "llm_checks": "LLM",
    "verify": "Vfy",
    "fetch": "Fetch",
    "parse": "Parse",
    "cited_vision": "CFig",
    "ref_internal": "RInt",
    "bib_verify": "Bib",
    "claims": "Claim",
    "unreliable": "Unrel",
    "contributions": "Contr",
}


def _build_stages_panel_from_path(paper: str, workspace_root: Path) -> "Panel | None":
    """Build stages panel from an explicit workspace root path."""
    from sciwrite_lint.references.workspace_db import db_path

    db_file = db_path(workspace_root)
    if not db_file.exists():
        return None
    return _build_stages_panel_impl(paper, workspace_root)


def _build_stages_panel(paper: str, config: LintConfig) -> "Panel | None":
    """Build stages panel by resolving workspace from config."""
    from sciwrite_lint.references.workspace_db import db_path

    ws = config.paper_workspace(paper)
    db_file = db_path(ws.root)
    if not db_file.exists():
        return None
    return _build_stages_panel_impl(paper, ws.root)


def _load_stages(workspace_root: Path) -> list[dict[str, str | float | None]] | None:
    """Load pipeline stages from workspace.db, or None on failure."""
    from sciwrite_lint.references.workspace_db import get_db, load_pipeline_stages

    try:
        with get_db(workspace_root) as conn:
            stages = load_pipeline_stages(conn)
    except sqlite3.Error as e:
        logger.debug(
            "pipeline-stages read failed for {}: {}: {}",
            workspace_root,
            type(e).__name__,
            e,
        )
        return None
    return stages or None


def _build_stages_panel_impl(paper: str, workspace_root: Path) -> "Panel | None":
    """Build a rich Panel showing pipeline stage progress for a paper."""
    import time as _time

    from rich.panel import Panel
    from rich.text import Text

    stages = _load_stages(workspace_root)
    if not stages:
        return None

    from datetime import datetime

    now = _time.time()
    content = Text()

    def _local_time(ts: float) -> str:
        """Format epoch timestamp as local HH:MM:SS."""
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    for i, s in enumerate(stages):
        name = str(s["stage"])
        status = str(s["status"])
        label = _STAGE_LABELS.get(name, name)
        start_t = s.get("start_time")
        end_t = s.get("end_time")

        if i > 0:
            content.append("  ")

        if status == "done":
            elapsed = ""
            if isinstance(start_t, float) and isinstance(end_t, float):
                elapsed = f" {_fmt_duration(end_t - start_t)}"
            content.append(f"[{label}{elapsed}]", style="green")
        elif status == "running":
            parts = label
            if isinstance(start_t, float):
                parts += f" {_fmt_duration(now - start_t)}"
            resource = _STAGE_RESOURCES.get(name, "")
            if resource:
                parts += f" {resource}"
            if isinstance(start_t, float):
                parts += f" @{_local_time(start_t)}"
            content.append(f"[{parts}]", style="bold yellow")
        elif status == "failed":
            content.append(f"[{label} fail]", style="indian_red")
        elif status == "skipped":
            content.append(f"({label})", style="dim")
        else:  # pending
            content.append(label, style="dim")

    # Add detail from currently running stage
    running = [s for s in stages if s["status"] == "running"]
    detail = str(running[0].get("detail", "")) if running else ""
    if detail:
        content.append(f"\n{detail}", style="dim italic")

    # Legend
    content.append("\n")
    content.append("[done]", style="green")
    content.append("  ", style="dim")
    content.append("[running]", style="bold yellow")
    content.append("  ", style="dim")
    content.append("[fail]", style="indian_red")
    content.append("  ", style="dim")
    content.append("pending", style="dim")

    return Panel(
        content,
        title=f"Pipeline Stages: {paper}",
        border_style="yellow",
        padding=(0, 2),
    )


def _build_batch_stages_panel(
    papers: list[tuple[str, Path]],
) -> "Panel | None":
    """Build a compact panel showing stage progress for a batch of papers.

    Renders a table with one row per paper and one column per pipeline stage.
    Each cell shows a status icon: green checkmark (done), yellow arrow
    (running), red X (failed), dash (skipped), or dot (pending).
    """
    import time as _time

    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from sciwrite_lint.references.workspace_db import PIPELINE_STAGES

    all_stages: list[tuple[str, list[dict[str, str | float | None]]]] = []
    for paper, ws_root in papers:
        stages = _load_stages(ws_root)
        if stages:
            all_stages.append((paper, stages))

    if not all_stages:
        return None

    now = _time.time()

    table = Table(box=None, padding=(0, 1), show_header=True)
    table.add_column("Paper", style="bold", min_width=12)
    for stage_name in PIPELINE_STAGES:
        table.add_column(
            _STAGE_SHORT.get(stage_name, stage_name),
            justify="center",
            min_width=5,
        )

    n_running = 0
    n_done = 0
    n_failed = 0

    for paper, stages in all_stages:
        stage_map = {str(s["stage"]): s for s in stages}
        cells: list[Text] = []
        for stage_name in PIPELINE_STAGES:
            s = stage_map.get(stage_name)
            if not s:
                cells.append(Text(".", style="dim"))
                continue
            status = str(s["status"])
            start_t = s.get("start_time")
            end_t = s.get("end_time")
            if status == "done":
                n_done += 1
                elapsed = ""
                if isinstance(start_t, float) and isinstance(end_t, float):
                    elapsed = f" {_fmt_duration(end_t - start_t)}"
                cells.append(Text(f"OK{elapsed}", style="green"))
            elif status == "running":
                n_running += 1
                elapsed = ""
                if isinstance(start_t, float):
                    elapsed = f" {_fmt_duration(now - start_t)}"
                cells.append(Text(f">>{elapsed}", style="bold yellow"))
            elif status == "failed":
                n_failed += 1
                cells.append(Text("FAIL", style="indian_red"))
            elif status == "skipped":
                cells.append(Text("--", style="dim"))
            else:  # pending
                cells.append(Text(".", style="dim"))
        table.add_row(paper, *cells)

    parts = [f"{len(all_stages)} papers"]
    if n_running:
        parts.append(f"{n_running} stages running")
    if n_done:
        parts.append(f"{n_done} done")
    if n_failed:
        parts.append(f"{n_failed} failed")
    title = f"Batch Pipeline ({', '.join(parts)})"

    border = "yellow" if n_running > 0 else "green" if n_failed == 0 else "red"

    return Panel(
        table,
        title=title,
        border_style=border,
        padding=(0, 2),
    )


def _run_containers_monitor(config: LintConfig, interval: float) -> int:
    """Live-refresh terminal monitor for GROBID + vLLM."""
    import time

    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    from sciwrite_lint.pdf.grobid import (
        CONTAINER_NAME as GROBID_CONTAINER,
        CONTAINER_RUNTIME,
        _read_cgroup_cpu_usage_usec,
        _read_cgroup_memory,
        _resolve_cgroup_dir,
        gpu_memory_status,
        gpu_utilization,
        is_grobid_running,
    )
    from sciwrite_lint.vllm.vllm_server import (
        _check_api_health,
        _container_name as vllm_container_name,
        _detect_container_runtime,
    )

    console = Console()
    endpoint = config.llm_endpoint
    from urllib.parse import urlparse

    _parsed_ep = urlparse(endpoint)
    vllm_text_port = _parsed_ep.port or 5001
    runtime = _detect_container_runtime()
    vllm_cname = vllm_container_name(config.llm_model)
    # Cache server launch args per container — they don't change while
    # the container runs, and ``podman inspect`` is too slow to call on
    # every refresh tick. Refresh keys lazily on first sight of a
    # container; ``--recreate`` produces a new container so the lookup
    # naturally re-runs once for the new instance.
    server_args_cache: dict[str, dict[str, str]] = {}

    def _server_args_for(cname: str) -> dict[str, str]:
        if cname not in server_args_cache and runtime:
            server_args_cache[cname] = _read_vllm_server_args(runtime, cname)
        return server_args_cache.get(cname, {})

    prev_prompt_tokens = 0.0
    prev_gen_tokens = 0.0
    prev_preemptions = 0.0
    prev_time = 0.0
    grobid_prev_cpu_us: int | None = None
    grobid_prev_cpu_time = 0.0
    vm_prev: dict[str, dict[str, float]] = {}
    runs_cache: list[dict] | None = None
    runs_cache_time = 0.0
    _RUNS_CACHE_TTL = 30.0  # reload run history every 30s

    legend = Text.from_markup(
        "[dim]TTFT: time to first token (how long before output starts)"
        "  ·  ITL: inter-token latency (speed per token)"
        "  ·  e2e: total request time\n"
        "KV cache: GPU memory for active requests"
        "  ·  Cache hit: reused prompt tokens vs freshly computed\n"
        "Swapped: requests paused, GPU memory moved to CPU"
        "  ·  Evictions: requests killed to free GPU memory[/]"
    )

    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                now = time.monotonic()
                panels: list[Panel | Text] = []

                # --- GROBID panel ---
                grobid_up = asyncio.run(is_grobid_running())
                grobid_used: int | None = None
                grobid_limit: int | None = None
                grobid_cpu_cores: float | None = None
                if grobid_up:
                    cgroup = _resolve_cgroup_dir(CONTAINER_RUNTIME, GROBID_CONTAINER)
                    if cgroup:
                        grobid_used, grobid_limit = _read_cgroup_memory(cgroup)
                        cur_cpu_us = _read_cgroup_cpu_usage_usec(cgroup)
                        if cur_cpu_us is not None:
                            if (
                                grobid_prev_cpu_us is not None
                                and grobid_prev_cpu_time > 0
                            ):
                                dt = now - grobid_prev_cpu_time
                                if dt > 0:
                                    grobid_cpu_cores = max(
                                        0.0,
                                        (cur_cpu_us - grobid_prev_cpu_us)
                                        / 1_000_000.0
                                        / dt,
                                    )
                            grobid_prev_cpu_us = cur_cpu_us
                            grobid_prev_cpu_time = now
                else:
                    # Container went down — invalidate previous sample so a
                    # restart doesn't get attributed a huge first-tick rate.
                    grobid_prev_cpu_us = None
                    grobid_prev_cpu_time = 0.0
                panels.append(
                    _build_grobid_panel(
                        grobid_up=grobid_up,
                        ram_used_bytes=grobid_used,
                        ram_limit_bytes=grobid_limit,
                        config_limit=config.grobid_memory,
                        cpu_cores=grobid_cpu_cores,
                    )
                )

                # --- Embedder subprocess panel ---
                # The embedder runs as a host-side Python subprocess (not a
                # container). When it's active during parse / ref_internal,
                # it owns the GPU and text vLLM is stopped (see
                # ``pipeline/swap.py::_needs_embedding_swap``). The panel
                # makes that visible so the operator doesn't read "vLLM
                # not running" as a failure.
                embedder_infos = _detect_embedder_processes()
                panels.append(_build_embedder_panel(embedder_infos))

                # --- Vision vLLM detection (lifted above text panel) ---
                # Knowing whether a vision vLLM container is up lets the text
                # panel distinguish "stopped because vision/embedder owns the
                # GPU" (expected) from "no sibling running" (likely a real
                # failure). The vision panel below reuses these results.
                from sciwrite_lint.vllm.vllm_server import MODELS, VISION_MODELS

                vision_health: dict[str, dict | None] = {}
                for vm in VISION_MODELS:
                    vm_port = MODELS[vm].get("port", 5002)
                    vm_endpoint_check = f"http://localhost:{vm_port}/v1"
                    vision_health[vm] = asyncio.run(
                        _check_api_health(vm_endpoint_check)
                    )
                vision_running = [vm for vm, h in vision_health.items() if h]

                # --- vLLM panel ---
                health = asyncio.run(_check_api_health(endpoint))

                if not health:
                    # Text vLLM being down is *expected* whenever the embedder
                    # subprocess or a vision vLLM container is up (the GPU is
                    # spoken for). Only flag in red when neither sibling is
                    # running. Describe observable state only — do not predict
                    # when text vLLM will come back, since that depends on
                    # pipeline wiring that may change.
                    msg = Text()
                    expected = bool(embedder_infos or vision_running)
                    if expected:
                        msg.append("stopped", style="yellow")
                        msg.append("  —  ", style="dim")
                        if embedder_infos and vision_running:
                            reason = "embedder + vision vLLM are running"
                        elif embedder_infos:
                            reason = "embedder is running"
                        else:
                            reason = "vision vLLM is running"
                        msg.append(reason, style="dim italic")
                        border = "yellow"
                    else:
                        msg.append("not running", style="red")
                        msg.append("  —  ", style="dim")
                        msg.append("sciwrite-lint vllm start", style="bold")
                        border = "red"
                    panels.append(
                        Panel(
                            msg,
                            title=f"vLLM (text) — localhost:{vllm_text_port}",
                            border_style=border,
                            padding=(0, 2),
                        )
                    )
                else:
                    models = [m["id"] for m in health.get("data", [])]
                    model_name = models[0] if models else "unknown"
                    metrics = _fetch_vllm_metrics_full(endpoint)

                    prompt_tok = float(metrics.get("prompt_tokens_total", 0.0))
                    gen_tok = float(metrics.get("generation_tokens_total", 0.0))
                    cur_preemptions = float(metrics.get("num_preemptions", 0.0))
                    prompt_rate = 0.0
                    gen_rate = 0.0
                    preemption_rate = 0.0
                    if prev_time > 0:
                        dt = now - prev_time
                        if dt > 0:
                            prompt_rate = (prompt_tok - prev_prompt_tokens) / dt
                            gen_rate = (gen_tok - prev_gen_tokens) / dt
                            preemption_rate = (cur_preemptions - prev_preemptions) / dt
                    prev_prompt_tokens = prompt_tok
                    prev_gen_tokens = gen_tok
                    prev_preemptions = cur_preemptions
                    prev_time = now

                    vram = gpu_memory_status()
                    gpu_util = gpu_utilization()
                    vllm_used: int | None = None
                    vllm_limit: int | None = None
                    if runtime:
                        cgroup = _resolve_cgroup_dir(runtime, vllm_cname)
                        if cgroup:
                            vllm_used, vllm_limit = _read_cgroup_memory(cgroup)

                    table = _build_monitor_table(
                        model_name=model_name,
                        endpoint=endpoint,
                        metrics=metrics,
                        vram=vram,
                        gpu_util_pct=gpu_util,
                        ram_used_bytes=vllm_used,
                        ram_limit_bytes=vllm_limit,
                        config_ram_limit=config.vllm_memory,
                        prompt_rate=prompt_rate,
                        gen_rate=gen_rate,
                        prompt_tok=prompt_tok,
                        gen_tok=gen_tok,
                        preemption_rate=preemption_rate,
                        server_args=_server_args_for(vllm_cname),
                    )
                    panels.append(
                        Panel(
                            table,
                            title=f"vLLM (text) — localhost:{vllm_text_port}",
                            border_style="blue",
                            padding=(1, 2),
                        )
                    )

                # --- Vision vLLM panel ---
                for vm in VISION_MODELS:
                    vm_profile = MODELS[vm]
                    vm_port = vm_profile.get("port", 5002)
                    vm_endpoint = f"http://localhost:{vm_port}/v1"
                    vm_health = vision_health[vm]
                    if not vm_health:
                        # Vision vLLM is normally stopped — its dim border
                        # already signals that. Surface whichever sibling is
                        # currently holding the GPU. Describe observable state
                        # only; do not predict when vision will come up.
                        msg = Text()
                        msg.append("not running", style="dim")
                        msg.append("  —  ", style="dim")
                        if embedder_infos:
                            reason = "embedder is running"
                        elif health:
                            reason = "text vLLM is running"
                        else:
                            reason = "no GPU sibling running"
                        msg.append(reason, style="dim italic")
                        panels.append(
                            Panel(
                                msg,
                                title=f"vLLM (vision) — localhost:{vm_port}",
                                border_style="dim",
                                padding=(0, 2),
                            )
                        )
                    else:
                        vm_models = [m["id"] for m in vm_health.get("data", [])]
                        vm_model_str = vm_models[0] if vm_models else vm
                        vm_cname = vllm_container_name(vm)
                        vm_metrics = _fetch_vllm_metrics_full(vm_endpoint)

                        vp = vm_prev.setdefault(
                            vm,
                            {
                                "prompt": 0.0,
                                "gen": 0.0,
                                "preempt": 0.0,
                                "time": 0.0,
                            },
                        )
                        vm_pt = float(vm_metrics.get("prompt_tokens_total", 0.0))
                        vm_gt = float(vm_metrics.get("generation_tokens_total", 0.0))
                        vm_pe = float(vm_metrics.get("num_preemptions", 0.0))
                        vm_pr = vm_gr = vm_per = 0.0
                        if vp["time"] > 0:
                            dt = now - vp["time"]
                            if dt > 0:
                                vm_pr = (vm_pt - vp["prompt"]) / dt
                                vm_gr = (vm_gt - vp["gen"]) / dt
                                vm_per = (vm_pe - vp["preempt"]) / dt
                        vp["prompt"] = vm_pt
                        vp["gen"] = vm_gt
                        vp["preempt"] = vm_pe
                        vp["time"] = now

                        vm_vram = gpu_memory_status()
                        vm_gpu_util = gpu_utilization()
                        vm_ram_used: int | None = None
                        vm_ram_limit: int | None = None
                        if runtime:
                            vm_cgroup = _resolve_cgroup_dir(runtime, vm_cname)
                            if vm_cgroup:
                                vm_ram_used, vm_ram_limit = _read_cgroup_memory(
                                    vm_cgroup
                                )

                        vm_table = _build_monitor_table(
                            model_name=vm_model_str,
                            endpoint=vm_endpoint,
                            metrics=vm_metrics,
                            vram=vm_vram,
                            gpu_util_pct=vm_gpu_util,
                            ram_used_bytes=vm_ram_used,
                            ram_limit_bytes=vm_ram_limit,
                            config_ram_limit=vm_profile.get("memory", "8g"),
                            prompt_rate=vm_pr,
                            gen_rate=vm_gr,
                            prompt_tok=vm_pt,
                            gen_tok=vm_gt,
                            preemption_rate=vm_per,
                            server_args=_server_args_for(vm_cname),
                        )
                        panels.append(
                            Panel(
                                vm_table,
                                title=f"vLLM (vision) — localhost:{vm_port}",
                                border_style="green",
                                padding=(1, 2),
                            )
                        )

                # --- Active runs + pipeline stages (DB-driven) ---
                from sciwrite_lint.usage import find_active_db_runs

                active_db_runs = find_active_db_runs()

                by_pid: dict[int, list[dict]] = {}
                for db_run in active_db_runs:
                    pid = db_run.get("pid", 0)
                    by_pid.setdefault(pid, []).append(db_run)

                for pid, runs in by_pid.items():
                    if len(runs) == 1:
                        r = runs[0]
                        p = _build_stages_panel_from_path(
                            r["paper"], Path(r["workspace_root"])
                        )
                        if p:
                            panels.append(p)
                    else:
                        batch_papers = [
                            (r["paper"], Path(r["workspace_root"])) for r in runs
                        ]
                        p = _build_batch_stages_panel(batch_papers)
                        if p:
                            panels.append(p)

                # --- Completed runs panel (cached, refreshed every 30s) ---
                from sciwrite_lint.usage import load_runs

                if runs_cache is None or (now - runs_cache_time) > _RUNS_CACHE_TTL:
                    runs_cache = load_runs(limit=5)
                    runs_cache_time = now

                _SERVICES = [
                    ("vLLM", "vllm"),
                    ("GROBID", "grobid"),
                    ("CrossRef", "crossref"),
                    ("OpenAlex", "openalex"),
                    ("S2", "semantic_scholar"),
                    ("Fetch", "fetch"),
                ]

                if runs_cache:
                    from rich.table import Table as RichTable

                    runs_table = RichTable(box=None, padding=(0, 1), show_header=True)
                    runs_table.add_column("Paper", style="bold")
                    runs_table.add_column("Time", justify="right")
                    runs_table.add_column("Cites", justify="right")
                    for label, _ in _SERVICES:
                        runs_table.add_column(label, justify="right")
                    runs_table.add_column("When", style="dim")
                    for run in runs_cache:
                        svc_cells: list[str] = []
                        for _, key in _SERVICES:
                            svc = run.get(key, {})
                            if not isinstance(svc, dict):
                                svc = {}
                            calls = svc.get("calls", 0)
                            errors = svc.get("errors", 0)
                            if calls > 0:
                                cell = str(calls)
                                if errors:
                                    cell += f"/{errors}err"
                                svc_cells.append(cell)
                            else:
                                svc_cells.append("—")
                        ts_raw = run.get("timestamp", "")
                        try:
                            from datetime import datetime, timezone

                            utc_dt = datetime.fromisoformat(ts_raw).replace(
                                tzinfo=timezone.utc
                            )
                            ts = utc_dt.astimezone().strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError):
                            ts = ts_raw[:16].replace("T", " ")
                        runs_table.add_row(
                            run.get("paper", "?"),
                            _fmt_duration(run.get("total_elapsed_s", 0)),
                            str(run.get("citations", 0)),
                            *svc_cells,
                            ts,
                        )
                    panels.append(
                        Panel(
                            runs_table,
                            title="✓ Completed runs (API calls per service)",
                            border_style="dim",
                            padding=(0, 2),
                        )
                    )

                footer = Text.from_markup(
                    f"[dim]Refreshing every {interval:.0f}s — Ctrl+C to exit[/]"
                )
                live.update(Group(*panels, legend, footer))
                time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\nMonitor stopped.", style="dim")

    return 0


def _run_vllm_monitor(config: LintConfig, interval: float) -> int:
    """Live-refresh terminal monitor (vLLM only, via ``sciwrite-lint vllm monitor``)."""
    return _run_containers_monitor(config, interval)
