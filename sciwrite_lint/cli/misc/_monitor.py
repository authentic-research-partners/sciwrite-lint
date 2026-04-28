"""Live-refresh terminal monitor for GROBID + vLLM.

Shared by ``sciwrite-lint containers monitor`` and ``sciwrite-lint vllm monitor``.
Contains the rich Table/Panel builders, stage-label dictionaries, small
formatting helpers, and the main event loop.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

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
    req_text.append(f"   waiting: {waiting}", style="red bold" if waiting else "dim")
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
    except Exception:
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

    prev_prompt_tokens = 0.0
    prev_gen_tokens = 0.0
    prev_preemptions = 0.0
    prev_time = 0.0
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
                if grobid_up:
                    cgroup = _resolve_cgroup_dir(CONTAINER_RUNTIME, GROBID_CONTAINER)
                    if cgroup:
                        grobid_used, grobid_limit = _read_cgroup_memory(cgroup)
                panels.append(
                    _build_grobid_panel(
                        grobid_up=grobid_up,
                        ram_used_bytes=grobid_used,
                        ram_limit_bytes=grobid_limit,
                        config_limit=config.grobid_memory,
                    )
                )

                # --- vLLM panel ---
                health = asyncio.run(_check_api_health(endpoint))

                if not health:
                    msg = Text()
                    msg.append("not running", style="red")
                    msg.append("  —  ", style="dim")
                    msg.append("sciwrite-lint vllm start", style="bold")
                    panels.append(
                        Panel(
                            msg,
                            title=f"vLLM (text) — localhost:{vllm_text_port}",
                            border_style="red",
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
                from sciwrite_lint.vllm.vllm_server import MODELS, VISION_MODELS

                for vm in VISION_MODELS:
                    vm_profile = MODELS[vm]
                    vm_port = vm_profile.get("port", 5002)
                    vm_endpoint = f"http://localhost:{vm_port}/v1"
                    vm_health = asyncio.run(_check_api_health(vm_endpoint))
                    if not vm_health:
                        msg = Text()
                        msg.append("not running", style="dim")
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
