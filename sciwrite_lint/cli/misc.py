"""CLI handlers for misc commands (init, parse, override, dismiss-claim, grobid, vllm, services)."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.panel import Panel
    from rich.table import Table

from loguru import logger

from sciwrite_lint.config import LintConfig, load_config


def run_init(args: argparse.Namespace) -> int:
    """Initialize a sciwrite-lint project in the current directory."""
    from sciwrite_lint.config import init_project

    success, message = init_project(force=args.force)
    print(message)
    return 0 if success else 1


def run_parse(args: argparse.Namespace) -> int:
    """Parse PDFs via GROBID and store results + embeddings."""
    from sciwrite_lint.references.reference_store import (
        parse_all_missing,
        parse_and_embed,
    )

    from sciwrite_lint.__main__ import _load_config, _resolve_paper

    config = _load_config(args)
    pc = _resolve_paper(config, args.paper)
    if not pc:
        return 2
    ws = config.paper_workspace(pc.name)
    ws.ensure_dirs()
    refs_dir = ws.root
    force = getattr(args, "fresh", False)
    embed = not getattr(args, "no_embed", False)

    if args.key:
        # Parse a single reference by key
        from sciwrite_lint.references.metadata import load_metadata

        meta = load_metadata(args.key, refs_dir)
        if not meta:
            print(f"No metadata for '{args.key}'. Run 'sciwrite-lint verify' first.")
            return 1
        local_file = meta.access.get("local_file", "")
        if not local_file or not local_file.endswith(".pdf"):
            print(f"'{args.key}' has no PDF (local_file={local_file!r})")
            return 1
        pdf_path = refs_dir / local_file
        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            return 1

        print(f"Parsing {args.key} ({pdf_path.name})...")
        text, chunks = asyncio.run(
            parse_and_embed(args.key, pdf_path, refs_dir, force=force, embed=embed)
        )
        if text:
            suffix = f", {chunks} chunks embedded" if chunks else ""
            print(
                f"  Done: {len(text)} chars, stored in references/parsed/{args.key}.md{suffix}"
            )
        else:
            print("  Failed (is GROBID running? sciwrite-lint containers start)")
            return 1
        return 0

    # Parse all PDFs with local files
    print("Parsing all references with local PDFs...")
    from sciwrite_lint.pdf.grobid import is_grobid_running

    if not asyncio.run(is_grobid_running()):
        logger.error("GROBID not running. Start with: sciwrite-lint containers start")
        return 1

    results = asyncio.run(parse_all_missing(refs_dir, force=force))

    cached = sum(1 for v in results.values() if v == "cached")
    parsed = sum(1 for v in results.values() if v == "parsed")
    failed = sum(1 for v in results.values() if v == "failed")

    # Embed newly parsed references
    if embed and parsed:
        logger.info(f"Computing embeddings for {parsed} newly parsed references...")
        for key, status in results.items():
            if status == "parsed":
                md_path = refs_dir / "parsed" / f"{key}.md"
                if md_path.exists():
                    try:
                        from sciwrite_lint.references.reference_store import (
                            compute_and_store_embeddings,
                        )

                        text = md_path.read_text(encoding="utf-8")
                        n = compute_and_store_embeddings(key, text, refs_dir)
                        print(f"    {key}: {n} chunks")
                    except ImportError:
                        print(
                            "    Embeddings skipped (pip install sentence-transformers)"
                        )
                        break
                    except Exception as e:
                        print(f"    {key}: error — {e}")

    print(f"\n  Summary: {cached} cached, {parsed} parsed, {failed} failed")
    if failed:
        for key, status in results.items():
            if status == "failed":
                print(f"    FAILED: {key}")

    return 0


def run_override(args: argparse.Namespace) -> int:
    """Manually override a citation's verification tier."""
    from datetime import date

    from sciwrite_lint.references.metadata import (
        compute_tier,
        load_metadata,
        save_metadata,
    )
    from sciwrite_lint.models import CitationMetadata

    from sciwrite_lint.__main__ import _load_config, _resolve_paper

    config = _load_config(args)
    pc = _resolve_paper(config, args.paper)
    if not pc:
        return 2
    ws = config.paper_workspace(pc.name)
    refs_dir = ws.root

    key = args.key
    meta = load_metadata(key, refs_dir)

    if args.clear:
        if not meta or not meta.manual_override:
            print(f"No override found for '{key}'.")
            return 1
        meta.manual_override = {}
        meta.access["tier"] = compute_tier(meta)
        save_metadata(meta, refs_dir)
        print(f"Cleared override for '{key}'. Tier reverted to {meta.access['tier']}.")
        return 0

    if not meta:
        meta = CitationMetadata(key=key)
        meta.bibitem = {"source_papers": []}
        meta.access = {
            "tier": "",
            "local_file": None,
            "oa_url": None,
            "oa_source": None,
        }
        meta.canonical = {}
        meta.api_match = "manual"

    meta.manual_override = {
        "tier": args.tier,
        "reason": args.reason,
        "date": str(date.today()),
    }
    meta.access["tier"] = compute_tier(meta)
    save_metadata(meta, refs_dir)

    print(f"Override set for '{key}':")
    print(f"  Tier: {args.tier}")
    print(f"  Reason: {args.reason}")
    print(f"  Date: {date.today()}")
    print()
    print("This override is preserved across verify runs.")
    return 0


def run_dismiss_claim(args: argparse.Namespace) -> int:
    """Dismiss a claim verification finding as false positive."""
    from datetime import date

    from sciwrite_lint.__main__ import _load_config

    config = _load_config(args)
    ws = config.paper_workspace(args.paper)
    if not ws.root.exists():
        print(
            f"No workspace found for paper '{args.paper}'. "
            f"Run 'sciwrite-lint check --paper {args.paper}' first."
        )
        return 1

    from sciwrite_lint.references.workspace_db import (
        clear_claim_dismissal,
        dismiss_claim,
        find_claim,
        get_db,
        list_claims_for_key,
    )

    with get_db(ws.root) as conn:
        claim = find_claim(conn, args.key, args.line)

        if not claim:
            print(f"No claim found for key='{args.key}' line={args.line}.")
            print(f"Available claims for '{args.key}':")
            for c in list_claims_for_key(conn, args.key):
                print(
                    f"  line {c.get('line')}: {c.get('verdict')} \u2014 "
                    f"{c.get('context', '')[:80]}"
                )
            return 1

        claim_id = claim["id"]

        if args.clear:
            if not claim.get("dismissed"):
                print(f"Claim not dismissed: {args.key} line {args.line}")
                return 1
            clear_claim_dismissal(conn, claim_id)
            print(f"Cleared dismissal for {args.key} (line {args.line}).")
            return 0

        dismiss_claim(conn, claim_id, reason=args.reason, date_str=str(date.today()))

    v = claim.get("verdict", "?")
    print(f"Dismissed: {args.key} (line {args.line}) \u2014 {v}")
    print(f"  Reason: {args.reason}")
    print(f"  Date: {date.today()}")
    print()
    print("This claim will be shown separately in summaries and UI.")
    return 0


def run_grobid(args: argparse.Namespace) -> int:
    """Manage GROBID container."""
    from sciwrite_lint.pdf.grobid import (
        CONTAINER_IMAGE,
        CONTAINER_NAME,
        CONTAINER_RUNTIME,
        is_grobid_running,
        start_grobid,
        stop_grobid,
    )

    config = load_config(
        Path(args.config) if hasattr(args, "config") and args.config else None
    )

    if args.action == "status":
        if asyncio.run(is_grobid_running()):
            print("GROBID: running at http://localhost:8070")
        else:
            print("GROBID: not running")
            print("  Start with: sciwrite-lint containers start")
            print(
                f"  Or manually: {CONTAINER_RUNTIME} run -d --name {CONTAINER_NAME} "
                f"--memory {config.grobid_memory} -p 8070:8070 {CONTAINER_IMAGE}"
            )
        return 0
    elif args.action == "start":
        print(f"Starting GROBID container (memory limit: {config.grobid_memory})...")
        if asyncio.run(
            start_grobid(memory=config.grobid_memory, image=config.grobid_image)
        ):
            print("GROBID: running at http://localhost:8070")
            return 0
        else:
            print("GROBID: failed to start within 60s")
            return 1
    elif args.action == "stop":
        stop_grobid()
        print("GROBID: stopped")
        return 0

    return 0


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

    # Load stages for all papers
    all_stages: list[tuple[str, list[dict[str, str | float | None]]]] = []
    for paper, ws_root in papers:
        stages = _load_stages(ws_root)
        if stages:
            all_stages.append((paper, stages))

    if not all_stages:
        return None

    now = _time.time()

    # Build table: Paper | Stage1 | Stage2 | ...
    table = Table(box=None, padding=(0, 1), show_header=True)
    table.add_column("Paper", style="bold", min_width=12)
    for stage_name in PIPELINE_STAGES:
        table.add_column(
            _STAGE_SHORT.get(stage_name, stage_name),
            justify="center",
            min_width=5,
        )

    # Count how many are running / done / failed across all papers
    n_running = 0
    n_done = 0
    n_failed = 0

    for paper, stages in all_stages:
        # Build a lookup: stage_name → stage dict
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

    # Title with summary counts
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
    # Extract port for consistent panel titles (e.g. "localhost:5001")
    from urllib.parse import urlparse

    _parsed_ep = urlparse(endpoint)
    vllm_text_port = _parsed_ep.port or 5001
    runtime = _detect_container_runtime()
    vllm_cname = vllm_container_name(config.llm_model)

    prev_prompt_tokens = 0.0
    prev_gen_tokens = 0.0
    prev_preemptions = 0.0
    prev_time = 0.0
    # Per-vision-model throughput state (keyed by model name)
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

                    # Throughput deltas
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

                        # Per-model throughput deltas
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
                # Detect runs with in-progress pipeline stages by scanning
                # workspace.db files referenced from usage.db. Works for
                # CLI runs, eval runs, and batch-staged runs alike —
                # no process name matching needed.
                from sciwrite_lint.usage import find_active_db_runs

                active_db_runs = find_active_db_runs()

                # Group by PID: single-paper runs get a detailed panel,
                # multi-paper batch runs get one compact table.
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

                # Footer
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


def run_containers(args: argparse.Namespace) -> int:
    """Manage both GROBID and vLLM containers together."""
    from sciwrite_lint.pdf.grobid import (
        CONTAINER_NAME as GROBID_CONTAINER,
        CONTAINER_RUNTIME,
        container_memory_status,
        gpu_memory_status,
        is_grobid_running,
        start_grobid,
        stop_grobid,
    )
    from sciwrite_lint.vllm.vllm_server import (
        VISION_MODELS,
        _check_api_health,
        _container_name as vllm_container_name,
        _container_running,
        _detect_container_runtime,
        start_container,
        stop_container,
    )

    config = load_config(
        Path(args.config) if hasattr(args, "config") and args.config else None
    )
    action = args.action

    if action == "status":
        # --- summary ---
        runtime = _detect_container_runtime()
        grobid_up = asyncio.run(is_grobid_running())
        if grobid_up:
            mem = container_memory_status(CONTAINER_RUNTIME, GROBID_CONTAINER)
            suffix = f"  RAM: {mem}" if mem else ""
            print(f"GROBID:  running at http://localhost:8070{suffix}")
        else:
            print("GROBID:  not running")

        endpoint = config.llm_endpoint
        health = asyncio.run(_check_api_health(endpoint))
        if health:
            models = [m["id"] for m in health.get("data", [])]
            vllm_name = vllm_container_name(config.llm_model)
            mem = container_memory_status(runtime, vllm_name) if runtime else None
            mem_str = f"  RAM: {mem}" if mem else ""
            print(
                f"vLLM (text):   running at {endpoint} ({', '.join(models)}){mem_str}"
            )
            # Show GPU details on second line
            vram = gpu_memory_status()
            metrics = _fetch_vllm_metrics(endpoint)
            if vram:
                used_gb = vram[0] / (1024**3)
                total_gb = vram[1] / (1024**3)
                vram_str = (
                    f"{used_gb:.1f}GB / {total_gb:.1f}GB ({vram[0] / vram[1]:.0%})"
                )
            else:
                vram_str = None
            gpu_parts = [f"VRAM: {vram_str}"] if vram_str else []
            if metrics:
                gpu_parts.append(metrics)
            if gpu_parts:
                print(f"               {', '.join(gpu_parts)}")
        else:
            print("vLLM (text):   not running")

        # Vision vLLM status
        runtime = _detect_container_runtime()
        vision_up = False
        for vm in VISION_MODELS:
            vm_name = vllm_container_name(vm)
            if runtime and _container_running(runtime, vm_name):
                from sciwrite_lint.vllm.vllm_server import MODELS

                vm_profile = MODELS[vm]
                vm_port = vm_profile.get("port", 5002)
                vm_endpoint = f"http://localhost:{vm_port}/v1"
                vm_health = asyncio.run(_check_api_health(vm_endpoint))
                if vm_health:
                    vm_models = [m["id"] for m in vm_health.get("data", [])]
                    mem = container_memory_status(runtime, vm_name) if runtime else None
                    mem_str = f"  RAM: {mem}" if mem else ""
                    print(
                        f"vLLM (vision): running at {vm_endpoint}"
                        f" ({', '.join(vm_models)}){mem_str}"
                    )
                else:
                    print("vLLM (vision): loading (container up, API not ready)")
                vision_up = True
            else:
                print("vLLM (vision): not running")

        # --- commands ---
        print()
        print("Commands:")
        if not grobid_up or not health:
            print(
                "  sciwrite-lint containers start            # start GROBID + text vLLM"
            )
        if not vision_up:
            print(
                "  sciwrite-lint containers start --vision   # also start vision vLLM"
            )
        print("  sciwrite-lint containers stop             # stop all")
        print("  sciwrite-lint grobid start|stop|status    # manage GROBID alone")
        print("  sciwrite-lint vllm start|stop|status      # manage vLLM alone")
        print("  sciwrite-lint vllm logs [-f]              # follow vLLM logs")

        # --- logs ---
        if runtime:
            log_containers = [
                ("GROBID", GROBID_CONTAINER),
                ("vLLM (text)", vllm_container_name(config.llm_model)),
            ]
            for vm in VISION_MODELS:
                log_containers.append(("vLLM (vision)", vllm_container_name(vm)))
            for label, name in log_containers:
                result = subprocess.run(
                    [runtime, "container", "inspect", name],
                    capture_output=True,
                )
                if result.returncode != 0:
                    continue
                print(f"\n{'─' * 60}")
                print(f"{label} logs (last 15 lines):")
                print(f"{'─' * 60}")
                _print_container_logs(runtime, name, tail=15)

        return 0

    elif action == "start":
        failed = False
        update = getattr(args, "update", False)
        vision = getattr(args, "vision", False)

        if update:
            print(f"Pulling GROBID image: {config.grobid_image}")
            subprocess.run([CONTAINER_RUNTIME, "pull", config.grobid_image])

        print(f"Starting GROBID container (memory limit: {config.grobid_memory})...")
        if asyncio.run(
            start_grobid(memory=config.grobid_memory, image=config.grobid_image)
        ):
            print("GROBID: running at http://localhost:8070")
        else:
            print("GROBID: failed to start within 60s")
            failed = True

        model = getattr(args, "model", None)
        ret = start_container(config, model=model, pull=update)
        if ret != 0:
            failed = True

        if vision:
            for vm in VISION_MODELS:
                ret = start_container(config, model=vm, pull=update)
                if ret != 0:
                    failed = True

        return 1 if failed else 0

    elif action == "stop":
        stop_grobid()
        print("GROBID: stopped")
        stop_container(config, model=getattr(args, "model", None))
        # Also stop any running vision containers
        for vm in VISION_MODELS:
            stop_container(config, model=vm)
        return 0

    elif action == "restart":
        model = getattr(args, "model", None)
        recreate = getattr(args, "recreate", False)
        runtime = _detect_container_runtime()

        stop_grobid()
        stop_container(config, model=model)
        for vm in VISION_MODELS:
            stop_container(config, model=vm)

        if recreate and runtime:
            # Remove containers so they get recreated with current config
            print("Removing containers to apply current config...")
            subprocess.run(
                [runtime, "rm", GROBID_CONTAINER],
                capture_output=True,
            )
            vllm_name = vllm_container_name(config.llm_model)
            subprocess.run(
                [runtime, "rm", vllm_name],
                capture_output=True,
            )
            for vm in VISION_MODELS:
                subprocess.run(
                    [runtime, "rm", vllm_container_name(vm)],
                    capture_output=True,
                )

        print("Containers stopped. Restarting...")
        args.action = "start"
        return run_containers(args)

    elif action == "monitor":
        return _run_containers_monitor(config, interval=getattr(args, "interval", 2))

    return 0


def run_vllm(args: argparse.Namespace) -> int:
    """Dispatch vllm subcommands."""
    from sciwrite_lint.vllm.vllm_server import (
        container_logs,
        remove_container,
        start_container,
        status,
        stop_container,
    )

    config = load_config(
        Path(args.config) if hasattr(args, "config") and args.config else None
    )
    action = args.vllm_action

    if action == "status":
        return status(config)
    elif action == "start":
        return start_container(config, model=args.model, pull=args.update)
    elif action == "stop":
        return stop_container(config, model=args.model)
    elif action == "logs":
        return container_logs(
            config, model=args.model, follow=args.follow, tail=args.tail
        )
    elif action == "rm":
        return remove_container(config, model=args.model, force=args.force)
    elif action == "monitor":
        return _run_vllm_monitor(config, interval=args.interval)

    return 0


def run_vision(args: argparse.Namespace) -> int:
    """Extract and describe manuscript figures.

    Supports two backends:
    - transformers (default): Qwen3-VL-2B in-process, no container needed
    - vllm: Qwen3-VL-8B-FP8 via container on port 5002

    Populates the vision cache (``vision_cache`` table in workspace.db) so
    that full-paper consistency checks can use figure descriptions.

    Normally runs automatically as part of ``sciwrite-lint check``.
    This command is for running vision separately (e.g. to pre-warm cache).
    """
    from sciwrite_lint.__main__ import _load_config, _resolve_paper

    config = _load_config(args)

    # CLI flags override config
    backend = getattr(args, "backend", None)
    if backend:
        config.vision_backend = backend
    device = getattr(args, "device", None) or config.vision_device
    fresh = getattr(args, "fresh", False)

    pc = _resolve_paper(config, args.paper)
    if not pc:
        return 2

    if not pc.file_path.exists():
        logger.error(f"File not found: {pc.file_path}")
        return 1

    from sciwrite_lint.vision.pipeline import run_vision_pipeline

    config.current_paper = pc.name
    result = run_vision_pipeline(
        pc.file_path,
        config,
        paper_name=pc.name,
        device=device,
        fresh=fresh,
    )

    if result:
        print(f"Described figures for {pc.name} — cached in workspace.")
        print(
            "Run 'sciwrite-lint check' to use figure descriptions in consistency checks."
        )
    else:
        print(f"No figures found in {pc.file_path.name}")

    return 0
