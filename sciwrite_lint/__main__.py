"""CLI entry point: sciwrite-lint check paper.tex

Command implementations live in sciwrite_lint.cli submodules:
- cli.check: check, checks
- cli.verify: verify, verify-claims, status
- cli.fetch: fetch
- cli.rank: contributions
- cli.misc: init, parse, override, dismiss-claim, grobid, vllm, vision
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from sciwrite_lint import __version__
from sciwrite_lint.config import LintConfig, PaperConfig, load_config
from sciwrite_lint.vllm.vllm_server import MODELS as _VLLM_MODELS

_ALL_MODELS = sorted(_VLLM_MODELS.keys())
_TEXT_MODELS = [k for k, v in _VLLM_MODELS.items() if v["kind"] == "text"]


# ---------------------------------------------------------------------------
# Helpers (used by CLI submodules)
# ---------------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> LintConfig:
    """Load config from --config flag or auto-discovery."""
    if getattr(args, "config", None):
        return load_config(Path(args.config))

    config = load_config(None)
    if config.config_path is None:
        from sciwrite_lint.config import _detect_papers

        logger.error("No .sciwrite-lint.toml found.")
        detected = _detect_papers()
        if detected:
            logger.error("  Detected .tex files:")
            for p in detected:
                bib = f" (bib: {p['bib']})" if p.get("bib") else ""
                logger.error(f"    {p['file_path']}{bib}")
        logger.error("  Run: sciwrite-lint init")
        logger.error("  Then review .sciwrite-lint.toml before running checks.")
    return config


def _resolve_paper(config: LintConfig, name: str) -> PaperConfig | None:
    """Resolve a paper name to its config. Print error if not found."""
    pc = config.get_paper(name)
    if not pc:
        if config.papers:
            names = ", ".join(p.name for p in config.papers)
            logger.error(f"Unknown paper '{name}'. Registered: {names}")
        else:
            logger.error(f"Unknown paper '{name}'. No papers registered.")
            logger.error(
                f"  Add [[papers]] to {config.config_path or '.sciwrite-lint.toml'}"
            )
    return pc


def _paper_names(config: LintConfig) -> list[str]:
    return [p.name for p in config.papers]


def _resolve_input_files(
    args: argparse.Namespace, config: LintConfig
) -> list[tuple[str, Path]]:
    """Resolve which files to check (.tex or .pdf).

    Priority: positional file > --paper (from config) > all papers in config.
    Returns list of (name, path) pairs.
    """
    if hasattr(args, "file") and args.file:
        p = Path(args.file)
        return [(p.stem, p)]

    paper_filter = getattr(args, "paper", None)
    if paper_filter:
        pc = _resolve_paper(config, paper_filter)
        if not pc:
            return []
        return [(pc.name, pc.file_path)]

    if config.papers:
        return [(pc.name, pc.file_path) for pc in config.papers]

    logger.error("No papers registered. Either:")
    logger.error("  sciwrite-lint check <file.tex|file.pdf> — check a specific file")
    logger.error(
        "  sciwrite-lint init                      — set up project with [[papers]]"
    )
    return []


def _classify_verify_issue(issue: str) -> tuple[str, str]:
    """Classify a verify issue string into (level, rule_id) for findings."""
    from sciwrite_lint.pipeline import _classify_verify_issue as _classify

    return _classify(issue)


def _setup_logging(config: LintConfig) -> None:
    """Configure loguru rotating file sink from config."""
    logger.add(
        "logs/sciwrite-lint.log",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        level=config.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    )


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from sciwrite_lint.cli.check import run_check, run_checks_list
    from sciwrite_lint.cli.config import run_config
    from sciwrite_lint.cli.fetch import run_fetch
    from sciwrite_lint.cli.misc import (
        run_containers,
        run_dismiss_claim,
        run_grobid,
        run_init,
        run_override,
        run_parse,
        run_vision,
        run_vllm,
    )
    from sciwrite_lint.cli.rank import run_contributions
    from sciwrite_lint.cli.verify import (
        run_ref_health,
        run_status,
        run_verify,
        run_verify_claims,
    )

    parser = argparse.ArgumentParser(
        prog="sciwrite-lint",
        description="A linter for scientific manuscripts.",
    )
    parser.add_argument(
        "--version", action="version", version=f"sciwrite-lint {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    # --- check ---
    p_check = sub.add_parser(
        "check",
        help="Full pipeline: verify references, check claims, assess reliability → SciLint Score",
    )
    p_check.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to .tex or .pdf file (.tex: text + LLM rules, .pdf: GROBID parse + checks)",
    )
    p_check.add_argument("--paper", help="Paper name from [[papers]] in config")
    p_check.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_check.add_argument("--format", choices=["terminal", "json"], default=None)
    p_check.add_argument(
        "--fresh",
        action="store_true",
        help="Start from scratch: re-verify references, re-fetch, re-parse, re-run claims (ignore all caches)",
    )
    p_check.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max papers in concurrent stages when checking 2+ papers "
        "(default: 2, validated up to 4). GPU stages always batched into one "
        "subprocess; vLLM and network stages run concurrently up to this "
        "limit. Above 4, the live monitor may fail to track progress and "
        "vLLM/API throughput plateaus.",
    )
    p_check.add_argument(
        "--vision-backend",
        choices=["transformers", "vllm"],
        default=None,
        help="Vision backend: transformers (2B, default) or vllm (8B FP8 on port 5002)",
    )
    p_check.set_defaults(func=run_check)

    # --- checks ---
    p_checks = sub.add_parser("checks", help="List all registered checks")
    p_checks.set_defaults(func=run_checks_list)

    # --- init ---
    p_init = sub.add_parser(
        "init", help="Initialize project: config + references directory"
    )
    p_init.add_argument(
        "--force", action="store_true", help="Overwrite existing config"
    )
    p_init.set_defaults(func=run_init)

    # --- config ---
    p_config = sub.add_parser("config", help="Manage polite email and API keys")
    config_sub = p_config.add_subparsers(dest="config_action")

    p_config_show = config_sub.add_parser(
        "show", help="Show current email and API key status"
    )
    p_config_show.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_config_show.set_defaults(func=run_config)

    p_config_email = config_sub.add_parser(
        "set-email", help="Set polite email (for CrossRef, Unpaywall, Retraction Watch)"
    )
    p_config_email.add_argument("email", help="Your email address")
    p_config_email.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_config_email.set_defaults(func=run_config)

    p_config_set_key = config_sub.add_parser(
        "set-key", help="Save an API key for a service"
    )
    p_config_set_key.add_argument(
        "service",
        choices=["semantic-scholar", "ncbi", "core"],
        help="Service name",
    )
    p_config_set_key.add_argument("key", help="API key value")
    p_config_set_key.set_defaults(func=run_config)

    p_config_rm_key = config_sub.add_parser(
        "remove-key", help="Remove a stored API key"
    )
    p_config_rm_key.add_argument(
        "service",
        choices=["semantic-scholar", "ncbi", "core"],
        help="Service to remove key for",
    )
    p_config_rm_key.set_defaults(func=run_config)

    # --- verify ---
    p_verify = sub.add_parser("verify", help="Verify citations against APIs (slow)")
    p_verify.add_argument("--paper", required=True, help="Paper to verify")
    p_verify.add_argument("--key", help="Verify a single citation key")
    p_verify.add_argument(
        "--fetch", action="store_true", help="Also attempt full-text download"
    )
    p_verify.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_verify.add_argument("--format", choices=["terminal", "json"], default=None)
    p_verify.set_defaults(func=run_verify)

    # --- status ---
    p_status = sub.add_parser("status", help="Show citation verification status")
    p_status.add_argument("--paper", required=True, help="Paper to show")
    p_status.add_argument("--tier", choices=["T1", "T2", "T3"], help="Filter by tier")
    p_status.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_status.set_defaults(func=run_status)

    # --- ref-health ---
    p_rh = sub.add_parser(
        "ref-health",
        help="Fast deterministic reference health check (no API calls)",
    )
    p_rh.add_argument("file", nargs="?", help=".tex file to check")
    p_rh.add_argument("--paper", help="Paper name from config")
    p_rh.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_rh.set_defaults(func=run_ref_health)

    # --- fetch ---
    p_fetch = sub.add_parser("fetch", help="Download full text for T2 citations")
    p_fetch.add_argument("--paper", required=True, help="Paper to fetch for")
    p_fetch.add_argument("--key", help="Fetch for a single citation key")
    p_fetch.add_argument(
        "--all-missing", action="store_true", help="Fetch all T2 citations"
    )
    p_fetch.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_fetch.set_defaults(func=run_fetch)

    # --- parse ---
    p_parse = sub.add_parser(
        "parse", help="Parse T1 PDFs via GROBID and compute embeddings"
    )
    p_parse.add_argument("--paper", required=True, help="Paper name from [[papers]]")
    p_parse.add_argument("--key", help="Parse a single citation key")
    p_parse.add_argument(
        "--fresh", action="store_true", help="Ignore cached results, re-parse all"
    )
    p_parse.add_argument(
        "--no-embed", action="store_true", help="Skip embedding computation"
    )
    p_parse.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_parse.set_defaults(func=run_parse)

    # --- verify-claims ---
    p_vc = sub.add_parser("verify-claims", help="Verify claims against cited sources")
    p_vc.add_argument("--paper", required=True, help="Paper to verify")
    p_vc.add_argument(
        "--backend",
        choices=["vllm", "claude"],
        default="vllm",
        help="vllm (local LLM, fast) or claude (Claude CLI, deep)",
    )
    p_vc.add_argument(
        "--model",
        choices=_TEXT_MODELS,
        default="",
        help="vLLM model preset (default: qwen3)",
    )
    p_vc.add_argument("--key", help="Verify only claims citing this key")
    p_vc.add_argument("--limit", type=int, help="Max claims to verify")
    p_vc.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore cached results, re-verify all claims",
    )
    p_vc.add_argument("--output-dir", default=None, help="Output directory")
    p_vc.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_vc.add_argument("--format", choices=["terminal", "json"], default=None)
    p_vc.set_defaults(func=run_verify_claims)

    # --- override ---
    p_override = sub.add_parser(
        "override", help="Manually override a citation's verification tier"
    )
    p_override.add_argument("--paper", required=True, help="Paper name from [[papers]]")
    p_override.add_argument("--key", required=True, help="Citation key to override")
    p_override.add_argument(
        "--tier",
        choices=["T1", "T2"],
        required=True,
        help="Target tier (T1 if you have full text, T2 if confirmed real)",
    )
    p_override.add_argument(
        "--reason", required=True, help="Why you're overriding (stored in metadata)"
    )
    p_override.add_argument(
        "--clear", action="store_true", help="Remove an existing override"
    )
    p_override.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_override.set_defaults(func=run_override)

    # --- dismiss-claim ---
    p_dismiss = sub.add_parser(
        "dismiss-claim", help="Dismiss a claim finding as false positive"
    )
    p_dismiss.add_argument("--paper", required=True, help="Paper the claim belongs to")
    p_dismiss.add_argument("--key", required=True, help="Citation key")
    p_dismiss.add_argument(
        "--line", type=int, required=True, help="Line number of the claim"
    )
    p_dismiss.add_argument(
        "--reason", required=True, help="Why this is a false positive"
    )
    p_dismiss.add_argument("--clear", action="store_true", help="Remove dismissal")
    p_dismiss.add_argument("--output-dir", default=None, help="Output directory")
    p_dismiss.set_defaults(func=run_dismiss_claim)

    # --- contributions ---
    p_score = sub.add_parser(
        "contributions",
        help="Compute contribution axes and update SciLint Score. Requires vLLM.",
    )
    p_score.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to .tex or .pdf file (standalone scoring, no check needed)",
    )
    p_score.add_argument("--paper", default=None, help="Paper to score")
    p_score.add_argument("--output-dir", default=None, help="Output directory")
    p_score.add_argument(
        "--findings", help="Path to check findings JSON (for internal score)"
    )
    p_score.add_argument(
        "--model", default="", help="vLLM model preset (default: from config)"
    )
    p_score.add_argument(
        "--format",
        choices=["terminal", "json"],
        default="terminal",
        help="Output format (default: terminal)",
        dest="format",
    )
    p_score.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_score.set_defaults(func=run_contributions)

    # --- vision ---
    p_vision = sub.add_parser(
        "vision",
        help="Extract and describe manuscript figures (transformers 2B or vLLM 8B FP8).",
    )
    p_vision.add_argument("--paper", required=True, help="Paper name from [[papers]]")
    p_vision.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Device for VL inference (default: auto — CUDA if available, transformers only)",
    )
    p_vision.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default=None,
        help="Vision backend: transformers (2B, default) or vllm (8B FP8 on port 5002)",
    )
    p_vision.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore vision cache and re-describe all figures",
    )
    p_vision.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_vision.set_defaults(func=run_vision)

    # --- containers (grobid + vllm together) ---
    p_containers = sub.add_parser(
        "containers", help="Manage GROBID and vLLM containers together"
    )
    p_containers.add_argument(
        "action",
        choices=["start", "stop", "restart", "status", "monitor"],
        help="start/stop/restart/status/monitor of all containers",
    )
    p_containers.add_argument("--model", choices=_ALL_MODELS, help="vLLM model to use")
    p_containers.add_argument(
        "--update",
        action="store_true",
        help="Pull latest container images before starting",
    )
    p_containers.add_argument(
        "-i",
        "--interval",
        type=float,
        default=2,
        help="Monitor refresh interval in seconds (default: 2)",
    )
    p_containers.add_argument(
        "--recreate",
        action="store_true",
        help="Remove and recreate containers (applies config changes like memory limits)",
    )
    p_containers.add_argument(
        "--vision",
        action="store_true",
        help="Also start vision vLLM (qwen3-vl-8b-fp8 on port 5002)",
    )
    p_containers.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_containers.set_defaults(func=run_containers)

    # --- grobid ---
    p_grobid = sub.add_parser("grobid", help="Manage GROBID container")
    p_grobid.add_argument(
        "action",
        choices=["start", "stop", "status"],
        help="start/stop/status of GROBID container",
    )
    p_grobid.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_grobid.set_defaults(func=run_grobid)

    # --- vllm ---
    p_vllm = sub.add_parser(
        "vllm", help="Manage local vLLM server for LLM-engine rules"
    )
    vllm_sub = p_vllm.add_subparsers(dest="vllm_action")

    p_vllm_status = vllm_sub.add_parser(
        "status", help="Show vLLM status and API health"
    )
    p_vllm_status.set_defaults(func=run_vllm)

    p_vllm_start = vllm_sub.add_parser("start", help="Start vLLM server")
    p_vllm_start.add_argument("--model", choices=_ALL_MODELS, help="Model to serve")
    p_vllm_start.add_argument(
        "--update",
        action="store_true",
        help="Pull latest vLLM image before starting",
    )
    p_vllm_start.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_vllm_start.set_defaults(func=run_vllm)

    p_vllm_stop = vllm_sub.add_parser("stop", help="Stop vLLM server")
    p_vllm_stop.add_argument("--model", choices=_ALL_MODELS, help="Model to stop")
    p_vllm_stop.set_defaults(func=run_vllm)

    p_vllm_logs = vllm_sub.add_parser("logs", help="Show vLLM container logs")
    p_vllm_logs.add_argument("--model", choices=_ALL_MODELS)
    p_vllm_logs.add_argument(
        "-f", "--follow", action="store_true", help="Follow log output"
    )
    p_vllm_logs.add_argument("-n", "--tail", type=int, default=50, help="Lines to show")
    p_vllm_logs.set_defaults(func=run_vllm)

    p_vllm_rm = vllm_sub.add_parser("rm", help="Remove vLLM container")
    p_vllm_rm.add_argument("--model", choices=_ALL_MODELS)
    p_vllm_rm.add_argument(
        "--force", action="store_true", help="Force remove even if running"
    )
    p_vllm_rm.set_defaults(func=run_vllm)

    p_vllm_monitor = vllm_sub.add_parser(
        "monitor", help="Live terminal monitor for vLLM server"
    )
    p_vllm_monitor.add_argument(
        "-i",
        "--interval",
        type=float,
        default=2,
        help="Refresh interval in seconds (default: 2)",
    )
    p_vllm_monitor.add_argument("--config", help="Path to .sciwrite-lint.toml")
    p_vllm_monitor.set_defaults(func=run_vllm)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "config" and not getattr(args, "config_action", None):
        p_config.print_help()
        return 0

    if args.command == "vllm" and not getattr(args, "vllm_action", None):
        p_vllm.print_help()
        return 0

    # Set up file logging from config (stderr sink stays at INFO by default)
    try:
        config = load_config(
            Path(args.config) if getattr(args, "config", None) else None
        )
    except Exception as e:
        logger.debug("Config load failed, using defaults: {}", e)
        config = LintConfig()
    _setup_logging(config)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
