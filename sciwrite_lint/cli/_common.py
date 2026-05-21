"""Shared CLI helpers used by every ``sciwrite_lint.cli.*`` handler.

These used to live at the top of ``sciwrite_lint/__main__.py``, which
forced every CLI submodule to do ``from sciwrite_lint.__main__ import
_load_config, ...`` inside its run-handler. Re-homing them here lets the
handlers import their helpers without reaching back into the entry-point
module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig, PaperConfig, load_config


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
