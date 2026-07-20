"""``sciwrite-lint init`` — scaffold a new project in the current directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def run_init(args: argparse.Namespace) -> int:
    """Initialize a sciwrite-lint project in the current directory.

    With a positional ``file``, register that specific manuscript (any
    supported type, including PDF). Without it, auto-detect.
    """
    from sciwrite_lint.config import init_project

    explicit_file: Path | None = None
    raw = getattr(args, "file", None)
    if raw:
        from sciwrite_lint.cli._common import (
            MANUSCRIPT_SUFFIXES,
            unsupported_manuscript_error,
        )

        explicit_file = Path(raw)
        if not explicit_file.exists():
            logger.error(f"Error: {explicit_file} not found")
            return 1
        if explicit_file.suffix.lower() not in MANUSCRIPT_SUFFIXES:
            logger.error(unsupported_manuscript_error(explicit_file))
            return 1

    success, message = init_project(force=args.force, explicit_file=explicit_file)
    print(message)
    return 0 if success else 1
