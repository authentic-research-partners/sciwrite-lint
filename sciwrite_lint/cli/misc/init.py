"""``sciwrite-lint init`` — scaffold a new project in the current directory."""

from __future__ import annotations

import argparse


def run_init(args: argparse.Namespace) -> int:
    """Initialize a sciwrite-lint project in the current directory."""
    from sciwrite_lint.config import init_project

    success, message = init_project(force=args.force)
    print(message)
    return 0 if success else 1
