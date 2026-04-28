"""``sciwrite-lint vllm`` — manage the vLLM container alone."""

from __future__ import annotations

import argparse
from pathlib import Path

from sciwrite_lint.config import load_config


def run_vllm(args: argparse.Namespace) -> int:
    """Dispatch vllm subcommands."""
    from sciwrite_lint.cli.misc._monitor import _run_vllm_monitor
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
