"""``sciwrite-lint grobid`` — start/stop/status the GROBID container alone."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sciwrite_lint.config import load_config


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
