"""``sciwrite-lint vision`` — extract and describe manuscript figures."""

from __future__ import annotations

import argparse

from loguru import logger


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
