"""``sciwrite-lint override`` — manually set a citation's tier."""

from __future__ import annotations

import argparse


def run_override(args: argparse.Namespace) -> int:
    """Manually override a citation's verification tier."""
    from datetime import date

    from sciwrite_lint.cli._common import _load_config, _resolve_paper
    from sciwrite_lint.models import CitationMetadata
    from sciwrite_lint.references.metadata import (
        compute_tier,
        load_metadata,
        save_metadata,
    )

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
