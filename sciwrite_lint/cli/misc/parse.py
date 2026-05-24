"""``sciwrite-lint parse`` — parse PDFs via GROBID and compute embeddings."""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger


def run_parse(args: argparse.Namespace) -> int:
    """Parse PDFs via GROBID and store results + embeddings."""
    from sciwrite_lint.cli._common import _load_config, _resolve_paper
    from sciwrite_lint.references.reference_store import (
        parse_all_missing,
        parse_and_embed,
    )

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

    print("Parsing all references with local PDFs...")
    from sciwrite_lint.pdf.grobid import is_grobid_running

    if not asyncio.run(is_grobid_running()):
        logger.error("GROBID not running. Start with: sciwrite-lint containers start")
        return 1

    results = asyncio.run(parse_all_missing(refs_dir, force=force))

    cached = sum(1 for v in results.values() if v == "cached")
    parsed = sum(1 for v in results.values() if v == "parsed")
    failed = sum(1 for v in results.values() if v == "failed")

    if embed and parsed:
        from sciwrite_lint.references.reference_store import (
            compute_and_store_embeddings,
        )

        logger.info(f"Computing embeddings for {parsed} newly parsed references...")
        for key, status in results.items():
            if status == "parsed":
                md_path = refs_dir / "parsed" / f"{key}.md"
                if md_path.exists():
                    try:
                        text = md_path.read_text(encoding="utf-8")
                        n = compute_and_store_embeddings(key, text, refs_dir)
                        print(f"    {key}: {n} chunks")
                    except Exception as e:
                        logger.warning("parse: embedding failed for {}: {}", key, e)

    print(f"\n  Summary: {cached} cached, {parsed} parsed, {failed} failed")
    if failed:
        for key, status in results.items():
            if status == "failed":
                print(f"    FAILED: {key}")

    return 0
