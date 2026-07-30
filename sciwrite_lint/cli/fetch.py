"""CLI handlers for 'fetch' command and fetch helpers."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.pipeline.fetch import _coerce_year as _coerce_year


def eager_parse(key: str, pdf_path: Path, refs_dir: Path) -> None:
    """Parse a newly-downloaded PDF and compute embeddings (best-effort)."""
    try:
        from sciwrite_lint.references.reference_store import parse_and_embed

        text, chunks = asyncio.run(parse_and_embed(key, pdf_path, refs_dir))
        if text:
            suffix = f", {chunks} chunks embedded" if chunks else ""
            logger.info(f"Parsed {key}: {len(text)} chars{suffix}")
        else:
            logger.warning(f"Parse failed for {key} (GROBID running?)")
    except Exception as e:
        logger.warning(f"Parse skipped for {key}: {e}")


def fetch_for_citations(
    citations: list,
    config: LintConfig,
    references_dir: Path,
) -> None:
    """Attempt full-text acquisition for citations missing local files."""
    from sciwrite_lint.fulltext import acquire_fulltext
    from sciwrite_lint.references.metadata import (
        compute_tier,
        load_metadata,
        save_metadata,
    )

    refs_dir = references_dir

    # Hash-aware drop-folder ingest before the OA waterfall. Runs for
    # every citation so a refreshed source on an already-T1 ref still
    # re-propagates through GROBID + embedding on the next run.
    from sciwrite_lint.local_sources import ingest_local_sources

    keys_titles_hashes: dict[str, tuple[str, str]] = {}
    for c in citations:
        if c.local_status != "none":
            continue
        meta = load_metadata(c.key, refs_dir)
        if not meta:
            continue
        keys_titles_hashes[c.key] = (
            c.title or meta.canonical.get("title", ""),
            meta.access.get("local_file_src_hash", ""),
        )

    if keys_titles_hashes:
        outcomes = ingest_local_sources(
            keys_titles_hashes,
            academic_dir=config.effective_local_pdfs_dir(),
            web_dir=config.effective_local_web_dir(),
            references_dir=refs_dir,
        )
        for key, outcome in outcomes.items():
            meta = load_metadata(key, refs_dir)
            if not meta:
                continue
            meta.access["local_file"] = outcome.local_file
            meta.access["local_file_src_hash"] = outcome.src_hash
            meta.access["tier"] = compute_tier(meta)
            save_metadata(meta, refs_dir)
            logger.info(
                f"{key}: ingested local {outcome.kind} source [{meta.access['tier']}]"
            )

    async def _do_fetch():
        for c in citations:
            if c.local_status != "none":
                continue

            meta = load_metadata(c.key, refs_dir)
            if not meta:
                continue

            tier = meta.access.get("tier", "")
            if tier == "T1":
                continue

            doi = meta.canonical.get("doi") or c.doi
            arxiv_id = meta.canonical.get("arxiv_id")
            oa_url = meta.access.get("oa_url")
            s2_pdf_url = meta.canonical.get("s2_pdf_url")
            pmcid = meta.canonical.get("pmcid")
            expected_title = meta.canonical.get("title", "") or c.title
            expected_authors = meta.canonical.get("authors") or c.authors
            expected_year = _coerce_year(
                meta.canonical.get("year") or meta.bibitem.get("year")
            )
            expected_entry_type = meta.bibitem.get("entry_type") or "article"

            logger.info(f"Fetching {c.key}...")
            result = await acquire_fulltext(
                c.key,
                refs_dir,
                config=config,
                doi=doi,
                arxiv_id=arxiv_id,
                oa_url=oa_url,
                s2_pdf_url=s2_pdf_url,
                pmcid=pmcid,
                expected_title=expected_title,
                expected_authors=expected_authors,
                expected_year=expected_year,
                expected_entry_type=expected_entry_type,
            )

            if result.found and result.local_path:
                meta.access["local_file"] = result.local_path
                meta.access["tier"] = compute_tier(meta)
                save_metadata(meta, refs_dir)
                logger.info(
                    f"{c.key}: downloaded {result.local_path} [now {meta.access['tier']}]"
                )

                # Eager parse + embed: cache GROBID output and compute embeddings
                if result.local_path.endswith(".pdf"):
                    from sciwrite_lint.references.reference_store import parse_and_embed

                    try:
                        text, chunks = await parse_and_embed(
                            c.key, refs_dir / result.local_path, refs_dir
                        )
                        if text:
                            suffix = f", {chunks} chunks embedded" if chunks else ""
                            logger.info(f"{c.key}: parsed {len(text)} chars{suffix}")
                    except Exception as e:
                        logger.warning(f"{c.key}: parse skipped: {e}")
            elif result.url:
                if result.is_oa:
                    meta.access["is_oa"] = True
                if result.oa_url:
                    meta.access["oa_url"] = result.oa_url
                if result.is_oa or result.oa_url:
                    save_metadata(meta, refs_dir)
                logger.warning(
                    f"{c.key}: manual download needed: {result.url} "
                    f"(save PDF to local_pdfs/ or MHTML to local_web/)"
                )
            else:
                logger.warning(f"{c.key}: no source found")

            if result.abstract and not meta.canonical.get("abstract"):
                meta.canonical["abstract"] = result.abstract
                meta.access["tier"] = compute_tier(meta)
                save_metadata(meta, refs_dir)

    asyncio.run(_do_fetch())


def run_fetch(args: argparse.Namespace) -> int:
    """Download full text for citations missing local files."""
    from sciwrite_lint.references.metadata import load_all_metadata
    from sciwrite_lint.pipeline import extract_citations_for_paper

    from sciwrite_lint.cli._common import _load_config, _resolve_paper
    from sciwrite_lint.cli.config import check_api_config

    config = _load_config(args)

    api_errors = check_api_config(config, needs_email=True)
    if api_errors:
        for e in api_errors:
            logger.error(f"  ✗ {e}")
        return 2
    pc = _resolve_paper(config, args.paper)
    if not pc:
        return 2

    ws = config.paper_workspace(pc.name)
    ws.ensure_dirs()
    refs_dir = ws.root
    config.current_paper = pc.name

    if args.key:
        all_meta = load_all_metadata(refs_dir)
        meta = all_meta.get(args.key)
        if not meta:
            print(f"No metadata for '{args.key}'. Run 'sciwrite-lint verify' first.")
            return 1
        fetch_single(
            args.key,
            meta,
            config,
            references_dir=refs_dir,
        )
        return 0

    if not pc.file_path.exists():
        logger.error(f"Error: {pc.file_path} not found")
        return 1

    citations = asyncio.run(
        extract_citations_for_paper(pc.file_path, config, bib_path=pc.bib)
    )

    logger.info(f"Fetching full text for {pc.name}...")
    fetch_for_citations(
        citations,
        config,
        references_dir=refs_dir,
    )
    return 0


def fetch_single(
    key: str,
    meta: object,
    config: LintConfig,
    references_dir: Path,
) -> None:
    """Fetch full text for a single citation."""
    from sciwrite_lint.fulltext import acquire_fulltext
    from sciwrite_lint.references.metadata import compute_tier, save_metadata

    refs_dir = references_dir

    doi = meta.canonical.get("doi")  # type: ignore[attr-defined]
    arxiv_id = meta.canonical.get("arxiv_id")  # type: ignore[attr-defined]
    oa_url = meta.access.get("oa_url")  # type: ignore[attr-defined]
    s2_pdf_url = meta.canonical.get("s2_pdf_url")  # type: ignore[attr-defined]
    pmcid = meta.canonical.get("pmcid")  # type: ignore[attr-defined]
    expected_title = meta.canonical.get("title", "")  # type: ignore[attr-defined]
    expected_authors = meta.canonical.get("authors")  # type: ignore[attr-defined]
    expected_year = _coerce_year(
        meta.canonical.get("year") or meta.bibitem.get("year")  # type: ignore[attr-defined]
    )
    expected_entry_type = (
        meta.bibitem.get("entry_type") or "article"  # type: ignore[attr-defined]
    )

    async def _do():
        logger.info(f"Fetching full text for {key}...")
        result = await acquire_fulltext(
            key,
            refs_dir,
            config=config,
            doi=doi,
            arxiv_id=arxiv_id,
            oa_url=oa_url,
            s2_pdf_url=s2_pdf_url,
            pmcid=pmcid,
            expected_title=expected_title,
            expected_authors=expected_authors,
            expected_year=expected_year,
            expected_entry_type=expected_entry_type,
        )

        if result.found and result.local_path:
            meta.access["local_file"] = result.local_path  # type: ignore[attr-defined]
            meta.access["tier"] = compute_tier(meta)  # type: ignore[attr-defined]
            save_metadata(meta, refs_dir)  # type: ignore[arg-type]
            logger.info(
                f"{key}: downloaded {result.local_path} [now {meta.access['tier']}]"  # type: ignore[attr-defined]
            )

            # Eager parse + embed
            if result.local_path.endswith(".pdf"):
                from sciwrite_lint.references.reference_store import parse_and_embed

                try:
                    text, chunks = await parse_and_embed(
                        key, refs_dir / result.local_path, refs_dir
                    )
                    if text:
                        suffix = f", {chunks} chunks embedded" if chunks else ""
                        logger.info(f"{key}: parsed {len(text)} chars{suffix}")
                except Exception as e:
                    logger.warning(f"{key}: parse skipped: {e}")
        elif result.url:
            logger.warning(
                f"{key}: manual download needed: {result.url} "
                f"(save PDF to local_pdfs/ or MHTML to local_web/)"
            )
        else:
            logger.warning(f"{key}: no source found")

    asyncio.run(_do())
