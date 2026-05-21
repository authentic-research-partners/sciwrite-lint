"""Stage 3: full-text acquisition for cited references.

Downloads missing PDFs for T1/T2 references (via OA, arXiv, S2, PMC, or
matched local PDF) and eagerly parses+embeds them inline unless the
orchestrator asks for batch embedding (``embed_inline=False``).

Negative-result cache
---------------------
Acquisition failures are classified by ``acquire_fulltext`` as either
``"exhausted"`` (every OA source was reached and returned a definitive
negative — not found / no match / title mismatch / too large) or
``"transient"`` (network error, timeout, 5xx, or no sources to try).

Only exhausted failures are cached, as ``acquisition_status`` plus
``acquisition_attempted_at`` on the citation's ``access`` metadata.
Within ``config.fetch_retry_ttl_days`` (default 30) such refs are
skipped by the ``need_fetch`` selection below, which avoids re-running
~14 source lookups per ref on every pipeline invocation. Transient
failures are never cached — they retry on the next run.

To force a retry of cached negatives before the TTL expires, use
``sciwrite-lint check --paper <name> --fresh`` (recreates the workspace,
dropping every cache including this one).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Citation


def _coerce_year(value: object) -> int | None:
    """Coerce a year-ish value from bib/canonical metadata to an int.

    Year fields are stored as ``str`` or ``int`` across the metadata
    surface. Returns ``None`` for empty, non-numeric, or out-of-range
    values — the validator treats ``None`` as "no year signal available".
    """
    if isinstance(value, int):
        return value if 1500 <= value <= 2100 else None
    if isinstance(value, str):
        match = value.strip()[:4]
        if match.isdigit():
            year = int(match)
            return year if 1500 <= year <= 2100 else None
    return None


def _is_fresh_negative(meta_access: dict, ttl_days: int) -> bool:
    """True if this ref has a cached exhausted failure still within the TTL.

    Returns False when the status is absent, not exhausted, the
    timestamp is missing or unparseable, or the TTL has elapsed. On any
    parse error we prefer a retry over silently trusting a corrupt
    timestamp — the cost is one more attempt, not a wrong result.
    """
    if meta_access.get("acquisition_status") != "exhausted":
        return False
    stamp = meta_access.get("acquisition_attempted_at")
    if not isinstance(stamp, str) or not stamp:
        return False
    try:
        attempted = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if attempted.tzinfo is None:
        attempted = attempted.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - attempted).total_seconds()
    return age_seconds < ttl_days * 86400


async def _stage_fetch(
    citations: list[Citation],
    config: LintConfig,
    references_dir: Path,
    embed_inline: bool = True,
) -> int:
    """Download PDFs for citations missing local files. Returns count fetched.

    Args:
        embed_inline: If True (default), eagerly embed parsed PDFs in-process.
            Set to False in batch-staged mode to avoid loading the embedding
            model's CUDA context in the parent process — the batch embedding
            subprocess handles all embeddings instead.
    """
    from sciwrite_lint.fulltext import acquire_fulltext
    from sciwrite_lint.local_sources import ingest_local_sources
    from sciwrite_lint.references.metadata import (
        compute_tier,
        load_metadata,
        save_metadata,
    )
    from sciwrite_lint.references.reference_store import parse_and_embed
    from sciwrite_lint.usage import tracked

    # Load metadata once for every citation so the drop-folder ingest
    # pass below and the need_fetch filter below share the same objects.
    meta_by_key: dict[str, Any] = {}
    for c in citations:
        meta = load_metadata(c.key, references_dir)
        if meta:
            meta_by_key[c.key] = meta

    # --- Hash-aware drop-folder ingest ---------------------------------
    # Runs for EVERY citation (regardless of tier) so a refreshed source
    # on an already-T1 ref is picked up: ``local_file_src_hash`` in
    # meta.access is compared against the current source SHA-256, and
    # only when they differ (or no prior hash exists) do we re-copy /
    # re-convert and recompute the tier. Unchanged sources cost a single
    # hash read and exit.
    keys_titles_hashes: dict[str, tuple[str, str]] = {
        c.key: (
            c.title or meta_by_key[c.key].canonical.get("title", ""),
            meta_by_key[c.key].access.get("local_file_src_hash", ""),
        )
        for c in citations
        if c.key in meta_by_key
    }
    outcomes = ingest_local_sources(
        keys_titles_hashes,
        academic_dir=config.effective_local_pdfs_dir(),
        web_dir=config.effective_local_web_dir(),
        references_dir=references_dir,
    )
    for key, outcome in outcomes.items():
        meta = meta_by_key[key]
        meta.access["local_file"] = outcome.local_file
        meta.access["local_file_src_hash"] = outcome.src_hash
        meta.access["tier"] = compute_tier(meta)
        save_metadata(meta, references_dir)
        logger.info(
            f"{key}: ingested local {outcome.kind} source [{meta.access['tier']}]"
        )

    # --- need_fetch filter (unchanged policy, but sees updated meta) ---
    need_fetch = []
    ttl_days = config.fetch_retry_ttl_days
    skipped_cached = 0

    for c in citations:
        meta = meta_by_key.get(c.key)
        if not meta:
            continue
        tier = meta.access.get("tier", "")
        local = meta.access.get("local_file", "")
        if tier == "T1" and local and (references_dir / local).exists():
            continue
        if tier == "T3" or meta.api_match in (
            "not_found",
            "web_verified",
            "web_dead",
            "web_blocked",
        ):
            continue
        # Negative-result cache: a previous run already exhausted every
        # OA source for this ref with definitive negatives. Re-running
        # the full source ladder (arXiv, S2, Unpaywall, PMC, CORE, …)
        # would hit the same answers and burn ~14 rate-limited API
        # calls per ref. Skip until the TTL expires or --fresh is used.
        if _is_fresh_negative(meta.access, ttl_days):
            skipped_cached += 1
            continue
        need_fetch.append((c, meta))

    if skipped_cached:
        logger.debug(
            f"Skipped {skipped_cached} refs with cached exhausted-acquisition "
            f"failures (TTL {ttl_days}d; use --fresh to retry)"
        )

    if not need_fetch:
        return 0

    logger.info(f"Fetching full text for {len(need_fetch)} citations")
    sem = asyncio.Semaphore(5)  # max 5 concurrent downloads
    fetched_keys: list[str] = []

    async def _fetch_one(c: Citation, meta) -> None:
        doi = meta.canonical.get("doi") or c.doi
        arxiv_id = meta.canonical.get("arxiv_id")
        oa_url = meta.access.get("oa_url")
        s2_pdf_url = meta.canonical.get("s2_pdf_url")
        pmcid = meta.canonical.get("pmcid")
        expected_title = c.title or meta.canonical.get("title", "")
        expected_authors = meta.canonical.get("authors") or c.authors
        expected_year = _coerce_year(
            meta.canonical.get("year") or meta.bibitem.get("year")
        )
        expected_entry_type = meta.bibitem.get("entry_type") or "article"

        async with sem:
            async with tracked("fetch"):
                result = await acquire_fulltext(
                    c.key,
                    references_dir,
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
            save_metadata(meta, references_dir)
            fetched_keys.append(c.key)
            logger.info(f"{c.key}: downloaded [{meta.access['tier']}]")

            # Eager parse (also concurrent — GROBID handles it).
            # In batch mode (embed_inline=False), skip in-process embedding
            # to avoid loading the CUDA context in the parent process.
            if result.local_path.endswith(".pdf"):
                try:
                    text, chunks = await parse_and_embed(
                        c.key,
                        references_dir / result.local_path,
                        references_dir,
                        embed=embed_inline,
                    )
                    if text:
                        suffix = f", {chunks} chunks" if chunks else ""
                        logger.debug(f"{c.key}: parsed {len(text)} chars{suffix}")

                    # Store formal classification in citation metadata
                    from sciwrite_lint.references.reference_store import (
                        is_formal_cached,
                    )

                    formal = is_formal_cached(c.key, references_dir)
                    meta.access["is_formal"] = formal
                    save_metadata(meta, references_dir)
                    if not formal:
                        logger.info(
                            f"{c.key}: non-formal document — "
                            f"text available, no reference extraction"
                        )
                except Exception as e:
                    logger.warning(f"{c.key}: parse failed: {e}")
        elif not result.found:
            if result.reason:
                meta.access["acquisition_reason"] = result.reason
            if result.is_oa:
                meta.access["is_oa"] = True
            if result.oa_url:
                meta.access["oa_url"] = result.oa_url
            # Only exhausted failures enter the skip-cache; transient
            # ones (timeouts, 5xx, connection errors) must retry next
            # run. We stamp the attempt time so the TTL is measured
            # from the last real attempt, not the first.
            status = getattr(result, "status", "transient")
            if status == "exhausted":
                meta.access["acquisition_status"] = "exhausted"
                meta.access["acquisition_attempted_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
            elif "acquisition_status" in meta.access:
                # A previously-cached exhausted result turned transient
                # on retry (e.g., a source is flapping). Drop the cache
                # so we keep retrying instead of silently returning to
                # an expired negative.
                meta.access.pop("acquisition_status", None)
                meta.access.pop("acquisition_attempted_at", None)
            if result.reason or result.is_oa or result.oa_url:
                save_metadata(meta, references_dir)
            logger.debug(
                f"{c.key}: PDF not acquired (stays T2)"
                + (f" — {result.reason}" if result.reason else "")
                + f" [{status}]"
            )

        if result.abstract and not meta.canonical.get("abstract"):
            meta.canonical["abstract"] = result.abstract
            meta.access["tier"] = compute_tier(meta)
            save_metadata(meta, references_dir)

    await asyncio.gather(*[_fetch_one(c, meta) for c, meta in need_fetch])
    return len(fetched_keys)
