"""CLI handlers for 'verify', 'verify-claims', 'status', and 'ref-health' commands."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from sciwrite_lint.models import CheckResult, Citation, CitationMetadata, Finding


def classify_t2_refs(
    citations: list[Citation],
    all_meta: dict[str, CitationMetadata],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split T2 refs into OA (manual download) and other (no source).

    Returns (t2_oa, t2_other) where each is a list of (key, url_or_reason).
    """
    t2_oa: list[tuple[str, str]] = []
    t2_other: list[tuple[str, str]] = []
    for c in sorted(citations, key=lambda x: x.key):
        meta = all_meta.get(c.key)
        if not meta:
            continue
        tier = meta.access.get("tier", "NC")
        if tier == "T2":
            if meta.access.get("is_oa") and meta.access.get("oa_url"):
                t2_oa.append((c.key, meta.access["oa_url"]))
            else:
                reason = meta.access.get("acquisition_reason", "")
                if reason:
                    t2_other.append((c.key, reason))
    return t2_oa, t2_other


_TIER_MARKERS = {
    "T1": ("\u2713", "has full text"),
    "T2": ("~", "confirmed"),
    "T3": ("\u2717", "unverified"),
}


def run_verify(args: argparse.Namespace) -> int:
    """Verify citations against APIs (CrossRef, OpenAlex, Semantic Scholar, Open Library, Library of Congress)."""
    from sciwrite_lint.api import CitationAPI, verify_citations
    from sciwrite_lint.references.citations import get_last_citation_source
    from sciwrite_lint.pipeline import extract_citations_for_paper

    from sciwrite_lint.cli._common import _load_config, _resolve_paper
    from sciwrite_lint.cli.config import check_api_config
    from sciwrite_lint.cli.fetch import fetch_for_citations

    config = _load_config(args)
    check_api_config(config, needs_email=False)

    pc = _resolve_paper(config, args.paper)
    if not pc:
        return 2

    if not pc.file_path.exists():
        logger.error(f"Error: {pc.file_path} not found")
        return 1

    ws = config.paper_workspace(pc.name)
    ws.ensure_dirs()
    refs_dir = ws.root
    config.current_paper = pc.name

    logger.info(f"Verifying {pc.name}...")
    citations = asyncio.run(
        extract_citations_for_paper(pc.file_path, config, bib_path=pc.bib)
    )
    src = get_last_citation_source()
    if src:
        logger.info(f"Citation source: {src}")

    if args.key:
        citations = [c for c in citations if c.key == args.key]
        if not citations:
            logger.error(f"Key '{args.key}' not found in {pc.name}")
            return 1

    async def _verify():
        async with CitationAPI(config=config) as api:
            await verify_citations(
                citations,
                api,
                config=config,
                references_dir=refs_dir,
            )

    asyncio.run(_verify())

    from sciwrite_lint.pipeline import _classify_verify_issue

    result = CheckResult(checker="verify", paper=pc.name)
    for c in citations:
        for issue in c.issues:
            level, rule_id = _classify_verify_issue(issue)
            result.findings.append(
                Finding(
                    level=level,
                    rule_id=rule_id,
                    message=f"{c.key}: {issue}",
                    file=pc.file_path.name,
                    context=f"Source: {c.api_source}" if c.api_source else "",
                )
            )

    # Reference-unreliable: fast path (metadata only)
    from sciwrite_lint.checks.reference_unreliable import (
        metadata_to_unreliable_findings,
    )
    from sciwrite_lint.references.metadata import load_all_metadata

    all_meta = load_all_metadata(refs_dir)
    if all_meta:
        result.findings.extend(metadata_to_unreliable_findings(all_meta, pc.file_path))

    from sciwrite_lint.report import format_findings

    label = f"{pc.name} ({pc.file_path.name})"
    format_findings(result.findings, label, fmt=args.format, color=config.color)

    if args.fetch:
        from sciwrite_lint.cli.config import check_api_config

        api_errors = check_api_config(config, needs_email=True)
        if api_errors:
            for e in api_errors:
                logger.error(f"  ✗ {e}")
            return 2
        fetch_for_citations(
            citations,
            config,
            references_dir=refs_dir,
        )

    return 1 if result.errors else 0


def run_status(args: argparse.Namespace) -> int:
    """Show citation verification status."""
    from sciwrite_lint.references.citations import get_last_citation_source
    from sciwrite_lint.references.metadata import load_all_metadata
    from sciwrite_lint.pipeline import extract_citations_for_paper

    from sciwrite_lint.cli._common import _load_config, _resolve_paper

    config = _load_config(args)
    pc = _resolve_paper(config, args.paper)
    if not pc:
        return 2

    ws = config.paper_workspace(pc.name)
    config.current_paper = pc.name
    all_meta = load_all_metadata(ws.root)

    if not all_meta:
        print("No metadata found. Run 'sciwrite-lint verify' first.")
        return 0

    if not pc.file_path.exists():
        logger.error(f"{pc.file_path} not found")
        return 1

    citations = asyncio.run(
        extract_citations_for_paper(pc.file_path, config, bib_path=pc.bib)
    )
    src = get_last_citation_source()

    print(f"\n{'=' * 70}")
    print(f"  {pc.name} \u2014 {len(citations)} citations")
    if src:
        print(f"  Source: {src}")
    print(f"{'=' * 70}")
    print(
        f"  {'Key':<25} {'Tier':<5} {'Type':<5} {'API Match':<15} {'Local File':<30} {'Verified'}"
    )
    print(f"  {'-' * 25} {'-' * 5} {'-' * 5} {'-' * 15} {'-' * 30} {'-' * 10}")

    tier_counts = {"T1": 0, "T2": 0, "T3": 0, "NC": 0}
    academic_count = 0
    web_count = 0

    for c in sorted(citations, key=lambda x: x.key):
        meta = all_meta.get(c.key)
        if meta:
            tier = meta.access.get("tier", "NC")
            api_match: str = meta.api_match or "?"
            local = meta.access.get("local_file") or "\u2014"
            verified = meta.verified_date or "\u2014"
            is_web = api_match in ("web_verified", "web_dead", "web_blocked")
        else:
            tier = "NC"
            api_match = "not checked"
            local = c.local_path.split("/")[-1] if c.local_path else "\u2014"
            verified = "\u2014"
            is_web = False

        if is_web:
            web_count += 1
        else:
            academic_count += 1

        if args.tier and tier != args.tier:
            continue

        tier_counts[tier if tier in tier_counts else "NC"] += 1

        marker, _ = _TIER_MARKERS.get(tier, ("\u00b7", ""))
        rtype = "web" if is_web else "acad"
        print(
            f"  {c.key:<25} {marker}{tier:<4} {rtype:<5} {api_match:<15} {local:<30} {verified}"
        )

    print()
    total = sum(tier_counts.values())
    t1, t2, t3, nc = (tier_counts.get(k, 0) for k in ("T1", "T2", "T3", "NC"))
    print(
        f"  Summary: {t1} T1 (has full text), "
        f"{t2} T2 (confirmed), "
        f"{t3} T3 (unverified), "
        f"{nc} not checked  |  {total} total"
    )
    print(f"  Types:   {academic_count} academic, {web_count} web resources")
    print()
    print("  T1 = verified + full text (can check claims)")
    print("  T2 = verified, no full text (confirmed real)")
    print("  T3 = not found in APIs or dead URL")

    # Show T2 refs split by OA status
    t2_oa, t2_other = classify_t2_refs(citations, all_meta)
    if t2_oa:
        print()
        print("  T2 open access — manual download needed:")
        print(
            "  (open URL in browser; save a PDF to local_pdfs/ OR, for HTML pages"
            " / JS-heavy sites, File -> Save As -> Webpage Single File (.mhtml)"
            " into local_web/ — filename should start with the citekey)"
        )
        for key, url in t2_oa:
            print(f"    {key}: {url}")
    if t2_other:
        print()
        print("  T2 acquisition details:")
        for key, reason in t2_other:
            print(f"    {key}: {reason}")

    return 0


def run_ref_health(args: argparse.Namespace) -> int:
    """Fast deterministic reference health check (no API calls).

    Parses the .tex and .bib, reports:
    - cite/bib mismatches (cited but missing from bib, in bib but never cited)
    - ID coverage (which refs have DOI, ISBN, arXiv, PMID, etc.)
    - entry type breakdown
    """
    from sciwrite_lint.references.citations import (
        extract_bibitems,
        find_orphans,
        is_web_resource,
    )

    # Resolve input: positional file or --paper
    tex_file = getattr(args, "file", None)
    paper_name = getattr(args, "paper", None)
    if not tex_file and not paper_name:
        logger.error("Provide a .tex file or --paper NAME")
        return 2

    if tex_file:
        tex_path = Path(tex_file).resolve()
        bib_path = None
        name = tex_path.stem
    else:
        from sciwrite_lint.cli._common import _load_config, _resolve_paper

        config = _load_config(args)
        assert paper_name is not None  # guarded above
        pc = _resolve_paper(config, paper_name)
        if not pc:
            return 2
        tex_path = pc.file_path
        bib_path = pc.bib
        name = pc.name

    if not tex_path.exists():
        logger.error(f"{tex_path} not found")
        return 1

    if tex_path.suffix != ".tex":
        logger.error(f"ref-health requires a .tex file, got {tex_path.suffix}")
        return 1

    citations = extract_bibitems(tex_path, bib_path=bib_path)

    if not citations:
        print(f"\n  {name}: no references found in {tex_path.name}")
        return 0

    cited_no_bib, bib_no_cite = find_orphans(citations, tex_path)

    # --- ID coverage ---
    id_fields = ["doi", "arxiv_id", "pmid", "pmc_id", "isbn", "lccn", "url"]
    has_any_id: list[str] = []
    no_id: list[str] = []
    id_counts: dict[str, int] = {f: 0 for f in id_fields}

    for c in citations:
        ids_present = [f for f in id_fields if getattr(c, f, "")]
        if ids_present:
            has_any_id.append(c.key)
            for f in ids_present:
                id_counts[f] += 1
        else:
            no_id.append(c.key)

    # --- Entry type breakdown ---
    type_counts: dict[str, int] = {}
    web_count = 0
    for c in citations:
        et = c.entry_type or "unknown"
        type_counts[et] = type_counts.get(et, 0) + 1
        if is_web_resource(c):
            web_count += 1

    # --- Local sources (academic + web) ---
    academic_matched: dict[str, Path] = {}
    academic_unmatched: list[Path] = []
    web_matched: dict[str, Path] = {}
    web_unmatched: list[Path] = []
    try:
        if tex_file:
            from sciwrite_lint.config import load_config as _load_cfg

            _cfg = _load_cfg()
        else:
            _cfg = config  # type: ignore[possibly-undefined]  # noqa: F841
        from sciwrite_lint.local_sources import (
            ACADEMIC_SUFFIXES,
            WEB_SUFFIXES,
            match_local_sources,
        )

        titles = {c.key: c.title for c in citations if c.title}
        academic_matched, academic_unmatched = match_local_sources(
            _cfg.local_pdfs_dir, titles, ACADEMIC_SUFFIXES
        )
        web_matched, web_unmatched = match_local_sources(
            _cfg.local_web_dir, titles, WEB_SUFFIXES
        )
    except Exception as e:
        logger.debug(
            f"ref-health: local source matching skipped ({type(e).__name__}: {e})"
        )

    # --- Output ---
    total = len(citations)
    print(f"\n{'=' * 60}")
    print(f"  {name} — reference health ({total} references)")
    print(f"{'=' * 60}")

    # Cite/bib mismatches
    if cited_no_bib or bib_no_cite:
        print("\n  MISMATCHES")
        if cited_no_bib:
            print(f"    Cited but missing from bib ({len(cited_no_bib)}):")
            for k in sorted(cited_no_bib):
                print(f"      - {k}")
        if bib_no_cite:
            print(f"    In bib but never cited ({len(bib_no_cite)}):")
            for k in sorted(bib_no_cite):
                print(f"      - {k}")
    else:
        print("\n  Cite/bib: all keys match")

    # ID coverage
    print("\n  ID COVERAGE")
    print(f"    With any ID:  {len(has_any_id)}/{total}")
    for f in id_fields:
        if id_counts[f]:
            print(f"      {f:<10} {id_counts[f]}")
    print(f"    No ID:        {len(no_id)}/{total}")
    if no_id:
        for k in sorted(no_id):
            print(f"      - {k}")

    # Entry types
    print("\n  ENTRY TYPES")
    for et, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {et:<15} {count}")
    if web_count:
        print(f"    ({web_count} web resources)")

    # Local sources — academic (PDFs + MD summaries) + web (MD + MHTML)
    if academic_matched or academic_unmatched:
        print("\n  LOCAL ACADEMIC SOURCES")
        if academic_matched:
            print(f"    Matched: {len(academic_matched)}")
            for key, src_path in sorted(academic_matched.items()):
                print(f"      {src_path.name} → {key}")
        if academic_unmatched:
            print(f"    Unmatched ({len(academic_unmatched)}):")
            for src_path in sorted(academic_unmatched):
                print(f"      - {src_path.name}  (no matching reference title)")
    if web_matched or web_unmatched:
        print("\n  LOCAL WEB CAPTURES")
        if web_matched:
            print(f"    Matched: {len(web_matched)}")
            for key, src_path in sorted(web_matched.items()):
                print(f"      {src_path.name} → {key}")
        if web_unmatched:
            print(f"    Unmatched ({len(web_unmatched)}):")
            for src_path in sorted(web_unmatched):
                print(f"      - {src_path.name}  (no matching reference title)")

    print()
    return 0


def run_verify_claims(args: argparse.Namespace) -> int:
    """Verify claims against cited sources using local LLM or Claude Opus."""
    from sciwrite_lint.cli._common import _load_config, _resolve_paper

    config = _load_config(args)
    pc = _resolve_paper(config, args.paper)
    if not pc:
        return 2

    ws = config.paper_workspace(pc.name)
    ws.ensure_dirs()
    refs_dir = ws.root
    config.current_paper = pc.name
    from sciwrite_lint.eval_claims import run_claim_verification

    results = asyncio.run(
        run_claim_verification(
            pc.name,
            pc.file_path,
            refs_dir,
            config=config,
            bib_path=pc.bib,
            backend=args.backend,
            model=args.model,
            key_filter=args.key,
            limit=args.limit,
            rerun=args.fresh,
        )
    )

    if results is None:
        return 1

    # Convert depth-1 results to findings
    from sciwrite_lint.pipeline import _claims_to_findings
    from sciwrite_lint.report import format_findings

    findings = _claims_to_findings(results, pc.file_path)

    # Reference-unreliable: deep path (claims + metadata)
    from sciwrite_lint.checks.reference_unreliable import (
        claims_to_unreliable_findings,
    )
    from sciwrite_lint.references.metadata import load_all_metadata

    all_meta = load_all_metadata(refs_dir)
    findings.extend(
        claims_to_unreliable_findings(
            results,
            pc.file_path,
            metadata_map=all_meta or None,
        )
    )

    # Print all findings in one pass
    label = f"{pc.name} ({pc.file_path.name})"
    format_findings(findings, label, fmt=args.format, color=config.color)

    return 1 if any(f.level == "error" for f in findings) else 0
