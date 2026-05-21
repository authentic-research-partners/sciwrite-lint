"""Stage 2: reference verification against external APIs.

Runs the five-phase verify pipeline (OpenAlex → S2 → CrossRef → OL/LoC →
URL), plus post-verify cross-validation, vLLM-confirmed venue matching,
Retraction Watch enrichment, and a standalone ``reference-accuracy``
check that re-runs on stored metadata (no API calls).
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Literal

import httpx
from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Citation, Finding


def _classify_verify_issue(
    issue: str,
) -> tuple[Literal["error", "warning", "info"], str]:
    """Classify a verify issue string into (level, check_id).

    Accuracy-related mismatches (title, author, year, venue) route to
    reference-accuracy. Existence issues (not found, dead URL, retracted)
    stay under reference-exists.
    """
    issue_lower = issue.lower()
    # Existence issues → reference-exists
    if "dead url" in issue_lower:
        return "error", "reference-exists"
    if "blocked by " in issue_lower:
        # Unverifiable URL: server refusal (4xx), server error (5xx), TLS
        # failure, timeout, connection error, decoding error, or oversized
        # response. URL may still be valid — we could not confirm. WARN
        # so the user checks manually, not ERROR.
        return "warning", "reference-exists"
    if "web content saved" in issue_lower:
        return "info", "reference-exists"
    if "content extraction failed" in issue_lower:
        return "warning", "reference-exists"
    if "not found" in issue_lower:
        return "error", "reference-exists"
    if "retracted" in issue_lower:
        return "error", "retracted-cite"
    # Accuracy issues → reference-accuracy
    if "mismatch" in issue_lower:
        return "warning", "reference-accuracy"
    # Unverified URL identifier — warning but no mismatch penalty
    if "unverified url identifier" in issue_lower:
        return "info", "reference-accuracy"
    # Informational (not problems)
    if "open access pdf available" in issue_lower:
        return "info", "reference-exists"
    if "doi available but missing" in issue_lower:
        return "info", "reference-exists"
    return "warning", "reference-exists"


def _register_ref_in_workspace(
    meta: Any,
    references_dir: Path,
) -> None:
    """Register a verified reference in the workspace DB for cross-depth dedup."""
    from sciwrite_lint.references.workspace_db import get_db, register_reference

    canonical = meta.canonical or {}
    bibitem = meta.bibitem or {}
    authors = canonical.get("authors") or bibitem.get("authors") or None

    try:
        with get_db(references_dir) as conn:
            register_reference(
                conn,
                ref_key=meta.key,
                workspace_path=".",
                depth=0,
                parent_key="",
                doi=canonical.get("doi") or bibitem.get("doi"),
                arxiv_id=canonical.get("arxiv_id") or bibitem.get("arxiv_id"),
                pmid=canonical.get("pmid") or bibitem.get("pmid"),
                pmcid=canonical.get("pmcid") or bibitem.get("pmc_id"),
                isbn=canonical.get("isbn") or bibitem.get("isbn"),
                lccn=canonical.get("lccn") or bibitem.get("lccn"),
                title=canonical.get("title") or bibitem.get("title"),
                authors=authors if isinstance(authors, list) else None,
            )
    except sqlite3.Error as e:
        logger.debug(
            "Failed to register {} in workspace DB ({}: {})",
            meta.key,
            type(e).__name__,
            e,
        )


async def _stage_verify(
    citations: list[Citation],
    config: LintConfig,
    references_dir: Path,
    *,
    fresh: bool = False,
) -> list[Finding]:
    """Verify citations against external sources. Returns ref-* findings.

    Verification proceeds through five phases, each narrowing the set of
    unresolved references before the next phase runs:

      Phase A  — OpenAlex batch      (single request, resolves DOI/arXiv/PMID)
      Phase B  — Semantic Scholar     (batch, resolves DOI/arXiv/PMID/PMC)
      Phase C  — CrossRef parallel    (per-citation, resolves DOI + title search)
      Phase C2 — Open Library + LoC   (per-citation, resolves ISBN → OL, LCCN → LoC;
                                       OL and LoC run in parallel per citation)
      Phase D  — URL verification     (per-citation, HEAD/GET check for refs with URLs;
                                       last resort before marking not_found)

    Web resources (@misc with URL, no DOI) bypass all phases and go directly
    to URL verification (concurrent, semaphore-limited).

    Each phase only processes citations still unresolved after prior phases.
    Results are validated by ``_id_result_matches`` (composite title/author/year
    score ≥ 0.40) to reject API results that don't match the bib entry.
    """
    from sciwrite_lint.api import (
        _id_result_matches,
        batch_openalex,
        batch_s2,
        cross_validate_ids,
        parallel_crossref,
    )
    from sciwrite_lint.references.citations import is_web_resource
    from sciwrite_lint.references.matching import compare_citation_detailed
    from sciwrite_lint.references.metadata import (
        build_metadata_from_citation,
        merge_source_paper,
        save_metadata,
    )

    findings: list[Finding] = []

    # Partition: already verified vs needs verification
    # Single DB query to get all already-verified metadata
    unverified: list[Citation] = []
    web_citations: list[Citation] = []
    if not fresh:
        from sciwrite_lint.references.workspace_db import (
            get_db,
            query_verified_metadata,
        )

        with get_db(references_dir) as conn:
            cached_meta = query_verified_metadata(conn)
    else:
        cached_meta = {}

    for c in citations:
        if not fresh:
            existing = cached_meta.get(c.key)
            if existing:
                # Already verified — collect findings from stored issues
                for issue in existing.issues:
                    level, rule_id = _classify_verify_issue(issue)
                    findings.append(
                        Finding(
                            level=level,
                            rule_id=rule_id,
                            message=f"{c.key}: {issue}",
                            file="",
                            context=f"Source: {existing.api_source}"
                            if existing.api_source
                            else "",
                        )
                    )
                c.api_match = existing.api_match
                c.tier = existing.access.get("tier", "")
                continue
        if is_web_resource(c):
            web_citations.append(c)
        else:
            unverified.append(c)

    if not unverified and not web_citations:
        return findings

    total = len(unverified) + len(web_citations)
    logger.info(
        f"Verifying {total} citations ({len(unverified)} academic, {len(web_citations)} web)"
    )

    # ------------------------------------------------------------------
    # Phases A–C: Batch academic API lookups (DOI, arXiv, PMID, title)
    # ------------------------------------------------------------------
    resolved: dict[str, dict[str, Any]] = {}

    if unverified:
        # Phase A: OpenAlex batch — single request, covers DOI + arXiv DOI
        oa_results = await batch_openalex(unverified, config)
        resolved.update(oa_results)
        oa_found = len(oa_results)

        # Phase B: Semantic Scholar batch — DOI, arXiv ID, PMID, PMC
        still_need = [c for c in unverified if c.key not in resolved]
        if still_need:
            s2_results = await batch_s2(still_need, config)
            resolved.update(s2_results)

        # Phase C: CrossRef parallel — DOI + title/author search
        cr_need = [c for c in unverified if c.key not in resolved]
        if cr_need:
            cr_results = await parallel_crossref(cr_need, config)
            for key, result in cr_results.items():
                if key not in resolved:
                    resolved[key] = result

        # ------------------------------------------------------------------
        # Phase C2: Open Library (ISBN) + Library of Congress (LCCN)
        #
        # Books and government reports often have ISBN/LCCN but no DOI, so
        # academic APIs miss them. Per-citation lookups run concurrently;
        # OL and LoC run in parallel per citation when both IDs are present.
        # ------------------------------------------------------------------
        from sciwrite_lint._network import is_valid_isbn, is_valid_lccn

        ol_need = [
            c
            for c in unverified
            if c.key not in resolved
            and (
                (c.isbn and is_valid_isbn(c.isbn)) or (c.lccn and is_valid_lccn(c.lccn))
            )
        ]
        if ol_need:
            from sciwrite_lint.api import CitationAPI

            async with CitationAPI(config=config) as cat_api:
                ol_sem = asyncio.Semaphore(5)

                async def _try_ol_loc(c: Citation) -> None:
                    async with ol_sem:
                        has_isbn = c.isbn and is_valid_isbn(c.isbn)
                        has_lccn = c.lccn and is_valid_lccn(c.lccn)
                        tasks: list[asyncio.Task] = []
                        if has_isbn:
                            tasks.append(
                                asyncio.create_task(cat_api._openlibrary_lookup(c))
                            )
                        if has_lccn:
                            tasks.append(asyncio.create_task(cat_api._loc_lookup(c)))
                        results = await asyncio.gather(*tasks)
                        for result in results:
                            if result and not result.get("error"):
                                if _id_result_matches(c, result):
                                    resolved[c.key] = result
                                    return

                await asyncio.gather(*[_try_ol_loc(c) for c in ol_need])
                ol_found = sum(1 for c in ol_need if c.key in resolved)

            if ol_found:
                logger.info(f"Open Library / LoC: {ol_found} found")

        logger.info(
            f"API results: {oa_found} OpenAlex, "
            f"{len(resolved) - oa_found} S2/CrossRef/OL/LoC, "
            f"{len(unverified) - len(resolved)} not found"
        )

    # ------------------------------------------------------------------
    # Phase D: URL verification (last resort)
    #
    # References not found in any academic/book API but carrying a URL
    # (techreports, journalism, books hosted online, conference pages).
    # Confirms the URL is alive (→ web_verified) or dead (→ web_dead)
    # before falling through to not_found.
    # ------------------------------------------------------------------
    url_need = [c for c in unverified if c.key not in resolved and c.url]
    if url_need:
        from sciwrite_lint.api import _verify_web_resource

        url_sem = asyncio.Semaphore(5)

        async def _try_url(c: Citation) -> None:
            async with url_sem:
                # Each task gets its own httpx client — sharing a client
                # across concurrent streaming requests causes gzip
                # decompression errors (corrupt shared decoder state).
                async with httpx.AsyncClient(
                    timeout=15.0, follow_redirects=True
                ) as client:
                    await _verify_web_resource(c, client, references_dir)

        await asyncio.gather(*[_try_url(c) for c in url_need])

        url_resolved_keys = {
            c.key
            for c in url_need
            if c.api_match in ("web_verified", "web_dead", "web_blocked")
        }
        logger.info(
            f"URL verification: {len(url_resolved_keys)}/{len(url_need)} resolved"
        )

    # Apply results and run metadata comparison
    for c in unverified:
        result = resolved.get(c.key)  # type: ignore[assignment]
        if result and not result.get("error"):
            c.api_data = result
            c.api_source = result.get("source", "")
            c.issues.extend(compare_citation_detailed(c, result))
        elif c.api_match not in ("web_verified", "web_dead", "web_blocked"):
            # Not found in any API and no URL (or URL not yet checked)
            c.api_match = "not_found"
            c.issues.append(
                "Not found in CrossRef, OpenAlex, Semantic Scholar, Open Library, or Library of Congress"
            )

    # ------------------------------------------------------------------
    # Phase E: Cross-validate identifiers
    #
    # For verified citations with multiple IDs, look up each ID
    # independently via OpenAlex and verify they all resolve to the
    # same paper. Catches LLM-mixed bib entries where DOI → paper A
    # but arXiv ID → paper B.
    # ------------------------------------------------------------------
    cross_need = [c for c in unverified if c.api_data and not c.api_data.get("error")]
    if cross_need:
        cross_sem = asyncio.Semaphore(5)

        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True
        ) as cross_client:

            async def _cross_validate(c: Citation) -> None:
                async with cross_sem:
                    cross_issues = await cross_validate_ids(
                        c, c.api_data, config, client=cross_client
                    )
                    c.issues.extend(cross_issues)

            await asyncio.gather(*[_cross_validate(c) for c in cross_need])

    # Set api_match based on all issues (including cross-validation).
    # Only for citations resolved via academic APIs — don't overwrite
    # web_verified / web_dead / web_blocked status set by URL verification.
    for c in unverified:
        result = resolved.get(c.key)  # type: ignore[assignment]
        if result and not result.get("error"):
            c.api_match = (
                "mismatch"
                if any("mismatch" in i.lower() for i in c.issues)
                else "verified"
            )

    # Persist and generate findings (single DB connection for all writes)
    from sciwrite_lint.references.workspace_db import (
        get_db,
        load_citation_metadata,
        save_citation_metadata,
    )

    with get_db(references_dir) as conn:
        for c in unverified:
            result = resolved.get(c.key)  # type: ignore[assignment]

            existing = load_citation_metadata(conn, c.key)
            meta = build_metadata_from_citation(
                c, result, references_dir=references_dir
            )
            if existing:
                merge_source_paper(meta, c.source_paper)
                for sp in existing.bibitem.get("source_papers", []):
                    merge_source_paper(meta, sp)
                if existing.manual_override:
                    meta.manual_override = existing.manual_override
            save_citation_metadata(conn, meta)
            _register_ref_in_workspace(meta, references_dir)
            c.tier = meta.access.get("tier", "")

            for issue in c.issues:
                level, rule_id = _classify_verify_issue(issue)
                findings.append(
                    Finding(
                        level=level,
                        rule_id=rule_id,
                        message=f"{c.key}: {issue}",
                        file="",
                        context=f"Source: {c.api_source}" if c.api_source else "",
                    )
                )

    # Web resources (concurrent — independent HTTP checks)
    if web_citations:
        from sciwrite_lint.api import _verify_web_resource

        sem = asyncio.Semaphore(5)

        async def _verify_one_web(c: Citation) -> None:
            async with sem:
                async with httpx.AsyncClient(
                    timeout=15.0, follow_redirects=True
                ) as client:
                    result = await _verify_web_resource(c, client, references_dir)
            meta = build_metadata_from_citation(
                c, result, references_dir=references_dir
            )
            save_metadata(meta, references_dir)
            c.tier = meta.access.get("tier", "")

        await asyncio.gather(*[_verify_one_web(c) for c in web_citations])

        for c in web_citations:
            for issue in c.issues:
                level, rule_id = _classify_verify_issue(issue)
                findings.append(
                    Finding(
                        level=level,
                        rule_id=rule_id,
                        message=f"{c.key}: {issue}",
                        file="",
                        context=f"Source: {c.api_source}" if c.api_source else "",
                    )
                )

    # Confirm venue mismatches via vLLM (suppresses false positives like
    # "NeurIPS" vs "Advances in Neural Information Processing Systems")
    findings = await _confirm_venue_findings(findings, config)

    # Retraction Watch enrichment: check all metadata against RW database
    from sciwrite_lint.references.retraction_watch import ensure_rw_database

    rw_db = await ensure_rw_database(config)
    if rw_db:
        from sciwrite_lint.references.metadata import enrich_retraction_status
        from sciwrite_lint.references.workspace_db import (
            get_db,
            load_all_citation_metadata,
            save_citation_metadata,
        )

        with get_db(references_dir) as conn:
            all_meta = load_all_citation_metadata(conn)
            for _key, meta in all_meta.items():
                if enrich_retraction_status(meta, rw_db):
                    save_citation_metadata(conn, meta)

    return findings


async def _confirm_venue_findings(
    findings: list[Finding],
    config: LintConfig,
) -> list[Finding]:
    """Filter venue mismatch findings through vLLM confirmation.

    If vLLM is available and says the venues match, the finding is dropped.
    If vLLM is unavailable, all findings pass through unchanged.
    """
    from sciwrite_lint.references.matching import venue_match_llm

    venue_findings = []
    other_findings = []
    for f in findings:
        if "Venue mismatch" in f.message:
            venue_findings.append(f)
        else:
            other_findings.append(f)

    if not venue_findings:
        return findings

    # Extract venue pairs from finding messages
    import re

    sem = asyncio.Semaphore(50)

    async def _confirm_one(f: Finding) -> Finding | None:
        m = re.search(r"tex='([^']*)', (?:canonical|API)='([^']*)'", f.message)
        if not m:
            return f  # can't parse, keep
        async with sem:
            try:
                same = await venue_match_llm(m.group(1), m.group(2), config=config)
            except (httpx.HTTPError, OSError, asyncio.TimeoutError) as e:
                # vLLM unreachable / timeout: keep the venue finding rather
                # than drop it silently. WARN so the operator sees that LLM
                # confirmation didn't run, not DEBUG which is usually muted.
                logger.warning(
                    "Venue match LLM unavailable ({}: {}); keeping finding",
                    type(e).__name__,
                    e,
                )
                return f
        if same is True:
            return None  # vLLM confirmed same venue — suppress
        return f

    results = await asyncio.gather(*[_confirm_one(f) for f in venue_findings])
    confirmed = [r for r in results if r is not None]

    suppressed = len(venue_findings) - len(confirmed)
    if suppressed:
        logger.info(f"Venue: {suppressed} false positive(s) suppressed by vLLM")

    return other_findings + confirmed


def _stage_reference_accuracy(
    config: LintConfig,
    references_dir: Path,
) -> list[Finding]:
    """Run reference-accuracy check on stored metadata. No API calls."""
    from sciwrite_lint.checks.reference_accuracy import (
        check_reference_accuracy_from_metadata,
    )
    from sciwrite_lint.references.metadata import load_all_metadata

    if not config.is_check_enabled("reference-accuracy"):
        return []
    if not references_dir.exists():
        return []
    all_metadata = load_all_metadata(references_dir)
    if not all_metadata:
        return []

    findings = check_reference_accuracy_from_metadata(all_metadata)

    # Apply severity override
    override = config.effective_severity("reference-accuracy", "warning")
    for f in findings:
        if f.level == "warning" and override != "warning":
            f.level = override  # type: ignore[assignment]

    if findings:
        logger.info(f"Reference accuracy: {len(findings)} issues")
    return findings
