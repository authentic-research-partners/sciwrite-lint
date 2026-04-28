"""Workspace-free OA PDF acquisition.

Public API. Tries identifier- and title-based sources in priority order and
writes the first validated PDF to the caller-provided ``out_path``. The
internal orchestrator :func:`sciwrite_lint.fulltext.acquire_fulltext` is a
thin wrapper that translates :class:`~sciwrite_lint.config.LintConfig` and a
workspace ``references_dir`` into a call here.

Priority order (matching the internal pipeline):

1. arXiv (``arxiv_id``)
2. Semantic Scholar openAccessPdf (``s2_pdf_url``)
3. OA URL from OpenAlex (``oa_url``) — smart path, handles landing pages
4. PubMed Central (``pmcid`` or ``doi``)
5. Europe PMC (``doi`` / ``pmcid``)
6. Unpaywall (``doi``) — smart path for oa landing pages
7. bioRxiv/medRxiv (``doi`` starting with ``10.1101/``)
8. NBER Working Papers (``title``)
9. RePEc / IDEAS (``title``)
10. HAL (``title``)
11. ERIC (``title``)
12. NASA ADS (``title``, key required)
13. OSF Preprints (``title``)
14. CORE (``doi``)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from loguru import logger

from sciwrite_lint._network import clean_and_validate_doi as _clean_and_validate_doi
from sciwrite_lint.fulltext import (
    AcquisitionResult,
    BibEvidence,
    SearchCandidate,
    _download_pdf,
    is_valid_arxiv_id,
    lookup_eric_by_title,
    lookup_europepmc,
    lookup_hal_by_title,
    lookup_ideas_by_title,
    lookup_nasa_ads_by_title,
    lookup_nber_by_title,
    lookup_osf_by_title,
    lookup_pmc,
    lookup_unpaywall,
    rank_candidates,
    search_core,
)
from sciwrite_lint.oa._models import DownloadResult, FetchConfig


async def download_pdf(
    out_path: Path,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    pmcid: str | None = None,
    s2_pdf_url: str | None = None,
    oa_url: str | None = None,
    title: str | None = None,
    authors: list[str] | None = None,
    year: int | None = None,
    entry_type: str = "article",
    email: str,
    config: FetchConfig | None = None,
) -> DownloadResult:
    """Acquire an OA PDF for the given identifiers and write to ``out_path``.

    At least one of ``doi``, ``arxiv_id``, ``pmcid``, ``s2_pdf_url``,
    ``oa_url``, or ``title`` must be provided for any source to be tried.

    :param out_path: Destination file path. Must end in ``.pdf`` (enforced
        so the smart download path writes to the same location).
    :param title: Reference title — drives title-search sources AND feeds
        the match-validation gate.
    :param authors: Reference authors (full names) — feeds the surname
        signal of the validation gate.
    :param year: Reference year — feeds the year signal of the validation
        gate.
    :param entry_type: Bib entry type (``article``, ``book``,
        ``techreport``, etc.) — controls the validator's body-length hard
        reject and the corporate-author carve-out for entries without
        named authors.
    :param email: Required for Unpaywall and as the ``mailto:`` hint in the
        polite User-Agent on title-search requests.
    :param config: Optional :class:`FetchConfig`. Supplies timeouts,
        user-agent, rate-limit intervals, and optional ``nasa_ads_key``.
        When ``nasa_ads_key`` is None, NASA ADS is skipped silently. Other
        API keys (S2, CORE, NCBI) are still read from the internal
        rate-limiter setup — they are not yet plumbed through
        :class:`FetchConfig`.

    :returns: :class:`DownloadResult`. ``found=True`` means the PDF bytes
        were written to ``out_path`` and passed
        :func:`sciwrite_lint.fulltext._validation.validate_pdf_match` —
        a multi-signal gate over title similarity, author surnames, DOI,
        and year, with template-pattern hard rejection for bibliographic-
        record landing pages. ``found=False`` with a populated
        ``failed_sources`` trail means every source was tried and none
        produced a validated PDF.
    :raises ValueError: If ``out_path`` does not end in ``.pdf``.
    """
    if out_path.suffix.lower() != ".pdf":
        raise ValueError(f"out_path must end in .pdf (got {out_path.name!r})")
    if not email:
        raise ValueError("email is required")

    cfg = config or FetchConfig()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    expected_title = title or ""
    evidence = BibEvidence(
        title=expected_title,
        authors=list(authors or []),
        doi=doi or "",
        year=year,
        entry_type=entry_type or "article",
    )
    failed_sources: list[str] = []

    async def _try_direct(url: str, source: str) -> AcquisitionResult:
        """Download a direct PDF URL to ``out_path`` with match validation."""
        return await _download_pdf(
            url,
            out_path,
            out_path.parent,
            source,
            timeout=cfg.timeout,
            evidence=evidence,
        )

    async def _try_smart(url: str, source: str) -> AcquisitionResult:
        """Download a URL that may be a landing page with embedded PDF."""
        from sciwrite_lint.pdf.pdf_download import download_and_validate

        dl = await download_and_validate(
            url,
            out_path.stem,
            out_path.parent,
            evidence=evidence,
            user_agent=cfg.user_agent,
        )
        if dl.get("found"):
            return AcquisitionResult(
                found=True,
                source=source,
                local_path=out_path.name,
                url=url,
            )
        return AcquisitionResult(
            found=False,
            source=source,
            url=url,
            reason=str(dl.get("reason") or "") or f"{source}: download failed",
        )

    def _record(result: AcquisitionResult, label: str) -> None:
        """Append a failure reason for a source that didn't produce a PDF."""
        if result.reason:
            failed_sources.append(result.reason)
        else:
            failed_sources.append(f"{label}: not available")

    # 1. arXiv
    if arxiv_id and is_valid_arxiv_id(arxiv_id.strip()):
        clean_id = arxiv_id.strip()
        result = await _try_direct(f"https://arxiv.org/pdf/{clean_id}.pdf", "arxiv")
        if result.found:
            return _ok(result, out_path, failed_sources)
        _record(result, f"arXiv ({clean_id})")

    # 2. Semantic Scholar openAccessPdf
    if s2_pdf_url:
        result = await _try_direct(s2_pdf_url, "semantic_scholar")
        if result.found:
            return _ok(result, out_path, failed_sources)
        _record(result, "Semantic Scholar PDF")

    # 3. OA URL from OpenAlex (may be a landing page)
    if oa_url:
        result = await _try_smart(oa_url, "openalex_oa")
        if result.found:
            return _ok(result, out_path, failed_sources)
        _record(result, "OA URL")

    # 4. PubMed Central
    pmc_id = pmcid
    if not pmc_id and doi:
        pmc_id = await lookup_pmc(doi)
    if pmc_id:
        result = await _try_direct(
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/",
            "pmc",
        )
        if result.found:
            return _ok(result, out_path, failed_sources)
        _record(result, f"PMC ({pmc_id})")

    # 5. Europe PMC
    if doi or pmc_id:
        epmc_url = await lookup_europepmc(doi or "", pmcid=pmc_id)
        if epmc_url:
            result = await _try_direct(epmc_url, "europepmc")
            if result.found:
                return _ok(result, out_path, failed_sources)
            _record(result, "Europe PMC")
        else:
            failed_sources.append("Europe PMC: no PDF URL found")

    # 6. Unpaywall
    uw_data = None
    is_oa = False
    oa_url_from_uw: str | None = None
    if doi:
        uw_data = await lookup_unpaywall(
            doi, polite_email=email, interval=cfg.unpaywall_interval
        )
        if uw_data:
            is_oa = True
            oa_url_from_uw = uw_data.get("oa_url") or uw_data.get("pdf_url")
            target = uw_data.get("pdf_url") or uw_data.get("oa_url")
            if target:
                result = await _try_smart(target, "unpaywall")
                if result.found:
                    dr = _ok(result, out_path, failed_sources)
                    dr.is_oa = True
                    dr.oa_url = oa_url_from_uw
                    return dr
                _record(result, "Unpaywall")
            else:
                failed_sources.append("Unpaywall: no PDF URL in response")
        else:
            failed_sources.append("Unpaywall: not found")

    # 7. bioRxiv / medRxiv
    if doi:
        clean_doi = _clean_and_validate_doi(doi)
        if clean_doi and clean_doi.startswith("10.1101/"):
            for host, source in (
                ("www.biorxiv.org", "biorxiv"),
                ("www.medrxiv.org", "medrxiv"),
            ):
                result = await _try_direct(
                    f"https://{host}/content/{clean_doi}v1.full.pdf", source
                )
                if result.found:
                    return _ok(result, out_path, failed_sources)
                _record(result, source)

    # 8-13. Title-based OA sources. Each returns a list of candidates with
    # whatever per-hit metadata the source exposes; a shared ranker picks
    # the best against the bib evidence, and only the winning URL hits
    # ``_try_direct``. NASA ADS is key-gated: only included when an API
    # key is configured.
    if expected_title:
        bib_authors = list(authors or [])
        title_sources: list[
            tuple[str, str, Callable[[], Awaitable[list[SearchCandidate]]]]
        ] = [
            (
                "nber",
                "NBER",
                lambda: lookup_nber_by_title(
                    expected_title,
                    polite_email=email,
                    authors=bib_authors or None,
                    year=year,
                ),
            ),
            (
                "ideas",
                "IDEAS",
                lambda: lookup_ideas_by_title(
                    expected_title,
                    polite_email=email,
                    authors=bib_authors or None,
                    year=year,
                ),
            ),
            (
                "hal",
                "HAL",
                lambda: lookup_hal_by_title(
                    expected_title,
                    polite_email=email,
                    authors=bib_authors or None,
                    year=year,
                ),
            ),
            (
                "eric",
                "ERIC",
                lambda: lookup_eric_by_title(
                    expected_title,
                    polite_email=email,
                    authors=bib_authors or None,
                    year=year,
                ),
            ),
        ]
        if cfg.nasa_ads_key:
            ads_key = cfg.nasa_ads_key
            title_sources.append(
                (
                    "nasa_ads",
                    "NASA ADS",
                    lambda: lookup_nasa_ads_by_title(
                        expected_title,
                        polite_email=email,
                        api_key=ads_key,
                        authors=bib_authors or None,
                        year=year,
                    ),
                )
            )
        title_sources.append(
            (
                "osf",
                "OSF Preprints",
                lambda: lookup_osf_by_title(
                    expected_title,
                    polite_email=email,
                    authors=bib_authors or None,
                    year=year,
                ),
            )
        )

        for source, label, resolve in title_sources:
            candidates = await resolve()
            if not candidates:
                failed_sources.append(f"{label}: no match")
                continue
            best = rank_candidates(candidates, evidence)
            if best is None:
                failed_sources.append(
                    f"{label}: {len(candidates)} candidate(s) failed pre-download ranking"
                )
                continue
            result = await _try_direct(best.url, source)
            if result.found:
                return _ok(result, out_path, failed_sources)
            _record(result, label)

    # 14. CORE
    core_abstract: str | None = None
    if doi:
        core_data = await search_core(doi, interval=cfg.core_interval)
        if core_data:
            core_abstract = core_data.get("abstract")
            download_url = core_data.get("download_url")
            if download_url:
                result = await _try_direct(download_url, "core")
                if result.found:
                    dr = _ok(result, out_path, failed_sources)
                    dr.abstract = core_abstract
                    return dr
                _record(result, "CORE")
            else:
                failed_sources.append("CORE: no download URL")
        else:
            failed_sources.append("CORE: not found")

    logger.debug(
        "download_pdf exhausted all sources for out_path={}; reasons: {}",
        out_path,
        "; ".join(failed_sources) or "no sources tried",
    )
    # Preserve the manual-download signal: a paper can be confirmed OA (via
    # Unpaywall *or* the caller-provided OpenAlex `oa_url`) even when every
    # automated download failed — publisher bot-blocks, CAPTCHA, JS-gated
    # PDFs, etc. Downstream (cli/fetch.py, pipeline.py) surfaces this as
    # "T2 open access — manual download needed" with the landing URL.
    known_oa = is_oa or bool(oa_url)
    best_oa_url = oa_url_from_uw or oa_url
    return DownloadResult(
        found=False,
        source="manual" if known_oa and best_oa_url else None,
        out_path=None,
        url_used=best_oa_url,
        failed_sources=failed_sources,
        abstract=core_abstract,
        is_oa=known_oa,
        oa_url=best_oa_url,
    )


def _ok(
    result: AcquisitionResult,
    out_path: Path,
    failed_sources: list[str],
) -> DownloadResult:
    """Build a found=True DownloadResult from a successful AcquisitionResult."""
    return DownloadResult(
        found=True,
        source=result.source or None,
        out_path=out_path,
        url_used=result.url,
        failed_sources=failed_sources,
    )
