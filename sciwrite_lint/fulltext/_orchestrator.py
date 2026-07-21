"""acquire_fulltext — workspace-aware wrapper around sciwrite_lint.oa.download_pdf."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.fulltext._common import (
    AcquisitionResult,
    _classify_acquisition_failure,
    _read_nasa_ads_key,
)


async def acquire_fulltext(
    key: str,
    references_dir: Path,
    config: LintConfig | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    oa_url: str | None = None,
    s2_pdf_url: str | None = None,
    pmcid: str | None = None,
    expected_title: str = "",
    expected_authors: list[str] | None = None,
    expected_year: int | None = None,
    expected_entry_type: str = "article",
) -> AcquisitionResult:
    """Acquire full text for a citation into ``references_dir / {key}.pdf``.

    Thin workspace-aware wrapper around :func:`sciwrite_lint.oa.download_pdf`.
    Translates :class:`LintConfig` and ``~/.sciwrite-lint/`` API keys into a
    :class:`~sciwrite_lint.oa.FetchConfig`, then adapts the
    :class:`~sciwrite_lint.oa.DownloadResult` back to the
    :class:`AcquisitionResult` shape expected by internal callers
    (``pipeline.py``, ``cli/fetch.py``).

    :param expected_title: Bib title — drives title-search adapters AND
        feeds the multi-signal match-validation gate.
    :param expected_authors: Bib authors (full names) — feeds the
        validator's surname signal.
    :param expected_year: Bib year — feeds the year signal.
    :param expected_entry_type: Bib entry type (``article``, ``book``,
        ``techreport``, ...). Controls the validator's body-length hard
        reject and the corporate-author carve-out.
    """
    from sciwrite_lint.oa import FetchConfig, download_pdf

    cfg = config or LintConfig()
    if not cfg.polite_email:
        # cli/fetch.py does not pass `config`, so a bare LintConfig() has
        # no email; fall back to the machine-wide ~/.sciwrite-lint/
        # polite_email (same dir API keys are read from), per the docstring.
        from sciwrite_lint.config import _read_user_polite_email
        cfg.polite_email = _read_user_polite_email()
    out_path = references_dir / f"{key}.pdf"

    fetch_cfg = FetchConfig(
        nasa_ads_key=_read_nasa_ads_key(),
        unpaywall_interval=cfg.unpaywall_interval,
        core_interval=cfg.core_interval,
    )

    dr = await download_pdf(
        out_path,
        doi=doi,
        arxiv_id=arxiv_id,
        pmcid=pmcid,
        s2_pdf_url=s2_pdf_url,
        oa_url=oa_url,
        title=expected_title or None,
        authors=expected_authors,
        year=expected_year,
        entry_type=expected_entry_type or "article",
        email=cfg.polite_email or "",
        config=fetch_cfg,
    )

    if dr.found and dr.out_path is not None:
        logger.info(
            "{}: downloaded via {} → {}",
            key,
            dr.source or "unknown",
            dr.out_path.name,
        )
        return AcquisitionResult(
            found=True,
            source=dr.source or "",
            local_path=str(dr.out_path.relative_to(references_dir)),
            url=dr.url_used,
            abstract=dr.abstract,
            is_oa=dr.is_oa,
            oa_url=dr.oa_url,
            status="found",
        )

    combined_reason = (
        "; ".join(dr.failed_sources) if dr.failed_sources else "no sources available"
    )
    failure_status = _classify_acquisition_failure(dr.failed_sources)
    if dr.oa_url:
        logger.info(
            "{}: manual download needed: {} "
            "(save PDF to local_pdfs/ or MHTML to local_web/)",
            key,
            dr.oa_url,
        )
        return AcquisitionResult(
            found=False,
            source="manual",
            url=dr.oa_url,
            abstract=dr.abstract,
            reason=combined_reason,
            is_oa=dr.is_oa,
            oa_url=dr.oa_url,
            status=failure_status,
        )

    return AcquisitionResult(
        found=False,
        abstract=dr.abstract,
        reason=combined_reason,
        status=failure_status,
    )
