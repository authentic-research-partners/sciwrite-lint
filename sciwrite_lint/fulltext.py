"""Full-text acquisition: download PDFs from open-access sources.

Downloads go to references/. Sources are tried in priority order — trusted,
validated sources first; less reliable sources last:

1. arXiv (direct PDF, most reliable)
2. Semantic Scholar openAccessPdf (URL from verify stage)
3. OA URL from OpenAlex (often arXiv or publisher open-access)
4. PubMed Central (NIH-funded, trusted; DOI→PMCID lookup if needed)
5. Europe PMC (broader than PMC, includes preprints)
6. Unpaywall (legal OA copy aggregator)
7. bioRxiv/medRxiv (preprint servers, DOI prefix 10.1101/)
8. CORE (institutional repos — least reliable, can return corrupt files)

All downloads are validated: minimum 5 KB size + PDF magic header check.
"""

from __future__ import annotations

from pydantic import BaseModel
from pathlib import Path
from collections.abc import Awaitable
from typing import Any

import httpx
from loguru import logger

from sciwrite_lint._network import (
    ResponseTooLarge,
    clean_and_validate_doi as _clean_and_validate_doi,
    is_valid_arxiv_id,
    ssrf_safe_client,
    stream_with_limit,
)
from sciwrite_lint.config import LintConfig
from sciwrite_lint.rate_limiter import (
    MonotonicRateLimiter,
    rate_limited_get,
    retry_on_transient,
)

_CORE_BASE = "https://api.core.ac.uk/v3"
_UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
_PMC_ID_CONVERTER = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
_EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

_MIN_PDF_SIZE = 5000  # bytes — reject anything smaller
_MAX_PDF_SIZE = 100 * 1024 * 1024  # 100 MB — reject downloads larger than this


def _is_valid_pdf(data: bytes) -> bool:
    """Check if bytes start with PDF magic header and meet min/max size."""
    return _MIN_PDF_SIZE <= len(data) <= _MAX_PDF_SIZE and data[:5] == b"%PDF-"


# ---------------------------------------------------------------------------
# API key helpers (read from ~/.sciwrite-lint/)
# ---------------------------------------------------------------------------


def _read_key_file(name: str) -> str | None:
    """Read an API key from ~/.sciwrite-lint/{name} if it exists."""
    import stat

    key_path = Path.home() / ".sciwrite-lint" / name
    if key_path.exists():
        # Warn if the key file is readable by group or others
        mode = key_path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            logger.warning(
                "API key file {} is readable by other users (mode {:o}). "
                "Run: chmod 600 {}",
                key_path,
                stat.S_IMODE(mode),
                key_path,
            )
        text = key_path.read_text().strip()
        return text if text else None
    return None


def _read_core_key() -> str | None:
    """Read CORE API key from ~/.sciwrite-lint/core_api_key."""
    return _read_key_file("core_api_key")


def _read_ncbi_key() -> str | None:
    """Read NCBI API key from ~/.sciwrite-lint/ncbi_api_key.

    With an NCBI API key, PMC rate limit increases from 3 to 10 req/s.
    Get one at: https://www.ncbi.nlm.nih.gov/account/settings/
    """
    return _read_key_file("ncbi_api_key")


def _read_s2_key() -> str | None:
    """Read Semantic Scholar API key from ~/.sciwrite-lint/s2_api_key.

    With an S2 API key, rate limit increases from 1 to 100 req/s.
    Request one at: https://www.semanticscholar.org/product/api#api-key
    """
    return _read_key_file("s2_api_key")


# Rate limiters: per-API, not per-download.
# PMC rate depends on whether NCBI API key is present (~/.sciwrite-lint/ncbi_api_key).
_core_limiter = MonotonicRateLimiter(1, 1.0)
_unpaywall_limiter = MonotonicRateLimiter(1, 0.1)
_pmc_limiter = MonotonicRateLimiter(
    10 if _read_ncbi_key() else 3,
    1.0,
)
_europepmc_limiter = MonotonicRateLimiter(10, 1.0)


class AcquisitionResult(BaseModel):
    """Result of a full-text acquisition attempt."""

    found: bool
    source: str = ""
    local_path: str | None = None
    url: str | None = None
    abstract: str | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Direct PDF download helper
# ---------------------------------------------------------------------------


async def _download_pdf(
    url: str,
    dest: Path,
    references_dir: Path,
    source: str,
    timeout: float = 30.0,
    expected_title: str = "",
    title_threshold: float = 0.65,
) -> AcquisitionResult:
    """Download a PDF from URL, validate, and save. Reuse cached file.

    When *expected_title* is provided the downloaded PDF's title is extracted
    via GROBID's processHeaderDocument endpoint and fuzzy-matched.  A score
    below *title_threshold* rejects the PDF so that wrong-paper downloads
    from APIs (e.g. CrossRef returning an unrelated DOI) are caught early.
    Threshold 0.65 cleanly separates correct matches (>0.85) from
    wrong-paper downloads (<0.60 in observed cases).
    """
    if dest.exists() and _is_valid_pdf(dest.read_bytes()):
        # Validate cached file against expected title
        if expected_title and not await _check_pdf_title(
            dest, expected_title, title_threshold
        ):
            logger.warning(
                "Cached PDF {} failed title check — removing",
                dest.name,
            )
            dest.unlink()
        else:
            return AcquisitionResult(
                found=True,
                source=source,
                local_path=str(dest.relative_to(references_dir)),
            )
    try:
        async with ssrf_safe_client(timeout=timeout) as client:
            resp = await retry_on_transient(
                lambda: stream_with_limit(client, url, _MAX_PDF_SIZE),
                label=f"PDF download ({source})",
            )
            if resp.status_code == 200 and _is_valid_pdf(resp.content):
                references_dir.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(resp.content)
                if expected_title and not await _check_pdf_title(
                    dest, expected_title, title_threshold
                ):
                    logger.warning(
                        "Downloaded PDF from {} failed title check — "
                        "expected '{}', rejecting",
                        source,
                        expected_title[:60],
                    )
                    dest.unlink()
                    return AcquisitionResult(
                        found=False,
                        source=source,
                        url=url,
                        reason=f"title mismatch from {source}",
                    )
                return AcquisitionResult(
                    found=True,
                    source=source,
                    local_path=str(dest.relative_to(references_dir)),
                )
    except ResponseTooLarge as e:
        logger.debug("PDF too large, skipping {}: {}", url, e)
        return AcquisitionResult(
            found=False,
            source=source,
            url=url,
            reason=f"too large ({source})",
        )
    except httpx.HTTPError as e:
        logger.debug("PDF download failed for {}: {}", url, e)
        return AcquisitionResult(
            found=False,
            source=source,
            url=url,
            reason=f"{type(e).__name__} from {source}",
        )
    return AcquisitionResult(found=False, source=source, url=url)


async def _check_pdf_title(
    pdf_path: Path,
    expected_title: str,
    threshold: float = 0.65,
) -> bool:
    """Return True if the PDF's title is close enough to *expected_title*.

    Uses GROBID's processHeaderDocument endpoint for title extraction.
    """
    if not expected_title:
        return True  # nothing to compare against

    from sciwrite_lint.pdf.grobid import extract_title_from_header
    from sciwrite_lint.pdf.pdf_download import _title_similarity

    grobid_title = await extract_title_from_header(pdf_path)
    if not grobid_title:
        # No title in header — likely non-formal document (news, guide, etc.).
        # Accept it; formal/non-formal classification happens after full parse.
        logger.debug(
            "GROBID could not extract title from {} — accepting "
            "(formal classification deferred to parse stage)",
            pdf_path.name,
        )
        return True

    score = _title_similarity(expected_title, grobid_title)
    if score >= threshold:
        return True

    logger.warning(
        "Title mismatch (score={:.2f}): expected '{}', got '{}'",
        score,
        expected_title[:60],
        grobid_title[:60],
    )
    return False


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


async def download_arxiv_pdf(
    arxiv_id: str,
    key: str,
    references_dir: Path,
    expected_title: str = "",
) -> AcquisitionResult:
    """Download PDF from arXiv."""
    if not arxiv_id:
        return AcquisitionResult(found=False)
    clean_id = arxiv_id.strip()
    if not is_valid_arxiv_id(clean_id):
        logger.debug("Invalid arXiv ID format, skipping: {}", clean_id[:40])
        return AcquisitionResult(found=False)
    url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    dest = references_dir / f"{key}_arxiv.pdf"
    return await _download_pdf(
        url, dest, references_dir, "arxiv", expected_title=expected_title
    )


# ---------------------------------------------------------------------------
# Semantic Scholar openAccessPdf
# ---------------------------------------------------------------------------


async def download_s2_pdf(
    s2_pdf_url: str,
    key: str,
    references_dir: Path,
    expected_title: str = "",
    expected_authors: list[str] | None = None,
) -> AcquisitionResult:
    """Download PDF from Semantic Scholar openAccessPdf URL."""
    if not s2_pdf_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_s2.pdf"
    return await _download_pdf(
        s2_pdf_url,
        dest,
        references_dir,
        "semantic_scholar",
        expected_title=expected_title,
    )


# ---------------------------------------------------------------------------
# PubMed Central (PMC)
# ---------------------------------------------------------------------------


async def lookup_pmc(doi: str) -> str | None:
    """Convert DOI to PMCID via NCBI ID Converter API. Returns PMCID or None."""
    if not doi:
        return None
    clean_doi = _clean_and_validate_doi(doi)
    if not clean_doi:
        return None
    params: dict[str, str] = {
        "ids": clean_doi,
        "format": "json",
        "tool": "sciwrite-lint",
    }
    ncbi_key = _read_ncbi_key()
    if ncbi_key:
        params["api_key"] = ncbi_key
    try:
        resp = await rate_limited_get(
            _pmc_limiter,
            _PMC_ID_CONVERTER,
            params=params,
            timeout=5.0,
            label="PMC ID converter",
            service="pmc",
        )
        if resp.status_code != 200:
            return None
        records = resp.json().get("records") or []
        for rec in records:
            pmcid = rec.get("pmcid")
            if pmcid:
                return pmcid
    except httpx.HTTPError as e:
        logger.debug("PMC ID converter failed for DOI {}: {}", doi, e)
    return None


async def download_pmc_pdf(
    pmcid: str,
    key: str,
    references_dir: Path,
    expected_title: str = "",
) -> AcquisitionResult:
    """Download PDF from PubMed Central."""
    if not pmcid:
        return AcquisitionResult(found=False)
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    dest = references_dir / f"{key}_pmc.pdf"
    return await _download_pdf(
        url, dest, references_dir, "pmc", expected_title=expected_title
    )


# ---------------------------------------------------------------------------
# Europe PMC
# ---------------------------------------------------------------------------


async def lookup_europepmc(
    doi: str,
    pmcid: str | None = None,
) -> str | None:
    """Look up paper on Europe PMC. Returns PDF URL or None."""
    if pmcid:
        return (
            f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        )

    if not doi:
        return None
    clean_doi = _clean_and_validate_doi(doi)
    if not clean_doi:
        return None

    try:
        resp = await rate_limited_get(
            _europepmc_limiter,
            f"{_EUROPEPMC_BASE}/search",
            params={
                "query": f"DOI:{clean_doi}",
                "format": "json",
                "resultType": "core",
            },
            label="Europe PMC",
            service="europepmc",
        )
        if resp.status_code != 200:
            return None
        results = (resp.json().get("resultList") or {}).get("result") or []
        for result in results:
            for url_info in result.get("fullTextUrlList", {}).get("fullTextUrl", []):
                if url_info.get("documentStyle") == "pdf":
                    return url_info.get("url")
            # If has PMCID, construct direct URL
            epmc_pmcid = result.get("pmcid")
            if epmc_pmcid:
                return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={epmc_pmcid}&blobtype=pdf"
    except httpx.HTTPError as e:
        logger.debug("Europe PMC lookup failed for DOI {}: {}", doi, e)
    return None


async def download_europepmc_pdf(
    pdf_url: str,
    key: str,
    references_dir: Path,
    expected_title: str = "",
) -> AcquisitionResult:
    """Download PDF from Europe PMC."""
    if not pdf_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_europepmc.pdf"
    return await _download_pdf(
        pdf_url, dest, references_dir, "europepmc", expected_title=expected_title
    )


# ---------------------------------------------------------------------------
# bioRxiv / medRxiv
# ---------------------------------------------------------------------------


async def download_biorxiv_pdf(
    doi: str,
    key: str,
    references_dir: Path,
    expected_title: str = "",
) -> AcquisitionResult:
    """Download from bioRxiv/medRxiv if DOI matches 10.1101/ prefix."""
    if not doi:
        return AcquisitionResult(found=False)
    clean_doi = _clean_and_validate_doi(doi)
    if not clean_doi or not clean_doi.startswith("10.1101/"):
        return AcquisitionResult(found=False)

    # bioRxiv and medRxiv both serve PDFs at this pattern
    url = f"https://www.biorxiv.org/content/{clean_doi}v1.full.pdf"
    dest = references_dir / f"{key}_biorxiv.pdf"
    result = await _download_pdf(
        url, dest, references_dir, "biorxiv", expected_title=expected_title
    )
    if result.found:
        return result

    # medRxiv uses the same DOI prefix but different domain
    url = f"https://www.medrxiv.org/content/{clean_doi}v1.full.pdf"
    dest = references_dir / f"{key}_medrxiv.pdf"
    return await _download_pdf(
        url, dest, references_dir, "medrxiv", expected_title=expected_title
    )


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------


async def search_core(
    doi: str,
    interval: float = 1.0,
) -> dict[str, Any] | None:
    """Look up a paper by DOI via CORE API."""
    if not doi:
        return None

    clean_doi = _clean_and_validate_doi(doi)
    if not clean_doi:
        return None
    core_key = _read_core_key()
    headers: dict[str, str] = {}
    if core_key:
        headers["Authorization"] = f"Bearer {core_key}"

    try:
        resp = await rate_limited_get(
            _core_limiter,
            f"{_CORE_BASE}/search/works/",
            params={"q": f'doi:"{clean_doi}"', "limit": "1"},
            headers=headers,
            timeout=15.0,
            label="CORE",
            service="core",
        )
        if resp.status_code in (401, 403):
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        work = results[0]
        for candidate in results:
            if candidate.get("abstract"):
                work = candidate
                break

        abstract = work.get("abstract")
        if not abstract:
            return None

        authors = [
            a.get("name", "") for a in (work.get("authors") or []) if a.get("name")
        ]

        return {
            "abstract": abstract,
            "download_url": work.get("downloadUrl"),
            "authors": authors,
            "year": work.get("yearPublished"),
            "core_id": work.get("id"),
            "source": "CORE",
        }
    except httpx.HTTPError as e:
        logger.debug("CORE API lookup failed for DOI {}: {}", doi, e)
        return None


async def download_core_pdf(
    download_url: str,
    key: str,
    references_dir: Path,
    expected_title: str = "",
) -> AcquisitionResult:
    """Download full text from CORE repository URL."""
    if not download_url:
        return AcquisitionResult(found=False)
    dest = references_dir / f"{key}_core.pdf"
    return await _download_pdf(
        download_url, dest, references_dir, "core", expected_title=expected_title
    )


# ---------------------------------------------------------------------------
# Unpaywall
# ---------------------------------------------------------------------------


async def lookup_unpaywall(
    doi: str,
    polite_email: str = "",
    interval: float = 0.1,
) -> dict[str, Any] | None:
    """Look up DOI on Unpaywall for legal OA copy."""
    if not polite_email or not doi:
        return None

    clean_doi = _clean_and_validate_doi(doi)
    if not clean_doi:
        return None

    try:
        resp = await rate_limited_get(
            _unpaywall_limiter,
            f"{_UNPAYWALL_BASE}/{clean_doi}",
            params={"email": polite_email},
            label="Unpaywall",
            service="unpaywall",
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()

        if not data.get("is_oa"):
            return None

        best_oa = data.get("best_oa_location") or {}
        oa_url = best_oa.get("url_for_landing_page") or best_oa.get("url")
        pdf_url = best_oa.get("url_for_pdf")

        if not oa_url and not pdf_url:
            return None

        return {
            "oa_url": oa_url,
            "pdf_url": pdf_url,
            "is_oa": True,
            "source": "Unpaywall",
        }
    except httpx.HTTPError as e:
        logger.debug("Unpaywall lookup failed for DOI {}: {}", doi, e)
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


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
    progress: bool = True,
) -> AcquisitionResult:
    """Try to acquire full text for a citation.

    Sources are tried in priority order: arXiv → S2 PDF → OA URL → PMC →
    Europe PMC → Unpaywall → bioRxiv/medRxiv → CORE.
    Trusted sources first; CORE is last (least reliable).
    """
    config = config or LintConfig()
    failed_sources: list[str] = []

    async def _try_source(
        label: str,
        coro: Awaitable[AcquisitionResult],
    ) -> AcquisitionResult | None:
        """Try a download source, print progress, return result if found."""
        if progress:
            print(f"    Trying {label}...", end="", flush=True)
        result = await coro
        if result.found:
            if progress:
                print(" downloaded")
            return result
        if progress:
            print(" not available")
        if result.reason:
            failed_sources.append(result.reason)
        else:
            failed_sources.append(f"{label}: not available")
        return None

    # 1. arXiv (most reliable)
    if arxiv_id:
        result = await _try_source(
            f"arXiv ({arxiv_id})",
            download_arxiv_pdf(
                arxiv_id, key, references_dir, expected_title=expected_title
            ),
        )
        if result:
            return result

    # 2. Semantic Scholar openAccessPdf
    if s2_pdf_url:
        result = await _try_source(
            "Semantic Scholar PDF",
            download_s2_pdf(
                s2_pdf_url, key, references_dir, expected_title, expected_authors
            ),
        )
        if result:
            return result

    # 3. OA URL from OpenAlex (often arXiv or publisher)
    if oa_url:
        if progress:
            print("    Trying OA URL...", end="", flush=True)
        dl = await _try_smart_download(
            oa_url, key, references_dir, expected_title, expected_authors, progress
        )
        if dl.found:
            return dl
        if dl.reason:
            failed_sources.append(dl.reason)
        else:
            failed_sources.append("OA URL: not available")

    # 4. PubMed Central
    pmc_id = pmcid
    if not pmc_id and doi:
        if progress:
            print("    Looking up PMC...", end="", flush=True)
        pmc_id = await lookup_pmc(doi)
        if progress:
            print(f" {pmc_id}" if pmc_id else " not found")
    if pmc_id:
        result = await _try_source(
            f"PMC ({pmc_id})",
            download_pmc_pdf(
                pmc_id, key, references_dir, expected_title=expected_title
            ),
        )
        if result:
            return result

    # 5. Europe PMC
    if doi or pmc_id:
        if progress:
            print("    Trying Europe PMC...", end="", flush=True)
        epmc_url = await lookup_europepmc(doi or "", pmcid=pmc_id)
        if epmc_url:
            result = await download_europepmc_pdf(
                epmc_url, key, references_dir, expected_title=expected_title
            )
            if result.found:
                if progress:
                    print(" downloaded")
                return result
            if result.reason:
                failed_sources.append(result.reason)
            else:
                failed_sources.append("Europe PMC: download failed")
        else:
            failed_sources.append("Europe PMC: no PDF URL found")
        if progress:
            print(" not found")

    # 6. Unpaywall
    uw_data = None
    if doi:
        if progress:
            print("    Trying Unpaywall...", end="", flush=True)
        uw_data = await lookup_unpaywall(
            doi,
            polite_email=config.polite_email,
            interval=config.unpaywall_interval,
        )
        if uw_data:
            url = uw_data.get("pdf_url") or uw_data.get("oa_url")
            if url:
                dl = await _try_smart_download(
                    url, key, references_dir, expected_title, expected_authors, progress
                )
                if dl.found:
                    return dl
                if dl.reason:
                    failed_sources.append(dl.reason)
                else:
                    failed_sources.append("Unpaywall: download failed")
            else:
                failed_sources.append("Unpaywall: no PDF URL in response")
        else:
            failed_sources.append("Unpaywall: not found")
        if progress:
            print(" not found")

    # 7. bioRxiv / medRxiv (only for 10.1101/ DOIs)
    if doi:
        clean_doi = _clean_and_validate_doi(doi)
        if clean_doi and clean_doi.startswith("10.1101/"):
            result = await _try_source(
                "bioRxiv/medRxiv",
                download_biorxiv_pdf(
                    doi, key, references_dir, expected_title=expected_title
                ),
            )
            if result:
                return result

    # 8. CORE (least reliable — can return corrupt files)
    core_abstract = None
    if doi:
        if progress:
            print("    Trying CORE...", end="", flush=True)
        core_data = await search_core(doi, interval=config.core_interval)
        if core_data:
            download_url = core_data.get("download_url")
            if download_url:
                result = await download_core_pdf(
                    download_url, key, references_dir, expected_title=expected_title
                )
                if result.found:
                    if progress:
                        print(" downloaded")
                    result.abstract = core_data.get("abstract")
                    return result
                if result.reason:
                    failed_sources.append(result.reason)
                else:
                    failed_sources.append("CORE: download failed")
            else:
                failed_sources.append("CORE: no download URL")
            core_abstract = core_data.get("abstract")
            if progress:
                print(" abstract only")
        else:
            failed_sources.append("CORE: not found")
            if progress:
                print(" not found")

    # 9. Suggest manual download
    combined_reason = (
        "; ".join(failed_sources) if failed_sources else "no sources available"
    )
    best_url = None
    if uw_data:
        best_url = uw_data.get("pdf_url") or uw_data.get("oa_url")
    if not best_url and oa_url:
        best_url = oa_url
    if best_url:
        if progress:
            print(f"    Manual download needed: {best_url}")
        return AcquisitionResult(
            found=False,
            source="manual",
            url=best_url,
            abstract=core_abstract,
            reason=combined_reason,
        )

    return AcquisitionResult(
        found=False, abstract=core_abstract, reason=combined_reason
    )


async def _try_smart_download(
    url: str,
    key: str,
    references_dir: Path,
    expected_title: str,
    expected_authors: list[str] | None,
    progress: bool,
) -> AcquisitionResult:
    """Download PDF from URL, validate title, save if match."""
    from sciwrite_lint.pdf.pdf_download import download_and_validate

    if progress:
        print(" downloading...", end="", flush=True)

    dl = await download_and_validate(
        url,
        key,
        references_dir,
        expected_title=expected_title,
        expected_authors=expected_authors,
    )

    if dl["found"]:
        if progress:
            score = dl.get("match_score", 0)
            print(f" validated (title match: {score:.0%})")
        return AcquisitionResult(
            found=True,
            source="download",
            local_path=dl["local_path"],
        )
    else:
        reason = dl.get("reason", "failed")
        if progress:
            print(f" {reason}")
        return AcquisitionResult(found=False, source="download", url=url, reason=reason)
