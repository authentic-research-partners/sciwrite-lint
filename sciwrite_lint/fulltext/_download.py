"""Direct PDF download + multi-signal match validation."""

from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger

from sciwrite_lint._network import (
    ResponseTooLarge,
    ssrf_safe_client,
    stream_with_limit,
)
from sciwrite_lint.fulltext._common import (
    AcquisitionResult,
    _MAX_PDF_SIZE,
    _is_valid_pdf,
)
from sciwrite_lint.fulltext._validation import (
    BibEvidence,
    ValidationResult,
    validate_pdf_match,
)
from sciwrite_lint.rate_limiter import retry_on_transient


async def _download_pdf(
    url: str,
    dest: Path,
    references_dir: Path,
    source: str,
    timeout: float = 30.0,
    evidence: BibEvidence | None = None,
) -> AcquisitionResult:
    """Download a PDF from URL, validate match, and save. Reuse cached file.

    When *evidence* is provided the downloaded PDF is run through
    :func:`validate_pdf_match`, which combines GROBID title extraction,
    pypdf first-page text (for surname / DOI / year signals), and
    template-pattern rejection. A failed validation removes the file and
    reports the failure reason so the caller can move to the next OA
    source in the waterfall.
    """
    if dest.exists() and _is_valid_pdf(dest.read_bytes()):
        cached_result = await _validate(dest, evidence)
        if cached_result.accepted:
            return AcquisitionResult(
                found=True,
                source=source,
                local_path=str(dest.relative_to(references_dir)),
            )
        logger.warning(
            "Cached PDF {} failed validation — removing ({})",
            dest.name,
            cached_result.reason,
        )
        dest.unlink(missing_ok=True)
        if cached_result.signals.get("body_chars") == 0:
            return AcquisitionResult(
                found=False,
                source=source,
                url=url,
                reason=f"unparseable PDF (from {source})",
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
                fresh_result = await _validate(dest, evidence)
                if not fresh_result.accepted:
                    logger.warning(
                        "Downloaded PDF from {} rejected — {}",
                        source,
                        fresh_result.reason,
                    )
                    dest.unlink(missing_ok=True)
                    return AcquisitionResult(
                        found=False,
                        source=source,
                        url=url,
                        reason=f"{fresh_result.reason} (from {source})",
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


async def _validate(pdf_path: Path, evidence: BibEvidence | None) -> ValidationResult:
    """Run :func:`validate_pdf_match` or return an auto-accept when no
    evidence is supplied.

    Adapters used in contexts where bib metadata is genuinely unknown
    (for example the public API being called with only a URL) pass
    ``evidence=None``; in that case we preserve bytes-level PDF validity
    without any content-match check. Internal callers in the pipeline
    always pass evidence built from the bib entry.
    """
    if evidence is None:
        return ValidationResult(accepted=True, confidence=0.0, reason="no evidence")
    return await validate_pdf_match(pdf_path, evidence)
