"""Unpaywall DOI lookup for legal OA copies."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from sciwrite_lint._network import clean_and_validate_doi as _clean_and_validate_doi
from sciwrite_lint.fulltext._common import UNPAYWALL_BASE, _unpaywall_limiter
from sciwrite_lint.rate_limiter import rate_limited_get


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
            f"{UNPAYWALL_BASE}/{clean_doi}",
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
