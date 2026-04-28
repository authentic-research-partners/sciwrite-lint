"""Workspace-free title-search across OA sources.

Public API. Queries NBER, RePEc/IDEAS, HAL, ERIC, NASA ADS, and OSF
Preprints in parallel and returns normalised :class:`SearchHit` candidates
without downloading any content. Callers can inspect candidates and feed
the selected ``pdf_url`` into :func:`download_pdf`, or simply use the
first hit.

NASA ADS is skipped when ``config.nasa_ads_key`` is not provided.

Unpaywall and CORE are identifier-based (DOI), not title-based, and are
therefore not queried here. Use :func:`download_pdf` with a ``doi`` to
reach those sources.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

from sciwrite_lint.fulltext import (
    SearchCandidate,
    lookup_eric_by_title,
    lookup_hal_by_title,
    lookup_ideas_by_title,
    lookup_nasa_ads_by_title,
    lookup_nber_by_title,
    lookup_osf_by_title,
)
from sciwrite_lint.oa._models import FetchConfig, SearchHit


async def search_by_title(
    title: str,
    authors: list[str] | None = None,
    *,
    email: str,
    config: FetchConfig | None = None,
) -> list[SearchHit]:
    """Search title-based OA sources in parallel and return candidate hits.

    :param title: Paper title to search for. Must be non-empty.
    :param authors: Bib author names. When provided, sources that support
        server-side author filtering (HAL, NASA ADS, ERIC) narrow their
        results on the first bib surname; sources that don't (NBER,
        IDEAS, OSF) include it in their free-text query as a ranking hint.
    :param email: Polite-contact email. Required for User-Agent headers.
    :param config: Optional :class:`FetchConfig`. ``nasa_ads_key`` enables
        NASA ADS; missing keys cause that source to be skipped silently.

    :returns: Flat list of :class:`SearchHit`, each with ``source``,
        ``pdf_url``, and whatever per-hit metadata (``title``, ``authors``,
        ``year``) the source exposed. Returned in the order sources
        respond; callers rank themselves or pick the first match. Empty
        list when no source returned a candidate.
    """
    if not title or not title.strip():
        return []

    cfg = config or FetchConfig()
    bib_authors = authors or None

    named_coros: list[tuple[str, Awaitable[list[SearchCandidate]]]] = [
        (
            "nber",
            lookup_nber_by_title(title, email, authors=bib_authors),
        ),
        (
            "ideas",
            lookup_ideas_by_title(title, email, authors=bib_authors),
        ),
        (
            "hal",
            lookup_hal_by_title(title, email, authors=bib_authors),
        ),
        (
            "eric",
            lookup_eric_by_title(title, email, authors=bib_authors),
        ),
        (
            "osf",
            lookup_osf_by_title(title, email, authors=bib_authors),
        ),
    ]
    if cfg.nasa_ads_key:
        named_coros.append(
            (
                "nasa_ads",
                lookup_nasa_ads_by_title(
                    title, email, api_key=cfg.nasa_ads_key, authors=bib_authors
                ),
            )
        )

    results = await asyncio.gather(*(c for _, c in named_coros), return_exceptions=True)

    hits: list[SearchHit] = []
    for (source_name, _), result in zip(named_coros, results, strict=True):
        if isinstance(result, BaseException):
            continue
        if not isinstance(result, list):
            continue
        for candidate in result:
            hits.append(
                SearchHit(
                    source=source_name,
                    title=candidate.title or None,
                    authors=list(candidate.authors),
                    year=candidate.year,
                    pdf_url=candidate.url,
                )
            )
    return hits
