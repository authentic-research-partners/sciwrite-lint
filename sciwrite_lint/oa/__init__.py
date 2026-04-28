"""Public OA-acquisition API.

Three async functions for open-access content acquisition, workspace-free:

- :func:`download_pdf` — try identifier/title-based sources in priority order,
  write the first validated PDF to an explicit ``out_path``.
- :func:`fetch_web` — fetch a URL and return extracted markdown in memory.
- :func:`search_by_title` — query all title-search sources, return normalised
  :class:`SearchHit` candidates without downloading anything.

All three take a single :class:`FetchConfig` for timeouts, user-agent, optional
API keys, and rate-limit intervals. They raise :class:`~sciwrite_lint.exceptions.SciWriteLintError`
subclasses only for infrastructure errors; "no source produced a PDF / hit" is
communicated via the return value (``DownloadResult.found = False`` /
``list[SearchHit] == []``), not an exception.

The internal orchestrators ``sciwrite_lint.fulltext.acquire_fulltext`` and
``sciwrite_lint.web.fetch_web_content`` are thin workspace-aware wrappers
around these functions.
"""

from __future__ import annotations

from sciwrite_lint.oa._models import (
    DownloadResult,
    FetchConfig,
    SearchHit,
    WebResult,
)
from sciwrite_lint.oa._pdf import download_pdf
from sciwrite_lint.oa._search import search_by_title
from sciwrite_lint.oa._web import fetch_web

__all__ = [
    "DownloadResult",
    "FetchConfig",
    "SearchHit",
    "WebResult",
    "download_pdf",
    "fetch_web",
    "search_by_title",
]
