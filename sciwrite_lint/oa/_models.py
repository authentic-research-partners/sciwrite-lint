"""Public-API models for ``sciwrite_lint.oa``.

These models are part of the public API contract. Changes here require a
SemVer bump.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)


class FetchConfig(BaseModel):
    """Tunables shared by :func:`download_pdf`, :func:`fetch_web`, :func:`search_by_title`.

    All fields are optional; defaults match the internal pipeline. Callers
    supply API keys explicitly — the public API never reads from
    ``~/.sciwrite-lint/``.

    Only keys that are actually honoured per-call appear here. S2 / CORE /
    NCBI rate-limit tokens are read from file by the internal rate-limiter
    setup; they are not yet plumbed through the public API, so exposing
    them here would be misleading.
    """

    timeout: float = 30.0
    user_agent: str = _DEFAULT_USER_AGENT

    nasa_ads_key: str | None = None

    unpaywall_interval: float = 0.1
    core_interval: float = 1.0


class DownloadResult(BaseModel):
    """Outcome of a :func:`download_pdf` call."""

    found: bool
    source: str | None = None
    out_path: Path | None = None
    url_used: str | None = None
    title_check_score: float | None = None
    failed_sources: list[str] = Field(default_factory=list)
    abstract: str | None = None
    is_oa: bool = False
    oa_url: str | None = None


class WebResult(BaseModel):
    """Outcome of a :func:`fetch_web` call.

    The extracted content is returned as a string in :attr:`markdown`; callers
    write it to disk if they want persistence.

    Three-valued outcome for failed fetches:
    - ``url_alive=True``: the server served content (2xx/3xx).
    - ``url_alive=False, blocked=True``: the server refused the request
      (401/403/429/451 — bot protection, paywall, or rate limit). The URL
      is likely still valid; we just could not verify it.
    - ``url_alive=False, blocked=False``: the URL is demonstrably unreachable
      (404/410, DNS failure, persistent 5xx, connection error).
    """

    url_alive: bool
    status_code: int | None = None
    content_type: str | None = None
    markdown: str | None = None
    title: str | None = None
    final_url: str | None = None
    error: str | None = None
    blocked: bool = False


class SearchHit(BaseModel):
    """Normalised per-source hit returned by :func:`search_by_title`."""

    source: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    landing_url: str | None = None
    abstract: str | None = None
    score: float | None = None
