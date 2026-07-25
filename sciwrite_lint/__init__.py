"""sciwrite-lint: a linter for scientific manuscripts."""

from sciwrite_lint.exceptions import LLMConnectionError, SciWriteLintError

__version__ = "0.6.1"

from sciwrite_lint.oa import (
    DownloadResult,
    FetchConfig,
    SearchHit,
    WebResult,
    download_pdf,
    fetch_web,
    search_by_title,
)

__all__ = [
    "DownloadResult",
    "FetchConfig",
    "LLMConnectionError",
    "SciWriteLintError",
    "SearchHit",
    "WebResult",
    "__version__",
    "download_pdf",
    "fetch_web",
    "search_by_title",
]
