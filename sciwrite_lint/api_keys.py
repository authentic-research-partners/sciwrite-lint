"""Optional-API-key registry, on-disk reader, and config validator.

Holds the cross-layer surface that both the CLI (``cli/config.py`` for
show / set-key / remove-key commands) and the pipeline preflight
(``pipeline/orchestration.py``) need to share. Living here keeps the
pipeline layer from importing the CLI layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from sciwrite_lint.config import LintConfig


# Services that accept API keys, stored in ~/.sciwrite-lint/<filename>
_API_KEY_SERVICES: dict[str, dict[str, str]] = {
    "semantic-scholar": {
        "file": "s2_api_key",
        "description": "Semantic Scholar (1 → 100 req/s)",
        "url": "https://www.semanticscholar.org/product/api#api-key",
    },
    "ncbi": {
        "file": "ncbi_api_key",
        "description": "NCBI / PubMed Central (3 → 10 req/s)",
        "url": "https://www.ncbi.nlm.nih.gov/account/settings/",
    },
    "core": {
        "file": "core_api_key",
        "description": "CORE (institutional repository access)",
        "url": "https://core.ac.uk/services/api",
    },
    "nasa-ads": {
        "file": "nasa_ads_api_key",
        "description": "NASA ADS (astronomy/astrophysics; 5000 req/day per token)",
        "url": "https://ui.adsabs.harvard.edu/user/settings/token",
    },
}

_KEY_DIR = Path.home() / ".sciwrite-lint"


def _read_key(filename: str) -> str | None:
    """Read an API key from ~/.sciwrite-lint/{filename}."""
    path = _KEY_DIR / filename
    if path.exists():
        val = path.read_text().strip()
        return val or None
    return None


def check_api_config(config: "LintConfig", needs_email: bool = False) -> list[str]:
    """Validate API configuration. Returns list of errors (hard stops).

    Also logs warnings for missing optional API keys.

    Args:
        config: Resolved LintConfig.
        needs_email: If True, missing polite_email is an error (not just a warning).
    """
    errors: list[str] = []

    # polite_email: required for Unpaywall + Retraction Watch
    if not config.polite_email:
        if needs_email:
            errors.append(
                "polite_email not set — Unpaywall and Retraction Watch will not work. "
                "Set with: sciwrite-lint config set-email you@example.com"
            )
        else:
            logger.warning(
                "polite_email not set — Unpaywall and Retraction Watch disabled. "
                "Set with: sciwrite-lint config set-email you@example.com"
            )

    # Optional API keys: warn about rate limit benefits
    missing_keys: list[str] = []
    for service, info in _API_KEY_SERVICES.items():
        if not _read_key(info["file"]):
            missing_keys.append(f"{service} ({info['description']})")

    if missing_keys:
        logger.info(
            "Optional API keys not configured (slower rate limits): {}. "
            "See: sciwrite-lint config show",
            ", ".join(missing_keys),
        )

    return errors
