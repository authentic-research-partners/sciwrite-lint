"""CLI handlers for config management (polite email, API keys)."""

from __future__ import annotations

import argparse
import re
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


def _read_key(filename: str) -> str | None:
    """Read an API key from ~/.sciwrite-lint/{filename}."""
    path = _KEY_DIR / filename
    if path.exists():
        val = path.read_text().strip()
        return val or None
    return None


def _mask(value: str) -> str:
    """Mask all but last 4 characters of a secret."""
    if len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


def run_config(args: argparse.Namespace) -> int:
    """Dispatch config subcommands."""
    action = getattr(args, "config_action", None)
    if action == "show":
        return _show(args)
    if action == "set-email":
        return _set_email(args)
    if action == "set-key":
        return _set_key(args)
    if action == "remove-key":
        return _remove_key(args)
    # No subcommand — print help (handled in __main__.py)
    return 0


def _show(args: argparse.Namespace) -> int:
    """Show current polite email and API key status."""
    from sciwrite_lint.__main__ import _load_config

    config = _load_config(args)

    print("Polite email and API key configuration\n")

    # Polite email
    email = config.polite_email
    if email:
        print(f"  polite email:      {email}")
        print("    → CrossRef polite pool, Unpaywall, Retraction Watch")
    else:
        print("  polite email:      (not set)")
        print("    ⚠ CrossRef: slower rate limits without polite pool")
        print("    ⚠ Unpaywall: will not work without an email")
        print("    ⚠ Retraction Watch: will not work without an email")
        print("    Set with: sciwrite-lint config set-email you@example.com")
    print()

    # API keys
    print("  API keys (stored in ~/.sciwrite-lint/):\n")
    for service, info in _API_KEY_SERVICES.items():
        key = _read_key(info["file"])
        if key:
            print(f"    {service:<20} {_mask(key)}  ({info['description']})")
        else:
            print(f"    {service:<20} (not set)  — {info['description']}")
            print(f"      Get one at: {info['url']}")

    print()
    return 0


def _set_email(args: argparse.Namespace) -> int:
    """Set polite email in .sciwrite-lint.toml."""
    email = args.email.strip()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        logger.error("Invalid email address: {}", email)
        return 1

    from sciwrite_lint.config import find_config

    toml_path = find_config()
    if toml_path is None:
        logger.error("No .sciwrite-lint.toml found. Run `sciwrite-lint init` first.")
        return 1

    content = toml_path.read_text()

    # Case 1: [api] section exists with polite_email (commented or not)
    if re.search(r"^#?\s*polite_email\s*=", content, re.MULTILINE):
        content = re.sub(
            r"^#?\s*polite_email\s*=.*$",
            f'polite_email = "{email}"',
            content,
            count=1,
            flags=re.MULTILINE,
        )
    # Case 2: [api] section exists but no polite_email line
    elif re.search(r"^\[api\]", content, re.MULTILINE):
        content = re.sub(
            r"^(\[api\].*)$",
            rf'\1\npolite_email = "{email}"',
            content,
            count=1,
            flags=re.MULTILINE,
        )
    # Case 3: no [api] section at all
    else:
        content = content.rstrip() + f'\n\n[api]\npolite_email = "{email}"\n'

    toml_path.write_text(content)
    print(f"Set polite email to: {email}")
    print("  → CrossRef polite pool (faster rate limits)")
    print("  → Unpaywall (full-text access)")
    print("  → Retraction Watch (retraction database)")
    return 0


def _set_key(args: argparse.Namespace) -> int:
    """Save an API key to ~/.sciwrite-lint/<service>."""
    service = args.service
    if service not in _API_KEY_SERVICES:
        names = ", ".join(_API_KEY_SERVICES)
        logger.error("Unknown service '{}'. Available: {}", service, names)
        return 1

    key = args.key.strip()
    if not key:
        logger.error("API key cannot be empty")
        return 1

    info = _API_KEY_SERVICES[service]
    _KEY_DIR.mkdir(exist_ok=True)
    key_path = _KEY_DIR / info["file"]
    key_path.write_text(key + "\n")
    key_path.chmod(0o600)

    print(f"Saved {service} API key to {key_path}")
    print(f"  → {info['description']}")
    return 0


def _remove_key(args: argparse.Namespace) -> int:
    """Remove a stored API key."""
    service = args.service
    if service not in _API_KEY_SERVICES:
        names = ", ".join(_API_KEY_SERVICES)
        logger.error("Unknown service '{}'. Available: {}", service, names)
        return 1

    info = _API_KEY_SERVICES[service]
    key_path = _KEY_DIR / info["file"]
    if key_path.exists():
        key_path.unlink()
        print(f"Removed {service} API key ({key_path})")
    else:
        print(f"No {service} API key found (nothing to remove)")
    return 0
