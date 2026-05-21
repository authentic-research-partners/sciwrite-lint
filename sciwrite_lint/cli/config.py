"""CLI handlers for config management (polite email, API keys)."""

from __future__ import annotations

import argparse
import re

from loguru import logger

# The shared API-key surface (registry, on-disk reader, preflight
# validator) lives in ``sciwrite_lint.api_keys`` so the pipeline layer
# can import ``check_api_config`` without depending on the CLI layer.
# Re-imported here for the show/set-key/remove-key handlers below.
from sciwrite_lint.api_keys import (
    _API_KEY_SERVICES,
    _KEY_DIR,
    _read_key,
    check_api_config as check_api_config,
)


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
    from sciwrite_lint.cli._common import _load_config

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
