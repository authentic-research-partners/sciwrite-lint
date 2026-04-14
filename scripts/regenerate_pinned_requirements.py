#!/usr/bin/env python3
"""Regenerate requirements-pinned.txt from the active environment.

Filters `pip freeze` output down to the top-level dependencies declared in
pyproject.toml (runtime + all optional extras). Transitive dependencies are
not pinned.

Run this from the environment you actively develop and test in (e.g. the
`sciwrite-lint` conda env) — that is the ground truth the pinned file
represents.

With --check: compare the committed file against what would be generated now
and exit non-zero if they differ. Used by scripts/release.sh as a gate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
OUTPUT = REPO_ROOT / "requirements-pinned.txt"

HEADER = """\
# Pinned versions of sciwrite-lint's top-level dependencies.
#
# This file pins ONLY the packages declared in pyproject.toml (runtime + all
# optional extras). Transitive dependencies are NOT pinned — pip will resolve
# them within each top-level package's own constraints.
#
# Purpose: reproduce the exact versions of packages that sciwrite-lint is
# actively developed and tested against. Generated from the maintainer's
# working `sciwrite-lint` conda env via `pip freeze`, filtered to declared
# deps.
#
# Regenerate with:
#   python scripts/regenerate_pinned_requirements.py
#
# Install with:
#   pip install -r requirements-pinned.txt
#   pip install -e . --no-deps
#
# See docs/README.md → "Reproducible environment" for details.

"""


def normalize(name: str) -> str:
    """Return the PEP 503 normalized form of a package name."""
    bare = re.split(r"[<>=!~ ;\[]", name, maxsplit=1)[0].strip()
    return bare.lower().replace("_", "-")


def load_declared_deps() -> set[str]:
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    project = data["project"]
    specs: list[str] = list(project.get("dependencies", []))
    for extra_name, extra_specs in project.get("optional-dependencies", {}).items():
        if extra_name == "all":
            continue  # recursive self-reference
        specs.extend(extra_specs)
    return {normalize(s) for s in specs}


def pip_freeze() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def build_expected_content() -> str:
    """Return the text that requirements-pinned.txt *should* contain for the active env."""
    declared = load_declared_deps()
    if not declared:
        raise SystemExit("error: no dependencies declared in pyproject.toml")

    kept: list[str] = []
    found: set[str] = set()
    for line in pip_freeze():
        if not line or line.startswith(("-e ", "#")):
            continue
        name = re.split(r"[=<>!~]", line, maxsplit=1)[0].strip()
        key = normalize(name)
        if key in declared:
            kept.append(line)
            found.add(key)

    missing = sorted(declared - found)
    if missing:
        raise SystemExit(
            f"error: declared deps not installed in current env: {missing}\n"
            "Run this script from the environment where sciwrite-lint[dev] is installed."
        )

    kept.sort(key=str.lower)
    return HEADER + "\n".join(kept) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify requirements-pinned.txt matches the current env; exit non-zero if not.",
    )
    args = parser.parse_args()

    expected = build_expected_content()

    if args.check:
        if not OUTPUT.exists():
            print(
                f"error: {OUTPUT.relative_to(REPO_ROOT)} is missing.\n"
                "Run: python scripts/regenerate_pinned_requirements.py",
                file=sys.stderr,
            )
            return 1
        actual = OUTPUT.read_text()
        if actual != expected:
            print(
                f"error: {OUTPUT.relative_to(REPO_ROOT)} is out of sync with the current env.\n"
                "Either:\n"
                "  1. (intentional bump) regenerate and commit:\n"
                "     python scripts/regenerate_pinned_requirements.py\n"
                "  2. (unintended drift) restore your env to match the committed file:\n"
                "     pip install -r requirements-pinned.txt",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {OUTPUT.relative_to(REPO_ROOT)} matches current env")
        return 0

    OUTPUT.write_text(expected)
    count = sum(
        1 for line in expected.splitlines() if line and not line.startswith("#")
    )
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({count} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
