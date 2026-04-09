"""Check registry — decorator-based registration and discovery.

Usage:
    from sciwrite_lint.checks.registry import check, get_checks, list_checks

    @check(
        id="dangling-cite",
        category="manuscript",
        description="\\cite{key} has no matching bibliography entry.",
    )
    def check_dangling_cite(tex_path, config):
        ...

Checks are discovered by importing check modules. Call ensure_checks_loaded()
before querying the registry.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Callable

from sciwrite_lint.models import CheckMeta

if TYPE_CHECKING:
    from sciwrite_lint.config import LintConfig

# Global registry: check_id -> (meta, callable)
_CHECKS: dict[str, tuple[CheckMeta, Callable]] = {}

# Track whether modules have been imported
_LOADED = False

# All check modules to auto-import
_CHECK_MODULES = [
    "sciwrite_lint.checks.dangling_cite",
    "sciwrite_lint.checks.dangling_ref",
    "sciwrite_lint.checks.cross_section_consistency",
    "sciwrite_lint.checks.structure_promises",
    "sciwrite_lint.checks.reference_exists",
    "sciwrite_lint.checks.reference_accuracy",
    "sciwrite_lint.checks.retracted_cite",
    "sciwrite_lint.checks.full_paper_consistency",
    "sciwrite_lint.checks.unreferenced_figure",
]

# Checks that run as pipeline stages, not from the registry.
# Listed separately so `sciwrite-lint checks` can show them.
PIPELINE_STAGE_CHECKS: list["CheckMeta"] = []


def _load_pipeline_stage_checks() -> None:
    """Import pipeline-stage check metadata (lazy, called by list_checks)."""
    if PIPELINE_STAGE_CHECKS:
        return
    from sciwrite_lint.checks.claim_support import CLAIM_SUPPORT_META
    from sciwrite_lint.checks.cite_purpose import CITE_PURPOSE_META
    from sciwrite_lint.checks.reference_unreliable import REFERENCE_UNRELIABLE_META

    PIPELINE_STAGE_CHECKS.extend(
        [
            CLAIM_SUPPORT_META,
            CITE_PURPOSE_META,
            REFERENCE_UNRELIABLE_META,
        ]
    )


def check(
    *,
    id: str,
    category: str,
    description: str,
    severity: str = "warning",
) -> Callable:
    """Decorator to register a check function."""

    def decorator(fn: Callable) -> Callable:
        meta = CheckMeta(
            id=id,
            severity=severity,  # type: ignore[arg-type]
            category=category,  # type: ignore[arg-type]
            description=description,
        )
        _CHECKS[id] = (meta, fn)
        fn.check_meta = meta  # type: ignore[attr-defined]
        return fn

    return decorator


def ensure_checks_loaded() -> None:
    """Import all check modules so their @check decorators fire."""
    global _LOADED
    if _LOADED:
        return
    for mod_name in _CHECK_MODULES:
        importlib.import_module(mod_name)
    _LOADED = True


def get_checks(
    config: LintConfig | None = None,
    category: str | None = None,
) -> list[tuple[CheckMeta, Callable]]:
    """Return all registered checks, optionally filtered."""
    ensure_checks_loaded()
    results = []
    for check_id, (meta, fn) in sorted(_CHECKS.items()):
        if category and meta.category != category:
            continue
        if config and not config.is_check_enabled(check_id):
            continue
        results.append((meta, fn))
    return results


def get_check(check_id: str) -> tuple[CheckMeta, Callable] | None:
    """Look up a single check by ID."""
    ensure_checks_loaded()
    return _CHECKS.get(check_id)


def list_checks() -> list[CheckMeta]:
    """List all checks — registry checks + pipeline-stage checks."""
    ensure_checks_loaded()
    _load_pipeline_stage_checks()
    registry = [meta for meta, _fn in sorted(_CHECKS.values(), key=lambda x: x[0].id)]
    return sorted(registry + PIPELINE_STAGE_CHECKS, key=lambda m: m.id)


def clear_registry() -> None:
    """Clear all checks (for testing)."""
    global _LOADED
    _CHECKS.clear()
    _LOADED = False
