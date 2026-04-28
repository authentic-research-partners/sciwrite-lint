"""Stage 4.5: internal consistency checks over cited papers (vLLM, thinking=low/medium)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sciwrite_lint.config import LintConfig


async def _stage_ref_internal(
    references_dir: Path,
    config: LintConfig,
    *,
    fresh: bool = False,
    ref_figure_descs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run consistency checks on cited papers (automatic in default pipeline)."""
    from sciwrite_lint.checks.ref_internal_checks import run_ref_internal_checks

    return await run_ref_internal_checks(
        references_dir,
        config,
        fresh=fresh,
        ref_figure_descs=ref_figure_descs,
    )
