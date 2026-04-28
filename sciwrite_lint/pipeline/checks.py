"""Stage 1: manuscript + local-LLM checks (the registered check engine).

``run_text_checks`` runs every non-LLM, non-reference-db check.
``run_llm_checks_batched`` dispatches the ``local-llm`` checks through
the batched vLLM runner so one query can cover multiple checks.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Finding


def run_text_checks(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Run all manuscript-engine checks (CPU-bound, no I/O). Returns findings."""
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_checks

    ensure_checks_loaded()
    findings: list[Finding] = []

    for meta, fn in get_checks(config=config):
        if meta.category in ("reference-db", "local-llm"):
            continue
        try:
            check_findings = fn(tex_path, config)
            for f in check_findings:
                override = config.effective_severity(meta.id, meta.severity)
                if override != f.level:
                    f.level = override  # type: ignore[assignment]
            findings.extend(check_findings)
        except Exception as e:
            logger.warning(f"Check {meta.id} skipped: {e}")
            findings.append(
                Finding(
                    level="info",
                    rule_id=meta.id,
                    message=f"Check {meta.id} could not run (internal error)",
                    context=f"{type(e).__name__}: {e!s}"[:200],
                )
            )

    return findings


async def run_llm_checks_batched(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Run all local-llm-engine checks via batched vLLM queries. Returns findings."""
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_checks
    from sciwrite_lint.cli.check import (
        run_llm_checks_batched as _run_llm_checks_batched,
    )

    ensure_checks_loaded()
    llm_checks = [
        (meta, fn)
        for meta, fn in get_checks(config=config)
        if meta.category == "local-llm"
    ]
    if not llm_checks:
        return []
    return await _run_llm_checks_batched(llm_checks, tex_path, config)
