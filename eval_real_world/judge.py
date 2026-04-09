"""Sonnet-based adjudication of linter findings.

For each finding the linter produces on a real paper, Sonnet decides
whether it is a true positive (TP), false positive (FP), or uncertain.
This drives the false-positive-rate measurement.

Requires the `claude` CLI installed and accessible.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from sciwrite_lint.claude_cli import (
    run_claude_async_validated,
)
from sciwrite_lint.models import Finding


class VerdictResponse(BaseModel):
    """Expected JSON response from Sonnet adjudication."""

    judgment: Literal["TP", "FP", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=500)


class Verdict(BaseModel):
    """Sonnet adjudication of a single finding."""

    rule_id: str
    finding_message: str
    judgment: str  # "TP", "FP", or "UNCERTAIN"
    confidence: float  # 0.0–1.0
    reasoning: str
    original_finding: dict | None = None


# Agent file for strong identity (--agent flag replaces Claude's default identity)
_AGENT_PATH = Path(__file__).parent / "agents" / "fpr_judge.md"


def _extract_context(tex_text: str, finding: Finding, context_lines: int = 10) -> str:
    """Extract manuscript context around a finding."""
    lines = tex_text.split("\n")

    if finding.line and finding.line <= len(lines):
        start = max(0, finding.line - context_lines - 1)
        end = min(len(lines), finding.line + context_lines)
        context_block = "\n".join(
            f"{i + 1}: {line}" for i, line in enumerate(lines[start:end], start=start)
        )
        return f"Lines {start + 1}–{end}:\n{context_block}"

    # No line number — search for context string in the text
    if finding.context:
        for i, line in enumerate(lines):
            if finding.context[:40] in line:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                return "\n".join(
                    f"{j + 1}: {ln}"
                    for j, ln in enumerate(lines[start:end], start=start)
                )

    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines[:50]))


async def judge_finding(
    finding: Finding,
    tex_text: str,
    project_dir: Path | None = None,
    timeout: int = 120,
) -> Verdict:
    """Ask Sonnet to adjudicate a single finding via async subprocess.

    Args:
        finding: The linter finding to judge.
        tex_text: Full LaTeX text of the paper.
        project_dir: Working directory for claude CLI.
        timeout: CLI timeout in seconds.

    Returns:
        Verdict with TP/FP/UNCERTAIN judgment.
    """
    context = _extract_context(tex_text, finding)
    user_prompt = (
        f"Evaluate this linter finding as TP, FP, or UNCERTAIN. "
        f"Respond with ONLY a JSON object: "
        f'{{"judgment":"TP","confidence":0.9,"reasoning":"why"}}\n\n'
        f"## Linter Finding\n\n"
        f"- Rule: {finding.rule_id}\n"
        f"- Level: {finding.level}\n"
        f"- Message: {finding.message}\n"
        f"- File: {finding.file}\n"
        f"- Line: {finding.line or 'N/A'}\n"
        f"- Context: {finding.context or 'N/A'}\n\n"
        f"## Manuscript Context\n\n```latex\n{context}\n```"
    )

    result = await run_claude_async_validated(
        user_prompt,
        VerdictResponse,
        agent=_AGENT_PATH,
        timeout=timeout,
        cwd=project_dir,
    )

    if result is None:
        return Verdict(
            rule_id=finding.rule_id,
            finding_message=finding.message,
            judgment="UNCERTAIN",
            confidence=0.0,
            reasoning="Claude CLI error, timeout, or validation failure",
        )

    return Verdict(
        rule_id=finding.rule_id,
        finding_message=finding.message,
        judgment=result.judgment,
        confidence=result.confidence,
        reasoning=result.reasoning,
        original_finding=finding.model_dump(),
    )


async def judge_findings(
    findings: list[Finding],
    tex_text: str,
    project_dir: Path | None = None,
    max_findings: int | None = None,
    concurrency: int = 3,
) -> list[Verdict]:
    """Judge multiple findings concurrently via async claude CLI calls.

    Args:
        findings: Findings to adjudicate.
        tex_text: Full LaTeX text.
        project_dir: Working directory for claude CLI.
        max_findings: Cap on how many to judge (cost control).
        concurrency: Max parallel Sonnet calls (default: 3).

    Returns:
        List of Verdicts in original order.
    """
    to_judge = findings[:max_findings] if max_findings else findings
    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int, finding: Finding) -> Verdict:
        async with sem:
            logger.info(
                f"Judging [{i}/{len(to_judge)}] {finding.rule_id}: {finding.message[:60]}"
            )
            verdict = await judge_finding(finding, tex_text, project_dir)
            logger.info(
                f"Verdict [{i}]: {verdict.judgment} (confidence: {verdict.confidence:.1f})"
            )
            return verdict

    verdicts = await asyncio.gather(*[_one(i, f) for i, f in enumerate(to_judge, 1)])
    return list(verdicts)
