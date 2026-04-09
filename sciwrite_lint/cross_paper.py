"""Cross-paper consistency checks between papers in the same project."""

from __future__ import annotations

import re
from pathlib import Path

from sciwrite_lint.models import CheckResult, Finding
from sciwrite_lint.tex_parser import (
    body_without_bibliography,
    find_all_cite_keys,
    strip_comments,
)


def check_citation_direction(
    paper_a_path: Path,
    paper_b_path: Path,
    a_ref_keys: set[str] | None = None,
    b_ref_keys: set[str] | None = None,
) -> list[Finding]:
    """Verify citation direction between two papers.

    Args:
        a_ref_keys: Keys in Paper B that cite Paper A (e.g. {"smith2024example"})
        b_ref_keys: Keys in Paper A that would cite Paper B (forbidden)
    """
    findings = []
    a_ref_keys = a_ref_keys or set()
    b_ref_keys = b_ref_keys or set()

    text_a = strip_comments(paper_a_path.read_text(encoding="utf-8"))
    text_b = strip_comments(paper_b_path.read_text(encoding="utf-8"))

    cite_keys_a = {k for _, k in find_all_cite_keys(text_a)}
    cite_keys_b = {k for _, k in find_all_cite_keys(text_b)}

    # Paper A must NOT cite Paper B
    a_cites_b = cite_keys_a & b_ref_keys
    for key in a_cites_b:
        findings.append(
            Finding(
                level="error",
                rule_id="ref-006",
                message=f"Paper A cites Paper B via \\cite{{{key}}} — forbidden",
                file=paper_a_path.name,
            )
        )

    # Paper B should cite Paper A
    if a_ref_keys and not (cite_keys_b & a_ref_keys):
        findings.append(
            Finding(
                level="warning",
                rule_id="ref-006",
                message="Paper B does not cite Paper A",
                file=paper_b_path.name,
            )
        )

    return findings


def check_shared_terminology(
    paper_a_path: Path,
    paper_b_path: Path,
    prohibited_in_a: list[str] | None = None,
    required_in_b: list[str] | None = None,
    shared_modes: list[str] | None = None,
) -> list[Finding]:
    """Check terminology consistency between papers."""
    findings = []

    body_a = body_without_bibliography(
        strip_comments(paper_a_path.read_text(encoding="utf-8"))
    )
    body_b = body_without_bibliography(
        strip_comments(paper_b_path.read_text(encoding="utf-8"))
    )

    # Check prohibited terms in Paper A
    for term in prohibited_in_a or []:
        matches = [
            m
            for m in re.finditer(re.escape(term), body_a)
            if not _is_in_job_title(body_a, m.start())
        ]
        for m in matches:
            line = body_a[: m.start()].count("\n") + 1
            findings.append(
                Finding(
                    level="error",
                    rule_id="style-005",
                    message=f"'{term}' found in Paper A body",
                    file=paper_a_path.name,
                    line=line,
                )
            )

    # Check required terms appear enough in Paper B
    for term in required_in_b or []:
        count = len(re.findall(re.escape(term), body_b))
        if count < 3:
            findings.append(
                Finding(
                    level="warning",
                    rule_id="con-007",
                    message=f"'{term}' appears only {count} times in Paper B (expected more)",
                    file=paper_b_path.name,
                )
            )

    # Check shared mode names are consistent
    for mode_pattern in shared_modes or []:
        in_a = bool(re.search(mode_pattern, body_a, re.IGNORECASE))
        in_b = bool(re.search(mode_pattern, body_b, re.IGNORECASE))
        if in_a and not in_b:
            findings.append(
                Finding(
                    level="info",
                    rule_id="con-007",
                    message=f"Mode '{mode_pattern}' found in Paper A but not Paper B",
                    file=paper_b_path.name,
                )
            )

    return findings


def _is_in_job_title(text: str, pos: int) -> bool:
    """Check if position is within a job title context."""
    start = max(0, pos - 100)
    context = text[start : pos + 50]
    return any(
        x in context.lower()
        for x in [
            "titled",
            "title",
            "advertised as",
            "``",
            "teacher and",
        ]
    )


def run_cross_paper_check(
    paper_a_path: Path,
    paper_b_path: Path,
    a_ref_keys: set[str] | None = None,
    b_ref_keys: set[str] | None = None,
    prohibited_in_a: list[str] | None = None,
    required_in_b: list[str] | None = None,
    shared_modes: list[str] | None = None,
) -> CheckResult:
    """Run all cross-paper consistency checks."""
    result = CheckResult(checker="cross_paper", paper="cross")
    result.findings.extend(
        check_citation_direction(
            paper_a_path,
            paper_b_path,
            a_ref_keys,
            b_ref_keys,
        )
    )
    result.findings.extend(
        check_shared_terminology(
            paper_a_path,
            paper_b_path,
            prohibited_in_a,
            required_in_b,
            shared_modes,
        )
    )
    return result
