"""Cross-paper consistency checks between papers in the same project."""

from __future__ import annotations

from pathlib import Path

from sciwrite_lint.models import Finding
from sciwrite_lint.tex_parser import (
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
