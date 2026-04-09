r"""Check: unreferenced-figure — figure exists but is never referenced.

Deterministic check: finds \label{fig:*} inside figure environments
that have no matching \ref{fig:*} anywhere in the document. The inverse
of dangling-ref (which catches \ref without \label).

LaTeX only — PDF figures don't have labels to cross-reference.
"""

from __future__ import annotations

import re
from pathlib import Path

from sciwrite_lint.checks.registry import check
from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Finding
from sciwrite_lint.tex_parser import extract_body, find_all_commands, strip_comments

# Figure environments that contain \label{fig:*}
_FIGURE_ENV_RE = re.compile(
    r"\\begin\{(?:figure|figure\*|subfigure)\}(.*?)\\end\{(?:figure|figure\*|subfigure)\}",
    re.DOTALL,
)
_LABEL_RE = re.compile(r"\\label\{(fig:[^}]+)\}")


def _figure_labels(body: str) -> set[str]:
    """Extract all fig:* labels from figure environments."""
    labels: set[str] = set()
    for env_match in _FIGURE_ENV_RE.finditer(body):
        for label_match in _LABEL_RE.finditer(env_match.group(1)):
            labels.add(label_match.group(1))
    return labels


@check(
    id="unreferenced-figure",
    category="manuscript",
    severity="warning",
    description="Figure is defined but never referenced in the text.",
)
def check_unreferenced_figure(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Check for figure labels that are never referenced."""
    if config.is_pdf:
        return []  # PDF figures don't have labels

    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    body = extract_body(text)

    fig_labels = _figure_labels(body)
    if not fig_labels:
        return []

    # All \ref{} targets in the document
    refs: set[str] = set()
    for cmd in ["ref", "eqref", "pageref"]:
        refs |= {arg for _, arg in find_all_commands(body, cmd)}

    findings: list[Finding] = []
    for label in sorted(fig_labels - refs):
        findings.append(
            Finding(
                level="warning",
                rule_id="unreferenced-figure",
                message=f"\\label{{{label}}} in figure environment is never referenced",
                file=tex_path.name,
            )
        )
    return findings
