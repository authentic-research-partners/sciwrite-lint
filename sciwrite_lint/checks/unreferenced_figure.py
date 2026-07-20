r"""Check: unreferenced-figure — figure exists but is never referenced.

Deterministic check: finds \label{fig:*} inside figure environments
that have no matching \ref{fig:*} anywhere in the document. The inverse
of dangling-ref (which catches \ref without \label).

For markdown: a figure labelled ``{#fig:*}`` that no ``@fig:*``
cross-reference points at. For PDF: a figure GROBID numbered (via its
``<label>``) whose number never appears in a "Figure N" mention in the
prose — conservative, since only labelled figures (not GROBID's
mis-detected ones) are considered.
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


# Figure references in rendered prose: "Figure 1", "Fig. 2", "Figs 3-5",
# "Figures 1, 2 and 3". Captures the run of numbers/separators after the word.
_PDF_FIG_REF_RE = re.compile(
    r"\bfig(?:ure)?s?\.?\s*((?:\d+\s*(?:[-–,]|and|&)\s*)*\d+)", re.IGNORECASE
)
_RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")


def _referenced_figure_numbers(body: str) -> set[str]:
    """Figure numbers mentioned in prose, expanding ranges and lists.

    Generous on purpose: a number in any "Fig N" context (incl. ranges and
    comma/and lists) counts as referenced, so the check does not falsely
    flag a figure that *is* cited inside e.g. "Figures 1-3".
    """
    referenced: set[str] = set()
    for match in _PDF_FIG_REF_RE.finditer(body):
        for part in re.split(r"\s*(?:,|and|&)\s*", match.group(1)):
            rng = _RANGE_RE.match(part)
            if rng:
                for num in range(int(rng.group(1)), int(rng.group(2)) + 1):
                    referenced.add(str(num))
            elif part.strip().isdigit():
                referenced.add(part.strip())
    return referenced


def _check_pdf(ctx: object) -> list[Finding]:
    """Flag GROBID-detected figures whose number never appears in the prose.

    Only figures GROBID gave a ``<label>`` (a clean number) are considered
    — its mis-detected figures have none — so this stays conservative. A
    figure number absent from every "Figure N" mention in the body/abstract
    is reported as unreferenced.
    """
    from sciwrite_lint.manuscript_store import ManuscriptContext

    assert isinstance(ctx, ManuscriptContext)
    defined = {n for n in ctx.figure_labels if n.isdigit()}
    if not defined:
        return []

    body = " ".join([ctx.abstract, *(sec.clean_text for sec in ctx.sections)])
    referenced = _referenced_figure_numbers(body)

    findings: list[Finding] = []
    for num in sorted(defined - referenced, key=int):
        findings.append(
            Finding(
                level="warning",
                rule_id="unreferenced-figure",
                message=f"Figure {num} is defined but never referenced in the text",
                file=ctx.source_path.name,
                context=num,
            )
        )
    return findings


def _check_markdown(md_path: Path) -> list[Finding]:
    """Flag ``{#fig:*}`` figure labels that no ``@fig:*`` reference points at."""
    from sciwrite_lint.markdown_cites import analyze_markdown

    analysis = analyze_markdown(md_path.read_text(encoding="utf-8"))
    if not analysis.figure_labels:
        return []
    referenced = {xref.target for xref in analysis.crossrefs}
    findings: list[Finding] = []
    for label in sorted(analysis.figure_labels - referenced):
        findings.append(
            Finding(
                level="warning",
                rule_id="unreferenced-figure",
                message=f"Figure '{label}' is defined but never referenced",
                file=md_path.name,
                context=label,
            )
        )
    return findings


@check(
    id="unreferenced-figure",
    category="manuscript",
    severity="warning",
    description="Figure is defined but never referenced in the text.",
)
def check_unreferenced_figure(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Check for figure labels that are never referenced."""
    if config.is_markdown:
        return _check_markdown(tex_path)
    if config.is_pdf:
        return _check_pdf(config.manuscript_context)

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
