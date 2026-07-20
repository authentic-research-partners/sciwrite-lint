r"""Check: dangling-ref — cross-reference has no matching target.

For LaTeX: \ref{X} has no matching \label{X}.
For PDF: rendered "??" patterns in section text (broken references).
For markdown: pandoc-crossref ``@fig:`` / ``@sec:`` / ``@tbl:`` reference
with no matching labelled element.
"""

from __future__ import annotations

import re
from pathlib import Path

from sciwrite_lint.checks.registry import check
from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Finding
from sciwrite_lint.tex_parser import extract_body, find_all_commands, strip_comments


def _check_latex(tex_path: Path) -> list[Finding]:
    r"""Check that all \ref{} have matching \label{}."""
    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    body = extract_body(text)

    labels = {arg for _, arg in find_all_commands(body, "label")}
    refs = set()
    for cmd in ["ref", "eqref", "pageref"]:
        refs |= {arg for _, arg in find_all_commands(body, cmd)}

    findings: list[Finding] = []
    for ref in sorted(refs - labels):
        findings.append(
            Finding(
                level="error",
                rule_id="dangling-ref",
                message=f"\\ref{{{ref}}} has no matching \\label",
                file=tex_path.name,
            )
        )
    return findings


# Pattern for broken references rendered as "??" in PDFs
_BROKEN_REF_PATTERN = re.compile(
    r"""(?:
        (?:Figure|Fig\.|Table|Tab\.|Section|Sec\.|Equation|Eq\.|
           Theorem|Lemma|Proposition|Corollary|Definition|
           Algorithm|Alg\.|Appendix|App\.|Chapter|Ch\.)\s*\?\?
        |
        \(\s*\?\?\s*\)       # (??), e.g. rendered \eqref
        |
        (?<!\?)\?\?(?!\?)     # standalone ?? (not part of ???+)
    )""",
    re.VERBOSE | re.IGNORECASE,
)


def _check_pdf(ctx: object) -> list[Finding]:
    r"""Check for broken references ("??") in PDF section text.

    In a rendered PDF, a dangling \ref{} appears as "??" (e.g.,
    "Figure ??", "Eq. (??)"). GROBID preserves these in section text.
    """
    from sciwrite_lint.manuscript_store import ManuscriptContext

    assert isinstance(ctx, ManuscriptContext)

    findings: list[Finding] = []
    seen_contexts: set[str] = set()

    for sec in ctx.sections:
        for match in _BROKEN_REF_PATTERN.finditer(sec.clean_text):
            matched = match.group(0).strip()
            # Dedup by matched text + section
            dedup = f"{sec.title}:{matched}"
            if dedup in seen_contexts:
                continue
            seen_contexts.add(dedup)

            findings.append(
                Finding(
                    level="error",
                    rule_id="dangling-ref",
                    message=f"Broken reference '{matched}' in section '{sec.title}'",
                    file=ctx.source_path.name,
                )
            )

    # Also check abstract
    for match in _BROKEN_REF_PATTERN.finditer(ctx.abstract):
        matched = match.group(0).strip()
        dedup = f"abstract:{matched}"
        if dedup not in seen_contexts:
            seen_contexts.add(dedup)
            findings.append(
                Finding(
                    level="error",
                    rule_id="dangling-ref",
                    message=f"Broken reference '{matched}' in abstract",
                    file=ctx.source_path.name,
                )
            )

    return findings


def _check_markdown(md_path: Path) -> list[Finding]:
    """Flag pandoc-crossref references with no matching labelled element.

    ``@fig:x`` / ``@sec:x`` / ``@tbl:x`` whose target is not defined by any
    block id in the document is dangling.
    """
    from sciwrite_lint.markdown_cites import analyze_markdown

    analysis = analyze_markdown(md_path.read_text(encoding="utf-8"))
    findings: list[Finding] = []
    seen: set[str] = set()
    for xref in analysis.crossrefs:
        if xref.target not in analysis.labels and xref.target not in seen:
            seen.add(xref.target)
            findings.append(
                Finding(
                    level="error",
                    rule_id="dangling-ref",
                    message=f"Cross-reference '@{xref.target}' has no matching label",
                    file=md_path.name,
                    context=xref.target,
                )
            )
    return findings


@check(
    id="dangling-ref",
    category="manuscript",
    severity="error",
    description="Cross-reference has no matching target.",
)
def check_dangling_ref(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Check dangling references. Supports LaTeX, PDF, and markdown input."""
    if config.is_markdown:
        return _check_markdown(tex_path)
    if config.is_pdf:
        return _check_pdf(config.manuscript_context)

    return _check_latex(tex_path)
