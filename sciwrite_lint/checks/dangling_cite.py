"""Check: dangling-cite — citation has no matching bibliography entry.

For LaTeX: \\cite{key} has no matching \\bibitem or .bib entry.
For PDF: inline citation marker not linked to any reference in GROBID TEI.
"""

from __future__ import annotations

from pathlib import Path

from sciwrite_lint.checks.registry import check
from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Finding


def _check_latex(tex_path: Path, config: LintConfig) -> list[Finding]:
    r"""Check that every \cite{key} has a matching \bibitem or .bib entry."""
    from sciwrite_lint.references.citations import extract_bibitems
    from sciwrite_lint.tex_parser import find_all_cite_keys, strip_comments

    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    cite_keys = find_all_cite_keys(text)

    try:
        citations = extract_bibitems(tex_path)
    except ValueError:
        return []

    bib_keys = {c.key for c in citations}
    if not bib_keys:
        return []

    findings: list[Finding] = []
    seen: set[str] = set()
    for line_no, key in cite_keys:
        if key not in bib_keys and key not in seen:
            seen.add(key)
            findings.append(
                Finding(
                    level="error",
                    rule_id="dangling-cite",
                    message=f"\\cite{{{key}}} has no matching bibliography entry",
                    file=tex_path.name,
                    line=line_no,
                    context=key,
                )
            )
    return findings


def _check_context(ctx: object) -> list[Finding]:
    """Check that inline citations link to a reference in the bibliography.

    Serves any ManuscriptContext that populates ``inline_citations`` and
    ``parsed_references`` — PDF (GROBID TEI <ref> tags) and markdown
    (pandoc ``[@key]`` + sibling ``.bib``) both do. Citations whose key is
    not in the reference list are dangling.
    """
    from sciwrite_lint.manuscript_store import ManuscriptContext

    assert isinstance(ctx, ManuscriptContext)
    ref_keys = {r.key for r in ctx.parsed_references}
    if not ref_keys:
        return []

    findings: list[Finding] = []
    seen: set[str] = set()
    for cite in ctx.inline_citations:
        if cite.key not in ref_keys and cite.key not in seen:
            seen.add(cite.key)
            findings.append(
                Finding(
                    level="error",
                    rule_id="dangling-cite",
                    message=f"Citation '{cite.key}' has no matching reference entry",
                    file=ctx.source_path.name,
                    context=cite.key,
                )
            )
    return findings


@check(
    id="dangling-cite",
    category="manuscript",
    severity="error",
    description="Citation has no matching bibliography entry.",
)
def check_dangling_cite(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Check dangling citations. Supports LaTeX, PDF, and markdown input.

    PDF and markdown both populate ``inline_citations`` +
    ``parsed_references`` on the context, so the same key-membership
    check (``_check_context``) serves both; LaTeX is parsed from source.
    """
    if config.is_pdf or config.is_markdown:
        return _check_context(config.manuscript_context)

    return _check_latex(tex_path, config)
