"""Markdown input support: ManuscriptContext setup.

For markdown manuscripts we parse the ``.md`` into a ManuscriptContext
(sections, abstract, paragraphs), cache it, and attach it to the config
so the prose/structure checks treat ``.md``, ``.tex``, and ``.pdf`` inputs
uniformly. Unlike ``.tex`` (pandoc cleaning) and ``.pdf`` (GROBID), this
needs no external service — the text is already clean markdown.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig


def build_markdown_context(md_path: Path, config: LintConfig) -> None:
    """Parse a markdown manuscript and set up ManuscriptContext for checks.

    Builds the context, attaches its pandoc ``[@key]`` citations and the
    sibling ``.bib`` references, seeds the manuscript-context cache so the
    LLM checks calling ``get_or_create_manuscript_context`` receive the
    markdown parse (a cache hit) rather than re-parsing the ``.md`` as
    LaTeX, and attaches it to config so checks can detect markdown mode
    via ``config.is_markdown``.
    """
    from sciwrite_lint.manuscript_store import (
        ManuscriptContext,
        install_manuscript_context,
    )

    ctx = ManuscriptContext.from_markdown(md_path)
    _attach_citations(md_path, ctx)
    install_manuscript_context(md_path, ctx, config)


def _attach_citations(md_path: Path, ctx: object) -> None:
    """Populate ``inline_citations`` (pandoc) + ``parsed_references`` (.bib).

    Citation checks (e.g. dangling-cite) read the same two fields on the
    context for markdown as for PDF. Pandoc ``[@key]`` citations are
    extracted from the document; the bibliography is the sibling
    ``{stem}.bib`` (the pandoc / JOSS convention). A document using a
    different citation convention, or one without a sibling ``.bib``, is
    reported with a loud warning so the coverage gap is not silent.
    """
    from sciwrite_lint.manuscript_store import (
        InlineCitation,
        ManuscriptContext,
        ParsedReference,
    )
    from sciwrite_lint.markdown_cites import (
        analyze_markdown,
        detect_citation_style,
        resolve_bib_paths,
    )
    from sciwrite_lint.references.citations import parse_bib_file

    assert isinstance(ctx, ManuscriptContext)
    md_text = md_path.read_text(encoding="utf-8")
    analysis = analyze_markdown(md_text)
    cites = list(analysis.citations)
    style = detect_citation_style(md_text, cites)
    if style == "numeric":
        logger.warning(
            "{}: markdown manuscript uses numeric [1] citations; only pandoc "
            "[@key] citations are extracted, so citation checks will not run.",
            md_path.name,
        )

    ctx.inline_citations = [
        InlineCitation(key=c.key, line=None, context=c.context, section=c.section)
        for c in cites
    ]

    # Reference keys come from the manuscript's bibliography: an explicit
    # config bib, the YAML `bibliography:` field, or the sibling {stem}.bib
    # (pandoc / JOSS convention). Multiple .bib files are merged.
    bib_paths = resolve_bib_paths(md_path, analysis)
    if bib_paths:
        refs = [
            r
            for bib in bib_paths
            for r in parse_bib_file(bib, source_paper=md_path.stem)
        ]
        ctx.parsed_references = [
            ParsedReference(
                key=r.key,
                title=r.title,
                authors=list(r.authors),
                year=r.year,
                venue=r.venue,
                doi=r.doi,
                url=r.url,
                isbn=r.isbn,
                lccn=r.lccn,
                raw=r.raw_text,
            )
            for r in refs
        ]
    elif cites:
        logger.warning(
            "{}: {} pandoc citation(s) found but no bibliography (.bib) resolved "
            "(checked config, YAML bibliography:, and sibling {}.bib) — citation "
            "checks cannot validate against a bibliography.",
            md_path.name,
            len(cites),
            md_path.stem,
        )
