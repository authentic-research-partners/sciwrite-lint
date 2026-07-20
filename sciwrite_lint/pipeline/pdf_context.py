"""PDF input support: ManuscriptContext setup and Citation extraction.

For PDF manuscripts we run GROBID once up front, cache the parsed
``ManuscriptContext`` on the config, then convert its parsed references
into ``Citation`` objects so the rest of the pipeline (verify, fetch,
etc.) can treat .tex and .pdf inputs uniformly.
"""

from __future__ import annotations

from pathlib import Path

from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Citation


async def build_pdf_context(
    pdf_path: Path,
    config: LintConfig,
) -> None:
    """Parse a PDF via GROBID and set up ManuscriptContext for all checks.

    Builds ManuscriptContext from GROBID output, caches it, and attaches
    it to config so checks can detect PDF mode.
    """
    from sciwrite_lint.manuscript_store import (
        ManuscriptContext,
        install_manuscript_context,
    )
    from sciwrite_lint.pdf.grobid import process_pdf

    grobid_result = await process_pdf(pdf_path)
    ctx = ManuscriptContext.from_grobid(pdf_path, grobid_result)
    install_manuscript_context(pdf_path, ctx, config)


def citations_from_pdf_context(config: LintConfig) -> list[Citation]:
    """Extract Citation objects from a PDF ManuscriptContext.

    Converts ParsedReference entries to Citation objects suitable for the
    API verification pipeline.
    """
    from sciwrite_lint.manuscript_store import ManuscriptContext

    ctx: ManuscriptContext = config.manuscript_context
    citations: list[Citation] = []
    for ref in ctx.parsed_references:
        citations.append(
            Citation(
                key=ref.key,
                raw_text=ref.raw,
                authors=list(ref.authors),
                title=ref.title,
                year=ref.year,
                venue=ref.venue,
                doi=ref.doi,
                url=ref.url,
                isbn=ref.isbn,
                lccn=ref.lccn,
                source_paper=ctx.source_path.stem,
                bib_format="grobid",
                entry_type="",
            )
        )
    return citations


async def extract_citations_for_paper(
    tex_path: Path,
    config: LintConfig,
    bib_path: Path | None = None,
) -> list[Citation]:
    """Extract citations from a paper, handling .tex, .pdf, and markdown inputs.

    For .tex, uses extract_bibitems. For .pdf, runs GROBID via
    build_pdf_context + citations_from_pdf_context. For markdown, parses
    the resolved bibliography (config / YAML ``bibliography:`` / sibling
    ``.bib``). All paths then attach local sources.
    """
    from sciwrite_lint.references.citations import check_local_sources, extract_bibitems

    suffix = tex_path.suffix.lower()
    if suffix == ".pdf":
        await build_pdf_context(tex_path, config)
        return citations_from_pdf_context(config)

    if suffix == ".md":
        from sciwrite_lint.markdown_cites import analyze_markdown, parse_markdown_bib

        analysis = analyze_markdown(tex_path.read_text(encoding="utf-8"))
        citations = parse_markdown_bib(tex_path, analysis, bib_path)
    else:
        citations = extract_bibitems(tex_path, "auto", bib_path=bib_path)
    check_local_sources(citations, config.effective_references_dir())
    return citations
