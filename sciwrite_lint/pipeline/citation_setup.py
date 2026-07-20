"""Source-aware citation extraction for the full pipeline.

One entry point — :func:`extract_pipeline_citations` — produces the
cited ``Citation`` list that the verify / fetch / reference-db stages
consume, branching on the manuscript source type:

- **PDF**      — citations come from the GROBID-parsed context.
- **markdown** — bibliography from the sibling ``.bib``, narrowed to the
  pandoc ``[@key]`` citations actually present in the document.
- **LaTeX**    — ``\\bibitem`` / ``.bib`` discovery, narrowed to ``\\cite``
  keys, plus ``\\footnote{\\url{}}`` source synthesis.

Once the list exists the downstream stages are source-agnostic — they
operate on ``Citation`` objects, not on the manuscript syntax — so this
is the single place where source type matters for citations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.models import Citation


def extract_pipeline_citations(
    paper_name: str,
    tex_path: Path,
    pc: Any,  # PaperConfig — Any to avoid a config import cycle at call sites
    config: LintConfig,
    refs_dir: Path,
) -> list[Citation]:
    """Return the cited references for the pipeline, by source type."""
    if config.is_pdf:
        from sciwrite_lint.pipeline.pdf_context import citations_from_pdf_context

        return citations_from_pdf_context(config)
    if config.is_markdown:
        return _markdown_citations(paper_name, tex_path, pc, config, refs_dir)
    return _latex_citations(paper_name, tex_path, pc, config, refs_dir)


def _markdown_citations(
    paper_name: str,
    tex_path: Path,
    pc: Any,
    config: LintConfig,
    refs_dir: Path,
) -> list[Citation]:
    """Bibliography (narrowed to pandoc cites) + footnote-URL web sources.

    Cited keys come from the markdown context's ``inline_citations``
    (pandoc ``[@key]`` occurrences populated by ``build_markdown_context``),
    so uncited bib entries are dropped before verify / fetch — the same
    contract as ``filter_to_cited`` for LaTeX. Markdown footnotes carrying
    a ``<url>`` / ``[text](url)`` link are also synthesized into T1 web
    citations, mirroring LaTeX ``\\footnote{\\url{}}``.
    """
    from sciwrite_lint.footnote_urls import ingest_footnote_sources
    from sciwrite_lint.manuscript_store import get_or_create_manuscript_context
    from sciwrite_lint.markdown_cites import analyze_markdown, parse_markdown_bib
    from sciwrite_lint.references.citations import check_local_sources

    explicit_bib = Path(pc.bib) if getattr(pc, "bib", None) else None
    analysis = analyze_markdown(tex_path.read_text(encoding="utf-8"))
    bib_citations = parse_markdown_bib(tex_path, analysis, explicit_bib)

    # Cited keys come from this paper's markdown context. Read it by path
    # from the cache (seeded by build_markdown_context) rather than from
    # the single config.manuscript_context slot, which the batch pipeline
    # overwrites per paper.
    ctx = get_or_create_manuscript_context(tex_path, config)
    cited_keys = {ic.key for ic in ctx.inline_citations}

    citations: list[Citation] = []
    if bib_citations:
        n_bib = len(bib_citations)
        cited = [c for c in bib_citations if c.key in cited_keys]
        check_local_sources(cited, refs_dir)
        citations.extend(cited)
        if n_bib > len(cited):
            logger.info(
                "{}: {}/{} bib entries are cited — skipping verify+fetch for {} uncited",
                tex_path.name,
                len(cited),
                n_bib,
                n_bib - len(cited),
            )
    elif cited_keys:
        logger.warning(
            "{}: pandoc citations found but no bibliography (.bib) resolved "
            "(config, YAML bibliography:, sibling {}.bib) — those citations "
            "cannot be verified.",
            tex_path.name,
            tex_path.stem,
        )

    # Footnote-URL web sources: markdown footnotes carrying a <url> /
    # [text](url) link, matched to an archived web capture — same T1
    # synthesis as LaTeX \footnote{\url{}}.
    citations.extend(
        ingest_footnote_sources(
            tex_path,
            config.effective_local_web_dir(paper_name),
            refs_dir,
            source_paper=tex_path.stem,
        )
    )
    return citations


def _latex_citations(
    paper_name: str,
    tex_path: Path,
    pc: Any,
    config: LintConfig,
    refs_dir: Path,
) -> list[Citation]:
    """``\\bibitem`` / ``.bib`` discovery narrowed to ``\\cite`` keys, plus
    ``\\footnote{\\url{}}`` source synthesis."""
    from sciwrite_lint.footnote_urls import ingest_footnote_sources
    from sciwrite_lint.references.citations import (
        check_local_sources,
        extract_bibitems,
        filter_to_cited,
    )

    citations = extract_bibitems(tex_path, "auto", bib_path=pc.bib)
    n_bib = len(citations)
    aux_path = tex_path.with_suffix(".aux")
    citations = filter_to_cited(
        citations, tex_path, aux_path=aux_path if aux_path.exists() else None
    )
    if n_bib > len(citations):
        logger.info(
            "{}: {}/{} bib entries are cited — skipping verify+fetch for {} uncited",
            paper_name,
            len(citations),
            n_bib,
            n_bib - len(citations),
        )
    check_local_sources(citations, refs_dir)

    # Footnote-URL sources: each \footnote{\url{URL}} whose URL appears in a
    # local_web_dir capture's Source: header becomes a synthetic T1
    # Citation, pre-registered to short-circuit verify + fetch.
    footnote_citations = ingest_footnote_sources(
        tex_path,
        config.effective_local_web_dir(paper_name),
        refs_dir,
        source_paper=tex_path.stem,
    )
    citations.extend(footnote_citations)
    return citations
