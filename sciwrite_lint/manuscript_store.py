"""Manuscript context: parse, strip, chunk, embed a manuscript.

Provides a cached ManuscriptContext that all rules share. Built from either
LaTeX source (.tex) or a GROBID-parsed PDF (GrobidResult). The first call
to get_or_create_manuscript_context() does the work; subsequent calls return
the cached result.

Embedding is optional (requires sentence-transformers). Without it, LLM
rules send full sections to vLLM instead.
"""

from __future__ import annotations

import re
from pydantic import BaseModel, Field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from sciwrite_lint.config import LintConfig
from sciwrite_lint.eval_claims import Section

if TYPE_CHECKING:
    from sciwrite_lint.models import Finding


# ---------------------------------------------------------------------------
# LaTeX → plain text for embedding and LLM consumption
# ---------------------------------------------------------------------------


def strip_latex_for_embedding(text: str) -> str:
    """Convert raw LaTeX body text to clean prose for embedding/LLM.

    More aggressive than eval_claims._clean_latex — strips float environments,
    math, cross-references, and all LaTeX commands.
    """
    # Strip float environments (figure, table, equation, align, etc.)
    for env in [
        "figure",
        "figure*",
        "table",
        "table*",
        "equation",
        "equation*",
        "align",
        "align*",
        "tikzpicture",
        "lstlisting",
    ]:
        text = re.sub(
            rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}",
            "",
            text,
            flags=re.DOTALL,
        )

    # Strip display math \[...\] and $$...$$
    text = re.sub(r"\\\[.*?\\\]", " [MATH] ", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", " [MATH] ", text, flags=re.DOTALL)

    # Strip inline math $...$
    text = re.sub(r"\$[^$]+\$", " [MATH] ", text)

    # Strip cross-references entirely
    for cmd in ["label", "ref", "eqref", "pageref"]:
        text = re.sub(rf"\\{cmd}\{{[^}}]*\}}", "", text)

    # Replace citations with marker
    text = re.sub(r"\\cite[tp]?(?:yearpar)?\{[^}]+\}", "[CITE]", text)

    # Strip footnotes
    text = re.sub(r"\\footnote\{[^}]*\}", "", text)

    # Unwrap formatting commands
    text = re.sub(r"\\textbf\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\texttt\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\underline\{([^}]+)\}", r"\1", text)

    # Unwrap any remaining \cmd{arg} → arg
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)

    # Strip bare commands (\item, \\, \noindent, etc.)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # Clean up braces, tildes, special chars
    text = re.sub(r"[{}~]", " ", text)
    text = re.sub(r"\\\\", " ", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def strip_latex_for_review(text: str) -> str:
    """Strip LaTeX formatting for LLM review — preserves tables and math.

    Lighter than :func:`strip_latex_for_embedding`: keeps table and equation
    environments intact so the LLM can cross-check numbers, formulas, and
    table values against running text.
    """
    # Strip figure environments (content requires vision model)
    for env in ["figure", "figure*", "tikzpicture", "lstlisting"]:
        text = re.sub(
            rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}",
            "",
            text,
            flags=re.DOTALL,
        )

    # Citations → [CITE]
    text = re.sub(r"\\cite[tp]?(?:yearpar)?\{[^}]+\}", "[CITE]", text)

    # Strip cross-references
    for cmd in ["label", "ref", "eqref", "pageref"]:
        text = re.sub(rf"\\{cmd}\{{[^}}]*\}}", "", text)

    # Strip footnotes
    text = re.sub(r"\\footnote\{[^}]*\}", "", text)

    # Unwrap formatting: \textbf{X} → X, \caption{X} → X
    for cmd in ["textbf", "emph", "textit", "texttt", "underline", "caption"]:
        text = re.sub(rf"\\{cmd}\{{([^}}]+)\}}", r"\1", text)

    # Unwrap remaining \cmd{arg} → arg (preserve \begin/\end)
    text = re.sub(r"\\(?!begin\b|end\b)[a-zA-Z]+\*?\{([^}]*)\}", r"\1", text)

    # Strip bare commands (\hline, \centering, etc.) but not \begin/\end
    text = re.sub(r"\\(?!begin\b|end\b)[a-zA-Z]+\*?", "", text)

    # Strip environment markers: \begin{X} and \end{X}
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}", "", text)

    # LaTeX line breaks
    text = re.sub(r"\\\\", " ", text)

    # Clean braces and tildes
    text = re.sub(r"[{}~]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# ManuscriptContext — shared across all LLM rules for one paper
# ---------------------------------------------------------------------------


class ParagraphEntry(BaseModel):
    """One prose paragraph with its source line number.

    ``text`` is already clean — LaTeX-stripped via pandoc for .tex input,
    passed through as-is for PDF/markdown input. ``line`` is 1-indexed
    against the original source file when known, ``None`` for PDF
    paragraphs where line numbers don't exist.
    """

    line: int | None = None
    text: str


class ManuscriptSection(BaseModel):
    """A section of the manuscript with both raw and clean text.

    ``paragraphs`` is the canonical paragraph-level view used by
    paragraph-oriented checks (e.g. prose-quality). For LaTeX input it
    is populated by the single batched pandoc pass in
    ``_build_context_latex``; ``clean_text`` is the blank-line-joined
    view of the same paragraphs for section-level consumers.
    """

    title: str
    raw_text: str  # original LaTeX (or plain text for PDF)
    clean_text: str  # stripped for LLM / embedding (joined paragraphs)
    paragraphs: list[ParagraphEntry] = Field(default_factory=list)
    start_line: int
    depth: int


class InlineCitation(BaseModel):
    """A citation occurrence in the manuscript text."""

    key: str  # symbolic key (LaTeX) or generated key (PDF)
    line: int | None  # line number (LaTeX) or None (PDF)
    context: str  # surrounding sentence or paragraph
    section: str = ""  # section title (from GROBID <head> or LaTeX \section)


class ParsedReference(BaseModel):
    """A bibliography entry parsed from the manuscript."""

    key: str  # bib key (LaTeX) or generated key (PDF)
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    isbn: str = ""
    lccn: str = ""
    raw: str = ""


class ManuscriptContext(BaseModel):
    """Parsed manuscript ready for rule consumption.

    Can be built from LaTeX (.tex) via ``from_latex()``, from a
    GROBID-parsed PDF via ``from_grobid()``, or from stored GROBID
    markdown via ``from_markdown()``. All checks consume this
    interface — they never touch raw .tex or GrobidResult directly.

    ``build_warnings`` carries non-fatal failures encountered while
    building the context (e.g. cite-context attach raised). The
    pipeline collects these and routes them to ``system_issues`` via
    ``persist_inline_citations`` so the user sees a system-issue
    Finding instead of a silent log line.
    """

    source_path: Path
    source_type: Literal["latex", "pdf", "markdown"] = "latex"
    sections: list[ManuscriptSection] = Field(default_factory=list)
    abstract: str = ""  # clean text
    abstract_raw: str = ""  # raw LaTeX or raw GROBID text
    bibliography_raw: str = ""  # raw LaTeX (empty for PDF)
    inline_citations: list[InlineCitation] = Field(default_factory=list)
    parsed_references: list[ParsedReference] = Field(default_factory=list)
    embeddings_available: bool = False
    build_warnings: list[str] = Field(default_factory=list)

    # ``tex_path`` reads more naturally than ``source_path`` in LaTeX-only
    # code paths; both are the same Path. Use either name based on which
    # reads clearer at the call site.
    @property
    def tex_path(self) -> Path:
        return self.source_path

    def get_section_by_title(self, *keywords: str) -> list[ManuscriptSection]:
        """Find sections whose title contains any of the keywords (case-insensitive)."""
        result = []
        for sec in self.sections:
            title_lower = sec.title.lower()
            if any(kw in title_lower for kw in keywords):
                result.append(sec)
        return result

    def as_eval_sections(self) -> list[Section]:
        """Convert to eval_claims.Section list for embedding retrieval."""
        return [
            Section(title=s.title, text=s.clean_text, index=i)
            for i, s in enumerate(self.sections)
        ]

    @classmethod
    def from_latex(
        cls,
        tex_path: Path,
        config: LintConfig | None = None,
    ) -> ManuscriptContext:
        """Build ManuscriptContext from a LaTeX .tex file."""
        return _build_context_latex(tex_path, config or LintConfig())

    @classmethod
    def from_grobid(
        cls,
        pdf_path: Path,
        grobid_result: Any,  # GrobidResult — Any to avoid circular import
    ) -> ManuscriptContext:
        """Build ManuscriptContext from a GROBID-parsed PDF."""
        return _build_context_grobid(pdf_path, grobid_result)

    @classmethod
    def from_markdown(
        cls,
        md_path: Path,
        ref_key: str = "",
    ) -> ManuscriptContext:
        """Build ManuscriptContext from stored GROBID markdown.

        Used to run consistency checks on cited papers at depth 1.
        The markdown was previously produced by GROBID and stored as
        ``references/{paper}/parsed/{key}.md``.
        """
        return _build_context_markdown(md_path, ref_key)


# ---------------------------------------------------------------------------
# Cached factory — singleton per tex_path per process
# ---------------------------------------------------------------------------

_cache: dict[str, ManuscriptContext] = {}


def set_manuscript_context(path: Path, ctx: ManuscriptContext) -> None:
    """Pre-set a ManuscriptContext in the cache (used for PDF input)."""
    _cache[str(path.resolve())] = ctx


def get_or_create_manuscript_context(
    tex_path: Path,
    config: LintConfig | None = None,
) -> ManuscriptContext:
    """Get cached ManuscriptContext, creating it on first call.

    Thread-safe for single-threaded rule execution (the normal case).
    For PDF input, call set_manuscript_context() first.
    """
    key = str(tex_path.resolve())
    if key in _cache:
        return _cache[key]

    ctx = _build_context_latex(tex_path, config or LintConfig())
    _cache[key] = ctx
    return ctx


def clear_cache() -> None:
    """Clear the manuscript context cache (for testing)."""
    _cache.clear()


def manuscript_ref_key(source_path: Path) -> str:
    """Stable ref_key for the manuscript itself.

    Used as the chunks/chunk_embeddings ``ref_key`` so the manuscript
    sits in the same retrieval store as parsed references but never
    collides with a real bib key (the ``_manuscript_`` prefix is
    reserved). Both the embedding pipeline and consistency-pair
    retrieval use this helper — keep them in sync.
    """
    return f"_manuscript_{source_path.stem}"


def write_manuscript_markdown(
    ctx: ManuscriptContext, references_dir: Path
) -> Path | None:
    """Serialize ``ctx`` to ``parsed/_manuscript_{stem}.md`` for embedding.

    The embedding subprocess resolves keys to ``parsed/{key}.md`` and
    runs ``compute_and_store_embeddings`` on the file. We feed the
    manuscript through the same path: write the abstract + every section
    as markdown headings, return the file path.

    Returns the written path, or ``None`` if the context has no usable
    text. The file is overwritten on each call — ``has_embeddings``
    keys off ``ref_status`` (the embedding stage owns idempotency).
    """
    parts: list[str] = []
    if ctx.abstract.strip():
        parts.append("# Abstract\n\n" + ctx.abstract.strip())
    for sec in ctx.sections:
        body = (sec.clean_text or "").strip()
        if not body:
            continue
        title = (sec.title or "Untitled").strip()
        parts.append(f"# {title}\n\n{body}")
    if not parts:
        return None

    out_dir = references_dir / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manuscript_ref_key(ctx.source_path)}.md"
    out_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return out_path


def persist_inline_citations(
    ctx: ManuscriptContext,
    references_dir: Path,
    source_hash: str = "",
) -> "Finding | None":
    """Persist ``ctx.inline_citations`` to ``workspace.db.manuscript_citations``.

    Writes one row per citation occurrence so the embedding subprocess can
    read claim contexts directly from the DB instead of receiving them
    across the process boundary as JSON. The .tex path populates
    ``InlineCitation.context`` during ``_build_context_latex`` (single
    parse, single source of truth via ``_extract_cite_claims_from_body``),
    so this function reads ``ic.context`` directly without re-parsing.
    ``source_hash`` is recorded for traceability — empty string means
    the caller had no hash to record.

    Returns a system-issue ``Finding`` (rule_id ``internal-error``) when
    ``ctx.build_warnings`` is non-empty — non-fatal build failures (e.g.
    cite-context attach raised) are surfaced in the report's
    ``system_issues`` bucket instead of being swallowed. Returns ``None``
    on a clean build.
    """
    from sciwrite_lint.references.workspace_db import (
        ManuscriptCitation,
        get_db,
        replace_manuscript_citations,
    )

    rows = [
        ManuscriptCitation(
            ref_key=ic.key,
            line=ic.line,
            section=ic.section,
            context=ic.context,
        )
        for ic in ctx.inline_citations
    ]

    with get_db(references_dir) as conn:
        replace_manuscript_citations(
            conn,
            source_type=ctx.source_type,
            source_hash=source_hash,
            citations=rows,
        )

    if ctx.build_warnings:
        from sciwrite_lint.models import Finding

        return Finding(
            level="warning",
            rule_id="internal-error",
            message="manuscript-context build produced warnings; "
            "claim retrieval may be degraded",
            file=str(ctx.source_path),
            context="; ".join(ctx.build_warnings)[:200],
        )
    return None


def _build_context_latex(tex_path: Path, config: LintConfig) -> ManuscriptContext:
    """Parse manuscript from LaTeX, clean markup via pandoc, build context.

    Runs exactly one pandoc subprocess per paper: all paragraphs across
    the abstract and every section are batched into a single
    ``paragraphs_to_markdown`` call. Results are distributed back to
    section-level ``paragraphs`` and ``clean_text``. This DRYs the
    conversion — every downstream LLM check sees the same pandoc-cleaned
    text, not a mix of regex-stripped and clean variants.
    """
    from sciwrite_lint.checks._section_utils import (
        analyze_sections_with_text,
        get_abstract_text,
    )
    from sciwrite_lint.latex_to_markdown import paragraphs_to_markdown
    from sciwrite_lint.tex_parser import (
        extract_bibliography,
        find_all_cite_keys,
        split_paragraphs,
        strip_comments,
        strip_non_prose_environments_preserve_lines,
        unwrap_hypertarget,
    )

    # Read + strip_comments exactly once, then pass through to the
    # section/abstract extractors. They each used to re-read the file
    # and re-strip comments — three round trips for the same data.
    # unwrap_hypertarget normalises pandoc's two-line ``\hypertarget{id}{%
    # \section{...}}`` heading envelope to plain LaTeX; without it the
    # section splitter strands the opening brace in the prior section and
    # pandoc rejects the unbalanced paragraph below.
    text = unwrap_hypertarget(strip_comments(tex_path.read_text(encoding="utf-8")))

    raw_sections = analyze_sections_with_text(tex_path, text=text)
    abstract_raw = get_abstract_text(tex_path, text=text)

    # Collect every paragraph from abstract + sections into one flat list
    # so pandoc runs once. Each entry carries an owner tag so we can
    # re-distribute the cleaned output.
    batch_raw: list[str] = []
    # Owner: ("abstract", paragraph_index) or ("section", section_index, abs_line)
    owners: list[tuple] = []

    if abstract_raw:
        for _para_line_rel, para_text in split_paragraphs(abstract_raw):
            batch_raw.append(para_text)
            owners.append(("abstract",))

    section_paragraph_owners: list[tuple[int, int]] = []
    for sec_idx, (info, raw_text) in enumerate(raw_sections):
        stripped = strip_non_prose_environments_preserve_lines(raw_text)
        for para_line_rel, para_text in split_paragraphs(stripped):
            abs_line = info.start_line + para_line_rel - 1
            batch_raw.append(para_text)
            owners.append(("section", sec_idx, abs_line))
            section_paragraph_owners.append((sec_idx, abs_line))

    cleaned: list[str] = paragraphs_to_markdown(batch_raw) if batch_raw else []

    # Build sections with paragraphs + joined clean_text
    section_paragraphs: dict[int, list[ParagraphEntry]] = {
        i: [] for i in range(len(raw_sections))
    }
    abstract_paragraphs: list[str] = []
    for owner, clean in zip(owners, cleaned):
        if owner[0] == "abstract":
            if clean.strip():
                abstract_paragraphs.append(clean)
        else:
            _, sec_idx, abs_line = owner
            if clean.strip():
                section_paragraphs[sec_idx].append(
                    ParagraphEntry(line=abs_line, text=clean)
                )

    sections = []
    for i, (info, raw_text) in enumerate(raw_sections):
        paragraphs = section_paragraphs[i]
        clean_joined = "\n\n".join(p.text for p in paragraphs)
        sections.append(
            ManuscriptSection(
                title=info.title,
                raw_text=raw_text,
                clean_text=clean_joined,
                paragraphs=paragraphs,
                start_line=info.start_line,
                depth=info.depth,
            )
        )

    abstract_clean = "\n\n".join(abstract_paragraphs)

    bibliography_raw = extract_bibliography(text)

    # Attach paragraph context to each inline citation now, while we
    # already have the body text in scope. Downstream consumers
    # (manuscript_citations persist, claim verification) rely on
    # InlineCitation.context being populated, and computing it here
    # eliminates a second .tex parse later in the pipeline. The shared
    # _extract_cite_claims_from_body helper is the single source of
    # truth so this pre-population matches what extract_claim_contexts
    # produces — query_vectors are keyed by sha256(context) and would
    # cache-miss if the two paths diverged.
    from sciwrite_lint.eval_claims import (
        _extract_cite_claims_from_body,
        _slice_tex_body,
    )

    cite_keys = find_all_cite_keys(text)
    build_warnings: list[str] = []
    try:
        body = _slice_tex_body(text, tex_path)
        line_offset = text[: text.find("\\begin{document}")].count("\n")
        ctx_by_keyline = {
            (cl.key, cl.line): cl.context
            for cl in _extract_cite_claims_from_body(body, line_offset=line_offset)
        }
    except Exception as e:
        from loguru import logger as _logger

        msg = f"Cite-context attach failed: {type(e).__name__}: {e}"
        _logger.warning("{} ({}); citations persist with empty contexts", msg, tex_path)
        build_warnings.append(msg)
        ctx_by_keyline = {}

    inline_citations = [
        InlineCitation(
            key=key,
            line=line_no,
            context=ctx_by_keyline.get((key, line_no), ""),
        )
        for line_no, key in cite_keys
    ]

    ctx = ManuscriptContext(
        source_path=tex_path,
        source_type="latex",
        sections=sections,
        abstract=abstract_clean,
        abstract_raw=abstract_raw,
        bibliography_raw=bibliography_raw,
        inline_citations=inline_citations,
        build_warnings=build_warnings,
    )

    # Embeddings are computed separately via `sciwrite-lint parse`.
    # Don't compute on the fly — it takes ~13s to load the model.
    # LLM rules work without embeddings (they send all sections to vLLM).

    return ctx


def _paragraphs_from_clean_text(text: str) -> list[ParagraphEntry]:
    """Split already-clean text into paragraph entries (no line numbers).

    Used by the GROBID and markdown builders where the source has no
    LaTeX markup and no reliable line numbers. Paragraphs are separated
    by blank lines — empty paragraphs are dropped.
    """
    entries: list[ParagraphEntry] = []
    for chunk in re.split(r"\n\s*\n", text):
        stripped = chunk.strip()
        if stripped:
            entries.append(ParagraphEntry(line=None, text=stripped))
    return entries


def _build_context_grobid(pdf_path: Path, grobid_result: Any) -> ManuscriptContext:
    """Build ManuscriptContext from a GROBID-parsed PDF.

    Maps GrobidResult fields to ManuscriptContext:
    - GrobidSection → ManuscriptSection (text is already clean)
    - GrobidReference → ParsedReference
    - TEI inline citations → InlineCitation (via raw_tei parsing)
    """
    from sciwrite_lint.pdf.grobid import GrobidResult

    result: GrobidResult = grobid_result

    # Sections — GROBID text is already clean (no LaTeX markup)
    sections = [
        ManuscriptSection(
            title=sec.title,
            raw_text=sec.text,
            clean_text=sec.text,  # already plain text
            paragraphs=_paragraphs_from_clean_text(sec.text),
            start_line=0,  # no line numbers in PDF
            depth=sec.level,
        )
        for sec in result.sections
    ]

    # References → ParsedReference with generated keys
    parsed_references = []
    key_counts: dict[str, int] = {}
    for ref in result.references:
        key = _generate_cite_key(ref.authors, ref.year, ref.title, key_counts)
        parsed_references.append(
            ParsedReference(
                key=key,
                title=ref.title,
                authors=ref.authors,
                year=ref.year,
                venue=ref.venue,
                doi=ref.doi,
                url=ref.url,
                isbn=ref.isbn,
                lccn=ref.lccn,
                raw=ref.raw,
            )
        )

    # Inline citations from TEI XML — GROBID links <ref> tags to <biblStruct>
    inline_citations = _extract_tei_inline_citations(result.raw_tei, parsed_references)

    return ManuscriptContext(
        source_path=pdf_path,
        source_type="pdf",
        sections=sections,
        abstract=result.abstract,
        abstract_raw=result.abstract,
        inline_citations=inline_citations,
        parsed_references=parsed_references,
    )


def _build_context_markdown(md_path: Path, ref_key: str) -> ManuscriptContext:
    """Build ManuscriptContext from stored GROBID markdown.

    Splits by markdown headings (``# Title`` / ``## Title``), separates
    abstract and bibliography sections, builds ManuscriptSection objects.
    Text is already clean (no LaTeX markup).
    """
    text = md_path.read_text(encoding="utf-8")

    # Split by markdown headings (h1-h3)
    heading_re = re.compile(r"(?m)^(#{1,3}\s+.+)$")
    parts = heading_re.split(text)

    sections: list[ManuscriptSection] = []
    abstract = ""
    bib_started = False
    current_title = "Preamble"
    current_text = ""

    _BIB_TITLES = {"references", "bibliography", "works cited", "literature cited"}

    for part in parts:
        if heading_re.match(part):
            # Flush previous section
            if current_text.strip() and not bib_started:
                sec_lower = current_title.lower()
                if sec_lower in ("abstract",):
                    abstract = current_text.strip()
                sections.append(
                    ManuscriptSection(
                        title=current_title,
                        raw_text=current_text.strip(),
                        clean_text=current_text.strip(),
                        paragraphs=_paragraphs_from_clean_text(current_text),
                        start_line=0,
                        depth=part.count("#") if heading_re.match(part) else 1,
                    )
                )
            current_title = part.strip().lstrip("#").strip()
            current_text = ""
            if current_title.lower() in _BIB_TITLES:
                bib_started = True
        else:
            current_text += part

    # Flush final section
    if current_text.strip() and not bib_started:
        sec_lower = current_title.lower()
        if sec_lower in ("abstract",):
            abstract = current_text.strip()
        sections.append(
            ManuscriptSection(
                title=current_title,
                raw_text=current_text.strip(),
                clean_text=current_text.strip(),
                paragraphs=_paragraphs_from_clean_text(current_text),
                start_line=0,
                depth=1,
            )
        )

    return ManuscriptContext(
        source_path=md_path,
        source_type="markdown",
        sections=sections,
        abstract=abstract,
        abstract_raw=abstract,
    )


def _generate_cite_key(
    authors: list[str],
    year: str,
    title: str,
    seen: dict[str, int],
) -> str:
    """Generate a citation key like 'smith2024' from author+year.

    Appends 'b', 'c', etc. for duplicates.
    """
    if authors:
        # Extract last name from first author
        # GROBID outputs "First Last" or "First M Last" — surname is
        # the last multi-char token (skip single-char initials at the end)
        first_author = authors[0]
        if "," in first_author:
            # "Last, First" format
            surname = first_author.split(",")[0].strip()
        else:
            parts = first_author.split()
            # Find last part that's more than one letter (skip initials)
            surname = "unknown"
            for part in reversed(parts):
                clean = re.sub(r"[^a-zA-Z]", "", part)
                if len(clean) > 1:
                    surname = clean
                    break
            if surname == "unknown" and parts:
                surname = re.sub(r"[^a-zA-Z]", "", parts[0])
        surname = re.sub(r"[^a-zA-Z]", "", surname).lower()
    else:
        # No author surname — use first word of title
        words = re.sub(r"[^a-zA-Z\s]", "", title).split()
        surname = words[0].lower() if words else "unknown"

    base_key = f"{surname}{year[:4]}" if year else surname
    if base_key not in seen:
        seen[base_key] = 1
        return base_key
    count = seen[base_key]
    seen[base_key] = count + 1
    suffix = chr(ord("a") + count)  # b, c, d, ...
    return f"{base_key}{suffix}"


def _extract_tei_inline_citations(
    raw_tei: str,
    parsed_references: list[ParsedReference],
) -> list[InlineCitation]:
    """Extract inline citation links from GROBID TEI XML.

    GROBID annotates inline citations as <ref type="bibr" target="#b5">
    linking to <biblStruct xml:id="b5">. We map these back to our
    generated keys via reference index.
    """
    from defusedxml.ElementTree import ParseError, fromstring as _safe_fromstring

    if not raw_tei:
        return []

    tei_ns = "http://www.tei-c.org/ns/1.0"
    ns = {"tei": tei_ns}

    try:
        root = _safe_fromstring(raw_tei)
    except ParseError:
        return []

    # Build index→key map from parsed_references (index matches GrobidReference.index)
    # GROBID uses xml:id="b0", "b1", ... matching reference index
    idx_to_key = {i: ref.key for i, ref in enumerate(parsed_references)}

    citations: list[InlineCitation] = []
    body = root.find(".//tei:text/tei:body", ns)
    if body is None:
        return []

    # Extract citations per <div> to capture section headings
    div_tag = f"{{{tei_ns}}}div"
    head_tag = f"{{{tei_ns}}}head"
    ref_tag = f"{{{tei_ns}}}ref"
    p_tag = f"{{{tei_ns}}}p"

    for div_el in body.iter(div_tag):
        head_el = div_el.find(head_tag)
        section_title = (head_el.text or "").strip() if head_el is not None else ""

        for ref_el in div_el.iter(ref_tag):
            if ref_el.get("type") != "bibr":
                continue
            target = ref_el.get("target", "")
            if not target.startswith("#b"):
                continue
            try:
                ref_idx = int(target[2:])
            except ValueError:
                continue
            key = idx_to_key.get(ref_idx)
            if key is None:
                continue

            # Get surrounding paragraph as context
            context = ref_el.tail or ""
            for p in div_el.iter(p_tag):
                if ref_el in list(p.iter(ref_tag)):
                    context = " ".join(p.itertext())[:200]
                    break

            citations.append(
                InlineCitation(
                    key=key,
                    line=None,
                    context=context.strip(),
                    section=section_title,
                )
            )

    return citations


def _compute_manuscript_embeddings(
    ctx: ManuscriptContext,
    config: LintConfig,
) -> None:
    """Compute and cache embeddings for manuscript sections."""
    from sciwrite_lint.references.embedding_store import (
        has_embeddings,
        store_embeddings,
    )
    from sciwrite_lint.references.reference_store import (
        _chunk_text,
        _get_embedding_config,
    )

    refs_dir = config.effective_references_dir()
    ms_key = f"_manuscript_{ctx.tex_path.stem}"

    model_name, dim, _ = _get_embedding_config()
    if has_embeddings(ms_key, refs_dir, model_name=model_name):
        return

    all_chunks = []
    for sec in ctx.sections:
        if sec.clean_text.strip():
            all_chunks.extend(_chunk_text(sec.clean_text, section_title=sec.title))

    if not all_chunks:
        return

    model_name, dim, _ = _get_embedding_config()
    chunk_dicts = [
        {
            "text": c.text,
            "section_title": c.section_title,
            "granularity": c.granularity,
            "start_char": c.start_char,
        }
        for c in all_chunks
    ]
    store_embeddings(ms_key, chunk_dicts, refs_dir, model_name, dim)


def retrieve_manuscript_sections(
    query: str,
    ctx: ManuscriptContext,
    config: LintConfig | None = None,
    top_k: int = 5,
) -> list[ManuscriptSection] | None:
    """Find manuscript sections relevant to a query using embeddings.

    Returns None if embeddings unavailable (caller should use all sections).
    """
    if not ctx.embeddings_available:
        return None

    config = config or LintConfig()
    refs_dir = config.effective_references_dir()
    ms_key = f"_manuscript_{ctx.tex_path.stem}"

    from sciwrite_lint.references.reference_store import retrieve_relevant_sections

    eval_sections = ctx.as_eval_sections()
    filtered = retrieve_relevant_sections(
        query,
        ms_key,
        refs_dir,
        eval_sections,
        top_k=top_k,
    )

    if filtered is None:
        return None

    # Map back to ManuscriptSection
    filtered_titles = {s.title for s in filtered}
    return [s for s in ctx.sections if s.title in filtered_titles]
