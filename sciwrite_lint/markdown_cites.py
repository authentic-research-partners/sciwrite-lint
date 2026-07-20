"""Deterministic citation / cross-reference analysis of markdown via pandoc.

Pandoc (already a dependency via ``pypandoc``) parses markdown — including
pandoc-style ``[@key]`` citations and pandoc-crossref ``@fig:`` / ``@sec:``
references — into a JSON AST. Reading that AST is deterministic and
code-fence-safe: pandoc does not emit ``Cite`` nodes for text inside
fenced or inline code, so ``data[@idx]`` in a code block is never
mistaken for a citation.

Pandoc represents BOTH bibliography citations (``[@smith2020]``) and
pandoc-crossref references (``[@fig:flow]``, ``@sec:methods``) as ``Cite``
nodes. They are distinguished by key prefix: a key beginning with one of
:data:`_CROSSREF_PREFIXES` is a cross-reference (its target is a figure /
section / table / equation / listing label), everything else is a
bibliography citation. :func:`analyze_markdown` does this split once and
also collects the labels actually defined in the document (block-level
ids on figures, headings, tables, …), so cross-references can be matched
against real targets.

Numeric ``[1]`` citations are *not* pandoc citations and produce no
``Cite`` nodes — :func:`detect_citation_style` reports that case so the
caller can warn rather than silently check nothing.

Numeric citations are an intentional non-goal, not a TODO. A ``[1]``
label carries no stable identifier: inserting one citation renumbers the
rest, which would churn the key-based caches (``workspace.db``,
``claim_results``, dedup all key off the citation key). The numbered
reference list is also free text rather than structured ``.bib`` entries,
so metadata would have to be recovered by heuristics — strictly less
reliable than the exact data ``[@key]`` + ``.bib`` already provides. The
supported path is pandoc ``[@key]`` with a bibliography; numeric-only
documents are reported, not half-parsed.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pypandoc
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from sciwrite_lint.models import Citation, Field

# pandoc-crossref reference kinds. A Cite key with one of these prefixes is
# a cross-reference to a labelled element, not a bibliography citation.
_CROSSREF_PREFIXES = ("fig:", "sec:", "tbl:", "eq:", "lst:")

# Numeric bracket citations, e.g. ``[1]``, ``[2, 3]``. Used only to
# recognise (not parse) the numeric convention so the caller can report
# it as unsupported.
_NUMERIC_CITE_RE = re.compile(r"\[\d+(?:[,;\s]+\d+)*\]")


class MarkdownCitation(BaseModel):
    """One bibliography ``[@key]`` occurrence with the prose around it."""

    key: str
    context: str = ""  # text of the enclosing block (paragraph / heading)
    section: str = ""  # most recent heading text above the occurrence


class CrossRef(BaseModel):
    """One pandoc-crossref reference occurrence (``@fig:x``, ``@sec:y`` …)."""

    target: str  # the referenced label, e.g. "fig:flow"
    kind: str  # "fig" | "sec" | "tbl" | "eq" | "lst"
    context: str = ""
    section: str = ""


class MarkdownAnalysis(BaseModel):
    """Citations, cross-references, and defined labels of one document.

    Immutable so it is safe to memoize and share across the checks that
    read it within a run.
    """

    model_config = ConfigDict(frozen=True)

    citations: tuple[MarkdownCitation, ...] = ()
    crossrefs: tuple[CrossRef, ...] = ()
    labels: frozenset[str] = frozenset()  # every defined block id
    figure_labels: frozenset[str] = frozenset()  # Figure-block ids only
    bibliography: tuple[str, ...] = ()  # YAML `bibliography:` field, as written


# Inline node types whose visible content is a nested inline list at c[1].
_PREFIXED_INLINE_TEXT = frozenset({"Quoted", "Cite", "Link", "Image"})
# Inline node types whose content IS a bare nested inline list.
_NESTED_INLINE = frozenset(
    {
        "Emph",
        "Strong",
        "Underline",
        "Strikeout",
        "Superscript",
        "Subscript",
        "SmallCaps",
    }
)


def _inline_text(node: Any) -> str:
    """Render a pandoc inline node (or list of them) to plain text."""
    if isinstance(node, list):
        return "".join(_inline_text(n) for n in node)
    if not isinstance(node, dict):
        return ""
    t = node.get("t")
    c = node.get("c")
    if t == "Str":
        return c if isinstance(c, str) else ""
    if t in ("Space", "SoftBreak", "LineBreak"):
        return " "
    if t in _NESTED_INLINE:
        return _inline_text(c)
    if t in _PREFIXED_INLINE_TEXT:
        return _inline_text(c[1]) if isinstance(c, list) and len(c) > 1 else ""
    if t in ("Code", "Math", "RawInline"):
        return c[-1] if isinstance(c, list) and c and isinstance(c[-1], str) else ""
    if t == "Note":
        return ""  # footnote body — not part of the surrounding sentence
    return _inline_text(c) if isinstance(c, (list, dict)) else ""


class _Acc(BaseModel):
    """Mutable accumulator used during the AST walk."""

    citations: list[MarkdownCitation] = Field(default_factory=list)
    crossrefs: list[CrossRef] = Field(default_factory=list)
    labels: set[str] = Field(default_factory=set)
    figure_labels: set[str] = Field(default_factory=set)


def _classify_cites(inlines: Any, context: str, section: str, acc: _Acc) -> None:
    """Find every ``Cite`` key under ``inlines`` and route it by prefix."""

    def rec(node: Any) -> None:
        if isinstance(node, list):
            for n in node:
                rec(n)
        elif isinstance(node, dict):
            if node.get("t") == "Cite":
                cites = node["c"][0] if isinstance(node.get("c"), list) else []
                for cd in cites:
                    key = cd.get("citationId", "")
                    if not key:
                        continue
                    prefix = next(
                        (p for p in _CROSSREF_PREFIXES if key.startswith(p)), None
                    )
                    if prefix:
                        acc.crossrefs.append(
                            CrossRef(
                                target=key,
                                kind=prefix.rstrip(":"),
                                context=context,
                                section=section,
                            )
                        )
                    else:
                        acc.citations.append(
                            MarkdownCitation(key=key, context=context, section=section)
                        )
            else:
                rec(node.get("c"))

    rec(inlines)


def _block_id(c: Any, idx: int) -> str:
    """Return the id from a block's attr (``[id, classes, kvs]``) at ``c[idx]``."""
    if isinstance(c, list) and len(c) > idx:
        attr = c[idx]
        if isinstance(attr, list) and attr and isinstance(attr[0], str):
            return attr[0]
    return ""


def _walk(blocks: list[Any], section: str, acc: _Acc) -> str:
    """Walk pandoc blocks in document order, filling ``acc``."""
    for block in blocks:
        if not isinstance(block, dict):
            continue
        t = block.get("t")
        c = block.get("c")
        if t == "Header":
            label = _block_id(c, 1)  # c = [level, attr, inlines]
            if label:
                acc.labels.add(label)
            section = _inline_text(c[2]).strip() if isinstance(c, list) else section
            _classify_cites(c[2] if isinstance(c, list) else [], section, section, acc)
        elif t in ("Para", "Plain"):
            text = _inline_text(c).strip()
            _classify_cites(c, text, section, acc)
        elif t == "Figure":
            label = _block_id(c, 0)  # c = [attr, caption, blocks]
            if label:
                acc.labels.add(label)
                acc.figure_labels.add(label)
            # Recurse into the caption + body for any citations.
            if isinstance(c, list) and len(c) > 2:
                _classify_cites(c[1], section, section, acc)
                _walk(c[2], section, acc)
        elif t == "Table":
            label = _block_id(c, 0)
            if label:
                acc.labels.add(label)
        elif t == "CodeBlock":
            label = _block_id(c, 0)
            if label:
                acc.labels.add(label)
        elif t == "Div":
            label = _block_id(c, 0)  # c = [attr, blocks]
            if label:
                acc.labels.add(label)
            if isinstance(c, list) and len(c) > 1:
                section = _walk(c[1], section, acc)
        elif t == "BlockQuote" and isinstance(c, list):
            section = _walk(c, section, acc)
        elif t == "BulletList" and isinstance(c, list):
            for item in c:
                if isinstance(item, list):
                    section = _walk(item, section, acc)
        elif t == "OrderedList" and isinstance(c, list) and len(c) > 1:
            for item in c[1]:
                if isinstance(item, list):
                    section = _walk(item, section, acc)
    return section


def _meta_strings(node: Any) -> list[str]:
    """Collect plain strings from a pandoc metadata value.

    Handles the ``bibliography:`` field in either form — a single value
    (``MetaInlines`` / ``MetaString``) or a list (``MetaList``).
    """
    if isinstance(node, list):
        out: list[str] = []
        for n in node:
            out.extend(_meta_strings(n))
        return out
    if not isinstance(node, dict):
        return []
    t = node.get("t")
    c = node.get("c")
    if t == "MetaString":
        return [c] if isinstance(c, str) else []
    if t == "MetaInlines":
        text = _inline_text(c).strip()
        return [text] if text else []
    if t == "MetaList":
        return _meta_strings(c)
    return []


@lru_cache(maxsize=16)
def analyze_markdown(md_text: str) -> MarkdownAnalysis:
    """Analyse markdown into citations, cross-references, and labels.

    One pandoc subprocess (markdown → JSON AST). Memoized by content so
    the checks that read it within a run share a single parse.
    """
    ast = json.loads(pypandoc.convert_text(md_text, to="json", format="markdown"))
    acc = _Acc()
    _walk(ast.get("blocks", []), "", acc)
    bibliography = _meta_strings(ast.get("meta", {}).get("bibliography"))
    return MarkdownAnalysis(
        citations=tuple(acc.citations),
        crossrefs=tuple(acc.crossrefs),
        labels=frozenset(acc.labels),
        figure_labels=frozenset(acc.figure_labels),
        bibliography=tuple(bibliography),
    )


def resolve_bib_paths(
    md_path: Path,
    analysis: MarkdownAnalysis,
    explicit_bib: Path | None = None,
) -> list[Path]:
    """Resolve the bibliography file(s) for a markdown manuscript.

    Priority: an explicitly configured ``.bib`` > the YAML
    ``bibliography:`` field (pandoc's canonical mechanism; resolved
    relative to the manuscript) > the sibling ``{stem}.bib`` convention.
    Only existing paths are returned.
    """
    if explicit_bib is not None:
        return [explicit_bib] if explicit_bib.exists() else []
    if analysis.bibliography:
        resolved = [(md_path.parent / b) for b in analysis.bibliography]
        return [p for p in resolved if p.exists()]
    sibling = md_path.with_suffix(".bib")
    return [sibling] if sibling.exists() else []


def parse_markdown_bib(
    md_path: Path,
    analysis: MarkdownAnalysis,
    explicit_bib: Path | None = None,
) -> list["Citation"]:
    """Parse and merge the markdown manuscript's resolved bibliography.

    Resolves the ``.bib`` file(s) via :func:`resolve_bib_paths` and returns
    the combined ``Citation`` list (empty when no bibliography resolves).
    The single place markdown citation entries are produced from a bib.
    """
    from sciwrite_lint.references.citations import parse_bib_file

    paths = resolve_bib_paths(md_path, analysis, explicit_bib)
    return [c for bib in paths for c in parse_bib_file(bib, source_paper=md_path.stem)]


def extract_markdown_citations(md_text: str) -> list[MarkdownCitation]:
    """Bibliography ``[@key]`` citations (cross-references excluded)."""
    return list(analyze_markdown(md_text).citations)


def detect_citation_style(md_text: str, citations: list[MarkdownCitation]) -> str:
    """Classify the citation convention of a markdown manuscript.

    - ``"pandoc"``  — pandoc ``[@key]`` bibliography citations are present.
    - ``"numeric"`` — no pandoc citations, but numeric ``[1]`` brackets
      appear (recognised but not extracted).
    - ``"none"``    — no recognisable citation markers at all.
    """
    if citations:
        return "pandoc"
    if _NUMERIC_CITE_RE.search(md_text):
        return "numeric"
    return "none"
