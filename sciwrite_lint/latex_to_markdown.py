"""LaTeX → markdown conversion for LLM consumption.

Uses pandoc (via pypandoc) to convert LaTeX fragments to clean markdown
text, then post-processes citation / reference / image syntax that pandoc
emits but the prose-quality LLM should not see verbatim. Batches many
paragraphs into a single pandoc invocation via sentinel markers, so one
paper → one subprocess, not one per paragraph.

Citations become ``[CITE]``, cross-references become ``[REF]``, images
are dropped, and stranded punctuation left behind by removed references
(``"Section ."``, ``"(Section )"``) is cleaned up. The output is what
reviewers of scientific prose actually see — no LaTeX macro syntax for
the LLM to mistake for a grammar error.
"""

from __future__ import annotations

import re

import pypandoc

# Sentinel: a LaTeX \paragraph*{} with a high-entropy token. Pandoc emits
# this as a level-4 heading, which we split on after conversion. The
# token is unlikely to collide with real manuscript content.
_MARKER_TEMPLATE = "SCIWRITE_PBREAK_%d_X7Q"
_MARKER_SPLIT_RE = re.compile(
    r"#{1,6}\s+SCIWRITE_PBREAK_\d+_X7Q(?:\s*\{[^}]*\})?\s*",
    re.MULTILINE,
)

# Pandoc citation forms: [@key], [-@key], [@key; @key2, p. 10]
_CITE_RE = re.compile(r"\[[-@]?@[^\]]+\]")

# Pandoc cross-reference link forms:
#   [\[ref\]](#ref){reference-type="ref" reference="ref"}
#   [ref](#ref)
_REFLINK_RE = re.compile(r"\[(?:\\?\[)?[^\]]*?(?:\\?\])?\]\(#[^)]+\)(?:\{[^}]*\})?")

# Image / figure markdown: ![caption](path){attrs}
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)(?:\{[^}]*\})?")

# Heading attribute suffixes like `{#sec:intro .unnumbered}` — drop.
_HEADING_ATTR_RE = re.compile(r"\s*\{[#.][^}]*\}\s*$", re.MULTILINE)

# Whitespace before punctuation — left behind when a \ref{} was removed
# and its trailing period/comma/paren stayed.
_ORPHAN_PUNCT_RE = re.compile(r"[ \t\xa0]+(?=[.,:;!?)])")

# Empty parentheses or parens containing only whitespace — often from
# "(Section~\ref{})" when the ref was stripped.
_EMPTY_PAREN_RE = re.compile(r"\([ \t\xa0]*\)")
# Dangling open-paren-with-space: "( foo)" → "(foo)"
_OPEN_PAREN_SPACE_RE = re.compile(r"\(\s+")

# Markdown heading prefixes remaining in body — strip leading "### ".
_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Non-breaking space (pandoc emits \xa0 for LaTeX ~).
_NBSP_RE = re.compile(r"\xa0")


def paragraphs_to_markdown(latex_paragraphs: list[str]) -> list[str]:
    """Convert a list of LaTeX paragraph strings to cleaned markdown.

    Returns a list of the same length; index i in the output is the
    cleaned markdown for paragraph i of the input. Empty strings survive
    as empty strings in the output so callers can preserve line-number
    mapping by list index.

    Raises if pandoc is unreachable or emits a malformed split — this is
    a pipeline primitive, not a best-effort helper.
    """
    if not latex_paragraphs:
        return []

    if len(latex_paragraphs) == 1:
        md = pypandoc.convert_text(
            latex_paragraphs[0],
            to="markdown",
            format="latex",
            extra_args=["--wrap=none"],
        )
        return [_clean_markdown(md)]

    parts: list[str] = []
    for i, para in enumerate(latex_paragraphs):
        parts.append(para)
        if i < len(latex_paragraphs) - 1:
            parts.append(f"\\paragraph*{{{_MARKER_TEMPLATE % i}}}")
    joined = "\n\n".join(parts)

    md = pypandoc.convert_text(
        joined,
        to="markdown",
        format="latex",
        extra_args=["--wrap=none"],
    )

    chunks = _MARKER_SPLIT_RE.split(md)
    if len(chunks) != len(latex_paragraphs):
        raise RuntimeError(
            f"pandoc paragraph split produced {len(chunks)} chunks, "
            f"expected {len(latex_paragraphs)} — markers were eaten or "
            "collided with manuscript content"
        )

    return [_clean_markdown(c) for c in chunks]


def _clean_markdown(text: str) -> str:
    """Normalise pandoc markdown for LLM prose review.

    Replaces citation and reference syntax with ``[CITE]`` / ``[REF]``
    placeholders (same as the legacy regex cleaner so the LLM prompt
    stays stable), drops images, fixes stranded punctuation, and strips
    heading attributes + blank-line accumulation.
    """
    text = _NBSP_RE.sub(" ", text)
    text = _IMAGE_RE.sub("", text)
    text = _CITE_RE.sub("[CITE]", text)
    text = _REFLINK_RE.sub("[REF]", text)
    text = _HEADING_ATTR_RE.sub("", text)
    text = _HEADING_PREFIX_RE.sub("", text)
    # Strip remaining markdown emphasis markers so they don't appear in
    # error spans — the LLM shouldn't have to learn markdown.
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    # Punctuation cleanup must run after ref/image stripping so the
    # whitespace left behind by those substitutions is collapsed.
    text = _EMPTY_PAREN_RE.sub("", text)
    text = _ORPHAN_PUNCT_RE.sub("", text)
    text = _OPEN_PAREN_SPACE_RE.sub("(", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
