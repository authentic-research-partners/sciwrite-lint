"""Synthetic error injection for detection-rate evaluation.

Injects known errors into clean papers and records ground truth
so detection rate can be measured.

Two modes:
- **LaTeX**: inject into raw .tex text (\\cite{}, \\ref{}, abstract numbers).
- **PDF**: inject into ManuscriptContext dataclasses built from GROBID output.
  Checks already have PDF-native code paths that consume ManuscriptContext.

Injection types (mapped to linter checks):
- Fake citations → dangling-cite
- Broken refs → dangling-ref
- Number drift between abstract and body → cross-section-consistency
"""

from __future__ import annotations

import copy
import random
import re
from pathlib import Path

from pydantic import BaseModel, Field

from sciwrite_lint.manuscript_store import ManuscriptContext


class Injection(BaseModel):
    """One injected error with ground truth."""

    rule_id: str
    description: str
    line: int | None = None
    context: str = ""


class InjectedPaper(BaseModel):
    """A paper with injected errors."""

    original_path: Path
    injected_text: str = ""
    injected_ctx: ManuscriptContext | None = None
    injections: list[Injection] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual injectors
# ---------------------------------------------------------------------------


def _inject_fake_citations(
    text: str, rng: random.Random
) -> tuple[str, list[Injection]]:
    """Insert fake \\cite{} commands with non-existent keys.

    Only injects when the paper has inline \\bibitem entries, since
    dangling-cite requires a parseable bibliography to detect orphans.
    """
    # Skip if no inline bibliography (external .bib may not be in the tarball)
    if "\\bibitem" not in text:
        return text, []

    fake_keys = [
        f"fakename{rng.randint(2020, 2026)}{suffix}"
        for suffix in ["deep", "novel", "framework", "analysis", "survey"]
    ]
    injections: list[Injection] = []

    # Find sentences in the body that don't already end with a citation
    sentences = list(
        re.finditer(
            r"([A-Z][^.!?]{30,120})\.",
            text,
        )
    )
    if not sentences:
        return text, []

    # Pick up to 3 locations
    targets = rng.sample(sentences, min(3, len(sentences)))
    offset = 0
    for match in sorted(targets, key=lambda m: m.start()):
        key = fake_keys.pop() if fake_keys else f"fake{rng.randint(1000, 9999)}"
        insert_pos = match.end() + offset
        citation = f" \\cite{{{key}}}"
        text = text[:insert_pos] + citation + text[insert_pos:]
        offset += len(citation)

        injections.append(
            Injection(
                rule_id="dangling-cite",
                description=f"Injected orphan citation: \\cite{{{key}}}",
                context=key,
            )
        )

    return text, injections


def _inject_broken_refs(text: str, rng: random.Random) -> tuple[str, list[Injection]]:
    """Insert \\ref{} to non-existent labels."""
    fake_labels = [
        f"fig:phantom{rng.randint(10, 99)}",
        f"tab:ghost{rng.randint(10, 99)}",
    ]
    injections: list[Injection] = []

    # Find places where "Figure" or "Table" appear (or similar)
    spots = list(re.finditer(r"(?:Figure|Table|Section)\s*\\ref\{", text))
    if not spots:
        # Insert near an existing \ref
        spots = list(re.finditer(r"\\ref\{", text))

    if not spots:
        # Insert after a sentence
        sentences = list(re.finditer(r"\.\s+[A-Z]", text))
        if sentences:
            pos = rng.choice(sentences).start() + 2
            label = fake_labels[0]
            insert = f"As shown in Figure~\\ref{{{label}}}, "
            text = text[:pos] + insert + text[pos:]
            injections.append(
                Injection(
                    rule_id="dangling-ref",
                    description=f"Injected broken ref: \\ref{{{label}}}",
                    context=label,
                )
            )
        return text, injections

    # Replace one existing \ref target with a fake label
    target = rng.choice(spots)
    # Find the closing brace
    brace_start = text.index("{", target.start())
    brace_end = text.index("}", brace_start)
    original_label = text[brace_start + 1 : brace_end]
    fake_label = fake_labels[0]

    # Only replace one occurrence — keep the original label elsewhere
    text = text[: brace_start + 1] + fake_label + text[brace_end:]

    injections.append(
        Injection(
            rule_id="dangling-ref",
            description=f"Injected broken ref: \\ref{{{fake_label}}} (was {original_label})",
            context=fake_label,
        )
    )

    return text, injections


def _inject_percentage_error(
    text: str, rng: random.Random
) -> tuple[str, list[Injection]]:
    """Alter a percentage so numbers don't sum to 100%."""
    # Find patterns like "X\%" in the body
    pct_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*\\?%", text))
    if len(pct_matches) < 2:
        return text, []

    # Pick one and alter it
    target = rng.choice(pct_matches)
    original = target.group(1)
    try:
        val = float(original)
    except ValueError:
        return text, []

    # Add 5-15% to the value
    delta = rng.uniform(5, 15)
    new_val = f"{val + delta:.1f}"

    text = text[: target.start(1)] + new_val + text[target.end(1) :]

    return text, [
        Injection(
            rule_id="percentage-error",
            description=f"Injected percentage error: {original}% → {new_val}%",
            context=f"{original} -> {new_val}",
        )
    ]


def _inject_number_drift(text: str, rng: random.Random) -> tuple[str, list[Injection]]:
    """Change a number in the abstract so it contradicts the body.

    Looks for numbers in the abstract and alters them.
    """
    # Find abstract
    abs_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        text,
        re.DOTALL,
    )
    if not abs_match:
        return text, []

    abstract = abs_match.group(1)
    # Find numbers with units or % in abstract
    num_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(?:\\?%|points?|pp)", abstract))
    if not num_matches:
        return text, []

    target = rng.choice(num_matches)
    original = target.group(1)
    try:
        val = float(original)
    except ValueError:
        return text, []

    # Alter by 20-40%
    factor = rng.choice([0.6, 0.7, 1.3, 1.4])
    new_val = f"{val * factor:.1f}"

    # Replace only in the abstract
    abs_start = abs_match.start(1)
    global_pos = abs_start + target.start(1)
    text = text[:global_pos] + new_val + text[global_pos + len(original) :]

    return text, [
        Injection(
            rule_id="cross-section-consistency",
            description=f"Injected number drift in abstract: {original} → {new_val}",
            context=f"{original} -> {new_val}",
        )
    ]


# ---------------------------------------------------------------------------
# Main injection pipeline
# ---------------------------------------------------------------------------

# Injectors for manuscript-engine checks (runnable without LLM)
TEXT_INJECTORS = [
    _inject_fake_citations,  # dangling-cite (manuscript engine)
    _inject_broken_refs,  # dangling-ref (manuscript engine)
]

# Injectors that require local-llm-engine checks
LLM_INJECTORS = [
    _inject_number_drift,  # cross-section-consistency (local-llm engine)
]

# _inject_percentage_error exists but has no matching check — kept as a
# standalone function for future use, not included in active injectors.

ALL_INJECTORS = TEXT_INJECTORS + LLM_INJECTORS


def inject_errors(
    tex_path: Path,
    seed: int | None = None,
    injectors: list | None = None,
    text_only: bool = True,
) -> InjectedPaper:
    """Inject synthetic errors into a LaTeX paper.

    Args:
        tex_path: Path to original .tex file.
        seed: Random seed for reproducibility.
        injectors: Which injectors to run (default: TEXT_INJECTORS).
        text_only: If True (default), only inject errors detectable by
            manuscript-engine rules. Set False to include local-llm-engine injections.

    Returns:
        InjectedPaper with modified text and ground truth.
    """
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    rng = random.Random(seed)
    if injectors is not None:
        fns = injectors
    else:
        fns = TEXT_INJECTORS if text_only else ALL_INJECTORS

    result = InjectedPaper(original_path=tex_path)

    for fn in fns:
        text, new_injections = fn(text, rng)
        result.injections.extend(new_injections)

    result.injected_text = text
    return result


# ---------------------------------------------------------------------------
# PDF injectors — operate on ManuscriptContext dataclasses
# ---------------------------------------------------------------------------


def _inject_fake_citations_pdf(
    ctx: ManuscriptContext,
    rng: random.Random,
) -> list[Injection]:
    """Add InlineCitations with fake keys not in parsed_references."""
    from sciwrite_lint.manuscript_store import InlineCitation

    if not ctx.parsed_references:
        return []

    ref_keys = {r.key for r in ctx.parsed_references}
    fake_keys = [
        f"fakename{rng.randint(2020, 2026)}{suffix}"
        for suffix in ["deep", "novel", "framework"]
    ]

    injections: list[Injection] = []
    for key in fake_keys[: rng.randint(1, 3)]:
        if key in ref_keys:
            continue
        ctx.inline_citations.append(
            InlineCitation(key=key, line=None, context=f"As shown by {key}.")
        )
        injections.append(
            Injection(
                rule_id="dangling-cite",
                description=f"Injected orphan citation: {key}",
                context=key,
            )
        )

    return injections


def _inject_broken_refs_pdf(
    ctx: ManuscriptContext,
    rng: random.Random,
) -> list[Injection]:
    """Insert '??' patterns into section text (how broken refs appear in PDFs)."""
    if not ctx.sections:
        return []

    target = rng.choice(ctx.sections)
    label = f"Figure {rng.randint(10, 99)}"
    broken = f"{label} ??"

    target.clean_text = f"{broken}. {target.clean_text}"
    target.raw_text = f"{broken}. {target.raw_text}"

    return [
        Injection(
            rule_id="dangling-ref",
            description=f"Injected broken ref: '{broken}' in '{target.title}'",
            context=broken,
        )
    ]


def _inject_number_drift_pdf(
    ctx: ManuscriptContext,
    rng: random.Random,
) -> list[Injection]:
    """Alter a number in the abstract so it contradicts the body."""
    if not ctx.abstract:
        return []

    num_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(?:%|points?|pp)", ctx.abstract))
    if not num_matches:
        return []

    target = rng.choice(num_matches)
    original = target.group(1)
    try:
        val = float(original)
    except ValueError:
        return []

    factor = rng.choice([0.6, 0.7, 1.3, 1.4])
    new_val = f"{val * factor:.1f}"

    ctx.abstract = (
        ctx.abstract[: target.start(1)] + new_val + ctx.abstract[target.end(1) :]
    )

    return [
        Injection(
            rule_id="cross-section-consistency",
            description=f"Injected number drift in abstract: {original} → {new_val}",
            context=f"{original} -> {new_val}",
        )
    ]


PDF_TEXT_INJECTORS = [
    _inject_fake_citations_pdf,
    _inject_broken_refs_pdf,
]

PDF_LLM_INJECTORS = [
    _inject_number_drift_pdf,
]

PDF_ALL_INJECTORS = PDF_TEXT_INJECTORS + PDF_LLM_INJECTORS


def inject_errors_pdf(
    ctx: ManuscriptContext,
    original_path: Path,
    seed: int | None = None,
    text_only: bool = True,
) -> InjectedPaper:
    """Inject synthetic errors into a GROBID-parsed PDF's ManuscriptContext.

    Works on a deep copy — the original ctx is not modified.

    Returns:
        InjectedPaper with injected_ctx set (injected_text is empty).
    """
    injected = copy.deepcopy(ctx)
    rng = random.Random(seed)
    fns = PDF_TEXT_INJECTORS if text_only else PDF_ALL_INJECTORS

    result = InjectedPaper(original_path=original_path)

    for fn in fns:
        new_injections = fn(injected, rng)
        result.injections.extend(new_injections)

    result.injected_ctx = injected
    return result
