"""Cross-format synthetic manuscript corpus.

One :class:`Scenario` describes manuscript content abstractly — citations,
figures, and cross-references, each flagged valid or broken — together
with the deterministic findings it should produce (derived from the
content, so the scenario is the single source of truth). Renderers turn a
scenario into a real ``.tex``, ``.md``, or ``.pdf`` so the *same* content
can be linted through each front-end (LaTeX parser, pandoc, GROBID).

tex/md are asserted symbol-for-symbol (``tests/test_synthetic_corpus.py``).
PDF is lossy — GROBID rebuilds from rendered text and drops the source's
symbolic ids — so :func:`pdf_coverage_report` measures it at the rule
level instead. Everything is generated on demand into a caller-supplied
directory; nothing is committed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from sciwrite_lint.models import Finding

from evals.synthetic_types import ExpectedFinding

# Deterministic, service-free checks this corpus asserts against.
DETERMINISTIC_CHECKS: frozenset[str] = frozenset(
    {"dangling-cite", "dangling-ref", "unreferenced-figure"}
)

# Rules a PDF round-trip cannot preserve, intrinsic to the format (not a
# check defect). ``dangling-cite`` needs the source cite *key*; a cite to a
# missing bib entry renders as ``[?]`` in the PDF, so GROBID has nothing to
# reconstruct a key from. The PDF coverage eval records these as known gaps,
# never as regressions.
PDF_UNDETECTABLE: frozenset[str] = frozenset({"dangling-cite"})


class Cite(BaseModel):
    """A citation in the manuscript. ``in_bib=False`` → dangling."""

    key: str
    in_bib: bool = True


class Figure(BaseModel):
    """A labelled figure. ``referenced=False`` → unreferenced.

    ``label`` carries the pandoc-crossref ``fig:`` prefix (e.g.
    ``"fig:flow"``) so the same value works as a LaTeX ``\\label`` and a
    markdown ``{#fig:flow}`` id.
    """

    label: str
    referenced: bool = True


class Scenario(BaseModel):
    """One logical manuscript, renderable to several formats."""

    name: str
    title: str = "Synthetic Manuscript"
    cites: list[Cite] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    # Cross-reference targets with no matching definition (→ dangling-ref);
    # each carries a crossref prefix, e.g. "fig:ghost" / "sec:ghost".
    dangling_refs: list[str] = Field(default_factory=list)
    # Sentences embedded verbatim in the body — used to seed LLM-engine
    # checks (e.g. a grammar error → prose-quality, a percentage set that
    # doesn't sum to 100 → percentages-sum). Empty for the deterministic
    # scenarios, so their rendered output is byte-identical to before.
    prose: list[str] = Field(default_factory=list)
    # URLs to embed in footnotes — LaTeX ``\footnote{\url{}}`` / markdown
    # ``^[<url>]`` — for footnote-URL extraction parity.
    footnote_urls: list[str] = Field(default_factory=list)
    # Bib keys present in the bibliography but never cited (→ orphan /
    # in-bib-never-cited). Rendered into the ``.bib`` without a citation.
    orphan_bib_keys: list[str] = Field(default_factory=list)
    # Markdown only: split the in-bib cites across two ``.bib`` files and a
    # frontmatter ``bibliography: [a, b]`` list — exercises multi-file bib
    # resolution. A cite from each file must resolve for both to be read.
    multi_bib: bool = False
    # One-line description of what a parse/extraction scenario demonstrates,
    # shown in the corpus MANIFEST when it produces no rule-level finding.
    demonstrates: str = ""
    # Rule ids the LLM-engine checks should surface for this content. Tracked
    # at the rule level only (recall): LLM output is non-deterministic, so
    # the exact context string and precision are not asserted.
    expected_llm_rules: list[str] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=lambda: ["tex", "md"])

    def expected_findings(self) -> list[ExpectedFinding]:
        """Derive the deterministic findings this content must produce."""
        expected: list[ExpectedFinding] = []
        for cite in self.cites:
            if not cite.in_bib:
                expected.append(
                    ExpectedFinding(rule_id="dangling-cite", context=cite.key)
                )
        for fig in self.figures:
            if not fig.referenced:
                expected.append(
                    ExpectedFinding(rule_id="unreferenced-figure", context=fig.label)
                )
        for target in self.dangling_refs:
            expected.append(ExpectedFinding(rule_id="dangling-ref", context=target))
        return expected


def matches(expected: ExpectedFinding, finding: Finding) -> bool:
    """True if ``finding`` satisfies ``expected`` (rule + needle in text).

    The needle is matched against the finding's context *or* message, so it
    works regardless of which field a given check populates per format.
    """
    return expected.rule_id == finding.rule_id and (
        expected.context in (finding.context or "")
        or expected.context in (finding.message or "")
    )


def _bib_entry(key: str) -> str:
    """One canonical ``@article`` bib entry for ``key``."""
    return f"@article{{{key}, title={{Study {key}}}, author={{Author, A.}}, year={{2020}}}}"


def _bib_keys(scenario: Scenario) -> list[str]:
    """All keys that belong in the ``.bib``: cited-and-in-bib, plus orphans."""
    return [c.key for c in scenario.cites if c.in_bib] + list(scenario.orphan_bib_keys)


def _bib_text(scenario: Scenario) -> str:
    """A ``.bib`` holding the in-bib cites and any uncited orphan entries."""
    entries = [_bib_entry(k) for k in _bib_keys(scenario)]
    return "\n".join(entries) + ("\n" if entries else "")


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in free-text prose.

    Embedded prose is plain English, but characters like ``%`` (comment),
    ``&``, ``$``, ``#``, ``_`` and braces are LaTeX-special — an unescaped
    ``40%`` silently comments out the rest of the line. Backslash is replaced
    first so the escapes this introduces are not themselves re-escaped.
    """
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for char, escaped in replacements:
        text = text.replace(char, escaped)
    return text


def render_tex(
    scenario: Scenario, dest: Path, *, figure_graphic: str = r"\rule{4cm}{3cm}"
) -> Path:
    """Render the scenario as a LaTeX manuscript (+ ``refs.bib``).

    ``figure_graphic`` is the body of each ``figure`` environment — a
    ``\\rule`` box by default (fine for source-level checks, which never
    compile), or ``\\includegraphics`` when :func:`render_pdf` needs a real
    image GROBID can detect.
    """
    body: list[str] = [
        f"We rely on prior work \\cite{{{c.key}}}." for c in scenario.cites
    ]
    body += [f"See \\ref{{{t}}}." for t in scenario.dangling_refs]
    # Prose is free text — escape LaTeX specials (notably "%", which would
    # otherwise comment out the rest of the line). Cite/ref lines above are
    # generated LaTeX and must keep their backslashes.
    body += [_latex_escape(p) for p in scenario.prose]
    body += [
        f"This claim is sourced.\\footnote{{\\url{{{u}}}}}"
        for u in scenario.footnote_urls
    ]

    figures: list[str] = []
    for fig in scenario.figures:
        figures.append(
            f"\\begin{{figure}}\n\\centering {figure_graphic}\n"
            f"\\caption{{Figure for {fig.label}.}}\\label{{{fig.label}}}\n"
            "\\end{figure}"
        )
        if fig.referenced:
            # "Figure~\ref" so the rendered PDF reads "Figure 1" — the form
            # the figure-reference matcher (and humans) recognise. Bare
            # "\ref" renders just "1", which reads as an unreferenced figure.
            body.append(f"As shown in Figure~\\ref{{{fig.label}}}, the result holds.")

    has_bib = bool(_bib_keys(scenario))
    if has_bib:
        (dest / "refs.bib").write_text(_bib_text(scenario), encoding="utf-8")

    # Built by concatenation (not one big f-string) to keep LaTeX
    # backslashes and braces readable and unescaped.
    lines = [
        r"\documentclass{article}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\begin{document}",
        r"\title{" + scenario.title + r"}\author{A}\maketitle",
        r"\section{Introduction}",
        " ".join(body),
        "\n".join(figures),
    ]
    if has_bib:
        lines.append(r"\bibliography{refs}")
    lines.append(r"\end{document}")

    path = dest / f"{scenario.name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_md(scenario: Scenario, dest: Path) -> Path:
    """Render the scenario as a pandoc markdown manuscript (+ ``refs.bib``)."""
    body: list[str] = [f"We rely on prior work [@{c.key}]." for c in scenario.cites]
    body += [f"See @{t}." for t in scenario.dangling_refs]
    body += scenario.prose
    body += [f"This claim is sourced.^[See <{u}>.]" for u in scenario.footnote_urls]

    figures: list[str] = []
    for fig in scenario.figures:
        figures.append(f"![Figure for {fig.label}.](fig.png){{#{fig.label}}}")
        if fig.referenced:
            body.append(f"As shown in @{fig.label}, the result holds.")

    bib_line = ""
    if scenario.multi_bib:
        # Split in-bib cites across two files; resolving both proves the
        # frontmatter list (not just the first entry) is honored.
        in_bib = [c.key for c in scenario.cites if c.in_bib]
        for fname, keys in (("refs_a.bib", in_bib[0::2]), ("refs_b.bib", in_bib[1::2])):
            text = "\n".join(_bib_entry(k) for k in keys)
            (dest / fname).write_text(text + ("\n" if keys else ""), encoding="utf-8")
        bib_line = "bibliography: [refs_a.bib, refs_b.bib]\n"
    elif _bib_keys(scenario):
        (dest / "refs.bib").write_text(_bib_text(scenario), encoding="utf-8")
        bib_line = "bibliography: refs.bib\n"

    front = f"---\ntitle: {scenario.title}\n{bib_line}---\n"

    md = (
        f"{front}\n# Introduction\n\n{' '.join(body)}\n\n" + "\n\n".join(figures) + "\n"
    )
    path = dest / f"{scenario.name}.md"
    path.write_text(md, encoding="utf-8")
    return path


def _write_png(path: Path, width: int = 240, height: int = 160) -> None:
    """Write a minimal valid grey-rectangle PNG (no third-party deps).

    A real raster (not a ``\\rule`` box) so GROBID detects each figure as a
    distinct figure rather than merging adjacent vector boxes.
    """
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    row = b"\x00" + b"\xa0\xa0\xa0" * width  # filter byte + grey pixels
    idat = zlib.compress(row * height, 9)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def render_pdf(scenario: Scenario, dest: Path) -> Path:
    """Render the scenario to a real PDF via ``pdflatex`` (+ ``bibtex``).

    Reuses :func:`render_tex` (with ``\\includegraphics`` figures) so the
    PDF carries the same content, then compiles. Needs a TeX toolchain;
    raises a clear error if ``pdflatex`` is absent.
    """
    import shutil
    import subprocess

    if shutil.which("pdflatex") is None:
        raise RuntimeError(
            "pdflatex not found — needed to render synthetic PDFs. "
            "Install a TeX distribution (e.g. texlive)."
        )

    if scenario.figures:
        _write_png(dest / "fig.png")
    tex_path = render_tex(
        scenario, dest, figure_graphic=r"\includegraphics[width=6cm]{fig.png}"
    )
    stem = tex_path.stem
    has_bib = any(c.in_bib for c in scenario.cites)

    def _latex() -> None:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=dest,
            capture_output=True,
            timeout=120,
            check=False,
        )

    _latex()
    if has_bib:
        subprocess.run(
            ["bibtex", stem], cwd=dest, capture_output=True, timeout=120, check=False
        )
        _latex()
    _latex()  # resolve \ref / \cite cross-references

    pdf_path = dest / f"{stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"pdflatex did not produce {pdf_path.name}")
    return pdf_path


RENDERERS: dict[str, Callable[[Scenario, Path], Path]] = {
    "tex": render_tex,
    "md": render_md,
    "pdf": render_pdf,
}


def materialize(scenario: Scenario, fmt: str, dest: Path) -> Path:
    """Write ``scenario`` as ``fmt`` into ``dest``; return the manuscript path."""
    if fmt not in RENDERERS:
        raise ValueError(f"unknown format {fmt!r}; available: {sorted(RENDERERS)}")
    return RENDERERS[fmt](scenario, dest)


def deterministic_findings(path: Path, config: object) -> list[Finding]:
    """Run the deterministic (service-free) checks on a materialized manuscript.

    Builds the markdown context first for ``.md`` input (so the markdown
    checks see it), then returns only the deterministic findings.
    """
    import asyncio

    from sciwrite_lint.config import LintConfig
    from sciwrite_lint.pipeline import (
        build_markdown_context,
        build_pdf_context,
        run_text_checks,
    )

    assert isinstance(config, LintConfig)
    suffix = path.suffix.lower()
    if suffix == ".md":
        build_markdown_context(path, config)
    elif suffix == ".pdf":
        # GROBID parse (local; needs the container) so config.is_pdf is set.
        asyncio.run(build_pdf_context(path, config))
    return [
        f for f in run_text_checks(path, config) if f.rule_id in DETERMINISTIC_CHECKS
    ]


# The standing deterministic scenario set. All three formats are meaningful
# (they declare ``pdf``); the service-free test asserts tex/md symbol-for-
# symbol (see SYMBOLIC_FORMATS), while pdf_coverage_report measures the same
# scenarios at the rule level after a GROBID round-trip.
_DET_FORMATS = ["tex", "md", "pdf"]
SCENARIOS: list[Scenario] = [
    Scenario(
        name="clean",
        cites=[Cite(key="smith2020")],
        figures=[Figure(label="fig:flow")],
        formats=_DET_FORMATS,
    ),
    Scenario(
        name="dangling_cite",
        cites=[Cite(key="smith2020"), Cite(key="ghost2099", in_bib=False)],
        formats=_DET_FORMATS,
    ),
    Scenario(
        name="unreferenced_figure",
        figures=[Figure(label="fig:orphan", referenced=False)],
        formats=_DET_FORMATS,
    ),
    Scenario(name="dangling_ref", dangling_refs=["fig:ghost"], formats=_DET_FORMATS),
    Scenario(
        name="combination",
        cites=[Cite(key="smith2020"), Cite(key="ghost2099", in_bib=False)],
        figures=[
            Figure(label="fig:shown"),
            Figure(label="fig:orphan", referenced=False),
        ],
        dangling_refs=["fig:ghost"],
        formats=_DET_FORMATS,
    ),
]

# Formats the service-free test suite asserts symbol-for-symbol (no GROBID).
SYMBOLIC_FORMATS: tuple[str, ...] = ("tex", "md")


class PdfCoverage(BaseModel):
    """Rule-level PDF detection coverage for one scenario.

    PDF is lossy — GROBID reconstructs from rendered text, losing the
    source's symbolic cite keys / figure labels and dropping a cite to a
    missing bib entry entirely (it renders as ``[?]``). So PDF coverage is
    reported at the *rule* level, descriptively, not asserted
    symbol-for-symbol the way tex/md are in the test suite.
    """

    name: str
    expected_rules: list[str]  # symbolic ground-truth rules (from tex/md)
    detected_rules: list[str]  # rules that fired on the parsed PDF

    @property
    def missed(self) -> list[str]:
        """Expected rules that did not fire on the parsed PDF."""
        return sorted(set(self.expected_rules) - set(self.detected_rules))

    @property
    def regressions(self) -> list[str]:
        """Missed rules that PDF *should* preserve — real failures."""
        return [r for r in self.missed if r not in PDF_UNDETECTABLE]

    @property
    def known_gaps(self) -> list[str]:
        """Missed rules that PDF intrinsically cannot preserve."""
        return [r for r in self.missed if r in PDF_UNDETECTABLE]

    @property
    def unexpected(self) -> list[str]:
        """Rules that fired but were not in the symbolic ground truth."""
        return sorted(set(self.detected_rules) - set(self.expected_rules))


def pdf_coverage_report(dest_root: Path) -> list[PdfCoverage]:
    """Render every scenario to PDF, lint via GROBID, report rule-level coverage.

    Requires ``pdflatex`` and a running GROBID. Descriptive (no pass/fail):
    surfaces which deterministic rules each scenario still triggers after a
    round-trip through PDF rendering + GROBID re-parsing, against its
    symbolic ground-truth rules — making PDF's extraction limits visible.
    """
    from sciwrite_lint.config import LintConfig

    report: list[PdfCoverage] = []
    for scenario in SCENARIOS:
        sub = dest_root / scenario.name
        sub.mkdir(parents=True, exist_ok=True)
        pdf = materialize(scenario, "pdf", sub)
        findings = deterministic_findings(pdf, LintConfig())
        report.append(
            PdfCoverage(
                name=scenario.name,
                expected_rules=sorted(
                    {e.rule_id for e in scenario.expected_findings()}
                ),
                detected_rules=sorted({f.rule_id for f in findings}),
            )
        )
    return report


# Rules this LLM corpus targets. Each scenario seeds one of these with
# self-contained prose (no external reference data). Other ``local-llm``
# checks — abstract-body-alignment, numbers-vs-tables, the rest of the
# full-paper-consistency family — reason over whole-paper structure and
# misfire on these deliberately minimal manuscripts; they need their own
# richer multi-section scenarios, so coverage is measured only within this
# scope (out-of-scope findings are not signal here).
LLM_RULES_UNDER_TEST: frozenset[str] = frozenset({"prose-quality", "percentages-sum"})

# LLM-engine scenarios. Content is chosen to trigger one targeted check
# reliably with no external reference data: a subject-verb grammar error
# (prose-quality) and a percentage set that sums to 93%, not 100%
# (percentages-sum, one of the full-paper-consistency family). The clean
# control uses hedged prose and a percentage set that balances. Rendered to
# tex and md — the formats that preserve prose verbatim — and each format's
# output is checked for the expected rule (recall per format), not compared
# to the other.
LLM_SCENARIOS: list[Scenario] = [
    Scenario(
        name="prose_grammar",
        prose=["The set of measurements demonstrate a clear upward trend."],
        expected_llm_rules=["prose-quality"],
    ),
    Scenario(
        name="percentages_sum",
        prose=[
            "The cohort comprised 40% early-career, 35% mid-career, and 18% "
            "senior researchers."
        ],
        expected_llm_rules=["percentages-sum"],
    ),
    Scenario(
        name="llm_clean",
        prose=[
            "Our results suggest that the method may improve performance.",
            "The sample included early-career (40%), mid-career (35%), and "
            "senior (25%) researchers.",
        ],
        expected_llm_rules=[],
    ),
]


class LlmCoverage(BaseModel):
    """Recall-oriented LLM-check coverage for one scenario in one format.

    LLM-engine checks are non-deterministic, so coverage is reported at the
    rule level (did the expected rule fire?) without asserting the exact
    finding context or precision — extra findings are informational, not
    failures. ``llm_clean`` (no expected rules) is the false-positive watch.
    """

    name: str
    fmt: str
    expected_rules: list[str]
    detected_rules: list[str]

    @property
    def hit(self) -> list[str]:
        """Expected rules that fired (recall numerator)."""
        return [r for r in self.expected_rules if r in self.detected_rules]

    @property
    def missed(self) -> list[str]:
        """Expected rules that did not fire."""
        return [r for r in self.expected_rules if r not in self.detected_rules]

    @property
    def unexpected(self) -> list[str]:
        """Rules that fired but were not expected (informational)."""
        return sorted(set(self.detected_rules) - set(self.expected_rules))


async def llm_coverage_report(
    dest_root: Path, formats: tuple[str, ...] = ("tex", "md")
) -> list[LlmCoverage]:
    """Render LLM scenarios, run the LLM-engine checks, report rule recall.

    Requires a running text vLLM. For each scenario × format, materializes
    the manuscript, builds the format's manuscript context, runs every
    ``local-llm`` check via :func:`run_llm_checks_batched`, and records which
    expected rules fired. A fresh :class:`LintConfig` per manuscript keeps
    one format's cached manuscript context from leaking into the next.
    Descriptive recall report — LLM non-determinism means a single miss is
    not a hard failure (see the eval's exit logic).
    """
    from sciwrite_lint.config import LintConfig
    from sciwrite_lint.pipeline import build_markdown_context, run_llm_checks_batched

    report: list[LlmCoverage] = []
    for scenario in LLM_SCENARIOS:
        for fmt in formats:
            sub = dest_root / f"{scenario.name}-{fmt}"
            sub.mkdir(parents=True, exist_ok=True)
            path = materialize(scenario, fmt, sub)
            config = LintConfig()
            if path.suffix.lower() == ".md":
                build_markdown_context(path, config)
            findings = await run_llm_checks_batched(path, config)
            # Restrict to the rules this corpus targets; other local-llm
            # checks need richer scenarios and only add noise on minimal docs.
            detected = {f.rule_id for f in findings} & LLM_RULES_UNDER_TEST
            report.append(
                LlmCoverage(
                    name=scenario.name,
                    fmt=fmt,
                    expected_rules=sorted(set(scenario.expected_llm_rules)),
                    detected_rules=sorted(detected),
                )
            )
    return report


# Parsing / extraction scenarios. These exercise lower-level extraction and
# classification (footnote-URL extraction, orphan-bib detection, citation-
# style classification, multi-file bib resolution, zero-reference handling)
# rather than producing run_text_checks findings, so they are asserted by
# dedicated tests against the real functions, not the symbolic loop. The
# footnote-URL and orphan-bib tests compare tex against md directly (genuine
# parity); numeric/multi-bib are markdown-only feature checks.
PARSE_SCENARIOS: list[Scenario] = [
    Scenario(
        name="footnote_urls",
        footnote_urls=[
            "https://example.com/about",
            # A long URL — LaTeX line-wraps it on compile; extraction must
            # still recover it whole (the deferred footnote-URL question).
            "https://data.example.org/datasets/2024/very/long/path/to/a/resource?id=12345&format=csv",
        ],
        demonstrates="footnote-URL extraction (tex + md)",
        formats=["tex", "md"],
    ),
    Scenario(
        name="uncited_bib",
        cites=[Cite(key="smith2020")],
        orphan_bib_keys=["lonely2020"],
        demonstrates="orphan bib entry: in-bib-never-cited (tex + md)",
        formats=["tex", "md"],
    ),
    Scenario(
        name="numeric_citations",
        prose=["Prior work [1] and later studies [2, 3] established this result."],
        demonstrates="numeric [1] citations classified as 'numeric' (md)",
        formats=["md"],
    ),
    Scenario(
        name="multi_bib",
        cites=[
            Cite(key="alpha2020"),
            Cite(key="beta2021"),
            Cite(key="ghost2099", in_bib=False),
        ],
        multi_bib=True,
        demonstrates="multi-file bibliography list resolution (md)",
        formats=["md"],
    ),
    Scenario(
        name="no_references",
        demonstrates="zero references → no-references notice",
        formats=["tex", "md"],
    ),
]


def detected_footnote_urls(path: Path) -> list[str]:
    """Footnote URLs extracted from a materialized manuscript (tex or md)."""
    from sciwrite_lint.footnote_urls import (
        extract_footnote_urls,
        extract_footnote_urls_markdown,
    )

    suffix = path.suffix.lower()
    if suffix == ".tex":
        pairs = extract_footnote_urls(path)
    elif suffix == ".md":
        pairs = extract_footnote_urls_markdown(path)
    else:
        raise ValueError(f"footnote-URL extraction not supported for {suffix}")
    return sorted(url for _, url in pairs)


def detected_orphan_bib_keys(path: Path) -> set[str]:
    """Bib keys present but never cited — the in-bib-never-cited orphan set.

    Mirrors the per-format split in ``cli/verify.py::run_ref_health``: LaTeX
    uses ``find_orphans`` (scans ``\\cite``), markdown diffs the bib keys
    against the pandoc cited keys.
    """
    from sciwrite_lint.references.citations import extract_bibitems, find_orphans

    suffix = path.suffix.lower()
    if suffix == ".tex":
        citations = extract_bibitems(path)
        _, bib_no_cite = find_orphans(citations, path)
        return bib_no_cite
    if suffix == ".md":
        from sciwrite_lint.markdown_cites import analyze_markdown, parse_markdown_bib

        analysis = analyze_markdown(path.read_text(encoding="utf-8"))
        citations = parse_markdown_bib(path, analysis)
        return {c.key for c in citations} - {c.key for c in analysis.citations}
    raise ValueError(f"orphan detection not supported for {suffix}")


class CorpusEntry(BaseModel):
    """One scenario written to disk, with what it is built to demonstrate."""

    name: str
    kind: str  # "deterministic", "llm", or "parse" (extraction/classification)
    expected_rules: list[str]  # rule ids it should trigger (empty for parse)
    summary: str  # one-line description of what it demonstrates
    paths: list[str]  # files written, relative to the corpus root

    @property
    def formats(self) -> list[str]:
        """Formats present, derived from the written file extensions."""
        return sorted({Path(p).suffix.lstrip(".") for p in self.paths})


def materialize_corpus(
    dest_root: Path, formats: tuple[str, ...] = ("tex", "md")
) -> list[CorpusEntry]:
    """Write every scenario to each requested format under ``dest_root``.

    Materialization needs no services for ``tex``/``md`` (pure file writes);
    ``pdf`` requires ``pdflatex``. Returns a manifest describing each scenario
    and the checks it is built to trigger — useful as living documentation of
    what the linter detects, generated on demand rather than committed.
    """
    bad = [f for f in formats if f not in RENDERERS]
    if bad:
        raise ValueError(f"unknown format(s) {bad}; available: {sorted(RENDERERS)}")

    entries: list[CorpusEntry] = []
    groups: tuple[tuple[str, list[Scenario]], ...] = (
        ("deterministic", SCENARIOS),
        ("llm", LLM_SCENARIOS),
        ("parse", PARSE_SCENARIOS),
    )
    for kind, scenarios in groups:
        for scenario in scenarios:
            # Render only the requested formats the scenario supports (e.g.
            # numeric/multi-bib are markdown-only); skip it if none overlap.
            chosen = [f for f in formats if f in scenario.formats]
            if not chosen:
                continue
            sub = dest_root / scenario.name
            sub.mkdir(parents=True, exist_ok=True)
            paths = [
                str(materialize(scenario, fmt, sub).relative_to(dest_root))
                for fmt in chosen
            ]
            if kind == "llm":
                expected = sorted(set(scenario.expected_llm_rules))
                summary = ", ".join(expected) or "clean control"
            elif kind == "parse":
                expected = []
                summary = scenario.demonstrates
            else:
                expected = sorted({e.rule_id for e in scenario.expected_findings()})
                summary = ", ".join(expected) or "clean control"
            entries.append(
                CorpusEntry(
                    name=scenario.name,
                    kind=kind,
                    expected_rules=expected,
                    summary=summary,
                    paths=sorted(paths),
                )
            )
    return entries


def render_manifest(entries: list[CorpusEntry]) -> str:
    """A human-readable MANIFEST.md describing the materialized corpus."""
    lines = [
        "# Synthetic manuscript corpus",
        "",
        "Generated on demand by `python -m evals synth-corpus`. Each scenario "
        "is one small manuscript in several formats; **demonstrates** is what "
        "it is built to surface (a clean control triggers nothing). Lint a "
        "file with `sciwrite-lint check <path>`.",
    ]
    for kind, heading in (
        ("deterministic", "Deterministic checks"),
        ("llm", "LLM-engine checks"),
        ("parse", "Parsing / extraction"),
    ):
        group = [e for e in entries if e.kind == kind]
        if not group:
            continue
        lines += ["", f"## {heading}", ""]
        for entry in group:
            fmts = ", ".join(entry.formats)
            lines.append(f"- **{entry.name}** ({fmts}) — {entry.summary}")
    return "\n".join(lines) + "\n"
