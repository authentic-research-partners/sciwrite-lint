"""Synthetic generators for PDF-mode checks (dangling-cite-pdf, dangling-ref-pdf)."""

from __future__ import annotations

from typing import Any

from evals.synthetic_types import ExpectedFinding, SyntheticCase


def _make_tei_xml(
    refs: list[dict[str, str]],
    body_sections: list[dict[str, str]],
    inline_refs: list[int] | None = None,
    abstract: str = "This paper studies X.",
) -> str:
    """Build synthetic GROBID TEI XML for eval cases.

    Args:
        refs: List of {"title": ..., "author": ..., "year": ...} dicts.
        body_sections: List of {"title": ..., "text": ...} dicts.
        inline_refs: List of reference indices to create <ref> tags for.
            These simulate GROBID-linked inline citations.
        abstract: Abstract text.
    """
    # Bibliography
    bib_parts = []
    for i, r in enumerate(refs):
        bib_parts.append(
            f'<biblStruct xml:id="b{i}">'
            f"<analytic><title>{r.get('title', '')}</title>"
            f"<author><persName><surname>{r.get('author', 'Unknown')}</surname></persName></author>"
            f"</analytic><monogr><imprint>"
            f'<date type="published" when="{r.get("year", "2020")}"/>'
            f"</imprint></monogr></biblStruct>"
        )

    # Body with optional inline citation refs
    body_parts = []
    for sec in body_sections:
        text = sec.get("text", "")
        body_parts.append(
            f"<div><head>{sec.get('title', 'Section')}</head><p>{text}</p></div>"
        )

    # Add inline refs to first section paragraph if specified
    if inline_refs and body_sections:
        ref_tags = " ".join(
            f'<ref type="bibr" target="#b{i}">[{i + 1}]</ref>' for i in inline_refs
        )
        text = body_sections[0].get("text", "Text")
        body_parts[0] = (
            f"<div><head>{body_sections[0].get('title', 'Section')}</head>"
            f"<p>{text} {ref_tags}</p></div>"
        )

    return (
        f'<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        f"<teiHeader><fileDesc><titleStmt><title>Synthetic Paper</title></titleStmt>"
        f"<publicationStmt><p/></publicationStmt><sourceDesc><p/></sourceDesc>"
        f"</fileDesc><profileDesc><abstract><p>{abstract}</p></abstract>"
        f"</profileDesc></teiHeader>"
        f"<text><body>{''.join(body_parts)}</body></text>"
        f"<listBibl>{''.join(bib_parts)}</listBibl></TEI>"
    )


def _make_grobid_result(
    refs: list[dict[str, str]],
    body_sections: list[dict[str, str]],
    inline_refs: list[int] | None = None,
    abstract: str = "This paper studies X.",
) -> Any:
    """Build a GrobidResult for eval cases."""
    from sciwrite_lint.pdf.grobid import GrobidReference, GrobidResult, GrobidSection

    tei = _make_tei_xml(refs, body_sections, inline_refs, abstract)

    sections = [
        GrobidSection(
            title=s.get("title", "Section"),
            text=s.get("text", ""),
            level=0,
            index=i,
        )
        for i, s in enumerate(body_sections)
    ]

    references = [
        GrobidReference(
            index=i,
            title=r.get("title", ""),
            authors=[r.get("author", "Unknown")],
            year=r.get("year", "2020"),
        )
        for i, r in enumerate(refs)
    ]

    return GrobidResult(
        title="Synthetic Paper",
        authors=["Test Author"],
        abstract=abstract,
        sections=sections,
        references=references,
        raw_tei=tei,
    )


def gen_dangling_cite_pdf_cases() -> list[SyntheticCase]:
    """PDF-mode dangling citation eval cases.

    PDF dangling-cite works differently from LaTeX: GROBID links inline
    <ref> tags to <biblStruct> entries. The check compares
    ManuscriptContext.inline_citations keys against parsed_references keys.

    Since both come from the same GrobidResult, dangling cites can only
    occur when:
    1. GROBID generates a <ref> that maps to a ref index with no
       corresponding entry in our parsed references
    2. The TEI has references that our parser couldn't handle

    We test: clean cases (no dangles), partially cited (still clean),
    and verify that the check itself is correctly dispatched in PDF mode.
    """
    cases: list[SyntheticCase] = []

    refs = [
        {"title": "Deep Learning Advances", "author": "Smith", "year": "2020"},
        {"title": "Neural Networks Revisited", "author": "Jones", "year": "2021"},
        {"title": "Transformers for NLP", "author": "Wang", "year": "2022"},
    ]

    sections = [
        {"title": "Introduction", "text": "We study deep learning advances."},
        {"title": "Methods", "text": "Our approach uses transformers."},
        {"title": "Results", "text": "Results show 45% improvement."},
    ]

    # TN: all citations properly linked
    cases.append(
        SyntheticCase(
            name="pdf_dangling_cite_clean",
            check_id="dangling-cite",
            description="PDF with all citations properly linked to references",
            expected=[],
            grobid_result=_make_grobid_result(refs, sections, inline_refs=[0, 1, 2]),
        )
    )

    # TN: no inline citations at all
    cases.append(
        SyntheticCase(
            name="pdf_dangling_cite_no_citations",
            check_id="dangling-cite",
            description="PDF with no inline citations (clean)",
            expected=[],
            grobid_result=_make_grobid_result(refs, sections, inline_refs=None),
        )
    )

    # TN: only some refs cited — uncited refs are NOT dangling
    cases.append(
        SyntheticCase(
            name="pdf_dangling_cite_partial_citation",
            check_id="dangling-cite",
            description="PDF citing only some refs — uncited refs are not dangling",
            expected=[],
            grobid_result=_make_grobid_result(refs, sections, inline_refs=[0]),
        )
    )

    # TN: many references, all cited — realistic CS paper
    many_refs = [
        {"title": f"Paper {i}", "author": f"Author{i}", "year": str(2018 + i)}
        for i in range(10)
    ]
    cases.append(
        SyntheticCase(
            name="pdf_dangling_cite_many_refs_clean",
            check_id="dangling-cite",
            description="PDF with 10 references, all properly cited",
            expected=[],
            grobid_result=_make_grobid_result(
                many_refs, sections, inline_refs=list(range(10))
            ),
        )
    )

    return cases


def gen_dangling_ref_pdf_cases() -> list[SyntheticCase]:
    """PDF-mode broken reference ('??') eval cases."""
    from sciwrite_lint.pdf.grobid import GrobidResult, GrobidSection

    cases: list[SyntheticCase] = []

    # TP: Figure ??
    cases.append(
        SyntheticCase(
            name="pdf_dangling_ref_figure",
            check_id="dangling-ref",
            description="PDF with 'Figure ??' broken reference",
            expected=[ExpectedFinding(rule_id="dangling-ref", context="Figure ??")],
            grobid_result=GrobidResult(
                title="Paper with broken ref",
                abstract="Results in Figure 1.",
                sections=[
                    GrobidSection(
                        title="Results",
                        text="As shown in Figure ??, the accuracy improved by 15%.",
                        level=0,
                        index=0,
                    ),
                ],
            ),
        )
    )

    # TP: Eq. ??
    cases.append(
        SyntheticCase(
            name="pdf_dangling_ref_equation",
            check_id="dangling-ref",
            description="PDF with 'Eq. ??' broken reference",
            expected=[ExpectedFinding(rule_id="dangling-ref", context="Eq. ??")],
            grobid_result=GrobidResult(
                title="Paper with broken eq ref",
                abstract="See Eq. 1.",
                sections=[
                    GrobidSection(
                        title="Methods",
                        text="The loss function in Eq. ?? is defined as cross-entropy.",
                        level=0,
                        index=0,
                    ),
                ],
            ),
        )
    )

    # TP: Table ??
    cases.append(
        SyntheticCase(
            name="pdf_dangling_ref_table",
            check_id="dangling-ref",
            description="PDF with 'Table ??' broken reference",
            expected=[ExpectedFinding(rule_id="dangling-ref", context="Table ??")],
            grobid_result=GrobidResult(
                title="Paper with broken table ref",
                abstract="Abstract.",
                sections=[
                    GrobidSection(
                        title="Results",
                        text="Table ?? summarizes the results across all datasets.",
                        level=0,
                        index=0,
                    ),
                ],
            ),
        )
    )

    # TP: multiple broken refs in one paper
    cases.append(
        SyntheticCase(
            name="pdf_dangling_ref_multiple",
            check_id="dangling-ref",
            description="PDF with multiple broken references across sections",
            expected=[
                ExpectedFinding(rule_id="dangling-ref", context="Figure ??"),
                ExpectedFinding(rule_id="dangling-ref", context="Table ??"),
            ],
            grobid_result=GrobidResult(
                title="Paper with multiple broken refs",
                abstract="Abstract.",
                sections=[
                    GrobidSection(
                        title="Results",
                        text="Figure ?? shows the architecture. Table ?? lists the hyperparameters.",
                        level=0,
                        index=0,
                    ),
                ],
            ),
        )
    )

    # TP: broken ref in abstract
    cases.append(
        SyntheticCase(
            name="pdf_dangling_ref_in_abstract",
            check_id="dangling-ref",
            description="PDF with broken reference in abstract",
            expected=[ExpectedFinding(rule_id="dangling-ref", context="Section ??")],
            grobid_result=GrobidResult(
                title="Paper",
                abstract="In Section ?? we describe our method.",
                sections=[
                    GrobidSection(
                        title="Methods", text="Our method is novel.", level=0, index=0
                    ),
                ],
            ),
        )
    )

    # TN: clean PDF with no broken references
    cases.append(
        SyntheticCase(
            name="pdf_dangling_ref_clean",
            check_id="dangling-ref",
            description="PDF with no broken references (all resolved)",
            expected=[],
            grobid_result=GrobidResult(
                title="Clean paper",
                abstract="We present results in Figure 1.",
                sections=[
                    GrobidSection(
                        title="Results",
                        text="Figure 1 shows results. Table 2 has metrics. See Section 3.",
                        level=0,
                        index=0,
                    ),
                ],
            ),
        )
    )

    # TN: question marks that are NOT broken refs
    cases.append(
        SyntheticCase(
            name="pdf_dangling_ref_questions_not_refs",
            check_id="dangling-ref",
            description="PDF with question marks that are not broken refs",
            expected=[],
            grobid_result=GrobidResult(
                title="Paper with questions",
                abstract="What is the best approach?",
                sections=[
                    GrobidSection(
                        title="Discussion",
                        text="Can we improve this? What if we tried a different approach? "
                        "The answer remains unclear.",
                        level=0,
                        index=0,
                    ),
                ],
            ),
        )
    )

    return cases
