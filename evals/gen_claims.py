"""Synthetic generators for claim-support and cite-purpose checks."""

from __future__ import annotations

from evals.synthetic_types import (
    ExpectedFinding,
    SyntheticCase,
    _BIBITEMS,
    _make_doc,
)
from evals.synthetic_templates import build_realistic_paper


def gen_claim_support_cases() -> list[SyntheticCase]:
    """Generate cases for claim-support check.

    Since claim-support requires the full vLLM + GROBID pipeline, these
    cases test the finding conversion logic (claims_to_findings) using
    pre-computed claim verification results stored in metadata.
    """
    cases: list[SyntheticCase] = []

    # TP: claim not supported
    cases.append(
        SyntheticCase(
            name="claim_support_not_supported",
            check_id="claim-support",
            description="Claim verdict is NOT_SUPPORTED",
            tex_content=_make_doc(
                r"Smith showed 90\% accuracy \cite{smith2020}.",
                [_BIBITEMS[0]],
            ),
            metadata={
                "_claim_results": [
                    {
                        "key": "smith2020",
                        "line": 5,
                        "verdict": "NOT_SUPPORTED",
                        "explanation": "Source reports 60% accuracy, not 90%",
                    },
                ],
            },
            expected=[
                ExpectedFinding(rule_id="claim-support", context="smith2020"),
            ],
        )
    )

    # TP: claim partially supported (warning)
    cases.append(
        SyntheticCase(
            name="claim_support_partial",
            check_id="claim-support",
            description="Claim verdict is PARTIALLY_SUPPORTS",
            tex_content=_make_doc(
                r"Jones demonstrated transformers are superior \cite{jones2021}.",
                [_BIBITEMS[1]],
            ),
            metadata={
                "_claim_results": [
                    {
                        "key": "jones2021",
                        "line": 5,
                        "verdict": "PARTIALLY_SUPPORTS",
                        "explanation": "Source shows improvement but not superiority",
                    },
                ],
            },
            expected=[
                ExpectedFinding(rule_id="claim-support", context="jones2021"),
            ],
        )
    )

    # TP: multiple claims, mixed verdicts
    cases.append(
        SyntheticCase(
            name="claim_support_mixed",
            check_id="claim-support",
            description="Two claims: one not supported, one supported",
            tex_content=_make_doc(
                r"\cite{smith2020} showed X. \cite{jones2021} showed Y.",
                _BIBITEMS[:2],
            ),
            metadata={
                "_claim_results": [
                    {
                        "key": "smith2020",
                        "line": 5,
                        "verdict": "NOT_SUPPORTED",
                        "explanation": "No evidence for X in source",
                    },
                    {
                        "key": "jones2021",
                        "line": 5,
                        "verdict": "SUPPORTS",
                        "explanation": "Source confirms Y",
                    },
                ],
            },
            expected=[
                ExpectedFinding(rule_id="claim-support", context="smith2020"),
            ],
        )
    )

    # TN: all claims supported
    cases.append(
        SyntheticCase(
            name="claim_support_all_ok",
            check_id="claim-support",
            description="All claims are supported (no issues)",
            tex_content=_make_doc(
                r"\cite{smith2020} confirms the hypothesis.",
                [_BIBITEMS[0]],
            ),
            metadata={
                "_claim_results": [
                    {
                        "key": "smith2020",
                        "line": 5,
                        "verdict": "SUPPORTS",
                        "explanation": "Source directly supports the claim",
                    },
                ],
            },
            expected=[],
        )
    )

    # TN: dismissed claim should not produce finding
    cases.append(
        SyntheticCase(
            name="claim_support_dismissed",
            check_id="claim-support",
            description="NOT_SUPPORTED but dismissed — should produce no finding",
            tex_content=_make_doc(
                r"\cite{smith2020} showed something.",
                [_BIBITEMS[0]],
            ),
            metadata={
                "_claim_results": [
                    {
                        "key": "smith2020",
                        "line": 5,
                        "verdict": "NOT_SUPPORTED",
                        "explanation": "Doesn't match",
                        "dismissed": True,
                    },
                ],
            },
            expected=[],
        )
    )

    # --- Realistic paper edge case ---

    # TP: CANNOT_DETERMINE should NOT produce a finding (TN)
    cases.append(
        SyntheticCase(
            name="claim_support_cannot_determine",
            check_id="claim-support",
            description="CANNOT_DETERMINE verdict produces no finding",
            tex_content=build_realistic_paper(),
            metadata={
                "_claim_results": [
                    {
                        "key": "vaswani2017",
                        "line": 25,
                        "verdict": "CANNOT_DETERMINE",
                        "explanation": "Source text too short to assess",
                    },
                ],
            },
            expected=[],
        )
    )

    # TP: realistic paper with multiple claims, mixed verdicts
    cases.append(
        SyntheticCase(
            name="claim_support_realistic_mixed",
            check_id="claim-support",
            description="Realistic paper: 3 claims, 1 unsupported, 1 partial, 1 ok",
            tex_content=build_realistic_paper(),
            metadata={
                "_claim_results": [
                    {
                        "key": "devlin2019",
                        "line": 15,
                        "verdict": "NOT_SUPPORTED",
                        "explanation": "Source reports different baseline numbers",
                    },
                    {
                        "key": "graves2016",
                        "line": 30,
                        "verdict": "PARTIALLY_SUPPORTS",
                        "explanation": "ACT is related but not identical",
                    },
                    {
                        "key": "vaswani2017",
                        "line": 45,
                        "verdict": "SUPPORTS",
                        "explanation": "Source confirms attention mechanism details",
                    },
                ],
            },
            expected=[
                ExpectedFinding(rule_id="claim-support", context="devlin2019"),
                ExpectedFinding(rule_id="claim-support", context="graves2016"),
            ],
        )
    )

    # TP: wrong PDF downloaded — NOT_SUPPORTED because parsed content is
    # from an unrelated paper (GROBID parsed a different PDF).
    # Real case: yuan2026citeaudit PDF was "Did you this read right?" about
    # cognitive psychology instead of the CiteAudit citation benchmark.
    cases.append(
        SyntheticCase(
            name="claim_support_wrong_pdf_yuan",
            check_id="claim-support",
            description="Wrong PDF: explanation mentions unrelated domain (cognitive psychology vs citation audit)",
            tex_content=_make_doc(
                r"CiteAudit finds 35\% of LLM-generated citations are fabricated \cite{yuan2026}.",
                [
                    r"\bibitem{yuan2026} Yuan et al. CiteAudit: You Cited It, But Did You Read It? arXiv, 2026."
                ],
            ),
            metadata={
                "_claim_results": [
                    {
                        "key": "yuan2026",
                        "line": 5,
                        "verdict": "NOT_SUPPORTED",
                        "explanation": "Source discusses reading comprehension effects (transposed-letter effect) unrelated to citation verification",
                    },
                ],
            },
            expected=[
                ExpectedFinding(rule_id="claim-support", context="yuan2026"),
            ],
        )
    )

    # TP: wrong PDF — ansari2026compound parsed as eyeblink deception study
    cases.append(
        SyntheticCase(
            name="claim_support_wrong_pdf_ansari",
            check_id="claim-support",
            description="Wrong PDF: eyeblink study instead of compound AI errors paper",
            tex_content=_make_doc(
                r"Compound errors in peer-reviewed AI papers are increasing \cite{ansari2026}.",
                [
                    r"\bibitem{ansari2026} Ansari et al. Compound Deception in Elite Peer Review. arXiv, 2026."
                ],
            ),
            metadata={
                "_claim_results": [
                    {
                        "key": "ansari2026",
                        "line": 5,
                        "verdict": "NOT_SUPPORTED",
                        "explanation": "Source discusses eyeblink frequency modulation for deception detection, unrelated to peer review",
                    },
                ],
            },
            expected=[
                ExpectedFinding(rule_id="claim-support", context="ansari2026"),
            ],
        )
    )

    # --- Live vLLM cases: context narrowing ---

    # TN: bundled paragraph, but source supports the specific sub-claim.
    # Without context narrowing, the full paragraph confuses the verifier.
    # With narrowing, the verifier should focus and return SUPPORTS.
    cases.append(
        SyntheticCase(
            name="claim_support_narrowing_bundled_supports",
            check_id="claim-support",
            description="Bundled paragraph: source supports the specific sub-claim (context narrowing should help)",
            tex_content=_make_doc(
                r"Three observations: inverse scaling worsens with size. "
                r"MiniCheck (770M) achieves GPT-4 accuracy on grounding \cite{tang2024}. "
                r"Self-correction degrades output.",
                [r"\bibitem{tang2024} Tang et al. MiniCheck. 2024."],
            ),
            metadata={
                "_live_context": {
                    "key": "tang2024",
                    "context": (
                        "Three observations support this hypothesis. "
                        "Larger models exhibit stronger priors that override "
                        "explicit instructions---inverse scaling causes "
                        "performance to worsen with scale on certain tasks. "
                        "Small specialized models match or outperform frontier "
                        "models on fact-checking: MiniCheck (770M) achieves "
                        "GPT-4-level accuracy on document grounding. "
                        "Intrinsic self-correction typically degrades output "
                        "because the model shares the same blind spots."
                    ),
                    "sections": [
                        {
                            "title": "Results",
                            "text": (
                                "We evaluate MiniCheck, a 770M-parameter model "
                                "fine-tuned for document grounding. On our benchmark, "
                                "MiniCheck achieves 96.2% accuracy, matching GPT-4 "
                                "(96.5%) at 400x lower cost. The model uses a DeBERTa "
                                "backbone with task-specific classification heads."
                            ),
                        },
                    ],
                    "line": 5,
                },
            },
            expected=[],  # No finding expected — source supports the claim
        )
    )

    # TP: bundled paragraph, source does NOT support any sub-claim.
    # Narrowing should not rescue this — still NOT_SUPPORTED.
    cases.append(
        SyntheticCase(
            name="claim_support_narrowing_still_unsupported",
            check_id="claim-support",
            description="Bundled paragraph: source is unrelated even after narrowing",
            tex_content=_make_doc(
                r"Novel graphene methods improve yields \cite{wang2025}.",
                [r"\bibitem{wang2025} Wang et al. Novelty Detection. 2025."],
            ),
            metadata={
                "_live_context": {
                    "key": "wang2025",
                    "context": (
                        "Research assessment tools like PageRank measure influence. "
                        "Automated novelty detection methods identify whether a paper "
                        "introduces new concepts. "
                        "SciLint Score excludes novelty in favor of structural properties."
                    ),
                    "sections": [
                        {
                            "title": "Introduction",
                            "text": (
                                "We propose a framework for adaptive learning using "
                                "LLM agents for personalized education. Our system "
                                "profiles student cognitive states and uses multi-agent "
                                "architectures to deliver customized instruction."
                            ),
                        },
                    ],
                    "line": 5,
                },
            },
            expected=[
                ExpectedFinding(rule_id="claim-support", context="wang2025"),
            ],
        )
    )

    return cases


def gen_cite_purpose_cases() -> list[SyntheticCase]:
    """Generate cases for cite-purpose check.

    Tests the finding conversion logic using pre-computed purpose results
    (same pattern as claim-support).
    """
    cases: list[SyntheticCase] = []

    # TP: padding citation (no substantive use)
    cases.append(
        SyntheticCase(
            name="cite_purpose_padding",
            check_id="cite-purpose",
            description="Citation used as padding — no substantive connection",
            tex_content=_make_doc(
                r"Many studies exist \cite{smith2020}.",
                [_BIBITEMS[0]],
            ),
            metadata={
                "_purpose_results": [
                    {
                        "key": "smith2020",
                        "line": 5,
                        "context": "Many studies exist",
                        "citation_purpose": "padding",
                    },
                ],
            },
            expected=[
                ExpectedFinding(rule_id="cite-purpose", context="smith2020"),
            ],
        )
    )

    # TP: ornamental citation
    cases.append(
        SyntheticCase(
            name="cite_purpose_ornamental",
            check_id="cite-purpose",
            description="Citation is ornamental — dropped without changing meaning",
            tex_content=_make_doc(
                r"Deep learning is popular \cite{jones2021}.",
                [_BIBITEMS[1]],
            ),
            metadata={
                "_purpose_results": [
                    {
                        "key": "jones2021",
                        "line": 5,
                        "context": "Deep learning is popular",
                        "citation_purpose": "ornamental",
                    },
                ],
            },
            expected=[
                ExpectedFinding(rule_id="cite-purpose", context="jones2021"),
            ],
        )
    )

    # TP: multiple citations, one padding
    cases.append(
        SyntheticCase(
            name="cite_purpose_mixed",
            check_id="cite-purpose",
            description="Three citations: evidence, padding, method — one finding",
            tex_content=_make_doc(
                r"\cite{smith2020} supports X. Others \cite{jones2021}. "
                r"We follow \cite{wang2022}.",
                _BIBITEMS[:3],
            ),
            metadata={
                "_purpose_results": [
                    {
                        "key": "smith2020",
                        "line": 5,
                        "context": "supports X",
                        "citation_purpose": "evidence",
                    },
                    {
                        "key": "jones2021",
                        "line": 5,
                        "context": "Others",
                        "citation_purpose": "padding",
                    },
                    {
                        "key": "wang2022",
                        "line": 5,
                        "context": "We follow",
                        "citation_purpose": "method",
                    },
                ],
            },
            expected=[
                ExpectedFinding(rule_id="cite-purpose", context="jones2021"),
            ],
        )
    )

    # TN: all legitimate purposes
    cases.append(
        SyntheticCase(
            name="cite_purpose_all_legitimate",
            check_id="cite-purpose",
            description="All citation purposes are legitimate — no findings",
            tex_content=_make_doc(
                r"\cite{smith2020} supports X. We use \cite{jones2021}. "
                r"Unlike \cite{wang2022}, we show Y.",
                _BIBITEMS[:3],
            ),
            metadata={
                "_purpose_results": [
                    {
                        "key": "smith2020",
                        "line": 5,
                        "context": "supports X",
                        "citation_purpose": "evidence",
                    },
                    {
                        "key": "jones2021",
                        "line": 5,
                        "context": "We use",
                        "citation_purpose": "method",
                    },
                    {
                        "key": "wang2022",
                        "line": 5,
                        "context": "Unlike",
                        "citation_purpose": "contrast",
                    },
                ],
            },
            expected=[],
        )
    )

    return cases
