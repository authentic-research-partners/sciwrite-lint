"""Synthetic generators for reference-accuracy and reference-exists checks."""

from __future__ import annotations

from evals.synthetic_types import (
    ExpectedFinding,
    SyntheticCase,
    _BIBITEMS,
    _make_doc,
)
from evals.synthetic_templates import build_realistic_paper


def gen_reference_accuracy_cases() -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []

    # TP: title mismatch (possible fabrication)
    cases.append(
        SyntheticCase(
            name="ref_accuracy_title_mismatch",
            check_id="reference-accuracy",
            description="Bibitem title doesn't match canonical API title",
            tex_content=_make_doc(
                r"Deep learning is effective \cite{smith2020}.",
                [
                    r"\bibitem{smith2020} Smith, J. (2020). A study of deep learning. In ICML."
                ],
            ),
            metadata={
                "smith2020": {
                    "key": "smith2020",
                    "api_match": "mismatch",
                    "bibitem": {
                        "title": "A study of deep learning",
                        "authors": ["Smith, J."],
                        "year": "2020",
                        "venue": "ICML",
                    },
                    "canonical": {
                        "title": "Reinforcement learning for robotics control",
                        "authors": ["Smith, J."],
                        "year": 2020,
                        "venue": "ICML",
                        "retracted": False,
                    },
                },
            },
            expected=[
                ExpectedFinding(rule_id="reference-accuracy", context="smith2020"),
            ],
        )
    )

    # TP: year mismatch
    cases.append(
        SyntheticCase(
            name="ref_accuracy_year_mismatch",
            check_id="reference-accuracy",
            description="Bibitem year off by 3 years from canonical",
            tex_content=_make_doc(
                r"Prior work \cite{jones2021} showed this.",
                [r"\bibitem{jones2021} Jones, A. (2021). Neural nets. In NeurIPS."],
            ),
            metadata={
                "jones2021": {
                    "key": "jones2021",
                    "api_match": "verified",
                    "bibitem": {
                        "title": "Neural networks revisited",
                        "authors": ["Jones, A."],
                        "year": "2021",
                        "venue": "NeurIPS",
                    },
                    "canonical": {
                        "title": "Neural networks revisited",
                        "authors": ["Jones, A."],
                        "year": 2018,
                        "venue": "NeurIPS",
                        "retracted": False,
                    },
                },
            },
            expected=[
                ExpectedFinding(rule_id="reference-accuracy", context="jones2021"),
            ],
        )
    )

    # TP: author mismatch
    cases.append(
        SyntheticCase(
            name="ref_accuracy_author_mismatch",
            check_id="reference-accuracy",
            description="Bibitem authors don't match canonical",
            tex_content=_make_doc(
                r"As \cite{chen2019} demonstrated, attention works well.",
                [r"\bibitem{chen2019} Chen, R. (2019). Attention mechanisms. In AAAI."],
            ),
            metadata={
                "chen2019": {
                    "key": "chen2019",
                    "api_match": "verified",
                    "bibitem": {
                        "title": "Attention mechanisms",
                        "authors": ["Chen, R."],
                        "year": "2019",
                        "venue": "AAAI",
                    },
                    "canonical": {
                        "title": "Attention mechanisms",
                        "authors": ["Zhang, W.", "Li, H.", "Kumar, S."],
                        "year": 2019,
                        "venue": "AAAI",
                        "retracted": False,
                    },
                },
            },
            expected=[
                ExpectedFinding(rule_id="reference-accuracy", context="chen2019"),
            ],
        )
    )

    # TN: all fields match
    cases.append(
        SyntheticCase(
            name="ref_accuracy_clean",
            check_id="reference-accuracy",
            description="All metadata fields match canonical (no issues)",
            tex_content=_make_doc(
                r"As shown by \cite{smith2020}, deep learning works.",
                [
                    r"\bibitem{smith2020} Smith, J. (2020). A study of deep learning. In ICML."
                ],
            ),
            metadata={
                "smith2020": {
                    "key": "smith2020",
                    "api_match": "verified",
                    "bibitem": {
                        "title": "A study of deep learning",
                        "authors": ["Smith, J."],
                        "year": "2020",
                        "venue": "ICML",
                    },
                    "canonical": {
                        "title": "A study of deep learning",
                        "authors": ["Smith, J."],
                        "year": 2020,
                        "venue": "ICML",
                        "retracted": False,
                    },
                },
            },
            expected=[],
        )
    )

    # TN: not-found references are skipped (not accuracy issues)
    cases.append(
        SyntheticCase(
            name="ref_accuracy_not_found_skipped",
            check_id="reference-accuracy",
            description="not_found references should not trigger accuracy check",
            tex_content=_make_doc(
                r"Some work \cite{unknown2023}.",
                [
                    r"\bibitem{unknown2023} Unknown, A. (2023). Mystery paper. In Workshop."
                ],
            ),
            metadata={
                "unknown2023": {
                    "key": "unknown2023",
                    "api_match": "not_found",
                    "bibitem": {
                        "title": "Mystery paper",
                        "authors": ["Unknown, A."],
                        "year": "2023",
                    },
                    "canonical": {},
                },
            },
            expected=[],
        )
    )

    # --- Realistic paper edge cases ---

    # TN: venue abbreviation vs full name (should match via fuzzy)
    cases.append(
        SyntheticCase(
            name="ref_accuracy_venue_abbreviation",
            check_id="reference-accuracy",
            description="Venue variant (NAACL-HLT vs NAACL) — should pass fuzzy match",
            tex_content=build_realistic_paper(),
            metadata={
                "devlin2019": {
                    "key": "devlin2019",
                    "api_match": "verified",
                    "bibitem": {
                        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                        "authors": ["Devlin, J."],
                        "year": "2019",
                        "venue": "NAACL-HLT",
                    },
                    "canonical": {
                        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                        "authors": ["Devlin, J."],
                        "year": 2019,
                        "venue": "NAACL",
                        "retracted": False,
                    },
                },
            },
            expected=[],
        )
    )

    # TN: year off by exactly 1 (within tolerance)
    cases.append(
        SyntheticCase(
            name="ref_accuracy_year_tolerance",
            check_id="reference-accuracy",
            description="Year off by 1 (within tolerance) — should pass",
            tex_content=build_realistic_paper(),
            metadata={
                "devlin2019": {
                    "key": "devlin2019",
                    "api_match": "verified",
                    "bibitem": {
                        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                        "authors": ["Devlin, J."],
                        "year": "2019",
                        "venue": "NAACL",
                    },
                    "canonical": {
                        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                        "authors": ["Devlin, J."],
                        "year": 2018,
                        "venue": "NAACL",
                        "retracted": False,
                    },
                },
            },
            expected=[],
        )
    )

    return cases


def gen_reference_exists_cases() -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []

    # TP: reference not found in any API
    cases.append(
        SyntheticCase(
            name="ref_exists_not_found",
            check_id="reference-exists",
            description="Reference not found in any API (T3)",
            tex_content=_make_doc(
                r"As shown by \cite{ghost2023}, this works.",
                [r"\bibitem{ghost2023} Ghost, A. (2023). Nonexistent paper. Workshop."],
            ),
            metadata={
                "ghost2023": {
                    "key": "ghost2023",
                    "api_match": "not_found",
                    "bibitem": {"title": "Nonexistent paper"},
                    "canonical": {},
                    "issues": [],
                },
            },
            expected=[
                ExpectedFinding(rule_id="reference-exists", context="ghost2023"),
            ],
        )
    )

    # TP: dead URL for web resource
    cases.append(
        SyntheticCase(
            name="ref_exists_dead_url",
            check_id="reference-exists",
            description="Web resource with dead URL",
            tex_content=_make_doc(
                r"See the documentation \cite{webdead2024}.",
                [
                    r"\bibitem{webdead2024} Dead Website. \url{https://example.com/gone}."
                ],
            ),
            metadata={
                "webdead2024": {
                    "key": "webdead2024",
                    "api_match": "",
                    "bibitem": {"title": "Dead Website"},
                    "canonical": {},
                    "issues": ["Dead URL: https://example.com/gone returned 404"],
                },
            },
            expected=[
                ExpectedFinding(rule_id="reference-exists", context="webdead2024"),
            ],
        )
    )

    # TP: content extraction failed
    cases.append(
        SyntheticCase(
            name="ref_exists_extraction_failed",
            check_id="reference-exists",
            description="URL alive but content extraction failed",
            tex_content=_make_doc(
                r"See \cite{webfail2024}.",
                [r"\bibitem{webfail2024} Some Site. \url{https://example.com}."],
            ),
            metadata={
                "webfail2024": {
                    "key": "webfail2024",
                    "api_match": "",
                    "bibitem": {"title": "Some Site"},
                    "canonical": {},
                    "issues": ["Content extraction failed for https://example.com"],
                },
            },
            expected=[
                ExpectedFinding(rule_id="reference-exists", context="webfail2024"),
            ],
        )
    )

    # TN: verified reference
    cases.append(
        SyntheticCase(
            name="ref_exists_verified",
            check_id="reference-exists",
            description="Reference verified in APIs (no issues)",
            tex_content=_make_doc(
                r"As shown by \cite{smith2020}, this works.",
                [_BIBITEMS[0]],
            ),
            metadata={
                "smith2020": {
                    "key": "smith2020",
                    "api_match": "verified",
                    "bibitem": {"title": "A study of deep learning"},
                    "canonical": {"title": "A study of deep learning"},
                    "issues": [],
                },
            },
            expected=[],
        )
    )

    # TN: multiple references, all found
    cases.append(
        SyntheticCase(
            name="ref_exists_all_found",
            check_id="reference-exists",
            description="Multiple references, all found in APIs",
            tex_content=_make_doc(
                r"\cite{smith2020} and \cite{jones2021} agree.",
                _BIBITEMS[:2],
            ),
            metadata={
                "smith2020": {
                    "key": "smith2020",
                    "api_match": "verified",
                    "bibitem": {},
                    "canonical": {},
                    "issues": [],
                },
                "jones2021": {
                    "key": "jones2021",
                    "api_match": "verified",
                    "bibitem": {},
                    "canonical": {},
                    "issues": [],
                },
            },
            expected=[],
        )
    )

    # --- Realistic paper edge case ---

    # TP: realistic paper, one of many references is not found
    cases.append(
        SyntheticCase(
            name="ref_exists_realistic_mixed",
            check_id="reference-exists",
            description="Realistic paper: 3 refs verified, 1 not found",
            tex_content=build_realistic_paper(),
            metadata={
                "vaswani2017": {
                    "key": "vaswani2017",
                    "api_match": "verified",
                    "bibitem": {},
                    "canonical": {},
                    "issues": [],
                },
                "devlin2019": {
                    "key": "devlin2019",
                    "api_match": "verified",
                    "bibitem": {},
                    "canonical": {},
                    "issues": [],
                },
                "graves2016": {
                    "key": "graves2016",
                    "api_match": "verified",
                    "bibitem": {},
                    "canonical": {},
                    "issues": [],
                },
                "socher2013": {
                    "key": "socher2013",
                    "api_match": "not_found",
                    "bibitem": {"title": "Recursive Deep Models"},
                    "canonical": {},
                    "issues": [],
                },
            },
            expected=[
                ExpectedFinding(rule_id="reference-exists", context="socher2013"),
            ],
        )
    )

    return cases


def gen_retracted_cite_cases() -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []

    # TP: retracted paper
    cases.append(
        SyntheticCase(
            name="retracted_cite_retraction",
            check_id="retracted-cite",
            description="Cited paper has been retracted",
            tex_content=_make_doc(
                r"Following \cite{wang2022}, we apply transformers.",
                [r"\bibitem{wang2022} Wang, L. (2022). Transformers for NLP. In ACL."],
            ),
            metadata={
                "wang2022": {
                    "key": "wang2022",
                    "api_match": "verified",
                    "bibitem": {
                        "title": "Transformers for NLP",
                        "authors": ["Wang, L."],
                        "year": "2022",
                        "venue": "ACL",
                    },
                    "canonical": {
                        "title": "Transformers for NLP",
                        "authors": ["Wang, L."],
                        "year": 2022,
                        "venue": "ACL",
                        "retracted": True,
                        "retraction_status": {
                            "nature": "Retraction",
                            "reason": "Fabrication",
                            "date": "2024-01-01",
                            "source": "retraction_watch",
                        },
                    },
                },
            },
            expected=[
                ExpectedFinding(rule_id="retracted-cite", context="wang2022"),
            ],
        )
    )

    # TN: clean reference (no retraction)
    cases.append(
        SyntheticCase(
            name="retracted_cite_clean",
            check_id="retracted-cite",
            description="Normal reference — no retraction status",
            tex_content=_make_doc(
                r"As shown by \cite{smith2020}, this works.",
                _BIBITEMS[:1],
            ),
            metadata={
                "smith2020": {
                    "key": "smith2020",
                    "api_match": "verified",
                    "bibitem": {"title": "A study"},
                    "canonical": {
                        "title": "A study of deep learning",
                        "retracted": False,
                    },
                },
            },
            expected=[],
        )
    )

    return cases
