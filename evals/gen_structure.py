"""Synthetic generators for cross-section-consistency and structure-promises checks."""

from __future__ import annotations

from evals.synthetic_types import (
    ExpectedFinding,
    SyntheticCase,
    _BIBITEMS,
    _make_sectioned_doc,
)
from evals.synthetic_templates import (
    CLEAN_ABSTRACT,
    CLEAN_CONCLUSION,
    CLEAN_INTRO,
    build_realistic_paper,
)


def gen_cross_section_cases() -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []

    # TP: number contradiction between abstract and results
    cases.append(
        SyntheticCase(
            name="cross_section_number_drift",
            check_id="cross-section-consistency",
            description="Abstract says 45% improvement, results say 23%",
            tex_content=_make_sectioned_doc(
                abstract=(
                    "We present a novel approach to text classification that achieves "
                    "a 45\\% improvement over the baseline on the benchmark dataset."
                ),
                intro=(
                    "Text classification remains a fundamental NLP task. "
                    "Prior work \\cite{smith2020} established strong baselines. "
                    "We propose a new method that significantly outperforms them."
                ),
                methods=(
                    "Our method uses a transformer encoder with a custom attention "
                    "mechanism. We train on 100K examples for 50 epochs with a "
                    "learning rate of 1e-4."
                ),
                results=(
                    "Our model achieves 78.3\\% accuracy on the test set, compared "
                    "to 63.5\\% for the baseline, representing a 23\\% improvement. "
                    "The difference is statistically significant (p < 0.01)."
                ),
                conclusion=(
                    "We presented a new text classification approach that "
                    "substantially outperforms prior work."
                ),
                bibitems=[_BIBITEMS[0]],
            ),
            expected=[
                ExpectedFinding(
                    rule_id="cross-section-consistency",
                    context="45",
                ),
            ],
        )
    )

    # TP: framing contradiction
    cases.append(
        SyntheticCase(
            name="cross_section_framing",
            check_id="cross-section-consistency",
            description="Abstract claims unsupervised, methods describe supervised training",
            tex_content=_make_sectioned_doc(
                abstract=(
                    "We introduce an unsupervised method for sentiment analysis "
                    "that requires no labeled data and achieves competitive results."
                ),
                intro=(
                    "Sentiment analysis is crucial for understanding user opinions. "
                    "Unsupervised methods are attractive because labeled data is expensive."
                ),
                methods=(
                    "We fine-tune a BERT model on 50,000 labeled sentiment examples "
                    "from the SST-2 dataset. The model is trained with cross-entropy "
                    "loss using supervised learning for 10 epochs."
                ),
                results=(
                    "Our supervised fine-tuning achieves 93.2\\% accuracy on SST-2, "
                    "outperforming prior supervised approaches."
                ),
                conclusion=(
                    "Our method shows that supervised fine-tuning of pre-trained "
                    "models is effective for sentiment analysis."
                ),
                bibitems=[_BIBITEMS[0]],
            ),
            expected=[
                ExpectedFinding(
                    rule_id="cross-section-consistency",
                    context="unsupervised",
                ),
            ],
        )
    )

    # TN: consistent document
    cases.append(
        SyntheticCase(
            name="cross_section_clean",
            check_id="cross-section-consistency",
            description="All sections are internally consistent",
            tex_content=_make_sectioned_doc(
                abstract=(
                    "We present a supervised text classification method that "
                    "achieves 92\\% accuracy on the benchmark dataset, a 15\\% "
                    "improvement over the previous best."
                ),
                intro=(
                    "Text classification is a core NLP task. Recent supervised "
                    "approaches using transformers have shown strong results. "
                    "We build on this work with an improved architecture."
                ),
                methods=(
                    "We use a transformer encoder with 12 layers, trained on "
                    "the standard benchmark with supervised cross-entropy loss."
                ),
                results=(
                    "Our model achieves 92\\% accuracy, compared to 80\\% for "
                    "the baseline, a 15\\% relative improvement."
                ),
                conclusion=(
                    "We demonstrated that our supervised transformer approach "
                    "achieves 92\\% accuracy, improving over prior work by 15\\%."
                ),
                bibitems=[_BIBITEMS[0]],
            ),
            expected=[],
        )
    )

    # --- Realistic paper cases ---

    # TP: abstract says 45% improvement, results say 5.4%
    abstract_inflated = CLEAN_ABSTRACT.replace("5.4\\%", "45.8\\%")
    cases.append(
        SyntheticCase(
            name="cross_section_realistic_number_drift",
            check_id="cross-section-consistency",
            description="Realistic paper: abstract inflates improvement (45.8% vs 5.4%)",
            tex_content=build_realistic_paper(abstract=abstract_inflated),
            expected=[
                ExpectedFinding(rule_id="cross-section-consistency", context="45"),
            ],
        )
    )

    # TP: abstract says unsupervised, methods describe supervised training
    abstract_unsup = (
        "We present AdaptiveAttend, an unsupervised attention mechanism "
        "for text classification that requires no labeled data. Our method "
        "achieves competitive accuracy on SST-2 and IMDB benchmarks using "
        "only unlabeled text, demonstrating that task-specific supervision "
        "is unnecessary for high-quality text classification."
    )
    cases.append(
        SyntheticCase(
            name="cross_section_realistic_framing",
            check_id="cross-section-consistency",
            description="Realistic paper: abstract says unsupervised, methods are supervised",
            tex_content=build_realistic_paper(abstract=abstract_unsup),
            expected=[
                ExpectedFinding(
                    rule_id="cross-section-consistency", context="unsupervised"
                ),
            ],
        )
    )

    # TN: realistic paper with consistent sections
    cases.append(
        SyntheticCase(
            name="cross_section_realistic_clean",
            check_id="cross-section-consistency",
            description="Realistic paper with internally consistent sections",
            tex_content=build_realistic_paper(),
            expected=[],
        )
    )

    return cases


def gen_structure_promises_cases() -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []

    # TP: claims 3 contributions, delivers 2
    cases.append(
        SyntheticCase(
            name="structure_promises_overclaim",
            check_id="structure-promises",
            description="Intro promises 3 contributions but paper delivers only 2",
            tex_content=_make_sectioned_doc(
                abstract="We make three contributions to the field of NLP.",
                intro=(
                    "In this paper, we make three key contributions:\n"
                    "\\begin{enumerate}\n"
                    "\\item A novel architecture for text classification.\n"
                    "\\item A new benchmark dataset with 100K examples.\n"
                    "\\item A theoretical analysis of convergence guarantees.\n"
                    "\\end{enumerate}"
                ),
                methods=(
                    "We describe our transformer-based architecture. "
                    "The model uses multi-head attention with 8 heads."
                ),
                results=(
                    "We evaluate on our new benchmark dataset. "
                    "Our model achieves state-of-the-art results."
                ),
                conclusion=(
                    "We presented a new architecture and benchmark dataset. "
                    "The theoretical analysis is left for future work."
                ),
                bibitems=[_BIBITEMS[0]],
            ),
            expected=[
                ExpectedFinding(rule_id="structure-promises", context="3"),
            ],
        )
    )

    # TN: claims match delivery
    cases.append(
        SyntheticCase(
            name="structure_promises_clean",
            check_id="structure-promises",
            description="Intro promises 2 contributions and delivers both",
            tex_content=_make_sectioned_doc(
                abstract="We make two contributions to text classification.",
                intro=(
                    "Our contributions are twofold:\n"
                    "\\begin{enumerate}\n"
                    "\\item A new transformer architecture for classification.\n"
                    "\\item An extensive evaluation on three benchmarks.\n"
                    "\\end{enumerate}"
                ),
                methods=(
                    "We describe our architecture in detail. "
                    "The model has 6 layers with 512-dimensional embeddings."
                ),
                results=(
                    "We evaluate on SST-2, IMDB, and Yelp. Our model "
                    "achieves new state-of-the-art on all three benchmarks."
                ),
                conclusion=(
                    "We presented a new architecture and evaluated it "
                    "extensively on three benchmarks, achieving strong results."
                ),
                bibitems=[_BIBITEMS[0]],
            ),
            expected=[],
        )
    )

    # --- Realistic paper cases ---

    # TP: claims 3 contributions but conclusion admits one is future work
    intro_overclaim = CLEAN_INTRO.replace(
        "\\item We present extensive experiments",
        "\\item A theoretical convergence proof for the adaptive mechanism "
        "(Section~\\ref{sec:conclusion}).\n"
        "\\item We present extensive experiments",
    ).replace("three key contributions", "four key contributions")
    conclusion_missing = CLEAN_CONCLUSION + (
        "\n\nThe theoretical convergence analysis is deferred to future work "
        "due to space constraints."
    )
    cases.append(
        SyntheticCase(
            name="structure_promises_realistic_overclaim",
            check_id="structure-promises",
            description="Realistic paper: claims 4 contributions, one deferred to future work",
            tex_content=build_realistic_paper(
                intro=intro_overclaim, conclusion=conclusion_missing
            ),
            expected=[
                ExpectedFinding(rule_id="structure-promises", context="4"),
            ],
        )
    )

    # TN: realistic paper with matching contributions
    cases.append(
        SyntheticCase(
            name="structure_promises_realistic_clean",
            check_id="structure-promises",
            description="Realistic paper with 3 contributions all delivered",
            tex_content=build_realistic_paper(),
            expected=[],
        )
    )

    return cases
