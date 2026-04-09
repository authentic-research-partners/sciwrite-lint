"""Synthetic generators for dangling-cite and dangling-ref checks."""

from __future__ import annotations

from evals.synthetic_types import (
    ExpectedFinding,
    SyntheticCase,
    _BIBITEMS,
    _make_doc,
)
from evals.synthetic_templates import (
    CLEAN_INTRO,
    CLEAN_METHODS_TRAINING,
    CLEAN_RESULTS,
    build_realistic_paper,
)


def gen_dangling_cite_cases() -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []

    # TP: single orphan cite
    cases.append(
        SyntheticCase(
            name="dangling_cite_single",
            check_id="dangling-cite",
            description="One citation with no matching bibitem",
            tex_content=_make_doc(
                r"Deep learning has advanced rapidly \cite{smith2020}. "
                r"Recent work shows promise \cite{phantom2023}.",
                [_BIBITEMS[0]],
            ),
            expected=[ExpectedFinding(rule_id="dangling-cite", context="phantom2023")],
        )
    )

    # TP: multiple orphan cites
    cases.append(
        SyntheticCase(
            name="dangling_cite_multiple",
            check_id="dangling-cite",
            description="Three citations, two orphaned",
            tex_content=_make_doc(
                r"Work by \cite{smith2020} builds on \cite{ghost2021} "
                r"and \cite{missing2022}.",
                [_BIBITEMS[0]],
            ),
            expected=[
                ExpectedFinding(rule_id="dangling-cite", context="ghost2021"),
                ExpectedFinding(rule_id="dangling-cite", context="missing2022"),
            ],
        )
    )

    # TN: all cites have bibitems
    cases.append(
        SyntheticCase(
            name="dangling_cite_clean",
            check_id="dangling-cite",
            description="All citations have matching bibitems (no issues)",
            tex_content=_make_doc(
                r"As shown by \cite{smith2020} and \cite{jones2021}, "
                r"transformers \cite{wang2022} improved results.",
                _BIBITEMS[:3],
            ),
            expected=[],  # no findings expected
        )
    )

    # TN: multi-key cite
    cases.append(
        SyntheticCase(
            name="dangling_cite_multikey_clean",
            check_id="dangling-cite",
            description="Multi-key cite where all keys exist",
            tex_content=_make_doc(
                r"Several studies \cite{smith2020,jones2021} confirm this.",
                _BIBITEMS[:2],
            ),
            expected=[],
        )
    )

    # TP: multi-key cite with one orphan
    cases.append(
        SyntheticCase(
            name="dangling_cite_multikey_orphan",
            check_id="dangling-cite",
            description="Multi-key cite where one key is orphaned",
            tex_content=_make_doc(
                r"Several studies \cite{smith2020,nonexistent2023} confirm this.",
                [_BIBITEMS[0]],
            ),
            expected=[
                ExpectedFinding(rule_id="dangling-cite", context="nonexistent2023")
            ],
        )
    )

    # --- Realistic paper cases ---

    # TP: orphan cite in realistic paper
    intro_with_orphan = CLEAN_INTRO.replace(
        r"\cite{graves2016}",
        r"\cite{graves2016}. Recent findings \cite{hallucinated2025}",
    )
    cases.append(
        SyntheticCase(
            name="dangling_cite_realistic_orphan",
            check_id="dangling-cite",
            description="Realistic paper with one orphan cite in introduction",
            tex_content=build_realistic_paper(intro=intro_with_orphan),
            expected=[
                ExpectedFinding(rule_id="dangling-cite", context="hallucinated2025"),
            ],
        )
    )

    # TP: cite in comment should NOT be detected (TN for that cite),
    # but a real orphan alongside it should be detected
    intro_with_comment = CLEAN_INTRO + (
        "\n% Removed: \\cite{commented_out2024}\n"
        "Furthermore, \\cite{realorphan2025} showed improvements."
    )
    cases.append(
        SyntheticCase(
            name="dangling_cite_comment_vs_real",
            check_id="dangling-cite",
            description="Commented cite ignored, real orphan detected",
            tex_content=build_realistic_paper(intro=intro_with_comment),
            expected=[
                ExpectedFinding(rule_id="dangling-cite", context="realorphan2025"),
            ],
        )
    )

    # TN: realistic paper with all cites matching
    cases.append(
        SyntheticCase(
            name="dangling_cite_realistic_clean",
            check_id="dangling-cite",
            description="Realistic paper with all citations matching bibitems",
            tex_content=build_realistic_paper(),
            expected=[],
        )
    )

    return cases


def gen_dangling_ref_cases() -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []

    # TP: ref to non-existent label
    cases.append(
        SyntheticCase(
            name="dangling_ref_single",
            check_id="dangling-ref",
            description="One ref with no matching label",
            tex_content=_make_doc(
                r"\section{Intro}\label{sec:intro}"
                "\n"
                r"As shown in Figure~\ref{fig:nonexistent}, results improve."
                "\n"
                r"See Section~\ref{sec:intro} for details.",
                [_BIBITEMS[0]],
            ),
            expected=[
                ExpectedFinding(rule_id="dangling-ref", context="fig:nonexistent")
            ],
        )
    )

    # TP: multiple dangling refs
    cases.append(
        SyntheticCase(
            name="dangling_ref_multiple",
            check_id="dangling-ref",
            description="Two refs with no matching labels",
            tex_content=_make_doc(
                r"\section{Intro}\label{sec:intro}"
                "\n"
                r"See Table~\ref{tab:ghost} and Figure~\ref{fig:phantom}."
                "\n"
                r"Section~\ref{sec:intro} describes our approach.",
                [_BIBITEMS[0]],
            ),
            expected=[
                ExpectedFinding(rule_id="dangling-ref", context="tab:ghost"),
                ExpectedFinding(rule_id="dangling-ref", context="fig:phantom"),
            ],
        )
    )

    # TN: all refs have labels
    cases.append(
        SyntheticCase(
            name="dangling_ref_clean",
            check_id="dangling-ref",
            description="All refs have matching labels (no issues)",
            tex_content=_make_doc(
                r"\section{Intro}\label{sec:intro}"
                "\n"
                r"\begin{figure}\label{fig:results}\end{figure}"
                "\n"
                r"See Figure~\ref{fig:results} and Section~\ref{sec:intro}.",
                [_BIBITEMS[0]],
            ),
            expected=[],
        )
    )

    # TP: eqref to missing equation
    cases.append(
        SyntheticCase(
            name="dangling_ref_eqref",
            check_id="dangling-ref",
            description="eqref to non-existent equation label",
            tex_content=_make_doc(
                r"\section{Methods}\label{sec:methods}"
                "\n"
                r"We use Equation~\eqref{eq:missing} to compute the loss.",
                [_BIBITEMS[0]],
            ),
            expected=[ExpectedFinding(rule_id="dangling-ref", context="eq:missing")],
        )
    )

    # --- Realistic paper cases ---

    # TP: ref to non-existent figure in realistic paper
    results_with_phantom = CLEAN_RESULTS.replace(
        r"See Figure~\ref{fig:heads}",
        r"See Figure~\ref{fig:ablation}",
    )
    cases.append(
        SyntheticCase(
            name="dangling_ref_realistic_phantom_fig",
            check_id="dangling-ref",
            description="Realistic paper with ref to non-existent figure",
            tex_content=build_realistic_paper(results=results_with_phantom),
            expected=[
                ExpectedFinding(rule_id="dangling-ref", context="fig:ablation"),
            ],
        )
    )

    # TP: ref to non-existent equation
    methods_with_phantom_eq = CLEAN_METHODS_TRAINING.replace(
        r"$\mathcal{L}_{\text{efficiency}} = \frac{1}{N} \sum_{i=1}^N c_i$",
        r"$\mathcal{L}_{\text{efficiency}}$ (see Equation~\eqref{eq:regularizer})",
    )
    cases.append(
        SyntheticCase(
            name="dangling_ref_realistic_phantom_eq",
            check_id="dangling-ref",
            description="Realistic paper with eqref to non-existent equation",
            tex_content=build_realistic_paper(methods_training=methods_with_phantom_eq),
            expected=[
                ExpectedFinding(rule_id="dangling-ref", context="eq:regularizer"),
            ],
        )
    )

    # TN: realistic paper with all refs matching
    cases.append(
        SyntheticCase(
            name="dangling_ref_realistic_clean",
            check_id="dangling-ref",
            description="Realistic paper with all refs matching labels",
            tex_content=build_realistic_paper(),
            expected=[],
        )
    )

    return cases
