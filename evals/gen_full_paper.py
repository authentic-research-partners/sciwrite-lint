"""Synthetic generators for full-paper consistency checks.

Scenario-based design: each scenario is ONE paper with multiple injected
errors.  All checks run on that paper in a single vLLM batch (APC caches
the shared prefix).  The eval caches LLM results by content hash, so
multiple SyntheticCase objects sharing the same tex_content only trigger
one vLLM round-trip.

Covers 12 checks: 7 mechanical/numerical + 5 figure.  Reasoning-heavy
checks (contradictions, scope, claims-vs-delivery) stay with the existing
pairwise approach (cross-section-consistency, structure-promises).
"""

from __future__ import annotations

from evals.synthetic_templates import (
    CLEAN_METHODS_ARCH,
    CLEAN_RESULTS,
    build_realistic_paper,
)
from evals.synthetic_types import (
    ExpectedFinding,
    SyntheticCase,
    _BIBITEMS,
    _make_sectioned_doc,
)

# The 12 full-paper check IDs (7 mechanical + 5 figure)
_MECHANICAL_CHECK_IDS = [
    "numbers-vs-tables",
    "percentages-sum",
    "sample-size-consistency",
    "arithmetic-consistency",
    "causal-language-audit",
    "abstract-body-alignment",
    "statistical-reporting",
]

_FIGURE_CHECK_IDS = [
    "caption-vs-content",
    "text-vs-figure",
    "axis-label-consistency",
    "figure-data-vs-table",
    # figure-numbering excluded: label-to-number mapping between
    # VL descriptions (fig:X labels) and stripped text (Figure N)
    # is unreliable. Needs deterministic pre-processing, not LLM.
]

_ALL_CHECK_IDS = _MECHANICAL_CHECK_IDS + _FIGURE_CHECK_IDS


def _scenario_cases(
    scenario_name: str,
    tex_content: str,
    description: str,
    expected_by_check: dict[str, list[ExpectedFinding]],
    figure_descriptions: str = "",
    check_ids: list[str] | None = None,
) -> list[SyntheticCase]:
    """Expand a scenario into one SyntheticCase per check.

    Checks in *expected_by_check* become TP cases; others become TN.
    All share *tex_content* — LLM cache ensures one vLLM run.

    If *figure_descriptions* is set, it is injected into the system prompt
    via a temp workspace (handled by the eval runner).
    """
    ids = check_ids or _ALL_CHECK_IDS
    cases: list[SyntheticCase] = []
    for check_id in ids:
        expected = expected_by_check.get(check_id, [])
        tag = "tp" if expected else "tn"
        cases.append(
            SyntheticCase(
                name=f"{scenario_name}_{check_id.replace('-', '_')}_{tag}",
                check_id=check_id,
                description=(
                    f"{description} "
                    f"[{check_id}: {'should fire' if expected else 'should be silent'}]"
                ),
                tex_content=tex_content,
                expected=expected,
                figure_descriptions=figure_descriptions,
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Scenario 1: Clean paper (TN for all)
# ---------------------------------------------------------------------------

_CLEAN_PAPER = build_realistic_paper()


def _gen_clean_scenario() -> list[SyntheticCase]:
    return _scenario_cases(
        "s1_clean",
        _CLEAN_PAPER,
        "Clean realistic paper with no injected errors",
        {},
    )


# ---------------------------------------------------------------------------
# Scenario 2: Numerical errors (numbers-vs-tables, percentages-sum, arithmetic)
# ---------------------------------------------------------------------------

_NUMERICAL_RESULTS = (
    "Table~\\ref{tab:results} summarizes our main results. AdaptiveAttend "
    "achieves 92.3\\% accuracy on SST-2, surpassing BERT-base (87.6\\%) by "
    "a margin of 4.7 percentage points, a 5.4\\% relative improvement. "
    "On IMDB, our model reaches 94.1\\%, compared to 91.3\\% for BERT-base. "
    "Importantly, AdaptiveAttend uses an average of 5.6 out of 8 attention "
    "heads per input, achieving 30\\% parameter reduction during inference.\n\n"
    "\\begin{table}[t]\n"
    "\\centering\n"
    "\\caption{Test accuracy (\\%) on classification benchmarks. "
    "Best results in \\textbf{bold}.}\n"
    "\\label{tab:results}\n"
    "\\begin{tabular}{lccc}\n"
    "\\toprule\n"
    "Model & SST-2 & IMDB & XNLI (avg) \\\\\n"
    "\\midrule\n"
    "BERT-base & 87.6 & 91.3 & 73.2 \\\\\n"
    "DistilBERT & 85.2 & 89.4 & 70.8 \\\\\n"
    "Universal Transformer & 88.1 & 91.7 & 74.0 \\\\\n"
    "\\midrule\n"
    "AdaptiveAttend (ours) & \\textbf{90.8} & \\textbf{94.1} & "
    "\\textbf{76.5} \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n\n"
    "Error analysis reveals that 45\\% of errors are lexical, "
    "38\\% are syntactic, and 24\\% are semantic in nature.\n\n"
    "We split participants into Group A (n=120) and Group B (n=125), "
    "for a total of 250 participants in the evaluation.\n\n"
    "On XNLI, AdaptiveAttend achieves 76.5\\% average accuracy across "
    "15 languages, outperforming BERT-base by 3.3 points."
)

_NUMERICAL_PAPER = build_realistic_paper(results=_NUMERICAL_RESULTS)


def _gen_numerical_scenario() -> list[SyntheticCase]:
    return _scenario_cases(
        "s2_numerical",
        _NUMERICAL_PAPER,
        "Paper with table mismatch (92.3 vs 90.8), bad percentages (107%), wrong arithmetic (120+125=250)",
        {
            "numbers-vs-tables": [
                ExpectedFinding(rule_id="numbers-vs-tables", context="92.3"),
            ],
            "percentages-sum": [
                ExpectedFinding(rule_id="percentages-sum", context=""),
            ],
            "arithmetic-consistency": [
                ExpectedFinding(rule_id="arithmetic-consistency", context="250"),
            ],
        },
    )


# ---------------------------------------------------------------------------
# Scenario 3: Sample size mismatch
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS = CLEAN_RESULTS.replace(
    "On XNLI, AdaptiveAttend achieves 76.5\\%",
    "Of the N=312 survey respondents, 89\\% preferred our interface. "
    "On XNLI, AdaptiveAttend achieves 76.5\\%",
)

_SAMPLE_METHODS = (
    "We recruited N=245 participants from three universities. "
    "AdaptiveAttend modifies the standard multi-head attention mechanism by "
    "introducing a lightweight gating network that determines which attention "
    "heads to activate for each input."
)

_SAMPLE_PAPER = build_realistic_paper(
    methods_arch=_SAMPLE_METHODS,
    results=_SAMPLE_RESULTS,
)


def _gen_sample_scenario() -> list[SyntheticCase]:
    return _scenario_cases(
        "s3_sample",
        _SAMPLE_PAPER,
        "Paper with N mismatch (245 in methods vs 312 in results)",
        {
            "sample-size-consistency": [
                ExpectedFinding(rule_id="sample-size-consistency", context="245"),
            ],
        },
    )


# ---------------------------------------------------------------------------
# Scenario 4: Causal language + abstract overclaim
# ---------------------------------------------------------------------------

_OVERREACH_PAPER = _make_sectioned_doc(
    abstract=(
        "We demonstrate that our attention mechanism causes improved "
        "classification accuracy. Our method achieves 92.3\\% on SST-2 "
        "and works across all NLP tasks and all languages."
    ),
    intro=(
        "Attention mechanisms are central to modern NLP. "
        "We propose an adaptive approach and study its effects."
    ),
    methods=(
        "We collect observational data from SST-2 usage logs. "
        "No controlled experiment or intervention is performed. "
        "We measure correlation between attention head usage and accuracy."
    ),
    results=(
        "We observe a positive correlation between adaptive head count and "
        "accuracy (r=0.42, p<0.001) on SST-2. No other datasets are evaluated."
    ),
    conclusion=(
        "Our correlational results suggest that adaptive attention is "
        "associated with better performance on SST-2."
    ),
    bibitems=[_BIBITEMS[0]],
)


def _gen_overreach_scenario() -> list[SyntheticCase]:
    return _scenario_cases(
        "s4_overreach",
        _OVERREACH_PAPER,
        "Paper with causal claims in observational study and abstract overclaim",
        {
            "causal-language-audit": [
                ExpectedFinding(rule_id="causal-language-audit", context="cause"),
            ],
            "abstract-body-alignment": [
                ExpectedFinding(rule_id="abstract-body-alignment", context=""),
            ],
        },
    )


# ---------------------------------------------------------------------------
# Scenario 5: Statistical misinterpretation
# ---------------------------------------------------------------------------

_STATS_PAPER = _make_sectioned_doc(
    abstract="We compare two classifiers on a benchmark dataset.",
    intro=(
        "Statistical significance testing is important for evaluating "
        "classifier performance. We use a paired t-test with alpha=0.05."
    ),
    methods=(
        "We train two models on SST-2 and compare accuracy. "
        "We use a paired t-test with alpha=0.05."
    ),
    results=(
        "Our model achieves 92.3\\% accuracy compared to 87.6\\% for the "
        "baseline. The difference was not statistically significant "
        "(t(98)=4.2, p=0.003)."
    ),
    conclusion=("The methods performed comparably on this benchmark."),
    bibitems=[_BIBITEMS[0]],
)


def _gen_stats_scenario() -> list[SyntheticCase]:
    return _scenario_cases(
        "s5_stats",
        _STATS_PAPER,
        "Paper with p=0.003 described as 'not statistically significant'",
        {
            "statistical-reporting": [
                ExpectedFinding(rule_id="statistical-reporting", context="0.003"),
            ],
        },
    )


# ---------------------------------------------------------------------------
# Scenario 6: Figure caption mismatch + text-vs-figure contradiction
# ---------------------------------------------------------------------------

# Paper text references figures with specific claims
_FIGURE_PAPER = build_realistic_paper(
    results=(
        "Table~\\ref{tab:results} summarizes our main results. AdaptiveAttend "
        "achieves 92.3\\% accuracy on SST-2, surpassing BERT-base (87.6\\%) by "
        "a margin of 4.7 percentage points.\n\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Test accuracy (\\%) on classification benchmarks.}\n"
        "\\label{tab:results}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Model & SST-2 & IMDB \\\\\n"
        "\\midrule\n"
        "BERT-base & 87.6 & 91.3 \\\\\n"
        "AdaptiveAttend & 92.3 & 94.1 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "As shown in Figure~1, latency increases linearly "
        "with sequence length, confirming that AdaptiveAttend maintains "
        "efficient scaling behavior.\n\n"
        "Figure~2 shows the accuracy distribution across "
        "runs, measured in milliseconds.\n\n"
        "On XNLI, AdaptiveAttend achieves 76.5\\% average accuracy across "
        "15 languages. Figure~3 presents the results. "
        "Note that Figure~4 compares our approach to prior work."
    ),
)

# VL descriptions that contradict the text claims
_FIGURE_DESCRIPTIONS = (
    "Figure (fig:heads)\n"
    "Visual content: This is a bar chart (not a line plot) showing "
    "the number of active attention heads per input for three models. "
    "X-axis: Model name (BERT, DistilBERT, AdaptiveAttend). "
    "Y-axis: Number of active heads (0-8). "
    "AdaptiveAttend uses 5.6 heads on average.\n\n"
    'Figure (fig:accuracy) — Caption: "Accuracy distribution across runs"\n'
    "Visual content: This is a scatter plot showing response times. "
    "X-axis: Run number (1-10). Y-axis: Response time (seconds). "
    "No accuracy values visible. The axis label says 'seconds' not 'milliseconds'.\n\n"
    'Figure (fig:xnli) — Caption: "XNLI results by language"\n'
    "Visual content: A bar chart showing accuracy by language. "
    "X-axis: Language (en, de, fr, es, zh, ...). "
    "Y-axis: Accuracy (%). Values range from 71% to 82%. "
    "The chart shows accuracy percentages matching Table 1 values."
)


def _gen_figure_scenario() -> list[SyntheticCase]:
    return _scenario_cases(
        "s6_figures",
        _FIGURE_PAPER,
        "Paper with figure issues: text says 'linearly' but fig is bar chart, "
        "caption says 'accuracy' but fig shows response times, axis says 'seconds' "
        "but text says 'milliseconds', Figure 4 referenced but doesn't exist",
        {
            "text-vs-figure": [
                ExpectedFinding(rule_id="text-vs-figure", context="linear"),
            ],
            "caption-vs-content": [
                ExpectedFinding(rule_id="caption-vs-content", context="accuracy"),
            ],
            "axis-label-consistency": [
                ExpectedFinding(
                    rule_id="axis-label-consistency", context="milliseconds"
                ),
            ],
        },
        figure_descriptions=_FIGURE_DESCRIPTIONS,
        check_ids=_FIGURE_CHECK_IDS,
    )


# ---------------------------------------------------------------------------
# Scenario 7: Clean paper with figures (TN for figure checks)
# ---------------------------------------------------------------------------

_CLEAN_FIG_DESCRIPTIONS = (
    'Figure (fig:architecture) — Caption: "Overview of AdaptiveAttend. '
    "The gating network computes a complexity score that determines how "
    'many attention heads are active for each input."\n'
    "Visual content: A block diagram showing the AdaptiveAttend architecture. "
    "Input flows through an embedding layer, then to a gating network that "
    "outputs a complexity score. The score controls which attention heads "
    "are activated. Active heads feed into a pooling layer and classifier.\n\n"
    'Figure (fig:heads) — Caption: "Average number of active attention heads '
    'per language on XNLI."\n'
    "Visual content: A bar chart showing average active heads per language. "
    "X-axis: Language (en, de, fr, es, zh, tr, fi, ...). "
    "Y-axis: Average active heads (0-8). "
    "Morphologically complex languages (Turkish, Finnish) show more active "
    "heads (6-7) while simpler languages (English, French) use fewer (4-5)."
)


# Build a variant of the clean paper with explicit figure numbers
# (LaTeX \ref{fig:X} gets stripped by strip_latex_for_review,
# so the LLM never sees the reference — use "Figure 1" etc.)
_CLEAN_FIG_PAPER = build_realistic_paper(
    methods_arch=CLEAN_METHODS_ARCH.replace(
        r"Figure~\ref{fig:architecture}",
        "Figure~1",
    ),
    results=CLEAN_RESULTS.replace(
        r"Figure~\ref{fig:heads}",
        "Figure~2",
    ),
)


def _gen_clean_figure_scenario() -> list[SyntheticCase]:
    return _scenario_cases(
        "s7_clean_figures",
        _CLEAN_FIG_PAPER,
        "Clean paper with consistent figure descriptions (no figure issues)",
        {},
        figure_descriptions=_CLEAN_FIG_DESCRIPTIONS,
        check_ids=_FIGURE_CHECK_IDS,
    )


# ---------------------------------------------------------------------------
# Scenario 8: Figure data contradicts table
# ---------------------------------------------------------------------------

_TABLE_FIG_PAPER = build_realistic_paper(
    results=(
        "Table~\\ref{tab:results} shows that AdaptiveAttend achieves "
        "92.3\\% on SST-2 and 94.1\\% on IMDB. Figure~1 "
        "visualizes the same comparison.\n\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Test accuracy (\\%) on classification benchmarks.}\n"
        "\\label{tab:results}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Model & SST-2 & IMDB \\\\\n"
        "\\midrule\n"
        "BERT-base & 87.6 & 91.3 \\\\\n"
        "AdaptiveAttend & 92.3 & 94.1 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    ),
)

_TABLE_FIG_DESCRIPTIONS = (
    'Figure (fig:comparison) — Caption: "Accuracy comparison"\n'
    "Visual content: A grouped bar chart comparing models. "
    "X-axis: Dataset (SST-2, IMDB). Y-axis: Accuracy (%). "
    "BERT-base: SST-2 = 87.6%, IMDB = 91.3%. "
    "AdaptiveAttend: SST-2 = 89.5%, IMDB = 93.2%. "
    "The bars show AdaptiveAttend values lower than the table reports."
)


def _gen_table_fig_scenario() -> list[SyntheticCase]:
    return _scenario_cases(
        "s8_table_fig",
        _TABLE_FIG_PAPER,
        "Paper where figure shows different values than table "
        "(92.3 in table vs 89.5 in figure for SST-2)",
        {
            "figure-data-vs-table": [
                ExpectedFinding(rule_id="figure-data-vs-table", context=""),
            ],
            # text-vs-figure: text says "visualizes the same comparison" but
            # figure shows different values. Borderline — model fires ~50% of
            # runs. Not included as expected to avoid flaky eval.
        },
        figure_descriptions=_TABLE_FIG_DESCRIPTIONS,
        check_ids=_FIGURE_CHECK_IDS,
    )


# ---------------------------------------------------------------------------
# Public generators — one per check, delegating to scenarios
# ---------------------------------------------------------------------------


def _all_scenario_cases() -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []
    # Mechanical checks (scenarios 1-5)
    cases.extend(_gen_clean_scenario())
    cases.extend(_gen_numerical_scenario())
    cases.extend(_gen_sample_scenario())
    cases.extend(_gen_overreach_scenario())
    cases.extend(_gen_stats_scenario())
    # Figure checks (scenarios 6-8)
    cases.extend(_gen_figure_scenario())
    cases.extend(_gen_clean_figure_scenario())
    cases.extend(_gen_table_fig_scenario())
    return cases


_ALL_CASES: list[SyntheticCase] | None = None


def _get_all_cases() -> list[SyntheticCase]:
    global _ALL_CASES
    if _ALL_CASES is None:
        _ALL_CASES = _all_scenario_cases()
    return _ALL_CASES


def _gen_for_check(check_id: str) -> list[SyntheticCase]:
    return [c for c in _get_all_cases() if c.check_id == check_id]


def gen_numbers_vs_tables_cases() -> list[SyntheticCase]:
    return _gen_for_check("numbers-vs-tables")


def gen_percentages_sum_cases() -> list[SyntheticCase]:
    return _gen_for_check("percentages-sum")


def gen_sample_size_consistency_cases() -> list[SyntheticCase]:
    return _gen_for_check("sample-size-consistency")


def gen_arithmetic_consistency_cases() -> list[SyntheticCase]:
    return _gen_for_check("arithmetic-consistency")


def gen_causal_language_audit_cases() -> list[SyntheticCase]:
    return _gen_for_check("causal-language-audit")


def gen_abstract_body_alignment_cases() -> list[SyntheticCase]:
    return _gen_for_check("abstract-body-alignment")


def gen_statistical_reporting_cases() -> list[SyntheticCase]:
    return _gen_for_check("statistical-reporting")


def gen_caption_vs_content_cases() -> list[SyntheticCase]:
    return _gen_for_check("caption-vs-content")


def gen_text_vs_figure_cases() -> list[SyntheticCase]:
    return _gen_for_check("text-vs-figure")


def gen_axis_label_consistency_cases() -> list[SyntheticCase]:
    return _gen_for_check("axis-label-consistency")


def gen_figure_data_vs_table_cases() -> list[SyntheticCase]:
    return _gen_for_check("figure-data-vs-table")
