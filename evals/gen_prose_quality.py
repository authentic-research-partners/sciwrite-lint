"""Synthetic cases for prose-quality (grammar + semantic word-choice).

Each case wraps a target sentence in a minimal single-section paper so
the LLM sees exactly one paragraph to review. True-positive cases have
a known grammar or semantic error in the target sentence; true-negative
cases have clean prose.

The prose-quality check emits findings with ``Finding.level`` driven by
the LLM-reported confidence (low dropped, medium → info, high → warning).
Eval cases here assume the LLM is calibrated to return ``high`` on clear
errors — P/R/F1 measured against that assumption guides prompt tuning.
"""

from __future__ import annotations

from evals.synthetic_types import (
    ExpectedFinding,
    SyntheticCase,
    _BIBITEMS,
    _make_doc,
)


def _case(
    name: str,
    target_sentence: str,
    *,
    filler_sentence: str = "",
    expected_span: str = "",
    description: str = "",
) -> SyntheticCase:
    """Wrap a target sentence in a minimal paper.

    Args:
        name: unique case name.
        target_sentence: the sentence the LLM is expected to review.
        filler_sentence: optional second sentence in the same paragraph
            (provides paragraph context without introducing extra TPs).
        expected_span: substring of the finding's context/message that
            identifies a TP. Empty string → TN case.
        description: short prose description for debugging.
    """
    para = target_sentence
    if filler_sentence:
        para = f"{filler_sentence} {target_sentence}"

    body = "\\section{Introduction}\n" + para + "\n"
    expected: list[ExpectedFinding] = []
    if expected_span:
        expected = [
            ExpectedFinding(rule_id="prose-quality", context=expected_span),
        ]
    return SyntheticCase(
        name=name,
        check_id="prose-quality",
        tex_content=_make_doc(body, _BIBITEMS[:1]),
        expected=expected,
        description=description or name,
    )


# ---------------------------------------------------------------------------
# Semantic word-choice true positives
# ---------------------------------------------------------------------------

_SEMANTIC_TP_CASES = [
    _case(
        "semantic_object_vs_purpose",
        "Current educational technology is built for a different object: content delivery.",
        expected_span="object",
        description=(
            "'object' used where 'purpose' or 'objective' is intended."
        ),
    ),
    _case(
        "semantic_comprise_of",
        "The pipeline comprises of three sequential processing stages.",
        expected_span="comprise",
        description="'comprise of' is non-idiomatic; should be 'comprises' or 'consists of'.",
    ),
    _case(
        "semantic_effect_vs_affect",
        "The treatment had a strong effect on outcomes and effected downstream metrics.",
        expected_span="effected",
        description="'effected' used where 'affected' is intended.",
    ),
    _case(
        "semantic_data_is_insufficient",
        "Our results demonstrate that the principle underlining this mechanism is sound.",
        expected_span="underlining",
        description="'underlining' used where 'underlying' is intended.",
    ),
    _case(
        "semantic_infer_vs_imply",
        "The authors infer from their model that adoption will increase.",
        filler_sentence=("The inference here belongs to the reader, not the authors."),
        expected_span="infer",
        description="Classic infer/imply swap — authors imply, readers infer.",
    ),
    _case(
        "semantic_fewer_vs_less",
        "Our model requires less parameters than the baseline while achieving comparable accuracy.",
        expected_span="less",
        description="'less' used with count noun 'parameters'; should be 'fewer'.",
    ),
    _case(
        "semantic_principle_vs_principal",
        "The principle contribution of this paper is a new sampling algorithm.",
        expected_span="principle",
        description="'principle' used where 'principal' (main) is intended.",
    ),
    _case(
        "semantic_its_apostrophe",
        "The network adjusts it's weights based on the loss signal.",
        expected_span="it's",
        description="'it's' (contraction of 'it is') used where possessive 'its' is intended.",
    ),
    _case(
        "semantic_discreet_vs_discrete",
        "We partition the input into discreet tokens using a byte-pair encoder.",
        expected_span="discreet",
        description="'discreet' (unobtrusive) used where 'discrete' (separate) is intended.",
    ),
]


# ---------------------------------------------------------------------------
# Grammar true positives
# ---------------------------------------------------------------------------

_GRAMMAR_TP_CASES = [
    _case(
        "grammar_sv_agreement",
        "The set of experiments demonstrate a consistent improvement across benchmarks.",
        expected_span="demonstrate",
        description="Subject-verb agreement: 'set' (singular) requires 'demonstrates'.",
    ),
    _case(
        "grammar_article_omission",
        "We propose novel architecture that generalizes across tasks.",
        expected_span="novel architecture",
        description="Missing article before 'novel architecture'.",
    ),
    _case(
        "grammar_tense_slip",
        "We trained the model on ImageNet and then evaluates it on CIFAR-100.",
        expected_span="evaluates",
        description="Tense slip — past 'trained' vs present 'evaluates'.",
    ),
    _case(
        "grammar_preposition",
        "These findings are consistent to prior work in the field.",
        expected_span="consistent to",
        description="'consistent to' should be 'consistent with'.",
    ),
    _case(
        "grammar_comma_splice",
        "The model converged quickly, it reached optimal loss within two hours.",
        expected_span="quickly,",
        description=(
            "Comma splice: two independent clauses joined by a comma. Should "
            "be a semicolon, period, or coordinating conjunction."
        ),
    ),
    _case(
        "grammar_dangling_modifier",
        "Having trained on 100k examples, the accuracy improved substantially.",
        expected_span="Having trained",
        description=(
            "Dangling modifier: 'having trained' grammatically attaches to "
            "'the accuracy' but accuracy cannot train."
        ),
    ),
    _case(
        "grammar_pronoun_case",
        "The paper was co-authored by my advisor and I.",
        expected_span=" I",
        description=(
            "Wrong pronoun case after preposition; should be 'my advisor and me'."
        ),
    ),
    _case(
        "grammar_compound_subject",
        "The encoder and the decoder works together to produce the output sequence.",
        expected_span="works",
        description=(
            "Compound subject ('encoder and decoder') requires plural verb 'work'."
        ),
    ),
]


# ---------------------------------------------------------------------------
# True negatives — clean scientific prose
# ---------------------------------------------------------------------------

_TN_CASES = [
    _case(
        "clean_hedged_claim",
        "Our results suggest that the proposed method may generalize across domains.",
        description="Hedged language must not be flagged as an error.",
    ),
    _case(
        "clean_passive_voice",
        "The experiments were conducted on a single A100 GPU over seven days.",
        description="Passive voice is valid academic register.",
    ),
    _case(
        "clean_long_sentence",
        (
            "We show, through a series of controlled experiments spanning three "
            "benchmark datasets and two model families, that the proposed "
            "intervention yields statistically significant improvements over "
            "the strongest baseline under every measured condition."
        ),
        description="Long but grammatically correct sentence.",
    ),
    _case(
        "clean_technical_jargon",
        "The transformer decoder attends to encoded token embeddings via scaled dot-product attention.",
        description="Technical jargon is not a prose error.",
    ),
    _case(
        "clean_parenthetical_aside",
        "The pretrained model (BERT-base, 110M parameters) is fine-tuned for 3 epochs on each task.",
        description="Parenthetical aside with numerals is routine technical prose.",
    ),
    _case(
        "clean_nested_clauses",
        (
            "Although prior work has reported strong zero-shot performance, we "
            "find that the accuracy drops sharply when the domain shifts, "
            "suggesting that the model relies on surface cues rather than robust "
            "generalization."
        ),
        description="Grammatically correct sentence with subordinating clauses.",
    ),
    _case(
        "clean_method_statement",
        "We evaluate on five standard benchmarks using top-1 accuracy, F1, and ROUGE-L.",
        description="Methods sentence with list is not a prose error.",
    ),
    _case(
        "clean_negation",
        "The observed drop in accuracy is not inconsistent with the theoretical upper bound.",
        description=(
            "Double negation is uncommon but grammatically valid; must not flag as awkward."
        ),
    ),
    _case(
        "clean_numbered_contributions",
        (
            "Our contributions are threefold: (1) a new benchmark, (2) a family "
            "of baselines, and (3) a detailed error analysis."
        ),
        description="Numbered-contributions sentence is a scientific-writing staple.",
    ),
    _case(
        "clean_formal_register",
        "We demonstrate empirically that the proposed optimizer converges in fewer steps than Adam.",
        description="Formal academic register; grammatically sound.",
    ),
    _case(
        "clean_here_opener",
        "Here we present a unified framework for evaluating claim-citation alignment.",
        description=(
            "'Here, we ...' is a common and accepted opener in scientific papers."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Borderline — stylistic / ambiguous (for FP measurement)
# ---------------------------------------------------------------------------

_BORDERLINE_CASES = [
    _case(
        "borderline_sentence_fragment",
        "A clear contribution to the field.",
        description=(
            "Sentence fragment — stylistically accepted in summary contexts. "
            "Must NOT be flagged as a grammar error."
        ),
    ),
    _case(
        "borderline_split_infinitive",
        "We aim to carefully characterize the failure modes of our method.",
        description="Split infinitive — widely accepted; must not flag.",
    ),
    _case(
        "borderline_starting_with_conjunction",
        "But the observed improvement vanishes once we control for dataset size.",
        description=(
            "Sentence starting with 'But' is stylistically accepted in "
            "contemporary scientific writing; must not flag."
        ),
    ),
    _case(
        "borderline_contraction",
        "It's unclear whether the effect survives under distribution shift.",
        description=(
            "Contractions in academic prose are informal but not ungrammatical; "
            "must not flag as a grammar error."
        ),
    ),
    _case(
        "borderline_nominalisation",
        "The establishment of a causal link requires an intervention study.",
        description=(
            "Nominalisation ('establishment') is verbose but not an error; "
            "must not flag as awkward."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------


def gen_prose_quality_cases() -> list[SyntheticCase]:
    return [
        *_SEMANTIC_TP_CASES,
        *_GRAMMAR_TP_CASES,
        *_TN_CASES,
        *_BORDERLINE_CASES,
    ]
