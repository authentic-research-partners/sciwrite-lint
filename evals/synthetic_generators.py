"""Synthetic test case generators for each check.

Each generator returns a list of SyntheticCase objects with both minimal
stub cases (fast regression) and realistic paper variants (edge cases).

Generator implementations are split by category:
- gen_dangling: dangling-cite, dangling-ref
- gen_structure: cross-section-consistency, structure-promises
- gen_references: reference-accuracy, reference-exists
- gen_claims: claim-support, cite-purpose
- gen_pdf: dangling-cite-pdf, dangling-ref-pdf
"""

from __future__ import annotations

from typing import Callable

from evals.synthetic_types import SyntheticCase


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------


def _load_generators() -> dict[str, Callable[[], list[SyntheticCase]]]:
    """Lazy-load generator functions from submodules."""
    from evals.gen_claims import (
        gen_cite_purpose_cases,
        gen_claim_support_cases,
    )
    from evals.gen_dangling import (
        gen_dangling_cite_cases,
        gen_dangling_ref_cases,
    )
    from evals.gen_pdf import (
        gen_dangling_cite_pdf_cases,
        gen_dangling_ref_pdf_cases,
    )
    from evals.gen_references import (
        gen_reference_accuracy_cases,
        gen_reference_exists_cases,
        gen_retracted_cite_cases,
    )
    from evals.gen_full_paper import (
        gen_abstract_body_alignment_cases,
        gen_arithmetic_consistency_cases,
        gen_axis_label_consistency_cases,
        gen_caption_vs_content_cases,
        gen_causal_language_audit_cases,
        gen_figure_data_vs_table_cases,
        gen_numbers_vs_tables_cases,
        gen_percentages_sum_cases,
        gen_sample_size_consistency_cases,
        gen_statistical_reporting_cases,
        gen_text_vs_figure_cases,
    )
    from evals.gen_prose_quality import gen_prose_quality_cases
    from evals.gen_structure import (
        gen_cross_section_cases,
        gen_structure_promises_cases,
    )

    return {
        "dangling-cite": gen_dangling_cite_cases,
        "dangling-ref": gen_dangling_ref_cases,
        "cross-section-consistency": gen_cross_section_cases,
        "structure-promises": gen_structure_promises_cases,
        "reference-exists": gen_reference_exists_cases,
        "reference-accuracy": gen_reference_accuracy_cases,
        "retracted-cite": gen_retracted_cite_cases,
        "claim-support": gen_claim_support_cases,
        "cite-purpose": gen_cite_purpose_cases,
        "dangling-cite-pdf": gen_dangling_cite_pdf_cases,
        "dangling-ref-pdf": gen_dangling_ref_pdf_cases,
        "numbers-vs-tables": gen_numbers_vs_tables_cases,
        "percentages-sum": gen_percentages_sum_cases,
        "sample-size-consistency": gen_sample_size_consistency_cases,
        "arithmetic-consistency": gen_arithmetic_consistency_cases,
        "causal-language-audit": gen_causal_language_audit_cases,
        "abstract-body-alignment": gen_abstract_body_alignment_cases,
        "statistical-reporting": gen_statistical_reporting_cases,
        "caption-vs-content": gen_caption_vs_content_cases,
        "text-vs-figure": gen_text_vs_figure_cases,
        "axis-label-consistency": gen_axis_label_consistency_cases,
        "figure-data-vs-table": gen_figure_data_vs_table_cases,
        "prose-quality": gen_prose_quality_cases,
    }


TEXT_CHECKS = ["dangling-cite", "dangling-ref"]
LLM_CHECKS = [
    "cross-section-consistency",
    "structure-promises",
    "numbers-vs-tables",
    "percentages-sum",
    "sample-size-consistency",
    "arithmetic-consistency",
    "causal-language-audit",
    "abstract-body-alignment",
    "statistical-reporting",
    "caption-vs-content",
    "text-vs-figure",
    "axis-label-consistency",
    "figure-data-vs-table",
    "prose-quality",
]
DB_CHECKS = [
    "reference-exists",
    "reference-accuracy",
    "retracted-cite",
    "claim-support",
    "cite-purpose",
]
PDF_CHECKS = ["dangling-cite-pdf", "dangling-ref-pdf"]


def _vllm_available() -> bool:
    """Check if vLLM is responding on the configured endpoint."""
    import asyncio

    from sciwrite_lint.vllm.vllm_server import _check_api_health

    result = asyncio.run(_check_api_health("http://localhost:5001/v1"))
    return result is not None


def generate_cases(
    checks: list[str] | None = None,
) -> list[SyntheticCase]:
    """Generate synthetic test cases for requested checks.

    Args:
        checks: Check IDs to generate for. None = auto-detect:
            always includes text + database + PDF checks, adds LLM checks
            if vLLM is available.
    """
    if checks is None:
        checks = list(TEXT_CHECKS + DB_CHECKS + PDF_CHECKS)
        if _vllm_available():
            checks.extend(LLM_CHECKS)

    generators = _load_generators()
    cases: list[SyntheticCase] = []
    for check_id in checks:
        gen = generators.get(check_id)
        if gen is None:
            raise ValueError(f"No generator for check: {check_id}")
        cases.extend(gen())

    return cases
