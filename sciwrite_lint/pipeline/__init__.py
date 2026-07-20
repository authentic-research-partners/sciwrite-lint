"""Unified check pipeline: text rules + API verification + claims in one pass.

Orchestrates all stages of manuscript verification:
1. Text + LLM rules (concurrent with stage 2)
2. Reference verification via batch APIs
3. Full-text acquisition
4. GROBID parsing + embeddings
5. Claim verification (sem-001)
6. Merged report

Supports both LaTeX (.tex) and PDF input. For PDF, the manuscript is parsed
via GROBID and a ManuscriptContext is built from the result.

Requires: vLLM running, GROBID running, network access.

This package is the split of the old single-file ``pipeline.py``. External
callers import public and "internal" names (used in tests, eval scripts,
and subprocess ``python -c "from sciwrite_lint.pipeline import ..."``
launchers) directly from ``sciwrite_lint.pipeline``; the names below
preserve that surface exactly.
"""

from sciwrite_lint.pipeline.checks import run_llm_checks_batched, run_text_checks
from sciwrite_lint.pipeline.claims import (
    _claims_to_findings,
    _collect_parse_hashes,
    _stage_bib_verify,
    _stage_claims,
)
from sciwrite_lint.pipeline.embeddings import (
    _run_embeddings_for_paper,
    _verify_embeddings_or_raise,
    ensure_claim_query_vectors,
    persist_manuscript_citations,
)
from sciwrite_lint.pipeline.fetch import _stage_fetch
from sciwrite_lint.pipeline.orchestration import (
    _backup_workspace,
    _format_usage_summary,
    preflight,
    run_full_check,
)
from sciwrite_lint.pipeline.staged import (
    StagedPaperResult,
    _PaperCtx,
    run_papers_staged,
)
from sciwrite_lint.pipeline.markdown_context import build_markdown_context
from sciwrite_lint.pipeline.parse import _stage_parse
from sciwrite_lint.pipeline.pdf_context import (
    build_pdf_context,
    citations_from_pdf_context,
    extract_citations_for_paper,
)
from sciwrite_lint.pipeline.ref_internal import _stage_ref_internal
from sciwrite_lint.pipeline.runners import (
    EmbedderWarmer,
    _batch_cited_vision,
    _batch_cited_vision_entry,
    _batch_embed,
    _batch_embed_entry,
    _batch_vision,
    _embed_keys,
    _embed_keys_via_stdin,
    _encode_missing_claim_queries,
    _run_embeddings_subprocess,
)
from sciwrite_lint.pipeline.swap import (
    _cited_needs_inference,
    _is_vllm_responding,
    _manuscript_needs_inference,
    _needs_embedding_swap,
    _restart_vllm_after_embedding,
    _stop_vllm_for_embedding,
    _swap_to_text_vllm,
    _swap_to_vision_vllm,
)
from sciwrite_lint.pipeline.tracking import (
    _StageStatus,
    _stage_failure_guard,
    _stage_tracking,
    _track,
)
from sciwrite_lint.pipeline.unreliable import _stage_unreliable
from sciwrite_lint.pipeline.verify import (
    _classify_verify_issue,
    _confirm_venue_findings,
    _register_ref_in_workspace,
    _stage_reference_accuracy,
    _stage_verify,
)
from sciwrite_lint.pipeline.vision import _stage_cited_vision, _stage_vision

__all__ = [
    # Public orchestrators
    "preflight",
    "run_full_check",
    "run_papers_staged",
    "StagedPaperResult",
    # Public stage helpers (LaTeX/PDF/markdown handling)
    "build_pdf_context",
    "build_markdown_context",
    "citations_from_pdf_context",
    "extract_citations_for_paper",
    # Public stage entry points
    "run_text_checks",
    "run_llm_checks_batched",
    "ensure_claim_query_vectors",
    "persist_manuscript_citations",
    # Internal names used by tests / evals / subprocess launchers
    "_backup_workspace",
    "_batch_cited_vision",
    "_batch_cited_vision_entry",
    "_batch_embed",
    "_batch_embed_entry",
    "_batch_vision",
    "_cited_needs_inference",
    "_claims_to_findings",
    "_classify_verify_issue",
    "_collect_parse_hashes",
    "_confirm_venue_findings",
    "_embed_keys",
    "_embed_keys_via_stdin",
    "_encode_missing_claim_queries",
    "EmbedderWarmer",
    "_format_usage_summary",
    "_is_vllm_responding",
    "_manuscript_needs_inference",
    "_needs_embedding_swap",
    "_PaperCtx",
    "_register_ref_in_workspace",
    "_restart_vllm_after_embedding",
    "_run_embeddings_for_paper",
    "_run_embeddings_subprocess",
    "_stage_bib_verify",
    "_stage_cited_vision",
    "_stage_claims",
    "_stage_failure_guard",
    "_stage_fetch",
    "_stage_parse",
    "_stage_ref_internal",
    "_stage_reference_accuracy",
    "_stage_tracking",
    "_stage_unreliable",
    "_stage_verify",
    "_stage_vision",
    "_StageStatus",
    "_stop_vllm_for_embedding",
    "_swap_to_text_vllm",
    "_swap_to_vision_vllm",
    "_track",
    "_verify_embeddings_or_raise",
]
