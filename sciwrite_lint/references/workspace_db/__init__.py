"""Per-paper workspace SQLite database.

Single DB per paper workspace (``references/{paper}/parsed/workspace.db``)
that holds all per-paper persistent data: embeddings, chunk metadata,
reference registry, citation metadata, parse cache, claim results,
vision cache, pipeline stage tracking, and query-vector cache.

This package is a compatibility surface: implementations live in
per-table submodules, but every symbol external callers and tests import
is re-exported here, so ``from sciwrite_lint.references.workspace_db
import X`` keeps working for every pre-split X.
"""

from __future__ import annotations

from sciwrite_lint.references.workspace_db._core import (
    DB_NAME,
    db_path,
    get_db,
    open_db,
    serialize_f32,
)
from sciwrite_lint.references.workspace_db.bib_checks import (
    load_bib_checks,
    save_bib_checks,
)
from sciwrite_lint.references.workspace_db.citation_metadata import (
    delete_citation_metadata,
    load_all_citation_metadata,
    load_citation_metadata,
    query_refs_by_match,
    query_refs_with_local_pdfs,
    query_refs_with_mismatches,
    query_retracted_refs,
    query_verified_metadata,
    save_citation_metadata,
)
from sciwrite_lint.references.workspace_db.claim_results import (
    clear_claim_dismissal,
    count_by_verdict,
    dismiss_claim,
    find_claim,
    list_claims_for_key,
    load_claim_results,
    save_claim_results,
)
from sciwrite_lint.references.workspace_db.parse_cache import (
    is_formal_cached_db,
    load_all_parse_cache,
    load_parse_cache,
    save_parse_cache,
    update_parse_cache_embeddings,
)
from sciwrite_lint.references.workspace_db.pipeline_stage import (
    PIPELINE_STAGES,
    init_pipeline_stages,
    load_pipeline_stages,
    update_pipeline_stage,
)
from sciwrite_lint.references.workspace_db.manuscript_citations import (
    ManuscriptCitation,
    count_manuscript_citations,
    find_unembedded_contexts,
    load_unique_contexts,
    replace_manuscript_citations,
)
from sciwrite_lint.references.workspace_db.query_vectors import (
    load_query_vector,
    save_query_vector,
)
from sciwrite_lint.references.workspace_db.ref_internal import (
    load_all_ref_internal_scores,
    load_ref_internal_cache,
    save_ref_internal_cache,
)
from sciwrite_lint.references.workspace_db.registry import (
    load_bibliography_entries,
    lookup_reference,
    register_reference,
)
from sciwrite_lint.references.workspace_db.vision_cache import (
    clear_vision_cache,
    load_all_vision_entries,
    load_vision_entry,
    save_vision_entry,
)

__all__ = [
    # Core
    "DB_NAME",
    "db_path",
    "get_db",
    "open_db",
    "serialize_f32",
    # Reference registry
    "load_bibliography_entries",
    "lookup_reference",
    "register_reference",
    # Bibliography verification
    "load_bib_checks",
    "save_bib_checks",
    # Citation metadata
    "delete_citation_metadata",
    "load_all_citation_metadata",
    "load_citation_metadata",
    "query_refs_by_match",
    "query_refs_with_local_pdfs",
    "query_refs_with_mismatches",
    "query_retracted_refs",
    "query_verified_metadata",
    "save_citation_metadata",
    # Parse cache
    "is_formal_cached_db",
    "load_all_parse_cache",
    "load_parse_cache",
    "save_parse_cache",
    "update_parse_cache_embeddings",
    # Ref internal cache
    "load_all_ref_internal_scores",
    "load_ref_internal_cache",
    "save_ref_internal_cache",
    # Claim results
    "clear_claim_dismissal",
    "count_by_verdict",
    "dismiss_claim",
    "find_claim",
    "list_claims_for_key",
    "load_claim_results",
    "save_claim_results",
    # Vision cache
    "clear_vision_cache",
    "load_all_vision_entries",
    "load_vision_entry",
    "save_vision_entry",
    # Pipeline stage tracking
    "PIPELINE_STAGES",
    "init_pipeline_stages",
    "load_pipeline_stages",
    "update_pipeline_stage",
    # Query vector cache
    "load_query_vector",
    "save_query_vector",
    # Manuscript inline citations
    "ManuscriptCitation",
    "count_manuscript_citations",
    "find_unembedded_contexts",
    "load_unique_contexts",
    "replace_manuscript_citations",
]
