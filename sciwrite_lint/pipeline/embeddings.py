"""Stage 4b helpers: embeddings for parsed refs + claim query pre-compute.

In ``run_full_check`` this runs inline after parse (one subprocess per
paper). In ``run_papers_staged`` the orchestrator runs ``_batch_embed``
directly and only reuses the claim-text extractor here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.pipeline.runners import EmbedderWarmer, _run_embeddings_subprocess
from sciwrite_lint.pipeline.swap import (
    _needs_embedding_swap,
    _restart_vllm_after_embedding,
    _stop_vllm_for_embedding,
)

if TYPE_CHECKING:
    from sciwrite_lint.models import Finding


def _resolve_manuscript_context(
    config: LintConfig,
    tex_path: Path | None,
) -> tuple["Any | None", "Finding | None"]:
    """Resolve a ManuscriptContext for the current run.

    Returns ``(ctx, system_issue_or_None)``. ``ctx`` is ``None`` when
    the run has nothing to embed (no PDF context preset, no .tex on
    disk). The system-issue finding is non-``None`` only when LaTeX
    parsing raised.
    """
    from sciwrite_lint.manuscript_store import ManuscriptContext

    if config.is_pdf and config.manuscript_context:
        return config.manuscript_context, None
    if tex_path and tex_path.suffix.lower() == ".tex":
        try:
            return ManuscriptContext.from_latex(tex_path, config), None
        except Exception as e:
            from sciwrite_lint.checks._diagnostics import internal_error_finding

            logger.warning("Manuscript context build failed: {}", e)
            return None, internal_error_finding("manuscript-context", e)
    return None, None


def persist_manuscript_citations(
    config: LintConfig,
    references_dir: Path,
    tex_path: Path | None = None,
) -> tuple[int, "Finding | None"]:
    """Persist the manuscript's inline citations to ``manuscript_citations``.

    Idempotent: replaces any prior rows for this workspace. Both .tex
    and PDF paths converge on the same persisted shape so the embedding
    subprocess can read claim contexts directly from the DB instead of
    receiving them as JSON.

    Returns ``(n_citations_persisted, system_issue_or_None)``. The
    ``Finding`` is non-``None`` when the manuscript build produced
    non-fatal warnings (e.g. cite-context attach raised) — callers in
    the pipeline append it to the per-paper findings list so it routes
    into the report's ``system_issues`` bucket via ``split_findings``.
    """
    from sciwrite_lint.manuscript_store import persist_inline_citations

    ctx, build_failure = _resolve_manuscript_context(config, tex_path)
    if build_failure is not None:
        return 0, build_failure
    if ctx is None:
        logger.debug(
            "No manuscript context to persist: is_pdf={}, has_ctx={}, tex_path={}",
            config.is_pdf,
            config.manuscript_context is not None,
            tex_path,
        )
        return 0, None

    finding = persist_inline_citations(ctx, references_dir)
    return len(ctx.inline_citations), finding


def prepare_manuscript_for_embedding(
    config: LintConfig,
    references_dir: Path,
    tex_path: Path | None,
    model_name: str,
) -> str | None:
    """Write ``parsed/_manuscript_{stem}.md`` and return its embedding key.

    The embedding subprocess resolves keys to ``parsed/{key}.md`` files,
    so dropping a manuscript markdown there is the cheapest way to put
    the manuscript through the same chunking + embedding code path used
    for parsed references — no extra subprocess entry point needed.

    Skips work entirely when embeddings already exist for the key
    (idempotent on re-runs without ``--fresh``).

    Returns the manuscript ref key, or ``None`` if nothing was prepared
    (no usable context, no text, or embeddings already up to date).
    """
    from sciwrite_lint.manuscript_store import (
        manuscript_ref_key,
        write_manuscript_markdown,
    )
    from sciwrite_lint.references.embedding_store import has_embeddings

    ctx, _ = _resolve_manuscript_context(config, tex_path)
    if ctx is None:
        return None
    ms_key = manuscript_ref_key(ctx.source_path)
    if has_embeddings(ms_key, references_dir, model_name=model_name):
        return None
    written = write_manuscript_markdown(ctx, references_dir)
    if written is None:
        return None
    return ms_key


def ensure_claim_query_vectors(
    references_dir: Path,
    config: LintConfig,
) -> None:
    """Ensure pre-computed query vectors exist for each persisted citation context.

    In the full pipeline, Stage 4b populates ``query_vectors`` so Stage 5's
    ``retrieve_similar()`` never has to load the embedding model in the
    parent process. Standalone entry points (``sciwrite-lint verify-claims``,
    library callers of ``run_claim_verification``) skip Stage 4b, so this
    helper spawns the same embedding subprocess. The subprocess reads
    contexts from ``manuscript_citations`` and encodes only those whose
    hash isn't already in ``query_vectors``.
    """
    from sciwrite_lint.references.reference_store import _get_embedding_config
    from sciwrite_lint.references.workspace_db import (
        find_unembedded_contexts,
        get_db,
    )

    model_name, _, _ = _get_embedding_config()
    with get_db(references_dir) as conn:
        missing = find_unembedded_contexts(conn, model_name)
    if not missing:
        return

    logger.info(
        "Pre-computing {} missing claim query vectors (subprocess)", len(missing)
    )
    err = _run_embeddings_subprocess(
        keys=[],
        references_dir=references_dir,
        config=config,
    )
    if err:
        logger.warning(
            "Query vector pre-compute subprocess failed: {} — "
            "claim retrieval will return CANNOT_DETERMINE",
            err,
        )


def _run_embeddings_for_paper(
    parse_results: dict[str, str],
    references_dir: Path,
    config: LintConfig,
    tex_path: Path | None = None,
    warmer: EmbedderWarmer | None = None,
) -> "Finding | None":
    """Run a single embedding subprocess for one paper's parsed + web keys.

    Also pre-computes claim query vectors in the same subprocess so
    Stage 5 never loads the embedding model in the parent process.

    Two paths:

    * ``warmer is None`` (direct): stop text vLLM here, launch a fresh
      one-shot subprocess via ``_run_embeddings_subprocess``, restart
      vLLM. Used by callers that don't pre-warm.
    * ``warmer is not None`` (pipelined): the caller already stopped
      text vLLM and launched the warming subprocess at parse start, so
      its model load overlapped with GROBID parse. Submit all keys to
      the warmer in one call and wait. The caller is responsible for
      restarting vLLM afterwards.

    Returns a system-issue ``Finding`` when ``persist_manuscript_citations``
    surfaces a non-fatal build warning; ``None`` otherwise. Callers should
    append the Finding to their per-paper findings list so it routes into
    the report's ``system_issues`` bucket.

    Raises ``RuntimeError`` if the embedding subprocess fails to actually
    produce embeddings for any requested key — verified by re-checking
    ``has_embeddings()`` against the DB afterwards (the authoritative
    source of truth). Failing loudly here avoids a cryptic "no
    embeddings found" crash later during claim verification.
    """
    from sciwrite_lint.references.embedding_store import has_embeddings
    from sciwrite_lint.references.metadata import load_all_metadata
    from sciwrite_lint.references.reference_store import _get_embedding_config
    from sciwrite_lint.references.workspace_db import (
        count_manuscript_citations,
        get_db,
    )

    _, build_finding = persist_manuscript_citations(
        config, references_dir, tex_path=tex_path
    )
    with get_db(references_dir) as conn:
        n_citations = count_manuscript_citations(conn)

    model_name, _, _ = _get_embedding_config()

    # Collect parsed-ref keys that need embedding.
    parsed_keys = []
    for key, status in parse_results.items():
        if status == "parsed":
            parsed_keys.append(key)
        elif status == "cached" and not has_embeddings(
            key, references_dir, model_name=model_name
        ):
            parsed_keys.append(key)

    # Collect web-summary keys that need embedding.
    all_meta = load_all_metadata(references_dir)
    web_keys = []
    for key, meta in all_meta.items():
        local_file = meta.access.get("local_file") or ""
        if not local_file.endswith("_web.md"):
            continue
        if has_embeddings(key, references_dir, model_name=model_name):
            continue
        web_path = references_dir / local_file
        if web_path.exists():
            web_keys.append(key)

    # Manuscript itself: write parsed/_manuscript_{stem}.md and add to
    # the keys list so the subprocess chunks + embeds it the same way it
    # handles parsed references. Idempotent — skipped when the key
    # already has embeddings.
    ms_key = prepare_manuscript_for_embedding(
        config, references_dir, tex_path, model_name
    )
    manuscript_keys = [ms_key] if ms_key else []

    all_keys = parsed_keys + web_keys + manuscript_keys

    if not all_keys and not n_citations:
        return build_finding

    # Stop vLLM if needed (warmer path: caller already did it; fresh
    # path: do it here).
    embed_swap = False
    if warmer is None and _needs_embedding_swap(config):
        embed_swap = True
        _stop_vllm_for_embedding(config)

    # Run the subprocess (warming or fresh).
    if warmer is not None:
        diag = warmer.submit_and_wait(all_keys)
    else:
        diag = _run_embeddings_subprocess(all_keys, references_dir, config)

    # Verify *before* restart so a real subprocess failure surfaces
    # via _verify_embeddings_or_raise instead of being masked by a
    # follow-on vLLM restart error. On verify failure the caller is
    # responsible for restart cleanup (warmer path: _stage_parse's
    # finally; fresh path: leak the stopped state, the next pipeline
    # invocation will restart vLLM).
    if parsed_keys:
        _verify_embeddings_or_raise(
            parsed_keys,
            references_dir,
            model_name,
            kind="parsed",
            subprocess_diag=diag,
        )
    if web_keys:
        _verify_embeddings_or_raise(
            web_keys,
            references_dir,
            model_name,
            kind="web summary",
            subprocess_diag=diag,
        )
    if manuscript_keys:
        # Manuscript embeddings are advisory: the consistency-pairs
        # check no-ops when they're missing, so a partial subprocess
        # failure on the manuscript shouldn't take the whole pipeline
        # down with it.
        from sciwrite_lint.references.embedding_store import has_embeddings

        for k in manuscript_keys:
            if not has_embeddings(k, references_dir, model_name=model_name):
                logger.warning(
                    "Manuscript embedding missing for {} — "
                    "internal-consistency-pairs will skip this run",
                    k,
                )

    if embed_swap:
        _restart_vllm_after_embedding(config)

    return build_finding


def _verify_embeddings_or_raise(
    keys: list[str],
    references_dir: Path,
    model_name: str,
    kind: str,
    subprocess_diag: str = "",
) -> None:
    """Verify that all requested keys have embeddings in the DB.

    Called after ``_run_embeddings_subprocess``. Raises ``RuntimeError``
    with a clear message if any keys are missing, including the subprocess
    stderr tail so the user sees the actual failure (CUDA OOM, subprocess
    timeout, model load failure, etc.) inline with the error.
    """
    from sciwrite_lint.references.embedding_store import has_embeddings

    missing = [
        k for k in keys if not has_embeddings(k, references_dir, model_name=model_name)
    ]
    if missing:
        sample = ", ".join(missing[:3])
        msg = (
            f"Embedding subprocess failed to produce {kind} embeddings: "
            f"{len(missing)}/{len(keys)} keys missing (first missing: "
            f"{sample})."
        )
        if subprocess_diag:
            msg += f"\n\nSubprocess diagnostic:\n{subprocess_diag}"
        else:
            msg += (
                "\n\nSubprocess exited cleanly but embeddings are absent — "
                "likely a silent internal skip (check for per-key exceptions "
                "in the subprocess log above)."
            )
        msg += (
            "\n\nCommon causes: CUDA OOM, subprocess timeout "
            "(auto-scaled by input size; see _compute_embed_timeout), "
            "model load failure, or sqlite-vec DB lock."
        )
        raise RuntimeError(msg)
