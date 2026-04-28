"""Stage 4b helpers: embeddings for parsed refs + claim query pre-compute.

In ``run_full_check`` this runs inline after parse (one subprocess per
paper). In ``run_papers_staged`` the orchestrator runs ``_batch_embed``
directly and only reuses the claim-text extractor here.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig
from sciwrite_lint.pipeline.runners import _run_embeddings_subprocess
from sciwrite_lint.pipeline.swap import (
    _needs_embedding_swap,
    _restart_vllm_after_embedding,
    _stop_vllm_for_embedding,
)


def _extract_claim_texts(config: LintConfig, tex_path: Path | None = None) -> list[str]:
    """Extract unique claim context strings for query vector pre-computation.

    Works for both LaTeX (.tex) and PDF input (ManuscriptContext).
    Returns deduplicated list of claim context strings.
    """
    try:
        if config.is_pdf and config.manuscript_context:
            texts = list(
                {ic.context for ic in config.manuscript_context.inline_citations}
            )
            logger.debug(
                "Extracted {} claim texts from PDF ManuscriptContext", len(texts)
            )
            return texts
        if tex_path and tex_path.suffix.lower() == ".tex":
            from sciwrite_lint.eval_claims import extract_claim_contexts

            claims = extract_claim_contexts(tex_path)
            texts = list({c.context for c in claims})
            logger.debug("Extracted {} claim texts from .tex", len(texts))
            return texts
        logger.debug(
            "No claim texts: is_pdf={}, has_ctx={}, tex_path={}",
            config.is_pdf,
            config.manuscript_context is not None,
            tex_path,
        )
    except Exception as e:
        logger.debug("Claim text extraction failed: {}", e)
        pass
    return []


def ensure_claim_query_vectors(
    claim_texts: list[str],
    references_dir: Path,
    config: LintConfig,
) -> None:
    """Ensure pre-computed query vectors exist for each unique claim context.

    In the full pipeline, Stage 4b populates ``query_vectors`` so Stage 5's
    ``retrieve_similar()`` never has to load the embedding model in the
    parent process. Standalone entry points (``sciwrite-lint verify-claims``,
    library callers of ``run_claim_verification``) skip Stage 4b, so this
    helper spawns the same embedding subprocess for any missing vectors.
    """
    import hashlib

    from sciwrite_lint.references.reference_store import _get_embedding_config
    from sciwrite_lint.references.workspace_db import get_db, load_query_vector

    unique_texts = sorted({t for t in claim_texts if t})
    if not unique_texts:
        return

    model_name, _, _ = _get_embedding_config()
    with get_db(references_dir) as conn:
        missing = [
            t
            for t in unique_texts
            if load_query_vector(
                conn, hashlib.sha256(t.encode()).hexdigest(), model_name
            )
            is None
        ]
    if not missing:
        return

    logger.info(
        "Pre-computing {} missing claim query vectors (subprocess)", len(missing)
    )
    err = _run_embeddings_subprocess(
        keys=[],
        references_dir=references_dir,
        config=config,
        claim_texts=missing,
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
) -> None:
    """Run embedding subprocesses for one paper's parsed + web keys.

    Also pre-computes claim query vectors in the same subprocess so
    Stage 5 never loads the embedding model in the parent process.

    Raises ``RuntimeError`` if the embedding subprocess fails to actually
    produce embeddings for any requested key. ``_run_embeddings_subprocess``
    swallows all subprocess failures (non-zero exit, timeout, exception)
    with just a warning log, so we verify by re-checking has_embeddings()
    against the DB afterwards — the authoritative source of truth. Failing
    loudly here avoids a cryptic "no embeddings found" crash 20+ minutes
    later during claim verification.
    """
    from sciwrite_lint.references.embedding_store import has_embeddings
    from sciwrite_lint.references.reference_store import _get_embedding_config

    # Extract claim texts for query vector pre-computation
    claim_texts = _extract_claim_texts(config, tex_path)

    model_name, _, _ = _get_embedding_config()
    keys_to_embed = []
    for key, status in parse_results.items():
        if status == "parsed":
            keys_to_embed.append(key)
        elif status == "cached" and not has_embeddings(
            key, references_dir, model_name=model_name
        ):
            keys_to_embed.append(key)

    # Run embedding in a subprocess to isolate the CUDA context.
    # On native Linux (no CUDA overcommit), stop text vLLM first to free
    # GPU, then force CUDA device for ~50x speedup over CPU.
    embed_swap = False
    if keys_to_embed or claim_texts:
        if _needs_embedding_swap(config):
            embed_swap = True
            _stop_vllm_for_embedding(config)
        diag = _run_embeddings_subprocess(
            keys_to_embed, references_dir, config, claim_texts=claim_texts
        )
        _verify_embeddings_or_raise(
            keys_to_embed,
            references_dir,
            model_name,
            kind="parsed",
            subprocess_diag=diag,
        )

    # Embed web summaries ({key}_web.md) that lack embeddings.
    from sciwrite_lint.references.metadata import load_all_metadata

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

    if web_keys:
        if not embed_swap and _needs_embedding_swap(config):
            embed_swap = True
            _stop_vllm_for_embedding(config)
        diag = _run_embeddings_subprocess(web_keys, references_dir, config)
        _verify_embeddings_or_raise(
            web_keys,
            references_dir,
            model_name,
            kind="web summary",
            subprocess_diag=diag,
        )

    if embed_swap:
        _restart_vllm_after_embedding(config)


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
