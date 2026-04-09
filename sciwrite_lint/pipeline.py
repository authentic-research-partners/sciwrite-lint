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
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from loguru import logger

from sciwrite_lint.config import LintConfig, PaperConfig, PaperWorkspace
from sciwrite_lint.models import Citation, Finding
from sciwrite_lint.references.workspace_db import (
    get_db,
    init_pipeline_stages,
    update_pipeline_stage,
)


def _track(refs_dir: Path, stage: str, status: str, detail: str = "") -> None:
    """Write a stage status update to workspace.db."""
    try:
        with get_db(refs_dir) as conn:
            update_pipeline_stage(conn, stage, status, detail)
    except Exception as e:
        logger.debug(f"pipeline stage tracking failed ({type(e).__name__}: {e})")


class _StageStatus:
    """Mutable detail holder yielded by :func:`_stage_tracking`.

    Assign ``status.detail`` inside the ``with`` block to set the detail
    string recorded when the stage is marked ``done``.
    """

    __slots__ = ("detail",)

    def __init__(self) -> None:
        self.detail: str = ""


@contextmanager
def _stage_tracking(
    refs_dirs: Path | list[Path],
    stages: str | list[str],
) -> Iterator[_StageStatus]:
    """Mark one or more pipeline stages ``running`` → ``done`` / ``failed``.

    On enter, every (refs_dir, stage) pair is marked ``running``. On clean
    exit, every pair is marked ``done`` with the detail set on the yielded
    ``_StageStatus``. On exception, every pair is marked ``failed`` with a
    truncated error message and the exception is re-raised so the caller
    can still set ``ctx.error`` or propagate.

    Use for simple stages where ``running`` and ``done`` are tracked in the
    same scope. For stages where ``done`` is deferred to a later step (e.g.
    multi-paper parse + batch embed), track manually with :func:`_track`.
    """
    dirs = [refs_dirs] if isinstance(refs_dirs, Path) else list(refs_dirs)
    stage_list = [stages] if isinstance(stages, str) else list(stages)

    for d in dirs:
        for s in stage_list:
            _track(d, s, "running")

    status = _StageStatus()
    try:
        yield status
    except Exception as e:
        msg = str(e)[:200]
        for d in dirs:
            for s in stage_list:
                _track(d, s, "failed", msg)
        raise
    else:
        for d in dirs:
            for s in stage_list:
                _track(d, s, "done", status.detail)


@contextmanager
def _stage_failure_guard(
    refs_dirs: Path | list[Path],
    stages: str | list[str],
) -> Iterator[None]:
    """Mark stages ``failed`` if the block raises, but never marks running/done.

    Use for stages where ``running`` and ``done`` transitions are managed
    per-item outside this scope (e.g. multi-paper parse + batch embed, or
    batch cited-vision where different ctxs have different done details).
    This guard only guarantees that an uncaught batch-level failure does
    not leave stages stuck in ``running`` in the monitor DB.
    """
    dirs = [refs_dirs] if isinstance(refs_dirs, Path) else list(refs_dirs)
    stage_list = [stages] if isinstance(stages, str) else list(stages)
    try:
        yield
    except Exception as e:
        msg = str(e)[:200]
        for d in dirs:
            for s in stage_list:
                _track(d, s, "failed", msg)
        raise


# ---------------------------------------------------------------------------
# Preflight: verify all required services
# ---------------------------------------------------------------------------


async def preflight(config: LintConfig) -> list[str]:
    """Check vLLM, GROBID, network, and API configuration. Return list of errors."""
    from sciwrite_lint.cli.config import check_api_config

    errors: list[str] = check_api_config(config, needs_email=True)

    from sciwrite_lint.pdf.grobid import is_grobid_running

    if not await is_grobid_running():
        errors.append("GROBID not running. Start with: sciwrite-lint containers start")

    try:
        from sciwrite_lint.llm_utils import get_model_config

        model_cfg = get_model_config(config)
        served_name = model_cfg["model"]

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{config.llm_endpoint}/models", timeout=5.0)
            if resp.status_code != 200:
                errors.append(f"vLLM not responding at {config.llm_endpoint}")
            else:
                model_ids = [m["id"] for m in resp.json().get("data", [])]
                if served_name not in model_ids:
                    errors.append(
                        f"vLLM model mismatch: config wants '{served_name}' "
                        f"but server has {model_ids}. "
                        f"Either change [llm] model in .sciwrite-lint.toml "
                        f"or restart vLLM with the right model."
                    )
    except httpx.HTTPError as e:
        errors.append(
            f"vLLM not responding at {config.llm_endpoint}: {e}. "
            "Start with: sciwrite-lint containers start"
        )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.openalex.org/works?per_page=1", timeout=10.0
            )
            if resp.status_code != 200:
                errors.append("Network: OpenAlex API unreachable")
    except httpx.HTTPError:
        errors.append("Network: cannot reach api.openalex.org")

    return errors


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Subprocess helpers for CUDA isolation
# ---------------------------------------------------------------------------


def _run_embeddings_subprocess(
    keys: list[str],
    references_dir: Path,
    config: LintConfig,
) -> str:
    """Run embedding computation in a subprocess for CUDA isolation.

    The embedding model brings batch data to VRAM; subprocess isolation
    ensures all CUDA allocations are released when embedding finishes.

    Returns:
        Empty string on success. On failure (non-zero exit, timeout, or
        crash), returns a human-readable diagnostic (subprocess stderr
        tail, timeout notice, or exception message) for inclusion in the
        RuntimeError raised by ``_verify_embeddings_or_raise``.
    """
    import subprocess
    import sys

    # Pass keys as comma-separated arg, references_dir, and config path
    keys_str = ",".join(keys)
    cmd = [
        sys.executable,
        "-c",
        "import sys; "
        "from sciwrite_lint.pipeline import _embed_keys; "
        f"_embed_keys({keys_str!r}, {str(references_dir)!r})",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(config.project_dir) if config.project_dir else None,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.warning("Embedding subprocess failed: {}", stderr)
            return f"non-zero exit {result.returncode}: {stderr}"
    except subprocess.TimeoutExpired:
        logger.warning("Embedding subprocess timed out (300s)")
        return "subprocess timed out after 300s"
    except Exception as e:
        logger.warning("Embedding subprocess error: {}", e)
        return f"{type(e).__name__}: {e}"
    return ""


def _embed_keys(keys_csv: str, references_dir_str: str) -> None:
    """Subprocess entry point: compute embeddings for given keys."""
    from sciwrite_lint.references.reference_store import (
        compute_and_store_embeddings,
        release_embedding_model,
    )

    references_dir = Path(references_dir_str)
    keys = keys_csv.split(",")

    for key in keys:
        # Check for parsed markdown
        md_path = references_dir / "parsed" / f"{key}.md"
        # Also check for web summaries
        web_path = references_dir / f"{key}_web.md"
        text_path = (
            md_path if md_path.exists() else (web_path if web_path.exists() else None)
        )
        if text_path is None:
            continue
        try:
            text = text_path.read_text(encoding="utf-8")
            compute_and_store_embeddings(key, text, references_dir)
        except ImportError:
            break  # sentence-transformers not installed
        except Exception as e:
            logger.debug(f"embedding skipped for {key} ({type(e).__name__}: {e})")
            continue

    release_embedding_model()


def _batch_embed_entry(manifest_path_str: str) -> None:
    """Subprocess entry point: embed keys for multiple papers in one process.

    Loads the embedding model once and iterates over papers. Each paper's
    keys are embedded and stored in its own workspace. Called by
    ``_batch_embed()`` via ``subprocess.run``.

    Manifest JSON: list of {"keys": [...], "references_dir": "..."}.
    """
    import json

    from sciwrite_lint.references.reference_store import (
        compute_and_store_embeddings,
        release_embedding_model,
    )

    manifest = json.loads(Path(manifest_path_str).read_text(encoding="utf-8"))

    for entry in manifest:
        references_dir = Path(entry["references_dir"])
        keys = entry["keys"]
        for key in keys:
            md_path = references_dir / "parsed" / f"{key}.md"
            web_path = references_dir / f"{key}_web.md"
            text_path = (
                md_path
                if md_path.exists()
                else (web_path if web_path.exists() else None)
            )
            if text_path is None:
                continue
            try:
                text = text_path.read_text(encoding="utf-8")
                compute_and_store_embeddings(key, text, references_dir)
            except ImportError:
                release_embedding_model()
                return
            except Exception as e:
                logger.debug(f"embedding skipped for {key} ({type(e).__name__}: {e})")
                continue

    release_embedding_model()


def _batch_cited_vision_entry(manifest_path_str: str) -> None:
    """Subprocess entry point: run VL inference on cited papers for multiple papers.

    Loads the VL model once and iterates over papers. Each paper's cited
    PDFs are processed and results cached in workspace.db. Called by
    ``_batch_cited_vision()`` via ``subprocess.run``.

    Manifest JSON: list of {"references_dir": "...", "fresh": bool}.
    """
    import json

    manifest = json.loads(Path(manifest_path_str).read_text(encoding="utf-8"))

    for entry in manifest:
        references_dir = Path(entry["references_dir"])
        fresh = entry.get("fresh", False)

        parsed_dir = references_dir / "parsed"
        if not parsed_dir.exists():
            continue

        keys = [f.stem for f in sorted(parsed_dir.glob("*.md"))]
        if not keys:
            continue

        from sciwrite_lint.vision.describe import describe_figures
        from sciwrite_lint.vision.image_extraction import (
            ExtractedImage,
            extract_images_from_pdf,
        )

        all_images: list[ExtractedImage] = []
        output_dir = references_dir / "parsed" / "ref_figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        for key in keys:
            candidates = sorted(references_dir.glob(f"{key}*.pdf"))
            if not candidates:
                continue
            images = extract_images_from_pdf(candidates[0], output_dir / key)
            all_images.extend(images)

        if all_images:
            describe_figures(all_images, references_dir=references_dir, fresh=fresh)


# ---------------------------------------------------------------------------
# Batch launchers — spawn ONE subprocess for all papers in a stage
# ---------------------------------------------------------------------------


def _batch_vision(
    papers: list[dict[str, Any]],
    timeout: int = 600,
) -> None:
    """Run vision for all papers in one subprocess (single model load).

    Args:
        papers: List of dicts with keys: paper_name, tex_path, config_path, fresh.
        timeout: Subprocess timeout in seconds.
    """
    import json
    import subprocess
    import sys
    import tempfile

    if not papers:
        return

    manifest = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(papers, manifest)
        manifest.close()
        cmd = [
            sys.executable,
            "-m",
            "sciwrite_lint.vision.pipeline",
            "--batch",
            manifest.name,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.error(
                "Batch vision subprocess failed (exit {}): {}\n"
                "Manuscript figures will be missing from full-paper LLM "
                "checks for all papers in this batch — checks will run with "
                "reduced visual context.",
                result.returncode,
                stderr,
            )
    except subprocess.TimeoutExpired:
        logger.error(
            "Batch vision subprocess timed out ({}s) — manuscript figures "
            "missing for batch",
            timeout,
        )
    except Exception as e:
        logger.error("Batch vision subprocess error: {}: {}", type(e).__name__, e)
    finally:
        Path(manifest.name).unlink(missing_ok=True)


def _batch_embed(
    papers: list[dict[str, Any]],
    timeout: int = 600,
) -> str:
    """Run embedding for all papers in one subprocess (single model load).

    Args:
        papers: List of dicts with keys: keys (list[str]), references_dir (str).
        timeout: Subprocess timeout in seconds.

    Returns:
        Empty string on success. On failure, returns a human-readable
        diagnostic (subprocess stderr tail, timeout notice, or exception
        message) for inclusion in any per-paper error surfaced by the
        post-embed has_embeddings() re-check in ``run_papers_staged``.
    """
    import json
    import subprocess
    import sys
    import tempfile

    if not papers:
        return ""

    # Filter out papers with no keys to embed
    papers = [p for p in papers if p.get("keys")]
    if not papers:
        return ""

    manifest = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(papers, manifest)
        manifest.close()
        cmd = [
            sys.executable,
            "-c",
            "from sciwrite_lint.pipeline import _batch_embed_entry; "
            f"_batch_embed_entry({manifest.name!r})",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Batch embedding subprocess timed out ({}s)", timeout)
            return f"subprocess timed out after {timeout}s"
        except Exception as e:
            logger.warning("Batch embedding subprocess error: {}", e)
            return f"{type(e).__name__}: {e}"
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.warning("Batch embedding subprocess failed: {}", stderr)
            return f"non-zero exit {result.returncode}: {stderr}"
        return ""
    finally:
        Path(manifest.name).unlink(missing_ok=True)


def _batch_cited_vision(
    papers: list[dict[str, Any]],
    timeout: int = 600,
) -> None:
    """Run cited-paper vision for all papers in one subprocess (single model load).

    Args:
        papers: List of dicts with keys: references_dir (str), fresh (bool).
        timeout: Subprocess timeout in seconds.
    """
    import json
    import subprocess
    import sys
    import tempfile

    if not papers:
        return

    manifest = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(papers, manifest)
        manifest.close()
        cmd = [
            sys.executable,
            "-c",
            "from sciwrite_lint.pipeline import _batch_cited_vision_entry; "
            f"_batch_cited_vision_entry({manifest.name!r})",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.error(
                "Batch cited vision subprocess failed (exit {}): {}\n"
                "Cited paper figures will be missing for all papers in this "
                "batch — ref-internal checks will run with reduced context.",
                result.returncode,
                stderr,
            )
    except subprocess.TimeoutExpired:
        logger.error(
            "Batch cited vision subprocess timed out ({}s) — cited figures "
            "missing for batch",
            timeout,
        )
    except Exception as e:
        logger.error("Batch cited vision subprocess error: {}: {}", type(e).__name__, e)
    finally:
        Path(manifest.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Stage 0.5: Vision — figure descriptions for full-paper consistency checks
# ---------------------------------------------------------------------------


def _stage_vision(
    tex_path: Path,
    config: LintConfig,
    paper_name: str,
    fresh: bool = False,
) -> None:
    """Extract and describe manuscript figures via Qwen3-VL-2B.

    Populates the vision cache so that full-paper consistency checks
    (Stage 2) can include figure descriptions in the shared prefix.

    Runs in a subprocess so the CUDA context (~500 MB) is fully released
    when the VL model finishes. Without subprocess isolation, the CUDA
    runtime context persists in the parent process even after del model +
    gc.collect() + empty_cache(), stealing VRAM from vLLM's KV cache and
    causing timeouts on LLM queries.

    Results are written to workspace.db (vision_cache table) by the
    subprocess; the parent reads them from DB — no return value transfer.
    """
    import subprocess
    import sys

    # Build a command that runs the vision pipeline in isolation.
    # Uses the same Python interpreter and config.
    cmd = [
        sys.executable,
        "-m",
        "sciwrite_lint.vision.pipeline",
        str(tex_path),
        "--paper",
        paper_name,
    ]
    if fresh:
        cmd.append("--fresh")
    if config.config_path:
        cmd.extend(["--config", str(config.config_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(config.project_dir) if config.project_dir else None,
        )
        if result.returncode == 0:
            logger.info("Vision: figure descriptions ready for full-paper checks")
        else:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.error(
                "Vision subprocess exited with code {}: {}\n"
                "Manuscript figures missing — full-paper checks will run "
                "with reduced visual context.",
                result.returncode,
                stderr,
            )
    except subprocess.TimeoutExpired:
        logger.error(
            "Vision subprocess timed out (300s) — manuscript figures missing, "
            "checks continue with reduced visual context"
        )
    except Exception as e:
        # Vision is best-effort — don't block the pipeline
        logger.error(
            "Vision pipeline failed ({}: {}) — checks continue without figures",
            type(e).__name__,
            e,
        )


# ---------------------------------------------------------------------------
# Stage 1: Manuscript + local-LLM checks (reuses existing logic)
# ---------------------------------------------------------------------------


def run_text_checks(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Run all manuscript-engine checks (CPU-bound, no I/O). Returns findings."""
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_checks

    ensure_checks_loaded()
    findings: list[Finding] = []

    for meta, fn in get_checks(config=config):
        if meta.category in ("reference-db", "local-llm"):
            continue
        try:
            check_findings = fn(tex_path, config)
            for f in check_findings:
                override = config.effective_severity(meta.id, meta.severity)
                if override != f.level:
                    f.level = override  # type: ignore[assignment]
            findings.extend(check_findings)
        except Exception as e:
            logger.warning(f"Check {meta.id} skipped: {e}")
            findings.append(
                Finding(
                    level="info",
                    rule_id=meta.id,
                    message=f"Check {meta.id} could not run (internal error)",
                    context=f"{type(e).__name__}: {e!s}"[:200],
                )
            )

    return findings


async def run_llm_checks_batched(tex_path: Path, config: LintConfig) -> list[Finding]:
    """Run all local-llm-engine checks via batched vLLM queries. Returns findings."""
    from sciwrite_lint.cli.check import (
        run_llm_checks_batched as _run_llm_checks_batched,
    )
    from sciwrite_lint.checks.registry import ensure_checks_loaded, get_checks

    ensure_checks_loaded()
    llm_checks = [
        (meta, fn)
        for meta, fn in get_checks(config=config)
        if meta.category == "local-llm"
    ]
    if not llm_checks:
        return []
    return await _run_llm_checks_batched(llm_checks, tex_path, config)


# ---------------------------------------------------------------------------
# Stage 2: Batch API verification
# ---------------------------------------------------------------------------


def _register_ref_in_workspace(
    meta: Any,
    references_dir: Path,
) -> None:
    """Register a verified reference in the workspace DB for cross-depth dedup."""
    from sciwrite_lint.references.workspace_db import get_db, register_reference

    canonical = meta.canonical or {}
    bibitem = meta.bibitem or {}
    authors = canonical.get("authors") or bibitem.get("authors") or None

    try:
        with get_db(references_dir) as conn:
            register_reference(
                conn,
                ref_key=meta.key,
                workspace_path=".",
                depth=0,
                parent_key="",
                doi=canonical.get("doi") or bibitem.get("doi"),
                arxiv_id=canonical.get("arxiv_id") or bibitem.get("arxiv_id"),
                pmid=canonical.get("pmid") or bibitem.get("pmid"),
                pmcid=canonical.get("pmcid") or bibitem.get("pmc_id"),
                isbn=canonical.get("isbn") or bibitem.get("isbn"),
                lccn=canonical.get("lccn") or bibitem.get("lccn"),
                title=canonical.get("title") or bibitem.get("title"),
                authors=authors if isinstance(authors, list) else None,
            )
    except Exception:
        logger.debug("Failed to register {} in workspace DB", meta.key)


async def _stage_verify(
    citations: list[Citation],
    config: LintConfig,
    references_dir: Path,
    *,
    fresh: bool = False,
) -> list[Finding]:
    """Verify citations against external sources. Returns ref-* findings.

    Verification proceeds through five phases, each narrowing the set of
    unresolved references before the next phase runs:

      Phase A  — OpenAlex batch      (single request, resolves DOI/arXiv/PMID)
      Phase B  — Semantic Scholar     (batch, resolves DOI/arXiv/PMID/PMC)
      Phase C  — CrossRef parallel    (per-citation, resolves DOI + title search)
      Phase C2 — Open Library + LoC   (per-citation, resolves ISBN → OL, LCCN → LoC;
                                       OL and LoC run in parallel per citation)
      Phase D  — URL verification     (per-citation, HEAD/GET check for refs with URLs;
                                       last resort before marking not_found)

    Web resources (@misc with URL, no DOI) bypass all phases and go directly
    to URL verification (concurrent, semaphore-limited).

    Each phase only processes citations still unresolved after prior phases.
    Results are validated by ``_id_result_matches`` (composite title/author/year
    score ≥ 0.40) to reject API results that don't match the bib entry.
    """
    from sciwrite_lint.api import (
        _id_result_matches,
        batch_openalex,
        batch_s2,
        cross_validate_ids,
        parallel_crossref,
    )
    from sciwrite_lint.references.citations import is_web_resource
    from sciwrite_lint.references.matching import compare_citation_detailed
    from sciwrite_lint.references.metadata import (
        build_metadata_from_citation,
        merge_source_paper,
        save_metadata,
    )

    findings: list[Finding] = []

    # Partition: already verified vs needs verification
    # Single DB query to get all already-verified metadata
    unverified: list[Citation] = []
    web_citations: list[Citation] = []
    if not fresh:
        from sciwrite_lint.references.workspace_db import (
            get_db,
            query_verified_metadata,
        )

        with get_db(references_dir) as conn:
            cached_meta = query_verified_metadata(conn)
    else:
        cached_meta = {}

    for c in citations:
        if not fresh:
            existing = cached_meta.get(c.key)
            if existing:
                # Already verified — collect findings from stored issues
                for issue in existing.issues:
                    level, rule_id = _classify_verify_issue(issue)
                    findings.append(
                        Finding(
                            level=level,
                            rule_id=rule_id,
                            message=f"{c.key}: {issue}",
                            file="",
                            context=f"Source: {existing.api_source}"
                            if existing.api_source
                            else "",
                        )
                    )
                c.api_match = existing.api_match
                c.tier = existing.access.get("tier", "")
                continue
        if is_web_resource(c):
            web_citations.append(c)
        else:
            unverified.append(c)

    if not unverified and not web_citations:
        return findings

    total = len(unverified) + len(web_citations)
    logger.info(
        f"Verifying {total} citations ({len(unverified)} academic, {len(web_citations)} web)"
    )

    # ------------------------------------------------------------------
    # Phases A–C: Batch academic API lookups (DOI, arXiv, PMID, title)
    # ------------------------------------------------------------------
    resolved: dict[str, dict[str, Any]] = {}

    if unverified:
        # Phase A: OpenAlex batch — single request, covers DOI + arXiv DOI
        oa_results = await batch_openalex(unverified, config)
        resolved.update(oa_results)
        oa_found = len(oa_results)

        # Phase B: Semantic Scholar batch — DOI, arXiv ID, PMID, PMC
        still_need = [c for c in unverified if c.key not in resolved]
        if still_need:
            s2_results = await batch_s2(still_need, config)
            resolved.update(s2_results)

        # Phase C: CrossRef parallel — DOI + title/author search
        cr_need = [c for c in unverified if c.key not in resolved]
        if cr_need:
            cr_results = await parallel_crossref(cr_need, config)
            for key, result in cr_results.items():
                if key not in resolved:
                    resolved[key] = result

        # ------------------------------------------------------------------
        # Phase C2: Open Library (ISBN) + Library of Congress (LCCN)
        #
        # Books and government reports often have ISBN/LCCN but no DOI, so
        # academic APIs miss them. Per-citation lookups run concurrently;
        # OL and LoC run in parallel per citation when both IDs are present.
        # ------------------------------------------------------------------
        from sciwrite_lint._network import is_valid_isbn, is_valid_lccn

        ol_need = [
            c
            for c in unverified
            if c.key not in resolved
            and (
                (c.isbn and is_valid_isbn(c.isbn)) or (c.lccn and is_valid_lccn(c.lccn))
            )
        ]
        if ol_need:
            from sciwrite_lint.api import CitationAPI

            async with CitationAPI(config=config) as cat_api:
                ol_sem = asyncio.Semaphore(5)

                async def _try_ol_loc(c: Citation) -> None:
                    async with ol_sem:
                        has_isbn = c.isbn and is_valid_isbn(c.isbn)
                        has_lccn = c.lccn and is_valid_lccn(c.lccn)
                        tasks: list[asyncio.Task] = []
                        if has_isbn:
                            tasks.append(
                                asyncio.create_task(cat_api._openlibrary_lookup(c))
                            )
                        if has_lccn:
                            tasks.append(asyncio.create_task(cat_api._loc_lookup(c)))
                        results = await asyncio.gather(*tasks)
                        for result in results:
                            if result and not result.get("error"):
                                if _id_result_matches(c, result):
                                    resolved[c.key] = result
                                    return

                await asyncio.gather(*[_try_ol_loc(c) for c in ol_need])
                ol_found = sum(1 for c in ol_need if c.key in resolved)

            if ol_found:
                logger.info(f"Open Library / LoC: {ol_found} found")

        logger.info(
            f"API results: {oa_found} OpenAlex, "
            f"{len(resolved) - oa_found} S2/CrossRef/OL/LoC, "
            f"{len(unverified) - len(resolved)} not found"
        )

    # ------------------------------------------------------------------
    # Phase D: URL verification (last resort)
    #
    # References not found in any academic/book API but carrying a URL
    # (techreports, journalism, books hosted online, conference pages).
    # Confirms the URL is alive (→ web_verified) or dead (→ web_dead)
    # before falling through to not_found.
    # ------------------------------------------------------------------
    url_need = [c for c in unverified if c.key not in resolved and c.url]
    if url_need:
        from sciwrite_lint.api import _verify_web_resource

        url_sem = asyncio.Semaphore(5)

        async def _try_url(c: Citation) -> None:
            async with url_sem:
                # Each task gets its own httpx client — sharing a client
                # across concurrent streaming requests causes gzip
                # decompression errors (corrupt shared decoder state).
                async with httpx.AsyncClient(
                    timeout=15.0, follow_redirects=True
                ) as client:
                    await _verify_web_resource(c, client, references_dir)

        await asyncio.gather(*[_try_url(c) for c in url_need])

        url_resolved_keys = {
            c.key for c in url_need if c.api_match in ("web_verified", "web_dead")
        }
        logger.info(
            f"URL verification: {len(url_resolved_keys)}/{len(url_need)} resolved"
        )

    # Apply results and run metadata comparison
    for c in unverified:
        result = resolved.get(c.key)  # type: ignore[assignment]
        if result and not result.get("error"):
            c.api_data = result
            c.api_source = result.get("source", "")
            c.issues.extend(compare_citation_detailed(c, result))
        elif c.api_match not in ("web_verified", "web_dead"):
            # Not found in any API and no URL (or URL not yet checked)
            c.api_match = "not_found"
            c.issues.append(
                "Not found in CrossRef, OpenAlex, Semantic Scholar, Open Library, or Library of Congress"
            )

    # ------------------------------------------------------------------
    # Phase E: Cross-validate identifiers
    #
    # For verified citations with multiple IDs, look up each ID
    # independently via OpenAlex and verify they all resolve to the
    # same paper. Catches LLM-mixed bib entries where DOI → paper A
    # but arXiv ID → paper B.
    # ------------------------------------------------------------------
    cross_need = [c for c in unverified if c.api_data and not c.api_data.get("error")]
    if cross_need:
        cross_sem = asyncio.Semaphore(5)

        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True
        ) as cross_client:

            async def _cross_validate(c: Citation) -> None:
                async with cross_sem:
                    cross_issues = await cross_validate_ids(
                        c, c.api_data, config, client=cross_client
                    )
                    c.issues.extend(cross_issues)

            await asyncio.gather(*[_cross_validate(c) for c in cross_need])

    # Set api_match based on all issues (including cross-validation).
    # Only for citations resolved via academic APIs — don't overwrite
    # web_verified / web_dead status set by URL verification.
    for c in unverified:
        result = resolved.get(c.key)  # type: ignore[assignment]
        if result and not result.get("error"):
            c.api_match = (
                "mismatch"
                if any("mismatch" in i.lower() for i in c.issues)
                else "verified"
            )

    # Persist and generate findings (single DB connection for all writes)
    from sciwrite_lint.references.workspace_db import (
        get_db,
        load_citation_metadata,
        save_citation_metadata,
    )

    with get_db(references_dir) as conn:
        for c in unverified:
            result = resolved.get(c.key)  # type: ignore[assignment]

            existing = load_citation_metadata(conn, c.key)
            meta = build_metadata_from_citation(
                c, result, references_dir=references_dir
            )
            if existing:
                merge_source_paper(meta, c.source_paper)
                for sp in existing.bibitem.get("source_papers", []):
                    merge_source_paper(meta, sp)
                if existing.manual_override:
                    meta.manual_override = existing.manual_override
            save_citation_metadata(conn, meta)
            _register_ref_in_workspace(meta, references_dir)
            c.tier = meta.access.get("tier", "")

            for issue in c.issues:
                level, rule_id = _classify_verify_issue(issue)
                findings.append(
                    Finding(
                        level=level,
                        rule_id=rule_id,
                        message=f"{c.key}: {issue}",
                        file="",
                        context=f"Source: {c.api_source}" if c.api_source else "",
                    )
                )

    # Web resources (concurrent — independent HTTP checks)
    if web_citations:
        from sciwrite_lint.api import _verify_web_resource

        sem = asyncio.Semaphore(5)

        async def _verify_one_web(c: Citation) -> None:
            async with sem:
                async with httpx.AsyncClient(
                    timeout=15.0, follow_redirects=True
                ) as client:
                    result = await _verify_web_resource(c, client, references_dir)
            meta = build_metadata_from_citation(
                c, result, references_dir=references_dir
            )
            save_metadata(meta, references_dir)
            c.tier = meta.access.get("tier", "")

        await asyncio.gather(*[_verify_one_web(c) for c in web_citations])

        for c in web_citations:
            for issue in c.issues:
                level, rule_id = _classify_verify_issue(issue)
                findings.append(
                    Finding(
                        level=level,
                        rule_id=rule_id,
                        message=f"{c.key}: {issue}",
                        file="",
                        context=f"Source: {c.api_source}" if c.api_source else "",
                    )
                )

    # Confirm venue mismatches via vLLM (suppresses false positives like
    # "NeurIPS" vs "Advances in Neural Information Processing Systems")
    findings = await _confirm_venue_findings(findings, config)

    # Retraction Watch enrichment: check all metadata against RW database
    from sciwrite_lint.references.retraction_watch import ensure_rw_database

    rw_db = await ensure_rw_database(config)
    if rw_db:
        from sciwrite_lint.references.metadata import enrich_retraction_status
        from sciwrite_lint.references.workspace_db import (
            get_db,
            load_all_citation_metadata,
            save_citation_metadata,
        )

        with get_db(references_dir) as conn:
            all_meta = load_all_citation_metadata(conn)
            for _key, meta in all_meta.items():
                if enrich_retraction_status(meta, rw_db):
                    save_citation_metadata(conn, meta)

    return findings


async def _confirm_venue_findings(
    findings: list[Finding],
    config: LintConfig,
) -> list[Finding]:
    """Filter venue mismatch findings through vLLM confirmation.

    If vLLM is available and says the venues match, the finding is dropped.
    If vLLM is unavailable, all findings pass through unchanged.
    """
    from sciwrite_lint.references.matching import venue_match_llm

    venue_findings = []
    other_findings = []
    for f in findings:
        if "Venue mismatch" in f.message:
            venue_findings.append(f)
        else:
            other_findings.append(f)

    if not venue_findings:
        return findings

    # Extract venue pairs from finding messages
    import re

    sem = asyncio.Semaphore(50)

    async def _confirm_one(f: Finding) -> Finding | None:
        m = re.search(r"tex='([^']*)', (?:canonical|API)='([^']*)'", f.message)
        if not m:
            return f  # can't parse, keep
        async with sem:
            try:
                same = await venue_match_llm(m.group(1), m.group(2), config=config)
            except Exception as e:
                logger.debug("Venue match LLM failed: {}", e)
                return f  # vLLM error, keep the finding
        if same is True:
            return None  # vLLM confirmed same venue — suppress
        return f

    results = await asyncio.gather(*[_confirm_one(f) for f in venue_findings])
    confirmed = [r for r in results if r is not None]

    suppressed = len(venue_findings) - len(confirmed)
    if suppressed:
        logger.info(f"Venue: {suppressed} false positive(s) suppressed by vLLM")

    return other_findings + confirmed


# ---------------------------------------------------------------------------
# Reference accuracy (post-verify, no API calls)
# ---------------------------------------------------------------------------


def _stage_reference_accuracy(
    config: LintConfig,
    references_dir: Path,
) -> list[Finding]:
    """Run reference-accuracy check on stored metadata. No API calls."""
    from sciwrite_lint.checks.reference_accuracy import (
        check_reference_accuracy_from_metadata,
    )
    from sciwrite_lint.references.metadata import load_all_metadata

    if not config.is_check_enabled("reference-accuracy"):
        return []
    if not references_dir.exists():
        return []

    all_metadata = load_all_metadata(references_dir)
    if not all_metadata:
        return []

    findings = check_reference_accuracy_from_metadata(all_metadata)

    # Apply severity override
    override = config.effective_severity("reference-accuracy", "warning")
    for f in findings:
        if f.level == "warning" and override != "warning":
            f.level = override  # type: ignore[assignment]

    if findings:
        logger.info(f"Reference accuracy: {len(findings)} issues")
    return findings


# ---------------------------------------------------------------------------
# Stage 3: Full-text acquisition
# ---------------------------------------------------------------------------


async def _stage_fetch(
    citations: list[Citation],
    config: LintConfig,
    references_dir: Path,
    embed_inline: bool = True,
) -> int:
    """Download PDFs for citations missing local files. Returns count fetched.

    Args:
        embed_inline: If True (default), eagerly embed parsed PDFs in-process.
            Set to False in batch-staged mode to avoid loading the embedding
            model's CUDA context in the parent process — the batch embedding
            subprocess handles all embeddings instead.
    """
    from sciwrite_lint.fulltext import acquire_fulltext
    from sciwrite_lint.references.metadata import (
        compute_tier,
        load_metadata,
        save_metadata,
    )
    from sciwrite_lint.references.reference_store import parse_and_embed
    from sciwrite_lint.usage import tracked

    need_fetch = []

    for c in citations:
        meta = load_metadata(c.key, references_dir)
        if not meta:
            continue
        tier = meta.access.get("tier", "")
        local = meta.access.get("local_file", "")
        if tier == "T1" and local and (references_dir / local).exists():
            continue
        if tier == "T3" or meta.api_match in ("not_found", "web_verified", "web_dead"):
            continue
        need_fetch.append((c, meta))

    if not need_fetch:
        return 0

    # Check local_pdfs_dir for user-provided PDFs before downloading
    from sciwrite_lint.local_pdfs import copy_local_pdf, match_local_pdfs

    local_pdfs_dir = config.local_pdfs_dir
    if local_pdfs_dir.is_dir() and any(local_pdfs_dir.glob("*.pdf")):
        titles = {
            c.key: (c.title or meta.canonical.get("title", ""))
            for c, meta in need_fetch
        }
        matched, _ = match_local_pdfs(local_pdfs_dir, titles)

        still_need = []
        for c, meta in need_fetch:
            if c.key in matched:
                local_path = copy_local_pdf(matched[c.key], c.key, references_dir)
                meta.access["local_file"] = local_path
                meta.access["tier"] = compute_tier(meta)
                save_metadata(meta, references_dir)
                logger.info(f"{c.key}: using local PDF [{meta.access['tier']}]")
            else:
                still_need.append((c, meta))
        need_fetch = still_need

    if not need_fetch:
        return 0

    logger.info(f"Fetching full text for {len(need_fetch)} citations")
    sem = asyncio.Semaphore(5)  # max 5 concurrent downloads
    fetched_keys: list[str] = []

    async def _fetch_one(c: Citation, meta) -> None:
        doi = meta.canonical.get("doi") or c.doi
        arxiv_id = meta.canonical.get("arxiv_id")
        oa_url = meta.access.get("oa_url")
        s2_pdf_url = meta.canonical.get("s2_pdf_url")
        pmcid = meta.canonical.get("pmcid")
        expected_title = c.title or meta.canonical.get("title", "")
        expected_authors = meta.canonical.get("authors") or c.authors

        async with sem:
            async with tracked("fetch"):
                result = await acquire_fulltext(
                    c.key,
                    references_dir,
                    config=config,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    oa_url=oa_url,
                    s2_pdf_url=s2_pdf_url,
                    pmcid=pmcid,
                    expected_title=expected_title,
                    expected_authors=expected_authors,
                    progress=False,
                )

        if result.found and result.local_path:
            meta.access["local_file"] = result.local_path
            meta.access["tier"] = compute_tier(meta)
            save_metadata(meta, references_dir)
            fetched_keys.append(c.key)
            logger.info(f"{c.key}: downloaded [{meta.access['tier']}]")

            # Eager parse (also concurrent — GROBID handles it).
            # In batch mode (embed_inline=False), skip in-process embedding
            # to avoid loading the CUDA context in the parent process.
            if result.local_path.endswith(".pdf"):
                try:
                    text, chunks = await parse_and_embed(
                        c.key,
                        references_dir / result.local_path,
                        references_dir,
                        embed=embed_inline,
                    )
                    if text:
                        suffix = f", {chunks} chunks" if chunks else ""
                        logger.debug(f"{c.key}: parsed {len(text)} chars{suffix}")

                    # Store formal classification in citation metadata
                    from sciwrite_lint.references.reference_store import (
                        is_formal_cached,
                    )

                    formal = is_formal_cached(c.key, references_dir)
                    meta.access["is_formal"] = formal
                    save_metadata(meta, references_dir)
                    if not formal:
                        logger.info(
                            f"{c.key}: non-formal document — "
                            f"text available, no reference extraction"
                        )
                except Exception as e:
                    logger.warning(f"{c.key}: parse failed: {e}")
        elif not result.found:
            if result.reason:
                meta.access["acquisition_reason"] = result.reason
                save_metadata(meta, references_dir)
            logger.debug(
                f"{c.key}: PDF not acquired (stays T2)"
                + (f" — {result.reason}" if result.reason else "")
            )

        if result.abstract and not meta.canonical.get("abstract"):
            meta.canonical["abstract"] = result.abstract
            meta.access["tier"] = compute_tier(meta)
            save_metadata(meta, references_dir)

    await asyncio.gather(*[_fetch_one(c, meta) for c, meta in need_fetch])
    return len(fetched_keys)


# ---------------------------------------------------------------------------
# Stage 4: GROBID parse + embeddings
# ---------------------------------------------------------------------------


async def _stage_parse(
    config: LintConfig,
    references_dir: Path,
    parse_sem: asyncio.Semaphore | None = None,
    skip_embeddings: bool = False,
) -> tuple[int, int]:
    """Parse unparsed PDFs via GROBID + build embeddings. Returns (parsed_count, cached_count).

    Args:
        skip_embeddings: If True, skip the embedding subprocess. Used by
            ``run_papers_staged()`` which runs embedding in a single batch
            subprocess across all papers (see ``_batch_embed``).
    """
    from sciwrite_lint.references.reference_store import parse_all_missing

    results = await parse_all_missing(references_dir, sem=parse_sem)

    cached = sum(1 for v in results.values() if v == "cached")
    parsed = sum(1 for v in results.values() if v == "parsed")
    failed = sum(1 for v in results.values() if v == "failed")

    if not skip_embeddings:
        _run_embeddings_for_paper(results, references_dir, config)

    parts = []
    if parsed:
        parts.append(f"{parsed} new")
    if cached:
        parts.append(f"{cached} cached")
    if failed:
        parts.append(f"{failed} failed")
    if parts:
        logger.info("Parse: {}", ", ".join(parts))

    return parsed, cached


def _run_embeddings_for_paper(
    parse_results: dict[str, str],
    references_dir: Path,
    config: LintConfig,
) -> None:
    """Run embedding subprocesses for one paper's parsed + web keys.

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
    # The embedding model (~1.2 GB on WSL2/CUDA) brings batch data to
    # VRAM; subprocess isolation ensures all CUDA allocations are released
    # when embedding finishes, freeing VRAM for vLLM claim verification.
    if keys_to_embed:
        diag = _run_embeddings_subprocess(keys_to_embed, references_dir, config)
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
        diag = _run_embeddings_subprocess(web_keys, references_dir, config)
        _verify_embeddings_or_raise(
            web_keys,
            references_dir,
            model_name,
            kind="web summary",
            subprocess_diag=diag,
        )


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
            "\n\nCommon causes: CUDA OOM, subprocess timeout (300s default), "
            "model load failure, or sqlite-vec DB lock."
        )
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Stage 4.6: Bibliography verification (existence + metadata + retraction)
# ---------------------------------------------------------------------------


def _collect_parse_hashes(references_dir: Path) -> dict[str, str]:
    """Collect MD5 hashes of parsed markdown files for cache invalidation."""
    import hashlib

    parsed_dir = references_dir / "parsed"
    if not parsed_dir.exists():
        return {}
    hashes: dict[str, str] = {}
    for md_path in parsed_dir.glob("*.md"):
        content = md_path.read_bytes()
        hashes[md_path.stem] = hashlib.md5(content, usedforsecurity=False).hexdigest()
    return hashes


async def _stage_bib_verify(
    config: LintConfig,
    references_dir: Path,
    fresh: bool = False,
) -> list[Any]:
    """Verify bibliography entries of parsed formal references via APIs.

    Runs after parse (Stage 4) and concurrently with claim verification
    (Stage 5). Uses structured GROBID data stored in workspace.db at
    parse time. No GPU needed — API calls only.

    Results are cached in workspace.db (bib_checks table), invalidated
    when a reference's parsed markdown changes (hash mismatch).
    """
    from sciwrite_lint.references.workspace_db import (
        get_db,
        load_bib_checks as load_bib_checks_db,
        save_bib_checks,
    )
    from sciwrite_lint.scoring.chain import RefBibCheck

    with get_db(references_dir) as conn:
        # Compute parse hashes for cache invalidation
        parse_hashes = _collect_parse_hashes(references_dir)

        if not fresh:
            cached = load_bib_checks_db(conn, parse_hashes=parse_hashes)
            if cached:
                logger.info("Bibliography verification: {} cached results", len(cached))
                return [RefBibCheck(**c) for c in cached]

        from sciwrite_lint.scoring.chain import run_bib_verification

        results = await run_bib_verification(references_dir, config)

        # Cache in workspace.db with parse hashes
        if results:
            save_bib_checks(
                conn,
                [r.model_dump() for r in results],
                parse_hashes=parse_hashes,
            )

        return results


# ---------------------------------------------------------------------------
# Stage 5: Claim verification → sem-001 findings
# ---------------------------------------------------------------------------


async def _stage_claims(
    paper_name: str,
    tex_path: Path,
    config: LintConfig,
    references_dir: Path,
    bib_path: Path | None = None,
    rerun: bool = False,
) -> tuple[list[Finding], list[dict]]:
    """Run claim verification. Returns (findings, raw_results)."""
    from sciwrite_lint.eval_claims import run_claim_verification

    results = await run_claim_verification(
        paper_name,
        tex_path,
        references_dir,
        config=config,
        bib_path=bib_path,
        backend="vllm",
        model=config.llm_model or "",
        rerun=rerun,
    )

    return _claims_to_findings(results, tex_path), results


def _claims_to_findings(results: list[dict], tex_path: Path) -> list[Finding]:
    """Convert claim verification results to findings. Delegates to check modules."""
    from sciwrite_lint.checks.claim_support import claims_to_findings
    from sciwrite_lint.checks.cite_purpose import cite_purposes_to_findings

    findings = claims_to_findings(results, tex_path)
    findings.extend(cite_purposes_to_findings(results, tex_path))
    return findings


# ---------------------------------------------------------------------------
# Stage 4.5: Reference internal consistency checks
# ---------------------------------------------------------------------------


def _stage_cited_vision(
    references_dir: Path,
    fresh: bool = False,
) -> dict[str, str]:
    """Stage 4.2: Describe figures from cited paper PDFs.

    Runs VL model in a subprocess (same isolation as _stage_vision) after
    the embedding model is unloaded (Stage 4) and before vLLM ref_internal
    queries (Stage 4.5).

    The subprocess runs VL inference and caches results in workspace.db.
    The parent process reads cached descriptions from DB afterwards.

    Returns {ref_key: figure_descriptions_str} for injection into
    ref_internal consistency queries.
    """
    import subprocess
    import sys

    from sciwrite_lint.vision.cache import format_descriptions_from_db
    from sciwrite_lint.vision.image_extraction import (
        ExtractedImage,
        extract_images_from_pdf,
    )

    parsed_dir = references_dir / "parsed"
    if not parsed_dir.exists():
        return {}

    keys = [f.stem for f in sorted(parsed_dir.glob("*.md"))]
    if not keys:
        return {}

    # Collect images and their ref_key ranges (lightweight, no GPU)
    all_images: list[ExtractedImage] = []
    ref_image_ranges: dict[str, tuple[int, int]] = {}

    output_dir = references_dir / "parsed" / "ref_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        candidates = sorted(references_dir.glob(f"{key}*.pdf"))
        if not candidates:
            continue
        start = len(all_images)
        images = extract_images_from_pdf(candidates[0], output_dir / key)
        all_images.extend(images)
        if images:
            ref_image_ranges[key] = (start, len(all_images))

    if not all_images:
        return {}

    # Dynamic timeout: ~5s per image (conservative, GPU batch=16), 60s for model load
    timeout = max(120, 60 + len(all_images) * 5)

    # Run VL inference in subprocess to isolate CUDA context
    cmd = [
        sys.executable,
        "-c",
        "from sciwrite_lint.checks.ref_internal_checks import _describe_cited_figures_vl; "
        f"_describe_cited_figures_vl({str(references_dir)!r}, {fresh!r})",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[-1500:] if result.stderr else ""
            logger.error(
                "Cited vision subprocess failed (exit {}): {}\n"
                "Cited figures missing — ref-internal checks will run with "
                "reduced visual context.",
                result.returncode,
                stderr,
            )
    except subprocess.TimeoutExpired:
        logger.error(
            "Cited vision subprocess timed out ({}s, {} images) — figures missing",
            timeout,
            len(all_images),
        )
    except Exception as e:
        logger.error("Cited vision subprocess error: {}: {}", type(e).__name__, e)

    # Read cached descriptions from DB (written by subprocess)
    descriptions: dict[str, str] = {}
    for key, (start, end) in ref_image_ranges.items():
        ref_images = all_images[start:end]
        desc = format_descriptions_from_db(ref_images, references_dir)
        if desc:
            descriptions[key] = desc

    if descriptions:
        logger.info(
            "Cited paper figures: {} papers, {} images described",
            len(descriptions),
            len(all_images),
        )

    return descriptions


async def _stage_ref_internal(
    references_dir: Path,
    config: LintConfig,
    *,
    fresh: bool = False,
    ref_figure_descs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run consistency checks on cited papers (automatic in default pipeline)."""
    from sciwrite_lint.checks.ref_internal_checks import run_ref_internal_checks

    return await run_ref_internal_checks(
        references_dir,
        config,
        fresh=fresh,
        ref_figure_descs=ref_figure_descs,
    )


# ---------------------------------------------------------------------------
# Stage 6: Reference-unreliable (aggregate reliability signals)
# ---------------------------------------------------------------------------


def _stage_unreliable(
    tex_path: Path,
    references_dir: Path,
    claim_results: list[dict],
    bib_checks: list[Any] | None = None,
) -> list[Finding]:
    """Aggregate reliability signals into reference-unreliable findings.

    Uses claim results (deep path) when available, otherwise metadata only.
    Bibliography checks are passed from the pipeline (Stage 4.6).
    """
    from sciwrite_lint.checks.reference_unreliable import (
        claims_to_unreliable_findings,
        metadata_to_unreliable_findings,
    )
    from sciwrite_lint.references.metadata import load_all_metadata

    all_meta = load_all_metadata(references_dir)
    if not all_meta:
        return []

    if claim_results:
        return claims_to_unreliable_findings(
            claim_results,
            tex_path,
            metadata_map=all_meta,
            bib_checks=bib_checks,
        )
    return metadata_to_unreliable_findings(
        all_meta,
        tex_path,
        bib_checks=bib_checks,
    )


# ---------------------------------------------------------------------------
# Issue classifier (moved from __main__.py, shared)
# ---------------------------------------------------------------------------


def _format_usage_summary(run: Any) -> str:
    """One-line summary of external tool calls for terminal output."""
    parts: list[str] = []

    # APIs
    api_parts: list[str] = []
    for name, svc in [
        ("OpenAlex", run.openalex),
        ("S2", run.semantic_scholar),
        ("CrossRef", run.crossref),
    ]:
        if svc.calls:
            api_parts.append(f"{svc.calls} {name}")
    if api_parts:
        parts.append(f"APIs: {', '.join(api_parts)}")

    # GROBID
    if run.grobid.calls:
        parts.append(f"GROBID: {run.grobid.calls} parsed")

    # Fetch
    if run.fetch.calls:
        parts.append(f"Fetch: {run.fetch.calls} downloads")

    # vLLM
    if run.vllm.calls:
        tokens = run.vllm.extra.get("prompt_tokens", 0) + run.vllm.extra.get(
            "completion_tokens", 0
        )
        token_str = f", {tokens:,} tokens" if tokens else ""
        err_str = f", {run.vllm.errors} errors" if run.vllm.errors else ""
        parts.append(f"vLLM: {run.vllm.calls} calls{token_str}{err_str}")

    return " | ".join(parts) if parts else "No external calls (all cached)"


def _classify_verify_issue(
    issue: str,
) -> tuple[Literal["error", "warning", "info"], str]:
    """Classify a verify issue string into (level, check_id).

    Accuracy-related mismatches (title, author, year, venue) route to
    reference-accuracy. Existence issues (not found, dead URL, retracted)
    stay under reference-exists.
    """
    issue_lower = issue.lower()
    # Existence issues → reference-exists
    if "dead url" in issue_lower:
        return "error", "reference-exists"
    if "web content saved" in issue_lower:
        return "info", "reference-exists"
    if "content extraction failed" in issue_lower:
        return "warning", "reference-exists"
    if "not found" in issue_lower:
        return "error", "reference-exists"
    if "retracted" in issue_lower:
        return "error", "retracted-cite"
    # Accuracy issues → reference-accuracy
    if "mismatch" in issue_lower:
        return "warning", "reference-accuracy"
    # Unverified URL identifier — warning but no mismatch penalty
    if "unverified url identifier" in issue_lower:
        return "info", "reference-accuracy"
    # Informational (not problems)
    if "open access pdf available" in issue_lower:
        return "info", "reference-exists"
    if "doi available but missing" in issue_lower:
        return "info", "reference-exists"
    return "warning", "reference-exists"


# ---------------------------------------------------------------------------
# PDF support: ManuscriptContext setup and Citation extraction
# ---------------------------------------------------------------------------


async def build_pdf_context(
    pdf_path: Path,
    config: LintConfig,
) -> None:
    """Parse a PDF via GROBID and set up ManuscriptContext for all checks.

    Builds ManuscriptContext from GROBID output, caches it, and attaches
    it to config so checks can detect PDF mode.
    """
    from sciwrite_lint.pdf.grobid import process_pdf
    from sciwrite_lint.manuscript_store import (
        ManuscriptContext,
        set_manuscript_context,
    )

    grobid_result = await process_pdf(pdf_path)
    ctx = ManuscriptContext.from_grobid(pdf_path, grobid_result)
    set_manuscript_context(pdf_path, ctx)
    config.manuscript_context = ctx


def citations_from_pdf_context(config: LintConfig) -> list[Citation]:
    """Extract Citation objects from a PDF ManuscriptContext.

    Converts ParsedReference entries to Citation objects suitable for the
    API verification pipeline.
    """
    from sciwrite_lint.manuscript_store import ManuscriptContext

    ctx: ManuscriptContext = config.manuscript_context
    citations: list[Citation] = []
    for ref in ctx.parsed_references:
        citations.append(
            Citation(
                key=ref.key,
                raw_text=ref.raw,
                authors=list(ref.authors),
                title=ref.title,
                year=ref.year,
                venue=ref.venue,
                doi=ref.doi,
                url=ref.url,
                isbn=ref.isbn,
                lccn=ref.lccn,
                source_paper=ctx.source_path.stem,
                bib_format="grobid",
                entry_type="",
            )
        )
    return citations


async def extract_citations_for_paper(
    tex_path: Path,
    config: LintConfig,
    bib_path: Path | None = None,
) -> list[Citation]:
    """Extract citations from a paper, handling both .tex and .pdf inputs.

    For .tex files, uses extract_bibitems + check_local_sources.
    For .pdf files, runs GROBID via build_pdf_context + citations_from_pdf_context.
    """
    from sciwrite_lint.references.citations import check_local_sources, extract_bibitems

    if tex_path.suffix.lower() == ".pdf":
        await build_pdf_context(tex_path, config)
        return citations_from_pdf_context(config)

    citations = extract_bibitems(tex_path, "auto", bib_path=bib_path)
    check_local_sources(citations, config.effective_references_dir())
    return citations


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def _backup_workspace(ws: PaperWorkspace) -> Path | None:
    """Back up a paper workspace as a zip archive before --fresh wipe.

    Creates references/{paper}_backup_YYYYMMDD_HHMMSS.zip from the workspace,
    then removes the original directory. Zip-first ensures the backup exists
    even if removal is interrupted.

    Returns the zip path, or None if nothing to back up.
    """
    if not ws.root.exists():
        return None
    import shutil
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = ws.root.parent / f"{ws.root.name}_backup_{stamp}"
    zip_path = Path(
        shutil.make_archive(str(archive_base), "zip", ws.root.parent, ws.root.name)
    )
    shutil.rmtree(ws.root)
    logger.info(f"Backed up {ws.root.name} → {zip_path.name}")
    return zip_path


async def run_full_check(
    paper_name: str,
    tex_path: Path,
    pc: PaperConfig,
    config: LintConfig,
    fresh: bool = False,
) -> list[Finding]:
    """Run the complete check pipeline for one paper.

    Stages 1 (text+LLM) and 2 (verify) run concurrently via asyncio.
    Stages 3-5 run sequentially after.

    **Not concurrency-safe.** GPU subprocess stages (vision, embedding,
    cited vision) assume exclusive VRAM access. Do not call this function
    concurrently for multiple papers — use ``run_papers_staged()`` instead,
    which coordinates GPU stages across papers.

    Args:
        fresh: Start from scratch — back up existing workspace, then
               re-verify, re-fetch, re-parse, and re-run claims.

    Supports both LaTeX (.tex) and PDF input. For PDF, call
    build_pdf_context() before this function.
    """
    from sciwrite_lint.usage import end_run, save_run_async, start_run

    # Per-paper workspace: references/{paper_name}/
    ws = config.paper_workspace(paper_name)

    # Check source compatibility (tex→pdf switch requires --fresh)
    if not fresh:
        ok, reason = ws.check_source(tex_path, bib_path=pc.bib)
        if not ok:
            raise RuntimeError(
                f"Source type changed ({reason}). Run with --fresh to start over."
            )

    if fresh:
        _backup_workspace(ws)
    ws.ensure_dirs()
    ws.save_source(tex_path, bib_path=pc.bib)
    refs_dir = ws.root
    # Set paper context for registered checks that need per-paper paths
    config.current_paper = paper_name

    # Extract citations — source depends on input type
    if config.is_pdf:
        citations = citations_from_pdf_context(config)
    else:
        from sciwrite_lint.references.citations import (
            check_local_sources,
            extract_bibitems,
        )

        citations = extract_bibitems(tex_path, "auto", bib_path=pc.bib)
        check_local_sources(citations, refs_dir)

    logger.info(f"{len(citations)} citations extracted")

    # Start usage tracking
    run = start_run(
        paper=paper_name,
        model=config.llm_model or "",
        workspace_root=str(refs_dir),
    )
    run.citations = len(citations)

    # Initialize pipeline stage tracking
    with get_db(refs_dir) as conn:
        init_pipeline_stages(conn)

    t0 = time.monotonic()
    all_findings: list[Finding] = []

    try:
        # Stage 0.5: Vision — describe figures for full-paper consistency checks.
        # Runs before LLM checks so descriptions are in the cache when
        # _build_system_prompt loads them.  On WSL2, the VL model (~4 GB)
        # shares VRAM with vLLM via memory overcommit — no container restart.
        with _stage_tracking(refs_dir, "vision"):
            _stage_vision(tex_path, config, paper_name, fresh=fresh)

        # Stage 1 + 2: concurrent (text checks in thread, LLM + verify async)
        with _stage_tracking(refs_dir, ["text_checks", "llm_checks", "verify"]):
            loop = asyncio.get_event_loop()
            text_task = loop.run_in_executor(None, run_text_checks, tex_path, config)
            llm_task = run_llm_checks_batched(tex_path, config)
            verify_task = _stage_verify(citations, config, refs_dir, fresh=fresh)
            text_findings, llm_findings, verify_findings = await asyncio.gather(
                text_task,
                llm_task,
                verify_task,
            )
            check_findings = text_findings + llm_findings

        t_checks = time.monotonic() - t0
        run.stage_rules_s = t_checks
        run.stage_verify_s = t_checks  # concurrent, same wall time
        logger.info(f"Checks + verify: {t_checks:.1f}s")

        # Note: reference-accuracy findings are produced by _stage_verify
        # via _classify_verify_issue routing mismatch issues. The registered
        # reference-accuracy check exists for standalone use (sciwrite-lint check).

        # Stage 3: fetch + Stage 4: parse (monitor GROBID memory)
        from sciwrite_lint.pdf.grobid import (
            MAX_PARSE_CONCURRENCY,
            monitor_container_memory,
        )

        parse_sem = asyncio.Semaphore(MAX_PARSE_CONCURRENCY)
        mem_monitor = asyncio.create_task(monitor_container_memory(parse_sem))
        try:
            t1 = time.monotonic()
            with _stage_tracking(refs_dir, "fetch") as st:
                fetched = await _stage_fetch(citations, config, refs_dir)
                st.detail = f"{fetched} downloaded"
            run.stage_fetch_s = time.monotonic() - t1

            t2 = time.monotonic()
            with _stage_tracking(refs_dir, "parse") as st:
                parsed, cached = await _stage_parse(
                    config, refs_dir, parse_sem=parse_sem
                )
                st.detail = f"{parsed} new, {cached} cached"
            run.stage_parse_s = time.monotonic() - t2
        finally:
            mem_monitor.cancel()

        if fetched or parsed:
            logger.info(f"Fetch + parse: {run.stage_fetch_s + run.stage_parse_s:.1f}s")

        # Stage 4.2: Vision on cited papers (VL model on GPU, sequential).
        # Never pass fresh — cited paper descriptions use hash-based
        # invalidation. Passing fresh=True would clear the entire
        # vision_cache table, destroying source paper descriptions
        # written by _stage_vision (Stage 0.5).
        with _stage_tracking(refs_dir, "cited_vision"):
            ref_figure_descs = _stage_cited_vision(refs_dir, fresh=False)

        # Stage 4.5: ref internal checks (vLLM, thinking=low/medium)
        t3 = time.monotonic()
        with _stage_tracking(refs_dir, "ref_internal"):
            ref_internal_results = await _stage_ref_internal(
                refs_dir, config, fresh=fresh, ref_figure_descs=ref_figure_descs
            )

        # Stage 4.6 + 5: bib verify (network) + claims (vLLM) — concurrent
        with _stage_tracking(refs_dir, ["bib_verify", "claims"]):
            bib_checks, (claim_findings, claim_results) = await asyncio.gather(
                _stage_bib_verify(config, refs_dir, fresh=fresh),
                _stage_claims(
                    paper_name,
                    tex_path,
                    config,
                    refs_dir,
                    bib_path=pc.bib,
                    rerun=fresh,
                ),
            )
        run.stage_claims_s = time.monotonic() - t3
        if claim_findings or ref_internal_results:
            logger.info(f"Claims + ref checks: {run.stage_claims_s:.1f}s")

        # Stage 6: reference-unreliable (aggregates signals from verify + claims)
        with _stage_tracking(refs_dir, "unreliable"):
            unreliable_findings = _stage_unreliable(
                tex_path,
                refs_dir,
                claim_results,
                bib_checks=bib_checks,
            )

        # Merge all findings
        all_findings = (
            list(check_findings)
            + verify_findings
            + claim_findings
            + unreliable_findings
        )

    finally:
        # Always save usage — even on crash, partial data is useful
        run.total_elapsed_s = time.monotonic() - t0
        logger.info(f"Total: {run.total_elapsed_s:.1f}s")
        logger.info(f"{_format_usage_summary(run)}")
        stats = end_run()
        if stats:
            await save_run_async(stats)
            config.last_run_stats = stats

    return all_findings


# ---------------------------------------------------------------------------
# Batch-staged orchestrator for multi-paper runs (evals)
# ---------------------------------------------------------------------------


@dataclass
class StagedPaperResult:
    """Result from ``run_papers_staged`` for one paper.

    Contains all pipeline outputs needed for scoring: findings from all
    stages (text, LLM, verify, claims, unreliable), raw claim verification
    results, and bibliography checks. Used by the real-world eval and
    calibration eval to compute SciLint Scores.
    """

    paper_name: str
    findings: list[Finding]
    claim_results: list[dict]
    bib_checks: list[Any]
    error: str | None = None


@dataclass
class _PaperCtx:
    """Per-paper mutable state carried across stages in ``run_papers_staged``.

    Each paper gets one context object at setup. Stages mutate it
    (appending findings, storing claim results, etc.) as they complete.
    """

    name: str
    tex_path: Path
    pc: PaperConfig
    config: LintConfig
    refs_dir: Path
    citations: list[Citation]
    check_findings: list[Finding]
    verify_findings: list[Finding]
    claim_findings: list[Finding]
    claim_results: list[dict]
    bib_checks: list[Any]
    ref_figure_descs: dict[str, str]
    run: Any  # RunStats from usage.py
    error: str | None = None


async def run_papers_staged(
    papers: list[tuple[str, Path, PaperConfig, LintConfig]],
    fresh: bool = False,
    concurrency: int = 0,
) -> list[StagedPaperResult]:
    """Run multiple papers through the pipeline with batch-by-stage orchestration.

    GPU stages (vision, embedding, cited vision) run in a single subprocess
    per batch — the model loads once and processes all papers sequentially.
    Non-GPU stages (vLLM checks, verify, fetch, GROBID, ref_internal, claims)
    run up to ``concurrency`` papers concurrently (0 = unlimited).

    Args:
        papers: List of (paper_name, tex_path, PaperConfig, LintConfig) tuples.
        fresh: Start from scratch for all papers.
        concurrency: Max papers in non-GPU concurrent stages. 0 = no limit.

    Returns:
        List of StagedPaperResult, one per paper.
    """
    from sciwrite_lint.usage import end_run, save_run_async, set_current, start_run

    if not papers:
        return []

    if concurrency < 0:
        raise ValueError(f"concurrency must be >= 0, got {concurrency}")

    if concurrency > 0 and concurrency >= len(papers):
        logger.info(
            "concurrency={} >= {} papers — all papers will run concurrently",
            concurrency,
            len(papers),
        )

    # Soft warning for high concurrency. On the single-GPU + single-vLLM-
    # server setup we ship with, validated baseline is up to 4 concurrent
    # papers. Above 4, the live monitor (which polls each workspace.db
    # plus the shared usage.db) saturates and can fail to surface stage
    # progress, vLLM queue depth grows, API rate limiters bite, and GROBID
    # memory pressure can stall parses. Larger numbers may still work on
    # bigger hardware but are not part of the validated baseline.
    _CONCURRENCY_TESTED_MAX = 4
    if concurrency > _CONCURRENCY_TESTED_MAX:
        logger.warning(
            "concurrency={} exceeds tested maximum ({}). Above this, the "
            "live monitor may fail to surface progress, vLLM queue depth "
            "grows, and API rate limits / GROBID memory may degrade "
            "throughput. The setting is allowed but not part of the "
            "validated baseline.",
            concurrency,
            _CONCURRENCY_TESTED_MAX,
        )

    # ------------------------------------------------------------------
    # Setup: workspace, citations, usage tracking for each paper
    # ------------------------------------------------------------------
    ctxs: list[_PaperCtx] = []
    for paper_name, tex_path, pc, config in papers:
        ws = config.paper_workspace(paper_name)
        if not fresh:
            ok, reason = ws.check_source(tex_path, bib_path=pc.bib)
            if not ok:
                raise RuntimeError(
                    f"[{paper_name}] Source type changed ({reason}). "
                    "Run with --fresh to start over."
                )
        if fresh:
            _backup_workspace(ws)
        ws.ensure_dirs()
        ws.save_source(tex_path, bib_path=pc.bib)
        refs_dir = ws.root
        config.current_paper = paper_name

        # Extract citations
        if config.is_pdf:
            citations = citations_from_pdf_context(config)
        else:
            from sciwrite_lint.references.citations import (
                check_local_sources,
                extract_bibitems,
            )

            citations = extract_bibitems(tex_path, "auto", bib_path=pc.bib)
            check_local_sources(citations, refs_dir)

        logger.info("[{}] {} citations extracted", paper_name, len(citations))

        run = start_run(
            paper=paper_name,
            model=config.llm_model or "",
            workspace_root=str(refs_dir),
        )
        run.citations = len(citations)

        with get_db(refs_dir) as conn:
            init_pipeline_stages(conn)

        ctxs.append(
            _PaperCtx(
                name=paper_name,
                tex_path=tex_path,
                pc=pc,
                config=config,
                refs_dir=refs_dir,
                citations=citations,
                check_findings=[],
                verify_findings=[],
                claim_findings=[],
                claim_results=[],
                bib_checks=[],
                ref_figure_descs={},
                run=run,
            )
        )

    t0 = time.monotonic()

    # Semaphore for non-GPU concurrent stages (0 = unlimited)
    paper_sem: asyncio.Semaphore | None = None
    if concurrency > 0:
        paper_sem = asyncio.Semaphore(concurrency)

    def _active() -> list[_PaperCtx]:
        """Papers that haven't failed yet."""
        return [ctx for ctx in ctxs if ctx.error is None]

    try:
        # ------------------------------------------------------------------
        # Stage 0.5: BATCH VISION — one subprocess, all papers
        # ------------------------------------------------------------------
        logger.info("=== Stage 0.5: Batch vision ({} papers) ===", len(ctxs))
        vision_manifest = [
            {
                "paper_name": ctx.name,
                "tex_path": str(ctx.tex_path),
                "config_path": str(ctx.config.config_path)
                if ctx.config.config_path
                else None,
                "fresh": fresh,
            }
            for ctx in ctxs
        ]
        with _stage_tracking([ctx.refs_dir for ctx in ctxs], "vision"):
            _batch_vision(vision_manifest)

        # ------------------------------------------------------------------
        # Stages 1+2: TEXT + LLM + VERIFY — all papers concurrent
        # ------------------------------------------------------------------
        logger.info("=== Stages 1+2: Checks + verify ({} papers) ===", len(ctxs))

        async def _checks_verify(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                with _stage_tracking(
                    ctx.refs_dir, ["text_checks", "llm_checks", "verify"]
                ):
                    loop = asyncio.get_event_loop()
                    text_task = loop.run_in_executor(
                        None, run_text_checks, ctx.tex_path, ctx.config
                    )
                    llm_task = run_llm_checks_batched(ctx.tex_path, ctx.config)
                    verify_task = _stage_verify(
                        ctx.citations, ctx.config, ctx.refs_dir, fresh=fresh
                    )
                    text_findings, llm_findings, verify_findings = await asyncio.gather(
                        text_task,
                        llm_task,
                        verify_task,
                    )
                    ctx.check_findings = list(text_findings) + list(llm_findings)
                    ctx.verify_findings = list(verify_findings)
            except Exception as e:
                logger.error("[{}] Checks+verify failed: {}", ctx.name, e)
                ctx.error = str(e)
            finally:
                if paper_sem:
                    paper_sem.release()

        await asyncio.gather(*[_checks_verify(ctx) for ctx in _active()])
        t_checks = time.monotonic() - t0
        logger.info("Checks + verify: {:.1f}s", t_checks)

        # ------------------------------------------------------------------
        # Stage 3: FETCH — all papers concurrent
        # ------------------------------------------------------------------
        logger.info("=== Stage 3: Fetch ({} papers) ===", len(ctxs))

        async def _fetch_paper(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                with _stage_tracking(ctx.refs_dir, "fetch") as st:
                    fetched = await _stage_fetch(
                        ctx.citations, ctx.config, ctx.refs_dir, embed_inline=False
                    )
                    st.detail = f"{fetched} downloaded"
            except Exception as e:
                logger.error("[{}] Fetch failed: {}", ctx.name, e)
                ctx.error = str(e)
            finally:
                if paper_sem:
                    paper_sem.release()

        await asyncio.gather(*[_fetch_paper(ctx) for ctx in _active()])

        # ------------------------------------------------------------------
        # Stage 4a: GROBID PARSE — concurrent with shared memory monitor
        # ------------------------------------------------------------------
        logger.info("=== Stage 4a: GROBID parse ({} papers) ===", len(ctxs))
        from sciwrite_lint.pdf.grobid import (
            MAX_PARSE_CONCURRENCY,
            monitor_container_memory,
        )

        parse_sem = asyncio.Semaphore(MAX_PARSE_CONCURRENCY)
        mem_monitor = asyncio.create_task(monitor_container_memory(parse_sem))

        # Store parse results for each paper (needed to collect embedding keys)
        parse_results_map: dict[str, dict[str, str]] = {}

        async def _parse_paper(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                _track(ctx.refs_dir, "parse", "running")
                from sciwrite_lint.references.reference_store import parse_all_missing

                pr = await parse_all_missing(ctx.refs_dir, sem=parse_sem)
                parse_results_map[ctx.name] = pr
                # Keep stage "running" — batch embedding runs after all papers
                # finish parsing, and its timing must be attributed to this stage.
                # Stage is marked "done" after `_batch_embed` completes below.
            except Exception as e:
                logger.error("[{}] Parse failed: {}", ctx.name, e)
                ctx.error = str(e)
                _track(ctx.refs_dir, "parse", "failed", str(e))
            finally:
                if paper_sem:
                    paper_sem.release()

        try:
            await asyncio.gather(*[_parse_paper(ctx) for ctx in _active()])
        finally:
            mem_monitor.cancel()

        # ------------------------------------------------------------------
        # Stage 4b: BATCH EMBEDDING — one subprocess, all papers
        # ------------------------------------------------------------------
        logger.info("=== Stage 4b: Batch embedding ({} papers) ===", len(ctxs))
        from sciwrite_lint.references.embedding_store import has_embeddings
        from sciwrite_lint.references.metadata import load_all_metadata
        from sciwrite_lint.references.reference_store import _get_embedding_config

        model_name, _, _ = _get_embedding_config()

        embed_manifest: list[dict[str, Any]] = []
        for ctx in _active():
            keys_to_embed: list[str] = []

            # Scan ALL parsed markdown files — catches refs parsed during
            # fetch (eager parse_and_embed) as well as _stage_parse.
            parsed_dir = ctx.refs_dir / "parsed"
            if parsed_dir.exists():
                for md_file in sorted(parsed_dir.glob("*.md")):
                    key = md_file.stem
                    if not has_embeddings(key, ctx.refs_dir, model_name=model_name):
                        keys_to_embed.append(key)

            # Web summary keys
            all_meta = load_all_metadata(ctx.refs_dir)
            for key, meta in all_meta.items():
                local_file = meta.access.get("local_file") or ""
                if not local_file.endswith("_web.md"):
                    continue
                if has_embeddings(key, ctx.refs_dir, model_name=model_name):
                    continue
                web_path = ctx.refs_dir / local_file
                if web_path.exists():
                    keys_to_embed.append(key)

            if keys_to_embed:
                embed_manifest.append(
                    {
                        "keys": keys_to_embed,
                        "references_dir": str(ctx.refs_dir),
                    }
                )

        with _stage_failure_guard([ctx.refs_dir for ctx in _active()], "parse"):
            embed_diag = _batch_embed(embed_manifest)

        # Mark parse stage done per paper, now that batch embedding has run.
        # _batch_embed swallows subprocess failures (crash, timeout, partial)
        # and returns a diagnostic string — we verify actual success by
        # re-checking has_embeddings() against the DB (authoritative source of
        # truth) and propagate the subprocess stderr into ctx.error so the
        # user sees the real failure reason (CUDA OOM, etc.) inline rather
        # than a cryptic "no embeddings" crash during claim verification.
        keys_requested: dict[str, list[str]] = {
            entry["references_dir"]: entry["keys"] for entry in embed_manifest
        }
        for ctx in _active():
            pr = parse_results_map.get(ctx.name, {})
            n_parsed = sum(1 for v in pr.values() if v == "parsed")
            n_cached = sum(1 for v in pr.values() if v == "cached")
            requested = keys_requested.get(str(ctx.refs_dir), [])
            n_embedded = sum(
                1
                for key in requested
                if has_embeddings(key, ctx.refs_dir, model_name=model_name)
            )
            if n_embedded < len(requested):
                missing = len(requested) - n_embedded
                _track(
                    ctx.refs_dir,
                    "parse",
                    "failed",
                    f"{n_parsed} new, {n_cached} cached, "
                    f"only {n_embedded}/{len(requested)} embedded",
                )
                if embed_diag:
                    ctx.error = (
                        f"batch embed: {missing}/{len(requested)} keys "
                        f"missing | {embed_diag}"
                    )
                else:
                    ctx.error = (
                        f"batch embed: {missing}/{len(requested)} keys "
                        "missing (subprocess exited cleanly — likely silent "
                        "per-key skip, check subprocess logs)"
                    )
                logger.error(
                    "[{}] Batch embed incomplete: {} of {} keys missing | {}",
                    ctx.name,
                    missing,
                    len(requested),
                    embed_diag or "subprocess exited cleanly",
                )
            else:
                _track(
                    ctx.refs_dir,
                    "parse",
                    "done",
                    f"{n_parsed} new, {n_cached} cached, {n_embedded} embedded",
                )

        # ------------------------------------------------------------------
        # Stage 4.2: BATCH CITED VISION — one subprocess, all papers
        # ------------------------------------------------------------------
        logger.info("=== Stage 4.2: Batch cited vision ({} papers) ===", len(ctxs))
        active_ctxs = _active()
        for ctx in active_ctxs:
            _track(ctx.refs_dir, "cited_vision", "running")

        cited_vis_manifest = [
            {"references_dir": str(ctx.refs_dir), "fresh": False} for ctx in active_ctxs
        ]
        # Count total images across all papers for dynamic timeout.
        # Image extraction is lightweight (no GPU) — just PDF page scanning.
        from sciwrite_lint.vision.image_extraction import extract_images_from_pdf

        total_images = 0
        for ctx in active_ctxs:
            parsed_dir = ctx.refs_dir / "parsed"
            if not parsed_dir.exists():
                continue
            for md_file in sorted(parsed_dir.glob("*.md")):
                key = md_file.stem
                candidates = sorted(ctx.refs_dir.glob(f"{key}*.pdf"))
                if candidates:
                    total_images += len(
                        extract_images_from_pdf(
                            candidates[0],
                            ctx.refs_dir / "parsed" / "ref_figures" / key,
                        )
                    )

        # ~5s per image (conservative, GPU batch=16), 60s for model load
        cited_vis_timeout = max(120, 60 + total_images * 5)
        logger.info(
            "Cited vision: {} images across {} papers, timeout={}s",
            total_images,
            len(active_ctxs),
            cited_vis_timeout,
        )
        with _stage_failure_guard(
            [ctx.refs_dir for ctx in active_ctxs], "cited_vision"
        ):
            _batch_cited_vision(cited_vis_manifest, timeout=cited_vis_timeout)

        # Read cited vision results from DB for each paper
        from sciwrite_lint.vision.cache import format_descriptions_from_db
        from sciwrite_lint.vision.image_extraction import extract_images_from_pdf

        for ctx in active_ctxs:
            parsed_dir = ctx.refs_dir / "parsed"
            if not parsed_dir.exists():
                _track(ctx.refs_dir, "cited_vision", "done")
                continue
            keys = [f.stem for f in sorted(parsed_dir.glob("*.md"))]
            descriptions: dict[str, str] = {}
            for key in keys:
                candidates = sorted(ctx.refs_dir.glob(f"{key}*.pdf"))
                if not candidates:
                    continue
                output_dir = ctx.refs_dir / "parsed" / "ref_figures" / key
                images = extract_images_from_pdf(candidates[0], output_dir)
                if images:
                    desc = format_descriptions_from_db(images, ctx.refs_dir)
                    if desc:
                        descriptions[key] = desc
            ctx.ref_figure_descs = descriptions
            _track(ctx.refs_dir, "cited_vision", "done")

        # ------------------------------------------------------------------
        # Stage 4.5: REF INTERNAL — all papers concurrent (vLLM)
        # ------------------------------------------------------------------
        logger.info("=== Stage 4.5: Ref internal ({} papers) ===", len(ctxs))

        async def _ref_internal_paper(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                with _stage_tracking(ctx.refs_dir, "ref_internal"):
                    await _stage_ref_internal(
                        ctx.refs_dir,
                        ctx.config,
                        fresh=fresh,
                        ref_figure_descs=ctx.ref_figure_descs,
                    )
            except Exception as e:
                logger.error("[{}] Ref internal failed: {}", ctx.name, e)
                ctx.error = str(e)
            finally:
                if paper_sem:
                    paper_sem.release()

        await asyncio.gather(*[_ref_internal_paper(ctx) for ctx in _active()])

        # ------------------------------------------------------------------
        # Stage 5: CLAIMS + BIB VERIFY — all papers concurrent (vLLM + network)
        # ------------------------------------------------------------------
        logger.info("=== Stage 5: Claims + bib verify ({} papers) ===", len(ctxs))

        async def _claims_bib_paper(ctx: _PaperCtx) -> None:
            set_current(ctx.run)
            if paper_sem:
                await paper_sem.acquire()
            try:
                with _stage_tracking(ctx.refs_dir, ["bib_verify", "claims"]):
                    bib, (c_findings, c_results) = await asyncio.gather(
                        _stage_bib_verify(ctx.config, ctx.refs_dir, fresh=fresh),
                        _stage_claims(
                            ctx.name,
                            ctx.tex_path,
                            ctx.config,
                            ctx.refs_dir,
                            bib_path=ctx.pc.bib,
                            rerun=fresh,
                        ),
                    )
                    ctx.bib_checks = bib
                    ctx.claim_findings = list(c_findings)
                    ctx.claim_results = list(c_results)
            except Exception as e:
                logger.error("[{}] Claims+bib failed: {}", ctx.name, e)
                ctx.error = str(e)
            finally:
                if paper_sem:
                    paper_sem.release()

        await asyncio.gather(*[_claims_bib_paper(ctx) for ctx in _active()])

        # ------------------------------------------------------------------
        # Stage 6: UNRELIABLE — all papers
        # ------------------------------------------------------------------
        active_for_unreliable = _active()
        logger.info(
            "=== Stage 6: Unreliable ({} papers) ===", len(active_for_unreliable)
        )
        for ctx in active_for_unreliable:
            try:
                with _stage_tracking(ctx.refs_dir, "unreliable"):
                    unreliable = _stage_unreliable(
                        ctx.tex_path,
                        ctx.refs_dir,
                        ctx.claim_results,
                        bib_checks=ctx.bib_checks,
                    )
            except Exception as e:
                logger.error("[{}] Unreliable failed: {}", ctx.name, e)
                ctx.error = str(e)
                continue

            # Merge all findings for this paper
            ctx.check_findings.extend(ctx.verify_findings)
            ctx.check_findings.extend(ctx.claim_findings)
            ctx.check_findings.extend(unreliable)

    finally:
        # Save usage tracking for all papers.
        # Note: each ctx.run was created by start_run() which wrote a
        # preliminary row to usage.db. We save stats directly from ctx.run
        # (not via end_run()) because the global _current only holds the
        # last paper's RunStats — the batch creates multiple runs.
        total_elapsed = time.monotonic() - t0
        for ctx in ctxs:
            ctx.run.total_elapsed_s = total_elapsed
            await save_run_async(ctx.run)
            ctx.config.last_run_stats = ctx.run
        # Clear the global so end_run() doesn't return stale data
        end_run()

    # Build results
    results: list[StagedPaperResult] = []
    for ctx in ctxs:
        results.append(
            StagedPaperResult(
                paper_name=ctx.name,
                findings=ctx.check_findings,
                claim_results=ctx.claim_results,
                bib_checks=ctx.bib_checks,
                error=ctx.error,
            )
        )

    failed = [ctx for ctx in ctxs if ctx.error]
    if failed:
        logger.warning(
            "Batch pipeline: {}/{} papers failed: {}",
            len(failed),
            len(ctxs),
            ", ".join(f"{ctx.name} ({ctx.error})" for ctx in failed),
        )
    logger.info(
        "Batch pipeline complete: {}/{} papers succeeded in {:.1f}s",
        len(ctxs) - len(failed),
        len(ctxs),
        time.monotonic() - t0,
    )
    return results
