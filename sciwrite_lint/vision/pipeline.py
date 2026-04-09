"""Vision pipeline: extract images → describe via VL model → cache results.

Orchestrates the full vision flow for a single paper. Runs automatically
as part of ``sciwrite-lint check`` (Stage 0.5, before LLM checks).

On WSL2, CUDA memory overcommit lets the VL model (~4 GB float16) share
VRAM with vLLM transparently — no container restart needed.  On native
Linux without overcommit, auto-resolves to CPU.

Can also run standalone: ``sciwrite-lint vision --paper paper_a``.
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from sciwrite_lint.config import LintConfig


def run_vision_pipeline(
    tex_path: Path,
    config: LintConfig,
    paper_name: str = "",
    device: str = "auto",
    fresh: bool = False,
) -> str:
    """Run the full vision pipeline for a manuscript.

    1. Extract images from the manuscript (LaTeX or PDF)
    2. Check workspace.db cache — skip images whose content hash hasn't changed
    3. Run VL inference on new/changed images
    4. Cache results in workspace.db (vision_cache table)
    5. Return formatted figure descriptions

    Args:
        tex_path: Path to the .tex or .pdf manuscript file.
        config: Lint configuration.
        paper_name: Paper name for workspace resolution. Uses
            ``config.current_paper`` if empty.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        fresh: Ignore cache and re-describe all images.

    Returns:
        Formatted figure descriptions string (empty if no images found).
    """
    from sciwrite_lint.vision.describe import describe_figures
    from sciwrite_lint.vision.image_extraction import (
        ExtractedImage,
        extract_images_from_latex,
        extract_images_from_pdf,
    )

    t0 = time.monotonic()
    paper = paper_name or config.current_paper

    # Determine workspace for DB caching
    references_dir: Path | None = None
    ws = None
    if paper:
        ws = config.paper_workspace(paper)
        ws.ensure_dirs()
        references_dir = ws.root

    # Extract images based on source type.
    # For LaTeX, prefer the workspace source copy (images are snapshotted
    # there by save_source) so the pipeline is independent of the live
    # source tree.
    images: list[ExtractedImage]
    if tex_path.suffix.lower() == ".pdf":
        if references_dir is None:
            raise RuntimeError(
                "PDF input requires a paper workspace for image extraction. "
                "Use --paper to specify the paper name."
            )
        output_dir = ws.parsed / "extracted_images"  # type: ignore[union-attr]
        images = extract_images_from_pdf(tex_path, output_dir)
    else:
        ws_tex = ws.source / tex_path.name if ws else None
        effective_tex = ws_tex if ws_tex and ws_tex.is_file() else tex_path
        images = extract_images_from_latex(effective_tex)

        # Also render TikZ/pgfplots figures from the compiled PDF.
        # Look for PDF alongside the original .tex (not the workspace copy).
        if ws:
            from sciwrite_lint.vision.image_extraction import (
                render_tikz_figures,
                _find_compiled_pdf,
            )

            pdf_path = _find_compiled_pdf(tex_path)
            if pdf_path:
                rendered_dir = ws.parsed / "rendered_figures"
                rendered = render_tikz_figures(
                    effective_tex, rendered_dir, pdf_path=pdf_path
                )
                images.extend(rendered)

    if not images:
        logger.info("No figures found in manuscript")
        return ""

    # Describe figures (with DB caching via get_db)
    result = describe_figures(
        images,
        references_dir=references_dir,
        device=device,
        fresh=fresh,
    )

    elapsed = time.monotonic() - t0
    logger.info("Vision pipeline: {} figures in {:.1f}s", len(images), elapsed)
    return result


# ---------------------------------------------------------------------------
# CLI entry point for subprocess isolation
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    """Entry point for ``python -m sciwrite_lint.vision.pipeline``.

    Called by _stage_vision() in a subprocess so the CUDA context
    is fully released when this process exits.

    Modes:
        Single paper: ``python -m sciwrite_lint.vision.pipeline <tex> --paper <name>``
        Batch: ``python -m sciwrite_lint.vision.pipeline --batch <manifest.json>``

    Batch mode loads the VL model once and processes all papers in the
    manifest sequentially. Used by ``run_papers_staged()`` to avoid
    loading the model N times.
    """
    import argparse
    import sys

    from sciwrite_lint.config import load_config

    parser = argparse.ArgumentParser(description="Vision pipeline (subprocess)")
    parser.add_argument("tex_path", type=Path, nargs="?", default=None)
    parser.add_argument("--paper", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--batch", type=Path, default=None, help="JSON manifest for batch mode"
    )
    args = parser.parse_args()

    if args.batch:
        _run_batch(args.batch)
    elif args.tex_path and args.paper:
        config = load_config(args.config)
        try:
            result = run_vision_pipeline(
                args.tex_path, config, paper_name=args.paper, fresh=args.fresh
            )
            if not result:
                logger.info("No figures found")
        except Exception as e:
            logger.error("Vision pipeline failed: {}", e)
            sys.exit(1)
    else:
        parser.error("Provide tex_path + --paper, or --batch <manifest.json>")


def _run_batch(manifest_path: Path) -> None:
    """Batch mode: load VL model once, process all papers in manifest.

    Manifest JSON: list of objects with keys:
        paper_name, tex_path, config_path (optional), fresh (optional)

    Results are written to each paper's workspace.db (vision_cache table).
    """
    import json
    import sys

    from sciwrite_lint.config import load_config

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        return

    failed = 0
    for i, entry in enumerate(manifest):
        paper_name = entry["paper_name"]
        tex_path = Path(entry["tex_path"])
        config_path = entry.get("config_path")
        fresh = entry.get("fresh", False)

        config = load_config(Path(config_path) if config_path else None)
        logger.info("[{}/{}] Vision: {}", i + 1, len(manifest), paper_name)
        try:
            run_vision_pipeline(tex_path, config, paper_name=paper_name, fresh=fresh)
        except Exception as e:
            logger.error("[{}] Vision failed: {}", paper_name, e)
            failed += 1

    if failed:
        logger.warning("Vision batch: {}/{} papers failed", failed, len(manifest))
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
