"""Extract figure images from LaTeX source or PDF files.

LaTeX: parses \\includegraphics paths from .tex source, resolves relative to
the source directory.  Gives clean, individual image files.

PDF: extracts embedded raster images via pypdfium2 (PDFium bindings).  PDFium
ships with all image codecs (JBIG2, JPEG2000, CCITT, etc.) statically linked,
so no system binaries are required.  Vector graphics (matplotlib/TikZ rendered
as PDF primitives) are not covered in V1.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class ExtractedImage(BaseModel):
    """An image extracted from a manuscript."""

    path: Path  # path to the image file (LaTeX) or temp file (PDF)
    label: str  # figure label if available (e.g., "fig:results")
    caption: str  # caption text if available
    source: str  # "latex" or "pdf"


# ---------------------------------------------------------------------------
# LaTeX: parse \includegraphics paths
# ---------------------------------------------------------------------------

# Match \includegraphics with optional args: \includegraphics[width=0.8\textwidth]{path}
_INCLUDEGRAPHICS_RE = re.compile(
    r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}",
)

# Match figure environment to extract label and caption alongside graphics
_FIGURE_ENV_RE = re.compile(
    r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}",
    re.DOTALL,
)

_CAPTION_RE = re.compile(r"\\caption\{([^}]+)\}")
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")

# Match subfigure environments
_SUBFIGURE_ENV_RE = re.compile(
    r"\\begin\{subfigure\}(?:\[[^\]]*\])?\{[^}]*\}(.*?)\\end\{subfigure\}",
    re.DOTALL,
)

# Common image extensions to try if the path has no extension
_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

# \graphicspath{{dir1/}{dir2/}} — each {dir} is a search path
# The outer braces belong to \graphicspath, inner braces wrap each directory.
_GRAPHICSPATH_RE = re.compile(
    r"\\graphicspath\s*\{\s*((?:\{[^}]*\}\s*)+)\}",
)
_GRAPHICSPATH_DIR_RE = re.compile(r"\{([^}]*)\}")


def _parse_graphicspath(tex_text: str, source_dir: Path) -> list[Path]:
    """Extract search directories from \\graphicspath in LaTeX source.

    Returns resolved absolute paths for each directory that exists.
    The source_dir itself is NOT included (caller prepends it).
    """
    m = _GRAPHICSPATH_RE.search(tex_text)
    if not m:
        return []
    dirs: list[Path] = []
    for dir_match in _GRAPHICSPATH_DIR_RE.finditer(m.group(1)):
        raw = dir_match.group(1).strip()
        if not raw:
            continue
        resolved = (source_dir / raw).resolve()
        if resolved.is_dir():
            dirs.append(resolved)
        else:
            logger.debug("graphicspath dir not found: {}", resolved)
    return dirs


def _resolve_image_path(
    raw_path: str,
    search_dirs: list[Path],
) -> Path | None:
    """Resolve an \\includegraphics path to an actual file.

    Searches each directory in ``search_dirs`` (source_dir first, then any
    \\graphicspath dirs).  LaTeX allows omitting extensions — tries common
    ones if the exact path doesn't exist.
    """
    clean = raw_path.strip()
    for base in search_dirs:
        candidate = base / clean
        if candidate.is_file():
            return candidate
        # Try adding common extensions
        if not candidate.suffix:
            for ext in _IMAGE_EXTENSIONS:
                with_ext = candidate.with_suffix(ext)
                if with_ext.is_file():
                    return with_ext

    return None


def extract_images_from_latex(tex_path: Path) -> list[ExtractedImage]:
    """Extract figure images from a LaTeX .tex file.

    Parses figure environments for labels and captions, then resolves
    \\includegraphics paths relative to the .tex file's directory.
    """
    text = tex_path.read_text(encoding="utf-8")
    source_dir = tex_path.parent
    images: list[ExtractedImage] = []

    # Build search path: source dir first, then \graphicspath dirs
    search_dirs = [source_dir, *_parse_graphicspath(text, source_dir)]

    # First pass: extract from figure environments (gets labels + captions)
    seen_paths: set[str] = set()
    for env_match in _FIGURE_ENV_RE.finditer(text):
        env_body = env_match.group(1)

        # Outer figure-level caption and label (used when no subfigure)
        fig_caption_m = _CAPTION_RE.search(env_body)
        fig_label_m = _LABEL_RE.search(env_body)
        fig_caption = fig_caption_m.group(1) if fig_caption_m else ""
        fig_label = fig_label_m.group(1) if fig_label_m else ""

        # Build subfigure map: for each \includegraphics inside a subfigure,
        # use that subfigure's own caption/label instead of the outer figure's.
        subfig_meta: dict[str, tuple[str, str]] = {}  # raw_path → (label, caption)
        for sub_match in _SUBFIGURE_ENV_RE.finditer(env_body):
            sub_body = sub_match.group(1)
            sub_caption_m = _CAPTION_RE.search(sub_body)
            sub_label_m = _LABEL_RE.search(sub_body)
            sub_caption = sub_caption_m.group(1) if sub_caption_m else ""
            sub_label = sub_label_m.group(1) if sub_label_m else ""
            for gfx_m in _INCLUDEGRAPHICS_RE.finditer(sub_body):
                subfig_meta[gfx_m.group(1)] = (sub_label, sub_caption)

        for gfx_match in _INCLUDEGRAPHICS_RE.finditer(env_body):
            raw_path = gfx_match.group(1)
            resolved = _resolve_image_path(raw_path, search_dirs)
            if resolved is None:
                logger.debug("Image not found: {} (from {})", raw_path, tex_path.name)
                continue

            # Skip non-raster formats (EPS, SVG, PDF figures need rendering)
            if resolved.suffix.lower() in {".eps", ".svg", ".pdf"}:
                logger.debug("Skipping vector image: {}", resolved.name)
                continue

            # Use subfigure metadata if available, otherwise outer figure
            label, caption = subfig_meta.get(raw_path, (fig_label, fig_caption))

            seen_paths.add(raw_path)
            images.append(
                ExtractedImage(
                    path=resolved,
                    label=label,
                    caption=caption,
                    source="latex",
                )
            )

    # Second pass: standalone \includegraphics not inside figure environments
    for gfx_match in _INCLUDEGRAPHICS_RE.finditer(text):
        raw_path = gfx_match.group(1)
        if raw_path in seen_paths:
            continue
        resolved = _resolve_image_path(raw_path, search_dirs)
        if resolved is None:
            continue
        if resolved.suffix.lower() in {".eps", ".svg", ".pdf"}:
            continue

        seen_paths.add(raw_path)
        images.append(
            ExtractedImage(
                path=resolved,
                label="",
                caption="",
                source="latex",
            )
        )

    logger.info("Extracted {} images from LaTeX source", len(images))
    return images


def collect_image_paths(tex_path: Path) -> list[tuple[Path, Path]]:
    """Collect all image files referenced by a .tex file.

    Returns a list of (absolute_path, relative_path) tuples where
    relative_path is relative to the .tex file's parent directory.
    This is used by ``save_source`` to copy images into the workspace
    while preserving directory structure so \\includegraphics paths
    still resolve.

    Includes all formats (raster + vector) — the workspace should be
    a complete snapshot of the source, not just what the vision pipeline
    processes.
    """
    text = tex_path.read_text(encoding="utf-8")
    source_dir = tex_path.parent
    search_dirs = [source_dir, *_parse_graphicspath(text, source_dir)]

    result: list[tuple[Path, Path]] = []
    seen: set[Path] = set()

    for gfx_match in _INCLUDEGRAPHICS_RE.finditer(text):
        raw_path = gfx_match.group(1)
        resolved = _resolve_image_path(raw_path, search_dirs)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        try:
            rel = resolved.resolve().relative_to(source_dir.resolve())
        except ValueError:
            # Image outside source tree — use just the filename
            rel = Path(resolved.name)
        result.append((resolved, rel))

    return result


# ---------------------------------------------------------------------------
# PDF: extract embedded raster images via pypdfium2
# ---------------------------------------------------------------------------


def extract_images_from_pdf(pdf_path: Path, output_dir: Path) -> list[ExtractedImage]:
    """Extract embedded raster images from a PDF file.

    Uses pypdfium2 (PDFium bindings) to iterate over image objects on each
    page, render them as bitmaps, and write PNGs to ``output_dir``.  PDFium
    bundles all image codecs statically (JBIG2, JPEG2000, CCITT, etc.), so
    no system binaries are required.

    One broken PDF must not abort a multi-paper batch: PDFium is a native
    binding and can raise unexpected exception types on malformed data.  We
    catch at the function boundary, log a clear warning with the failing
    PDF's name, and return an empty list.  This is isolation of external
    input corruption, not silent degradation — the user sees which PDF
    failed and why.

    V1 limitation: only raster images.  Vector graphics rendered as PDF
    drawing commands are not extracted.
    """
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_raw

    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[ExtractedImage] = []

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                try:
                    img_idx = 0
                    for obj in page.get_objects(
                        filter=(pdfium_raw.FPDF_PAGEOBJ_IMAGE,)
                    ):
                        img_idx += 1
                        # Cheap dimension check before rendering.
                        # Scientific figures are typically 300+ px on each side.
                        try:
                            w, h = obj.get_px_size()
                        except Exception as e:
                            logger.debug(
                                "image dimension probe skipped ({}: {})",
                                type(e).__name__,
                                e,
                            )
                            continue
                        if w < 200 or h < 200:
                            continue

                        # Encoded-size filter: skip mostly-blank or trivial
                        # images that pass the pixel filter but are decorations.
                        # Scientific figures are typically ≥5 KB once PNG-encoded.
                        # Encode in memory first so we don't write small images
                        # to disk and immediately delete them.
                        bitmap = obj.get_bitmap(render=True)
                        pil_img = bitmap.to_pil()
                        from io import BytesIO

                        buf = BytesIO()
                        pil_img.save(buf, format="PNG")
                        png_bytes = buf.getvalue()
                        if len(png_bytes) < 5120:
                            continue

                        out_path = output_dir / f"p{page_idx + 1}_img{img_idx}.png"
                        out_path.write_bytes(png_bytes)

                        images.append(
                            ExtractedImage(
                                path=out_path,
                                label=f"page{page_idx + 1}_img{img_idx}",
                                caption="",
                                source="pdf",
                            )
                        )
                finally:
                    page.close()
        finally:
            pdf.close()
    except Exception as e:
        logger.warning(
            "Skipping {}: PDF image extraction failed ({}: {})",
            pdf_path.name,
            type(e).__name__,
            e,
        )
        return []

    logger.info("Extracted {} raster images from PDF", len(images))
    return images


# ---------------------------------------------------------------------------
# Rendered figures: TikZ/pgfplots via compiled PDF page rendering
# ---------------------------------------------------------------------------

# "Figure 1:" or "Figure 1." in PDF text — indicates a figure caption on that page
_FIGURE_CAPTION_PDF_RE = re.compile(r"Figure\s+(\d+)\s*[.:]")


def _find_compiled_pdf(tex_path: Path) -> Path | None:
    """Find a compiled PDF alongside the .tex file.

    Looks for a PDF with the same stem, then any PDF in the same directory.
    Returns None if no PDF found.
    """
    same_stem = tex_path.with_suffix(".pdf")
    if same_stem.is_file():
        return same_stem

    pdfs = sorted(tex_path.parent.glob("*.pdf"), key=lambda p: p.stat().st_mtime)
    if pdfs:
        return pdfs[-1]  # most recently modified

    return None


def _find_tikz_figures(tex_path: Path) -> list[tuple[str, str]]:
    """Find figure environments that have NO \\includegraphics (pure TikZ/pgfplots).

    Returns list of (label, caption) for each vector-only figure.
    """
    text = tex_path.read_text(encoding="utf-8")
    results: list[tuple[str, str]] = []

    for env_match in _FIGURE_ENV_RE.finditer(text):
        env_body = env_match.group(1)
        # Skip figures that have \includegraphics — already handled
        if _INCLUDEGRAPHICS_RE.search(env_body):
            continue

        caption_m = _CAPTION_RE.search(env_body)
        label_m = _LABEL_RE.search(env_body)
        caption = caption_m.group(1) if caption_m else ""
        label = label_m.group(1) if label_m else ""
        results.append((label, caption))

    return results


def _figure_pages_from_pdf(pdf_path: Path) -> dict[int, tuple[int, str]]:
    """Identify pages containing figure captions in a compiled PDF.

    Returns {figure_number: (page_index_0based, caption_snippet)}.
    Uses pypdf text extraction — fast, no rendering.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    result: dict[int, tuple[int, str]] = {}

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for m in _FIGURE_CAPTION_PDF_RE.finditer(text):
            fig_num = int(m.group(1))
            if fig_num not in result:
                # Grab ~80 chars after "Figure N:" as caption snippet
                start = m.end()
                snippet = text[start : start + 80].strip().split("\n")[0]
                result[fig_num] = (page_idx, snippet)

    return result


def render_tikz_figures(
    tex_path: Path,
    output_dir: Path,
    pdf_path: Path | None = None,
    dpi: int = 150,
) -> list[ExtractedImage]:
    """Render TikZ/pgfplots figures by finding them in the compiled PDF.

    1. Finds figure environments in .tex that have no \\includegraphics
    2. Locates the compiled PDF (explicit ``pdf_path``, or alongside .tex)
    3. Identifies which PDF pages contain those figures (via caption text)
    4. Renders those pages at ``dpi`` resolution via pdf2image (poppler)

    Returns ExtractedImage entries for each rendered figure page.
    """
    from pdf2image import convert_from_path

    tikz_figs = _find_tikz_figures(tex_path)
    if not tikz_figs:
        return []

    if pdf_path is None:
        pdf_path = _find_compiled_pdf(tex_path)
    if pdf_path is None:
        logger.debug("No compiled PDF found for {}", tex_path.name)
        return []

    # Map figure numbers to pages
    fig_pages = _figure_pages_from_pdf(pdf_path)
    if not fig_pages:
        logger.debug("No figure captions found in PDF text")
        return []

    # Match TikZ figures to PDF pages by order
    # TikZ figures appear in .tex order → Figure 1, 2, 3...
    # fig_pages maps figure number → page
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[ExtractedImage] = []

    for fig_idx, (label, caption) in enumerate(tikz_figs):
        fig_num = fig_idx + 1
        page_info = fig_pages.get(fig_num)
        if page_info is None:
            # Numbering may not start at 1, or mixed with raster figures.
            # Try matching by caption text.
            for num, (pidx, snippet) in fig_pages.items():
                if caption and snippet and caption[:30].lower() in snippet.lower():
                    page_info = (pidx, snippet)
                    break
        if page_info is None:
            logger.debug(
                "Could not find PDF page for TikZ figure {} (label={})",
                fig_num,
                label,
            )
            continue

        page_idx, _snippet = page_info
        # Render single page
        page_images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_idx + 1,  # 1-based
            last_page=page_idx + 1,
            fmt="png",
        )
        if not page_images:
            continue

        out_path = output_dir / f"tikz_fig{fig_num}_p{page_idx + 1}.png"
        page_images[0].save(str(out_path), "PNG")

        images.append(
            ExtractedImage(
                path=out_path,
                label=label,
                caption=caption,
                source="rendered",
            )
        )
        logger.debug(
            "Rendered TikZ figure {} from page {} → {}",
            fig_num,
            page_idx + 1,
            out_path.name,
        )

    logger.info("Rendered {} TikZ/vector figure(s) from compiled PDF", len(images))
    return images
