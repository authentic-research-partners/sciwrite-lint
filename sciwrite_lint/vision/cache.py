"""Cache for vision model figure descriptions.

Stores descriptions in workspace.db (``vision_cache`` table), keyed by
image filename stem + SHA-256 content hash.  Only re-runs VL inference
when images change.

The vLLM backend produces structured JSON (figure_type, description,
readability_issues); the transformers backend produces free text stored
in the description field only.

All DB access goes through ``get_db()`` — the universal pattern.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
from pydantic import BaseModel, Field, StringConstraints

from sciwrite_lint.vision.image_extraction import ExtractedImage


# Per-item char limit for ``readability_issues``. Individual issues are
# short phrases like "y-axis label partially obscured" (~30 chars); 150
# leaves headroom for slightly longer notes without inviting essays.
ReadabilityIssue = Annotated[str, StringConstraints(max_length=150)]


class VisionResult(BaseModel):
    """Structured output from vision model figure analysis.

    The vLLM backend populates all three fields from its JSON response.
    The transformers backend populates only ``description`` (free text)
    and leaves ``readability_issues`` as an empty list.

    Field bounds are decoder-enforced: the vLLM backend sends this model
    as a ``json_schema`` response format with ``strict=True`` (see
    ``vision/describe.py:_vision_response_format``), so Pydantic's
    ``max_length`` shows up as ``maxLength`` / ``maxItems`` in the
    generated JSON schema and the constrained decoder respects both
    during token generation.

    - ``description``: ~1000 words cap (4000 chars) — plenty of room
      for a dense figure write-up.
    - ``readability_issues``: list of at most 5 short notes, each up to
      150 chars. Empty list means "no issues". This is a list rather
      than a string because a figure can legitimately have several
      distinct problems (obscured label + cut-off legend + unreadable
      tick marks), and making each an explicit item keeps the
      downstream consumer's formatting clean and lets the prompt cap
      "at most 5, most important first".
    - ``figure_type``: short enum-style label (~10-20 chars natural).

    Python-side defaults (``default=""``, ``default_factory=list``)
    exist so the transformers backend and test fixtures can construct
    instances without all three fields.
    """

    figure_type: str = Field(default="", max_length=80)
    description: str = Field(default="", max_length=4000)
    readability_issues: list[ReadabilityIssue] = Field(
        default_factory=list, max_length=5
    )


def _hash_image_and_caption(path: Path, caption: str = "") -> str:
    """SHA-256 hash of image bytes + caption text (first 32 hex chars).

    The VL prompt includes the caption, so changing the caption should
    invalidate the cache (the model may focus on different details).
    """
    h = hashlib.sha256(path.read_bytes())
    h.update(caption.encode("utf-8"))
    return h.hexdigest()[:32]


def split_cached_and_new(
    images: list[ExtractedImage],
    references_dir: Path,
    source: str = "manuscript",
) -> list[ExtractedImage]:
    """Return images that need VL inference (not in cache or hash changed).

    Uses ``get_db()`` to check the vision_cache table.  ``source``
    scopes the lookup to manuscript or a specific cited ref.
    """
    from sciwrite_lint.references.workspace_db import get_db, load_vision_entry

    new_images: list[ExtractedImage] = []
    with get_db(references_dir) as conn:
        for img in images:
            key = img.path.stem
            current_hash = _hash_image_and_caption(img.path, img.caption)
            entry = load_vision_entry(
                conn, key, expected_hash=current_hash, source=source
            )
            if entry is not None:
                continue  # Valid cache hit
            new_images.append(img)

    return new_images


def update_cache(
    images: list[ExtractedImage],
    results: list[VisionResult],
    references_dir: Path,
    source: str = "manuscript",
) -> None:
    """Save new vision results to workspace.db.

    ``source`` tags entries as manuscript (``"manuscript"``) or cited ref
    (the ref_key).  Part of the composite PK — same image_key from
    different sources won't collide.
    """
    from sciwrite_lint.references.workspace_db import get_db, save_vision_entry

    with get_db(references_dir) as conn:
        for img, result in zip(images, results):
            save_vision_entry(
                conn,
                img.path.stem,
                image_hash=_hash_image_and_caption(img.path, img.caption),
                source=source,
                label=img.label,
                caption=img.caption,
                description=result.description,
                figure_type=result.figure_type,
                readability_issues=result.readability_issues,
            )


def _format_entry(
    label: str,
    caption: str,
    entry: dict[str, Any],
) -> str:
    """Format a single vision cache entry for the LLM system prompt.

    If structured fields (figure_type, readability_issues) are present,
    uses the structured format. Otherwise formats free-text description only.
    """
    header = "Figure"
    if label:
        header += f" ({label})"
    if caption:
        header += f' — Caption: "{caption}"'

    figure_type = entry.get("figure_type", "")
    readability = entry.get("readability_issues") or []
    description = entry["description"]

    lines = [header]
    if figure_type:
        lines.append(f"Type: {figure_type}")
    lines.append(f"Visual content: {description}")
    if readability:
        if len(readability) == 1:
            lines.append(f"Readability: {readability[0]}")
        else:
            lines.append("Readability issues:")
            lines.extend(f"  - {issue}" for issue in readability)

    return "\n".join(lines)


def format_descriptions_from_db(
    images: list[ExtractedImage],
    references_dir: Path,
    source: str = "manuscript",
) -> str:
    """Format figure descriptions for the LLM system prompt.

    VL descriptions come from workspace.db.  Captions and labels come from
    the *current* ``images`` list (freshly parsed from .tex), not from the
    DB — so the formatted output always reflects the latest source even if
    VL inference was cached from an earlier caption.

    ``source`` scopes to manuscript or a specific cited ref.
    """
    from sciwrite_lint.references.workspace_db import get_db, load_all_vision_entries

    with get_db(references_dir) as conn:
        all_entries = load_all_vision_entries(conn, source=source)

    parts: list[str] = []
    for img in images:
        key = img.path.stem
        entry = all_entries.get(key)
        if entry is None:
            continue

        parts.append(_format_entry(img.label, img.caption, entry))

    return "\n\n".join(parts)


def load_all_descriptions(
    references_dir: Path,
    source: str = "manuscript",
) -> str:
    """Load cached figure descriptions as formatted text.

    Used by full-paper consistency checks to inject into the system prompt
    without needing the original ExtractedImage list.

    ``source`` defaults to ``"manuscript"`` — only the paper's own figures.
    Pass a ref_key to load a cited paper's figures.
    """
    from sciwrite_lint.references.workspace_db import get_db, load_all_vision_entries

    with get_db(references_dir) as conn:
        all_entries = load_all_vision_entries(conn, source=source)

    if not all_entries:
        return ""

    parts: list[str] = []
    for _key, entry in all_entries.items():
        parts.append(_format_entry(entry["label"], entry["caption"], entry))

    logger.info("Loaded {} cached figure descriptions (source={})", len(parts), source)
    return "\n\n".join(parts)


def clear_cache(references_dir: Path) -> None:
    """Delete all vision cache entries (for --fresh)."""
    from sciwrite_lint.references.workspace_db import clear_vision_cache, get_db

    with get_db(references_dir) as conn:
        clear_vision_cache(conn)
