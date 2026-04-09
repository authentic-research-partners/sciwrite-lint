"""Cache for vision model figure descriptions.

Stores descriptions in workspace.db (``vision_cache`` table), keyed by
image filename stem + SHA-256 content hash.  Only re-runs VL inference
when images change.

All DB access goes through ``get_db()`` — the universal pattern.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from sciwrite_lint.vision.image_extraction import ExtractedImage


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
) -> list[ExtractedImage]:
    """Return images that need VL inference (not in cache or hash changed).

    Uses ``get_db()`` to check the vision_cache table.
    """
    from sciwrite_lint.references.workspace_db import get_db, load_vision_entry

    new_images: list[ExtractedImage] = []
    with get_db(references_dir) as conn:
        for img in images:
            key = img.path.stem
            current_hash = _hash_image_and_caption(img.path, img.caption)
            entry = load_vision_entry(conn, key, expected_hash=current_hash)
            if entry is not None:
                continue  # Valid cache hit
            new_images.append(img)

    return new_images


def update_cache(
    images: list[ExtractedImage],
    descriptions: list[str],
    references_dir: Path,
) -> None:
    """Save new descriptions to workspace.db."""
    from sciwrite_lint.references.workspace_db import get_db, save_vision_entry

    with get_db(references_dir) as conn:
        for img, desc in zip(images, descriptions):
            save_vision_entry(
                conn,
                img.path.stem,
                image_hash=_hash_image_and_caption(img.path, img.caption),
                label=img.label,
                caption=img.caption,
                description=desc,
            )


def format_descriptions_from_db(
    images: list[ExtractedImage],
    references_dir: Path,
) -> str:
    """Format figure descriptions for the LLM system prompt.

    VL descriptions come from workspace.db.  Captions and labels come from
    the *current* ``images`` list (freshly parsed from .tex), not from the
    DB — so the formatted output always reflects the latest source even if
    VL inference was cached from an earlier caption.
    """
    from sciwrite_lint.references.workspace_db import get_db, load_all_vision_entries

    with get_db(references_dir) as conn:
        all_entries = load_all_vision_entries(conn)

    parts: list[str] = []
    for img in images:
        key = img.path.stem
        entry = all_entries.get(key)
        if entry is None:
            continue

        # Use current caption/label from extraction, not stale DB values
        header = "Figure"
        if img.label:
            header += f" ({img.label})"
        if img.caption:
            header += f' — Caption: "{img.caption}"'
        parts.append(f"{header}\nVisual content: {entry['description']}")

    return "\n\n".join(parts)


def load_all_descriptions(references_dir: Path) -> str:
    """Load all cached figure descriptions as formatted text.

    Used by full-paper consistency checks to inject into the system prompt
    without needing the original ExtractedImage list.
    """
    from sciwrite_lint.references.workspace_db import get_db, load_all_vision_entries

    with get_db(references_dir) as conn:
        all_entries = load_all_vision_entries(conn)

    if not all_entries:
        return ""

    parts: list[str] = []
    for key, entry in all_entries.items():
        header = "Figure"
        if entry["label"]:
            header += f" ({entry['label']})"
        if entry["caption"]:
            header += f' — Caption: "{entry["caption"]}"'
        parts.append(f"{header}\nVisual content: {entry['description']}")

    logger.info("Loaded {} cached figure descriptions", len(parts))
    return "\n\n".join(parts)


def clear_cache(references_dir: Path) -> None:
    """Delete all vision cache entries (for --fresh)."""
    from sciwrite_lint.references.workspace_db import clear_vision_cache, get_db

    with get_db(references_dir) as conn:
        clear_vision_cache(conn)
