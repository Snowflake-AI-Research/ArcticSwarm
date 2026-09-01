"""Shared image media-type helpers.

Single source of truth for supported image extensions and their MIME
media types. Used by :mod:`arcticswarm.tools.read_file` (for the vision
variant of ``read_file``) and by the evaluation runner's multimodal
user-message builder.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "gif", "webp"})

IMAGE_MEDIA_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


def media_type_for_extension(ext: str, *, default: str = "image/png") -> str:
    """Return the MIME media type for a lowercase file extension (no dot)."""
    return IMAGE_MEDIA_TYPES.get(ext.lstrip(".").lower(), default)


def media_type_for_path(path: str | Path, *, default: str = "image/png") -> str:
    """Return the MIME media type for an image file path."""
    ext = Path(path).suffix.lstrip(".").lower()
    return IMAGE_MEDIA_TYPES.get(ext, default)
