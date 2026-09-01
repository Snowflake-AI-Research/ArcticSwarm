# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
