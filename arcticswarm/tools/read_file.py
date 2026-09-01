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

"""ReadFile tool — read a local file by absolute path."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from arcticswarm.tools._image_media import IMAGE_EXTENSIONS, IMAGE_MEDIA_TYPES
from arcticswarm.tools.base import BaseTool, ToolResult

_MAX_LINES = 2000
_MAX_LINE_LEN = 500

_DESC_TEXT_ONLY = (
    "Read a file from the local filesystem. Returns numbered lines. "
    "You can optionally specify offset (1-based line number) and limit "
    "(number of lines) to read a slice of a large file."
)
_DESC_WITH_VISION = (
    "Read a file from the local filesystem. Returns numbered lines for text files, "
    "or renders the image for image files (png, jpg, gif, webp). "
    "You can optionally specify offset (1-based line number) and limit "
    "(number of lines) to read a slice of a large text file."
)


class ReadFileTool(BaseTool):
    name = "read_file"

    def __init__(
        self,
        *,
        enable_vision: bool = False,
    ) -> None:
        self._enable_vision = enable_vision

    @property
    def description(self) -> str:
        return _DESC_WITH_VISION if self._enable_vision else _DESC_TEXT_ONLY

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read.",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from. Default: 1.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max lines to return. Default: {_MAX_LINES}.",
                },
            },
            "required": ["file_path"],
        }

    def execute(self, *, file_path: str, offset: int = 1, limit: int = _MAX_LINES, **_: Any) -> ToolResult:
        p = Path(file_path).expanduser()
        if not p.exists():
            return ToolResult(error=f"File not found: {file_path}", is_error=True)
        if p.is_dir():
            return ToolResult(error=f"Path is a directory, not a file: {file_path}", is_error=True)

        if self._enable_vision:
            ext = p.suffix.lstrip(".").lower()
            if ext in IMAGE_EXTENSIONS:
                return self._read_image(p, ext)

        try:
            text = p.read_text(errors="replace")
        except Exception as exc:
            return ToolResult(error=f"Cannot read {file_path}: {exc}", is_error=True)

        lines = text.splitlines()
        start = max(0, offset - 1)
        end = start + limit
        selected = lines[start:end]

        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            truncated = line[:_MAX_LINE_LEN] + ("…" if len(line) > _MAX_LINE_LEN else "")
            numbered.append(f"{i:6d}\t{truncated}")

        total = len(lines)
        header = f"[{file_path}] lines {start + 1}-{min(end, total)} of {total}"
        return ToolResult(output=header + "\n" + "\n".join(numbered))

    @staticmethod
    def _read_image(p: Path, ext: str) -> ToolResult:
        """Return an image as a base64 content block for multimodal LLMs."""
        try:
            data = p.read_bytes()
        except Exception as exc:
            return ToolResult(error=f"Cannot read image {p}: {exc}", is_error=True)

        media_type = IMAGE_MEDIA_TYPES.get(ext, "image/png")
        b64 = base64.standard_b64encode(data).decode("ascii")

        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        }
        return ToolResult(
            output=f"[Image: {p.name} ({len(data)} bytes)]",
            extra_content=[image_block],
        )
