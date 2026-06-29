"""EditFile tool — exact string replacement in files (like Claude Code's Edit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arcticswarm.tools.base import BaseTool, ToolResult


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Perform an exact string replacement in a file. "
        "Provide old_string (must be unique in the file) and new_string. "
        "Set replace_all to true to replace every occurrence."
    )

    def __init__(self) -> None:
        pass

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find in the file.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace all occurrences. Default: false.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    def execute(
        self,
        *,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **_: Any,
    ) -> ToolResult:
        p = Path(file_path).expanduser()
        if not p.exists():
            return ToolResult(error=f"File not found: {file_path}", is_error=True)

        try:
            content = p.read_text()
        except Exception as exc:
            return ToolResult(error=f"Cannot read {file_path}: {exc}", is_error=True)

        count = content.count(old_string)
        if count == 0:
            return ToolResult(
                error=f"old_string not found in {file_path}. Make sure it matches exactly.",
                is_error=True,
            )
        if count > 1 and not replace_all:
            return ToolResult(
                error=(
                    f"old_string appears {count} times in {file_path}. "
                    "Provide more context to make it unique, or set replace_all=true."
                ),
                is_error=True,
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replaced = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replaced = 1

        try:
            p.write_text(new_content)
        except Exception as exc:
            return ToolResult(error=f"Cannot write {file_path}: {exc}", is_error=True)

        return ToolResult(output=f"Replaced {replaced} occurrence(s) in {file_path}")
