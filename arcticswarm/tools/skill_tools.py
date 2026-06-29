"""Skill tools — LoadSkillTool, ReadSkillFileTool, PerSkillTool.

``LoadSkillTool`` exposes SKILL.md files via the standard ``BaseTool``
interface with rich structured results (metadata, instructions, file listing).

``ReadSkillFileTool`` allows agents to read individual files from a
skill's directory using ``skill://name/path`` URIs.

``PerSkillTool`` presents each skill as its own named tool (one tool per
skill) instead of routing through a single ``load_skill`` dispatcher.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from arcticswarm.skill_loader import (
    SkillRegistry,
    _format_file_size,
    build_load_skill_tool_description,
    build_load_skill_tool_description_legacy,
    get_default_registry,
)
from arcticswarm.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_SKILL_URI_RE = re.compile(
    r"^skill://([^/]+)/(.+)$"
)


# ---------------------------------------------------------------------------
# LoadSkillTool
# ---------------------------------------------------------------------------


class LoadSkillTool(BaseTool):
    """Load a skill by name, returning structured instructions + file listing.

    The tool description lists all available skills so the LLM can
    discover them (progressive disclosure).  Calling the tool with a
    specific ``skill_name`` returns a rich result matching SI's
    ``RenderForMemory`` format.
    """

    def __init__(
        self,
        skill_names: list[str],
        registry: SkillRegistry | None = None,
        legacy_format: bool = False,
        tool_name_overrides: dict[str, str] | None = None,
    ) -> None:
        self._skill_names = list(skill_names)
        self._registry = registry or get_default_registry()
        self._legacy_format = legacy_format
        self._tool_name_overrides = tool_name_overrides or {}
        metadata = self._registry.get_all_metadata(self._skill_names)
        self._valid_names = [m["name"] for m in metadata]

    @property
    def name(self) -> str:
        return "load_skill"

    @property
    def description(self) -> str:
        if self._legacy_format:
            return build_load_skill_tool_description_legacy(
                self._skill_names, self._registry,
            )
        return build_load_skill_tool_description(
            self._skill_names, self._registry,
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["skill_name"],
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to load.",
                    "enum": self._valid_names,
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        skill_name = kwargs.get("skill_name", "")
        if not skill_name:
            return ToolResult(
                error="'skill_name' is required.",
                is_error=True,
            )
        try:
            loaded = self._registry.load_skill(skill_name)
        except FileNotFoundError:
            return ToolResult(
                error=f"Skill '{skill_name}' not found.",
                is_error=True,
            )

        content = loaded.content
        for old, new in self._tool_name_overrides.items():
            content = content.replace(old, new)

        if self._legacy_format:
            return ToolResult(output=content)

        parts: list[str] = [
            f"# Skill: {loaded.metadata['name']}",
            "",
            f"**Description**: {loaded.metadata['description']}",
            "",
            "## Instructions",
            "",
            content,
        ]

        non_dir_files = [f for f in loaded.file_list if not f.is_dir]
        if non_dir_files:
            parts.extend([
                "",
                "## Skill Directory Contents",
                "",
                "| File Path | Size |",
                "|-----------|------|",
            ])
            for fi in non_dir_files:
                uri = f"{loaded.skill_path}{fi.path}"
                parts.append(
                    f"| `{uri}` | {_format_file_size(fi.size)} |"
                )
            parts.extend([
                "",
                "**To read a file**, use the `read_skill_file` tool with "
                "the file path shown above.",
            ])

        return ToolResult(output="\n".join(parts))


# ---------------------------------------------------------------------------
# PerSkillTool  (Claude Code-style: one tool per skill)
# ---------------------------------------------------------------------------


SKILL_TOOL_PREFIX = "skill-"


class PerSkillTool(BaseTool):
    """A single-skill tool that the model invokes by a prefixed skill name.

    Instead of ``load_skill(skill_name="sql-building")``, the model calls
    ``skill-sql-building()`` directly.  The ``skill-`` prefix makes it
    unambiguous that this is an instruction/knowledge tool, not an
    execution tool.
    """

    def __init__(
        self,
        skill_name: str,
        registry: SkillRegistry | None = None,
        legacy_format: bool = False,
        tool_name_overrides: dict[str, str] | None = None,
    ) -> None:
        self._skill_name = skill_name
        self._registry = registry or get_default_registry()
        self._legacy_format = legacy_format
        self._tool_name_overrides = tool_name_overrides or {}
        self._metadata = self._registry.get_metadata(skill_name)

    @property
    def name(self) -> str:
        return f"{SKILL_TOOL_PREFIX}{self._skill_name}"

    @property
    def description(self) -> str:
        desc = self._metadata.get("description", "")
        if not desc:
            desc = f"Specialized instructions for {self._skill_name}."
        return f"[Skill] {desc} Invoke this tool to load detailed instructions."

    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            loaded = self._registry.load_skill(self._skill_name)
        except FileNotFoundError:
            return ToolResult(
                error=f"Skill '{self._skill_name}' not found.",
                is_error=True,
            )

        content = loaded.content
        for old, new in self._tool_name_overrides.items():
            content = content.replace(old, new)

        if self._legacy_format:
            return ToolResult(output=content)

        parts: list[str] = [
            f"# Skill: {loaded.metadata['name']}",
            "",
            f"**Description**: {loaded.metadata['description']}",
            "",
            "## Instructions",
            "",
            content,
        ]

        non_dir_files = [f for f in loaded.file_list if not f.is_dir]
        if non_dir_files:
            parts.extend([
                "",
                "## Skill Directory Contents",
                "",
                "| File Path | Size |",
                "|-----------|------|",
            ])
            for fi in non_dir_files:
                uri = f"{loaded.skill_path}{fi.path}"
                parts.append(
                    f"| `{uri}` | {_format_file_size(fi.size)} |"
                )
            parts.extend([
                "",
                "**To read a file**, use the `read_skill_file` tool with "
                "the file path shown above.",
            ])

        return ToolResult(output="\n".join(parts))


def make_per_skill_tools(
    skill_names: list[str],
    registry: SkillRegistry | None = None,
    legacy_format: bool = False,
    tool_name_overrides: dict[str, str] | None = None,
) -> dict[str, "PerSkillTool"]:
    """Create one :class:`PerSkillTool` per skill name, keyed by tool name.

    Keys use the ``skill-`` prefixed name (matching ``tool.name``).
    """
    reg = registry or get_default_registry()
    tools: dict[str, PerSkillTool] = {}
    for sn in skill_names:
        try:
            tool = PerSkillTool(
                sn,
                registry=reg,
                legacy_format=legacy_format,
                tool_name_overrides=tool_name_overrides,
            )
            tools[tool.name] = tool
        except FileNotFoundError:
            logger.warning("Skill '%s' not found — skipping PerSkillTool", sn)
    return tools


# ---------------------------------------------------------------------------
# ReadSkillFileTool
# ---------------------------------------------------------------------------


class ReadSkillFileTool(BaseTool):
    """Read a file from a loaded skill's directory.

    Accepts ``skill://skill-name/relative/path`` URIs and returns the
    raw file content.  Registered alongside ``LoadSkillTool`` so agents
    can inspect supplementary skill files (scripts, examples, etc.).
    """

    def __init__(self, registry: SkillRegistry | None = None) -> None:
        self._registry = registry or get_default_registry()

    @property
    def name(self) -> str:
        return "read_skill_file"

    @property
    def description(self) -> str:
        return (
            "Read a file from a skill's directory. Use the file paths "
            "shown in the Skill Directory Contents table returned by "
            "load_skill. The file_path should be a skill:// URI like "
            "'skill://skill-name/path/to/file'."
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["file_path"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "The skill:// URI to the file, e.g. "
                        "'skill://sql-building/examples/query.sql'."
                    ),
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        file_path = kwargs.get("file_path", "")
        if not file_path:
            return ToolResult(
                error="'file_path' is required.",
                is_error=True,
            )

        match = _SKILL_URI_RE.match(file_path)
        if not match:
            return ToolResult(
                error=(
                    f"Invalid file path: {file_path!r}. "
                    f"Expected format: skill://skill-name/relative/path"
                ),
                is_error=True,
            )

        skill_name = match.group(1)
        relative_path = match.group(2)

        try:
            content = self._registry.read_skill_file(skill_name, relative_path)
            return ToolResult(output=content)
        except FileNotFoundError as exc:
            return ToolResult(error=str(exc), is_error=True)
        except ValueError as exc:
            return ToolResult(error=str(exc), is_error=True)
