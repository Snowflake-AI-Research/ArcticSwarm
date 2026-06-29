"""Local skill loading with tool-based progressive disclosure.

Reads SKILL.md files from the local filesystem and provides a
:class:`SkillRegistry` for discovery and loading — matching the pattern
used by the cortex Go orchestrator (``ServerSkillTool``).

Two-stage progressive disclosure:
  1. Tool description lists all available skills (name + description)
  2. Agent calls ``load_skill(skill_name=...)`` to get full instructions

See ``DIFFERENCES.md`` in this directory for remaining gaps vs the
cortexagent implementation and the rationale for each.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# The skill library lives one level up, in ``arcticswarm/skills/``.  Production
# call sites always pass an explicit ``skills_dir``; this default backs the
# lazily-created module-level registry (``get_default_registry``).
SKILLS_DIR = Path(__file__).resolve().parent / "skills"

SKILL_PATH_PREFIX = "skill://"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Data classes (matching SI's EmbeddedSkill / SkillFileInfo)
# ---------------------------------------------------------------------------


@dataclass
class SkillLocation:
    """Where a skill lives on disk."""
    name: str
    base_path: Path


@dataclass
class SkillFileInfo:
    """A file in a skill directory (excluding SKILL.md)."""
    path: str
    size: int
    is_dir: bool


@dataclass
class LoadedSkill:
    """A fully loaded skill with content and file listing.

    Mirrors cortexagent's ``EmbeddedSkill`` struct.
    """
    metadata: dict[str, str]
    content: str
    file_list: list[SkillFileInfo] = field(default_factory=list)
    skill_path: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_skill_name(name: str) -> str:
    """Normalize a skill name for comparison (lowercase + strip).

    Matches SI's ``normalizeSkillName()``.
    """
    return name.strip().lower()


def _parse_skill_md(skill_path: Path) -> tuple[dict[str, str], str]:
    """Parse a SKILL.md file into (frontmatter_dict, body_markdown)."""
    text = skill_path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()
    frontmatter = yaml.safe_load(match.group(1)) or {}
    body = text[match.end():].strip()
    return frontmatter, body


def _list_skill_files(skill_dir: Path, relative: Path | None = None) -> list[SkillFileInfo]:
    """Recursively list all files in a skill directory, excluding SKILL.md."""
    file_list: list[SkillFileInfo] = []
    if relative is None:
        relative = Path("")

    current = skill_dir / relative
    try:
        for entry in sorted(current.iterdir()):
            entry_rel = relative / entry.name
            if entry.is_dir():
                file_list.append(SkillFileInfo(
                    path=str(entry_rel) + "/",
                    size=0,
                    is_dir=True,
                ))
                file_list.extend(_list_skill_files(skill_dir, entry_rel))
            else:
                if entry.name == "SKILL.md" and relative == Path(""):
                    continue
                file_list.append(SkillFileInfo(
                    path=str(entry_rel),
                    size=entry.stat().st_size,
                    is_dir=False,
                ))
    except (PermissionError, OSError) as exc:
        logger.warning("Error listing directory %s: %s", current, exc)

    return file_list


def _format_file_size(size: int) -> str:
    """Format a file size in human-readable form."""
    if size < 1024:
        return f"{size} B"
    for unit in ("KB", "MB", "GB"):
        size_f = size / 1024
        if size_f < 1024 or unit == "GB":
            return f"{size_f:.1f} {unit}"
        size = int(size_f)
    return f"{size} B"


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Discovers and manages skills from a local directory.

    Walks ``skills_dir`` at init time to find all sub-directories
    containing a ``SKILL.md`` file.  Provides methods to list, look up,
    and fully load skills — matching the cortexagent
    ``ServerSkillTool`` / sandbox ``SkillExecutor`` patterns.
    """

    def __init__(
        self,
        skills_dir: Path | str = SKILLS_DIR,
        enabled_skills: list[str] | None = None,
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._registry: dict[str, SkillLocation] = {}
        self._enabled_skills: list[str] | None = enabled_skills
        self._last_config: list[str] | None = None
        self.discover()
        if enabled_skills is not None:
            self.configure(enabled_skills)

    # -- discovery -----------------------------------------------------------

    def discover(self) -> None:
        """Walk ``skills_dir`` for directories containing SKILL.md."""
        self._registry.clear()
        if not self._skills_dir.is_dir():
            logger.warning("Skills directory does not exist: %s", self._skills_dir)
            return

        for entry in sorted(self._skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if skill_md.exists():
                name = normalize_skill_name(entry.name)
                self._registry[name] = SkillLocation(
                    name=entry.name,
                    base_path=entry,
                )

        logger.info(
            "SkillRegistry discovered %d skill(s) in %s",
            len(self._registry),
            self._skills_dir,
        )

    # -- configure / reconfigure ---------------------------------------------

    def configure(self, enabled_skills: list[str]) -> None:
        """Restrict the registry to only the given skills.

        Caches the config to skip redundant reconfiguration (matching
        sandbox ``SkillExecutor``'s ``_last_config`` pattern).
        """
        if self._last_config == enabled_skills:
            return

        self._last_config = list(enabled_skills)
        self._enabled_skills = list(enabled_skills)

        # Re-discover to rebuild the full set, then filter
        self.discover()
        if not enabled_skills:
            return

        normalized = {normalize_skill_name(s) for s in enabled_skills}
        self._registry = {
            k: v for k, v in self._registry.items()
            if k in normalized
        }

    # -- lookup / listing ----------------------------------------------------

    def get(self, name: str) -> SkillLocation | None:
        """Look up a skill by name (normalized)."""
        return self._registry.get(normalize_skill_name(name))

    def list_skills(self) -> list[str]:
        """Return sorted list of registered skill names."""
        return sorted(loc.name for loc in self._registry.values())

    def get_metadata(self, name: str) -> dict[str, str]:
        """Read YAML frontmatter from a skill's SKILL.md.

        Returns ``{"name": ..., "description": ..., "location": ...}``.
        """
        loc = self.get(name)
        if loc is None:
            raise FileNotFoundError(f"Skill not found: {name}")
        skill_md = loc.base_path / "SKILL.md"
        fm, _ = _parse_skill_md(skill_md)
        return {
            "name": fm.get("name", loc.name),
            "description": fm.get("description", "").strip(),
            "location": fm.get("location", ""),
        }

    def get_all_metadata(
        self,
        skill_names: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Return metadata for all (or specified) registered skills."""
        names = skill_names or self.list_skills()
        return [self.get_metadata(n) for n in names]

    # -- full load -----------------------------------------------------------

    def load_skill(self, name: str) -> LoadedSkill:
        """Fully load a skill: content, metadata, file listing, path prefix.

        Mirrors cortexagent ``loadEmbeddedSkill()``.
        """
        loc = self.get(name)
        if loc is None:
            raise FileNotFoundError(f"Skill not found: {name}")

        skill_md = loc.base_path / "SKILL.md"
        fm, body = _parse_skill_md(skill_md)

        metadata = {
            "name": fm.get("name", loc.name),
            "description": fm.get("description", "").strip(),
            "location": fm.get("location", ""),
        }

        file_list = _list_skill_files(loc.base_path)
        skill_path = f"{SKILL_PATH_PREFIX}{loc.name}/"

        return LoadedSkill(
            metadata=metadata,
            content=body,
            file_list=file_list,
            skill_path=skill_path,
        )

    # -- file reading --------------------------------------------------------

    def read_skill_file(self, skill_name: str, relative_path: str) -> str:
        """Read a file from a skill directory.

        ``relative_path`` is relative to the skill root (e.g.
        ``scripts/foo.py``).
        """
        loc = self.get(skill_name)
        if loc is None:
            raise FileNotFoundError(f"Skill not found: {skill_name}")

        target = (loc.base_path / relative_path).resolve()
        # Ensure we don't escape the skill directory
        if not str(target).startswith(str(loc.base_path.resolve())):
            raise ValueError(
                f"Path escapes skill directory: {relative_path}"
            )

        if not target.exists():
            raise FileNotFoundError(
                f"File not found in skill '{skill_name}': {relative_path}"
            )

        return target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Default registry instance + backward-compat helpers
# ---------------------------------------------------------------------------

_default_registry: SkillRegistry | None = None


def get_default_registry() -> SkillRegistry:
    """Return (and lazily create) the module-level default registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry(SKILLS_DIR)
    return _default_registry


SKILL_NAMES: list[str] = []  # populated on first access via __getattr__


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ to make ``SKILL_NAMES`` dynamic."""
    if name == "SKILL_NAMES":
        return get_default_registry().list_skills()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_skill_metadata(skill_name: str) -> dict[str, str]:
    """Read YAML frontmatter from a SKILL.md file (backward compat)."""
    return get_default_registry().get_metadata(skill_name)


def load_skill_content(skill_name: str) -> str:
    """Read the full body of a SKILL.md file (backward compat)."""
    loaded = get_default_registry().load_skill(skill_name)
    return loaded.content


def get_all_skill_metadata(
    skill_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """Return metadata for all (or specified) skills (backward compat)."""
    return get_default_registry().get_all_metadata(skill_names)


# ---------------------------------------------------------------------------
# Tool description construction (SI-style format)
# ---------------------------------------------------------------------------


def build_load_skill_tool_description_legacy(
    skill_names: list[str],
    registry: SkillRegistry | None = None,
) -> str:
    """Old compact format: 2-line intro + self-closing ``<skill .../>`` tags.

    Used for A/B comparison against the SI-aligned format.
    """
    reg = registry or get_default_registry()
    lines = [
        "Load a skill by name to get its full instructions. "
        "Review the available skills below and load the ones relevant "
        "to your current task.",
        "",
        "<available_skills>",
    ]
    for name in skill_names:
        try:
            meta = reg.get_metadata(name)
        except FileNotFoundError:
            continue
        desc = meta["description"].replace('"', "&quot;")
        lines.append(f'<skill name="{meta["name"]}" description="{desc}" />')
    lines.append("</available_skills>")
    return "\n".join(lines)


def build_load_skill_tool_description(
    skill_names: list[str],
    registry: SkillRegistry | None = None,
) -> str:
    """Build the ``load_skill`` tool description with available skills.

    Uses the cortexagent ``ServerSkillTool`` format: nested
    ``<skill><name>/<description>`` XML wrapped in a
    ``<skills_instructions>`` block.
    """
    reg = registry or get_default_registry()

    skill_blocks: list[str] = []
    for name in skill_names:
        try:
            meta = reg.get_metadata(name)
        except FileNotFoundError:
            continue
        skill_blocks.append(
            "<skill>\n"
            f"<name>\n{meta['name']}\n</name>\n"
            f"<description>\n{meta['description']}\n</description>\n"
            "</skill>"
        )

    skills_xml = "\n".join(skill_blocks)

    return (
        "Load and execute a skill to get specialized instructions and knowledge.\n"
        "\n"
        "<skills_instructions>\n"
        "When users ask you to perform tasks, check if any of the available "
        "skills below can help complete the task more effectively. Skills "
        "provide specialized capabilities and domain knowledge.\n"
        "\n"
        "IMPORTANT: Skills marked with **[REQUIRED]** tag MUST be invoked as "
        "your FIRST action when the task matches their domain. DO NOT attempt "
        "to handle these tasks with direct tool usage - always invoke the "
        "required skill first.\n"
        "\n"
        "How to use skills:\n"
        "- Invoke skills using this tool with the skill name only\n"
        "- The skill's prompt will expand and provide detailed instructions "
        "on how to complete the task\n"
        "\n"
        "Important:\n"
        "- Only use skills listed in <available_skills> below\n"
        "- Do not invoke a skill that is already running\n"
        "</skills_instructions>\n"
        "\n"
        "<available_skills>\n"
        f"{skills_xml}\n"
        "</available_skills>"
    )


def build_system_reminder(
    skill_names: list[str],
    registry: SkillRegistry | None = None,
) -> str:
    """Build a ``<system-reminder>`` with the same skill listing as the tool description.

    Intended for periodic injection into the conversation to keep the
    LLM aware of available skills (mirrors Claude Code's pattern).
    """
    reg = registry or get_default_registry()
    skill_blocks: list[str] = []
    for name in skill_names:
        try:
            meta = reg.get_metadata(name)
        except FileNotFoundError:
            continue
        skill_blocks.append(
            "<skill>\n"
            f"<name>\n{meta['name']}\n</name>\n"
            f"<description>\n{meta['description']}\n</description>\n"
            "</skill>"
        )
    skills_xml = "\n".join(skill_blocks)
    return (
        "<system-reminder>\n"
        "<available_skills>\n"
        f"{skills_xml}\n"
        "</available_skills>\n"
        "</system-reminder>"
    )


def make_load_skill_tool_schema(
    skill_names: list[str],
    registry: SkillRegistry | None = None,
) -> dict[str, Any]:
    """Return the Anthropic-format tool definition for ``load_skill``.

    The ``skill_name`` parameter is constrained to an enum of the
    available skill names so the LLM can only request valid skills.
    """
    reg = registry or get_default_registry()
    metadata = reg.get_all_metadata(skill_names)
    valid_names = [m["name"] for m in metadata]

    return {
        "name": "load_skill",
        "description": build_load_skill_tool_description(skill_names, reg),
        "input_schema": {
            "type": "object",
            "required": ["skill_name"],
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to load.",
                    "enum": valid_names,
                },
            },
        },
    }
