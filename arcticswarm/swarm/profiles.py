"""Tool profiles for swarm subagents.

Each profile defines what tools, system prompt, and idle behaviour a
subagent gets when it claims a task tagged with that profile.

Profiles are frozen dataclasses — immutable and thread-safe by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolProfile:
    """A named tool profile controlling subagent capabilities per task."""

    name: str
    """Short identifier: ``"browsing"``, ``"coding"``, ``"reasoning"``."""

    included_tools: frozenset[str]
    """Tools the subagent KEEPS.  Empty means keep all registered tools."""

    excluded_tools: frozenset[str] = frozenset()
    """Tools to DROP (applied after *included_tools*)."""

    system_prompt_key: str = ""
    """Key used to look up the profile-specific system prompt fragment."""

    idle_review_key: str = ""
    """``"research"`` — controls idle-check behaviour."""

    orchestrator_description: str = ""
    """Human-readable blurb shown in the orchestrator system prompt."""

    bbs_channels: frozenset[str] = frozenset()
    """BBS channels this profile introduces (beyond the always-present core set)."""

    skill_names: tuple[str, ...] = ()
    """Composable skills loaded via LoadSkillTool for this profile."""

    supports_reflection: bool = False
    """When True, subagent uses the structured Search→Reflect→Summarize loop."""

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ToolProfile":
        """Build a ToolProfile from a YAML-sourced dictionary.

        Expected keys: ``tools``, ``skills``, ``description``,
        ``bbs_channels``.  Missing keys fall through to built-in defaults
        for this profile name (if one exists).
        """
        base = PROFILES.get(name)
        return cls(
            name=name,
            included_tools=frozenset(data["tools"]) if "tools" in data else (base.included_tools if base else frozenset()),
            skill_names=tuple(data["skills"]) if "skills" in data else (base.skill_names if base else ()),
            orchestrator_description=data.get("description", base.orchestrator_description if base else ""),
            bbs_channels=frozenset(data["bbs_channels"]) if "bbs_channels" in data else (base.bbs_channels if base else frozenset()),
            system_prompt_key=base.system_prompt_key if base else name,
            idle_review_key=base.idle_review_key if base else "research",
            supports_reflection=base.supports_reflection if base else False,
        )


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

PROFILE_BROWSING = ToolProfile(
    name="browsing",
    included_tools=frozenset({"web_search", "web_fetch", "pdf_read", "read_file", "calculator"}),
    system_prompt_key="browsing",
    idle_review_key="research",
    orchestrator_description=(
        "Web research specialist. Tools: web_search, web_fetch, pdf_read, "
        "read_file, calculator. Use for questions requiring external "
        "information, fact-checking, or current events. Can search the web, "
        "fetch full page content, and read PDF documents."
    ),
    skill_names=("web-research", "tool-usage-policy-browsing", "task-completion-web"),
    supports_reflection=True,
    )

PROFILE_CODING = ToolProfile(
    name="coding",
    included_tools=frozenset({
        "bash", "python_execute",
        "read_file", "edit_file", "calculator",
    }),
    system_prompt_key="coding",
    idle_review_key="research",
    orchestrator_description=(
        "Code execution specialist. Tools: bash, python_execute, "
        "read_file, edit_file, calculator. "
        "Use for computation, scripting, data processing, and file operations."
    ),
    skill_names=("coding-execution", "tool-usage-policy-coding"),
    )

PROFILE_REASONING = ToolProfile(
    name="reasoning",
    included_tools=frozenset({"reasoning"}),
    system_prompt_key="reasoning",
    idle_review_key="research",
    orchestrator_description=(
        "Deep reasoning specialist. Tools: reasoning (extended thinking). "
        "Use for hard math, logic puzzles, complex analysis, or "
        "verifying findings that require deep chain-of-thought."
    ),
    skill_names=("deep-reasoning",),
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROFILES: dict[str, ToolProfile] = {
    p.name: p
    for p in [
        PROFILE_BROWSING, PROFILE_CODING, PROFILE_REASONING,
    ]
}

DEFAULT_PROFILE_NAME = "browsing"


def load_profiles_from_config(
    tool_profiles: dict[str, dict],
) -> dict[str, ToolProfile]:
    """Build a complete profile dict from YAML overrides + built-in defaults.

    Profiles specified in *tool_profiles* (from ``config.tool_profiles``)
    override matching built-in profiles.  Built-in profiles not overridden
    are kept as-is.
    """
    result = dict(PROFILES)  # shallow copy of built-in defaults
    for name, data in tool_profiles.items():
        result[name] = ToolProfile.from_dict(name, data)
    return result


# ---------------------------------------------------------------------------
# File / shell tools that can be stripped from subagent profiles
# ---------------------------------------------------------------------------

FILE_SHELL_TOOLS: frozenset[str] = frozenset({
    "bash", "read_file", "edit_file",
})
"""Tools stripped from subagent profiles when ``--subagent-file-tools`` is off.

``python_execute`` is intentionally excluded — it is useful for computation
and data processing even without filesystem access.
"""

# ---------------------------------------------------------------------------
# BBS channel mapping
# ---------------------------------------------------------------------------

CORE_CHANNELS: frozenset[str] = frozenset({
    "tasks", "discussion", "consensus", "discoveries", "key-findings",
})
"""BBS channels that are always present regardless of active profiles."""


def channels_for_profiles(
    profile_names: list[str],
    tool_profiles: dict[str, Any] | None = None,
) -> frozenset[str]:
    """Compute the active BBS channels given a set of available profiles.

    Returns the core set plus any profile-specific channels introduced by
    the given profiles.
    """
    profiles = load_profiles_from_config(tool_profiles)
    channels = set(CORE_CHANNELS)
    for name in profile_names:
        p = profiles.get(name)
        if p and p.bbs_channels:
            channels |= p.bbs_channels
    return frozenset(channels)


def get_profile(name: str) -> ToolProfile | None:
    """Look up a profile by name.  Returns ``None`` if not found."""
    return PROFILES.get(name)


def available_profile_names() -> list[str]:
    """Return sorted list of registered profile names."""
    return sorted(PROFILES.keys())


def resolve_orchestrator_skill(
    *,
    has_bbs: bool,
    has_web_search: bool = False,
    orchestrator_realtime: bool = False,
    skill_overrides: dict[str, str] | None = None,
) -> str:
    """Return the orchestration skill name for the given swarm configuration.

    Subagents are always spawned dynamically (on demand), so the dynamic
    orchestration skills are selected here.

    ``skill_overrides`` (``{original_name: variant_name}``) remaps the
    resolved name — used by ablation arms to swap in a gate-stripped SKILL.md
    variant. Empty/None = baseline behavior.
    """
    if not has_bbs:
        name = "swarm-orchestration-dynamic-dm"
    elif has_web_search:
        name = "swarm-orchestration-dynamic-web"
    else:
        name = "swarm-orchestration-dynamic"
    if skill_overrides:
        name = skill_overrides.get(name, name)
    return name


def resolve_profile_skills(
    domain_skills: tuple[str, ...],
    profile_tools: frozenset[str],
    *,
    has_bbs: bool,
    has_dm: bool,
    is_duo: bool = False,
    registry: "SkillRegistry | None" = None,
    skill_overrides: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Compose the full skill list for a subagent profile.

    Prepends the appropriate coordination and task-completion skills
    (determined by comm mode) to the profile's domain skills.  YAML
    profiles only need to declare domain skills; this function adds
    the comm-layer skills automatically.

    The web variant (``bbs-coordination-web``, ``task-completion-web``)
    is selected when ``web_search`` is in *profile_tools*.

    Domain skills pass through unchanged — comm-topology-specific
    variants (e.g. leader vs auditor) belong in the comm skill, not
    in a domain-skill swap.
    """
    is_web = "web_search" in profile_tools
    result: list[str] = []

    if is_duo:
        result.append("duo-coordination")
    elif has_bbs:
        result.append("bbs-coordination-web")
    elif has_dm:
        result.append("dm-coordination")

    if is_web:
        # In DM/Duo mode there is no BBS — the original task-completion-web
        # skill instructs agents to "post results to the appropriate BBS
        # channel", which is misleading and provokes calls to a tool
        # (post_to_bbs) that is not registered. The -dm-duo variant
        # rewrites that step to use complete_task summaries and targeted
        # send_message DMs instead.
        if not has_bbs and (has_dm or is_duo):
            result.append("task-completion-web-dm-duo")
        else:
            result.append("task-completion-web")
    else:
        result.append("task-completion")

    for skill in domain_skills:
        if not has_bbs and (has_dm or is_duo) and skill == "tool-usage-policy-browsing":
            # Same rationale as task-completion-web above: the original
            # browsing policy ends with a "Post web search findings to
            # `#discoveries`" instruction that is invalid in DM/Duo mode.
            result.append("tool-usage-policy-browsing-dm-duo")
        elif not has_bbs and (has_dm or is_duo) and skill == "task-completion-web":
            # The default browsing profile lists task-completion-web as a
            # domain skill (profiles.py PROFILE_BROWSING.skill_names), so
            # without this swap the BBS-bearing original would be loaded
            # alongside the auto-injected -dm-duo variant.
            result.append("task-completion-web-dm-duo")
        else:
            result.append(skill)

    # Dedup while preserving order (auto-injection above and the domain
    # loop can both produce the same skill name when a YAML happens to
    # list a skill that is already injected).
    result = list(dict.fromkeys(result))

    # Ablation skill remap: swap in gate-stripped SKILL.md variants BEFORE
    # the registry filter so the variant name is what gets validated/loaded.
    if skill_overrides:
        result = [skill_overrides.get(s, s) for s in result]

    if registry is not None:
        result = [s for s in result if registry.get(s) is not None]
    return tuple(result)
