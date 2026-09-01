"""SubAgent — persistent pool member that claims tasks from the board.

Each SubAgent is a long-running thread that:
1. Polls the TaskBoard for claimable tasks.
2. Claims a task, executes it via an LLM conversation, posts results to BBS.
3. Returns to idle and optionally verifies other agents' BBS posts.
4. Loops until signalled to shut down.

A shared ``SnowflakeClient`` is reused across all SubAgents (it is already
connection-pooled and thread-safe).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from arcticswarm.agent import Agent, StreamEvent, TokenUsage, ToolCallEnd, ToolCallStart, TurnComplete
from arcticswarm.config import ArcticswarmConfig
from arcticswarm.snowflake_client import SnowflakeClient
from arcticswarm.swarm.bbs import (
    ALL_CHANNELS,
    BBS,
    CHANNEL_CONSENSUS,
    CHANNEL_DISCOVERIES,
    CHANNEL_KEY_FINDINGS,
)
from arcticswarm.swarm.mailbox import (
    DM_LANE_CONTROL,
    DM_TYPE_IDLE_NOTIFICATION,
    DM_TYPE_TASK_FAILED,
    Mailbox,
)
from arcticswarm.swarm.prompts import (
    IDLE_REVIEW_MESSAGE_RESEARCH,
    IDLE_REVIEW_MESSAGE_RESEARCH_ADVERSARIAL,
    PROFILE_SYSTEM_PROMPTS,
    build_comm_protocol_inline,
    build_skill_recommendations,
    get_profile_task_prompt,
)
from arcticswarm.swarm.profiles import (
    DEFAULT_PROFILE_NAME,
    get_profile,
    resolve_profile_skills,
)
from arcticswarm.swarm.task import AgentRegistry, AgentStatus, TaskBoard, TaskSpec, TaskStatus
from arcticswarm.swarm.tools import (
    CompleteTaskTool,
    ListTasksTool,
    PostToBBSTool,
    ReadBBSTool,
    ReadDMTool,
    SendMessageTool,
    UpdateTaskSummaryTool,
)

logger = logging.getLogger(__name__)

# How often idle subagents poll for new tasks (seconds)
_POLL_INTERVAL = 2.0

# Minimum seconds between idle BBS review LLM calls to avoid burning tokens
_IDLE_CHECK_COOLDOWN = 5.0

# Maximum consecutive idle BBS reviews triggered by #discussion-only traffic.
# Without this cap, idle agents can enter an infinite feedback loop — Agent A
# posts to #discussion, Agent B reviews and posts, Agent A sees B's post, etc.
# The counter resets automatically when new *result* posts appear (discoveries,
# consensus, key-findings) because those represent genuine new findings.
_MAX_CONSECUTIVE_IDLE_REVIEWS = 3

# Maximum consecutive idle DM handles before the agent stops responding to DMs.
# Mirrors _MAX_CONSECUTIVE_IDLE_REVIEWS but for DM-only cascades.  Resets when
# the agent claims a task.
_MAX_CONSECUTIVE_DM_HANDLES = 2

# Channels whose new messages reset the idle-review counter — these carry real
# findings from completed tasks, not back-and-forth chatter.
_RESULT_CHANNELS = frozenset({
    CHANNEL_DISCOVERIES, CHANNEL_CONSENSUS, CHANNEL_KEY_FINDINGS,
})

# Global counter to limit subagent prompt logging to first N executions
_subagent_prompt_log_count = 0
_subagent_prompt_log_lock = threading.Lock()


# ---------------------------------------------------------------------------
# One-shot tool wrapper (used to cap reasoning calls in idle review)
# ---------------------------------------------------------------------------


class _CappedToolWrapper:
    """Wraps a tool so it can only be called a limited number of times.

    On the first N calls, delegates to the inner tool.  On subsequent calls,
    returns an error telling the agent the tool is exhausted.  This
    prevents idle reviewers from calling the expensive reasoning tool
    repeatedly in a review spiral.
    """

    def __init__(self, inner: Any, max_calls: int = 3) -> None:
        self._inner = inner
        self._call_count = 0
        self._max_calls = max_calls
        # Forward attributes the agent framework expects
        self.name = inner.name
        self.description = inner.description

    def parameters_schema(self) -> dict[str, Any]:
        return self._inner.parameters_schema()

    def to_anthropic_tool(self) -> dict[str, Any]:
        return self._inner.to_anthropic_tool()

    def to_openai_tool(self) -> dict[str, Any]:
        return self._inner.to_openai_tool()

    def execute(self, **kwargs: Any) -> Any:
        if self._call_count >= self._max_calls:
            from arcticswarm.tools.base import ToolResult
            return ToolResult(
                error=(
                    f"The reasoning tool can only be used {self._max_calls} times per review cycle. "
                    f"You have used all {self._max_calls} calls. Share any remaining review feedback "
                    "through the coordination tools available in this mode, and say "
                    "'Already reviewed — nothing new to add' if there is nothing else to do."
                ),
                is_error=True,
            )
        self._call_count += 1
        return self._inner.execute(**kwargs)


# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------


class _TimingCollector:
    """Intercepts stream events to record per-turn LLM and per-tool timing.

    - ``llm_durations``: list of seconds for each LLM response (one per
      assistant turn).  The duration covers the time from the start of
      the turn (or end of the previous tool call) to the first
      ``ToolCallStart`` (or ``TurnComplete`` if no tools).
    - ``tool_durations``: dict mapping ``tool_use_id`` to execution
      seconds for each tool call.

    When constructed with ``task_board`` + ``task_id`` the collector also
    emits a progress heartbeat on every ``ToolCallStart`` so the
    orchestrator's ``list_tasks`` / ``wait_for_tasks`` output can show live
    subagent activity instead of a static "running" label (Claude Code
    ``ProgressTracker`` pattern).
    """

    def __init__(
        self,
        inner_on_event: Any = None,
        *,
        task_board: Any = None,
        task_id: str = "",
        agent: Any = None,
    ) -> None:
        self._inner = inner_on_event
        self._llm_start: float = 0.0
        self._tool_start: float = 0.0
        self.llm_durations: list[float] = []
        self.tool_durations: dict[str, float] = {}
        self.tool_names: dict[str, str] = {}
        self._task_board = task_board
        self._task_id = task_id
        self._agent = agent

    def start(self) -> None:
        """Mark the beginning of a new ``run_turn_streaming`` call."""
        self._llm_start = time.monotonic()

    def on_event(self, event: StreamEvent) -> None:
        now = time.monotonic()
        if isinstance(event, ToolCallStart):
            if self._llm_start > 0:
                self.llm_durations.append(round(now - self._llm_start, 2))
                self._llm_start = 0.0
            self._tool_start = now
            self.tool_names[event.tool_use_id] = event.tool_name
            if self._task_board is not None and self._task_id:
                try:
                    preview = _summarize_tool_input(
                        event.tool_name, event.tool_input,
                    )
                    tokens = 0
                    if self._agent is not None:
                        usage = getattr(self._agent, "last_turn_usage", None)
                        if usage is not None:
                            tokens = getattr(usage, "total_tokens", 0) or 0
                    self._task_board.bump_progress(
                        self._task_id,
                        tool_name=event.tool_name,
                        input_preview=preview,
                        tokens=tokens,
                    )
                except Exception:
                    # Progress bookkeeping must never break a tool call.
                    pass
        elif isinstance(event, ToolCallEnd):
            self.tool_durations[event.tool_use_id] = round(
                now - self._tool_start, 2,
            )
            if event.tool_use_id not in self.tool_names:
                self.tool_names[event.tool_use_id] = event.tool_name
            self._llm_start = now
        elif isinstance(event, TurnComplete):
            if self._llm_start > 0:
                self.llm_durations.append(round(now - self._llm_start, 2))
        if self._inner:
            self._inner(event)

    def latency_breakdown(self) -> dict[str, float]:
        """Aggregate timing into a breakdown by category.

        Returns a dict with keys ``llm_planning``, per-tool-name entries, and
        ``overhead`` (computed as ``total - llm - tools`` when *total* is
        supplied later by the caller).
        """
        breakdown: dict[str, float] = {"llm_planning": round(sum(self.llm_durations), 2)}
        for uid, dur in self.tool_durations.items():
            name = self.tool_names.get(uid, "unknown_tool")
            breakdown[name] = round(breakdown.get(name, 0.0) + dur, 2)
        return breakdown


def _summarize_tool_input(tool_name: str, tool_input: Any) -> str:
    """Return a short (<= 120 char) preview of a tool call for progress render.

    Prefers common "meaningful" fields (``url``, ``query``, ``code``,
    ``task_name``, ``prompt``, ``content``) before falling back to
    the first string arg.  Shortened and stripped so
    ``list_tasks`` output stays one line per activity.
    """
    if not isinstance(tool_input, dict):
        return str(tool_input)[:120]
    for key in (
        "url", "urls", "query", "search_query", "q",
        "code",
        "task_name", "task_id", "prompt",
        "recipient", "to_agent", "content",
    ):
        val = tool_input.get(key)
        if isinstance(val, str) and val:
            cleaned = " ".join(val.split())
            return cleaned[:120]
        if isinstance(val, list) and val:
            cleaned = ", ".join(str(v) for v in val[:3])
            return cleaned[:120]
    # Fallback: first non-empty string value.
    for val in tool_input.values():
        if isinstance(val, str) and val:
            return " ".join(val.split())[:120]
    return ""


def _inject_timings_into_messages(
    messages: list[dict[str, Any]],
    collector: _TimingCollector,
    msg_start_idx: int,
) -> None:
    """Inject ``_llm_duration_seconds`` / ``_tool_duration_seconds`` in-place.

    Walks messages added since *msg_start_idx* and annotates:
    - Each ``assistant`` message with ``_llm_duration_seconds``.
    - Each ``tool_result`` block with ``_tool_duration_seconds``.
    """
    llm_idx = 0
    durations = collector.llm_durations
    for msg in messages[msg_start_idx:]:
        if msg["role"] == "assistant":
            if llm_idx >= len(durations):
                continue
            n_tools = 0
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        n_tools += 1
            n_entries = max(n_tools, 1)
            end_idx = min(llm_idx + n_entries, len(durations))
            msg["_llm_duration_seconds"] = round(
                sum(durations[llm_idx:end_idx]), 2,
            )
            llm_idx = end_idx
        elif msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        uid = block.get("tool_use_id", "")
                        if uid in collector.tool_durations:
                            block["_tool_duration_seconds"] = collector.tool_durations[uid]


class SubAgent:
    """Persistent subagent that runs a task-claiming loop in a thread.

    Parameters
    ----------
    name:
        Human name for this subagent (e.g. "Alice", "Bob").
    config:
        Arcticswarm configuration (API key, model, Snowflake connection, etc.).
    bbs:
        Shared BBS instance.
    task_board:
        Shared TaskBoard instance.
    agent_registry:
        Shared registry for tracking subagent status/activity.
    question:
        The user's original question (provided as context for each task).
    shutdown:
        Threading event — when set, the subagent exits its loop.
    sf_client:
        Shared SnowflakeClient.  Reused to avoid creating duplicate
        connections.
    on_event:
        Optional callback for streaming events (tool calls, text deltas).
    on_status_change:
        Optional callback fired when the subagent's status changes.
        Signature: ``(name, status_str, activity_str) -> None``.
    role:
        Optional free-text role description, injected as ``## Your Role`` in the system prompt.
    is_duo:
        When True, use duo coordination protocol and skill adjustments for two-agent runs.
    """

    def __init__(
        self,
        name: str,
        config: ArcticswarmConfig,
        bbs: BBS | None,
        task_board: TaskBoard,
        agent_registry: AgentRegistry,
        question: str,
        shutdown: threading.Event,
        sf_client: SnowflakeClient | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,
        on_status_change: Callable[[str, str, str], None] | None = None,
        web_source_tracker: Any | None = None,
        active_channels: frozenset[str] | None = None,
        mailbox: Mailbox | None = None,
        has_bbs: bool = True,
        has_dm: bool = False,
        system_reminder_interval: int = -1,
        dynamic_mode: bool = False,
        initial_profile: str = "",
        on_task_complete: Callable[[str], None] | None = None,
        role: str = "",
        is_duo: bool = False,
        content_cache: Any | None = None,
        question_images: list[dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.bbs = bbs
        self.task_board = task_board
        self.agent_registry = agent_registry
        self.question = question
        # Image content blocks from the original user question. When
        # non-empty, ``execute_task`` will prepend them to the first user
        # message so subagents see attached images (multimodal cases).
        self.question_images: list[dict[str, Any]] = list(question_images or [])
        self._shutdown = shutdown
        self._on_event = on_event
        self._on_status_change = on_status_change
        self._owns_sf_client = False
        self._active_channels = active_channels or ALL_CHANNELS
        self._last_idle_check: float = 0.0
        self._consecutive_idle_reviews: int = 0
        self._consecutive_dm_handles: int = 0
        self._lifetime_idle_reviews: int = 0
        self._last_result_msg_count: int = 0
        self._last_task_count: int = 0
        self._idle_review_concluded: bool = False
        self._mailbox = mailbox
        self._has_bbs = has_bbs
        self._has_dm = has_dm
        self._is_duo = is_duo
        self._role = role
        self._registered_skill_tool_names: set[str] = set()
        self._status: AgentStatus | None = None
        self._prev_status: AgentStatus | None = None

        # Dynamic-mode fields
        self._dynamic_mode = dynamic_mode
        self._pending_task: TaskSpec | None = None
        self._task_event = threading.Event()
        self._tasks_completed: int = 0
        self._initial_profile = initial_profile
        self._agent_shutdown = threading.Event()
        self._on_task_complete = on_task_complete
        self._spawn_time: float = time.monotonic()

        # Effective default profile: prefer the first YAML-declared profile
        # so subagents use the correct tool set.  Fall back to the legacy
        # "browsing" heuristic only when no custom profiles exist.
        if config.swarm_profiles:
            self._effective_default = config.swarm_profiles[0]
        elif config.has_web_search_capability():
            self._effective_default = "browsing"
        else:
            self._effective_default = DEFAULT_PROFILE_NAME
        self._restore_profile_target = initial_profile if dynamic_mode else self._effective_default

        # Cumulative token usage across all LLM turns (task execution + idle).
        self.token_usage: TokenUsage = TokenUsage()
        self.total_num_steps: int = 0

        # Archive of messages before context resets — so trajectory capture
        # still has the full conversation even after clear_history().
        self._message_archive: list[dict[str, Any]] = []

        # Reflection statistics (browsing tasks only)
        self.reflection_calls: int = 0
        self.reflection_sufficient: int = 0  # ended with is_sufficient=True
        self.reflection_insufficient: int = 0  # exhausted loops without sufficient
        self.reflection_confidence_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        self.reflection_total_gaps: int = 0
        self.reflection_total_queries: int = 0

        # Create the underlying Agent (marked as swarm subagent to exclude GAIA FINAL ANSWER format).
        self.agent = Agent(
            config,
            is_swarm_subagent=True,
        )

        # Set the web source tracker for capturing web_search results
        self.agent.web_source_tracker = web_source_tracker

        # Set the shared content cache for web_fetch/pdf_read deduplication
        self.agent.content_cache = content_cache

        # Share the SnowflakeClient if one was provided
        if sf_client is not None and self.agent.sf_client is not sf_client:
            if self.agent.sf_client is not None:
                try:
                    self.agent.sf_client.close()
                except Exception:
                    pass
            self.agent.sf_client = sf_client
            self.agent._register_tools()
        else:
            self._owns_sf_client = True
            # Re-register tools if content_cache was set (Agent.__init__ ran
            # _register_tools before content_cache was assigned).
            if content_cache is not None:
                self.agent._register_tools()

        # -- Snapshot task tools BEFORE stripping anything ----------------------
        self._all_task_tools: dict[str, Any] = dict(self.agent._tools)

        # Strip file/shell tools from the legacy pool.
        # YAML profiles declare exact tool sets — stripping is done in _apply_profile.
        if not self.config.tool_profiles:
            from arcticswarm.swarm.profiles import FILE_SHELL_TOOLS
            for tool_name in FILE_SHELL_TOOLS:
                self._all_task_tools.pop(tool_name, None)
                self.agent._tools.pop(tool_name, None)

        # Register BBS tools (conditional on has_bbs)
        self._read_bbs_tool: ReadBBSTool | None = None
        if has_bbs and bbs is not None:
            self.agent._tools["post_to_bbs"] = PostToBBSTool(
                bbs, author=name, channels=self._active_channels,
                has_dm=has_dm, is_web=config.has_web_search_capability(),
            )
            self._read_bbs_tool = ReadBBSTool(bbs, channels=self._active_channels)
            self._read_bbs_tool.initialize_cursor()
            self.agent._tools["read_bbs"] = self._read_bbs_tool
            self.agent._auto_bbs_check = self._read_bbs_tool.check_new_messages

        # Register task board tools (always present)
        _broadcast = getattr(config, "submit_findings_broadcast", False)
        _mb = mailbox if has_dm else None
        self.agent._tools["complete_task"] = CompleteTaskTool(
            task_board,
            profile=initial_profile,
            mailbox=_mb,
            sender=name,
            broadcast=_broadcast,
        )
        self.agent._tools["list_tasks"] = ListTasksTool(task_board)
        self.agent._tools["update_task_summary"] = UpdateTaskSummaryTool(
            task_board,
            author=name,
            mailbox=_mb,
            sender=name,
            broadcast=_broadcast,
        )
        self._all_task_tools["update_task_summary"] = self.agent._tools["update_task_summary"]

        # Register DM tools (conditional on has_dm)
        self._read_dm_tool: ReadDMTool | None = None
        if has_dm and mailbox is not None:
            other_names = [n for n in mailbox.registered_names if n != name]
            self.agent._tools["send_message"] = SendMessageTool(
                mailbox,
                sender=name,
                agent_names=other_names,
                has_bbs=has_bbs,
                peer_dm_summary=getattr(config, "peer_dm_summary", False),
            )
            self._read_dm_tool = ReadDMTool(mailbox, agent_name=name)
            self.agent._tools["read_dm"] = self._read_dm_tool
            self.agent._auto_dm_check = lambda: self._check_dms()

        from arcticswarm.swarm.profiles import load_profiles_from_config
        _profiles = load_profiles_from_config(self.config.tool_profiles)
        _default_profile = _profiles.get(self._effective_default) or get_profile(self._effective_default)

        # -- Overwrite skill tools with swarm-aware skill list ------------------
        _has_skill_tools = "load_skill" in self.agent._tools or self.config.per_skill_tools
        if _has_skill_tools:
            from arcticswarm.tools.skill_tools import LoadSkillTool, make_per_skill_tools as _make_ps
            from arcticswarm.skill_loader import SkillRegistry
            from pathlib import Path
            skills_dir = Path(__file__).resolve().parent.parent / "skills"
            registry = SkillRegistry(skills_dir=skills_dir)
            adjusted = resolve_profile_skills(
                _default_profile.skill_names if _default_profile else (),
                _default_profile.included_tools if _default_profile else frozenset(),
                has_bbs=has_bbs, has_dm=has_dm, is_duo=is_duo,
                registry=registry,
                skill_overrides=getattr(self.config, "skill_overrides", None),
            )
            if self.config.per_skill_tools:
                from arcticswarm.tools.skill_tools import PerSkillTool as _PST
                self.agent._tools.pop("load_skill", None)
                for k in list(self.agent._tools):
                    if isinstance(self.agent._tools[k], _PST):
                        del self.agent._tools[k]
                per_tools = _make_ps(
                    list(adjusted),
                    registry=registry,
                    legacy_format=self.config.skill_legacy_format,
                )
                self.agent._tools.update(per_tools)
                for tool_name, tool in per_tools.items():
                    self._all_task_tools[tool_name] = tool
                self._registered_skill_tool_names = set(per_tools.keys())
            else:
                self.agent._tools["load_skill"] = LoadSkillTool(
                    list(adjusted),
                    registry=registry,
                    legacy_format=self.config.skill_legacy_format,
                )
                self._all_task_tools["load_skill"] = self.agent._tools["load_skill"]

        # -- System-reminder injection (periodic skill listing) ----------------
        if system_reminder_interval >= 1 and _default_profile is not None:
            from arcticswarm.skill_loader import build_system_reminder
            reminder_skills = list(resolve_profile_skills(
                _default_profile.skill_names, _default_profile.included_tools,
                has_bbs=has_bbs, has_dm=has_dm, is_duo=is_duo,
                skill_overrides=getattr(self.config, "skill_overrides", None),
            ))
            reminder_text = build_system_reminder(reminder_skills)
            _counter = [0]

            def _reminder_callback() -> str | None:
                _counter[0] += 1
                if _counter[0] % system_reminder_interval == 0:
                    return reminder_text
                return None

            self.agent._auto_system_reminder = _reminder_callback

        # -- Build per-profile system prompts (cached for reuse) ---------------
        from datetime import date, datetime
        if config.date_override:
            try:
                current_date = datetime.strptime(config.date_override, "%Y-%m-%d").date()
            except ValueError:
                current_date = date.today()
        else:
            current_date = date.today()

        base_prompt = self.agent.system_prompt
        role_prefix = f"\n## Your Role\n\n{role}\n\n" if role else ""
        self._profile_prompts: dict[str, str] = {}
        for profile_name, prompt_template in PROFILE_SYSTEM_PROMPTS.items():
            _prompt_profile = _profiles.get(profile_name) or get_profile(profile_name)
            _prompt_skills = _prompt_profile.skill_names if _prompt_profile else ()
            _skill_overrides = getattr(config, "skill_overrides", None)
            _disable_idle_review = (
                getattr(config, "disable_auditor", False)
                or getattr(config, "disable_builder_idle", False)
            )
            comm_protocol = build_comm_protocol_inline(
                has_bbs,
                has_dm,
                per_skill_tools=config.per_skill_tools,
                is_duo=is_duo,
                profile_name=profile_name,
                disable_idle_review=_disable_idle_review,
                skill_overrides=_skill_overrides,
            )
            self._profile_prompts[profile_name] = (
                base_prompt + role_prefix + prompt_template.format(
                    agent_name=name,
                    current_date=current_date.isoformat(),
                    skill_recommendations=build_skill_recommendations(
                        profile_name,
                        skill_names=_prompt_skills,
                        per_skill_tools=config.per_skill_tools,
                        skill_overrides=_skill_overrides,
                    ),
                    comm_protocol=comm_protocol,
                )
            )
            # anti-anchoring: append the reframe block to the browsing
            # agent prompt only, gated by config.reframe_prompt (default off, so
            # other models/runs are unaffected). See prompts.ANTI_ANCHOR_BLOCK.
            if profile_name == "browsing" and getattr(config, "reframe_prompt", False):
                from arcticswarm.swarm.prompts import ANTI_ANCHOR_BLOCK
                self._profile_prompts[profile_name] += ANTI_ANCHOR_BLOCK

        # Set default profile
        self._current_profile_name: str = self._effective_default
        default_prompt_key = self._effective_default
        if default_prompt_key in self._profile_prompts:
            self._base_system_prompt = self._profile_prompts[default_prompt_key]
        else:
            self._base_system_prompt = base_prompt + role_prefix
        self.agent.system_prompt = self._base_system_prompt

        if self.config.tool_profiles:
            self._apply_profile(self._effective_default)

    # -- profile swapping ----------------------------------------------------

    def _apply_profile(self, profile_name: str) -> None:
        """Swap tools and system prompt for *profile_name*.

        **Declarative path** (``config.tool_profiles`` non-empty and profile
        found there): build the exact tool set fresh via ``ToolFactory``.
        The YAML profile is the single source of truth — no inheritance
        from the agent pool, no FILE_SHELL_TOOLS stripping.

        **Legacy path** (no YAML profiles): filter from ``_all_task_tools``
        using built-in profile whitelists, lazily instantiate missing tools,
        and strip FILE_SHELL_TOOLS.

        Both paths then re-add swarm infrastructure tools and update skills.
        """
        from arcticswarm.swarm.profiles import load_profiles_from_config

        profiles = load_profiles_from_config(self.config.tool_profiles)
        profile = profiles.get(profile_name)
        if profile is None:
            profile = get_profile(profile_name)
        if profile is None:
            logger.warning(
                "Unknown profile %r — falling back to default %r",
                profile_name, self._effective_default,
            )
            profile = profiles.get(self._effective_default) or get_profile(self._effective_default)
            profile_name = self._effective_default
            if profile is None:
                return

        is_yaml_profile = bool(self.config.tool_profiles) and profile_name in self.config.tool_profiles

        if is_yaml_profile:
            # Declarative path — YAML profile declares the exact tool set.
            from arcticswarm.tools.factory import ToolFactory
            factory = ToolFactory(
                self.config,
                sf_client=self.agent.sf_client,
                agent_client=self.agent.client,
                content_cache=self.agent.content_cache,
            )
            tool_names = [t for t in profile.included_tools if t != "load_skill"]
            filtered = factory.build(tool_names)
        else:
            # Legacy path — filter from agent pool + lazy instantiation.
            if profile.included_tools:
                filtered = {
                    k: v for k, v in self._all_task_tools.items()
                    if k in profile.included_tools
                }
                missing = profile.included_tools - filtered.keys()
                if missing:
                    for name, tool in self._create_missing_tools(missing).items():
                        filtered[name] = tool
            else:
                filtered = dict(self._all_task_tools)

            if profile.excluded_tools:
                for tool_name in profile.excluded_tools:
                    filtered.pop(tool_name, None)

            from arcticswarm.swarm.profiles import FILE_SHELL_TOOLS
            for tool_name in FILE_SHELL_TOOLS:
                filtered.pop(tool_name, None)

        if getattr(self.config, "no_web_fetch", False):
            filtered.pop("web_fetch", None)

        # Re-add swarm infrastructure tools (always present)
        swarm_tool_names = [
            "post_to_bbs", "read_bbs", "complete_task",
            "list_tasks", "send_message", "read_dm",
            "update_task_summary",
        ]
        swarm_tools = {
            k: self.agent._tools.get(k) for k in swarm_tool_names
        }
        filtered.update({k: v for k, v in swarm_tools.items() if v is not None})

        # Update skill tools with profile-specific skills (adjusted for DM / duo)
        _has_any_skill = (
            "load_skill" in self._all_task_tools or bool(self._registered_skill_tool_names)
        )
        if _has_any_skill:
            from arcticswarm.tools.skill_tools import LoadSkillTool, ReadSkillFileTool
            from arcticswarm.skill_loader import SkillRegistry
            from pathlib import Path
            skills_dir = Path(__file__).resolve().parent.parent / "skills"
            registry = SkillRegistry(skills_dir=skills_dir)
            domain_skills = profile.skill_names or get_profile(self._effective_default).skill_names

            # Conditionally add vision guidance skill when vision is enabled
            # and the profile supports image files (coding, browsing).
            if (
                self.config.enable_vision
                and profile.name in ("coding", "browsing")
                and "vision-guidance" not in domain_skills
            ):
                domain_skills = tuple(domain_skills) + ("vision-guidance",)

            adjusted = resolve_profile_skills(
                domain_skills, profile.included_tools,
                has_bbs=self._has_bbs, has_dm=self._has_dm,
                is_duo=self._is_duo,
                registry=registry,
                skill_overrides=getattr(self.config, "skill_overrides", None),
            )
            if self.config.per_skill_tools:
                from arcticswarm.tools.skill_tools import make_per_skill_tools as _make_ps
                for old_name in self._registered_skill_tool_names:
                    filtered.pop(old_name, None)
                new_tools = _make_ps(
                    list(adjusted),
                    registry=registry,
                    legacy_format=self.config.skill_legacy_format,
                )
                filtered.update(new_tools)
                self._registered_skill_tool_names = set(new_tools.keys())
            else:
                filtered["load_skill"] = LoadSkillTool(
                    list(adjusted),
                    registry=registry,
                    legacy_format=self.config.skill_legacy_format,
                )
            filtered["read_skill_file"] = ReadSkillFileTool(registry=registry)

        self.agent._tools = filtered

        # Swap system prompt
        if profile_name in self._profile_prompts:
            self.agent.system_prompt = self._profile_prompts[profile_name]

        self._current_profile_name = profile_name
        logger.debug(
            "SubAgent %s applied profile %r — tools: %s",
            self.name, profile_name, sorted(filtered.keys()),
        )

    def _restore_default_profile(self) -> None:
        """Restore the default profile after task completion."""
        self._apply_profile(self._restore_profile_target)

    def _create_missing_tools(self, names: set[str]) -> dict[str, Any]:
        """Lazily instantiate tools that a profile needs but weren't registered.

        Uses :class:`ToolFactory` to construct tools on demand. Created
        tools are cached in ``_all_task_tools`` for reuse across profile
        switches.
        """
        from arcticswarm.tools.factory import ToolFactory

        factory = ToolFactory(
            self.config,
            sf_client=self.agent.sf_client,
            agent_client=self.agent.client,
            content_cache=self.agent.content_cache,
        )

        created: dict[str, Any] = {}
        for name in names:
            if name in self._all_task_tools:
                continue
            tool = factory.make(name)
            if tool is not None:
                created[tool.name] = tool
                self._all_task_tools[tool.name] = tool

        if created:
            logger.debug(
                "SubAgent %s lazily created tools: %s",
                self.name, sorted(created.keys()),
            )
        return created

    @property
    def all_messages(self) -> list:
        """Full message history including messages from before context resets."""
        return self._message_archive + list(self.agent.messages)

    # -- main loop -----------------------------------------------------------

    def run_loop(self) -> None:
        """Main loop — poll for tasks, execute, idle-check, repeat.

        Designed to run in a ``ThreadPoolExecutor`` worker thread.
        """
        logger.info("SubAgent %s started", self.name)
        self._set_status(AgentStatus.IDLE, "ready")

        while not self._shutdown.is_set():
            # 1. Try to claim a ready task
            task = self._try_claim_next_task()
            if task is not None:
                task_desc = task.prompt[:120] + "..." if len(task.prompt) > 120 else task.prompt
                self._set_status(AgentStatus.WORKING, f"{task.name}: {task_desc}")
                try:
                    self._execute_task(task)
                except Exception as exc:
                    logger.error("SubAgent %s failed task %s: %s", self.name, task.name, exc)
                    self.task_board.fail(task.id, error=str(exc))
                    if self.bbs is not None:
                        self.bbs.post(
                            channel="discussion",
                            author=self.name,
                            content=f"FAILED task '{task.name}': {exc}",
                            structured_data={"error": str(exc)},
                            tags=["error"],
                        )
                    elif self._has_dm and self._mailbox is not None:
                        fail_msg = f"FAILED task '{task.name}': {exc}"
                        for recipient in self._mailbox.registered_names:
                            if recipient != self.name:
                                try:
                                    self._mailbox.send(
                                        from_agent=self.name,
                                        to_agent=recipient,
                                        content=fail_msg,
                                        lane=DM_LANE_CONTROL,
                                        message_type=DM_TYPE_TASK_FAILED,
                                        payload={
                                            "task_id": task.id,
                                            "task_name": task.name,
                                            "status": TaskStatus.FAILED.value,
                                            "failure_reason": str(exc),
                                        },
                                    )
                                except Exception:
                                    pass
                self._set_status(AgentStatus.IDLE, "ready")
                if getattr(self.config, "orchestrator_realtime", False):
                    completed_status = TaskStatus.COMPLETED.value
                    idle_reason = "available"
                    failure_reason = None
                    if task.status == TaskStatus.FAILED:
                        completed_status = TaskStatus.FAILED.value
                        idle_reason = "failed"
                        failure_reason = task.error or "task failed"
                    self._notify_leader_idle(
                        idle_reason=idle_reason,
                        completed_task_id=task.id,
                        completed_task_name=task.name,
                        completed_status=completed_status,
                        failure_reason=failure_reason,
                    )
                self._consecutive_idle_reviews = 0  # reset after real work
                self._consecutive_dm_handles = 0
                self._idle_review_concluded = False  # allow fresh reviews
                # Mirror the dynamic-mode loop — fire the
                # post-task hook so callers can run cleanup keyed to task
                # completion (e.g. dynamic-mode task queue management). The
                # callback receives the SubAgent name, not the task name,
                # because the same SubAgent may handle multiple tasks
                # over its lifetime and callers usually key off the
                # agent identity (workspace handle, mailbox, etc.).
                if self._on_task_complete is not None:
                    try:
                        self._on_task_complete(self.name)
                    except Exception:
                        # The callback is fire-and-forget — a failure
                        # inside the post-task hook must not crash the
                        # subagent's loop.
                        logger.exception(
                            "SubAgent %s: on_task_complete callback raised",
                            self.name,
                        )
                continue  # immediately check for more tasks

            # 2. No tasks available — idle
            self._idle_check()

            # 3. Wait before polling again (mailbox-aware if DM is enabled)
            if self._has_dm and self._mailbox is not None:
                got_dm = self._mailbox.wait_for_message(
                    self.name, timeout=_POLL_INTERVAL,
                )
                if got_dm and not self._shutdown.is_set():
                    self._handle_dm()
            else:
                self._shutdown.wait(timeout=_POLL_INTERVAL)

        logger.info("SubAgent %s shutting down", self.name)

    # -- dynamic-mode loop ---------------------------------------------------

    def run_loop_dynamic(self) -> None:
        """Event-driven loop for dynamic scaling mode.

        Waits on ``_task_event`` (set by :meth:`assign_task`) or mailbox.
        Exits when ``_shutdown`` is set.
        """
        logger.info("SubAgent %s started (dynamic mode)", self.name)
        self._set_status(AgentStatus.IDLE, "ready")

        while not self._shutdown.is_set() and not self._agent_shutdown.is_set():
            # Check for a pending task (set by assign_task)
            if self._pending_task is not None:
                task = self._pending_task
                self._pending_task = None
                self._task_event.clear()
                task_desc = task.prompt[:120] + "..." if len(task.prompt) > 120 else task.prompt
                self._set_status(AgentStatus.WORKING, f"{task.name}: {task_desc}")
                try:
                    self._execute_task(task)
                except Exception as exc:
                    logger.error("SubAgent %s failed task %s: %s", self.name, task.name, exc)
                    self.task_board.fail(task.id, error=str(exc))
                self._tasks_completed += 1
                self._set_status(AgentStatus.IDLE, "ready")
                if getattr(self.config, "orchestrator_realtime", False):
                    completed_status = TaskStatus.COMPLETED.value
                    idle_reason = "available"
                    failure_reason = None
                    if task.status == TaskStatus.FAILED:
                        completed_status = TaskStatus.FAILED.value
                        idle_reason = "failed"
                        failure_reason = task.error or "task failed"
                    self._notify_leader_idle(
                        idle_reason=idle_reason,
                        completed_task_id=task.id,
                        completed_task_name=task.name,
                        completed_status=completed_status,
                        failure_reason=failure_reason,
                    )
                self._consecutive_idle_reviews = 0
                self._consecutive_dm_handles = 0
                self._idle_review_concluded = False  # allow fresh reviews
                if self._on_task_complete is not None:
                    self._on_task_complete(self.name)
                continue

            # No pending task — check for DMs
            if self._has_dm and self._mailbox is not None:
                msgs = self._mailbox.check_new(self.name)
                if msgs:
                    self._handle_dm()
                    continue

            # Idle check (BBS review)
            self._idle_check()

            # Wait for a task assignment or DM signal
            self._task_event.wait(timeout=_POLL_INTERVAL)
            self._task_event.clear()

        logger.info("SubAgent %s shutting down (dynamic)", self.name)

    def assign_task(self, task: TaskSpec) -> None:
        """Assign a task to this dynamic-mode worker (called by SwarmContext)."""
        self._pending_task = task
        self.task_board.claim(task.id, self.name)
        self.task_board.mark_running(task.id)
        self._task_event.set()
        # Also signal the mailbox in case the agent is blocked on wait_for_message
        if self._has_dm and self._mailbox is not None:
            self._mailbox.signal(self.name)

    # -- task execution ------------------------------------------------------

    def _try_claim_next_task(self) -> TaskSpec | None:
        """Find and claim the first ready, unclaimed task.

        Tasks with ``assigned_to`` set are only claimable by the named
        agent.  Assigned-to-me tasks are tried first.
        """
        ready = self.task_board.ready_tasks()
        mine_first = sorted(ready, key=lambda t: t.assigned_to != self.name)
        for task in mine_first:
            if task.assigned_to and task.assigned_to != self.name:
                continue
            if self.task_board.claim(task.id, self.name):
                self.task_board.mark_running(task.id)
                self._consecutive_dm_handles = 0
                logger.info("SubAgent %s claimed task %s", self.name, task.name)
                return task
        return None

    def _execute_task(self, task: TaskSpec) -> None:
        """Run an LLM conversation to complete the claimed task.

        Conversation history is preserved across tasks so the agent
        retains context from previous work and BBS interactions.

        The task's ``profile`` field controls which tools and system
        prompt the subagent uses for this execution.  The profile is
        restored to default after completion (or failure).
        """
        # Apply the task's profile (swap tools + system prompt)
        profile_name = task.profile or self._effective_default
        self._apply_profile(profile_name)

        # Set source scoring context so auto-scoring knows the query
        self.agent.source_scoring_query = task.prompt[:500]
        self.agent._pending_sources.clear()

        # --- BBS isolation: suppress all read paths for isolated tasks ---
        # Isolation is either requested per task (metadata["isolated"], set by
        # the create_task tool) OR forced for browsing-profile EXPLORATION
        # executions via the force_bbs_isolation ablation. Enforcing force HERE
        # — the single point where reads are actually suppressed — covers
        # browsing tasks created outside the create_task tool too: the alt-task
        # / candidate-emergence contrarian sweeps spawn profile="browsing" tasks
        # directly (answer_verification.py, tools.py alt_task_gate), and their
        # prompts embed the leading candidate + question inline, so they run
        # correctly without BBS reads.
        # REVIEWER TASKS ARE EXEMPT: the builder-reviewer gate runs a
        # profile="browsing" task whose job is to re-verify the leading
        # candidate the team CONVERGED ON (read from #key-findings/#consensus);
        # it must keep BBS access so "a builder acting as reviewer still sees the
        # BBS". Reviewer tasks carry a reviewer_kind marker (tools.py reviewer
        # gate); the dedicated reviewer is profile="reasoning" and is skipped by
        # the profile check anyway. Scoped to browsing + restored per execution
        # in the finally block below. disable_bbs_isolation overrides and wins.
        is_isolated = task.metadata.get("isolated", False)
        if (
            self.config.force_bbs_isolation
            and profile_name == "browsing"
            and not task.metadata.get("reviewer_kind")
        ):
            is_isolated = True
        is_isolated = is_isolated and not self.config.disable_bbs_isolation
        _saved_auto_bbs = None
        _saved_read_bbs_tool = None
        if is_isolated and self._read_bbs_tool is not None:
            _saved_auto_bbs = self.agent._auto_bbs_check
            _saved_read_bbs_tool = self.agent._tools.pop("read_bbs", None)
            self.agent._auto_bbs_check = None
            logger.info(
                "SubAgent %s executing task %s in ISOLATED mode (BBS reads suppressed)",
                self.name, task.name,
            )

        # Build profile-specific user prompt
        if is_isolated:
            bbs_context = ""
        else:
            bbs_context = self.bbs.render_for_llm(limit=30) if self.bbs is not None else ""
        task_board_status = self.task_board.render_status()

        prompt_text = get_profile_task_prompt(
            profile_name=profile_name,
            task=task,
            question=self.question,
            bbs_context=bbs_context,
            task_board_status=task_board_status,
            has_bbs=self._has_bbs,
            has_dm=self._has_dm,
            no_web_fetch=getattr(self.config, "no_web_fetch", False),
            is_duo=self._is_duo,
            per_skill_tools=self.config.per_skill_tools,
            agent_name=self.name,
            disable_self_reflection=getattr(
                self.config, "disable_self_reflection", False
            ),
        )

        # Attach the original question's image blocks to the first user
        # message so the subagent sees them directly on turn 0. The image
        # blocks come first (Anthropic convention) and the profile task
        # text follows. When there are no images, the prompt stays a
        # plain string — matching the legacy behavior exactly.
        prompt: str | list[dict[str, Any]]
        if self.question_images:
            prompt = [
                *(dict(block) for block in self.question_images),
                {"type": "text", "text": prompt_text},
            ]
        else:
            prompt = prompt_text

        msg_start_idx = len(self.agent.messages)
        collector = _TimingCollector(
            inner_on_event=self._on_event,
            task_board=self.task_board,
            task_id=task.id,
            agent=self.agent,
        )
        collector.start()

        # Log subagent input prompt for first 5 tasks (for debugging)
        global _subagent_prompt_log_count
        with _subagent_prompt_log_lock:
            if _subagent_prompt_log_count < 5:
                _subagent_prompt_log_count += 1
                from pprint import pformat
                logger.info('='*100)
                logger.info(f"Subagent {self.name} [{profile_name}] input prompt example #{_subagent_prompt_log_count}")
                logger.info('='*100)
                # Don't dump base64 image payloads into logs — summarize
                # attachments and print only the text.
                if isinstance(prompt, list):
                    n_images = sum(
                        1 for b in prompt
                        if isinstance(b, dict) and b.get("type") == "image"
                    )
                    user_prompt_preview: Any = {
                        "attached_images": n_images,
                        "text": prompt_text,
                    }
                else:
                    user_prompt_preview = prompt
                stream_kwargs_preview = {
                    'model': self.config.model,
                    'profile': profile_name,
                    'system_prompt_length': len(self.agent.system_prompt),
                    'system_prompt_preview': self.agent.system_prompt[:500] + "..." if len(self.agent.system_prompt) > 500 else self.agent.system_prompt,
                    'user_prompt': user_prompt_preview,
                    'tool_count': len(self.agent._tools),
                    'available_tools': list(self.agent._tools.keys()),
                }
                logger.info(pformat(stream_kwargs_preview))
                logger.info('='*100)

        try:
            profile = get_profile(profile_name)
            use_reflection = (
                profile is not None
                and profile.supports_reflection
                and not self.config.disable_self_reflection
            )
            if use_reflection:
                response = self._execute_task_with_reflection(
                    task, prompt, collector,
                    msg_start_idx=msg_start_idx,
                )
            else:
                # Lever D: cap non-reflection subagents (e.g. auditor) too. 0 -> no-op.
                _nr_max = getattr(self.config, "subagent_max_turns", 0) or None
                response = self.agent.run_turn_streaming(prompt, on_event=collector.on_event, max_turns=_nr_max)
            self.token_usage += self.agent.last_turn_usage
            self.total_num_steps += self.agent.last_num_steps
            _inject_timings_into_messages(self.agent.messages, collector, msg_start_idx)
            # If the subagent's turn loop exhausted ``max_turns`` the
            # ``response`` text is just the partial working output plus a
            # "(Reached max turns: N)" marker — it is NOT a completion.
            # Mark the task FAILED with a clean diagnostic so the
            # orchestrator can spawn a retry instead of trusting a garbage
            # summary.  Mirrors Claude Code's distinct ``failAgentTask``
            # terminal transition.
            hit_max_turns = (
                getattr(self.agent, "last_stop_reason", "") == "max_turns"
            )
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                if hit_max_turns:
                    max_turns_limit = getattr(self.config, "max_turns", 0)
                    diagnostic = (
                        f"Subagent exhausted max_turns ({max_turns_limit}) "
                        f"before completing.  Partial work below.\n\n"
                        f"{response}"
                    )
                    failed_ok = self.task_board.fail(task.id, error=diagnostic)
                    if failed_ok:
                        logger.warning(
                            "SubAgent %s FAILED task %s (max_turns=%d) [profile=%s]",
                            self.name, task.name, max_turns_limit, profile_name,
                        )
                else:
                    self.task_board.complete(task.id, summary=response)
                    logger.info(
                        "SubAgent %s completed task %s [profile=%s]",
                        self.name, task.name, profile_name,
                    )
            else:
                logger.info(
                    "SubAgent %s finished run_turn for task %s "
                    "(already in terminal state %s)",
                    self.name, task.name, task.status.value,
                )
            self._checkpoint_and_reset(task)
        except Exception:
            self.token_usage += self.agent.last_turn_usage
            self.total_num_steps += self.agent.last_num_steps
            _inject_timings_into_messages(self.agent.messages, collector, msg_start_idx)
            raise
        finally:
            # Restore BBS access after isolated task
            if is_isolated:
                if _saved_auto_bbs is not None:
                    self.agent._auto_bbs_check = _saved_auto_bbs
                if _saved_read_bbs_tool is not None:
                    self.agent._tools["read_bbs"] = _saved_read_bbs_tool
            # Always restore default profile so the next task starts fresh
            self._restore_default_profile()

    def _checkpoint_and_reset(self, task: TaskSpec) -> None:
        """Reset agent context after task completion, if context is heavy.

        Findings are already posted to BBS by the agent during execution.
        The next task's prompt includes a fresh BBS snapshot (via
        ``bbs.render_for_llm``), so all prior findings remain accessible.

        We only reset when the agent has accumulated significant context
        (≥ 20 messages), because short tasks don't cause context pressure
        and preserving recent context can help with follow-up tasks.
        """
        if not getattr(self.config, "subagent_context_reset", False):
            logger.debug(
                "SubAgent %s: context reset disabled by config, keeping %d messages",
                self.name, len(self.agent.messages),
            )
            return
        if len(self.agent.messages) < 20:
            logger.debug(
                "SubAgent %s: only %d messages after task %s, skipping context reset",
                self.name, len(self.agent.messages), task.name,
            )
            return

        logger.info(
            "SubAgent %s: resetting context after task %s (%d messages)",
            self.name, task.name, len(self.agent.messages),
        )
        self._message_archive.extend(self.agent.messages)
        self.agent.clear_history()

    # -- reflection loop (browsing tasks) ------------------------------------

    def _execute_task_with_reflection(
        self,
        task: TaskSpec,
        initial_prompt: str | list[dict[str, Any]],
        collector: _TimingCollector,
        msg_start_idx: int = 0,
    ) -> str:
        """Execute a task using structured Search→Reflect→Summarize phases.

        Implements OpenJiuwen-style reflection loops with hard structural
        ceilings enforced in Python (not prompts):

        - ``max_search_plans``  outer loop iterations  (default 2)
        - ``max_reflection_loops``  inner loop per plan (default 2)
        - ``step_budget``  max_turns per search phase  (dynamic formula)

        The reflection call is a **separate** lightweight LLM call that
        does not pollute the agent's conversation history.
        """
        from arcticswarm.swarm.reflection import (
            MAX_SEARCH_PLANS,
            MAX_REFLECTION_LOOPS,
            RESERVE_TURNS_SUMMARIZE,
            ReflectionResult,
            compute_step_budget,
            run_reflection,
        )

        # Allow config overrides for eval tuning
        max_plans = self.config.browsing_max_search_plans or MAX_SEARCH_PLANS
        max_loops = self.config.browsing_max_reflection_loops or MAX_REFLECTION_LOOPS
        reflect_model = self.config.browsing_reflection_model or self.config.model

        # Lever D: cap per-subagent turns. subagent_max_turns=0 -> use max_turns (no-op).
        _eff_max_turns = getattr(self.config, "subagent_max_turns", 0) or self.config.max_turns
        step_budget = compute_step_budget(_eff_max_turns, max_plans=max_plans, max_loops=max_loops)

        accumulated_findings: list[str] = []
        all_queries_used: list[str] = []
        total_response = ""
        last_reflection: ReflectionResult | None = None
        search_prompt = initial_prompt
        did_break = False

        for plan_idx in range(max_plans):
            for loop_idx in range(max_loops):
                # Bail out early if the swarm is shutting down (avoids
                # making LLM calls on a client that may be closed soon).
                if self._shutdown.is_set():
                    did_break = True
                    break

                # ---- Phase: SEARCH / EXECUTE ---------------------------------
                collector.start()
                search_response = self.agent.run_turn_streaming(
                    search_prompt,
                    on_event=collector.on_event,
                    max_turns=step_budget,
                )
                self.token_usage += self.agent.last_turn_usage
                self.total_num_steps += self.agent.last_num_steps

                total_response += search_response
                accumulated_findings.append(search_response)

                # Extract queries used in this phase
                new_queries = self._extract_search_queries(self.agent.messages)
                all_queries_used.extend(new_queries)

                # Skip reflection if search produced no meaningful new content
                if len(search_response.strip()) < 100 and len(accumulated_findings) > 1:
                    logger.info(
                        "SubAgent %s: skipping reflection [plan=%d loop=%d] "
                        "(no new findings, %d chars)",
                        self.name, plan_idx, loop_idx, len(search_response.strip()),
                    )
                    continue

                # ---- Phase: REFLECT ------------------------------------------
                if self._shutdown.is_set():
                    did_break = True
                    break

                # Browsing-specific reflection using accumulated findings
                findings_for_reflect = "\n---\n".join(accumulated_findings[-3:])
                if len(findings_for_reflect) > 8000:
                    findings_for_reflect = findings_for_reflect[:8000] + "\n...(truncated)"

                reflection, reflect_usage = run_reflection(
                    client=self.agent.client,
                    model=reflect_model,
                    question=self.question,
                    task_prompt=task.prompt,
                    findings=findings_for_reflect,
                    queries_used=all_queries_used[-10:],
                    compact=getattr(self.config, "enable_compact_reflection", True),
                )
                # Roll the reflection LLM call's usage into the subagent's
                # token bucket so it shows up in `swarm_token_usage_breakdown`.
                # Previously this was captured but discarded.
                self.token_usage += reflect_usage
                last_reflection = reflection

                logger.info(
                    "SubAgent %s reflection [plan=%d loop=%d]: "
                    "sufficient=%s confidence=%s gaps=%d queries=%d",
                    self.name, plan_idx, loop_idx,
                    reflection.is_sufficient, reflection.confidence,
                    len(reflection.knowledge_gaps), len(reflection.next_queries),
                )

                # Track reflection statistics
                self.reflection_calls += 1
                conf = reflection.confidence.lower() if reflection.confidence else "low"
                if conf in self.reflection_confidence_counts:
                    self.reflection_confidence_counts[conf] += 1
                self.reflection_total_gaps += len(reflection.knowledge_gaps)
                self.reflection_total_queries += len(reflection.next_queries)

                if reflection.is_sufficient:
                    self.reflection_sufficient += 1
                    did_break = True
                    break

                # Build follow-up prompt from reflection gaps
                search_prompt = self._build_followup_prompt(
                    reflection, plan_idx, loop_idx, has_bbs=self._has_bbs,
                )

            if did_break:
                break

        # Track tasks that exhausted reflection loops without reaching sufficient
        if not did_break and self.reflection_calls > 0:
            self.reflection_insufficient += 1

        # ---- Phase: SUMMARIZE + POST -----------------------------------------
        # When the per-turn cap resolves to 1, tell the model to sequence
        # post_to_bbs and complete_task across separate turns (see prompt).
        _ov = getattr(self.agent, "max_tool_calls_per_turn_override", None)
        _eff_max_tc = _ov if _ov is not None else getattr(self.config, "max_tool_calls_per_turn", 0)
        summarize_prompt = self._build_summarize_prompt(
            task, accumulated_findings, last_reflection, has_bbs=self._has_bbs,
            single_tool_call=(_eff_max_tc == 1),
        )

        # Remove domain-specific tools for summarize phase — force synthesis only
        _STRIP_TOOLS = {"web_search", "web_fetch", "pdf_read"}
        saved_tools = dict(self.agent._tools)
        for tool_name in _STRIP_TOOLS:
            self.agent._tools.pop(tool_name, None)

        try:
            collector.start()
            summary_response = self.agent.run_turn_streaming(
                summarize_prompt,
                on_event=collector.on_event,
                max_turns=RESERVE_TURNS_SUMMARIZE,
            )
            self.token_usage += self.agent.last_turn_usage
            self.total_num_steps += self.agent.last_num_steps
        finally:
            # Always restore tools
            self.agent._tools = saved_tools

        return summary_response

    # -- reflection helpers --------------------------------------------------

    @staticmethod
    def _extract_search_queries(messages: list[dict[str, Any]]) -> list[str]:
        """Extract web_search query strings from recent conversation messages."""
        queries: list[str] = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "web_search"
                ):
                    query = block.get("input", {}).get("query", "")
                    if query:
                        queries.append(query)
        return queries

    @staticmethod
    def _build_followup_prompt(
        reflection: Any,  # ReflectionResult
        plan_idx: int,
        loop_idx: int,
        *,
        has_bbs: bool,
    ) -> str:
        """Build a follow-up search prompt from reflection gaps."""
        gaps = "\n".join(f"- {g}" for g in reflection.knowledge_gaps) if reflection.knowledge_gaps else "(none)"
        queries = "\n".join(f"- {q}" for q in reflection.next_queries) if reflection.next_queries else "(none)"
        completion_line = (
            "After searching, post findings to BBS. Do NOT call complete_task yet."
            if has_bbs else
            "After searching, keep a clear record of your sourced findings for the final "
            "task summary. Do NOT call complete_task yet."
        )

        return (
            f"## Follow-Up Search (round {plan_idx + 1}.{loop_idx + 2})\n\n"
            f"Your previous search was evaluated and found INSUFFICIENT.\n\n"
            f"### What We Have So Far\n{reflection.summary_of_findings}\n\n"
            f"### Knowledge Gaps to Fill\n{gaps}\n\n"
            f"### Suggested Search Queries\n{queries}\n\n"
            f"PRIORITY: Search for evidence that CONTRADICTS or DISPROVES the "
            f"current leading candidate, not just evidence that confirms it. "
            f"Also consider alternative entities that might match the "
            f"constraints.\n\n"
            f"Search specifically for the information gaps above. "
            f"Do NOT repeat previous searches. Focus on filling the specific "
            f"gaps listed.\n\n"
            f"{completion_line}"
        )

    @staticmethod
    def _build_summarize_prompt(
        task: TaskSpec,
        findings: list[str],
        reflection: Any | None,  # ReflectionResult | None
        *,
        has_bbs: bool,
        single_tool_call: bool = False,
    ) -> str:
        """Build the final summarize-and-post prompt."""
        confidence = reflection.confidence if reflection else "unknown"
        gaps: list[str] = []
        if reflection and not reflection.is_sufficient:
            gaps = reflection.knowledge_gaps or []
        gaps_text = "\n".join(f"- {g}" for g in gaps) if gaps else "None identified."
        synthesis_line = (
            "into a concise, high-quality summary and post it to the BBS."
            if has_bbs else
            "into a concise, high-quality summary for the orchestrator."
        )
        step_two = (
            f"2. **Post to BBS** using `post_to_bbs` with channel "
            f"\"key-findings\". Include:\n"
            f"   - A clear, concise summary of what you found\n"
            f"   - Source URLs for every factual claim\n"
            f"   - Confidence level: {confidence}\n"
            f"   - Remaining gaps: {gaps_text}\n"
            if has_bbs else
            f"2. **Prepare the completion summary** so it includes:\n"
            f"   - A clear, concise summary of what you found\n"
            f"   - Source URLs for every factual claim\n"
            f"   - Confidence level: {confidence}\n"
            f"   - Remaining gaps: {gaps_text}\n"
        )

        # When the per-turn cap is 1, post_to_bbs and complete_task must NOT be
        # batched into one turn (the second is dropped + re-issued — wasted
        # latency, and if the model deems itself done the post can be lost).
        # Sequence them across turns: post first, confirm, then complete.
        _sep = (
            " IN A SEPARATE TURN (after your post_to_bbs result returns)"
            if single_tool_call and has_bbs else ""
        )
        _seq_note = (
            "\n**ONE TOOL CALL PER TURN:** make `post_to_bbs` (step 2) your "
            "ONLY tool call this turn. Wait for its result, then do step 3 in a "
            "separate turn. Never call `post_to_bbs` and `complete_task` in the "
            "same turn.\n"
            if single_tool_call and has_bbs else ""
        )

        if task.status == TaskStatus.COMPLETED:
            # BBS Phase 1 wrapper-autocomplete already marked this task
            # COMPLETED.  Calling complete_task again would either be
            # dropped as a no-op or fall back to append_summary under
            # the hood.  Ask the LLM to use update_task_summary directly
            # so the intent is explicit and a task_summary_updated DM
            # is emitted instead of a stale task_completed duplicate.
            step_three = (
                f"3. **Call `update_task_summary`**{_sep} with "
                f"task_name=\"{task.name}\" and your final synthesis. "
                f"The task was already marked complete earlier in this "
                f"workflow; use `update_task_summary` to append your "
                f"polished final entry.\n\n"
            )
        else:
            step_three = (
                f"3. **Call `complete_task`**{_sep} with task_id=\"{task.id}\" "
                f"and a brief summary.\n\n"
            )
        return (
            f"## Final Summary Phase\n\n"
            f"You have completed your search. Now synthesize your findings "
            f"{synthesis_line}\n\n"
            f"### Instructions\n\n"
            f"1. **Synthesize** all your search findings into a focused summary "
            f"that directly addresses the task: \"{task.name}\"\n"
            f"{step_two}"
            f"{step_three}"
            f"{_seq_note}"
            f"### Quality Requirements\n"
            f"- Do NOT dump raw search results. Synthesize into a coherent narrative.\n"
            f"- Include specific facts, numbers, and dates when available.\n"
            f"- Note conflicting information explicitly.\n"
            f"- Keep the summary under 500 words — concise and focused."
        )

    # -- idle behaviour ------------------------------------------------------

    def _idle_check(self) -> None:
        """When idle, check BBS for new messages and react if needed.

        Runs a lightweight LLM turn with a reduced tool set (BBS / DM comm
        tools plus a capped ``reasoning`` tool for adversarial review).

        A cap on consecutive idle reviews prevents an infinite feedback loop
        where agents keep responding to each other's #discussion posts.
        The cap resets when new result-channel posts (discoveries,
        consensus, key-findings) appear, so agents always review genuine new
        findings.

        If this agent has already posted "VERIFIED" or "Already reviewed"
        to the BBS, skip further idle reviews — the work is done.
        """
        if not self._has_bbs or self.bbs is None or self._read_bbs_tool is None:
            return

        # --disable-builder-idle / --builder-idle-lifetime: skip or cap idle
        # reviews for builder agents (those that completed >= 1 task) while
        # leaving the dedicated auditor (_tasks_completed == 0) unaffected.
        is_builder = self._tasks_completed > 0
        is_auditor = self._dynamic_mode and self._tasks_completed == 0
        if is_builder and getattr(self.config, "disable_builder_idle", False):
            return

        # Early exit: if this agent already posted a conclusive review,
        # there's no value in re-reviewing the same findings.
        # Only applies to web search mode.
        # Skip this check for auditors when builder idle is reduced — the
        # auditor must keep reviewing since builders won't.
        builder_idle_reduced = (
            getattr(self.config, "disable_builder_idle", False)
            or getattr(self.config, "builder_idle_lifetime", -1) >= 0
        )
        reset_auditor = getattr(self.config, "reset_auditor_history", False)
        auditor_uncapped = is_auditor and (builder_idle_reduced or reset_auditor)

        if not auditor_uncapped:
            if self._idle_review_concluded and getattr(self.config, "web_search_enabled", False):
                return

        # Reset counter when genuinely new findings appear on result channels.
        result_count = sum(
            1 for m in self.bbs.read_all()
            if m.channel in _RESULT_CHANNELS
        )
        if result_count > self._last_result_msg_count:
            self._last_result_msg_count = result_count
            self._consecutive_idle_reviews = 0

        # Determine max idle reviews.
        # - Auditors are uncapped when builder idle is reduced or history resets.
        # - Builders use --builder-idle-lifetime as a LIFETIME cap (not reset
        #   by new findings), so `--builder-idle-lifetime 2` means at most 2
        #   total idle reviews for that builder's entire lifetime.
        # - The default consecutive cap (_MAX_CONSECUTIVE_IDLE_REVIEWS) still
        #   applies to all agents not covered by the lifetime cap.
        builder_limit = getattr(self.config, "builder_idle_lifetime", -1)
        if is_builder and builder_limit >= 0:
            if self._lifetime_idle_reviews >= builder_limit:
                return
        elif not auditor_uncapped:
            if self._consecutive_idle_reviews >= _MAX_CONSECUTIVE_IDLE_REVIEWS:
                return

        now = time.monotonic()
        if now - self._last_idle_check < _IDLE_CHECK_COOLDOWN:
            return

        new_content = self._read_bbs_tool.check_new_messages()
        if new_content is None:
            return

        self._last_idle_check = now
        self._consecutive_idle_reviews += 1
        self._lifetime_idle_reviews += 1
        self._set_status(AgentStatus.SURFING, "reviewing BBS updates")

        # --reset-auditor-history: archive existing messages before the idle
        # review turn so the LLM starts with a clean context.
        if is_auditor and reset_auditor and self.agent.messages:
            self._message_archive.extend(self.agent.messages)
            self.agent.clear_history()

        # Determine idle tool set and review message for the research path.
        full_tools = self.agent._tools
        _skill_names_for_idle: set[str] = (
            self._registered_skill_tool_names
            if self.config.per_skill_tools
            else {"load_skill"}
        )
        # Runtime-only comm tools (BBS / DM) are not constructible via
        # ToolFactory.  Keep the idle-review allowlist aligned with the
        # actual comm topology so BBS-only runs do not try to lazy-build
        # DM tools (`send_message`, `read_dm`) and DM-only runs do not try
        # to lazy-build BBS tools (`post_to_bbs`, `read_bbs`).
        _idle_comm_tools: set[str] = {"update_task_summary"}
        if self._has_bbs:
            _idle_comm_tools |= {"post_to_bbs", "read_bbs"}
        if self._has_dm:
            _idle_comm_tools |= {"send_message", "read_dm"}

        # YAML override: tools.idle_reviewer in the config takes precedence
        # over the hardcoded research default below.  An explicit list gives
        # operators control over idle-review cost / scope (e.g. drop
        # ``reasoning`` for cheaper runs, or add ``web_fetch`` so reviewers
        # can verify findings firsthand).  When the user lists ``load_skill``
        # we expand it to the per-skill tool names if ``per_skill_tools`` is
        # on, so the reviewer sees the same skill surface as the default path.
        user_idle_list = list(getattr(self.config, "idle_reviewer_tools", None) or [])
        if user_idle_list:
            idle_tool_names: set[str] = set(user_idle_list)
            if "load_skill" in idle_tool_names:
                idle_tool_names |= _skill_names_for_idle
        else:
            idle_tool_names = (
                _idle_comm_tools
                | {"read_skill_file", "reasoning"}
                | _skill_names_for_idle
            )
        # Pick a research review prompt so the LLM gets the right guidance,
        # regardless of tool choice.
        if self.agent.config.web_search_enabled:
            idle_msg = IDLE_REVIEW_MESSAGE_RESEARCH_ADVERSARIAL
        else:
            idle_msg = IDLE_REVIEW_MESSAGE_RESEARCH

        idle_tools = {
            k: v for k, v in full_tools.items()
            if k in idle_tool_names
        }
        # Lazily instantiate any idle-review tool that isn't already in
        # the agent's main toolset (e.g. ``reasoning`` for a browsing
        # agent, or ``web_fetch`` added to a reviewer via
        # ``tools.idle_reviewer``).  ``reasoning`` is additionally
        # wrapped in a one-shot guard so the agent can only call it a
        # few times per idle review cycle — prevents the costly
        # "review spiral" — while other tools are injected as-is.
        missing_names = [n for n in idle_tool_names if n not in idle_tools]
        if missing_names:
            from arcticswarm.tools.factory import ToolFactory as _IdleTF
            _idle_factory = _IdleTF(
                self.config,
                sf_client=self.agent.sf_client,
                agent_client=self.agent.client,
            )
            for _name in missing_names:
                try:
                    _tool = _idle_factory.make(_name)
                except Exception as _exc:
                    logger.debug(
                        "Idle-review tool '%s' could not be lazy-built: %s", _name, _exc
                    )
                    continue
                if _tool is None:
                    continue
                if _name == "reasoning":
                    _tool = _CappedToolWrapper(_tool, max_calls=3)
                idle_tools[_name] = _tool

        self.agent._tools = idle_tools

        prompt = idle_msg.format(
            new_messages=new_content,
            question=self.question,
        )

        msg_start_idx = len(self.agent.messages)
        collector = _TimingCollector(inner_on_event=self._on_event)
        collector.start()
        try:
            self.agent.run_turn_streaming(prompt, on_event=collector.on_event)
        except Exception as exc:
            logger.debug("SubAgent %s idle check failed: %s", self.name, exc)
        finally:
            self.token_usage += self.agent.last_turn_usage
            self.total_num_steps += self.agent.last_num_steps
            _inject_timings_into_messages(self.agent.messages, collector, msg_start_idx)
            self.agent._tools = full_tools
            self._set_status(AgentStatus.IDLE, "ready")

        # Check if this agent just posted a conclusive review — if so,
        # mark it done so we skip future idle cycles.
        if not auditor_uncapped:
            self._check_review_concluded()

    def _check_review_concluded(self) -> None:
        """Scan this agent's recent BBS posts for conclusive markers.

        Requires at least 2 conclusive posts before marking idle review
        as done — a single "Already reviewed" during an early idle cycle
        should not permanently shut down the agent's review capability.
        """
        if self.bbs is None:
            return
        _CONCLUSIVE = ("VERIFIED", "Already reviewed")
        conclusive_count = 0
        for msg in reversed(self.bbs.read_all()):
            if msg.author != self.name:
                continue
            if any(marker in msg.content for marker in _CONCLUSIVE):
                conclusive_count += 1
                if conclusive_count >= 2:
                    self._idle_review_concluded = True
                    logger.debug(
                        "SubAgent %s: idle review concluded (%d conclusive posts)",
                        self.name, conclusive_count,
                    )
                    return

    # -- DM helpers ----------------------------------------------------------

    def _check_dms(self) -> str | None:
        """Auto-injection callback: return formatted DMs or None."""
        if self._mailbox is None:
            return None
        msgs = self._mailbox.check_new(self.name)
        if not msgs:
            return None
        return self._mailbox.render_for_llm(msgs)

    # Tools allowed during idle DM handling — keeps the turn lightweight.
    _DM_HANDLE_TOOL_ALLOWLIST: frozenset[str] = frozenset({
        "send_message", "read_dm", "update_task_summary", "list_tasks",
        "complete_task", "read_skill_file",
        "post_to_bbs", "read_bbs",
    })

    def _handle_dm(self) -> None:
        """Process incoming DMs while idle — lightweight review turn.

        Uses a restricted tool set to prevent expensive execution
        (web_search, web_fetch, python_execute, bash, etc.) during what
        should be a quick verification pass.  The allowlist mirrors the
        idle-review tool set from ``_idle_check`` so the agent can post a
        targeted comment if needed.

        Capped by ``_MAX_CONSECUTIVE_DM_HANDLES`` to prevent cascading
        spirals where idle agents keep verifying each other's findings.
        The counter resets when the agent claims a real task.
        """
        if self._mailbox is None:
            return
        msgs = self._mailbox.check_new(self.name)
        if not msgs:
            return

        if self._consecutive_dm_handles >= _MAX_CONSECUTIVE_DM_HANDLES:
            logger.debug(
                "SubAgent %s hit DM handle cap (%d), skipping",
                self.name, _MAX_CONSECUTIVE_DM_HANDLES,
            )
            return

        self._consecutive_dm_handles += 1
        dm_content = self._mailbox.render_for_llm(msgs)
        self._set_status(AgentStatus.SURFING, "responding to DM")

        task_board_status = self.task_board.render_status()

        prompt = (
            f"## Overall Question\n{self.question}\n\n"
            f"## Task Board Status\n{task_board_status}\n\n"
            f"## Direct Message(s) Received\n\n"
            f"You are currently idle (not assigned to any task).\n\n"
            f"{dm_content}\n\n"
            "Review the message and respond helpfully. "
            "Do NOT re-run searches unless you spot a clear methodology problem.\n"
            "- If the message asks you to verify findings, check for obvious errors "
            "(unsupported claim, missing source, arithmetic mistake).\n"
            "- If everything looks reasonable, say 'Nothing to flag.'\n"
            "- If you find a real error, call update_task_summary and send a "
            "targeted DM to the original author only. Do NOT broadcast to 'all'."
        )

        full_tools = self.agent._tools
        _skill_names: set[str] = (
            self._registered_skill_tool_names
            if self.config.per_skill_tools
            else {"load_skill"}
        )
        dm_tools = {
            k: v for k, v in full_tools.items()
            if k in self._DM_HANDLE_TOOL_ALLOWLIST or k in _skill_names
        }
        self.agent._tools = dm_tools

        msg_start_idx = len(self.agent.messages)
        collector = _TimingCollector(inner_on_event=self._on_event)
        collector.start()
        try:
            self.agent.run_turn_streaming(prompt, on_event=collector.on_event)
        except Exception as exc:
            logger.debug("SubAgent %s DM handling failed: %s", self.name, exc)
        finally:
            self.token_usage += self.agent.last_turn_usage
            self.total_num_steps += self.agent.last_num_steps
            _inject_timings_into_messages(self.agent.messages, collector, msg_start_idx)
            self.agent._tools = full_tools
            self._set_status(AgentStatus.IDLE, "ready")

    def _notify_leader_idle(
        self,
        *,
        idle_reason: str = "available",
        completed_task_id: str | None = None,
        completed_task_name: str | None = None,
        completed_status: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """Send a lightweight idle DM to the leader (realtime mode only).

        Wakes the orchestrator's ``wait_for_message("leader")`` so it can
        retry ``prepare_report`` instead of blocking until timeout.
        """
        if self._status != AgentStatus.IDLE or self._prev_status == AgentStatus.IDLE:
            return
        if self._mailbox is not None and self._has_dm:
            peer_summary = self._mailbox.consume_last_peer_summary(self.name)
            payload = {
                "agent_name": self.name,
                "status": AgentStatus.IDLE.value,
                "idle_reason": idle_reason,
            }
            if completed_task_id is not None:
                payload["completed_task_id"] = completed_task_id
            if completed_task_name is not None:
                payload["completed_task_name"] = completed_task_name
            if completed_status is not None:
                payload["completed_status"] = completed_status
            if failure_reason is not None:
                payload["failure_reason"] = failure_reason
            if peer_summary is not None:
                payload["peer_summary"] = peer_summary
            try:
                self._mailbox.send(
                    from_agent=self.name,
                    to_agent="leader",
                    content=f"[idle] {self.name}",
                    lane=DM_LANE_CONTROL,
                    message_type=DM_TYPE_IDLE_NOTIFICATION,
                    payload=payload,
                )
            except Exception:
                pass

    # -- helpers -------------------------------------------------------------

    def _set_status(self, status: AgentStatus, activity: str) -> None:
        """Update agent registry and fire status-change callback."""
        self._prev_status = self._status
        self._status = status
        self.agent_registry.set_status(self.name, status, activity)
        if self._on_status_change:
            self._on_status_change(self.name, status.value, activity)

    def close(self) -> None:
        """Release resources.

        If a shared SnowflakeClient was provided, we only close the
        Anthropic HTTP client (the shared SF client is owned by the
        orchestrator).
        """
        try:
            self.agent.client.close()
        except Exception:
            pass
        if self.agent._orchestration_client is not self.agent.client:
            try:
                self.agent._orchestration_client.close()
            except Exception:
                pass
        if self._owns_sf_client and self.agent.sf_client is not None:
            try:
                self.agent.sf_client.close()
            except Exception:
                pass


# Backward-compatible alias (used in __init__.py and tools.py TYPE_CHECKING)
Teammate = SubAgent
