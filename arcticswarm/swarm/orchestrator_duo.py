"""Duo mode for the swarm orchestrator.

Two-agent "duo" mode (main worker + auditor, no orchestrator LLM) extracted
from :mod:`arcticswarm.swarm.orchestrator` into a mixin.  The
:class:`DuoMixin` is composed into ``SwarmOrchestrator`` via the MRO so the
``self.``-attribute surface (``self.config``, ``self._shared_sf_client``,
``self.last_*`` counters, ``self._capture_trajectories`` /
``self._aggregate_reflection_stats`` helpers) resolves on the orchestrator
instance exactly as before.  The method body is moved verbatim.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Callable

from arcticswarm.agent import (
    Agent,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCallStart,
)
from arcticswarm.logging_utils import aggregate_tool_role_usage
from arcticswarm.swarm.empty_answer_recovery import (
    extract_answer_from_messages as _extract_answer_from_messages,
)
from arcticswarm.swarm.mailbox import DM_TYPE_IDLE_NOTIFICATION, Mailbox
from arcticswarm.swarm.prompts import build_duo_system_prompt
from arcticswarm.swarm.task import AgentRegistry, TaskBoard
from arcticswarm.swarm.teammate import _TimingCollector, _inject_timings_into_messages
from arcticswarm.swarm.tools import (
    ListTasksTool,
    ReadDMTool,
    SendMessageTool,
    SendReportTool,
    SwarmContext,
)

# Symbols defined in orchestrator.py.  This import is safe from a
# circular-import standpoint because orchestrator.py imports this module only
# AFTER defining these event classes and helpers (see the import placement in
# orchestrator.py just before ``class SwarmOrchestrator``).
from arcticswarm.swarm.orchestrator import (
    OrchestratorTextDelta,
    OrchestratorToolCall,
    SubagentClaimedTask,
    SubagentIdle,
    SubagentSpawned,
    SwarmComplete,
    SwarmStarted,
    TeammateToolCall,
    _images_of,
    _make_peer_tool_observer,
    _summarize_tool_call,
    _text_of,
    _with_text_replaced,
)

logger = logging.getLogger(__name__)


class DuoMixin:
    """Duo-mode orchestration (main worker + auditor)."""

    def _run_duo_turn(
        self,
        question: str | list[dict[str, Any]],
        *,
        on_event: Callable[[StreamEvent], None] | None = None,
        on_swarm_event: Callable[..., None] | None = None,
        is_followup: bool = False,
        turn_number: int = 1,
    ) -> str:
        """Two-agent 'duo' mode — main worker + auditor, no orchestrator LLM.

        Both agents independently analyse the question in parallel, share
        findings via DM, reconcile, and the main worker submits the report.
        """
        if getattr(self.config, "disable_auditor", False):
            raise ValueError(
                "swarm.disable_auditor is not supported in duo mode: duo mode "
                "is leader + auditor by construction. Use dynamic/BBS swarm "
                "mode (comm: [bbs]) to run without a dedicated auditor."
            )
        from concurrent.futures import Future
        from datetime import date, datetime

        from arcticswarm.swarm.teammate import SubAgent

        t0 = time.monotonic()
        timings: dict[str, float] = {}
        duo_config = replace(self.config, orchestrator_realtime=True)

        # Text-only view of the question for subagent task descriptions,
        # the auditor's SubAgent context, and UI event payloads. The main
        # worker keeps the full multimodal content (image blocks + text)
        # so it sees attached images directly on turn 0. The auditor
        # (a SubAgent) receives ``question_images`` separately so it also
        # sees the image on its first turn.
        question_text = _text_of(question)
        question_images = _images_of(question)

        # ---- Names -----------------------------------------------------------
        main_name = "leader"
        auditor_name = "auditor"

        # ---- Infrastructure (reuse existing primitives) ----------------------
        mailbox = Mailbox()
        mailbox.register(main_name)
        mailbox.register(auditor_name)

        task_board = TaskBoard(num_agents=1)
        self._task_board = task_board  # expose for partial trajectory recovery on timeout
        agent_registry = AgentRegistry()
        agent_registry.register(auditor_name)

        pool = ThreadPoolExecutor(max_workers=1)

        has_web_search = self.config.has_web_search_capability()

        ctx = SwarmContext(
            bbs=None,
            task_board=task_board,
            agent_registry=agent_registry,
            config=duo_config,
            pool=pool,
            sf_client=self._shared_sf_client,
            on_swarm_event=on_swarm_event,
            question=question_text,
            question_images=question_images,
            max_teammates=1,
            active_channels=frozenset(),
            mailbox=mailbox,
            has_bbs=False,
            has_dm=True,
            system_reminder_interval=getattr(
                self.config, "system_reminder_interval", -1,
            ),
        )

        # ---- Pre-create the auditor's task -----------------------------------
        from arcticswarm.swarm.task import TaskSpec

        auditor_task = TaskSpec(
            id="task-1",
            name="Analyze the question",
            prompt=question_text,
            assigned_to=auditor_name,
            profile=self.config.swarm_profiles[0] if self.config.swarm_profiles else "",
        )
        task_board.add_task(auditor_task)

        # ---- Resolve current date --------------------------------------------
        if self.config.date_override:
            try:
                current_date = datetime.strptime(
                    self.config.date_override, "%Y-%m-%d",
                ).date()
            except ValueError:
                current_date = date.today()
        else:
            current_date = date.today()

        # ---- Spawn the auditor SubAgent on the thread pool -------------------
        t_spawn = time.monotonic()

        # Peer-tool-call observation: when one duo agent runs an observable
        # tool (e.g. ``edit_file``), the orchestrator drops a synthetic DM
        # into the other agent's mailbox so the peer learns about the
        # change before its next LLM turn. Gated behind
        # ``config.peer_tool_observation`` (default off). Tools observed
        # come from ``config.peer_tool_observation_tools`` — file mutators
        # by default. See ``_make_peer_tool_observer`` for the
        # implementation. Motivation: in duo mode a large fraction of
        # second-comer edits operate from a stale view, and many overlapping
        # edits happen with zero communication between the two agents.
        _peer_obs_enabled = bool(getattr(self.config, "peer_tool_observation", False))
        _peer_obs_tools = frozenset(
            getattr(self.config, "peer_tool_observation_tools", None) or [
                "edit_file", "str_replace_based_edit_tool", "bash",
            ]
        )

        def _make_auditor_on_event() -> Callable:
            from arcticswarm.agent import ToolCallStart as _TCS

            def _on_event(event: Any) -> None:
                if isinstance(event, _TCS) and on_swarm_event:
                    desc = _summarize_tool_call(event.tool_name, event.tool_input)
                    on_swarm_event(TeammateToolCall(
                        name=auditor_name,
                        tool_name=event.tool_name,
                        description=desc,
                    ))
                # Forward all stream events (ToolCallStart/End/etc.) to the
                # caller so auditor tool calls show up alongside leader's
                # in any caller-side trajectory collector.
                if on_event is not None:
                    on_event(event)

            if _peer_obs_enabled:
                return _make_peer_tool_observer(
                    emitter=auditor_name,
                    mailbox=mailbox,
                    peers=[main_name],
                    observed_tools=_peer_obs_tools,
                    base_on_event=_on_event,
                )
            return _on_event

        def _make_auditor_on_status(name: str) -> Callable:
            def _on_status_change(_name: str, status: str, activity: str) -> None:
                if on_swarm_event:
                    if status == "working":
                        on_swarm_event(SubagentClaimedTask(name=_name, activity=activity))
                    elif status == "idle":
                        on_swarm_event(SubagentIdle(name=_name, activity=activity))

            return _on_status_change

        auditor_config = duo_config.for_subagent()

        # ``self.config.auditor_role`` drives whether the auditor sees
        # the author-mode prompt (parallel patch producer) or the
        # reviewer-mode prompt (critic + test validator). Matched on
        # the report side by the reviewer-stall gate in
        # ``SendReportTool`` (which waits for the reviewer's first DM).
        _auditor_role = getattr(self.config, "auditor_role", "author")

        auditor = SubAgent(
            name=auditor_name,
            config=auditor_config,
            bbs=None,
            task_board=task_board,
            agent_registry=agent_registry,
            question=question_text,
            question_images=question_images,
            shutdown=ctx.shutdown,
            sf_client=self._shared_sf_client,
            on_event=_make_auditor_on_event(),
            on_status_change=_make_auditor_on_status(auditor_name),
            web_source_tracker=ctx.web_sources,
            active_channels=frozenset(),
            mailbox=mailbox,
            has_bbs=False,
            has_dm=True,
            system_reminder_interval=getattr(
                self.config, "system_reminder_interval", -1,
            ),
            is_duo=True,
            content_cache=self._content_cache,
        )

        # broadcast=True so complete_task / update_task_summary DMs reach "leader".
        from arcticswarm.swarm.tools import CompleteTaskTool, UpdateTaskSummaryTool

        auditor.agent._tools["complete_task"] = CompleteTaskTool(
            task_board, mailbox=mailbox, sender=auditor_name,
            broadcast=True,
        )
        auditor.agent._tools["update_task_summary"] = UpdateTaskSummaryTool(
            task_board, author=auditor_name, mailbox=mailbox, sender=auditor_name,
            broadcast=True,
        )

        # Override the auditor's system prompt with the duo auditor prompt
        _duo_profile = self.config.swarm_profiles[0] if self.config.swarm_profiles else "browsing"
        # ``self.config.auditor_role`` drives whether the auditor sees
        # the author-mode prompt (parallel patch producer) or the
        # reviewer-mode prompt (critic + test validator).
        auditor.agent.system_prompt = build_duo_system_prompt(
            agent_name=auditor_name,
            partner_name=main_name,
            current_date=current_date.isoformat(),
            is_main_worker=False,
            per_skill_tools=self.config.per_skill_tools,
            base_prompt=auditor.agent.system_prompt,
            profile_name=_duo_profile,
            peer_tool_observation=_peer_obs_enabled,
            auditor_role=_auditor_role,
        )
        auditor._base_system_prompt = auditor.agent.system_prompt

        # ``SubAgent._apply_profile`` (teammate.py) resets
        # ``self.agent.system_prompt = self._profile_prompts[profile_name]``
        # whenever a task is claimed, which silently undoes our orchestrator-
        # side prompt augmentation. Mirror the peer-observation note onto
        # every per-profile prompt the SubAgent caches so the note survives
        # profile switches. The DUO_AUDITOR_PROMPT body is intentionally
        # NOT re-applied here — the SubAgent already builds duo-aware
        # ``comm_protocol`` into each profile prompt via
        # ``build_comm_protocol_inline(..., is_duo=True)``.
        if _peer_obs_enabled and hasattr(auditor, "_profile_prompts"):
            from arcticswarm.swarm.prompts import PEER_TOOL_OBSERVATION_NOTE
            _peer_obs_note = PEER_TOOL_OBSERVATION_NOTE.format(partner_name=main_name)
            for _pk in list(auditor._profile_prompts.keys()):
                if _peer_obs_note not in auditor._profile_prompts[_pk]:
                    auditor._profile_prompts[_pk] = (
                        auditor._profile_prompts[_pk] + _peer_obs_note
                    )

        ctx.subagents.append(auditor)
        auditor_future: Future[None] = pool.submit(auditor.run_loop)
        ctx.futures.append(auditor_future)

        if on_swarm_event:
            on_swarm_event(SubagentSpawned(name=auditor_name))
        timings["subagent_spawn"] = round(time.monotonic() - t_spawn, 2)

        # ---- Build the main worker Agent ------------------------------------
        t_agent_setup = time.monotonic()
        agent = Agent(self.config)
        agent.web_source_tracker = ctx.web_sources

        # Share the SnowflakeClient
        if self._shared_sf_client is not None and agent.sf_client is not self._shared_sf_client:
            if agent.sf_client is not None:
                try:
                    agent.sf_client.close()
                except Exception:
                    pass
            agent.sf_client = self._shared_sf_client
            agent._register_tools()

        # Main worker keeps ALL its worker tools (web, skills, etc.)
        # Add DM tools
        agent._tools["send_message"] = SendMessageTool(
            mailbox=mailbox,
            sender=main_name,
            agent_names=[auditor_name],
            has_bbs=False,
        )
        read_dm_tool = ReadDMTool(mailbox, agent_name=main_name)
        agent._tools["read_dm"] = read_dm_tool

        def _main_check_dms() -> str | None:
            msgs = mailbox.check_new(main_name)
            if not msgs:
                return None
            return mailbox.render_for_llm(msgs)

        agent._auto_dm_check = _main_check_dms

        # Overwrite skill tools with duo-aware skill list
        _has_skill_tools = "load_skill" in agent._tools or self.config.per_skill_tools
        if _has_skill_tools:
            from arcticswarm.tools.skill_tools import LoadSkillTool, make_per_skill_tools as _make_ps
            from arcticswarm.skill_loader import SkillRegistry
            from arcticswarm.swarm.profiles import resolve_profile_skills, load_profiles_from_config, get_profile, DEFAULT_PROFILE_NAME
            from pathlib import Path

            skills_dir = Path(__file__).resolve().parent.parent / "skills"
            registry = SkillRegistry(skills_dir=skills_dir)
            _profiles = load_profiles_from_config(self.config.tool_profiles)
            _pname = (
                self.config.swarm_profiles[0]
                if self.config.swarm_profiles
                else DEFAULT_PROFILE_NAME
            )
            _def_profile = _profiles.get(_pname) or get_profile(_pname) or get_profile(DEFAULT_PROFILE_NAME)
            adjusted = resolve_profile_skills(
                _def_profile.skill_names, _def_profile.included_tools,
                has_bbs=False, has_dm=True, is_duo=True,
            )
            main_skills = list(dict.fromkeys([
                *adjusted,
                *self.config.orchestrator_skills,
            ]))
            if self.config.per_skill_tools:
                from arcticswarm.tools.skill_tools import PerSkillTool as _PST

                agent._tools.pop("load_skill", None)
                for k in list(agent._tools):
                    if isinstance(agent._tools[k], _PST):
                        del agent._tools[k]
                per_tools = _make_ps(
                    main_skills,
                    registry=registry,
                    legacy_format=self.config.skill_legacy_format,
                )
                agent._tools.update(per_tools)
            else:
                agent._tools["load_skill"] = LoadSkillTool(
                    main_skills,
                    registry=registry,
                    legacy_format=self.config.skill_legacy_format,
                )

        # Add report tool.
        #
        # Duo deliberately does NOT register ``prepare_report``: that tool
        # blocks inside a single LLM turn (waiting for tasks/agents to
        # settle), which freezes the leader's context and prevents
        # ``_auto_dm_check`` from injecting auditor DMs mid-wait.  Instead
        # we let the leader end its turn naturally when it wants to wait;
        # the outer loop's ``mailbox.wait_for_message`` call (below) is the
        # runtime-level wait, mirroring Claude Code's
        # ``waitForNextPromptOrShutdown`` pattern.
        #
        # ``send_user_markdown_report`` is registered from the start (no
        # "unlock" step required).  ``strict_dm_drain=True`` rejects the
        # submission when auditor DMs arrived mid-composition and remain
        # unread in the mailbox, forcing one more turn so those findings
        # aren't lost.  ``list_tasks`` stays on the toolkit so the leader
        # can inspect the auditor's status and activity whenever it wants.
        report_tool = SendReportTool(
            has_web_search=has_web_search,
            mailbox=mailbox,
            agent_name=main_name,
            strict_dm_drain=True,
            # Reviewer-mode "wait for first review" stall. When
            # ``auditor_role == "reviewer"``, ``SendReportTool``
            # blocks submission until the auditor has delivered at
            # least one peer-lane DM (its STATUS/FINDINGS review) or
            # the budget expires. Author mode passes these through
            # but they have no effect because the stall block is
            # gated on ``auditor_role == "reviewer"``.
            auditor_role=_auditor_role,
            peer_agent_name=auditor_name,
            reviewer_stall_budget_s=getattr(
                self.config, "auditor_review_stall_s", 60.0,
            ),
            reject_refusal=getattr(
                self.config, "reject_refusal_reports", False),
            question=question_text,
        )
        agent._tools["send_user_markdown_report"] = report_tool

        # Give the leader read-only visibility into the auditor's task so
        # it can decide when to wait (end turn) vs. submit.  ``_run_duo_turn``
        # builds ``agent`` from scratch via ``Agent(self.config)`` and does
        # NOT go through the main orchestrator setup path where
        # ``list_tasks`` is normally registered, so we must add it here
        # explicitly.  The tool is cheap and non-blocking — it just renders
        # the TaskBoard state — so it does not reintroduce the
        # prepare_report / wait_for_tasks latency pathology.
        agent._tools["list_tasks"] = ListTasksTool(task_board)

        # Remove orchestrator-only tools that don't apply to duo main
        # worker.  We intentionally do NOT strip ``list_tasks`` (registered
        # just above) and we DO strip ``wait_for_tasks`` (it blocks inside
        # a tool call — same pathology as the removed prepare_report).
        for tool_name in ("create_task", "wait_for_tasks", "create_agent"):
            agent._tools.pop(tool_name, None)

        # Set the duo main worker system prompt.
        agent.system_prompt = build_duo_system_prompt(
            agent_name=main_name,
            partner_name=auditor_name,
            current_date=current_date.isoformat(),
            is_main_worker=True,
            per_skill_tools=self.config.per_skill_tools,
            base_prompt=agent.system_prompt,
            profile_name=_duo_profile,
            peer_tool_observation=_peer_obs_enabled,
            auditor_role=_auditor_role,
        )

        # Expose the main-worker agent BEFORE the execution loop so the eval
        # runner's _capture_partial_trajectory can still dump conversation
        # history when the outer watchdog fires.  Previously this was only
        # set after wait_and_cleanup, which meant DUO timeout trajectories
        # contained nothing but the [eval error] marker.
        self._agent = agent
        self._ctx = ctx
        if mailbox is not None:
            try:
                mailbox.attach_task_board(task_board)
            except AttributeError:
                pass

        timings["agent_setup"] = round(time.monotonic() - t_agent_setup, 2)

        # ---- Activate swarm UI -----------------------------------------------
        if on_swarm_event:
            on_swarm_event(SwarmStarted(question=question_text, bbs=None))

        # ---- Streaming helpers (reuse from main path) ------------------------
        def _on_agent_event(event: StreamEvent) -> None:
            if on_swarm_event:
                if isinstance(event, TextDelta):
                    on_swarm_event(OrchestratorTextDelta(text=event.text))
                elif isinstance(event, ToolCallStart):
                    desc = _summarize_tool_call(event.tool_name, event.tool_input)
                    on_swarm_event(OrchestratorToolCall(
                        tool_name=event.tool_name, description=desc,
                    ))
            # Forward raw stream events (including ToolCallStart/End) to the
            # caller's on_event so swarm callers can collect tool-call
            # trajectories the same way single-agent callers do.
            if on_event is not None:
                on_event(event)

        # ---- Execution loop (realtime pattern) --------------------------------
        # Only rewrite the text portion on follow-up turns; keep any image
        # blocks intact so the main worker still sees the image.
        if is_followup:
            enriched_question_text = (
                f"[Turn {turn_number} — follow-up request]\n{question_text}"
            )
            enriched_question: str | list[dict[str, Any]] = _with_text_replaced(
                question, enriched_question_text,
            )
        else:
            enriched_question = question

        msg_start_idx = len(agent.messages)
        # When peer-tool-observation is on, wrap the leader's event handler
        # so its ``edit_file``/``write_file``/``bash`` tool calls get
        # mirrored as DMs into the auditor's mailbox (symmetric to the
        # auditor → leader path wired around line ~2580).
        _leader_inner_on_event: Callable[[Any], None] = _on_agent_event
        if _peer_obs_enabled:
            _leader_inner_on_event = _make_peer_tool_observer(
                emitter=main_name,
                mailbox=mailbox,
                peers=[auditor_name],
                observed_tools=_peer_obs_tools,
                base_on_event=_on_agent_event,
            )
        orch_collector = _TimingCollector(inner_on_event=_leader_inner_on_event)
        _realtime_timeout = getattr(self.config, "orchestrator_realtime_timeout", 300)

        try:
            prompt: str | list[dict[str, Any]] | None = enriched_question
            while True:
                orch_collector.start()
                agent.run_turn_streaming(
                    prompt, on_event=orch_collector.on_event,
                )

                if report_tool.captured_report:
                    break

                # Wait for DMs from auditor (same pattern as orchestrator realtime loop)
                prompt = None
                while prompt is None:
                    # Pre-check: if auditor already finished, skip the wait
                    if task_board.all_completed() and agent_registry.all_idle():
                        peer_summary = mailbox.drain_peer_summaries()
                        idle_parts: list[str] = []
                        if peer_summary:
                            idle_parts.append(peer_summary)
                        idle_parts.append(
                            '<swarm_notification type="all_idle">'
                            "The auditor has finished their analysis. "
                            "Review their findings, reconcile if needed, "
                            "then call send_user_markdown_report with "
                            "your final answer."
                            "</swarm_notification>"
                        )
                        prompt = "\n\n".join(idle_parts)
                        break

                    got_dm = mailbox.wait_for_message(
                        main_name, timeout=_realtime_timeout,
                    )
                    if not got_dm:
                        peer_summary = mailbox.drain_peer_summaries()
                        timeout_parts: list[str] = []
                        if peer_summary:
                            timeout_parts.append(peer_summary)
                        timeout_parts.append(
                            '<swarm_notification type="timeout">'
                            "Timeout waiting for auditor. "
                            "Call send_user_markdown_report with the "
                            "data collected so far."
                            "</swarm_notification>"
                        )
                        prompt = "\n\n".join(timeout_parts)
                        orch_collector.start()
                        agent.run_turn_streaming(
                            prompt, on_event=orch_collector.on_event,
                        )
                        prompt = None  # sentinel: outer loop should break
                        break

                    dm_msgs = mailbox.check_new(main_name) or []
                    idle_msgs = [
                        m for m in dm_msgs
                        if m.message_type == DM_TYPE_IDLE_NOTIFICATION
                    ]
                    real_msgs = [
                        m for m in dm_msgs
                        if m.message_type != DM_TYPE_IDLE_NOTIFICATION
                    ]

                    actionable_idle_msgs = [
                        m for m in idle_msgs
                        if (
                            m.payload.get("completed_task_id")
                            or m.payload.get("failure_reason")
                            or m.payload.get("peer_summary")
                        )
                    ]

                    if not real_msgs and actionable_idle_msgs:
                        peer_summary = mailbox.drain_peer_summaries()
                        parts: list[str] = []
                        if peer_summary:
                            parts.append(peer_summary)
                        parts.append(mailbox.render_for_llm(actionable_idle_msgs))
                        prompt = "\n\n".join(parts)
                    elif not real_msgs and idle_msgs:
                        if not agent_registry.all_idle():
                            continue
                        peer_summary = mailbox.drain_peer_summaries()
                        parts: list[str] = []
                        if peer_summary:
                            parts.append(peer_summary)
                        parts.append(
                            '<swarm_notification type="all_idle">'
                            "The auditor has finished their analysis. "
                            "Review their findings, reconcile if needed, "
                            "then call send_user_markdown_report with "
                            "your final answer."
                            "</swarm_notification>"
                        )
                        prompt = "\n\n".join(parts)
                    elif real_msgs:
                        peer_summary = mailbox.drain_peer_summaries()
                        parts = []
                        if peer_summary:
                            parts.append(peer_summary)
                        parts.append(mailbox.render_for_llm(real_msgs))
                        prompt = "\n\n".join(parts)
                    else:
                        continue

                if prompt is None:
                    break

            _inject_timings_into_messages(agent.messages, orch_collector, msg_start_idx)

            answer = report_tool.captured_report or ""

            # Fallback: recover answer from orchestrator messages when
            # the report tool was never called (swarm bypass).
            if not answer.strip():
                answer = _extract_answer_from_messages(agent.messages)
                if answer:
                    logger.info(
                        "Swarm bypass fallback (dm-realtime): recovered "
                        "%d-char answer from orchestrator messages",
                        len(answer),
                    )

            # ---- Cleanup -----------------------------------------------------
            t_cleanup = time.monotonic()
            ctx.wait_and_cleanup(timeout=300)
            timings["wait_and_cleanup"] = round(
                time.monotonic() - t_cleanup, 2,
            )

            # ---- Trajectory capture ------------------------------------------
            # NOTE: self._agent was assigned earlier (before the execution
            # loop) so _capture_partial_trajectory works on eval timeouts.
            # Re-asserting here is harmless and documents the invariant.
            self._agent = agent
            self._capture_trajectories(agent, ctx, task_board, bbs=None)

            # ---- Token usage aggregation ------------------------------------
            total_usage = agent.last_turn_usage
            breakdown: dict[str, TokenUsage] = {}
            breakdown["orchestrator"] = agent.last_turn_usage
            for sa in ctx.subagents:
                total_usage += sa.token_usage
                if sa.token_usage.total_tokens > 0:
                    breakdown[sa.name] = sa.token_usage
            # Commit the orch + subagent breakdown before tool-role aggregation
            # so a drain failure can't lose it (mirrors the BBS-mode path).
            self.last_token_usage = total_usage
            self.last_token_usage_breakdown = breakdown
            try:
                tool_role_totals, tool_role_calls = aggregate_tool_role_usage(agent, ctx.subagents)
                for role_name, role_usage in tool_role_totals.items():
                    if role_usage.total_tokens > 0 or tool_role_calls.get(role_name, 0) > 0:
                        breakdown[role_name] = role_usage
                        total_usage += role_usage
                self.last_token_usage = total_usage
                self.last_token_usage_breakdown = breakdown
            except Exception:
                logger.warning(
                    "Tool-role token aggregation failed (duo) — keeping orch+subagent breakdown",
                    exc_info=True,
                )
            self.last_num_steps = agent.last_num_steps + sum(
                sa.total_num_steps for sa in ctx.subagents
            )
            orch_tokens = breakdown["orchestrator"].total_tokens
            max_sa_tokens = max(
                (sa.token_usage.total_tokens for sa in ctx.subagents), default=0,
            )
            self.last_total_token_e2e = orch_tokens + max_sa_tokens
            self.last_saturation_events = task_board.saturation_events
            # Mirror the dynamic-mode aggregation for these counters so duo
            # callers see the same ``last_*`` surface area. Without this,
            # ``last_total_llm_calls`` etc. silently stay at 0 because the
            # duo branch previously skipped them.
            self.last_compaction_count = agent.compaction_count + sum(
                sa.agent.compaction_count for sa in ctx.subagents
            )
            self.last_total_llm_calls = agent.total_llm_calls + sum(
                sa.agent.total_llm_calls for sa in ctx.subagents
            )
            self.last_safety_refusal_count = agent.safety_refusal_count + sum(
                sa.agent.safety_refusal_count for sa in ctx.subagents
            )
            self.last_content_filter_count = agent.content_filter_count + sum(
                sa.agent.content_filter_count for sa in ctx.subagents
            )

            # Reflection stats (likely empty for duo, but kept for compat)
            self.last_reflection_stats = self._aggregate_reflection_stats(
                ctx.subagents
            )

            elapsed = time.monotonic() - t0
            timings["total"] = round(elapsed, 2)
            self.phase_timings = timings

            if on_swarm_event:
                on_swarm_event(SwarmComplete(
                    answer=answer[:200],
                    duration_seconds=elapsed,
                    subagent_count=len(ctx.subagents),
                    bbs_message_count=0,
                    report=answer,
                    token_usage=total_usage,
                    web_source_tracker=ctx.web_sources,
                ))

            self._last_report = answer
            return answer

        except Exception:
            _inject_timings_into_messages(agent.messages, orch_collector, msg_start_idx)
            timings["total"] = round(time.monotonic() - t0, 2)
            timings["error"] = True  # type: ignore[assignment]
            self.phase_timings = timings
            try:
                total_usage = agent.last_turn_usage
                breakdown = {}
                breakdown["orchestrator"] = agent.last_turn_usage
                for sa in ctx.subagents:
                    total_usage += sa.token_usage
                    if sa.token_usage.total_tokens > 0:
                        breakdown[sa.name] = sa.token_usage
                # Commit before tool-role aggregation — see matching comment
                # in the BBS exception path.
                self.last_token_usage = total_usage
                self.last_token_usage_breakdown = breakdown
                try:
                    tool_role_totals, tool_role_calls = aggregate_tool_role_usage(agent, ctx.subagents)
                    for role_name, role_usage in tool_role_totals.items():
                        if role_usage.total_tokens > 0 or tool_role_calls.get(role_name, 0) > 0:
                            breakdown[role_name] = role_usage
                            total_usage += role_usage
                    self.last_token_usage = total_usage
                    self.last_token_usage_breakdown = breakdown
                except Exception:
                    pass
            except Exception:
                pass
            ctx.wait_and_cleanup(timeout=30)
            raise
        finally:
            pool.shutdown(wait=False)

