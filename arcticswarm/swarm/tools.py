"""BBS and task board tools for swarm subagents and the orchestrator.

These tools extend :class:`~arcticswarm.tools.base.BaseTool` so they integrate
with the existing Anthropic tool-use protocol.  Each subagent ``Agent`` gets
these registered alongside the standard arcticswarm tools.

The orchestrator also gets orchestration-specific tools (``DynamicCreateTaskTool``,
``ListTasksTool``, ``WaitForTasksTool``) that let it post tasks to the board
and monitor their progress.  Subagents are spawned on demand as tasks are created.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

from arcticswarm.tools.base import BaseTool, ToolResult
from arcticswarm.swarm.bbs import (
    ALL_CHANNELS,
    BBS,
    CHANNEL_CONSENSUS,
    CHANNEL_DISCOVERIES,
    CHANNEL_DISCUSSION,
    CHANNEL_KEY_FINDINGS,
    CHANNEL_TASKS,
    is_verified_consensus_verdict,
)
from arcticswarm.swarm.mailbox import (
    DM_LANE_PEER,
    DM_LANE_RESULT,
    DM_TYPE_IDLE_NOTIFICATION,
    DM_TYPE_PEER_MESSAGE,
    DM_TYPE_SUBAGENT_COMPLETE,
    DM_TYPE_TASK_COMPLETED,
    DM_TYPE_TASK_FAILED,
    DM_TYPE_TASK_SUMMARY_UPDATED,
    Mailbox,
)

# DM types that the strict-DM-drain guard on send_user_markdown_report
# should silently swallow rather than treat as "new findings". In
# ``dm_realtime_direct_report`` mode the task-completion summary is
# already delivered as the ``create_task`` tool result, so the matching
# result-lane DM is redundant; idle pings and worktree-harvest notices
# are pure control signals (the harvest is also surfaced separately via
# the pre-submit reminder). Without this filter, every successful task
# emits 1-3 redundant DMs that bounce the orchestrator's first submit
# attempt and trigger a re-send loop. Substantive types
# (PEER_MESSAGE, TASK_SUMMARY_UPDATED) are NOT in this set and still
# block submission, because they may carry corrections the leader has
# not yet read.
_NON_SUBSTANTIVE_REPORT_DRAIN_DM_TYPES = frozenset({
    DM_TYPE_IDLE_NOTIFICATION,
    DM_TYPE_TASK_COMPLETED,
    DM_TYPE_TASK_FAILED,
    DM_TYPE_SUBAGENT_COMPLETE,
})
from arcticswarm.swarm.references import ReferenceRegistry
from arcticswarm.swarm.task import (
    STALE_HEARTBEAT_THRESHOLD_SECONDS,
    AgentRegistry,
    AgentStatus,
    TaskBoard,
    TaskSpec,
    TaskStatus,
    task_is_alt,
)
from arcticswarm.swarm.web_sources import WebSourceTracker

if TYPE_CHECKING:
    from arcticswarm.config import ArcticswarmConfig
    from arcticswarm.snowflake_client import SnowflakeClient
    from arcticswarm.swarm.teammate import SubAgent

logger = logging.getLogger(__name__)


def _normalize_bbs_content(content: Any) -> str:
    """Normalise BBS post content to a plain string.

    LLMs sometimes pass content as ``[{"text": "..."}]`` (Anthropic
    content-block format) instead of a plain string.  This function
    extracts the text in those cases.
    """
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts) if parts else str(content)

    if not isinstance(content, str):
        return str(content)

    stripped = content.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                parts = []
                for item in parsed:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(str(item["text"]))
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return " ".join(parts)
        except (ValueError, TypeError):
            pass

    return content


# ---------------------------------------------------------------------------
# PostToBBS
# ---------------------------------------------------------------------------


class PostToBBSTool(BaseTool):
    """Post a discovery, work log, or discussion to the shared BBS."""

    def __init__(
        self,
        bbs: BBS,
        author: str,
        *,
        channels: frozenset[str] | None = None,
        has_dm: bool = False,
        is_web: bool = False,
        is_auditor: bool = False,
    ) -> None:
        self._bbs = bbs
        self._author = author
        self._channels = channels or ALL_CHANNELS
        self._has_dm = has_dm
        self._is_web = is_web
        self._is_auditor = is_auditor

    @property
    def name(self) -> str:
        return "post_to_bbs"

    @property
    def description(self) -> str:
        channel_descs: list[str] = []
        if "discoveries" in self._channels:
            channel_descs.append(
                "'discoveries' for found sources, data structures, "
                "relevant context, and detailed work (SQL queries with "
                "their results, methodology notes)"
            )
        if "key-findings" in self._channels:
            channel_descs.append("'key-findings' for critical facts and candidate answers")
        if "consensus" in self._channels:
            channel_descs.append("'consensus' for resolved decisions")
        if "discussion" in self._channels:
            channel_descs.append("'discussion' for challenging findings")

        base = (
            "Post a message to the shared Bulletin Board System (BBS) so other "
            "agents in the swarm can see your findings. Channels: "
            + ", ".join(channel_descs) + "."
        )
        if self._has_dm:
            base += (
                " If you need a specific agent to re-verify or correct something, "
                "use send_message instead of posting to BBS."
            )
        return base

    def parameters_schema(self) -> dict[str, Any]:
        sd_parts: list[str] = [
            "For discoveries: {source_url, title, key_facts} or "
            "{table_fqn, columns, sample_data}. SQL evidence posts may "
            "include {sql, row_count, result_preview, methodology}.",
        ]
        sd_parts.append("For consensus: {topic, decision, votes}.")
        structured_data_desc = "Machine-readable payload. " + " ".join(sd_parts)

        tags_example = "'source:web', 'topic:analysis'"

        return {
            "type": "object",
            "required": ["channel", "content"],
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "BBS channel to post to.",
                    "enum": sorted(self._channels),
                },
                "content": {
                    "type": "string",
                    "description": "Human-readable summary of the post.",
                },
                "structured_data": {
                    "type": "object",
                    "description": structured_data_desc,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Tags for filtering (e.g. {tags_example}).",
                },
                "in_reply_to": {
                    "type": "string",
                    "description": "Message ID this post replies to (for threaded discussion).",
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        channel = kwargs.get("channel", "")
        content = kwargs.get("content", "")
        structured_data = kwargs.get("structured_data")
        tags = kwargs.get("tags")
        in_reply_to = kwargs.get("in_reply_to")

        if isinstance(content, str):
            content_stripped = content.strip()
        else:
            content_stripped = content
        if not channel or not content_stripped:
            allowed = sorted(self._channels)
            missing = []
            if not channel:
                missing.append("`channel`")
            if not content_stripped:
                missing.append("`content`")
            return ToolResult(
                error=(
                    f"ERROR: `post_to_bbs` is missing required parameter(s): "
                    f"{', '.join(missing)}.\n\n"
                    "COMMON MISTAKE: you may have written the message text as "
                    "plain assistant text alongside this tool call. That text "
                    "is IGNORED — only the JSON arguments of this tool call "
                    "are posted to the BBS. Both `channel` and `content` MUST "
                    "be passed inside the tool call's JSON arguments.\n\n"
                    "Required JSON shape:\n"
                    f'  {{"channel": "<one of {allowed}>", '
                    '"content": "<the message text>", '
                    '"structured_data": {...optional JSON object...}, '
                    '"tags": ["optional", "string", "tags"]}\n\n'
                    f"You provided: channel={channel!r}, content={content!r}.\n"
                    "Retry this call with both parameters populated in the "
                    "JSON arguments."
                ),
                is_error=True,
            )

        # Normalise content: LLMs sometimes pass [{"text": "..."}] instead
        # of a plain string.
        content = _normalize_bbs_content(content)
        if channel not in self._channels:
            return ToolResult(
                error=f"Unknown channel '{channel}'. Use one of: {sorted(self._channels)}",
                is_error=True,
            )

        msg = self._bbs.post(
            channel=channel,
            author=self._author,
            content=content,
            structured_data=structured_data,
            in_reply_to=in_reply_to,
            tags=tags,
        )
        return ToolResult(
            output=f"Posted to #{channel} (id={msg.id})",
            metadata={"message_id": msg.id},
        )


# ---------------------------------------------------------------------------
# ReadBBS
# ---------------------------------------------------------------------------


class ReadBBSTool(BaseTool):
    """Read recent posts from the BBS (optionally filtered by channel/tags).

    Supports incremental reads via an internal ``_last_seen_id`` cursor.
    When no explicit ``since_id`` is provided by the caller, the tool
    automatically returns only messages posted after the last read.
    Both :meth:`execute` (manual LLM calls) and :meth:`check_new_messages`
    (auto-injection) share the same cursor so there are no duplicates.

    Includes backoff to prevent agents from burning tokens polling an
    empty BBS hundreds of times.  After ``_EMPTY_READ_BACKOFF_THRESHOLD``
    consecutive empty reads the tool starts injecting a delay and a
    stronger hint to stop polling.
    """

    # After this many consecutive empty reads, start backoff
    _EMPTY_READ_BACKOFF_THRESHOLD = 10
    # After this many, return a hard stop message
    _EMPTY_READ_HARD_LIMIT = 20

    def __init__(self, bbs: BBS, *, channels: frozenset[str] | None = None) -> None:
        self._bbs = bbs
        self._channels = channels or ALL_CHANNELS
        # Incremental-read cursor: tracks the ID of the last message
        # this agent has seen.  ``None`` means "show everything".
        self._last_seen_id: str | None = None
        # Backoff state for consecutive empty reads
        self._consecutive_empty_reads: int = 0

    # -- cursor management ---------------------------------------------------

    def initialize_cursor(self) -> None:
        """Advance the cursor to the latest BBS message.

        Call this at spawn time so that subsequent reads (both manual and
        auto-injected) skip messages already included in the initial BBS
        snapshot embedded in the user prompt.
        """
        msgs = self._bbs.read_all()
        if msgs:
            self._last_seen_id = msgs[-1].id

    def check_new_messages(self) -> str | None:
        """Return formatted new BBS messages since last read, or ``None``.

        Used by the auto-injection hook in the agent loop.  Updates the
        internal cursor so the same messages are not returned twice.
        """
        msgs = self._bbs.read(since_id=self._last_seen_id)
        if not msgs:
            return None
        self._last_seen_id = msgs[-1].id
        return self._format_messages(msgs)

    # -- BaseTool interface --------------------------------------------------

    @property
    def name(self) -> str:
        return "read_bbs"

    @property
    def description(self) -> str:
        return (
            "Read recent messages from the shared Bulletin Board System. "
            "Use this to see what other agents have discovered or posted. "
            "Only returns messages you haven't seen yet (incremental). "
            "Filter by channel or tags to focus on specific topics."
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Filter by BBS channel.",
                    "enum": sorted(self._channels),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags (returns posts matching any tag).",
                },
                "since_id": {
                    "type": "string",
                    "description": (
                        "Only return messages posted after this ID. "
                        "If omitted, returns messages since your last read."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of messages to return (default 50).",
                    "default": 50,
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        channel = kwargs.get("channel")
        tags = kwargs.get("tags")
        since_id = kwargs.get("since_id")
        limit = kwargs.get("limit", 50)

        # Use the incremental cursor when no explicit since_id is provided
        effective_since_id = since_id if since_id is not None else self._last_seen_id

        msgs = self._bbs.read(
            channel=channel,
            tags=tags,
            since_id=effective_since_id,
            limit=limit,
        )

        # Advance the cursor to the latest returned message
        if msgs:
            self._last_seen_id = msgs[-1].id
            self._consecutive_empty_reads = 0  # reset backoff

        if not msgs:
            self._consecutive_empty_reads += 1

            # Hard limit: tell the agent to stop polling entirely
            if self._consecutive_empty_reads >= self._EMPTY_READ_HARD_LIMIT:
                return ToolResult(
                    output=(
                        "(no new messages — you have polled "
                        f"{self._consecutive_empty_reads} times with no "
                        "results. STOP reading BBS and focus on your "
                        "current task. Post your findings or say you "
                        "have nothing new to add.)"
                    ),
                )

            # Soft backoff: add a delay and a gentle hint
            if self._consecutive_empty_reads >= self._EMPTY_READ_BACKOFF_THRESHOLD:
                delay = min(
                    2.0 * (self._consecutive_empty_reads - self._EMPTY_READ_BACKOFF_THRESHOLD + 1),
                    10.0,
                )
                time.sleep(delay)
                return ToolResult(
                    output=(
                        "(no new messages — consider focusing on your "
                        "task instead of polling the BBS repeatedly)"
                    ),
                )

            return ToolResult(output="(no new messages)")

        return ToolResult(
            output=self._format_messages(msgs),
            metadata={"count": len(msgs)},
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _format_messages(msgs: list[Any]) -> str:
        """Format a list of :class:`BBSMessage` objects as readable text."""
        lines: list[str] = []
        for m in msgs:
            header = f"[{m.channel}] {m.author} (id={m.id})"
            if m.in_reply_to:
                header += f" re:{m.in_reply_to}"
            lines.append(header)
            lines.append(f"  {m.content}")
            if m.structured_data:
                lines.append(f"  data: {json.dumps(m.structured_data, default=str)}")
            if m.tags:
                lines.append(f"  tags: {list(m.tags)}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SendMessageTool — targeted DM between subagents
# ---------------------------------------------------------------------------


class SendMessageTool(BaseTool):
    """Send a direct message to a specific subagent (or all, in DM-only mode)."""

    def __init__(
        self,
        mailbox: Mailbox,
        sender: str,
        agent_names: list[str],
        has_bbs: bool = True,
        peer_dm_summary: bool = False,
        dynamic_names: bool = False,
    ) -> None:
        self._mailbox = mailbox
        self._sender = sender
        self._agent_names = sorted(agent_names)
        self._has_bbs = has_bbs
        self._peer_dm_summary = peer_dm_summary
        self._dynamic_names = dynamic_names

    @property
    def name(self) -> str:
        return "send_message"

    @property
    def description(self) -> str:
        if self._has_bbs:
            peer_hint = ""
            if self._peer_dm_summary:
                peer_hint = (
                    "\n\nWhen messaging a peer about their task, briefly restate the relevant "
                    "task name and the specific figure or claim you are questioning so they "
                    "can respond without re-reading the full board.\n"
                )
            return (
                "Send a private direct message to a **single specific teammate** by name. "
                "The recipient gets a dedicated turn to respond with their full tool set.\n\n"
                "**When to use send_message:**\n"
                "- You spotted a potential error in another agent's BBS post and want THEM to re-verify\n"
                "- You need a specific agent to re-run a query, check a number, or clarify methodology\n"
                "- You want to ask a specific agent a follow-up question about their findings\n\n"
                "**When NOT to use send_message — use post_to_bbs instead:**\n"
                "- Sharing your own findings, results, or SQL output (use BBS)\n"
                "- General announcements the whole team should see (use BBS)\n"
                "- Posting discoveries or key findings (use BBS)\n\n"
                "Default to post_to_bbs for sharing; use send_message for targeted requests to a specific agent."
                f"{peer_hint}"
            )
        return (
            "Send a direct message to a specific teammate by name, or "
            "use to='all' to broadcast to every teammate. Use targeted "
            "messages when you need a specific agent to act; use 'all' to "
            "share findings with the whole team. Recipients get a dedicated "
            "turn to respond with full tools.\n\n"
            "**Message quality**: Include specific details — numbers, SQL "
            "snippets, table names, or concrete evidence. Avoid vague "
            "messages like 'I verified your findings'. Instead: 'I re-ran "
            "your query with HAVING SUM(...) > 0 and got 3 users instead "
                "of 5 — ADARABI was miscounted.'"
        )

    def _resolve_agent_names(self) -> list[str]:
        """Return current teammate names, using mailbox if dynamic."""
        if self._dynamic_names:
            return sorted(
                n for n in self._mailbox.registered_names
                if n != self._sender
            )
        return self._agent_names

    def parameters_schema(self) -> dict[str, Any]:
        to_field: dict[str, Any] = {"type": "string"}
        if self._dynamic_names:
            to_field["description"] = (
                "Name of the recipient agent (use the name you gave "
                "when creating the agent), or 'all' to broadcast."
            )
        else:
            to_enum = list(self._agent_names)
            if not self._has_bbs:
                to_enum = ["all"] + to_enum
            to_field["description"] = (
                "Name of the recipient agent, or 'all' to broadcast to every teammate."
            )
            to_field["enum"] = to_enum
        props: dict[str, Any] = {
            "to": to_field,
            "content": {
                "type": "string",
                "description": (
                    "The message to send. Be specific about what you need "
                    "the recipient to do (re-verify, adjust, check, etc.)."
                ),
            },
        }
        required = ["to", "content"]
        if self._peer_dm_summary:
            props["summary"] = {
                "type": "string",
                "description": (
                    "A 5-15 word summary of this message for the team leader's "
                    "visibility. Example: 'Asking Patricia to re-check revenue date filter'"
                ),
            }
            required.append("summary")
        return {
            "type": "object",
            "required": required,
            "properties": props,
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        to = kwargs.get("to", "")
        content = kwargs.get("content", "")

        if not to or not content:
            return ToolResult(
                error="Both 'to' and 'content' are required.",
                is_error=True,
            )

        current_names = self._resolve_agent_names()

        # Broadcast to all teammates (DM-only mode)
        if to == "all":
            if self._has_bbs:
                return ToolResult(
                    error="Broadcast is not available when BBS is active. "
                    "Use post_to_bbs to share with the whole team.",
                    is_error=True,
                )
            recipients = [n for n in current_names if n != self._sender]
            if not recipients:
                return ToolResult(
                    error="No other agents to broadcast to.",
                    is_error=True,
                )
            for name in recipients:
                try:
                    self._mailbox.send(
                        from_agent=self._sender,
                        to_agent=name,
                        content=content,
                        lane=DM_LANE_PEER,
                        message_type=DM_TYPE_PEER_MESSAGE,
                    )
                except ValueError:
                    pass
            preview = content[:80] + "..." if len(content) > 80 else content
            return ToolResult(
                output=f'Message broadcast to {len(recipients)} agent(s). Content: "{preview}"',
            )

        if to not in current_names:
            return ToolResult(
                error=f"Unknown agent '{to}'. Available: {current_names}",
                is_error=True,
            )

        if to == self._sender:
            return ToolResult(
                error="You cannot send a message to yourself.",
                is_error=True,
            )

        try:
            msg = self._mailbox.send(
                from_agent=self._sender,
                to_agent=to,
                content=content,
                lane=DM_LANE_PEER,
                message_type=DM_TYPE_PEER_MESSAGE,
            )
        except ValueError as exc:
            return ToolResult(error=str(exc), is_error=True)

        if self._peer_dm_summary and to != "leader":
            summary = kwargs.get("summary", "")
            if summary:
                self._mailbox.log_peer_summary(self._sender, to, summary)

        preview = content[:80] + "..." if len(content) > 80 else content
        return ToolResult(
            output=f'Message delivered to {to}. Content: "{preview}"',
            metadata={
                "message_id": msg.id,
                "lane": msg.lane,
                "message_type": msg.message_type,
            },
        )


class DynamicSendMessageTool(SendMessageTool):
    """SendMessageTool variant that resolves recipients dynamically.

    In dynamic scaling mode, new subagents may be spawned after the tool
    is created.  This subclass reads ``Mailbox.registered_names`` at call
    time instead of using a frozen list captured at init.
    """

    def parameters_schema(self) -> dict[str, Any]:
        if self._dynamic_names:
            return super().parameters_schema()
        current_names = sorted(
            n for n in self._mailbox.registered_names if n != self._sender
        )
        to_enum = current_names
        if not self._has_bbs:
            to_enum = ["all"] + current_names
        props: dict[str, Any] = {
            "to": {
                "type": "string",
                "description": "Name of the recipient agent, or 'all' to broadcast to every teammate.",
                "enum": to_enum,
            },
            "content": {
                "type": "string",
                "description": (
                    "The message to send. Be specific about what you need "
                    "the recipient to do (re-verify, adjust, check, etc.)."
                ),
            },
        }
        required = ["to", "content"]
        if self._peer_dm_summary:
            props["summary"] = {
                "type": "string",
                "description": (
                    "A 5-15 word summary of this message for the team leader's "
                    "visibility. Example: 'Asking Patricia to re-check revenue date filter'"
                ),
            }
            required.append("summary")
        return {
            "type": "object",
            "required": required,
            "properties": props,
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        # Refresh _agent_names from the live mailbox before dispatching
        self._agent_names = sorted(
            n for n in self._mailbox.registered_names if n != self._sender
        )
        return super().execute(**kwargs)


# ---------------------------------------------------------------------------
# ReadDMTool — check DM inbox
# ---------------------------------------------------------------------------


class ReadDMTool(BaseTool):
    """Check for new direct messages from other agents.

    Registered as a real tool so that auto-injected ``read_dm`` fake
    tool_use/tool_result pairs reference a known tool definition.
    Agents can also call it voluntarily mid-task.
    """

    def __init__(self, mailbox: Mailbox, agent_name: str) -> None:
        self._mailbox = mailbox
        self._agent_name = agent_name

    @property
    def name(self) -> str:
        return "read_dm"

    @property
    def description(self) -> str:
        return (
            "Check for new direct messages from other agents. Messages are "
            "also auto-delivered between tool calls, so you rarely need to "
            "call this manually."
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        messages = self._mailbox.check_new(self._agent_name)
        if not messages:
            return ToolResult(output="No new direct messages.")
        return ToolResult(output=self._mailbox.render_for_llm(messages))


# ---------------------------------------------------------------------------
# ClaimTaskTool
# ---------------------------------------------------------------------------


class ClaimTaskTool(BaseTool):
    """Claim a pending task from the task board."""

    def __init__(self, task_board: TaskBoard, agent_name: str) -> None:
        self._task_board = task_board
        self._agent_name = agent_name

    @property
    def name(self) -> str:
        return "claim_task"

    @property
    def description(self) -> str:
        return (
            "Claim a pending task from the shared task board. "
            "Only tasks whose dependencies are all completed can be claimed."
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["task_id"],
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "ID of the task to claim.",
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return ToolResult(error="'task_id' is required.", is_error=True)

        # Resolve by ID first, then fall back to name lookup
        task = self._task_board.resolve_task_id(task_id)
        if task is not None:
            task_id = task.id

        success = self._task_board.claim(task_id, self._agent_name)
        if success:
            self._task_board.mark_running(task_id)
            return ToolResult(output=f"Claimed task '{task_id}'")
        else:
            task = self._task_board.get_task(task_id)
            if task is None:
                return ToolResult(error=f"Task '{task_id}' not found.", is_error=True)
            return ToolResult(
                error=(
                    f"Cannot claim task '{task_id}' "
                    f"(status={task.status.value}, dependencies may not be met)."
                ),
                is_error=True,
            )


# ---------------------------------------------------------------------------
# CompleteTaskTool
# ---------------------------------------------------------------------------


class CompleteTaskTool(BaseTool):
    """Mark a task as completed with a summary.

    When a mailbox is configured (DM mode), the completion summary can be
    delivered to peers so they share context — analogous to BBS posts in
    BBS mode.
    """

    def __init__(
        self,
        task_board: TaskBoard,
        *,
        profile: str = "reasoning",
        mailbox: Mailbox | None = None,
        sender: str = "",
        broadcast: bool = True,
        on_complete_callback: "Callable[[str], None] | None" = None,
    ) -> None:
        self._task_board = task_board
        self._profile = profile
        self._mailbox = mailbox
        self._sender = sender
        self._broadcast = broadcast
        # Optional post-transition / pre-broadcast hook. Called with the
        # ``sender`` name immediately after the task_board state flips to
        # COMPLETED but *before* the broadcast ``task_completed`` DM goes
        # out to peers. The window between state-flip and broadcast is
        # the only safe place to surface side effects (e.g. a worktree
        # harvest DM) that the receiving peer should see BEFORE it learns
        # the task is done — otherwise the peer can race ahead on the
        # ``task_completed`` notification and submit a final report
        # before the side-effect data is available.
        #
        # Smoke evidence: harvest DMs wired to ``SubAgent.run_loop``'s
        # post-task hook (which fires AFTER the LLM turn ends) lost the
        # race in 4/10 trials — the leader saw the task_completed DM and
        # submitted before the auditor's run_loop got around to firing
        # the workspace cleanup.
        self._on_complete_callback = on_complete_callback

    @property
    def name(self) -> str:
        return "complete_task"

    @property
    def description(self) -> str:
        if self._profile == "browsing":
            return (
                "Mark a claimed task as completed with a detailed summary. "
                "The orchestrator relies on this summary to understand your work. "
                "Include: (1) the answer/finding with key numbers, "
                "(2) the web searches and sources you consulted, "
                "(3) source URLs for every claim, "
                "(4) any caveats or uncertainties."
            )
        return (
            "Mark a claimed task as completed with a detailed summary. "
            "The orchestrator relies on this summary to understand your work. "
            "Include: (1) the answer/finding with key numbers, "
            "(2) the SQL queries you ran (or methodology used), "
            "(3) data source (table/view names), "
            "(4) any caveats or uncertainties."
        )

    def parameters_schema(self) -> dict[str, Any]:
        if self._profile == "browsing":
            summary_desc = (
                "Detailed summary of findings. Include: key numbers/answer, "
                "web searches performed, source URLs, and any caveats or "
                "open questions. This is the orchestrator's primary window "
                "into your work."
            )
        else:
            summary_desc = (
                "Detailed summary of findings. Include: key numbers/answer, "
                "SQL queries or methodology used, data sources, and any "
                "caveats or open questions. This is the orchestrator's "
                "primary window into your work."
            )
        return {
            "type": "object",
            "required": ["task_name"],
            "properties": {
                "task_name": {
                    "type": "string",
                    "description": "Name of the task to complete (as shown in list_tasks).",
                },
                "summary": {
                    "type": "string",
                    "description": summary_desc,
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        task_name = kwargs.get("task_name", "") or kwargs.get("task_id", "")
        summary = kwargs.get("summary", "")
        if not task_name:
            return ToolResult(error="'task_name' is required.", is_error=True)

        # Resolve by ID first, then fall back to name lookup
        task = self._task_board.resolve_task_id(task_name)
        if task is None:
            task = self._task_board.get_task(task_name)
        if task is None:
            return ToolResult(error=f"Task '{task_name}' not found.", is_error=True)

        # Claude Code's ``completeAgentTask``/``enqueueAgentNotification``
        # invariant: mutate task-board state atomically *before* any DM
        # notification.  ``TaskBoard.complete`` now returns ``True`` only on
        # the actual non-terminal -> COMPLETED edge; on an already-terminal
        # task it returns ``False`` and we fall back to an append-only path
        # so a second call's polished summary is not silently dropped.
        transitioned = self._task_board.complete(task.id, summary=summary)

        # Fire post-transition / pre-broadcast hook on the actual edge.
        # Must happen BEFORE ``_send_result_dm`` so any side-effect DMs
        # the hook emits (e.g. worktree harvest) land in the peer's
        # mailbox ahead of the broadcast ``task_completed`` notification.
        # Idempotency-by-side-effect lives in the callback itself; this
        # site only guards against exceptions bubbling up and turning a
        # successful completion into a tool error.
        if transitioned and self._on_complete_callback is not None:
            try:
                self._on_complete_callback(self._sender)
            except Exception:
                logger.exception(
                    "CompleteTaskTool on_complete_callback raised for "
                    "sender=%r task=%r; broadcast will proceed",
                    self._sender, task.id,
                )

        if not transitioned:
            # Task was already terminal.  Two sub-cases:
            #   1. COMPLETED + non-empty summary: append as a new summary
            #      entry and emit a ``task_summary_updated`` DM (matches
            #      UpdateTaskSummaryTool semantics).  This recovers the
            #      second-complete_task summary that BBS Phase 3 produces.
            #   2. FAILED, or empty summary: preserve the original no-op
            #      behavior — do not resurrect failed tasks, do not send
            #      a DM with no content.
            if task.status == TaskStatus.COMPLETED and summary:
                appended = self._task_board.append_summary(
                    task.id, author=self._sender or task.claimed_by or "unknown",
                    content=summary,
                )
                if appended:
                    self._send_result_dm(
                        task=task,
                        task_name=task_name,
                        summary=summary,
                        message_type=DM_TYPE_TASK_SUMMARY_UPDATED,
                        label="Task updated",
                    )
                    return ToolResult(
                        output=(
                            f"Task '{task.id}' already completed; your "
                            f"summary was appended as update entry "
                            f"#{len(task.summaries)}. For future additions, "
                            f"prefer update_task_summary."
                        ),
                    )
            return ToolResult(
                output=(
                    f"Task '{task.id}' already in terminal state "
                    f"({task.status.value}); no notification sent."
                ),
            )

        if summary:
            self._send_result_dm(
                task=task,
                task_name=task_name,
                summary=summary,
                message_type=DM_TYPE_TASK_COMPLETED,
                label="Task completed",
            )

        return ToolResult(output=f"Task '{task.id}' marked as completed.")

    def _send_result_dm(
        self,
        *,
        task: TaskSpec,
        task_name: str,
        summary: str,
        message_type: str,
        label: str,
    ) -> None:
        """Broadcast a result-lane DM to peers.

        Shared between the first-time ``task_completed`` path and the
        fallback ``task_summary_updated`` path so the send loop is not
        duplicated.  Silently no-ops when no mailbox is configured.
        """
        if self._mailbox is None or not summary:
            return
        content = f"[{label}: {task_name}] {summary}"
        if self._broadcast:
            recipients = [r for r in self._mailbox.registered_names if r != self._sender]
        else:
            recipients = ["leader"]
        for recipient in recipients:
            try:
                self._mailbox.send(
                    from_agent=self._sender,
                    to_agent=recipient,
                    content=content,
                    lane=DM_LANE_RESULT,
                    message_type=message_type,
                    payload={
                        "task_id": task.id,
                        "task_name": task.name,
                        "status": task.status.value,
                    },
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# UpdateTaskSummaryTool
# ---------------------------------------------------------------------------


class UpdateTaskSummaryTool(BaseTool):
    """Append a correction or updated finding to a completed task's summary.

    When a mailbox is configured, updates can be broadcast to peers —
    mirroring :class:`CompleteTaskTool` completion notifications.
    """

    def __init__(
        self,
        task_board: TaskBoard,
        author: str,
        mailbox: Mailbox | None = None,
        sender: str = "",
        broadcast: bool = True,
    ) -> None:
        self._task_board = task_board
        self._author = author
        self._mailbox = mailbox
        self._sender = sender
        self._broadcast = broadcast

    @property
    def name(self) -> str:
        return "update_task_summary"

    @property
    def description(self) -> str:
        return (
            "Append a correction or updated finding to a completed task's "
            "summary. Use this when you receive new information (e.g., via DM) "
            "that changes or supplements your earlier findings. The orchestrator "
            "will see all summary entries."
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["task_name", "summary"],
            "properties": {
                "task_name": {
                    "type": "string",
                    "description": "Name of the completed task to update.",
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "The correction or updated finding to append. "
                        "Describe what changed and why."
                    ),
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        task_name = kwargs.get("task_name", "")
        summary = kwargs.get("summary", "")
        if not task_name:
            return ToolResult(error="'task_name' is required.", is_error=True)
        if not summary:
            return ToolResult(error="'summary' is required.", is_error=True)

        task = self._task_board.find_by_name(task_name)
        if task is None:
            task = self._task_board.get_task(task_name)
        if task is None:
            return ToolResult(error=f"Task '{task_name}' not found.", is_error=True)
        if task.status != TaskStatus.COMPLETED:
            # Allow updates on RUNNING infrastructure tasks (dynamic mode)
            is_infra_running = (
                task.status == TaskStatus.RUNNING
                and task.metadata.get("infrastructure", False)
            )
            if not is_infra_running:
                return ToolResult(
                    error=f"Task '{task_name}' is not completed (status={task.status.value}). "
                          "Only completed tasks can be updated.",
                    is_error=True,
                )

        self._task_board.append_summary(task.id, author=self._author, content=summary)

        if self._mailbox is not None and summary:
            content = f"[Task updated: {task_name}] {summary}"
            if self._broadcast:
                recipients = [r for r in self._mailbox.registered_names if r != self._sender]
            else:
                recipients = ["leader"]
            for recipient in recipients:
                try:
                    self._mailbox.send(
                        from_agent=self._sender,
                        to_agent=recipient,
                        content=content,
                        lane=DM_LANE_RESULT,
                        message_type=DM_TYPE_TASK_SUMMARY_UPDATED,
                        payload={
                            "task_id": task.id,
                            "task_name": task.name,
                            "status": task.status.value,
                        },
                    )
                except Exception:
                    pass

        return ToolResult(
            output=f"Summary updated for task '{task_name}' (entry #{len(task.summaries)})."
        )


# ---------------------------------------------------------------------------
# SwarmContext — shared mutable state for a swarm turn
# ---------------------------------------------------------------------------


class SwarmContext:
    """Shared mutable state for a single swarm turn.

    Bundles BBS, TaskBoard, AgentRegistry, ThreadPoolExecutor, and subagent
    lifecycle management.  Orchestration tools (``DynamicCreateTaskTool``, etc.)
    receive a reference to this context.

    Pre-spawned subagents run persistent loops claiming tasks from the board.
    The orchestrator posts tasks; subagents pick them up autonomously.
    """

    def __init__(
        self,
        bbs: BBS | None,
        task_board: TaskBoard,
        agent_registry: AgentRegistry,
        config: ArcticswarmConfig,
        pool: ThreadPoolExecutor,
        sf_client: SnowflakeClient | None,
        on_swarm_event: Callable[..., None] | None,
        question: str,
        max_teammates: int = 5,
        active_channels: frozenset[str] | None = None,
        mailbox: Mailbox | None = None,
        has_bbs: bool = True,
        has_dm: bool = False,
        system_reminder_interval: int = -1,
        dynamic_mode: bool = False,
        deadline: float | None = None,
        content_cache: Any | None = None,
        question_images: list[dict[str, Any]] | None = None,
    ) -> None:
        self.bbs = bbs
        self.task_board = task_board
        self.agent_registry = agent_registry
        self.config = config
        self.pool = pool
        self.sf_client = sf_client
        self.on_swarm_event = on_swarm_event
        self.question = question
        # Image content blocks from the original user question. Empty list
        # for text-only runs. Propagated to every SubAgent so subagents see
        # attached images on their first turn (e.g. image cases).
        self.question_images: list[dict[str, Any]] = list(question_images or [])
        self.max_teammates = max_teammates
        self.active_channels = active_channels
        self.mailbox = mailbox
        self.has_bbs = has_bbs
        self.has_dm = has_dm
        self.system_reminder_interval = system_reminder_interval
        self.dynamic_mode = dynamic_mode
        self.content_cache = content_cache

        self.subagents: list[SubAgent] = []
        self.futures: list[Future[None]] = []
        self._task_counter: int = 0
        self._lock = threading.Lock()
        self.shutdown = threading.Event()
        self.wrapping_up = threading.Event()
        self.winding_down = threading.Event()  # 90% timeout — subagents must wrap up

        # Dynamic-mode bookkeeping
        self._subagent_map: dict[str, SubAgent] = {}   # name -> SubAgent
        self._auditor_spawned: bool = False
        self._pending_queue: list[TaskSpec] = []

        # Spawn/assignment event log for dynamic-vs-static analysis.
        # Each entry records a spawn_or_assign decision (dynamic mode) or
        # a dequeue event when a queued task is assigned to a freed worker.
        self.spawn_events: list[dict[str, Any]] = []
        self._swarm_t0: float = time.monotonic()
        self.deadline: float | None = deadline

        # Track web sources from web_search tool calls
        self.web_sources = WebSourceTracker()

    def next_task_id(self) -> str:
        """Generate a unique task ID (thread-safe)."""
        with self._lock:
            self._task_counter += 1
            return f"task-{self._task_counter}"

    @property
    def tasks_created(self) -> int:
        """Number of tasks created so far."""
        return self.task_board.task_count

    # -- dynamic-mode helpers ------------------------------------------------

    def _make_callbacks(self, agent_name: str) -> tuple[Callable, Callable]:
        """Create on_event and on_status_change callbacks for a subagent."""
        from arcticswarm.agent import ToolCallStart
        from arcticswarm.swarm.orchestrator import (
            SubagentClaimedTask,
            SubagentIdle,
            SubagentSpawned,
            SubagentSurfing,
            TeammateToolCall,
            _summarize_tool_call,
        )

        def _on_tool_event(event: Any) -> None:
            if isinstance(event, ToolCallStart) and self.on_swarm_event:
                desc = _summarize_tool_call(event.tool_name, event.tool_input)
                self.on_swarm_event(TeammateToolCall(
                    name=agent_name,
                    tool_name=event.tool_name,
                    description=desc,
                ))

        def _on_status_change(name: str, status: str, activity: str) -> None:
            if self.on_swarm_event:
                if status == "working":
                    self.on_swarm_event(SubagentClaimedTask(name=name, activity=activity))
                elif status == "surfing":
                    self.on_swarm_event(SubagentSurfing(name=name, activity=activity))
                elif status == "idle":
                    self.on_swarm_event(SubagentIdle(name=name, activity=activity))

        return _on_tool_event, _on_status_change

    def _spawn_subagent(self, name: str, profile: str = "", config_override: "ArcticswarmConfig | None" = None) -> "SubAgent":
        """Spawn a single dynamic-mode subagent and start its loop."""
        from dataclasses import replace
        from arcticswarm.swarm.orchestrator import SubagentSpawned
        from arcticswarm.swarm.teammate import SubAgent

        self.agent_registry.register(name)
        if self.mailbox is not None:
            self.mailbox.register(name)

        if self.on_swarm_event:
            self.on_swarm_event(SubagentSpawned(name=name))

        on_event_cb, on_status_cb = self._make_callbacks(name)

        sa_config = config_override or self.config.for_subagent()

        subagent = SubAgent(
            name=name,
            config=sa_config,
            bbs=self.bbs,
            task_board=self.task_board,
            agent_registry=self.agent_registry,
            question=self.question,
            question_images=self.question_images,
            shutdown=self.shutdown,
            sf_client=self.sf_client,
            on_event=on_event_cb,
            on_status_change=on_status_cb,
            web_source_tracker=self.web_sources,
            active_channels=self.active_channels,
            mailbox=self.mailbox,
            has_bbs=self.has_bbs,
            has_dm=self.has_dm,
            system_reminder_interval=self.system_reminder_interval,
            dynamic_mode=True,
            initial_profile=profile,
            on_task_complete=self._on_worker_complete,
            content_cache=self.content_cache,
        )
        with self._lock:
            self.subagents.append(subagent)
            self._subagent_map[name] = subagent

        future = self.pool.submit(subagent.run_loop_dynamic)
        with self._lock:
            self.futures.append(future)

        return subagent

    def _log_spawn_event(
        self,
        task: TaskSpec,
        decision: str,
        assigned_to: str | None,
    ) -> None:
        """Record a spawn/assignment decision for post-hoc analysis."""
        idle_count = sum(
            1 for sa in self.subagents
            if sa._dynamic_mode
            and sa._pending_task is None
            and sa.agent_registry.get_status(sa.name) == AgentStatus.IDLE
        )
        self.spawn_events.append({
            "task_id": task.id,
            "task_name": task.name,
            "task_profile": task.profile or "reasoning",
            "decision": decision,
            "assigned_to": assigned_to,
            "timestamp": round(time.monotonic() - self._swarm_t0, 3),
            "active_subagents": len(self.subagents),
            "idle_subagents": idle_count,
            "queue_depth": len(self._pending_queue),
        })

    def spawn_or_assign(self, task: TaskSpec) -> str:
        """Assign *task* to an existing worker or spawn a new one.

        Returns the name of the assigned subagent.

        Selection priority:
        1. Idle worker with matching profile (unless mixing allowed).
        2. Idle worker with capacity (if mixing allowed).
        3. Spawn a new worker (up to hard cap).
        4. Queue the task (all workers busy and at cap).
        """
        task_profile = task.profile or "reasoning"

        # Auto-spawn auditor on first task creation.
        # Use reasoning profile for web-search swarms (auditor reviews
        # search findings); otherwise fall back to the task's profile.
        if not self._auditor_spawned and not getattr(self.config, "disable_auditor", False):
            auditor_profile = "reasoning" if self.config.has_web_search_capability() else task_profile
            self._spawn_auditor(profile=auditor_profile)

        profile = task_profile
        max_tasks = self.config.max_subagent_tasks

        with self._lock:
            # 1 — Try to find an idle worker with matching profile
            for sa in self.subagents:
                if not sa._dynamic_mode:
                    continue
                if sa._pending_task is not None:
                    continue
                if sa.agent_registry.get_status(sa.name) != AgentStatus.IDLE:
                    continue
                if max_tasks > 0 and sa._tasks_completed >= max_tasks:
                    continue
                if sa._initial_profile == profile:
                    sa.assign_task(task)
                    self._log_spawn_event(task, "reused_matching", sa.name)
                    return sa.name

            # 2 — Spawn a new worker (if under cap)
            hard_cap = self.config.max_subagents
            if len(self.subagents) < hard_cap:
                from arcticswarm.swarm.names import assign_names
                used = {sa.name for sa in self.subagents}
                new_names = assign_names(1, exclude=used)
                new_name = new_names[0]

        # Spawning happens outside the lock (may involve I/O)
        if len(self.subagents) < self.config.max_subagents:
            sa = self._spawn_subagent(new_name, profile=profile)
            sa.assign_task(task)
            self._log_spawn_event(task, "spawned_new", sa.name)
            return sa.name

        # 3 — At cap, queue the task
        with self._lock:
            self._pending_queue.append(task)
        self._log_spawn_event(task, "queued", None)
        return "(queued)"

    def _spawn_auditor(self, profile: str = "reasoning") -> None:
        """Spawn a dedicated auditor subagent.

        The auditor uses the same profile as workers (e.g. ``sql``) so it
        has SQL tools for idle-review verification.  It receives no
        explicit task — it picks up work through the idle review loop
        which triggers when other agents post findings to BBS.
        """
        if self._auditor_spawned:
            return
        self._auditor_spawned = True

        num_auditors = 1

        from arcticswarm.swarm.names import assign_names
        used = {sa.name for sa in self.subagents}
        names = assign_names(num_auditors, exclude=used)
        auditor_config = self.config.for_auditor()
        for n in names:
            sa = self._spawn_subagent(n, profile=profile, config_override=auditor_config)
            # Mark the auditor's post_to_bbs tool as the auditor instance.
            post_tool = sa.agent._tools.get("post_to_bbs")
            if isinstance(post_tool, PostToBBSTool):
                post_tool._is_auditor = True

    def _on_worker_complete(self, worker_name: str) -> None:
        """Callback when a dynamic-mode worker finishes a task.

        Drains the pending queue if there are waiting tasks, assigning
        the next queued task to the just-freed worker.
        """
        with self._lock:
            if not self._pending_queue:
                return
            task = self._pending_queue.pop(0)
        sa = self._subagent_map.get(worker_name)
        if sa is not None and sa._pending_task is None:
            sa.assign_task(task)
            self._log_spawn_event(task, "dequeued", sa.name)
        else:
            with self._lock:
                self._pending_queue.insert(0, task)

    def stop_infrastructure_tasks(self) -> None:
        """Mark any RUNNING infrastructure tasks as completed.

        Called by PrepareReportTool after the wait loop so that
        long-running background tasks do not block report generation.
        """
        for task in self.task_board.all_tasks():
            if (
                task.status == TaskStatus.RUNNING
                and task.metadata.get("infrastructure", False)
            ):
                self.task_board.complete(
                    task.id,
                    summary="(infrastructure task stopped at report time)",
                )

    def wait_and_cleanup(self, timeout: float = 300) -> None:
        """Wait for all tasks to finish, signal shutdown, clean up subagents."""
        # Wait for all board tasks to complete
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.task_board.task_count == 0 or self.task_board.all_completed():
                break
            # Early exit: pending tasks but no live subagents to claim them
            if (
                not self.subagents
                and any(
                    t.status == TaskStatus.PENDING
                    for t in self.task_board.all_tasks()
                )
            ):
                logger.warning(
                    "Pending tasks but no subagents alive — "
                    "exiting wait_and_cleanup early"
                )
                break
            time.sleep(0.5)

        # Signal all subagents to exit their loops
        self.shutdown.set()

        # Wait for subagent threads to finish (brief grace period)
        for future in self.futures:
            try:
                future.result(timeout=20)
            except Exception:
                pass

        # Close subagent resources
        for sa in self.subagents:
            try:
                sa.close()
            except Exception:
                pass


class DynamicCreateTaskTool(BaseTool):
    """Create a task and dispatch it via ``SwarmContext.spawn_or_assign``.

    When ``expose_blocking=True`` (set by the orchestrator for DM-realtime
    direct-report runs, see :func:`Orchestrator._setup_swarm_agent_tools`)
    the tool exposes two extra parameters:

    - ``blocking`` (default ``True``): wait inside this tool call until
      the spawned task reaches a terminal state (``COMPLETED`` /
      ``FAILED``), the per-call ``blocking_timeout`` elapses, or every
      tracked task becomes ``STALE``.  When the wait returns, the tool's
      ``ToolResult`` carries the subagent's ``complete_task`` summary
      inline so the leader's next reasoning step starts with the
      finding already in tool-result context.  This closes the
      leader-reviewer sequencing gap we measured on django-13023 +
      7 similar duo-vs-DM regressions, where the reviewer's findings
      arrived as a mailbox DM after the leader had already drafted its
      patch and so were dismissed as "non-blocking observations".
    - ``blocking_timeout`` (default ``_WAIT_DEFAULT_TIMEOUT``, capped at
      ``_WAIT_MAX_TIMEOUT``): per-call budget.  Independent across calls
      (no cumulative anchor like the harvest-stall budget on
      ``SendReportTool``).  On timeout the spawned task keeps running
      in the background and the LLM can re-wait via another
      ``create_task`` follow-up or ``list_tasks``.

    When ``expose_blocking=False`` (default, used by BBS-realtime and
    dm_exec) the tool behaves exactly as before: the schema does not
    include ``blocking``/``blocking_timeout`` and ``execute`` returns
    immediately after ``spawn_or_assign``.  Out-of-tool waiting on the
    mailbox / ``wait_for_tasks`` is still available there.
    """

    def __init__(
        self,
        ctx: SwarmContext,
        active_profiles: list[str] | None = None,
        *,
        has_web_search: bool = False,
        disable_bbs_isolation: bool = False,
        force_bbs_isolation: bool = False,
        expose_blocking: bool = False,
        enforce_alt_task: bool = True,
    ) -> None:
        self._ctx = ctx
        self._active_profiles: list[str] = active_profiles or []
        self._has_web_search = has_web_search
        self._disable_bbs_isolation = disable_bbs_isolation
        self._force_bbs_isolation = force_bbs_isolation
        self._expose_blocking = expose_blocking
        self._enforce_alt_task = enforce_alt_task

    @property
    def name(self) -> str:
        return "create_task"

    @property
    def description(self) -> str:
        return (
            "Create a task and assign it to a subagent. In dynamic scaling "
            "mode, a new subagent is spawned on demand if no idle worker is "
            "available (up to the hard cap). Use the 'profile' parameter to "
            "control what tools the executing subagent gets. Optionally use "
            "'assign_to' to target a specific subagent by name."
        )

    def parameters_schema(self) -> dict[str, Any]:
        from arcticswarm.swarm.profiles import get_profile, DEFAULT_PROFILE_NAME

        profile_names = self._active_profiles
        profile_descs = []
        for pname in profile_names:
            p = get_profile(pname)
            if p and p.orchestrator_description:
                profile_descs.append(f"'{pname}' = {p.orchestrator_description}")
        desc_text = " ".join(profile_descs) if profile_descs else ", ".join(profile_names)
        default = DEFAULT_PROFILE_NAME if DEFAULT_PROFILE_NAME in profile_names else (profile_names[0] if profile_names else "reasoning")

        schema: dict[str, Any] = {
            "type": "object",
            "required": ["name", "prompt", "profile"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Unique short name for the task "
                        "(e.g. 'revenue-analysis', 'web-researcher', 'verify-logic')."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Detailed, specific instructions for the task. "
                        "Include what to investigate, what BBS channel to post "
                        "results to, and what format to use."
                    ),
                },
                "profile": {
                    "type": "string",
                    "description": (
                        f"Tool profile for this task. Options: {', '.join(profile_names)}. "
                        f"{desc_text} "
                        f"Default: '{default}'."
                    ),
                    "enum": profile_names,
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of task *names* that must complete before this "
                        "task starts. Leave empty or omit for independent tasks."
                    ),
                },
                "assign_to": {
                    "type": "string",
                    "description": (
                        "Optional: name of a specific subagent to assign the "
                        "task to. Omit to let the system choose automatically."
                    ),
                },
            },
        }
        # Expose the ``alt`` (alternative/contrarian) option only when the
        # alt-task premature-commitment gate is enforced. When it is off
        # (ablation), the ``_check_alt_task_gate`` backstop is a no-op, so the
        # schema must not advertise "required before reporting" — mirrors the
        # ``isolated`` option gating below.
        if self._enforce_alt_task:
            schema["properties"]["alt"] = {
                "type": "boolean",
                "description": (
                    "Set true if this task explores an ALTERNATIVE or "
                    "CONTRARIAN hypothesis — i.e. it deliberately looks for "
                    "a candidate DIFFERENT from the team's current leading "
                    "answer. At least one such task is required before "
                    "reporting; marking it here (or naming the task with an "
                    "'alt'/'alternative'/'contrarian' token) satisfies that."
                ),
            }
        # Expose the per-task ``isolated`` choice only when isolation is
        # left to the orchestrator. Both ablation flags remove the choice:
        # ``disable`` => nothing is ever isolated; ``force`` => every
        # browsing task is auto-isolated by the harness. In either case the
        # leader has no decision to make, so keep the option out of the
        # schema (and out of the prompt — see build_orchestrator_system_prompt).
        if (
            self._has_web_search
            and not self._disable_bbs_isolation
            and not self._force_bbs_isolation
        ):
            schema["properties"]["isolated"] = {
                "type": "boolean",
                "description": (
                    "If true, the subagent executes this task WITHOUT "
                    "reading the BBS — it cannot see other agents' "
                    "findings and must search independently. Use "
                    "isolated=true for initial exploration tasks to "
                    "prevent premature convergence on a single candidate. "
                    "Use isolated=false (default) for verification, "
                    "cross-referencing, and follow-up tasks that benefit "
                    "from seeing prior findings."
                ),
            }
        if self._expose_blocking:
            # See class docstring for the rationale; in short, DM-realtime
            # leaders previously had no way to receive a subagent's
            # ``complete_task`` summary inline (the only path was a
            # mailbox DM in the next turn).  ``blocking=true`` plugs the
            # gap and matches Claude Code's ``Task`` tool default.
            schema["properties"]["blocking"] = {
                "type": "boolean",
                "description": (
                    "If true (DEFAULT), wait inside this tool call until "
                    "the subagent finishes (or fails / times out) and "
                    "return its complete_task summary as the tool result. "
                    "Use this for single dependent spawns — e.g. a "
                    "reviewer whose findings should drive your NEXT "
                    "reasoning step. Set blocking=false for parallel "
                    "fan-out (e.g. two 'author' candidates in one turn) "
                    "where you want to keep working while they run; in "
                    "that case findings arrive later as a "
                    "task-completion DM. Note: multiple blocking "
                    "calls emitted in the SAME tool-use turn serialize "
                    "at dispatch — for true parallelism, mark all but "
                    "one as blocking=false."
                ),
            }
            schema["properties"]["blocking_timeout"] = {
                "type": "integer",
                "description": (
                    f"Seconds to wait for the subagent to finish when "
                    f"blocking=true. Default {_WAIT_DEFAULT_TIMEOUT}, "
                    f"capped at {_WAIT_MAX_TIMEOUT}. On timeout the task "
                    "keeps running in the background; the tool returns "
                    "with a 'still running' note and you can re-wait "
                    "via another create_task follow-up or check "
                    "list_tasks. Ignored when blocking=false."
                ),
            }
        return schema

    def execute(self, **kwargs: Any) -> ToolResult:
        # Block new tasks once the soft deadline has fired.
        if self._ctx.wrapping_up.is_set():
            return ToolResult(
                error=(
                    "Time budget exhausted — cannot create new tasks. "
                    "Call prepare_report and then send_user_markdown_report "
                    "with your best candidate NOW."
                ),
                is_error=True,
            )

        task_name = kwargs.get("name", "")
        prompt = kwargs.get("prompt", "")
        depends_on_names: list[str] = kwargs.get("depends_on", []) or []
        profile: str = kwargs.get("profile", "")
        assign_to: str = kwargs.get("assign_to", "")
        # ``isolated`` here reflects only the orchestrator's EXPLICIT per-task
        # choice (baseline mode). The force_bbs_isolation ablation is NOT applied
        # here: it is enforced uniformly for every browsing-profile execution in
        # teammate.py (which also covers browsing tasks spawned outside this
        # tool), and its option is stripped from the schema above. disable wins.
        isolated: bool = bool(kwargs.get("isolated", False)) and not self._disable_bbs_isolation

        if not task_name or not prompt:
            return ToolResult(
                error="Both 'name' and 'prompt' are required.",
                is_error=True,
            )

        if not profile:
            return ToolResult(
                error="'profile' is required. Choose one of: "
                      + ", ".join(self._active_profiles),
                is_error=True,
            )

        if profile not in self._active_profiles:
            return ToolResult(
                error=(
                    f"Profile '{profile}' is not available for this run. "
                    f"Available: {', '.join(self._active_profiles)}"
                ),
                is_error=True,
            )

        if profile == "browsing" and not self._ctx.config.has_web_search_capability():
            return ToolResult(
                error="Profile 'browsing' requires a web search API key. Not configured.",
                is_error=True,
            )

        if self._ctx.task_board.find_by_name(task_name) is not None:
            return ToolResult(
                error=f"A task named '{task_name}' already exists.",
                is_error=True,
            )

        # Resolve depends_on names to task IDs
        depends_on_ids: list[str] = []
        for dep_name in depends_on_names:
            dep_task = self._ctx.task_board.find_by_name(dep_name)
            if dep_task is None:
                return ToolResult(
                    error=(
                        f"Dependency '{dep_name}' not found on the task board. "
                        f"Make sure you create dependent tasks first."
                    ),
                    is_error=True,
                )
            depends_on_ids.append(dep_task.id)

        task_id = self._ctx.next_task_id()
        task_metadata: dict[str, Any] = {}
        if isolated:
            task_metadata["isolated"] = True
        if kwargs.get("alt"):
            # Explicit alternative/contrarian marker (satisfies the alt-task
            # gate in PrepareReportTool). Name-token detection also applies.
            task_metadata["alt"] = True
        spec = TaskSpec(
            id=task_id,
            name=task_name,
            prompt=prompt,
            depends_on=depends_on_ids,
            profile=profile,
            metadata=task_metadata,
        )
        self._ctx.task_board.add_task(spec)

        # Mirror to BBS
        if self._ctx.bbs is not None:
            prompt_preview = prompt[:2000] + "..." if len(prompt) > 2000 else prompt
            deps_note = ""
            if depends_on_ids:
                deps_note = f" (depends on: {', '.join(depends_on_names)})"
            profile_note = f" [profile: {profile}]" if profile else ""
            self._ctx.bbs.post(
                channel="tasks",
                author="orchestrator",
                content=f"[task] {task_name}{profile_note}: {prompt_preview}{deps_note}",
            )

        # Dispatch via spawn_or_assign
        assigned = self._ctx.spawn_or_assign(spec)

        profile_msg = f" with profile '{profile}'" if profile else ""

        # ---- Optional blocking wait (DM-realtime direct-report only) -------
        # Gated on ``self._expose_blocking`` to keep BBS-realtime and
        # dm_exec callers unaffected.  ``blocking`` defaults to True so
        # the common single-spawn pattern returns the subagent's
        # ``complete_task`` summary inline as this tool's result —
        # closing the leader-reviewer sequencing gap diagnosed on
        # django-13023.  Leaders that want true parallel fan-out should
        # mark all but one call ``blocking=false``.
        if self._expose_blocking:
            should_block = bool(kwargs.get("blocking", True))
        else:
            should_block = False

        if should_block:
            requested_timeout = kwargs.get("blocking_timeout", _WAIT_DEFAULT_TIMEOUT)
            if requested_timeout is None:
                requested_timeout = _WAIT_DEFAULT_TIMEOUT
            wait_tool = WaitForTasksTool(self._ctx.task_board, swarm_ctx=self._ctx)
            wait_result = wait_tool.execute(
                task_names=[task_name],
                timeout=int(requested_timeout),
            )
            header = (
                f"Task '{task_name}' (id={task_id}) created{profile_msg} "
                f"and assigned to {assigned}. Blocked until completion:"
            )
            combined = header + "\n" + (wait_result.output or "")
            merged_meta: dict[str, Any] = {
                "task_id": task_id,
                "assigned_to": assigned,
                "blocking": True,
            }
            if wait_result.metadata:
                merged_meta.update(wait_result.metadata)
            return ToolResult(
                output=combined,
                metadata=merged_meta,
                is_error=wait_result.is_error,
            )

        return ToolResult(
            output=(
                f"Task '{task_name}' (id={task_id}) created{profile_msg} "
                f"and assigned to {assigned}."
            ),
            metadata={"task_id": task_id, "assigned_to": assigned},
        )


# ---------------------------------------------------------------------------
# ListTasksTool — orchestrator checks task status
# ---------------------------------------------------------------------------


def _render_task_activity_lines(
    task: TaskSpec,
    now: float,
    *,
    include_summary: bool = False,
) -> list[str]:
    """Render one task as orchestrator-friendly status + activity lines.

    Shared between :class:`WaitForTasksTool` and :class:`PrepareReportTool`
    so the orchestrator sees consistent heartbeat info wherever it asks.

    The returned list always starts with a ``"  <name>: <status>"`` line
    and may include indented follow-up lines for live activity, STALE
    flags, completion summaries (when ``include_summary=True``), and
    errors.  Callers typically ``extend(...)`` this into their own output
    buffer.
    """
    lines: list[str] = [f"  {task.name}: {task.status.value}"]
    if (
        task.status in (TaskStatus.CLAIMED, TaskStatus.RUNNING)
        and task.last_heartbeat > 0.0
    ):
        age = max(0, int(now - task.last_heartbeat))
        activity = task.last_activity_tool or "(waiting for tool call)"
        preview = (
            f" {task.last_activity_input}"
            if task.last_activity_input
            else ""
        )
        stale_tag = (
            " STALE"
            if age > STALE_HEARTBEAT_THRESHOLD_SECONDS
            else ""
        )
        lines.append(
            f"    activity: {activity}{preview} "
            f"({age}s ago, {task.tool_use_count} tool uses)"
            f"{stale_tag}"
        )
    if include_summary and task.summary:
        lines.append(f"    summary: {task.summary}")
    if task.error:
        lines.append(f"    error: {task.error}")
    return lines


class ListTasksTool(BaseTool):
    """Show the current status of all tasks on the board."""

    def __init__(self, task_board: TaskBoard) -> None:
        self._task_board = task_board

    @property
    def name(self) -> str:
        return "list_tasks"

    @property
    def description(self) -> str:
        return (
            "List all tasks on the shared task board with their current "
            "status (pending, running, completed, failed). Use this to "
            "monitor teammate progress before reviewing team updates."
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        status = self._task_board.render_status()
        return ToolResult(output=status)


# ---------------------------------------------------------------------------
# WaitForTasksTool — blocking wait for tasks to complete
# ---------------------------------------------------------------------------

_WAIT_POLL_INTERVAL = 2.0   # seconds between polls
_WAIT_DEFAULT_TIMEOUT = 1500  # 25 minutes
# NOTE: was 300 s; raised after gap analysis on browsecomp showed the
# orchestrator was burning ~50% of its 6300 s wallclock looping on the
# 5-minute timeout while subagents were still doing legitimate research.
_WAIT_MAX_TIMEOUT = 1800      # hard ceiling regardless of LLM request


class WaitForTasksTool(BaseTool):
    """
    Polls the task board internally every few seconds.  The tool call
    does not return until all requested tasks have reached a terminal
    state (completed or failed), the timeout is exceeded, or every
    task has become STALE (progress heartbeat older than the
    board-wide staleness threshold).  Prevents the orchestrator from
    burning LLM round-trips on repeated status checks while also
    preventing it from burning *wall-clock* budget waiting on a
    subagent that has silently stopped making progress.
    """

    def __init__(self, task_board: TaskBoard, swarm_ctx: "SwarmContext | None" = None) -> None:
        self._task_board = task_board
        self._swarm_ctx = swarm_ctx

    @property
    def name(self) -> str:
        return "wait_for_tasks"

    @property
    def description(self) -> str:
        return (
            "Block until one or more tasks finish. The call waits "
            "(up to the timeout) for every listed task to reach a "
            "terminal state (completed or failed), then returns the "
            "final status and summary for each task. Use this after "
            "creating tasks to wait for subagents to finish their work. "
            "Returns early with a STALLED warning if every named task "
            "stops making tool-use progress (heartbeat stale for > "
            "the board staleness threshold)."
        )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["task_names"],
            "properties": {
                "task_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task names to wait for.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Maximum seconds to wait (default "
                        f"{_WAIT_DEFAULT_TIMEOUT}, capped at "
                        f"{_WAIT_MAX_TIMEOUT}). If the timeout is "
                        "reached before all tasks finish, the current "
                        "status is returned with a warning. "
                        "For heavy research subagents, "
                        "prefer the default; for quick tasks you can pass a "
                        "smaller value."
                    ),
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        task_names: list[str] = kwargs.get("task_names", [])
        requested_timeout: int = (
            kwargs.get("timeout", _WAIT_DEFAULT_TIMEOUT) or _WAIT_DEFAULT_TIMEOUT
        )
        # Cap so an LLM that passes ``timeout=3600`` cannot starve the
        # orchestrator's wall-clock budget.
        timeout = max(1, min(int(requested_timeout), _WAIT_MAX_TIMEOUT))

        if not task_names:
            return ToolResult(
                error="'task_names' is required and must be non-empty.",
                is_error=True,
            )

        # If wrapping up, skip the blocking wait — return current status
        # immediately so the orchestrator can move to prepare_report.
        _wrapping_up = (
            self._swarm_ctx is not None
            and self._swarm_ctx.wrapping_up.is_set()
        )

        # Validate that all names exist
        for tn in task_names:
            if self._task_board.find_by_name(tn) is None:
                return ToolResult(
                    error=f"Task '{tn}' not found on the task board.",
                    is_error=True,
                )

        # ---- Poll until all tasks reach a terminal state or go stale -------
        deadline = time.monotonic() + timeout
        timed_out = False
        stalled_exit = False
        task_ids = [
            t.id
            for t in (
                self._task_board.find_by_name(tn) for tn in task_names
            )
            if t is not None
        ]

        if not _wrapping_up:
            while True:
                all_done = True
                for tn in task_names:
                    task = self._task_board.find_by_name(tn)
                    if task is not None and task.status not in (
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                    ):
                        all_done = False
                        break

                if all_done:
                    break

                if time.monotonic() >= deadline:
                    timed_out = True
                    break

                # Also break out if wrapping_up fires mid-wait.
                if (self._swarm_ctx is not None
                        and self._swarm_ctx.wrapping_up.is_set()):
                    timed_out = True
                    break

                # Early-exit: every still-running task's last tool-use event is
                # older than the staleness threshold (configured on TaskSpec).
                # Prevents the orchestrator from wasting remaining budget when
                # subagents are definitely stuck.
                if (
                    hasattr(self._task_board, "any_running_stale")
                    and task_ids
                    and self._task_board.any_running_stale(task_ids)
                ):
                    stalled_exit = True
                    break

                time.sleep(_WAIT_POLL_INTERVAL)

        # ---- Build result summary ------------------------------------------
        lines: list[str] = []
        now = time.monotonic()
        for tn in task_names:
            task = self._task_board.find_by_name(tn)
            if task is None:
                lines.append(f"  {tn}: NOT FOUND")
                continue
            lines.extend(
                _render_task_activity_lines(task, now, include_summary=True)
            )

        if stalled_exit:
            lines.append(
                "\nSTALLED: every task has been idle (no tool-use "
                "progress) for longer than the staleness threshold. "
                "Consider marking the tasks failed and re-delegating, "
                "or finalizing with partial results."
            )
        elif timed_out:
            lines.append(
                f"\nWARNING: Timed out after {timeout}s. Some tasks "
                "are still in progress."
            )

        if _wrapping_up:
            lines.append(
                "\nTime budget exhausted. Call prepare_report and then "
                "send_user_markdown_report with your best candidate NOW."
            )

        return ToolResult(output="\n".join(lines))


# ---------------------------------------------------------------------------
# PrepareReportTool — blocks until all work is done, then unlocks the report
# ---------------------------------------------------------------------------

_PREPARE_POLL_INTERVAL = 2.0   # seconds between polls
_PREPARE_DEFAULT_TIMEOUT = 300  # 5 minutes (overridable per-instance via constructor)

# After first detecting "all idle", wait this many seconds and re-check.
# This gives subagents time to do their idle BBS review (which toggles them
# briefly to SURFING) before we commit to generating the report.
_SETTLE_DELAY = 8.0


class PrepareReportTool(BaseTool):
    """Block until all tasks are complete and all subagents are idle.

    The orchestrator calls this tool when it believes enough work has been
    delegated.  The tool polls the task board and agent registry every few
    seconds.  Once all tasks have reached a terminal state (completed or
    failed) **and** every subagent is idle, it dynamically registers the
    ``send_user_markdown_report`` tool so the orchestrator can deliver the
    final report.

    While this tool is blocking, subagents continue to finish up and post
    to the BBS.  After the tool returns, ``_maybe_inject_bbs_update()``
    delivers all remaining BBS messages to the orchestrator, giving the
    LLM full context before it writes the report.
    """

    def __init__(
        self,
        task_board: TaskBoard,
        agent_registry: AgentRegistry,
        report_tool: "SendReportTool",
        agent_tools: dict[str, BaseTool],
        bbs: "BBS | None" = None,
        is_followup: bool = False,
        web_source_tracker: Any | None = None,
        swarm_ctx: "SwarmContext | None" = None,
        realtime: bool = False,
        mailbox: Mailbox | None = None,
        agent_name: str | None = None,
        enable_force_submit: bool = False,
        blocking: bool = True,
        late_register_tools: dict[str, BaseTool] | None = None,
        default_timeout: int = _PREPARE_DEFAULT_TIMEOUT,
        min_dedicated_reviewers: int = 0,
        min_builder_reviewers: int = 0,
        max_reviewer_remediations: int = 2,
        has_web_search: bool = False,
        enforce_alt_task: bool = False,
        question_text: str = "",
        surface_bbs_candidates: bool = False,
    ) -> None:
        self._task_board = task_board
        self._agent_registry = agent_registry
        self._report_tool = report_tool
        self._agent_tools = agent_tools
        self._bbs = bbs
        self._is_followup = is_followup
        self._web_source_tracker = web_source_tracker
        self._swarm_ctx = swarm_ctx
        self._realtime = realtime
        self._mailbox = mailbox
        self._agent_name = agent_name
        self._deadline_exceeded = False
        # Tools to register on the orchestrator alongside
        # ``send_user_markdown_report`` once ``execute()`` decides to
        # unlock the report path.  Empty by default.
        self._late_register_tools: dict[str, BaseTool] = (
            late_register_tools or {}
        )
        # When False (default), hides the ``force`` knob from the schema and
        # silently drops ``force=True`` if passed anyway.  Duo-style configs
        # leave this False so the leader cannot short-circuit the
        # wait-for-teammate barrier — there is only one teammate in duo mode,
        # so the "straggler" concept that motivates force= does not apply.
        # See :meth:`execute` for the enforcement. YAML:
        # ``swarm.enable_force_submit: true``.
        self._enable_force_submit = enable_force_submit
        # When False, the realtime mailbox path skips ``wait_for_message``
        # and returns immediately with a snapshot of task state + any
        # pending DMs.  Matches the Claude-Code agent model: messages are
        # delivered between tool rounds by a background poll, so the
        # leader never needs to sleep inside a tool call.  Saves hundreds
        # of seconds of wall-clock per case (see YAML:
        # ``swarm.blocking_prepare_report: false``).
        self._blocking = blocking
        # Per-instance default for the wait-loop timeout, overridable by
        # the LLM via the tool's ``timeout`` argument. Wired from
        # ``EvalConfig.prepare_timeout`` (yaml: ``eval.prepare_timeout``).
        self._default_timeout = default_timeout
        # Reviewer-diversity gate (web-research swarms). Require at least
        # ``_min_builder_reviewers`` VERIFIED #consensus verdicts from builders
        # (subagents that did first-hand web search) AND
        # ``_min_dedicated_reviewers`` from dedicated reviewers (reasoning
        # auditors reviewing from the BBS) before unlocking the report. When a
        # source is short, the gate auto-spawns a targeted reviewer task and
        # blocks (via the existing wait loop on retry), bounded by
        # ``_max_reviewer_remediations`` rounds then advisory-degrade. 0/0 =
        # disabled. ``_has_web_search`` scopes the gate to web runs. See
        # :meth:`_check_reviewer_diversity_gate`.
        self._min_dedicated_reviewers = int(min_dedicated_reviewers)
        self._min_builder_reviewers = int(min_builder_reviewers)
        self._max_reviewer_remediations = int(max_reviewer_remediations)
        self._has_web_search = has_web_search
        # Number of remediation rounds (spawns) the gate has already issued
        # this turn; bounds the spawn-and-wait loop so it never hangs.
        self._reviewer_remediation_attempts = 0
        # Set to a non-empty WARNING when the gate advisory-degrades (budget /
        # deadline exhausted); appended to the final prepare_report output so
        # the orchestrator (and trajectory) records that diversity was not met.
        self._reviewer_degrade_note = ""
        # Premature-commitment guard (web/BBS runs). When True, the
        # ``_check_alt_task_gate`` refuses to unlock send_user_markdown_report
        # until at least one ALTERNATIVE/CONTRARIAN task exists on the board,
        # auto-spawning one if the orchestrator never opened one. Targets the
        # arcticswarm "premature commitment correlates with failure" finding.
        # ``_has_web_search`` scopes the gate to web runs (set by the caller as
        # ``enforce_alt_task and has_web_search``). ``_alt_gate_spawned`` bounds
        # it to a single auto-spawn; ``_alt_gate_degrade_note`` records a WARNING
        # when the guard degrades (deadline / could-not-spawn) so it never hangs.
        self._enforce_alt_task = enforce_alt_task
        self._question_text = question_text or ""
        self._alt_gate_spawned = False
        self._alt_gate_degrade_note = ""
        # answer-retention: append a deterministic BBS candidate digest to
        # the report-unlock message so a found-but-compacted-away answer is
        # re-surfaced verbatim before the final answer is written. Default off.
        self._surface_bbs_candidates = surface_bbs_candidates

    @property
    def name(self) -> str:
        return "prepare_report"

    @property
    def description(self) -> str:
        force_hint = (
            " Pass force=true to skip waiting and report immediately "
            "with partial data."
            if self._enable_force_submit else ""
        )
        no_force_hint = (
            "" if self._enable_force_submit else
            " You MUST wait for the teammate(s) to finish — do not try to "
            "short-circuit this barrier, the teammate's review is the "
            "whole point of this mode."
        )
        if self._realtime and self._mailbox is not None:
            if not self._blocking:
                # Non-blocking (Claude-Code-style) snapshot mode: this tool
                # returns immediately with current status + any pending DMs.
                # Teammate messages are also delivered automatically between
                # your tool rounds — you do NOT need to poll aggressively.
                return (
                    "Snapshot the current state of your teammate's work. "
                    "Returns IMMEDIATELY (non-blocking) with: the task "
                    "status, any new messages from your teammate since "
                    "your last call, and — when all tasks are complete "
                    "— unlocks send_user_markdown_report. Teammate "
                    "messages are ALSO delivered automatically between "
                    "your tool rounds, so you don't need to poll in a "
                    "tight loop. Call this when you want an explicit "
                    "status check or believe you have enough data to "
                    "finalise."
                    + no_force_hint
                )
            return (
                "Check whether all tasks are complete. This call will "
                "wait for your teammate to send findings before returning "
                "— it may take a while. When all tasks are done it "
                "unlocks the send_user_markdown_report tool. Call this "
                "when you believe you have enough data to write the "
                "report."
                + force_hint + no_force_hint
            )
        if self._realtime:
            return (
                "Check whether all tasks are complete and all subagents "
                "are idle. If not ready, returns the pending tasks so you "
                "can wait for more DMs from teammates. If ready, unlocks "
                "the send_user_markdown_report tool. Call this when you "
                "believe you have enough data to write the report."
                + force_hint + no_force_hint
            )
        return (
            "Wait until all tasks are complete and all subagents are idle, "
            "then unlock the send_user_markdown_report tool. Call this BEFORE "
            "writing your final report. The call blocks until the swarm is "
            "ready — you will receive any remaining team updates after it "
            "returns. Once it succeeds, call send_user_markdown_report to "
            "deliver the report."
            + no_force_hint
        )

    def parameters_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "timeout": {
                "type": "integer",
                "description": (
                    f"Maximum seconds to wait (default {self._default_timeout}). "
                    "If the timeout is reached, the current status is returned "
                    "with a warning."
                ),
            },
        }
        if self._enable_force_submit:
            properties["force"] = {
                "type": "boolean",
                "description": (
                    "When true, skip waiting for remaining tasks and "
                    "immediately unlock send_user_markdown_report so "
                    "you can report with partial data. Use when you "
                    "already have strong results from completed tasks "
                    "and remaining tasks are stragglers."
                ),
            }
        return {"type": "object", "properties": properties}

    def execute(self, **kwargs: Any) -> ToolResult:
        # On follow-up turns with no new tasks, skip the wait and
        # immediately enable the report tool for light editing.
        if self._task_board.task_count == 0:
            if not self._is_followup:
                return ToolResult(
                    error=(
                        "No tasks have been created yet. You must delegate "
                        "work to subagents via create_task before preparing "
                        "a report."
                    ),
                    is_error=True,
                )
            # Follow-up turn with no new tasks — light edit path.
            # Build references from the full (persistent) BBS.
            ref_section = ""
            if self._bbs is not None:
                registry = ReferenceRegistry.from_bbs(self._bbs, web_source_tracker=self._web_source_tracker)
                if len(registry) > 0:
                    ref_section = "\n" + registry.render_for_prompt()
                    self._report_tool.reference_registry = registry

            self._agent_tools["send_user_markdown_report"] = self._report_tool
            for _name, _tool in self._late_register_tools.items():
                self._agent_tools[_name] = _tool
            return ToolResult(
                output=(
                    "No new tasks this turn. The send_user_markdown_report "
                    "tool is now available. Take the previous report from "
                    "your conversation history, apply only the user's "
                    "requested changes, and call send_user_markdown_report "
                    "with the full updated report."
                    + ref_section
                ),
            )

        timeout: int = kwargs.get("timeout", self._default_timeout) or self._default_timeout
        force: bool = bool(kwargs.get("force", False)) if self._enable_force_submit else False

        # When ``enable_force_submit`` is disabled (e.g. duo-style configs set
        # ``swarm.enable_force_submit: false``), reject ``force=True`` even if
        # the LLM passes it.  The schema hides the knob, but some providers
        # occasionally forward unknown args anyway (provider-dependent tool-
        # call serialisation).  We silently downgrade here rather than erroring
        # — the goal is to prevent bypass, not surface a noisy failure.
        if not self._enable_force_submit and force:
            logger.info(
                "[prepare_report] ignoring force=True; the teammate's "
                "review is required before reporting (enable_force_submit=False)."
            )
            force = False

        if force:
            non_infra_tasks = [
                t for t in self._task_board.all_tasks()
                if not t.metadata.get("infrastructure", False)
            ]
            tasks_done = (
                len(non_infra_tasks) == 0
                or all(
                    t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                    for t in non_infra_tasks
                )
            )
            agents_idle = self._agent_registry.all_idle()
            timed_out = not (tasks_done and agents_idle)
        elif self._realtime:
            non_infra_tasks = [
                t for t in self._task_board.all_tasks()
                if not t.metadata.get("infrastructure", False)
            ]
            tasks_done = (
                len(non_infra_tasks) == 0
                or all(
                    t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                    for t in non_infra_tasks
                )
            )
            agents_idle = self._agent_registry.all_idle()

            if not (tasks_done and agents_idle) and not self._deadline_exceeded:
                if self._mailbox is not None and self._agent_name is not None:
                    # Non-blocking mode: drain any currently-pending DMs and
                    # return immediately.  The teammate's subsequent DMs are
                    # still delivered via ``_auto_dm_check`` between the
                    # leader's tool rounds (same mechanism Claude Code uses
                    # with its 1s inbox poller), so we're not losing any
                    # information — we're just not burning wall clock
                    # sleeping in ``wait_for_message``.
                    if not self._blocking:
                        dm_msgs = self._mailbox.check_new(self._agent_name) or []
                        dm_text = ""
                        if dm_msgs:
                            dm_text = (
                                "\n\nYour teammate sent new message(s) "
                                "since your last check:\n\n"
                                + self._mailbox.render_for_llm(dm_msgs)
                            )
                        return ToolResult(
                            output=self._build_not_ready_output(
                                agents_idle=agents_idle,
                                dm_text=dm_text,
                                guidance=(
                                    "Status check returned immediately "
                                    "(non-blocking). Continue your own "
                                    "analysis — new teammate DMs will be "
                                    "injected between your tool rounds "
                                    "automatically — or call "
                                    "prepare_report again later to poll "
                                    "for task completion."
                                ),
                            ),
                        )
                    got_dm = self._mailbox.wait_for_message(
                        self._agent_name,
                        timeout=timeout,
                    )
                    if not got_dm:
                        self._deadline_exceeded = True
                    else:
                        dm_msgs = self._mailbox.check_new(self._agent_name)
                        non_infra_tasks = [
                            t for t in self._task_board.all_tasks()
                            if not t.metadata.get("infrastructure", False)
                        ]
                        tasks_done = (
                            len(non_infra_tasks) == 0
                            or all(
                                t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                                for t in non_infra_tasks
                            )
                        )
                        agents_idle = self._agent_registry.all_idle()
                        if not (tasks_done and agents_idle):
                            dm_text = ""
                            if dm_msgs:
                                dm_text = (
                                    "\n\nHowever, your teammate sent you a message:\n\n"
                                    + self._mailbox.render_for_llm(dm_msgs)
                                )
                            return ToolResult(
                                output=self._build_not_ready_output(
                                    agents_idle=agents_idle,
                                    dm_text=dm_text,
                                    guidance=(
                                        "Review their findings. Continue your "
                                        "analysis or call prepare_report again "
                                        "when ready."
                                    ),
                                ),
                            )
                else:
                    return ToolResult(
                        output=self._build_not_ready_output(
                            agents_idle=agents_idle,
                            dm_text="",
                            guidance=(
                                "Wait for more DMs from teammates, then try "
                                "again."
                            ),
                        ),
                    )
            timed_out = self._deadline_exceeded and not (tasks_done and agents_idle)
        else:
            deadline = time.monotonic() + timeout
            timed_out = False

            # Two-phase wait:
            #   Phase 1 — poll until tasks are done and agents are idle.
            #   Phase 2 — "settling" period.  After first seeing all-idle, wait
            #     _SETTLE_DELAY seconds and re-check.  If any agent went back to
            #     SURFING (idle BBS review), reset and go back to Phase 1.  This
            #     ensures agents have time to do their post-task discussion before
            #     we snapshot BBS and write the report.
            settle_start: float | None = None

            while True:
                # Exclude infrastructure tasks from the completion check
                non_infra_tasks = [
                    t for t in self._task_board.all_tasks()
                    if not t.metadata.get("infrastructure", False)
                ]
                tasks_done = (
                    len(non_infra_tasks) == 0
                    or all(
                        t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                        for t in non_infra_tasks
                    )
                )
                agents_idle = self._agent_registry.all_idle()

                if tasks_done and agents_idle:
                    if settle_start is None:
                        settle_start = time.monotonic()
                    elif time.monotonic() - settle_start >= _SETTLE_DELAY:
                        break
                else:
                    settle_start = None

                if time.monotonic() >= deadline:
                    timed_out = True
                    break

                time.sleep(_PREPARE_POLL_INTERVAL)

        # Stop infrastructure tasks in dynamic mode
        if self._swarm_ctx is not None:
            self._swarm_ctx.stop_infrastructure_tasks()

        # Build a status summary for the orchestrator
        lines: list[str] = []
        for task in self._task_board.all_tasks():
            line = f"  {task.name}: {task.status.value}"
            if task.summary:
                summary = task.summary
                line += f" — {summary}"
            if task.error:
                line += f" [error: {task.error}]"
            lines.append(line)

        if timed_out:
            lines.append(
                f"\nWARNING: Timed out after {timeout}s. Some tasks or "
                "subagents may still be in progress."
            )

        # Build a reference registry from BBS data so the LLM can cite sources
        ref_section = ""
        if self._bbs is not None:
            registry = ReferenceRegistry.from_bbs(self._bbs, web_source_tracker=self._web_source_tracker)
            if len(registry) > 0:
                ref_section = "\n" + registry.render_for_prompt()
                # Share the registry with the report tool for safety-net footer
                self._report_tool.reference_registry = registry

        # Always register the report tool — even on timeout the orchestrator
        # must be able to deliver a report with whatever data was collected.
        # Reviewer-diversity gate (web-research swarms): require a VERIFIED
        # #consensus verdict from BOTH a builder and a dedicated reviewer
        # before unlocking the report.  A non-None return is a refusal that
        # leaves send_user_markdown_report unregistered, forcing the
        # orchestrator to retry prepare_report after the auto-spawned
        # reviewer(s) finish.  No-op for non-web runs and when the feature
        # is disabled.
        reviewer_msg = self._check_reviewer_diversity_gate(
            force=force, timed_out=timed_out,
        )
        if reviewer_msg is not None:
            return ToolResult(output=reviewer_msg)
        # Premature-commitment guard: refuse to unlock the report until the
        # swarm has opened (or the harness has auto-spawned) at least one
        # alternative/contrarian task.  Like the gates above, a non-None
        # return leaves send_user_markdown_report unregistered, forcing the
        # orchestrator to retry prepare_report after the contrarian task
        # finishes.  No-op when disabled / non-web / followup / degraded.
        alt_msg = self._check_alt_task_gate(force=force, timed_out=timed_out)
        if alt_msg is not None:
            return ToolResult(output=alt_msg)
        self._agent_tools["send_user_markdown_report"] = self._report_tool
        for _name, _tool in self._late_register_tools.items():
            self._agent_tools[_name] = _tool

        status_preamble = (
            "All tasks are complete and all subagents are idle."
            if not timed_out
            else (
                "Timed out waiting for all work to finish, but the "
                "send_user_markdown_report tool is now available. "
                "Write the report using the data collected so far."
            )
        )

        return ToolResult(
            output=(
                f"{status_preamble} "
                "The send_user_markdown_report tool is now available.\n"
                "Task summary:\n" + "\n".join(lines)
                + ref_section
                + self._reviewer_degrade_note
                + self._alt_gate_degrade_note
                + self._build_candidate_digest()
            ),
        )

    def _build_candidate_digest(self) -> str:
        """deterministic BBS candidate digest.

        Returns a text block summarizing the candidate(s) the team converged on,
        harvested from the BBS (VERIFIED ``#consensus`` verdicts first, then top
        ``#key-findings`` / ``#discoveries`` posts). Appended to the report-unlock
        message so a finding that history-compaction or BBS burst-truncation
        dropped from the leader's working context is re-surfaced verbatim right
        before the final answer is written. Empty string when disabled, no BBS,
        or nothing on the board. Never raises (best-effort).
        """
        if not self._surface_bbs_candidates or self._bbs is None:
            return ""
        try:
            seen: set[str] = set()
            verified: list[str] = []
            for m in self._bbs.read(channel=CHANNEL_CONSENSUS, limit=60):
                content = (getattr(m, "content", "") or "").strip()
                if content and is_verified_consensus_verdict(content):
                    key = content[:160]
                    if key not in seen:
                        seen.add(key)
                        verified.append(f"- [VERIFIED by {getattr(m, 'author', '?')}] {content[:400]}")
            findings: list[str] = []
            for ch in (CHANNEL_KEY_FINDINGS, CHANNEL_DISCOVERIES):
                for m in self._bbs.read(channel=ch, limit=40):
                    content = (getattr(m, "content", "") or "").strip()
                    if not content:
                        continue
                    key = content[:160]
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(f"- [{ch}/{getattr(m, 'author', '?')}] {content[:300]}")
            if not verified and not findings:
                return ""
            out = [
                "\n\n=== CANDIDATE FINDINGS THE TEAM CONVERGED ON (from the shared "
                "blackboard) ===",
                "Use this as the authoritative source for your FINAL ANSWER. A definite "
                "answer exists; commit to the single candidate that satisfies the MOST "
                "hard constraints. Do NOT answer 'no answer found' if any candidate "
                "below plausibly fits.",
            ]
            if verified:
                out.append("\nVERIFIED consensus verdicts:")
                out.extend(verified[:8])
            if findings:
                out.append("\nKey findings / discoveries:")
                out.extend(findings[:12])
            return "\n".join(out)
        except Exception:  # never let the digest break report unlock
            return ""

    def _check_reviewer_diversity_gate(
        self, *, force: bool, timed_out: bool,
    ) -> str | None:
        """Reviewer-diversity gate (web-research swarms).

        Require a VERIFIED ``#consensus`` verdict from BOTH a *builder*
        reviewer (a subagent that did first-hand web investigation) and a
        *dedicated* reviewer (a reasoning auditor reviewing from the BBS)
        before ``send_user_markdown_report`` is unlocked.  Eliminates the
        builder-only / dedicated-only / no-reviewer outcomes.

        Returns a refusal string (report stays locked) when a source is
        missing — after auto-spawning a targeted reviewer task for it, so the
        orchestrator's next ``prepare_report`` call blocks on the existing
        two-phase wait until that reviewer posts.  Returns ``None`` when the
        gate is satisfied, disabled, not applicable (hybrid / non-web), or has
        exhausted its remediation budget / the run timed out (advisory degrade
        — never hang).
        """
        # Disabled?
        if max(self._min_dedicated_reviewers, self._min_builder_reviewers) <= 0:
            return None
        if not self._has_web_search:
            return None
        if self._bbs is None or self._swarm_ctx is None:
            return None
        # Force escape hatch.
        if force and timed_out:
            return None

        n_builder, n_dedicated = self._classify_consensus_reviewers()
        need_builder = n_builder < self._min_builder_reviewers
        need_dedicated = n_dedicated < self._min_dedicated_reviewers
        if not (need_builder or need_dedicated):
            return None  # both sources satisfied

        missing = (
            (["dedicated"] if need_dedicated else [])
            + (["builder"] if need_builder else [])
        )

        # Advisory-degrade: never block past the run deadline or the
        # remediation budget.  Unlock with a recorded WARNING.
        if (
            timed_out
            or self._reviewer_remediation_attempts >= self._max_reviewer_remediations
        ):
            self._reviewer_degrade_note = (
                "\n\nWARNING: reviewer-diversity not fully satisfied "
                f"(missing a VERIFIED #consensus verdict from: {', '.join(missing)}; "
                f"have builder={n_builder}/{self._min_builder_reviewers}, "
                f"dedicated={n_dedicated}/{self._min_dedicated_reviewers}). "
                "Proceeding after exhausting the reviewer remediation budget."
            )
            logger.info(
                "[reviewer_gate] advisory-degrade (attempts=%d, timed_out=%s): "
                "builder=%d/%d dedicated=%d/%d",
                self._reviewer_remediation_attempts, timed_out,
                n_builder, self._min_builder_reviewers,
                n_dedicated, self._min_dedicated_reviewers,
            )
            return None

        # Spawn the missing source(s) and refuse this round.
        spawned: list[tuple[str, str]] = []
        for kind in missing:
            name = self._spawn_reviewer_task(kind)
            if name:
                spawned.append((kind, name))
        self._reviewer_remediation_attempts += 1

        if not spawned:
            # Could not spawn (no swarm_ctx / cap+queue) — degrade rather than
            # refuse forever.
            self._reviewer_degrade_note = (
                "\n\nWARNING: reviewer-diversity not satisfied and no reviewer "
                f"could be spawned for: {', '.join(missing)}. Proceeding."
            )
            return None

        spawn_desc = "; ".join(f"a {src} reviewer (task {nm})" for src, nm in spawned)
        return (
            "Reviewer-diversity check: this answer still lacks a VERIFIED "
            "#consensus verdict from "
            + " and ".join(s for s, _ in spawned)
            + " reviewer(s) (have builder="
            + f"{n_builder}/{self._min_builder_reviewers}, dedicated="
            + f"{n_dedicated}/{self._min_dedicated_reviewers}). "
            "I spawned " + spawn_desc + " to obtain the missing verdict(s). "
            "Call prepare_report again — it will wait for the reviewer(s) to "
            "finish and then re-check. Do NOT write the report yet."
        )

    def _classify_consensus_reviewers(self) -> tuple[int, int]:
        """Count distinct VERIFIED ``#consensus`` posters by reviewer source.

        Returns ``(n_builder, n_dedicated)``.  A poster is a **builder** if its
        live agent shows first-hand web investigation
        (``web_search`` + ``web_fetch`` > 0); otherwise it is **dedicated**
        (a reasoning auditor reviewing from the BBS).  Authors not resolvable
        to a subagent (e.g. the orchestrator) are not counted as a source.
        """
        if self._bbs is None or self._swarm_ctx is None:
            return 0, 0
        try:
            msgs = self._bbs.read(channel="consensus", limit=200)
        except Exception:  # noqa: BLE001 — gate is best-effort
            return 0, 0
        smap = getattr(self._swarm_ctx, "_subagent_map", {}) or {}
        builders: set[str] = set()
        dedicated: set[str] = set()
        for m in msgs:
            if not is_verified_consensus_verdict(getattr(m, "content", "") or ""):
                continue
            author = getattr(m, "author", "") or ""
            sa = smap.get(author)
            if sa is None:
                continue
            counts = getattr(getattr(sa, "agent", None), "tool_calls_by_name", {}) or {}
            web = counts.get("web_search", 0) + counts.get("web_fetch", 0)
            if web > 0:
                builders.add(author)
            else:
                dedicated.add(author)
        return len(builders), len(dedicated)

    def _latest_candidate_snippet(self, max_chars: int = 1200) -> str:
        """Short 'leading candidate' context block harvested from the BBS,
        injected into the spawned reviewer's task prompt."""
        if self._bbs is None:
            return "## Leading candidate\n(see #key-findings and #consensus on the BBS)"
        parts: list[str] = []
        for ch in ("consensus", "key-findings"):
            try:
                msgs = self._bbs.read(channel=ch, limit=4)
            except Exception:  # noqa: BLE001
                msgs = []
            for m in msgs[-2:]:
                txt = (getattr(m, "content", "") or "").strip()
                if txt:
                    parts.append(f"[{ch}|{getattr(m, 'author', '')}] {txt[:400]}")
        blob = "\n".join(parts)[:max_chars]
        if blob:
            return "## Leading candidate / latest findings (from BBS)\n" + blob
        return "## Leading candidate\n(see #key-findings and #consensus on the BBS)"

    def _spawn_reviewer_task(self, kind: str) -> str | None:
        """Auto-spawn a targeted reviewer task for a missing source.

        ``kind="dedicated"`` → ``profile=reasoning`` task that reviews the
        leading candidate FROM THE BBS (no new search).
        ``kind="builder"`` → ``profile=browsing`` task that must re-verify the
        candidate first-hand via ``web_search`` / ``web_fetch``.
        Returns the assigned subagent name, or ``None`` on failure.
        """
        ctx = self._swarm_ctx
        if ctx is None:
            return None
        try:
            cand = self._latest_candidate_snippet()
            n = self._reviewer_remediation_attempts + 1
            if kind == "dedicated":
                profile = "reasoning"
                name = f"reviewer-dedicated-{n}"
                prompt = (
                    "## Dedicated verification review\n\n"
                    "Independently review the leading candidate answer the team "
                    "has converged on, using ONLY the evidence already on the "
                    "BBS (#key-findings, #discoveries, #consensus). Do NOT run "
                    "new web searches — use the `reasoning` tool to audit the "
                    "logic.\n\n"
                    f"{cand}\n\n"
                    "Check EVERY hard constraint in the original question "
                    "against the cited BBS evidence, then post exactly one "
                    "verdict:\n"
                    "- If every constraint is verified with evidence: post to "
                    "#consensus: \"VERIFIED: Reviewed [candidate] — all "
                    "constraints verified with evidence.\"\n"
                    "- If any constraint is unverified or contradicted: post a "
                    "CHALLENGE to #discussion naming the failing constraint.\n"
                    "Then complete the task."
                )
            else:
                profile = "browsing"
                name = f"reviewer-builder-{n}"
                prompt = (
                    "## Independent first-hand verification\n\n"
                    "Independently re-verify the leading candidate answer the "
                    "team has converged on by gathering your OWN first-hand "
                    "evidence — you MUST run web_search / web_fetch to confirm "
                    "the key constraints from primary sources (do not rely only "
                    "on other agents' BBS posts).\n\n"
                    f"{cand}\n\n"
                    "After confirming the constraints first-hand, post exactly "
                    "one verdict:\n"
                    "- If your own searches confirm every constraint: post to "
                    "#consensus: \"VERIFIED: Reviewed [candidate] — all "
                    "constraints verified with evidence.\"\n"
                    "- If your searches contradict any constraint: post a "
                    "CHALLENGE to #discussion naming the failing constraint.\n"
                    "Then complete the task."
                )
            spec = TaskSpec(
                id=ctx.next_task_id(),
                name=name,
                prompt=prompt,
                profile=profile,
                metadata={
                    "source": "reviewer_diversity_gate",
                    "reviewer_kind": kind,
                },
            )
            ctx.task_board.add_task(spec)
            assigned = ctx.spawn_or_assign(spec)
            logger.info(
                "[reviewer_gate] spawned %s reviewer task %s -> %s",
                kind, spec.id, assigned,
            )
            return assigned or name
        except Exception:
            logger.exception("[reviewer_gate] failed to spawn %s reviewer", kind)
            return None

    def _check_alt_task_gate(
        self, *, force: bool, timed_out: bool,
    ) -> str | None:
        """Premature-commitment guard: require >=1 alternative/contrarian task.

        The arcticswarm paper found that runs which never open an alternative /
        contrarian task commit to whichever candidate emerged first and lose
        accuracy, with the gap growing on harder questions.  This gate
        guarantees at least one such task on every web/BBS run: if the
        orchestrator never opened one, the harness auto-spawns a contrarian
        "find a DIFFERENT candidate" task (tagged ``metadata['alt']=True``) and
        refuses to unlock ``send_user_markdown_report`` until it is on the
        board.  The orchestrator's next ``prepare_report`` call blocks on the
        existing two-phase wait until that task finishes, then re-checks.

        Mirrors :meth:`_check_reviewer_diversity_gate`.  Returns ``None`` (pass)
        when: disabled, follow-up turn, non-web run, ``force and timed_out``, an
        alt task already exists, or the guard has degraded (single auto-spawn
        already issued / deadline / could-not-spawn) — never hangs.
        """
        if not self._enforce_alt_task:
            return None
        if self._is_followup:
            return None
        if not self._has_web_search:
            return None
        # Force escape hatch, mirroring the other gates.
        if force and timed_out:
            return None
        # Already have an alternative/contrarian task on the board?
        if any(task_is_alt(t) for t in self._task_board.all_tasks()):
            return None
        # Advisory-degrade: never block past the deadline or after the single
        # auto-spawn this guard is allowed.  Unlock with a recorded WARNING.
        if timed_out or self._alt_gate_spawned:
            self._alt_gate_degrade_note = (
                "\n\nWARNING: no ALTERNATIVE/CONTRARIAN task was opened before "
                "reporting (premature-commitment guard degraded after the run "
                "deadline or its single auto-spawn). The answer may reflect the "
                "first candidate found without a rival hypothesis being explored."
            )
            logger.info(
                "[alt_gate] advisory-degrade (timed_out=%s, already_spawned=%s)",
                timed_out, self._alt_gate_spawned,
            )
            return None
        # Auto-spawn a single contrarian task and refuse this round.
        name = self._spawn_contrarian_task()
        self._alt_gate_spawned = True
        if not name:
            self._alt_gate_degrade_note = (
                "\n\nWARNING: alternative/contrarian task could not be spawned "
                "(no swarm context / dispatch failure); proceeding without it."
            )
            return None
        return (
            "Premature-commitment guard: the team has not opened any "
            "ALTERNATIVE / CONTRARIAN search task, so it risks committing to "
            "the first candidate it found without exploring a rival. I spawned "
            f"a contrarian 'alternative-candidate-sweep' task ({name}) that "
            "searches for a DIFFERENT answer than the current leader. Call "
            "prepare_report again — it will wait for that task to finish and "
            "then re-check. Do NOT write the report yet."
        )

    def _spawn_contrarian_task(self) -> str | None:
        """Auto-spawn the contrarian 'find a different candidate' task.

        ``profile=browsing`` task tagged ``metadata={'alt': True}`` so it both
        satisfies :meth:`_check_alt_task_gate` and is counted as an alternative
        task in trajectories.  Mirrors :meth:`_spawn_reviewer_task` and the
        rival-sweep in ``wire_candidate_emergence_hook``.  Works in dynamic mode
        (dispatch via ``swarm_ctx.spawn_or_assign``) and fixed-pool mode (add to
        the board for an idle subagent to claim).  Returns the assigned subagent
        name (or the task name in fixed-pool mode), or ``None`` on failure.
        """
        ctx = self._swarm_ctx
        board = ctx.task_board if ctx is not None else self._task_board
        if board is None:
            return None
        try:
            cand = self._latest_candidate_snippet()
            q = (self._question_text or "").strip()[:1500]
            prompt = (
                "## Alternative-candidate sweep (contrarian)\n\n"
                "The team has converged toward a leading candidate without ever "
                "exploring an alternative. Before we commit, find at least 3 "
                "PLAUSIBLE ALTERNATIVE candidates that satisfy the question's "
                "hard constraints but are DIFFERENT from the current leader. "
                "Do NOT prefer fame — obscure answers are common.\n\n"
                + (f"## Question\n{q}\n\n" if q else "")
                + f"{cand}\n\n"
                "## Instructions\n"
                "1. Identify the 1-2 hard constraints that most narrowly "
                "distinguish the answer (specific dates, places, unusual "
                "biographical details).\n"
                "2. Run web searches that EXCLUDE the leading candidate's name; "
                "bias queries toward less-famous matches.\n"
                "3. For each alternative you find, post ONE message to "
                "#discoveries: `ALT: <name> | matches: <which constraints> | "
                "obscurity: <mention level>`.\n"
                "4. If after a genuine search no alternative survives the hard "
                "constraints, post to #consensus that the leading candidate "
                "withstood a contrarian check. Do NOT commit a final answer; "
                "complete the task with a one-sentence summary."
            )
            task_id = (
                ctx.next_task_id()
                if ctx is not None
                else f"alt-sweep-{uuid.uuid4().hex[:8]}"
            )
            spec = TaskSpec(
                id=task_id,
                name="alternative-candidate-sweep",
                prompt=prompt,
                profile="browsing",
                metadata={"source": "alt_task_gate", "alt": True},
            )
            board.add_task(spec)
            assigned = ctx.spawn_or_assign(spec) if ctx is not None else None
            logger.info(
                "[alt_gate] spawned contrarian task %s -> %s", spec.id, assigned,
            )
            return assigned or spec.name
        except Exception:
            logger.exception("[alt_gate] failed to spawn contrarian task")
            return None

    def _build_not_ready_output(
        self,
        *,
        agents_idle: bool,
        dm_text: str,
        guidance: str,
    ) -> str:
        """Compose the ``Not ready`` response shared by both realtime branches.

        The orchestrator used to see a bare list of pending task names
        (or worse, a single-sentence "some subagents are still active"
        message).  Reusing :func:`_render_task_activity_lines` here gives
        it the same per-task activity + STALE flags that ``list_tasks``
        and ``wait_for_tasks`` already emit, so the orchestrator can
        make an informed ``force=true`` decision on its next turn
        without an extra ``list_tasks`` round-trip.
        """
        _terminal = (TaskStatus.COMPLETED, TaskStatus.FAILED)
        pending = [
            t for t in self._task_board.all_tasks()
            if t.status not in _terminal
            and not t.metadata.get("infrastructure", False)
        ]
        now = time.monotonic()
        task_lines: list[str] = []
        for task in pending:
            task_lines.extend(
                _render_task_activity_lines(task, now, include_summary=False)
            )

        header = (
            f"Not ready yet — {len(pending)} task(s) still in progress"
            if pending
            else "Not ready yet"
        )
        if not agents_idle:
            header += "; some subagents are still active"
        header += "."

        parts: list[str] = [header]
        if task_lines:
            parts.append("\n".join(task_lines))
        if dm_text:
            # dm_text already starts with \n\n to set off the DM block.
            parts.append(dm_text.lstrip("\n"))
        parts.append(guidance)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# SendReportTool
# ---------------------------------------------------------------------------


class SendReportTool(BaseTool):
    """Deliver the final markdown report to the user.

    The orchestrator calls this tool once when it has gathered enough
    information from subagents to answer the user's question.  The
    ``report`` parameter is streamed token-by-token to the terminal via
    the ``ToolInputDelta`` event path.

    After execution, ``captured_report`` holds the full report text so
    that ``run_swarm_turn`` can return it as the answer string.
    """

    def __init__(
        self,
        has_web_search: bool = False,
        *,
        mailbox: Mailbox | None = None,
        agent_name: str | None = None,
        strict_dm_drain: bool = False,
        auditor_role: str = "author",
        peer_agent_name: str | None = None,
        reviewer_stall_budget_s: float = 60.0,
        reject_refusal: bool = False,
        question: str | None = None,
        max_refusal_bounces: int = 3,
    ) -> None:
        self.captured_report: str | None = None
        self.reference_registry: ReferenceRegistry | None = None
        self._has_web_search = has_web_search
        # Duo / DM-strict flag: if ``strict_dm_drain`` is True and the agent's
        # mailbox has unread DMs at the moment the leader tries to submit its
        # report, block the submission and surface the pending messages as a
        # tool result instead.  This forces the leader to take at least one
        # more turn to read the teammate's findings before finalising — which
        # closes the "leader-finishes-while-auditor-is-still-writing" race.
        self._mailbox = mailbox
        self._agent_name = agent_name
        self._strict_dm_drain = strict_dm_drain
        # Reviewer-mode "wait for first review" stall. Only active when
        # ``auditor_role == "reviewer"`` and a peer agent name is
        # supplied. Trips when the leader tries to finalize before the
        # auditor has delivered any peer-lane DM. Bounded by a wall
        # budget (default 60s) so a silent/crashed auditor cannot
        # deadlock submission.
        #
        # Smoke evidence (n=10, reviewer mode): in 5/10 trials the
        # leader called ``send_user_markdown_report`` 8-50s BEFORE the
        # auditor's first review DM. The reviewer is structurally
        # slower than the leader (it has to wait for the leader's
        # candidate to exist, then verify+interpret), so
        # without a stall the leader systematically out-races the
        # auditor on easy tasks and never sees its findings.
        #
        # ``_reviewer_stall_start_ts`` records the monotonic clock at
        # the FIRST stall fire, not at tool construction. The budget
        # measures patience-after-the-leader-tried-to-finish, not
        # total run time.
        self._auditor_role = auditor_role
        self._peer_agent_name = peer_agent_name
        self._reviewer_stall_budget_s = reviewer_stall_budget_s
        self._reviewer_stall_start_ts: float | None = None
        self._reviewer_stall_count = 0
        # anti-give-up + canonical-answer (qwen-gated via reject_refusal):
        # when True, append answer-form guidance to the instructions and bounce
        # a refusal/give-up FINAL ANSWER for a committed retry (bounded).
        self._reject_refusal = reject_refusal
        self._question = question
        self._max_refusal_bounces = max_refusal_bounces
        self._refusal_bounce_count = 0

    def _answer_form_guidance(self) -> str:
        """Extra FINAL ANSWER guidance, only when reject_refusal is on (qwen).

        Targets (a) under-specification judge artifacts — emit the fullest
        canonical form, honoring the question's requested format — and (b)
        give-ups — commit to the best-supported candidate, never refuse.
        """
        if not self._reject_refusal:
            return ""
        fmt = (
            " The question may specify the exact form required (e.g. \"full "
            "name including first, middle and last name\", an exact title, or "
            "units); your FINAL ANSWER MUST match that requested form."
            if self._question else ""
        )
        return (
            "\n\nFINAL ANSWER form (important): give the answer's FULLEST, most "
            "canonical/official form, reproduced VERBATIM from your verified "
            "findings — a person's complete legal name including middle "
            "names/suffixes (not a stage name or abbreviation), a work/event's "
            "exact full title (do not drop or add words), an organization's "
            "most specific name as the source states it. Do not paraphrase, "
            "shorten, or normalize the canonical string." + fmt +
            "\nA definite answer ALWAYS exists for this task. NEVER answer "
            "\"no answer found\", \"unable to determine\", \"no candidate "
            "satisfies all constraints\", or any refusal. A single unverified "
            "or contested constraint is NOT grounds to give up — commit to the "
            "single candidate that satisfies the MOST hard constraints."
        )

    @property
    def name(self) -> str:
        return "send_user_markdown_report"

    @property
    def description(self) -> str:
        if self._has_web_search:
            queryinfo = 'research findings'
        else:
            queryinfo = 'SQL queries'
        return (
            "Send the final markdown report to the user. Call this ONCE "
            "when you have enough information to answer the user's question. "
            "The report should be a complete, well-formatted markdown document "
            f"that includes the answer, relevant {queryinfo}, key data, and any "
            "caveats. Include ```vega-lite code fences with a Vega-Lite v5 "
            "JSON spec for data charts (bar, line, pie, scatter) — these "
            "render as interactive charts with hover tooltips. Charts are "
            "rendered in a full HTML report; the terminal shows [Chart] "
            "placeholders. "
            "Do NOT include a ## References section — it will be generated "
            "automatically. Just use [N] citations inline to reference sources. "
            "End the report with these two lines, in this order, each on its "
            "own line (so the grader and calibration metric can extract them "
            "cleanly):\n"
            "Confidence: <integer 0-100>\n"
            "FINAL ANSWER: <your answer>\n"
            "This is the ONLY way to deliver your final answer."
            + self._answer_form_guidance()
        )

    def parameters_schema(self) -> dict[str, Any]:
        if self._has_web_search:
            queryinfo = 'research findings'
        else:
            queryinfo = 'SQL queries'

        report_desc = (
                "The complete markdown report to send to the user. "
                f"Include the answer, {queryinfo}, key evidence, "
                "Vega-Lite charts where they add value, "
                "and any caveats or assumptions. Use [N] inline "
                "citations to reference sources. Do NOT include a "
                "## References section - it will be auto-generated."
                "For multiple sources, use separate brackets: [1][2][3] or [1], [2], [3]. "
                "NEVER concatenate numbers like [784] when you mean [7], [8], [4].\n\n"
                + self._answer_form_guidance()
            )
        return {
            "type": "object",
            "required": ["report"],
            "properties": {
                "report": {
                    "type": "string",
                    "description": report_desc,
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        report = kwargs.get("report", "")
        if isinstance(report, str):
            report = report.strip()
        if not report:
            return ToolResult(
                error=(
                    "ERROR: `send_user_markdown_report` was called with an empty "
                    "`report` parameter.\n\n"
                    "COMMON MISTAKE: you may have written the report content as "
                    "plain text in your assistant message alongside this tool "
                    "call. That text is IGNORED — only the JSON arguments of "
                    "this tool call are delivered to the user. The full report "
                    "MUST be passed as the `report` field in this tool call's "
                    "JSON arguments.\n\n"
                    "Required JSON shape:\n"
                    '  {"report": "<complete markdown report ending with '
                    "Confidence: <int> and FINAL ANSWER: <answer> on the last "
                    'two lines>"}\n\n'
                    "Retry this call with the full markdown report inside the "
                    "`report` parameter."
                ),
                is_error=True,
            )

        # Duo / strict-DM mode: drain the mailbox before accepting the report.
        # If the teammate has sent findings while the leader was composing the
        # report, surface them now and decline the submission — the leader
        # must take one more turn to read them.  We consume the messages
        # (check_new is destructive) so the next call can submit freely.
        #
        # Non-substantive DMs (idle pings, task-completed echoes that the
        # orchestrator already received as the ``create_task`` tool result,
        # worktree-harvest notices that are surfaced separately) are
        # dropped silently so they don't trigger a re-send loop. Without
        # this filter the leader's first ``send_user_markdown_report`` call
        # was being bounced 1-3 times per question on idle/result-lane echoes
        # with no new content — see ``_NON_SUBSTANTIVE_REPORT_DRAIN_DM_TYPES``.
        if (
            self._strict_dm_drain
            and self._mailbox is not None
            and self._agent_name is not None
        ):
            pending = self._mailbox.check_new(self._agent_name) or []
            substantive = [
                m for m in pending
                if m.message_type not in _NON_SUBSTANTIVE_REPORT_DRAIN_DM_TYPES
            ]
            if substantive:
                rendered = self._mailbox.render_for_llm(substantive)
                return ToolResult(
                    error=(
                        "Your teammate sent new findings while you were "
                        "composing the report.  Read them carefully, revise "
                        "the report if needed, and then call "
                        "send_user_markdown_report again.\n\n"
                        f"{rendered}"
                    ),
                    is_error=True,
                )

        # Reviewer-mode pre-submit "wait for first review" stall.
        # Only fires in duo + ``auditor_role == "reviewer"``. Trips when
        # the leader has not yet received any peer-lane DM from the
        # auditor (i.e. no STATUS/FINDINGS body has landed yet) and the
        # wall budget has not been exceeded.
        if (
            self._auditor_role == "reviewer"
            and self._peer_agent_name is not None
            and self._mailbox is not None
            and self._agent_name is not None
        ):
            received_any = self._mailbox.has_received_from(
                self._agent_name,
                self._peer_agent_name,
                lane=DM_LANE_PEER,
            )
            if not received_any:
                now = time.monotonic()
                if self._reviewer_stall_start_ts is None:
                    self._reviewer_stall_start_ts = now
                waited = now - self._reviewer_stall_start_ts
                if waited < self._reviewer_stall_budget_s:
                    self._reviewer_stall_count += 1
                    remaining = max(
                        0.0,
                        self._reviewer_stall_budget_s - waited,
                    )
                    return ToolResult(
                        error=(
                            f"Hold on — your reviewer-mode auditor "
                            f"({self._peer_agent_name}) has not yet "
                            f"delivered a review DM. Reviewer-mode "
                            f"auditors are structurally slower than "
                            f"you: they replay your work, verify it, "
                            f"and write structured findings. "
                            f"That typically takes 2-4 minutes total.\n\n"
                            f"Take one more turn — call ``read_dm`` "
                            f"(or any short tool) to keep your turn "
                            f"alive while the auditor finishes; "
                            f"``send_user_markdown_report`` will "
                            f"unblock as soon as the first auditor "
                            f"peer DM arrives. If no DM arrives "
                            f"within {remaining:.0f}s, this stall "
                            f"auto-clears and the next submit "
                            f"attempt goes through unconditionally."
                        ),
                        is_error=True,
                    )
                # Budget exceeded — log and fall through to allow submit.
                # Subsequent submit attempts in this tool instance will
                # also pass through because ``has_received_from`` is
                # still False AND the budget remains exceeded.
                logger.info(
                    "SendReportTool: reviewer stall budget (%.0fs) "
                    "exceeded with no DM from %s — allowing submit. "
                    "stall_count=%d",
                    self._reviewer_stall_budget_s,
                    self._peer_agent_name,
                    self._reviewer_stall_count,
                )

        # Second strict-DM drain. Peer DMs may have arrived between the
        # top-of-execute drain and now (the top drain only catches DMs
        # received BEFORE the call started). Cheap when nothing arrived
        # (one mailbox.check_new returning []).
        if (
            self._strict_dm_drain
            and self._mailbox is not None
            and self._agent_name is not None
        ):
            pending = self._mailbox.check_new(self._agent_name) or []
            substantive = [
                m for m in pending
                if m.message_type not in _NON_SUBSTANTIVE_REPORT_DRAIN_DM_TYPES
            ]
            if substantive:
                rendered = self._mailbox.render_for_llm(substantive)
                return ToolResult(
                    error=(
                        "Your teammate sent new findings while you "
                        "were composing the report. "
                        "Read them carefully, revise the report if "
                        "needed, and then call "
                        "send_user_markdown_report again.\n\n"
                        f"{rendered}"
                    ),
                    is_error=True,
                )

        # anti-give-up bounce (qwen-gated). If the FINAL ANSWER is a
        # refusal/give-up, reject the report so the orchestrator retries with a
        # committed best-candidate answer — bounded by max_refusal_bounces so a
        # genuinely-stuck case can't loop forever (a committed wrong answer
        # scores the same as a give-up, so accepting after the bound is safe).
        # Ordered AFTER the DM drains / reviewer stall so a pending teammate
        # finding is surfaced first.
        if self._reject_refusal:
            from arcticswarm.swarm.empty_answer_recovery import (
                final_answer_is_giveup,
            )
            if final_answer_is_giveup(report):
                if self._refusal_bounce_count < self._max_refusal_bounces:
                    self._refusal_bounce_count += 1
                    return ToolResult(
                        error=(
                            "Your FINAL ANSWER is a refusal / give-up, but a "
                            "definite answer ALWAYS exists for this task. Do "
                            "NOT submit \"no answer found\" / \"unable to "
                            "determine\" / \"no candidate satisfies all "
                            "constraints\".\n\n"
                            "Re-read your team's #key-findings / #consensus and "
                            "task summaries, pick the SINGLE candidate that "
                            "satisfies the MOST hard constraints, and commit to "
                            "it. A single unverified or contested constraint "
                            "(a distance measurement, a date-range boundary, or "
                            "a \"could not find\" result) is NOT grounds to give "
                            "up — treat \"could not find evidence for X\" as "
                            "MISSING evidence, not as X being false.\n\n"
                            "Give the candidate's fullest canonical form "
                            "(verbatim from the findings) on the FINAL ANSWER "
                            "line, then call send_user_markdown_report again."
                        ),
                        is_error=True,
                    )
                logger.warning(
                    "SendReportTool: refusal-bounce budget (%d) exhausted — "
                    "accepting give-up report as-is.",
                    self._max_refusal_bounces,
                )

        # Always strip any ## References or ## AI Deep Dives section the LLM
        # may have written — we render these programmatically from the registry.
        report = re.sub(
            r"\n##\s*(?:References|AI Deep Dives)\b.*",
            "",
            report,
            flags=re.IGNORECASE | re.DOTALL,
        ).rstrip()

        # Always append the programmatic footer from the reference registry
        if (
            self.reference_registry is not None
            and len(self.reference_registry) > 0
        ):
            report += self.reference_registry.render_footer()

        self.captured_report = report
        return ToolResult(output="Report delivered.")

