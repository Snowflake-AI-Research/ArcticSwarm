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

"""Swarm orchestrator — pre-spawns a pool of named subagents.

The orchestrator is the "lead agent" that:
1. Pre-spawns N subagents with random human names at swarm start.
2. Runs its own agentic loop with access to planning tools and BBS.
3. Posts tasks to the shared board — subagents claim them autonomously.
4. Monitors the BBS and task board to decide when to act or delegate.
5. Produces the final answer directly when it has enough information.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from arcticswarm.agent import (
    Agent,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCallEnd,
    ToolCallStart,
    ToolInputDelta,
    TurnComplete,
)
from arcticswarm.config import ArcticswarmConfig
from arcticswarm.logging_utils import aggregate_tool_role_usage
from arcticswarm.snowflake_client import SnowflakeClient
from arcticswarm.swarm.bbs import BBS
from arcticswarm.swarm.empty_answer_recovery import (
    extract_answer_from_messages as _extract_answer_from_messages,
    is_empty_or_refusal as _is_empty_or_refusal,
    run_empty_answer_recovery_turn,
)
from arcticswarm.swarm.answer_verification import (
    wire_candidate_emergence_hook,
)
from arcticswarm.swarm.mailbox import DM_LANE_CONTROL, Mailbox
from arcticswarm.swarm.names import assign_names
from arcticswarm.swarm.prompts import (
    build_orchestrator_system_prompt,
)
from arcticswarm.swarm.task import AgentRegistry, TaskBoard
from arcticswarm.swarm.teammate import _TimingCollector, _inject_timings_into_messages
from arcticswarm.swarm.tools import (
    DynamicCreateTaskTool,
    ListTasksTool,
    PostToBBSTool,
    PrepareReportTool,
    ReadBBSTool,
    ReadDMTool,
    SendMessageTool,
    SendReportTool,
    SwarmContext,
    WaitForTasksTool,
)
from arcticswarm.tools.reasoning import ReasoningTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multimodal user-content helpers
# ---------------------------------------------------------------------------
#
# The orchestrator accepts a question as either a plain string (text-only,
# the historical path) or a list of Anthropic-shape content blocks built by
# :func:`arcticswarm.eval.runner._build_user_message_content` for multimodal
# inputs (e.g. image cases).  These helpers let the orchestrator reason
# about the two shapes uniformly: ``_text_of`` extracts the text portion for
# log lines / subagent task descriptions, while ``_with_text_replaced``
# rewrites the final text block in-place so prefixes like "[Turn N —
# follow-up request]\n" keep image blocks
# attached to the orchestrator's initial user message.


def _text_of(content: str | list[dict[str, Any]]) -> str:
    """Return the text portion of a ``str | list[dict]`` user content value.

    For a plain string, returns it unchanged. For a list, concatenates the
    text from every ``{"type": "text", ...}`` block (in order, joined with
    newlines) and drops image / other blocks. This is intended for text-only
    consumers like subagent task descriptions and UI event payloads.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


def _images_of(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return image content blocks from a user-content value.

    For a plain string, returns ``[]``. For a list, returns each
    ``{"type": "image", ...}`` block in order. Used to forward the
    original question's image attachments to subagents so they can see
    the same visual context the orchestrator / leader sees.
    """
    if isinstance(content, str):
        return []
    return [
        dict(b) for b in content
        if isinstance(b, dict) and b.get("type") == "image"
    ]


def _with_text_replaced(
    content: str | list[dict[str, Any]], new_text: str
) -> str | list[dict[str, Any]]:
    """Return a copy of *content* whose final text block is *new_text*.

    - For a plain string, simply returns *new_text*.
    - For a list, preserves every image block (and any non-final text
      blocks such as ``"Image 1:"`` markers) and replaces the final text
      block with ``{"type": "text", "text": new_text}``. This lets the
      orchestrator re-enrich the question (e.g. follow-up
      prefix) without losing its attached images.
    """
    if isinstance(content, str):
        return new_text
    new_blocks = [dict(b) if isinstance(b, dict) else b for b in content]
    for i in range(len(new_blocks) - 1, -1, -1):
        block = new_blocks[i]
        if isinstance(block, dict) and block.get("type") == "text":
            new_blocks[i] = {"type": "text", "text": new_text}
            return new_blocks
    new_blocks.append({"type": "text", "text": new_text})
    return new_blocks


# ---------------------------------------------------------------------------
# Pre-shutdown subagent wrapping (90% timeout)
# ---------------------------------------------------------------------------


def _wrap_subagent_for_winding_down(
    sa: Any,
    remaining_secs: float,
    blocked_tools: frozenset[str],
) -> None:
    """Wrap a subagent's ``_execute_tool`` to block web tools and inject warnings.

    Called from the pre-shutdown timer thread at 90% of the timeout budget.
    Thread-safe: Python attribute assignment is atomic under the GIL, so the
    subagent thread picks up the new method on its next tool call.
    """
    from arcticswarm.tools.base import ToolResult as _TR

    _orig = sa.agent._execute_tool
    _first_call = [True]  # mutable flag — detailed warning on first non-blocked call

    def _winding_down_execute(
        name: str,
        input_data: dict[str, Any],
        _orig: Any = _orig,
    ) -> "_TR":
        if name in blocked_tools:
            return _TR(
                error=(
                    f"SYSTEM SHUTTING DOWN in ~{int(remaining_secs)}s. "
                    f"{name} is now DISABLED.\n"
                    "You MUST post your findings to BBS via post_to_bbs "
                    "immediately. Include:\n"
                    "  1. Your best candidate/answer so far\n"
                    "  2. Key evidence found\n"
                    "  3. What still needs investigation\n"
                    "If you don't post, ALL your findings will be LOST."
                ),
                is_error=True,
            )
        result = _orig(name, input_data)
        if _first_call[0]:
            _first_call[0] = False
            suffix = (
                f"\n\n⚠️ APPROACHING SHUTDOWN (~{int(remaining_secs)}s "
                "remaining). Post your findings to BBS NOW via "
                "post_to_bbs before time runs out. If you have partial "
                "conclusions, post them — note what still needs "
                "investigation."
            )
            return _TR(
                output=(result.output or "") + suffix,
                metadata=result.metadata,
            )
        return result

    sa.agent._execute_tool = _winding_down_execute  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Orchestrator event types (for the UI layer)
# ---------------------------------------------------------------------------


@dataclass
class SwarmEvent:
    """Base class for swarm-level events."""
    pass


@dataclass
class SwarmStarted(SwarmEvent):
    """The orchestrator has started processing a question."""
    question: str = ""
    bbs: Any = None  # BBS reference for the viewer (avoids circular import)


@dataclass
class SubagentSpawned(SwarmEvent):
    """A subagent has been pre-spawned at swarm start."""
    name: str = ""


@dataclass
class SubagentClaimedTask(SwarmEvent):
    """A subagent claimed a task from the board."""
    name: str = ""
    activity: str = ""


@dataclass
class SubagentIdle(SwarmEvent):
    """A subagent has become idle (finished a task or waiting)."""
    name: str = ""
    activity: str = ""


@dataclass
class SubagentSurfing(SwarmEvent):
    """A subagent is surfing the BBS (reading/posting during idle time)."""
    name: str = ""
    activity: str = ""


# Keep legacy names for backward compatibility with viewer
@dataclass
class TeammateSpawned(SwarmEvent):
    """A teammate has been spawned (legacy — see SubagentSpawned)."""
    name: str = ""
    prompt: str = ""


@dataclass
class TeammateCompleted(SwarmEvent):
    """A teammate finished its task."""
    name: str = ""
    summary: str = ""


@dataclass
class TeammateFailed(SwarmEvent):
    """A teammate's task failed."""
    name: str = ""
    error: str = ""


@dataclass
class TeammateToolCall(SwarmEvent):
    """A subagent invoked a tool."""
    name: str = ""          # subagent name
    tool_name: str = ""     # e.g. "web_search"
    description: str = ""   # short human-readable summary


@dataclass
class OrchestratorToolCall(SwarmEvent):
    """The orchestrator invoked a tool."""
    tool_name: str = ""     # e.g. "create_task", "web_search"
    description: str = ""   # short human-readable summary


@dataclass
class OrchestratorTextDelta(SwarmEvent):
    """A streaming text delta from the orchestrator / main worker."""
    text: str = ""


@dataclass
class OrchestratorMessage(SwarmEvent):
    """Intermediate reasoning text from the orchestrator (shown in Swarm Live)."""
    text: str = ""


@dataclass
class ReportStarted(SwarmEvent):
    """The orchestrator is streaming its final report — tear down the swarm panel."""
    pass


@dataclass
class ReportDelta(SwarmEvent):
    """Incremental text chunk for the streamed final report."""
    text: str = ""


@dataclass
class SwarmComplete(SwarmEvent):
    """The swarm run is complete."""
    answer: str = ""
    duration_seconds: float = 0.0
    subagent_count: int = 0
    bbs_message_count: int = 0
    report: str = ""
    token_usage: TokenUsage | None = None
    web_source_tracker: Any = None  # WebSourceTracker with captured URLs


# ---------------------------------------------------------------------------
# Tool-call summariser (for the Swarm Live feed)
# ---------------------------------------------------------------------------


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from content that may be a JSON content-block array.

    LLMs sometimes pass ``[{"text": "..."}]`` (Anthropic content block
    format) instead of a plain string.  This helper normalises both forms
    to a single string.
    """
    if isinstance(content, list):
        parts = []
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
        import json as _json
        try:
            parsed = _json.loads(stripped)
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


def _make_peer_tool_observer(
    *,
    emitter: str,
    mailbox: Mailbox,
    peers: list[str],
    observed_tools: frozenset[str],
    base_on_event: Callable[[Any], None] | None,
) -> Callable[[Any], None]:
    """Return an ``on_event`` wrapper that mirrors observable tool calls as DMs.

    For every ``ToolCallStart``/``ToolCallEnd`` pair where ``tool_name`` is in
    ``observed_tools``, the wrapper drops a synthetic DM into every peer's
    mailbox describing the call (tool name, file path / shell command,
    success/error). The receiver's existing ``_auto_dm_check`` /
    ``check_new`` mechanism surfaces the DM as a system reminder before its
    next LLM turn, so the peer learns about its teammate's file edits /
    shell commands without having to poll.

    Wire-level detail: ``ToolCallEnd`` does NOT carry ``tool_input`` — it
    only carries ``tool_name``, ``tool_use_id`` and the ``ToolResult``. We
    therefore stash inputs at ``ToolCallStart`` (keyed by ``tool_use_id``)
    and look them up when the matching ``ToolCallEnd`` arrives.

    The mechanism is fire-and-forget: failures inside the mailbox send are
    swallowed so a transient mailbox issue never breaks the originating
    agent's tool-execution path. ``base_on_event`` is always invoked, so
    upstream UI/trajectory collectors keep working unchanged.

    See SwarmConfig.peer_tool_observation in run_config.py for the rationale
    (the duo stale-view edit race) that motivated this.
    """

    pending_inputs: dict[str, dict[str, Any]] = {}

    def _peer_tool_call_text(tool_name: str, tool_input: dict[str, Any], ok: bool, err_text: str) -> tuple[str, str]:
        """Return (path-or-empty, rendered DM body) for a single tool call."""
        path = ""
        if isinstance(tool_input, dict):
            path = tool_input.get("file_path") or tool_input.get("path") or ""
        cmd = ""
        if isinstance(tool_input, dict) and tool_name == "bash":
            cmd = tool_input.get("command") or tool_input.get("cmd") or ""
            cmd = (cmd[:200] + "...") if len(cmd) > 200 else cmd
        lines = [
            "<peer_tool_call>",
            f"Your teammate `{emitter}` just ran the `{tool_name}` tool.",
        ]
        if path:
            lines.append(f"  target path: `{path}`")
        if cmd:
            lines.append(f"  shell command: `{cmd}`")
        lines.append(f"  outcome: {'SUCCESS' if ok else 'ERROR'}")
        if (not ok) and err_text:
            snip = err_text[:200] + ("..." if len(err_text) > 200 else "")
            lines.append(f"  error: {snip}")
        lines.append(
            "If this affects a file you are about to edit, re-read it before "
            "your next `edit_file`/`write_file` call — your prior view of the "
            "file may be stale and `edit_file` will fail with `old_string not "
            "found` (or, worse, silently clobber your teammate's change)."
        )
        lines.append("</peer_tool_call>")
        return path, "\n".join(lines)

    def _on_event(event: Any) -> None:
        try:
            if isinstance(event, ToolCallStart) and event.tool_name in observed_tools:
                # Stash input keyed by tool_use_id so we can build the
                # broadcast on the matching ToolCallEnd.
                inp = getattr(event, "tool_input", None)
                if isinstance(inp, dict):
                    pending_inputs[event.tool_use_id] = inp
            elif isinstance(event, ToolCallEnd) and event.tool_name in observed_tools:
                inp = pending_inputs.pop(event.tool_use_id, {}) or {}
                result = getattr(event, "result", None)
                ok = True
                err_text = ""
                if result is not None:
                    ok = not bool(getattr(result, "is_error", False))
                    if not ok:
                        err_text = getattr(result, "error", "") or getattr(result, "output", "") or ""
                path, body = _peer_tool_call_text(event.tool_name, inp, ok, err_text)
                for peer in peers:
                    if peer == emitter:
                        continue
                    try:
                        mailbox.send(
                            from_agent=emitter,
                            to_agent=peer,
                            content=body,
                            lane=DM_LANE_CONTROL,
                            message_type="peer_tool_call",
                            payload={
                                "tool_name": event.tool_name,
                                "path": path,
                                "ok": ok,
                                "emitter": emitter,
                            },
                        )
                    except Exception:
                        # Mailbox issues must NOT break the originating
                        # agent's tool-execution path. Swallow and continue.
                        pass
        except Exception:
            pass
        if base_on_event is not None:
            base_on_event(event)

    return _on_event


def _summarize_tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Produce a short human-readable description from a tool call."""
    if tool_name == "post_to_bbs":
        channel = tool_input.get("channel", "")
        content = _extract_text_from_content(tool_input.get("content", ""))
        snippet = content[:200] + "..." if len(content) > 200 else content
        return f'posted to #{channel}: "{snippet}"'

    if tool_name == "read_bbs":
        channel = tool_input.get("channel", "")
        if channel:
            return f"reading BBS #{channel}"
        return "reading BBS"

    if tool_name == "web_search":
        query = str(tool_input.get("query", "")).strip()
        if query:
            snippet = query[:160] + "..." if len(query) > 160 else query
            return f'web_search "{snippet}"'
        return "web_search"

    if tool_name == "web_fetch":
        url = str(tool_input.get("url", "")).strip()
        if url:
            snippet = url[:200] + "..." if len(url) > 200 else url
            return f"web_fetch {snippet}"
        return "web_fetch"

    if tool_name == "pdf_read":
        target = str(tool_input.get("url") or tool_input.get("file_path") or "").strip()
        if target:
            snippet = target[:200] + "..." if len(target) > 200 else target
            return f"pdf_read {snippet}"
        return "pdf_read"

    if tool_name == "complete_task":
        task_id = tool_input.get("task_id", "")
        return f"completed task '{task_id}'"

    if tool_name == "bash":
        cmd = tool_input.get("command", "")
        if cmd:
            snippet = cmd[:200] + "..." if len(cmd) > 200 else cmd
            return f"$ {snippet}"
        return "running bash command"

    if tool_name == "read_file":
        return "exploring files (read_file)"

    if tool_name == "calculator":
        return "calculating"

    if tool_name == "create_task":
        task_name = tool_input.get("name", "")
        prompt = tool_input.get("prompt", "")
        if prompt:
            snippet = prompt[:120] + "..." if len(prompt) > 120 else prompt
            return f"posting task '{task_name}': {snippet}"
        return f"posting task '{task_name}'"

    if tool_name == "list_tasks":
        return "checking task status"

    if tool_name == "wait_for_tasks":
        task_names = tool_input.get("task_names", [])
        return f"waiting on tasks: {', '.join(str(n) for n in task_names)}"

    if tool_name == "prepare_report":
        return "preparing final report"

    if tool_name == "reasoning":
        question = tool_input.get("question", "")
        if question:
            snippet = question[:200] + "..." if len(question) > 200 else question
            return f"deep reasoning: {snippet}"
        return "deep reasoning"

    if tool_name == "send_message":
        to = tool_input.get("to", "")
        content = _extract_text_from_content(tool_input.get("content", ""))
        snippet = content[:120] + "..." if len(content) > 120 else content
        return f'DM to {to}: "{snippet}"'

    if tool_name == "read_dm":
        return "checking DMs"

    return f"called {tool_name}"


# ---------------------------------------------------------------------------
# Final-message extraction
# ---------------------------------------------------------------------------


def _extract_final_message_text(messages: list[dict[str, Any]]) -> str:
    """Extract text content from the last assistant message.

    After ``Agent.run_turn()`` completes, the conversation history contains
    one assistant message per LLM round-trip.  Only the *last* one holds the
    final answer (the one with no tool_use blocks).  This function walks the
    history in reverse and returns just that text, discarding intermediate
    reasoning produced between tool calls.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                # Anthropic Pydantic objects have a .type attribute
                if hasattr(block, "type") and block.type == "text":
                    text_parts.append(block.text)
                # Serialised dicts (from saved sessions)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "".join(text_parts)
    return ""


# ---------------------------------------------------------------------------
# Incremental JSON parser for streaming the report field
# ---------------------------------------------------------------------------


class _ReportStreamParser:
    """Extract the ``report`` string value from incremental JSON chunks.

    The ``send_user_markdown_report`` tool input is
    ``{"report": "…long markdown…"}``.  As the Anthropic streaming API
    delivers ``input_json_delta`` chunks, this parser incrementally
    decodes the ``report`` string value — handling JSON string escapes
    (``\\"``, ``\\\\``, ``\\n``, etc.) — and returns parsed text on each
    :meth:`feed` call.

    State machine
    ~~~~~~~~~~~~~
    1. **Preamble** — accumulate until ``"report"`` + ``:`` + ``"`` found.
    2. **In-string** — emit decoded characters until the closing ``"``.
    3. **Done** — ignore further input.
    """

    _JSON_ESCAPES = {
        '"': '"', "\\": "\\", "/": "/",
        "n": "\n", "r": "\r", "t": "\t",
        "b": "\b", "f": "\f",
    }

    def __init__(self) -> None:
        self._buf = ""
        self._state = "preamble"   # preamble | in_string | done
        self._escape_next = False
        self.started = False

    def feed(self, chunk: str) -> str:
        """Feed a partial-JSON chunk; return any decoded report text."""
        if self._state == "done":
            return ""

        self._buf += chunk

        if self._state == "preamble":
            # Look for the opening of the report string value
            idx = self._buf.find('"report"')
            if idx == -1:
                return ""
            rest = self._buf[idx + len('"report"'):]
            # Skip whitespace and colon
            colon = rest.find(":")
            if colon == -1:
                return ""
            rest = rest[colon + 1:].lstrip()
            if not rest or rest[0] != '"':
                return ""
            self._state = "in_string"
            self.started = True
            self._buf = rest[1:]  # everything after the opening quote

        # Now parse the string value
        result: list[str] = []
        i = 0
        while i < len(self._buf):
            c = self._buf[i]
            if self._escape_next:
                result.append(self._JSON_ESCAPES.get(c, c))
                self._escape_next = False
            elif c == "\\":
                self._escape_next = True
            elif c == '"':
                # Closing quote — report is complete
                self._state = "done"
                self._buf = ""
                break
            else:
                result.append(c)
            i += 1
        else:
            # Consumed the whole buffer; clear it for the next chunk
            self._buf = ""

        return "".join(result)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# Imported AFTER the event classes + module helpers above so the mixin
# submodules can import those names from this module without hitting a
# partial-module error.  orchestrator_dm imports nothing from here;
# orchestrator_duo imports the event classes + helpers defined above.
from arcticswarm.swarm.orchestrator_dm import DmMixin
from arcticswarm.swarm.orchestrator_duo import DuoMixin


class SwarmOrchestrator(DmMixin, DuoMixin):
    """Lead agent that coordinates a pool of pre-spawned subagents.

    The orchestrator runs its own agentic loop with access to:

    - Standard planning tools (calculator, reasoning)
    - BBS tools (post_to_bbs, read_bbs)
    - Orchestration tools (create_task, list_tasks, wait_for_tasks)

    At swarm start, N subagents are spawned with random human names.
    They run persistent loops, claiming tasks from the board and posting
    results to the BBS.

    Parameters
    ----------
    config:
        Arcticswarm configuration (API key, model, Snowflake connection, etc.).
    max_teammates:
        Number of subagents to pre-spawn.
    """

    def __init__(
        self,
        config: ArcticswarmConfig,
        max_teammates: int = 5,
    ) -> None:
        self.config = config
        self.max_teammates = max_teammates

        # Exposed for the REPL to inspect via /bbs command
        self.last_bbs: BBS | None = None

        # Populated after run_swarm_turn() completes — exposes the
        # orchestrator's conversation and subagent summaries for
        # trajectory capture during eval.
        self.last_orchestrator_messages: list[dict[str, Any]] = []
        self.last_subagent_summaries: list[dict[str, Any]] = []
        self.last_task_summaries: list[dict[str, Any]] = []

        # Per-phase timing breakdown (seconds) for bottleneck analysis.
        # Populated after run_swarm_turn() completes.
        self.phase_timings: dict[str, float] = {}

        # Rival-audit telemetry slot, read by the eval trajectory writer.
        # Stays empty now that the Layer 4b rival audit has been removed;
        # kept so the telemetry reader remains backward-compatible.
        self._last_rival_audit: dict[str, Any] = {}

        # Aggregated token usage for the most recent swarm turn
        # (orchestrator + all subagents).
        self.last_token_usage: TokenUsage | None = None
        self.last_num_steps: int = 0
        # e2e total token estimate for swarm mode: orch_tokens + max(subagent_tokens).
        # Approximates the token load on the longest execution path when subagents run in parallel.
        self.last_total_token_e2e: int = 0

        # Swarm saturation: number of times tasks were ready but all agents were busy
        self.last_saturation_events: int = 0

        # Context compaction stats (aggregated across orchestrator + subagents)
        self.last_compaction_count: int = 0
        self.last_total_llm_calls: int = 0
        # Safety refusal count (aggregated across orchestrator + subagents)
        self.last_safety_refusal_count: int = 0
        self.last_content_filter_count: int = 0
        # O5: thinking-only count (aggregated across orchestrator + subagents).
        self.last_thinking_only_count: int = 0
        # O2: split-compaction + peak context size (aggregated)
        self.last_proactive_compaction_count: int = 0
        self.last_reactive_compaction_count: int = 0
        self.last_peak_input_tokens: int = 0

        # Reflection stats (aggregated across all subagents)
        self.last_reflection_stats: dict[str, Any] = {}

        # Spawn/assignment event log (dynamic mode only).
        self.last_spawn_events: list[dict[str, Any]] = []

        # Per-agent breakdown: maps agent name → TokenUsage.
        # Keys are "orchestrator" and subagent names.
        self.last_token_usage_breakdown: dict[str, TokenUsage] = {}

        # Shared SnowflakeClient for all agents (connection-pooled,
        # thread-safe).  Created once and reused across swarm turns.
        self._shared_sf_client: SnowflakeClient | None = None
        if config.sf_params:
            try:
                self._shared_sf_client = SnowflakeClient(config.sf_params)
            except Exception:
                pass  # will fail later when tools try to use it

        # Persistent state for multiturn conversations.
        # The orchestrator Agent and BBS carry over between turns so the
        # LLM has conversation memory and subagent findings persist.
        self._orchestrator_agent: Agent | None = None
        self._persistent_bbs: BBS | None = None
        self._turn_count: int = 0
        self._last_report: str | None = None
        self._used_names: set[str] = set()  # names assigned in prior turns

        # Single-shot guards for cheap-win recovery and rival sweep.
        self._cheap_win_fired: bool = False
        self._rival_sweep_fired: bool = False

        # Layer 4a (now critic-refine in step 4) telemetry.  Persisted for
        # the confidence detector that step 5 reads.  Keys: fired (bool),
        # clean (bool), found_falsification (bool), failed_constraint (str).
        self._last_layer4a: dict[str, Any] = {
            "fired": False,
            "clean": False,
            "found_falsification": False,
            "failed_constraint": None,
        }

        # Shared content cache for web_fetch/pdf_read deduplication (question-level).
        # Created externally per-case (with conv_id isolation) and set before run_swarm_turn().
        self._content_cache: Any | None = None

    @staticmethod
    def _aggregate_reflection_stats(
        subagents: Iterable,
    ) -> dict[str, Any]:
        """Build reflection stats dict from a list of subagents.

        Isolated into a helper so that a single ``AttributeError`` from
        a corrupted subagent does not leave ``r_calls`` unbound and crash
        the caller with ``UnboundLocalError``.
        """
        r_calls = sum(sa.reflection_calls for sa in subagents)
        r_sufficient = sum(sa.reflection_sufficient for sa in subagents)
        r_insufficient = sum(sa.reflection_insufficient for sa in subagents)
        r_conf: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        r_gaps = 0
        r_queries = 0
        for sa in subagents:
            for k, v in sa.reflection_confidence_counts.items():
                r_conf[k] = r_conf.get(k, 0) + v
            r_gaps += sa.reflection_total_gaps
            r_queries += sa.reflection_total_queries
        return {
            "total_calls": r_calls,
            "sufficient": r_sufficient,
            "insufficient": r_insufficient,
            "confidence_distribution": r_conf,
            "avg_gaps": round(r_gaps / r_calls, 1) if r_calls else 0,
            "avg_queries": round(r_queries / r_calls, 1) if r_calls else 0,
        }

    def run_swarm_turn(
        self,
        question: str | list[dict[str, Any]],
        *,
        on_event: Callable[[StreamEvent], None] | None = None,
        on_swarm_event: Callable[[SwarmEvent], None] | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """Run a full swarm turn with pre-spawned subagents.

        Pre-spawn N subagents, run the orchestrator's agentic loop
        (plan → create tasks → wait → read BBS → report).

        Parameters
        ----------
        on_event:
            (Unused — kept for API compat.)
        on_swarm_event:
            Callback for swarm-level events for the live UI.

        Returns the final answer text.
        """
        t0 = time.monotonic()
        timings: dict[str, float] = {}
        is_followup = self._turn_count > 0
        self._turn_count += 1

        # Per-turn flag reset.  These flags gate single-shot
        # behaviour (cheap-win recovery, rival sweep) and need to be
        # re-armed for each new turn even within a multiturn conversation.
        self._cheap_win_fired = False
        self._rival_sweep_fired = False
        self._last_layer4a = {
            "fired": False, "clean": False,
            "found_falsification": False, "failed_constraint": None,
        }

        # Compute absolute deadline from eval timeout (if provided).
        _deadline: float | None = (t0 + timeout_seconds) if timeout_seconds else None
        # Cache the text-only view of the question once: subagent task
        # descriptions, UI event payloads, and BBS seed messages all want
        # plain text even when the orchestrator is running with a
        # multimodal list content (image + text blocks).
        question_text = _text_of(question)
        # Preserve image blocks separately so we can also attach them to
        # every subagent's initial user message (subagents otherwise run
        # blind on image questions — see the image-propagation fix).
        question_images = _images_of(question)

        # Derive communication channel flags from config
        swarm_comm: list[str] = getattr(self.config, "swarm_comm", None) or ["bbs"]
        is_duo = "duo" in swarm_comm
        has_bbs = "bbs" in swarm_comm and not is_duo
        has_dm = "dm" in swarm_comm or is_duo

        if is_duo:
            return self._run_duo_turn(
                question,
                on_event=on_event,
                on_swarm_event=on_swarm_event,
                is_followup=is_followup,
                turn_number=self._turn_count,
            )

        # ---- BBS: persistent across turns (conditional) ----------------------
        bbs: BBS | None = None
        if has_bbs:
            if self._persistent_bbs is None:
                self._persistent_bbs = BBS()
            bbs = self._persistent_bbs
        self.last_bbs = bbs

        # ---- Mailbox: fresh each turn (conditional) --------------------------
        mailbox: Mailbox | None = None
        if has_dm:
            mailbox = Mailbox()

        # Subagents are always spawned dynamically (on demand).
        pool_size = self.config.max_subagents

        # TaskBoard, AgentRegistry, and thread pool are fresh each turn.
        task_board = TaskBoard(num_agents=pool_size)
        self._task_board = task_board  # expose for partial trajectory recovery on timeout
        agent_registry = AgentRegistry()
        pool = ThreadPoolExecutor(max_workers=pool_size)

        # Determine which profiles (and thus BBS channels) are available
        # for this swarm run.  This must happen before subagent spawn so
        # BBS tools are created with the correct channel enum.
        has_web_search = self.config.has_web_search_capability()

        # Active profile list — fully data-driven.
        #
        # Resolution order:
        #
        #   1. If the YAML pinned ``swarm.profiles`` (non-empty), honor it
        #      exactly. This includes single-profile setups like
        #      ``swarm.profiles: [browsing]`` and custom-profile topologies
        #      like BBS-worktree's ``[author, reviewer]``.
        #   2. Otherwise (YAML left ``swarm.profiles`` unset / empty), fall
        #      back to a sensible default derived from capabilities so
        #      ``create_task`` is never wired to an empty enum.
        if self.config.swarm_profiles:
            active_profile_names: list[str] = list(self.config.swarm_profiles)
        elif has_web_search:
            # Default fallback for web-search runs that didn't pin profiles
            # — preserves the historical web triple so create_task isn't
            # left with an empty enum.
            active_profile_names = ["browsing", "coding", "reasoning"]
        else:
            active_profile_names = []

        from arcticswarm.swarm.profiles import channels_for_profiles
        active_channels = channels_for_profiles(
            active_profile_names,
            self.config.tool_profiles,
        )
        if self.config.swarm_bbs_channels:
            active_channels = active_channels | frozenset(self.config.swarm_bbs_channels)

        ctx = SwarmContext(
            bbs=bbs,
            task_board=task_board,
            agent_registry=agent_registry,
            config=self.config,
            pool=pool,
            sf_client=self._shared_sf_client,
            on_swarm_event=on_swarm_event,
            question=question_text,
            question_images=question_images,
            max_teammates=self.max_teammates,
            active_channels=active_channels,
            mailbox=mailbox,
            has_bbs=has_bbs,
            has_dm=has_dm,
            system_reminder_interval=getattr(
                self.config, "system_reminder_interval", -1,
            ),
            dynamic_mode=True,
            deadline=_deadline,
            content_cache=self._content_cache,
        )
        # Expose ctx BEFORE subagent spawn / agent setup so
        # _capture_partial_trajectory can reach live subagents even if the
        # outer watchdog fires mid-turn.  Mirrors the DUO-mode Fix 0 invariant
        # (self._agent is exposed early); without ctx the eval runner cannot
        # harvest subagent conversations on timeout.
        self._ctx = ctx
        if mailbox is not None:
            try:
                mailbox.attach_task_board(task_board)
            except AttributeError:
                pass

        # ---- Disagreement gate -----------------------------------------------
        # When the first qualifying candidate post lands on BBS, post a
        # rival-sweep task to the board.  An existing browsing subagent will
        # pick it up.  Single-shot per swarm turn.  (Per-turn flag reset is
        # at the top of run_swarm_turn so this branch doesn't need to.)
        if (
            bbs is not None
            and has_bbs
            and getattr(self.config, "enable_candidate_emergence_sweep", True)
            and has_web_search
            and not is_followup
        ):
            wire_candidate_emergence_hook(
                self,
                bbs=bbs,
                task_board=task_board,
                question_text=question_text,
            )

        # ---- Subagents are spawned on demand (dynamic scaling) ---------------
        t_spawn = time.monotonic()
        # Dynamic mode: subagents are spawned on demand when tasks are created.
        subagent_names: list[str] = []

        # Register all participants on the mailbox
        if mailbox is not None:
            mailbox.register("leader")  # orchestrator receives broadcasts
            for sa_name in subagent_names:
                mailbox.register(sa_name)

        timings["subagent_spawn"] = round(time.monotonic() - t_spawn, 2)

        # ---- Build or reuse the orchestrator Agent ---------------------------
        t_agent_setup = time.monotonic()
        orchestrator_realtime = (
            has_dm
            and getattr(self.config, "orchestrator_realtime", False)
        )
        dm_realtime_direct_report = orchestrator_realtime and has_dm and not has_bbs

        if self._orchestrator_agent is None:
            # First turn: create a new Agent and strip all task-execution
            # tools.  The orchestrator is DELEGATE-ONLY — it thinks via
            # the reasoning tool and delegates everything else to subagents.
            agent = Agent(self.config)

            # Role-aware per-turn tool-call budget: the orchestrator is a
            # coordinator (batch create_task + wait_for_tasks), so a cap meant
            # to discipline browsing subagents silently drops its batched
            # fan-out calls (the dropped intent is never re-queued).  When
            # ``orchestrator_max_tool_calls_per_turn`` is set (>= 0), apply it
            # to the orchestrator agent ONLY via a per-agent override — never
            # mutate the shared config, which subagents read for their own cap.
            #   -1 = inherit config.max_tool_calls_per_turn (default, no-op)
            #    0 = unlimited orchestrator;  >=1 = explicit orchestrator cap.
            _orch_max_tc = getattr(
                self.config, "orchestrator_max_tool_calls_per_turn", -1,
            )
            if _orch_max_tc >= 0:
                agent.max_tool_calls_per_turn_override = _orch_max_tc

            # Set the web source tracker for capturing web_search results
            agent.web_source_tracker = ctx.web_sources

            # Set the shared content cache for web_fetch/pdf_read deduplication
            agent.content_cache = self._content_cache

            # Share the SnowflakeClient (avoid creating duplicate connections)
            if self._shared_sf_client is not None and agent.sf_client is not self._shared_sf_client:
                if agent.sf_client is not None:
                    try:
                        agent.sf_client.close()
                    except Exception:
                        pass
                agent.sf_client = self._shared_sf_client
                agent._register_tools()
            elif self._content_cache is not None:
                # Re-register tools so they pick up the content_cache
                agent._register_tools()

            # Snapshot skill availability before clearing
            from arcticswarm.tools.skill_tools import PerSkillTool as _PST
            _had_skill_tools = (
                "load_skill" in agent._tools
                or any(isinstance(t, _PST) for t in agent._tools.values())
            )
            _saved_read_skill_file = agent._tools.get("read_skill_file")

            # Strip ALL task-execution tools — the orchestrator only
            # delegates via create_task (and optionally reasons via the
            # reasoning tool when --reasoning-tool is enabled).
            agent._tools.clear()

            # Declarative path — YAML controls the orchestrator tool set.
            # Skill tools are restored separately below.
            from arcticswarm.tools.factory import ToolFactory as _OTF
            _orch_factory = _OTF(
                self.config,
                sf_client=self._shared_sf_client or agent.sf_client,
                agent_client=agent.client,
            )
            orch_factory_tools = [t for t in self.config.orchestrator_tools if t != "load_skill"]
            agent._tools.update(_orch_factory.build(orch_factory_tools))

            # Restore skill tools for the orchestrator
            if _had_skill_tools:
                from arcticswarm.tools.skill_tools import LoadSkillTool, PerSkillTool
                from arcticswarm.skill_loader import SkillRegistry
                from pathlib import Path
                skills_dir = Path(__file__).resolve().parent.parent / "skills"
                registry = SkillRegistry(skills_dir=skills_dir)
                from arcticswarm.swarm.profiles import resolve_orchestrator_skill
                orch_skill = resolve_orchestrator_skill(
                    has_bbs=has_bbs,
                    has_web_search=has_web_search,
                    orchestrator_realtime=orchestrator_realtime,
                    skill_overrides=getattr(self.config, "skill_overrides", None),
                )
                orch_skills = list(dict.fromkeys(
                    [orch_skill, *self.config.orchestrator_skills]
                ))
                if self.config.per_skill_tools:
                    for skill_name in orch_skills:
                        _pst = PerSkillTool(
                            skill_name,
                            registry=registry,
                            legacy_format=self.config.skill_legacy_format,
                        )
                        agent._tools[_pst.name] = _pst
                else:
                    agent._tools["load_skill"] = LoadSkillTool(
                        orch_skills,
                        registry=registry,
                        legacy_format=self.config.skill_legacy_format,
                    )
            if _saved_read_skill_file is not None:
                agent._tools["read_skill_file"] = _saved_read_skill_file

            self._orchestrator_agent = agent
        else:
            # Follow-up turn: reuse the existing agent (preserves
            # conversation history). Remove any stale report-preparation
            # tools from the previous turn; the active mode below
            # re-registers the correct reporting path.
            agent = self._orchestrator_agent
            agent._tools.pop("send_user_markdown_report", None)
            agent._tools.pop("prepare_report", None)

        self._agent = agent  # expose for partial trajectory capture on timeout
        self._report_tool: SendReportTool | None = None  # set below; exposed for timeout recovery

        # Register BBS tools (conditional on has_bbs)
        if has_bbs and bbs is not None:
            agent._tools["post_to_bbs"] = PostToBBSTool(bbs, author="orchestrator", channels=active_channels)
            read_bbs_tool = ReadBBSTool(bbs, channels=active_channels)
            read_bbs_tool.initialize_cursor()
            agent._tools["read_bbs"] = read_bbs_tool
            agent._auto_bbs_check = read_bbs_tool.check_new_messages
        else:
            agent._auto_bbs_check = None

        # Register DM tools and auto-injection for orchestrator.
        # In realtime mode the outer event-driven loop manages DM
        # delivery exclusively, so auto-injection is disabled and
        # read_dm is NOT registered (to prevent the LLM from polling).
        if has_dm and mailbox is not None:
            if not orchestrator_realtime:
                agent._tools["read_dm"] = ReadDMTool(mailbox, agent_name="leader")

                def _orch_check_dms() -> str | None:
                    msgs = mailbox.check_new("leader")
                    if not msgs:
                        return None
                    return mailbox.render_for_llm(msgs)

                agent._auto_dm_check = _orch_check_dms
            else:
                agent._auto_dm_check = None
        else:
            agent._auto_dm_check = None

        # Register orchestration tools (fresh each turn — new TaskBoard).
        # Subagents are spawned on demand via DynamicCreateTaskTool.
        # Expose the ``blocking`` create_task parameter only for
        # DM-realtime direct-report runs.  In every other config the
        # leader either has ``wait_for_tasks`` (non-realtime) or a
        # BBS coordination channel for follow-up, so the schema
        # surface area stays unchanged there.  See the
        # ``DynamicCreateTaskTool`` class docstring + the
        # `dm_create_task_blocking` plan for the leader-reviewer
        # sequencing rationale.
        agent._tools["create_task"] = DynamicCreateTaskTool(
            ctx,
            active_profiles=active_profile_names,
            has_web_search=has_web_search,
            disable_bbs_isolation=self.config.disable_bbs_isolation,
            force_bbs_isolation=self.config.force_bbs_isolation,
            expose_blocking=dm_realtime_direct_report,
            enforce_alt_task=getattr(self.config, "enforce_alt_task", True),
        )
        agent._tools["list_tasks"] = ListTasksTool(task_board)
        if not orchestrator_realtime:
            agent._tools["wait_for_tasks"] = WaitForTasksTool(task_board, swarm_ctx=ctx)
        else:
            agent._tools.pop("wait_for_tasks", None)

        # In real-time mode, the orchestrator can send messages to
        # specific subagents (or broadcast) for mediation / follow-up.
        if orchestrator_realtime and mailbox is not None:
            agent._tools["send_message"] = SendMessageTool(
                mailbox=mailbox,
                sender="leader",
                agent_names=list(subagent_names),
                has_bbs=has_bbs,
                peer_dm_summary=getattr(self.config, "peer_dm_summary", False),
                dynamic_names=False,
            )

        prepare_report_tool: PrepareReportTool | None = None
        _late_register_tools: dict[str, Any] = {}
        if dm_realtime_direct_report:
            # Plain DM realtime now mirrors Duo: waiting happens at the
            # runtime layer (outer mailbox loop), not via a tool-gated
            # prepare_report barrier. The leader can inspect `list_tasks`
            # on demand and submit directly when ready; unread teammate
            # DMs still block submission via strict_dm_drain.
            report_tool = SendReportTool(
                has_web_search=has_web_search,
                mailbox=mailbox,
                agent_name="leader",
                strict_dm_drain=True,
                reject_refusal=getattr(
                    self.config, "reject_refusal_reports", False),
                question=question_text,
            )
            agent._tools["send_user_markdown_report"] = report_tool
        else:
            # The report tool is the orchestrator's ONLY way to deliver the
            # final answer.  It is NOT registered yet — the orchestrator must
            # first call prepare_report, which blocks until all tasks are
            # complete and all subagents are idle, then dynamically registers
            # send_user_markdown_report for the next LLM turn.
            report_tool = SendReportTool(
                has_web_search=has_web_search,
                reject_refusal=getattr(
                    self.config, "reject_refusal_reports", False),
                question=question_text,
            )
            prepare_report_tool = PrepareReportTool(
                task_board=task_board,
                agent_registry=agent_registry,
                report_tool=report_tool,
                agent_tools=agent._tools,
                bbs=bbs,
                is_followup=is_followup,
                web_source_tracker=ctx.web_sources,
                swarm_ctx=ctx,
                realtime=orchestrator_realtime,
                # Plumbing the mailbox lets prepare_report block inside the tool
                # on ``wait_for_message`` (matches the DUO main-worker path).
                # Without these args execute() falls back to a one-sentence
                # "Not ready" response; with them the orchestrator receives
                # the rendered DM and a richer pending-tasks list.
                mailbox=mailbox if orchestrator_realtime else None,
                agent_name="leader" if orchestrator_realtime else None,
                enable_force_submit=self.config.enable_force_submit,
                blocking=self.config.blocking_prepare_report,
                late_register_tools=_late_register_tools,
                default_timeout=getattr(self.config, "prepare_report_timeout", 300),
                # Reviewer-diversity gate (web-research swarms): require a
                # VERIFIED #consensus verdict from BOTH a builder and a
                # dedicated reviewer before unlocking the report. ``has_web_search``
                # scopes the gate to web runs; it is a no-op when there is no
                # web capability and when the mins are 0.
                min_dedicated_reviewers=(
                    0 if getattr(self.config, "disable_auditor", False)
                    else getattr(self.config, "min_dedicated_reviewers", 0)
                ),
                min_builder_reviewers=getattr(
                    self.config, "min_builder_reviewers", 0,
                ),
                max_reviewer_remediations=getattr(
                    self.config, "max_reviewer_remediations", 2,
                ),
                has_web_search=has_web_search,
                # Premature-commitment guard: require >=1 alternative/contrarian
                # task before unlocking the report; auto-spawn one if the
                # orchestrator never opened one. ANDed with ``has_web_search``
                # so non-web runs are unaffected. See
                # PrepareReportTool._check_alt_task_gate.
                enforce_alt_task=(
                    getattr(self.config, "enforce_alt_task", True)
                    and has_web_search
                ),
                question_text=question_text,
                surface_bbs_candidates=getattr(
                    self.config, "surface_bbs_candidates", False,
                ),
            )
            agent._tools["prepare_report"] = prepare_report_tool
        # Exposed for timeout recovery (set regardless of branch).
        self._report_tool = report_tool
        timings["agent_setup"] = round(time.monotonic() - t_agent_setup, 2)

        # Build two parallel views of the question:
        #   * ``enriched_question_text``: plain text used everywhere the
        #     orchestrator composes strings (task descriptions, log lines).
        #   * ``enriched_question``: the LLM-facing content, either the
        #     same string (text-only case) or a list of blocks that
        #     preserves attached image blocks and whose final text block
        #     mirrors ``enriched_question_text``.
        enriched_question_text = question_text

        # On follow-up turns, prefix the message with a clear turn
        # boundary so the LLM knows the previous report delivery does
        # not satisfy this turn — it must call the tool again.
        if is_followup:
            enriched_question_text = (
                f"[Turn {self._turn_count} — follow-up request]\n"
                f"{enriched_question_text}"
            )

        enriched_question = _with_text_replaced(question, enriched_question_text)

        # ---- Activate the Swarm Live UI ----------------------------------
        # On follow-up turns, defer the panel UI until the orchestrator
        # actually creates a task (Path B — new investigation).  For light
        # edits (Path A) the panel never appears and the report streams
        # directly via phase-1 rolling console output.
        swarm_ui_activated = False
        if on_swarm_event and not is_followup:
            on_swarm_event(SwarmStarted(question=question_text, bbs=bbs))
            swarm_ui_activated = True

        # ---- Build the orchestrator system prompt ----------------------------
        # The web/corpus swarm has no semantic-model schema to summarise.
        schema_summary = ""

        # Resolve current date for temporal consistency checks
        from datetime import date, datetime
        if self.config.date_override:
            try:
                current_date = datetime.strptime(self.config.date_override, "%Y-%m-%d").date()
            except ValueError:
                current_date = date.today()
        else:
            current_date = date.today()

        agent.system_prompt = build_orchestrator_system_prompt(
            max_teammates=self.max_teammates,
            schema_summary=schema_summary,
            subagent_names=subagent_names,
            is_followup=is_followup,
            turn_number=self._turn_count,
            has_web_search=has_web_search,
            no_web_fetch=getattr(self.config, "no_web_fetch", False),
            dataset=self.config.dataset,
            current_date=current_date.isoformat(),
            active_channels=active_channels,
            has_bbs=has_bbs,
            has_dm=has_dm,
            has_reasoning_tool="reasoning" in agent._tools,
            active_profiles=active_profile_names,
            orchestrator_realtime=orchestrator_realtime,
            per_skill_tools=self.config.per_skill_tools,
            orchestrator_prompt_mode=self.config.orchestrator_prompt_mode,
            enable_vision=self.config.enable_vision,
            pre_loaded_tasks=None,
            tool_profiles=self.config.tool_profiles,
            disable_bbs_isolation=self.config.disable_bbs_isolation,
            force_bbs_isolation=self.config.force_bbs_isolation,
            enforce_alt_task=getattr(self.config, "enforce_alt_task", True),
            skill_overrides=getattr(self.config, "skill_overrides", None),
        )

        # Forward orchestrator events to the swarm UI.
        #
        # TextDelta arrives token-by-token during streaming, but the Swarm
        # Live feed should show one compact line per reasoning block — not
        # one line per token.  We accumulate text and flush it as a single
        # OrchestratorMessage when the LLM switches to a tool call or
        # finishes its turn.
        #
        # ToolInputDelta for the report tool → ReportStarted + ReportDelta
        #   (viewer tears down the swarm panel and streams Rich Markdown).
        # ToolCallStart → OrchestratorToolCall (existing behaviour).
        report_parser = _ReportStreamParser()
        report_started_fired = False
        text_accumulator: list[str] = []

        def _flush_text() -> None:
            """Emit accumulated orchestrator reasoning as one feed line."""
            if not text_accumulator or not on_swarm_event:
                return
            full = "".join(text_accumulator)
            text_accumulator.clear()
            if full.strip():
                on_swarm_event(OrchestratorMessage(text=full))

        def _on_agent_event(event: StreamEvent) -> None:
            nonlocal report_started_fired, swarm_ui_activated

            if not on_swarm_event:
                return

            if isinstance(event, TextDelta):
                # Accumulate streaming tokens.  Suppress any text emitted
                # *after* the report has started (LLM acknowledgement).
                if not report_parser.started:
                    if orchestrator_realtime and on_swarm_event:
                        on_swarm_event(OrchestratorTextDelta(text=event.text))
                    else:
                        text_accumulator.append(event.text)

            elif isinstance(event, ToolInputDelta):
                if event.tool_name == "send_user_markdown_report":
                    # Flush any buffered reasoning before the report starts
                    _flush_text()
                    parsed = report_parser.feed(event.partial_json)
                    if report_parser.started and not report_started_fired:
                        on_swarm_event(ReportStarted())
                        report_started_fired = True
                    if parsed:
                        on_swarm_event(ReportDelta(text=parsed))

            elif isinstance(event, ToolCallStart):
                # Flush accumulated reasoning that preceded this tool call
                _flush_text()

                # On follow-up turns, lazily activate the Swarm Live panel
                # when the orchestrator creates its first task (Path B).
                # For light edits (Path A) this never fires and the viewer
                # stays in phase-1 rolling-log mode.
                if not swarm_ui_activated and event.tool_name == "create_task":
                    on_swarm_event(SwarmStarted(question=question_text, bbs=bbs))
                    swarm_ui_activated = True

                # Don't emit a feed line for the report tool — the
                # streamed report itself is already displayed.
                if event.tool_name != "send_user_markdown_report":
                    desc = _summarize_tool_call(event.tool_name, event.tool_input)
                    on_swarm_event(OrchestratorToolCall(
                        tool_name=event.tool_name,
                        description=desc,
                    ))

            elif isinstance(event, ToolCallEnd):
                # Show a completion message for long-running blocking tools
                if event.tool_name == "prepare_report":
                    on_swarm_event(OrchestratorToolCall(
                        tool_name=event.tool_name,
                        description="✅ All tasks done — writing report...",
                    ))
                elif event.tool_name == "send_user_markdown_report":
                    # Report delivery is done — re-enable text accumulation
                    # so any conversational reply the orchestrator generates
                    # afterwards is displayed in the terminal.
                    report_parser.started = False

            elif isinstance(event, TurnComplete):
                # Flush any trailing text from the final LLM round
                _flush_text()

        # ---- Run the orchestrator's agentic loop (streaming) -----------------
        msg_start_idx = len(agent.messages)
        orch_collector = _TimingCollector(inner_on_event=_on_agent_event)
        orch_collector.start()
        try:
            if orchestrator_realtime:
                self._run_realtime_loop(
                    agent=agent,
                    mailbox=mailbox,
                    report_tool=report_tool,
                    prepare_report_tool=prepare_report_tool,
                    task_board=task_board,
                    agent_registry=agent_registry,
                    enriched_question=enriched_question,
                    dm_realtime_direct_report=dm_realtime_direct_report,
                    orch_collector=orch_collector,
                )
            else:
                # -- Soft-deadline timer ----------------------------------
                # When a timeout is configured, fire a soft deadline at
                # exactly timeout_seconds.  The eval runner's hard timeout
                # is extended by 300s to give the model time to wrap up.
                WRAP_UP_PERIOD = 300  # noqa: N806

                def _soft_timeout() -> None:
                    logger.warning(
                        "Soft deadline reached (%ss) — signalling wrap-up",
                        timeout_seconds,
                    )
                    ctx.wrapping_up.set()
                    ctx.shutdown.set()
                    prepare_report_tool._deadline_exceeded = True

                _soft_timer: threading.Timer | None = None
                if timeout_seconds is not None:
                    _soft_timer = threading.Timer(timeout_seconds, _soft_timeout)
                    _soft_timer.daemon = True
                    _soft_timer.start()

                    # Wrap tool execution so that every tool result carries a
                    # wrap-up reminder once the soft deadline has fired.  This
                    # prevents the model from endlessly reading BBS / reasoning
                    # instead of submitting the report.
                    _WRAP_EXEMPT = {"send_user_markdown_report"}
                    _WRAP_SUFFIX = (
                        "\n\n⚠️ TIME BUDGET EXHAUSTED. You MUST call "
                        "send_user_markdown_report NOW with your best answer. "
                        "Do NOT call any other tool."
                    )
                    _original_execute = agent._execute_tool

                    def _wrapped_execute(
                        name: str,
                        input_data: dict[str, Any],
                        _orig: Any = _original_execute,
                    ) -> "ToolResult":
                        result = _orig(name, input_data)
                        if (
                            ctx.wrapping_up.is_set()
                            and name not in _WRAP_EXEMPT
                        ):
                            from arcticswarm.tools.base import ToolResult as _TR
                            return _TR(
                                output=(result.output or "") + _WRAP_SUFFIX,
                                metadata=result.metadata,
                            )
                        return result

                    agent._execute_tool = _wrapped_execute  # type: ignore[assignment]

                    # -- Hard-force timer -------------------------------------
                    # If the LLM still hasn't called send_user_markdown_report
                    # 200s after the soft deadline, bypass the LLM entirely:
                    # build a fallback report from BBS content and set it on
                    # the report tool.  The remaining 100s of the 300s grace
                    # period lets run_turn_streaming exit naturally.
                    _FORCE_REPORT_DELAY = 200

                    def _force_report() -> None:
                        if report_tool.captured_report:
                            return  # LLM already submitted — nothing to do
                        logger.warning(
                            "Force-report timer fired — building fallback "
                            "report from BBS"
                        )
                        # When surface_bbs_candidates is on, build a
                        # COMMITTED answer that LEADS with the team's VERIFIED
                        # #consensus verdicts + #key-findings (judge-extractable),
                        # instead of dumping raw posts. Converts the found-but-
                        # blocked-at-timeout cases (the correct answer is on the
                        # BBS but the LLM never reached send_user_markdown_report).
                        # Also take this committed-answer path when
                        # reject_refusal_reports is on, so the timeout never
                        # finalizes a give-up.
                        # Best-effort; falls back to the raw dump on any issue.
                        _commit_force = (
                            getattr(self.config, "surface_bbs_candidates", False)
                            or getattr(self.config, "reject_refusal_reports", False)
                        )
                        if _commit_force and bbs is not None:
                            try:
                                from arcticswarm.swarm.bbs import (
                                    CHANNEL_CONSENSUS, CHANNEL_KEY_FINDINGS,
                                    CHANNEL_DISCOVERIES, is_verified_consensus_verdict,
                                )
                                seen: set[str] = set()
                                verified: list[str] = []
                                for m in bbs.read(channel=CHANNEL_CONSENSUS, limit=60):
                                    c = (m.content or "").strip()
                                    if c and is_verified_consensus_verdict(c) and c[:160] not in seen:
                                        seen.add(c[:160]); verified.append(f"- [VERIFIED by {m.author}] {c[:500]}")
                                findings: list[str] = []
                                for ch in (CHANNEL_KEY_FINDINGS, CHANNEL_DISCOVERIES):
                                    for m in bbs.read(channel=ch, limit=40):
                                        c = (m.content or "").strip()
                                        if c and c[:160] not in seen:
                                            seen.add(c[:160]); findings.append(f"- [{ch}/{m.author}] {c[:350]}")
                                if verified or findings:
                                    blk = ["# Final Answer (auto-generated at time limit)", "",
                                           "Based on the team's verified findings, commit to the single "
                                           "best-supported candidate below as the FINAL ANSWER (a definite "
                                           "answer exists; do not say 'no answer found'):"]
                                    if verified:
                                        blk += ["", "## Verified consensus verdicts", *verified[:8]]
                                    if findings:
                                        blk += ["", "## Key findings", *findings[:12]]
                                    report_tool.captured_report = "\n".join(blk)
                                    return
                            except Exception as exc:  # fall back to raw dump
                                logger.warning("candidate-led force-report failed (%s); raw dump", exc)
                        # Collect all BBS findings (legacy fallback)
                        parts: list[str] = []
                        if bbs is not None:
                            for msg in bbs.read_all():
                                parts.append(
                                    f"[{msg.channel}] {msg.author}: "
                                    f"{msg.content}"
                                )
                        # When reject_refusal_reports is on, never finalize
                        # a bare dump / give-up — wrap with a commit instruction
                        # and a synthetic FINAL ANSWER line so the judge can
                        # still extract a committed answer at the wall.
                        _commit_at_wall = getattr(
                            self.config, "reject_refusal_reports", False)
                        if parts:
                            body = "\n\n---\n\n".join(parts[-20:])  # last 20 posts
                            if _commit_at_wall:
                                report_tool.captured_report = (
                                    "# Final Answer (auto-generated at time limit)\n\n"
                                    "A definite answer exists. Based on the team's "
                                    "findings below, the single best-supported "
                                    "candidate is committed as the answer (do NOT "
                                    "say 'no answer found').\n\n"
                                    f"{body}\n\n"
                                    "Confidence: 20\n"
                                    "FINAL ANSWER: the best-supported candidate in "
                                    "the findings above."
                                )
                            else:
                                report_tool.captured_report = (
                                    "# Findings (auto-generated — time limit reached)\n\n"
                                    + body
                                )
                        else:
                            report_tool.captured_report = (
                                "Time limit reached. No findings were posted "
                                "to the BBS."
                            )

                    _force_timer = threading.Timer(
                        timeout_seconds + _FORCE_REPORT_DELAY, _force_report,
                    )
                    _force_timer.daemon = True
                    _force_timer.start()

                # -- Pre-shutdown timer (90% of timeout) -------------------
                # Give subagents a heads-up before the hard shutdown:
                # block web_search/web_fetch/pdf_read and inject a warning
                # so they post partial findings to BBS.
                _PRE_SHUTDOWN_FRACTION = 0.90
                _BLOCKED_SUBAGENT_TOOLS = frozenset({
                    "web_search", "web_fetch", "pdf_read",
                })
                _pre_timer: threading.Timer | None = None
                if (
                    timeout_seconds is not None
                    and self.config.has_web_search_capability()
                ):
                    def _pre_shutdown() -> None:
                        ctx.winding_down.set()
                        remaining = (
                            _deadline - time.monotonic()
                            if _deadline else 0
                        )
                        remaining = max(remaining, 0)
                        logger.warning(
                            "Pre-shutdown fired (90%% of %ss) — "
                            "wrapping subagent tools (~%.0fs remaining)",
                            timeout_seconds, remaining,
                        )
                        # Snapshot subagents under lock to avoid
                        # concurrent-modification from dynamic spawns.
                        with ctx._lock:
                            sa_snapshot = list(ctx.subagents)

                        for sa in sa_snapshot:
                            _wrap_subagent_for_winding_down(
                                sa, remaining, _BLOCKED_SUBAGENT_TOOLS,
                            )

                    _pre_delay = timeout_seconds * _PRE_SHUTDOWN_FRACTION
                    _pre_timer = threading.Timer(_pre_delay, _pre_shutdown)
                    _pre_timer.daemon = True
                    _pre_timer.start()

                try:
                    agent.run_turn_streaming(
                        enriched_question, on_event=orch_collector.on_event,
                    )
                finally:
                    if _soft_timer is not None:
                        _soft_timer.cancel()
                    if timeout_seconds is not None:
                        _force_timer.cancel()
                    if _pre_timer is not None:
                        _pre_timer.cancel()

            _inject_timings_into_messages(agent.messages, orch_collector, msg_start_idx)

            # The report is ONLY taken from the tool — orchestrator text
            # is displayed in the terminal but never used as the report.
            answer = report_tool.captured_report or ""

            # Fallback: when the orchestrator solved the question directly
            # (e.g. via reasoning tool) without delegating to subagents,
            # the report tool was never called.  Recover the answer from
            # the orchestrator's own assistant messages.
            if not answer.strip():
                answer = _extract_answer_from_messages(agent.messages)
                if answer:
                    logger.info(
                        "Swarm bypass fallback: recovered %d-char answer "
                        "from orchestrator messages",
                        len(answer),
                    )

            # Cheap-win: if the answer is still empty or a
            # refusal, inject ONE recovery turn asking for a best-guess.
            # In an internal calibration run, many wrong cases were empty/refusal
            # answers that never reached Layer 4a.
            # reject_refusal_reports also enables this post-hoc recovery
            # (independent of enable_empty_answer_recovery) so the
            # bounce -> natural-turn-exit -> empty-answer path is still caught.
            if (
                (
                    getattr(self.config, "enable_empty_answer_recovery", True)
                    or getattr(self.config, "reject_refusal_reports", False)
                )
                and not is_followup
                and not self._cheap_win_fired
                and not ctx.wrapping_up.is_set()
                and _is_empty_or_refusal(answer)
            ):
                self._cheap_win_fired = True
                answer = run_empty_answer_recovery_turn(
                    agent=agent,
                    answer=answer,
                    report_tool=report_tool,
                    on_agent_event=_on_agent_event,
                )

            # ---- Constraint verification (code-enforced) -----------------
            # After the orchestrator produces an answer, use the reasoning
            # tool to verify whether all constraints from the original
            # question are satisfied.  If verification finds gaps, inject
            # a follow-up prompt asking the orchestrator to address them.
            if (
                not is_followup
                and answer
                and answer.strip()
                and "reasoning" in agent._tools
                and not ctx.wrapping_up.is_set()
                and not getattr(
                    self.config, "disable_final_verification", False
                )
            ):
                reasoning_tool = agent._tools["reasoning"]
                if on_swarm_event:
                    on_swarm_event(OrchestratorToolCall(
                        tool_name="reasoning",
                        description="Verifying answer constraints...",
                    ))

                verification_prompt = (
                    "You are a constraint verification auditor. "
                    "Analyze the following answer against the "
                    "original question.\n\n"
                    f"## Question\n{enriched_question_text}\n\n"
                    "## Answer (first 2000 chars)\n"
                    f"{answer[:2000]}\n\n"
                    "For each factual constraint in the question, "
                    "determine:\n"
                    "- VERIFIED: Answer explicitly addresses this "
                    "constraint with evidence\n"
                    "- UNVERIFIED: Answer does not address this "
                    "constraint\n"
                    "- CONTRADICTED: Answer contradicts this "
                    "constraint\n\n"
                    "Output a JSON object:\n"
                    '{"verified_count": N, "unverified_count": N, '
                    '"contradicted_count": N, '
                    '"unverified_constraints": ["constraint1", ...], '
                    '"candidate_count": N, '
                    '"summary": "brief assessment"}'
                )

                verification_result = reasoning_tool.execute(
                    question=verification_prompt,
                )
                verification_text = (
                    verification_result.output
                    if not verification_result.is_error
                    else ""
                )

                # Persist Layer 4a telemetry for the gated-retry confidence
                # detector.  `clean` is filled from the loose count parse
                # below (see verification_had_gaps).
                self._last_layer4a["fired"] = True

                # If verification found gaps, inject a follow-up to
                # address them.  Parse loosely — look for trouble signals.
                verification_had_gaps = False
                needs_followup = False
                if verification_text:
                    # Parse counts from the JSON response and apply
                    # numeric thresholds instead of exact-string matching.
                    # The LLM may produce malformed JSON (missing braces,
                    # stray quotes, markdown fences), so we use a regex-
                    # first approach that just needs the key: number pair.

                    def _extract_int(
                        key: str, text: str,
                    ) -> int | None:
                        """Extract integer for *key* from JSON-ish text.

                        Uses a regex that tolerates malformed JSON — it only
                        needs ``"key": <digits>`` (or ``key: <digits>``)
                        somewhere in *text*.  Quotes around the key are
                        optional so we survive e.g. ``{unverified_count: 3}``.
                        """
                        # Allow optional quotes around the key and flexible
                        # whitespace / colon variants.
                        m = re.search(
                            rf"""(?:"|')?{re.escape(key)}(?:"|')?\s*:\s*(\d+)""",
                            text,
                        )
                        return int(m.group(1)) if m else None

                    unverified = _extract_int(
                        "unverified_count", verification_text,
                    )
                    contradicted = _extract_int(
                        "contradicted_count", verification_text,
                    )
                    candidates = _extract_int(
                        "candidate_count", verification_text,
                    )

                    # Trigger thresholds for web-search verification.
                    if (unverified is not None and unverified >= 2) or (
                        contradicted is not None and contradicted >= 1
                    ):
                        needs_followup = True
                    if candidates is not None and candidates <= 1:
                        needs_followup = True

                    # Telemetry: log verification counts for threshold tuning
                    logger.info(
                        "Constraint verification counts: "
                        "unverified=%s contradicted=%s candidates=%s "
                        "needs_followup=%s",
                        unverified, contradicted, candidates,
                        needs_followup,
                    )
                    if unverified is not None and unverified == 1:
                        logger.info(
                            "Constraint verification: unverified_count==1 "
                            "(below threshold, logged for future tuning)"
                        )

                    # If we couldn't parse ANY of the expected fields,
                    # the response is too malformed to trust — trigger
                    # followup to be safe.
                    if (
                        unverified is None
                        and contradicted is None
                        and candidates is None
                    ):
                        logger.warning(
                            "Could not parse any constraint counts from "
                            "verification response — triggering followup"
                        )
                        needs_followup = True

                    # Track whether thresholds genuinely triggered.
                    verification_had_gaps = needs_followup

                # Persist the legacy gap signal for the gated-retry detector.
                self._last_layer4a["clean"] = not verification_had_gaps

                if needs_followup and verification_text:
                    logger.info(
                        "Constraint verification found gaps — "
                        "requesting orchestrator follow-up"
                    )

                    followup_msg = (
                        "## Constraint Verification Alert\n\n"
                        "An automated audit of your answer found "
                        "potential issues:\n\n"
                        f"{verification_text}\n\n"
                        "Please address these gaps:\n"
                        "1. If constraints are UNVERIFIED, create "
                        "targeted search tasks to verify them "
                        "specifically.\n"
                        "2. If only ONE candidate was found, create "
                        "an 'alternative search' task that searches "
                        "for OTHER entities matching the unverified "
                        "constraints (exclude the current candidate "
                        "by name).\n"
                        "3. After new tasks complete, call "
                        "`prepare_report` and "
                        "`send_user_markdown_report` with an "
                        "updated answer.\n\n"
                        "Remember: the answer ALWAYS exists. "
                        "Unverified constraints suggest you may "
                        "have the wrong candidate.\n\n"
                        "IMPORTANT: Do NOT discard your current "
                        "best candidate unless you find a NEW "
                        "candidate that matches MORE constraints. "
                        "If follow-up tasks fail to find a better "
                        "alternative, KEEP your original answer. "
                        "A partially-verified answer is better "
                        "than switching to an unverified one."
                    )
                    # Remove send_user_markdown_report so orchestrator
                    # must go through prepare_report again.
                    agent._tools.pop(
                        "send_user_markdown_report", None,
                    )

                    msg_start_idx_fu = len(agent.messages)
                    orch_collector_fu = _TimingCollector(
                        inner_on_event=_on_agent_event,
                    )
                    orch_collector_fu.start()
                    agent.run_turn_streaming(
                        followup_msg,
                        on_event=orch_collector_fu.on_event,
                    )
                    _inject_timings_into_messages(
                        agent.messages,
                        orch_collector_fu,
                        msg_start_idx_fu,
                    )

                    # Take updated answer if available
                    answer_fu = report_tool.captured_report
                    if answer_fu and answer_fu != answer:
                        answer = answer_fu
                        logger.info(
                            "Constraint verification follow-up "
                            "produced updated answer (len=%d)",
                            len(answer),
                        )

            # Wait for any tasks still running, then shut down subagents
            t_cleanup = time.monotonic()
            ctx.wait_and_cleanup(timeout=300)
            timings["wait_and_cleanup"] = round(
                time.monotonic() - t_cleanup, 2,
            )

            # Clear the BBS candidate-emergence observer so its closure
            # (which captures task_board, question_text) does not leak into
            # the next multiturn turn.
            if bbs is not None:
                bbs.set_on_post(None)

            # Capture trajectory data for eval debugging before resources
            # are cleaned up.
            self._capture_trajectories(agent, ctx, task_board, bbs)

            # Aggregate token usage: orchestrator + all subagents
            all_subagents = list(ctx.subagents)

            total_usage = agent.last_turn_usage
            breakdown: dict[str, TokenUsage] = {}
            breakdown["orchestrator"] = agent.last_turn_usage
            for sa in all_subagents:
                total_usage += sa.token_usage
                if sa.token_usage.total_tokens > 0:
                    breakdown[sa.name] = sa.token_usage
            # Commit the orch + subagent breakdown BEFORE attempting tool-role
            # aggregation, so that any drain_token_ledger failure can't abandon
            # the partial state we already have.  Tool-role totals are merged
            # in afterwards if aggregation succeeds.
            self.last_token_usage = total_usage
            self.last_token_usage_breakdown = breakdown
            # Per-role tool tokens (compactor / source_scorer) — drained from
            # every agent's tool factory and aggregated across the swarm.
            try:
                tool_role_totals, tool_role_calls = aggregate_tool_role_usage(agent, all_subagents)
                for role_name, role_usage in tool_role_totals.items():
                    if role_usage.total_tokens > 0 or tool_role_calls.get(role_name, 0) > 0:
                        breakdown[role_name] = role_usage
                        total_usage += role_usage
                self.last_token_usage = total_usage
                self.last_token_usage_breakdown = breakdown
            except Exception:
                logger.warning(
                    "Tool-role token aggregation failed — keeping orch+subagent breakdown",
                    exc_info=True,
                )
            self.last_num_steps = agent.last_num_steps + sum(
                sa.total_num_steps for sa in all_subagents
            )
            orch_tokens = breakdown["orchestrator"].total_tokens
            max_sa_tokens = max(
                (sa.token_usage.total_tokens for sa in all_subagents), default=0,
            )
            self.last_total_token_e2e = orch_tokens + max_sa_tokens
            self.last_saturation_events = task_board.saturation_events
            self.last_compaction_count = agent.compaction_count + sum(
                sa.agent.compaction_count for sa in all_subagents
            )
            self.last_total_llm_calls = agent.total_llm_calls + sum(
                sa.agent.total_llm_calls for sa in all_subagents
            )
            self.last_safety_refusal_count = agent.safety_refusal_count + sum(
                sa.agent.safety_refusal_count for sa in all_subagents
            )
            self.last_content_filter_count = agent.content_filter_count + sum(
                sa.agent.content_filter_count for sa in all_subagents
            )
            # O5
            self.last_thinking_only_count = (
                getattr(agent, "thinking_only_count", 0)
                + sum(getattr(sa.agent, "thinking_only_count", 0) for sa in all_subagents)
            )
            # O2: aggregate split-compaction + peak context size.
            self.last_proactive_compaction_count = (
                getattr(agent, "proactive_compaction_count", 0)
                + sum(getattr(sa.agent, "proactive_compaction_count", 0) for sa in all_subagents)
            )
            self.last_reactive_compaction_count = (
                getattr(agent, "reactive_compaction_count", 0)
                + sum(getattr(sa.agent, "reactive_compaction_count", 0) for sa in all_subagents)
            )
            _peaks = [getattr(agent._context_budget, "peak_input_tokens", 0)]
            for sa in all_subagents:
                cb = getattr(sa.agent, "_context_budget", None)
                if cb is not None:
                    _peaks.append(getattr(cb, "peak_input_tokens", 0))
            self.last_peak_input_tokens = max(_peaks) if _peaks else 0

            # Aggregate reflection stats across all subagents
            self.last_reflection_stats = self._aggregate_reflection_stats(
                all_subagents
            )

            elapsed = time.monotonic() - t0
            timings["total"] = round(elapsed, 2)
            self.phase_timings = timings

            if on_swarm_event:
                on_swarm_event(SwarmComplete(
                    answer=answer[:200],
                    duration_seconds=elapsed,
                    subagent_count=len(ctx.subagents),
                    bbs_message_count=bbs.message_count if bbs is not None else 0,
                    report=answer,
                    token_usage=total_usage,
                    web_source_tracker=ctx.web_sources,
                ))

            # Store the report for multiturn context
            self._last_report = answer

            # Export BBS for post-hoc analysis
            self._export_bbs(bbs)

            return answer
        except Exception:
            # Best-effort: inject whatever timing we collected so far
            _inject_timings_into_messages(agent.messages, orch_collector, msg_start_idx)
            timings["total"] = round(time.monotonic() - t0, 2)
            timings["error"] = True
            self.phase_timings = timings
            # Best-effort token usage aggregation
            try:
                total_usage = agent.last_turn_usage
                breakdown = {}
                breakdown["orchestrator"] = agent.last_turn_usage
                for sa in ctx.subagents:
                    total_usage += sa.token_usage
                    if sa.token_usage.total_tokens > 0:
                        breakdown[sa.name] = sa.token_usage
                # Commit basic breakdown immediately — see matching comment in
                # the success path.  Tool-role aggregation can still raise
                # while iterating drained ledgers; preserving the orch +
                # subagent breakdown is more valuable than re-trying it.
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
                        "Tool-role token aggregation failed in exception path",
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
                # O5
                self.last_thinking_only_count = (
                    getattr(agent, "thinking_only_count", 0)
                    + sum(getattr(sa.agent, "thinking_only_count", 0) for sa in ctx.subagents)
                )
                # O2
                self.last_proactive_compaction_count = (
                    getattr(agent, "proactive_compaction_count", 0)
                    + sum(getattr(sa.agent, "proactive_compaction_count", 0) for sa in ctx.subagents)
                )
                self.last_reactive_compaction_count = (
                    getattr(agent, "reactive_compaction_count", 0)
                    + sum(getattr(sa.agent, "reactive_compaction_count", 0) for sa in ctx.subagents)
                )
                _peaks2 = [getattr(agent._context_budget, "peak_input_tokens", 0)]
                for sa in ctx.subagents:
                    cb = getattr(sa.agent, "_context_budget", None)
                    if cb is not None:
                        _peaks2.append(getattr(cb, "peak_input_tokens", 0))
                self.last_peak_input_tokens = max(_peaks2) if _peaks2 else 0
                self.last_reflection_stats = self._aggregate_reflection_stats(
                    ctx.subagents
                )
            except Exception:
                pass
            try:
                self._capture_trajectories(agent, ctx, task_board, bbs)
            except Exception:
                pass
            ctx.wait_and_cleanup(timeout=30)
            raise
        finally:
            # Don't close the orchestrator's Anthropic client — it is
            # reused across turns for multiturn conversations.  The
            # client is cleaned up in close() or reset() instead.
            pool.shutdown(wait=False)

    # -- helpers --------------------------------------------------------------

    def _aggregate_tool_role_usage(
        self,
        orchestrator_agent: Agent,
        subagents: Iterable,
    ) -> dict[str, TokenUsage]:
        """Drain compactor / source_scorer token ledgers from every agent.

        Each agent (orchestrator + each subagent) owns its own SourceScorer
        and ContentCompactor instance via its tool factory.  Each ledger is
        keyed by role label ("source_scorer", "compactor", ...).  This
        helper merges them across the swarm into ``{role: TokenUsage}``.
        """
        totals: dict[str, TokenUsage] = {}

        def _drain(obj: Any) -> None:
            if obj is None or not hasattr(obj, "drain_token_ledger"):
                return
            try:
                ledger = obj.drain_token_ledger()
            except Exception:
                return
            for role, counts in (ledger or {}).items():
                bucket = totals.setdefault(role, TokenUsage())
                bucket.input_tokens += int(counts.get("input_tokens", 0) or 0)
                bucket.output_tokens += int(counts.get("output_tokens", 0) or 0)
                bucket.cache_creation_input_tokens += int(
                    counts.get("cache_creation_input_tokens", 0) or 0,
                )
                bucket.cache_read_input_tokens += int(
                    counts.get("cache_read_input_tokens", 0) or 0,
                )

        agents = [orchestrator_agent]
        for sa in subagents:
            agent = getattr(sa, "agent", None)
            if agent is not None:
                agents.append(agent)
        for ag in agents:
            _drain(getattr(ag, "_source_scorer", None))
            _drain(getattr(ag, "_fetch_compactor", None))
            # _pdf_compactor often points at the same singleton as
            # _fetch_compactor; only drain when distinct.
            pdf = getattr(ag, "_pdf_compactor", None)
            if pdf is not None and pdf is not getattr(ag, "_fetch_compactor", None):
                _drain(pdf)
        return totals

    def _capture_trajectories(
        self,
        agent: Agent,
        ctx: SwarmContext,
        task_board: TaskBoard,
        bbs: BBS | None,
    ) -> None:
        """Capture orchestrator and subagent trajectory data for eval debugging.

        Populates ``last_orchestrator_messages``, ``last_subagent_summaries``,
        and ``last_task_summaries`` so the eval runner can serialise them into
        the trajectory JSON file.

        Thread-safety: on timeout, subagent threads may still be alive.
        We snapshot the subagent list under lock and guard every per-agent
        access individually so a single failure doesn't lose all data.
        """
        # Orchestrator conversation (may contain Anthropic Pydantic objects)
        self.last_orchestrator_messages = list(agent.messages)

        # Snapshot the subagent list under lock to avoid
        # "list changed size during iteration" from concurrent spawns.
        with ctx._lock:
            subagents_snapshot = list(ctx.subagents)

        # Capture each subagent individually — guard every access so one
        # failing subagent doesn't prevent capturing the rest.
        summaries: list[dict[str, Any]] = []
        for sa in subagents_snapshot:
            try:
                try:
                    tools = sa.agent._get_tool_definitions()
                except (RuntimeError, Exception):
                    tools = []
                try:
                    tool_counts = dict(sa.agent.tool_calls_by_name)
                except (RuntimeError, Exception):
                    tool_counts = {}
                try:
                    messages = list(sa.all_messages)
                except (RuntimeError, Exception):
                    # Fallback: try just the archive (already finalized)
                    messages = list(getattr(sa, "_message_archive", []))
                entry = {
                    "name": sa.name,
                    "model": getattr(sa.config, "model", ""),
                    "reasoning_effort": getattr(sa.config, "reasoning_effort", ""),
                    "messages": messages,
                    "system_prompt": getattr(sa.agent, "system_prompt", ""),
                    "tools": tools,
                    "tool_calls_by_name": tool_counts,
                    # Lifecycle metadata for dynamic-vs-static analysis
                    "dynamic_mode": sa._dynamic_mode,
                    "initial_profile": sa._initial_profile,
                    "tasks_completed": sa._tasks_completed,
                }
                # Drain Brave fallback log from subagent's web_search tool
                # Snapshot _tools to avoid race with concurrent profile switches.
                try:
                    tools_snapshot = dict(sa.agent._tools) if sa.agent._tools else {}
                except RuntimeError:
                    tools_snapshot = {}
                ws_tool = tools_snapshot.get("web_search")
                if ws_tool and hasattr(ws_tool, "drain_fallback_log"):
                    fallback_log = ws_tool.drain_fallback_log()
                    if fallback_log:
                        entry["web_search_fallback_log"] = fallback_log
                if ws_tool and hasattr(ws_tool, "drain_search_log"):
                    search_log = ws_tool.drain_search_log()
                    if search_log:
                        entry["web_search_log"] = search_log
                # Drain fetch log from subagent's web_fetch tool
                wf_tool = tools_snapshot.get("web_fetch")
                if wf_tool and hasattr(wf_tool, "drain_fetch_log"):
                    fetch_log = wf_tool.drain_fetch_log()
                    if fetch_log:
                        entry["web_fetch_log"] = fetch_log
                # Drain compactor log from subagent's content compactor
                sa_compactor = (
                    getattr(sa.agent, "_fetch_compactor", None)
                    or getattr(sa.agent, "_pdf_compactor", None)
                )
                if sa_compactor is not None and hasattr(
                    sa_compactor, "drain_compactor_log"
                ):
                    comp_log = sa_compactor.drain_compactor_log()
                    if comp_log:
                        entry["compactor_log"] = comp_log
                # Drain content filter log from subagent's source scorer
                cf_log = sa.agent.drain_content_filter_log()
                if cf_log:
                    entry["content_filter_log"] = cf_log
                summaries.append(entry)
            except Exception:
                logger.warning(
                    "Failed to capture trajectory for subagent %s",
                    getattr(sa, "name", "?"), exc_info=True,
                )
        self.last_subagent_summaries = summaries

        # Task board final state — snapshot under lock.
        try:
            self.last_task_summaries = [
                task.to_dict() for task in task_board.all_tasks()
            ]
        except Exception:
            logger.warning("Failed to capture task board state", exc_info=True)
            self.last_task_summaries = []

        # Spawn/assignment event log (dynamic mode only)
        try:
            self.last_spawn_events = list(ctx.spawn_events)
        except Exception:
            self.last_spawn_events = []

    def _export_bbs(self, bbs: BBS | None) -> None:
        """Export BBS to disk for post-hoc analysis."""
        if bbs is None:
            return
        from pathlib import Path
        import uuid

        export_dir = Path.home() / ".arcticswarm" / "bbs"
        try:
            session_id = uuid.uuid4().hex[:12]
            bbs.export(export_dir / f"{session_id}.json")
        except Exception as exc:
            logger.debug("BBS export failed (non-fatal): %s", exc)

    def reset(self) -> None:
        """Reset multiturn state, starting a fresh conversation.

        Closes the persistent orchestrator agent's client and clears BBS,
        turn count, and last report.  The ``SwarmOrchestrator`` itself
        remains usable — the next ``run_swarm_turn()`` call will create
        fresh state as if it were the first turn.
        """
        if self._orchestrator_agent is not None:
            try:
                self._orchestrator_agent.client.close()
            except Exception:
                pass
            if self._orchestrator_agent._orchestration_client is not self._orchestrator_agent.client:
                try:
                    self._orchestrator_agent._orchestration_client.close()
                except Exception:
                    pass
            self._orchestrator_agent = None
        self._persistent_bbs = None
        self._turn_count = 0
        self._last_report = None
        self.last_bbs = None

    def close(self) -> None:
        """Release all resources (including multiturn state)."""
        self.reset()
        if self._shared_sf_client is not None:
            try:
                self._shared_sf_client.close()
            except Exception:
                pass
