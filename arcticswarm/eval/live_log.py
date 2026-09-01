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

"""Live, per-case activity feed for ``arcticswarm-eval``.

The eval CLI runs many cases in parallel through a thread pool and, by
default, only shows a progress bar plus one summary line per *finished*
case.  This module adds the interactive, streaming feed you get from the
``arcticswarm`` REPL (the :class:`~arcticswarm.swarm.viewer.SwarmRenderer`
"Swarm Live" panel), but flattened into single prefixed lines so the
output from several concurrent cases can be multiplexed onto one console.

Each swarm / single-agent event becomes one line::

    [browsecomp_0a3f      ] (orchestrator) posting task 'find_founding_year'
    [browsecomp_0a3f      ] worker-2 claimed - find_founding_year
    [browsecomp_0a3f      ] worker-2 web_search "acme corp founded"
    [browsecomp_91c2      ] worker-1 FAILED - Timed out after 300s

Design choices (see the conversation that introduced this):

* **conv_id prefix** — every line is prefixed with the case's ``conv_id``
  so you can tell which parallel case is doing what.  Each ``conv_id`` is
  assigned a stable colour (deterministic via CRC32) so a glance groups
  lines by case.  The colour palette deliberately excludes red.
* **errors in red** — failures (``TeammateFailed``, tool errors) render
  the message in red.  Everything else is plain.
* **normal progress is plain** — orchestrator tool calls
  (``(orchestrator) create task ...``) and subagent actions print with the
  default style.  Low-signal lifecycle events (spawned / joined / started)
  are dimmed.
* **curated by default** — only high-signal events are shown: orchestrator
  tool calls, subagent claim/complete/fail, and substantive subagent
  actions (search / fetch / pdf / bash).  Reasoning text, BBS reads and
  file reads are skipped to keep the feed readable at high parallelism.

The renderer is intentionally decoupled from the event classes: it
dispatches on ``type(event).__name__`` and reads attributes via
``getattr`` so it never imports the orchestrator / agent modules (no
import cycle, no import-time cost when the feed is disabled).

Thread-safety: :meth:`LiveEvalLogger.on_event` is called from worker and
subagent threads.  All console output goes through a single
:class:`rich.console.Console`, whose ``print`` is internally locked, so
lines never interleave mid-line.  When the eval is showing a
``rich.progress`` bar, pass that progress bar's ``console`` so the feed
lines render *above* the live bar.
"""

from __future__ import annotations

import threading
import zlib
from typing import Any

from rich.console import Console
from rich.text import Text

__all__ = ["LiveEvalLogger"]


# Per-case prefix colours.  Red is excluded (reserved for errors) and so are
# plain white / default (reserved for normal message text), so the prefix
# stays visually distinct from both.
_PREFIX_PALETTE: tuple[str, ...] = (
    "cyan",
    "green",
    "yellow",
    "magenta",
    "blue",
    "bright_cyan",
    "bright_green",
    "bright_yellow",
    "bright_magenta",
    "bright_blue",
)

# Subagent tool calls worth surfacing in the curated feed — the substantive
# "what is this worker actually doing" actions.  Everything else (read_bbs,
# read_dm, read_file, glob, grep, list_tasks, reasoning, calculator, ...) is
# noise at high parallelism and is dropped.
_CURATED_TEAMMATE_TOOLS: frozenset[str] = frozenset({
    "web_search",
    "web_fetch",
    "pdf_read",
    "bash",
    "post_to_bbs",
    "complete_task",
})

# Orchestrator tool calls to drop from the curated feed — pure status reads
# that add no signal.  All other orchestrator tool calls (create_task,
# wait_for_tasks, post_to_bbs, prepare_report, web_search, ...) are shown.
_ORCH_TOOLS_SKIP: frozenset[str] = frozenset({
    "list_tasks",
    "read_bbs",
    "read_dm",
})

# Single-agent (non-swarm) tool calls worth surfacing.
_CURATED_AGENT_TOOLS: frozenset[str] = frozenset({
    "web_search",
    "web_fetch",
    "pdf_read",
    "bash",
})

# Message style sentinels returned by ``_format``.
_PLAIN = None       # default terminal colour
_ERROR = "red"      # failures / tool errors
_META = "dim"       # low-signal lifecycle (spawned / started / joined)

# Cap a single feed line's message so a giant bash command / error does not
# flood the terminal.  Newlines are collapsed to keep one event per line;
# the line is additionally ellipsis-cropped at the console width on print.
_MAX_MSG_CHARS = 200


def _truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _summarize_agent_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """One-line summary of a single-agent tool call (mirrors the swarm summary)."""
    if tool_name == "web_search":
        query = str(tool_input.get("query", "")).strip()
        return f'web_search "{_truncate(query, 160)}"' if query else "web_search"
    if tool_name == "web_fetch":
        url = str(tool_input.get("url", "")).strip()
        return f"web_fetch {_truncate(url, 200)}" if url else "web_fetch"
    if tool_name == "pdf_read":
        target = str(tool_input.get("url") or tool_input.get("file_path") or "").strip()
        return f"pdf_read {_truncate(target, 200)}" if target else "pdf_read"
    if tool_name == "bash":
        cmd = str(tool_input.get("command", "")).strip()
        return f"$ {_truncate(cmd, 200)}" if cmd else "bash"
    return tool_name


class LiveEvalLogger:
    """Render eval swarm / single-agent events as prefixed console lines.

    Parameters
    ----------
    console:
        The :class:`rich.console.Console` to print on.  When a
        ``rich.progress.Progress`` bar is active, pass ``progress.console``
        so feed lines render above the bar.
    prefix_width:
        Fixed column width for the ``[conv_id]`` prefix so message columns
        align.  ``conv_id`` values longer than this are truncated.
    """

    def __init__(self, console: Console, *, prefix_width: int = 24) -> None:
        self._console = console
        self._prefix_width = max(8, prefix_width)
        self._lock = threading.Lock()
        # Stable conv_id -> colour assignment cache (CRC32 is deterministic
        # across processes, unlike the salted built-in ``hash``).
        self._colors: dict[str, str] = {}

    # -- public API ----------------------------------------------------------

    def on_event(self, conv_id: str, event: Any) -> None:
        """Format and print one feed line for *event* (skips low-signal events)."""
        try:
            formatted = self._format(event)
        except Exception:
            # The feed is observability only — never let a formatting bug
            # take down an eval case.
            return
        if formatted is None:
            return
        message, style = formatted
        message = _truncate(message, _MAX_MSG_CHARS)
        if not message:
            return

        line = Text()
        line.append(self._prefix(conv_id), style=self._color_for(conv_id))
        line.append(" ")
        line.append(message, style=style)
        # One physical line per event so every line carries a conv_id prefix
        # (continuation lines without a prefix are confusing once several
        # parallel cases interleave).  Over-long lines are ellipsis-cropped at
        # the console width; the full detail lives in the trajectory JSON.
        self._console.print(line, no_wrap=True, overflow="ellipsis", crop=True)

    # -- internals -----------------------------------------------------------

    def _prefix(self, conv_id: str) -> str:
        w = self._prefix_width
        short = conv_id if len(conv_id) <= w else conv_id[: w - 1] + "…"
        return f"[{short.ljust(w)}]"

    def _color_for(self, conv_id: str) -> str:
        with self._lock:
            color = self._colors.get(conv_id)
            if color is None:
                idx = zlib.crc32(conv_id.encode("utf-8")) % len(_PREFIX_PALETTE)
                color = _PREFIX_PALETTE[idx]
                self._colors[conv_id] = color
            return color

    def _format(self, event: Any) -> tuple[str, str | None] | None:
        """Map *event* to ``(message, style)`` or ``None`` to skip it.

        Dispatches on the class name so this module stays decoupled from the
        orchestrator / agent event hierarchies.
        """
        name = type(event).__name__

        # ---- Swarm orchestrator events -----------------------------------
        if name == "SwarmStarted":
            return ("swarm started", _META)

        if name == "OrchestratorToolCall":
            tool = getattr(event, "tool_name", "")
            if tool in _ORCH_TOOLS_SKIP:
                return None
            desc = getattr(event, "description", "") or f"called {tool}"
            return (f"(orchestrator) {desc}", _PLAIN)

        if name == "ReportStarted":
            return ("(orchestrator) writing report...", _PLAIN)

        if name == "SubagentSpawned":
            return (f"{getattr(event, 'name', '?')} joined", _META)

        if name == "TeammateSpawned":
            who = getattr(event, "name", "?")
            prompt = _truncate(getattr(event, "prompt", "") or "", 120)
            return (f"{who} spawned - {prompt}" if prompt else f"{who} spawned", _META)

        if name == "SubagentClaimedTask":
            who = getattr(event, "name", "?")
            activity = _truncate(getattr(event, "activity", "") or "", 160)
            return (f"{who} claimed - {activity}" if activity else f"{who} claimed", _PLAIN)

        if name == "TeammateToolCall":
            tool = getattr(event, "tool_name", "")
            if tool not in _CURATED_TEAMMATE_TOOLS:
                return None
            who = getattr(event, "name", "?")
            desc = getattr(event, "description", "") or f"called {tool}"
            return (f"{who} {desc}", _PLAIN)

        if name == "TeammateCompleted":
            return (f"{getattr(event, 'name', '?')} completed task", _PLAIN)

        if name == "TeammateFailed":
            who = getattr(event, "name", "?")
            err = _truncate(getattr(event, "error", "") or "", 200)
            return (f"{who} FAILED - {err}", _ERROR)

        # ---- Single-agent (non-swarm) stream events ----------------------
        if name == "ToolCallStart":
            tool = getattr(event, "tool_name", "")
            if tool not in _CURATED_AGENT_TOOLS:
                return None
            summary = _summarize_agent_tool(tool, getattr(event, "tool_input", {}) or {})
            return (summary, _PLAIN)

        if name == "ToolCallEnd":
            result = getattr(event, "result", None)
            if result is not None and getattr(result, "is_error", False):
                tool = getattr(event, "tool_name", "")
                err = _truncate(getattr(result, "error", "") or "", 200)
                return (f"{tool} error: {err}", _ERROR)
            return None

        # Everything else (OrchestratorMessage, OrchestratorTextDelta,
        # ReportDelta, SubagentIdle, SubagentSurfing, SwarmComplete,
        # TextDelta, TurnComplete, ToolInputDelta, ...) is intentionally
        # skipped in the curated feed.
        return None
