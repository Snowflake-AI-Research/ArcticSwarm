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

"""Tests for the live eval feed (:mod:`arcticswarm.eval.live_log`).

Covers the curated event filtering, the conv_id prefix, error colouring,
and the enhanced web-tool summaries that feed it.
"""

from __future__ import annotations

import io

from rich.console import Console

from arcticswarm.agent import ToolCallEnd, ToolCallStart, TextDelta
from arcticswarm.eval.live_log import LiveEvalLogger
from arcticswarm.swarm.orchestrator import (
    OrchestratorMessage,
    OrchestratorToolCall,
    SubagentClaimedTask,
    SubagentIdle,
    SubagentSpawned,
    SwarmComplete,
    TeammateCompleted,
    TeammateFailed,
    TeammateToolCall,
    _summarize_tool_call,
)
from arcticswarm.tools.base import ToolResult


def _make_logger(*, force_terminal: bool = False, width: int = 200):
    """Return ``(logger, get_output)`` writing to an in-memory console."""
    buf = io.StringIO()
    console = Console(
        file=buf,
        width=width,
        force_terminal=force_terminal,
        color_system="standard" if force_terminal else None,
        highlight=False,
    )
    return LiveEvalLogger(console, prefix_width=18), (lambda: buf.getvalue())


# ---------------------------------------------------------------------------
# Curated filtering: which events produce a line
# ---------------------------------------------------------------------------


def test_shown_events_produce_one_prefixed_line():
    log, out = _make_logger()
    cid = "browsecomp_0a3f"

    log.on_event(cid, OrchestratorToolCall(tool_name="create_task", description="posting task 'x'"))
    log.on_event(cid, SubagentClaimedTask(name="worker-2", activity="find year"))
    log.on_event(cid, TeammateToolCall(name="worker-2", tool_name="web_search", description='web_search "acme"'))
    log.on_event(cid, TeammateCompleted(name="worker-2"))

    text = out()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 4
    # Every line is prefixed with the (truncated/padded) conv_id.
    assert all(ln.startswith("[browsecomp_0a3f") for ln in lines)
    assert "(orchestrator) posting task 'x'" in text
    assert "worker-2 claimed - find year" in text
    assert 'worker-2 web_search "acme"' in text
    assert "worker-2 completed task" in text


def test_low_signal_events_are_skipped():
    log, out = _make_logger()
    cid = "c1"

    # Status-read orchestrator tools, low-signal teammate tools, reasoning
    # text, idle, and completion summary are all curated out.
    log.on_event(cid, OrchestratorToolCall(tool_name="list_tasks", description="checking task status"))
    log.on_event(cid, TeammateToolCall(name="w", tool_name="read_bbs", description="reading BBS"))
    log.on_event(cid, OrchestratorMessage(text="let me think"))
    log.on_event(cid, SubagentIdle(name="w", activity="ready"))
    log.on_event(cid, SwarmComplete(subagent_count=3, bbs_message_count=10, duration_seconds=1.0))
    log.on_event(cid, TextDelta(text="hello"))

    assert out().strip() == ""


def test_substantive_single_agent_tool_calls_shown_reads_skipped():
    log, out = _make_logger()
    cid = "c1"

    log.on_event(cid, ToolCallStart(tool_name="web_search", tool_input={"query": "acme corp"}))
    log.on_event(cid, ToolCallStart(tool_name="read_file", tool_input={"file_path": "/x"}))  # skipped
    log.on_event(cid, ToolCallEnd(tool_name="web_search", result=ToolResult(output="ok", is_error=False)))  # skipped

    text = out()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert 'web_search "acme corp"' in text


# ---------------------------------------------------------------------------
# Error colouring (red) vs normal progress (plain)
# ---------------------------------------------------------------------------


def test_teammate_failure_is_red():
    log, out = _make_logger(force_terminal=True)
    log.on_event("c1", TeammateFailed(name="worker-1", error="Timed out after 300s"))
    text = out()
    assert "worker-1 FAILED - Timed out after 300s" in text
    assert "\x1b[31m" in text  # red SGR somewhere on the line


def test_tool_error_is_red():
    log, out = _make_logger(force_terminal=True)
    log.on_event(
        "c1",
        ToolCallEnd(tool_name="web_fetch", result=ToolResult(output="", error="403 Forbidden", is_error=True)),
    )
    text = out()
    assert "web_fetch error: 403 Forbidden" in text
    assert "\x1b[31m" in text


def test_normal_progress_has_no_red():
    log, out = _make_logger(force_terminal=True)
    log.on_event("c1", OrchestratorToolCall(tool_name="create_task", description="posting task 'x'"))
    assert "\x1b[31m" not in out()  # plain message, no red


# ---------------------------------------------------------------------------
# Per-case prefix colour is stable and distinguishes cases
# ---------------------------------------------------------------------------


def test_prefix_colour_is_stable_per_case_and_differs_between_cases():
    log, _ = _make_logger()
    a1 = log._color_for("browsecomp_0a3f")
    a2 = log._color_for("browsecomp_0a3f")
    b = log._color_for("totally_different_id_xyz")
    assert a1 == a2  # stable
    assert "red" not in (a1, b)  # red reserved for errors
    # Not asserting a1 != b in general (palette collisions possible), but the
    # two chosen ids must hash to different palette slots here.
    assert a1 != b


def test_formatting_error_never_raises():
    log, _ = _make_logger()

    class Boom:
        @property
        def tool_name(self):  # noqa: D401 - trigger an exception in _format
            raise RuntimeError("boom")

    # Must swallow the error rather than propagate into the eval case.
    log.on_event("c1", OrchestratorToolCall.__new__(OrchestratorToolCall))  # missing attrs ok
    log.on_event("c1", Boom())


# ---------------------------------------------------------------------------
# Enhanced _summarize_tool_call web branches (feed depends on these)
# ---------------------------------------------------------------------------


def test_summarize_web_tool_calls_show_query_and_url():
    assert _summarize_tool_call("web_search", {"query": "acme corp founded"}) == 'web_search "acme corp founded"'
    assert _summarize_tool_call("web_fetch", {"url": "https://example.com"}) == "web_fetch https://example.com"
    assert _summarize_tool_call("pdf_read", {"url": "https://x/y.pdf"}) == "pdf_read https://x/y.pdf"
    assert _summarize_tool_call("pdf_read", {"file_path": "/tmp/a.pdf"}) == "pdf_read /tmp/a.pdf"
    # No args → bare tool name, no crash.
    assert _summarize_tool_call("web_search", {}) == "web_search"
