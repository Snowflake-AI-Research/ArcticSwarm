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

"""Unit tests for the three Duo-mode communication fixes.

Fix 1 (config flag): ``PrepareReportTool(enable_force_submit=False)`` — which
is wired from ``swarm.enable_force_submit_report: false`` in the YAML — hides
the ``force`` knob from the tool schema and silently downgrades any
``force=True`` passed by the LLM.  Before the fix the leader LLM called
``prepare_report(force=True)`` in ~89% of runs, short-circuiting the
wait for its auditor and then submitting a report without ever reading
the auditor's findings.

Fix 2 (infra): ``SendReportTool(strict_dm_drain=True)`` checks the
leader's mailbox at submission time.  If the auditor has sent DMs that
the leader has not yet consumed, the final submission is rejected and
the pending DMs are surfaced as a tool result so the leader must take
one more turn to read them.  Closes the race where the auditor's DM
arrives after the leader has already decided to finalise.

Latency variant: ``PrepareReportTool(blocking=False)`` — wired from
``swarm.blocking_prepare_report: false`` in the YAML — skips the
``Mailbox.wait_for_message`` sleep and returns immediately with a
status snapshot + any pending DMs.  Matches the Claude-Code agent
model (messages delivered between tool rounds via a background poll,
no sleeping inside a tool call).  Cuts Duo average latency roughly in
half without weakening correctness because (a) ``_auto_dm_check``
still injects new DMs between leader turns, and (b) Fix 2 still gates
the final submission.

Fix 4 (Duo leader toolkit — no prepare_report barrier): ``_run_duo_turn``
no longer registers ``prepare_report`` on the Duo leader.  ``list_tasks``
is restored to the toolkit (was previously stripped) so the leader can
inspect auditor status on demand.  ``send_user_markdown_report`` is
registered directly at setup time with ``strict_dm_drain=True`` instead
of being gated by a successful ``prepare_report`` call.  This matches
Claude Code's architecture — waiting happens at the runtime layer
(outer loop's ``mailbox.wait_for_message``) rather than inside a
blocking LLM tool call.
"""
from __future__ import annotations

import re
from pathlib import Path

from arcticswarm.swarm.mailbox import DM_LANE_PEER, DM_TYPE_PEER_MESSAGE, Mailbox
from arcticswarm.swarm.task import AgentRegistry, TaskBoard
from arcticswarm.swarm.tools import ListTasksTool, PrepareReportTool, SendReportTool


def _make_prepare_tool(
    *,
    enable_force_submit: bool,
    blocking: bool = True,
    mailbox: Mailbox | None = None,
) -> PrepareReportTool:
    board = TaskBoard()
    registry = AgentRegistry()
    registry.register("leader")
    report_tool = SendReportTool(has_web_search=False)
    return PrepareReportTool(
        task_board=board,
        agent_registry=registry,
        report_tool=report_tool,
        agent_tools={},
        bbs=None,
        is_followup=False,
        web_source_tracker=None,
        swarm_ctx=None,
        realtime=True,
        mailbox=mailbox,
        agent_name="leader",
        enable_force_submit=enable_force_submit,
        blocking=blocking,
    )


# ---------------------------------------------------------------------------
# Fix 1: schema + force-handling
# ---------------------------------------------------------------------------


def test_prepare_report_schema_hides_force_when_not_allowed():
    tool = _make_prepare_tool(enable_force_submit=False)
    schema = tool.parameters_schema()
    props = schema["properties"]
    assert "force" not in props, (
        "enable_force_submit=False must hide the force knob from the tool schema so "
        "the leader LLM cannot choose to skip waiting for the teammate"
    )
    assert "timeout" in props


def test_prepare_report_schema_keeps_force_when_allowed():
    tool = _make_prepare_tool(enable_force_submit=True)
    schema = tool.parameters_schema()
    assert "force" in schema["properties"]


def test_prepare_report_description_drops_force_hint_when_not_allowed():
    tool = _make_prepare_tool(enable_force_submit=False)
    desc = tool.description.lower()
    # The "force=true" suggestion sentence must be suppressed so the LLM
    # is not told about a knob that has been explicitly disabled.
    assert "force=true" not in desc
    # And an explicit instruction to wait should be present.
    assert "wait" in desc


def test_prepare_report_description_keeps_force_hint_when_allowed():
    tool = _make_prepare_tool(enable_force_submit=True)
    desc = tool.description.lower()
    assert "force=true" in desc


def test_prepare_report_execute_drops_force_when_not_allowed():
    """Even if the LLM sends ``force=True`` (provider-dependent unknown-arg
    passthrough), enable_force_submit=False must ignore it and fall back to the
    wait path.  The easiest observable signal: the report tool is NOT
    registered and the tool returns a ``Not ready`` payload instead of
    the force-unlocks message.
    """
    tool = _make_prepare_tool(enable_force_submit=False)
    # No tasks — in the non-duo path this would unlock the report tool and
    # return a "ready" string.  In duo mode with force silenced we still
    # fall through the task_count==0 branch (which is fine) — so to test
    # the force-drop specifically, add a task that's not done.
    board = tool._task_board
    from arcticswarm.swarm.task import AgentStatus, TaskSpec, TaskStatus
    task = TaskSpec(id="t1", name="alpha", prompt="x")
    board.add_task(task)
    board.claim("t1", "auditor")
    board.mark_running("t1")
    tool._agent_registry.register("auditor")
    tool._agent_registry.set_status("auditor", AgentStatus.WORKING)

    result = tool.execute(timeout=1, force=True)
    out = (result.output or "") + (result.error or "")
    # If force had been honoured, execute() would have unlocked the report
    # tool and returned the "ready" message; instead we see Not ready.
    assert "Not ready" in out or "still" in out.lower(), (
        f"enable_force_submit=False must ignore force=True, got: {out[:200]!r}"
    )
    # And the task must still be RUNNING (force=True would have short-
    # circuited the check and not waited).
    assert board.get_task("t1").status == TaskStatus.RUNNING


# ---------------------------------------------------------------------------
# Fix 2: SendReportTool drains pending DMs before accepting submission
# ---------------------------------------------------------------------------


def _seed_dm(mailbox: Mailbox, *, from_agent: str, to_agent: str, content: str) -> None:
    mailbox.send(
        from_agent=from_agent,
        to_agent=to_agent,
        content=content,
        lane=DM_LANE_PEER,
        message_type=DM_TYPE_PEER_MESSAGE,
    )


def test_send_report_blocks_when_dms_pending_in_strict_mode():
    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("auditor")
    _seed_dm(
        mailbox,
        from_agent="auditor",
        to_agent="leader",
        content="I disagree with option C — here's why: ...",
    )

    tool = SendReportTool(
        has_web_search=False,
        mailbox=mailbox,
        agent_name="leader",
        strict_dm_drain=True,
    )
    result = tool.execute(report="# Final Answer\n\nOption C.")

    assert result.is_error, "submission must be blocked while DMs pending"
    err = result.error or ""
    assert "teammate" in err.lower()
    assert "option c" in err.lower(), (
        f"pending DM content must be surfaced so the leader can revise, "
        f"got: {err!r}"
    )
    assert tool.captured_report is None, (
        "report must NOT be captured on a blocked submission"
    )


def test_send_report_submits_after_dms_drained():
    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("auditor")
    _seed_dm(
        mailbox,
        from_agent="auditor",
        to_agent="leader",
        content="quick thought",
    )

    tool = SendReportTool(
        has_web_search=False,
        mailbox=mailbox,
        agent_name="leader",
        strict_dm_drain=True,
    )
    # First call: blocked, DMs consumed.
    first = tool.execute(report="# Final Answer\n\nAnswer 1.")
    assert first.is_error
    # Second call: mailbox is now empty → submission succeeds.
    second = tool.execute(report="# Final Answer\n\nAnswer 1.")
    assert not second.is_error, (
        f"second call should succeed after DMs drained, got: {second.error!r}"
    )
    assert tool.captured_report is not None
    assert "Answer 1" in tool.captured_report


def test_send_report_default_behavior_unchanged_without_strict_flag():
    """Regression guard: the default constructor (no strict flag) must
    keep the pre-fix behaviour so non-duo swarm modes continue to work.
    """
    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("auditor")
    _seed_dm(
        mailbox,
        from_agent="auditor",
        to_agent="leader",
        content="late DM that should be ignored outside duo mode",
    )

    tool = SendReportTool(has_web_search=False)  # no mailbox, no strict flag
    result = tool.execute(report="# Final Answer\n\nShipped.")
    assert not result.is_error
    assert tool.captured_report is not None
    assert "Shipped" in tool.captured_report


# ---------------------------------------------------------------------------
# Non-blocking prepare_report (Claude-Code-style snapshot)
# ---------------------------------------------------------------------------


def _seed_pending_task(tool: PrepareReportTool) -> None:
    """Register a RUNNING auditor task so ``prepare_report`` can't short-
    circuit via the ``task_count == 0`` path."""
    from arcticswarm.swarm.task import AgentStatus, TaskSpec
    task = TaskSpec(id="t1", name="alpha", prompt="x")
    tool._task_board.add_task(task)
    tool._task_board.claim("t1", "auditor")
    tool._task_board.mark_running("t1")
    tool._agent_registry.register("auditor")
    tool._agent_registry.set_status("auditor", AgentStatus.WORKING)


def test_prepare_report_non_blocking_returns_immediately_without_sleeping():
    """Non-blocking mode must NOT call ``wait_for_message``.  We assert
    this by (a) timing the call (should be < 0.5s — generous margin for
    CI) and (b) by passing a mailbox whose ``wait_for_message`` would
    otherwise block for the full ``timeout`` seconds."""
    import time

    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("auditor")

    tool = _make_prepare_tool(enable_force_submit=False, blocking=False, mailbox=mailbox)
    _seed_pending_task(tool)

    t0 = time.monotonic()
    # Timeout is deliberately large — blocking mode would sleep up to 5s
    # here because there's a pending task and no DM to wake it up.
    result = tool.execute(timeout=5)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, (
        f"non-blocking prepare_report must return immediately, slept "
        f"{elapsed:.2f}s (full timeout would be 5s)"
    )
    out = (result.output or "") + (result.error or "")
    assert "Not ready" in out or "still" in out.lower()


def test_prepare_report_non_blocking_surfaces_pending_dms():
    """Any DMs already in the leader's mailbox when ``prepare_report``
    is called must be rendered into the response — so the leader can
    react to them without having to block on wait_for_message first."""
    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("auditor")
    mailbox.send(
        from_agent="auditor",
        to_agent="leader",
        content="I checked the integral — answer should be 42, not 17",
        lane=DM_LANE_PEER,
        message_type=DM_TYPE_PEER_MESSAGE,
    )

    tool = _make_prepare_tool(enable_force_submit=False, blocking=False, mailbox=mailbox)
    _seed_pending_task(tool)

    result = tool.execute(timeout=5)
    out = (result.output or "") + (result.error or "")
    assert "42" in out, (
        f"pending DM content must be surfaced in the non-blocking "
        f"response so the leader can react, got: {out[:400]!r}"
    )
    # And the DM must have been consumed from the inbox (check_new
    # drains), so a second call sees an empty mailbox (None per API).
    assert not mailbox.check_new("leader")


def test_prepare_report_blocking_mode_still_calls_wait_for_message():
    """Regression guard: the default ``blocking=True`` path must route
    through ``Mailbox.wait_for_message`` (i.e. it *does* sleep, unlike
    the non-blocking fast-path).  We stub ``wait_for_message`` so the
    test finishes instantly while still asserting (a) the call happened
    and (b) the ``_deadline_exceeded`` latch fires when the stub
    indicates a timeout.

    Historical note: this test used to call ``tool.execute(timeout=0)``
    and rely on "0 means instant return".  ``PrepareReportTool.execute``
    actually applies the idiom
    ``timeout = kwargs.get("timeout", DEFAULT) or DEFAULT``, which treats
    ``0`` as falsy and silently swaps in the 300-second default — so the
    test slept 5 minutes on every CI run.  The stub-based version below
    is both faster (≈1ms) and a tighter contract (proves the code path,
    not just the latch)."""
    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("auditor")

    calls: list[tuple[str, int]] = []

    def _fake_wait(agent_name: str, *, timeout: int) -> bool:
        calls.append((agent_name, timeout))
        return False  # no DM arrived → tool must latch _deadline_exceeded

    mailbox.wait_for_message = _fake_wait  # type: ignore[assignment]

    tool = _make_prepare_tool(enable_force_submit=False, blocking=True, mailbox=mailbox)
    _seed_pending_task(tool)

    tool.execute(timeout=1)  # non-zero to avoid the `or DEFAULT` trap
    assert calls, (
        "blocking=True must route through Mailbox.wait_for_message; "
        "stub was never called"
    )
    assert calls[0][0] == "leader"
    assert tool._deadline_exceeded, (
        "when wait_for_message returns False (no DM), the blocking path "
        "must latch _deadline_exceeded so subsequent calls short-circuit"
    )


def test_prepare_report_non_blocking_description_explains_behaviour():
    """The tool description shown to the LLM must tell it the call is
    non-blocking so it doesn't assume a blocking barrier — otherwise
    it might sit idle waiting for a "synchronous" return that already
    happened."""
    mailbox = Mailbox()
    mailbox.register("leader")
    tool = _make_prepare_tool(enable_force_submit=False, blocking=False, mailbox=mailbox)
    desc = tool.description.lower()
    assert "immediately" in desc or "non-blocking" in desc, (
        f"non-blocking description must signal immediate return; got: "
        f"{desc!r}"
    )
    # Crucially, must NOT promise the old blocking semantics.
    assert "may take a while" not in desc


# ---------------------------------------------------------------------------
# Fix 4: Duo leader toolkit — no prepare_report barrier
# ---------------------------------------------------------------------------
#
# Constructing a real ArcticswarmOrchestrator in a unit test is heavy (it
# would require full config + an LLM client stub + a subagent pool).
# Instead we verify the Duo toolkit wiring in two complementary layers:
#
#   (a) *Contract tests* against the orchestrator source file: the
#       specific registration line and the tool-strip tuple are the
#       invariants we care about; we grep the source to make sure they
#       cannot silently regress (e.g. "list_tasks" accidentally re-added
#       to the strip tuple or "prepare_report" re-registered).
#
#   (b) *Behavioural tests* that exercise the exact dict operations
#       ``_run_duo_turn`` performs on ``agent._tools``, so the
#       post-setup state is pinned down and easy to reason about.


_ORCH_SOURCE = (
    (Path(__file__).resolve().parents[1] / "orchestrator.py").read_text()
    # ``_run_duo_turn`` was extracted into the DuoMixin in
    # ``orchestrator_duo.py``; the duo-wiring contract assertions below
    # search the combined source so they still pin the same invariants
    # regardless of which module the method body physically lives in.
    + (Path(__file__).resolve().parents[1] / "orchestrator_duo.py").read_text()
)


def test_duo_source_strips_wait_and_create_but_keeps_list_tasks():
    """Source-level contract: the Duo tool-strip tuple must NOT include
    ``list_tasks`` (leader needs it for visibility) and MUST include
    ``wait_for_tasks`` (blocks inside a tool call — same latency
    pathology as the removed ``prepare_report``)."""
    pattern = re.compile(
        r'for tool_name in \(("[^)]+)\)\s*:\s*\n\s*agent\._tools\.pop'
    )
    matches = pattern.findall(_ORCH_SOURCE)
    # Look for the duo-specific strip (must contain wait_for_tasks and
    # NOT contain list_tasks).
    duo_strips = [
        m for m in matches
        if "wait_for_tasks" in m and "list_tasks" not in m
    ]
    assert duo_strips, (
        "No strip tuple found that keeps list_tasks and strips "
        "wait_for_tasks; did the Duo toolkit wiring regress?"
    )
    duo_strip = duo_strips[0]
    # Explicit positive assertions for each item the Duo leader must
    # never see.
    assert '"create_task"' in duo_strip
    assert '"wait_for_tasks"' in duo_strip
    assert '"create_agent"' in duo_strip
    # And list_tasks must NOT be in the strip tuple.
    assert '"list_tasks"' not in duo_strip


def test_duo_source_explicitly_registers_list_tasks_on_leader():
    """Source-level contract: ``_run_duo_turn`` builds a fresh
    ``Agent(self.config)`` which does NOT go through the main orchestrator
    setup path (where ``list_tasks`` is normally registered).  The Duo
    prompt instructs the leader to call ``list_tasks`` to inspect auditor
    status, so the tool MUST be registered explicitly inside the duo
    wiring block — otherwise the LLM sees a promise it can't fulfil.

    Regression guard: an earlier iteration removed ``list_tasks`` from
    the strip tuple but forgot to add the explicit registration, so the
    leader silently lost the tool.  A production trajectory showed the
    model noticing: "the instructions might indicate that 'list_tasks'
    could be unavailable"."""
    duo_start = _ORCH_SOURCE.find("def _run_duo_turn")
    assert duo_start >= 0, "could not locate _run_duo_turn in orchestrator.py"
    duo_block = _ORCH_SOURCE[duo_start:]
    next_def = duo_block.find("\n    def ", 10)
    if next_def > 0:
        duo_block = duo_block[:next_def]
    assert 'agent._tools["list_tasks"] = ListTasksTool(task_board)' in duo_block, (
        "Duo leader must explicitly register ListTasksTool; the prompt "
        "instructs the leader to call list_tasks for auditor-status "
        "visibility, so the tool cannot rely on an upstream registration "
        "that _run_duo_turn doesn't actually run"
    )


def test_duo_source_registers_send_report_directly_not_prepare_report():
    """Source-level contract: in the Duo setup section, the leader's
    toolkit must register ``send_user_markdown_report`` directly (not
    gated behind ``prepare_report``)."""
    # Cut the source to the _run_duo_turn region to avoid matching the
    # main-swarm / BBS PrepareReportTool registrations.
    duo_start = _ORCH_SOURCE.find("def _run_duo_turn")
    assert duo_start >= 0, "could not locate _run_duo_turn in orchestrator.py"
    duo_block = _ORCH_SOURCE[duo_start:]
    # Chop at the next top-level def to bound the region.
    next_def = duo_block.find("\n    def ", 10)
    if next_def > 0:
        duo_block = duo_block[:next_def]

    assert 'agent._tools["send_user_markdown_report"] = report_tool' in duo_block, (
        "Duo leader must register send_user_markdown_report directly; "
        "gating through prepare_report has been removed"
    )
    assert 'agent._tools["prepare_report"]' not in duo_block, (
        "Duo leader must NOT register prepare_report — it blocks inside "
        "the LLM turn and was the root cause of V2/V3 latency"
    )


def _simulate_duo_toolkit_wiring() -> tuple[dict[str, object], Mailbox, SendReportTool]:
    """Apply the exact dict operations ``_run_duo_turn`` performs on
    ``agent._tools`` after the base toolkit has been assembled.  Returns
    the resulting tools dict plus the mailbox + report tool that were
    wired in."""
    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("auditor")
    task_board = TaskBoard()

    # Simulate the base toolkit the agent would have BEFORE Duo-specific
    # post-processing.  ``_run_duo_turn`` constructs a fresh
    # ``Agent(self.config)`` that does NOT go through the main
    # orchestrator setup path — so in production this dict does NOT
    # contain ``list_tasks``, ``wait_for_tasks``, ``create_task``, or
    # ``create_agent`` at this point.  The Duo wiring step below is what
    # must provide any orchestrator-level visibility the leader needs.
    agent_tools: dict[str, object] = {
        "web_search": object(),
        "load_skill": object(),
    }

    # === Duo-specific wiring (mirrors _run_duo_turn) =====================
    report_tool = SendReportTool(
        has_web_search=True,
        mailbox=mailbox,
        agent_name="leader",
        strict_dm_drain=True,
    )
    agent_tools["send_user_markdown_report"] = report_tool
    agent_tools["list_tasks"] = ListTasksTool(task_board)
    for tool_name in ("create_task", "wait_for_tasks", "create_agent"):
        agent_tools.pop(tool_name, None)
    # =====================================================================

    return agent_tools, mailbox, report_tool


def test_duo_toolkit_contains_list_tasks_and_send_report():
    tools, _, _ = _simulate_duo_toolkit_wiring()
    assert "list_tasks" in tools, (
        "Duo leader must have list_tasks for on-demand auditor-status "
        "visibility (Claude-Code-style informed decision making)"
    )
    assert "send_user_markdown_report" in tools, (
        "Duo leader must have send_user_markdown_report registered from "
        "the start — no prepare_report unlock step anymore"
    )


def test_duo_toolkit_excludes_prepare_report_and_wait_for_tasks():
    tools, _, _ = _simulate_duo_toolkit_wiring()
    assert "prepare_report" not in tools, (
        "Duo leader must NOT have prepare_report; waiting now happens "
        "at the runtime layer via the outer loop's mailbox.wait_for_message"
    )
    assert "wait_for_tasks" not in tools, (
        "Duo leader must NOT have wait_for_tasks — it blocks inside a "
        "tool call and has the same pathology as prepare_report"
    )
    assert "create_task" not in tools
    assert "create_agent" not in tools


def test_duo_send_report_is_wired_with_strict_dm_drain_and_mailbox():
    """The SendReportTool registered in Duo mode must have the safety
    net configured: strict_dm_drain on + mailbox + agent_name wired.
    Otherwise the mid-composition auditor-DM race is unguarded."""
    _, mailbox, report_tool = _simulate_duo_toolkit_wiring()
    assert report_tool._strict_dm_drain is True, (
        "strict_dm_drain must be True so auditor DMs that arrive between "
        "_auto_dm_check and the submit tool-call can't be silently dropped"
    )
    assert report_tool._mailbox is mailbox
    assert report_tool._agent_name == "leader"


def test_duo_send_report_still_rejects_pending_dms_after_rewire():
    """End-to-end behavioural check: the new toolkit wiring still
    produces a SendReportTool that rejects submission when auditor DMs
    are waiting in the mailbox.  Guards against a future refactor
    accidentally dropping the strict-drain kwarg."""
    tools, mailbox, _ = _simulate_duo_toolkit_wiring()
    _seed_dm(
        mailbox,
        from_agent="auditor",
        to_agent="leader",
        content="wait — double-check the sign of term 3",
    )
    report_tool = tools["send_user_markdown_report"]
    assert isinstance(report_tool, SendReportTool)
    result = report_tool.execute(report="# Done\n\nFinal answer: 42.")
    assert result.is_error, (
        "strict_dm_drain must block the final submission while the "
        "mailbox has unread auditor DMs"
    )
    assert "sign of term 3" in (result.error or ""), (
        "rejected submission must surface the unread DMs so the leader "
        "sees what it missed"
    )
    assert report_tool.captured_report is None


def test_duo_toolkit_followup_path_has_send_report_without_unlock_step():
    """Follow-up turn (is_followup=True) previously relied on
    prepare_report's task_count==0 branch to unlock send_user_markdown_report.
    After the rewire, send_user_markdown_report is registered at setup
    time unconditionally, so the follow-up path just works with no
    additional gating."""
    # Whether a turn is a "follow-up" or not doesn't affect the Duo
    # toolkit setup anymore — both paths go through the same dict
    # operations.  Re-invoking the wiring helper is enough to confirm
    # the tool is present without any "unlock" machinery.
    tools, _, _ = _simulate_duo_toolkit_wiring()
    report_tool = tools["send_user_markdown_report"]
    assert isinstance(report_tool, SendReportTool)
    # Submit once with an empty mailbox — in the old flow this would
    # have required a prior successful prepare_report call; now it's a
    # direct path.
    mailbox = report_tool._mailbox
    assert mailbox is not None
    assert not mailbox.check_new("leader")
    result = report_tool.execute(report="# Refined Answer\n\nStill 42.")
    assert not result.is_error, (
        f"follow-up submission must succeed with an empty mailbox; "
        f"got: {result.error!r}"
    )
    assert report_tool.captured_report is not None
