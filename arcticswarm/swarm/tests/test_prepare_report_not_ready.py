"""Unit tests for PrepareReportTool's enriched ``Not ready`` payload and
the shared :func:`_render_task_activity_lines` helper.

Before this change, the DM-rt orchestrator's ``prepare_report`` returned
a one-sentence response ("some subagents are still active") that hid
every useful signal.  With the helper reused from ``WaitForTasksTool``,
both the mailbox-plumbed branch and the legacy else branch now surface:

* pending task names + live status,
* per-task ``activity:`` heartbeat (tool + input preview + age),
* ``STALE`` flag when the heartbeat is older than the threshold.

This matches what ``list_tasks`` / ``wait_for_tasks`` show, so the
orchestrator can make an informed ``force=true`` decision without a
second tool call.
"""
from __future__ import annotations

import time

from arcticswarm.swarm.mailbox import DM_LANE_RESULT, DM_TYPE_TASK_COMPLETED, Mailbox
from arcticswarm.swarm.task import (
    STALE_HEARTBEAT_THRESHOLD_SECONDS,
    AgentRegistry,
    AgentStatus,
    TaskBoard,
    TaskSpec,
    TaskStatus,
)
from arcticswarm.swarm.tools import (
    PrepareReportTool,
    SendReportTool,
    WaitForTasksTool,
    _render_task_activity_lines,
)


def _build_tool(
    *,
    with_mailbox: bool,
) -> tuple[TaskBoard, AgentRegistry, Mailbox | None, PrepareReportTool]:
    board = TaskBoard()
    task = TaskSpec(id="t1", name="alpha", prompt="do alpha")
    board.add_task(task)
    board.claim("t1", "wallace")
    board.mark_running("t1")
    # One tool-use heartbeat so the activity line has something to render.
    board.bump_progress(
        "t1", tool_name="web_search", input_preview="arrhenius", tokens=12_345,
    )

    registry = AgentRegistry()
    registry.register("wallace")
    registry.set_status("wallace", AgentStatus.WORKING, activity="web_search")

    mailbox: Mailbox | None = None
    agent_name: str | None = None
    if with_mailbox:
        mailbox = Mailbox()
        mailbox.register("leader")
        mailbox.register("wallace")
        agent_name = "leader"

    report_tool = SendReportTool(has_web_search=True)
    agent_tools: dict = {}
    tool = PrepareReportTool(
        task_board=board,
        agent_registry=registry,
        report_tool=report_tool,
        agent_tools=agent_tools,
        bbs=None,
        is_followup=False,
        web_source_tracker=None,
        swarm_ctx=None,
        realtime=True,
        mailbox=mailbox,
        agent_name=agent_name,
    )
    return board, registry, mailbox, tool


def test_not_ready_no_mailbox_renders_activity():
    """Legacy else branch (mailbox=None) must show the task activity line
    in the response, not just "some subagents are still active".
    """
    _, _, _, tool = _build_tool(with_mailbox=False)
    result = tool.execute(timeout=1)
    out = result.output or ""
    assert "Not ready" in out
    assert "alpha" in out                             # task name rendered
    assert "activity:" in out                         # heartbeat line rendered
    assert "web_search" in out                        # last tool name rendered
    assert "tool uses" in out                         # heartbeat counter


def test_not_ready_with_mailbox_renders_activity_and_dm():
    """Mailbox-plumbed branch: a DM arriving during the wait is rendered
    below the pending task list so the orchestrator can act on the
    teammate's findings without calling list_tasks first.
    """
    _, _, mailbox, tool = _build_tool(with_mailbox=True)
    assert mailbox is not None
    # Deliver a DM BEFORE execute so wait_for_message returns True
    # immediately.  Not ready because task alpha is still RUNNING.
    mailbox.send(
        from_agent="wallace",
        to_agent="leader",
        content="[Task completed: alpha] early peek",
        lane=DM_LANE_RESULT,
        message_type=DM_TYPE_TASK_COMPLETED,
        payload={"task_id": "t1", "task_name": "alpha", "status": "running"},
    )

    result = tool.execute(timeout=2)
    out = result.output or ""
    assert "Not ready" in out
    assert "alpha" in out
    assert "activity:" in out
    assert "web_search" in out
    assert "early peek" in out                        # DM content surfaced
    assert "your teammate" in out.lower()             # DM framing line


def test_not_ready_shows_stale_flag():
    """Aged heartbeats must trigger the STALE flag, mirroring
    wait_for_tasks behavior.
    """
    board, _, _, tool = _build_tool(with_mailbox=False)
    stored = board.get_task("t1")
    assert stored is not None
    stored.last_heartbeat = time.monotonic() - (
        STALE_HEARTBEAT_THRESHOLD_SECONDS + 60
    )
    result = tool.execute(timeout=1)
    out = result.output or ""
    assert "STALE" in out


def test_render_helper_parity_with_wait_for_tasks():
    """The shared helper must produce lines byte-identical to the ones
    ``WaitForTasksTool`` would render for the same task.  Guards against
    drift when the renderer is changed in the future.
    """
    board = TaskBoard()
    task = TaskSpec(id="t1", name="alpha", prompt="x")
    board.add_task(task)
    board.claim("t1", "wallace")
    board.mark_running("t1")
    board.bump_progress(
        "t1", tool_name="python_execute", input_preview="print(42)", tokens=1,
    )

    wait_tool = WaitForTasksTool(board)
    # Task is already running; use timeout=0 so wait_for_tasks exits
    # immediately without blocking.
    wait_result = wait_tool.execute(task_names=["alpha"], timeout=1)
    wait_output = wait_result.output or ""
    # Extract just the per-task lines (strip any trailing WARNING/STALLED
    # footer).
    wait_lines = [
        line for line in wait_output.splitlines()
        if line.startswith("  ") or line.startswith("    ")
    ]

    now = time.monotonic()
    helper_lines = _render_task_activity_lines(
        board.get_task("t1"), now, include_summary=True,
    )
    # Heartbeat age in seconds is allowed to differ by 1s between the two
    # snapshots (clock progressed between rendering calls).  Normalize by
    # stripping the "(Xs ago," substring to something stable.
    import re
    def _norm(line: str) -> str:
        return re.sub(r"\(\d+s ago,", "(Ns ago,", line)
    assert [_norm(l) for l in helper_lines] == [_norm(l) for l in wait_lines], (
        f"renderer drift between helper and WaitForTasksTool:\n"
        f"  helper: {helper_lines}\n"
        f"  wait:   {wait_lines}"
    )
