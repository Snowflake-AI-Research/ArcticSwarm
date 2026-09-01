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

"""Unit tests for CompleteTaskTool atomicity and summary-fallback.

The tool MUST gate its peer-DM broadcast on the bool returned by
``TaskBoard.complete`` so that:

1. The first caller to win the race causes exactly one ``task_completed``
   broadcast.
2. A second call on a COMPLETED task with a *non-empty* summary falls
   back to ``append_summary`` and emits a ``task_summary_updated`` DM
   (not a duplicate ``task_completed``).  This recovers the polished
   Phase 3 summary that BBS mode's multi-phase workflow produces.
3. A second call with an *empty* summary, or any call on a FAILED task,
   remains a silent no-op (no append, no DM, no resurrection).
4. Task-board state is mutated BEFORE any DM is sent (Claude Code's
   ``enqueueAgentNotification`` pattern).
"""
from __future__ import annotations

from arcticswarm.swarm.mailbox import (
    DM_TYPE_TASK_COMPLETED,
    DM_TYPE_TASK_SUMMARY_UPDATED,
    Mailbox,
)
from arcticswarm.swarm.task import TaskBoard, TaskSpec, TaskStatus
from arcticswarm.swarm.tools import CompleteTaskTool


def _setup() -> tuple[TaskBoard, TaskSpec, Mailbox, CompleteTaskTool]:
    board = TaskBoard()
    task = TaskSpec(id="t1", name="alpha", prompt="x")
    board.add_task(task)
    board.claim("t1", "wallace")
    board.mark_running("t1")

    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("wallace")
    mailbox.register("bob")
    mailbox.attach_task_board(board)

    tool = CompleteTaskTool(
        board,
        mailbox=mailbox,
        sender="wallace",
        broadcast=True,
    )
    return board, task, mailbox, tool


def test_first_complete_transitions_and_broadcasts():
    board, task, mailbox, tool = _setup()
    result = tool.execute(task_name="alpha", summary="alpha done")
    assert not result.is_error
    stored = board.get_task(task.id)
    assert stored is not None
    assert stored.status is TaskStatus.COMPLETED
    # Peer DMs delivered to non-sender registered agents.
    for recipient in ("leader", "bob"):
        inbox = mailbox.check_new(recipient) or []
        types = [m.message_type for m in inbox]
        assert DM_TYPE_TASK_COMPLETED in types, (
            f"expected completion DM in {recipient} inbox, got {types}"
        )


def test_second_complete_with_summary_falls_back_to_append():
    """Second complete_task on a COMPLETED task with non-empty summary
    MUST append as an update entry and emit task_summary_updated (not
    task_completed).  Recovers the dropped-summary bug on BBS Phase 3.
    """
    board, task, mailbox, tool = _setup()
    tool.execute(task_name="alpha", summary="first draft")
    mailbox.check_new("leader")
    mailbox.check_new("bob")

    result2 = tool.execute(task_name="alpha", summary="polished final synthesis")
    assert not result2.is_error
    output = (result2.output or "").lower()
    assert "already completed" in output
    assert "update entry #2" in output
    assert "update_task_summary" in output

    # Task now has TWO summaries (original + appended).
    stored = board.get_task(task.id)
    assert stored is not None
    assert len(stored.summaries) == 2
    assert stored.summaries[-1].content == "polished final synthesis"


def test_second_complete_emits_summary_updated_dm():
    """The fallback must emit DM_TYPE_TASK_SUMMARY_UPDATED, not a duplicate
    DM_TYPE_TASK_COMPLETED.  Orchestrators disambiguate these via DM type.
    """
    board, task, mailbox, tool = _setup()
    tool.execute(task_name="alpha", summary="first")
    mailbox.check_new("leader")
    mailbox.check_new("bob")

    tool.execute(task_name="alpha", summary="appended")
    for recipient in ("leader", "bob"):
        inbox = mailbox.check_new(recipient) or []
        types = [m.message_type for m in inbox]
        assert DM_TYPE_TASK_SUMMARY_UPDATED in types, (
            f"expected summary-updated DM in {recipient} inbox, got {types}"
        )
        assert DM_TYPE_TASK_COMPLETED not in types, (
            f"fallback must NOT re-emit task_completed; got {types}"
        )


def test_second_complete_with_empty_summary_is_noop():
    """A second complete_task with no summary content keeps the original
    no-op behavior: no append, no DM.  Avoids spamming peers with empty
    updates when the subagent just re-confirms completion.
    """
    board, task, mailbox, tool = _setup()
    tool.execute(task_name="alpha", summary="first")
    mailbox.check_new("leader")
    mailbox.check_new("bob")

    result2 = tool.execute(task_name="alpha", summary="")
    assert not result2.is_error
    assert "already in terminal state" in (result2.output or "").lower()
    assert mailbox.check_new("leader") is None
    assert mailbox.check_new("bob") is None

    stored = board.get_task(task.id)
    assert stored is not None
    assert len(stored.summaries) == 1


def test_complete_after_failure_is_noop_even_with_summary():
    """FAILED tasks must NOT be resurrected by a summary-bearing
    complete_task call.  Returns the existing 'already terminal' message
    and sends no DM.
    """
    board, task, mailbox, tool = _setup()
    assert board.fail(task.id, error="boom") is True
    result = tool.execute(task_name="alpha", summary="too late")
    assert not result.is_error
    assert "already in terminal state" in (result.output or "").lower()
    stored = board.get_task(task.id)
    assert stored is not None
    assert stored.status is TaskStatus.FAILED
    # No DMs at all (neither completed nor summary-updated).
    assert mailbox.check_new("leader") is None
    assert mailbox.check_new("bob") is None


# ---------------------------------------------------------------------------
# on_complete_callback — race-fix for duo's worktree harvest
# ---------------------------------------------------------------------------


def test_on_complete_callback_fires_on_clean_transition():
    """The post-transition / pre-broadcast hook MUST fire exactly once on
    a successful first ``complete_task`` call, receiving the sender name.

    Duo mode wires the worktree-harvest DM through this hook so the
    harvest data lands in SwarmContext BEFORE the leader sees the
    broadcast ``task_completed`` notification and decides to submit.
    """
    board = TaskBoard()
    board.add_task(TaskSpec(id="t1", name="alpha", prompt="x"))
    board.claim("t1", "wallace")
    board.mark_running("t1")

    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("wallace")
    mailbox.attach_task_board(board)

    fired: list[str] = []
    tool = CompleteTaskTool(
        board, mailbox=mailbox, sender="wallace", broadcast=True,
        on_complete_callback=fired.append,
    )

    result = tool.execute(task_name="alpha", summary="done")
    assert not result.is_error
    assert fired == ["wallace"], (
        f"callback should fire exactly once with sender name; got {fired}"
    )


def test_on_complete_callback_fires_before_broadcast_dm():
    """The hook MUST run BEFORE ``_send_result_dm`` so any DMs the hook
    enqueues (e.g. harvest) arrive in the leader's mailbox ahead of the
    ``task_completed`` broadcast. This is the property that closes the
    race observed in the smoke run.
    """
    board = TaskBoard()
    board.add_task(TaskSpec(id="t1", name="alpha", prompt="x"))
    board.claim("t1", "wallace")
    board.mark_running("t1")

    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("wallace")
    mailbox.attach_task_board(board)

    def _harvest_hook(sender: str) -> None:
        # Simulate the harvest DM the orchestrator wires in real duo runs.
        mailbox.send(
            from_agent="system", to_agent="leader",
            content="<auditor_worktree_harvest/>",
            message_type="auditor_worktree_harvest",
        )

    tool = CompleteTaskTool(
        board, mailbox=mailbox, sender="wallace", broadcast=True,
        on_complete_callback=_harvest_hook,
    )
    tool.execute(task_name="alpha", summary="done")

    inbox = mailbox.check_new("leader") or []
    types = [m.message_type for m in inbox]
    # The harvest DM MUST arrive before task_completed; otherwise the
    # leader can act on completion before the harvest data exists.
    assert "auditor_worktree_harvest" in types
    assert DM_TYPE_TASK_COMPLETED in types
    harvest_idx = types.index("auditor_worktree_harvest")
    completed_idx = types.index(DM_TYPE_TASK_COMPLETED)
    assert harvest_idx < completed_idx, (
        f"harvest DM ({harvest_idx}) must precede task_completed DM "
        f"({completed_idx}) in the leader's inbox order; got {types}"
    )


def test_on_complete_callback_does_not_fire_on_terminal_task():
    """When the task is already in a terminal state (e.g. FAILED, or
    appended-to-COMPLETED), ``_task_board.complete`` returns False —
    the hook MUST NOT fire so it can't double-emit harvest DMs in
    re-entrant paths.
    """
    board = TaskBoard()
    board.add_task(TaskSpec(id="t1", name="alpha", prompt="x"))
    board.claim("t1", "wallace")
    board.mark_running("t1")
    assert board.fail("t1", error="boom") is True

    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("wallace")
    mailbox.attach_task_board(board)

    fired: list[str] = []
    tool = CompleteTaskTool(
        board, mailbox=mailbox, sender="wallace", broadcast=True,
        on_complete_callback=fired.append,
    )
    tool.execute(task_name="alpha", summary="too late")
    assert fired == [], (
        f"callback must not fire for an already-terminal task; got {fired}"
    )


def test_on_complete_callback_exception_does_not_break_completion():
    """A raising callback must not turn a successful completion into a
    tool error or skip the broadcast — harvest-side bugs must not corrupt
    the task lifecycle."""
    board = TaskBoard()
    board.add_task(TaskSpec(id="t1", name="alpha", prompt="x"))
    board.claim("t1", "wallace")
    board.mark_running("t1")

    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("wallace")
    mailbox.attach_task_board(board)

    def _bad_hook(sender: str) -> None:
        raise RuntimeError("simulated harvest failure")

    tool = CompleteTaskTool(
        board, mailbox=mailbox, sender="wallace", broadcast=True,
        on_complete_callback=_bad_hook,
    )
    result = tool.execute(task_name="alpha", summary="done")
    assert not result.is_error
    # Broadcast still fired — leader receives task_completed even though
    # the harvest hook bombed out.
    inbox = mailbox.check_new("leader") or []
    types = [m.message_type for m in inbox]
    assert DM_TYPE_TASK_COMPLETED in types
