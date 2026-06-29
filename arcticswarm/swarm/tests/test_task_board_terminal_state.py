"""Unit tests for idempotent TaskBoard terminal-state transitions.

Fix #2 / Fix #4 rely on the invariant that:

- ``TaskBoard.complete`` returns ``True`` only on the actual
  pending/claimed/running -> COMPLETED edge and is a no-op (returns
  ``False``) on any subsequent call or on an already-failed task.
- ``TaskBoard.fail`` has the same idempotent semantics.

Without this, ``CompleteTaskTool`` can't safely gate its peer DM
notification on "did I actually cause this transition?" and the
orchestrator can receive duplicate/racing completion DMs (the DM-rt
desync bug).
"""
from __future__ import annotations

from arcticswarm.swarm.task import TaskBoard, TaskSpec, TaskStatus


def _make_board(task_id: str = "t1", name: str = "sample") -> tuple[TaskBoard, TaskSpec]:
    board = TaskBoard()
    task = TaskSpec(id=task_id, name=name, prompt="dummy")
    board.add_task(task)
    board.claim(task_id, "alice")
    board.mark_running(task_id)
    return board, task


def test_complete_returns_true_on_first_transition():
    board, task = _make_board()
    assert board.complete(task.id, summary="hello") is True
    stored = board.get_task(task.id)
    assert stored is not None
    assert stored.status is TaskStatus.COMPLETED
    assert stored.summary == "hello"


def test_complete_is_idempotent():
    board, task = _make_board()
    assert board.complete(task.id, summary="first") is True
    assert board.complete(task.id, summary="second") is False
    # Second summary must not be appended on the no-op path.
    stored = board.get_task(task.id)
    assert stored is not None
    assert len(stored.summaries) == 1


def test_complete_on_already_failed_is_noop():
    board, task = _make_board()
    assert board.fail(task.id, error="boom") is True
    assert board.complete(task.id, summary="too late") is False
    stored = board.get_task(task.id)
    assert stored is not None
    assert stored.status is TaskStatus.FAILED


def test_fail_returns_true_on_first_transition():
    board, task = _make_board()
    assert board.fail(task.id, error="boom") is True
    stored = board.get_task(task.id)
    assert stored is not None
    assert stored.status is TaskStatus.FAILED
    assert stored.error == "boom"


def test_fail_is_idempotent():
    board, task = _make_board()
    assert board.fail(task.id, error="first") is True
    assert board.fail(task.id, error="second") is False
    stored = board.get_task(task.id)
    assert stored is not None
    assert stored.error == "first"


def test_fail_on_already_completed_is_noop():
    board, task = _make_board()
    assert board.complete(task.id, summary="ok") is True
    assert board.fail(task.id, error="too late") is False
    stored = board.get_task(task.id)
    assert stored is not None
    assert stored.status is TaskStatus.COMPLETED
