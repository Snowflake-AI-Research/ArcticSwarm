"""Unit tests for the per-task progress heartbeat (Fix #4).

The orchestrator relies on ``TaskBoard.bump_progress`` + ``render_status``
to see *live* subagent activity instead of a static "running" label.  A
``STALE`` tag must appear once the heartbeat is older than the
board-wide threshold so ``list_tasks`` / ``wait_for_tasks`` can surface
stuck subagents.
"""
from __future__ import annotations

import time

from arcticswarm.swarm.task import (
    STALE_HEARTBEAT_THRESHOLD_SECONDS,
    TaskBoard,
    TaskSpec,
    TaskStatus,
)


def _running_task() -> tuple[TaskBoard, TaskSpec]:
    board = TaskBoard()
    task = TaskSpec(id="t1", name="poly-theory-analysis", prompt="x")
    board.add_task(task)
    board.claim("t1", "wallace")
    board.mark_running("t1")
    return board, task


def test_bump_progress_populates_fields():
    board, task = _running_task()
    board.bump_progress(
        task.id,
        tool_name="web_fetch",
        input_preview="https://rpg.stackexchange.com/questions/12345",
        tokens=340_000,
    )
    stored = board.get_task(task.id)
    assert stored is not None
    assert stored.tool_use_count == 1
    assert stored.last_activity_tool == "web_fetch"
    assert "rpg.stackexchange.com" in stored.last_activity_input
    assert stored.last_heartbeat > 0.0
    assert stored.token_count == 340_000


def test_bump_progress_increments_count():
    board, task = _running_task()
    for i in range(3):
        board.bump_progress(
            task.id, tool_name="python_execute",
            input_preview=f"step {i}",
        )
    assert board.get_task(task.id).tool_use_count == 3


def test_bump_progress_ignored_on_terminal_task():
    board, task = _running_task()
    board.complete(task.id, summary="done")
    board.bump_progress(task.id, tool_name="web_fetch", input_preview="late")
    stored = board.get_task(task.id)
    assert stored is not None
    # Heartbeat and activity fields remain unset (no leak after completion).
    assert stored.tool_use_count == 0
    assert stored.last_activity_tool == ""


def test_render_status_shows_activity_line():
    board, task = _running_task()
    board.bump_progress(
        task.id, tool_name="web_fetch",
        input_preview="https://example.com/page",
    )
    rendered = board.render_status()
    lines = rendered.splitlines()
    assert any("poly-theory-analysis: running" in ln for ln in lines)
    activity_lines = [ln for ln in lines if "activity:" in ln]
    assert activity_lines, f"expected activity line in:\n{rendered}"
    assert "web_fetch" in activity_lines[0]
    # Fresh heartbeat: must not be marked STALE.
    assert "STALE" not in activity_lines[0]


def test_render_status_marks_stale_after_threshold():
    board, task = _running_task()
    board.bump_progress(
        task.id, tool_name="web_fetch", input_preview="https://example.com",
    )
    # Simulate an old heartbeat by pushing ``now`` forward.
    future = time.monotonic() + STALE_HEARTBEAT_THRESHOLD_SECONDS + 10
    rendered = board.render_status(now=future)
    activity_lines = [ln for ln in rendered.splitlines() if "activity:" in ln]
    assert activity_lines, f"expected activity line in:\n{rendered}"
    assert "STALE" in activity_lines[0]


def test_any_running_stale_detects_all_idle_tasks():
    board, task = _running_task()
    board.bump_progress(task.id, tool_name="web_fetch")
    future = time.monotonic() + STALE_HEARTBEAT_THRESHOLD_SECONDS + 5
    assert board.any_running_stale([task.id], now=future) is True


def test_any_running_stale_false_when_any_task_fresh():
    board = TaskBoard()
    t1 = TaskSpec(id="t1", name="stale", prompt="x")
    t2 = TaskSpec(id="t2", name="fresh", prompt="y")
    board.add_task(t1)
    board.add_task(t2)
    board.claim("t1", "a"); board.mark_running("t1")
    board.claim("t2", "b"); board.mark_running("t2")
    # t1 is ancient; t2 was just seeded by mark_running.
    t1.last_heartbeat = 1.0
    assert board.any_running_stale(["t1", "t2"]) is False
