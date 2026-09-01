"""Unit tests for SubAgent._build_summarize_prompt status branching.

The BBS Final Summary phase can run after the wrapper-autocomplete has
already marked the task COMPLETED.  In that case the prompt must
instruct the subagent to use ``update_task_summary`` (which appends
to ``task.summaries`` and emits ``task_summary_updated``) rather than
``complete_task`` (which would hit the tool-level fallback).  For
non-terminal statuses the prompt must keep the original
``complete_task`` instruction.
"""
from __future__ import annotations

from arcticswarm.swarm.task import TaskSpec, TaskStatus
from arcticswarm.swarm.teammate import SubAgent


def _make_task(status: TaskStatus) -> TaskSpec:
    task = TaskSpec(id="t1", name="alpha", prompt="x")
    task.status = status
    return task


def test_prompt_for_completed_task_uses_update_task_summary():
    task = _make_task(TaskStatus.COMPLETED)
    prompt = SubAgent._build_summarize_prompt(
        task, findings=[], reflection=None, has_bbs=True,
    )
    # Step 3 must direct the subagent to update_task_summary.
    assert "update_task_summary" in prompt
    # The stale complete_task instruction must NOT appear.
    # (Allow "task" / "complete" in other words; just ensure the exact
    # "Call `complete_task`" instruction block is absent.)
    assert "Call `complete_task`" not in prompt


def test_prompt_for_running_task_uses_complete_task():
    task = _make_task(TaskStatus.RUNNING)
    prompt = SubAgent._build_summarize_prompt(
        task, findings=[], reflection=None, has_bbs=True,
    )
    assert "Call `complete_task`" in prompt
    assert "update_task_summary" not in prompt


def test_prompt_for_claimed_task_uses_complete_task():
    task = _make_task(TaskStatus.CLAIMED)
    prompt = SubAgent._build_summarize_prompt(
        task, findings=[], reflection=None, has_bbs=False,
    )
    assert "Call `complete_task`" in prompt
    assert "update_task_summary" not in prompt
