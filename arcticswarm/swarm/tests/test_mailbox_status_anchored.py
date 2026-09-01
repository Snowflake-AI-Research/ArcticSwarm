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

"""Unit tests for status-anchored DM rendering (Fix #4d-iii).

When a Mailbox has a TaskBoard attached, DM headers for messages that
reference a ``task_id`` must be annotated with the *current* task status
from the board, not derived from the message body.  Prevents the DM-rt
race where a peer DM announces completion before ``complete_task``
actually fires.
"""
from __future__ import annotations

from arcticswarm.swarm.mailbox import (
    DM_LANE_PEER,
    DM_LANE_RESULT,
    DM_TYPE_PEER_MESSAGE,
    DM_TYPE_TASK_COMPLETED,
    Mailbox,
)
from arcticswarm.swarm.task import TaskBoard, TaskSpec


def _setup() -> tuple[Mailbox, TaskBoard, TaskSpec]:
    board = TaskBoard()
    task = TaskSpec(id="t1", name="poly-theory-analysis", prompt="x")
    board.add_task(task)
    board.claim("t1", "wallace")
    board.mark_running("t1")

    mb = Mailbox()
    mb.register("leader")
    mb.register("wallace")
    mb.attach_task_board(board)
    return mb, board, task


def test_peer_dm_with_task_id_gets_running_tag_when_task_running():
    mb, board, task = _setup()
    msg = mb.send(
        from_agent="wallace",
        to_agent="leader",
        content="I think the answer is alpha=4",
        lane=DM_LANE_PEER,
        message_type=DM_TYPE_PEER_MESSAGE,
        payload={"task_id": task.id, "task_name": task.name},
    )
    rendered = mb.render_for_llm([msg])
    assert "status=running" in rendered
    assert "poly-theory-analysis" in rendered
    # Must NOT mislabel as completed based on prose.
    assert "status=completed" not in rendered


def test_completion_dm_tag_reflects_board_after_transition():
    mb, board, task = _setup()
    board.complete(task.id, summary="alpha=4")
    msg = mb.send(
        from_agent="wallace",
        to_agent="leader",
        content="[Task completed: poly-theory-analysis] alpha=4",
        lane=DM_LANE_RESULT,
        message_type=DM_TYPE_TASK_COMPLETED,
        payload={"task_id": task.id, "task_name": task.name,
                 "status": "completed"},
    )
    rendered = mb.render_for_llm([msg])
    assert "status=completed" in rendered


def test_no_tag_when_no_task_board_attached():
    mb = Mailbox()
    mb.register("leader")
    mb.register("wallace")
    msg = mb.send(
        from_agent="wallace",
        to_agent="leader",
        content="something",
        lane=DM_LANE_PEER,
        payload={"task_id": "t1", "task_name": "x"},
    )
    rendered = mb.render_for_llm([msg])
    assert "status=" not in rendered


def test_no_tag_when_payload_lacks_task_id():
    mb, board, _ = _setup()
    msg = mb.send(
        from_agent="wallace",
        to_agent="leader",
        content="hello",
        lane=DM_LANE_PEER,
    )
    rendered = mb.render_for_llm([msg])
    assert "status=" not in rendered
