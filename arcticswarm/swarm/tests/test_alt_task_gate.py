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

"""Unit tests for the premature-commitment guard (alt-task gate).

The arcticswarm paper found that orchestrator runs which never open an
ALTERNATIVE / CONTRARIAN task commit to whichever candidate emerged first and
lose accuracy, with the gap widening on harder questions.  The guard
(``PrepareReportTool._check_alt_task_gate``) requires at least one such task on
every web/BBS run before ``send_user_markdown_report`` is unlocked, auto-spawning
a contrarian ``alternative-candidate-sweep`` task if the orchestrator never
opened one (then re-checking on the next ``prepare_report`` call).

Also covers the ``arcticswarm.swarm.task.task_is_alt`` detector.
"""
from __future__ import annotations

from arcticswarm.swarm.bbs import BBS
from arcticswarm.swarm.task import (
    AgentRegistry,
    TaskBoard,
    TaskSpec,
    task_is_alt,
)
from arcticswarm.swarm.tools import PrepareReportTool, SendReportTool


# ---------------------------------------------------------------------------
# Detector: task_is_alt
# ---------------------------------------------------------------------------


class TestTaskIsAlt:

    def test_name_tokens_match(self):
        for name in (
            "alt-angle",
            "alternative-candidates",
            "alternative-interpretation-search",
            "contrarian-search",
            "search-alternative-candidates",
            "ALT-Hypothesis",  # case-insensitive
        ):
            assert task_is_alt(TaskSpec(id="t", name=name, prompt="x")) is True, name

    def test_metadata_flag_matches(self):
        spec = TaskSpec(id="t", name="rival-sweep", prompt="x", metadata={"alt": True})
        assert task_is_alt(spec) is True

    def test_non_alt_names_rejected(self):
        # "salt"/"halt"/"default"/"maraltro" contain the substring "alt" but are
        # not standalone 'alt' tokens, and "alternate" lacks the 'alternativ' stem.
        for name in (
            "author-road-accident",
            "verify-logic",
            "salt-lake-search",
            "default-handler",
            "maraltro-designer-research",
            "ken-walibora-verification",
            "",
        ):
            assert task_is_alt(TaskSpec(id="t", name=name, prompt="x")) is False, name

    def test_none_is_false(self):
        assert task_is_alt(None) is False


# ---------------------------------------------------------------------------
# Fakes for the live swarm state the gate inspects
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, web: bool = True):
        self._web = web

    def has_web_search_capability(self) -> bool:
        return self._web


class _FakeCtx:
    """Duck-typed stand-in for ``SwarmContext`` used by the gate."""

    def __init__(self, board: TaskBoard, *, raise_on_spawn: bool = False):
        self.task_board = board
        self.config = _FakeConfig(web=True)
        self._tid = 0
        self._raise = raise_on_spawn
        self.spawned: list = []

    def next_task_id(self) -> str:
        self._tid += 1
        return f"at-{self._tid}"

    def spawn_or_assign(self, spec) -> str:
        if self._raise:
            raise RuntimeError("dispatch failed")
        self.spawned.append(spec)
        return f"worker-{spec.profile}"


def _make_tool(
    *,
    board: TaskBoard | None = None,
    ctx: _FakeCtx | None = None,
    enforce_alt_task: bool = True,
    has_web_search: bool = True,
    is_followup: bool = False,
) -> tuple[PrepareReportTool, _FakeCtx]:
    board = board if board is not None else TaskBoard()
    ctx = ctx if ctx is not None else _FakeCtx(board)
    tool = PrepareReportTool(
        task_board=board,
        agent_registry=AgentRegistry(),
        report_tool=SendReportTool(has_web_search=has_web_search),
        agent_tools={},
        bbs=BBS(),
        is_followup=is_followup,
        swarm_ctx=ctx,
        has_web_search=has_web_search,
        enforce_alt_task=enforce_alt_task,
        question_text="Who is the African author who died in a road accident?",
    )
    return tool, ctx


def _add(board: TaskBoard, name: str, *, alt_meta: bool = False) -> None:
    md = {"alt": True} if alt_meta else {}
    board.add_task(TaskSpec(id=name, name=name, prompt="x", profile="browsing", metadata=md))


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------


class TestAltTaskGate:

    def test_passes_when_alt_task_present_by_name(self):
        board = TaskBoard()
        _add(board, "author-search")
        _add(board, "alternative-candidates")  # alt by name token
        tool, ctx = _make_tool(board=board)
        assert tool._check_alt_task_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []

    def test_passes_when_alt_task_present_by_metadata(self):
        board = TaskBoard()
        _add(board, "author-search")
        _add(board, "rival-sweep", alt_meta=True)  # alt by metadata flag
        tool, ctx = _make_tool(board=board)
        assert tool._check_alt_task_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []

    def test_autospawns_contrarian_when_none(self):
        board = TaskBoard()
        _add(board, "author-search")
        _add(board, "verify-logic")
        tool, ctx = _make_tool(board=board)
        msg = tool._check_alt_task_gate(force=False, timed_out=False)
        assert msg is not None
        assert "alternative-candidate-sweep" in msg
        assert tool._alt_gate_spawned is True
        # Exactly one contrarian task spawned, tagged alt + browsing profile.
        assert len(ctx.spawned) == 1
        spawned = ctx.spawned[0]
        assert spawned.profile == "browsing"
        assert spawned.metadata.get("alt") is True
        assert task_is_alt(spawned) is True
        # It is now on the board, so a re-check passes without a second spawn.
        assert tool._check_alt_task_gate(force=False, timed_out=False) is None
        assert len(ctx.spawned) == 1

    def test_noop_when_disabled(self):
        board = TaskBoard()
        _add(board, "author-search")
        tool, ctx = _make_tool(board=board, enforce_alt_task=False)
        assert tool._check_alt_task_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []

    def test_noop_on_followup(self):
        board = TaskBoard()
        _add(board, "author-search")
        tool, ctx = _make_tool(board=board, is_followup=True)
        assert tool._check_alt_task_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []

    def test_noop_on_non_web_run(self):
        board = TaskBoard()
        _add(board, "revenue-analysis")
        tool, ctx = _make_tool(board=board, has_web_search=False)
        assert tool._check_alt_task_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []

    def test_force_and_timed_out_escape(self):
        board = TaskBoard()
        _add(board, "author-search")
        tool, ctx = _make_tool(board=board)
        assert tool._check_alt_task_gate(force=True, timed_out=True) is None
        assert ctx.spawned == []

    def test_timed_out_degrades_without_spawn(self):
        board = TaskBoard()
        _add(board, "author-search")
        tool, ctx = _make_tool(board=board)
        assert tool._check_alt_task_gate(force=False, timed_out=True) is None
        assert ctx.spawned == []
        assert "WARNING" in tool._alt_gate_degrade_note

    def test_single_shot_degrades_on_second_miss(self):
        # If a prior spawn somehow left no alt task on the board, the guard must
        # not keep spawning — it degrades after its single auto-spawn.
        board = TaskBoard()
        _add(board, "author-search")
        tool, ctx = _make_tool(board=board)
        tool._alt_gate_spawned = True  # pretend we already spawned once
        assert tool._check_alt_task_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []
        assert "WARNING" in tool._alt_gate_degrade_note

    def test_spawn_failure_degrades(self):
        board = TaskBoard()
        _add(board, "author-search")
        ctx = _FakeCtx(board, raise_on_spawn=True)
        tool, _ = _make_tool(board=board, ctx=ctx)
        msg = tool._check_alt_task_gate(force=False, timed_out=False)
        # Dispatch raised -> _spawn_contrarian_task returns None -> degrade.
        assert msg is None
        assert "WARNING" in tool._alt_gate_degrade_note
