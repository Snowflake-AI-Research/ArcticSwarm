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

"""Unit test for the role-aware orchestrator tool-call budget.

The orchestrator inherits ``max_tool_calls_per_turn`` from the shared config,
which silently truncates its batched ``create_task`` fan-out + ``wait_for_tasks``
when that cap is 1 (the dropped intent is never re-queued).  The fix gives the
orchestrator a SEPARATE per-agent override so the leader can run unlimited tool
calls per turn while browsing subagents keep the disciplined cap.

Following the repo convention (see test_agent_stop_reason.py), we test the
pure resolution helper + config defaults + the bridge + source invariants,
rather than stubbing a full Agent turn loop.
"""
from __future__ import annotations

import inspect


# --- pure resolution semantics ---------------------------------------------

def test_resolve_prefers_override_when_set():
    from arcticswarm.agent import _resolve_max_tool_calls

    # override None => inherit the shared config value (default behavior)
    assert _resolve_max_tool_calls(None, 1) == 1
    assert _resolve_max_tool_calls(None, 0) == 0
    # override set => wins regardless of config (0 = unlimited orchestrator)
    assert _resolve_max_tool_calls(0, 1) == 0
    assert _resolve_max_tool_calls(3, 1) == 3
    # override 0 must beat a config cap of 1 — the core of the fix
    assert _resolve_max_tool_calls(0, 1) == 0, (
        "orchestrator override=0 (unlimited) must override subagent cap=1"
    )


# --- config field defaults + bridge ----------------------------------------

def test_flat_config_defaults_to_inherit():
    from arcticswarm.config import ArcticswarmConfig

    cfg = ArcticswarmConfig()
    assert getattr(cfg, "orchestrator_max_tool_calls_per_turn", None) == -1, (
        "default -1 must mean 'inherit max_tool_calls_per_turn' so existing "
        "runs/models are unchanged"
    )


def test_run_config_bridges_orchestrator_override():
    from arcticswarm.run_config import LLMConfig

    llm = LLMConfig()
    assert llm.orchestrator_max_tool_calls_per_turn == -1
    # the bridge must copy the YAML-facing field onto the flat config
    bridge_src = inspect.getsource(
        __import__("arcticswarm.run_config", fromlist=["RunConfig"]).RunConfig.to_arcticswarm_config
    )
    assert "config.orchestrator_max_tool_calls_per_turn = self.llm.orchestrator_max_tool_calls_per_turn" in bridge_src, (
        "to_arcticswarm_config must bridge orchestrator_max_tool_calls_per_turn"
    )


# --- Agent + orchestrator wiring (source invariants) ------------------------

def test_agent_declares_override_attribute():
    from arcticswarm.agent import Agent

    src = inspect.getsource(Agent.__init__)
    assert "self.max_tool_calls_per_turn_override" in src, (
        "Agent.__init__ must declare max_tool_calls_per_turn_override so the "
        "orchestrator path can set it and the turn loops can read it"
    )
    # default must be None (inherit) so subagents are unaffected
    assert "max_tool_calls_per_turn_override: int | None = None" in src


def test_turn_loops_use_the_resolver():
    """Both turn loops must resolve the cap via the override-aware helper,
    not read self.config.max_tool_calls_per_turn directly."""
    from arcticswarm.agent import Agent

    for name in ("run_turn", "run_turn_streaming"):
        src = inspect.getsource(getattr(Agent, name))
        assert "_resolve_max_tool_calls(" in src, (
            f"Agent.{name} must resolve max_tc via _resolve_max_tool_calls "
            "so the orchestrator override is honored"
        )
        assert "self.max_tool_calls_per_turn_override" in src


def test_orchestrator_applies_override_gated_on_flag():
    """Orchestrator construction must apply the override ONLY when the flag is
    >= 0 (>= 0 because 0 = unlimited), and never mutate the shared config."""
    from arcticswarm.swarm import orchestrator as orch_mod

    src = inspect.getsource(orch_mod)
    assert "orchestrator_max_tool_calls_per_turn" in src
    assert "agent.max_tool_calls_per_turn_override = _orch_max_tc" in src, (
        "must set the per-agent override on the orchestrator agent"
    )
    assert "if _orch_max_tc >= 0:" in src, (
        "override must be gated on flag >= 0 so default -1 is a no-op"
    )
    # guard against regressing to a shared-config mutation
    assert "self.config.max_tool_calls_per_turn =" not in src, (
        "must NOT mutate the shared config (subagents read it for their cap)"
    )


# --- privileged always-execute tools (post_to_bbs bypasses the cap) ---------

def _tc(name):
    return {"id": name, "name": name, "input": {}}


def test_split_keeps_first_n_plus_privileged():
    from arcticswarm.agent import _split_capped_tool_calls

    priv = frozenset({"post_to_bbs"})
    # the user's example: web_search + post_to_bbs + list_tasks, cap=1
    batch = [_tc("web_search"), _tc("post_to_bbs"), _tc("list_tasks")]
    kept, dropped = _split_capped_tool_calls(batch, 1, priv)
    assert [c["name"] for c in kept] == ["web_search", "post_to_bbs"]
    assert [c["name"] for c in dropped] == ["list_tasks"]


def test_split_post_first_is_unchanged():
    from arcticswarm.agent import _split_capped_tool_calls

    priv = frozenset({"post_to_bbs"})
    # post first (within budget) -> kept; complete_task past cap -> dropped (re-issued)
    kept, dropped = _split_capped_tool_calls(
        [_tc("post_to_bbs"), _tc("complete_task")], 1, priv,
    )
    assert [c["name"] for c in kept] == ["post_to_bbs"]
    assert [c["name"] for c in dropped] == ["complete_task"]


def test_split_post_after_complete_still_executes():
    from arcticswarm.agent import _split_capped_tool_calls

    priv = frozenset({"post_to_bbs"})
    # the loss case the fix targets: complete_task first, post second.
    # post_to_bbs must STILL execute (privileged) so the finding always lands.
    kept, dropped = _split_capped_tool_calls(
        [_tc("complete_task"), _tc("post_to_bbs")], 1, priv,
    )
    assert [c["name"] for c in kept] == ["complete_task", "post_to_bbs"]
    assert dropped == []


def test_split_no_privileged_is_plain_truncation():
    from arcticswarm.agent import _split_capped_tool_calls

    kept, dropped = _split_capped_tool_calls(
        [_tc("web_search"), _tc("web_fetch"), _tc("reasoning")], 1, frozenset(),
    )
    assert [c["name"] for c in kept] == ["web_search"]
    assert [c["name"] for c in dropped] == ["web_fetch", "reasoning"]


def test_split_unlimited_keeps_all():
    from arcticswarm.agent import _split_capped_tool_calls

    batch = [_tc("a"), _tc("b"), _tc("c")]
    kept, dropped = _split_capped_tool_calls(batch, 0, frozenset({"post_to_bbs"}))
    assert kept == batch and dropped == []


def test_config_default_empty_and_qwen_enables_post_to_bbs():
    from arcticswarm.config import ArcticswarmConfig
    from arcticswarm.run_config import load_run_config

    # default: empty -> strict cap, no behavior change for other models/runs
    assert ArcticswarmConfig().always_execute_tools_per_turn == []
    # qwen config enables post_to_bbs bypass
    cfg = load_run_config(["conf/bench/browsecomp_qwen.yaml"]).to_arcticswarm_config()
    assert "post_to_bbs" in cfg.always_execute_tools_per_turn


def test_turn_loops_apply_privileged_split():
    from arcticswarm.agent import Agent

    for name in ("run_turn", "run_turn_streaming"):
        src = inspect.getsource(getattr(Agent, name))
        assert "_split_capped_tool_calls(" in src, (
            f"Agent.{name} must use _split_capped_tool_calls so privileged "
            "tools bypass the cap"
        )
        assert "always_execute_tools_per_turn" in src


# --- prompt sequences post -> complete across turns -------------------------

def test_summarize_prompt_sequences_post_then_complete():
    from arcticswarm.swarm.teammate import SubAgent
    from arcticswarm.swarm.task import TaskSpec, TaskStatus

    task = TaskSpec(id="task-1", name="find-x", prompt="find x", profile="browsing")
    task.status = TaskStatus.RUNNING
    # single_tool_call=True (cap is 1) must add the separation guidance
    p = SubAgent._build_summarize_prompt(
        task, ["a finding"], None, has_bbs=True, single_tool_call=True,
    )
    assert "SEPARATE TURN" in p
    assert "Never call `post_to_bbs` and `complete_task` in the same turn" in p
    # when uncapped, no separation note (batching is fine)
    p2 = SubAgent._build_summarize_prompt(
        task, ["a finding"], None, has_bbs=True, single_tool_call=False,
    )
    assert "SEPARATE TURN" not in p2


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
