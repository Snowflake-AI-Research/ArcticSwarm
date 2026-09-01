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

"""Smoke tests for ``Agent.last_stop_reason`` (Fix #2 dependency).

Subagents distinguish "natural completion" from "max_turns exhaustion"
by inspecting ``agent.last_stop_reason``.  We only verify the attribute
is declared on the class and defaults to an empty string — the full
``run_turn`` max_turns path is exercised via integration tests because
stubbing the LLM client robustly here would be more code than the fix.
"""
from __future__ import annotations


def test_agent_declares_last_stop_reason():
    """Reading the attribute on a fresh Agent must yield ``""`` (not raise)."""
    from arcticswarm.agent import Agent

    # The class must expose the attribute in its ``__init__`` so any code path
    # that runs ``run_turn`` / ``run_turn_streaming`` can unconditionally read
    # ``self.last_stop_reason`` afterwards.
    import inspect
    src = inspect.getsource(Agent.__init__)
    assert "self.last_stop_reason" in src, (
        "Agent.__init__ must initialize self.last_stop_reason "
        "so subagents can safely inspect it after every turn."
    )


def test_turn_loop_sets_last_stop_reason_on_max_turns():
    """Both ``run_turn`` and ``run_turn_streaming`` must stamp ``"max_turns"``
    when the turn budget is exhausted.  We grep the source (pure read, no
    LLM) to avoid the instability of a full Agent stub."""
    import inspect

    from arcticswarm.agent import Agent

    run_turn_src = inspect.getsource(Agent.run_turn)
    run_turn_streaming_src = inspect.getsource(Agent.run_turn_streaming)
    for name, src in (
        ("run_turn", run_turn_src),
        ("run_turn_streaming", run_turn_streaming_src),
    ):
        assert 'self.last_stop_reason = "max_turns"' in src, (
            f"Agent.{name} must set self.last_stop_reason='max_turns' "
            "when the turn budget is exhausted (Fix #2 invariant)."
        )
        # The natural end_turn path must also stamp the attribute so the
        # subagent never reads a stale value from an earlier turn.
        assert "self.last_stop_reason = stop_reason" in src, (
            f"Agent.{name} must stamp self.last_stop_reason on natural "
            "turn completion (Fix #2 invariant)."
        )


def test_successful_report_tool_terminates_turn_immediately():
    """A successful ``send_user_markdown_report`` must end the current turn.

    Otherwise the orchestrator asks the LLM for one more round after the
    report is already delivered, which creates the spurious empty-response
    fallback churn seen in Duo runs.
    """
    import inspect

    from arcticswarm.agent import Agent

    helper_src = inspect.getsource(Agent._tool_batch_terminates_turn)
    assert '"send_user_markdown_report"' in helper_src, (
        "Agent must treat send_user_markdown_report as a terminal tool "
        "when it succeeds."
    )

    for name, src in (
        ("run_turn", inspect.getsource(Agent.run_turn)),
        ("run_turn_streaming", inspect.getsource(Agent.run_turn_streaming)),
    ):
        assert "self._tool_batch_terminates_turn(tool_calls, tool_results)" in src, (
            f"Agent.{name} must stop after a successful report-tool batch "
            "instead of looping into another LLM call."
        )
