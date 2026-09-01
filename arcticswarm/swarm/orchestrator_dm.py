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

"""Realtime ("DM-realtime") orchestration loop for the swarm orchestrator.

The event-driven multi-turn orchestrator loop (Claude Code pattern) extracted
from :meth:`arcticswarm.swarm.orchestrator.SwarmOrchestrator.run_swarm_turn`
into a mixin.  :class:`DmMixin` is composed into ``SwarmOrchestrator`` via the
MRO; the loop reads only ``self.config`` plus the keyword params passed in by
``run_swarm_turn``.  The loop body is moved verbatim (dedented one level).

This module intentionally does NOT import ``orchestrator`` — the loop uses no
orchestrator-defined symbols — so it adds no circular-import edge.
"""

from __future__ import annotations

from arcticswarm.swarm.mailbox import DM_TYPE_IDLE_NOTIFICATION
from arcticswarm.swarm.task import TaskStatus


class DmMixin:
    """Realtime ("DM-realtime") orchestration loop."""

    def _run_realtime_loop(
        self,
        *,
        agent,
        mailbox,
        report_tool,
        prepare_report_tool,
        task_board,
        agent_registry,
        enriched_question,
        dm_realtime_direct_report,
        orch_collector,
    ) -> None:
        # Event-driven multi-turn loop (Claude Code pattern).
        # Each iteration runs one LLM turn, then waits for the
        # next DM from subagents before starting the next turn.
        # The orchestrator is never blocked inside a tool call.
        prompt = enriched_question
        _realtime_timeout = getattr(self.config, "orchestrator_realtime_timeout", 300)
        while True:
            orch_collector.start()
            agent.run_turn_streaming(
                prompt, on_event=orch_collector.on_event,
            )

            if report_tool.captured_report:
                break

            # Inner loop: wait for actionable DMs before running
            # the next LLM turn.  ``continue`` here goes back to
            # wait_for_message, NOT to run_turn_streaming.
            prompt = None
            while prompt is None:
                assert mailbox is not None
                got_dm = mailbox.wait_for_message(
                    "leader", timeout=_realtime_timeout,
                )
                if not got_dm:
                    if prepare_report_tool is not None:
                        prepare_report_tool._deadline_exceeded = True
                    peer_summary = mailbox.drain_peer_summaries()
                    timeout_parts: list[str] = []
                    if peer_summary:
                        timeout_parts.append(peer_summary)
                    _terminal = (TaskStatus.COMPLETED, TaskStatus.FAILED)
                    _completed = [
                        t for t in task_board.all_tasks()
                        if t.status in _terminal
                    ]
                    _pending = [
                        t for t in task_board.all_tasks()
                        if t.status not in _terminal
                    ]
                    _status_lines: list[str] = []
                    if _completed:
                        _status_lines.append(
                            "Completed: "
                            + ", ".join(
                                f"{t.name} ({t.claimed_by or '?'})"
                                for t in _completed
                            )
                        )
                    if _pending:
                        _status_lines.append(
                            "Still running: "
                            + ", ".join(
                                f"{t.name} ({t.claimed_by or '?'})"
                                for t in _pending
                            )
                        )
                    _task_status = "\n".join(_status_lines)
                    timeout_parts.append(
                        (
                            '<swarm_notification type="timeout">'
                            "Timeout waiting for teammates.\n"
                        )
                        + _task_status + "\n"
                        + (
                            "Call send_user_markdown_report with data "
                            "collected so far."
                            if dm_realtime_direct_report else
                            "Call prepare_report and "
                            "send_user_markdown_report with data "
                            "collected so far."
                        )
                        + "</swarm_notification>"
                    )
                    prompt = "\n\n".join(timeout_parts)
                    orch_collector.start()
                    agent.run_turn_streaming(
                        prompt, on_event=orch_collector.on_event,
                    )
                    prompt = None  # sentinel: outer loop should break
                    break

                dm_msgs = mailbox.check_new("leader") or []
                idle_msgs = [
                    m for m in dm_msgs
                    if m.message_type == DM_TYPE_IDLE_NOTIFICATION
                ]
                real_msgs = [
                    m for m in dm_msgs
                    if m.message_type != DM_TYPE_IDLE_NOTIFICATION
                ]

                actionable_idle_msgs = [
                    m for m in idle_msgs
                    if (
                        m.payload.get("completed_task_id")
                        or m.payload.get("failure_reason")
                        or m.payload.get("peer_summary")
                    )
                ]

                if not real_msgs and actionable_idle_msgs:
                    peer_summary = mailbox.drain_peer_summaries()
                    parts: list[str] = []
                    if peer_summary:
                        parts.append(peer_summary)
                    parts.append(mailbox.render_for_llm(actionable_idle_msgs))
                    prompt = "\n\n".join(parts)
                elif not real_msgs and idle_msgs:
                    if not agent_registry.all_idle():
                        continue
                    peer_summary = mailbox.drain_peer_summaries()
                    parts: list[str] = []
                    if peer_summary:
                        parts.append(peer_summary)
                    parts.append(
                        (
                            '<swarm_notification type="all_idle">'
                            "All teammates are now idle. "
                        )
                        + (
                            "Review list_tasks and decide whether to "
                            "submit or wait."
                            if dm_realtime_direct_report else
                            "Call prepare_report to check readiness."
                        )
                        + "</swarm_notification>"
                    )
                    prompt = "\n\n".join(parts)
                elif real_msgs:
                    peer_summary = mailbox.drain_peer_summaries()
                    parts = []
                    if peer_summary:
                        parts.append(peer_summary)
                    parts.append(mailbox.render_for_llm(real_msgs))
                    prompt = "\n\n".join(parts)
                else:
                    continue

            if prompt is None:
                break

