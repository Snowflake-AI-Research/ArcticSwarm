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

"""Direct Messaging (DM) mailbox for swarm subagents.

Thread-safe in-memory mailbox with per-agent queues and wake-up events.
Agents register at spawn time, then send/receive messages through the
shared ``Mailbox`` instance.

Design notes:
- ``send()`` is non-blocking for the sender — it enqueues and signals.
- ``wait_for_message()`` blocks the recipient until a message arrives or
  timeout, using a ``threading.Event`` per agent.
- ``check_new()`` returns unread messages and marks them read (used by
  auto-injection and the ``ReadDMTool``).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any


DM_LANE_CONTROL = "control"
DM_LANE_RESULT = "result"
DM_LANE_PEER = "peer"
DM_ALL_LANES = frozenset({DM_LANE_CONTROL, DM_LANE_RESULT, DM_LANE_PEER})

DM_TYPE_IDLE_NOTIFICATION = "idle_notification"
DM_TYPE_TASK_COMPLETED = "task_completed"
DM_TYPE_TASK_SUMMARY_UPDATED = "task_summary_updated"
DM_TYPE_TASK_FAILED = "task_failed"
DM_TYPE_PEER_MESSAGE = "peer_message"
# Legacy worktree-merge harvest notice type. No code currently produces
# this message type (the worktree-merge harvest subsystem was removed);
# the constant and its render branch are retained as defensive rendering
# and for the report-drain filter that classifies it as non-substantive.
DM_TYPE_SUBAGENT_COMPLETE = "subagent_complete"


@dataclass(frozen=True)
class DirectMessage:
    """A single DM between two agents.  Immutable once created."""

    id: str
    from_agent: str
    to_agent: str
    timestamp: float
    content: str
    lane: str = DM_LANE_PEER
    message_type: str = DM_TYPE_PEER_MESSAGE
    payload: dict[str, Any] = field(default_factory=dict)
    read: bool = False


class Mailbox:
    """Thread-safe per-agent message queues with wake-up signalling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inboxes: dict[str, list[DirectMessage]] = {}
        self._events: dict[str, threading.Event] = {}
        self._peer_summaries: list[str] = []
        self._latest_peer_summary_by_sender: dict[str, str] = {}
        # Optional TaskBoard reference used by :meth:`render_for_llm` so that
        # every DM referencing a ``task_id`` can be annotated with the
        # *current* task status at render time instead of trusting the
        # subagent's prose.  Anchored on the board, not the message body,
        # to prevent the DM-rt race where a subagent's peer DM claims
        # completion before ``complete_task`` is called.
        self._task_board_ref: Any = None

    def attach_task_board(self, task_board: Any) -> None:
        """Register a task board so DM headers include live task status."""
        self._task_board_ref = task_board

    def register(self, agent_name: str) -> None:
        """Create an inbox and wake-up event for *agent_name*."""
        with self._lock:
            if agent_name not in self._inboxes:
                self._inboxes[agent_name] = []
                self._events[agent_name] = threading.Event()

    @property
    def registered_names(self) -> list[str]:
        """Return sorted list of registered agent names."""
        with self._lock:
            return sorted(self._inboxes.keys())

    def send(
        self,
        *,
        from_agent: str,
        to_agent: str,
        content: str,
        lane: str = DM_LANE_PEER,
        message_type: str = DM_TYPE_PEER_MESSAGE,
        payload: dict[str, Any] | None = None,
    ) -> DirectMessage:
        """Enqueue a message and signal the recipient's wake-up event.

        Raises ``ValueError`` if *to_agent* is not registered.
        """
        if lane not in DM_ALL_LANES:
            raise ValueError(
                f"Unknown DM lane '{lane}'. Use one of: {sorted(DM_ALL_LANES)}"
            )
        msg = DirectMessage(
            id=uuid.uuid4().hex[:12],
            from_agent=from_agent,
            to_agent=to_agent,
            timestamp=time.monotonic(),
            content=content,
            lane=lane,
            message_type=message_type,
            payload=dict(payload or {}),
        )
        with self._lock:
            inbox = self._inboxes.get(to_agent)
            if inbox is None:
                raise ValueError(
                    f"Agent '{to_agent}' is not registered on the mailbox."
                )
            inbox.append(msg)
            event = self._events.get(to_agent)
        if event is not None:
            event.set()
        return msg

    def has_received_from(
        self,
        agent_name: str,
        sender: str,
        *,
        lane: str | None = None,
    ) -> bool:
        """Return True if *agent_name*'s inbox contains any DM from *sender*.

        Read state is ignored — what matters is whether the message has
        ever landed, not whether the recipient has surfaced it in a turn
        yet. Optional ``lane`` filter narrows the check to a specific
        DM lane (e.g. ``DM_LANE_PEER`` to exclude automated
        ``task_completed`` / ``task_summary_updated`` lifecycle DMs
        which are emitted on the control / result lanes).

        Introduced for the reviewer-mode pre-submit stall in
        ``SendReportTool``: before allowing ``send_user_markdown_report``
        to finalize, we want to confirm the auditor has actually sent
        at least one peer-lane review DM. The check is read-state
        agnostic so that a recipient that has already consumed the DM
        in an earlier turn still trips the "yes, I've seen the
        reviewer" branch.
        """
        with self._lock:
            inbox = self._inboxes.get(agent_name, [])
            for m in inbox:
                if m.from_agent != sender:
                    continue
                if lane is not None and m.lane != lane:
                    continue
                return True
            return False

    def check_new(self, agent_name: str) -> list[DirectMessage] | None:
        """Return unread messages for *agent_name*, marking them read.

        Returns ``None`` if there are no unread messages.
        """
        with self._lock:
            inbox = self._inboxes.get(agent_name, [])
            unread = [m for m in inbox if not m.read]
            if not unread:
                return None
            read_msgs: list[DirectMessage] = []
            for i, m in enumerate(inbox):
                if not m.read:
                    marked = replace(m, read=True)
                    inbox[i] = marked
                    read_msgs.append(marked)
            # Clear the wake-up event now that messages have been consumed
            event = self._events.get(agent_name)
            if event is not None:
                event.clear()
            return read_msgs

    def wait_for_message(
        self,
        agent_name: str,
        timeout: float,
    ) -> bool:
        """Block until a message arrives for *agent_name* or *timeout* expires.

        Returns ``True`` if a message is available, ``False`` on timeout.
        Used by idle subagents to sleep efficiently instead of polling.
        """
        event = self._events.get(agent_name)
        if event is None:
            return False
        return event.wait(timeout=timeout)

    def signal(self, agent_name: str) -> None:
        """Wake the agent's wait_for_message without sending a DM."""
        event = self._events.get(agent_name)
        if event is not None:
            event.set()

    def log_peer_summary(self, from_agent: str, to_agent: str, summary: str) -> None:
        """Append a peer DM summary (does NOT trigger wake-up events)."""
        payload = json.dumps(
            {
                "lane": DM_LANE_PEER,
                "message_type": "peer_summary",
                "from": from_agent,
                "to": to_agent,
                "summary": summary,
            },
            sort_keys=True,
        )
        with self._lock:
            self._peer_summaries.append(payload)
            self._latest_peer_summary_by_sender[from_agent] = payload

    def consume_last_peer_summary(self, agent_name: str) -> dict[str, Any] | None:
        """Return and clear the last peer-summary payload sent by *agent_name*."""
        with self._lock:
            raw = self._latest_peer_summary_by_sender.pop(agent_name, None)
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"summary": raw}

    def drain_peer_summaries(self) -> str | None:
        """Return accumulated peer summaries and clear the buffer.

        Returns ``None`` if the buffer is empty.
        """
        with self._lock:
            if not self._peer_summaries:
                return None
            text = (
                '<swarm_notification type="peer_activity" lane="peer">\n'
                "Peer activity summary — your teammates exchanged "
                "the following messages among themselves:\n"
                + "\n".join(self._peer_summaries)
                + "\n</swarm_notification>"
            )
            self._peer_summaries.clear()
            return text

    def render_for_llm(self, messages: list[DirectMessage]) -> str:
        """Format DMs as lane-aware XML-wrapped text for LLM injection."""
        if not messages:
            return "No new direct messages."
        lane_titles = {
            DM_LANE_CONTROL: "## DM Control Notifications",
            DM_LANE_RESULT: "## DM Result Notifications",
            DM_LANE_PEER: "## DM Peer Messages",
        }
        lane_order = {
            DM_LANE_CONTROL: 0,
            DM_LANE_RESULT: 1,
            DM_LANE_PEER: 2,
        }
        parts: list[str] = []
        grouped: dict[str, list[DirectMessage]] = {}
        for m in messages:
            grouped.setdefault(m.lane, []).append(m)
        for lane in sorted(grouped.keys(), key=lambda value: lane_order.get(value, 99)):
            if parts:
                parts.append("")
            parts.append(lane_titles.get(lane, f"## DM {lane.title()} Messages"))
            for m in grouped[lane]:
                # Status-anchored header: if the DM references a task, look
                # it up on the board and include the *current* status so the
                # orchestrator cannot be misled by a subagent's prose (e.g.
                # "I completed X" before ``complete_task`` actually fires).
                status_tag = self._resolve_task_status_tag(m)
                if m.message_type == DM_TYPE_TASK_COMPLETED:
                    parts.append(
                        f"Task completion notification from {m.from_agent}"
                        f"{status_tag}:"
                    )
                elif m.message_type == DM_TYPE_TASK_SUMMARY_UPDATED:
                    parts.append(
                        f"Task summary update from {m.from_agent}"
                        f"{status_tag}:"
                    )
                elif m.message_type == DM_TYPE_TASK_FAILED:
                    parts.append(
                        f"Task failure notification from {m.from_agent}"
                        f"{status_tag}:"
                    )
                elif m.message_type == DM_TYPE_IDLE_NOTIFICATION:
                    parts.append(f"Idle notification from {m.from_agent}:")
                elif m.message_type == DM_TYPE_SUBAGENT_COMPLETE:
                    parts.append(
                        f"Subagent completion notice (worktree harvest)"
                        f"{status_tag}:"
                    )
                elif m.from_agent == "leader":
                    parts.append("The orchestrator (leader) has sent you a message:")
                else:
                    parts.append(
                        f"Subagent {m.from_agent} has sent you a message"
                        f"{status_tag}:"
                    )
                parts.append(
                    f'<swarm_dm from="{m.from_agent}" lane="{m.lane}" '
                    f'type="{m.message_type}">'
                )
                if m.payload:
                    parts.append("<swarm_dm_payload>")
                    parts.append(json.dumps(m.payload, sort_keys=True))
                    parts.append("</swarm_dm_payload>")
                parts.append(m.content)
                parts.append("</swarm_dm>")
        return "\n".join(parts).rstrip()

    def _resolve_task_status_tag(self, m: DirectMessage) -> str:
        """Return "" or " [task=<name> status=<current>]" for DM headers.

        Source of truth is the *current* TaskBoard status, not the DM body.
        Prevents the DM-rt race where a peer DM announces completion before
        ``complete_task`` actually mutates the board.
        """
        if self._task_board_ref is None:
            return ""
        payload = m.payload or {}
        task_id = payload.get("task_id") or ""
        task_name = payload.get("task_name") or ""
        if not task_id and not task_name:
            return ""
        try:
            task = None
            if task_id and hasattr(self._task_board_ref, "get_task"):
                task = self._task_board_ref.get_task(task_id)
            if task is None and task_name and hasattr(
                self._task_board_ref, "find_by_name",
            ):
                task = self._task_board_ref.find_by_name(task_name)
            if task is None:
                return ""
            status_val = getattr(task.status, "value", str(task.status))
            display = task_name or task_id
            return f" [task={display} status={status_val}]"
        except Exception:
            return ""

    def export(self) -> dict[str, Any]:
        """Dump all inboxes for human debugging.  NOT used by LLM or scoring."""
        with self._lock:
            return {
                agent: [
                    {
                        "id": m.id,
                        "from": m.from_agent,
                        "to": m.to_agent,
                        "content": m.content,
                        "lane": m.lane,
                        "message_type": m.message_type,
                        "payload": m.payload,
                        "read": m.read,
                    }
                    for m in msgs
                ]
                for agent, msgs in self._inboxes.items()
            }
