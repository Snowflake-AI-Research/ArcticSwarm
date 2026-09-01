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

"""Parse arcticswarm result trajectories into unified timelines and BBS snapshots."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class QuestionSummary:
    conv_id: str
    dataset: str
    question: str
    duration_seconds: float
    total_tokens: int
    judge_correct: bool | None
    score: float
    has_error: bool
    error: str | None
    had_timeout: bool
    swarm_teammates_spawned: int
    swarm_bbs_message_count: int
    swarm_subagent_tool_counts: dict[str, dict[str, int]]
    response_text: str
    reference_answer: str
    judge_comment: str
    judge_raw_output: str


@dataclass
class TimelineEvent:
    step: int
    timestamp: str  # ISO 8601
    actor: str  # "orchestrator" or subagent name
    event_type: str  # tool_call, tool_result, bbs_post, bbs_read, text, spawn, task_create
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_input_summary: str | None = None
    tool_result_text: str | None = None
    tool_result_summary: str | None = None
    tool_duration: float | None = None
    llm_output_tokens: int | None = None
    llm_duration: float | None = None
    bbs_channel: str | None = None
    bbs_content: str | None = None
    text: str | None = None
    is_error: bool = False


@dataclass
class BBSPost:
    channel: str
    author: str
    timestamp: str
    content: str
    estimated_tokens: int


@dataclass
class BBSSnapshot:
    step: int
    posts: list[BBSPost]
    total_estimated_tokens: int
    posts_by_channel: dict[str, list[BBSPost]]


@dataclass
class TaskInfo:
    id: str
    name: str
    prompt: str
    status: str
    profile: str
    claimed_by: str | None
    summary: str | None


@dataclass
class AgentInfo:
    name: str
    profile: str | None
    tool_calls_by_name: dict[str, int]
    tasks_completed: int
    first_timestamp: str | None
    is_idle_reviewer: bool = False


@dataclass
class TimelineData:
    question: str
    response_text: str
    total_duration_seconds: float
    agents: list[AgentInfo]
    tasks: list[TaskInfo]
    events: list[TimelineEvent]
    bbs_snapshots: dict[int, BBSSnapshot]  # step -> snapshot
    bbs_post_steps: list[int]


# ---------------------------------------------------------------------------
# Report loader
# ---------------------------------------------------------------------------

def load_report(run_dir: Path) -> list[QuestionSummary]:
    """Load report.json and return question summaries."""
    report_path = run_dir / "report.json"
    with open(report_path) as f:
        report = json.load(f)

    summaries = []
    for case in report.get("per_case", []):
        error = case.get("error")
        has_error = bool(error)
        had_timeout = (
            has_error and "timeout" in str(error).lower()
        ) or (
            "timeout" in str(case.get("phase_timings", {})).lower()
        )

        # Extract reference answer: prefer explicit field, fall back to regex on judge_raw_output
        judge_raw = case.get("judge_raw_output", "")
        reference_answer = case.get("reference_answer", "")
        if not reference_answer:
            ref_match = re.search(r'correct answer (?:is |of |")(?:"?)([^""\n]+?)(?:"|\.|\n|$)', judge_raw)
            reference_answer = ref_match.group(1).strip() if ref_match else ""

        summaries.append(QuestionSummary(
            conv_id=case["conv_id"],
            dataset=case.get("dataset", ""),
            question=case.get("question", ""),
            duration_seconds=case.get("duration_seconds", 0),
            total_tokens=case.get("total_token_e2e", case.get("total_tokens", 0)),
            judge_correct=case.get("judge_correct"),
            score=case.get("score", 0),
            has_error=has_error,
            error=error,
            had_timeout=had_timeout,
            swarm_teammates_spawned=case.get("swarm_teammates_spawned", 0),
            swarm_bbs_message_count=case.get("swarm_bbs_message_count", 0),
            swarm_subagent_tool_counts=case.get("swarm_subagent_tool_counts", {}),
            response_text=case.get("response_text", ""),
            reference_answer=reference_answer,
            judge_comment=case.get("judge_comment", ""),
            judge_raw_output=judge_raw,
        ))

    return summaries


# ---------------------------------------------------------------------------
# Timeline builder
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Approximate token count (~4 chars/token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _truncate(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _extract_events_from_messages(
    messages: list[dict[str, Any]],
    actor: str,
) -> list[TimelineEvent]:
    """Extract timeline events from a conversation message list."""
    events: list[TimelineEvent] = []
    # Map tool_use_id -> tool_name for resolving tool_results
    tool_id_map: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "")
        ts = msg.get("_timestamp", "")
        llm_dur = msg.get("_llm_duration_seconds")
        llm_tokens = msg.get("_llm_output_tokens")

        content = msg.get("content", [])
        if isinstance(content, str):
            # Plain text message (first user message is often a string)
            continue

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            btype = block.get("type", "")

            if btype == "text" and role == "assistant":
                text = block.get("text", "")
                if text.strip():
                    events.append(TimelineEvent(
                        step=0,
                        timestamp=ts,
                        actor=actor,
                        event_type="text",
                        text=text,
                        llm_output_tokens=llm_tokens,
                        llm_duration=llm_dur,
                    ))

            elif btype == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                tool_use_id = block.get("id", "")
                tool_id_map[tool_use_id] = tool_name

                # Determine event type
                if tool_name == "post_to_bbs":
                    channel = tool_input.get("channel", "")
                    bbs_content = tool_input.get("content", "")
                    events.append(TimelineEvent(
                        step=0,
                        timestamp=ts,
                        actor=actor,
                        event_type="bbs_post",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_input_summary=f"#{channel}: {_truncate(bbs_content, 100)}",
                        bbs_channel=channel,
                        bbs_content=bbs_content,
                        llm_output_tokens=llm_tokens,
                        llm_duration=llm_dur,
                    ))
                elif tool_name == "create_task":
                    task_name = tool_input.get("name", tool_input.get("task_name", ""))
                    profile = tool_input.get("profile", "")
                    events.append(TimelineEvent(
                        step=0,
                        timestamp=ts,
                        actor=actor,
                        event_type="task_create",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_input_summary=f"[{profile}] {task_name}",
                        llm_output_tokens=llm_tokens,
                        llm_duration=llm_dur,
                    ))
                else:
                    input_summary = ""
                    if tool_name == "web_search":
                        input_summary = tool_input.get("query", "")
                    elif tool_name == "web_fetch":
                        input_summary = tool_input.get("url", "")[:100]
                    elif tool_name == "reasoning":
                        input_summary = _truncate(tool_input.get("thought", tool_input.get("text", "")), 150)
                    else:
                        input_summary = _truncate(json.dumps(tool_input, default=str), 150)

                    events.append(TimelineEvent(
                        step=0,
                        timestamp=ts,
                        actor=actor,
                        event_type="tool_call",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_input_summary=input_summary,
                        llm_output_tokens=llm_tokens,
                        llm_duration=llm_dur,
                    ))

            elif btype == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                tool_name = tool_id_map.get(tool_use_id, block.get("tool_name", ""))
                result_content = block.get("content", "")
                is_error = block.get("is_error", False)
                tool_dur = block.get("_tool_duration_seconds")

                result_text = ""
                if isinstance(result_content, str):
                    result_text = result_content
                elif isinstance(result_content, list):
                    # Extract text from content blocks
                    parts = []
                    for rb in result_content:
                        if isinstance(rb, dict) and rb.get("type") == "text":
                            parts.append(rb.get("text", ""))
                    result_text = "\n".join(parts)
                else:
                    result_text = str(result_content)

                if tool_name == "read_bbs":
                    events.append(TimelineEvent(
                        step=0,
                        timestamp=ts,
                        actor=actor,
                        event_type="bbs_read",
                        tool_name=tool_name,
                        tool_result_text=result_text,
                        tool_result_summary=_truncate(result_text, 200),
                        tool_duration=tool_dur,
                        is_error=is_error,
                    ))
                else:
                    events.append(TimelineEvent(
                        step=0,
                        timestamp=ts,
                        actor=actor,
                        event_type="tool_result",
                        tool_name=tool_name,
                        tool_result_text=result_text,
                        tool_result_summary=_truncate(result_text, 200),
                        tool_duration=tool_dur,
                        is_error=is_error,
                    ))

    return events


def build_timeline(run_dir: Path, conv_id: str) -> TimelineData | None:
    """Build a unified timeline from a trajectory file."""
    safe_name = conv_id.replace("/", "_").replace("\\", "_")[:200]
    traj_path = run_dir / "trajectories" / f"{safe_name}.json"
    if not traj_path.exists():
        return None

    with open(traj_path) as f:
        raw = json.load(f)

    # Handle both formats
    if isinstance(raw, dict):
        phase = raw.get("trajectory", [])
        phase_timings = raw.get("phase_timings", {})
    elif isinstance(raw, list):
        phase = raw
        phase_timings = {}
    else:
        return None

    if not phase:
        return None

    t0 = phase[0]
    if not isinstance(t0, dict):
        return None

    # Extract orchestrator events
    orch_msgs = t0.get("orchestrator", [])
    all_events = _extract_events_from_messages(orch_msgs, "orchestrator")

    # Extract subagent events
    agents: list[AgentInfo] = []
    task_claims = {
        t.get("claimed_by") for t in t0.get("tasks", []) if t.get("claimed_by")
    }
    for sa in t0.get("subagents", []):
        sa_name = sa.get("name", "unknown")
        sa_msgs = sa.get("messages", [])
        sa_events = _extract_events_from_messages(sa_msgs, sa_name)
        all_events.extend(sa_events)

        first_ts = None
        if sa_msgs:
            first_ts = sa_msgs[0].get("_timestamp")

        # Detect idle reviewers: reasoning profile with no claimed task
        is_idle = (
            sa.get("initial_profile") == "reasoning"
            and sa_name not in task_claims
        )

        agents.append(AgentInfo(
            name=sa_name,
            profile=sa.get("initial_profile"),
            tool_calls_by_name=sa.get("tool_calls_by_name", {}),
            tasks_completed=sa.get("tasks_completed", 0),
            first_timestamp=first_ts,
            is_idle_reviewer=is_idle,
        ))

    # Add spawn events
    spawn_agent_ts: dict[str, str] = {}
    for a in agents:
        if a.first_timestamp:
            spawn_agent_ts[a.name] = a.first_timestamp

    for se in t0.get("spawn_events", []):
        assigned = se.get("assigned_to", "")
        # Use subagent's first message timestamp if available
        ts = spawn_agent_ts.get(assigned, "")
        all_events.append(TimelineEvent(
            step=0,
            timestamp=ts,
            actor="orchestrator",
            event_type="spawn",
            text=f"Spawned {assigned} for [{se.get('task_profile', '')}] {se.get('task_name', '')}",
        ))

    # Sort by timestamp
    all_events.sort(key=lambda e: e.timestamp or "")

    # Assign step indices
    for i, event in enumerate(all_events):
        event.step = i

    # Build tasks
    tasks: list[TaskInfo] = []
    for t in t0.get("tasks", []):
        summaries = t.get("summaries", [])
        summary_text = t.get("summary", "")
        if summaries and not summary_text:
            summary_text = " | ".join(s.get("content", "") for s in summaries if isinstance(s, dict))

        tasks.append(TaskInfo(
            id=t.get("id", ""),
            name=t.get("name", ""),
            prompt=t.get("prompt", ""),
            status=t.get("status", ""),
            profile=t.get("profile", ""),
            claimed_by=t.get("claimed_by"),
            summary=summary_text or None,
        ))

    # Build BBS snapshots
    bbs_posts: list[BBSPost] = []
    bbs_snapshots: dict[int, BBSSnapshot] = {}
    bbs_post_steps: list[int] = []

    for event in all_events:
        new_post = None
        if event.event_type == "bbs_post" and event.bbs_content:
            new_post = BBSPost(
                channel=event.bbs_channel or "",
                author=event.actor,
                timestamp=event.timestamp,
                content=event.bbs_content,
                estimated_tokens=_estimate_tokens(event.bbs_content),
            )
        elif event.event_type == "task_create" and event.tool_input:
            # Task creation shows up as #tasks channel on BBS
            task_name = event.tool_input.get("name", event.tool_input.get("task_name", ""))
            profile = event.tool_input.get("profile", "")
            prompt = event.tool_input.get("prompt", event.tool_input.get("description", ""))
            content = f"[task] {task_name} [profile: {profile}]: {prompt}"
            new_post = BBSPost(
                channel="tasks",
                author=event.actor,
                timestamp=event.timestamp,
                content=content,
                estimated_tokens=_estimate_tokens(content),
            )

        if new_post:
            bbs_posts.append(new_post)

            by_channel: dict[str, list[BBSPost]] = {}
            for p in bbs_posts:
                by_channel.setdefault(p.channel, []).append(p)

            total_tokens = sum(p.estimated_tokens for p in bbs_posts)
            bbs_snapshots[event.step] = BBSSnapshot(
                step=event.step,
                posts=list(bbs_posts),
                total_estimated_tokens=total_tokens,
                posts_by_channel={k: list(v) for k, v in by_channel.items()},
            )
            bbs_post_steps.append(event.step)

    # Get question text from first user message
    question = ""
    if orch_msgs:
        first_content = orch_msgs[0].get("content", "")
        if isinstance(first_content, str):
            question = first_content
        elif isinstance(first_content, list):
            for b in first_content:
                if isinstance(b, dict) and b.get("type") == "text":
                    question = b.get("text", "")
                    break

    total_dur = phase_timings.get("total", 0)

    return TimelineData(
        question=question,
        response_text="",  # set by caller from report
        total_duration_seconds=total_dur,
        agents=agents,
        tasks=tasks,
        events=all_events,
        bbs_snapshots=bbs_snapshots,
        bbs_post_steps=bbs_post_steps,
    )
