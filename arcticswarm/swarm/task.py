"""Task board — shared task lifecycle management for swarm agents.

Tasks are created by the orchestrator, claimed by teammates, and completed
with a summary.  Dependencies between tasks are tracked so the orchestrator
knows when blocked tasks become ready.
"""

from __future__ import annotations

import enum
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any


class TaskStatus(enum.Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SummaryEntry:
    """A single summary entry appended to a task."""

    author: str
    content: str
    seq: int


@dataclass
class TaskSpec:
    """A single task on the board."""

    id: str
    name: str
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    claimed_by: str | None = None
    summaries: list[SummaryEntry] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    profile: str = ""  # Tool profile name ("browsing", "coding", "reasoning", or "" for default)
    assigned_to: str | None = None  # If set, only this subagent may claim the task.

    # Progress heartbeat (Claude-Code-style ``ProgressTracker``).  Populated
    # by :meth:`TaskBoard.bump_progress` on every subagent tool-use event so
    # the orchestrator can see *live* activity in ``list_tasks`` rendering
    # instead of a binary "running/completed" label.  Used to detect stuck
    # subagents (stale heartbeat) and to let ``wait_for_tasks`` exit early.
    tool_use_count: int = 0
    last_activity_tool: str = ""       # e.g. "web_fetch", "python_execute"
    last_activity_input: str = ""      # short preview, <= 120 chars
    last_heartbeat: float = 0.0        # ``time.monotonic()`` of last update
    token_count: int = 0               # cumulative tokens charged to this task

    @property
    def summary(self) -> str:
        if not self.summaries:
            return ""
        if len(self.summaries) == 1:
            return self.summaries[0].content
        lines = []
        for entry in self.summaries:
            tag = f"[{entry.seq}] {entry.author}"
            lines.append(f"{tag}: {entry.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "claimed_by": self.claimed_by,
            "summary": self.summary,
            "summaries": [
                {"seq": e.seq, "author": e.author, "content": e.content}
                for e in self.summaries
            ],
            "error": self.error,
            "profile": self.profile,
            "assigned_to": self.assigned_to,
            "tool_use_count": self.tool_use_count,
            "last_activity_tool": self.last_activity_tool,
            "last_activity_input": self.last_activity_input,
            "last_heartbeat": self.last_heartbeat,
            "token_count": self.token_count,
        }


# ---------------------------------------------------------------------------
# Alternative / contrarian task detection
# ---------------------------------------------------------------------------

# Tokens that mark a task as an alternative / contrarian exploration.  A task
# whose name (split on non-alphanumerics) yields a token equal to ``alt`` or
# containing ``alternativ`` / ``contrarian`` counts.  This mirrors the
# behavioural signal measured in the arcticswarm paper ("premature commitment"):
# the number of orchestrator tasks tagged ``alt`` / ``alternative`` /
# ``contrarian``.  Cases that never open such a task commit to the first
# plausible candidate and lose accuracy, especially on harder questions.
_ALT_NAME_RE = re.compile(r"[^a-z0-9]+")


def task_is_alt(spec: "TaskSpec") -> bool:
    """Return True if *spec* is an alternative / contrarian exploration task.

    A task qualifies if it is explicitly flagged (``metadata['alt']`` truthy —
    set by the ``create_task`` ``alt`` parameter or by a harness auto-spawn) or
    if its ``name`` carries an ``alt`` / ``alternativ*`` / ``contrarian`` token.
    Used by ``PrepareReportTool``'s alt-task gate so every web/BBS run is
    guaranteed at least one genuinely independent rival hypothesis.
    """
    if spec is None:
        return False
    try:
        if spec.metadata.get("alt"):
            return True
    except AttributeError:
        pass
    for tok in _ALT_NAME_RE.split((spec.name or "").lower()):
        if tok == "alt" or "alternativ" in tok or "contrarian" in tok:
            return True
    return False


# Threshold: a running task whose last_heartbeat is older than this (seconds)
# is flagged ``STALE`` by ``list_tasks``/``wait_for_tasks`` rendering.  Keeps
# the orchestrator from blindly waiting on a subagent that has stopped making
# tool-use progress (e.g. looping or hung LLM call).
STALE_HEARTBEAT_THRESHOLD_SECONDS = 300.0


class TaskBoard:
    """Thread-safe shared task board.

    The orchestrator adds tasks; teammates claim and complete them.
    Dependency resolution is handled via :meth:`ready_tasks`.
    """

    def __init__(self, num_agents: int = 0) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskSpec] = {}
        # Saturation tracking: how many times a task became ready but all agents were busy
        self._num_agents = num_agents
        self.saturation_events: int = 0

    def add_task(self, task: TaskSpec) -> None:
        """Add a new task to the board."""
        with self._lock:
            self._tasks[task.id] = task
            self._check_saturation_locked()

    def add_tasks(self, tasks: list[TaskSpec]) -> None:
        """Bulk-add tasks."""
        with self._lock:
            for task in tasks:
                self._tasks[task.id] = task
            self._check_saturation_locked()

    def get_task(self, task_id: str) -> TaskSpec | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_task_by_name(self, name: str) -> TaskSpec | None:
        """Fallback lookup by task name (for when agents pass name instead of ID)."""
        with self._lock:
            for task in self._tasks.values():
                if task.name == name:
                    return task
            return None

    def resolve_task_id(self, task_id_or_name: str) -> TaskSpec | None:
        """Look up a task by ID first, then fall back to name."""
        task = self.get_task(task_id_or_name)
        if task is None:
            task = self.get_task_by_name(task_id_or_name)
        return task

    def _check_saturation_locked(self) -> None:
        """Check if there are more ready tasks than available agents (must hold lock)."""
        if self._num_agents <= 0:
            return
        # Count ready (PENDING with deps met) tasks
        ready_count = 0
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self._tasks.get(dep_id) is not None
                and self._tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.depends_on
            )
            if deps_met:
                ready_count += 1
        # Count busy agents (tasks in CLAIMED or RUNNING state)
        busy_agents = sum(
            1 for t in self._tasks.values()
            if t.status in (TaskStatus.CLAIMED, TaskStatus.RUNNING)
        )
        # Saturation: ready tasks waiting but all agents are occupied
        if ready_count > 0 and busy_agents >= self._num_agents:
            self.saturation_events += 1

    def find_by_name(self, name: str) -> TaskSpec | None:
        """Find a task by its human-readable name."""
        with self._lock:
            for task in self._tasks.values():
                if task.name == name:
                    return task
            return None

    def claim(self, task_id: str, agent_name: str) -> bool:
        """Try to claim a task.  Returns True if successful."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.PENDING:
                return False
            # Check dependencies
            for dep_id in task.depends_on:
                dep = self._tasks.get(dep_id)
                if dep is None or dep.status != TaskStatus.COMPLETED:
                    return False
            if task.assigned_to is not None and task.assigned_to != agent_name:
                return False
            task.status = TaskStatus.CLAIMED
            task.claimed_by = agent_name
            return True

    def mark_running(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.RUNNING
                # Seed a heartbeat so the first render shows "0s ago" instead
                # of a misleading stale marker before the first tool call.
                task.last_heartbeat = time.monotonic()

    def bump_progress(
        self,
        task_id: str,
        *,
        tool_name: str,
        input_preview: str = "",
        tokens: int = 0,
    ) -> None:
        """Record a live progress heartbeat for a running task.

        Called by the subagent turn loop on every ``ToolUseStart`` event so
        the orchestrator can see what each subagent is currently doing via
        ``list_tasks`` / ``wait_for_tasks`` output.  Mirrors Claude Code's
        ``updateProgressFromMessage`` / ``ProgressTracker`` pattern.

        No-op on terminal-state tasks so stray late events from a finished
        subagent cannot corrupt progress metadata.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
            ):
                return
            task.tool_use_count += 1
            task.last_activity_tool = tool_name
            task.last_activity_input = (input_preview or "")[:120]
            task.last_heartbeat = time.monotonic()
            if tokens > 0:
                task.token_count = tokens

    def complete(self, task_id: str, summary: str = "") -> bool:
        """Mark a task as completed, appending the initial summary entry.

        Idempotent: returns ``True`` only on the actual pending/claimed/running
        -> completed transition.  Subsequent calls (or calls on an already
        failed task) are no-ops that return ``False``.  Callers such as
        :class:`CompleteTaskTool` rely on this to gate peer DM notifications
        so duplicate completions do not double-notify (Claude Code's
        ``completeAgentTask`` idempotency pattern).
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
            ):
                return False
            task.status = TaskStatus.COMPLETED
            if summary:
                seq = len(task.summaries) + 1
                author = task.claimed_by or "unknown"
                task.summaries.append(SummaryEntry(author=author, content=summary, seq=seq))
            self._check_saturation_locked()
            return True

    def append_summary(self, task_id: str, author: str, content: str) -> bool:
        """Append a correction or update to a completed task's summary.

        Returns True if successful, False if the task doesn't exist or
        isn't completed.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.COMPLETED:
                return False
            seq = len(task.summaries) + 1
            task.summaries.append(SummaryEntry(author=author, content=content, seq=seq))
            return True

    def fail(self, task_id: str, error: str = "") -> bool:
        """Mark a task as failed.

        Idempotent: returns ``True`` only on the actual non-terminal ->
        failed transition.  Calls on an already-terminal task (completed
        or failed) are no-ops that return ``False``.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
            ):
                return False
            task.status = TaskStatus.FAILED
            task.error = error
            return True

    def ready_tasks(self) -> list[TaskSpec]:
        """Return tasks whose dependencies are all completed (topological readiness)."""
        with self._lock:
            ready: list[TaskSpec] = []
            for task in self._tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                deps_met = all(
                    self._tasks.get(dep_id) is not None
                    and self._tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.depends_on
                )
                if deps_met:
                    ready.append(task)
            return ready

    def all_completed(self) -> bool:
        """Return True if every task is completed or failed."""
        with self._lock:
            return all(
                t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                for t in self._tasks.values()
            )

    def all_tasks(self) -> list[TaskSpec]:
        """Return a snapshot of all tasks."""
        with self._lock:
            return list(self._tasks.values())

    @property
    def task_count(self) -> int:
        with self._lock:
            return len(self._tasks)

    def render_status(
        self,
        *,
        now: float | None = None,
        stale_threshold: float = STALE_HEARTBEAT_THRESHOLD_SECONDS,
    ) -> str:
        """Human-readable summary of task statuses.

        For running/claimed tasks with a populated heartbeat, a second line
        per task is appended describing the last tool-use event::

            poly-theory-analysis: running [Wallace] [profile: reasoning]
              activity: web_fetch https://... (12s ago, 18 tool uses)

        If the heartbeat is older than *stale_threshold* seconds the activity
        line is also tagged ``STALE`` so the orchestrator can distinguish a
        productive subagent from a stuck one.
        """
        with self._lock:
            tasks = list(self._tasks.values())
        if not tasks:
            return "(no tasks)"
        if now is None:
            now = time.monotonic()
        lines: list[str] = []
        for t in tasks:
            deps = f" (depends: {', '.join(t.depends_on)})" if t.depends_on else ""
            owner = f" [{t.claimed_by}]" if t.claimed_by else ""
            profile_tag = f" [profile: {t.profile}]" if t.profile else ""
            lines.append(f"  {t.name}: {t.status.value}{owner}{deps}{profile_tag}")
            if (
                t.status in (TaskStatus.CLAIMED, TaskStatus.RUNNING)
                and t.last_heartbeat > 0.0
            ):
                age = max(0.0, now - t.last_heartbeat)
                stale_tag = " STALE" if age > stale_threshold else ""
                activity = t.last_activity_tool or "(waiting for first tool call)"
                preview = (
                    f" {t.last_activity_input}"
                    if t.last_activity_input
                    else ""
                )
                lines.append(
                    f"    activity: {activity}{preview} "
                    f"({int(age)}s ago, {t.tool_use_count} tool uses"
                    f"{f', {t.token_count // 1000}k tok' if t.token_count else ''})"
                    f"{stale_tag}"
                )
        return "\n".join(lines)

    def any_running_stale(
        self,
        task_ids: list[str] | None = None,
        *,
        now: float | None = None,
        stale_threshold: float = STALE_HEARTBEAT_THRESHOLD_SECONDS,
    ) -> bool:
        """Return True if ALL named running tasks have gone stale.

        If *task_ids* is None, inspects every running task on the board.
        Returns False if any named task is still in-progress (heartbeat
        younger than *stale_threshold*) or is already terminal.  Used by
        :class:`WaitForTasksTool` to exit the polling loop early when no
        subagent is making progress.
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            if task_ids is None:
                running = [
                    t for t in self._tasks.values()
                    if t.status in (TaskStatus.CLAIMED, TaskStatus.RUNNING)
                ]
            else:
                running = []
                for tid in task_ids:
                    t = self._tasks.get(tid)
                    if t is None:
                        # Fall back to name lookup
                        t = next(
                            (tt for tt in self._tasks.values() if tt.name == tid),
                            None,
                        )
                    if t is None:
                        continue
                    if t.status in (TaskStatus.CLAIMED, TaskStatus.RUNNING):
                        running.append(t)
            if not running:
                return False
            for t in running:
                if t.last_heartbeat <= 0.0:
                    return False
                if (now - t.last_heartbeat) <= stale_threshold:
                    return False
            return True


# ---------------------------------------------------------------------------
# Subagent status tracking
# ---------------------------------------------------------------------------


class AgentStatus(enum.Enum):
    """Status of a persistent subagent in the pool."""

    IDLE = "idle"
    WORKING = "working"
    SURFING = "surfing"


class AgentRegistry:
    """Thread-safe registry of subagent statuses and activities.

    The orchestrator and viewer read this to display the current state of
    each subagent in the pool.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._statuses: dict[str, AgentStatus] = {}
        self._activities: dict[str, str] = {}

    def register(self, name: str) -> None:
        """Register a new subagent with IDLE status."""
        with self._lock:
            self._statuses[name] = AgentStatus.IDLE
            self._activities[name] = ""

    def set_status(
        self,
        name: str,
        status: AgentStatus,
        activity: str = "",
    ) -> None:
        """Update a subagent's status and activity description."""
        with self._lock:
            self._statuses[name] = status
            self._activities[name] = activity

    def set_activity(self, name: str, activity: str) -> None:
        """Update only the activity description (status unchanged)."""
        with self._lock:
            self._activities[name] = activity

    def get_status(self, name: str) -> tuple[AgentStatus, str]:
        """Return ``(status, activity)`` for a subagent."""
        with self._lock:
            return (
                self._statuses.get(name, AgentStatus.IDLE),
                self._activities.get(name, ""),
            )

    def all_idle(self) -> bool:
        """Return True if every registered subagent is idle."""
        with self._lock:
            return all(
                s == AgentStatus.IDLE for s in self._statuses.values()
            )

    def names(self) -> list[str]:
        """Return a snapshot of all registered subagent names."""
        with self._lock:
            return list(self._statuses.keys())
