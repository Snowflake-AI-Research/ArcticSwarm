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

"""Rich-based BBS viewer for the REPL.

Provides:
- Static rendering of BBS state (for ``/bbs`` command)
- Rich ``Layout`` + ``Live`` split panel during swarm execution:
  - Top: subagent status table (name, state, current activity)
  - Bottom: "Swarm Live" feed showing tool calls, BBS posts, and votes
"""

from __future__ import annotations

import threading
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from arcticswarm.swarm.bbs import BBS, BBSMessage
from arcticswarm.swarm.references import ReferenceRegistry
from arcticswarm.swarm.report import (
    open_report,
    save_report,
)
from arcticswarm.swarm.orchestrator import (
    OrchestratorMessage,
    OrchestratorToolCall,
    ReportDelta,
    ReportStarted,
    SubagentClaimedTask,
    SubagentIdle,
    SubagentSpawned,
    SubagentSurfing,
    SwarmComplete,
    SwarmEvent,
    SwarmStarted,
    TeammateCompleted,
    TeammateFailed,
    TeammateSpawned,
    TeammateToolCall,
)

# BBS / coordination tools get highlighted to show inter-agent activity
_BBS_TOOLS = frozenset({"post_to_bbs", "read_bbs", "complete_task"})

# Orchestration tools get a distinctive style in the feed
_ORCH_TOOLS = frozenset({"create_task", "list_tasks", "wait_for_tasks"})

# Channel colors for the live BBS Board panel (consistent with existing styles)
_CHANNEL_COLORS: dict[str, str] = {
    "discoveries": "cyan",
    "consensus": "green",
    "tasks": "blue",
    "discussion": "magenta",
}

_CHANNEL_ORDER: list[str] = [
    "tasks",
    "discoveries",
    "key-findings",
    "discussion",
    "consensus",
]

# Max messages shown per channel in the BBS Board panel
_BBS_CHANNEL_ROWS: int = 4


# ---------------------------------------------------------------------------
# Static BBS rendering (for /bbs command)
# ---------------------------------------------------------------------------


def render_bbs(
    bbs: BBS,
    console: Console,
    *,
    channel: str | None = None,
    limit: int = 30,
) -> None:
    """Print BBS messages to the console in a rich panel."""
    msgs = bbs.read(channel=channel, limit=limit)

    if not msgs:
        console.print("[dim]No BBS messages yet.[/dim]")
        return

    table = Table(
        title=f"BBS {'#' + channel if channel else '(all channels)'}",
        show_lines=True,
        expand=True,
    )
    table.add_column("Channel", style="cyan", width=14)
    table.add_column("Author", style="bold", width=20)
    table.add_column("Content", ratio=1)

    for m in msgs:
        content = m.content
        if len(content) > 200:
            content = content[:197] + "..."
        tags_str = ""
        if m.tags:
            tags_str = f"\n[dim]tags: {', '.join(m.tags)}[/dim]"
        reply_str = ""
        if m.in_reply_to:
            reply_str = f"\n[dim]re: {m.in_reply_to}[/dim]"

        table.add_row(
            f"#{m.channel}",
            m.author,
            content + tags_str + reply_str,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Swarm event renderer — Rich Layout + Live split panel
# ---------------------------------------------------------------------------


class SwarmRenderer:
    """Real-time swarm progress display using Rich Layout + Live.

    Shows a three-panel vertical split during swarm execution:
      - Top: subagent status table (name, status, current activity)
      - Middle: live BBS Board showing recent messages by channel
      - Bottom: "Swarm Live" feed showing tool calls, BBS posts, and votes

    All panels update as ``SwarmEvent``s arrive from the orchestrator.
    Thread-safe: Rich's ``Live`` handles rendering from any thread.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        # RLock (reentrant) because on_swarm_event holds the lock and calls
        # _add_feed_line / _update_layout which also acquire it.
        self._lock = threading.RLock()

        # Agent tracking
        self._agent_status: dict[str, str] = {}
        self._agent_activity: dict[str, str] = {}
        self._agent_task: dict[str, str] = {}
        self._task_plan: list[dict[str, Any]] = []

        # Live feed (latest N lines shown in the bottom panel)
        self._feed: list[str] = []
        self._max_feed_lines: int = 30

        # BBS reference (set when SwarmStarted arrives with the BBS instance)
        self._bbs: BBS | None = None

        # Live display (created lazily when the panels activate)
        self._live: Live | None = None
        self._layout: Layout | None = None

        # Two-phase display:
        #   phase 1 (panels_active=False): pre-``SwarmStarted`` actions are
        #       printed directly to the console as a rolling log.
        #   phase 2 (panels_active=True): full three-panel Live layout showing
        #       agent status, BBS board, and activity feed.
        # The transition happens on ``SwarmStarted``.
        self._panels_active: bool = False

        # Report accumulation state.  When the orchestrator streams its
        # final report, chunks are buffered here and printed statically
        # once the swarm completes.
        self._report_buffer: str = ""
        self.streamed_report: str = ""

    def start(self) -> None:
        """Arm the renderer in phase 1 (rolling console output).

        The full three-panel Live display activates when ``SwarmStarted``
        arrives, which fires once the orchestrator begins its turn.
        """
        # Pre-populate the orchestrator so it appears from the very first frame.
        self._agent_status["orchestrator"] = "working"
        self._agent_task["orchestrator"] = "lead agent"
        self._agent_activity["orchestrator"] = "starting..."
        # No Live or Layout yet — phase 1 prints directly to console.

    def _upgrade_to_panels(self) -> None:
        """Transition from phase 1 (rolling log) to the full panel display."""
        self.console.print()  # visual break before the panels appear
        self._panels_active = True
        self._layout = Layout()
        self._layout.split_column(
            Layout(name="status", size=5),  # 4 overhead + 1 for orchestrator
            Layout(name="bbs", ratio=1),    # live BBS Board
            Layout(name="feed", ratio=1),
        )
        self._update_layout()
        self._live = Live(
            self._layout,
            console=self.console,
            refresh_per_second=15,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._layout = None

    # -- phase 1 helpers -------------------------------------------------------

    def _console_line(self, text: str) -> None:
        """Print a line directly to the console (phase 1 only)."""
        if not self._panels_active:
            self.console.print(text)

    # -- event handling --------------------------------------------------------

    def on_swarm_event(self, event: SwarmEvent) -> None:
        """Handle a swarm event and update the display.

        In **phase 1** (before ``SwarmStarted``) actions are printed
        directly to the console as a rolling log.
        Once ``SwarmStarted`` arrives, the renderer
        transitions to **phase 2** — the full three-panel Live layout.

        Thread-safe: the lock serialises all dict mutations and layout
        rebuilds so that concurrent events from pool threads cannot cause
        ``RuntimeError: dictionary changed size during iteration``.
        """
        with self._lock:
            if isinstance(event, SwarmStarted):
                # Capture the BBS reference for the live BBS Board panel
                if event.bbs is not None:
                    self._bbs = event.bbs
                # Add the orchestrator as the first row in the status table
                self._agent_status["orchestrator"] = "working"
                self._agent_task["orchestrator"] = "lead agent"
                self._agent_activity["orchestrator"] = "analyzing question..."
                self._add_feed_line(
                    "[bold cyan][swarm][/bold cyan] Orchestrator started"
                )
                # Activate the full panel UI immediately once the
                # orchestrator's turn begins.
                if not self._panels_active:
                    self._upgrade_to_panels()
                    return

            elif isinstance(event, SubagentSpawned):
                # Pre-spawned subagent — starts as idle.
                # Tracked internally; silent in phase 1 (not interesting
                # until tasks are out).
                self._agent_status[event.name] = "idle"
                self._agent_task[event.name] = ""
                self._agent_activity[event.name] = "ready"
                self._add_feed_line(
                    f"[bold blue]>[/bold blue] "
                    f"[bold]{event.name}[/bold] joined the team"
                )

            elif isinstance(event, SubagentClaimedTask):
                # Subagent picked up a task
                self._agent_status[event.name] = "working"
                self._agent_task[event.name] = event.activity
                self._agent_activity[event.name] = "starting..."
                self._add_feed_line(
                    f"[bold green]>[/bold green] "
                    f"[bold]{event.name}[/bold] claimed — [dim]{event.activity}[/dim]"
                )

            elif isinstance(event, SubagentSurfing):
                # Subagent is reading/posting on BBS during idle time
                self._agent_status[event.name] = "surfing"
                self._agent_activity[event.name] = event.activity or "surfing BBS"
                self._add_feed_line(
                    f"  [magenta]{event.name}[/magenta] surfing BBS..."
                )

            elif isinstance(event, SubagentIdle):
                # Subagent finished a task or is waiting
                self._agent_status[event.name] = "idle"
                self._agent_task[event.name] = ""
                self._agent_activity[event.name] = event.activity or "ready"

            elif isinstance(event, OrchestratorToolCall):
                # Update orchestrator activity in the status table
                self._agent_activity["orchestrator"] = event.description
                # Show in the feed with a distinctive style
                if event.tool_name in _ORCH_TOOLS:
                    # Orchestration tools — highlighted in cyan
                    self._add_feed_line(
                        f"  [bold cyan]orchestrator[/bold cyan] {event.description}"
                    )
                elif event.tool_name in _BBS_TOOLS:
                    # BBS tools — highlighted in magenta
                    self._add_feed_line(
                        f"  [magenta]orchestrator[/magenta] {event.description}"
                    )
                else:
                    # Data tools — subtler
                    self._add_feed_line(
                        f"  [dim cyan]orchestrator[/dim cyan] [dim yellow]{event.description}[/dim yellow]"
                    )
                # Phase 1: print orchestrator tool calls to console
                self._console_line(
                    f"  [dim cyan]▸[/dim cyan] {event.description}"
                )

            # Legacy event — keep for backward compatibility
            elif isinstance(event, TeammateSpawned):
                self._agent_status[event.name] = "working"
                task_desc = event.prompt[:200] + "..." if len(event.prompt) > 200 else event.prompt
                self._agent_task[event.name] = task_desc
                self._agent_activity[event.name] = "starting..."
                self._add_feed_line(
                    f"[bold green]>[/bold green] "
                    f"[bold]{event.name}[/bold] spawned — [dim]{task_desc}[/dim]"
                )

            elif isinstance(event, TeammateToolCall):
                is_bbs = event.tool_name in _BBS_TOOLS
                if is_bbs:
                    # BBS/coordination tools — highlighted in magenta
                    self._add_feed_line(
                        f"  [magenta]{event.name}[/magenta] {event.description}"
                    )
                else:
                    # Regular tools — subtler
                    self._add_feed_line(
                        f"  [dim]{event.name}[/dim] [dim yellow]{event.description}[/dim yellow]"
                    )
                # Update the Activity column with the latest tool
                if self._agent_status.get(event.name) in ("working", "idle"):
                    self._agent_activity[event.name] = event.description

            elif isinstance(event, TeammateCompleted):
                # Task completed — subagent returns to idle (not "done")
                self._agent_status[event.name] = "idle"
                self._agent_task[event.name] = ""
                self._agent_activity[event.name] = "ready"
                self._add_feed_line(
                    f"[bold green]OK[/bold green] "
                    f"[bold]{event.name}[/bold] completed task"
                )

            elif isinstance(event, TeammateFailed):
                self._agent_status[event.name] = "idle"
                self._agent_activity[event.name] = f"Error: {event.error[:200]}"
                self._add_feed_line(
                    f"[bold red]FAIL[/bold red] "
                    f"[bold]{event.name}[/bold] failed — {event.error[:200]}"
                )

            elif isinstance(event, OrchestratorMessage):
                # Intermediate reasoning text — add a truncated feed line
                text = event.text.replace("\n", " ").replace("\r", " ").strip()
                if text:
                    snippet = text[:200] + "..." if len(text) > 200 else text
                    self._add_feed_line(
                        f"  [dim italic]orchestrator:[/dim italic] [dim]{snippet}[/dim]"
                    )
                    self._console_line(f"  [dim]{text}[/dim]")

            elif isinstance(event, ReportStarted):
                # Tear down the swarm panel; report will be printed
                # statically once SwarmComplete arrives.
                if self._panels_active:
                    self._update_layout()
                self.stop()
                self._report_buffer = ""
                self._agent_activity["orchestrator"] = "writing report..."
                return  # panel is gone — no layout to update

            elif isinstance(event, ReportDelta):
                # Just accumulate text; we print it all at once on SwarmComplete.
                self._report_buffer += event.text
                return

            elif isinstance(event, SwarmComplete):
                # If the swarm panel is still up (report tool was never
                # called), tear it down now.
                if self._live is not None:
                    self._agent_status["orchestrator"] = "done"
                    self._agent_activity["orchestrator"] = "done"
                    for name in list(self._agent_status):
                        if name != "orchestrator":
                            self._agent_status[name] = "done"
                            self._agent_activity[name] = "done"
                    self._update_layout()
                    self.stop()

                # Save the report as a self-contained HTML file and open
                # it in the default browser instead of rendering markdown
                # in the terminal.  Prefer the full report from
                # SwarmComplete (which includes safety-net references
                # appended by SendReportTool) over the raw streamed
                # content in _report_buffer.
                report_text = (event.report.strip() or self._report_buffer.strip())
                if report_text:
                    # Build reference registry from BBS for modal HTML rendering
                    registry = None
                    if self._bbs is not None:
                        registry = ReferenceRegistry.from_bbs(
                            self._bbs,
                            web_source_tracker=event.web_source_tracker
                        )
                        if len(registry) == 0:
                            registry = None
                    report_path = save_report(
                        report_text, reference_registry=registry,
                    )
                    self.console.print()
                    self.console.print(
                        f"[bold cyan]Report saved:[/bold cyan] "
                        f"[link=file://{report_path}]file://{report_path}[/link]"
                    )
                    open_report(report_path)

                self.streamed_report = report_text or self._report_buffer

                self.console.print()
                self.console.print(
                    f"[dim]  Swarm complete: "
                    f"{event.subagent_count} subagent(s), "
                    f"{event.bbs_message_count} BBS message(s), "
                    f"{event.duration_seconds:.1f}s[/dim]"
                )
                self.console.print()
                return  # Don't try to update layout after stop

            if self._panels_active:
                self._update_layout()

    def _build_bbs_panel(self) -> Panel:
        """Build the BBS Board panel organised by channel.

        Each known channel gets a header and up to ``_BBS_CHANNEL_ROWS``
        most-recent messages (FIFO — oldest are dropped when new ones arrive).
        Channels with no messages are shown as dim placeholders so the layout
        stays stable.
        """
        if self._bbs is None:
            return Panel("[dim]Waiting...[/dim]", title="BBS Board", border_style="magenta")

        count = self._bbs.message_count
        if count == 0:
            return Panel(
                "[dim]No messages yet.[/dim]",
                title="BBS Board",
                border_style="magenta",
            )

        # Bucket all messages by channel
        all_msgs = self._bbs.read(limit=200)
        by_channel: dict[str, list[BBSMessage]] = {}
        for m in all_msgs:
            by_channel.setdefault(m.channel, []).append(m)

        # Use a grid table so each line is auto-truncated with ellipsis
        # at the panel width — no manual char-count needed, and every line
        # stays on a single row just like the Swarm Live feed.
        grid = Table.grid(expand=True)
        grid.add_column(no_wrap=True, overflow="ellipsis")

        rendered_channels: set[str] = set()

        for ch in _CHANNEL_ORDER:
            rendered_channels.add(ch)
            for line in self._render_channel_lines(ch, by_channel.get(ch)):
                grid.add_row(line)

        # Any channels not in the canonical list (agents can invent channels)
        for ch in sorted(by_channel):
            if ch not in rendered_channels:
                for line in self._render_channel_lines(ch, by_channel[ch]):
                    grid.add_row(line)

        return Panel(
            grid,
            title=f"BBS Board ({count})",
            border_style="magenta",
        )

    @staticmethod
    def _render_channel_lines(
        channel: str,
        msgs: list[BBSMessage] | None,
    ) -> list[str]:
        """Return per-line markup strings for one channel section."""
        color = _CHANNEL_COLORS.get(channel, "dim")
        msg_count = len(msgs) if msgs else 0
        header = f"[{color} bold]#{channel}[/{color} bold] ({msg_count})"

        if not msgs:
            return [f"{header}  [dim]—[/dim]"]

        # Keep only the most recent _BBS_CHANNEL_ROWS messages (FIFO)
        recent = msgs[-_BBS_CHANNEL_ROWS:]
        lines = [header]
        for m in recent:
            text = m.content.replace("\n", " ").replace("\r", " ")
            lines.append(f"  [bold]{m.author}[/bold]  {text}")
        return lines

    def _add_feed_line(self, line: str) -> None:
        """Add a line to the live feed, trimming old entries."""
        # Collapse newlines so each feed entry stays on a single line
        line = line.replace("\n", " ").replace("\r", " ")
        with self._lock:
            self._feed.append(line)
            if len(self._feed) > self._max_feed_lines:
                self._feed = self._feed[-self._max_feed_lines:]

    def _resize_status_panel(self) -> None:
        """Resize the top panel to fit the number of agents."""
        if self._layout is not None:
            # 4 = panel border (2) + header row (1) + header rule (1); +1 per agent
            n = max(len(self._agent_status), 1)
            self._layout["status"].size = 4 + n

    def _update_layout(self) -> None:
        """Rebuild the layout panels from current state."""
        if self._layout is None:
            return

        # Keep the status panel sized to fit all agent rows
        self._resize_status_panel()

        # -- Top panel: agent status table -------------------------------------
        status_table = Table(expand=True, show_header=True, show_edge=False)
        status_table.add_column("Agent", style="bold", width=20, no_wrap=True)
        status_table.add_column("Task", style="dim", ratio=1, no_wrap=True, overflow="ellipsis")
        status_table.add_column("Status", width=10, no_wrap=True)
        status_table.add_column("Activity", ratio=1, style="dim", no_wrap=True, overflow="ellipsis")

        for name, status in self._agent_status.items():
            style_map = {
                "idle": "[dim]idle[/dim]",
                "pending": "[dim]pending[/dim]",
                "surfing": "[bold magenta]surfing[/bold magenta]",
                "working": "[bold yellow]working[/bold yellow]",
                "done": "[bold green]done[/bold green]",
                "failed": "[bold red]failed[/bold red]",
            }
            status_str = style_map.get(status, status)
            task = self._agent_task.get(name, "").replace("\n", " ").replace("\r", " ")
            if len(task) > 200:
                task = task[:197] + "..."
            activity = self._agent_activity.get(name, "").replace("\n", " ").replace("\r", " ")
            status_table.add_row(name, task, status_str, activity)

        self._layout["status"].update(
            Panel(status_table, title="Swarm Status", border_style="cyan")
        )

        # -- Middle panel: BBS Board --------------------------------------------
        self._layout["bbs"].update(self._build_bbs_panel())

        # -- Bottom panel: Swarm Live feed ------------------------------------
        with self._lock:
            feed_text = "\n".join(self._feed) if self._feed else "[dim]Waiting...[/dim]"
        self._layout["feed"].update(
            Panel(feed_text, title="Swarm Live", border_style="blue")
        )
