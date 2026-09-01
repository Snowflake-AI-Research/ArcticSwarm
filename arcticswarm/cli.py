"""Rich interactive REPL for Arcticswarm.

Provides:
  - Streaming markdown output
  - Tool-call progress indicators (spinner + status)
  - Slash commands: /model, /connection, /clear, /quit, /help
  - Multiline input (paste-friendly; submit with Enter on an empty line)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys

import time as _time
import tty
import termios
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arcticswarm.swarm.bbs import BBS
    from arcticswarm.swarm.orchestrator import SwarmOrchestrator

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from arcticswarm import __version__
from arcticswarm.agent import (
    Agent,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCallEnd,
    ToolCallStart,
    TurnComplete,
)
from arcticswarm.logging_utils import (
    SearchApiUnhealthyError,
    check_search_api_health,
)
from arcticswarm.config import (
    ArcticswarmConfig,
    AVAILABLE_MODELS,
    get_model_info,
    ModelInfo,
    load_snowflake_connections,
    get_snowflake_connection_params,
    settings_json_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_token_count(s: str) -> int:
    """Parse "200k", "1m", or "200000" into an int token count."""
    if not s:
        return 0
    s = s.strip().lower().replace(",", "").replace("_", "")
    mult = 1
    if s.endswith("k"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid token count {s!r}: expected int, '200k', or '1m'"
        ) from exc
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arcticswarm",
        description="Arcticswarm — a research agent",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--config", "-c",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "YAML config file (repeatable, merged left-to-right; same format as "
            "arcticswarm-eval --config). When given, the YAML provides the base "
            "configuration and any other CLI flags below override it."
        ),
    )
    p.add_argument(
        "--no-stream", action="store_true",
        help="Disable streaming (wait for full response before displaying)",
    )
    p.add_argument(
        "--swarm", action="store_true",
        help="Enable swarm mode: questions use parallel agent teams",
    )
    p.add_argument(
        "--web-search", action="store_true",
        default=False,
        help="Enable web search tools and skills (requires brave_api_key or serper_api_key in settings)",
    )
    p.add_argument(
        "--web-provider",
        choices=["native", "corpus"],
        default="native",
        help="Web search backend: 'native' (direct APIs) or 'corpus' (BrowseComp-Plus document corpus via --corpus-backend). Default: native.",
    )
    p.add_argument(
        "--web-fetch-backend",
        choices=["native", "corpus"],
        default="native",
        help=(
            "web_fetch backend: 'native' uses the Jina→Serper→requests chain, "
            "'corpus' retrieves full documents from the corpus (see --corpus-backend). "
            "Default: native."
        ),
    )
    p.add_argument(
        "--corpus-backend",
        choices=["stub", "cortex", "local"],
        default=None,
        help=(
            "Corpus retrieval backend when --web-provider/--web-fetch-backend is 'corpus': "
            "'stub' (no-op placeholder, default), 'cortex' (Snowflake Cortex Search), or "
            "'local' (local JSONL via --corpus-local-path). See the README."
        ),
    )
    p.add_argument(
        "--swarm-comm", nargs="+", choices=["bbs", "dm", "duo"], default=["bbs"],
        help="Swarm communication channels (default: bbs). Combine: --swarm-comm bbs dm. Use 'duo' for 2-agent peer mode.",
    )
    p.add_argument(
        "--enable-vision", action="store_true",
        default=False,
        help="Enable image understanding in read_file (sends base64 image blocks to the LLM)",
    )
    p.add_argument(
        "--disable-source-scorer", action="store_true",
        default=False,
        help="Disable the source content scorer (no [Source Quality] annotations and no search-result judge gating). On by default; pass this to cut the per-call judge latency.",
    )
    p.add_argument(
        "--use-fetch-compactor", action="store_true",
        default=False,
        help="Route web_fetch results through the chunking compactor LLM "
             "(chunks the full page into ~1000-char sentence-aware blocks; "
             "the LLM picks relevant chunk indices plus a composite quality score). "
             "Replaces the source scorer for web_fetch only; search judge gating is unchanged.",
    )
    p.add_argument(
        "--use-pdf-compactor", action="store_true",
        default=False,
        help="Same as --use-fetch-compactor but for pdf_read results. "
             "Chunks the full PDF text (~1000 chars/chunk) and keeps only the "
             "LLM-selected chunks. Can be combined with --use-fetch-compactor.",
    )
    p.add_argument(
        "--context-compact-tokens",
        type=_parse_token_count,
        default=0,
        metavar="N",
        help="Trigger proactive compaction at this absolute input-token count "
             "instead of the default 90%% of the model's context limit. Accepts "
             "raw integers or k/m suffixes (e.g. 200k, 1m, 200000). Especially "
             "useful with --enable-1m-context where 90%% = 900K is too late.",
    )
    p.add_argument(
        "--disable-self-reflection", action="store_true",
        default=False,
        help="Disable adversarial self-reflection loop for browsing subagents (single search pass only)",
    )
    p.add_argument(
        "--enable-1m-context", action="store_true",
        default=False,
        help=(
            "Enable the 1M token context window (experimental). "
            "Passes experimental={Enable1MContextModel: true} to the Anthropic API "
            "and raises the compaction threshold from 180K to 900K tokens."
        ),
    )
    p.add_argument(
        "--max-tool-calls-per-turn", type=int, default=0,
        help="Max tool calls per LLM turn. 0=unlimited (default), 1=enforce single tool call.",
    )
    p.add_argument(
        "--num_teammates", type=int, default=5,
        help="Number of teammates to spawn in swarm mode (default: 5)",
    )
    p.add_argument(
        "--max-subagents", type=int, default=16,
        help="Hard cap on subagents in dynamic mode (default: 16)",
    )
    p.add_argument(
        "--max-subagent-tasks", type=int, default=3,
        help="Tasks before a dynamic worker is context-full (default: 3, -1=no limit)",
    )
    p.add_argument(
        "--session",
        help="Resume a previous session by UUID",
    )
    p.add_argument(
        "--usage", action="store_true",
        help="Show token usage stats after each turn",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        help="Override max turns per conversation (default: 150)",
    )
    p.add_argument(
        "--model",
        help="Override the model (default: from settings). Supports Claude and GPT models.",
    )
    p.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help=(
            "Dot-notation overrides applied on top of --config: "
            "key.subkey=value (e.g. llm.model=claude-opus-4-7 web.enabled=true). "
            "Ignored unless --config is also passed."
        ),
    )
    return p


def _explicit_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """Return the set of action.dest names whose option strings appear in *argv*.

    Used to distinguish "user explicitly passed this flag" from "argparse
    populated the default" — needed when YAML provides the base config and
    we only want CLI flags to win when they were actually given.
    """
    flag_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            flag_to_dest[opt] = action.dest
    explicit: set[str] = set()
    for tok in argv:
        flag = tok.split("=", 1)[0]
        dest = flag_to_dest.get(flag)
        if dest is not None:
            explicit.add(dest)
    return explicit


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


_HELP_TEXT = """\
**Slash commands:**
| Command | Description |
|---------|-------------|
| `/help` | Show this help |
| `/model` | Model & reasoning effort picker |
| `/connection` | Switch Snowflake connection |
| `/clear` | Clear conversation history |
| `/config` | Show current configuration |
| `/save [path]` | Save last response to a local file |
| `/swarm on/off` | Toggle swarm mode (parallel agent teams) |
| `/bbs [channel]` | Show BBS messages (swarm mode) |
| `/quit` or `/exit` | Exit Arcticswarm |

**Tips:**
- Paste multiline input freely; press Enter twice (blank line) to submit.
- Use `--swarm` or `/swarm on` to enable parallel agent teams.
"""


_SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/model", "Model & reasoning effort picker"),
    ("/connection", "Switch Snowflake connection"),
    ("/clear", "Clear conversation history"),
    ("/config", "Show current configuration"),
    ("/save", "Save last response to a local file"),
    ("/swarm", "Toggle swarm mode on/off"),
    ("/bbs", "Show BBS messages (swarm mode)"),
    ("/quit", "Exit Arcticswarm"),
    ("/exit", "Exit Arcticswarm"),
]


class _SlashCompleter(Completer):
    """Auto-complete slash commands when the line starts with ``/``."""

    def get_completions(self, document: Document, complete_event: object) -> Completion:
        text = document.text_before_cursor.lstrip()
        # Only complete on the first "word" and only when it starts with /
        if " " in text or not text.startswith("/"):
            return
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )


class _ReplState:
    """Mutable state shared across the REPL loop."""

    def __init__(self, swarm_enabled: bool = False) -> None:
        self.swarm_enabled = swarm_enabled
        self.orchestrator: "SwarmOrchestrator | None" = None
        self.last_bbs: "BBS | None" = None
        self.last_response: str = ""

    def get_orchestrator(self, config: ArcticswarmConfig) -> "SwarmOrchestrator":
        """Lazily create or return the swarm orchestrator."""
        from arcticswarm.swarm.orchestrator import SwarmOrchestrator
        if self.orchestrator is None:
            self.orchestrator = SwarmOrchestrator(
                config, max_teammates=config.max_teammates,
            )
        return self.orchestrator


# ---------------------------------------------------------------------------
# Interactive model + effort picker
# ---------------------------------------------------------------------------

_EFFORT_LEVELS = ["none", "low", "medium", "high", "xhigh"]


def _read_key() -> str:
    """Read a single keypress from stdin (raw mode). Returns a string tag."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home"}.get(ch3, "")
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\t":
            return "\t"
        if ch in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
            return "esc"
        if ch == "q":
            return "esc"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


@dataclass
class _PickerResult:
    """Result from the model picker."""
    model: str
    effort: str | None
    subagent_model: str           # "" means same as orchestrator
    subagent_effort: str | None   # None means same as orchestrator


def _run_model_picker(
    console: Console,
    current_model: str,
    current_effort: str | None,
    current_sub_model: str,
    current_sub_effort: str | None,
) -> _PickerResult | None:
    """Interactive two-tab model picker with effort adjustment.

    Uses direct ANSI escape codes to redraw in-place so it works
    reliably regardless of terminal quirks with Rich Live.

    Returns a ``_PickerResult`` on selection, or ``None`` on cancel.
    """
    n_models = len(AVAILABLE_MODELS)
    # Fixed height so we can always erase the same number of lines,
    # regardless of which tab/state is displayed.
    #   1 blank + 1 header + 1 blank + 1 tab bar + 1 blank
    #   + max(4 model rows, 2 "same" lines + 1 blank + 4 model rows) + 1 blank
    _FIXED_HEIGHT = 5 + max(n_models, 2 + 1 + n_models) + 1

    def _find_model_idx(model_id: str) -> int:
        for i, m in enumerate(AVAILABLE_MODELS):
            if m.id == model_id:
                return i
        return 0

    def _find_effort_idx(effort: str | None) -> int:
        if effort is None:
            return 0  # "none"
        return _EFFORT_LEVELS.index(effort) if effort in _EFFORT_LEVELS else 0

    # Tab state: 0 = Orchestrator, 1 = Subagents
    active_tab = 0

    # Orchestrator tab state
    orch_cursor = _find_model_idx(current_model)
    orch_efforts = [_find_effort_idx(current_effort)] * n_models

    # Subagent tab state — "same" checkbox when no override is set
    sub_same = not current_sub_model and not current_sub_effort
    sub_cursor = _find_model_idx(current_sub_model or current_model)
    sub_efforts = [_find_effort_idx(current_sub_effort or current_effort)] * n_models

    _TAB_NAMES = ["Orchestrator", "Subagents"]

    def _render_plain() -> list[str]:
        """Build the picker as a list of plain ANSI-styled strings (one per line)."""
        lines: list[str] = [
            "",
            "  \033[1mModel selector\033[0m  \033[2mTab: switch tab  \u2191\u2193: model  \u2190\u2192: effort  Enter: confirm  Esc: cancel\033[0m",
            "",
        ]
        # Tab bar
        tab_parts = ["  "]
        for i, name in enumerate(_TAB_NAMES):
            if i == active_tab:
                tab_parts.append(f"\033[1;36m[  {name}  ]\033[0m")
            else:
                tab_parts.append(f"\033[2m   {name}   \033[0m")
            tab_parts.append(" ")
        lines.append("".join(tab_parts))
        lines.append("")

        if active_tab == 0:
            lines.extend(_model_rows(orch_cursor, orch_efforts))
        else:
            if sub_same:
                orch_name = AVAILABLE_MODELS[orch_cursor].display_name
                orch_eff = _EFFORT_LEVELS[orch_efforts[orch_cursor]]
                lines.append(
                    f"  \033[1;36m>\033[0m \033[1mSame as Orchestrator\033[0m"
                    f"  \033[2m({orch_name}, {orch_eff})\033[0m"
                )
                lines.append("    \033[2mPress \u2193 to override\033[0m")
            else:
                lines.append("    \033[2mSame as Orchestrator  (press Home to reset)\033[0m")
                lines.append("")
                lines.extend(_model_rows(sub_cursor, sub_efforts))

        lines.append("")
        # Pad to fixed height so erasing is consistent
        while len(lines) < _FIXED_HEIGHT:
            lines.append("")
        return lines

    def _model_rows(cursor: int, efforts: list[int]) -> list[str]:
        rows: list[str] = []
        for i, m in enumerate(AVAILABLE_MODELS):
            selected = i == cursor
            # Effort display
            eff_parts = []
            for j, level in enumerate(_EFFORT_LEVELS):
                if j == efforts[i]:
                    eff_parts.append(f"\033[1;36m{level}\033[0m")
                else:
                    eff_parts.append(f"\033[2m{level}\033[0m")
            effort_display = "  ".join(eff_parts)

            pointer = "\033[1;36m>\033[0m" if selected else " "
            padded_name = m.display_name.ljust(12)
            if selected:
                name = f"\033[1m{padded_name}\033[0m"
            else:
                name = padded_name
            price = f"${m.input_per_mtok:g}/${m.output_per_mtok:g} per MTok"
            padded_price = price.ljust(20)

            if selected:
                rows.append(
                    f"  {pointer} {name}  \033[2m{padded_price}\033[0m  "
                    f"\033[2m<\033[0m {effort_display} \033[2m>\033[0m"
                )
            else:
                rows.append(
                    f"  {pointer} {name}  \033[2m{padded_price}\033[0m    {effort_display}"
                )
        return rows

    # --- Main loop: draw, read key, erase, redraw ---
    prev_line_count = 0
    try:
        while True:
            rendered = _render_plain()
            # Erase previous frame: move cursor up and clear each line
            if prev_line_count > 0:
                sys.stdout.write(f"\033[{prev_line_count}A")  # move up
                for _ in range(prev_line_count):
                    sys.stdout.write("\033[2K\n")  # clear line, move down
                sys.stdout.write(f"\033[{prev_line_count}A")  # move back up
            # Draw
            for line in rendered:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()
            prev_line_count = len(rendered)

            key = _read_key()

            if key == "\t":
                active_tab = (active_tab + 1) % 2
            elif key == "enter":
                orch_model = AVAILABLE_MODELS[orch_cursor].id
                orch_eff_raw = _EFFORT_LEVELS[orch_efforts[orch_cursor]]
                orch_eff = None if orch_eff_raw == "none" else orch_eff_raw
                if sub_same:
                    return _PickerResult(orch_model, orch_eff, "", None)
                else:
                    sub_model = AVAILABLE_MODELS[sub_cursor].id
                    sub_eff = _EFFORT_LEVELS[sub_efforts[sub_cursor]]
                    return _PickerResult(orch_model, orch_eff, sub_model, sub_eff)
            elif key == "esc":
                return None
            elif active_tab == 0:
                if key == "up":
                    orch_cursor = (orch_cursor - 1) % n_models
                elif key == "down":
                    orch_cursor = (orch_cursor + 1) % n_models
                elif key == "left":
                    orch_efforts[orch_cursor] = max(0, orch_efforts[orch_cursor] - 1)
                elif key == "right":
                    orch_efforts[orch_cursor] = min(len(_EFFORT_LEVELS) - 1, orch_efforts[orch_cursor] + 1)
            else:
                if sub_same:
                    if key == "down":
                        sub_same = False
                else:
                    if key == "up":
                        if sub_cursor == 0:
                            sub_same = True
                        else:
                            sub_cursor = sub_cursor - 1
                    elif key == "down":
                        sub_cursor = (sub_cursor + 1) % n_models
                    elif key == "left":
                        sub_efforts[sub_cursor] = max(0, sub_efforts[sub_cursor] - 1)
                    elif key == "right":
                        sub_efforts[sub_cursor] = min(len(_EFFORT_LEVELS) - 1, sub_efforts[sub_cursor] + 1)
                    elif key == "home":
                        sub_same = True
    finally:
        # Erase the picker on exit (Enter, Esc, or exception)
        if prev_line_count > 0:
            sys.stdout.write(f"\033[{prev_line_count}A")
            for _ in range(prev_line_count):
                sys.stdout.write("\033[2K\n")
            sys.stdout.write(f"\033[{prev_line_count}A")
            sys.stdout.flush()


def _run_connection_picker(
    console: Console,
    current_connection: str,
) -> str | None:
    """Interactive connection picker.

    Returns the chosen connection name, or ``None`` on cancel.
    """
    connections = load_snowflake_connections()
    if not connections:
        console.print("[bold red]No connections found in ~/.snowflake/connections.toml[/bold red]")
        return None
    names = list(connections.keys())
    n = len(names)

    def _find_current_idx() -> int:
        for i, name in enumerate(names):
            if name == current_connection:
                return i
        return 0

    cursor = _find_current_idx()
    _FIXED_HEIGHT = 4 + n + 1

    def _render() -> list[str]:
        lines: list[str] = [
            "",
            "  \033[1mConnection selector\033[0m  \033[2m\u2191\u2193: navigate  Enter: confirm  Esc: cancel\033[0m",
            "",
        ]
        for i, name in enumerate(names):
            selected = i == cursor
            pointer = "\033[1;36m>\033[0m" if selected else " "
            label = f"\033[1m{name}\033[0m" if selected else name
            current_marker = "  \033[2m(current)\033[0m" if name == current_connection else ""
            # Show account/host hint from connection params
            params = connections[name]
            hint_parts = []
            if "account" in params:
                hint_parts.append(params["account"])
            elif "host" in params:
                hint_parts.append(params["host"])
            if "database" in params:
                hint_parts.append(params["database"])
            hint = f"  \033[2m({', '.join(hint_parts)})\033[0m" if hint_parts else ""
            lines.append(f"  {pointer} {label}{current_marker}{hint}")
        lines.append("")
        while len(lines) < _FIXED_HEIGHT:
            lines.append("")
        return lines

    prev_line_count = 0
    try:
        while True:
            rendered = _render()
            if prev_line_count > 0:
                sys.stdout.write(f"\033[{prev_line_count}A")
                for _ in range(prev_line_count):
                    sys.stdout.write("\033[2K\n")
                sys.stdout.write(f"\033[{prev_line_count}A")
            for line in rendered:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()
            prev_line_count = len(rendered)

            key = _read_key()
            if key == "enter":
                return names[cursor]
            elif key == "esc":
                return None
            elif key == "up":
                cursor = (cursor - 1) % n
            elif key == "down":
                cursor = (cursor + 1) % n
    finally:
        if prev_line_count > 0:
            sys.stdout.write(f"\033[{prev_line_count}A")
            for _ in range(prev_line_count):
                sys.stdout.write("\033[2K\n")
            sys.stdout.write(f"\033[{prev_line_count}A")
            sys.stdout.flush()


def _build_status_panel(
    config: ArcticswarmConfig,
    session_id: str,
    repl_state: _ReplState | None = None,
) -> Panel:
    """Build the status panel showing current model, effort, and connection info."""
    orch_info = get_model_info(config.model)
    orch_name = orch_info.display_name if orch_info else config.model
    orch_effort = config.reasoning_effort or "none"

    # Orchestrator line — always shown
    model_line = f"Orchestrator: [cyan]{orch_name}[/cyan] [dim]({orch_effort})[/dim]"

    # Subagent line — always shown
    sub_model_id = config.subagent_model or config.model
    sub_effort = (config.subagent_reasoning_effort if config.subagent_reasoning_effort is not None else config.reasoning_effort) or "none"
    sub_info = get_model_info(sub_model_id)
    sub_name = sub_info.display_name if sub_info else sub_model_id
    model_line += f"  |  Subagents: [cyan]{sub_name}[/cyan] [dim]({sub_effort})[/dim]"

    parts = [
        f"v{__version__}  —  a research agent\n",
        model_line,
    ]
    if config.sf_params:
        parts.append(f"  |  Snowflake: [cyan]{config.sf_connection_name}[/cyan]")
    if repl_state is not None and repl_state.swarm_enabled:
        parts.append(f"  |  [bold magenta]Swarm: ON ({config.max_teammates} teammates, dynamic)[/bold magenta]")
    parts.append(f"\n[dim]Session: {session_id}[/dim]")
    parts.append("\n[dim]Type /help for commands. Press Enter twice to submit.[/dim]")

    return Panel("".join(parts), border_style="blue")


_LOGO = r"""
 ____  __ _   __   _  _  ____  _  _   __   ____  _  _
/ ___)(  ( \ /  \ / )( \/ ___)/ )( \ / _\ (  _ \( \/ )
\___ \/    /(  O )\ /\ /\___ \\ /\ //    \ )   // \/ \
(____/\_)__) \__/ (_/\_)(____/(_/\_)\_/\_/(__\_)\_)(_/
"""


def _redraw_banner(
    console: Console,
    config: ArcticswarmConfig,
    session_id: str,
    repl_state: _ReplState | None = None,
) -> None:
    """Clear the screen and redraw the logo + status panel."""
    console.clear()
    console.print(f"[bold cyan]{_LOGO}[/bold cyan]", highlight=False)
    console.print(_build_status_panel(config, session_id, repl_state))


def _handle_slash(
    command: str,
    agent: Agent,
    console: Console,
    session_id: str = "",
    created_at: str = "",
    repl_state: "_ReplState | None" = None,
) -> bool:
    """Handle slash commands. Returns True if the input was a slash command."""
    parts = command.strip().split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        if agent.messages:
            _save_session(session_id, agent, agent.config, created_at)
        _print_session_goodbye(console, session_id)
        sys.exit(0)

    if cmd == "/help":
        console.print(Markdown(_HELP_TEXT))
        return True

    if cmd == "/clear":
        agent.clear_history()
        if repl_state is not None and repl_state.orchestrator is not None:
            repl_state.orchestrator.reset()
            console.print("[dim]Conversation history and swarm state cleared.[/dim]")
        else:
            console.print("[dim]Conversation history cleared.[/dim]")
        return True

    if cmd == "/model":
        if arg:
            # Direct shortcut: /model <name>
            info = get_model_info(arg)
            if info is None:
                console.print(f"[bold red]Unknown model:[/bold red] {arg}")
                console.print("[dim]Available models:[/dim]")
                for m in AVAILABLE_MODELS:
                    console.print(f"  [dim]{m.id}[/dim]")
            else:
                agent.config.model = arg
                _redraw_banner(console, agent.config, session_id, repl_state)
        else:
            result = _run_model_picker(
                console,
                agent.config.model,
                agent.config.reasoning_effort,
                agent.config.subagent_model,
                agent.config.subagent_reasoning_effort,
            )
            if result is not None:
                agent.config.model = result.model
                agent.config.reasoning_effort = result.effort
                agent.config.subagent_model = result.subagent_model
                agent.config.subagent_reasoning_effort = result.subagent_effort
                _redraw_banner(console, agent.config, session_id, repl_state)
            else:
                console.print("[dim]Cancelled.[/dim]")
        return True

    if cmd == "/connection":
        if arg:
            # Direct shortcut: /connection <name>
            try:
                sf_params = get_snowflake_connection_params(arg)
                agent.switch_connection(arg, sf_params)
                _redraw_banner(console, agent.config, session_id, repl_state)
            except KeyError as exc:
                console.print(f"[bold red]{exc}[/bold red]")
        else:
            chosen = _run_connection_picker(console, agent.config.sf_connection_name)
            if chosen is not None and chosen != agent.config.sf_connection_name:
                try:
                    sf_params = get_snowflake_connection_params(chosen)
                    agent.switch_connection(chosen, sf_params)
                    _redraw_banner(console, agent.config, session_id, repl_state)
                except KeyError as exc:
                    console.print(f"[bold red]{exc}[/bold red]")
            elif chosen is None:
                console.print("[dim]Cancelled.[/dim]")
        return True

    if cmd == "/config":
        info = {
            "settings_file": settings_json_path(),
            "model": agent.config.model,
            "reasoning_effort": agent.config.reasoning_effort,
            "base_url": agent.config.base_url,
            "max_tokens": agent.config.max_tokens,
            "max_turns": agent.config.max_turns,
            "snowflake_connection": agent.config.sf_connection_name,
            "swarm_mode": repl_state.swarm_enabled if repl_state else False,
        }
        console.print(Panel(
            "\n".join(f"  {k}: {v}" for k, v in info.items()),
            title="Arcticswarm Configuration",
            border_style="blue",
        ))
        return True

    if cmd == "/swarm" and repl_state is not None:
        if arg.lower() in ("on", "true", "1"):
            repl_state.swarm_enabled = True
            console.print("[dim]Swarm mode enabled. Questions will use parallel agent teams.[/dim]")
        elif arg.lower() in ("off", "false", "0"):
            repl_state.swarm_enabled = False
            console.print("[dim]Swarm mode disabled. Questions use a single agent.[/dim]")
        else:
            status = "on" if repl_state.swarm_enabled else "off"
            console.print(f"[dim]Swarm mode:[/dim] {status}")
            console.print("[dim]Usage: /swarm on  or  /swarm off[/dim]")
        return True

    if cmd == "/save" and repl_state is not None:
        if not repl_state.last_response.strip():
            console.print("[dim]Nothing to save. Run a question first.[/dim]")
            return True
        if arg.strip():
            save_path = Path(arg.strip()).expanduser()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(f"arcticswarm_{ts}.md")
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(repl_state.last_response)
            console.print(f"[dim]Saved to:[/dim] [bold]{save_path}[/bold]")
        except OSError as exc:
            console.print(f"[bold red]Error saving file:[/bold red] {exc}")
        return True

    if cmd == "/bbs" and repl_state is not None:
        from arcticswarm.swarm.viewer import render_bbs
        if repl_state.last_bbs is None:
            console.print("[dim]No BBS data yet. Run a question in swarm mode first.[/dim]")
        else:
            channel = arg.strip() if arg.strip() else None
            render_bbs(repl_state.last_bbs, console, channel=channel)
        return True

    return False


# ---------------------------------------------------------------------------
# REPL rendering
# ---------------------------------------------------------------------------


class _ReplRenderer:
    """Collects stream events and renders them via Rich."""

    def __init__(self, console: Console, streaming: bool = True) -> None:
        self.console = console
        self.streaming = streaming
        self._text_buffer = ""
        self._live: Live | None = None

    def start(self) -> None:
        if self.streaming:
            self._text_buffer = ""
            self._live = Live(
                Markdown(""),
                console=self.console,
                refresh_per_second=15,
                vertical_overflow="ellipsis",
            )
            self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            # Clear the live region before stopping to prevent leftover
            # fragments in the terminal scrollback.  Then print the full
            # content as a static element so the scrollback contains
            # exactly one clean copy of the response.
            self._live.update(Text(""))
            self._live.stop()
            self._live = None
            if self._text_buffer.strip():
                self.console.print(Markdown(self._text_buffer.strip()))
        elif self._text_buffer.strip():
            # Non-streaming path: just print once.
            self.console.print()
            self.console.print(Markdown(self._text_buffer.strip()))

    def on_event(self, event: StreamEvent) -> None:
        if isinstance(event, TextDelta):
            self._text_buffer += event.text
            if self._live is not None:
                # Render markdown in-place; Live will refresh the region.
                self._live.update(Markdown(self._text_buffer))

        elif isinstance(event, ToolCallStart):
            if self._live is not None:
                # Stop the live region cleanly before tool output.  Print
                # accumulated text as a static element to avoid scrollback
                # duplication artefacts.
                self._live.update(Text(""))
                self._live.stop()
                self._live = None
                if self._text_buffer.strip():
                    self.console.print(Markdown(self._text_buffer.strip()))
                self._text_buffer = ""
            self.console.print()
            _input_summary = _summarize_tool_input(event.tool_name, event.tool_input)
            self.console.print(
                f"  [bold cyan]{event.tool_name}[/bold cyan] [dim]{_input_summary}[/dim]",
            )

        elif isinstance(event, ToolCallEnd):
            if event.result and event.result.is_error:
                self.console.print(f"  [red]Error:[/red] {rich_escape(event.result.error or '')}")
            elif event.result:
                preview = _smart_preview(event.result.output, event.tool_name)
                # Use Text() so Rich treats the preview as literal content —
                # no markup interpretation, no highlight mangling of numbers,
                # and whitespace (table alignment padding) is preserved exactly.
                self.console.print(Text(f"  {preview}", style="dim"))
            self.console.print()
            # Restart live for subsequent text
            if self.streaming:
                self._text_buffer = ""
                self._live = Live(
                    Markdown(""),
                    console=self.console,
                    refresh_per_second=15,
                    vertical_overflow="ellipsis",
                )
                self._live.start()

        elif isinstance(event, TurnComplete):
            pass  # handled by stop()


def _smart_preview(output: str, tool_name: str, max_lines: int = 10) -> str:
    """Build a line-based preview of a tool result (first N lines)."""
    if not output:
        return "(no output)"
    return _preview_by_lines(output.split("\n"), max_lines)


def _preview_by_lines(lines: list[str], max_lines: int) -> str:
    """Generic line-based preview: first N non-empty logical lines."""
    if len(lines) <= max_lines:
        return "\n".join(lines)

    shown = lines[:max_lines]
    remaining = len(lines) - max_lines
    return "\n".join(shown) + f"\n... ({remaining} more lines)"


def _summarize_tool_input(tool_name: str, tool_input: dict[str, Any]) -> str:
    """One-line summary of a tool call's input."""
    if tool_name == "bash":
        return tool_input.get("command", "")[:80]
    if tool_name in ("read_file", "edit_file"):
        return tool_input.get("file_path", "")
    # Fallback: compact JSON
    try:
        return json.dumps(tool_input, separators=(",", ":"))[:80]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Multiline input helper
# ---------------------------------------------------------------------------


def _read_multiline(session: PromptSession) -> str:
    """Read input, supporting multiline paste. Blank line submits."""
    lines: list[str] = []
    first = True
    while True:
        try:
            prompt = [("bold fg:ansicyan", "arcticswarm> ")] if first else [("", "       ... ")]
            line = session.prompt(prompt)
            first = False
        except EOFError:
            if lines:
                break
            raise
        except KeyboardInterrupt:
            return ""

        # A blank line after content submits
        if line == "" and lines:
            break
        lines.append(line)

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

_HISTORY_DIR = Path.home() / ".arcticswarm" / "history"


def _serialize_content(content: Any) -> Any:
    """Convert Anthropic Pydantic content blocks to JSON-native dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [b.model_dump() if hasattr(b, "model_dump") else b for b in content]
    return content


def _save_session(
    session_id: str,
    agent: Agent,
    config: ArcticswarmConfig,
    created_at: str,
) -> Path:
    """Persist the current conversation to disk. Returns the file path."""
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _HISTORY_DIR / f"{session_id}.json"

    data = {
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "messages": [
            {"role": m["role"], "content": _serialize_content(m["content"])}
            for m in agent.messages
        ],
    }

    # Atomic write: write to a tmp file then rename
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, default=str))
    tmp_path.rename(path)
    return path


def _load_session(session_id: str) -> dict[str, Any]:
    """Load a saved session. Raises FileNotFoundError if not found."""
    path = _HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No session file found at {path}")
    with open(path) as f:
        return json.load(f)


def _replay_history(messages: list[dict[str, Any]], console: Console) -> None:
    """Render saved conversation history so the user sees prior context.

    Displays user questions, assistant text (as Markdown), and tool-call
    summaries in the same style the REPL uses during live execution.
    """
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            # User messages are either a plain string (typed question) or a
            # list of tool_result blocks (internal plumbing — skip those).
            if isinstance(content, str):
                console.print(f"\n[bold cyan]arcticswarm>[/bold cyan] {rich_escape(content)}")
            # list content = tool results → nothing to show the user
            continue

        if role == "assistant":
            # Content may be a plain string or a list of text / tool_use blocks.
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str):
                blocks = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # Could be Pydantic model objects (if not yet serialised) or dicts
                blocks = [
                    b.model_dump() if hasattr(b, "model_dump") else b
                    for b in content
                ]

            text_parts: list[str] = []
            for block in blocks:
                btype = block.get("type", "")

                if btype == "text":
                    text = block.get("text", "")
                    if text.strip():
                        text_parts.append(text)

                elif btype == "tool_use":
                    # Flush any accumulated text before showing tool call
                    if text_parts:
                        console.print(Markdown("\n".join(text_parts)))
                        text_parts.clear()
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})
                    summary = _summarize_tool_input(tool_name, tool_input)
                    console.print(
                        f"\n  [bold cyan]{tool_name}[/bold cyan] [dim]{summary}[/dim]",
                    )

            # Flush remaining text
            if text_parts:
                console.print(Markdown("\n".join(text_parts)))

    # Visual separator before new input
    if messages:
        console.print()
        console.print("[dim]─── end of history ───[/dim]")
        console.print()


def _extract_final_message_text(messages: list[dict[str, Any]]) -> str:
    """Extract text content from the last assistant message.

    Walks the conversation history in reverse and returns the text of the
    most recent assistant message (the final answer), ignoring tool-use
    blocks and intermediate reasoning.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if hasattr(block, "type") and block.type == "text":
                    text_parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "".join(text_parts)
    return ""


def _print_session_goodbye(console: Console, session_id: str) -> None:
    """Print session ID and resume hint on exit."""
    console.print(f"\n[dim]Session:[/dim] [bold]{session_id}[/bold]")
    console.print(f"[dim]To resume:[/dim] arcticswarm --session {session_id}")


def _format_usage_line(usage: TokenUsage, model_info: ModelInfo | None = None) -> str:
    """Format a single TokenUsage into a compact string."""
    line = (
        f"{usage.input_tokens:,} in + {usage.output_tokens:,} out"
        f" = {usage.total_tokens:,} total"
    )
    if usage.reasoning_tokens > 0:
        line += f" ({usage.reasoning_tokens:,} reasoning)"
    line += f" | cache: {usage.cache_read_input_tokens:,} read, {usage.cache_creation_input_tokens:,} write"
    if model_info is not None:
        cost = usage.cost(model_info)
        line += f" | ${cost:.2f}"
    return line


def _print_token_usage(
    console: Console,
    usage: TokenUsage,
    breakdown: dict[str, TokenUsage] | None = None,
    model_info: ModelInfo | None = None,
) -> None:
    """Print a token usage summary, optionally with per-agent breakdown."""
    console.print(f"[dim]Tokens: {_format_usage_line(usage, model_info)}[/dim]")
    if breakdown:
        for name, agent_usage in breakdown.items():
            console.print(f"[dim]  {name:20s} {_format_usage_line(agent_usage, model_info)}[/dim]")


# ---------------------------------------------------------------------------
# Swarm execution helper
# ---------------------------------------------------------------------------


def _run_swarm_turn(
    user_input: str,
    config: ArcticswarmConfig,
    console: Console,
    streaming: bool,
    repl_state: _ReplState,
    prompt_session: PromptSession | None = None,
    show_usage: bool = False,
) -> None:
    """Execute a single turn through the swarm orchestrator.

    The orchestrator runs a dynamic agentic loop — all activity is shown
    in the Swarm Live panel.  When the orchestrator finishes, the panel
    stops and the final answer is printed directly.
    """
    from arcticswarm.swarm.orchestrator import OrchestratorToolCall, SwarmComplete
    from arcticswarm.swarm.viewer import SwarmRenderer

    orchestrator = repl_state.get_orchestrator(config)
    swarm_renderer = SwarmRenderer(console)

    _warmup_spinner: Status | None = None
    _report_spinner: Status | None = None

    def _on_swarm_event_wrapper(event: object) -> None:
        """Forward swarm events to the renderer, managing lifecycle spinners."""
        nonlocal _warmup_spinner, _report_spinner
        if _warmup_spinner is not None:
            _warmup_spinner.stop()
            _warmup_spinner = None
        swarm_renderer.on_swarm_event(event)
        if (isinstance(event, OrchestratorToolCall)
                and event.tool_name == "prepare_report"
                and "All tasks done" in event.description):
            swarm_renderer.stop()
            _report_spinner = Status("Generating final report...", console=console)
            _report_spinner.start()
        elif isinstance(event, SwarmComplete) and _report_spinner is not None:
            _report_spinner.stop()
            _report_spinner = None

    # Arm the renderer (phase 1 — rolling console output).
    # The full panel display activates on SwarmStarted.
    swarm_renderer.start()

    try:
        answer = orchestrator.run_swarm_turn(
            user_input,
            on_swarm_event=_on_swarm_event_wrapper,
        )
        # Capture the BBS for /bbs inspection
        repl_state.last_bbs = orchestrator.last_bbs

        # Store the response for /save
        repl_state.last_response = answer

        # The viewer has already saved and opened the HTML report in the
        # SwarmComplete event handler.  Check if SendReportTool appended
        # extra content (e.g. a ## References footer) after the streaming
        # finished and show the delta to the user.
        if answer and swarm_renderer.streamed_report and answer != swarm_renderer.streamed_report:
            extra = answer[len(swarm_renderer.streamed_report):]
            if extra.strip():
                console.print(Markdown(extra.strip()))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as exc:
        import traceback
        logger.debug("Swarm error traceback:\n%s", traceback.format_exc())
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
    finally:
        if _warmup_spinner is not None:
            _warmup_spinner.stop()
        if _report_spinner is not None:
            _report_spinner.stop()
        swarm_renderer.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    explicit = _explicit_dests(parser, sys.argv[1:])

    # Build base config: from YAML when --config is given, otherwise the
    # standard resolve() flow. CLI flags below are applied as overrides
    # only when explicitly passed (so YAML values aren't clobbered by
    # argparse defaults like ``store_true`` → False).
    if args.config:
        from arcticswarm.run_config import load_run_config

        try:
            run_cfg = load_run_config(args.config, args.overrides or [])
        except (FileNotFoundError, ValueError) as exc:
            Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)
        config = run_cfg.to_arcticswarm_config()
    else:
        if args.overrides:
            Console(stderr=True).print(
                "[bold red]Error:[/bold red] dot-notation overrides "
                f"({', '.join(args.overrides)}) require --config."
            )
            sys.exit(1)
        config = ArcticswarmConfig.resolve()

    if "model" in explicit and args.model:
        config.model = args.model
    if "swarm" in explicit:
        config.swarm_enabled = args.swarm
    if "web_search" in explicit:
        config.web_search_enabled = args.web_search
    if "web_provider" in explicit:
        config.web_search_provider = args.web_provider
    if "web_fetch_backend" in explicit:
        config.web_fetch_backend = args.web_fetch_backend
    if getattr(args, "corpus_backend", None):
        config.corpus_backend = args.corpus_backend
    if "enable_vision" in explicit:
        config.enable_vision = args.enable_vision
    if "num_teammates" in explicit:
        config.max_teammates = args.num_teammates
    if "max_subagents" in explicit:
        config.max_subagents = args.max_subagents
    if "max_subagent_tasks" in explicit:
        config.max_subagent_tasks = args.max_subagent_tasks
    if "swarm_comm" in explicit:
        config.swarm_comm = args.swarm_comm
    if "disable_source_scorer" in explicit:
        config.disable_source_scorer = args.disable_source_scorer
    if "use_fetch_compactor" in explicit:
        config.use_fetch_compactor = args.use_fetch_compactor
    if "use_pdf_compactor" in explicit:
        config.use_pdf_compactor = args.use_pdf_compactor
    if args.context_compact_tokens:
        config.context_compact_tokens = args.context_compact_tokens
    if "disable_self_reflection" in explicit:
        config.disable_self_reflection = args.disable_self_reflection
    if args.enable_1m_context:
        config.enable_1m_context_model = True
    if args.max_tool_calls_per_turn:
        config.max_tool_calls_per_turn = args.max_tool_calls_per_turn
    show_usage = args.usage

    if args.max_turns is not None:
        config.max_turns = args.max_turns

    if not config.api_key:
        Console(stderr=True).print(
            "[bold red]Error:[/bold red] api_key is not set. "
            f"Add it to {settings_json_path()} "
            "(or set ARCTICSWARM_SETTINGS_PATH), e.g.:\n"
            '  {"api_key": "sk-ant-..."}'
        )
        sys.exit(1)

    # Probe configured Brave/Serper/Tavily/Jina keys before starting so a
    # broken billing/quota (e.g. "credit card required") fails fast rather
    # than silently degrading the run.
    try:
        check_search_api_health(config)
    except SearchApiUnhealthyError as exc:
        Console(stderr=True).print(
            f"[bold red]API health check failed:[/bold red] {exc}"
        )
        sys.exit(2)

    console = Console()

    # Session ID
    session_id = args.session or uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    agent = Agent(config)

    # Swarm state
    repl_state = _ReplState(swarm_enabled=config.swarm_enabled)

    # Resume previous session if requested
    if args.session:
        try:
            saved = _load_session(args.session)
            agent.messages = saved.get("messages", [])
            created_at = saved.get("created_at", created_at)
            console.print(f"[dim]Resumed session [bold]{session_id}[/bold] ({len(agent.messages)} messages)[/dim]")
            _replay_history(agent.messages, console)
        except FileNotFoundError as exc:
            Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)

    # Banner
    _redraw_banner(console, config, session_id, repl_state)

    streaming = not args.no_stream
    # Persist prompt history to disk so up-arrow recalls previous sessions.
    _PROMPT_HISTORY_PATH = Path.home() / ".arcticswarm" / "prompt_history"
    _PROMPT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(_PROMPT_HISTORY_PATH)),
        completer=_SlashCompleter(),
        complete_while_typing=True,
    )

    while True:
        try:
            user_input = _read_multiline(prompt_session)
        except (EOFError, KeyboardInterrupt):
            if agent.messages:
                _save_session(session_id, agent, config, created_at)
            _print_session_goodbye(console, session_id)
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            if _handle_slash(user_input, agent, console, session_id, created_at, repl_state):
                continue

        # Swarm mode path
        if repl_state.swarm_enabled:
            _run_swarm_turn(user_input, config, console, streaming, repl_state, prompt_session, show_usage=show_usage)
            # Show token usage if enabled (with per-agent breakdown)
            if show_usage:
                orchestrator = repl_state.orchestrator
                if orchestrator is not None and orchestrator.last_token_usage is not None:
                    _print_token_usage(
                        console,
                        orchestrator.last_token_usage,
                        breakdown=orchestrator.last_token_usage_breakdown,
                        model_info=get_model_info(config.model),
                    )
            # Save the orchestrator's conversation for session persistence
            orchestrator = repl_state.orchestrator
            if orchestrator is not None and orchestrator._orchestrator_agent is not None:
                orch_agent = orchestrator._orchestrator_agent
                if orch_agent.messages:
                    _save_session(session_id, orch_agent, config, created_at)
            continue

        # Standard single-agent path
        renderer = _ReplRenderer(console, streaming=streaming)
        renderer.start()
        try:
            if streaming:
                agent.run_turn_streaming(user_input, on_event=renderer.on_event)
            else:
                text = agent.run_turn(user_input, on_event=renderer.on_event)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
        except Exception as exc:
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
        finally:
            renderer.stop()

        # Show token usage if enabled
        if show_usage and agent.last_turn_usage.total_tokens > 0:
            _print_token_usage(console, agent.last_turn_usage,
                               model_info=get_model_info(config.model))

        # Store the last response for /save
        repl_state.last_response = _extract_final_message_text(agent.messages)

        # Save after every turn so nothing is lost on crash
        if agent.messages:
            _save_session(session_id, agent, config, created_at)


if __name__ == "__main__":
    main()
