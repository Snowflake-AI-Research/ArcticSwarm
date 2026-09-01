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

"""Interactive tool usage breakdown analyzer for arcticswarm eval results.

Analyzes tool call patterns from eval trajectories: shows per-tool counts,
success/failure correlation with judge scores, and lets you interactively
browse real input/output of each tool call.

Usage::

    python -m arcticswarm.eval.analyze_breakdown --results-dir results/
    python -m arcticswarm.eval.analyze_breakdown --results-dir results/ --tool execute_python
    python -m arcticswarm.eval.analyze_breakdown --results-dir results/ --tool web_search --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool call extracted from a trajectory."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    is_error: bool
    # Context
    conv_id: str  # which eval case this came from
    trajectory_file: str  # path to the trajectory JSON
    message_index: int  # index of the assistant message in the trajectory


@dataclass
class CaseInfo:
    """Summary of an eval case from report.json."""

    conv_id: str
    score: float  # judge score
    tools_used: list[str]
    question: str = ""
    dataset: str = ""
    num_tool_calls: int = 0
    judge_correct: bool = False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _extract_tool_calls_from_messages(
    messages: list[dict[str, Any]],
    conv_id: str,
    trajectory_file: str,
) -> list[ToolCall]:
    """Extract tool calls from a flat list of messages."""
    calls: list[ToolCall] = []

    # Build a map: tool_use_id -> (tool_name, tool_input, msg_index)
    pending: dict[str, tuple[str, dict[str, Any], int]] = {}

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tid = block.get("id", "")
                    pending[tid] = (
                        block.get("name", ""),
                        block.get("input", {}),
                        i,
                    )

        elif role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    if tid in pending:
                        name, inp, msg_idx = pending.pop(tid)
                        # Extract output text
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            parts = []
                            for rc in result_content:
                                if isinstance(rc, dict):
                                    parts.append(rc.get("text", ""))
                                elif isinstance(rc, str):
                                    parts.append(rc)
                            output_text = "\n".join(parts)
                        elif isinstance(result_content, str):
                            output_text = result_content
                        else:
                            output_text = str(result_content)

                        calls.append(ToolCall(
                            tool_name=name,
                            tool_input=inp,
                            tool_output=output_text,
                            is_error=block.get("is_error", False),
                            conv_id=conv_id,
                            trajectory_file=trajectory_file,
                            message_index=msg_idx,
                        ))

    return calls


def _extract_tool_calls_from_trajectory(
    traj_data: Any,
    conv_id: str,
    trajectory_file: str,
) -> list[ToolCall]:
    """Extract tool calls from a trajectory JSON (handles all formats)."""
    # Handle wrapped format: {phase_timings: ..., trajectory: [...]}
    if isinstance(traj_data, dict) and "trajectory" in traj_data:
        traj_data = traj_data["trajectory"]

    if not isinstance(traj_data, list) or not traj_data:
        return []

    first = traj_data[0]

    # Format 1: flat list of messages [{role, content}, ...]
    if isinstance(first, dict) and "role" in first:
        return _extract_tool_calls_from_messages(traj_data, conv_id, trajectory_file)

    # Format 2: wrapped [{system_prompt, tools, messages}, ...]
    if isinstance(first, dict) and "messages" in first:
        msgs = first.get("messages", [])
        return _extract_tool_calls_from_messages(msgs, conv_id, trajectory_file)

    # Format 3: swarm trajectory [{orchestrator: [...], subagents: [...], ...}]
    if isinstance(first, dict) and "orchestrator" in first:
        calls: list[ToolCall] = []
        entry = first
        # Orchestrator messages
        orch_msgs = entry.get("orchestrator", [])
        calls.extend(_extract_tool_calls_from_messages(
            orch_msgs, conv_id, trajectory_file,
        ))
        # Subagent messages
        for sa in entry.get("subagents", []):
            sa_msgs = sa.get("messages", [])
            sa_name = sa.get("name", "subagent")
            calls.extend(_extract_tool_calls_from_messages(
                sa_msgs, f"{conv_id}:{sa_name}", trajectory_file,
            ))
        return calls

    return []


def load_results(results_dir: Path) -> tuple[list[CaseInfo], list[ToolCall]]:
    """Load case info from report.json and tool calls from trajectories."""
    report_path = results_dir / "report.json"
    if not report_path.exists():
        print(f"Error: {report_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    # Handle multi-run reports
    if "runs" in report:
        per_case_list = report["runs"][0].get("per_case", [])
    else:
        per_case_list = report.get("per_case", [])

    # Build case info
    cases: list[CaseInfo] = []
    for pc in per_case_list:
        cid = pc.get("conv_id", "")
        score = pc.get("score", 0.0)
        judge_correct = pc.get("judge_correct")
        case = CaseInfo(
            conv_id=cid,
            score=score,
            tools_used=pc.get("tools_used", []),
            question=pc.get("question", "")[:200],
            dataset=pc.get("dataset", ""),
            num_tool_calls=pc.get("num_tool_calls", 0),
            judge_correct=judge_correct if judge_correct is not None else (score > 0),
        )
        cases.append(case)

    # Load trajectories
    traj_dir = results_dir / "trajectories"
    all_calls: list[ToolCall] = []

    if traj_dir.exists():
        for traj_file in sorted(traj_dir.glob("*.json")):
            conv_id = traj_file.stem
            try:
                with open(traj_file) as f:
                    traj_data = json.load(f)
                calls = _extract_tool_calls_from_trajectory(
                    traj_data, conv_id, str(traj_file),
                )
                all_calls.extend(calls)
            except Exception as e:
                print(f"  Warning: failed to parse {traj_file.name}: {e}", file=sys.stderr)

    return cases, all_calls


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def print_tool_headcount(
    cases: list[CaseInfo],
    all_calls: list[ToolCall],
) -> None:
    """Print a compact one-line-per-tool headcount summary.

    For each unique tool that appeared in any trajectory, shows:
    - How many cases used it
    - Total call count
    - Accuracy when tool was used vs not used
    """
    # Collect per-tool, per-case stats
    tool_case_calls: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tc in all_calls:
        base_conv_id = tc.conv_id.split(":")[0]
        tool_case_calls[tc.tool_name][base_conv_id] += 1

    case_success: dict[str, bool] = {}
    for c in cases:
        case_success[c.conv_id] = c.judge_correct

    total_cases = len(cases)
    total_correct = sum(1 for c in cases if c.judge_correct)

    print(f"\n{'=' * 95}")
    print(f" Tool Headcount Summary  ({total_correct}/{total_cases} cases correct overall)")
    print(f"{'=' * 95}")
    print(f" {'Tool':<28s} {'Cases':>5s} {'Calls':>6s} "
          f"{'Acc(used)':>18s} {'Acc(unused)':>18s} {'Avg/case':>8s}")
    print(f" {'-' * 27} {'-' * 5} {'-' * 6} {'-' * 18} {'-' * 18} {'-' * 8}")

    sorted_tools = sorted(
        tool_case_calls.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for tool_name, case_map in sorted_tools:
        n_cases = len(case_map)
        total_tool_calls = sum(case_map.values())
        avg_per_case = total_tool_calls / n_cases if n_cases else 0

        # Accuracy when tool was used
        used_ids = set(case_map.keys())
        used_correct = sum(1 for cid in used_ids if case_success.get(cid, False))
        used_acc = (used_correct / n_cases * 100) if n_cases else 0
        used_str = f"{used_acc:5.1f}% ({used_correct:>3d}/{n_cases:<3d})"

        # Accuracy when tool was NOT used
        unused_ids = set(case_success.keys()) - used_ids
        n_unused = len(unused_ids)
        unused_correct = sum(1 for cid in unused_ids if case_success.get(cid, False))
        unused_acc = (unused_correct / n_unused * 100) if n_unused else 0
        unused_str = f"{unused_acc:5.1f}% ({unused_correct:>3d}/{n_unused:<3d})" if n_unused else f"{'N/A':>18s}"

        print(
            f" {tool_name:<28s} {n_cases:>5d} {total_tool_calls:>6d} "
            f"{used_str:>18s} {unused_str:>18s} {avg_per_case:>8.1f}"
        )

    print()


# ---------------------------------------------------------------------------
# Analysis (detailed)
# ---------------------------------------------------------------------------


def print_overview(
    cases: list[CaseInfo],
    all_calls: list[ToolCall],
) -> None:
    """Print an overview of tool usage across all cases."""
    total_calls = len(all_calls)
    tool_counts: dict[str, int] = defaultdict(int)
    tool_error_counts: dict[str, int] = defaultdict(int)
    tool_cases: dict[str, set[str]] = defaultdict(set)

    for tc in all_calls:
        tool_counts[tc.tool_name] += 1
        tool_cases[tc.tool_name].add(tc.conv_id)
        if tc.is_error:
            tool_error_counts[tc.tool_name] += 1

    # Build case success map
    case_success: dict[str, bool] = {}
    for c in cases:
        case_success[c.conv_id] = c.judge_correct

    # Tool success/fail correlation
    tool_in_correct: dict[str, int] = defaultdict(int)
    tool_in_incorrect: dict[str, int] = defaultdict(int)
    for tc in all_calls:
        base_conv_id = tc.conv_id.split(":")[0]  # strip subagent name for swarm
        if case_success.get(base_conv_id, False):
            tool_in_correct[tc.tool_name] += 1
        else:
            tool_in_incorrect[tc.tool_name] += 1

    print(f"\n{'=' * 80}")
    print(f" Tool Usage Breakdown")
    print(f"{'=' * 80}")
    print(f" Total eval cases:  {len(cases)}")
    print(f" Total tool calls:  {total_calls}")
    print(f" Cases correct:     {sum(1 for c in cases if c.judge_correct)}/{len(cases)}")
    print(f"{'=' * 80}\n")

    # Sort by count descending
    sorted_tools = sorted(tool_counts.items(), key=lambda x: -x[1])

    # Header
    print(f" {'Tool':<25s} {'Count':>7s} {'% Total':>8s} {'Cases':>6s} "
          f"{'Errors':>7s} {'In Correct':>11s} {'In Wrong':>9s}")
    print(f" {'-' * 24} {'-' * 7} {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 11} {'-' * 9}")

    for tool_name, count in sorted_tools:
        pct = (count / total_calls * 100) if total_calls > 0 else 0.0
        n_cases = len(tool_cases[tool_name])
        errors = tool_error_counts.get(tool_name, 0)
        in_correct = tool_in_correct.get(tool_name, 0)
        in_wrong = tool_in_incorrect.get(tool_name, 0)
        print(
            f" {tool_name:<25s} {count:>7d} {pct:>7.1f}% {n_cases:>6d} "
            f"{errors:>7d} {in_correct:>11d} {in_wrong:>9d}"
        )

    print()


def print_tool_detail(
    tool_name: str,
    all_calls: list[ToolCall],
    cases: list[CaseInfo],
) -> None:
    """Print details for a specific tool."""
    filtered = [tc for tc in all_calls if tc.tool_name == tool_name]
    if not filtered:
        print(f"No calls found for tool '{tool_name}'")
        return

    case_success: dict[str, bool] = {}
    for c in cases:
        case_success[c.conv_id] = c.judge_correct

    correct_calls = [tc for tc in filtered if case_success.get(tc.conv_id.split(":")[0], False)]
    wrong_calls = [tc for tc in filtered if not case_success.get(tc.conv_id.split(":")[0], False)]
    error_calls = [tc for tc in filtered if tc.is_error]

    print(f"\n{'=' * 80}")
    print(f" Tool Detail: {tool_name}")
    print(f"{'=' * 80}")
    print(f" Total calls:       {len(filtered)}")
    print(f" In correct cases:  {len(correct_calls)}")
    print(f" In wrong cases:    {len(wrong_calls)}")
    print(f" Error calls:       {len(error_calls)}")
    print(f"{'=' * 80}\n")


def interactive_browse(
    tool_name: str | None,
    all_calls: list[ToolCall],
    cases: list[CaseInfo],
) -> None:
    """Interactively browse tool calls with n/b/d/q navigation."""
    if tool_name:
        filtered = [tc for tc in all_calls if tc.tool_name == tool_name]
    else:
        filtered = list(all_calls)

    if not filtered:
        print(f"No tool calls found" + (f" for '{tool_name}'" if tool_name else ""))
        return

    case_success: dict[str, bool] = {}
    case_question: dict[str, str] = {}
    for c in cases:
        case_success[c.conv_id] = c.judge_correct
        case_question[c.conv_id] = c.question

    # Keep a reference to the original full list for filter resets
    base_calls = list(filtered)
    idx = 0

    while True:
        if not filtered:
            print("  No calls match the current filter.")
            break

        tc = filtered[idx]
        base_conv_id = tc.conv_id.split(":")[0]
        success = case_success.get(base_conv_id, False)
        success_label = "\033[32mCORRECT\033[0m" if success else "\033[31mWRONG\033[0m"

        separator = "-" * 80
        print(f"\n{separator}")
        print(f" [{idx + 1}/{len(filtered)}] Tool: {tc.tool_name}  |  "
              f"Case: {tc.conv_id}  |  {success_label}")
        print(f" Question: {case_question.get(base_conv_id, '?')[:120]}")
        print(f" Error: {tc.is_error}")
        print(separator)

        # Input
        print(f"\n\033[1mINPUT:\033[0m")
        try:
            input_str = json.dumps(tc.tool_input, indent=2, default=str)
        except Exception:
            input_str = str(tc.tool_input)
        if len(input_str) > 2000:
            print(input_str[:2000])
            print(f"... (truncated, {len(input_str)} chars total)")
        else:
            print(input_str)

        # Output
        print(f"\n\033[1mOUTPUT:\033[0m")
        output = tc.tool_output
        if len(output) > 3000:
            print(output[:3000])
            print(f"... (truncated, {len(output)} chars total)")
        else:
            print(output)

        # Navigation
        print(f"\n  [n] next  [b] back  [d] adjacent calls in same case  "
              f"[f] filter correct  [w] filter wrong  [a] show all  [q] quit")
        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd == "q":
            break
        elif cmd == "n":
            idx = min(idx + 1, len(filtered) - 1)
        elif cmd == "b":
            idx = max(idx - 1, 0)
        elif cmd == "d":
            # Show adjacent tool calls from the same case
            same_case = [
                (i, c) for i, c in enumerate(filtered)
                if c.conv_id == tc.conv_id
            ]
            print(f"\n  Tool calls in case '{tc.conv_id}':")
            for j, (fi, fc) in enumerate(same_case):
                marker = " >>>" if fi == idx else "    "
                err_tag = " [ERROR]" if fc.is_error else ""
                print(f"  {marker} [{j + 1}] {fc.tool_name}{err_tag}")
            try:
                choice = input("  Jump to # (or Enter to stay): ").strip()
                if choice.isdigit():
                    ci = int(choice) - 1
                    if 0 <= ci < len(same_case):
                        idx = same_case[ci][0]
            except (EOFError, KeyboardInterrupt):
                print()
                break
        elif cmd == "f":
            filtered = [
                tc for tc in base_calls
                if case_success.get(tc.conv_id.split(":")[0], False)
            ]
            idx = 0
            print(f"  Filtered to {len(filtered)} calls in correct cases")
        elif cmd == "w":
            filtered = [
                tc for tc in base_calls
                if not case_success.get(tc.conv_id.split(":")[0], False)
            ]
            idx = 0
            print(f"  Filtered to {len(filtered)} calls in wrong cases")
        elif cmd == "a":
            filtered = list(base_calls)
            idx = 0
            print(f"  Showing all {len(filtered)} calls")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_breakdown",
        description="Interactive tool usage breakdown analyzer for arcticswarm eval results",
    )
    p.add_argument(
        "--results-dir", "-r",
        required=True,
        help="Path to eval results directory (containing report.json and trajectories/)",
    )
    p.add_argument(
        "--tool", "-t",
        default=None,
        help="Filter to a specific tool name (e.g. execute_python, web_search)",
    )
    p.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=False,
        help="Enter interactive browsing mode (n-next, b-back, d-adjacent, q-quit)",
    )
    p.add_argument(
        "--tool-headcount", "-H",
        action="store_true",
        default=False,
        help="Show compact per-tool headcount: cases used, calls, accuracy when used vs unused",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from {results_dir} ...")
    cases, all_calls = load_results(results_dir)
    print(f"  Loaded {len(cases)} cases, {len(all_calls)} tool calls")

    # Headcount mode: compact summary only, then exit
    if args.tool_headcount:
        print_tool_headcount(cases, all_calls)
        return

    # Always print overview
    print_overview(cases, all_calls)

    # If a specific tool is specified, show detail
    if args.tool:
        print_tool_detail(args.tool, all_calls, cases)

    # Interactive mode
    if args.interactive or args.tool:
        interactive_browse(args.tool, all_calls, cases)


if __name__ == "__main__":
    main()
