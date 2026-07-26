"""Eval recovery, resume, and rebuild utilities.

Extracted from :mod:`arcticswarm.eval.cli` to keep the CLI module focused on
argument parsing, execution orchestration, and reporting.  Everything here
deals with loading/reconstructing :class:`EvalResult` objects from prior
runs — either from ``report.json`` or from raw trajectory files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from arcticswarm.eval.runner import EvalResult, ToolCallRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rejudge — load results for re-judging (strips judge verdicts)
# ---------------------------------------------------------------------------


def load_results_for_rejudge(
    rejudge_dir: Path,
) -> tuple[list[list[EvalResult]], bool]:
    """Load previous eval results from *rejudge_dir* for re-judging.

    Reconstructs :class:`EvalResult` objects from ``report.json`` per-case
    data and trajectory files.  Tool call records are rebuilt from trajectory
    JSON so that ``result.response_full`` works for full FLEX judging.

    Returns ``(grouped_results, is_swarm)`` where *is_swarm* is detected
    from the presence of ``swarm_teammates_spawned`` in per-case data.
    """
    from arcticswarm.eval.data_loader import EvalCase

    report_path = rejudge_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"No report.json found in {rejudge_dir}")

    with open(report_path) as f:
        report = json.load(f)

    # Detect single-run vs multi-run report
    if "runs" in report:
        runs_data = report["runs"]
    else:
        runs_data = [report]

    grouped: list[list[EvalResult]] = []
    for run_idx, run_data in enumerate(runs_data):
        per_case = run_data.get("per_case", [])
        results: list[EvalResult] = []

        # Determine trajectory directory
        if len(runs_data) > 1:
            traj_dir = rejudge_dir / f"run_{run_idx}" / "trajectories"
        else:
            traj_dir = rejudge_dir / "trajectories"

        for case_dict in per_case:
            conv_id = case_dict.get("conv_id", "")
            case = EvalCase(
                conv_id=conv_id,
                turn_index=0,
                question=case_dict.get("question", ""),
                reference_answer="",  # Will be resolved from CSV
                attributes={
                    "dataset": case_dict.get("dataset", ""),
                    "eval_mode": case_dict.get("eval_mode", "QA"),
                    "is_vip": True,
                },
            )

            result = EvalResult(
                case=case,
                response_text=case_dict.get("response_text", ""),
                tools_used=case_dict.get("tools_used", []),
                duration_seconds=case_dict.get("duration_seconds", 0.0),
                error=case_dict.get("error"),
            )

            # Rebuild tool_calls from trajectory for response_full
            safe_name = conv_id.replace("/", "_").replace("\\", "_")[:200]
            traj_path = traj_dir / f"{safe_name}.json"
            if traj_path.exists():
                try:
                    with open(traj_path) as f:
                        traj = json.load(f)
                    # Handle wrapped trajectory format (with phase_timings)
                    if isinstance(traj, dict) and "trajectory" in traj:
                        traj = traj["trajectory"]
                    if isinstance(traj, list):
                        for msg in traj:
                            if not isinstance(msg, dict):
                                continue
                            content = msg.get("content", [])
                            if not isinstance(content, list):
                                continue
                            for block in content:
                                if not isinstance(block, dict):
                                    continue
                                if block.get("type") == "tool_result":
                                    tc = ToolCallRecord(
                                        name=block.get("tool_name", ""),
                                        input={},
                                        output=str(block.get("content", ""))[:2000],
                                        is_error=block.get("is_error", False),
                                    )
                                    result.tool_calls.append(tc)
                except Exception:
                    pass

            results.append(result)
        grouped.append(results)

    # Detect swarm mode from report metadata (preferred), falling back to
    # heuristic (swarm_teammates_spawned) for reports written before this field.
    is_swarm = report.get("swarm_enabled", False)
    if not is_swarm:
        is_swarm = any(
            case_dict.get("swarm_teammates_spawned", 0) > 0
            for run_data in runs_data
            for case_dict in run_data.get("per_case", [])
        )

    return grouped, is_swarm


# ---------------------------------------------------------------------------
# Resume — load results with judge scores intact
# ---------------------------------------------------------------------------


def load_results_for_resume(
    resume_dir: Path,
) -> tuple[list[list[EvalResult]], bool]:
    """Load previous eval results from *resume_dir* with judge scores intact.

    Unlike :func:`load_results_for_rejudge` which strips judge results (so
    they can be re-computed), this function reconstructs the full
    :class:`EvalResult` including ``qa_result``, ``flex_result``, token usage,
    swarm metrics, etc.  The returned results are ready to be merged with new
    results without any additional judging.

    Source of truth is ``report.json`` — trajectory files that exist without a
    corresponding ``per_case`` entry are ignored (they will be re-run).

    Returns ``(grouped_results, is_swarm)``.
    """
    from arcticswarm.agent import TokenUsage
    from arcticswarm.eval.data_loader import EvalCase
    from arcticswarm.eval.judge import (
        FlexJudgeResult,
        InsightJudgeResult,
        QAJudgeResult,
    )

    report_path = resume_dir / "report.json"

    # For eval.repeat>1 (num_runs>1) the freshest data lives in per-run
    # checkpoints (run_0/report.json, run_1/report.json, …). The top-level
    # report.json with a "runs" key is written ONLY when the whole eval
    # finishes — during the run, and after a mid-run kill, it does not exist.
    # Prefer the per-run checkpoints when present so resume recovers completed
    # work across all runs instead of restarting from scratch.
    per_run_reports: list[dict] = []
    i = 0
    while (resume_dir / f"run_{i}" / "report.json").exists():
        try:
            with open(resume_dir / f"run_{i}" / "report.json") as f:
                per_run_reports.append(json.load(f))
        except Exception:
            break
        i += 1

    if per_run_reports:
        report = {
            "runs": per_run_reports,
            "swarm_enabled": any(
                r.get("swarm_enabled", False) for r in per_run_reports
            ),
        }
        runs_data = per_run_reports
        multi_run = True
    elif report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        # Detect single-run vs multi-run report
        if "runs" in report:
            runs_data = report["runs"]
            multi_run = True
        else:
            runs_data = [report]
            multi_run = False
    else:
        raise FileNotFoundError(f"No report.json found in {resume_dir}")

    grouped: list[list[EvalResult]] = []
    for run_idx, run_data in enumerate(runs_data):
        per_case = run_data.get("per_case", [])
        results: list[EvalResult] = []

        # Determine trajectory directory
        if multi_run:
            traj_dir = resume_dir / f"run_{run_idx}" / "trajectories"
        else:
            traj_dir = resume_dir / "trajectories"

        for case_dict in per_case:
            conv_id = case_dict.get("conv_id", "")
            case = EvalCase(
                conv_id=conv_id,
                turn_index=0,
                question=case_dict.get("question", ""),
                reference_answer="",  # Will be resolved from CSV later
                attributes={
                    "dataset": case_dict.get("dataset", ""),
                    "eval_mode": case_dict.get("eval_mode", "QA"),
                    "is_vip": True,
                },
            )

            result = EvalResult(
                case=case,
                response_text=case_dict.get("response_text", ""),
                tools_used=case_dict.get("tools_used", []),
                duration_seconds=case_dict.get("duration_seconds", 0.0),
                error=case_dict.get("error"),
                num_steps=case_dict.get("num_steps(turns)", 0),
            )

            # --- Restore judge results ---
            judge_mode = case_dict.get("judge_mode")
            # Fallback for old reports where judge_mode was missing due to
            # an indentation bug in metrics.py: infer from score field.
            if judge_mode is None and "score" in case_dict and not result.error:
                judge_mode = "QA"
            if judge_mode == "QA":
                result.qa_result = QAJudgeResult(
                    correct=case_dict.get("judge_correct") if case_dict.get("judge_correct") is not None else case_dict.get("score", 0) == 1.0,
                    comment=case_dict.get("judge_comment", ""),
                    raw_output=case_dict.get("judge_raw_output", ""),
                )
            if judge_mode == "INSIGHT":
                result.insight_result = InsightJudgeResult(
                    rating=case_dict.get("judge_rating", 0),
                    analysis=case_dict.get("judge_analysis", ""),
                    reasoning=case_dict.get("judge_reasoning", ""),
                )
            if "answer_only_rating" in case_dict:
                result.answer_only_result = InsightJudgeResult(
                    rating=case_dict.get("answer_only_rating", 0),
                    analysis=case_dict.get("answer_only_analysis", ""),
                )

            flex_acc = case_dict.get("flex_answer_accuracy", -1.0)
            if flex_acc != -1.0:
                result.flex_result = FlexJudgeResult(
                    flex_answer_accuracy=int(flex_acc),
                    answer_groundedness=float(case_dict.get("answer_groundedness", 0.0)),
                    answer_relevancy=int(case_dict.get("answer_relevancy", 0)),
                    methodology_soundness=int(case_dict.get("methodology_soundness", 0)),
                    accuracy_reasoning=case_dict.get("accuracy_reasoning", ""),
                )

            # --- Restore unit test results ---
            if "unit_test_score" in case_dict:
                result.unit_test_score = case_dict["unit_test_score"]
            if "unit_test_extracted_json" in case_dict:
                result.unit_test_extracted_json = case_dict["unit_test_extracted_json"]

            # --- Restore token usage ---
            tu_dict = case_dict.get("token_usage")
            if tu_dict and isinstance(tu_dict, dict):
                result.token_usage = TokenUsage(
                    input_tokens=tu_dict.get("input_tokens", 0),
                    output_tokens=tu_dict.get("output_tokens", 0),
                    cache_creation_input_tokens=tu_dict.get("cache_creation_input_tokens", 0),
                    cache_read_input_tokens=tu_dict.get("cache_read_input_tokens", 0),
                    reasoning_tokens=tu_dict.get("reasoning_tokens", 0),
                )
            if case_dict.get("token_breakdown"):
                result.token_breakdown = case_dict["token_breakdown"]
            if case_dict.get("tool_call_distribution"):
                result.tool_call_distribution = case_dict["tool_call_distribution"]

            # --- Restore swarm metrics ---
            result.swarm_teammates_spawned = case_dict.get("swarm_teammates_spawned", 0)
            result.swarm_bbs_message_count = case_dict.get("swarm_bbs_message_count", 0)
            result.swarm_saturation_events = case_dict.get("swarm_saturation_events", 0)
            if case_dict.get("swarm_subagent_tool_counts"):
                result.swarm_subagent_tool_counts = case_dict["swarm_subagent_tool_counts"]
            if case_dict.get("reflection"):
                result.swarm_reflection_stats = case_dict["reflection"]

            # --- Restore other metrics ---
            if case_dict.get("phase_timings"):
                result.phase_timings = case_dict["phase_timings"]
            if case_dict.get("latency_breakdown"):
                result.latency_breakdown = case_dict["latency_breakdown"]
            result.compaction_count = case_dict.get("compaction_count", 0)
            result.total_llm_calls = case_dict.get("total_llm_calls", 0)
            result.total_token_e2e = case_dict.get("total_token_e2e", 0)

            # --- Rebuild tool_calls from trajectory for response_full ---
            safe_name = conv_id.replace("/", "_").replace("\\", "_")[:200]
            traj_path = traj_dir / f"{safe_name}.json"
            if traj_path.exists():
                try:
                    with open(traj_path) as f:
                        traj = json.load(f)
                    if isinstance(traj, dict) and "trajectory" in traj:
                        traj = traj["trajectory"]
                    if isinstance(traj, list):
                        result.trajectory = traj
                        for msg in traj:
                            if not isinstance(msg, dict):
                                continue
                            content = msg.get("content", [])
                            if not isinstance(content, list):
                                continue
                            for block in content:
                                if not isinstance(block, dict):
                                    continue
                                if block.get("type") == "tool_result":
                                    tc = ToolCallRecord(
                                        name=block.get("tool_name", ""),
                                        input={},
                                        output=str(block.get("content", ""))[:2000],
                                        is_error=block.get("is_error", False),
                                    )
                                    result.tool_calls.append(tc)
                except Exception:
                    pass

            results.append(result)
        grouped.append(results)

    # Detect swarm mode
    is_swarm = report.get("swarm_enabled", False)
    if not is_swarm:
        is_swarm = any(
            case_dict.get("swarm_teammates_spawned", 0) > 0
            for run_data in runs_data
            for case_dict in run_data.get("per_case", [])
        )

    return grouped, is_swarm


# ---------------------------------------------------------------------------
# Rebuild from trajectories (crash recovery)
# ---------------------------------------------------------------------------


def load_results_from_trajectories(
    results_dir: Path,
) -> tuple[list[list[EvalResult]], bool]:
    """Reconstruct EvalResult objects from trajectory files only.

    Useful when a run crashed before any report.json checkpoint was saved.
    Scans ``{results_dir}/trajectories/*.json``, extracts question, response,
    tools, and timing from the saved agent conversation, and returns results
    ready for re-judging.

    Token usage is **not** available in trajectory files and will be ``None``.

    Returns ``(grouped_results, is_swarm)``.
    """
    from arcticswarm.eval.data_loader import EvalCase

    traj_dir = results_dir / "trajectories"
    if not traj_dir.exists():
        raise FileNotFoundError(f"No trajectories/ directory found in {results_dir}")

    traj_files = sorted(traj_dir.glob("*.json"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {traj_dir}")

    results: list[EvalResult] = []
    is_swarm = False

    for traj_path in traj_files:
        conv_id = traj_path.stem  # e.g. browsecomp_1011

        try:
            with open(traj_path) as f:
                raw = json.load(f)
        except Exception:
            continue

        # Unwrap trajectory wrapper
        phase_timings: dict[str, float] = {}
        if isinstance(raw, dict) and "trajectory" in raw:
            phase_timings = raw.get("phase_timings", {})
            traj_data = raw["trajectory"]
        else:
            traj_data = raw

        if not isinstance(traj_data, list) or not traj_data:
            continue

        # traj_data is a list of turns; each turn may have orchestrator messages
        # (swarm format) or be a flat list of messages (non-swarm).
        turn = traj_data[0]

        # Determine if this is swarm format (dict with "orchestrator" key)
        if isinstance(turn, dict) and "orchestrator" in turn:
            messages = turn["orchestrator"]
            subagents = turn.get("subagents", [])
            if subagents:
                is_swarm = True
        elif isinstance(turn, dict) and "role" in turn:
            # Non-swarm: flat list of messages
            messages = traj_data
            subagents = []
        else:
            continue

        if not isinstance(messages, list):
            continue

        # Extract question from first user message.  Multimodal image cases
        # build a content list in the order ``[text "Image 1:", image, ...,
        # text <question>]`` — so the last text block is the real question,
        # not the leading image marker.
        question = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    question = content
                elif isinstance(content, list):
                    for block in reversed(content):
                        if isinstance(block, dict) and block.get("type") == "text":
                            question = block.get("text", "")
                            break
                break

        # Extract response_text.  For SWARM runs the final answer is the
        # ``send_user_markdown_report`` tool input (its ``report`` field), NOT
        # an assistant text block — the orchestrator delivers the report via
        # that tool (see orchestrator.py / report_parser), so scan for the LAST
        # such tool_use first.  Falls back to the last assistant text block for
        # non-swarm runs (flat message lists) or swarm runs that timed out
        # before calling the report tool.
        response_text = ""
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in reversed(content):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "send_user_markdown_report"
                ):
                    rep = (block.get("input") or {}).get("report")
                    if isinstance(rep, str) and rep.strip():
                        response_text = rep
                        break
            if response_text:
                break

        if not response_text:
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                response_text = block["text"]
                                break
                    break

        # Extract tools_used from assistant tool_use blocks
        tools_used: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_name = block.get("name", "")
                    if tool_name and tool_name not in tools_used:
                        tools_used.append(tool_name)
                if block.get("type") == "tool_result":
                    tc = ToolCallRecord(
                        name=block.get("tool_name", ""),
                        input={},
                        output=str(block.get("content", ""))[:2000],
                        is_error=block.get("is_error", False),
                    )
                    tool_calls.append(tc)

        # Infer dataset from conv_id prefix (e.g. browsecomp_1011 -> BROWSECOMP)
        dataset = conv_id.split("_")[0].upper() if "_" in conv_id else ""

        case = EvalCase(
            conv_id=conv_id,
            turn_index=0,
            question=question,
            reference_answer="",  # Will be resolved from CSV later
            attributes={
                "dataset": dataset,
                "eval_mode": "QA",
                "is_vip": True,
            },
        )

        result = EvalResult(
            case=case,
            response_text=response_text,
            tools_used=tools_used,
            tool_calls=tool_calls,
            duration_seconds=phase_timings.get("total", 0.0),
            phase_timings=phase_timings,
        )
        result.trajectory = traj_data

        # Swarm metrics from subagents
        if isinstance(subagents, list):
            result.swarm_teammates_spawned = len(subagents)

        results.append(result)

    return [results], is_swarm


# ---------------------------------------------------------------------------
# Reference answer resolution
# ---------------------------------------------------------------------------


def resolve_reference_answers(
    grouped: list[list[EvalResult]],
    csv_path: str | None = None,
) -> None:
    """Fill in reference answers and date_override from the eval CSV.

    The saved report.json doesn't include reference answers (they're large),
    so we re-load them from the CSV by conv_id.
    """
    from arcticswarm.eval.data_loader import load_eval_cases

    # Load all cases (no filters) to build a lookup
    all_cases = load_eval_cases(csv_path=csv_path, vip_only=False)
    by_conv_id = {c.conv_id: c for c in all_cases}

    for results in grouped:
        for result in results:
            source = by_conv_id.get(result.case.conv_id)
            if source:
                result.case.reference_answer = source.reference_answer
                result.case.date_override = source.date_override
                result.case.unit_test = source.unit_test
                result.case.attributes = source.attributes


# ---------------------------------------------------------------------------
# Trajectory I/O
# ---------------------------------------------------------------------------


def save_trajectory(output_dir: Path, result: EvalResult) -> None:
    """Write a single case's agent trajectory to the trajectories folder."""
    if not result.trajectory:
        return
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    # Sanitise conv_id for use as a filename
    safe_name = result.case.conv_id.replace("/", "_").replace("\\", "_")[:200]
    traj_path = traj_dir / f"{safe_name}.json"

    # Wrap trajectory with phase_timings metadata when available
    payload: Any
    if result.phase_timings:
        payload = {
            "phase_timings": result.phase_timings,
            "trajectory": result.trajectory,
        }
    else:
        payload = result.trajectory

    traj_path.write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Result classification helpers
# ---------------------------------------------------------------------------


def result_is_complete(r: EvalResult) -> bool:
    """Return True if a result has a response or an explicit error."""
    return bool(r.response_text) or bool(r.error)


def result_has_rerunnable_error(
    r: EvalResult, output_dir: Path,
) -> tuple[bool, str]:
    """Check if a completed result has an error worth re-running.

    Returns ``(should_rerun, reason)`` tuple.
    """
    # (1) Execution error (timeout, exception, no answer)
    if r.error:
        return True, f"error: {r.error[:80]}"

    # (2) Web search rate limits from failure log files
    safe_name = r.case.conv_id.replace("/", "_").replace("\\", "_")[:200]
    wsf_path = output_dir / "web_search_failures" / f"{safe_name}.json"
    if wsf_path.exists():
        try:
            content = wsf_path.read_text()
            if "399530" in content or "rate limit" in content.lower():
                return True, "web search rate limit (399530)"
        except OSError:
            pass

    return False, ""

def result_is_timeout(
    case_dict: dict[str, Any],
    timeout_seconds: float,
) -> bool:
    """Detect timeout cases from a prior report's per-case dict.

    A case is considered a timeout if its ``duration_seconds`` equals or
    exceeds ``timeout_seconds``.  This catches both unrecovered timeouts
    (which have an ``error`` field) and *recovered* timeouts where the
    force-report fallback produced an answer and cleared the error — the
    latter are invisible to ``--rerun-errors``.
    """
    dur = case_dict.get("duration_seconds", 0)
    return dur >= timeout_seconds


def result_is_wrong(r: "EvalResult") -> bool:
    """Return True if the result was judged incorrect (not an error/incomplete)."""
    if r.error:
        return False
    if r.qa_result is not None:
        return not r.qa_result.correct
    if r.insight_result is not None:
        return r.insight_result.rating == 0
    if r.flex_result is not None:
        return r.flex_result.flex_answer_accuracy == 0
    return False


# ---------------------------------------------------------------------------
# Rerun-errors summary
# ---------------------------------------------------------------------------


def print_rerun_errors_summary(
    console: Any,
    meta: dict[str, Any],
    report: Any,
) -> None:
    """Print a summary of --rerun-errors results."""
    per_case = report.to_dict().get("per_case", [])
    error_ids = meta.get("error_ids", set())

    # Compute accuracy for re-run cases only
    rerun_correct = 0
    rerun_total = 0
    for c in per_case:
        if c.get("conv_id") in error_ids:
            rerun_total += 1
            correct = c.get("judge_correct")
            if correct is None:
                correct = c.get("score", 0) > 0
            if correct:
                rerun_correct += 1

    console.print("[bold]Rerun-errors Summary[/bold]")
    console.print(f"  Clean (preserved):  {meta['clean_count']}")
    console.print(
        f"  Re-run (errors):    {meta['error_count']}"
        + (f"  → {rerun_correct}/{rerun_total} correct "
           f"({100 * rerun_correct / rerun_total:.1f}%)"
           if rerun_total > 0 else "")
    )
    if meta.get("incomplete_count"):
        console.print(f"  Incomplete stubs:   {meta['incomplete_count']}")
    console.print()
    console.print("  Error breakdown:")
    for cat, cnt in sorted(meta.get("error_categories", {}).items()):
        console.print(f"    {cat:30s} {cnt}")
    console.print()
