"""CLI entry point for the arcticswarm evaluation suite.

Usage::

    arcticswarm-eval --config conf/bench/browsecomp.yaml eval.output=results/run1
    arcticswarm-eval --config conf/bench/browsecomp_plus.yaml eval.output=results/bcp
    arcticswarm-eval --config conf/bench/browsecomp.yaml --config conf/override.yaml  # composition

Or equivalently::

    python -m arcticswarm.eval --config conf/bench/browsecomp.yaml eval.output=results/run1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from typing import Any

from arcticswarm.logging_utils import SearchApiUnhealthyError, capture_git_snapshot, check_search_api_health, compute_grounding_fetch_stats, get_contamination_stats, print_judge_fallback_stats, reset_contamination_stats, save_compactor_log, save_content_filter_log, save_web_fetch_log, save_web_search_failures, save_web_search_log, silence_noisy_loggers
from arcticswarm.config import ArcticswarmConfig, settings_json_path
from arcticswarm.eval.data_loader import EvalCase, load_eval_cases
from arcticswarm.eval.judge import LLMJudge
from arcticswarm.eval.live_log import LiveEvalLogger
from arcticswarm.eval.metrics import (
    DatasetMetrics,
    EvalReport,
    compute_aggregated_report,
    compute_metrics,
    compute_multi_run_report,
    generate_breakdown_md,
)
from arcticswarm.eval.runner import EvalResult, ToolCallRecord, run_eval_repeated
from arcticswarm.eval.recovery import (
    load_results_for_rejudge,
    load_results_for_resume,
    load_results_from_trajectories,
    print_rerun_errors_summary,
    resolve_reference_answers,
    result_has_rerunnable_error,
    result_is_complete,
    result_is_timeout,
    result_is_wrong,
    save_trajectory,
)
from arcticswarm.run_config import load_run_config, RunConfig

logger = logging.getLogger(__name__)

_DIFFICULTY_LEVELS = {"easy", "medium", "hard", "extreme"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arcticswarm-eval",
        description=(
            "Arcticswarm offline evaluation suite.\n\n"
            "All config is specified via YAML files + dot-notation overrides.\n"
            "Example: arcticswarm-eval --config conf/bench/browsecomp.yaml eval.output=results/bc"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", "-c",
        action="append",
        required=True,
        help="YAML config file (repeatable, merged left to right). E.g. -c conf/bench/browsecomp.yaml -c conf/override.yaml",
    )
    p.add_argument(
        "overrides",
        nargs="*",
        help="Dot-notation overrides: key.subkey=value (e.g. eval.output=results/run1 eval.parallel=8)",
    )
    p.add_argument(
        "--rejudge",
        default=None,
        metavar="DIR",
        help=(
            "Re-run judges on a previous eval run's results (skip agent execution). "
            "DIR is the output directory from a prior run."
        ),
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return p


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------


def _print_summary(
    console: Console,
    report: EvalReport,
    num_runs: int | None = None,
) -> None:
    """Print a rich summary table to the console."""
    is_multi = num_runs is not None and num_runs > 1
    console.print()
    console.print("[bold]Evaluation Summary[/bold]")
    console.print(f"  Cases per run: {report.total_cases}")
    if is_multi:
        console.print(f"  Avg errors/run: {report.total_errors}")
    else:
        console.print(f"  Total errors: {report.total_errors}")
    if report.total_prompt_too_long_errors > 0:
        console.print(f"  Prompt-too-long errors: {report.total_prompt_too_long_errors}")
    console.print()

    title = "Per-Dataset Metrics"
    if is_multi:
        title += f" ({num_runs} runs, mean ± stddev)"

    errors_col = "Avg Errors" if is_multi else "Errors"

    # Show unit test column only when at least one dataset has scores.
    has_unit_test = any(
        dm.unit_test_accuracy >= 0 for dm in list(report.datasets.values()) + [report.overall]
    )

    # Show QA LLM Acc column only when qa_llm judge actually ran.
    has_qa_llm = any(
        dm.qa_llm_accuracy > 0 for dm in list(report.datasets.values()) + [report.overall]
    )

    # Show Flex Acc column only when at least one dataset has FLEX scores.
    has_flex = any(
        dm.flex_answer_accuracy >= 0 for dm in list(report.datasets.values()) + [report.overall]
    )

    has_bi_phases = any(
        dm.bi_phase1_rate >= 0 for dm in list(report.datasets.values()) + [report.overall]
    )

    table = Table(title=title)
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Cases", justify="right", no_wrap=True)
    table.add_column(errors_col, justify="right", no_wrap=True)
    if has_qa_llm:
        table.add_column("QA LLM Acc", justify="right", no_wrap=True)
    table.add_column("Ans Only Acc", justify="right", no_wrap=True)
    if has_unit_test:
        table.add_column("Unit Test Acc", justify="right", no_wrap=True)
    if has_flex:
        table.add_column("Flex Acc", justify="right", no_wrap=True)
    if has_bi_phases:
        table.add_column("Phase1 Rate", justify="right", no_wrap=True)
        table.add_column("Phase2 Rate", justify="right", no_wrap=True)
    table.add_column("Total Lat(s)", justify="right", no_wrap=True)
    table.add_column("Avg Lat(s)", justify="right", no_wrap=True)
    table.add_column("Avg Steps", justify="right", no_wrap=True)
    table.add_column("Calls/Step", justify="right", no_wrap=True)
    table.add_column("Avg Tokens", justify="right", no_wrap=True)

    def _fmt_val(val: float, stddev: float | None, fmt: str = ".3f") -> str:
        if stddev is not None:
            return f"{val:{fmt}} ± {stddev:{fmt}}"
        return f"{val:{fmt}}"

    def _fmt_metric(dm: DatasetMetrics) -> tuple[str, ...]:
        base = (
            dm.dataset,
            str(dm.num_cases),
            str(dm.num_errors),
        )
        qa = ()
        if has_qa_llm:
            qa = (_fmt_val(dm.qa_llm_accuracy, dm.qa_llm_accuracy_stddev),)
        ao = ("N/A" if dm.answer_only_accuracy == -1.0 else _fmt_val(dm.answer_only_accuracy, dm.answer_only_accuracy_stddev),)
        ut = ()
        if has_unit_test:
            if dm.unit_test_accuracy >= 0:
                ut = (_fmt_val(dm.unit_test_accuracy, dm.unit_test_accuracy_stddev),)
            else:
                ut = ("N/A",)
        fx = ()
        if has_flex:
            if dm.flex_answer_accuracy >= 0:
                fx = (_fmt_val(dm.flex_answer_accuracy, dm.flex_answer_accuracy_stddev),)
            else:
                fx = ("N/A",)
        bi = ()
        if has_bi_phases:
            if dm.bi_phase1_rate >= 0:
                bi = (
                    _fmt_val(dm.bi_phase1_rate, dm.bi_phase1_rate_stddev),
                    _fmt_val(dm.bi_phase2_rate, dm.bi_phase2_rate_stddev),
                )
            else:
                bi = ("N/A", "N/A")
        tail = (
            _fmt_val(dm.total_duration_seconds, None, ".1f"),
            _fmt_val(dm.avg_duration_seconds, dm.avg_duration_seconds_stddev, ".1f"),
            _fmt_val(dm.avg_steps, None, ".2f"),
            _fmt_val(dm.avg_tool_calls_per_step, None, ".2f"),
            _fmt_val(dm.avg_total_tokens, None, ",.0f"),
        )
        return base + qa + ao + ut + fx + bi + tail

    for dm in report.datasets.values():
        table.add_row(*_fmt_metric(dm))

    # Overall row
    table.add_section()
    table.add_row(*_fmt_metric(report.overall), style="bold")

    console.print(table)
    console.print()


def _print_performance(
    console: Console,
    report: EvalReport,
    num_runs: int | None = None,
) -> None:
    """Print per-dataset details not shown in the summary table."""
    if report.overall.num_cases == 0:
        return

    header = "Per-Dataset Performance Details"
    if num_runs and num_runs > 1:
        header += f" (across {num_runs} runs)"
    console.print(f"[bold]{header}[/bold]")

    def _print_dm(dm: DatasetMetrics) -> None:
        console.print(f"  [cyan]{dm.dataset}[/cyan] ({dm.num_cases} cases):")
        if dm.bi_phase1_rate >= 0:
            p1_str = f"{dm.bi_phase1_rate:.3f}"
            if dm.bi_phase1_rate_stddev is not None:
                p1_str += f" ± {dm.bi_phase1_rate_stddev:.3f}"
            p2_str = f"{dm.bi_phase2_rate:.3f}"
            if dm.bi_phase2_rate_stddev is not None:
                p2_str += f" ± {dm.bi_phase2_rate_stddev:.3f}"
            console.print(f"    Phase 1 Completion:   {p1_str}")
            console.print(f"    Phase 2 Completion:   {p2_str}")
        console.print(f"    P50 Latency (Median): {dm.p50_duration_seconds:.3f} s")
        console.print(f"    P90 Latency:          {dm.p90_duration_seconds:.3f} s")
        if dm.avg_latency_breakdown:
            bd = dm.avg_latency_breakdown
            total = dm.avg_duration_seconds or sum(v for k, v in bd.items() if not k.startswith(" "))
            console.print(f"    Avg Latency Breakdown ({total:.1f}s):")
            for key, val in bd.items():
                pct = (val / total * 100) if total > 0 else 0.0
                if key.startswith("  "):
                    console.print(f"        {key:25s} {val:8.1f}s  ({pct:4.1f}%)")
                else:
                    console.print(f"      - {key:25s} {val:8.1f}s  ({pct:4.1f}%)")
        console.print(f"    Avg Orch Tokens (orchestration model):")
        console.print(f"      - Input Tokens:          {dm.avg_input_tokens:,.0f}")
        console.print(f"      - Output Tokens:         {dm.avg_output_tokens:,.0f}")
        console.print(f"      - Cache Read Tokens:     {dm.avg_cache_read_tokens:,.0f}")
        console.print(f"      - Cache Creation Tokens: {dm.avg_cache_creation_tokens:,.0f}")
        console.print(f"      - Total:                 {dm.avg_total_tokens:,.0f}")
        if dm.avg_total_token_e2e > 0:
            console.print(f"      - Total E2E (swarm):     {dm.avg_total_token_e2e:,.0f}")
        has_turn_tokens = dm.avg_output_tokens_other_tool_turns > 0 or dm.avg_output_tokens_text_turns > 0
        if has_turn_tokens:
            _ototal = dm.avg_output_tokens_other_tool_turns + dm.avg_output_tokens_text_turns
            def _opct(v: float) -> str:
                return f"({v / _ototal * 100:4.1f}%)" if _ototal > 0 else ""
            console.print(f"    Avg Orch Output Token Breakdown (per-turn):")
            console.print(f"      - Tool turns:       {dm.avg_output_tokens_other_tool_turns:>8,.0f}  {_opct(dm.avg_output_tokens_other_tool_turns)}")
            console.print(f"      - Text-only turns:  {dm.avg_output_tokens_text_turns:>8,.0f}  {_opct(dm.avg_output_tokens_text_turns)}")
        if dm.avg_tool_call_distribution:
            console.print(f"    Avg Tool Call Distribution:")
            for tool, avg_count in dm.avg_tool_call_distribution.items():
                console.print(f"      - {tool:30s} {avg_count:.1f}")

    for dm in report.datasets.values():
        _print_dm(dm)
    _print_dm(report.overall)
    console.print()


def _print_browsecomp_difficulty_breakdown(console: Console, report: EvalReport) -> None:
    """Print accuracy breakdown by difficulty for BrowseComp datasets."""
    per_case = report.to_dict().get("per_case", [])
    if not per_case:
        return

    # Check if difficulty annotations are present
    has_difficulty = any(c.get("difficulty") for c in per_case)
    if not has_difficulty:
        return

    from collections import defaultdict

    by_diff: dict[str, list[bool]] = defaultdict(list)
    for c in per_case:
        diff = c.get("difficulty")
        if not diff:
            continue
        correct = c.get("judge_correct")
        if correct is None:
            correct = c.get("score", 0) > 0
        by_diff[diff].append(bool(correct))

    if not by_diff:
        return

    console.print("[bold]Accuracy by Difficulty[/bold]")
    total_correct = sum(sum(v) for v in by_diff.values())
    total_cases = sum(len(v) for v in by_diff.values())
    for diff in ["easy", "medium", "hard", "extreme"]:
        vals = by_diff.get(diff, [])
        if not vals:
            continue
        correct = sum(vals)
        n = len(vals)
        console.print(f"  {diff:<8} {correct:>3}/{n:<3} ({100 * correct / n:.1f}%)")
    console.print(
        f"  [bold]overall  {total_correct:>3}/{total_cases:<3} "
        f"({100 * total_correct / total_cases:.1f}%)[/bold]"
    )
    console.print()

def _format_score(result: EvalResult) -> str:
    """Return a human-readable score string for a completed result."""
    parts: list[str] = []
    if result.insight_result is not None:
        parts.append(f"rating={result.insight_result.rating}")
    if result.answer_only_result is not None:
        parts.append(f"ao={result.answer_only_result.rating}")
    if result.qa_result is not None:
        parts.append("correct" if result.qa_result.correct else "incorrect")
    return " ".join(parts)



# ---------------------------------------------------------------------------
# Grounding stats
# ---------------------------------------------------------------------------


def _print_grounding_fetch_stats(
    console: "Console",
    grouped_results: "list[list[Any]]",
) -> None:
    """Print aggregate grounding fetch tier stats across all eval cases."""
    all_fetch_logs = [
        result.web_fetch_log
        for run in grouped_results
        for result in run
        if result.web_fetch_log
    ]
    if not all_fetch_logs:
        return

    stats = compute_grounding_fetch_stats(all_fetch_logs)
    if stats["grounding_attempts"] == 0:
        return

    attempts = stats["grounding_attempts"]
    successes = stats["grounding_success"]
    fallbacks = stats["grounding_fallback"]
    fail_pct = stats["grounding_fail_rate_pct"]

    console.print()
    console.print("[bold]Grounding Web Fetch Stats[/bold]")
    console.print(f"  Grounding attempts : {attempts}")
    console.print(f"  Grounding success  : {successes} ({100.0 - fail_pct:.1f}%)")
    console.print(f"  Fallback rate      : {fallbacks}/{attempts} ({fail_pct:.1f}%)")
    if fallbacks > 0:
        console.print(f"    └─ to Jina     : {stats['fallback_to_jina']}")
        console.print(f"    └─ to Serper   : {stats['fallback_to_serper']}")
        console.print(f"    └─ to requests : {stats['fallback_to_requests']}")
        console.print(f"    └─ total fail  : {stats['fallback_to_none']}")


# ---------------------------------------------------------------------------
# Post-filters (--subset-filter: difficulty, WHY subset)
# ---------------------------------------------------------------------------


def _apply_subset_filter(choice: str, datasets: list[str]) -> tuple[str, set[str]]:
    """Resolve a --subset-filter choice into (label, allowed_conv_ids).

    Two ad-hoc forms are supported for targeted reruns:
      * ``file:<path>`` reads one ``conv_id`` per line from *path*
        (lines are stripped; blanks and lines starting with ``#`` are ignored).
      * ``ids:<cid1>,<cid2>,...`` parses a comma-separated list inline.
    Neither form requires the ids to belong to any difficulty / why-subset
    registry, so it is the right escape hatch for re-running a custom
    set of cases (e.g. content-filter-affected questions).
    """
    if choice.startswith("file:"):
        path = choice[len("file:"):].strip()
        if not path:
            raise ValueError("subset_filter='file:' requires a non-empty path")
        with open(path, encoding="utf-8") as f:
            ids = {
                ln.strip()
                for ln in f
                if ln.strip() and not ln.strip().startswith("#")
            }
        if not ids:
            raise ValueError(f"subset_filter file {path!r} yielded no conv_ids")
        return (f"file={path} ({len(ids)} ids)", ids)

    if choice.startswith("ids:"):
        raw = choice[len("ids:"):]
        ids = {tok.strip() for tok in raw.split(",") if tok.strip()}
        if not ids:
            raise ValueError("subset_filter='ids:' requires at least one conv_id")
        return (f"ids ({len(ids)} ids)", ids)

    raise ValueError(
        f"Unsupported subset_filter {choice!r}; use 'file:<path>' or 'ids:<id1>,<id2>,...'."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    silence_noisy_loggers()

    term_size = shutil.get_terminal_size((220, 50))
    console = Console(width=max(term_size.columns, 220), height=term_size.lines)

    # Load hierarchical config from YAML and convert to flat ArcticswarmConfig
    try:
        run_cfg = load_run_config(args.config, args.overrides or [])
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    ev = run_cfg.eval  # shorthand for eval config section
    config = run_cfg.to_arcticswarm_config()
    if ev.verbose and not args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    num_runs = max(1, ev.repeat)

    if (
        not ev.conv_id
        and not ev.datasets
        and not args.rejudge
        and not ev.rebuild_from_trajectories
    ):
        console.print(
            "[bold red]Error:[/bold red] eval.datasets is required in config "
            "unless eval.conv_id or --rejudge is provided"
        )
        sys.exit(1)

    if not ev.output and not args.rejudge:
        console.print(
            "[bold red]Error:[/bold red] eval.output is required. "
            "Set it in your YAML or via override: eval.output=results/my_run"
        )
        sys.exit(1)

    # Node-local cache mirror: when caching is enabled, automatically mirror the
    # shared master cache to node-local disk and sync deltas back periodically.
    # Avoids the SIGBUS/deadlock from a shared WAL SQLite on Lustre across hosts.
    # Mutates config.fetch_cache_path to the local mirror.
    #
    # should_auto_mirror() engages it (web.cache_local_mirror defaults True) only
    # when caching is active AND node-local fast storage exists, so a dev box /
    # CPU pod without a fast-disk mount silently uses the master cache directly.
    # Set web.cache_local_mirror=false to force it off.
    _cache_mirror = None
    from arcticswarm.tools.cache_sync import should_auto_mirror
    if should_auto_mirror(config) and ev.output and not args.rejudge:
        try:
            from arcticswarm.tools.cache_sync import CacheMirrorManager
            _cache_mirror = CacheMirrorManager(config, output_dir=ev.output)
            _cache_mirror.setup_and_start()
            console.print(
                f"[dim]cache mirror on: shared caches copied to "
                f"{getattr(config, 'cache_local_dir', '')}; deltas synced back to "
                f"the master cache every {getattr(config, 'cache_sync_every', 5)} cases[/dim]"
            )
        except Exception as exc:  # never let cache plumbing block a run
            console.print(f"[yellow]cache mirror setup failed ({exc}); continuing without it[/yellow]")

    if not config.api_key:
        console.print(
            "[bold red]Error:[/bold red] api_key is not set. "
            f"Add it to [cyan]{settings_json_path()}[/cyan] "
            "(or set env [cyan]ARCTICSWARM_SETTINGS_PATH[/cyan] to that file), e.g.:\n"
            '  {"api_key": "sk-ant-..."}'
        )
        sys.exit(1)

    # Probe configured Brave/Serper/Tavily/Jina keys before doing any real
    # work so a broken billing/quota fails fast rather than silently degrading
    # every case in the run.
    try:
        check_search_api_health(config)
    except SearchApiUnhealthyError as exc:
        console.print(f"[bold red]API health check failed:[/bold red] {exc}")
        sys.exit(2)

    # Load eval cases (skip when --rejudge or standalone --rebuild-from-trajectories is active)
    if not args.rejudge and not (ev.rebuild_from_trajectories and not ev.resume):
        if ev.conv_id:
            console.print(f"[bold]Loading eval case[/bold] (conv_id={ev.conv_id})")
        else:
            datasets_str = ", ".join(ev.datasets)
            console.print(f"[bold]Loading eval cases[/bold] (datasets={datasets_str}, vip_only={ev.vip_only})")
        try:
            cases = load_eval_cases(
                csv_path=ev.csv_path or None,
                datasets=ev.datasets or None,
                vip_only=ev.vip_only,
                eval_mode=ev.eval_mode,
                limit=ev.limit,
                offset=ev.offset,
                conv_id=ev.conv_id or None,
            )
        except FileNotFoundError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)

        if not cases:
            console.print("[yellow]No eval cases found matching the filters.[/yellow]")
            sys.exit(0)

        console.print(f"  Found [cyan]{len(cases)}[/cyan] eval cases")

        # Apply post-filter (difficulty or WHY subset) — skip when conv_id targets a single case
        if ev.subset_filter and not ev.conv_id:
            label, allowed_ids = _apply_subset_filter(ev.subset_filter, ev.datasets)
            before = len(cases)
            cases = [c for c in cases if c.conv_id in allowed_ids]
            console.print(
                f"  Filter [cyan]{label}[/cyan]: {len(cases)}/{before} cases selected"
            )
            if not cases:
                console.print(f"[yellow]No eval cases match the {label} filter.[/yellow]")
                sys.exit(0)

    # --resume / --rerun-errors: detect completed cases from prior report.json (or trajectory files)
    resumed_results: list[list[EvalResult]] | None = None
    cases_by_run: list[list[Any]] | None = None
    needs_rejudge_by_run: list[list[EvalResult]] = []
    rerun_error_meta: dict[str, Any] | None = None  # populated by --rerun-errors / --rerun-timeouts / --rerun-wrong
    if not args.rejudge and (ev.resume or ev.rerun_errors or ev.rerun_timeouts or ev.rerun_wrong):
        output_dir = Path(ev.output)
        prior_report = output_dir / "report.json"
        # For eval.repeat>1 the top-level report.json is only written when the
        # whole eval finishes; mid-run (or after a kill) the checkpoints live in
        # per-run run_N/report.json files. Treat those as a valid resume source.
        prior_per_run = (output_dir / "run_0" / "report.json").exists()
        # Choose data source: trajectory files or report.json
        if ev.rebuild_from_trajectories:
            console.print(
                f"[bold]Rebuilding from trajectory files[/bold] in {output_dir / 'trajectories'}"
            )
            prior_grouped, _prior_is_swarm = load_results_from_trajectories(output_dir)
            resolve_reference_answers(prior_grouped, csv_path=ev.csv_path or None)
        elif prior_report.exists() or prior_per_run:
            prior_grouped, _prior_is_swarm = load_results_for_resume(output_dir)
            resolve_reference_answers(prior_grouped, csv_path=ev.csv_path or None)
        else:
            prior_grouped = None

        if prior_grouped is not None:
            if len(prior_grouped) != num_runs:
                console.print(
                    f"  [yellow]Resume run count mismatch:[/yellow] prior report has "
                    f"{len(prior_grouped)} run(s), config repeat={num_runs}. "
                    "Missing runs will start fresh; extra prior runs will be ignored."
                )
            if len(prior_grouped) < num_runs:
                prior_grouped.extend([[] for _ in range(num_runs - len(prior_grouped))])
            elif len(prior_grouped) > num_runs:
                prior_grouped = prior_grouped[:num_runs]

            all_cases_count = len(cases)
            cases_by_run: list[list[EvalCase]] = []
            per_run_status: list[tuple[int, int, int, int]] = []
            needs_rejudge_by_run = [[] for _ in range(num_runs)]

            # Aggregated across all runs for the rerun-errors/timeouts summary.
            from collections import Counter as _Counter
            _agg_clean = 0
            _agg_incomplete = 0
            _agg_rerun_reasons: dict[str, str] = {}
            _agg_error_categories: _Counter[str] = _Counter()
            _agg_error_ids: set[str] = set()

            for run_idx in range(num_runs):
                complete_results = [
                    r for r in prior_grouped[run_idx] if result_is_complete(r)
                ]
                incomplete_ids = {
                    r.case.conv_id
                    for r in prior_grouped[run_idx]
                    if not result_is_complete(r)
                }

                run_needs_rejudge = [
                    r for r in complete_results
                    if r.qa_result is None
                    and r.insight_result is None
                    and r.flex_result is None
                    and not (ev.rerun_errors and r.error)
                ]
                needs_rejudge_ids = {r.case.conv_id for r in run_needs_rejudge}
                complete_results = [
                    r for r in complete_results
                    if r.case.conv_id not in needs_rejudge_ids
                ]

                if ev.rerun_errors or ev.rerun_timeouts or ev.rerun_wrong:
                    # --rerun-errors / --rerun-timeouts: split complete results
                    # into clean (preserved) vs rerun (re-executed). Applied to
                    # EVERY run so eval.repeat>1 reruns errors/timeouts in all
                    # runs, not just run 0.
                    clean_results = []
                    rerun_reasons: dict[str, str] = {}

                    # Timeout threshold = the configured eval.timeout for THIS
                    # rerun (pass the same eval.timeout as the original run).
                    # NOTE: this used to be *inferred* from max(duration)
                    # (max_dur-300, or max_dur*0.9 for a single max). That broke
                    # whenever an earlier rerun_timeouts pass had already double-run
                    # a case to ~2x the limit: max_dur then sat near ~18000s, so the
                    # inferred threshold (~16700s) sat far above the real ~9000s wall
                    # and only the single longest case got re-flagged (the "1 rerun
                    # instead of ~30" bug). Use the explicit configured value.
                    _prior_timeout: float = ev.timeout
                    if ev.rerun_timeouts and run_idx == 0:
                        logger.info(
                            "rerun_timeouts threshold: %.0fs (eval.timeout)", _prior_timeout,
                        )

                    for r in complete_results:
                        should_rerun = False
                        reason = ""

                        # Check errors first
                        if ev.rerun_errors:
                            should_rerun, reason = result_has_rerunnable_error(r, output_dir)

                        # Check timeouts (catches recovered timeouts invisible to
                        # rerun_errors). Duration comes from the restored result
                        # itself, so this works per-run without the top-level report.
                        if not should_rerun and ev.rerun_timeouts:
                            _case_dict = {"duration_seconds": r.duration_seconds}
                            if result_is_timeout(_case_dict, _prior_timeout):
                                should_rerun = True
                                reason = f"timeout: {r.duration_seconds:.0f}s (prior limit {_prior_timeout:.0f}s)"

                        # Check wrong answers (judged incorrect)
                        if not should_rerun and ev.rerun_wrong:
                            if result_is_wrong(r):
                                should_rerun = True
                                reason = "wrong answer"

                        if should_rerun:
                            rerun_reasons[r.case.conv_id] = reason
                            if "rate limit" in reason.lower() or "399530" in reason:
                                _agg_error_categories["web search rate limit"] += 1
                            elif "timeout" in reason.lower() or "timed out" in reason.lower():
                                _agg_error_categories["timeout"] += 1
                            elif "wrong" in reason.lower():
                                _agg_error_categories["wrong answer"] += 1
                            else:
                                _agg_error_categories["execution error"] += 1
                        else:
                            clean_results.append(r)

                    error_ids = set(rerun_reasons.keys())
                    completed_ids = {r.case.conv_id for r in clean_results}
                    remaining_cases = [
                        c for c in cases
                        if c.conv_id not in completed_ids
                        and c.conv_id not in needs_rejudge_ids
                    ]
                    cases_by_run.append(remaining_cases)
                    needs_rejudge_by_run[run_idx] = run_needs_rejudge
                    prior_grouped[run_idx] = clean_results
                    per_run_status.append(
                        (len(completed_ids), len(run_needs_rejudge), len(incomplete_ids), len(remaining_cases))
                    )

                    # Accumulate into the cross-run summary.
                    _agg_clean += len(clean_results)
                    _agg_incomplete += len(incomplete_ids)
                    _agg_rerun_reasons.update(rerun_reasons)
                    _agg_error_ids |= error_ids
                    rerun_error_meta = {
                        "clean_count": _agg_clean,
                        "error_count": len(_agg_error_ids),
                        "incomplete_count": _agg_incomplete,
                        "rerun_reasons": dict(_agg_rerun_reasons),
                        "error_categories": dict(_agg_error_categories),
                        "error_ids": set(_agg_error_ids),
                    }
                else:
                    completed_ids = {r.case.conv_id for r in complete_results}
                    remaining_cases = [
                        c for c in cases
                        if c.conv_id not in completed_ids
                        and c.conv_id not in needs_rejudge_ids
                    ]
                    cases_by_run.append(remaining_cases)
                    needs_rejudge_by_run[run_idx] = run_needs_rejudge
                    prior_grouped[run_idx] = complete_results
                    per_run_status.append(
                        (
                            len(completed_ids),
                            len(run_needs_rejudge),
                            len(incomplete_ids),
                            len(remaining_cases),
                        )
                    )

            resumed_results = prior_grouped

            if (ev.rerun_errors or ev.rerun_timeouts or ev.rerun_wrong) and rerun_error_meta:
                parts = []
                if ev.rerun_errors:
                    parts.append("errors")
                if ev.rerun_timeouts:
                    parts.append("timeouts")
                if ev.rerun_wrong:
                    parts.append("wrong")
                label = "Rerun-" + "+".join(parts) if parts else "Rerun"
                console.print(
                    f"  [bold]{label}:[/bold] {rerun_error_meta['clean_count']} clean (preserved), "
                    f"{rerun_error_meta['error_count']} to rerun, "
                    f"{rerun_error_meta['incomplete_count']} incomplete stubs"
                )
                for cat, cnt in sorted(rerun_error_meta.get("error_categories", {}).items()):
                    console.print(f"    [yellow]→ {cat}: {cnt}[/yellow]")
            elif num_runs == 1:
                completed_n, rejudge_n, incomplete_n, remaining_n = per_run_status[0]
                console.print(
                    f"  [bold]Resuming:[/bold] {completed_n} completed, "
                    f"{remaining_n} remaining out of {all_cases_count} total"
                )
                if incomplete_n:
                    console.print(
                        f"  [yellow]→ {incomplete_n} incomplete stubs will be re-run[/yellow]"
                    )
                if rejudge_n:
                    console.print(
                        f"  [yellow]→ {rejudge_n} completed cases missing judge verdict, will re-judge[/yellow]"
                    )
            else:
                console.print(
                    f"  [bold]Resuming {num_runs} runs:[/bold] "
                    f"{all_cases_count} cases per run"
                )
                for run_idx, (completed_n, rejudge_n, incomplete_n, remaining_n) in enumerate(per_run_status):
                    console.print(
                        f"    run {run_idx}: {completed_n} completed, "
                        f"{remaining_n} remaining"
                    )
                    if incomplete_n:
                        console.print(
                            f"      [yellow]→ {incomplete_n} incomplete stubs will be re-run[/yellow]"
                        )
                    if rejudge_n:
                        console.print(
                            f"      [yellow]→ {rejudge_n} completed cases missing judge verdict, will re-judge[/yellow]"
                        )
        else:
            console.print(
                "[yellow]Warning:[/yellow] --resume/--rerun-errors/--rerun-timeouts/--rerun-wrong specified but no report.json "
                f"found in {output_dir}. Running from scratch."
            )

    # Initialize judge.
    #
    # Azure GPT judge (e.g. ``azure.enabled=true eval.judge_model=gpt-4-1-dev``)
    # takes precedence over a self-hosted ``judge_model_base_url``: when the
    # judge model is a GPT deployment and Azure is enabled, route through Azure
    # and ignore any vLLM judge URL baked into the preset. Otherwise the
    # ``judge_base_url`` branch in LLMJudge.__init__ would win and try to serve
    # the GPT model from the (Qwen) vLLM endpoint. This keeps the two-flag
    # recipe robust regardless of the preset's default judge endpoint.
    _use_azure_judge = config.use_azure_openai and ev.judge_model.startswith("gpt")
    judge = LLMJudge(
        api_key=config.api_key,
        base_url=config.base_url,
        model=ev.judge_model,
        use_azure_openai=_use_azure_judge,
        judge_base_url="" if _use_azure_judge else ev.judge_model_base_url,
        custom_judge_prompt=ev.custom_judge_prompt,
    )

    # Re-judge resumed cases that are missing judge verdicts.
    # On failure (or if the judge still produces no verdict), re-run the
    # whole question from scratch.
    if resumed_results and any(needs_rejudge_by_run):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from arcticswarm.eval.runner import judge_result

        rejudge_succeeded: list[tuple[int, EvalResult]] = []
        rejudge_failed: list[tuple[int, EvalResult]] = []

        with ThreadPoolExecutor(max_workers=ev.parallel) as pool:
            futures = {
                pool.submit(
                    judge_result, result, judge,
                    _prior_is_swarm, ev.qa_llm,
                ): (run_idx, result)
                for run_idx, run_results in enumerate(needs_rejudge_by_run)
                for result in run_results
            }
            for future in as_completed(futures):
                run_idx, r = futures[future]
                try:
                    future.result()
                    # Verify the judge actually produced a verdict
                    if r.qa_result is not None or r.insight_result is not None or r.flex_result is not None:
                        status = "correct" if r.qa_result and r.qa_result.correct else "wrong"
                        prefix = f"run {run_idx} " if num_runs > 1 else ""
                        console.print(f"  [dim]→ Re-judged {prefix}{r.case.conv_id}: {status}[/dim]")
                        rejudge_succeeded.append((run_idx, r))
                    else:
                        prefix = f"run {run_idx} " if num_runs > 1 else ""
                        console.print(f"  [yellow]→ Re-judge produced no verdict for {prefix}{r.case.conv_id}, will re-run[/yellow]")
                        rejudge_failed.append((run_idx, r))
                except Exception as exc:
                    prefix = f"run {run_idx} " if num_runs > 1 else ""
                    console.print(f"  [yellow]→ Re-judge failed for {prefix}{r.case.conv_id}: {exc}, will re-run[/yellow]")
                    rejudge_failed.append((run_idx, r))

        # Add successfully re-judged cases back to resumed results,
        # BUT if rerun_wrong is enabled, send wrong cases to the run queue instead.
        for run_idx, result in rejudge_succeeded:
            if ev.rerun_wrong and result_is_wrong(result):
                if cases_by_run is None:
                    cases_by_run = [[] for _ in range(num_runs)]
                cases_by_run[run_idx].append(result.case)
                console.print(
                    f"  [dim]→ Re-judged {result.case.conv_id}: wrong → will re-run[/dim]"
                )
            else:
                resumed_results[run_idx].append(result)

        # Push failed cases back to the run queue for full re-run
        if rejudge_failed:
            if cases_by_run is None:
                cases_by_run = [[] for _ in range(num_runs)]
            for run_idx, result in rejudge_failed:
                cases_by_run[run_idx].append(result.case)
            console.print(
                f"  [yellow]→ {len(rejudge_failed)} cases will be fully re-run[/yellow]"
            )

    # --rejudge: re-run judges on a previous eval run's saved results
    if args.rejudge:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from arcticswarm.eval.runner import judge_result

        rejudge_dir = Path(args.rejudge)
        console.print(f"[bold]Re-judging from[/bold] {rejudge_dir}")

        grouped_results, is_swarm = load_results_for_rejudge(rejudge_dir)
        resolve_reference_answers(grouped_results, csv_path=ev.csv_path or None)

        num_runs = len(grouped_results)
        total_cases = sum(len(r) for r in grouped_results)
        console.print(
            f"  Loaded {total_cases} results across {num_runs} run(s), "
            f"is_swarm={is_swarm}"
        )

        # Re-judge all results
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Re-judging", total=total_cases)

            for run_idx, results in enumerate(grouped_results):
                with ThreadPoolExecutor(max_workers=ev.parallel) as pool:
                    futures = {
                        pool.submit(
                            judge_result, result, judge,
                            is_swarm, ev.qa_llm,
                        ): result
                        for result in results
                    }
                    for future in as_completed(futures):
                        future.result()
                        progress.advance(task_id)

        # Compute metrics and print
        all_reports = [compute_metrics(results) for results in grouped_results]
        for rpt in all_reports:
            rpt.swarm_enabled = is_swarm
            rpt.swarm_scaling_mode = "dynamic" if is_swarm else ""
        if num_runs > 1:
            agg_report = compute_aggregated_report(all_reports)
            _print_summary(console, agg_report, num_runs=num_runs)
        else:
            _print_summary(console, all_reports[0])
            if any("browsecomp" in d.lower() for d in ev.datasets):
                _print_browsecomp_difficulty_breakdown(console, all_reports[0])
        output_dir = Path(ev.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "report.json"
        breakdown_path = output_dir / "breakdown.md"
        if num_runs == 1:
            report_path.write_text(all_reports[0].to_json())
            breakdown_path.write_text(generate_breakdown_md(all_reports[0]))
        else:
            multi = compute_multi_run_report(all_reports)
            report_path.write_text(multi.to_json())
            breakdown_path.write_text(generate_breakdown_md(agg_report))

        console.print(f"  Results written to [bold]{output_dir}[/bold]")
        _print_grounding_fetch_stats(console, grouped_results)
        print_judge_fallback_stats(console, judge)
        console.print()
        os._exit(0)

    # --rebuild-from-trajectories (standalone, without --resume): reconstruct
    # report.json by loading trajectory files, re-judging, and writing a report.
    if ev.rebuild_from_trajectories and not ev.resume:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from arcticswarm.eval.runner import judge_result

        rebuild_dir = Path(ev.output)
        console.print(
            f"[bold]Rebuilding from trajectory files[/bold] in {rebuild_dir / 'trajectories'}"
        )

        grouped_results, is_swarm = load_results_from_trajectories(rebuild_dir)
        resolve_reference_answers(grouped_results, csv_path=ev.csv_path or None)

        num_runs = len(grouped_results)
        total_cases = sum(len(r) for r in grouped_results)
        console.print(
            f"  Loaded {total_cases} results from trajectories, "
            f"is_swarm={is_swarm}"
        )

        # Re-judge all results
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Judging from trajectories", total=total_cases)

            for run_idx, results in enumerate(grouped_results):
                with ThreadPoolExecutor(max_workers=ev.parallel) as pool:
                    futures = {
                        pool.submit(
                            judge_result, result, judge,
                            is_swarm, ev.qa_llm,
                        ): result
                        for result in results
                    }
                    for future in as_completed(futures):
                        future.result()
                        progress.advance(task_id)

        # Compute metrics and print
        all_reports = [compute_metrics(results) for results in grouped_results]
        for rpt in all_reports:
            rpt.swarm_enabled = is_swarm
            rpt.swarm_scaling_mode = "dynamic" if is_swarm else ""
        if num_runs > 1:
            agg_report = compute_aggregated_report(all_reports)
            _print_summary(console, agg_report, num_runs=num_runs)
        else:
            _print_summary(console, all_reports[0])
            if any("browsecomp" in d.lower() for d in ev.datasets):
                _print_browsecomp_difficulty_breakdown(console, all_reports[0])
        _print_grounding_fetch_stats(console, grouped_results)
        output_dir = rebuild_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "report.json"
        breakdown_path = output_dir / "breakdown.md"
        if num_runs == 1:
            report_path.write_text(all_reports[0].to_json())
            breakdown_path.write_text(generate_breakdown_md(all_reports[0]))
        else:
            multi = compute_multi_run_report(all_reports)
            report_path.write_text(multi.to_json())
            breakdown_path.write_text(generate_breakdown_md(agg_report))

        console.print(f"  Results written to [bold]{output_dir}[/bold]")
        print_judge_fallback_stats(console, judge)
        console.print()
        os._exit(0)

    # Output directory setup
    output_dir = Path(ev.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Launch metadata: how many trajectories already on disk, and which
    # resume-mode flags fired this launch. The snapshot writer uses these
    # to disambiguate the original launch from later resumes/reruns and
    # records them inside the JSON for future audits.
    traj_dir = output_dir / "trajectories"
    prior_trajectory_count = (
        sum(1 for p in traj_dir.glob("*.json")) if traj_dir.exists() else 0
    )
    resume_flags = {
        "resume": bool(getattr(ev, "resume", False)),
        "rerun_errors": bool(getattr(ev, "rerun_errors", False)),
        "rerun_wrong": bool(getattr(ev, "rerun_wrong", False)),
        "rerun_timeouts": bool(getattr(ev, "rerun_timeouts", False)),
        "rebuild_from_trajectories": bool(
            getattr(ev, "rebuild_from_trajectories", False),
        ),
    }

    # Capture git state for reproducibility (never overwrites; later
    # launches into the same output_dir append a timestamped sibling).
    capture_git_snapshot(
        output_dir,
        prior_trajectory_count=prior_trajectory_count,
        resume_flags=resume_flags,
    )

    # Dump the fully resolved config for debugging / reproducibility.
    # Same overwrite-safety rule as the code snapshot.
    try:
        import dataclasses as _dc
        from datetime import datetime, timezone
        import yaml as _yaml
        base_resolved = output_dir / "resolved_config.yaml"
        if base_resolved.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            resolved_path = output_dir / f"resolved_config_{ts}.yaml"
        else:
            resolved_path = base_resolved
        resolved_path.write_text(
            _yaml.dump(_dc.asdict(run_cfg), default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
    except Exception:
        pass  # best-effort; don't block the eval

    # Build per-run trajectory directories
    run_output_dirs = [
        output_dir / f"run_{i}" if num_runs > 1 else output_dir
        for i in range(num_runs)
    ]

    # Count resumed cases for progress bar
    resumed_count_per_run = (
        [len(resumed_results[run_idx]) for run_idx in range(num_runs)]
        if resumed_results else [0] * num_runs
    )
    remaining_count_per_run = (
        [len(cases_by_run[run_idx]) for run_idx in range(num_runs)]
        if cases_by_run is not None else [len(cases)] * num_runs
    )
    # Total cases per run = resumed + remaining (for display and checkpoints)
    total_cases_per_run = [
        resumed_count_per_run[run_idx] + remaining_count_per_run[run_idx]
        for run_idx in range(num_runs)
    ]
    total_tasks = sum(total_cases_per_run)
    run_label = (
        f"Evaluation ({num_runs} runs x {total_cases_per_run[0]} cases)"
        if num_runs > 1 and len(set(total_cases_per_run)) == 1
        else (f"Evaluation ({num_runs} runs, {total_tasks} total cases)" if num_runs > 1 else "Running evaluation")
    )
    console.print(
        f"\n[bold]{run_label}[/bold] "
        f"(agent_model={config.model}, parallel={ev.parallel}, timeout={ev.timeout}s, "
        f"max_retries={ev.max_retries}, judge_model={ev.judge_model}, "
        f"config={' '.join(args.config)})"
    )
    if getattr(ev, "stream", True):
        console.print(
            "  [dim]Live feed on: each line is prefixed with its conv_id; "
            "errors in red. Disable with eval.stream=false.[/dim]"
        )
    console.print()

    checkpoint_interval = ev.checkpoint_interval

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(run_label, total=total_tasks)

        # Track results for intermediate metrics (per run)
        accumulated_results: list[list[EvalResult]] = [[] for _ in range(num_runs)]
        _checkpoint_lock = threading.Lock()

        def _save_checkpoint(run_idx: int, completed: int, total_per_run: int) -> None:
            """Save intermediate report.json with metrics computed so far."""
            results_so_far = list(accumulated_results[run_idx])  # snapshot
            if not results_so_far:
                return
            try:
                checkpoint_report = compute_metrics(results_so_far)
                checkpoint_report.swarm_enabled = config.swarm_enabled
                checkpoint_report.swarm_scaling_mode = "dynamic" if config.swarm_enabled else ""
                checkpoint_report.total_elapsed_seconds = sum(
                    r.duration_seconds for r in results_so_far
                )
                report_dict = checkpoint_report.to_dict()
                report_dict["checkpoint"] = {
                    "completed": completed,
                    "total": total_per_run,
                    "timestamp": time.time(),
                }
                report_dir = run_output_dirs[run_idx]
                report_path = report_dir / "report.json"
                # Atomic write: temp file + rename
                fd, tmp_path = tempfile.mkstemp(
                    dir=report_dir, suffix=".json.tmp"
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(report_dict, f, indent=2, default=str)
                    Path(tmp_path).replace(report_path)
                except Exception:
                    Path(tmp_path).unlink(missing_ok=True)
                    raise
                progress.console.print(
                    f"  [dim]→ Checkpoint saved ({completed}/{total_per_run})[/dim]"
                )
            except Exception as exc:
                logger.debug("Checkpoint save failed: %s", exc)

        def on_progress(
            run_idx: int,
            completed: int,
            total_per_run: int,
            result: EvalResult,
        ) -> None:
            # Adjust counters to account for resumed cases
            adj_completed = completed + resumed_count_per_run[run_idx]
            adj_total = total_per_run + resumed_count_per_run[run_idx]

            progress.advance(task_id)
            run_tag = f"run {run_idx}" if num_runs > 1 else ""
            score = _format_score(result)
            status = (
                    "[green]OK[/green]"
                    if not result.error
                    else f"[red]ERR: {result.error[:60]}[/red]"
             )

            retry_info = f" attempt {result.attempt}" if result.attempt > 1 else ""
            tok_info = ""
            if result.token_usage is not None:
                u = result.token_usage
                tok_info = f"  tok={u.input_tokens:,}in+{u.output_tokens:,}out={u.total_tokens:,}"
            progress.console.print(
                f"  [{run_tag + ' ' if run_tag else ''}"
                f"{adj_completed}/{adj_total}] "
                f"{result.case.conv_id[:40]:40s}  "
                f"{status} {score} ({result.duration_seconds:.1f}s{retry_info}){tok_info}"
            )
            save_trajectory(run_output_dirs[run_idx], result)
            save_web_search_failures(
                run_output_dirs[run_idx],
                result.case.conv_id,
                result.web_search_failures,
                trajectory=result.trajectory,
            )
            save_web_search_log(
                run_output_dirs[run_idx],
                result.case.conv_id,
                result.web_search_log,
            )
            save_web_fetch_log(
                run_output_dirs[run_idx],
                result.case.conv_id,
                result.web_fetch_log,
            )
            save_compactor_log(
                run_output_dirs[run_idx],
                result.case.conv_id,
                result.compactor_log,
            )
            save_content_filter_log(
                run_output_dirs[run_idx],
                result.case.conv_id,
                result.content_filter_log,
            )

            # Accumulate results for intermediate metrics
            accumulated_results[run_idx].append(result)

            # Save checkpoint every N cases
            if checkpoint_interval > 0 and adj_completed % checkpoint_interval == 0:
                with _checkpoint_lock:
                    _save_checkpoint(run_idx, adj_completed, adj_total)

            # Print intermediate QA accuracy every N cases (for single-run mode only)
            if num_runs == 1 and checkpoint_interval > 0 and adj_completed % checkpoint_interval == 0:
                results_so_far = accumulated_results[0]
                qa_correct = sum(1 for r in results_so_far if r.qa_result and r.qa_result.correct)
                qa_total = sum(1 for r in results_so_far if r.qa_result is not None)
                if qa_total > 0:
                    qa_accuracy = qa_correct / qa_total
                    progress.console.print(
                        f"  [bold cyan]→ Intermediate QA Accuracy ({adj_completed}/{adj_total}): "
                        f"{qa_accuracy:.1%} ({qa_correct}/{qa_total} correct)[/bold cyan]"
                    )

        t0 = time.monotonic()
        # Reset eval-awareness contamination stats for this run
        reset_contamination_stats()

        # Phase 1: Pre-seed accumulated_results with resumed cases (no re-judging)
        if resumed_results:
            for run_idx in range(min(num_runs, len(resumed_results))):
                for result in resumed_results[run_idx]:
                    accumulated_results[run_idx].append(result)
                    progress.advance(task_id)
                resumed_n = len(resumed_results[run_idx])
                progress.console.print(
                    f"  [dim]→ Loaded {resumed_n} resumed cases for run {run_idx} "
                    f"(judge scores preserved, no re-judging)[/dim]"
                )
                # Save initial checkpoint with resumed data so report.json
                # reflects the full state immediately.
                with _checkpoint_lock:
                    _save_checkpoint(run_idx, resumed_n, total_cases_per_run)

        # Phase 2: Run remaining cases
        if any(remaining_count_per_run):
            # Live per-case activity feed (CLI-style).  Bound to the
            # progress bar's console so feed lines render above the live bar;
            # disabled via ``eval.stream=false``.
            live_logger = (
                LiveEvalLogger(progress.console) if getattr(ev, "stream", True) else None
            )
            new_grouped = run_eval_repeated(
                cases=None if cases_by_run is not None else cases,
                base_config=config,
                judge=judge,
                num_runs=num_runs,
                cases_by_run=cases_by_run,
                parallel=ev.parallel,
                on_progress=on_progress,
                timeout_seconds=ev.timeout,
                max_retries=ev.max_retries,
                use_qa_llm=ev.qa_llm,
                gated_retry=getattr(ev, "gated_retry", False),
                retry_threshold=getattr(ev, "retry_threshold", -0.38),
                max_retry_fraction=getattr(ev, "max_retry_fraction", 0.5),
                on_case_event=live_logger.on_event if live_logger is not None else None,
            )
        else:
            new_grouped = [[] for _ in range(num_runs)]

        elapsed = time.monotonic() - t0

        # Merge resumed + new results
        if resumed_results:
            grouped_results = []
            for run_idx in range(num_runs):
                resumed = resumed_results[run_idx] if run_idx < len(resumed_results) else []
                new = new_grouped[run_idx] if run_idx < len(new_grouped) else []
                grouped_results.append(resumed + new)
        else:
            grouped_results = new_grouped

    # Compute per-run reports.
    #
    # These are built *after* ``run_eval_repeated`` has drained the judge
    # pool, so every per-case verdict (``judge_correct``, ``score``,
    # ``qa_llm_score``) is finalised here.  This is important because the
    # intermediate ``run_N/report.json`` files written by ``_save_checkpoint``
    # inside ``on_progress`` are taken the moment the *agent* finishes a
    # case — the judge may still be running, so the last case in each run
    # gets snapshotted as ``judge=None`` (which ``compute_metrics`` treats
    # as ``score=0.0``) and no further checkpoint is ever written.  We
    # therefore overwrite each ``run_N/report.json`` below with this
    # fully-judged snapshot so per-run files match the top-level aggregate.
    all_reports = [compute_metrics(results) for results in grouped_results]
    for rpt in all_reports:
        # Use sum of per-case durations so per-run files and the embedded
        # ``runs[]`` array in the multi-run report agree on
        # ``total_elapsed_seconds``.  The previous ``elapsed / num_runs``
        # value was the wall-clock time of the *whole* eval divided across
        # runs, which is misleading when runs share a thread pool.
        rpt.total_elapsed_seconds = rpt.overall.total_duration_seconds
        rpt.swarm_enabled = config.swarm_enabled
        rpt.swarm_scaling_mode = "dynamic" if config.swarm_enabled else ""

    # Re-write per-run report.json with the fully-judged snapshot so it
    # supersedes whatever ``_save_checkpoint`` wrote last.  Only meaningful
    # when ``num_runs > 1``: for single-run mode the per-run dir IS the
    # output dir and gets the final report written further down anyway.
    if num_runs > 1:
        for run_idx, rpt in enumerate(all_reports):
            try:
                final_path = run_output_dirs[run_idx] / "report.json"
                fd, tmp_path = tempfile.mkstemp(
                    dir=run_output_dirs[run_idx], suffix=".json.tmp"
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(rpt.to_dict(), f, indent=2, default=str)
                    Path(tmp_path).replace(final_path)
                except Exception:
                    Path(tmp_path).unlink(missing_ok=True)
                    raise
            except Exception as exc:
                logger.warning(
                    "Failed to write final per-run report for run %d: %s",
                    run_idx, exc,
                )

    if num_runs > 1:
        agg_report = compute_aggregated_report(all_reports)
        _print_summary(console, agg_report, num_runs=num_runs)
        _print_performance(console, agg_report, num_runs=num_runs)
    else:
        _print_summary(console, all_reports[0])
        _print_performance(console, all_reports[0])
    _print_grounding_fetch_stats(console, grouped_results)

    console.print(f"  Total time: {elapsed:.1f}s")

    # Print eval-awareness contamination stats
    contamination = get_contamination_stats()
    console.print(f"  {contamination.summary()}")

    # Print swarm saturation summary
    if config.swarm_enabled:
        report_for_sat = all_reports[0]
        sat_data = report_for_sat.to_dict().get("swarm_saturation", {})
        sat_cases = sat_data.get("cases_with_saturation", 0)
        sat_total = sat_data.get("total_cases", 0)
        sat_pct = sat_data.get("saturation_pct", 0.0)
        sat_events = sat_data.get("total_saturation_events", 0)
        console.print(
            f"  Swarm saturation: {sat_cases}/{sat_total} cases ({sat_pct:.1f}%) "
            f"had more pending tasks than available agents "
            f"({sat_events} total saturation events)"
        )

    # Print context compaction summary
    report_for_compact = agg_report if num_runs > 1 else all_reports[0]
    compact_data = report_for_compact.to_dict().get("context_compaction", {})
    if compact_data:
        compact_cases = compact_data.get("cases_compacted", 0)
        compact_total = compact_data.get("total_cases", 0)
        compact_pct = compact_data.get("compaction_pct", 0.0)
        total_compactions = compact_data.get("total_compactions", 0)
        total_llm_calls = compact_data.get("total_llm_calls", 0)
        rate_pct = compact_data.get("compaction_rate_pct", 0.0)
        console.print(
            f"  Context compaction: {compact_cases}/{compact_total} cases ({compact_pct:.1f}%) "
            f"triggered compaction "
            f"({total_compactions}/{total_llm_calls} LLM calls compacted, {rate_pct:.2f}%)"
        )

    # Print reflection summary
    report_for_reflect = agg_report if num_runs > 1 else all_reports[0]
    reflect_data = report_for_reflect.to_dict().get("reflection", {})
    if reflect_data:
        r_cases = reflect_data.get("cases_with_reflection", 0)
        r_calls = reflect_data.get("total_calls", 0)
        r_avg = reflect_data.get("avg_calls_per_case", 0)
        r_suf = reflect_data.get("sufficient_rate", 0)
        r_insuf = reflect_data.get("insufficient_tasks", 0)
        r_conf_pct = reflect_data.get("confidence_pct", {})
        console.print(
            f"  Reflection: {r_cases} cases, {r_calls} total calls "
            f"(avg {r_avg}/case), {r_suf:.0f}% sufficient, "
            f"{r_insuf} tasks exhausted loops"
        )
        console.print(
            f"    Confidence: {r_conf_pct.get('high', 0):.0f}% high, "
            f"{r_conf_pct.get('medium', 0):.0f}% medium, "
            f"{r_conf_pct.get('low', 0):.0f}% low"
        )

        # Print accuracy by reflection call count
        per_case_data = report_for_reflect.to_dict().get("per_case", [])
        from collections import defaultdict
        bucket_correct: dict[int, list[bool]] = defaultdict(list)
        for c in per_case_data:
            r = c.get("reflection", {})
            n_calls = r.get("total_calls", 0)
            correct = c.get("judge_correct")
            if correct is None:
                continue
            bucket_correct[n_calls].append(bool(correct))
        if bucket_correct:
            parts = []
            for n_calls in sorted(bucket_correct.keys()):
                vals = bucket_correct[n_calls]
                acc = sum(vals) / len(vals)
                parts.append(f"{n_calls} calls: {acc:.0%} ({sum(vals)}/{len(vals)})")
            console.print(f"    Accuracy by reflection count: {', '.join(parts)}")

    # Write report
    report_path = output_dir / "report.json"
    breakdown_path = output_dir / "breakdown.md"

    # Embed code snapshot in report for easy access
    snapshot_path = output_dir / "code_snapshot.json"
    code_snapshot = None
    if snapshot_path.exists():
        code_snapshot = json.loads(snapshot_path.read_text())

    if num_runs == 1:
        report_dict = all_reports[0].to_dict()
        report_dict["eval_awareness"] = contamination.to_dict()
        if code_snapshot:
            report_dict["code_snapshot"] = code_snapshot
        report_path.write_text(json.dumps(report_dict, indent=2, default=str))
        breakdown_path.write_text(generate_breakdown_md(all_reports[0]))
    else:
        multi = compute_multi_run_report(all_reports)
        multi_dict = multi.to_dict()
        multi_dict["eval_awareness"] = contamination.to_dict()
        if code_snapshot:
            multi_dict["code_snapshot"] = code_snapshot
        report_path.write_text(json.dumps(multi_dict, indent=2, default=str))
        breakdown_path.write_text(generate_breakdown_md(agg_report))

    if any("browsecomp" in d.lower() for d in ev.datasets):
        _print_browsecomp_difficulty_breakdown(console, report_for_reflect)

    # Print --rerun-errors summary when applicable
    if rerun_error_meta:
        print_rerun_errors_summary(console, rerun_error_meta, report_for_reflect)

    console.print(f"  Results written to [bold]{output_dir}[/bold]")
    print_judge_fallback_stats(console, judge)
    console.print()

    # Force-exit to avoid blocking on Snowflake connector atexit hooks that
    # try to close sessions via network calls which may hang indefinitely.
    os._exit(0)


if __name__ == "__main__":
    main()
