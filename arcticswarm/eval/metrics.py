"""Metrics computation for arcticswarm eval results.

Computes per-dataset aggregate metrics:
  - **QALLMAccuracy**: Average LLM judge score (QA → 0 or 1; Insight → 0, 1, or 2).
  - **AnswerOnlyAccuracy**: Average answer-only judge score (Insight cases only, 0–2).
  - **ToolPrecision**: Fraction of agent tool calls that were expected.
  - **ToolRecall**: Fraction of expected tools that were actually called.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from arcticswarm.eval.runner import EvalResult


# ---------------------------------------------------------------------------
# Per-case tool metrics
# ---------------------------------------------------------------------------


def _tool_precision_recall(
    tools_used: list[str],
    reference_tools: list[str],
) -> tuple[float, float]:
    """Compute precision and recall for tool usage.

    Returns (precision, recall) as floats in [0, 1].
    Both are 1.0 when the sets are empty (vacuous truth).
    """
    if not tools_used and not reference_tools:
        return 1.0, 1.0

    used_set = set(tools_used)
    expected_set = set(reference_tools)

    if not used_set:
        # Agent used no tools but some were expected
        return 1.0, 0.0

    if not expected_set:
        # No expected tools specified → cannot compute recall meaningfully
        return 1.0, 1.0

    tp = len(used_set & expected_set)
    precision = tp / len(used_set) if used_set else 1.0
    recall = tp / len(expected_set) if expected_set else 1.0
    return precision, recall


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class DatasetMetrics:
    """Aggregate metrics for one dataset."""

    dataset: str = ""
    num_cases: int = 0
    num_errors: int = 0
    # NOTE: this counter sums ``compaction_count`` across cases — both
    # proactive (utilization ≥ threshold) and reactive (context-too-long
    # error) compactions.  Despite the legacy name, it is NOT a count of
    # prompt-too-long failures.  See O6 in CONSOLIDATED_CODE_TODOS.md.
    # ``num_compaction_events`` is the new, accurate name; the legacy
    # ``num_prompt_too_long_errors`` is preserved as an alias.
    num_prompt_too_long_errors: int = 0
    num_compaction_events: int = 0
    # Number of questions where at least one safety refusal was detected
    num_safety_refusal_questions: int = 0
    # Number of questions where Azure content filter blocked a call
    num_content_filter_questions: int = 0

    # LLM accuracy
    qa_llm_accuracy: float = 0.0  # average score across all cases
    qa_llm_accuracy_stddev: float | None = None

    # Answer-only accuracy (Insight cases only; ignores methodology/tools)
    answer_only_accuracy: float = 0.0
    answer_only_accuracy_stddev: float | None = None

    # Tool precision / recall (macro-averaged)
    tool_precision: float = 0.0
    tool_precision_stddev: float | None = None
    tool_recall: float = 0.0
    tool_recall_stddev: float | None = None

    # Unit-test accuracy (verifiable datasets only; -1 means N/A)
    unit_test_accuracy: float = -1.0
    unit_test_accuracy_stddev: float | None = None

    # FLEX judge metrics (-1.0 = N/A / not evaluated)
    flex_answer_accuracy: float = -1.0
    flex_answer_accuracy_stddev: float | None = None
    answer_groundedness: float = -1.0
    answer_groundedness_stddev: float | None = None
    answer_relevancy: float = -1.0
    answer_relevancy_stddev: float | None = None
    methodology_soundness: float = -1.0
    methodology_soundness_stddev: float | None = None

    # Timing
    total_duration_seconds: float = 0.0
    avg_duration_seconds: float = 0.0
    avg_duration_seconds_stddev: float | None = None
    p50_duration_seconds: float = 0.0
    p90_duration_seconds: float = 0.0

    # Steps / tool calls (per-question averages)
    avg_steps: float = 0.0
    avg_tool_calls_per_step: float = 0.0

    # Latency breakdown (averages across cases, seconds)
    avg_latency_breakdown: dict[str, float] = field(default_factory=dict)

    # Token breakdown — per-turn attribution (from _llm_output_tokens on assistant msgs)
    avg_output_tokens_other_tool_turns: float = 0.0
    avg_output_tokens_text_turns: float = 0.0

    # Tool call distribution (avg per case, keyed by tool name)
    avg_tool_call_distribution: dict[str, float] = field(default_factory=dict)

    # Token usage (averages across cases)
    avg_input_tokens: float = 0.0
    avg_input_tokens_stddev: float | None = None
    avg_output_tokens: float = 0.0
    avg_output_tokens_stddev: float | None = None
    avg_cache_read_tokens: float = 0.0
    avg_cache_read_tokens_stddev: float | None = None
    avg_cache_creation_tokens: float = 0.0
    avg_total_tokens: float = 0.0
    avg_total_tokens_stddev: float | None = None
    avg_reasoning_tokens: float = 0.0
    avg_reasoning_tokens_stddev: float | None = None
    avg_total_token_e2e: float = 0.0
    avg_total_token_e2e_stddev: float | None = None

    # Bird-Interact phase completion rates (-1.0 = N/A for non-BI datasets)
    bi_phase1_rate: float = -1.0
    bi_phase1_rate_stddev: float | None = None
    bi_phase2_rate: float = -1.0
    bi_phase2_rate_stddev: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise with human-friendly key names for steps/turns."""
        d: dict[str, Any] = {
            "dataset": self.dataset,
            "num_cases": self.num_cases,
            "num_errors": self.num_errors,
            "num_prompt_too_long_errors": self.num_prompt_too_long_errors,
            "num_compaction_events": self.num_compaction_events,
            "num_safety_refusal_questions": self.num_safety_refusal_questions,
            "num_content_filter_questions": self.num_content_filter_questions,
            "qa_llm_accuracy": self.qa_llm_accuracy,
            "answer_only_accuracy": "N/A" if self.answer_only_accuracy == -1.0 else self.answer_only_accuracy,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "unit_test_accuracy": self.unit_test_accuracy,
            "flex_answer_accuracy": "N/A" if self.flex_answer_accuracy == -1.0 else self.flex_answer_accuracy,
            "answer_groundedness": "N/A" if self.answer_groundedness == -1.0 else self.answer_groundedness,
            "answer_relevancy": "N/A" if self.answer_relevancy == -1.0 else self.answer_relevancy,
            "methodology_soundness": "N/A" if self.methodology_soundness == -1.0 else self.methodology_soundness,
            "total_duration_seconds": self.total_duration_seconds,
            "avg_duration_seconds": self.avg_duration_seconds,
            "p50_duration_seconds": self.p50_duration_seconds,
            "p90_duration_seconds": self.p90_duration_seconds,
            "avg_steps(turns)": self.avg_steps,
            "avg_tool_calls_per_step(turn)": self.avg_tool_calls_per_step,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "avg_cache_read_tokens": self.avg_cache_read_tokens,
            "avg_cache_creation_tokens": self.avg_cache_creation_tokens,
            "avg_reasoning_tokens": self.avg_reasoning_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "bi_phase1_rate": "N/A" if self.bi_phase1_rate < 0 else self.bi_phase1_rate,
            "bi_phase2_rate": "N/A" if self.bi_phase2_rate < 0 else self.bi_phase2_rate,
        }
        if self.avg_total_token_e2e > 0:
            d["avg_total_token_e2e"] = self.avg_total_token_e2e
        if self.avg_latency_breakdown:
            d["avg_latency_breakdown"] = self.avg_latency_breakdown
        # Token breakdown fields
        d["avg_token_breakdown"] = {
            "output_tokens_other_tool_turns": self.avg_output_tokens_other_tool_turns,
            "output_tokens_text_turns": self.avg_output_tokens_text_turns,
        }
        if self.avg_tool_call_distribution:
            d["avg_tool_call_distribution"] = self.avg_tool_call_distribution
        for field_name in (
            "qa_llm_accuracy", "answer_only_accuracy",
            "tool_precision", "tool_recall", "unit_test_accuracy",
            "flex_answer_accuracy", "answer_groundedness",
            "answer_relevancy", "methodology_soundness",
            "bi_phase1_rate", "bi_phase2_rate",
            "avg_duration_seconds",
            "avg_input_tokens", "avg_output_tokens",
            "avg_reasoning_tokens",
            "avg_cache_read_tokens", "avg_total_tokens",
            "avg_total_token_e2e",
        ):
            stddev = getattr(self, f"{field_name}_stddev")
            if stddev is not None:
                d[f"{field_name}_stddev"] = stddev
        return d


@dataclass
class EvalReport:
    """Full evaluation report."""

    total_cases: int = 0
    total_errors: int = 0
    # Legacy: total compaction events across all cases.  See O6.
    total_prompt_too_long_errors: int = 0
    # Same value, accurate name.  Both are written so consumers can migrate
    # at their own pace.
    total_compaction_events: int = 0
    total_elapsed_seconds: float = 0.0
    swarm_enabled: bool = False
    swarm_scaling_mode: str = ""  # always "dynamic"
    datasets: dict[str, DatasetMetrics] = field(default_factory=dict)
    overall: DatasetMetrics = field(default_factory=DatasetMetrics)
    per_case: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "total_cases": self.total_cases,
            "total_errors": self.total_errors,
            "total_prompt_too_long_errors": self.total_prompt_too_long_errors,
            "total_compaction_events": self.total_compaction_events,
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 1),
            "swarm_enabled": self.swarm_enabled,
            "swarm_scaling_mode": self.swarm_scaling_mode,
            "overall": self.overall.to_dict(),
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "per_case": self.per_case,
        }
        # Add swarm saturation summary when available
        if self.swarm_enabled:
            saturated = sum(
                1 for c in self.per_case
                if c.get("swarm_saturation_events", 0) > 0
            )
            total_sat_events = sum(
                c.get("swarm_saturation_events", 0) for c in self.per_case
            )
            d["swarm_saturation"] = {
                "cases_with_saturation": saturated,
                "total_cases": self.total_cases,
                "saturation_pct": round(
                    (saturated / self.total_cases * 100) if self.total_cases > 0 else 0.0, 1,
                ),
                "total_saturation_events": total_sat_events,
            }

            # Add aggregate reflection stats
            cases_with_reflection = [
                c for c in self.per_case if c.get("reflection", {}).get("total_calls", 0) > 0
            ]
            if cases_with_reflection:
                total_r_calls = sum(c["reflection"]["total_calls"] for c in cases_with_reflection)
                total_sufficient = sum(c["reflection"]["sufficient"] for c in cases_with_reflection)
                total_insufficient = sum(c["reflection"]["insufficient"] for c in cases_with_reflection)
                agg_conf = {"low": 0, "medium": 0, "high": 0}
                for c in cases_with_reflection:
                    for k, v in c["reflection"].get("confidence_distribution", {}).items():
                        agg_conf[k] = agg_conf.get(k, 0) + v
                total_conf = sum(agg_conf.values()) or 1
                d["reflection"] = {
                    "cases_with_reflection": len(cases_with_reflection),
                    "total_calls": total_r_calls,
                    "avg_calls_per_case": round(total_r_calls / len(cases_with_reflection), 1),
                    "sufficient_rate": round(total_sufficient / total_r_calls * 100, 1) if total_r_calls else 0,
                    "insufficient_tasks": total_insufficient,
                    "confidence_distribution": agg_conf,
                    "confidence_pct": {
                        k: round(v / total_conf * 100, 1)
                        for k, v in agg_conf.items()
                    },
                }

        # Add context compaction summary when any compaction occurred
        total_compactions = sum(
            c.get("compaction_count", 0) for c in self.per_case
        )
        total_llm_calls = sum(
            c.get("total_llm_calls", 0) for c in self.per_case
        )
        cases_compacted = sum(
            1 for c in self.per_case if c.get("compaction_count", 0) > 0
        )
        if total_compactions > 0:
            d["context_compaction"] = {
                "cases_compacted": cases_compacted,
                "total_cases": self.total_cases,
                "compaction_pct": round(
                    (cases_compacted / self.total_cases * 100) if self.total_cases > 0 else 0.0, 1,
                ),
                "total_compactions": total_compactions,
                "total_llm_calls": total_llm_calls,
                "compaction_rate_pct": round(
                    (total_compactions / total_llm_calls * 100) if total_llm_calls > 0 else 0.0, 2,
                ),
            }

        # Discovery accuracy (force discovery mode)
        discovery_cases = [c for c in self.per_case if c.get("discovery_expected_models")]
        if discovery_cases:
            matched = sum(1 for c in discovery_cases if c.get("discovery_match"))
            d["discovery_accuracy"] = {
                "cases_with_expected_model": len(discovery_cases),
                "correct_view_found": matched,
                "accuracy": round(matched / len(discovery_cases), 3) if discovery_cases else 0,
            }

        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------


def _score_for_result(result: EvalResult) -> float:
    """Return the numeric judge score for a single result."""
    if result.insight_result is not None:
        return float(result.insight_result.rating)
    if result.qa_result is not None:
        return 1.0 if result.qa_result.correct else 0.0
    return 0.0


def _answer_only_score(result: EvalResult) -> float | None:
    """Return the answer-only judge score, or ``None`` if not evaluated."""
    if result.answer_only_result is not None:
        return float(result.answer_only_result.rating)
    return None


def _flex_scores(result: EvalResult) -> tuple[float, float, float, float] | None:
    """Return (accuracy, groundedness, relevancy, soundness) from FLEX result.

    Returns ``None`` if FLEX was not evaluated.  Preserves ``-1.0`` sentinels
    for N/A process metrics (swarm mode).
    """
    fr = result.flex_result
    if fr is None:
        return None
    return (
        float(fr.flex_answer_accuracy),
        fr.answer_groundedness,
        float(fr.answer_relevancy),
        float(fr.methodology_soundness),
    )


def _extract_bi_phase_completion(result: EvalResult) -> tuple[bool, bool]:
    """Extract Bird-Interact phase1/phase2 completion from submit_sql tool metadata."""
    phase1 = False
    phase2 = False
    for traj_entry in result.trajectory:
        messages = traj_entry.get("messages", []) if isinstance(traj_entry, dict) else []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                meta = block.get("metadata") or {}
                if meta.get("passed") is True:
                    phase = meta.get("phase")
                    if phase == 1:
                        phase1 = True
                    elif phase == 2:
                        phase2 = True
    return phase1, phase2


def _case_summary(result: EvalResult) -> dict[str, Any]:
    """Build a per-case dict for the report."""
    summary: dict[str, Any] = {
        "conv_id": result.case.conv_id,
        "dataset": result.case.dataset,
        "eval_mode": result.case.eval_mode,
        "question": result.case.question,  # Full question (no truncation)
        "reference_answer": result.case.reference_answer,
        "response_text": result.response_text,  # Full response (no truncation)
        "tools_used": result.tools_used,
        "reference_tools": result.case.reference_tools,
        "duration_seconds": round(result.duration_seconds, 2),
        "score": _score_for_result(result),
        "num_steps(turns)": result.num_steps,
        "num_tool_calls": len(result.tool_calls),
        "total_tokens": result.total_tokens,
    }
    # Surface difficulty annotation for BrowseComp breakdown reporting
    if result.case.attributes and "difficulty" in result.case.attributes:
        summary["difficulty"] = result.case.attributes["difficulty"]
    if result.total_token_e2e > 0:
        summary["total_token_e2e"] = result.total_token_e2e

    if result.swarm_teammates_spawned > 0:
        summary["swarm_teammates_spawned"] = result.swarm_teammates_spawned
        summary["swarm_bbs_message_count"] = result.swarm_bbs_message_count
        summary["swarm_saturation_events"] = result.swarm_saturation_events

    if result.swarm_subagent_tool_counts:
        summary["swarm_subagent_tool_counts"] = result.swarm_subagent_tool_counts

    if result.swarm_spawn_events:
        summary["swarm_spawn_events"] = result.swarm_spawn_events

    if result.swarm_token_usage_breakdown:
        summary["swarm_token_usage_breakdown"] = result.swarm_token_usage_breakdown

    if result.swarm_reflection_stats:
        summary["reflection"] = result.swarm_reflection_stats

    if getattr(result, "swarm_rival_audit", None):
        summary["rival_audit"] = result.swarm_rival_audit

    # Telemetry surfaced for the gated-retry detector.
    if getattr(result, "swarm_layer4a", None):
        summary["layer4a"] = result.swarm_layer4a
    if getattr(result, "swarm_cheap_win_fired", False):
        summary["cheap_win_fired"] = True
    if getattr(result, "swarm_rival_sweep_fired", False):
        summary["rival_sweep_fired"] = True

    if result.phase_timings:
        summary["phase_timings"] = result.phase_timings

    if result.latency_breakdown:
        summary["latency_breakdown"] = result.latency_breakdown

    if result.error:
        summary["error"] = result.error

    if result.compaction_count > 0:
        summary["compaction_count"] = result.compaction_count
        summary["total_llm_calls"] = result.total_llm_calls

    # O2: surface proactive vs reactive split + peak context size when
    # any compaction or peak signal was recorded.  Both fields default
    # to 0 on EvalResult, so the gating preserves clean per_case output
    # for non-compactor-equipped runs.
    if getattr(result, "proactive_compaction_count", 0) > 0:
        summary["proactive_compaction_count"] = result.proactive_compaction_count
    if getattr(result, "reactive_compaction_count", 0) > 0:
        summary["reactive_compaction_count"] = result.reactive_compaction_count
    peak_input = getattr(result, "peak_input_tokens", 0)
    if peak_input > 0:
        summary["peak_input_tokens"] = peak_input

    if result.safety_refusal_count > 0:
        summary["safety_refusal_count"] = result.safety_refusal_count

    # Distinguishes "intermediate refusals → recovered final answer" from
    # "final answer is itself a refusal".  Set unconditionally (also when
    # False) so consumers don't have to guess about absence vs zero.
    summary["final_answer_is_refusal"] = bool(
        getattr(result, "final_answer_is_refusal", False)
    )

    if result.content_filter_count > 0:
        summary["content_filter_count"] = result.content_filter_count

    # O5 (Q7): expose thinking-only count when non-zero — lets us verify
    # Q3 (disable_extended_thinking) is suppressing the failure mode it targets.
    thinking_only = getattr(result, "thinking_only_count", 0)
    if thinking_only > 0:
        summary["thinking_only_count"] = thinking_only

    # O4: surface multi-attempt timeouts that were silently absorbed by
    # Layer-1/3 force-report recovery.  Without this you cannot tell
    # 12624s slow-but-clean from 6300+6300+force-reported.
    timeout_attempts = getattr(result, "timeout_attempts", 0)
    if timeout_attempts > 0:
        summary["timeout_attempts"] = timeout_attempts
    recovery_mode = getattr(result, "recovery_mode", "")
    if recovery_mode:
        summary["recovery_mode"] = recovery_mode

    if result.qa_result:
        summary["judge_mode"] = "QA"
        summary["judge_correct"] = result.qa_result.correct
        summary["judge_comment"] = result.qa_result.comment
        if result.qa_result.raw_output:
            summary["judge_raw_output"] = result.qa_result.raw_output[:16000]
        if result.qa_result.judge_confidence is not None:
            summary["hle_judge_confidence"] = result.qa_result.judge_confidence
        if result.qa_result.extracted_final_answer is not None:
            summary["hle_extracted_final_answer"] = result.qa_result.extracted_final_answer

    if result.insight_result:
        summary["judge_mode"] = "INSIGHT"
        summary["judge_rating"] = result.insight_result.rating
        summary["judge_analysis"] = result.insight_result.analysis[:500]
        summary["judge_reasoning"] = result.insight_result.reasoning

    if result.answer_only_result:
        summary["answer_only_rating"] = result.answer_only_result.rating
        summary["answer_only_analysis"] = result.answer_only_result.analysis[:500]

    if result.token_usage is not None:
        summary["token_usage"] = {
            "input_tokens": result.token_usage.input_tokens,
            "output_tokens": result.token_usage.output_tokens,
            "cache_creation_input_tokens": result.token_usage.cache_creation_input_tokens,
            "cache_read_input_tokens": result.token_usage.cache_read_input_tokens,
            "reasoning_tokens": result.token_usage.reasoning_tokens,
        }
    if result.token_breakdown:
        summary["token_breakdown"] = result.token_breakdown
    if result.tool_call_distribution:
        summary["tool_call_distribution"] = result.tool_call_distribution
    if result.unit_test_score is not None:
        summary["unit_test_score"] = result.unit_test_score
    if result.unit_test_extracted_json is not None:
        summary["unit_test_extracted_json"] = result.unit_test_extracted_json

    if result.flex_result is not None:
        summary["flex_answer_accuracy"] = result.flex_result.flex_answer_accuracy
        summary["answer_groundedness"] = result.flex_result.answer_groundedness
        summary["answer_relevancy"] = result.flex_result.answer_relevancy
        summary["methodology_soundness"] = result.flex_result.methodology_soundness
        summary["accuracy_reasoning"] = result.flex_result.accuracy_reasoning
        # Pre-analysis metadata (mirrors Go's QualityScoreResult)
        summary["order_sensitivity"] = result.flex_result.order_sensitivity
        summary["question_parts"] = result.flex_result.question_parts
        summary["question_type"] = result.flex_result.question_type
        summary["acceptable_alternatives"] = result.flex_result.acceptable_alternatives
        summary["expected_facts"] = result.flex_result.expected_facts
        summary["response_facts"] = result.flex_result.response_facts
    else:
        summary["flex_answer_accuracy"] = -1.0
        summary["answer_groundedness"] = -1.0
        summary["answer_relevancy"] = -1.0
        summary["methodology_soundness"] = -1.0
        summary["accuracy_reasoning"] = ""

    # Bird-Interact phase completion
    if result.case.dataset == "BIRD_INTERACT":
        p1, p2 = _extract_bi_phase_completion(result)
        summary["bi_phase1_completed"] = p1
        summary["bi_phase2_completed"] = p2

    return summary


def _populate_performance_fields(
    dm: DatasetMetrics,
    durations: list[float],
    steps: list[int],
    tool_call_counts: list[int],
    input_tokens: list[float],
    output_tokens: list[float],
    cache_read_tokens: list[float],
    cache_creation_tokens: list[float],
    total_tokens: list[float],
    total_token_e2e: list[float] | None = None,
    reasoning_tokens: list[float] | None = None,
) -> None:
    """Fill the latency-percentile, step, tool-call, and token fields on *dm*."""
    if durations:
        sorted_durations = sorted(durations)
        dm.p50_duration_seconds = _percentile(sorted_durations, 50)
        dm.p90_duration_seconds = _percentile(sorted_durations, 90)

    if steps:
        dm.avg_steps = _safe_mean([float(s) for s in steps])
        total_steps = sum(steps)
        if total_steps > 0 and tool_call_counts:
            dm.avg_tool_calls_per_step = sum(tool_call_counts) / total_steps

    if input_tokens:
        dm.avg_input_tokens = _safe_mean(input_tokens)
    if output_tokens:
        dm.avg_output_tokens = _safe_mean(output_tokens)
    if reasoning_tokens:
        dm.avg_reasoning_tokens = _safe_mean(reasoning_tokens)
    if cache_read_tokens:
        dm.avg_cache_read_tokens = _safe_mean(cache_read_tokens)
    if cache_creation_tokens:
        dm.avg_cache_creation_tokens = _safe_mean(cache_creation_tokens)
    if total_tokens:
        dm.avg_total_tokens = _safe_mean(total_tokens)
    if total_token_e2e:
        dm.avg_total_token_e2e = _safe_mean(total_token_e2e)


def _aggregate_latency_breakdowns(breakdowns: list[dict[str, float]]) -> dict[str, float]:
    """Average per-category latency across cases, preserving a stable key order."""
    if not breakdowns:
        return {}
    all_keys: list[str] = []
    for bd in breakdowns:
        for k in bd:
            if k not in all_keys:
                all_keys.append(k)
    key_order = ["llm_planning"] + [k for k in all_keys if k not in ("llm_planning", "overhead")] + ["overhead"]
    key_order = [k for k in key_order if k in all_keys]
    result: dict[str, float] = {}
    for k in key_order:
        vals = [bd.get(k, 0.0) for bd in breakdowns]
        result[k] = round(_safe_mean(vals), 3)
    return result


_TOKEN_BREAKDOWN_KEYS = [
    "output_tokens_other_tool_turns",
    "output_tokens_text_turns",
]

_TOKEN_BREAKDOWN_FIELD_MAP = {
    "output_tokens_other_tool_turns": "avg_output_tokens_other_tool_turns",
    "output_tokens_text_turns": "avg_output_tokens_text_turns",
}


def _populate_token_breakdown(
    dm: DatasetMetrics,
    breakdowns: list[dict[str, float]],
    tool_distributions: list[dict[str, int]],
) -> None:
    """Aggregate per-case token breakdowns and tool call distributions onto *dm*."""
    if breakdowns:
        for key in _TOKEN_BREAKDOWN_KEYS:
            vals = [bd.get(key, 0.0) for bd in breakdowns]
            setattr(dm, _TOKEN_BREAKDOWN_FIELD_MAP[key], _safe_mean(vals))

    if tool_distributions:
        all_tools: set[str] = set()
        for td in tool_distributions:
            all_tools.update(td.keys())
        dm.avg_tool_call_distribution = {
            tool: _safe_mean([float(td.get(tool, 0)) for td in tool_distributions])
            for tool in sorted(all_tools)
        }


def compute_metrics(results: list[EvalResult]) -> EvalReport:
    """Compute aggregate metrics from a list of :class:`EvalResult`."""
    report = EvalReport(total_cases=len(results))

    # Group by dataset
    by_dataset: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        ds = r.case.dataset or "(unknown)"
        by_dataset[ds].append(r)
        report.per_case.append(_case_summary(r))
        if r.error:
            report.total_errors += 1
        report.total_prompt_too_long_errors += r.compaction_count
        report.total_compaction_events += r.compaction_count

    # Per-dataset metrics
    all_scores: list[float] = []
    all_answer_only: list[float] = []
    all_precisions: list[float] = []
    all_recalls: list[float] = []
    all_durations: list[float] = []
    all_steps: list[int] = []
    all_tool_call_counts: list[int] = []
    all_input_tokens: list[float] = []
    all_output_tokens: list[float] = []
    all_reasoning_tokens: list[float] = []
    all_cache_read_tokens: list[float] = []
    all_cache_creation_tokens: list[float] = []
    all_total_tokens: list[float] = []
    all_total_token_e2e: list[float] = []
    all_unit_test: list[float] = []
    all_breakdowns: list[dict[str, float]] = []
    all_token_breakdowns: list[dict[str, float]] = []
    all_tool_distributions: list[dict[str, int]] = []
    all_flex_acc: list[float] = []
    all_flex_gnd: list[float] = []
    all_flex_rel: list[float] = []
    all_flex_snd: list[float] = []
    all_bi_phase1: list[float] = []
    all_bi_phase2: list[float] = []

    for ds_name, ds_results in sorted(by_dataset.items()):
        scores: list[float] = []
        answer_only_scores: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        durations: list[float] = []
        ds_steps: list[int] = []
        ds_tool_calls: list[int] = []
        ds_input_tok: list[float] = []
        ds_output_tok: list[float] = []
        ds_reasoning_tok: list[float] = []
        ds_cache_read_tok: list[float] = []
        ds_cache_creation_tok: list[float] = []
        ds_total_tok: list[float] = []
        ds_total_token_e2e: list[float] = []
        unit_test_scores: list[float] = []
        ds_breakdowns: list[dict[str, float]] = []
        ds_token_breakdowns: list[dict[str, float]] = []
        ds_tool_distributions: list[dict[str, int]] = []
        ds_flex_acc: list[float] = []
        ds_flex_gnd: list[float] = []
        ds_flex_rel: list[float] = []
        ds_flex_snd: list[float] = []
        ds_bi_phase1: list[float] = []
        ds_bi_phase2: list[float] = []
        num_errors = 0

        for r in ds_results:
            scores.append(_score_for_result(r))
            ao = _answer_only_score(r)
            # Use -1.0 as sentinel for "not applicable" (e.g., canonical-metric-only benchmarks)
            answer_only_scores.append(ao if ao is not None else -1.0)
            p, rc = _tool_precision_recall(r.tools_used, r.case.reference_tools)
            precisions.append(p)
            recalls.append(rc)
            ds_steps.append(r.num_steps)
            ds_tool_calls.append(len(r.tool_calls))
            has_perf = r.duration_seconds > 0 or r.token_usage is not None
            if has_perf:
                durations.append(r.duration_seconds)
                if r.token_usage is not None:
                    ds_input_tok.append(float(r.token_usage.input_tokens))
                    ds_output_tok.append(float(r.token_usage.output_tokens))
                    ds_reasoning_tok.append(float(r.token_usage.reasoning_tokens))
                    ds_cache_read_tok.append(float(r.token_usage.cache_read_input_tokens))
                    ds_cache_creation_tok.append(float(r.token_usage.cache_creation_input_tokens))
                    ds_total_tok.append(float(r.total_tokens))
                if r.total_token_e2e > 0:
                    ds_total_token_e2e.append(float(r.total_token_e2e))
            if r.latency_breakdown:
                ds_breakdowns.append(r.latency_breakdown)
            if r.token_breakdown:
                ds_token_breakdowns.append(r.token_breakdown)
            if r.tool_call_distribution:
                ds_tool_distributions.append(r.tool_call_distribution)
            if r.case.unit_test:
                unit_test_scores.append(r.unit_test_score if r.unit_test_score is not None else 0.0)
            fs = _flex_scores(r)
            # Use -1.0 sentinel for cases without a FLEX result (errored /
            # unjudged). _safe_mean_with_na will count these as 0 when FLEX is
            # active for the dataset, or preserve N/A when no case was judged.
            if fs is None:
                ds_flex_acc.append(-1.0)
                ds_flex_gnd.append(-1.0)
                ds_flex_rel.append(-1.0)
                ds_flex_snd.append(-1.0)
            else:
                ds_flex_acc.append(fs[0])
                ds_flex_gnd.append(fs[1])
                ds_flex_rel.append(fs[2])
                ds_flex_snd.append(fs[3])
            if ds_name == "BIRD_INTERACT":
                p1, p2 = _extract_bi_phase_completion(r)
                ds_bi_phase1.append(1.0 if p1 else 0.0)
                ds_bi_phase2.append(1.0 if p2 else 0.0)
            if r.error:
                num_errors += 1

        ds_prompt_too_long = sum(r.compaction_count for r in ds_results)
        ds_safety_refusal_questions = sum(
            1 for r in ds_results if r.safety_refusal_count > 0
        )
        ds_content_filter_questions = sum(
            1 for r in ds_results if r.content_filter_count > 0
        )

        dm = DatasetMetrics(
            dataset=ds_name,
            num_cases=len(ds_results),
            num_errors=num_errors,
            num_prompt_too_long_errors=ds_prompt_too_long,
            num_compaction_events=ds_prompt_too_long,
            num_safety_refusal_questions=ds_safety_refusal_questions,
            num_content_filter_questions=ds_content_filter_questions,
            qa_llm_accuracy=_safe_mean(scores),
            answer_only_accuracy=_safe_mean_with_na(answer_only_scores),
            tool_precision=_safe_mean(precisions),
            tool_recall=_safe_mean(recalls),
            total_duration_seconds=sum(durations),
            avg_duration_seconds=_safe_mean(durations),
            unit_test_accuracy=_safe_mean(unit_test_scores) if unit_test_scores else -1.0,
            flex_answer_accuracy=_safe_mean_with_na(ds_flex_acc) if ds_flex_acc else -1.0,
            answer_groundedness=_safe_mean_with_na(ds_flex_gnd) if ds_flex_gnd else -1.0,
            answer_relevancy=_safe_mean_with_na(ds_flex_rel) if ds_flex_rel else -1.0,
            methodology_soundness=_safe_mean_with_na(ds_flex_snd) if ds_flex_snd else -1.0,
        )
        _populate_performance_fields(
            dm, durations, ds_steps, ds_tool_calls,
            ds_input_tok, ds_output_tok, ds_cache_read_tok, ds_cache_creation_tok,
            ds_total_tok, ds_total_token_e2e, ds_reasoning_tok,
        )
        dm.avg_latency_breakdown = _aggregate_latency_breakdowns(ds_breakdowns)
        _populate_token_breakdown(dm, ds_token_breakdowns, ds_tool_distributions)
        if ds_bi_phase1:
            dm.bi_phase1_rate = _safe_mean(ds_bi_phase1)
            dm.bi_phase2_rate = _safe_mean(ds_bi_phase2)
        report.datasets[ds_name] = dm

        all_breakdowns.extend(ds_breakdowns)
        all_token_breakdowns.extend(ds_token_breakdowns)
        all_tool_distributions.extend(ds_tool_distributions)
        all_scores.extend(scores)
        all_answer_only.extend(answer_only_scores)
        all_precisions.extend(precisions)
        all_recalls.extend(recalls)
        all_durations.extend(durations)
        all_steps.extend(ds_steps)
        all_tool_call_counts.extend(ds_tool_calls)
        all_input_tokens.extend(ds_input_tok)
        all_output_tokens.extend(ds_output_tok)
        all_reasoning_tokens.extend(ds_reasoning_tok)
        all_cache_read_tokens.extend(ds_cache_read_tok)
        all_cache_creation_tokens.extend(ds_cache_creation_tok)
        all_total_tokens.extend(ds_total_tok)
        all_total_token_e2e.extend(ds_total_token_e2e)
        all_unit_test.extend(unit_test_scores)
        all_flex_acc.extend(ds_flex_acc)
        all_flex_gnd.extend(ds_flex_gnd)
        all_flex_rel.extend(ds_flex_rel)
        all_flex_snd.extend(ds_flex_snd)
        all_bi_phase1.extend(ds_bi_phase1)
        all_bi_phase2.extend(ds_bi_phase2)

    # Overall metrics
    report.overall = DatasetMetrics(
        dataset="(overall)",
        num_cases=len(results),
        num_errors=report.total_errors,
        num_prompt_too_long_errors=report.total_prompt_too_long_errors,
        num_compaction_events=report.total_compaction_events,
        num_safety_refusal_questions=sum(
            1 for r in results if r.safety_refusal_count > 0
        ),
        num_content_filter_questions=sum(
            1 for r in results if r.content_filter_count > 0
        ),
        qa_llm_accuracy=_safe_mean(all_scores),
        answer_only_accuracy=_safe_mean_with_na(all_answer_only),
        tool_precision=_safe_mean(all_precisions),
        tool_recall=_safe_mean(all_recalls),
        total_duration_seconds=sum(all_durations),
        avg_duration_seconds=_safe_mean(all_durations),
        unit_test_accuracy=_safe_mean(all_unit_test) if all_unit_test else -1.0,
        flex_answer_accuracy=_safe_mean_with_na(all_flex_acc) if all_flex_acc else -1.0,
        answer_groundedness=_safe_mean_with_na(all_flex_gnd) if all_flex_gnd else -1.0,
        answer_relevancy=_safe_mean_with_na(all_flex_rel) if all_flex_rel else -1.0,
        methodology_soundness=_safe_mean_with_na(all_flex_snd) if all_flex_snd else -1.0,
    )
    _populate_performance_fields(
        report.overall, all_durations, all_steps, all_tool_call_counts,
        all_input_tokens, all_output_tokens, all_cache_read_tokens,
        all_cache_creation_tokens, all_total_tokens, all_total_token_e2e,
        all_reasoning_tokens,
    )
    report.overall.avg_latency_breakdown = _aggregate_latency_breakdowns(all_breakdowns)
    _populate_token_breakdown(report.overall, all_token_breakdowns, all_tool_distributions)
    if all_bi_phase1:
        report.overall.bi_phase1_rate = _safe_mean(all_bi_phase1)
        report.overall.bi_phase2_rate = _safe_mean(all_bi_phase2)

    return report


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_mean_with_na(values: list[float]) -> float:
    """Compute mean across per-case scores, honouring the -1.0 "not applicable"
    sentinel and counting missing/errored cases as 0.

    Rules:
    - Empty list → 0.0 (degenerate).
    - All values are -1.0 (judge not active for this dataset/mode) → -1.0 (N/A).
    - Otherwise the metric IS applicable: any -1.0 entry is treated as 0
      (errored case or unjudged case counts against accuracy). The denominator
      is always ``len(values)``.

    This avoids silently shrinking the denominator when some cases failed to
    produce a judge result — a timeout or empty response should count against
    the benchmark, not be excluded from it.
    """
    if not values:
        return 0.0
    if all(v == -1.0 for v in values):
        return -1.0
    return sum(0.0 if v == -1.0 else v for v in values) / len(values)


def _safe_stddev(values: list[float]) -> float:
    """Sample standard deviation; returns 0 if fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _percentile(sorted_values: list[float], p: int) -> float:
    """Compute the p-th percentile using the same index-based method as
    agent-eval-e2e's ``calculatePercentile`` in Go."""
    if not sorted_values:
        return 0.0
    index = (p * len(sorted_values)) // 100
    if index >= len(sorted_values):
        index = len(sorted_values) - 1
    return sorted_values[index]


# ---------------------------------------------------------------------------
# Multi-run aggregation (--repeat N)
# ---------------------------------------------------------------------------


_METRIC_FIELDS = [
    "qa_llm_accuracy",
    "answer_only_accuracy",
    "tool_precision",
    "tool_recall",
    "unit_test_accuracy",
    "flex_answer_accuracy",
    "answer_groundedness",
    "answer_relevancy",
    "methodology_soundness",
    "bi_phase1_rate",
    "bi_phase2_rate",
    "total_duration_seconds",
    "avg_duration_seconds",
    "p50_duration_seconds",
    "p90_duration_seconds",
    "avg_steps",
    "avg_tool_calls_per_step",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_reasoning_tokens",
    "avg_cache_read_tokens",
    "avg_cache_creation_tokens",
    "avg_total_tokens",
    "avg_total_token_e2e",
    "avg_output_tokens_other_tool_turns",
    "avg_output_tokens_text_turns",
]

# Subset of _METRIC_FIELDS that have a corresponding _stddev field on DatasetMetrics.
_STDDEV_FIELDS = [
    "qa_llm_accuracy",
    "answer_only_accuracy",
    "tool_precision",
    "tool_recall",
    "unit_test_accuracy",
    "flex_answer_accuracy",
    "answer_groundedness",
    "answer_relevancy",
    "methodology_soundness",
    "bi_phase1_rate",
    "bi_phase2_rate",
    "avg_duration_seconds",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_reasoning_tokens",
    "avg_cache_read_tokens",
    "avg_total_tokens",
    "avg_total_token_e2e",
]


@dataclass
class AggregatedBreakdown:
    """min / avg / max / total for one slice (dataset or overall) across runs."""

    total_overall: dict[str, Any] = field(default_factory=dict)
    min_overall: DatasetMetrics = field(default_factory=DatasetMetrics)
    avg_overall: DatasetMetrics = field(default_factory=DatasetMetrics)
    max_overall: DatasetMetrics = field(default_factory=DatasetMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_overall": self.total_overall,
            "min_overall": self.min_overall.to_dict(),
            "avg_overall": self.avg_overall.to_dict(),
            "max_overall": self.max_overall.to_dict(),
        }


def _build_breakdown(metrics_list: list[DatasetMetrics]) -> AggregatedBreakdown:
    """Build an :class:`AggregatedBreakdown` from per-run metrics for one slice."""
    bd = AggregatedBreakdown()
    if not metrics_list:
        return bd

    dataset_name = metrics_list[0].dataset
    num_cases = metrics_list[0].num_cases

    # min
    min_dm = DatasetMetrics(
        dataset=dataset_name,
        num_cases=num_cases,
        num_errors=min(m.num_errors for m in metrics_list),
    )
    for f in _METRIC_FIELDS:
        setattr(min_dm, f, min(getattr(m, f) for m in metrics_list))
    bd.min_overall = min_dm

    # max
    max_dm = DatasetMetrics(
        dataset=dataset_name,
        num_cases=num_cases,
        num_errors=max(m.num_errors for m in metrics_list),
    )
    for f in _METRIC_FIELDS:
        setattr(max_dm, f, max(getattr(m, f) for m in metrics_list))
    bd.max_overall = max_dm

    # avg + stddev
    avg_dm = DatasetMetrics(
        dataset=dataset_name,
        num_cases=num_cases,
        num_errors=_safe_mean([float(m.num_errors) for m in metrics_list]),
    )
    for f in _METRIC_FIELDS:
        values = [getattr(m, f) for m in metrics_list]
        setattr(avg_dm, f, _safe_mean(values))
    for f in _STDDEV_FIELDS:
        values = [getattr(m, f) for m in metrics_list]
        setattr(avg_dm, f"{f}_stddev", _safe_stddev(values))
    bd_list = [m.avg_latency_breakdown for m in metrics_list if m.avg_latency_breakdown]
    avg_dm.avg_latency_breakdown = _aggregate_latency_breakdowns(bd_list)
    # Aggregate tool call distributions across runs
    all_tools: set[str] = set()
    for m in metrics_list:
        all_tools.update(m.avg_tool_call_distribution.keys())
    if all_tools:
        avg_dm.avg_tool_call_distribution = {
            tool: _safe_mean([m.avg_tool_call_distribution.get(tool, 0.0) for m in metrics_list])
            for tool in sorted(all_tools)
        }

    bd.avg_overall = avg_dm

    # total — only metrics meaningful as sums
    bd.total_overall = {
        "total_samples": len(metrics_list) * num_cases,
        "total_errors": sum(m.num_errors for m in metrics_list),
        "total_duration_seconds": sum(m.total_duration_seconds for m in metrics_list),
    }

    return bd


@dataclass
class MultiRunReport:
    """Report aggregating multiple evaluation runs."""

    num_runs: int = 0
    swarm_enabled: bool = False
    swarm_scaling_mode: str = ""
    performance_over_all_datasets: AggregatedBreakdown = field(
        default_factory=AggregatedBreakdown,
    )
    performance_per_dataset: dict[str, AggregatedBreakdown] = field(
        default_factory=dict,
    )
    runs: list[dict[str, Any]] = field(default_factory=list)

    # Per-conv_id aggregation across runs
    per_conv_id_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_runs": self.num_runs,
            "swarm_enabled": self.swarm_enabled,
            "swarm_scaling_mode": self.swarm_scaling_mode,
            "performance_over_all_datasets": self.performance_over_all_datasets.to_dict(),
            "performance_per_dataset": {
                k: v.to_dict() for k, v in self.performance_per_dataset.items()
            },
            "per_conv_id_summary": self.per_conv_id_summary,
            "runs": self.runs,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def compute_aggregated_report(reports: list[EvalReport]) -> EvalReport:
    """Aggregate multiple run reports into one with mean and stddev per dataset.

    Used for the console summary table (not the JSON report).
    """
    if not reports:
        return EvalReport()
    if len(reports) == 1:
        return reports[0]

    agg = EvalReport(
        total_cases=reports[0].total_cases,
        total_errors=_safe_mean([float(r.total_errors) for r in reports]),
    )

    # Collect all dataset names across runs
    all_datasets: set[str] = set()
    for r in reports:
        all_datasets.update(r.datasets.keys())

    for ds_name in sorted(all_datasets):
        per_run = [r.datasets[ds_name] for r in reports if ds_name in r.datasets]
        if not per_run:
            continue

        dm = DatasetMetrics(
            dataset=ds_name,
            num_cases=per_run[0].num_cases,
            num_errors=_safe_mean([float(m.num_errors) for m in per_run]),
        )
        for f in _METRIC_FIELDS:
            values = [getattr(m, f) for m in per_run]
            setattr(dm, f, _safe_mean(values))
        for f in _STDDEV_FIELDS:
            values = [getattr(m, f) for m in per_run]
            setattr(dm, f"{f}_stddev", _safe_stddev(values))
        bd_list = [m.avg_latency_breakdown for m in per_run if m.avg_latency_breakdown]
        dm.avg_latency_breakdown = _aggregate_latency_breakdowns(bd_list)
        agg_tools: set[str] = set()
        for m in per_run:
            agg_tools.update(m.avg_tool_call_distribution.keys())
        if agg_tools:
            dm.avg_tool_call_distribution = {
                tool: _safe_mean([m.avg_tool_call_distribution.get(tool, 0.0) for m in per_run])
                for tool in sorted(agg_tools)
            }
        agg.datasets[ds_name] = dm

    # Overall
    overall_list = [r.overall for r in reports]
    agg.overall = DatasetMetrics(
        dataset="(overall)",
        num_cases=reports[0].overall.num_cases,
        num_errors=_safe_mean([float(m.num_errors) for m in overall_list]),
    )
    for f in _METRIC_FIELDS:
        values = [getattr(m, f) for m in overall_list]
        setattr(agg.overall, f, _safe_mean(values))
    for f in _STDDEV_FIELDS:
        values = [getattr(m, f) for m in overall_list]
        setattr(agg.overall, f"{f}_stddev", _safe_stddev(values))
    bd_list = [m.avg_latency_breakdown for m in overall_list if m.avg_latency_breakdown]
    agg.overall.avg_latency_breakdown = _aggregate_latency_breakdowns(bd_list)
    overall_tools: set[str] = set()
    for m in overall_list:
        overall_tools.update(m.avg_tool_call_distribution.keys())
    if overall_tools:
        agg.overall.avg_tool_call_distribution = {
            tool: _safe_mean([m.avg_tool_call_distribution.get(tool, 0.0) for m in overall_list])
            for tool in sorted(overall_tools)
        }

    return agg


def compute_multi_run_report(reports: list[EvalReport]) -> MultiRunReport:
    """Build a :class:`MultiRunReport` from a list of per-run reports."""
    multi = MultiRunReport(
        num_runs=len(reports),
        swarm_enabled=any(r.swarm_enabled for r in reports),
        swarm_scaling_mode=next((r.swarm_scaling_mode for r in reports if r.swarm_scaling_mode), ""),
    )

    # Store each run's full report
    for i, report in enumerate(reports):
        run_data = report.to_dict()
        run_data["run_index"] = i
        multi.runs.append(run_data)

    # Overall breakdown across all datasets
    multi.performance_over_all_datasets = _build_breakdown(
        [r.overall for r in reports],
    )

    # Per-dataset breakdowns
    all_datasets: set[str] = set()
    for r in reports:
        all_datasets.update(r.datasets.keys())

    for ds_name in sorted(all_datasets):
        per_run = [r.datasets[ds_name] for r in reports if ds_name in r.datasets]
        if per_run:
            multi.performance_per_dataset[ds_name] = _build_breakdown(per_run)

    # Per-conv_id aggregation: group scores across all runs
    conv_id_data: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "dataset": "",
        "eval_mode": "",
        "per_run_scores": [],
        "per_run_unit_test_scores": [],
    })
    for i, report in enumerate(reports):
        for case_dict in report.per_case:
            cid = case_dict.get("conv_id", "")
            if not cid:
                continue
            entry = conv_id_data[cid]
            if not entry["dataset"]:
                entry["dataset"] = case_dict.get("dataset", "")
                entry["eval_mode"] = case_dict.get("eval_mode", "")
            entry["per_run_scores"].append(case_dict.get("score"))
            entry["per_run_unit_test_scores"].append(
                case_dict.get("unit_test_score"),
            )

    for cid, entry in conv_id_data.items():
        scores = [s for s in entry["per_run_scores"] if s is not None]
        entry["mean_score"] = round(_safe_mean(scores), 4) if scores else None
        entry["stddev_score"] = round(_safe_stddev(scores), 4) if scores else None

        ut_scores = [s if s is not None else 0.0 for s in entry["per_run_unit_test_scores"]]
        if ut_scores:
            entry["mean_unit_test_score"] = round(_safe_mean(ut_scores), 4)
            entry["stddev_unit_test_score"] = round(_safe_stddev(ut_scores), 4)

    multi.per_conv_id_summary = dict(conv_id_data)

    return multi


# ---------------------------------------------------------------------------
# Markdown breakdown report
# ---------------------------------------------------------------------------


def generate_breakdown_md(report: EvalReport) -> str:
    """Return markdown content with latency and token breakdown tables."""
    datasets = [dm for dm in report.datasets.values() if dm.num_cases > 0]
    if not datasets:
        return ""

    lines: list[str] = ["# Evaluation Breakdown\n"]

    # --- Summary Table ---
    has_flex = any(dm.flex_answer_accuracy >= 0 for dm in datasets)
    has_bi_phases = any(dm.bi_phase1_rate >= 0 for dm in datasets)
    summary_cols = []
    if has_flex:
        summary_cols.append("Flex Answer")
    summary_cols.append("AO Acc")
    if has_bi_phases:
        summary_cols.extend(["Phase1 Rate", "Phase2 Rate"])
    summary_cols += [
        "Avg Lat", "P50 Lat", "P90 Lat",
        "Errors", "Avg Tokens", "Avg Steps",
    ]
    lines.append("## Summary\n")
    header = "| Dataset | " + " | ".join(summary_cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in summary_cols) + "|"
    lines.append(header)
    lines.append(sep)
    for dm in datasets:
        cells: list[str] = []
        if has_flex:
            cells.append(
                f"{dm.flex_answer_accuracy:.3f}" if dm.flex_answer_accuracy >= 0 else "-"
            )
        cells.append(f"{dm.answer_only_accuracy:.3f}")
        if has_bi_phases:
            if dm.bi_phase1_rate >= 0:
                p1 = f"{dm.bi_phase1_rate:.3f}"
                if dm.bi_phase1_rate_stddev is not None:
                    p1 += f" ± {dm.bi_phase1_rate_stddev:.3f}"
                p2 = f"{dm.bi_phase2_rate:.3f}"
                if dm.bi_phase2_rate_stddev is not None:
                    p2 += f" ± {dm.bi_phase2_rate_stddev:.3f}"
                cells.extend([p1, p2])
            else:
                cells.extend(["N/A", "N/A"])
        cells += [
            f"{dm.avg_duration_seconds:.1f}s",
            f"{dm.p50_duration_seconds:.1f}s",
            f"{dm.p90_duration_seconds:.1f}s",
            str(dm.num_errors),
            f"{dm.avg_total_tokens:,.0f}",
            f"{dm.avg_steps:.1f}",
        ]
        lines.append(f"| {dm.dataset} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- Latency Breakdown Table ---
    all_keys: list[str] = []
    seen: set[str] = set()
    for dm in datasets:
        for k in dm.avg_latency_breakdown:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    if all_keys:
        lines.append("## Latency Breakdown\n")
        col_names = []
        for k in all_keys:
            col_names.append(f"..{k.strip()}" if k.startswith("  ") else k)
        header = "| Dataset | " + " | ".join(col_names) + " |"
        sep = "|---|" + "|".join("---:" for _ in col_names) + "|"
        lines.append(header)
        lines.append(sep)

        for dm in datasets:
            bd = dm.avg_latency_breakdown
            total = dm.avg_duration_seconds or sum(
                v for k, v in bd.items() if not k.startswith(" ")
            )
            cells: list[str] = []
            for k in all_keys:
                val = bd.get(k, 0.0)
                pct = (val / total * 100) if total > 0 else 0.0
                cells.append(f"{val:.1f}s ({pct:.1f}%)")
            lines.append(f"| {dm.dataset} | " + " | ".join(cells) + " |")
        lines.append("")

    # --- Token Breakdown Table ---
    has_turn_tokens = any(
        dm.avg_output_tokens_other_tool_turns > 0
        or dm.avg_output_tokens_text_turns > 0
        for dm in datasets
    )

    lines.append("## Token Breakdown\n")
    cols = [
        "Orch Input", "Orch Output", "Cache Read", "Cache Create",
        "Reasoning", "Orch Total",
    ]
    if has_turn_tokens:
        cols += ["Out: Tool", "Out: Text-only"]

    header = "| Dataset | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in cols) + "|"
    lines.append(header)
    lines.append(sep)

    for dm in datasets:
        cells: list[str] = [
            f"{dm.avg_input_tokens:,.0f}",
            f"{dm.avg_output_tokens:,.0f}",
            f"{dm.avg_cache_read_tokens:,.0f}",
            f"{dm.avg_cache_creation_tokens:,.0f}",
            f"{dm.avg_reasoning_tokens:,.0f}",
            f"{dm.avg_total_tokens:,.0f}",
        ]
        if has_turn_tokens:
            ototal = (
                dm.avg_output_tokens_other_tool_turns
                + dm.avg_output_tokens_text_turns
            )

            def _pct(v: float) -> str:
                return f"{v:,.0f} ({v / ototal * 100:.1f}%)" if ototal > 0 else f"{v:,.0f}"

            cells += [
                _pct(dm.avg_output_tokens_other_tool_turns),
                _pct(dm.avg_output_tokens_text_turns),
            ]
        lines.append(f"| {dm.dataset} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- Tool Call Distribution Table ---
    all_tools: list[str] = []
    seen_tools: set[str] = set()
    for dm in datasets:
        for t in sorted(dm.avg_tool_call_distribution):
            if t not in seen_tools:
                all_tools.append(t)
                seen_tools.add(t)

    if all_tools:
        lines.append("## Avg Tool Call Distribution\n")
        header = "| Dataset | " + " | ".join(all_tools) + " | Total |"
        sep = "|---|" + "|".join("---:" for _ in all_tools) + "|---:|"
        lines.append(header)
        lines.append(sep)
        for dm in datasets:
            dist = dm.avg_tool_call_distribution
            cells = [f"{dist.get(t, 0.0):.1f}" for t in all_tools]
            total = sum(dist.values())
            cells.append(f"{total:.1f}")
            lines.append(f"| {dm.dataset} | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)
