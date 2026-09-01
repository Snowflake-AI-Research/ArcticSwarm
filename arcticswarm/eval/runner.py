"""Eval runner — execute eval cases through the arcticswarm Agent.

For each :class:`~arcticswarm.eval.data_loader.EvalCase`, the runner:
  1. Builds a :class:`~arcticswarm.config.ArcticswarmConfig` for the case.
  2. Instantiates an :class:`~arcticswarm.agent.Agent`.
  3. Calls ``agent.run_turn(question)`` and records the full response,
     tool call trace, and any errors.
  4. Returns an :class:`EvalResult` for downstream judging.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from arcticswarm.agent import Agent, StreamEvent, TokenUsage, ToolCallEnd, ToolCallStart
from arcticswarm.config import ArcticswarmConfig, load_snowflake_connections
from arcticswarm.logging_utils import serialize_content, serialize_messages
from arcticswarm.swarm.teammate import _TimingCollector, _inject_timings_into_messages
from arcticswarm.tools._image_media import media_type_for_path

from arcticswarm.eval.data_loader import EvalCase
from arcticswarm.eval.judge import FlexJudgeResult, InsightJudgeResult, LLMJudge, QAJudgeResult

logger = logging.getLogger(__name__)

# Extra wall-clock seconds to wait after the per-case deadline before giving
# up and declaring a timeout.  Guards against the narrow race where the
# worker thread finishes a turn just past ``timeout_seconds``; without this
# grace the case's full response, token counts, and step counts would be
# discarded even though the trajectory contains a real final answer.
_TIMEOUT_GRACE_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Question hint enrichment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multimodal user-message construction
# ---------------------------------------------------------------------------


def _encode_image_block(image_path: str) -> dict[str, Any] | None:
    """Encode a local image file as an Anthropic-shape image content block.

    Returns ``None`` when the file cannot be read — callers should skip it
    and fall back to text-only. The Anthropic canonical shape is
    auto-translated to OpenAI Chat (``image_url``) and OpenAI Responses
    (``input_image``) formats by :mod:`arcticswarm.llm_client`.
    """
    p = Path(image_path).expanduser()
    try:
        raw = p.read_bytes()
    except OSError as exc:
        logger.warning("Skipping attached image %s: %s", image_path, exc)
        return None
    media_type = media_type_for_path(p)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64,
        },
    }


def _build_user_message_content(
    enriched_question: str, case: EvalCase, enable_vision: bool = True
) -> str | list[dict[str, Any]]:
    """Build the initial user-message content for an :class:`EvalCase`.

    When the case has no attached images, returns *enriched_question*
    unchanged so the downstream message path stays a plain string (the
    existing fast path).

    When images are attached, returns a list of content blocks in the
    Anthropic canonical shape, ordered image-before-text per the vision
    best-practice.

    - Single image: ``[image, text question]`` (the ``"Image 1:"``
      marker would be pure noise — the image block itself already tells
      the model it is looking at an image).
    - Multiple images: ``[text "Image 1:", image, text "Image 2:",
      image, ..., text question]`` — per the Anthropic "Multiple
      images" example, the markers let the model disambiguate which
      image the question refers to.

    Images that fail to load are skipped (a warning is logged); the
    caller still receives the text block so the model can attempt a
    text-only answer rather than crashing the case.

    When *enable_vision* is False (e.g. a text-only self-hosted vLLM model
    served with ``--language-model-only``), attached
    images are dropped entirely and the plain text question is returned.
    Sending an image to a text-only server otherwise hard-errors the case
    (vLLM ``400 BadRequestError: At most 0 vision_chunk(s) may be
    provided``); a text-only attempt at least produces some signal.
    """
    if not case.attached_images:
        return enriched_question

    if not enable_vision:
        logger.warning(
            "Vision disabled (enable_vision=False): dropping %d attached image(s) "
            "for case %s; attempting text-only.",
            len(case.attached_images),
            getattr(case, "conv_id", "?"),
        )
        return enriched_question

    image_blocks: list[dict[str, Any]] = []
    for image_path in case.attached_images:
        image_block = _encode_image_block(image_path)
        if image_block is not None:
            image_blocks.append(image_block)

    # If every image failed to load, fall back to plain text so the run
    # still produces some signal instead of a degenerate list.
    if not image_blocks:
        return enriched_question

    blocks: list[dict[str, Any]] = []
    if len(image_blocks) == 1:
        blocks.append(image_blocks[0])
    else:
        for idx, image_block in enumerate(image_blocks, start=1):
            blocks.append({"type": "text", "text": f"Image {idx}:"})
            blocks.append(image_block)

    blocks.append({"type": "text", "text": enriched_question})
    return blocks


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# Datasets that use only their canonical/original evaluation metric.
# These skip the Answer-Only judge to avoid redundant or non-standard metrics.
# Add new external benchmarks here as they're integrated.
CANONICAL_METRIC_ONLY_DATASETS = frozenset([
    "BROWSECOMP_V1",  # OpenAI's BrowseComp benchmark - uses specialized binary judge
    "HYBRID_V1",  # Hybrid (search + SQL): SQL-result comparison or browsecomp text judge
])


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """A single tool call observed during the agent turn."""

    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    is_error: bool = False


@dataclass
class EvalResult:
    """Full result for a single eval case."""

    case: EvalCase
    response_text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0

    # How many attempts were needed (1 = succeeded on first try)
    attempt: int = 1

    # Populated after judging
    qa_result: QAJudgeResult | None = None
    insight_result: InsightJudgeResult | None = None
    answer_only_result: InsightJudgeResult | None = None
    flex_result: FlexJudgeResult | None = None

    # Populated after unit-test judging (verifiable datasets only)
    unit_test_score: float | None = None
    unit_test_extracted_json: dict[str, Any] | None = None

    # Full agent conversation trajectory (serialised messages)
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    # Optional metadata extracted from the response (e.g. answer type, boxed
    # answer). Empty unless a postprocessing step populates it.
    extracted_metadata: dict[str, Any] = field(default_factory=dict)

    # Swarm-specific metrics (populated when config.swarm_enabled is True)
    swarm_teammates_spawned: int = 0
    swarm_bbs_message_count: int = 0
    swarm_saturation_events: int = 0
    swarm_reflection_stats: dict[str, Any] = field(default_factory=dict)
    # Layer 4b — rival-candidate audit telemetry (per-case).
    swarm_rival_audit: dict[str, Any] = field(default_factory=dict)
    # Layer 4a / critic-refine telemetry.  Read by the
    # gated-retry confidence detector.
    swarm_layer4a: dict[str, Any] = field(default_factory=dict)
    # Cheap-win recovery turn fired.
    swarm_cheap_win_fired: bool = False
    # Disagreement-gate rival sweep was triggered.
    swarm_rival_sweep_fired: bool = False
    # Per-subagent tool call counts: {subagent_name: {tool_name: count}}
    swarm_subagent_tool_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Spawn/assignment event log (dynamic mode only)
    swarm_spawn_events: list[dict[str, Any]] = field(default_factory=list)
    # Per-agent token usage breakdown: {agent_name: {input_tokens, output_tokens, ...}}
    swarm_token_usage_breakdown: dict[str, Any] = field(default_factory=dict)

    # Per-phase timing breakdown (seconds) for bottleneck analysis
    phase_timings: dict[str, float] = field(default_factory=dict)

    # Latency breakdown
    latency_breakdown: dict[str, float] = field(default_factory=dict)

    # Token usage for this eval case (input, output, cache read/write)
    token_usage: TokenUsage | None = None

    # Per-turn token breakdown (populated from trajectory by _extract_token_stats)
    token_breakdown: dict[str, float] = field(default_factory=dict)

    # Per-tool call counts (populated from trajectory by _extract_token_stats)
    tool_call_distribution: dict[str, int] = field(default_factory=dict)

    # Number of LLM round-trips (steps/turns) in the agent loop
    num_steps: int = 0

    # Context compaction stats
    compaction_count: int = 0
    total_llm_calls: int = 0
    # O2: split proactive vs reactive + peak input tokens.  Lets reports
    # tell whether proactive compaction (at 90% of the context limit) ever fires
    # and what the largest single-call context size reached.
    proactive_compaction_count: int = 0
    reactive_compaction_count: int = 0
    peak_input_tokens: int = 0

    # Safety refusal count (primary model refused, fell back to Sonnet 4)
    safety_refusal_count: int = 0
    # True iff the agent's final response_text matches the safety-refusal
    # regex (Agent.is_refusal_text).  Distinguishes "100 intermediate
    # refusals → recovered final answer" (working as intended) from
    # "100 intermediate refusals → final answer is a refusal" (doomed
    # case).  Read from response_text after the agent loop ends — works
    # for both single-agent and swarm modes.
    final_answer_is_refusal: bool = False
    # Azure content filter blocks (source scorer, etc.)
    content_filter_count: int = 0
    # O5 (Q7): count of LLM responses that returned only thinking blocks
    # (no text, no tool_use).  Helps verify Q3 (disable_extended_thinking)
    # actually disables the failure mode it targets.
    thinking_only_count: int = 0
    # O4: count of attempts on this case that ended in TimeoutError.
    # Set even when force-report recovery (Layer 1/3) clears ``error`` so
    # the case is judged — without this we cannot distinguish a slow
    # successful case from a multi-attempt timeout-then-recovered case.
    timeout_attempts: int = 0
    # O4: free-form recovery marker.  Currently one of:
    #   ""                                       (no recovery, normal completion)
    #   "force_report_after_timeout"             (Layer 1/3 recovery cleared error)
    recovery_mode: str = ""

    # e2e total token estimate for swarm mode: orch_tokens + max(subagent_tokens).
    # For non-swarm mode this equals total_tokens.
    total_token_e2e: int = 0

    # Brave search fallback events (populated by drain_fallback_log on web_search tool)
    web_search_failures: list[dict[str, Any]] = field(default_factory=list)

    # Full per-query search log (all providers, not just failures)
    web_search_log: list[dict[str, Any]] = field(default_factory=list)

    # Full per-fetch log (url, tier, latency, content_chars)
    web_fetch_log: list[dict[str, Any]] = field(default_factory=list)

    # Per-call compactor log (source, indices, scores, fallback_reason, latency)
    compactor_log: list[dict[str, Any]] = field(default_factory=list)

    # Source scorer content filter rejection log
    content_filter_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens: input + output + cache_read + cache_creation (matches Go AvgTotalTokens)."""
        if self.token_usage is None:
            return 0
        return self.token_usage.total_tokens

    @property
    def response_full(self) -> str:
        """Build a text representation of tool uses + results for the judge."""
        parts: list[str] = []
        for tc in self.tool_calls:
            parts.append(f"Tool: {tc.name}")
            try:
                input_str = json.dumps(tc.input, indent=2, default=str)
            except Exception:
                input_str = str(tc.input)
            parts.append(f"Input: {input_str}")
            if tc.is_error:
                parts.append(f"Error: {tc.output}")
            else:
                parts.append(f"Output: {tc.output[:2000]}")
            parts.append("")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Event collector
# ---------------------------------------------------------------------------


class _EventCollector:
    """Collects tool call events emitted by the Agent during a turn."""

    def __init__(self) -> None:
        self.tool_calls: list[ToolCallRecord] = []
        self._pending: dict[str, ToolCallRecord] = {}

    def on_event(self, event: StreamEvent) -> None:
        if isinstance(event, ToolCallStart):
            record = ToolCallRecord(name=event.tool_name, input=event.tool_input)
            self._pending[event.tool_use_id] = record
        elif isinstance(event, ToolCallEnd):
            record = self._pending.pop(event.tool_use_id, None)
            if record is None:
                record = ToolCallRecord(name=event.tool_name)
            if event.result:
                record.output = event.result.output if not event.result.is_error else event.result.error or ""
                record.is_error = event.result.is_error
            self.tool_calls.append(record)


# ---------------------------------------------------------------------------
# Per-case token breakdown extraction
# ---------------------------------------------------------------------------


def _extract_token_stats(
    trajectory: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    """Extract per-case token breakdown and tool call distribution from trajectory.

    Walks the serialised messages and uses ``_llm_output_tokens`` on assistant
    messages for per-turn attribution.

    Two-way turn classification:
    - **Tool turns**: assistant turns containing tool calls.
    - **Text-only turns**: assistant turns with no tool calls.

    Returns ``(token_breakdown, tool_call_distribution)``.
    """
    if not trajectory:
        return {}, {}

    messages = trajectory[0].get("messages", []) if isinstance(trajectory[0], dict) else []
    if not messages:
        return {}, {}

    output_tokens_other_tool_turns = 0.0
    output_tokens_text_turns = 0.0

    tool_counts: dict[str, int] = defaultdict(int)

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        llm_output = msg.get("_llm_output_tokens", 0) or 0

        tool_names_in_turn: set[str] = set()
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    tool_names_in_turn.add(name)
                    tool_counts[name] += 1

        if tool_names_in_turn:
            output_tokens_other_tool_turns += llm_output
        else:
            output_tokens_text_turns += llm_output

    token_breakdown = {
        "output_tokens_other_tool_turns": output_tokens_other_tool_turns,
        "output_tokens_text_turns": output_tokens_text_turns,
    }

    return token_breakdown, dict(tool_counts)


def _enrich_latency_breakdown(
    breakdown: dict[str, float],
    trajectory: list[dict[str, Any]],
) -> None:
    """Add sub-timings to *breakdown* in-place.

    Splits ``llm_planning`` into ``llm_other_tool_turns`` and
    ``llm_text_turns`` based on whether each assistant turn invokes a tool.
    """
    if not trajectory or not breakdown:
        return
    messages = trajectory[0].get("messages", []) if isinstance(trajectory[0], dict) else []
    if not messages:
        return

    llm_other_tool_turns = 0.0
    llm_text_turns = 0.0

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        llm_dur = msg.get("_llm_duration_seconds", 0.0) or 0.0
        has_tool = isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in content
        )
        if has_tool:
            llm_other_tool_turns += llm_dur
        else:
            llm_text_turns += llm_dur

    new_bd: dict[str, float] = {}
    for key, val in breakdown.items():
        new_bd[key] = val
        if key == "llm_planning":
            new_bd["  llm_other_tool_turns"] = round(llm_other_tool_turns, 3)
            new_bd["  llm_text_turns"] = round(llm_text_turns, 3)
    breakdown.clear()
    breakdown.update(new_bd)


# ---------------------------------------------------------------------------
# Single-case execution
# ---------------------------------------------------------------------------


def _build_retry_config(base_config: ArcticswarmConfig) -> ArcticswarmConfig:
    """Mutate *base_config* for a gated-retry attempt.

    Drop the coding profile and bump subagent reasoning effort to xhigh.
    Keeps the auditor reasoning at default to avoid entrenchment.
    """
    from copy import deepcopy
    cfg = deepcopy(base_config)
    # Drop coding profile from swarm_profiles if present.
    profiles = list(getattr(cfg, "swarm_profiles", []) or [])
    cfg.swarm_profiles = [p for p in profiles if p != "coding"]
    # Bump subagent reasoning effort.
    cfg.subagent_reasoning_effort = "xhigh"
    return cfg


def _build_config(
    case: EvalCase,
    base_config: ArcticswarmConfig,
) -> ArcticswarmConfig:
    """Build a config for this eval case, inheriting API/SF settings from *base_config*."""
    return replace(
        base_config,
        brave_api_key=base_config.brave_api_key,
        date_override=case.date_override,
        dataset=case.dataset,
        swarm_enabled=base_config.swarm_enabled,
        max_teammates=base_config.max_teammates,
        subagent_model=base_config.subagent_model,
        subagent_reasoning_effort=base_config.subagent_reasoning_effort,
        swarm_comm=base_config.swarm_comm,
    )


def _run_agent(
    case: EvalCase,
    config: ArcticswarmConfig,
    collector: _EventCollector,
    agent_ref: list[Agent | None] | None = None,
    timing_ref: list[_TimingCollector | None] | None = None,
    on_event: Any | None = None,
) -> tuple[str, list[dict[str, Any]], TokenUsage, int, dict[str, float], str, list[dict[str, Any]]]:
    """Run the agent for a single case.

    Returns ``(response_text, messages, token_usage, num_steps,
    latency_breakdown, system_prompt, tool_definitions)``.

    This is the inner function executed inside a thread so it can be timed out.

    *on_event*, when given, is invoked with every :class:`StreamEvent` for
    the live eval feed (in addition to the result ``collector``).
    """
    agent: Agent | None = None
    try:
        agent = Agent(config)
        if agent_ref is not None: # store agent instance for partial trajectory record
            agent_ref[0] = agent

        # Set up content cache for same-agent dedup (even without swarm).
        # Engages when a per-question output dir OR the global fetch cache is
        # configured (cache_dir=None => global-only).
        if getattr(config, "enable_content_cache", True) and (
            config.output_dir or getattr(config, "fetch_cache_path", "")
        ):
            from arcticswarm.tools.content_cache import ContentCache
            safe_id = case.conv_id.replace("/", "_").replace("\\", "_")[:200]
            agent.content_cache = ContentCache(
                cache_dir=config.output_dir or None, case_id=safe_id,
                global_db_path=getattr(config, "fetch_cache_path", ""),
            )
            agent._register_tools()

        msg_start_idx = len(agent.messages)
        # Fan tool events to the result collector and, optionally, the live
        # eval feed.  ``_TimingCollector`` forwards each event to this inner
        # callback after stamping timings.
        if on_event is not None:
            def _inner_on_event(ev: StreamEvent) -> None:
                collector.on_event(ev)
                try:
                    on_event(ev)
                except Exception:
                    pass  # feed is observability-only; never break the case
        else:
            _inner_on_event = collector.on_event
        timing = _TimingCollector(inner_on_event=_inner_on_event)
        if timing_ref is not None:
            timing_ref[0] = timing
        timing.start()
        enriched_question = case.question
        user_content = _build_user_message_content(enriched_question, case, config.enable_vision)
        if config.use_streaming:
            response_text = agent.run_turn_streaming(user_content, on_event=timing.on_event)
        else:
            response_text = agent.run_turn(user_content, on_event=timing.on_event)
        extracted_metadata: dict[str, Any] = {}
        _inject_timings_into_messages(agent.messages, timing, msg_start_idx)
        system_prompt = agent.system_prompt
        tool_defs = agent._get_tool_definitions()
        return (
            response_text, agent.messages, agent.last_turn_usage,
            agent.last_num_steps, timing.latency_breakdown(),
            system_prompt, tool_defs, extracted_metadata,
            agent.compaction_count, agent.total_llm_calls,
            agent.safety_refusal_count, agent.content_filter_count,
            getattr(agent, "thinking_only_count", 0),
            getattr(agent, "proactive_compaction_count", 0),
            getattr(agent, "reactive_compaction_count", 0),
            getattr(agent._context_budget, "peak_input_tokens", 0),
        )
    finally:
        if agent is not None:
            agent.close()


@dataclass
class _SwarmResult:
    """Extended result from a swarm run including swarm-specific metrics."""
    response_text: str = ""
    teammates_spawned: int = 0
    bbs_message_count: int = 0
    saturation_events: int = 0
    # Context compaction stats (aggregated across orchestrator + subagents)
    compaction_count: int = 0
    total_llm_calls: int = 0
    # O2: split proactive vs reactive + peak input tokens
    proactive_compaction_count: int = 0
    reactive_compaction_count: int = 0
    peak_input_tokens: int = 0
    # Safety refusal count (aggregated across orchestrator + subagents)
    safety_refusal_count: int = 0
    # Azure content filter blocks (aggregated across orchestrator + subagents)
    content_filter_count: int = 0
    # O5: thinking-only count (aggregated across orchestrator + subagents)
    thinking_only_count: int = 0
    # Reflection stats (aggregated across all subagents)
    reflection_stats: dict[str, Any] = field(default_factory=dict)
    # Layer 4a / critic-refine telemetry.
    layer4a: dict[str, Any] = field(default_factory=dict)
    # Cheap-win + rival-sweep telemetry.
    cheap_win_fired: bool = False
    rival_sweep_fired: bool = False
    # Layer 4b — rival-candidate audit telemetry (per-case).
    rival_audit: dict[str, Any] = field(default_factory=dict)
    # Trajectory data for debugging
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    # Per-phase timing breakdown (seconds)
    phase_timings: dict[str, float] = field(default_factory=dict)
    # Aggregated token usage
    token_usage: TokenUsage | None = None
    num_steps: int = 0
    total_token_e2e: int = 0
    # Final answer formatter metadata
    extracted_metadata: dict[str, Any] = field(default_factory=dict)
    # Per-subagent tool call counts: {subagent_name: {tool_name: count}}
    subagent_tool_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Orchestrator's own tool calls: {tool_name: count}.  Source for
    # synthesizing ``EvalResult.tool_calls`` in
    # ``_apply_swarm_result_to_eval_result`` so swarm runs report a real
    # ``num_tool_calls`` instead of 0.
    orchestrator_tool_counts: dict[str, int] = field(default_factory=dict)
    # Brave search fallback events from all subagents
    web_search_failures: list[dict[str, Any]] = field(default_factory=list)
    # Full per-query search log from all subagents
    web_search_log: list[dict[str, Any]] = field(default_factory=list)
    # Full per-fetch log from all subagents
    web_fetch_log: list[dict[str, Any]] = field(default_factory=list)
    # Per-call compactor log from all subagents
    compactor_log: list[dict[str, Any]] = field(default_factory=list)
    # Source scorer content filter rejection log from all subagents
    content_filter_log: list[dict[str, Any]] = field(default_factory=list)
    # Spawn/assignment event log (dynamic mode only)
    spawn_events: list[dict[str, Any]] = field(default_factory=list)
    # Per-agent token usage breakdown: {agent_name: {input_tokens, output_tokens, ...}}
    token_usage_breakdown: dict[str, Any] = field(default_factory=dict)


def _serialize_swarm_trajectory(
    orchestrator_messages: list[dict[str, Any]],
    subagent_summaries: list[dict[str, Any]],
    task_summaries: list[dict[str, Any]],
    *,
    orchestrator_system_prompt: str = "",
    orchestrator_model: str = "",
    orchestrator_reasoning_effort: str | None = None,
    orchestrator_tools: list[dict[str, Any]] | None = None,
    spawn_events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a JSON-safe trajectory structure for a swarm run.

    The returned list has a single dict with keys:

    - ``orchestrator_system_prompt``: the orchestrator's system prompt.
    - ``orchestrator_model`` / ``orchestrator_reasoning_effort``: the
      orchestrator backend configuration.
    - ``orchestrator_tools``: tool definitions available to the orchestrator.
    - ``orchestrator``: the orchestrator's full conversation (serialised).
    - ``subagents``: list of ``{name, model, reasoning_effort,
      system_prompt, tools, messages}``
      for each subagent.
    - ``tasks``: final task board state (name, status, summary, etc.).
    - ``spawn_events``: chronological log of spawn/assignment decisions
      (dynamic mode only).

    This mirrors the single-agent trajectory format (a JSON-serialisable
    list) but wraps the multi-agent data into a structured envelope.
    """
    serialized_subagents = []
    for sa in subagent_summaries:
        entry: dict[str, Any] = {"name": sa["name"]}
        for key in ("model", "reasoning_effort"):
            if key in sa:
                entry[key] = sa[key]
        if "system_prompt" in sa:
            entry["system_prompt"] = sa["system_prompt"]
        if "tools" in sa:
            entry["tools"] = sa["tools"]
        # Include persistent tool call counter (survives clear_history)
        if "tool_calls_by_name" in sa:
            entry["tool_calls_by_name"] = sa["tool_calls_by_name"]
        # Subagent lifecycle metadata for dynamic-vs-static analysis
        for key in ("dynamic_mode", "initial_profile", "tasks_completed"):
            if key in sa:
                entry[key] = sa[key]
        entry["messages"] = serialize_messages(sa.get("messages", []))
        serialized_subagents.append(entry)

    result: dict[str, Any] = {
        "orchestrator_system_prompt": orchestrator_system_prompt,
        "orchestrator_model": orchestrator_model,
        "orchestrator_reasoning_effort": orchestrator_reasoning_effort,
        "orchestrator_tools": orchestrator_tools or [],
        "orchestrator": serialize_messages(orchestrator_messages),
        "subagents": serialized_subagents,
        "tasks": task_summaries,
    }
    if spawn_events:
        result["spawn_events"] = spawn_events
    return [result]


def _run_swarm(
    case: EvalCase,
    config: ArcticswarmConfig,
    orchestrator_ref: list[Any] | None = None,
    timeout_seconds: float | None = None,
    on_event: Any | None = None,
) -> _SwarmResult:
    """Run a swarm for a single case.

    The swarm orchestrator manages its own agents internally.
    Returns a ``_SwarmResult`` with the response text, swarm metrics,
    and trajectory data for debugging.

    *on_event*, when given, is forwarded every :class:`SwarmEvent` for the
    live eval feed (in addition to the internal ``SwarmComplete`` capture).
    """
    from arcticswarm.swarm.orchestrator import SwarmOrchestrator, SwarmComplete
    orchestrator = SwarmOrchestrator(config, max_teammates=config.max_teammates)
    if orchestrator_ref is not None: # store orchestrator instance for partial trajectory record
        orchestrator_ref[0] = orchestrator

    # Set up per-question content cache (isolated by conv_id)
    if getattr(config, "enable_content_cache", True) and (
        config.output_dir or getattr(config, "fetch_cache_path", "")
    ):
        from arcticswarm.tools.content_cache import ContentCache
        safe_id = case.conv_id.replace("/", "_").replace("\\", "_")[:200]
        orchestrator._content_cache = ContentCache(
            cache_dir=config.output_dir or None, case_id=safe_id,
            global_db_path=getattr(config, "fetch_cache_path", ""),
        )

    swarm_result = _SwarmResult()

    def _on_swarm_event(event: object) -> None:
        if isinstance(event, SwarmComplete):
            swarm_result.teammates_spawned = event.subagent_count
            swarm_result.bbs_message_count = event.bbs_message_count
        if on_event is not None:
            try:
                on_event(event)
            except Exception:
                pass  # feed is observability-only; never break the case

    try:
        enriched_question = case.question
        user_content = _build_user_message_content(enriched_question, case, config.enable_vision)
        swarm_result.response_text = orchestrator.run_swarm_turn(
            user_content,
            on_swarm_event=_on_swarm_event,
            timeout_seconds=timeout_seconds,
        )
        orch_agent = orchestrator._orchestrator_agent or getattr(orchestrator, "_agent", None)
        swarm_result.trajectory = _serialize_swarm_trajectory(
            orchestrator.last_orchestrator_messages,
            orchestrator.last_subagent_summaries,
            orchestrator.last_task_summaries,
            orchestrator_system_prompt=(
                orch_agent.system_prompt if orch_agent is not None else ""
            ),
            orchestrator_model=(
                getattr(getattr(orch_agent, "config", None), "model", "")
                if orch_agent is not None else ""
            ),
            orchestrator_reasoning_effort=(
                getattr(getattr(orch_agent, "config", None), "reasoning_effort", None)
                if orch_agent is not None else None
            ),
            orchestrator_tools=(
                orch_agent._get_tool_definitions() if orch_agent is not None else []
            ),
            spawn_events=orchestrator.last_spawn_events or None,
        )
        swarm_result.phase_timings = dict(orchestrator.phase_timings)
        swarm_result.token_usage = orchestrator.last_token_usage
        swarm_result.num_steps = orchestrator.last_num_steps
        swarm_result.total_token_e2e = orchestrator.last_total_token_e2e
        swarm_result.saturation_events = orchestrator.last_saturation_events
        swarm_result.compaction_count = orchestrator.last_compaction_count
        swarm_result.total_llm_calls = orchestrator.last_total_llm_calls
        swarm_result.safety_refusal_count = orchestrator.last_safety_refusal_count
        swarm_result.content_filter_count = orchestrator.last_content_filter_count
        swarm_result.thinking_only_count = getattr(
            orchestrator, "last_thinking_only_count", 0
        )
        # O2
        swarm_result.proactive_compaction_count = getattr(
            orchestrator, "last_proactive_compaction_count", 0
        )
        swarm_result.reactive_compaction_count = getattr(
            orchestrator, "last_reactive_compaction_count", 0
        )
        swarm_result.peak_input_tokens = getattr(
            orchestrator, "last_peak_input_tokens", 0
        )
        swarm_result.reflection_stats = orchestrator.last_reflection_stats
        swarm_result.spawn_events = orchestrator.last_spawn_events
        # Layer 4b — rival-candidate audit telemetry (web-search swarm only).
        swarm_result.rival_audit = dict(
            getattr(orchestrator, "_last_rival_audit", {}) or {}
        )
        # Cheap-win + rival-sweep + critic-refine telemetry.
        swarm_result.layer4a = dict(
            getattr(orchestrator, "_last_layer4a", {}) or {}
        )
        swarm_result.cheap_win_fired = bool(
            getattr(orchestrator, "_cheap_win_fired", False)
        )
        swarm_result.rival_sweep_fired = bool(
            getattr(orchestrator, "_rival_sweep_fired", False)
        )
        # Per-agent token usage breakdown
        breakdown = orchestrator.last_token_usage_breakdown
        if breakdown:
            from dataclasses import asdict
            swarm_result.token_usage_breakdown = {
                name: asdict(usage) if hasattr(usage, "__dataclass_fields__") else str(usage)
                for name, usage in breakdown.items()
            }

        # -- Per-subagent tool call counting --------------------------------
        # Prefer the persistent counter (agent.tool_calls_by_name) which
        # survives clear_history() and compaction.  Fall back to scanning
        # messages only if the persistent counter is missing (older runs).
        from collections import Counter
        subagent_tool_counts: dict[str, dict[str, int]] = {}
        total_tools: Counter[str] = Counter()
        for sa in orchestrator.last_subagent_summaries:
            sa_name = sa.get("name", "?")
            persistent = sa.get("tool_calls_by_name", {})
            if persistent:
                # Use the persistent counter — accurate across clear_history
                sa_counts = Counter(persistent)
            else:
                # Fallback: scan remaining messages (may undercount due to
                # clear_history / compaction wiping tool_use blocks)
                sa_counts = Counter()
                for msg in sa.get("messages", []):
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                sa_counts[block.get("name", "")] += 1
            subagent_tool_counts[sa_name] = dict(sa_counts)
            total_tools += sa_counts

        swarm_result.subagent_tool_counts = subagent_tool_counts

        # -- Orchestrator's own tool call counts ----------------------------
        # Source for synthesizing ``EvalResult.tool_calls`` so swarm runs
        # report a non-zero ``num_tool_calls`` (the single-agent path
        # already does this via the event collector at runner.py:1429).
        orch_agent = getattr(orchestrator, "_orchestrator_agent", None)
        if orch_agent is not None:
            swarm_result.orchestrator_tool_counts = dict(
                getattr(orch_agent, "tool_calls_by_name", {}) or {},
            )

        # -- Collect Brave fallback logs from all subagents ------------------
        all_fallbacks: list[dict[str, Any]] = []
        for sa in orchestrator.last_subagent_summaries:
            sa_name = sa.get("name", "?")
            sa_fallbacks = sa.get("web_search_fallback_log", [])
            for entry in sa_fallbacks:
                entry["subagent"] = sa_name
            all_fallbacks.extend(sa_fallbacks)
        swarm_result.web_search_failures = all_fallbacks

        # -- Collect full search logs from all subagents ---------------------
        all_search_log: list[dict[str, Any]] = []
        for sa in orchestrator.last_subagent_summaries:
            sa_name = sa.get("name", "?")
            sa_search_log = sa.get("web_search_log", [])
            for entry in sa_search_log:
                entry["subagent"] = sa_name
            all_search_log.extend(sa_search_log)
        swarm_result.web_search_log = all_search_log

        # -- Collect full fetch logs from all subagents ----------------------
        all_fetch_log: list[dict[str, Any]] = []
        for sa in orchestrator.last_subagent_summaries:
            sa_name = sa.get("name", "?")
            sa_fetch_log = sa.get("web_fetch_log", [])
            for entry in sa_fetch_log:
                entry["subagent"] = sa_name
            all_fetch_log.extend(sa_fetch_log)
        swarm_result.web_fetch_log = all_fetch_log

        # -- Collect compactor logs from all subagents ------------------------
        all_compactor_log: list[dict[str, Any]] = []
        for sa in orchestrator.last_subagent_summaries:
            sa_name = sa.get("name", "?")
            sa_comp_log = sa.get("compactor_log", [])
            for entry in sa_comp_log:
                entry["subagent"] = sa_name
            all_compactor_log.extend(sa_comp_log)
        swarm_result.compactor_log = all_compactor_log

        # -- Collect content filter logs from all subagents -------------------
        all_cf_log: list[dict[str, Any]] = []
        for sa in orchestrator.last_subagent_summaries:
            sa_name = sa.get("name", "?")
            sa_cf_log = sa.get("content_filter_log", [])
            for entry in sa_cf_log:
                entry["subagent"] = sa_name
            all_cf_log.extend(sa_cf_log)
        swarm_result.content_filter_log = all_cf_log

        # Log summary so it's easy to spot missing web_search usage
        browsing_tools = {t: total_tools[t] for t in ("web_search", "web_fetch", "pdf_read") if total_tools[t]}
        logger.info(
            "Case %s tool calls: %s | per-subagent: %s",
            case.conv_id,
            dict(total_tools) if total_tools else "(none)",
            {name: {k: v for k, v in counts.items() if v}
             for name, counts in subagent_tool_counts.items()
             if any(counts.values())},
        )
        if not browsing_tools:
            logger.warning(
                "Case %s: NO browsing tool calls (web_search/web_fetch/pdf_read) "
                "from any subagent!",
                case.conv_id,
            )

        return swarm_result
    finally:
        orchestrator.close()


def _apply_swarm_result_to_eval_result(
    result: EvalResult,
    swarm_result: _SwarmResult,
    config_build_time: float,
) -> None:
    """Copy all fields from a completed ``_SwarmResult`` onto an ``EvalResult``.

    Shared by both the normal success path and the grace-period rescue path
    (when ``future.result()`` completes just past the outer timeout deadline).
    """
    result.response_text = swarm_result.response_text
    result.final_answer_is_refusal = Agent.is_refusal_text(swarm_result.response_text)
    result.extracted_metadata = swarm_result.extracted_metadata
    result.swarm_teammates_spawned = swarm_result.teammates_spawned
    result.swarm_bbs_message_count = swarm_result.bbs_message_count
    result.swarm_saturation_events = swarm_result.saturation_events
    result.swarm_reflection_stats = swarm_result.reflection_stats
    result.swarm_rival_audit = swarm_result.rival_audit
    result.swarm_layer4a = swarm_result.layer4a
    result.swarm_cheap_win_fired = swarm_result.cheap_win_fired
    result.swarm_rival_sweep_fired = swarm_result.rival_sweep_fired
    result.swarm_subagent_tool_counts = swarm_result.subagent_tool_counts
    result.swarm_spawn_events = swarm_result.spawn_events
    result.swarm_token_usage_breakdown = swarm_result.token_usage_breakdown
    result.compaction_count = swarm_result.compaction_count
    result.total_llm_calls = swarm_result.total_llm_calls
    result.safety_refusal_count = swarm_result.safety_refusal_count
    result.content_filter_count = swarm_result.content_filter_count
    result.thinking_only_count = swarm_result.thinking_only_count
    result.proactive_compaction_count = swarm_result.proactive_compaction_count
    result.reactive_compaction_count = swarm_result.reactive_compaction_count
    result.peak_input_tokens = swarm_result.peak_input_tokens
    result.trajectory = swarm_result.trajectory
    result.phase_timings = swarm_result.phase_timings
    result.phase_timings["config_build"] = round(config_build_time, 2)
    result.token_usage = swarm_result.token_usage
    result.num_steps = swarm_result.num_steps
    result.total_token_e2e = swarm_result.total_token_e2e
    result.web_search_failures = swarm_result.web_search_failures
    result.web_search_log = swarm_result.web_search_log
    result.web_fetch_log = swarm_result.web_fetch_log
    result.compactor_log = swarm_result.compactor_log
    result.content_filter_log = swarm_result.content_filter_log

    # Synthesize ``tool_calls`` / ``tools_used`` from per-subagent +
    # orchestrator tool counters.  Without this, the swarm path leaves
    # ``result.tool_calls = []`` (the single-agent path populates it from
    # the event collector at runner.py:1429), which makes downstream
    # metrics report ``num_tool_calls = 0`` and
    # ``avg_tool_calls_per_step(turn) = 0`` for every swarm run
    # (see ``metrics.py:824, 429``).  We don't have per-call timing /
    # input / output for the swarm path, so the records carry only the
    # tool name; that's enough for count and tools-used metrics.
    records: list[ToolCallRecord] = []
    for name, count in (swarm_result.orchestrator_tool_counts or {}).items():
        if not name:
            continue
        records.extend(ToolCallRecord(name=name) for _ in range(int(count)))
    for sa_counts in (swarm_result.subagent_tool_counts or {}).values():
        for name, count in (sa_counts or {}).items():
            if not name:
                continue
            records.extend(ToolCallRecord(name=name) for _ in range(int(count)))
    if records:
        result.tool_calls = records
        result.tools_used = list(dict.fromkeys(r.name for r in records))


def _apply_agent_tuple_to_eval_result(
    result: EvalResult,
    agent_tuple: tuple,
    collector: _EventCollector,
    agent_ref: list[Agent | None],
) -> None:
    """Copy single-agent ``_run_agent`` return tuple onto an ``EvalResult``.

    Shared by both the normal success path and the grace-period rescue path.
    """
    (response_text, messages, usage, steps, breakdown, sys_prompt,
     tool_defs, extracted_meta, compact_count, llm_calls,
     safety_refusals, content_filters, thinking_only,
     proactive_compactions, reactive_compactions, peak_input_tokens) = agent_tuple
    result.response_text = response_text
    result.extracted_metadata = extracted_meta
    result.tool_calls = collector.tool_calls
    result.tools_used = list(dict.fromkeys(tc.name for tc in collector.tool_calls))
    result.trajectory = [{
        "system_prompt": sys_prompt,
        "tools": tool_defs,
        "messages": serialize_messages(messages),
    }]
    result.token_breakdown, result.tool_call_distribution = (
        _extract_token_stats(result.trajectory)
    )
    result.token_usage = usage
    result.num_steps = steps
    result.latency_breakdown = breakdown
    _enrich_latency_breakdown(result.latency_breakdown, result.trajectory)
    result.compaction_count = compact_count
    result.total_llm_calls = llm_calls
    result.safety_refusal_count = safety_refusals
    result.content_filter_count = content_filters
    result.final_answer_is_refusal = Agent.is_refusal_text(response_text)
    result.thinking_only_count = thinking_only
    result.proactive_compaction_count = proactive_compactions
    result.reactive_compaction_count = reactive_compactions
    result.peak_input_tokens = peak_input_tokens

    agent = agent_ref[0] if agent_ref else None
    ws_tool = agent._tools.get("web_search") if agent else None
    if ws_tool and hasattr(ws_tool, "drain_fallback_log"):
        result.web_search_failures = ws_tool.drain_fallback_log()
    if ws_tool and hasattr(ws_tool, "drain_search_log"):
        result.web_search_log = ws_tool.drain_search_log()
    wf_tool = agent._tools.get("web_fetch") if agent else None
    if wf_tool and hasattr(wf_tool, "drain_fetch_log"):
        result.web_fetch_log = wf_tool.drain_fetch_log()
    if agent is not None:
        compactor = (
            getattr(agent, "_fetch_compactor", None)
            or getattr(agent, "_pdf_compactor", None)
        )
        if compactor is not None and hasattr(compactor, "drain_compactor_log"):
            result.compactor_log = compactor.drain_compactor_log()
    if agent and hasattr(agent, "drain_content_filter_log"):
        result.content_filter_log = agent.drain_content_filter_log()


def _extract_final_assistant_text(messages: list[dict[str, Any]]) -> str:
    """Pull the final assistant text block from a message list.

    Returns ``""`` if no assistant text is present.  Used by the timeout
    partial-harvest path so that runs which completed past the deadline
    (and therefore never saw their success path execute) still surface
    the orchestrator's last-spoken answer in ``EvalResult.response_text``.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            if content.strip():
                return content
            continue
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = block.get("text", "") or ""
                    if txt.strip():
                        texts.append(txt)
            if texts:
                return "\n".join(texts)
    return ""


def _harvest_live_swarm_metrics(orch: Any) -> dict[str, Any]:
    """Best-effort pull of aggregated metrics from a live swarm orchestrator.

    Prefers the ``last_*`` fields populated by ``run_swarm_turn`` (set on
    both the success path and the internal-exception path).  Falls back to
    per-agent aggregation (orchestrator agent + live subagents) when those
    fields are still at their defaults — the normal case when the outer
    timeout fires mid-turn.
    """
    metrics: dict[str, Any] = {
        "response_text": "",
        "token_usage": None,
        "num_steps": 0,
        "compaction_count": 0,
        "total_llm_calls": 0,
        "total_token_e2e": 0,
        "teammates_spawned": 0,
        "bbs_message_count": 0,
        "saturation_events": 0,
        "reflection_stats": {},
        "subagent_tool_counts": {},
        "spawn_events": [],
        "token_usage_breakdown": {},
        # Surface peak context size and the per-failure-mode counters in the
        # partial-harvest path so timeout cases still populate them.  Without
        # these, post-mortem questions like "did this run keep
        # thinking_only_count == 0?" can't be answered for any case that
        # didn't hit the success path.
        "peak_input_tokens": 0,
        "thinking_only_count": 0,
        "proactive_compaction_count": 0,
        "reactive_compaction_count": 0,
        "safety_refusal_count": 0,
        "content_filter_count": 0,
    }
    if orch is None:
        return metrics

    orch_agent = getattr(orch, "_agent", None)
    ctx = getattr(orch, "_ctx", None)
    subagents = list(getattr(ctx, "subagents", []) or []) if ctx is not None else []

    # --- response_text: prefer orchestrator's last assistant text -----------
    if orch_agent is not None and getattr(orch_agent, "messages", None):
        metrics["response_text"] = _extract_final_assistant_text(orch_agent.messages)

    # --- token aggregation --------------------------------------------------
    last_tu = getattr(orch, "last_token_usage", None)
    if last_tu is not None and getattr(last_tu, "total_tokens", 0) > 0:
        metrics["token_usage"] = last_tu
        metrics["num_steps"] = int(getattr(orch, "last_num_steps", 0) or 0)
        metrics["compaction_count"] = int(getattr(orch, "last_compaction_count", 0) or 0)
        metrics["total_llm_calls"] = int(getattr(orch, "last_total_llm_calls", 0) or 0)
        metrics["total_token_e2e"] = int(getattr(orch, "last_total_token_e2e", 0) or 0)
        metrics["saturation_events"] = int(getattr(orch, "last_saturation_events", 0) or 0)
        metrics["reflection_stats"] = dict(getattr(orch, "last_reflection_stats", {}) or {})
        try:
            from dataclasses import asdict
            raw_breakdown = getattr(orch, "last_token_usage_breakdown", {}) or {}
            metrics["token_usage_breakdown"] = {
                name: (asdict(usage) if hasattr(usage, "__dataclass_fields__") else str(usage))
                for name, usage in raw_breakdown.items()
            }
        except Exception:
            metrics["token_usage_breakdown"] = {}
    else:
        # Mid-turn timeout: aggregate from live agents.  Mirrors the logic in
        # SwarmOrchestrator.run_swarm_turn so the numbers are comparable.
        try:
            orch_usage = getattr(orch_agent, "last_turn_usage", None) if orch_agent else None
            total = orch_usage
            breakdown: dict[str, Any] = {}
            if total is not None:
                breakdown["orchestrator"] = total
            num_steps = int(getattr(orch_agent, "last_num_steps", 0) or 0) if orch_agent else 0
            compaction = int(getattr(orch_agent, "compaction_count", 0) or 0) if orch_agent else 0
            llm_calls = int(getattr(orch_agent, "total_llm_calls", 0) or 0) if orch_agent else 0
            max_sa_tokens = 0
            for sa in subagents:
                sa_usage = getattr(sa, "token_usage", None)
                if sa_usage is None:
                    continue
                if total is None:
                    total = sa_usage
                else:
                    total = total + sa_usage
                if getattr(sa_usage, "total_tokens", 0) > 0:
                    breakdown[sa.name] = sa_usage
                    if sa_usage.total_tokens > max_sa_tokens:
                        max_sa_tokens = sa_usage.total_tokens
                num_steps += int(getattr(sa, "total_num_steps", 0) or 0)
                sa_agent = getattr(sa, "agent", None)
                compaction += int(getattr(sa_agent, "compaction_count", 0) or 0)
                llm_calls += int(getattr(sa_agent, "total_llm_calls", 0) or 0)
            if total is not None and getattr(total, "total_tokens", 0) > 0:
                metrics["token_usage"] = total
                metrics["num_steps"] = num_steps
                metrics["compaction_count"] = compaction
                metrics["total_llm_calls"] = llm_calls
                orch_tokens = (
                    breakdown.get("orchestrator").total_tokens
                    if "orchestrator" in breakdown else 0
                )
                metrics["total_token_e2e"] = orch_tokens + max_sa_tokens
                try:
                    from dataclasses import asdict
                    metrics["token_usage_breakdown"] = {
                        name: (asdict(u) if hasattr(u, "__dataclass_fields__") else str(u))
                        for name, u in breakdown.items()
                    }
                except Exception:
                    metrics["token_usage_breakdown"] = {}
        except Exception:
            logger.debug(
                "Could not aggregate live swarm token usage", exc_info=True,
            )

    # --- swarm structural metrics ------------------------------------------
    metrics["teammates_spawned"] = len(subagents)
    metrics["spawn_events"] = list(getattr(ctx, "spawn_events", []) or []) if ctx is not None else []
    # Per-subagent tool call counts from persistent counter (survives compaction)
    try:
        from collections import Counter
        sa_counts: dict[str, dict[str, int]] = {}
        for sa in subagents:
            sa_agent = getattr(sa, "agent", None)
            persistent = dict(getattr(sa_agent, "tool_calls_by_name", {}) or {})
            if persistent:
                sa_counts[sa.name] = dict(Counter(persistent))
        if sa_counts:
            metrics["subagent_tool_counts"] = sa_counts
    except Exception:
        logger.debug("Could not harvest subagent tool counts", exc_info=True)

    # --- per-failure-mode counters + peak context size ---------------------
    # Prefer the orchestrator's aggregated last_* fields (set by run_swarm_turn
    # on both success and exception paths).  When those are still at default
    # (mid-turn timeout, breakdown population aborted before reaching them),
    # aggregate directly from the live agents — same shape as the orchestrator
    # would have produced, just without the post-cleanup snapshot.
    def _agg_field_from_agents(field_name: str) -> int:
        total = 0
        if orch_agent is not None:
            try:
                total += int(getattr(orch_agent, field_name, 0) or 0)
            except Exception:
                pass
        for sa in subagents:
            sa_agent = getattr(sa, "agent", None)
            if sa_agent is None:
                continue
            try:
                total += int(getattr(sa_agent, field_name, 0) or 0)
            except Exception:
                pass
        return total

    def _harvest_or_aggregate(last_attr: str, agent_attr: str) -> int:
        live = getattr(orch, last_attr, 0)
        try:
            live = int(live or 0)
        except Exception:
            live = 0
        if live > 0:
            return live
        return _agg_field_from_agents(agent_attr)

    metrics["thinking_only_count"] = _harvest_or_aggregate(
        "last_thinking_only_count", "thinking_only_count",
    )
    metrics["proactive_compaction_count"] = _harvest_or_aggregate(
        "last_proactive_compaction_count", "proactive_compaction_count",
    )
    metrics["reactive_compaction_count"] = _harvest_or_aggregate(
        "last_reactive_compaction_count", "reactive_compaction_count",
    )
    metrics["safety_refusal_count"] = _harvest_or_aggregate(
        "last_safety_refusal_count", "safety_refusal_count",
    )
    metrics["content_filter_count"] = _harvest_or_aggregate(
        "last_content_filter_count", "content_filter_count",
    )

    # peak_input_tokens lives on agent._context_budget, not on the agent
    # directly, so it has its own aggregation shape (max across agents,
    # not sum).  Match what SwarmOrchestrator.run_swarm_turn computes.
    peak_live = getattr(orch, "last_peak_input_tokens", 0)
    try:
        peak_live = int(peak_live or 0)
    except Exception:
        peak_live = 0
    if peak_live > 0:
        metrics["peak_input_tokens"] = peak_live
    else:
        peaks: list[int] = []
        if orch_agent is not None:
            cb = getattr(orch_agent, "_context_budget", None)
            if cb is not None:
                try:
                    peaks.append(int(getattr(cb, "peak_input_tokens", 0) or 0))
                except Exception:
                    pass
        for sa in subagents:
            sa_agent = getattr(sa, "agent", None)
            if sa_agent is None:
                continue
            cb = getattr(sa_agent, "_context_budget", None)
            if cb is None:
                continue
            try:
                peaks.append(int(getattr(cb, "peak_input_tokens", 0) or 0))
            except Exception:
                pass
        metrics["peak_input_tokens"] = max(peaks) if peaks else 0

    return metrics


def _capture_partial_trajectory(
    result: EvalResult,
    collector: _EventCollector,
    use_swarm: bool,
    orchestrator_ref: list[Any],
    agent_ref: list[Agent | None],
    conv_id: str,
    timing_ref: list[_TimingCollector | None] | None = None,
) -> None:
    """Best-effort capture of the agent's conversation and metrics so far.

    Called when a case times out or raises an unexpected exception so the
    trajectory JSON still contains whatever messages were exchanged before
    the failure, AND the report's ``response_text`` / ``token_usage`` /
    ``num_steps`` / swarm counters reflect the work actually performed.

    Previously this helper only populated ``result.trajectory`` and
    ``result.tool_calls`` — so a case that emitted a final answer right at
    the deadline still surfaced as ``response_text=""``, ``total_tokens=0``,
    ``num_steps=0`` in the report, which severely distorted aggregate
    metrics whenever timeouts were hit.
    """
    try:
        partial_msgs: list[dict[str, Any]] = []
        subagent_summaries: list[dict[str, Any]] = []
        orch_system_prompt = ""
        orch_tools: list[dict[str, Any]] = []
        orch = None
        agent = None

        task_summaries: list[dict[str, Any]] = []

        if use_swarm:
            orch = orchestrator_ref[0] if orchestrator_ref else None
            if orch is not None and hasattr(orch, "_agent") and orch._agent is not None:
                partial_msgs = list(orch._agent.messages)
                orch_system_prompt = orch._agent.system_prompt
                orch_tools = orch._agent._get_tool_definitions()
                # Best-effort capture of subagent/task data on timeout.
                # Try calling _capture_trajectories() which safely iterates
                # over ctx.subagents with try/except guards.  Fall back to
                # any previously-populated summaries.
                try:
                    ctx = getattr(orch, "_ctx", None)
                    task_board = getattr(orch, "_task_board", None)
                    bbs = getattr(orch, "last_bbs", None)
                    if ctx is not None and task_board is not None:
                        orch._capture_trajectories(
                            orch._agent, ctx, task_board, bbs,
                        )
                except Exception:
                    logger.warning(
                        "Could not run _capture_trajectories on timeout for %s",
                        conv_id, exc_info=True,
                    )
                if hasattr(orch, "last_subagent_summaries") and orch.last_subagent_summaries:
                    subagent_summaries = list(orch.last_subagent_summaries)
                    # Mark timeout-captured subagents as partial so the
                    # trajectory consumer can distinguish them from
                    # clean-finished conversations.
                    for entry in subagent_summaries:
                        entry.setdefault("partial", True)
                if hasattr(orch, "last_task_summaries") and orch.last_task_summaries:
                    task_summaries = list(orch.last_task_summaries)
        elif agent_ref[0] is not None:
            agent = agent_ref[0]
            partial_msgs = list(agent.messages)
            orch_system_prompt = agent.system_prompt
            orch_tools = agent._get_tool_definitions()

        if partial_msgs:
            if timing_ref and timing_ref[0] is not None:
                _inject_timings_into_messages(partial_msgs, timing_ref[0], 0)
            if use_swarm:
                result.trajectory = _serialize_swarm_trajectory(
                    partial_msgs, subagent_summaries, task_summaries,
                    orchestrator_system_prompt=orch_system_prompt,
                    orchestrator_model=(
                        getattr(getattr(orch._agent, "config", None), "model", "")
                        if orch is not None and getattr(orch, "_agent", None) is not None
                        else ""
                    ),
                    orchestrator_reasoning_effort=(
                        getattr(getattr(orch._agent, "config", None), "reasoning_effort", None)
                        if orch is not None and getattr(orch, "_agent", None) is not None
                        else None
                    ),
                    orchestrator_tools=orch_tools,
                    spawn_events=getattr(orch, "last_spawn_events", None) or None,
                )
            else:
                result.trajectory = [{
                    "system_prompt": orch_system_prompt,
                    "tools": orch_tools,
                    "messages": serialize_messages(partial_msgs),
                }]
            result.tool_calls = collector.tool_calls
            result.tools_used = list(
                dict.fromkeys(tc.name for tc in collector.tool_calls)
            )

        # ------------------------------------------------------------------
        # Harvest aggregated metrics so the report is not zeroed out on
        # timeout.  Only overwrite fields that are still at their defaults
        # so a caller that has already populated (e.g. rescue path) wins.
        # ------------------------------------------------------------------
        if use_swarm and orch is not None:
            metrics = _harvest_live_swarm_metrics(orch)
            if not result.response_text and metrics["response_text"]:
                result.response_text = metrics["response_text"]
                result.final_answer_is_refusal = Agent.is_refusal_text(metrics["response_text"])
            if result.token_usage is None and metrics["token_usage"] is not None:
                result.token_usage = metrics["token_usage"]
            if result.num_steps == 0:
                result.num_steps = metrics["num_steps"]
            if result.compaction_count == 0:
                result.compaction_count = metrics["compaction_count"]
            if result.total_llm_calls == 0:
                result.total_llm_calls = metrics["total_llm_calls"]
            if result.total_token_e2e == 0:
                result.total_token_e2e = metrics["total_token_e2e"]
            if result.swarm_teammates_spawned == 0:
                result.swarm_teammates_spawned = metrics["teammates_spawned"]
            if not result.swarm_reflection_stats:
                result.swarm_reflection_stats = metrics["reflection_stats"]
            if not result.swarm_subagent_tool_counts:
                result.swarm_subagent_tool_counts = metrics["subagent_tool_counts"]
            if not result.swarm_spawn_events:
                result.swarm_spawn_events = metrics["spawn_events"]
            if not result.swarm_token_usage_breakdown:
                result.swarm_token_usage_breakdown = metrics["token_usage_breakdown"]
            # Surface peak context size + per-failure-mode counters for cases
            # that exited via the timeout/exception path.
            if getattr(result, "peak_input_tokens", 0) == 0:
                result.peak_input_tokens = metrics["peak_input_tokens"]
            if getattr(result, "thinking_only_count", 0) == 0:
                result.thinking_only_count = metrics["thinking_only_count"]
            if getattr(result, "proactive_compaction_count", 0) == 0:
                result.proactive_compaction_count = metrics["proactive_compaction_count"]
            if getattr(result, "reactive_compaction_count", 0) == 0:
                result.reactive_compaction_count = metrics["reactive_compaction_count"]
            if getattr(result, "safety_refusal_count", 0) == 0:
                result.safety_refusal_count = metrics["safety_refusal_count"]
            if getattr(result, "content_filter_count", 0) == 0:
                result.content_filter_count = metrics["content_filter_count"]
            # Per-turn token breakdown + tool distribution from the captured
            # trajectory (keeps parity with the success path).
            if result.trajectory and not result.token_breakdown:
                try:
                    tb, tcd = _extract_token_stats(result.trajectory)
                    result.token_breakdown = tb
                    result.tool_call_distribution = tcd
                except Exception:
                    logger.debug(
                        "Could not extract token stats from partial trajectory",
                        exc_info=True,
                    )
        elif not use_swarm and agent is not None:
            # Single-agent partial harvest.
            if not result.response_text and partial_msgs:
                result.response_text = _extract_final_assistant_text(partial_msgs)
                result.final_answer_is_refusal = Agent.is_refusal_text(result.response_text)
            if result.token_usage is None:
                last_tu = getattr(agent, "last_turn_usage", None)
                if last_tu is not None and getattr(last_tu, "total_tokens", 0) > 0:
                    result.token_usage = last_tu
            if result.num_steps == 0:
                result.num_steps = int(getattr(agent, "last_num_steps", 0) or 0)
            if result.compaction_count == 0:
                result.compaction_count = int(getattr(agent, "compaction_count", 0) or 0)
            if result.total_llm_calls == 0:
                result.total_llm_calls = int(getattr(agent, "total_llm_calls", 0) or 0)
            if result.trajectory and not result.token_breakdown:
                try:
                    tb, tcd = _extract_token_stats(result.trajectory)
                    result.token_breakdown = tb
                    result.tool_call_distribution = tcd
                except Exception:
                    logger.debug(
                        "Could not extract token stats from partial trajectory",
                        exc_info=True,
                    )
    except Exception:
        logger.debug(
            "Could not capture partial trajectory for %s",
            conv_id, exc_info=True,
        )


def _recover_partial_metrics(
    result: EvalResult,
    use_swarm: bool,
    orchestrator_ref: list[Any],
    agent_ref: list[Agent | None],
    conv_id: str,
) -> None:
    """Best-effort recovery of swarm metrics after timeout/exception.

    On the normal path these are set from the SwarmResult, but on timeout
    we can still read them from the live orchestrator/context objects.
    """
    if not use_swarm:
        return
    try:
        orch = orchestrator_ref[0] if orchestrator_ref else None
        if orch is None:
            return

        ctx = getattr(orch, "_ctx", None)
        agent = getattr(orch, "_agent", None)

        # Token usage from orchestrator (best-effort)
        if hasattr(orch, "last_token_usage") and orch.last_token_usage is not None:
            result.token_usage = orch.last_token_usage
            result.num_steps = getattr(orch, "last_num_steps", 0)
            result.total_token_e2e = getattr(orch, "last_total_token_e2e", 0)

        # Estimate tokens from orchestrator + max(subagent) if not already set
        if ctx is not None and result.total_token_e2e == 0:
            subagents = list(getattr(ctx, "subagents", []))
            orch_tokens = 0
            max_sa = 0
            if agent is not None:
                try:
                    orch_tokens = getattr(agent, "total_input_tokens", 0) + getattr(agent, "total_output_tokens", 0)
                except Exception:
                    pass
            for sa in subagents:
                try:
                    sa_tok = getattr(sa.agent, "total_input_tokens", 0) + getattr(sa.agent, "total_output_tokens", 0)
                    max_sa = max(max_sa, sa_tok)
                except Exception:
                    pass
            result.total_token_e2e = orch_tokens + max_sa

        # Recover swarm-specific metrics that the normal path sets
        if ctx is not None:
            subagents = list(getattr(ctx, "subagents", []))
            result.swarm_teammates_spawned = len(subagents)

            # BBS message count
            _bbs = getattr(orch, "last_bbs", None) or getattr(orch, "_persistent_bbs", None)
            if _bbs is not None:
                try:
                    result.swarm_bbs_message_count = _bbs.message_count
                except Exception:
                    pass

            # Per-subagent tool counts
            subagent_tool_counts: dict[str, dict[str, int]] = {}
            for sa in subagents:
                try:
                    counts = getattr(sa.agent, "tool_calls_by_name", {})
                    if counts:
                        subagent_tool_counts[sa.name] = dict(counts)
                except Exception:
                    pass
            if subagent_tool_counts:
                result.swarm_subagent_tool_counts = subagent_tool_counts

    except Exception:
        logger.debug(
            "Could not recover partial metrics for %s",
            conv_id, exc_info=True,
        )


def run_single_case(
    case: EvalCase,
    base_config: ArcticswarmConfig,
    *,
    timeout_seconds: float = 300,
    max_retries: int = 0,
    agent_base_url_override: str = "",
    on_event: Any | None = None,
) -> EvalResult:
    """Execute one eval case through the arcticswarm Agent.

    Returns an :class:`EvalResult` with the response text, tool call records,
    and timing information.  Errors are captured rather than raised.

    Parameters
    ----------
    timeout_seconds:
        Maximum wall-clock seconds allowed per attempt.  When exceeded the
        attempt is abandoned and retried.
    max_retries:
        Number of *retries* after the first attempt fails (so total attempts
        is ``1 + max_retries``).
    on_event:
        Optional ``(conv_id, event) -> None`` callback for the live eval
        feed.  Each swarm / single-agent event is forwarded with this
        case's ``conv_id`` so a multiplexed feed can label the line.
    """
    result = EvalResult(case=case)

    # Bind this case's conv_id to the live-feed callback so the swarm /
    # single-agent paths can stay conv_id-agnostic.
    _case_cb = (
        (lambda ev: on_event(case.conv_id, ev)) if on_event is not None else None
    )

    t_config = time.monotonic()
    try:
        config = _build_config(case, base_config)
    except Exception as exc:
        result.error = str(exc)
        result.duration_seconds = time.monotonic() - t_config
        result.trajectory.append({
            "role": "system",
            "content": f"[eval error] Config build failed: {exc}",
            "metadata": {
                "error_type": "config_build",
                "duration_seconds": round(result.duration_seconds, 1),
            },
        })
        logger.error("Case %s config build failed: %s", case.conv_id, exc)
        return result
    config_build_time = time.monotonic() - t_config

    # Per-case agent endpoint override (multi-endpoint load distribution).
    # When the eval distributes cases across several vLLM base URLs, the
    # worker has already picked this case's endpoint; pin the whole agent
    # (orchestrator + every subagent + auditor, which all inherit
    # ``agent_model_base_url`` from this per-case config) to it.
    if agent_base_url_override:
        config.agent_model_base_url = agent_base_url_override

    # -- Per-case memory isolation -----------------------------------------
    last_error: str | None = None

    total_attempts = 1 + max_retries
    t0 = time.monotonic()

    use_swarm = config.swarm_enabled

    def _force_close_refs(
        agent_ref: list[Agent | None],
        orchestrator_ref: list[Any],
    ) -> None:
        """Forcibly close agent/orchestrator connections to interrupt zombie threads."""
        orch = orchestrator_ref[0] if orchestrator_ref else None
        if orch is not None:
            try:
                orch.close()
            except Exception:
                pass
        agent = agent_ref[0] if agent_ref else None
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass

    for attempt in range(1, total_attempts + 1):
        # Enable force_submit only on the final attempt so the orchestrator
        # can force a partial report rather than timing out again.
        if total_attempts > 1:
            config.enable_force_submit = (attempt == total_attempts)

        collector = _EventCollector()
        agent_ref: list[Agent | None] = [None]
        orchestrator_ref: list[Any] = [None] if use_swarm else []
        timing_ref: list[_TimingCollector | None] = [None]
        attempt_t0 = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1)

        # Backup timer: if future.result(timeout=...) gets stuck due to GIL
        # contention, this timer fires after an extra 30s grace period and
        # forcibly closes connections to unblock the zombie thread.
        # For swarm runs, the soft-deadline mechanism adds 300s of wrap-up
        # time, so the watchdog must account for that.
        _swarm_wrap_up = 300 if use_swarm else 0
        watchdog = threading.Timer(
            timeout_seconds + _swarm_wrap_up + 30,
            _force_close_refs,
            args=[agent_ref, orchestrator_ref],
        )
        watchdog.daemon = True
        watchdog.start()

        try:
            if use_swarm:
                future = executor.submit(_run_swarm, case, config, orchestrator_ref, timeout_seconds, _case_cb)
                # Allow extra 300s for the soft-deadline wrap-up period
                swarm_result: _SwarmResult = future.result(timeout=timeout_seconds + 300)
                _apply_swarm_result_to_eval_result(result, swarm_result, config_build_time)
            else:
                future = executor.submit(_run_agent, case, config, collector, agent_ref, timing_ref, _case_cb)
                agent_tuple = future.result(timeout=timeout_seconds)
                _apply_agent_tuple_to_eval_result(result, agent_tuple, collector, agent_ref)

            result.attempt = attempt
            result.duration_seconds = time.monotonic() - t0
            if result.latency_breakdown:
                accounted = sum(v for k, v in result.latency_breakdown.items() if not k.startswith(" "))
                result.latency_breakdown["overhead"] = round(result.duration_seconds - accounted, 2)
            last_error = None
            break  # success

        except TimeoutError:
            # Grace-period rescue: the worker thread might have finished the
            # turn a hair past the deadline (we've seen ~0.1s races at
            # 1200s budgets where future.result() raises even though the
            # orchestrator has already emitted a FINAL ANSWER).  Give it a
            # short window to surface the real result before falling back
            # to the partial-harvest path.
            rescued = False
            try:
                if use_swarm:
                    swarm_result = future.result(timeout=_TIMEOUT_GRACE_SECONDS)
                    _apply_swarm_result_to_eval_result(result, swarm_result, config_build_time)
                    rescued = True
                else:
                    agent_tuple = future.result(timeout=_TIMEOUT_GRACE_SECONDS)
                    _apply_agent_tuple_to_eval_result(result, agent_tuple, collector, agent_ref)
                    rescued = True
            except TimeoutError:
                rescued = False
            except Exception as grace_exc:
                # Worker raised within grace window — treat as failed attempt,
                # not rescued, but record the real error for visibility.
                rescued = False
                logger.debug(
                    "Case %s worker raised during timeout grace window: %s",
                    case.conv_id, grace_exc, exc_info=True,
                )

            if rescued:
                result.attempt = attempt
                result.duration_seconds = time.monotonic() - t0
                if result.latency_breakdown:
                    accounted = sum(
                        v for k, v in result.latency_breakdown.items() if not k.startswith(" ")
                    )
                    result.latency_breakdown["overhead"] = round(
                        result.duration_seconds - accounted, 2
                    )
                last_error = None
                logger.warning(
                    "Case %s finished within %ds grace past the %ds deadline — "
                    "rescued from timeout",
                    case.conv_id, _TIMEOUT_GRACE_SECONDS, int(timeout_seconds),
                )
                break  # rescued success

            elapsed = time.monotonic() - attempt_t0
            last_error = f"Timed out after {elapsed:.0f}s (limit {timeout_seconds}s)"
            # O4: count every timeout-attempt so the silent multi-attempt
            # case stays visible even after Layer-1/3 recovery clears
            # ``last_error``.  Reports can then distinguish a 12624s
            # slow-clean run from a 6300+6300+force-reported run.
            result.timeout_attempts += 1

            # Capture trajectory and metrics BEFORE force-closing.
            # close() → reset() clears last_bbs, _persistent_bbs, and
            # _orchestrator_agent, making subagent/BBS data unreachable.
            _capture_partial_trajectory(
                result, collector, use_swarm, orchestrator_ref, agent_ref, case.conv_id,
                timing_ref=timing_ref,
            )
            _recover_partial_metrics(
                result, use_swarm, orchestrator_ref, agent_ref, case.conv_id,
            )

            # ---- Force-report recovery ----------------------------------
            # Guarantee an answer even on timeout.  Three fallback layers:
            #   1. The force-report timer may have already set captured_report
            #   2. Build a report from BBS content (always available on orch)
            #   3. Extract from agent messages as last resort
            #
            # Layer 1 (LLM-generated report) is high-quality → stops retries.
            # Layer 2 (raw BBS dump) is low-quality → saved as fallback but
            #   does NOT stop retries.  A fresh attempt may produce a real answer.
            # Layer 3 (message extraction) is medium-quality → stops retries.
            if use_swarm:
                orch = orchestrator_ref[0] if orchestrator_ref else None
                if orch is not None:
                    # Layer 1: check if LLM or force-timer already wrote a report
                    _recovered = getattr(
                        getattr(orch, "_report_tool", None),
                        "captured_report", None,
                    )
                    # The force-report timer also writes BBS dumps to
                    # captured_report — detect these so we can allow retries.
                    _BBS_DUMP_PREFIX = "# Findings (auto-generated"
                    _is_bbs_dump = bool(
                        _recovered and _recovered.startswith(_BBS_DUMP_PREFIX)
                    )

                    # Layer 2: build from BBS (deterministic, no timer needed)
                    if not _recovered:
                        _bbs = getattr(orch, "last_bbs", None) or getattr(orch, "_persistent_bbs", None)
                        if _bbs is not None:
                            try:
                                bbs_msgs = _bbs.read_all()
                                if bbs_msgs:
                                    parts = []
                                    for msg in bbs_msgs:
                                        parts.append(
                                            f"[{msg.channel}] {msg.author}: "
                                            f"{msg.content}"
                                        )
                                    _recovered = (
                                        "# Findings (auto-generated — time limit reached)\n\n"
                                        + "\n\n---\n\n".join(parts[-20:])
                                    )
                                    _is_bbs_dump = True
                            except Exception:
                                pass  # BBS may be in inconsistent state

                    # Layer 3: extract from orchestrator messages
                    if not _recovered:
                        try:
                            _agent = getattr(orch, "_agent", None)
                            if _agent is not None:
                                from arcticswarm.swarm.orchestrator import _extract_answer_from_messages
                                _recovered = _extract_answer_from_messages(_agent.messages)
                        except Exception:
                            pass

                    if _recovered:
                        logger.info(
                            "Case %s: recovered answer (%d chars, bbs_dump=%s) after timeout",
                            case.conv_id, len(_recovered), _is_bbs_dump,
                        )
                        result.response_text = _recovered
                        result.final_answer_is_refusal = Agent.is_refusal_text(_recovered)
                        if not _is_bbs_dump:
                            # Layer 1 or 3: high-quality recovery → stop retries
                            last_error = None
                            # O4: tag the case so reports can distinguish
                            # silently-recovered timeouts from clean runs.
                            result.recovery_mode = "force_report_after_timeout"
                        else:
                            # Layer 2 (BBS dump): low-quality fallback — keep
                            # as response_text but allow retries to produce a
                            # better answer.  If all retries fail, this BBS
                            # dump will be the final response.
                            logger.info(
                                "Case %s: BBS dump recovered but allowing retry for better answer",
                                case.conv_id,
                            )

            # Now safe to tear down connections.
            _force_close_refs(agent_ref, orchestrator_ref)

            if last_error is None:
                # Force-report recovery succeeded — treat as success
                result.duration_seconds = time.monotonic() - t0
                break

            if attempt < total_attempts:
                logger.warning(
                    "Case %s timed out on attempt %d/%d (%.0fs), retrying...",
                    case.conv_id, attempt, total_attempts, elapsed,
                )
            else:
                logger.error(
                    "Case %s timed out on all %d attempts", case.conv_id, total_attempts,
                )

        except Exception as exc:
            import traceback
            last_error = str(exc)
            logger.error(
                "Case %s attempt %d exception traceback:\n%s",
                case.conv_id, attempt, traceback.format_exc(),
            )

            _capture_partial_trajectory(
                result, collector, use_swarm, orchestrator_ref, agent_ref, case.conv_id,
                timing_ref=timing_ref,
            )
            _recover_partial_metrics(
                result, use_swarm, orchestrator_ref, agent_ref, case.conv_id,
            )

            if attempt < total_attempts:
                logger.warning(
                    "Case %s failed on attempt %d/%d (%s), retrying...",
                    case.conv_id, attempt, total_attempts, exc,
                )
            else:
                logger.error(
                    "Case %s failed on all %d attempts: %s",
                    case.conv_id, total_attempts, exc,
                )

        finally:
            watchdog.cancel()
            _force_close_refs(agent_ref, orchestrator_ref)
            executor.shutdown(wait=False, cancel_futures=True)

    result.duration_seconds = time.monotonic() - t0

    if last_error is not None:
        result.error = last_error
        result.attempt = total_attempts

        # Append error entry to trajectory so it appears in the saved JSON
        result.trajectory.append({
            "role": "system",
            "content": f"[eval error] {last_error}",
            "metadata": {
                "error_type": "timeout" if "Timed out" in last_error else "exception",
                "attempt": total_attempts,
                "duration_seconds": round(result.duration_seconds, 1),
            },
        })

    return result


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def judge_result(
    result: EvalResult,
    judge: LLMJudge,
    is_swarm: bool = False,
    use_qa_llm: bool = False,
) -> None:
    """Run the LLM judge on an :class:`EvalResult` (mutates in-place).

    The primary judge (QA or Insight) is selected by the case's eval_mode
    but only runs when *use_qa_llm* is True (off by default).
    For BROWSECOMP_V1 / BROWSECOMP_PLUS_V1, we use the specialized browsecomp judge.
    The answer-only judge always runs alongside — it evaluates purely on
    final answer correctness (0/1/2), ignoring methodology and tool usage.
    """
    t_judge = time.monotonic()
    case = result.case
    dataset = case.dataset

    if dataset == "BROWSECOMP_V1":
        # Use specialized BrowseComp judge (matches OpenAI's grading logic)
        result.qa_result = judge.judge_browsecomp(
            question=case.question,
            answer=result.response_text,
            expected_answer=case.reference_answer,
        )
    elif dataset == "BROWSECOMP_PLUS_V1":
        # Use official BrowseComp-Plus judge (from texttron/BrowseComp-Plus)
        result.qa_result = judge.judge_browsecomp_plus(
            question=case.question,
            answer=result.response_text,
            expected_answer=case.reference_answer,
        )
    elif use_qa_llm:
        result.qa_result = judge.judge_qa(
            question=case.question,
            answer=result.response_text,
            expected_answer=case.reference_answer,
        )

    # Answer-only judge: evaluates purely on final answer correctness,
    # ignoring methodology, tools, and SQL.  Skip for external benchmarks
    # that define their own canonical metric (see CANONICAL_METRIC_ONLY_DATASETS).
    if dataset not in CANONICAL_METRIC_ONLY_DATASETS:
        result.answer_only_result = judge.judge_answer_only(
            question=case.question,
            answer=result.response_text,
            expected_answer=case.reference_answer,
            analysis_date=case.date_override,
        )

    result.phase_timings["judging"] = round(time.monotonic() - t_judge, 2)

    # Run verifiable unit test when the case provides one.
    if result.case.unit_test:
        _run_unit_test(result)


def _run_unit_test(result: EvalResult) -> None:
    """Run the verifiable unit test against the agent's JSON response.

    Tries three extraction strategies (``<json>`` tags, markdown fenced block,
    raw JSON) on :attr:`result.response_text`.  On success, sets
    :attr:`result.unit_test_score` and :attr:`result.unit_test_extracted_json`.
    """
    from arcticswarm.eval.rule_eval import (
        extract_json_from_tags,
        extract_json_from_markdown,
        extract_raw_json,
        run_one,
    )

    conv_id = result.case.conv_id
    test_code = result.case.unit_test
    text = result.response_text

    extractors = [extract_json_from_tags, extract_json_from_markdown, extract_raw_json]
    for extractor in extractors:
        try:
            parsed = extractor(text)
            if parsed is not None:
                grade = run_one(conv_id, test_code, parsed)
                result.unit_test_score = grade["score"]
                result.unit_test_extracted_json = parsed
                return
        except json.JSONDecodeError:
            continue
        except Exception:
            logger.warning("Unit test failed for %s", conv_id, exc_info=True)
            result.unit_test_score = None
            return

    # No valid JSON could be extracted from the response.
    logger.warning("No JSON extracted from response for unit test: %s", conv_id)
    result.unit_test_score = None


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-endpoint load distribution
# ---------------------------------------------------------------------------


class EndpointPool:
    """Thread-safe least-connections balancer over multiple agent base URLs.

    Each eval case acquires an endpoint when its worker thread *starts* and
    releases it when the case finishes, so the in-flight counts reflect what
    is actually running (not submit order).  ``acquire`` returns the endpoint
    with the fewest in-flight cases (ties broken round-robin), which keeps the
    ``parallel`` concurrent cases spread evenly and routes a freshly-freed
    endpoint the next queued case — i.e. each endpoint behaves as its own
    continuous queue.  With 2 endpoints and parallel=4 this runs 2 cases per
    endpoint and, when one finishes, sends the next case to that endpoint.
    """

    def __init__(self, urls: list[str]) -> None:
        self._urls = list(urls)
        self._inflight: dict[str, int] = {}
        for u in self._urls:
            self._inflight.setdefault(u, 0)
        self._rr = 0
        self._lock = threading.Lock()

    @property
    def urls(self) -> list[str]:
        return list(self._urls)

    def acquire(self) -> str:
        """Return the least-loaded endpoint (round-robin tie-break) and mark it busy."""
        with self._lock:
            n = len(self._urls)
            best: str | None = None
            for i in range(n):
                u = self._urls[(self._rr + i) % n]
                if best is None or self._inflight[u] < self._inflight[best]:
                    best = u
            self._rr = (self._rr + 1) % n
            self._inflight[best] += 1  # type: ignore[index]
            return best  # type: ignore[return-value]

    def release(self, url: str) -> None:
        with self._lock:
            if self._inflight.get(url, 0) > 0:
                self._inflight[url] -= 1


def _parse_endpoint_spec(spec: str) -> list[str]:
    """Split a (possibly comma-separated) agent_model_base_url into clean URLs."""
    if not spec:
        return []
    return [u.strip() for u in spec.split(",") if u.strip()]


def _build_endpoint_pool(base_config: ArcticswarmConfig) -> EndpointPool | None:
    """Return an EndpointPool when >=2 agent base URLs are configured, else None.

    A single endpoint (or none) keeps the existing behaviour exactly: the
    per-case config's ``agent_model_base_url`` is used unchanged.
    """
    urls = _parse_endpoint_spec(getattr(base_config, "agent_model_base_url", "") or "")
    if len(urls) < 2:
        return None
    logger.info(
        "Distributing eval cases across %d agent endpoints (least-connections): %s",
        len(urls), ", ".join(urls),
    )
    return EndpointPool(urls)


def _run_single_case_balanced(
    case: EvalCase,
    base_config: ArcticswarmConfig,
    *,
    endpoint_pool: "EndpointPool | None" = None,
    **kwargs: Any,
) -> EvalResult:
    """Run one case, acquiring/releasing an endpoint from *endpoint_pool*.

    Executes in the worker thread so the acquire reflects real concurrency.
    When ``endpoint_pool`` is None, calls :func:`run_single_case` unchanged.
    """
    if endpoint_pool is None:
        return run_single_case(case, base_config, **kwargs)
    url = endpoint_pool.acquire()
    try:
        logger.info("Case %s -> agent endpoint %s", case.conv_id, url)
        return run_single_case(case, base_config, agent_base_url_override=url, **kwargs)
    finally:
        endpoint_pool.release(url)


def run_eval(
    cases: list[EvalCase],
    base_config: ArcticswarmConfig,
    judge: LLMJudge,
    *,
    parallel: int = 3,
    on_progress: Any | None = None,
    timeout_seconds: float = 300,
    max_retries: int = 0,
    use_qa_llm: bool = False,
    on_case_event: Any | None = None,
) -> list[EvalResult]:
    """Run all *cases* through the agent and judge, returning results.

    Parameters
    ----------
    cases:
        Eval cases to run.
    base_config:
        Base config (API key, Snowflake connection, model, etc.).
    judge:
        LLM judge instance.
    parallel:
        Number of concurrent cases to run.
    on_progress:
        Optional callback ``(completed: int, total: int, result: EvalResult) -> None``.
    timeout_seconds:
        Maximum wall-clock seconds per case attempt before timeout.
    max_retries:
        Number of retries per case after the first attempt fails.
    on_case_event:
        Optional ``(conv_id, event) -> None`` callback for the live eval
        feed, forwarded to every case.
    """
    total = len(cases)
    results: list[EvalResult] = []

    run_kwargs = dict(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        on_event=on_case_event,
    )
    endpoint_pool = _build_endpoint_pool(base_config)

    if parallel <= 1:
        for i, case in enumerate(cases):
            logger.info("[%d/%d] Running %s ...", i + 1, total, case.conv_id)
            result = _run_single_case_balanced(
                case, base_config, endpoint_pool=endpoint_pool, **run_kwargs
            )
            judge_result(
                result,
                judge,
                is_swarm=base_config.swarm_enabled,
                use_qa_llm=use_qa_llm,
            )
            results.append(result)
            if on_progress:
                on_progress(i + 1, total, result)
        return results

    # Parallel execution
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        future_to_case = {
            pool.submit(
                _run_single_case_balanced, case, base_config,
                endpoint_pool=endpoint_pool, **run_kwargs,
            ): case
            for case in cases
        }
        completed = 0
        for future in as_completed(future_to_case):
            completed += 1
            result = future.result()
            logger.info(
                "[%d/%d] Completed %s (%.1fs, attempt %d)",
                completed,
                total,
                result.case.conv_id,
                result.duration_seconds,
                result.attempt,
            )
            # Judge sequentially to avoid hammering the API
            judge_result(
                result,
                judge,
                is_swarm=base_config.swarm_enabled,
                use_qa_llm=use_qa_llm,
            )
            results.append(result)
            if on_progress:
                on_progress(completed, total, result)

    return results


def run_eval_repeated(
    cases: list[EvalCase] | None,
    base_config: ArcticswarmConfig,
    judge: LLMJudge,
    *,
    num_runs: int = 1,
    cases_by_run: list[list[EvalCase]] | None = None,
    parallel: int = 3,
    on_progress: Any | None = None,
    timeout_seconds: float = 300,
    max_retries: int = 0,
    use_qa_llm: bool = False,
    gated_retry: bool = False,
    retry_threshold: float = -0.38,
    max_retry_fraction: float = 0.5,
    on_case_event: Any | None = None,
) -> list[list[EvalResult]]:
    """Run *cases* across *num_runs* repeats in a single thread pool.

    Unlike calling :func:`run_eval` in a loop, all ``num_runs * len(cases)``
    tasks share one pool so workers stay busy even when one run's last case
    is slow.

    Parameters
    ----------
    on_progress:
        Optional callback
        ``(run_idx: int, completed: int, total_per_run: int, result: EvalResult) -> None``.
    on_case_event:
        Optional ``(conv_id, event) -> None`` callback for the live eval
        feed, forwarded to every case (and gated-retry re-runs).
    """
    if cases_by_run is not None:
        if len(cases_by_run) != num_runs:
            raise ValueError(
                f"cases_by_run has {len(cases_by_run)} runs but num_runs={num_runs}"
            )
        per_run_cases = [list(run_cases) for run_cases in cases_by_run]
    else:
        per_run_cases = [list(cases or []) for _ in range(num_runs)]

    per_run_totals = [len(run_cases) for run_cases in per_run_cases]
    # Build (run_idx, case) pairs for every task
    tasks: list[tuple[int, EvalCase]] = [
        (run_idx, case)
        for run_idx, run_cases in enumerate(per_run_cases)
        for case in run_cases
    ]

    run_kwargs = dict(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        on_event=on_case_event,
    )
    endpoint_pool = _build_endpoint_pool(base_config)

    # Pre-allocate per-run result lists and completion counters
    grouped_results: list[list[EvalResult]] = [[] for _ in range(num_runs)]
    run_completed: list[int] = [0] * num_runs

    if parallel <= 1:
        for run_idx, case in tasks:
            run_completed[run_idx] += 1
            logger.info(
                "[run %d][%d/%d] Running %s ...",
                run_idx, run_completed[run_idx], per_run_totals[run_idx], case.conv_id,
            )
            result = _run_single_case_balanced(
                case, base_config, endpoint_pool=endpoint_pool, **run_kwargs
            )
            judge_result(
                result,
                judge,
                is_swarm=base_config.swarm_enabled,
                use_qa_llm=use_qa_llm,
            )
            grouped_results[run_idx].append(result)
            if on_progress:
                on_progress(run_idx, run_completed[run_idx], per_run_totals[run_idx], result)
        return grouped_results

    # Parallel execution — agent pool + judge pool running concurrently.
    # Results are handed off to the judge pool as they arrive so judging
    # does not block the result-processing loop.
    with ThreadPoolExecutor(max_workers=parallel) as agent_pool, \
         ThreadPoolExecutor(max_workers=parallel) as judge_pool:
        future_to_task = {
            agent_pool.submit(
                _run_single_case_balanced, case, base_config,
                endpoint_pool=endpoint_pool, **run_kwargs,
            ): (run_idx, case)
            for run_idx, case in tasks
        }
        judge_futures: list[Future] = []
        retry_count = 0
        max_retries_total = int(max_retry_fraction * len(tasks)) if gated_retry else 0
        for future in as_completed(future_to_task):
            run_idx, case = future_to_task[future]
            result = future.result()

            # Gated retry.  Score the result with the
            # calibrated detector; if below threshold and budget allows,
            # re-run synchronously with a different config and pick the
            # better answer.
            if gated_retry and retry_count < max_retries_total:
                from arcticswarm.eval.confidence_detector import (
                    compute_confidence_score, pick_better,
                )
                score = compute_confidence_score(result)
                if score < retry_threshold:
                    retry_count += 1
                    retry_config = _build_retry_config(base_config)
                    logger.info(
                        "[gated-retry] %s: score=%.3f < %.3f — retrying "
                        "(retry %d/%d)",
                        result.case.conv_id, score, retry_threshold,
                        retry_count, max_retries_total,
                    )
                    try:
                        retry_result = _run_single_case_balanced(
                            case, retry_config,
                            endpoint_pool=endpoint_pool, **run_kwargs,
                        )
                        better = pick_better(result, retry_result)
                        # Annotate so the runner reports retry telemetry.
                        better.swarm_layer4a = dict(
                            better.swarm_layer4a or {},
                        )
                        better.swarm_layer4a["retry_fired"] = True
                        better.swarm_layer4a["retry_score_original"] = score
                        better.swarm_layer4a["retry_score_alternative"] = (
                            compute_confidence_score(retry_result)
                        )
                        result = better
                    except Exception:
                        logger.exception(
                            "[gated-retry] %s: retry failed; using original",
                            result.case.conv_id,
                        )

            run_completed[run_idx] += 1
            logger.info(
                "[run %d][%d/%d] Completed %s (%.1fs, attempt %d)",
                run_idx,
                run_completed[run_idx],
                per_run_totals[run_idx],
                result.case.conv_id,
                result.duration_seconds,
                result.attempt,
            )
            judge_futures.append(
                judge_pool.submit(
                    judge_result,
                    result,
                    judge,
                    base_config.swarm_enabled,
                    use_qa_llm,
                )
            )
            grouped_results[run_idx].append(result)
            if on_progress:
                on_progress(run_idx, run_completed[run_idx], per_run_totals[run_idx], result)

        # Wait for all judging to finish; propagate any exception.
        for jf in as_completed(judge_futures):
            jf.result()

    return grouped_results
