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

"""Core agentic loop for Arcticswarm.

Follows a provider-agnostic tool-use pattern:
  1. Build messages with system prompt + conversation history.
  2. Call the LLM (Anthropic or OpenAI) via :class:`BaseLLMClient`.
  3. If the response contains tool_use blocks, execute each tool and append results.
  4. Loop until the model produces a final text-only response or we hit max turns.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

from arcticswarm.config import ArcticswarmConfig
from arcticswarm.llm_client import (
    BaseLLMClient,
    LLMResponse,
    create_llm_client,
    detect_provider,
)
from arcticswarm.logging_utils import (
    BrowsingContaminationStats,
    get_contamination_stats,
    log_compaction_stats_for_agent,
    log_web_fetch_stats_for_agent,
    log_web_search_stats_for_agent,
)
from arcticswarm.snowflake_client import SnowflakeClient
from arcticswarm import fallback
from arcticswarm.fallback import FallbackMixin, RepeatedEmptyFallbackError
from arcticswarm.context_management import (
    ContextManagementMixin,
    TokenUsage,
    _extract_token_usage,
)
from arcticswarm.system_prompt import build_system_prompt
from arcticswarm.tools.base import (
    BaseTool,
    ToolResult,
)
# ToolExecutionMixin powers the tool-dispatch half of the loop; _CONTAMINATION_KEYWORDS
# is re-exported here for backward-compat (tests import it from arcticswarm.agent).
from arcticswarm.tools.execution import ToolExecutionMixin, _CONTAMINATION_KEYWORDS  # noqa: F401

logger = logging.getLogger(__name__)


def _resolve_max_tool_calls(override: int | None, config_value: int) -> int:
    """Effective per-turn tool-call cap for an agent.

    A per-agent ``override`` (set on the orchestrator so the swarm leader can
    batch fan-out calls) takes precedence; ``None`` means inherit the shared
    ``config_value``.  ``0`` = unlimited, ``>=1`` = hard cap (downstream the
    cap only bites when ``value > 0``).
    """
    return override if override is not None else config_value


def _split_capped_tool_calls(
    tool_calls: list[dict[str, Any]],
    max_tc: int,
    privileged: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a tool-call batch into ``(kept, dropped)`` under a per-turn cap.

    Keeps the first ``max_tc`` calls in order, PLUS any call whose tool name is
    in ``privileged`` (regardless of position), so a finding the model tries to
    post (e.g. ``post_to_bbs``) always executes even when it isn't the first
    call.  Everything else past the cap is dropped (and stubbed by the caller so
    the model can re-issue it).  ``max_tc <= 0`` means unlimited (keep all).

    Example: ``[web_search, post_to_bbs, list_tasks]`` with ``max_tc=1`` and
    ``privileged={"post_to_bbs"}`` keeps ``[web_search, post_to_bbs]`` and drops
    ``[list_tasks]``.
    """
    if max_tc <= 0 or len(tool_calls) <= max_tc:
        return list(tool_calls), []
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    budget = max_tc
    for tc in tool_calls:
        if budget > 0:
            kept.append(tc)
            budget -= 1
        elif tc.get("name") in privileged:
            kept.append(tc)
        else:
            dropped.append(tc)
    return kept, dropped


# ---------------------------------------------------------------------------
# Event types for the UI layer
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """Base class for events emitted during a turn."""
    pass


@dataclass
class TextDelta(StreamEvent):
    """Incremental text chunk from the model."""
    text: str = ""


@dataclass
class ToolCallStart(StreamEvent):
    """A tool call is about to be executed."""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str = ""


@dataclass
class ToolCallEnd(StreamEvent):
    """A tool call finished."""
    tool_name: str = ""
    tool_use_id: str = ""
    result: ToolResult | None = None


@dataclass
class ToolInputDelta(StreamEvent):
    """Incremental JSON chunk for a tool call being streamed."""
    tool_name: str = ""
    tool_use_id: str = ""
    partial_json: str = ""


@dataclass
class TurnComplete(StreamEvent):
    """The model finished its turn (no more tool calls)."""
    stop_reason: str = ""
    token_usage: TokenUsage | None = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent(ToolExecutionMixin, ContextManagementMixin, FallbackMixin):
    """Stateful agent that manages conversation history and the tool-use loop."""

    def _make_llm_client(self, *, model: str, base_url: str = "", api_key: str = "", **kwargs: Any) -> "BaseLLMClient":
        """Construct an LLM client, centralising self-hosted vLLM routing.

        For vLLM-provider models (e.g. Qwen3.5) this ignores the caller's
        ``base_url`` (which at several sites is the Anthropic/Cortex default)
        and routes to the configured vLLM endpoint
        (``config.agent_model_base_url``), injecting the vLLM thinking/sampling
        knobs from config.  Closed-model-only kwargs (``openai_base_url``,
        ``azure_*``, ``enable_1m_context_model``, ``disable_extended_thinking``)
        are simply not forwarded on the vLLM path.

        For every other provider the call is forwarded unchanged, so existing
        behaviour is byte-for-byte preserved.
        """
        cfg = self.config
        if detect_provider(model) == "vllm":
            # Resolve the vLLM endpoint: honour an explicit self-hosted URL the
            # caller passed (e.g. the orchestration client's own endpoint), but
            # when the caller passed the Anthropic/Cortex default (sites that
            # build from config.base_url), substitute the configured vLLM
            # endpoint.
            passed = base_url or ""
            is_closed_default = (
                not passed or "anthropic" in passed or "/cortex/" in passed
            )
            vllm_url = (
                (getattr(cfg, "agent_model_base_url", "") or passed)
                if is_closed_default
                else passed
            )
            return create_llm_client(
                model=model,
                api_key="EMPTY",
                base_url=vllm_url,
                timeout=kwargs.get("timeout", getattr(cfg, "llm_timeout", None)),
                vllm_enable_thinking=getattr(cfg, "enable_thinking", True),
                vllm_temperature=getattr(cfg, "vllm_temperature", 0.6),
                vllm_top_p=getattr(cfg, "vllm_top_p", 0.95),
                vllm_top_k=getattr(cfg, "vllm_top_k", 20),
                vllm_presence_penalty=getattr(cfg, "vllm_presence_penalty", 0.0),
                vllm_served_model_id=getattr(cfg, "vllm_served_model_id", ""),
                vllm_max_model_len=getattr(cfg, "vllm_max_model_len", 262144),
                vllm_max_output_tokens=getattr(cfg, "vllm_max_output_tokens", 0),
            )
        return create_llm_client(model=model, base_url=base_url, api_key=api_key, **kwargs)

    def __init__(
        self,
        config: ArcticswarmConfig,
        is_swarm_subagent: bool = False,
    ) -> None:
        self.config = config
        self._is_swarm_subagent = is_swarm_subagent
        # Per-agent override for ``max_tool_calls_per_turn``.  ``None`` =
        # inherit ``config.max_tool_calls_per_turn`` (default).  Set to an int
        # by the orchestrator construction path so the swarm leader can have a
        # different (e.g. unlimited) per-turn budget than browsing subagents,
        # WITHOUT mutating the shared config object the subagents read from.
        self.max_tool_calls_per_turn_override: int | None = None
        self.messages: list[dict[str, Any]] = []

        # Eval-awareness: contamination filter stats tracker
        self.contamination_stats: BrowsingContaminationStats = get_contamination_stats()

        # Source quality scoring — evaluates web_fetch results automatically
        # after each tool-call batch.  Scores are appended as text annotations
        # to the web_fetch tool results (no documents are dropped).
        self._source_scorer: Any = None  # SourceScorer, set in _register_tools
        self._pending_sources: list[dict[str, Any]] = []  # accumulated web_fetch results
        self.source_scoring_query: str = ""  # task/question context for scoring

        # Web-fetch / pdf_read content compactor (opt-in via
        # use_fetch_compactor / use_pdf_compactor flags).  When set, the
        # corresponding tool's results are routed through the chunking
        # compactor instead of being accumulated for batch source scoring.
        # Both refs may point to the same singleton ContentCompactor.
        # Search-result judging on _source_scorer is unaffected.
        self._fetch_compactor: Any = None  # ContentCompactor for web_fetch
        self._pdf_compactor: Any = None    # ContentCompactor for pdf_read

        # Shared content cache for web_fetch/pdf_read deduplication.
        # Set externally by SwarmOrchestrator or eval runner before _register_tools().
        self.content_cache: Any | None = None

        # Responses API chaining: after each call the server stores the
        # conversation under response_id.  Subsequent calls pass
        # previous_response_id so only new items need to be sent.
        self._last_response_id: str | None = None
        self._msg_checkpoint: int = 0

        # Token usage for the most recent run_turn / run_turn_streaming call.
        # Reset at the start of each turn; callers can read it after the call.
        self.last_turn_usage: TokenUsage = TokenUsage()
        self.last_num_steps: int = 0
        # Populated at the end of every ``run_turn`` / ``run_turn_streaming``
        # call with the LLM's reported ``stop_reason`` (``"end_turn"``,
        # ``"max_turns"``, ``"tool_use"``, ...).  Subagents inspect this to
        # distinguish genuine completions from ``max_turns`` exhaustion so
        # that a task can be marked FAILED instead of COMPLETED with a
        # garbage partial response.
        self.last_stop_reason: str = ""

        # Context compaction stats — how many times compaction was triggered
        # vs. total LLM calls.  Read by eval harness for reporting.
        self.compaction_count: int = 0
        # Split proactive (utilization-threshold) from reactive
        # (context-too-long error) compactions.  Helps verify that
        # proactive compaction (at 90% of the context limit) ever fires.
        self.proactive_compaction_count: int = 0
        self.reactive_compaction_count: int = 0
        self.total_llm_calls: int = 0

        # Persistent tool call counter — survives clear_history() and compaction.
        # Incremented in _execute_tool() for every successful dispatch.
        self.tool_calls_by_name: dict[str, int] = {}

        # Count of duplicate tool-call results collapsed-to-stub in the outbound
        # history (when ``collapse_duplicate_tool_history`` is on). Observability.
        self._dup_collapsed_count: int = 0

        # Per-role token ledger for direct LLM calls made *by the agent itself*
        # outside the normal turn loop (currently: history compaction in
        # ``_call_compaction_llm``).  Drained by the swarm orchestrator's
        # ``_aggregate_tool_role_usage`` and surfaced as a
        # ``swarm_token_usage_breakdown`` row.  Same shape as
        # ``SourceScorer._token_ledger``.
        self._token_ledger: dict[str, dict[str, int]] = {}

        # Safety refusal count — incremented when _is_safety_refusal detects
        # a refusal from the primary model.  Read by eval harness / orchestrator.
        self.safety_refusal_count: int = 0

        # Thinking-only count — incremented when an LLM call
        # returns only thinking blocks (no text, no tool_use).  This is
        # the failure mode where adaptive-thinking opus consumes the full
        # max_tokens budget on a thinking block and emits nothing
        # actionable.  Surfaced in eval reports for observability.
        self.thinking_only_count: int = 0

        # Bail out after N consecutive empty-response fallbacks in
        # the same case.  Prevents pathological retry loops on questions
        # the model refuses to answer (e.g. browsecomp_1126's refusal
        # cascade).  Reset to 0 on any non-empty response.
        self.consecutive_empty_fallbacks: int = 0

        # Azure prompt-shield refusal count — increments only when the
        # response matches the canonical canned refusal stub (see
        # ``_AZURE_FILTER_REFUSAL_RE``).  Distinct from the broader
        # ``safety_refusal_count`` so that experiments measuring the
        # cortex-proxy fallback aren't conflated with
        # model-trained refusals.
        self.azure_refusal_count: int = 0

        # Proactive context budget tracker
        from arcticswarm.context_management import ContextBudget
        self._context_budget = ContextBudget(
            model=config.model,
            threshold_fraction=0.90,
            threshold_tokens_override=getattr(config, "context_compact_tokens", 0),
            enable_1m_context_model=getattr(config, "enable_1m_context_model", False),
        )

        # LLM client (provider-agnostic)
        # Per-model toggle: disable extended thinking for adaptive-thinking
        # models on the BrowseComp-style path (system card §2.21.1).
        _disable_thinking = getattr(
            config, "disable_extended_thinking_by_model", {}
        ).get(config.model, False)
        self.client: BaseLLMClient = self._make_llm_client(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            openai_base_url=getattr(config, "openai_base_url", ""),
            openai_api_key=getattr(config, "openai_api_key", ""),
            use_azure_openai=getattr(config, "use_azure_openai", False),
            azure_openai_api_key=getattr(config, "azure_openai_api_key", ""),
            azure_openai_endpoint=getattr(config, "azure_openai_endpoint", ""),
            azure_openai_api_version=getattr(config, "azure_openai_api_version", "2025-04-01-preview"),
            use_chat_completions=getattr(config, "use_chat_completions", False),
            timeout=getattr(config, "llm_timeout", None),
            enable_1m_context_model=getattr(config, "enable_1m_context_model", False),
            disable_extended_thinking=_disable_thinking,
        )

        # Orchestration client — may point to a self-hosted model (e.g. vLLM).
        # Tools keep using self.client with the default API credentials.
        if config.agent_model_base_url:
            self._orchestration_client: BaseLLMClient = self._make_llm_client(
                model=config.model,
                api_key="dummy",
                base_url=config.agent_model_base_url,
            )
        else:
            self._orchestration_client = self.client

        # Align the context budget with the orchestration model's real window
        # (e.g. a vLLM server's /v1/models max_model_len). Without this, a
        # served model whose window differs from the name-based table (e.g.
        # Tongyi-DeepResearch's 131072 vs the qwen* default 262144) gets a
        # wrong utilization(), so the reactive empty-response compaction
        # (which fires at >0.85 utilization) never triggers and the run
        # spirals on empty responses once the prompt nears the real window.
        _orch_window = getattr(self._orchestration_client, "_max_model_len", 0)
        if _orch_window:
            self._context_budget.context_limit_override = int(_orch_window)

        # Snowflake client (may be None if not configured). Used by the
        # Cortex corpus/web auth path — not SQL-specific.
        self.sf_client: SnowflakeClient | None = None
        if config.sf_params:
            try:
                self.sf_client = SnowflakeClient(config.sf_params)
            except Exception:
                pass  # will report when tools try to use it

        # Per-instance tool registry (avoids global state races when
        # multiple Agents run in parallel, e.g. during eval).
        self._tools: dict[str, BaseTool] = {}
        self._register_tools()

        # Optional callback for auto-injecting BBS updates between tool
        # rounds.  When set, the agent loop calls this after each tool-call
        # batch.  If it returns a non-None string, a simulated ``read_bbs``
        # tool_use + tool_result pair is injected into the conversation.
        self._auto_bbs_check: Callable[[], str | None] | None = None

        # Optional callback for auto-injecting DM updates between tool
        # rounds (same pattern as BBS).  Injected as a simulated ``read_dm``
        # tool_use + tool_result pair.
        self._auto_dm_check: Callable[[], str | None] | None = None

        # Optional callback for periodic system-reminder injection.
        # When set, called after each tool-call batch; if it returns a
        # non-None string the content is appended as a plain ``user``
        # message (wrapped in ``<system-reminder>`` tags by the caller).
        self._auto_system_reminder: Callable[[], str | None] | None = None

        # Web source tracker (optional — set by SwarmContext for tracking web_search results)
        self.web_source_tracker: Any | None = None

        # System prompt
        has_sf = self.sf_client is not None or bool(config.sf_params)
        self.system_prompt = build_system_prompt(
            has_snowflake=has_sf,
            has_web_search=config.has_web_search_capability(),
            no_web_fetch=config.no_web_fetch,
            date_override=config.date_override,
            dataset=config.dataset,
            is_swarm_subagent=is_swarm_subagent,
            model=config.model,
            prompt_style=config.prompt_style,
            max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        )

    @property
    def content_filter_count(self) -> int:
        """Number of Azure content-filter blocks (from source scorer)."""
        ss = self._source_scorer
        return ss.content_filter_count if ss is not None else 0

    def drain_content_filter_log(self) -> list[dict]:
        """Return and clear content filter rejection logs from source scorer."""
        ss = self._source_scorer
        return ss.drain_content_filter_log() if ss is not None else []

    def _register_tools(self) -> None:
        """Instantiate and register tools via :class:`ToolFactory`.

        ``config.agent_tools`` (populated from YAML or defaults) provides
        the list of tool names; the factory builds exactly those tools.
        """
        self._tools.clear()

        from arcticswarm.tools.factory import ToolFactory

        factory = ToolFactory(
            self.config,
            sf_client=self.sf_client,
            agent_client=self.client,
            messages_ref=self.messages,
            content_cache=self.content_cache,
        )

        # Declarative path — YAML controls the tool set.
        # Skill tools (load_skill, per-skill) are handled by
        # _register_skill_tools(), so filter them out here.
        factory_tools = [t for t in self.config.agent_tools if t != "load_skill"]
        self._tools.update(factory.build(factory_tools))
        self._source_scorer = factory._source_scorer
        # Both flags share one ContentCompactor singleton (see factory).
        # Each ref is None when its flag is off.
        compactor = factory._ensure_content_compactor()
        self._fetch_compactor = compactor if self.config.use_fetch_compactor else None
        self._pdf_compactor = compactor if self.config.use_pdf_compactor else None

        self._register_skill_tools()

    def _register_skill_tools(self) -> None:
        """Register skill tools (LoadSkillTool or per-skill tools).

        Uses ``config.agent_skills`` which is always populated from
        ``ToolsConfig`` defaults (with conditional skills appended by
        ``RunConfig.to_arcticswarm_config()``).
        """
        from arcticswarm.tools.skill_tools import LoadSkillTool, ReadSkillFileTool, make_per_skill_tools
        from arcticswarm.skill_loader import SkillRegistry

        skills_dir = Path(__file__).parent / "skills"
        registry = SkillRegistry(skills_dir=skills_dir)
        skill_names = list(self.config.agent_skills)

        if self.config.per_skill_tools:
            per_tools = make_per_skill_tools(
                skill_names,
                registry=registry,
                legacy_format=self.config.skill_legacy_format,
            )
            self._tools.update(per_tools)
        else:
            self._tools["load_skill"] = LoadSkillTool(
                skill_names,
                registry=registry,
                legacy_format=self.config.skill_legacy_format,
            )
        self._tools["read_skill_file"] = ReadSkillFileTool(registry=registry)

    # -- per-instance tool helpers --------------------------------------------

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions in internal (Anthropic-style) format."""
        # Snapshot values to avoid RuntimeError if another thread mutates
        # _tools concurrently (e.g. subagent profile switch during
        # _capture_trajectories).
        return [t.to_anthropic_tool() for t in list(self._tools.values())]









    def switch_connection(self, connection_name: str, sf_params: dict[str, Any]) -> None:
        """Hot-swap the Snowflake connection and re-register tools."""
        if self.sf_client is not None:
            try:
                self.sf_client.close()
            except Exception:
                pass
        self.config.sf_connection_name = connection_name
        self.config.sf_params = sf_params
        self.sf_client = SnowflakeClient(sf_params)
        self._register_tools()
        # Rebuild system prompt with updated Snowflake state
        has_sf = self.sf_client is not None or bool(self.config.sf_params)
        self.system_prompt = build_system_prompt(
            has_snowflake=has_sf,
            has_web_search=self.config.has_web_search_capability(),
            no_web_fetch=self.config.no_web_fetch,
            date_override=self.config.date_override,
            dataset=self.config.dataset,
            is_swarm_subagent=self._is_swarm_subagent,
            model=self.config.model,
            prompt_style=self.config.prompt_style,
            max_tool_calls_per_turn=self.config.max_tool_calls_per_turn,
        )

    def close(self) -> None:
        """Release resources held by the agent (Snowflake connection)."""
        if self.sf_client is not None:
            try:
                self.sf_client.close()
            except Exception:
                pass
        try:
            self.client.close()
        except Exception:
            pass
        if self._orchestration_client is not self.client:
            try:
                self._orchestration_client.close()
            except Exception:
                pass

    def clear_history(self) -> None:
        self.messages.clear()
        self._last_response_id = None
        self._msg_checkpoint = 0
        self._context_budget.reset()

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        """Server-suggested retry delay for a 429 (see ``fallback`` module)."""
        return fallback.retry_after_seconds(exc)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True for transient failures worth retrying (see ``fallback``)."""
        return fallback.is_retryable(exc)

    @staticmethod
    def _is_context_too_long(exc: Exception) -> bool:
        """True if the error indicates the prompt exceeded context limits."""
        return fallback.is_context_too_long(exc)

    @staticmethod
    def _is_max_tokens_overflow(exc: Exception) -> bool:
        """True if the error is about ``max_tokens`` being too large (see ``fallback``)."""
        return fallback.is_max_tokens_overflow(exc)

    @staticmethod
    def _has_visible_response_content(response: "LLMResponse") -> bool:
        """True when response contains actionable visible content (see ``fallback``)."""
        return fallback.has_visible_response_content(response)

    @staticmethod
    def _is_safety_refusal(response: "LLMResponse") -> bool:
        """True if the response is a safety/ethics refusal (see ``fallback``)."""
        return fallback.is_safety_refusal(response)

    @staticmethod
    def is_refusal_text(text: str) -> bool:
        """True if ``text`` matches the safety/ethics refusal regex.

        Public string counterpart of :meth:`_is_safety_refusal`, used by the
        eval runner to flag cases whose final answer was a refusal.
        """
        return fallback.is_refusal_text(text)

    @staticmethod
    def _is_no_actionable_content(response: "LLMResponse") -> bool:
        """True if blocks exist but none are actionable text/tool_use (see ``fallback``)."""
        return fallback.is_no_actionable_content(response)

    @staticmethod
    def _is_azure_filter_refusal(response: "LLMResponse") -> bool:
        """True if the response looks like Azure's prompt-shield stub (see ``fallback``)."""
        return fallback.is_azure_filter_refusal(response)

    # Cross-model fallback target (Step 3 of the empty-response fallback
    # ladder, after same-model retries with reduced / no reasoning and the
    # Azure->Cortex non-Azure reroute have failed).
    #
    # History:
    #   - Originally ``claude-4-sonnet`` (200K input context).
    #   - Switched to ``claude-sonnet-4-6`` (1M-context) on 2026-04-xx because
    #     nearly all fallback failures were "Error code: 400 ...
    #     max tokens of 200000 exceeded" from Sonnet 4 receiving > 200K input
    #     after a GPT-5.4 primary in 1M-context mode.
    #   - Reverted to ``claude-4-sonnet`` on 2026-05-19 — the current eval
    #     setup runs GPT-5.4 with ``enable_1m_context: false`` and Sonnet 4.5
    #     primary is itself 200K, so compacted history is bounded to 200K and
    #     never overflows the Sonnet 4 fallback window.  Sonnet 4 is also the
    #     judge model (eval/judge.py:28) and the codebase-wide default
    #     cross-model safety net, so it's the most-exercised target.
    #
    # Trimming contract: when the primary accumulated > 200K tokens we keep
    # the first 50K + a sentinel + the last 50K so the fallback fits within
    # Sonnet 4's window — see ``_trim_messages_for_sonnet4_fallback`` below.
    #
    # IMPORTANT: if anyone re-enables ``enable_1m_context: true`` on a GPT-5.4
    # primary, flip this back to ``claude-sonnet-4-6`` (or make the choice
    # conditional on ``self.config.enable_1m_context_model``) — otherwise the
    # original > 200K fallback regression returns.  The ``fallback_enable_1m``
    # branch below already short-circuits when the fallback target is 1M-aware,
    # so swapping the constant is sufficient.
    _EMPTY_RESPONSE_FALLBACK_MODEL = "claude-4-sonnet"

    # Sonnet 4 has a 200K input context; trigger trimming whenever the
    # primary model's input exceeds this many tokens.  After trimming we
    # send ~100K tokens of messages plus the system prompt and tools, which
    # comfortably fits the 200K cap.
    _FALLBACK_TRIM_TRIGGER_TOKENS = 200_000
    _FALLBACK_TRIM_HEAD_TOKENS = 50_000
    _FALLBACK_TRIM_TAIL_TOKENS = 50_000
    _FALLBACK_TRIM_SENTINEL = "[message trimmed because of context limit]"

    # Process-lifetime cache of (model, reasoning_effort) combinations that
    # have hit "unknown model" or 404 errors.  The Snowflake Azure proxy
    # doesn't deploy GPT-5.4 with reasoning=medium / no-reasoning, so those
    # rungs of the empty-response fallback ladder almost always fail.
    # Caching the failures avoids repeating known-dead fallback rungs.
    # Process-local (not persisted) — avoids cross-deploy poisoning.
    _DEAD_FALLBACK_COMBOS: ClassVar[set[tuple[str, str | None]]] = set()

    # Cap on consecutive empty-response fallbacks per agent instance.
    # When exceeded, ``_fallback_on_empty_response`` raises
    # ``RepeatedEmptyFallbackError`` instead of looping again.
    _MAX_CONSECUTIVE_EMPTY_FALLBACKS: ClassVar[int] = 3

    # Block types Anthropic's Messages API accepts inside message.content.
    # Any other block type (notably ``reasoning`` from the OpenAI Responses
    # API, which we preserve in ``self.messages`` for chaining) must be
    # stripped before we hand the history to Anthropic, or the request is
    # rejected at the schema layer (``messages.N.content: Field required``
    # on proxies that re-validate after stripping unknown blocks).
    _ANTHROPIC_ALLOWED_BLOCK_TYPES: frozenset[str] = frozenset({
        "text",
        "image",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
    })

    # Fallback calls do NOT enable thinking, so thinking blocks from the
    # primary model's history must be stripped or the API rejects with 400.
    _FALLBACK_ALLOWED_BLOCK_TYPES: frozenset[str] = frozenset({
        "text",
        "image",
        "tool_use",
        "tool_result",
    })










    # -- Score-aware truncation -------------------------------------------


    # -- Structured (proactive) compaction --------------------------------




    def drain_token_ledger(self) -> dict[str, dict[str, int]]:
        """Return and clear the per-role token ledger.

        Mirrors :meth:`SourceScorer.drain_token_ledger`.  Drained by
        :meth:`SwarmOrchestrator._aggregate_tool_role_usage` so direct LLM
        calls made by the agent (e.g. history compaction) show up in
        ``swarm_token_usage_breakdown``.
        """
        out = self._token_ledger
        self._token_ledger = {}
        return out

    def _record_role_usage(self, role: str, response: Any) -> None:
        """Record per-role token usage from an LLM response into the ledger.

        Same shape as :meth:`SourceScorer._record_usage`.  Reads the four
        token fields off ``response`` defensively (some providers omit
        cache fields).
        """
        bucket = self._token_ledger.setdefault(role, {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "calls": 0,
        })
        bucket["input_tokens"] += int(getattr(response, "input_tokens", 0) or 0)
        bucket["output_tokens"] += int(getattr(response, "output_tokens", 0) or 0)
        bucket["cache_creation_input_tokens"] += int(
            getattr(response, "cache_creation_input_tokens", 0) or 0,
        )
        bucket["cache_read_input_tokens"] += int(
            getattr(response, "cache_read_input_tokens", 0) or 0,
        )
        bucket["calls"] += 1





    def _append_msg(self, msg: dict[str, Any]) -> None:
        """Append a message with a wall-clock timestamp for trajectory debugging."""
        from datetime import datetime, timezone
        msg["_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        self.messages.append(msg)






    def _save_response_chain(self, response: "LLMResponse") -> None:
        """Update chaining state after appending the assistant message."""
        if getattr(self.config, "disable_responses_chaining", False):
            # Chaining off: re-send the full conversation each call.
            self._last_response_id = None
            self._msg_checkpoint = 0
            return
        # Don't extend the chain off a likely-empty turn (output_tokens < 10
        # or no content_blocks).  GPT-5.4 frequently returns end_turn with
        # ~4 output tokens; reusing that response_id propagates the bad
        # chain state into the next call, which then also empties out.
        output_tokens = getattr(response, "output_tokens", None) or 0
        if not response.content_blocks or output_tokens < 10:
            self._last_response_id = None
            self._msg_checkpoint = 0
            return
        self._last_response_id = response.response_id or None
        self._msg_checkpoint = len(self.messages)


    def _call_llm_with_retry(self) -> LLMResponse:
        """Call the LLM with targeted retry on idle-timeout and context-overflow errors.

        Backoff schedule (exponential ×2 starting at 5.0 s, total 5 attempts):
        attempts fire at t = 0, 5, 15, 35, 75 s.

        Tuned for Azure OpenAI 429 rate-limit windows (typically 60 s): the
        longer base delay plus 5 attempts span >2 minutes, allowing the quota
        bucket to refill. When the server returns a ``Retry-After`` header,
        we honor it (using the larger of the hint and our local delay).
        """
        max_attempts = 5
        delay_s = 5.0
        last_exc: Exception | None = None
        compacted = False
        self.total_llm_calls += 1

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._orchestration_client.call(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system_prompt=self.system_prompt,
                    tools=self._get_tool_definitions(),
                    messages=self._msgs_for_llm(),
                    reasoning_effort=self.config.reasoning_effort,
                    previous_response_id=self._last_response_id,
                )
                # Empty content / refusal recovery: web-search runs and
                # refusals use the cross-model fallback (claude-4-sonnet).
                is_refusal = self._is_safety_refusal(response)
                # Defensive backstop for adaptive-thinking opus that
                # fills max_tokens with a thinking block and emits no
                # actionable text/tool_use.  Treat the same as empty.
                is_no_actionable = self._is_no_actionable_content(response)
                if is_no_actionable and not is_refusal:
                    self.thinking_only_count += 1
                    logger.warning(
                        "Primary model returned thinking-only content "
                        "(no text, no tool_use). Treating as empty for "
                        "fallback purposes (count=%d).",
                        self.thinking_only_count,
                    )
                empty_or_refusal = (
                    (not self._has_visible_response_content(response)) or is_refusal
                )
                if empty_or_refusal and self.config.web_search_enabled:
                    is_azure_refusal = self._is_azure_filter_refusal(response)
                    if is_azure_refusal:
                        self.azure_refusal_count += 1
                    if is_refusal:
                        self.safety_refusal_count += 1
                        logger.warning(
                            "Primary model returned safety refusal — "
                            "triggering fallback (same as empty response). "
                            "azure_filter_refusal=%s",
                            is_azure_refusal,
                        )
                    try:
                        response = self._fallback_on_empty_response(
                            streaming=False,
                            primary_stop_reason=response.stop_reason,
                            primary_usage={
                                "input_tokens": response.input_tokens,
                                "output_tokens": response.output_tokens,
                                "cache_read": response.cache_read_input_tokens,
                                "cache_create": response.cache_creation_input_tokens,
                            },
                            is_azure_refusal=is_azure_refusal,
                        )
                    except RepeatedEmptyFallbackError as bail_exc:
                        logger.error(
                            "Empty-fallback bail-out: %s — returning the "
                            "last empty response so the caller can end "
                            "the case gracefully.", bail_exc,
                        )
                    except Exception:
                        logger.warning(
                            "All empty-response fallbacks failed.",
                            exc_info=True,
                        )
                else:
                    # Clean primary response — reset the streak.
                    self.consecutive_empty_fallbacks = 0
                return response
            except Exception as exc:
                last_exc = exc

                # Context too long — compact and retry once (web-search mode only).
                # For "max_tokens overflow" specifically (input_tokens +
                # max_tokens > model context window, OR max_tokens > model
                # parameter cap), compaction may not be sufficient because the
                # bumped thinking budget alone can exceed the cap. After
                # compacting (or if compaction is unavailable), route to the
                # reduced-thinking-budget fallback chain.
                is_max_tokens_overflow = self._is_max_tokens_overflow(exc)
                if (not compacted
                    and self.config.has_web_search_capability()
                    and self._is_context_too_long(exc)):
                    if self._compact_context():
                        compacted = True
                        self.compaction_count += 1
                        self.reactive_compaction_count += 1  # context-too-long
                        if not is_max_tokens_overflow:
                            # Pure context-too-long: compaction alone is enough.
                            continue
                        # max_tokens overflow: compaction may help if input was
                        # the dominant term, but we still want to fall through
                        # to the fallback chain on the next failure.
                        continue
                    # Compaction failed (or wasn't applicable). For
                    # max_tokens overflow, try the reduced-thinking fallback
                    # chain before giving up.
                    if is_max_tokens_overflow and self.config.web_search_enabled:
                        logger.warning(
                            "max_tokens overflow with no compaction headroom — "
                            "routing to reduced-thinking-budget fallback chain."
                        )
                        try:
                            return self._fallback_on_empty_response(
                                streaming=False,
                                primary_stop_reason="max_tokens_overflow",
                                primary_usage=None,
                            )
                        except Exception:
                            logger.warning(
                                "max_tokens-overflow fallback chain also failed.",
                                exc_info=True,
                            )
                    raise

                # max_tokens overflow that didn't trigger compaction (e.g.
                # web search disabled or compaction already used). Try the
                # fallback chain directly.
                if (is_max_tokens_overflow
                        and self.config.web_search_enabled):
                    logger.warning(
                        "max_tokens overflow detected (compaction unavailable "
                        "or already used) — routing to reduced-thinking-budget "
                        "fallback chain. Error: %s",
                        exc,
                    )
                    try:
                        return self._fallback_on_empty_response(
                            streaming=False,
                            primary_stop_reason="max_tokens_overflow",
                            primary_usage=None,
                        )
                    except Exception:
                        logger.warning(
                            "max_tokens-overflow fallback chain also failed.",
                            exc_info=True,
                        )
                    raise

                if not self._is_retryable(exc) or attempt == max_attempts:
                    raise
                err_text = str(exc).lower()
                if self._last_response_id and (
                    "reasoning" in err_text
                    or "previous_response_not_found" in err_text
                    or "previous_response_id" in err_text
                    or (
                        "previous response with id" in err_text
                        and "not found" in err_text
                    )
                    # Azure (cortex-eastus-dev) wording for the same
                    # "stored response evicted" condition. See the
                    # matching note in _is_retryable above.
                    or (
                        "item with id" in err_text
                        and "not found" in err_text
                    )
                ):
                    self._last_response_id = None
                    self._msg_checkpoint = 0
                server_hint = self._retry_after_seconds(exc)
                actual_delay = max(server_hint or 0.0, delay_s)
                logger.warning(
                    "LLM call failed (attempt %s/%s: %s), retrying in %.1fs%s...",
                    attempt, max_attempts, exc, actual_delay,
                    f" (server Retry-After={server_hint:.1f}s)" if server_hint else "",
                )
                time.sleep(actual_delay)
                delay_s *= 2

        # Unreachable; kept for typing completeness.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable")

    def run_turn(
        self,
        user_message: str | list[dict[str, Any]],
        on_event: Callable[[StreamEvent], None] | None = None,
        max_turns: int | None = None,
    ) -> str:
        """Run one full user turn (may involve multiple LLM round-trips for tool use).

        Parameters
        ----------
        user_message:
            Either a plain string (text-only user turn) or a pre-built list
            of Anthropic-shape content blocks (e.g. image + text) for
            multimodal inputs. List content is passed through unchanged and
            translated to the active provider's shape by the LLM client.
        max_turns:
            Override ``self.config.max_turns`` for this call only.
            Useful for budgeting turns across multiple phases.

        Returns the final text response.
        """
        self._append_msg({"role": "user", "content": user_message})

        effective_max_turns = max_turns if max_turns is not None else self.config.max_turns

        full_text = ""
        turns = 0
        turn_usage = TokenUsage()

        while turns < effective_max_turns:
            turns += 1
            self._nudge_retried = False
            response = self._call_llm_with_retry()

            # Accumulate token usage from this LLM call
            turn_usage += _extract_token_usage(response)
            self.last_turn_usage = turn_usage
            if self.config.has_web_search_capability():
                self._context_budget.update(
                    response.input_tokens
                    + response.cache_creation_input_tokens
                    + response.cache_read_input_tokens
                )

            # Collect text and tool-use blocks from the response
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for block in response.content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                    if on_event:
                        on_event(TextDelta(text=block.get("text", "")))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "name": block["name"],
                        "input": block.get("input", {}),
                    })

            text_chunk = "".join(text_parts)
            full_text += text_chunk

            # Append the assistant message (normalised plain-dict blocks)
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content_blocks}
            assistant_msg["_llm_output_tokens"] = response.output_tokens
            assistant_msg["_llm_reasoning_tokens"] = response.reasoning_tokens
            self._append_msg(assistant_msg)
            self._save_response_chain(response)

            if not tool_calls:
                # If the response was completely empty (no text, no tools)
                # and we haven't nudged yet, inject a nudge and retry
                # inline.  Roll back if the nudge also fails.
                if (
                    not text_parts
                    and not getattr(self, "_nudge_retried", False)
                    and self.config.web_search_enabled
                    and not self._is_safety_refusal(response)
                ):
                    self._nudge_retried = True
                    nudge_user = {"role": "user", "content": (
                        "Your previous response was empty. You MUST respond "
                        "with either a tool call (e.g. web_search, "
                        "post_to_bbs, complete_task) or a text message. "
                        "Do not return an empty response."
                    )}
                    self._append_msg(nudge_user)
                    logger.info(
                        "Empty response — injecting nudge message and "
                        "retrying primary model (non-streaming, turn %d).",
                        turns,
                    )
                    nudge_response = self._call_llm_with_retry()
                    # A thinking-only nudge response is no better than
                    # the original empty turn — gate on actionable content.
                    nudge_actionable = (
                        nudge_response.content_blocks
                        and not self._is_no_actionable_content(nudge_response)
                    )
                    if nudge_actionable:
                        # Nudge worked — replace response and continue
                        # processing tool calls / text below.
                        response = nudge_response
                        text_parts = [
                            b.get("text", "") for b in response.content_blocks
                            if b.get("type") == "text"
                        ]
                        tool_calls = [
                            {"id": b["id"], "name": b["name"], "input": b.get("input", {})}
                            for b in response.content_blocks
                            if b.get("type") == "tool_use"
                        ]
                        text_chunk = "".join(text_parts)
                        full_text += text_chunk
                        # Re-append assistant msg (the empty one was already
                        # appended above; now append the real one)
                        assistant_msg = {"role": "assistant", "content": response.content_blocks}
                        assistant_msg["_llm_output_tokens"] = response.output_tokens
                        assistant_msg["_llm_reasoning_tokens"] = response.reasoning_tokens
                        self._append_msg(assistant_msg)
                        self._save_response_chain(response)
                        if not tool_calls:
                            stop_reason = response.stop_reason or "end_turn"
                            self.last_stop_reason = stop_reason
                            if on_event:
                                on_event(TurnComplete(
                                    stop_reason=stop_reason,
                                    token_usage=turn_usage,
                                ))
                            break
                        # Fall through to tool execution below
                    else:
                        # Nudge failed — roll back
                        self.messages.pop()  # nudge user
                        logger.info("Nudge retry also returned empty — rolled back.")
                        stop_reason = response.stop_reason or "end_turn"
                        self.last_stop_reason = stop_reason
                        if on_event:
                            on_event(TurnComplete(
                                stop_reason=stop_reason,
                                token_usage=turn_usage,
                            ))
                        break
                else:
                    stop_reason = response.stop_reason or "end_turn"
                    self.last_stop_reason = stop_reason
                    if on_event:
                        on_event(TurnComplete(
                            stop_reason=stop_reason,
                            token_usage=turn_usage,
                        ))
                    break

            # Enforce max tool calls per turn if configured.  An optional
            # per-agent override (set by the orchestrator construction path)
            # takes precedence over the shared config so the swarm leader can
            # batch fan-out calls while browsing subagents stay capped.  Tools
            # in ``always_execute_tools_per_turn`` (e.g. post_to_bbs) bypass the
            # cap regardless of position so a posted finding always lands.
            max_tc = _resolve_max_tool_calls(
                self.max_tool_calls_per_turn_override,
                self.config.max_tool_calls_per_turn,
            )
            dropped: list[dict[str, Any]] = []
            if max_tc > 0 and len(tool_calls) > max_tc:
                privileged = frozenset(
                    getattr(self.config, "always_execute_tools_per_turn", None) or ()
                )
                tool_calls, dropped = _split_capped_tool_calls(
                    tool_calls, max_tc, privileged,
                )
                if dropped:
                    logger.info(
                        "Capped tool calls: kept %d (max=%d, +%d privileged), "
                        "dropped %d (max_tool_calls_per_turn)",
                        len(tool_calls), max_tc,
                        max(0, len(tool_calls) - max_tc), len(dropped),
                    )

            # Execute this turn's tool calls (sequentially; see _execute_tools_batch).
            tool_results = self._execute_tools_batch(tool_calls, on_event)

            # Add stub results for dropped tool calls so the message history
            # stays consistent (required by OpenAI Responses API chaining).
            for tc in dropped:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": "(skipped — max tool calls per turn reached)",
                    "is_error": True,
                })
            self._append_msg({"role": "user", "content": tool_results})

            if self._tool_batch_terminates_turn(tool_calls, tool_results):
                stop_reason = "end_turn"
                self.last_stop_reason = stop_reason
                if on_event:
                    on_event(TurnComplete(
                        stop_reason=stop_reason,
                        token_usage=turn_usage,
                    ))
                break

            # Auto-score web_fetch sources from this batch
            self._maybe_score_sources(tool_calls, tool_results)

            # Collapse consecutive empty BBS/DM polls to save context tokens
            self._prune_empty_polls()

            # Auto-inject BBS updates as a simulated read_bbs tool call
            self._maybe_inject_bbs_update()

            # Proactively compact if approaching context limit (web-search mode only)
            if self.config.has_web_search_capability():
                self._maybe_proactive_compact()
        else:
            # Exceeded max turns
            full_text += f"\n\n(Reached max turns: {effective_max_turns})"
            self.last_stop_reason = "max_turns"
            if on_event:
                on_event(TurnComplete(stop_reason="max_turns", token_usage=turn_usage))

        self.last_turn_usage = turn_usage
        self.last_num_steps = turns
        log_compaction_stats_for_agent(self)
        log_web_search_stats_for_agent(self)
        log_web_fetch_stats_for_agent(self)
        return full_text


    # -- Empty-poll pruning --------------------------------------------------

    _EMPTY_POLL_TOOLS = frozenset({"read_bbs", "read_dm"})
    _EMPTY_POLL_TEXTS = frozenset({
        "(no new messages)",
        "No new direct messages.",
    })
    _EMPTY_POLL_THRESHOLD = 3  # minimum consecutive empties before collapsing

    def _prune_empty_polls(self) -> None:
        """Collapse consecutive empty read_bbs/read_dm pairs in self.messages.

        Scans backwards from the end of ``self.messages``.  When N consecutive
        (assistant tool_use, user tool_result) pairs are found where the tool
        is in ``_EMPTY_POLL_TOOLS`` and the result text matches
        ``_EMPTY_POLL_TEXTS``, the oldest N-1 pairs are replaced with a single
        collapsed note.  The most recent empty poll is always preserved so the
        LLM knows BBS/DM was just checked.

        Only triggers when there are ≥ ``_EMPTY_POLL_THRESHOLD`` consecutive
        empty polls (to avoid churn on occasional checks).
        """
        msgs = self.messages
        if len(msgs) < self._EMPTY_POLL_THRESHOLD * 2:
            return  # not enough messages to bother

        # Walk backwards collecting indices of consecutive empty poll pairs.
        # Each pair is (assistant_idx, user_idx) where user_idx = assistant_idx + 1.
        empty_pairs: list[tuple[int, int]] = []
        i = len(msgs) - 1
        while i >= 1:
            user_msg = msgs[i]
            asst_msg = msgs[i - 1]

            if (
                asst_msg.get("role") != "assistant"
                or user_msg.get("role") != "user"
            ):
                break

            # Check assistant message: exactly one tool_use in _EMPTY_POLL_TOOLS
            asst_content = asst_msg.get("content")
            if not isinstance(asst_content, list) or len(asst_content) != 1:
                break
            block = asst_content[0]
            if (
                not isinstance(block, dict)
                or block.get("type") != "tool_use"
                or block.get("name") not in self._EMPTY_POLL_TOOLS
            ):
                break

            # Check user message: exactly one tool_result with empty-poll text
            user_content = user_msg.get("content")
            if not isinstance(user_content, list) or len(user_content) != 1:
                break
            result_block = user_content[0]
            if (
                not isinstance(result_block, dict)
                or result_block.get("type") != "tool_result"
            ):
                break

            result_text = self._extract_tool_result_text(result_block)
            if result_text not in self._EMPTY_POLL_TEXTS:
                break

            empty_pairs.append((i - 1, i))
            i -= 2

        if len(empty_pairs) < self._EMPTY_POLL_THRESHOLD:
            return

        # Keep the most recent pair (first in our reversed list).
        pairs_to_remove = empty_pairs[1:]  # remove all but the newest
        if not pairs_to_remove:
            return

        # Determine tool name for the collapsed note
        first_pair_asst = msgs[empty_pairs[0][0]]
        tool_name = first_pair_asst["content"][0].get("name", "read_bbs")
        count = len(pairs_to_remove)

        # Collect indices to remove (sorted descending so we can pop in order)
        indices_to_remove: list[int] = []
        for asst_idx, user_idx in pairs_to_remove:
            indices_to_remove.append(user_idx)
            indices_to_remove.append(asst_idx)
        indices_to_remove.sort(reverse=True)

        for idx in indices_to_remove:
            msgs.pop(idx)

        # Insert a collapsed note pair where the removed pairs were.
        # The remaining (newest) pair is at the end, so insert before it.
        insert_pos = len(msgs) - 2  # before the kept (assistant, user) pair
        fake_id = f"pruned_{uuid.uuid4().hex[:8]}"
        msgs.insert(insert_pos, {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": fake_id, "name": tool_name, "input": {}},
            ],
        })
        msgs.insert(insert_pos + 1, {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": fake_id,
                    "content": f"({count} earlier empty {tool_name} polls collapsed)",
                },
            ],
        })

        logger.info("Pruned %d empty %s polls from context", count, tool_name)

    @staticmethod
    def _extract_tool_result_text(result_block: dict[str, Any]) -> str:
        """Extract the text from a tool_result content field.

        Handles both formats:
        - String: ``{"content": "(no new messages)"}``
        - List:   ``{"content": [{"type": "text", "text": "(no new messages)"}]}``
        """
        content = result_block.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            for sub in content:
                if isinstance(sub, dict) and sub.get("type") == "text":
                    return sub.get("text", "").strip()
        return ""

    # -- BBS auto-injection --------------------------------------------------

    def _maybe_inject_bbs_update(self) -> None:
        """Inject simulated tool calls for new BBS and/or DM messages.

        Called after each tool-call batch.  For each active check callback
        (``_auto_bbs_check``, ``_auto_dm_check``), if it returns content,
        we append a fake assistant ``tool_use`` message followed by a user
        ``tool_result`` message so the LLM sees the new posts/messages
        exactly as if it had called the tool itself.
        """
        # BBS injection
        if self._auto_bbs_check is not None:
            new_content = self._auto_bbs_check()
            if new_content is not None:
                fake_id = f"auto_bbs_{uuid.uuid4().hex[:8]}"
                self._append_msg({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": fake_id, "name": "read_bbs", "input": {}},
                    ],
                })
                self._append_msg({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": fake_id,
                            "content": new_content,
                        },
                    ],
                })

        # DM injection
        if self._auto_dm_check is not None:
            dm_content = self._auto_dm_check()
            if dm_content is not None:
                fake_id = f"auto_dm_{uuid.uuid4().hex[:8]}"
                self._append_msg({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": fake_id, "name": "read_dm", "input": {}},
                    ],
                })
                self._append_msg({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": fake_id,
                            "content": dm_content,
                        },
                    ],
                })

        # System-reminder injection (plain user message, not fake tool call)
        if self._auto_system_reminder is not None:
            reminder = self._auto_system_reminder()
            if reminder is not None:
                self._append_msg({"role": "user", "content": reminder})

    def _call_llm(self) -> LLMResponse:
        """Make a single LLM call (convenience alias)."""
        return self._call_llm_with_retry()

    def run_turn_streaming(
        self,
        user_message: str | list[dict[str, Any]],
        on_event: Callable[[StreamEvent], None] | None = None,
        max_turns: int | None = None,
    ) -> str:
        """Like :meth:`run_turn` but streams text deltas token-by-token.

        Parameters
        ----------
        user_message:
            Either a plain string or a pre-built list of Anthropic-shape
            content blocks (e.g. image + text) for multimodal inputs.
        max_turns:
            Override ``self.config.max_turns`` for this call only.
        """
        self._append_msg({"role": "user", "content": user_message})

        effective_max_turns = max_turns if max_turns is not None else self.config.max_turns

        full_text = ""
        turns = 0
        turn_usage = TokenUsage()

        while turns < effective_max_turns:
            turns += 1
            self._nudge_retried = False

            tool_calls: list[dict[str, Any]] = []

            # See ``_call_llm_with_retry`` for the rationale: 5 attempts,
            # exponential ×2 starting at 5.0 s, with Retry-After honoring.
            max_attempts = 5
            delay_s = 5.0
            compacted_streaming = False
            self.total_llm_calls += 1
            for attempt in range(1, max_attempts + 1):
                try:
                    def _on_text(text: str) -> None:
                        nonlocal full_text
                        full_text += text
                        if on_event:
                            on_event(TextDelta(text=text))

                    def _on_tool_input(name: str, tid: str, partial: str) -> None:
                        if on_event:
                            on_event(ToolInputDelta(
                                tool_name=name,
                                tool_use_id=tid,
                                partial_json=partial,
                            ))

                    response = self._orchestration_client.call_streaming(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        system_prompt=self.system_prompt,
                        tools=self._get_tool_definitions(),
                        messages=self._msgs_for_llm(),
                        reasoning_effort=self.config.reasoning_effort,
                        previous_response_id=self._last_response_id,
                        on_text_delta=_on_text,
                        on_tool_input_delta=_on_tool_input,
                    )
                    break
                except Exception as exc:
                    # Context too long — compact and retry once (web-search mode only).
                    # See _call_llm_with_retry for the full rationale; we mirror
                    # its max_tokens-overflow routing to the reduced-thinking
                    # fallback chain when compaction is unavailable or
                    # insufficient.
                    is_max_tokens_overflow = self._is_max_tokens_overflow(exc)
                    if (not compacted_streaming
                        and self.config.has_web_search_capability()
                        and self._is_context_too_long(exc)):
                        if self._compact_context():
                            compacted_streaming = True
                            self.compaction_count += 1
                            self.reactive_compaction_count += 1  # context-too-long
                            continue
                        # Compaction unavailable — for max_tokens overflow,
                        # try the reduced-thinking fallback before giving up.
                        if is_max_tokens_overflow and self.config.web_search_enabled:
                            logger.warning(
                                "Streaming max_tokens overflow with no "
                                "compaction headroom — routing to "
                                "reduced-thinking-budget fallback chain."
                            )
                            try:
                                response = self._fallback_on_empty_response(
                                    streaming=True,
                                    on_text_delta=_on_text,
                                    on_tool_input_delta=_on_tool_input,
                                    primary_stop_reason="max_tokens_overflow",
                                    primary_usage=None,
                                )
                                break
                            except Exception:
                                logger.warning(
                                    "Streaming max_tokens-overflow fallback "
                                    "chain also failed.",
                                    exc_info=True,
                                )
                        raise

                    # max_tokens overflow that didn't trigger compaction (e.g.
                    # web search disabled or compaction already used). Try the
                    # fallback chain directly.
                    if (is_max_tokens_overflow
                            and self.config.web_search_enabled):
                        logger.warning(
                            "Streaming max_tokens overflow detected "
                            "(compaction unavailable or already used) — "
                            "routing to reduced-thinking-budget fallback "
                            "chain. Error: %s",
                            exc,
                        )
                        try:
                            response = self._fallback_on_empty_response(
                                streaming=True,
                                on_text_delta=_on_text,
                                on_tool_input_delta=_on_tool_input,
                                primary_stop_reason="max_tokens_overflow",
                                primary_usage=None,
                            )
                            break
                        except Exception:
                            logger.warning(
                                "Streaming max_tokens-overflow fallback "
                                "chain also failed.",
                                exc_info=True,
                            )
                        raise

                    if not self._is_retryable(exc) or attempt == max_attempts:
                        raise
                    err_text = str(exc).lower()
                    if self._last_response_id and (
                        "reasoning" in err_text
                        or "previous_response_not_found" in err_text
                        or "previous_response_id" in err_text
                        or (
                            "previous response with id" in err_text
                            and "not found" in err_text
                        )
                        # Azure (cortex-eastus-dev) wording for the same
                        # "stored response evicted" condition. See the
                        # matching note in _is_retryable above.
                        or (
                            "item with id" in err_text
                            and "not found" in err_text
                        )
                    ):
                        self._last_response_id = None
                        self._msg_checkpoint = 0
                    server_hint = self._retry_after_seconds(exc)
                    actual_delay = max(server_hint or 0.0, delay_s)
                    logger.warning(
                        "Streaming LLM call failed (attempt %s/%s: %s), retrying in %.1fs%s...",
                        attempt, max_attempts, exc, actual_delay,
                        f" (server Retry-After={server_hint:.1f}s)" if server_hint else "",
                    )
                    time.sleep(actual_delay)
                    delay_s *= 2

            # Handle empty response or safety refusal: web-search runs and
            # refusals go through the cross-model fallback (claude-4-sonnet).
            is_refusal = self._is_safety_refusal(response)
            # Defensive backstop for adaptive-thinking opus.
            is_no_actionable = self._is_no_actionable_content(response)
            if is_no_actionable and not is_refusal:
                self.thinking_only_count += 1
                logger.warning(
                    "Streaming: primary model returned thinking-only "
                    "content (count=%d).", self.thinking_only_count,
                )
            empty_or_refusal = (
                (not self._has_visible_response_content(response)) or is_refusal
            )

            if empty_or_refusal and self.config.web_search_enabled:
                if is_refusal:
                    self.safety_refusal_count += 1
                    logger.warning(
                        "Streaming: primary model returned safety refusal — "
                        "triggering fallback."
                    )
                # Empty responses often happen when context is near-full.
                # Try compaction + retry with primary model before falling back.
                if (
                    not is_refusal  # don't retry primary model for refusals
                    and not compacted_streaming
                    and self._context_budget.utilization() > 0.85
                    and self._compact_context()
                ):
                    compacted_streaming = True
                    self.compaction_count += 1
                    self.reactive_compaction_count += 1  # empty-response @ high util
                    logger.info(
                        "Empty response at %.0f%% context — compacted and retrying primary model.",
                        self._context_budget.utilization() * 100,
                    )
                    continue

                # --- Nudge retry: inject a short user message and retry the
                # same model+reasoning.  The full prompt is already cached so
                # this is very cheap.  The nudge breaks the model out of the
                # "think then say nothing" state.
                # If the nudge also fails, ROLL BACK the nudge messages so
                # the fallback chain sees a clean history.
                if not is_refusal and not getattr(self, "_nudge_retried", False):
                    self._nudge_retried = True
                    nudge_assistant = {"role": "assistant", "content": [
                        {"type": "text", "text": "(empty response)"},
                    ]}
                    nudge_user = {"role": "user", "content": (
                        "Your previous response was empty. You MUST respond with "
                        "either a tool call (e.g. web_search, post_to_bbs, "
                        "complete_task) or a text message. Do not return an "
                        "empty response."
                    )}
                    self._append_msg(nudge_assistant)
                    self._append_msg(nudge_user)
                    logger.info(
                        "Empty response — injecting nudge message and retrying "
                        "primary model (turn %d).", turns,
                    )
                    # Retry inline with the same model+reasoning
                    try:
                        nudge_response = self._orchestration_client.call_streaming(
                            model=self.config.model,
                            max_tokens=self.config.max_tokens,
                            system_prompt=self.system_prompt,
                            tools=self._get_tool_definitions(),
                            messages=self._msgs_for_llm(),
                            reasoning_effort=self.config.reasoning_effort,
                            on_text_delta=_on_text,
                            on_tool_input_delta=_on_tool_input,
                        )
                    except Exception:
                        nudge_response = None
                    # A thinking-only nudge response is no better than the
                    # original empty turn — gate on actionable content, mirroring
                    # the non-streaming path.
                    nudge_actionable = (
                        nudge_response is not None
                        and nudge_response.content_blocks
                        and not self._is_no_actionable_content(nudge_response)
                    )
                    if nudge_actionable:
                        response = nudge_response
                        # Success — keep nudge messages, fall through to
                        # normal tool_call / text processing below.
                    else:
                        # Nudge failed — roll back the 2 nudge messages
                        # so the fallback chain sees clean history.
                        self.messages.pop()  # nudge user
                        self.messages.pop()  # nudge assistant
                        logger.info("Nudge retry also returned empty — rolled back nudge messages.")

                # Context isn't the issue (or compaction/nudge already tried) — try fallback model
                # Refusal also triggers fallback even though content_blocks
                # are non-empty (the canned refusal text is not a real answer).
                # Thinking-only responses also trigger fallback — content
                # blocks present but no text/tool_use means no actionable output.
                final_is_refusal = self._is_safety_refusal(response)
                final_is_no_actionable = self._is_no_actionable_content(response)
                final_is_azure_refusal = (
                    final_is_refusal and self._is_azure_filter_refusal(response)
                )
                if final_is_azure_refusal:
                    self.azure_refusal_count += 1
                if (
                    not response.content_blocks
                    or final_is_refusal
                    or final_is_no_actionable
                ):
                    try:
                        response = self._fallback_on_empty_response(
                            on_text_delta=_on_text,
                            on_tool_input_delta=_on_tool_input,
                            streaming=True,
                            primary_raw_event_log=response._raw_event_log,
                            primary_stop_reason=response.stop_reason,
                            primary_usage={
                                "input_tokens": response.input_tokens,
                                "output_tokens": response.output_tokens,
                                "cache_read": response.cache_read_input_tokens,
                                "cache_create": response.cache_creation_input_tokens,
                            },
                            is_azure_refusal=final_is_azure_refusal,
                        )
                    except Exception:
                        logger.warning("All empty-response fallbacks failed.")
            else:
                # Clean primary response — reset the streak so a few empty
                # fallback events spread across many turns don't accumulate
                # indefinitely. Mirrors the non-streaming reset in
                # ``_call_llm_with_retry``.
                self.consecutive_empty_fallbacks = 0

            # Accumulate token usage
            turn_usage += _extract_token_usage(response)
            self.last_turn_usage = turn_usage
            if self.config.has_web_search_capability():
                self._context_budget.update(
                    response.input_tokens
                    + response.cache_creation_input_tokens
                    + response.cache_read_input_tokens
                )

            # Extract tool calls from the response blocks
            for block in response.content_blocks:
                if block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "name": block["name"],
                        "input": block.get("input", {}),
                    })

            # Append assistant message to history (normalised plain-dict blocks)
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content_blocks}
            assistant_msg["_llm_output_tokens"] = response.output_tokens
            assistant_msg["_llm_reasoning_tokens"] = response.reasoning_tokens
            self._append_msg(assistant_msg)
            self._save_response_chain(response)

            if not tool_calls:
                stop_reason = response.stop_reason or "end_turn"
                self.last_stop_reason = stop_reason
                if on_event:
                    on_event(TurnComplete(
                        stop_reason=stop_reason,
                        token_usage=turn_usage,
                    ))
                break

            # Enforce max tool calls per turn if configured.  An optional
            # per-agent override (set by the orchestrator construction path)
            # takes precedence over the shared config so the swarm leader can
            # batch fan-out calls while browsing subagents stay capped.  Tools
            # in ``always_execute_tools_per_turn`` (e.g. post_to_bbs) bypass the
            # cap regardless of position so a posted finding always lands.
            max_tc = _resolve_max_tool_calls(
                self.max_tool_calls_per_turn_override,
                self.config.max_tool_calls_per_turn,
            )
            dropped: list[dict[str, Any]] = []
            if max_tc > 0 and len(tool_calls) > max_tc:
                privileged = frozenset(
                    getattr(self.config, "always_execute_tools_per_turn", None) or ()
                )
                tool_calls, dropped = _split_capped_tool_calls(
                    tool_calls, max_tc, privileged,
                )
                if dropped:
                    logger.info(
                        "Capped tool calls: kept %d (max=%d, +%d privileged), "
                        "dropped %d (max_tool_calls_per_turn)",
                        len(tool_calls), max_tc,
                        max(0, len(tool_calls) - max_tc), len(dropped),
                    )

            # Execute this turn's tool calls (sequentially; see _execute_tools_batch).
            tool_results = self._execute_tools_batch(tool_calls, on_event)

            # Add stub results for dropped tool calls so the message history
            # stays consistent (required by OpenAI Responses API chaining).
            for tc in dropped:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": "(skipped — max tool calls per turn reached)",
                    "is_error": True,
                })
            self._append_msg({"role": "user", "content": tool_results})

            if self._tool_batch_terminates_turn(tool_calls, tool_results):
                stop_reason = "end_turn"
                self.last_stop_reason = stop_reason
                if on_event:
                    on_event(TurnComplete(
                        stop_reason=stop_reason,
                        token_usage=turn_usage,
                    ))
                break

            # Auto-score web_fetch sources from this batch
            self._maybe_score_sources(tool_calls, tool_results)

            # Collapse consecutive empty BBS/DM polls to save context tokens
            self._prune_empty_polls()

            # Auto-inject BBS updates as a simulated read_bbs tool call
            self._maybe_inject_bbs_update()

            # Proactively compact if approaching context limit (web-search mode only)
            if self.config.has_web_search_capability():
                self._maybe_proactive_compact()
        else:
            full_text += f"\n\n(Reached max turns: {effective_max_turns})"
            self.last_stop_reason = "max_turns"
            if on_event:
                on_event(TurnComplete(stop_reason="max_turns", token_usage=turn_usage))

        self.last_turn_usage = turn_usage
        self.last_num_steps = turns
        log_compaction_stats_for_agent(self)
        log_web_search_stats_for_agent(self)
        log_web_fetch_stats_for_agent(self)
        return full_text
