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

"""YAML-based run configuration for arcticswarm-eval.

Loads hierarchical YAML configs, applies CLI dot-notation overrides,
and bridges to the flat :class:`ArcticswarmConfig` used internally.

Usage::

    from arcticswarm.run_config import load_run_config

    cfg = load_run_config(["conf/swarm_duo.yaml"], ["eval.output=results/duo"])
    arcticswarm_config = cfg.to_arcticswarm_config()
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from arcticswarm.config import ArcticswarmConfig


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    model: str = "claude-sonnet-4-5"
    # Optional Azure OpenAI endpoint override for eval runs.
    # Accepts either a full URL (e.g. https://resource.openai.azure.com/)
    # or a resource shortname (e.g. resource).
    openai_endpoint: str = ""
    # Deprecated alias for openai_endpoint; kept for backward compatibility.
    backend: str = ""
    max_tokens: int = 16384
    max_turns: int = 150
    # Per-SUBAGENT turn cap (lever D, anti-flailing). 0 = use max_turns (no-op).
    # Global max_turns is needed for the orchestrator's many wait/read turns, but
    # applied to a single browsing subagent it permits ~O(max_turns) tool calls
    # => O(N^2) context re-send on the flailing tail. Capping subagents (e.g. 150)
    # bounds that without touching the orchestrator.
    subagent_max_turns: int = 0
    reasoning_effort: str | None = None
    streaming: bool = True
    timeout: float | None = 120.0
    # Absolute token count at which to trigger compaction.  When 0,
    # compaction falls back to 90% of the model's context limit.  Useful
    # for 1M-context models where 90% (=900K) is too late to compact.
    compact_tokens: int = 0
    enable_1m_context: bool = False
    # Set True to disable Responses API ``previous_response_id`` chaining.
    # Use for models where chaining causes empty responses (e.g. GPT-5.4).
    disable_responses_chaining: bool = False
    # Optional override for the model used during context-compaction
    # summarisation.  Empty = reuse primary.  Set to a stable model
    # (e.g. ``claude-sonnet-4-6``) when the primary is the failing path.
    compaction_model: str = ""
    max_tool_calls_per_turn: int = 0  # 0 = unlimited; 1 = enforce single tool call
    # Role-aware override for the ORCHESTRATOR only. -1 = inherit
    # ``max_tool_calls_per_turn`` (default); 0 = unlimited for the orchestrator
    # (subagents keep their own cap); >=1 = explicit per-turn cap. Lets the
    # swarm leader batch create_task + wait_for_tasks while browsing subagents
    # stay disciplined at 1.
    orchestrator_max_tool_calls_per_turn: int = -1
    # Tool names that bypass the per-turn cap and always execute (e.g.
    # ["post_to_bbs"] so a posted finding always lands even when the model
    # batches it after another tool). Empty = strict cap (default).
    always_execute_tools_per_turn: list[str] = field(default_factory=list)
    subagent_model: str = ""
    subagent_reasoning_effort: str | None = None
    agent_model_base_url: str = ""
    # --- Self-hosted vLLM (Qwen3.5) knobs --------------------------------
    # Consulted only when the model routes to the vLLM provider (model name
    # contains "qwen" or "tongyi").  ``enable_thinking`` toggles thinking mode
    # (Qwen <think> via chat_template_kwargs.enable_thinking); the sampling
    # knobs follow the model card's thinking-mode recommendation.
    # ``disable_closed_model_fallback`` (auto-enabled for vLLM runs in
    # ``to_arcticswarm_config``) forbids any closed-model (Claude/GPT) call
    # during the agent run.
    enable_thinking: bool = True
    vllm_temperature: float = 0.6
    vllm_top_p: float = 0.95
    vllm_top_k: int = 20
    vllm_presence_penalty: float = 0.0
    vllm_max_model_len: int = 262144
    vllm_served_model_id: str = ""
    vllm_max_output_tokens: int = 0
    disable_closed_model_fallback: bool = False
    # Per-model overrides for reasoning effort.  When the resolved model
    # name is a key here, this value wins over ``reasoning_effort`` /
    # ``subagent_reasoning_effort`` / ``swarm.auditor_reasoning_effort``.
    # Lets a single YAML target both sonnet and opus with model-specific
    # effort levels (system-card recipe: opus BrowseComp wants ``max``).
    # See CONSOLIDATED_CODE_TODOS.md Q2.
    reasoning_effort_by_model: dict[str, str] = field(default_factory=dict)
    subagent_reasoning_effort_by_model: dict[str, str] = field(default_factory=dict)
    auditor_reasoning_effort_by_model: dict[str, str] = field(default_factory=dict)
    # Per-model toggle for extended thinking on adaptive-thinking models
    # (Opus 4.6+).  When True, effort is still routed via
    # ``output_config.effort`` but the ``thinking={"type":"adaptive"}``
    # hint is NOT attached.  Per Opus 4.6 system card §2.21.1:
    # "All reported BrowseComp scores in this section were obtained with
    # thinking disabled" — opus does better on BrowseComp without
    # extended thinking.  See CONSOLIDATED_CODE_TODOS.md Q3.
    disable_extended_thinking_by_model: dict[str, bool] = field(default_factory=dict)


@dataclass
class SwarmConfig:
    enabled: bool = False
    comm: list[str] = field(default_factory=lambda: ["bbs"])
    orchestrator_prompt_mode: str = "default"
    max_teammates: int = 5
    max_subagents: int = 16
    max_subagent_tasks: int = 3
    disable_builder_idle: bool = False
    builder_idle_lifetime: int = -1
    reset_auditor_history: bool = False
    auditor_model: str = ""
    auditor_reasoning_effort: str | None = None
    # When True, do NOT spawn the always-on dedicated auditor subagent
    # (dynamic/BBS mode) that reviews findings via idle-review, AND force
    # the reviewer-diversity gate's dedicated side off (no reasoning-only
    # reviewer is auto-spawned on demand). Net effect: the run has NO
    # dedicated reviewers — only builder subagents (which can act as
    # reviewers) run. Unsupported in duo mode (leader+auditor by
    # construction) — raises at duo entry.
    disable_auditor: bool = False
    realtime: bool = False
    realtime_timeout: int = 300
    enable_force_submit: bool = False
    # When True (default), enforce at least one alternative/contrarian search
    # task per web/BBS run.  ``PrepareReportTool`` auto-spawns a contrarian
    # task if the orchestrator never opened one.  See
    # ``ArcticswarmConfig.enforce_alt_task``.
    enforce_alt_task: bool = True
    # When True, skip the orchestrator's post-answer, code-enforced
    # constraint-verification re-loop (see ``ArcticswarmConfig``). Ablation
    # knob for the "final verification" review gate; independent of the
    # reviewer-diversity / alt-task gates.
    disable_final_verification: bool = False
    # Per-run skill-name remap ``{original_skill_name: variant_skill_name}``.
    # Applied in ``resolve_orchestrator_skill`` / ``resolve_profile_skills`` so
    # an ablation arm can swap in a gate-stripped SKILL.md variant without
    # touching the baseline skill files. Empty = no remap (baseline).
    skill_overrides: dict[str, str] = field(default_factory=dict)
    broadcast_findings: bool = False
    peer_dm_summary: bool = False
    context_reset: bool = False
    system_reminder_interval: int = -1
    profiles: list[str] = field(default_factory=lambda: ["browsing", "reasoning"])
    bbs_channels: list[str] = field(default_factory=list)
    enable_content_cache: bool = True
    # When True (default), ``prepare_report`` blocks inside its mailbox
    # ``wait_for_message`` until a teammate DM arrives or the tool's
    # timeout elapses.  When False, the tool returns immediately with a
    # non-blocking snapshot (+ any pending DMs) — matching the Claude-Code
    # agent model where messages are delivered via a background poll
    # between tool rounds and the leader never sleeps inside a tool call.
    # Duo configs should set this False to avoid wasting wall clock.
    blocking_prepare_report: bool = True
    # Peer tool-call observation. When True, every time agent A executes a
    # tool in ``peer_tool_observation_tools``, the orchestrator drops a
    # synthetic DM into agent B's mailbox describing the call (tool name,
    # file path if any, success/error). The recipient's existing
    # auto-DM-check hook surfaces it before B's next LLM turn, closing the
    # "I have no idea my teammate edited foo.py" gap that drives the duo
    # stale-view race. Default off for
    # backward compatibility — opt in per-run via YAML or CLI override.
    peer_tool_observation: bool = False
    # Tools whose ToolCallEnd events get broadcast when
    # ``peer_tool_observation`` is True. File-mutating tools by default;
    # widen to include ``read_file`` etc. for full visibility
    # at the cost of more DM volume.
    peer_tool_observation_tools: list[str] = field(default_factory=lambda: [
        "edit_file", "str_replace_based_edit_tool", "bash",
    ])

    # Auditor role in duo mode. ``"author"`` (default) preserves today's
    # behavior: the auditor produces an independent patch in its own
    # worktree, and the orchestrator harvests the diff into a
    # ``<auditor_worktree_harvest>`` DM to the leader (full diff +
    # git apply commands) plus stashes it for the pre-submit reminder.
    # ``"reviewer"`` recasts the auditor as a critic + test validator:
    # it pulls the leader's diff, applies it into its worktree, runs
    # tests, probes edges, and DMs concrete findings to the leader. The
    # worktree is informational scratch space, NOT a harvest target.
    # On task completion the orchestrator captures the diff for logs
    # but sends only a thin ``<auditor_complete>`` notice (path + line
    # count, no full diff) and does NOT enqueue a pre-submit reminder.
    # Only meaningful in duo mode.
    auditor_role: str = "author"  # "author" | "reviewer"

    # Reviewer-mode pre-submit stall budget (seconds). Only meaningful
    # when ``auditor_role == "reviewer"``. When the leader tries to
    # call ``send_user_markdown_report`` before its reviewer-auditor
    # has delivered any peer-lane DM, ``SendReportTool`` returns a
    # stall error and re-queues. The leader keeps its turn alive
    # (e.g. with ``read_dm``) until either (a) the auditor sends a
    # review DM, or (b) the wall budget below elapses since the FIRST
    # stall fire, after which the next submit attempt is allowed
    # unconditionally. Defaults to 60s which comfortably covers the
    # 22-50s lateness observed in the n=10 reviewer-mode smoke and
    # leaves headroom for slower tasks; set to ``0`` to disable.
    auditor_review_stall_s: float = 60.0

    # Reviewer-diversity gate (web-research swarms). Before the orchestrator's
    # final report is unlocked, require at least this many distinct VERIFIED
    # ``#consensus`` verdicts from each reviewer *source*:
    #   - ``min_builder_reviewers``   — verdicts from builders (subagents that
    #     did first-hand web investigation: web_search/web_fetch > 0).
    #   - ``min_dedicated_reviewers`` — verdicts from dedicated reviewers (the
    #     reasoning auditor that reviews from the BBS without searching).
    # When a source is short, ``PrepareReportTool`` auto-spawns a targeted
    # reviewer task (profile=reasoning for dedicated, profile=browsing for
    # builder) and blocks the report until it posts, bounded by
    # ``max_reviewer_remediations`` rounds (then degrades to advisory so the
    # run never hangs). 0 disables a requirement. Defaults to 1/1: every
    # web-research case must end with both a builder and a dedicated VERIFIED
    # verdict. No-op for SQL-only runs (no web builder/auditor split), and
    # structurally absent in duo mode (no ``prepare_report`` barrier). Motivated by the
    # reviewer-diversity finding: cases with verdicts from both reviewer sources
    # score substantially higher than cases with no verdict.
    min_dedicated_reviewers: int = 1
    min_builder_reviewers: int = 1
    max_reviewer_remediations: int = 2


@dataclass
class WebConfig:
    enabled: bool = False
    provider: str = "native"
    no_js: bool = False
    no_fetch: bool = False
    fetch_backend: str = "native"
    # Provider try-order for web_search. Default = Brave -> Serper -> Tavily.
    search_provider_order: list[str] = field(
        default_factory=lambda: ["brave", "tavily", "serper"]
    )
    disable_source_scorer: bool = False
    disable_bbs_isolation: bool = False
    # Force BBS isolation for all browsing-profile task executions (ablation).
    # Scoped per task; reviewer/reasoning tasks still read the BBS. Ignored
    # when disable_bbs_isolation is also set.
    force_bbs_isolation: bool = False
    # --- Node-local cache mirror (multi-host runs) -------------------------
    # The fetch cache is mirrored to node-local disk (cache_local_dir)
    # at startup and used from there (WAL on local xfs is safe); new rows are
    # synced back to the shared master every cache_sync_every cases
    # (rollback-journal + flock, no SIGBUS). This is essential whenever an eval
    # runs across multiple hosts with caches enabled — a shared WAL SQLite on a
    # network filesystem (e.g. Lustre) crashes with a bus error otherwise.
    #
    # Default True = auto: the eval CLI engages the mirror only when caching is
    # enabled AND node-local fast storage (cache_local_dir's mount) exists, so a
    # dev box / CPU pod without a fast-disk mount silently uses the master cache
    # directly. Empty cache_local_dir leaves the mirror off unless a fast-disk
    # mount exists; set cache_local_mirror=false to force it off everywhere. See
    # ENVIRONMENT.md for the cache env vars and node-local mirror setup.
    cache_local_mirror: bool = True
    cache_local_dir: str = ""
    cache_sync_every: int = 5
    # When True, the web_search repeat-guard escalates to a hard stop (forces a
    # looping subagent to finalize) once it is unambiguously stuck. Set False to
    # keep only the soft nudge (no forced bail).
    search_repeat_guard_hard_stop: bool = True
    # Runaway near-duplicate (reformulation-loop) force-stop threshold. Once one
    # search intent has been reworded this many times, the repeat-guard forces a
    # stop (only when search_repeat_guard_hard_stop is on). 40 = WebSearchTool
    # default; lower (e.g. 12) to bite a churning small model harder.
    search_neardup_hard_stop: int = 40
    # Collapse duplicate tool-call RESULTS in the outbound LLM history (the
    # bulky body of all-but-last-N identical web_search/web_fetch/pdf_read
    # results is stubbed). Saves context on models that re-issue the same call.
    collapse_duplicate_tool_history: bool = False
    dup_history_keep_last: int = 1
    use_fetch_compactor: bool = False
    use_pdf_compactor: bool = False
    # Hard cap (in TOKENS, ~4 chars/token) on a single web_fetch / pdf_read
    # tool result and on the ContentCompactor's selected output. 0 disables.
    # See ArcticswarmConfig.max_tool_output_tokens. Bench configs set 5000.
    max_tool_output_tokens: int = 24_000
    disable_self_reflection: bool = False
    browsing_max_search_plans: int = 2
    browsing_max_reflection_loops: int = 2
    browsing_reflection_model: str = ""
    # --- BrowseComp-Plus corpus retrieval (provider: corpus) ---------------
    # Backend selector: "stub" (default), "cortex", or "local". See
    # arcticswarm/tools/corpus_retriever.py and the README.
    corpus_backend: str = "stub"
    corpus_account: str = ""
    corpus_db: str = ""
    corpus_schema: str = ""
    corpus_chunked_service: str = ""
    corpus_service: str = ""
    corpus_pat_connection: str = "default"
    corpus_local_path: str = ""
    # --- Global cross-run fetch cache --------------------------------------
    # Override the shared web_fetch/pdf_read SQLite cache path for this run
    # (default resolved from settings ``fetch_cache_path`` / env
    # ARCTICSWARM_FETCH_CACHE / the built-in default). Set to "off" / "none"
    # to disable the global cache for this run.
    fetch_cache_path: str = ""
    # --- Cortex web-search provider (provider: cortex / cortex-grounding) ----
    # Snowflake account used to reach the Cortex ``agent:run`` web-search
    # passthrough.  Empty => fall back to the settings/env-resolved
    # ``cortex_account`` (CORTEX_ACCOUNT) or the sf_client session token
    # (preferred for in-cluster runs).  Set here only to override per-run.
    cortex_account: str = ""


@dataclass
class ProfileConfig:
    """Declarative tool/skill definition for a single swarm subagent profile."""
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    description: str = ""
    bbs_channels: list[str] = field(default_factory=list)


@dataclass
class ToolsConfig:
    """Declarative tool lists for agents, orchestrator, and subagent profiles.

    Defaults target the web-research path; presets override these in their
    YAML files.

    ``reasoning`` is excluded from the orchestrator defaults because it
    gives the orchestrator access to extended-thinking on the agent's
    model, which may be undesirable for controlled experiments.  Web
    swarm presets that need it should add it explicitly.
    """
    agent: list[str] = field(default_factory=lambda: [
        "load_skill", "reasoning", "web_search", "web_fetch", "pdf_read",
        "calculator", "read_file", "python_execute",
    ])
    orchestrator: list[str] = field(default_factory=lambda: [
        "load_skill",
    ])
    agent_skills: list[str] = field(default_factory=lambda: [
        "web-research", "tool-usage-policy-browsing",
    ])
    orchestrator_skills: list[str] = field(default_factory=list)
    # Tools available to subagents during idle review cycles (BBS/DM review
    # of teammates' findings, triggered when the subagent has no assigned
    # tasks).  When empty, the hardcoded research default in
    # ``Teammate._idle_check`` is used.  When non-empty, this list replaces
    # it verbatim.  Tools not in the agent's main toolset (e.g. ``reasoning``
    # for a browsing agent) are lazy-instantiated via ``ToolFactory``.
    idle_reviewer: list[str] = field(default_factory=list)
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)


@dataclass
class SkillsConfig:
    legacy_format: bool = False
    per_skill_tools: bool = False


@dataclass
class AzureConfig:
    enabled: bool = False
    use_chat_completions: bool = False


@dataclass
class ODLConfig:
    hybrid: str = "docling-fast"
    hybrid_url: str = ""
    hybrid_timeout: int = 60000
    hybrid_fallback_timeout: int = 300
    force_ocr: bool = False


@dataclass
class EvalConfig:
    datasets: list[str] = field(default_factory=list)
    csv_path: str = ""
    output: str = ""
    parallel: int = 3
    timeout: float = 300
    # Default wait timeout (seconds) for the orchestrator's ``prepare_report``
    # tool. Bumping this lets late subagent tasks finish before the
    # orchestrator finalizes. The eval soft-deadline (``timeout`` above)
    # still bounds total runtime.
    prepare_timeout: int = 300
    repeat: int = 1
    max_retries: int = 0
    judge_model: str = "claude-4-sonnet"
    # Base URL for a self-hosted (OpenAI-compatible, e.g. vLLM) judge model.
    # When set, the judge talks to this endpoint via Chat Completions instead
    # of the Cortex proxy. Used to run the judge on a
    # self-hosted Qwen (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507 at http://host:port/v1).
    judge_model_base_url: str = ""
    # Optional path to a user-supplied custom judge-rubric template
    # (yaml: eval.custom_judge_prompt). When set, LLMJudge grades every case
    # with this rubric, overriding the built-in per-dataset judges. See
    # docs/custom_evaluation.md.
    custom_judge_prompt: str = ""
    qa_llm: bool = False
    vip_only: bool = True
    limit: int = 0
    # Skip the first ``offset`` matching cases before applying ``limit``.
    # ``eval.limit=100 eval.offset=100`` selects cases 100..199 — useful for
    # extending a previous ``limit=100`` run without re-running the first 100.
    # Case ordering is deterministic (no shuffling), so the slice is stable
    # across runs and reproducible across modes.
    offset: int = 0
    conv_id: str = ""
    eval_mode: str | None = None
    subset_filter: str | None = None
    resume: bool = False
    rerun_errors: bool = False
    rerun_timeouts: bool = False
    rerun_wrong: bool = False
    checkpoint_interval: int = 5
    rebuild_from_trajectories: bool = False
    verbose: bool = False
    # Live, per-case activity feed (CLI-style) during the run.
    # Each swarm/agent event prints one console line prefixed with the case's
    # conv_id (errors in red, normal progress plain) so parallel cases can be
    # followed live.  Set ``eval.stream=false`` to disable and keep only the
    # progress bar + per-case summary lines.
    stream: bool = True
    # Gated retry wrapper.
    gated_retry: bool = False
    retry_threshold: float = -0.38
    max_retry_fraction: float = 0.5



@dataclass
class RunConfig:
    """Top-level hierarchical config loaded from YAML."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    web: WebConfig = field(default_factory=WebConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)
    odl: ODLConfig = field(default_factory=ODLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    vision: bool = False
    date_override: str = ""
    prompt_style: str = ""
    # best-of-N alternatives plan flags.
    enable_empty_answer_recovery: bool = False
    # answer-retention: prepare_report appends a BBS candidate-findings
    # digest so a found-but-compacted-away answer is re-surfaced before the final
    # answer is written. Default False; enabled only in the qwen browsecomp YAML.
    surface_bbs_candidates: bool = False
    # selective-delete compaction (user request): prune certainly-wrong
    # tool-result paths from the proactive-compaction summarizer input. Default
    # False; enabled only in the qwen browsecomp YAML.
    compaction_prune_junk: bool = False
    enable_compact_reflection: bool = False
    enable_candidate_emergence_sweep: bool = False
    candidate_emergence_min_chars: int = 60
    candidate_emergence_max_turns: int = 8
    # also dispatch the candidate-emergence rival sweep via spawn_or_assign
    # (not just add_task) so it actually runs. Default False. See ArcticswarmConfig.
    alt_task_force_dispatch: bool = False
    # anti-anchoring browsing-prompt block. Default False; qwen-gated.
    reframe_prompt: bool = False
    # anti-give-up + canonical-answer report behavior. When True: (1) the
    # send_user_markdown_report instructions tell the orchestrator to emit the
    # fullest canonical/official form of the answer and honor the question's
    # explicitly-requested format (fixes under-specification judge artifacts like
    # "Richard Todd" vs "Richard Andrew Palethorpe-Todd"); (2) a report whose
    # FINAL ANSWER is a refusal/give-up is bounced back for a committed retry
    # (bounded by SendReportTool.max_refusal_bounces); (3) the timeout
    # force-report + post-hoc recovery paths never finalize a give-up. Default
    # False; enabled only in the qwen browsecomp YAML (BrowseComp answers always
    # exist, so committing the best candidate can only help the binary score).
    reject_refusal_reports: bool = False

    def to_arcticswarm_config(self) -> ArcticswarmConfig:
        """Convert to the flat ArcticswarmConfig used internally.

        Calls ``ArcticswarmConfig.resolve()`` first to pick up API keys
        and Snowflake params from the settings file / env vars, then
        overlays every field from this hierarchical config.
        """
        config = ArcticswarmConfig.resolve()

        # LLM
        config.model = self.llm.model
        config.max_tokens = self.llm.max_tokens
        config.max_turns = self.llm.max_turns
        config.subagent_max_turns = self.llm.subagent_max_turns
        config.reasoning_effort = (
            None if self.llm.reasoning_effort == "none"
            else self.llm.reasoning_effort
        )
        # Per-model main-agent effort override (Q2): keyed by the resolved
        # model name.  Wins over the global ``reasoning_effort`` so a single
        # YAML can target sonnet+opus with model-specific levels.
        _per_model_effort = self.llm.reasoning_effort_by_model.get(self.llm.model)
        if _per_model_effort is not None:
            config.reasoning_effort = (
                None if _per_model_effort == "none" else _per_model_effort
            )
        config.use_streaming = self.llm.streaming
        config.llm_timeout = self.llm.timeout
        config.context_compact_tokens = self.llm.compact_tokens
        config.enable_1m_context_model = self.llm.enable_1m_context
        config.disable_responses_chaining = self.llm.disable_responses_chaining
        config.compaction_model = self.llm.compaction_model
        config.max_tool_calls_per_turn = self.llm.max_tool_calls_per_turn
        config.orchestrator_max_tool_calls_per_turn = self.llm.orchestrator_max_tool_calls_per_turn
        config.always_execute_tools_per_turn = list(self.llm.always_execute_tools_per_turn)
        config.subagent_model = self.llm.subagent_model
        config.subagent_reasoning_effort = (
            None if self.llm.subagent_reasoning_effort == "none"
            else self.llm.subagent_reasoning_effort
        )
        config.agent_model_base_url = self.llm.agent_model_base_url
        # Self-hosted vLLM (Qwen) knobs.
        config.enable_thinking = self.llm.enable_thinking
        config.vllm_temperature = self.llm.vllm_temperature
        config.vllm_top_p = self.llm.vllm_top_p
        config.vllm_top_k = self.llm.vllm_top_k
        config.vllm_presence_penalty = self.llm.vllm_presence_penalty
        config.vllm_max_model_len = self.llm.vllm_max_model_len
        config.vllm_served_model_id = self.llm.vllm_served_model_id
        config.vllm_max_output_tokens = self.llm.vllm_max_output_tokens
        # For vLLM (self-hosted Qwen) runs, forbid closed-model calls by
        # default — keeps the whole agent run on Qwen (the eval judge is a
        # separate, post-hoc client and is unaffected). An explicit YAML
        # ``disable_closed_model_fallback`` still wins.
        from arcticswarm.llm_client import detect_provider
        config.disable_closed_model_fallback = (
            self.llm.disable_closed_model_fallback
            or detect_provider(self.llm.model) == "vllm"
        )
        # Per-model effort overrides (CONSOLIDATED_CODE_TODOS.md Q2).
        config.reasoning_effort_by_model = dict(self.llm.reasoning_effort_by_model)
        config.subagent_reasoning_effort_by_model = dict(self.llm.subagent_reasoning_effort_by_model)
        config.auditor_reasoning_effort_by_model = dict(self.llm.auditor_reasoning_effort_by_model)
        config.disable_extended_thinking_by_model = dict(self.llm.disable_extended_thinking_by_model)
        # Endpoint override — routes Azure traffic to a named deployment
        # (e.g. ``my-azure-deployment``) instead of the default
        # ``AZURE_OPENAI_ENDPOINT`` env var.
        endpoint_override = (self.llm.openai_endpoint or self.llm.backend).strip()
        if endpoint_override:
            if "://" not in endpoint_override:
                endpoint_override = f"https://{endpoint_override}.openai.azure.com/"
            config.azure_openai_endpoint = endpoint_override
            config.use_azure_openai = True

        # Swarm
        config.swarm_enabled = self.swarm.enabled
        config.swarm_comm = list(self.swarm.comm)
        config.orchestrator_prompt_mode = self.swarm.orchestrator_prompt_mode
        config.max_teammates = self.swarm.max_teammates
        config.max_subagents = self.swarm.max_subagents
        config.max_subagent_tasks = self.swarm.max_subagent_tasks
        config.disable_builder_idle = self.swarm.disable_builder_idle
        config.builder_idle_lifetime = self.swarm.builder_idle_lifetime
        config.reset_auditor_history = self.swarm.reset_auditor_history
        config.auditor_model = self.swarm.auditor_model
        config.disable_auditor = self.swarm.disable_auditor
        config.auditor_reasoning_effort = (
            None if self.swarm.auditor_reasoning_effort == "none"
            else self.swarm.auditor_reasoning_effort
        )
        config.orchestrator_realtime = self.swarm.realtime
        config.orchestrator_realtime_timeout = self.swarm.realtime_timeout
        config.enable_force_submit = self.swarm.enable_force_submit
        config.enforce_alt_task = self.swarm.enforce_alt_task
        config.disable_final_verification = self.swarm.disable_final_verification
        config.skill_overrides = dict(self.swarm.skill_overrides)
        config.submit_findings_broadcast = self.swarm.broadcast_findings
        config.peer_dm_summary = self.swarm.peer_dm_summary
        config.subagent_context_reset = self.swarm.context_reset
        config.system_reminder_interval = self.swarm.system_reminder_interval
        config.swarm_profiles = list(self.swarm.profiles)
        config.swarm_bbs_channels = list(self.swarm.bbs_channels)
        config.enable_content_cache = self.swarm.enable_content_cache
        config.blocking_prepare_report = self.swarm.blocking_prepare_report
        config.peer_tool_observation = self.swarm.peer_tool_observation
        config.peer_tool_observation_tools = list(self.swarm.peer_tool_observation_tools)
        config.auditor_role = self.swarm.auditor_role
        config.auditor_review_stall_s = self.swarm.auditor_review_stall_s
        config.min_dedicated_reviewers = self.swarm.min_dedicated_reviewers
        config.min_builder_reviewers = self.swarm.min_builder_reviewers
        config.max_reviewer_remediations = self.swarm.max_reviewer_remediations

        # Web
        config.web_search_enabled = self.web.enabled
        config.cache_local_mirror = self.web.cache_local_mirror
        config.cache_local_dir = self.web.cache_local_dir
        config.cache_sync_every = self.web.cache_sync_every
        config.web_search_provider = self.web.provider
        config.no_js = self.web.no_js
        config.no_web_fetch = self.web.no_fetch
        config.web_fetch_backend = self.web.fetch_backend
        config.corpus_backend = self.web.corpus_backend
        config.corpus_account = self.web.corpus_account
        config.corpus_db = self.web.corpus_db
        config.corpus_schema = self.web.corpus_schema
        config.corpus_chunked_service = self.web.corpus_chunked_service
        config.corpus_service = self.web.corpus_service
        config.corpus_pat_connection = self.web.corpus_pat_connection
        config.corpus_local_path = self.web.corpus_local_path
        # Coerce to a list of provider tokens.  A bare string override
        # (e.g. ``web.search_provider_order=serper`` or ``=serper,tavily``)
        # arrives as a str; ``list("serper")`` would shred it into single
        # characters, so split on commas instead of iterating the string.
        _spo = self.web.search_provider_order
        if isinstance(_spo, str):
            config.search_provider_order = [
                tok.strip() for tok in _spo.split(",") if tok.strip()
            ]
        else:
            config.search_provider_order = list(_spo)
        config.disable_source_scorer = self.web.disable_source_scorer
        config.disable_bbs_isolation = self.web.disable_bbs_isolation
        config.force_bbs_isolation = self.web.force_bbs_isolation
        if self.web.disable_bbs_isolation and self.web.force_bbs_isolation:
            # Contradictory ablation flags: disable => never isolate,
            # force => always isolate browsing. Fail loud rather than
            # silently letting disable win and producing a "force" run that
            # isolated nothing.
            raise ValueError(
                "web.disable_bbs_isolation and web.force_bbs_isolation are "
                "mutually exclusive — set at most one."
            )
        config.search_repeat_guard_hard_stop = self.web.search_repeat_guard_hard_stop
        config.search_neardup_hard_stop = self.web.search_neardup_hard_stop
        config.collapse_duplicate_tool_history = self.web.collapse_duplicate_tool_history
        config.dup_history_keep_last = self.web.dup_history_keep_last
        # Cortex web-search provider account: a non-empty web.cortex_account
        # overrides the settings/env-resolved value; otherwise keep what
        # ArcticswarmConfig.resolve() already loaded (settings / CORTEX_ACCOUNT).
        if self.web.cortex_account:
            config.cortex_account = self.web.cortex_account
        # Global fetch-cache override: a non-empty web.fetch_cache_path wins
        # over the settings/env/default resolved in ArcticswarmConfig.resolve();
        # "off"/"none"/"disabled"/"false" disables the global cache this run.
        _fcp = (self.web.fetch_cache_path or "").strip()
        if _fcp:
            config.fetch_cache_path = (
                "" if _fcp.lower() in ("off", "none", "disabled", "false") else _fcp
            )
        config.use_fetch_compactor = self.web.use_fetch_compactor
        config.use_pdf_compactor = self.web.use_pdf_compactor
        config.max_tool_output_tokens = self.web.max_tool_output_tokens
        config.disable_self_reflection = self.web.disable_self_reflection
        config.browsing_max_search_plans = self.web.browsing_max_search_plans
        config.browsing_max_reflection_loops = self.web.browsing_max_reflection_loops
        config.browsing_reflection_model = self.web.browsing_reflection_model

        # Skills
        config.skill_legacy_format = self.skills.legacy_format
        config.per_skill_tools = self.skills.per_skill_tools

        # Azure
        config.use_azure_openai = self.azure.enabled
        config.use_chat_completions = self.azure.use_chat_completions

        # ODL
        config.odl_hybrid = self.odl.hybrid
        config.odl_hybrid_url = self.odl.hybrid_url
        config.odl_hybrid_timeout = self.odl.hybrid_timeout
        config.odl_hybrid_fallback_timeout = self.odl.hybrid_fallback_timeout
        config.odl_force_ocr = self.odl.force_ocr

        # Declarative tools
        config.agent_tools = list(self.tools.agent)
        config.orchestrator_tools = list(self.tools.orchestrator)
        config.agent_skills = list(self.tools.agent_skills)
        config.orchestrator_skills = list(self.tools.orchestrator_skills)
        config.idle_reviewer_tools = list(self.tools.idle_reviewer)
        config.tool_profiles = {
            name: asdict(prof)
            for name, prof in self.tools.profiles.items()
        }

        # Top-level
        config.enable_vision = self.vision
        config.date_override = self.date_override
        config.prompt_style = self.prompt_style

        # plan flags
        config.enable_empty_answer_recovery = self.enable_empty_answer_recovery
        config.surface_bbs_candidates = self.surface_bbs_candidates
        config.compaction_prune_junk = self.compaction_prune_junk
        config.enable_compact_reflection = self.enable_compact_reflection
        config.enable_candidate_emergence_sweep = self.enable_candidate_emergence_sweep
        config.candidate_emergence_min_chars = self.candidate_emergence_min_chars
        config.candidate_emergence_max_turns = self.candidate_emergence_max_turns
        config.alt_task_force_dispatch = self.alt_task_force_dispatch
        config.reframe_prompt = self.reframe_prompt
        config.reject_refusal_reports = self.reject_refusal_reports

        # Eval output dir — used for diagnostic logs (empty fallback cases, etc.)
        config.output_dir = self.eval.output

        # prepare_report wait-loop default timeout (yaml: eval.prepare_timeout)
        config.prepare_report_timeout = self.eval.prepare_timeout

        return config


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_SECTION_CLASSES: dict[str, type] = {
    "llm": LLMConfig,
    "swarm": SwarmConfig,
    "web": WebConfig,
    "tools": ToolsConfig,
    "skills": SkillsConfig,
    "azure": AzureConfig,
    "odl": ODLConfig,
    "eval": EvalConfig,
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into *base* (overlay wins)."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce_value(raw: str) -> Any:
    """Best-effort coerce a CLI override string to a Python value.

    Handles booleans, None/null, ints, floats, and lists.
    Falls back to plain string if nothing matches.
    """
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none", "~"):
        return None
    if raw.startswith("[") and raw.endswith("]"):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            # Fall back to a bare comma-separated list of strings, so an
            # unquoted CLI override like ``[serper,tavily,brave]`` works
            # without shell-quoting each element.  Empty list => [].
            inner = raw[1:-1].strip()
            if not inner:
                return []
            return [tok.strip().strip("\"'") for tok in inner.split(",") if tok.strip()]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_overrides(data: dict[str, Any], overrides: list[str]) -> None:
    """Apply dot-notation overrides (``key.subkey=value``) in place."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Override must be key=value or key.subkey=value, got: {override!r}"
            )
        key_path, _, raw_value = override.partition("=")
        parts = key_path.split(".")
        target = data
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = _coerce_value(raw_value)


def _dict_to_run_config(data: dict[str, Any]) -> RunConfig:
    """Convert a raw dict (from YAML + overrides) to a typed RunConfig."""
    kwargs: dict[str, Any] = {}

    for section_name, cls in _SECTION_CLASSES.items():
        section_data = data.pop(section_name, {})
        if isinstance(section_data, dict):
            if cls is ToolsConfig:
                # Nested profiles need to be converted to ProfileConfig
                profiles_raw = section_data.pop("profiles", {})
                tools_cfg = ToolsConfig(**section_data)
                if profiles_raw and isinstance(profiles_raw, dict):
                    tools_cfg.profiles = {
                        name: ProfileConfig(**pdata) if isinstance(pdata, dict) else ProfileConfig()
                        for name, pdata in profiles_raw.items()
                    }
                kwargs[section_name] = tools_cfg
            else:
                kwargs[section_name] = cls(**section_data)
        else:
            kwargs[section_name] = cls()

    # Remaining top-level keys go directly to RunConfig
    for key, value in data.items():
        kwargs[key] = value

    return RunConfig(**kwargs)


def _find_config_file(path_str: str) -> Path:
    """Resolve a config file path, searching common locations."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p

    # Search relative to the arcticswarm package conf/ directory
    pkg_conf = Path(__file__).resolve().parent.parent / "conf"
    candidate = pkg_conf / path_str
    if candidate.exists():
        return candidate

    # Search with .yaml extension appended
    if not path_str.endswith((".yaml", ".yml")):
        for ext in (".yaml", ".yml"):
            candidate = pkg_conf / (path_str + ext)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Config file not found: {path_str!r} "
        f"(searched cwd and {pkg_conf})"
    )


def load_run_config(
    config_paths: list[str],
    overrides: list[str] | None = None,
) -> RunConfig:
    """Load one or more YAML config files, merge them, apply overrides.

    Parameters
    ----------
    config_paths:
        YAML files to load, merged left-to-right (later files override
        earlier ones).
    overrides:
        CLI dot-notation overrides, e.g. ``["eval.output=results/run1",
        "eval.parallel=8"]``.

    Returns
    -------
    RunConfig
        Fully-resolved hierarchical config.
    """
    merged: dict[str, Any] = {}
    for path_str in config_paths:
        filepath = _find_config_file(path_str)
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping in {filepath}, got {type(data).__name__}")
        merged = _deep_merge(merged, data)

    if overrides:
        _apply_overrides(merged, overrides)

    return _dict_to_run_config(merged)
