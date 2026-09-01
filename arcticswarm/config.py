"""Configuration for Arcticswarm.

Resolution order (highest priority first):
  1. CLI flags (--no-stream)
  2. Settings file — see :func:`settings_json_path` (default
     ``./config_files.json``, or ``ARCTICSWARM_SETTINGS_PATH``)
  3. Built-in defaults
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[reportMissingImports]
    except ModuleNotFoundError:  # pragma: no cover - container fallback
        from pip._vendor import tomli as tomllib


# ---------------------------------------------------------------------------
# Model registry with pricing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """An available model with its pricing (per million tokens)."""

    id: str            # e.g. "claude-opus-4-6"
    display_name: str  # e.g. "Opus 4.6"
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float


AVAILABLE_MODELS: list[ModelInfo] = [
    # Anthropic (Claude)
    ModelInfo("claude-opus-4-6",   "Opus 4.6",   5.0,  25.0, 0.50,  6.25),
    ModelInfo("claude-sonnet-4-6", "Sonnet 4.6", 3.0,  15.0, 0.30,  3.75),
    ModelInfo("claude-opus-4-5",   "Opus 4.5",   5.0,  25.0, 0.50,  6.25),
    ModelInfo("claude-sonnet-4-5", "Sonnet 4.5", 3.0,  15.0, 0.30,  3.75),
    ModelInfo("claude-4-sonnet",   "Sonnet 4",   3.0,  15.0, 0.30,  3.75),
    ModelInfo("claude-4-opus",     "Opus 4",    15.0,  75.0, 1.50, 18.75),
    ModelInfo("claude-haiku-4-5",  "Haiku 4.5",  1.0,   5.0, 0.10,  1.25),
    # OpenAI (GPT) — pricing approximate; cache fields unused
    ModelInfo("openai-gpt-5-chat", "GPT-5 Chat", 2.0,  8.0, 0.0, 0.0),
    ModelInfo("openai-gpt-5",      "GPT-5",      2.0,  8.0, 0.0, 0.0),
    ModelInfo("gpt-5",             "GPT-5",      2.0,  8.0, 0.0, 0.0),
    ModelInfo("gpt-5.2",           "GPT-5.2",    2.0,  8.0, 0.0, 0.0),
    ModelInfo("gpt-5.4",           "GPT-5.4",    2.0,  8.0, 0.0, 0.0),
    ModelInfo("gpt-5.4-pro",       "GPT-5.4 Pro", 2.0,  8.0, 0.0, 0.0),
]


def get_model_info(model_id: str) -> ModelInfo | None:
    """Look up pricing for a model ID. Returns None if unknown."""
    for m in AVAILABLE_MODELS:
        if m.id == model_id:
            return m
    return None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-sonnet-4-5"
_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MAX_TOKENS = 16384
_DEFAULT_MAX_TURNS = 50
_DEFAULT_SF_CONNECTION = "default"
_CONNECTIONS_TOML = Path.home() / ".snowflake" / "connections.toml"
_SETTINGS_JSON = Path("config_files.json")

# Default location of the global, cross-run web_fetch/pdf_read cache (a single
# SQLite file). A plain local SQLite at a configurable path. Override via
# settings ``fetch_cache_path`` / env ARCTICSWARM_FETCH_CACHE / ``web.fetch_cache_path``;
# set to a path to enable. Opens lazily and degrades to disabled if the path
# isn't writable, so enabling it is always safe. Empty (the default) keeps every
# fetch live; see ENVIRONMENT.md for the optional cross-run cache setup.
_DEFAULT_FETCH_CACHE_PATH = ""


# ---------------------------------------------------------------------------
# Settings file helper
# ---------------------------------------------------------------------------


def settings_json_path() -> Path:
    """Path to ``config_files.json``.

    If the environment variable ``ARCTICSWARM_SETTINGS_PATH`` is set, that file
    is used (expanded ``~``). Otherwise the default is ``./config_files.json``
    (relative to the current working directory).
    """
    env = os.environ.get("ARCTICSWARM_SETTINGS_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _SETTINGS_JSON


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load JSON settings for API keys, model, and optional feature keys.

    Returns an empty dict if the file does not exist.
    """
    path = path or settings_json_path()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Snowflake connection TOML helpers
# ---------------------------------------------------------------------------


def load_snowflake_connections(path: Path | None = None) -> dict[str, Any]:
    """Load *all* connections from ``~/.snowflake/connections.toml``."""
    path = path or _CONNECTIONS_TOML
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def get_snowflake_connection_params(
    connection_name: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Return the params dict for a single named connection."""
    connection_name = connection_name or _DEFAULT_SF_CONNECTION
    connections = load_snowflake_connections(path)
    if connection_name not in connections:
        raise KeyError(
            f"Connection '{connection_name}' not found in {path or _CONNECTIONS_TOML}. "
            f"Available: {list(connections.keys())}"
        )
    return dict(connections[connection_name])


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class ArcticswarmConfig:
    """Immutable bag of resolved configuration."""

    # LLM provider
    api_key: str = ""
    base_url: str = _DEFAULT_BASE_URL
    openai_base_url: str = ""  # explicit OpenAI endpoint; defaults to public OpenAI if empty
    openai_api_key: str = ""  # separate key for OpenAI/GPT (else falls back to OPENAI_API_KEY)
    model: str = _DEFAULT_MODEL

    # Base URL for a self-hosted agent model (e.g. vLLM). When set, the
    # agent's orchestration LLM calls go here while tools and judge remain
    # on the default API credentials.
    agent_model_base_url: str = ""

    # --- Self-hosted vLLM (Qwen3.5) knobs --------------------------------
    # Consulted only when the model routes to the "vllm" provider (model name
    # contains "qwen" or "tongyi"). See ``llm_client``'s ``VLLMChatLLMClient``.
    enable_thinking: bool = True
    vllm_temperature: float = 0.6
    vllm_top_p: float = 0.95
    vllm_top_k: int = 20
    vllm_presence_penalty: float = 0.0
    vllm_max_model_len: int = 262144
    vllm_served_model_id: str = ""
    vllm_max_output_tokens: int = 0
    # When True, NO closed-model (Claude/GPT) API call is made during the
    # agent run: the cross-model empty-response fallback and the source
    # scorer / content compactor GPT fallback are disabled (same-model
    # retries stay alive). Auto-enabled for vLLM runs by RunConfig.
    disable_closed_model_fallback: bool = False

    # Azure OpenAI — used instead of the Cortex proxy
    # when use_azure_openai=True
    use_azure_openai: bool = False
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2025-04-01-preview"
    max_tokens: int = _DEFAULT_MAX_TOKENS
    max_turns: int = _DEFAULT_MAX_TURNS
    # Per-subagent turn cap (lever D). 0 = use max_turns (no-op).
    subagent_max_turns: int = 0

    # Absolute compaction trigger in tokens.  When 0, compaction falls back
    # to 90% of the model's context limit (the reactive compaction in
    # _call_llm_with_retry() is the safety net).  Useful for 1M context
    # models where 90% (= 900K) is too late to compact.
    context_compact_tokens: int = 0
    subagent_context_reset: bool = False     # reset context between swarm tasks

    # Snowflake
    sf_connection_name: str = _DEFAULT_SF_CONNECTION
    sf_params: dict[str, Any] = field(default_factory=dict)

    # Brave Search API key (resolved from settings key `brave_api_key` or env var `BRAVE_API_KEY`)
    brave_api_key: str = ""

    # Jina Reader API key (resolved from settings key `jina_api_key` or env var `JINA_API_KEY`)
    # Used as primary fetch method by WebFetchTool and PdfReadTool.
    jina_api_key: str = ""

    # Google Serper API key (resolved from settings key `serper_api_key` or env var `SERPER_API_KEY`)
    # Used for web scraping fallback (MCP) and search result gating.
    serper_api_key: str = ""

    # Tavily Search API key (resolved from settings key `tavily_api_key` or env var `TAVILY_API_KEY`)
    # Used as fallback search provider between Brave and Serper.
    tavily_api_key: str = ""

    # Order in which web_search tries providers.  Default is Brave primary,
    # then Serper (Google) as the first fallback, then Tavily as a last-resort
    # safety net.  Serper outranks Tavily because Google results are terser
    # (less context bloat) and align with SOTA deep-research agents; Tavily's
    # verbose advanced+answer payloads are kept only as a final backstop.
    # Reorder freely, e.g. ["serper","tavily","brave"]; unkeyed providers are
    # skipped.
    search_provider_order: list[str] = field(
        default_factory=lambda: ["brave", "tavily", "serper"]
    )

    # When True, use Jina "direct" engine (no JS rendering) instead of
    # the default "browser" engine for web_fetch.
    no_js: bool = False

    # When True, the web_fetch tool is disabled.  web_search and pdf_read
    # remain available.
    no_web_fetch: bool = False

    # web_fetch backend: "native" uses Jina→Serper→requests chain.
    web_fetch_backend: str = "native"

    # When True, web search tools and skills are enabled.  Disabled by
    # default — use ``--web-search`` to enable.  When False, the agent
    # has no web_search tool and web-related skills are hidden.
    web_search_enabled: bool = False

    # Web search provider: "native" (Brave/Tavily/Serper) or a managed
    # inference endpoint.
    web_search_provider: str = "native"

    # Account for Cortex (agent:run) (e.g. "myaccount").
    cortex_account: str = ""

    # --- BrowseComp-Plus corpus retrieval (web.provider: corpus) -----------
    # The corpus benchmark searches a fixed document corpus via a pluggable
    # backend (see arcticswarm/tools/corpus_retriever.py):
    #   "stub"    (default) — no-op placeholder; harness runs, no real retrieval
    #   "managed"           — Snowflake Cortex Search (coords below)
    #   "local"             — local JSONL corpus + reference scorer (template)
    corpus_backend: str = "stub"
    # Cortex Search service coordinates (only used when corpus_backend="cortex";
    # falls back to cortex_account when corpus_account is empty). These are NOT
    # hardcoded — set them in the run config / env. The values the original
    # BrowseComp-Plus runs used are documented (commented) in
    # conf/bench/browsecomp_plus*.yaml and the README.
    corpus_account: str = ""
    corpus_db: str = ""
    corpus_schema: str = ""
    corpus_chunked_service: str = ""   # chunked snippets -> web_search
    corpus_service: str = ""           # full documents  -> web_fetch
    corpus_pat_connection: str = "default"
    # Local corpus JSONL path (only used when corpus_backend="local").
    corpus_local_path: str = ""

    # When True, read_file returns base64 image content blocks for image
    # files (png, jpg, gif, webp) so the LLM can see them.  Off by default
    # for backward compatibility — use ``--enable-vision`` to enable.
    enable_vision: bool = False

    # Date override (YYYY-MM-DD) — used during eval to simulate "today" for
    # time-relative queries.
    date_override: str = ""

    # Dataset name (e.g., "GAIA_V1", "BROWSECOMP_V1") — used during eval to
    # customize system prompt for benchmark-specific requirements.
    dataset: str = ""

    # Swarm mode — when True, questions route through the SwarmOrchestrator
    # instead of a single Agent.
    swarm_enabled: bool = False
    max_teammates: int = 5
    # Hard cap on subagents in dynamic mode.
    max_subagents: int = 16
    # Tasks before a dynamic-mode worker is considered context-full (-1 = no limit).
    max_subagent_tasks: int = 3
    # Disable idle-review for builder subagents (agents that completed a task).
    # The dedicated auditor is unaffected.
    disable_builder_idle: bool = False
    # Max total idle-review turns per builder subagent over its lifetime
    # (-1 = use default consecutive cap).  The dedicated auditor is unaffected.
    builder_idle_lifetime: int = -1
    # Clear auditor conversation history after each idle-review cycle so each
    # review starts with a fresh context (no accumulated history).
    reset_auditor_history: bool = False
    # When non-empty, the dedicated auditor subagent uses this model instead
    # of subagent_model / main model.  Other subagents are unaffected.
    auditor_model: str = ""
    # Reasoning effort for the auditor.  None = inherit from subagent.
    auditor_reasoning_effort: str | None = None
    # When True, suppress the always-on dedicated auditor subagent (spawned
    # on first task creation in dynamic/BBS mode) AND the reviewer-diversity
    # gate's dedicated-reviewer auto-spawn, so the run has NO dedicated
    # reviewers — only builder subagents run.  Guarded (raises) in duo mode.
    disable_auditor: bool = False
    # Disable BBS isolation (isolated=true on create_task) for ablation experiments.
    disable_bbs_isolation: bool = False
    # Force BBS isolation for browsing-profile EXPLORATION task executions
    # (ablation). Scoped per task execution: reviewer tasks (reviewer_kind) and
    # a subagent later running a reviewer/reasoning task still read the BBS.
    # Mutually exclusive with disable_bbs_isolation — setting both is rejected
    # at config load (run_config.to_arcticswarm_config).
    force_bbs_isolation: bool = False
    # Communication channels for swarm mode (list of "bbs", "dm", and/or "duo").
    swarm_comm: list[str] = field(default_factory=lambda: ["bbs"])
    # When True and DM is enabled, the orchestrator uses an event-driven
    # multi-turn loop (Claude Code pattern) instead of blocking in
    # wait_for_tasks.  The orchestrator ends its turn after creating tasks
    # and gets woken up by subagent DM notifications.
    orchestrator_realtime: bool = False
    # Timeout (seconds) for the realtime orchestrator DM wait loop.
    orchestrator_realtime_timeout: int = 300
    # Default wait timeout (seconds) for ``prepare_report`` before it returns
    # with a "Timed out waiting for all work to finish" status. The LLM may
    # still pass a smaller ``timeout`` argument; this is just the default
    # baked into the tool schema. Wired from ``eval.prepare_timeout``.
    prepare_report_timeout: int = 300
    # When False (default), complete_task / update_task_summary only DM the
    # leader.  When True, findings are broadcast to every agent (peer review).
    submit_findings_broadcast: bool = False
    # Periodic system-reminder injection interval (-1 = disabled, N = every N tool rounds).
    system_reminder_interval: int = -1
    # When True, SendMessageTool exposes a ``summary`` parameter and peer DM
    # summaries are accumulated for the orchestrator's visibility.
    peer_dm_summary: bool = False
    # When True, ``prepare_report`` exposes a ``force`` parameter that lets
    # the leader skip waiting for stragglers / remaining tasks and submit a
    # partial report with data collected so far.  Duo-style configs should
    # leave this False so the leader must wait for the single auditor's
    # findings before reporting (there are no "stragglers" when there is
    # only one teammate).
    enable_force_submit: bool = False
    # When True (default), the orchestrator must open at least one
    # alternative/contrarian search task (name tagged ``alt`` /
    # ``alternative`` / ``contrarian``, or ``create_task(alt=true)``) before
    # ``prepare_report`` will unlock ``send_user_markdown_report``.  If none
    # exists, ``PrepareReportTool`` auto-spawns a contrarian "find a different
    # candidate" task so the swarm cannot commit to its first candidate
    # unchallenged.  Only enforced on web-capable runs (ANDed with
    # ``has_web_search`` at the construction site).  Targets the
    # "premature commitment correlates with failure" finding.
    enforce_alt_task: bool = True
    # When True, skip the post-answer code-enforced constraint-verification
    # re-loop in the orchestrator (Layer 4a). Ablation knob for the "final
    # verification" review gate; leaves the reviewer-diversity / alt-task
    # gates independent.
    disable_final_verification: bool = False
    # Per-run skill-name remap ``{original: variant}`` applied at skill
    # resolution so ablation arms can swap gate-stripped SKILL.md variants.
    skill_overrides: dict[str, str] = field(default_factory=dict)
    # When True (default), ``prepare_report`` in realtime mode blocks
    # inside ``Mailbox.wait_for_message`` for up to ``timeout`` seconds
    # waiting for a teammate DM before returning "Not ready".  When
    # False, the tool returns immediately with a snapshot of task status
    # and any pending DMs (matching the Claude-Code pattern where
    # messages are delivered between tool rounds via a background poll,
    # not by sleeping inside a tool call).  Duo configs should set this
    # to False — the leader can iterate/poll without burning wall clock,
    # and ``_auto_dm_check`` already injects new teammate DMs between
    # tool rounds.
    blocking_prepare_report: bool = True

    # Peer tool-call observation (see SwarmConfig.peer_tool_observation in
    # run_config.py for full rationale). When True, every observable tool
    # call by one swarm agent is mirrored as a DM into the peer agent's
    # mailbox so the peer learns about file edits / shell commands made by
    # its teammate. Closes the "I have no idea my teammate edited foo.py"
    # stale-view race observed in duo trajectories.
    peer_tool_observation: bool = False
    peer_tool_observation_tools: list[str] = field(default_factory=lambda: [
        "edit_file", "str_replace_based_edit_tool", "bash",
    ])

    # Auditor role in duo mode — see ``SwarmConfig.auditor_role`` in
    # run_config.py for the full rationale. ``"author"`` keeps today's
    # heavy harvest path; ``"reviewer"`` makes the auditor a critic +
    # test validator with a thin pull-on-demand notification instead.
    auditor_role: str = "author"  # "author" | "reviewer"

    # Reviewer-mode pre-submit stall budget (seconds). See
    # ``SwarmConfig.auditor_review_stall_s`` in run_config.py for the
    # full rationale. Defaults to 60s; set to 0 to disable. Only
    # consulted by ``SendReportTool`` when ``auditor_role == "reviewer"``.
    auditor_review_stall_s: float = 60.0

    # Reviewer-diversity gate — see ``SwarmConfig.min_dedicated_reviewers`` in
    # run_config.py for the full rationale. Require at least this many distinct
    # VERIFIED ``#consensus`` verdicts from builder vs dedicated reviewers
    # before ``PrepareReportTool`` unlocks the final report; missing sources
    # are auto-spawned and waited on, bounded by ``max_reviewer_remediations``.
    # Defaults 1/1 (no-op for hybrid / SQL-only / duo runs).
    min_dedicated_reviewers: int = 1
    min_builder_reviewers: int = 1
    max_reviewer_remediations: int = 2

    # Browsing subagent reflection loop — structural ceilings for the
    # Search→Reflect→Summarize loop.  Defaults match constants in
    # arcticswarm.swarm.reflection.
    browsing_max_search_plans: int = 2
    browsing_max_reflection_loops: int = 2
    browsing_reflection_model: str = ""  # empty = use subagent's model

    # When True, use the pre-alignment (legacy) tool description and result
    # format for skills. Used for A/B comparison against the SI-aligned format.
    skill_legacy_format: bool = False
    # When True, present each skill as its own named tool (Claude Code style)
    # instead of a centralized load_skill tool with skill_name enum.
    per_skill_tools: bool = False

    # When True, use the legacy Chat Completions API for GPT models instead
    # of the Responses API.  The Responses API is recommended for reasoning
    # models because it preserves reasoning items across tool-calling turns.
    use_chat_completions: bool = False

    # When True, use streaming LLM calls (call_streaming / run_turn_streaming).
    # When False, use non-streaming calls (call / run_turn), which gives the
    # OpenAI SDK a chance to retry on read-timeout errors automatically.
    use_streaming: bool = True

    # Per-request HTTP timeout (seconds) for OpenAI / Azure OpenAI calls.
    # Lowering from the SDK default (600s) makes stalled calls fail fast so
    # the agent-level retry can re-attempt without wasting minutes.
    # None means use the SDK default.
    llm_timeout: float | None = 120.0

    # When True, completely disables the source content scorer: no
    # "[Source Quality: ...]" annotations are appended to web_fetch results
    # and search-result judge gating is disabled.  Enabled by default; pass
    # --disable-source-scorer to turn it off (e.g. to cut the per-call judge
    # latency, which can be >10x on every web_search / web_fetch call).
    disable_source_scorer: bool = False

    # When True, the web_search repeat-guard escalates to a hard stop (forces a
    # subagent that is stuck looping the same query to finalize) rather than
    # only emitting a soft nudge it can ignore.
    search_repeat_guard_hard_stop: bool = True

    # Runaway near-duplicate (reformulation-loop) force-stop threshold for
    # web_search.  Once a single search intent has been reworded this many
    # times, the repeat-guard forces a stop (only when
    # ``search_repeat_guard_hard_stop`` is on).  Default 40 matches
    # ``WebSearchTool._NEARDUP_HARD_STOP``; lower it (e.g. 12) to bite a
    # churning small model harder, at the cost of possibly truncating a legit
    # broad sweep.
    search_neardup_hard_stop: int = 40

    # When True, collapse duplicate tool-call RESULTS in the outbound LLM
    # history: for each web_search / web_fetch / pdf_read signature issued more
    # than ``dup_history_keep_last`` times, the bulky body of the EARLIER
    # ``tool_result`` blocks is replaced with a compact stub (the last N are
    # kept in full).  Tool_use blocks (tiny) are untouched, so the
    # tool_use<->tool_result pairing is preserved.  Saves context on small
    # models that re-issue identical calls.  See
    # ``Agent._collapse_duplicate_tool_results``.
    collapse_duplicate_tool_history: bool = False
    dup_history_keep_last: int = 1

    # When True, web_fetch results are routed through ContentCompactor
    # instead of SourceScorer.  The compactor chunks the full page (~1000
    # chars/chunk, sentence-aware) and asks an LLM to pick the relevant
    # chunk indices plus a composite source-quality score; only the
    # selected chunks are returned to the agent.  Source scorer is NOT
    # run for web_fetch in this mode (search-result judge gating is
    # unaffected).  See arcticswarm/tools/content_compactor.py.
    use_fetch_compactor: bool = False

    # Same as ``use_fetch_compactor`` but for ``pdf_read`` results.  Both
    # flags share a single ContentCompactor instance under the hood.
    use_pdf_compactor: bool = False

    # Hard cap (in TOKENS, using the codebase-wide ~4-chars/token estimate, so
    # the char budget is this * 4) on the agent-visible output of a single
    # ``web_fetch`` / ``pdf_read`` tool result, AND on the ContentCompactor's
    # selected output.  Bounds any one tool turn so a huge page / docling PDF /
    # compactor selection can't push context past the model window before the
    # reactive compaction net can run.  0 disables.  Enforced as a hard backstop
    # in ``Agent._cap_tool_output`` (covers raw, compactor, fallback, and cache
    # paths) and, for the compactor specifically, in
    # ``ContentCompactor`` (selected chunks are capped to this budget).  The
    # 24_000-token (~96 KB) default is a protective ceiling that won't trim
    # normal pages; the browsecomp bench configs override it to 5_000.
    max_tool_output_tokens: int = 24_000

    # When True, disables the adversarial self-reflection loop for
    # browsing subagents.  Tasks run as a single search pass with no
    # reflect→follow-up cycle.  Useful for ablation.
    disable_self_reflection: bool = False

    # Cheap-win: when answer is empty/refusal at end-of-run, inject one
    # extra orchestrator turn asking for a best-guess. Catches the empty/refusal
    # wrong cases that never reach Layer 4a because there's no answer to verify.
    enable_empty_answer_recovery: bool = False

    # answer-retention fix: when True, prepare_report appends a
    # deterministic "Candidate findings the team converged on" digest, harvested
    # from the BBS (#consensus VERIFIED verdicts + top #key-findings/#discoveries
    # posts), into the report-unlock message. Counters the "the correct answer
    # was present in the swarm's own findings but not in the final
    # answer" failure: even if history compaction or BBS burst-truncation dropped
    # the finding from the leader's working context, the candidate is re-surfaced
    # verbatim right before the final answer is written. Default False preserves
    # behavior for all other runs; enabled only in the qwen browsecomp YAML.
    surface_bbs_candidates: bool = False

    # selective-delete compaction (user request): before the proactive
    # structured-compaction LLM call, deterministically prune CERTAINLY-WRONG
    # tool-result paths (empty "(no output)" searches, "SYSTEM SHUTTING DOWN"
    # timeout notices, "skipped — max tool calls" stubs, is_error results) from
    # the COPY fed to the summarizer — so the model spends its summary budget on
    # real findings instead of junk, and certainly-wrong paths are dropped rather
    # than lossily summarized. Operates on the throwaway compaction input only;
    # never mutates the live message history (no tool_use/tool_result pairing
    # risk). Default False; enabled only in the qwen browsecomp YAML.
    compaction_prune_junk: bool = False

    # Step 2 (compact reflection): use the new constraint-checklist schema
    # instead of the prose JSON schema. Cuts reflection tokens by ~60% per
    # call. Preserves the reflection_stats output shape for downstream
    # consumers (the confidence detector depends on high_conf_ratio).
    enable_compact_reflection: bool = False

    # Step 3 (disagreement gate): on first qualifying BBS post to
    # discoveries / key-findings / consensus, spawn one isolated rival-sweep
    # subagent that hunts for ≥3 candidates without fame bias.
    enable_candidate_emergence_sweep: bool = False
    candidate_emergence_min_chars: int = 60
    candidate_emergence_max_turns: int = 8
    # alt-task dispatch fix: the candidate-emergence rival sweep only
    # task_board.add_task()s the contrarian task — it never dispatches a worker,
    # so on a saturated vLLM (no worker goes idle to pull it) it dies PENDING with
    # 0 tool_uses in a large fraction of cases. When True, the hook ALSO calls
    # ctx.spawn_or_assign() (mirroring the working enforce_alt_task gate path), so
    # the contrarian/diversity task is actually dispatched + run. Targets the
    # recall/anchoring failure bucket. Default False preserves current behavior.
    alt_task_force_dispatch: bool = False
    # anti-anchoring: append the ANTI_ANCHOR_BLOCK to the browsing agent
    # system prompt (decompose constraints, consider multiple interpretations,
    # search each separately, drop+reframe on disconfirm, read pages don't just
    # scan snippets). Targets the dominant recall-miss failure (never-found
    # cases anchored on a wrong framing). Default False; qwen-gated via YAML.
    reframe_prompt: bool = False

    # anti-give-up + canonical-answer report behavior. When True,
    # SendReportTool (1) instructs the orchestrator to emit the fullest
    # canonical/official answer form and honor the question's requested format,
    # and (2) bounces a refusal/give-up FINAL ANSWER back for a committed retry
    # (bounded); the timeout force-report + post-hoc recovery paths are also made
    # to never finalize a give-up. Default False; enabled only in the qwen
    # browsecomp YAML. See RunConfig.reject_refusal_reports.
    reject_refusal_reports: bool = False

    # Reasoning effort: None, "low", "medium", or "high", or "xhigh" (GPT-5 family).
    # Provider-agnostic — mapped to thinking={"type":"enabled", "budget_tokens":N}
    # for Claude and reasoning={"effort": ...} for GPT (Responses API) by the LLM client.
    # None means non-reasoning (parameter omitted from API call).
    reasoning_effort: str | None = None

    # Subagent overrides — when set, subagents use these instead of the
    # main model/reasoning_effort.
    subagent_model: str = ""
    # None means inherit from reasoning_effort.
    subagent_reasoning_effort: str | None = None

    # Per-model effort overrides (resolved by model name AFTER
    # ``model`` / ``subagent_model`` / ``auditor_model`` is applied).
    # Empty dict = no override.  See CONSOLIDATED_CODE_TODOS.md Q2.
    reasoning_effort_by_model: dict[str, str] = field(default_factory=dict)
    subagent_reasoning_effort_by_model: dict[str, str] = field(default_factory=dict)
    auditor_reasoning_effort_by_model: dict[str, str] = field(default_factory=dict)
    # When True for a given model, ``thinking={"type":"adaptive"}`` is
    # NOT attached — effort still routes via ``output_config.effort``.
    # System-card recipe for opus 4.6 BrowseComp.  Q3.
    disable_extended_thinking_by_model: dict[str, bool] = field(default_factory=dict)

    # Available subagent profiles the orchestrator can assign tasks to.
    # Default: ["browsing", "reasoning"].  Choices: browsing, reasoning, coding.
    swarm_profiles: list[str] = field(default_factory=lambda: ["browsing", "reasoning"])
    # Prompt variant for the swarm orchestrator. "default" keeps the
    # delegate-only wording; "exec_enabled" tells the LLM it may use its
    # own execution tools directly when the run intentionally exposes them.
    orchestrator_prompt_mode: str = "default"

    # Extra BBS channels to add beyond core + profile channels.
    swarm_bbs_channels: list[str] = field(default_factory=list)

    # Experimental model flags (e.g., 1M context window).
    # When True, passes experimental={"Enable1MContextModel": True} to the
    # Anthropic API, enabling the 1M context window for supported models.
    enable_1m_context_model: bool = False

    # OpenAI Responses API: when True, never pass ``previous_response_id``
    # so each call re-sends the full conversation instead of relying on the
    # server-stored chain. Set this for models where chaining produces
    # silent empty responses (e.g. GPT-5.4).
    disable_responses_chaining: bool = False

    # Compaction model: when set, summarisation calls
    # (``Agent._call_compaction_llm``) route through a *separate* client built
    # from this model rather than reusing the primary.  Needed when the
    # primary is in a degraded state (e.g. GPT-5.4 emitting empty responses)
    # and compaction itself fails, blocking the empty-response fallback path.
    # Empty string = reuse primary (legacy behaviour).
    compaction_model: str = ""

    # Maximum tool calls the agent will execute per LLM turn.
    # 0 = unlimited (default).  1 = enforce single tool call per turn.
    max_tool_calls_per_turn: int = 0

    # Role-aware override for the ORCHESTRATOR only (swarm leader).  The
    # orchestrator's job is fan-out + coordination (batch ``create_task`` +
    # ``wait_for_tasks``), so a per-turn cap meant to discipline browsing
    # subagents silently drops its batched calls (the dropped intent is never
    # re-queued).  This applies a SEPARATE limit to the orchestrator agent
    # without touching what subagents inherit from ``max_tool_calls_per_turn``.
    #   -1 = inherit ``max_tool_calls_per_turn`` (default; preserves all
    #        existing runs/models byte-for-byte).
    #    0 = unlimited for the orchestrator (subagents keep their own cap).
    #  >=1 = explicit per-turn cap for the orchestrator.
    orchestrator_max_tool_calls_per_turn: int = -1

    # Tool names that ALWAYS execute in a turn even when ``max_tool_calls_per_turn``
    # would otherwise truncate them — they bypass the cap regardless of position
    # in the batch.  The first ``max_tc`` calls are kept as usual, PLUS any call
    # whose name is listed here.  Purpose: guarantee a finding the model tries to
    # post (``post_to_bbs``) always lands, even if the model emits it after
    # another tool in the same turn (e.g. ``web_search`` + ``post_to_bbs`` +
    # ``list_tasks`` keeps the first two, drops ``list_tasks``).  Empty (default)
    # = strict cap, no bypass (preserves prior behavior for other models/runs).
    always_execute_tools_per_turn: list[str] = field(default_factory=list)

    # OpenDataLoader PDF hybrid mode settings.
    # Requires Java 11+. The hybrid backend server is auto-started on a random
    # free port when odl_hybrid_url is empty. Set odl_hybrid_url to an explicit
    # URL (e.g. "http://localhost:5002") to use a pre-started server instead.
    odl_hybrid: str = "docling-fast"        # "docling-fast" or "off" (Java-only)
    odl_hybrid_url: str = ""                # empty = auto-start server on random port
    odl_hybrid_timeout: int = 60000         # milliseconds
    odl_hybrid_fallback_timeout: int = 300  # seconds — fall back to Java-only if hybrid exceeds this
    odl_force_ocr: bool = False             # enable OCR for scanned PDFs

    # Agent identity / prompt style: "web", "general", or "" (auto-infer).
    # When set, build_system_prompt() uses this directly instead of inferring
    # from the tool list.  Empty string = fall back to auto-inference.
    prompt_style: str = ""

    # Output directory for experiment artifacts (empty fallback logs, etc.).
    # Populated from ``eval.output`` when running via the eval CLI.
    output_dir: str = ""

    # Content cache for web_fetch / pdf_read deduplication.
    # When True, results are cached to disk under {output_dir}/cache/content/
    # and shared across all agents working on the same question.
    enable_content_cache: bool = True
    # Node-local cache mirror for multi-host runs: mirror the shared caches to
    # node-local disk and sync deltas back periodically. Avoids the
    # SIGBUS / deadlock from a shared WAL SQLite on a network filesystem (e.g.
    # Lustre) across hosts. See arcticswarm/tools/cache_sync.py.
    #
    # Default True = "mirror automatically whenever caching is enabled AND
    # node-local fast storage exists" (the eval CLI gates on cache_local_dir's
    # mount being present, so a dev box / CPU pod without a fast-disk mount
    # silently uses the master cache directly). Set False to force the mirror
    # off. Empty cache_local_dir disables the mirror unless a fast-disk mount
    # exists. See ENVIRONMENT.md for the cache env vars and node-local mirror
    # setup (paths, optional S3 bucket/prefix).
    cache_local_mirror: bool = True
    cache_local_dir: str = ""
    cache_sync_every: int = 5

    # Global, cross-run web_fetch / pdf_read cache: a single SQLite file shared
    # by EVERY run on the machine, so a URL fetched (or PDF read) once is never
    # re-fetched. Layered under the per-question cache: read global-first,
    # write successes through. Failures are never stored globally (so a
    # transient network error never poisons future runs); on key conflict the
    # longer content wins. Empty string disables it. See
    # arcticswarm/tools/content_cache.py and scripts/build_fetch_cache.py.
    fetch_cache_path: str = _DEFAULT_FETCH_CACHE_PATH

    # --- Declarative tool lists (populated from YAML ToolsConfig) ----------
    # When non-empty, these lists drive tool registration instead of boolean
    # flags.  Agent._register_tools() and SubAgent._apply_profile() read them.
    agent_tools: list[str] = field(default_factory=list)
    orchestrator_tools: list[str] = field(default_factory=list)
    agent_skills: list[str] = field(default_factory=list)
    orchestrator_skills: list[str] = field(default_factory=list)
    # Tools for subagent idle-review cycles (BBS/DM review of teammate
    # findings when the subagent has no assigned tasks).  Empty = use
    # hardcoded defaults in ``Teammate._start_idle_review`` (research vs
    # SQL set, chosen by profile ``idle_review_key``).  Non-empty =
    # replace both defaults verbatim; tools missing from the agent's
    # main toolset are lazy-instantiated via ``ToolFactory``.
    idle_reviewer_tools: list[str] = field(default_factory=list)
    # Per-profile overrides from YAML (profile_name -> {tools, skills, ...}).
    tool_profiles: dict[str, dict] = field(default_factory=dict)

    def has_web_search_capability(self) -> bool:
        """Return True if web search is enabled AND a viable provider exists."""
        if not self.web_search_enabled:
            return False
        # Corpus backends authenticate independently of the Cortex agent:run
        # path: "local" reads a JSONL file, and "cortex" authenticates with a
        # PAT from ~/.snowflake/connections.toml (corpus_pat_connection) using
        # corpus_account/db/schema — NOT sf_params or cortex_account. Gate on
        # the corpus coordinates instead so browsing subagents aren't refused.
        if self.web_search_provider in ("corpus", "cortex-corpus"):
            backend = (getattr(self, "corpus_backend", "") or "stub").strip().lower()
            if backend == "local":
                return bool(getattr(self, "corpus_local_path", "").strip())
            if backend == "cortex":
                return bool(
                    self.corpus_account.strip()
                    and self.corpus_db.strip()
                    and self.corpus_schema.strip()
                )
            return False  # stub -> no real retrieval
        if self.web_search_provider in ("cortex", "cortex-grounding"):
            return bool(self.sf_params) or (bool(self.api_key.strip()) and bool(self.cortex_account.strip()))
        has_brave = bool(self.brave_api_key.strip())
        has_tavily = bool(self.tavily_api_key.strip())
        has_serper = bool(self.serper_api_key.strip())
        return has_brave or has_tavily or has_serper

    def for_subagent(self) -> "ArcticswarmConfig":
        """Return a config copy with subagent overrides applied.

        If no overrides are set, returns ``self`` unchanged.
        ``subagent_reasoning_effort`` values: ``None`` = inherit,
        ``"none"`` = non-reasoning, ``"low"``/``"medium"``/``"high"``/``"xhigh"``/``"max"`` = that level.

        Per-model overrides (``subagent_reasoning_effort_by_model``) win
        over the global ``subagent_reasoning_effort`` when the resolved
        subagent model matches a key.  Q2.
        """
        overrides: dict[str, Any] = {}
        if self.subagent_model:
            overrides["model"] = self.subagent_model
        if self.subagent_reasoning_effort is not None:
            overrides["reasoning_effort"] = (
                None if self.subagent_reasoning_effort == "none"
                else self.subagent_reasoning_effort
            )
        # Per-model override: looks up the resolved subagent model name.
        target_model = overrides.get("model", self.model)
        per_model = self.subagent_reasoning_effort_by_model.get(target_model)
        if per_model is not None:
            overrides["reasoning_effort"] = (
                None if per_model == "none" else per_model
            )
        if not overrides:
            return self
        return replace(self, **overrides)

    def for_auditor(self) -> "ArcticswarmConfig":
        """Return a config copy for the dedicated auditor subagent.

        Uses ``auditor_model`` / ``auditor_reasoning_effort`` if set,
        otherwise falls back to :meth:`for_subagent`.  When
        ``auditor_model`` is a GPT model and no explicit reasoning
        effort is provided, defaults to ``"xhigh"``.  GPT models also
        default to the Chat Completions API (Responses API may not be
        available for all deployments).

        Per-model overrides (``auditor_reasoning_effort_by_model``)
        win over the global ``auditor_reasoning_effort`` when the
        resolved auditor model matches a key.  Q2.
        """
        base = self.for_subagent()
        overrides: dict[str, Any] = {}
        if self.auditor_model:
            overrides["model"] = self.auditor_model
        if self.auditor_reasoning_effort is not None:
            overrides["reasoning_effort"] = (
                None if self.auditor_reasoning_effort == "none"
                else self.auditor_reasoning_effort
            )
        elif self.auditor_model and self.auditor_model.startswith("gpt"):
            overrides["reasoning_effort"] = "xhigh"
        # Per-model override: takes precedence over the global setting.
        target_model = overrides.get("model", base.model)
        per_model = self.auditor_reasoning_effort_by_model.get(target_model)
        if per_model is not None:
            overrides["reasoning_effort"] = (
                None if per_model == "none" else per_model
            )
        # GPT auditor defaults to Chat Completions API — the Responses
        # API may not be available for all Azure deployments.
        if self.auditor_model and self.auditor_model.startswith("gpt"):
            overrides.setdefault("use_chat_completions", True)
        if not overrides:
            return base
        return replace(base, **overrides)

    @classmethod
    def resolve(cls) -> "ArcticswarmConfig":
        """Build config from settings file + defaults.

        Model/API/connection settings come from :func:`settings_json_path`
        (default ``./config_files.json``, or
        ``ARCTICSWARM_SETTINGS_PATH``).
        """
        settings = load_settings()

        api_key = settings.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = settings.get("base_url", _DEFAULT_BASE_URL)
        openai_base_url = settings.get("openai_base_url", "")
        openai_api_key = settings.get("openai_api_key", "") or os.environ.get("OPENAI_API_KEY", "")

        # Public OpenAI by default: the standard endpoint + OPENAI_API_KEY.
        # Override openai_base_url in settings (or OPENAI_BASE_URL) to point at
        # another OpenAI-compatible endpoint.
        if not openai_base_url:
            openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        chosen_model = settings.get("model", _DEFAULT_MODEL)
        max_tokens = int(settings.get("max_tokens", _DEFAULT_MAX_TOKENS))
        max_turns = int(settings.get("max_turns", _DEFAULT_MAX_TURNS))

        sf_conn_name = settings.get("connection", _DEFAULT_SF_CONNECTION)

        # Brave Search API key (prefer settings file; fall back to env)
        brave_api_key = settings.get("brave_api_key", "") or os.environ.get("BRAVE_API_KEY", "")

        # Jina Reader API key (prefer settings file; fall back to env)
        jina_api_key = settings.get("jina_api_key", "") or os.environ.get("JINA_API_KEY", "")

        # Google Serper API key (prefer settings file; fall back to env)
        serper_api_key = settings.get("serper_api_key", "") or os.environ.get("SERPER_API_KEY", "")

        # Tavily Search API key (prefer settings file; fall back to env)
        tavily_api_key = settings.get("tavily_api_key", "") or os.environ.get("TAVILY_API_KEY", "")

        # Account for Cortex (agent:run)
        cortex_account = settings.get("cortex_account", "") or os.environ.get("CORTEX_ACCOUNT", "")

        # Global cross-run fetch cache (SQLite). Precedence: settings file, then
        # env var, then the built-in default. An explicit empty string in
        # settings/env disables it.
        if "fetch_cache_path" in settings:
            fetch_cache_path = settings.get("fetch_cache_path", "")
        elif "ARCTICSWARM_FETCH_CACHE" in os.environ:
            fetch_cache_path = os.environ.get("ARCTICSWARM_FETCH_CACHE", "")
        else:
            fetch_cache_path = _DEFAULT_FETCH_CACHE_PATH

        # Azure OpenAI credentials (optional)
        azure_openai_api_key = (
            settings.get("AZURE_OPENAI_API_KEY", "")
            or settings.get("azure_openai_api_key", "")
        )
        azure_openai_endpoint = (
            settings.get("AZURE_OPENAI_ENDPOINT", "")
            or settings.get("azure_openai_endpoint", "")
        )
        azure_openai_api_version = (
            settings.get("OPENAI_API_VERSION", "")
            or settings.get("AZURE_OPENAI_API_VERSION", "")
            or settings.get("azure_openai_api_version", "")
            or "2025-04-01-preview"
        )
        use_azure_openai = bool(settings.get("use_azure_openai", False))

        # Experimental flags
        enable_1m_context_model = bool(settings.get("Enable1MContextModel", False))

        # Try to load Snowflake params — failures are non-fatal at config time
        sf_params: dict[str, Any] = {}
        try:
            sf_params = get_snowflake_connection_params(sf_conn_name)
        except (KeyError, FileNotFoundError):
            pass  # will fail later when we actually try to connect

        return cls(
            api_key=api_key,
            base_url=base_url,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            model=chosen_model,
            max_tokens=max_tokens,
            max_turns=max_turns,
            use_azure_openai=use_azure_openai,
            azure_openai_api_key=azure_openai_api_key,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_api_version=azure_openai_api_version,
            sf_connection_name=sf_conn_name,
            sf_params=sf_params,
            brave_api_key=brave_api_key,
            jina_api_key=jina_api_key,
            serper_api_key=serper_api_key,
            tavily_api_key=tavily_api_key,
            cortex_account=cortex_account,
            enable_1m_context_model=enable_1m_context_model,
            fetch_cache_path=fetch_cache_path,
        )
