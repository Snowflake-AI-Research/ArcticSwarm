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

"""Context management for the agent loop.

Everything that governs the conversation context fed to the model:

  * :class:`ContextBudget` — proactive token-utilisation tracker that signals
    when to compact *before* a prompt-too-long error occurs (uses the
    ``input_tokens`` every Anthropic / OpenAI response already returns).
  * :class:`TokenUsage` / :func:`_extract_token_usage` — per-turn token
    accounting and cost.
  * :class:`ContextManagementMixin` — the stateful history machinery mixed into
    :class:`arcticswarm.agent.Agent`: compaction (single-pass, structured, and
    split-with-fallback) plus the message-history plumbing that prepares and
    trims the history sent to each provider.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

from arcticswarm.logging_utils import score_aware_truncate, truncate_tool_results

if TYPE_CHECKING:
    from arcticswarm.config import ModelInfo
    from arcticswarm.llm_client import LLMResponse

logger = logging.getLogger(__name__)

# Conservative context limits per model family.  These leave headroom
# for the output (max_tokens) and internal overhead.
_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic Claude 4.x family
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5": 400_000,
    "gpt-5.2": 400_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.4-pro": 1_000_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    # Self-hosted vLLM (Qwen3.5) — native 256K window.  Keyed lowercase; the
    # prefix match below lowercases the model name so the "qwen3.5-27B" alias
    # (capital B) still resolves here instead of the 200K default.
    "qwen3.5-27b": 262_144,
    "qwen": 262_144,
}

_DEFAULT_CONTEXT_LIMIT = 200_000
_1M_CONTEXT_LIMIT = 1_000_000

# Anthropic models that support the ``context-1m-2025-08-07`` beta header.
# Per Anthropic docs, only Sonnet 4 / 4.5 expose 1M directly; the 4-6 entries
# are aliases for endpoints that already route 1M context, in which case
# attaching the header is a harmless no-op.
_ANTHROPIC_1M_CONTEXT_MODEL_PREFIXES: tuple[str, ...] = (
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-sonnet-4-5",
    "claude-sonnet-4",
)


def supports_anthropic_1m_context(model: str) -> bool:
    """Return True if ``model`` is an Anthropic model that supports the
    ``context-1m-2025-08-07`` beta header."""
    bare = model
    for provider_prefix in ("anthropic-", "azure-"):
        if bare.startswith(provider_prefix):
            bare = bare[len(provider_prefix):]
            break
    return any(bare.startswith(p) for p in _ANTHROPIC_1M_CONTEXT_MODEL_PREFIXES)


@dataclass
class ContextBudget:
    """Track context token utilisation and signal when compaction is needed.

    After each LLM call, the caller updates ``last_input_tokens`` with the
    ``input_tokens`` value from the API response.  ``should_compact()``
    returns True when the context has consumed ≥ ``threshold_fraction`` of
    the model's context limit.

    Parameters
    ----------
    model:
        Model identifier used to look up the context window size.
    threshold_fraction:
        Fraction of the context limit at which to trigger compaction.
        Default 0.90 — at 90 % of 200 K = 180 K tokens, only ~20 K
        tokens of headroom remain.  This is deliberately conservative:
        we only compact when genuinely close to the limit, not earlier,
        because compaction destroys information and costs an LLM call.
        The existing reactive compaction in ``_call_llm_with_retry()``
        remains as a safety net for the final ~10 %.
    threshold_tokens_override:
        Absolute token count at which to trigger compaction.  When set
        (non-zero), wins over ``threshold_fraction``.  Useful for 1 M
        context models where 90 % = 900 K is far too late to start
        summarising — set this to e.g. 200_000 to compact early.
    enable_1m_context_model:
        When True, override the context limit to 1 M tokens.
    """

    model: str = ""
    threshold_fraction: float = 0.90
    threshold_tokens_override: int = 0
    # Actual served context window (e.g. a vLLM server's /v1/models
    # max_model_len). When > 0 it overrides the model-name table lookup,
    # keeping utilization() and the reactive empty-response compaction trigger
    # accurate for served models whose window differs from the table (e.g.
    # Tongyi-DeepResearch's 131072 vs the qwen* default of 262144).
    context_limit_override: int = 0
    enable_1m_context_model: bool = False
    last_input_tokens: int = 0
    peak_input_tokens: int = 0

    # --- derived helpers ---------------------------------------------------

    @property
    def context_limit(self) -> int:
        """Return the context window size for the current model."""
        if self.enable_1m_context_model:
            return _1M_CONTEXT_LIMIT
        # The actual served window (vLLM /v1/models) wins over the name table.
        if self.context_limit_override > 0:
            return self.context_limit_override
        # Try exact match first, then prefix match for versioned names.
        # Strip any provider prefix ("openai-", "anthropic-", "azure-") before
        # the prefix match so e.g. "openai-gpt-5.4" still resolves to the
        # gpt-5.4 entry's 1M limit instead of falling through to the 200K
        # default. Defense in depth: when enable_1m_context_model is False
        # (or hasn't propagated), this prevents a silent budget downgrade.
        if self.model in _MODEL_CONTEXT_LIMITS:
            return _MODEL_CONTEXT_LIMITS[self.model]
        bare_model = self.model
        for provider_prefix in ("openai-", "anthropic-", "azure-"):
            if bare_model.startswith(provider_prefix):
                bare_model = bare_model[len(provider_prefix):]
                break
        if bare_model != self.model and bare_model in _MODEL_CONTEXT_LIMITS:
            return _MODEL_CONTEXT_LIMITS[bare_model]
        # Lowercase for the prefix match so case-variant aliases (e.g.
        # "qwen3.5-27B" vs the lowercase "qwen3.5-27b" key) still resolve.
        bare_lower = bare_model.lower()
        for prefix, limit in _MODEL_CONTEXT_LIMITS.items():
            if bare_lower.startswith(prefix):
                return limit
        return _DEFAULT_CONTEXT_LIMIT

    @property
    def threshold_tokens(self) -> int:
        """Absolute token count at which compaction should trigger."""
        if self.threshold_tokens_override > 0:
            return self.threshold_tokens_override
        return int(self.context_limit * self.threshold_fraction)

    # --- public API --------------------------------------------------------

    def update(self, input_tokens: int) -> None:
        """Record the input_tokens count from the latest LLM response."""
        self.last_input_tokens = input_tokens
        if input_tokens > self.peak_input_tokens:
            self.peak_input_tokens = input_tokens

    def should_compact(self) -> bool:
        """Return True if context utilisation has reached the threshold."""
        return self.last_input_tokens >= self.threshold_tokens

    def utilization(self) -> float:
        """Return context utilisation as a fraction (0.0 – 1.0+)."""
        limit = self.context_limit
        if limit <= 0:
            return 0.0
        return self.last_input_tokens / limit

    def reset(self) -> None:
        """Reset tracking after a compaction (actual count unknown until next call)."""
        self.last_input_tokens = 0


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Accumulated token usage for an agentic turn or session."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.reasoning_tokens += other.reasoning_tokens
        return self

    def cost(self, model: "ModelInfo") -> float:
        """Compute API cost in USD for this usage at the given model's rates."""
        return (
            self.input_tokens * model.input_per_mtok
            + self.output_tokens * model.output_per_mtok
            + self.cache_read_input_tokens * model.cache_read_per_mtok
            + self.cache_creation_input_tokens * model.cache_write_per_mtok
        ) / 1_000_000


def _extract_token_usage(response: LLMResponse) -> TokenUsage:
    """Convert an :class:`LLMResponse` to our :class:`TokenUsage`."""
    return TokenUsage(
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cache_creation_input_tokens=response.cache_creation_input_tokens,
        cache_read_input_tokens=response.cache_read_input_tokens,
        reasoning_tokens=response.reasoning_tokens,
    )


# ---------------------------------------------------------------------------
# Context-management mixin (compaction + message-history plumbing)
# ---------------------------------------------------------------------------


class ContextManagementMixin:
    """Conversation-context machinery mixed into :class:`arcticswarm.agent.Agent`.

    Compaction (single-pass / structured / split-with-fallback) plus the
    message-history plumbing that prepares and trims the history sent to each
    provider. All methods operate on ``Agent`` instance state via ``self`` and
    are resolved on the combined class via the MRO; cross-cutting class
    constants (fallback / block-type sets) remain defined on ``Agent``.
    """

    _COMPACTION_PROMPT = (
        "The conversation history has grown too long and needs to be summarized. "
        "Please provide a detailed summary of everything that has happened so far, "
        "including:\n"
        "- The original user question/task\n"
        "- All key findings, facts, and data points discovered\n"
        "- All tool calls made and their important results\n"
        "- Any conclusions or partial answers reached\n"
        "- What still needs to be done to complete the task\n\n"
        "Be thorough — this summary will replace the full conversation history. "
        "Preserve all specific facts, numbers, URLs, and quotes."
    )

    _SPLIT_COMPACTION_PROMPT = (
        "This is the {which_half} of a conversation history that has grown "
        "too long. The conversation is being summarized in two halves and the "
        "results will be merged.\n\n"
        "Please provide a detailed summary of THIS HALF of the conversation, "
        "including:\n"
        "- All key findings, facts, and data points discovered\n"
        "- All tool calls made and their important results\n"
        "- Any conclusions or partial answers reached\n"
        "- Any important context needed to understand the other half\n\n"
        "Do NOT summarize the original question — that will be preserved "
        "separately. Focus only on summarizing the work done in this half.\n"
        "Be thorough — preserve all specific facts, numbers, URLs, and quotes."
    )

    _STRUCTURED_COMPACTION_PROMPT = (
        "The conversation history needs to be compacted to continue research. "
        "Produce a STRUCTURED summary preserving the following sections. "
        "This summary will replace the full conversation — anything not "
        "retained here will be lost permanently.\n\n"
        "## Original Question Constraints\n"
        "Restate every distinct factual constraint from the question as a "
        "numbered list. Tag each one with:\n"
        "- [DISCRIMINATING] — narrow constraint that few entities would "
        "satisfy (e.g., a specific year, episode, exact title, unusual "
        "biographical detail).\n"
        "- [BROAD] — many entities could satisfy this (e.g., \"religious\", "
        "\"American\", \"academic\").\n"
        "At least one constraint MUST be tagged [DISCRIMINATING].\n\n"
        "## Candidates Considered\n"
        "List EVERY candidate answer that has been proposed, evaluated, OR "
        "rejected during the conversation — including ones briefly mentioned "
        "and dismissed. Do NOT drop rejected candidates; they are critical "
        "for downstream rival audit. For each candidate include:\n"
        "- name: exact spelling, including original-language form "
        "(e.g., \"Wróblewska\" not \"Wroblewski\"; preserve diacritics)\n"
        "- status: ACTIVE_LEADER | ACTIVE_RIVAL | REJECTED | UNCERTAIN\n"
        "- mention_count: approximate number of times surfaced across teammates\n"
        "- first_surfaced_by: which teammate / which search query first "
        "surfaced it\n"
        "- rejection_reason: if REJECTED, one sentence on why AND which "
        "constraint triggered the rejection (e.g., \"REJECTED: born 1975, "
        "fails [DISCRIMINATING] constraint #3 'born in the 80s'\")\n\n"
        "## Constraint × Candidate Verification Matrix\n"
        "For each constraint (rows) × each candidate listed above (columns), "
        "record one of:\n"
        "- EXACT — primary-source quote directly verifying the constraint "
        "(cite URL + ≤30-word quote)\n"
        "- PARTIAL — match by inference, proxy, or secondary source "
        "(cite URL + reasoning)\n"
        "- CONTRADICTED — primary-source evidence directly disproving "
        "(cite URL + quote)\n"
        "- UNKNOWN — not yet investigated for this candidate\n\n"
        "Preserve ALL exact numbers, dates, currencies, unit symbols, and "
        "proper-noun spellings VERBATIM. Do NOT normalize: keep \"£500\" as "
        "£500 (do not write \"500 pounds\"), keep \"0.1%\" as 0.1% (do not "
        "write \"0.1\"), keep original-language episode/title spellings, "
        "keep \"S2 E04\" not \"Season 2 Episode 4\".\n\n"
        "## Discriminating Evidence\n"
        "For each [DISCRIMINATING] constraint, summarize the evidence that "
        "distinguishes the leader from the top rival(s). If a "
        "[DISCRIMINATING] constraint has not been checked against any "
        "candidate other than the leader, flag this explicitly: "
        "\"⚠ Single-candidate verification — rivals not checked on "
        "constraint #N.\"\n\n"
        "## Search Strategies Tried\n"
        "List exact queries used (so they are not re-run) and label each:\n"
        "- PRODUCTIVE: yielded a candidate or clue\n"
        "- DEAD_END: no useful results — also note WHY (no hits, paywall, "
        "off-topic) so future searches pivot rather than retry.\n\n"
        "## Sources Consulted\n"
        "List unique URLs visited. For each high-relevance URL include:\n"
        "- 1-sentence content summary\n"
        "- which candidate(s) it provides evidence for or against\n"
        "- which constraint(s) it bears on\n\n"
        "## Open Investigations\n"
        "- Constraints marked UNKNOWN for any ACTIVE candidate\n"
        "- Candidates marked UNCERTAIN that need follow-up\n"
        "- [DISCRIMINATING] constraint dimensions never explicitly explored "
        "for rivals (single-candidate fixation risk)\n\n"
        "## Compaction Integrity Check\n"
        "Before finishing, confirm:\n"
        "- [ ] Every distinct candidate name that appeared in the conversation "
        "is in \"Candidates Considered\" (rejected ones included).\n"
        "- [ ] Every [DISCRIMINATING] constraint has at least one row in the "
        "matrix.\n"
        "- [ ] Every exact number, date, currency, and proper-noun spelling "
        "from the conversation appears in the summary verbatim somewhere.\n"
        "- [ ] The leader's verdict on every [DISCRIMINATING] constraint is "
        "EXACT or PARTIAL with a citation, not unsupported.\n\n"
        "If any check fails, fix the summary before returning.\n\n"
        "Be exhaustive. Losing a rival or a [DISCRIMINATING] fact will make "
        "downstream verification blind to it."
    )

    @classmethod
    def _filter_blocks_for_provider(
        cls,
        blocks: list[dict[str, Any]],
        provider: str,
        *,
        is_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        """Drop content blocks that aren't native to the target provider."""
        if provider == "anthropic":
            allowed = cls._FALLBACK_ALLOWED_BLOCK_TYPES if is_fallback else cls._ANTHROPIC_ALLOWED_BLOCK_TYPES
            return [
                b for b in blocks
                if not isinstance(b, dict)
                or b.get("type") in allowed
            ]
        return blocks

    @classmethod
    def _messages_for_provider(
        cls,
        msgs: list[dict[str, Any]],
        provider: str,
        *,
        is_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        """Return a copy of *msgs* safe to send to *provider*.

        Strips provider-incompatible blocks (e.g. OpenAI Responses-API
        ``reasoning`` items when targeting Anthropic) and then runs the
        standard sanitize pass so messages that become empty after
        filtering are dropped rather than triggering a 400.

        When *is_fallback* is True, also strips thinking/redacted_thinking
        blocks since the fallback model does not enable thinking.
        """
        out: list[dict[str, Any]] = []
        for m in msgs:
            content = m.get("content")
            if isinstance(content, list):
                filtered = cls._filter_blocks_for_provider(content, provider, is_fallback=is_fallback)
                if len(filtered) != len(content):
                    m = {**m, "content": filtered}
            out.append(m)
        return cls._sanitize_messages(out)

    @staticmethod
    def _estimate_msg_tokens(m: dict[str, Any]) -> int:
        """Cheap char-based token estimate for one message.

        Anthropic does not ship a free-standing tokenizer for Claude, so we
        fall back to the canonical "~4 characters per token" rule of thumb
        (matches ``llm_client.LLMClient._estimate_input_tokens``).  Used only
        for trim-budget bookkeeping, not for billing or hard limits.
        """
        try:
            chars = len(json.dumps(m, default=str))
        except (TypeError, ValueError):
            chars = len(str(m))
        return chars // 4

    @classmethod
    def _trim_messages_for_sonnet4_fallback(
        cls,
        msgs: list[dict[str, Any]],
        *,
        head_tokens: int | None = None,
        tail_tokens: int | None = None,
        sentinel: str | None = None,
    ) -> list[dict[str, Any]]:
        """Keep first ``head_tokens`` + sentinel + last ``tail_tokens`` of msgs.

        Walks message-by-message (we cannot trim mid-message without
        breaking content-block schemas) using the 4-chars-per-token estimate
        and stops as soon as the next message would exceed the budget.  A
        synthetic ``user`` message containing the sentinel marker is
        inserted between the kept head and tail so the model sees that the
        history was truncated.

        The caller is responsible for running ``_sanitize_messages`` on the
        result — trimming may leave dangling tool_use / tool_result pairs
        that the sanitizer's stub-injection pass will repair.
        """
        head_budget = head_tokens if head_tokens is not None else cls._FALLBACK_TRIM_HEAD_TOKENS
        tail_budget = tail_tokens if tail_tokens is not None else cls._FALLBACK_TRIM_TAIL_TOKENS
        sentinel_text = sentinel if sentinel is not None else cls._FALLBACK_TRIM_SENTINEL

        if not msgs:
            return msgs
        total = sum(cls._estimate_msg_tokens(m) for m in msgs)
        if total <= head_budget + tail_budget:
            return msgs

        head: list[dict[str, Any]] = []
        head_used = 0
        head_idx = 0
        for i, m in enumerate(msgs):
            mt = cls._estimate_msg_tokens(m)
            if head_used + mt > head_budget and head:
                break
            head.append(m)
            head_used += mt
            head_idx = i + 1

        tail: list[dict[str, Any]] = []
        tail_used = 0
        for m in reversed(msgs[head_idx:]):
            mt = cls._estimate_msg_tokens(m)
            if tail_used + mt > tail_budget and tail:
                break
            tail.insert(0, m)
            tail_used += mt

        sentinel_msg = {"role": "user", "content": sentinel_text}
        return head + [sentinel_msg] + tail

    @staticmethod
    def _score_aware_truncate(
        msgs: list[dict[str, Any]],
        high_score_threshold: float = 30.0,
        mid_high_score_threshold: float = 22.0,
        mid_low_score_threshold: float = 12.0,
        low_score_max_chars: int = 500,
        mid_low_score_max_chars: int = 2000,
        mid_high_score_max_chars: int = 4000,
        high_score_max_chars: int = 8000,
        no_score_max_chars: int = 2000,
    ) -> list[dict[str, Any]]:
        """Delegate to :func:`logging_utils.score_aware_truncate`."""
        return score_aware_truncate(
            msgs,
            high_score_threshold=high_score_threshold,
            mid_high_score_threshold=mid_high_score_threshold,
            mid_low_score_threshold=mid_low_score_threshold,
            low_score_max_chars=low_score_max_chars,
            mid_low_score_max_chars=mid_low_score_max_chars,
            mid_high_score_max_chars=mid_high_score_max_chars,
            high_score_max_chars=high_score_max_chars,
            no_score_max_chars=no_score_max_chars,
        )

    def _compact_context_structured(self) -> bool:
        """Structured compaction that preserves candidate answers and evidence.

        Uses :meth:`_score_aware_truncate` to compress low-value tool results
        first, then sends the structured compaction prompt to the LLM.
        Falls back to the generic :meth:`_compact_context` cascade on failure.
        """
        if len(self.messages) <= 2:
            return False

        old_count = len(self.messages)
        logger.warning(
            "Proactive structured compaction (%d messages, %d input tokens)…",
            old_count, self._context_budget.last_input_tokens,
        )

        # Pre-truncate low-value content so the compaction LLM call fits
        try:
            truncated_msgs = self._score_aware_truncate(
                self.messages,
            )
        except Exception as exc:
            logger.warning(
                "Score-aware truncation failed (%s); using raw messages for compaction.",
                exc,
            )
            truncated_msgs = copy.deepcopy(self.messages)

        # selective-delete: prune certainly-wrong tool-result paths from the
        # (throwaway) summarizer input so the model summarizes real findings, not
        # junk. Operates on truncated_msgs only — never the live history.
        if getattr(self.config, "compaction_prune_junk", False):
            try:
                n_pruned = self._prune_certainly_wrong(truncated_msgs)
                if n_pruned:
                    logger.info("Selective-delete pruned %d junk tool-results before compaction.", n_pruned)
            except Exception as exc:  # never let pruning break compaction
                logger.warning("Junk-prune failed (%s); proceeding without it.", exc)

        # Attempt single-pass structured compaction
        try:
            summary_text = self._call_compaction_llm(
                truncated_msgs + [
                    {"role": "user", "content": self._STRUCTURED_COMPACTION_PROMPT}
                ],
            )
            if summary_text:
                self._apply_compacted_summary(summary_text, old_count)
                return True
            # None → context too long even after truncation; fall through
            logger.warning(
                "Structured compaction too long; falling back to generic compaction."
            )
        except Exception as exc:
            logger.warning("Structured compaction failed (%s); falling back.", exc)

        # Fall back to the existing split-then-merge cascade
        return self._compact_context()

    # Certainly-wrong tool-result signatures: empty results, the timeout
    # shutdown notice, the per-turn-cap skip stub, and connection errors. These
    # carry no investigative value, so pruning them from the summarizer input
    # frees budget for real findings (selective-delete).
    _JUNK_RESULT_MARKERS = (
        "(no output)",
        "SYSTEM SHUTTING DOWN",
        "is now DISABLED",
        "skipped — max tool calls",
        "skipped - max tool calls",
        "Connection error",
        "(no results)",
    )

    def _prune_certainly_wrong(self, msgs: list[dict[str, Any]]) -> int:
        """Replace certainly-wrong tool_result CONTENT with a tiny placeholder.

        Mutates ``msgs`` (the throwaway compaction input copy) in place; preserves
        every message and every tool_result block (so tool_use/tool_result pairing
        is untouched), only shrinking junk content. Returns the number pruned.
        """
        pruned = 0
        placeholder = "(pruned: empty/failed tool result — no useful content)"
        for m in msgs:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                is_err = bool(block.get("is_error"))
                inner = block.get("content")
                if isinstance(inner, list):
                    text = " ".join(
                        b.get("text", "") for b in inner if isinstance(b, dict)
                    )
                elif isinstance(inner, str):
                    text = inner
                else:
                    text = ""
                stripped = text.strip()
                is_junk = (
                    is_err
                    or stripped == ""
                    or any(mk in text for mk in self._JUNK_RESULT_MARKERS)
                )
                # Only prune SHORT junk — a long result that merely contains an
                # error string elsewhere may still hold useful content.
                if is_junk and len(stripped) < 600:
                    block["content"] = placeholder
                    pruned += 1
        return pruned

    def _last_oversized_fetch_tool(self, char_threshold: int) -> str | None:
        """Return ``"web_fetch"`` / ``"pdf_read"`` if a recent result of that tool alone exceeds ``char_threshold``.

        Used by ``_maybe_proactive_compact`` to skip full-history
        compaction when one un-chunked tool result is itself dominating
        the context budget — full-history compaction wouldn't help in
        that case; the per-tool ContentCompactor (``--use-fetch-compactor``
        / ``--use-pdf-compactor``) is the right remedy.

        Walks the last ~10 messages, builds a ``tool_use_id -> tool_name``
        map from assistant messages, then scans tool_result blocks in
        user messages, returning the name of the largest oversized
        ``web_fetch`` / ``pdf_read`` result (None if none).
        """
        recent = self.messages[-10:]
        id_to_name: dict[str, str] = {}
        for msg in recent:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type")
                if t == "tool_use":
                    cid, name = blk.get("id"), blk.get("name")
                    if cid and name:
                        id_to_name[cid] = name
                elif t == "function_call":
                    cid, name = blk.get("call_id"), blk.get("name")
                    if cid and name:
                        id_to_name[cid] = name

        for msg in reversed(recent):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type")
                if t == "tool_result":
                    cid = blk.get("tool_use_id")
                    inner = blk.get("content")
                elif t == "function_call_output":
                    cid = blk.get("call_id")
                    inner = blk.get("output")
                else:
                    continue
                name = id_to_name.get(cid or "", "")
                if name not in ("web_fetch", "pdf_read"):
                    continue
                if isinstance(inner, str):
                    size = len(inner)
                elif isinstance(inner, list):
                    size = sum(
                        len(b.get("text", "")) if isinstance(b, dict)
                        else len(str(b))
                        for b in inner
                    )
                else:
                    size = len(str(inner)) if inner else 0
                if size >= char_threshold:
                    return name
        return None

    def _maybe_proactive_compact(self) -> bool:
        """Check context budget and proactively compact if approaching limit.

        Called after each tool-call batch (between LLM round-trips).
        Returns True if compaction was performed.
        """
        if not self._context_budget.should_compact():
            return False
        if len(self.messages) <= 4:
            return False

        # Guard: if a single recent web_fetch / pdf_read result is itself
        # at or above the compaction budget, full-history compaction is
        # the wrong tool — it would summarise everything *except* the
        # huge raw result, leaving the agent in the same state. The
        # per-tool ContentCompactor is the right remedy. Skip proactive
        # compaction here; reactive compaction remains the safety net if
        # the next call still hits prompt_too_long.
        # ~4 chars/token is a coarse but standard rule of thumb.
        oversized = self._last_oversized_fetch_tool(
            char_threshold=self._context_budget.threshold_tokens * 4,
        )
        if oversized is not None:
            logger.warning(
                "Skipping proactive compaction: a single %s result alone "
                "exceeds the compaction budget (%d tokens). Enable "
                "%s to chunk the content before it reaches the agent.",
                oversized,
                self._context_budget.threshold_tokens,
                "--use-fetch-compactor" if oversized == "web_fetch"
                else "--use-pdf-compactor",
            )
            return False

        logger.info(
            "Proactive compaction triggered: %d input tokens (%.0f%% of %d limit)",
            self._context_budget.last_input_tokens,
            self._context_budget.utilization() * 100,
            self._context_budget.context_limit,
        )
        success = self._compact_context_structured()
        if success:
            self.compaction_count += 1
            self.proactive_compaction_count += 1
            self._context_budget.reset()
            logger.info(
                "Proactive compaction succeeded (compaction #%d). Context budget reset.",
                self.compaction_count,
            )
        else:
            logger.warning(
                "Proactive compaction FAILED — next LLM call may hit prompt-too-long. "
                "input_tokens=%d, limit=%d",
                self._context_budget.last_input_tokens,
                self._context_budget.context_limit,
            )
            # Safety: a failed compaction may have left the message history
            # in a state that's inconsistent with the server-side chained
            # context (e.g. truncated tool_results without the chain being
            # reset).  Force the next call to send the full history instead
            # of a stale delta so we don't trip
            # "No tool call found for function call output with call_id ..."
            self._last_response_id = None
            self._msg_checkpoint = 0
        return success

    def _call_compaction_llm(self, messages: list[dict[str, Any]]) -> str | None:
        """Send a compaction request and return the summary text, or None on failure."""
        stripped = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        sanitized = self._sanitize_messages(stripped)

        # Debug: log any messages with suspicious content before sending
        for i, m in enumerate(sanitized):
            c = m.get("content")
            if c is None or ("content" not in m):
                logger.error(
                    "COMPACTION DEBUG: msg[%d] role=%s has content=%r (type=%s), keys=%s",
                    i, m.get("role"), c, type(c).__name__, list(m.keys()),
                )
            elif isinstance(c, list) and len(c) == 0:
                logger.error(
                    "COMPACTION DEBUG: msg[%d] role=%s has empty list content, keys=%s",
                    i, m.get("role"), list(m.keys()),
                )

        # When ``config.compaction_model`` is set, build a separate client so
        # compaction doesn't reuse the failing primary.
        # When GPT-5.4 emits empty primary responses, it also emits
        # empty for compaction, and the agent then ships uncompacted history
        # to the cross-model fallback (which then 400s with "max tokens of
        # 200000 exceeded").
        compaction_model = (
            getattr(self.config, "compaction_model", "") or self.config.model
        )
        if compaction_model != self.config.model:
            compaction_client = self._make_llm_client(
                model=compaction_model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                openai_base_url=getattr(self.config, "openai_base_url", ""),
                openai_api_key=getattr(self.config, "openai_api_key", ""),
                use_azure_openai=False,  # stable summariser, not the primary's deployment
                # Stable Anthropic summariser supports 1M; force it on so a
                # large pre-compaction history doesn't trip the 200K cap.
                enable_1m_context_model=(
                    compaction_model.startswith("claude-sonnet-4-6")
                    or compaction_model.startswith("claude-opus-4-6")
                    or getattr(self.config, "enable_1m_context_model", False)
                ),
            )
            close_after = True
        else:
            compaction_client = self.client
            close_after = False

        try:
            response = compaction_client.call(
                model=compaction_model,
                max_tokens=8192,
                system_prompt=self.system_prompt,
                tools=[],
                messages=sanitized,
                reasoning_effort=None,
            )
            self._record_role_usage("history_compactor", response)
            parts = []
            for block in response.content_blocks:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = "\n".join(parts).strip()
            return text if text else None
        except Exception as exc:
            if self._is_context_too_long(exc):
                return None  # signal to caller to try split approach
            # Log the failing message for debugging "Field required" errors
            import re
            match = re.search(r'messages\.(\d+)\.content', str(exc))
            if match:
                idx = int(match.group(1))
                if idx < len(sanitized):
                    bad = sanitized[idx]
                    logger.error(
                        "COMPACTION DEBUG: API rejected msg[%d] role=%s "
                        "content_type=%s keys=%s content_preview=%s",
                        idx, bad.get("role"),
                        type(bad.get("content")).__name__,
                        list(bad.keys()),
                        repr(bad.get("content"))[:300],
                    )
            raise  # non-context errors propagate
        finally:
            if close_after:
                try:
                    compaction_client.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass

    @staticmethod
    def _truncate_tool_results(
        msgs: list[dict[str, Any]], max_chars: int = 2000,
    ) -> list[dict[str, Any]]:
        """Delegate to :func:`logging_utils.truncate_tool_results`."""
        return truncate_tool_results(msgs, max_chars=max_chars)

    def _compaction_with_fallback(
        self,
        prepared_msgs: list[dict[str, Any]],
        which_half: str,
        raw_msg_count: int,
    ) -> str | None:
        """Try progressively more aggressive strategies to summarize a half.

        Returns the summary text, or ``None`` if all attempts fail.

        Strategies (in order):
        1. Full messages as-is.
        2. Truncate tool_result content to 2 000 chars.
        3. Drop oldest 2/3 of messages, keep only the most recent third.
        4. Hard fallback: return a minimal text-only extract.
        """
        # --- Strategy 1: full messages ----------------------------------------
        try:
            summary = self._call_compaction_llm(prepared_msgs)
            if summary:
                logger.info(
                    "Split compaction: %s summarized (%d msgs → %d chars).",
                    which_half, raw_msg_count, len(summary),
                )
                return summary
        except Exception as exc:
            logger.warning("Split compaction (%s) attempt 1 failed: %s", which_half, exc)

        # --- Strategy 2: truncated tool results --------------------------------
        logger.warning(
            "Split compaction (%s): full messages too long, truncating tool results...",
            which_half,
        )
        truncated = self._truncate_tool_results(prepared_msgs, max_chars=2000)
        try:
            summary = self._call_compaction_llm(truncated)
            if summary:
                logger.info(
                    "Split compaction: %s summarized with truncation (%d msgs → %d chars).",
                    which_half, raw_msg_count, len(summary),
                )
                return summary
        except Exception as exc:
            logger.warning("Split compaction (%s) attempt 2 (truncated) failed: %s", which_half, exc)

        # --- Strategy 3: keep only recent 1/3 of messages ----------------------
        keep = max(len(truncated) // 3, 2)
        trimmed = truncated[-keep:]
        # Ensure it starts with a user message
        if trimmed and trimmed[0].get("role") != "user":
            trimmed.insert(0, {
                "role": "user",
                "content": f"[Trimmed context for {which_half} — only the most recent messages are shown.]",
            })
        logger.warning(
            "Split compaction (%s): truncation insufficient, trimming to %d/%d messages...",
            which_half, len(trimmed), len(truncated),
        )
        try:
            summary = self._call_compaction_llm(trimmed)
            if summary:
                logger.info(
                    "Split compaction: %s summarized after trimming (%d msgs → %d chars).",
                    which_half, raw_msg_count, len(summary),
                )
                return summary
        except Exception as exc:
            logger.warning("Split compaction (%s) attempt 3 (trimmed) failed: %s", which_half, exc)

        # --- Strategy 4: hard fallback — extract text from last assistant ------
        logger.error(
            "Split compaction (%s): all LLM attempts failed, using hard fallback.",
            which_half,
        )
        for m in reversed(prepared_msgs):
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            return f"[Partial summary — {which_half}, extracted from last assistant response]\n\n{text[:4000]}"
            elif isinstance(content, str) and content.strip():
                return f"[Partial summary — {which_half}, extracted from last assistant response]\n\n{content[:4000]}"
        return f"[No summary available for {which_half} — messages were too large to process.]"

    def _compact_context(self) -> bool:
        """Summarize conversation history to reduce context size.

        Replaces ``self.messages`` with a single user message containing the
        summary.  Returns True if compaction succeeded, False otherwise.

        **Strategy:**

        1. **Single-pass**: Try summarizing the full conversation in one call.
        2. **Split-then-merge**: If the full conversation exceeds the context
           limit, split messages (excluding the first/question message) into
           two halves. Summarize each half independently in parallel-safe
           sequential calls, then merge both summaries with the original
           question into one compacted message. This preserves ALL information
           without dropping any messages.
        """
        if len(self.messages) <= 2:
            return False

        old_count = len(self.messages)
        logger.warning(
            "Context too long (%d messages). Compacting conversation history...",
            old_count,
        )

        # --- Attempt 1: single-pass compaction --------------------------------
        try:
            summary_text = self._call_compaction_llm(
                self.messages + [
                    {"role": "user", "content": self._COMPACTION_PROMPT}
                ],
            )
            if summary_text:
                self._apply_compacted_summary(summary_text, old_count)
                return True
            # None means context too long — fall through to split approach
            logger.warning(
                "Single-pass compaction failed (context too long). "
                "Trying split-then-merge approach..."
            )
        except Exception as exc:
            logger.error("Single-pass compaction failed: %s", exc)
            return False

        # --- Attempt 2: split-then-merge compaction ---------------------------
        first_msg = self.messages[0]  # original question
        rest = self.messages[1:]

        if len(rest) < 2:
            logger.error("Not enough messages to split for compaction.")
            return False

        # Find a safe split point at a turn boundary.  Messages alternate
        # assistant (tool_use) → user (tool_result).  A "turn" is an
        # (assistant, user) pair.  We must split BETWEEN turns — i.e.
        # after a user message and before the next assistant message —
        # so neither half has a broken assistant/tool_result pair.
        mid = len(rest) // 2
        # Scan forward from midpoint to find a user→assistant boundary
        split_idx = mid
        while split_idx < len(rest) - 1:
            if rest[split_idx].get("role") == "user":
                break  # split after this user message
            split_idx += 1
        else:
            # Scan backward instead
            split_idx = mid
            while split_idx > 0:
                if rest[split_idx - 1].get("role") == "user":
                    split_idx -= 1
                    break
                split_idx -= 1

        # split_idx is the last message of the first half (inclusive)
        first_half = rest[:split_idx + 1]
        second_half = rest[split_idx + 1:]

        if not first_half or not second_half:
            # Degenerate split — one half is empty
            logger.error("Could not find valid split point for compaction.")
            return False

        # Helper to ensure valid message alternation for each half
        def _prepare_half(msgs: list[dict[str, Any]], which: str) -> list[dict[str, Any]]:
            """Wrap a message slice into a valid user→assistant… sequence."""
            result = []
            # Start with a bridging user message providing context
            result.append({
                "role": "user",
                "content": (
                    f"[This is the {which} of a long conversation. "
                    f"The original question/task is provided for context.]\n\n"
                    f"{first_msg.get('content', '')}"
                ),
            })
            for m in msgs:
                # Skip messages with missing or empty content
                c = m.get("content")
                if c is None or c == [] or c == "" or "content" not in m:
                    continue
                # Ensure alternation: if we'd have same role twice, insert bridge
                if result and result[-1].get("role") == m.get("role"):
                    bridge_role = "assistant" if m.get("role") == "user" else "user"
                    result.append({"role": bridge_role, "content": "(continued)"})
                result.append(m)
            # Must end with user message (we'll append the compaction prompt)
            if result[-1].get("role") != "user":
                result.append({"role": "user", "content": "(end of this half)"})
            return result

        def _append_prompt(msgs: list[dict[str, Any]], which: str) -> None:
            """Append the compaction prompt, merging if last msg is also user."""
            prompt = self._SPLIT_COMPACTION_PROMPT.format(which_half=which)
            if msgs and msgs[-1].get("role") == "user":
                last_content = msgs[-1].get("content", "")
                if isinstance(last_content, str):
                    msgs[-1]["content"] = last_content + "\n\n" + prompt
                elif isinstance(last_content, list):
                    # Don't corrupt list content — append prompt as a text block
                    msgs[-1]["content"] = last_content + [{"type": "text", "text": "\n\n" + prompt}]
                else:
                    msgs.append({"role": "user", "content": prompt})
            else:
                msgs.append({"role": "user", "content": prompt})

        # Summarize first half
        half1_msgs = _prepare_half(first_half, "FIRST HALF")
        half1_msgs = self._sanitize_messages(half1_msgs)
        _append_prompt(half1_msgs, "FIRST HALF")
        summary_1 = self._compaction_with_fallback(half1_msgs, "FIRST HALF", len(first_half))

        # Summarize second half
        half2_msgs = _prepare_half(second_half, "SECOND HALF")
        half2_msgs = self._sanitize_messages(half2_msgs)
        _append_prompt(half2_msgs, "SECOND HALF")
        summary_2 = self._compaction_with_fallback(half2_msgs, "SECOND HALF", len(second_half))

        if not summary_1 and not summary_2:
            logger.error("Split compaction: both halves failed, giving up.")
            return False

        # Merge: original question + both summaries
        merged_summary = (
            f"[CONTEXT COMPACTED — Split-then-merge summary of prior conversation]\n\n"
            f"## Summary of earlier work (first half)\n\n"
            f"{summary_1}\n\n"
            f"## Summary of recent work (second half)\n\n"
            f"{summary_2}"
        )

        self._apply_compacted_summary(merged_summary, old_count)
        return True

    def _apply_compacted_summary(self, summary_text: str, old_count: int) -> None:
        """Replace message history with a compacted summary."""
        self.messages.clear()
        self._append_msg({
            "role": "user",
            "content": (
                "[CONTEXT COMPACTED — Summary of prior conversation]\n\n"
                f"{summary_text}\n\n"
                "[END OF SUMMARY — Continue the task from where we left off.]"
            ),
        })
        self._last_response_id = None
        self._msg_checkpoint = 0
        logger.info(
            "Context compacted: %d messages → 1 summary message.", old_count,
        )

    @staticmethod
    def _sanitize_messages(
        msgs: list[dict[str, Any]],
        *,
        is_chaining_delta: bool = False,
    ) -> list[dict[str, Any]]:
        """Remove empty-content messages and fix role alternation.

        When *is_chaining_delta* is True the messages are a partial slice
        sent alongside ``previous_response_id`` — the server holds earlier
        assistant tool_use blocks.  Pass 4 (orphaned tool_result removal)
        is skipped because tool_results in the delta may legitimately
        reference tool_use IDs that live only in the chained context.

        Returns a new list.  Messages with ``content`` that is ``None``,
        ``[]``, or ``""`` are dropped (these arise from the mutable-list
        trick when an exception interrupts tool execution).  Consecutive
        messages with the same role are de-duplicated by keeping the later
        one.
        """
        sanitized: list[dict[str, Any]] = []
        for m in msgs:
            content = m.get("content")
            if content is None or content == [] or content == "" or "content" not in m:
                logger.warning(
                    "Dropping message with empty/missing content (role=%s)",
                    m.get("role", "?"),
                )
                continue
            # Strip empty text blocks from list content — these cause
            # API 400 "text content blocks must be non-empty" after
            # compaction leaves behind {"type": "text", "text": ""}.
            if isinstance(content, list):
                cleaned = [
                    b for b in content
                    if not (isinstance(b, dict) and b.get("type") == "text"
                            and not b.get("text"))
                ]
                if not cleaned:
                    logger.warning(
                        "Dropping message with only empty text blocks (role=%s)",
                        m.get("role", "?"),
                    )
                    continue
                if len(cleaned) != len(content):
                    m = {**m, "content": cleaned}
            sanitized.append(m)

        if len(sanitized) >= 2:
            deduped: list[dict[str, Any]] = [sanitized[0]]
            for m in sanitized[1:]:
                if m.get("role") == deduped[-1].get("role"):
                    prev = deduped[-1]
                    prev_content = prev.get("content", "")
                    new_content = m.get("content", "")

                    # Only merge when BOTH are plain strings (e.g. BBS injections).
                    # Never touch messages with list content (tool_use/tool_result
                    # blocks) — dropping or replacing them breaks the required
                    # tool_use↔tool_result pairing and causes API 400 errors.
                    if isinstance(prev_content, str) and isinstance(new_content, str):
                        prev["content"] = prev_content + "\n\n" + new_content
                        logger.debug(
                            "Merged consecutive %s messages to fix alternation",
                            m.get("role", "?"),
                        )
                    else:
                        # Insert a bridge message of the opposite role to
                        # maintain alternation without losing either message.
                        bridge_role = "assistant" if m.get("role") == "user" else "user"
                        deduped.append({"role": bridge_role, "content": "(continued)"})
                        deduped.append(m)
                else:
                    deduped.append(m)
            sanitized = deduped

        # --- Pass 3: repair orphaned tool_use ↔ tool_result pairs ---------------
        # The Anthropic API requires that every tool_use block in an assistant
        # message has a corresponding tool_result in the immediately following
        # user message.  Missing results cause 400 errors.  We inject stub
        # tool_results for any orphaned IDs.
        repaired: list[dict[str, Any]] = []
        for i, m in enumerate(sanitized):
            repaired.append(m)
            if m.get("role") != "assistant":
                continue
            content = m.get("content", [])
            if not isinstance(content, list):
                continue
            tool_use_ids = [
                b["id"] for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use" and "id" in b
            ]
            if not tool_use_ids:
                continue
            # Check next message for matching tool_results
            next_msg = sanitized[i + 1] if i + 1 < len(sanitized) else None
            existing_result_ids: set[str] = set()
            if next_msg and next_msg.get("role") == "user":
                next_content = next_msg.get("content", [])
                if isinstance(next_content, list):
                    existing_result_ids = {
                        b.get("tool_use_id", "")
                        for b in next_content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
            missing_ids = [tid for tid in tool_use_ids if tid not in existing_result_ids]
            if not missing_ids:
                continue
            # Inject stub tool_results
            if next_msg and next_msg.get("role") == "user" and isinstance(next_msg.get("content"), list):
                for tid in missing_ids:
                    next_msg["content"].append({
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": "(result unavailable — tool execution was interrupted)",
                        "is_error": True,
                    })
            else:
                # No user message follows — insert one with all stubs
                stub_blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": "(result unavailable — tool execution was interrupted)",
                        "is_error": True,
                    }
                    for tid in missing_ids
                ]
                repaired.append({"role": "user", "content": stub_blocks})
            logger.debug("Injected %d stub tool_results for orphaned tool_use IDs", len(missing_ids))
        sanitized = repaired

        # --- Pass 4: remove orphaned tool_result blocks -------------------------
        # The reverse problem: a user message contains tool_result blocks whose
        # tool_use_id does not appear in any preceding assistant message.  This
        # happens when messages are split for compaction and the tool_use ended
        # up in the other half.  The API rejects these with:
        #   "unexpected tool_use_id found in tool_result blocks"
        #
        # When chaining (previous_response_id), the delta may contain
        # tool_results whose tool_use lives in the server-side chained
        # context.  Skip Pass 4 entirely for chaining deltas.
        if is_chaining_delta:
            return sanitized

        has_assistant = any(m.get("role") == "assistant" for m in sanitized)
        if not has_assistant:
            has_tool_results = any(
                isinstance(m.get("content"), list)
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in (m.get("content") or [])
                )
                for m in sanitized
            )
            if has_tool_results:
                return sanitized

        all_tool_use_ids: set[str] = set()
        cleaned: list[dict[str, Any]] = []
        for m in sanitized:
            if m.get("role") == "assistant":
                content = m.get("content", [])
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_use" and "id" in b:
                            all_tool_use_ids.add(b["id"])
                cleaned.append(m)
            elif m.get("role") == "user":
                content = m.get("content", [])
                if isinstance(content, list):
                    filtered_blocks = [
                        b for b in content
                        if not (isinstance(b, dict)
                                and b.get("type") == "tool_result"
                                and b.get("tool_use_id") not in all_tool_use_ids)
                    ]
                    orphaned_count = len(content) - len(filtered_blocks)
                    if orphaned_count > 0:
                        logger.debug(
                            "Removed %d orphaned tool_result blocks (no matching tool_use)",
                            orphaned_count,
                        )
                    if filtered_blocks:
                        cleaned.append({**m, "content": filtered_blocks})
                    elif orphaned_count > 0:
                        # All blocks were orphaned tool_results — drop the message
                        pass
                    else:
                        cleaned.append(m)
                else:
                    cleaned.append(m)
            else:
                cleaned.append(m)
        sanitized = cleaned

        return sanitized

    def _msgs_for_llm(self) -> list[dict[str, Any]]:
        """Return the message slice to send to the LLM.

        When chaining via ``previous_response_id``, only messages added
        after the last stored response need to be sent.

        In web-search mode, includes a sanitization pass to remove messages
        with missing or empty ``content`` fields that would cause API 400
        errors (these can arise from interrupted tool executions during long
        browsing sessions).

        Internal metadata keys (prefixed with ``_``, e.g. ``_timestamp``)
        are stripped so they are never sent to the LLM API.  The ``metadata``
        key carried on ``tool_result`` blocks is dropped at the LLM-client
        boundary (see ``llm_client._strip_metadata``).
        """
        chaining = bool(self._last_response_id)
        msgs = self.messages[self._msg_checkpoint:] if chaining else self.messages
        cleaned = [{k: v for k, v in m.items() if not k.startswith("_")} for m in msgs]
        if getattr(self.config, "collapse_duplicate_tool_history", False):
            cleaned = self._collapse_duplicate_tool_results(cleaned)
        if self.config.has_web_search_capability():
            return self._sanitize_messages(cleaned, is_chaining_delta=chaining)
        return cleaned

    def _dup_signature(self, name: str, inp: dict[str, Any]) -> str | None:
        """Stable dedup key for a tool call, or None if the tool isn't deduped.

        Reuses the existing normalizers so two effectively-identical calls map to
        the same key: web_search query (``_normalize_query``), web_fetch URL and
        pdf_read source (``content_cache._normalize_url``). Imported locally to
        avoid any import-cycle at module load.
        """
        if not isinstance(inp, dict):
            return None
        try:
            if name == "web_search":
                from arcticswarm.tools.web_search import WebSearchTool
                q = (inp.get("query") or "").strip()
                return f"web_search:{WebSearchTool._normalize_query(q)}" if q else None
            if name == "web_fetch":
                from arcticswarm.tools.content_cache import _normalize_url
                u = (inp.get("url") or "").strip()
                return f"web_fetch:{_normalize_url(u)}" if u else None
            if name == "pdf_read":
                from arcticswarm.tools.content_cache import _normalize_url
                s = (inp.get("source") or "").strip()
                if not s:
                    return None
                pages = (inp.get("pages") or "").strip()
                return f"pdf_read:{_normalize_url(s)}|pages={pages}"
        except Exception:
            return None
        return None

    def _collapse_duplicate_tool_results(
        self, msgs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Collapse duplicate tool-call RESULTS in an outbound message slice.

        For each dedup signature (web_search/web_fetch/pdf_read) issued more than
        ``dup_history_keep_last`` times, the bulky ``content`` of the EARLIER
        ``tool_result`` blocks is replaced with a compact stub; the last N are
        kept in full. Tool_use blocks (tiny) are left intact, so the
        tool_use<->tool_result pairing and role-alternation invariants are
        trivially preserved (no ``_sanitize_messages`` dependency). Operates on a
        rebuilt copy — never mutates ``self.messages`` or the shared block dicts.

        Only single-``tool_use`` assistant messages are considered (our runs use
        ``max_tool_calls_per_turn=1``); batched multi-tool_use messages are left
        untouched for safety.
        """
        keep_last = max(1, int(getattr(self.config, "dup_history_keep_last", 1)))
        # 1) tool_use_id -> dedup signature
        id2sig: dict[str, str] = {}
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            if len(uses) != 1:  # skip batched / non-tool assistant messages
                continue
            b = uses[0]
            sig = self._dup_signature(b.get("name", ""), b.get("input") or {})
            if sig and b.get("id"):
                id2sig[b["id"]] = sig
        if not id2sig:
            return msgs
        # 2) ordered tool_result occurrences per signature
        occ: dict[str, list[tuple[int, int]]] = {}
        for mi, m in enumerate(msgs):
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for bi, b in enumerate(content):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    sig = id2sig.get(b.get("tool_use_id"))
                    if sig:
                        occ.setdefault(sig, []).append((mi, bi))
        # 3) mark all-but-last-N occurrences of each over-repeated signature
        to_stub: dict[int, set[int]] = {}
        for lst in occ.values():
            if len(lst) <= keep_last:
                continue
            for (mi, bi) in lst[:-keep_last]:
                to_stub.setdefault(mi, set()).add(bi)
        if not to_stub:
            return msgs
        # 4) rebuild only the affected messages (don't mutate shared block dicts)
        out: list[dict[str, Any]] = []
        for mi, m in enumerate(msgs):
            if mi not in to_stub:
                out.append(m)
                continue
            new_content = []
            for bi, b in enumerate(m.get("content", [])):
                if bi in to_stub[mi] and isinstance(b, dict):
                    tool = id2sig.get(b.get("tool_use_id"), "").split(":", 1)[0] or "tool"
                    stub = (
                        f"[duplicate {tool} call collapsed to save context — an "
                        f"identical call is issued again later in the conversation; "
                        f"this earlier result was omitted]"
                    )
                    new_content.append({**b, "content": stub})
                    self._dup_collapsed_count += 1
                else:
                    new_content.append(b)
            out.append({**m, "content": new_content})
        return out
