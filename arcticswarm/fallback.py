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

"""Fallback / resilience layer for the agent loop.

Consolidates the model-level fallback and error-classification helpers used by
:class:`arcticswarm.agent.Agent`:

  * Refusal / empty-response detection (safety refusals, Azure prompt-shield
    stubs, thinking-only responses, visible-content checks).
  * Transient-error classification + ``Retry-After`` extraction for the
    retry/backoff loop.
  * Context-overflow / max-tokens-overflow detection for the reactive
    compaction path.
  * The reduced-reasoning-effort ladder used by the empty-response fallback.

These are pure helpers (no ``Agent`` state) so they can be unit-tested and
reused without constructing an Agent.  The stateful orchestrators that *use*
them (the empty-response fallback ladder, context compaction) live on
``Agent`` and call into these functions.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from arcticswarm.llm_client import _VLLM_THINKING_OFF_EFFORTS, detect_provider
from arcticswarm.logging_utils import (
    log_empty_fallback,
    log_empty_fallback_for_agent,
)

if TYPE_CHECKING:
    from arcticswarm.llm_client import LLMResponse


# ---------------------------------------------------------------------------
# Refusal / filter detection patterns
# ---------------------------------------------------------------------------

# Patterns indicating the LLM refused to answer on safety/ethics grounds.
# Used to trigger fallback just like empty-response fallback.
SAFETY_REFUSAL_RE = re.compile(
    r"\b(?:I\s+(?:can'?t|won'?t|cannot|will not)\s+(?:research|provide|help|assist|answer|engage|proceed)"
    r"|ethical\s+(?:concern|grounds|reason)"
    r"|safety\s+(?:refusal|concern|reason|filter)"
    r"|I'm\s+(?:stopping|not\s+proceeding))\b",
    re.IGNORECASE,
)

# Canonical Azure prompt-shield refusal stub.  Azure injects this text as a
# normal HTTP 200 response (stop_reason=stop, ~1-5 output tokens) when its
# content filter blocks input or output for the deployment.  We treat hits as
# "filter, not model" — i.e. retry the same prompt through the non-Azure
# (Cortex) proxy where the filter is absent.  Anchored to the start of the
# response text so genuine model-trained refusals that *contain* the phrase
# don't trigger the cross-deployment fallback.
AZURE_FILTER_REFUSAL_RE = re.compile(
    r"^\s*I'?m sorry,?\s+but I (?:cannot|can't|won't) assist with that request\.?\s*$",
    re.IGNORECASE,
)

# Azure deployment name -> Cortex /openai model name.
# Cortex's OpenAI proxy registers the canonical model name (e.g. ``gpt-5.4``)
# while Azure exposes deployment slots that often carry a suffix indicating
# the reasoning variant (e.g. ``gpt-5.4-re`` for the reasoning-enabled slot).
# The cross-deployment fallback (Azure prompt shield -> Cortex /openai) must
# remap any such Azure-only deployment name to a name Cortex recognises,
# otherwise Cortex returns a 400 ``unknown model``.  Keep this map tiny and
# explicit: only models that genuinely differ between Azure and Cortex.
AZURE_TO_CORTEX_MODEL_ALIASES = {
    "gpt-5.4-re": "gpt-5.4",
}

# Reduced-reasoning-effort ladder for the empty-response fallback (Step 1):
# step the current effort down one rung before dropping reasoning entirely.
REDUCED_REASONING = {
    "max": "high",     # adaptive max → high
    "xhigh": "high",   # 64K → 32K
    "high": "medium",  # 32K → 16K
    "medium": "low",   # 16K → 5K
}


# ---------------------------------------------------------------------------
# Refusal / empty-response detection
# ---------------------------------------------------------------------------


def has_visible_response_content(response: "LLMResponse") -> bool:
    """Return True when response contains actionable visible content.

    ``reasoning``-only blocks are not actionable for the orchestrator; they
    should be treated the same as an empty response so fallback/retry logic
    can engage.
    """
    blocks = response.content_blocks or []
    return any(b.get("type") in ("text", "tool_use") for b in blocks)


def is_safety_refusal(response: "LLMResponse") -> bool:
    """Return True if the response is a safety/ethics refusal.

    A safety refusal produces non-empty content (the refusal text) but no
    tool calls.  We detect it by pattern-matching the text and trigger the
    same fallback path as empty responses.
    """
    if not response.content_blocks:
        return False
    has_tool_use = any(b.get("type") == "tool_use" for b in response.content_blocks)
    if has_tool_use:
        return False
    text = " ".join(
        b.get("text", "") for b in response.content_blocks
        if b.get("type") == "text"
    )
    return bool(SAFETY_REFUSAL_RE.search(text))


def is_refusal_text(text: str) -> bool:
    """Return True if ``text`` matches the safety/ethics refusal regex.

    String counterpart of :func:`is_safety_refusal` (e.g. for a final
    ``response_text`` after the agent loop ends).
    """
    if not text:
        return False
    return bool(SAFETY_REFUSAL_RE.search(text))


def is_no_actionable_content(response: "LLMResponse") -> bool:
    """True if blocks exist but none are actionable text or tool_use.

    Specifically: only thinking / redacted_thinking blocks (no text, no
    tool_use, or only empty text).  This is the failure mode where
    adaptive-thinking-enabled opus consumes the full ``max_tokens`` budget on
    a thinking block and emits nothing actionable.  Defensive backstop for
    ``disable_extended_thinking_by_model``.
    """
    if not response.content_blocks:
        # An empty response is already handled upstream — return False here so
        # we don't double-count.
        return False
    for block in response.content_blocks:
        t = block.get("type")
        if t == "text" and (block.get("text") or "").strip():
            return False
        if t == "tool_use":
            return False
    # Only thinking / redacted_thinking blocks survived (or all-empty text).
    return True


def is_azure_filter_refusal(response: "LLMResponse") -> bool:
    """Return True if the response looks like Azure's prompt-shield stub.

    Azure injects ``"I'm sorry, but I cannot assist with that request."`` as
    the entire response text (stop_reason=stop, output_tokens ≈ 1-5) when its
    content filter blocks a call.  Distinct from a model-trained refusal.
    Anchored match — we don't want to trip on the phrase appearing inside a
    longer model-authored response.
    """
    if not response.content_blocks:
        return False
    if any(b.get("type") == "tool_use" for b in response.content_blocks):
        return False
    text = "".join(
        b.get("text", "") for b in response.content_blocks
        if b.get("type") == "text"
    )
    return bool(AZURE_FILTER_REFUSAL_RE.match(text.strip()))


# ---------------------------------------------------------------------------
# Transient-error classification + retry backoff
# ---------------------------------------------------------------------------


def retry_after_seconds(exc: Exception) -> float | None:
    """Return the server-suggested retry delay (seconds) for a 429, if any.

    Azure OpenAI / OpenAI return ``Retry-After`` (seconds) on rate-limit
    responses.  Honoring it is critical: Azure's quota window is typically
    60 s, so a fixed exponential schedule starting at <1 s spends all retries
    inside the same throttled bucket.  Returns ``None`` when no usable hint is
    available.
    """
    try:
        import openai
    except ImportError:
        return None
    if not isinstance(exc, openai.APIStatusError):
        return None
    if getattr(exc, "status_code", None) != 429:
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        val = headers.get(key) if hasattr(headers, "get") else None
        if not val:
            continue
        try:
            return max(0.0, float(val))
        except (TypeError, ValueError):
            continue
    return None


def is_retryable(exc: Exception) -> bool:
    """Return True for transient failures worth retrying."""
    try:
        import openai
        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code in (429, 500, 502, 503, 529):
                return True
            # Azure OpenAI reasoning-chain corruption — retryable after
            # clearing previous_response_id.
            if exc.status_code == 400 and "reasoning" in str(exc):
                return True
            # OpenAI/Azure Responses API: stored response evicted, never
            # persisted, or chain pointer stale/missing. Retryable after
            # clearing previous_response_id and resending the full
            # conversation.
            exc_text_lower = str(exc).lower()
            if exc.status_code == 400 and (
                "previous_response_not_found" in exc_text_lower
                or "previous_response_id" in exc_text_lower
                or "previous response with id" in exc_text_lower
                or ("item with id" in exc_text_lower and "not found" in exc_text_lower)
            ):
                return True
            # Anthropic: model generated malformed tool-call JSON —
            # stochastic, succeeds on retry.
            if exc.status_code == 400 and "unable to parse tool parameter" in str(exc).lower():
                return True
    except ImportError:
        pass

    text = str(exc)
    # OpenAI streaming: 429 raised as bare APIError (not APIStatusError)
    if "too many requests" in text.lower():
        return True
    # Proxy idle-timeout (500)
    if "Idle timeout expired" in text or "messageCode=370001" in text:
        return True
    # Transient 403 — occasionally fires under load but succeeds on retry
    if "403" in text and "Not authorized" in text:
        return True
    # HTTP read timeout from Azure OpenAI / httpcore
    if "read operation timed out" in text.lower():
        return True
    # Anthropic: model produced malformed tool-parameter JSON
    if "unable to parse tool parameter" in text.lower():
        return True
    # OpenAI/Azure Responses API chain pointer stale/missing (bare APIError,
    # typical for streaming).
    if "previous_response_not_found" in text.lower():
        return True
    if "previous response with id" in text.lower() and "not found" in text.lower():
        return True
    if "item with id" in text.lower() and "not found" in text.lower():
        return True
    # Stale socket in httpx connection pool under heavy concurrent load.
    if "Bad file descriptor" in text:
        return True
    # Peer dropped the connection mid-stream (overloaded proxy/server).
    if "incomplete chunked read" in text or "peer closed connection" in text:
        return True
    # SSL transport failure — transient under high concurrency.
    if "record layer failure" in text:
        return True
    # Upstream proxy 500s — typically transient (one-shot upstream blip).
    if "Error code: 500" in text and "internal error" in text.lower():
        return True
    return False


# ---------------------------------------------------------------------------
# Context-overflow detection (reactive compaction triggers)
# ---------------------------------------------------------------------------


def is_context_too_long(exc: Exception) -> bool:
    """Return True if the error indicates the prompt exceeded context limits."""
    text = str(exc).lower()
    # Anthropic: "prompt is too long: ..."
    if "prompt is too long" in text:
        return True
    # Anthropic: "... exceeds the maximum number of tokens"
    if "exceeds the maximum number of tokens" in text:
        return True
    # Anthropic legacy enforcement: "max tokens of 200000 exceeded"
    # (input_tokens + max_tokens > model context window)
    if "max tokens of" in text and "exceeded" in text:
        return True
    # Anthropic max_tokens-parameter cap: "the maximum tokens you requested
    # exceeds the model limit". Compaction alone won't fix this — caller
    # should fall back to reduced thinking budget — but we still classify it
    # so the retry path engages instead of failing fatally.
    if "exceeds the model limit" in text:
        return True
    # OpenAI: "maximum context length" / "context_length_exceeded"
    if "maximum context length" in text or "context_length_exceeded" in text:
        return True
    # Azure GPT-5 (Responses API) wording — distinct from the classic OpenAI
    # "maximum context length" string. Observed: "Your input exceeds the
    # context window of this model."
    if "exceeds the context window" in text:
        return True
    return False


def is_max_tokens_overflow(exc: Exception) -> bool:
    """Return True if the error is specifically about ``max_tokens`` being
    too large (input fits, but ``max_tokens`` parameter or ``input +
    max_tokens`` exceeds a backend cap).

    These cannot be fixed by compacting input alone — the recovery path is to
    reduce the thinking budget (or disable thinking entirely).
    """
    text = str(exc).lower()
    if "max tokens of" in text and "exceeded" in text:
        return True
    if "exceeds the model limit" in text:
        return True
    return False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RepeatedEmptyFallbackError(RuntimeError):
    """Raised by :meth:`Agent._fallback_on_empty_response` after the
    same agent instance has burned ``_MAX_CONSECUTIVE_EMPTY_FALLBACKS``
    consecutive empty-fallback firings on the same case.

    Caller is expected to catch this, log a final BBS marker, and end
    the case rather than loop forever on a model-side refusal.
    """


# ---------------------------------------------------------------------------
# Empty-response fallback mixin (stateful orchestration)
# ---------------------------------------------------------------------------


class FallbackMixin:
    """Empty-response / refusal recovery, mixed into :class:`arcticswarm.agent.Agent`.

    The stateful orchestrators of the empty-response fallback ladder
    (reduced-reasoning retries, the Azure->Cortex reroute, the cross-model
    fallback target, and per-case logging). They lean on the pure helpers in
    this module (refusal/overflow detection) and on ``Agent`` state via
    ``self``; the fallback tuning constants remain defined on ``Agent``.
    """

    def _fallback_on_empty_response(
        self,
        *,
        on_text_delta: Any | None = None,
        on_tool_input_delta: Any | None = None,
        streaming: bool = True,
        primary_raw_event_log: list[dict[str, Any]] | None = None,
        primary_stop_reason: str = "",
        primary_usage: dict[str, Any] | None = None,
        is_azure_refusal: bool = False,
    ) -> "LLMResponse":
        """Retry the current prompt when the primary model returns empty content.

        Fallback chain:
        0. (only when ``is_azure_refusal=True`` and primary uses Azure)
           Same primary model, but routed through the **non-Azure Cortex
           proxy** (chat completions).  Azure's prompt shield silently
           injects a canned refusal for sensitive content; the same call
           through Cortex bypasses the Azure-deployment filter entirely.
           Quality on Cortex is lower (chat completions vs. Responses API)
           but a worse answer beats no answer.
        1. If reasoning_effort was set, retry the **same primary model** with
           reasoning disabled (thinking blocks stripped from history).
        2. Fall back to Sonnet 4 (thinking blocks stripped).

        The caller is responsible for replacing the empty response in the
        conversation history.
        """
        # Bail out after 3 consecutive empty-fallback firings on the
        # same case to prevent pathological retry loops on questions the
        # model truly refuses (e.g. browsecomp_1126's refusal cascade ×
        # however many subagents triggered it).  Counter is reset to 0
        # on any non-empty response (handled at receive sites).
        self.consecutive_empty_fallbacks += 1
        if self.consecutive_empty_fallbacks >= self._MAX_CONSECUTIVE_EMPTY_FALLBACKS:
            logger.error(
                "Repeated empty-response fallback (%d in a row) — bailing "
                "out to avoid burning the rest of the case.  Caller should "
                "treat as terminal and write best-effort to BBS.",
                self.consecutive_empty_fallbacks,
            )
            raise RepeatedEmptyFallbackError(
                f"{self.consecutive_empty_fallbacks} consecutive empty-response "
                f"fallbacks; primary_stop_reason={primary_stop_reason!r}"
            )

        # Build ordered fallback list.  The nudge retry (inject user message
        # and retry same model+reasoning) happens *before* this function is
        # called, in the streaming/non-streaming turn loop.  By the time we
        # get here, the nudge already failed.
        #
        # Fallback chain:
        #   0. (Azure-refusal only) Same model via non-Azure Cortex proxy
        #   1. Same model, reduced thinking budget (if reasoning was on)
        #   2. Same model, no reasoning at all
        #   3. Different model (Sonnet 4), no reasoning
        _REDUCED_REASONING = {
            "max": "high",     # adaptive max → high
            "xhigh": "high",   # 64K → 32K
            "high": "medium",  # 32K → 16K
            "medium": "low",   # 16K → 5K
        }
        fallback_specs: list[dict[str, Any]] = []
        # Step 0: Cortex-proxy retry, only when the trigger was an Azure
        # prompt-shield refusal AND the primary path is Azure.  Same model,
        # same reasoning_effort, just routed through the non-Azure client so
        # the deployment-side filter doesn't fire again.
        if is_azure_refusal and getattr(self.config, "use_azure_openai", False):
            # Remap Azure deployment names (e.g. ``gpt-5.4-re``) to the
            # canonical Cortex name (e.g. ``gpt-5.4``) so the non-Azure
            # proxy actually recognises the model.  Models not in the
            # alias table pass through unchanged.
            cortex_model = AZURE_TO_CORTEX_MODEL_ALIASES.get(
                self.config.model, self.config.model
            )
            fallback_specs.append({
                "model": cortex_model,
                "reasoning_effort": self.config.reasoning_effort,
                "label": f"{cortex_model} (cortex-proxy, non-azure)",
                "force_non_azure": True,
            })
        # Step 0.5: on a model-side refusal, prepend a same-model
        # ``reasoning=low`` rung.  The opus refusal cascade in
        # ``empty_fallback/case_001`` (browsecomp_450, recovered) shows
        # opus-low succeeds when opus-medium silently empties.  The
        # opus-1126 refusal cascade tried medium → no-reasoning → sonnet
        # without ever trying low and lost the case.  Putting low at the
        # head gives refusal cases the same recovery rung that empty
        # cases have already proved works.
        if (
            primary_stop_reason == "refusal"
            and self.config.reasoning_effort
            and self.config.reasoning_effort != "low"
        ):
            fallback_specs.append({
                "model": self.config.model,
                "reasoning_effort": "low",
                "label": f"{self.config.model} (reasoning=low, refusal-recovery)",
            })
        if self.config.reasoning_effort:
            # Step 1: reduced thinking budget
            reduced = _REDUCED_REASONING.get(self.config.reasoning_effort)
            # vLLM models gate thinking with a BINARY chat-template toggle
            # (enable_thinking), not a token budget. Reducing e.g. xhigh->high
            # leaves thinking ON, so the retry re-issues the identical
            # think-only turn that just emptied (in practice reasoning=high
            # recovered no empties on such vLLM models, while the
            # no-reasoning rung recovered all of them). Skip the reduced rung
            # for vLLM unless the reduced effort actually crosses the
            # thinking-off threshold, so we go straight to no-reasoning.
            if (
                reduced
                and detect_provider(self.config.model) == "vllm"
                and reduced not in _VLLM_THINKING_OFF_EFFORTS
            ):
                reduced = None
            if reduced:
                fallback_specs.append({
                    "model": self.config.model,
                    "reasoning_effort": reduced,
                    "label": f"{self.config.model} (reasoning={reduced})",
                })
            # Step 2: no reasoning
            fallback_specs.append({
                "model": self.config.model,
                "reasoning_effort": None,
                "label": f"{self.config.model} (no reasoning)",
            })
        # Step 3: different (closed) model — skipped when closed-model calls
        # are forbidden for this run (e.g. self-hosted vLLM/Qwen). The
        # same-model reduced-reasoning rungs above stay as the recovery path.
        if not getattr(self.config, "disable_closed_model_fallback", False):
            fallback_specs.append({
                "model": self._EMPTY_RESPONSE_FALLBACK_MODEL,
                "reasoning_effort": None,
                "label": self._EMPTY_RESPONSE_FALLBACK_MODEL,
            })

        # Full history for fallback — the fallback model has no prior context
        # (no previous_response_id), so it needs the complete conversation,
        # not just the chaining delta that _msgs_for_llm() returns.
        full_msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in self.messages]
        if self.config.has_web_search_capability():
            full_msgs = self._sanitize_messages(full_msgs)

        # Sonnet 4 has a 200K input context.  When the primary model
        # accumulated more (e.g. GPT-5.4 / Sonnet 4.6 in 1M mode) we trim
        # the message history to "first 50K tokens + sentinel + last 50K
        # tokens" so the fallback fits its window.  Trim is non-destructive
        # — we only mutate the local ``full_msgs`` copy, not ``self.messages``.
        # Use the higher of last_input_tokens and peak — last_input_tokens
        # may be stale (updated after fallback check) or zero on the first turn.
        estimated_tokens = max(
            self._context_budget.last_input_tokens,
            self._context_budget.peak_input_tokens,
        )
        if estimated_tokens == 0:
            estimated_tokens = sum(
                self._estimate_msg_tokens(m) for m in full_msgs
            )
        if estimated_tokens > self._FALLBACK_TRIM_TRIGGER_TOKENS:
            logger.info(
                "Context too large for Sonnet 4 fallback (%d tokens > %d) "
                "— trimming to first %d + sentinel + last %d tokens.",
                estimated_tokens,
                self._FALLBACK_TRIM_TRIGGER_TOKENS,
                self._FALLBACK_TRIM_HEAD_TOKENS,
                self._FALLBACK_TRIM_TAIL_TOKENS,
            )
            full_msgs = self._trim_messages_for_sonnet4_fallback(full_msgs)
            full_msgs = self._sanitize_messages(full_msgs)

        # ``last_user_text`` is only used for per-case log triage, so it's
        # fine to compute it from the pre-provider-filter history.
        last_user_text = ""
        for msg in reversed(full_msgs):
            role = msg.get("role", "")
            if role in ("user", "tool"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_text = content
                elif isinstance(content, list):
                    last_user_text = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if last_user_text.strip():
                    break

        fallback_attempts: list[dict] = []

        for spec in fallback_specs:
            fallback_model = spec["model"]
            fallback_re = spec["reasoning_effort"]
            label = spec["label"]

            # Skip combos that have permanently failed earlier in this
            # process with "unknown model" / 404.
            combo_key = (fallback_model, fallback_re)
            if combo_key in self._DEAD_FALLBACK_COMBOS:
                logger.info(
                    "Skipping fallback %s — combo previously returned "
                    "'unknown model' / 404 in this process.",
                    label,
                )
                continue

            logger.warning(
                "Primary model returned empty content_blocks. "
                "Trying fallback: %s for this turn.",
                label,
            )

            # When a spec carries force_non_azure (Cortex-proxy retry rung),
            # override the agent's Azure config so the call routes through
            # the standard OpenAI client instead.
            spec_use_azure = (
                False if spec.get("force_non_azure")
                else getattr(self.config, "use_azure_openai", False)
            )
            # The default cross-model fallback target is Sonnet 4 (200K) —
            # we trim ``full_msgs`` above so the input fits without the 1M
            # beta header.  Only opt in to 1M when an explicitly-configured
            # fallback model still requires it (Sonnet 4.6 / Opus 4.6
            # aliases on the Cortex proxy).
            fallback_enable_1m = (
                fallback_model.startswith("claude-sonnet-4-6")
                or fallback_model.startswith("claude-opus-4-6")
            )

            # Strip provider-incompatible blocks (e.g. OpenAI ``reasoning``
            # items when falling back to Anthropic) and drop any messages
            # that become empty after filtering.  Sending ``reasoning``
            # blocks to Anthropic produces a schema-level 400:
            # ``messages.N.content: Field required``.
            fallback_provider = detect_provider(fallback_model)
            input_messages = self._messages_for_provider(full_msgs, fallback_provider, is_fallback=True)

            fallback_client = self._make_llm_client(
                model=fallback_model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                openai_base_url=getattr(self.config, "openai_base_url", ""),
                openai_api_key=getattr(self.config, "openai_api_key", ""),
                use_azure_openai=spec_use_azure,
                azure_openai_api_key=getattr(self.config, "azure_openai_api_key", ""),
                azure_openai_endpoint=getattr(self.config, "azure_openai_endpoint", ""),
                azure_openai_api_version=getattr(self.config, "azure_openai_api_version", "2025-04-01-preview"),
                # Only Sonnet 4.6 / Opus 4.6 cross-model fallbacks need the
                # 1M-context beta header — the default Sonnet 4 fallback
                # uses the standard 200K window after trimming above.
                enable_1m_context_model=fallback_enable_1m,
            )
            # claude-4-sonnet has an 8192 max output token limit
            fallback_max_tokens = min(self.config.max_tokens, 8192)
            fallback_error: str | None = None
            fallback_response: LLMResponse | None = None
            try:
                if streaming:
                    fallback_response = fallback_client.call_streaming(
                        model=fallback_model,
                        max_tokens=fallback_max_tokens,
                        system_prompt=self.system_prompt,
                        tools=self._get_tool_definitions(),
                        messages=input_messages,
                        reasoning_effort=fallback_re,
                        on_text_delta=on_text_delta,
                        on_tool_input_delta=on_tool_input_delta,
                    )
                else:
                    fallback_response = fallback_client.call(
                        model=fallback_model,
                        max_tokens=fallback_max_tokens,
                        system_prompt=self.system_prompt,
                        tools=self._get_tool_definitions(),
                        messages=input_messages,
                        reasoning_effort=fallback_re,
                    )
            except Exception as exc:
                fallback_error = str(exc)
                # Mark this (model, reasoning_effort) dead for the rest of
                # the process if the error indicates the deployment is
                # missing — avoids repeating known-dead fallback rungs on
                # GPT-5.4 (Azure proxy doesn't deploy reasoning=medium / no
                # reasoning variants).
                err_lower = fallback_error.lower()
                if (
                    "unknown model" in err_lower
                    or "deployment" in err_lower and "not" in err_lower and "found" in err_lower
                    or " 404" in err_lower
                ):
                    self._DEAD_FALLBACK_COMBOS.add(combo_key)
                    logger.warning(
                        "Marking fallback combo %s as DEAD for this process "
                        "(error: %s).",
                        combo_key,
                        fallback_error[:200],
                    )
            finally:
                fallback_client.close()

            fallback_attempts.append({
                "model": label,
                "response": fallback_response,
                "error": fallback_error,
            })

            # If we got a non-empty response, return it
            if fallback_response and fallback_response.content_blocks:
                log_empty_fallback_for_agent(
                    self,
                    last_user_text=last_user_text,
                    num_messages=len(input_messages),
                    fallback_attempts=fallback_attempts,
                    streaming=streaming,
                    primary_raw_event_log=primary_raw_event_log,
                    primary_stop_reason=primary_stop_reason,
                    primary_usage=primary_usage,
                )
                # Reset chaining — fallback response is from a different
                # model/config so the old previous_response_id is invalid.
                self._last_response_id = None
                self._msg_checkpoint = 0
                return fallback_response

            # If an exception occurred, log and try the next model
            if fallback_error:
                logger.warning(
                    "Fallback to %s failed: %s. Trying next.",
                    label,
                    fallback_error,
                )
            else:
                logger.warning(
                    "Fallback to %s also returned empty content_blocks.",
                    label,
                )

        # All fallback models exhausted — log and raise.
        # Reset chaining state so the next call (if error is caught)
        # sends full context instead of stale delta.
        fallback_labels = [s["label"] for s in fallback_specs]
        self._last_response_id = None
        self._msg_checkpoint = 0
        log_empty_fallback_for_agent(
            self,
            last_user_text=last_user_text,
            num_messages=len(input_messages),
            fallback_attempts=fallback_attempts,
            streaming=streaming,
            primary_raw_event_log=primary_raw_event_log,
            primary_stop_reason=primary_stop_reason,
            primary_usage=primary_usage,
        )
        raise RuntimeError(
            f"All fallback models ({', '.join(fallback_labels)}) "
            "returned empty content or failed."
        )

    def _log_empty_fallback(
        self,
        *,
        last_user_text: str,
        num_messages: int,
        fallback_attempts: list[dict],
        streaming: bool,
    ) -> None:
        """Write a per-case log file for empty-response fallback events."""
        log_empty_fallback(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            total_llm_calls=self.total_llm_calls,
            system_prompt=self.system_prompt,
            input_messages=self._msgs_for_llm(),
            tool_definitions=self._get_tool_definitions(),
            last_user_text=last_user_text,
            num_messages=num_messages,
            fallback_attempts=fallback_attempts,
            streaming=streaming,
            output_dir=self.config.output_dir,
        )
