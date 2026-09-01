"""LLM client abstraction — supports Anthropic (Claude) and OpenAI (GPT).

Provides a unified interface so :class:`~arcticswarm.agent.Agent` can work
with either provider.  Messages are stored in an internal format based on
Anthropic's structure (content-block lists), and the OpenAI clients
convert at the API boundary.

Two OpenAI backends are available:
  - ``OpenAIResponsesLLMClient`` (default) — uses the Responses API which
    preserves reasoning items across tool-calling turns.
  - ``OpenAIChatLLMClient`` — legacy Chat Completions API fallback.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalised response
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Provider-agnostic response from an LLM call.

    ``content_blocks`` uses Anthropic-style dicts:
      - ``{"type": "text", "text": "..."}``
      - ``{"type": "tool_use", "id": "...", "name": "...", "input": {...}}``
    """

    content_blocks: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    response_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    # Diagnostic: raw streaming event summaries (only populated on empty responses)
    _raw_event_log: list[dict[str, Any]] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Abstract LLM client that Agent talks to."""

    @abstractmethod
    def call(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        force_json: bool = False,
    ) -> LLMResponse:
        """Synchronous (non-streaming) LLM call."""

    @abstractmethod
    def call_streaming(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        on_text_delta: Any | None = None,
        on_tool_input_delta: Any | None = None,
    ) -> LLMResponse:
        """Streaming LLM call.

        Callbacks are invoked during streaming:
          - ``on_text_delta(text: str)``
          - ``on_tool_input_delta(tool_name: str, tool_use_id: str, partial_json: str)``

        Returns the final aggregated :class:`LLMResponse`.
        """

    @abstractmethod
    def close(self) -> None:
        """Release SDK resources."""


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------

def _patch_httpx_client_del() -> None:
    """Suppress ``AttributeError: '_state'`` in SyncHttpxClientWrapper.__del__.

    Known anthropic SDK issue: the httpx wrapper can be GC'd before __init__
    fully sets ``_state``, causing a noisy but harmless traceback.  We wrap
    ``__del__`` once so it swallows the AttributeError silently.
    """
    try:
        from anthropic._base_client import SyncHttpxClientWrapper
    except ImportError:
        return
    orig_del = getattr(SyncHttpxClientWrapper, "__del__", None)
    if orig_del is None or getattr(orig_del, "_patched", False):
        return

    def _safe_del(self: Any) -> None:
        try:
            orig_del(self)
        except AttributeError:
            pass

    _safe_del._patched = True  # type: ignore[attr-defined]
    SyncHttpxClientWrapper.__del__ = _safe_del  # type: ignore[attr-defined]

_patch_httpx_client_del()


class AnthropicLLMClient(BaseLLMClient):
    """Wraps ``anthropic.Anthropic`` and normalises responses."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        enable_1m_context_model: bool = False,
        disable_extended_thinking: bool = False,
    ) -> None:
        import anthropic

        client_kwargs: dict[str, Any] = {}
        is_custom = base_url and base_url != "https://api.anthropic.com"
        if api_key:
            if is_custom:
                client_kwargs["auth_token"] = api_key
            else:
                client_kwargs["api_key"] = api_key
        if is_custom:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**client_kwargs)
        self._enable_1m_context_model = enable_1m_context_model
        self._disable_extended_thinking = disable_extended_thinking

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _strip_metadata(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove internal fields from tool_result blocks before sending.

        Strips 'metadata' and any underscore-prefixed keys (e.g.
        '_tool_duration_seconds') that are used for internal bookkeeping.
        """
        cleaned = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        block = {k: v for k, v in block.items() if k != "metadata" and not k.startswith("_")}
                    new_content.append(block)
                msg = {**msg, "content": new_content}
            cleaned.append(msg)
        return cleaned

    @staticmethod
    def _system_for_api(system_prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @staticmethod
    def _tools_for_api(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not tools:
            return tools
        out = list(tools)
        out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
        return out

    @staticmethod
    def _inject_cache_breakpoint(
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for msg in reversed(messages):
            if msg["role"] != "user":
                continue
            content = msg["content"]
            if isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                return {"message": msg, "original": content}
            if isinstance(content, list) and content:
                last_block = content[-1]
                if isinstance(last_block, dict) and "cache_control" not in last_block:
                    last_block["cache_control"] = {"type": "ephemeral"}
                    return {"message": msg, "block": last_block}
            break
        return None

    @staticmethod
    def _restore_cache_breakpoint(saved: dict[str, Any] | None) -> None:
        if saved is None:
            return
        if "original" in saved:
            saved["message"]["content"] = saved["original"]
        elif "block" in saved:
            saved["block"].pop("cache_control", None)

    # Budget tokens by effort level (for budget-based extended thinking).
    # ``"max"`` is the system-card vocabulary for adaptive-thinking models
    # (Opus 4.6+); on the budget-based path it aliases to ``"xhigh"`` (64K)
    # so a single YAML config can target both adaptive and budget paths.
    # Without this, ``reasoning_effort="max"`` silently fell back to the
    # 16K medium default — see CONSOLIDATED_CODE_TODOS.md Q1.
    _THINKING_BUDGETS: dict[str, int] = {
        "low": 5_000,
        "medium": 16_000,
        "high": 32_000,
        "xhigh": 64_000,
        "max": 64_000,
    }

    # Models that use adaptive thinking (effort-based) instead of budget-based.
    _ADAPTIVE_THINKING_MODELS: frozenset[str] = frozenset({
        "claude-opus-4-6",
        "claude-opus-4-6-20250501",
    })

    # Models that should NEVER use extended/adaptive thinking.
    _NO_THINKING_MODELS: frozenset[str] = frozenset({
    })

    @classmethod
    def _thinking_params(
        cls, reasoning_effort: str | None, model: str,
        *,
        disable_extended_thinking: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return ``(thinking_kwarg, extra_body)`` for the Anthropic Messages API.

        Two modes depending on model:
        - **Adaptive** (Opus 4.6+): ``thinking={"type": "adaptive"}``,
          extra_body={"output_config": {"effort": reasoning_effort}}.
          When ``disable_extended_thinking`` is True (system-card BrowseComp
          recipe per Q3), ``thinking`` is omitted but the effort still
          flows through ``output_config``.
        - **Budget-based** (Sonnet 4.5 etc.): ``thinking={"type": "enabled",
          "budget_tokens": N}``, extra_body=None.  ``disable_extended_thinking``
          is ignored on this path (budget-based thinking is opt-in via the
          ``thinking`` block; not attaching it already disables it).
        """
        if not reasoning_effort:
            return None, None

        # Models that should never use thinking.
        if any(model.startswith(m) for m in cls._NO_THINKING_MODELS):
            return None, None

        if any(model.startswith(m) for m in cls._ADAPTIVE_THINKING_MODELS):
            # Adaptive thinking: effort goes in output_config
            extra = {"output_config": {"effort": reasoning_effort}}
            if disable_extended_thinking:
                # Effort still routed via output_config; no thinking hint.
                # Per Opus 4.6 system card §2.21.1 BrowseComp recipe.
                return None, extra
            return ({"type": "adaptive"}, extra)

        # Budget-based thinking
        budget = cls._THINKING_BUDGETS.get(reasoning_effort, 16_000)
        return {"type": "enabled", "budget_tokens": budget}, None

    # ------------------------------------------------------------------
    # max_tokens guards
    # ------------------------------------------------------------------
    # Some providers reject requests whose ``max_tokens`` exceeds the model's
    # own parameter limit ("The maximum tokens you requested exceeds the model
    # limit"). Cap conservatively, leaving a buffer.
    _MAX_TOKENS_PARAM_CAP: int = 100_000

    # Standard Anthropic context window when 1M is not in effect.
    _STANDARD_CONTEXT_LIMIT: int = 200_000

    # Headroom we leave free of input + max_tokens — covers minor token-count
    # rounding between our estimate and the API's accounting.
    _CONTEXT_LIMIT_SAFETY_HEADROOM: int = 5_000

    # Anthropic beta header that unlocks the 1M context window on Sonnet 4 / 4.5.
    _1M_CONTEXT_BETA_HEADER_VALUE: str = "context-1m-2025-08-07"

    def _maybe_attach_1m_context_beta(
        self, kwargs: dict[str, Any], model: str
    ) -> None:
        """Attach the ``anthropic-beta: context-1m-2025-08-07`` header when
        ``_enable_1m_context_model`` is True and ``model`` supports it.

        Without this header Anthropic enforces the standard 200K window even
        if our local guards are relaxed, so a prompt above 200K would hit
        ``prompt_too_long`` regardless of ``enable_1m_context_model``.
        """
        if not self._enable_1m_context_model:
            return
        from arcticswarm.context_management import supports_anthropic_1m_context

        if not supports_anthropic_1m_context(model):
            logger.warning(
                "enable_1m_context_model=True but model %r does not support "
                "Anthropic's context-1m beta. Header NOT attached; the API "
                "will reject prompts above 200K.",
                model,
            )
            return
        extra_headers = dict(kwargs.get("extra_headers") or {})
        existing = extra_headers.get("anthropic-beta", "")
        if self._1M_CONTEXT_BETA_HEADER_VALUE in existing:
            return
        extra_headers["anthropic-beta"] = (
            f"{existing},{self._1M_CONTEXT_BETA_HEADER_VALUE}"
            if existing
            else self._1M_CONTEXT_BETA_HEADER_VALUE
        )
        kwargs["extra_headers"] = extra_headers

    def _apply_max_tokens_guards(
        self, kwargs: dict[str, Any]
    ) -> None:
        """Cap ``kwargs['max_tokens']`` so the request stays within known limits.

        Called after the thinking-budget bump but before the API call. Two
        guards apply:

        1. **Parameter cap** (always): ``max_tokens`` cannot exceed the model's
           own ``max_tokens`` parameter limit. We cap at
           ``_MAX_TOKENS_PARAM_CAP`` (100k) to leave a safety buffer.

        2. **Standard context window** (when ``_enable_1m_context_model`` is
           False): ``input + max_tokens`` cannot exceed 200k. We estimate
           input tokens from the JSON-serialised messages/system/tools and
           cap so the sum stays under 200k.

        The guard never reduces ``max_tokens`` below ``budget_tokens + 1024``
        (the minimum the Anthropic API requires when thinking is enabled). If
        even that minimum doesn't fit, we leave the original value alone — the
        agent's reduced-thinking-budget fallback chain will pick up the API
        rejection.
        """
        requested = int(kwargs.get("max_tokens", 0) or 0)
        if requested <= 0:
            return

        thinking = kwargs.get("thinking") or {}
        budget = int(thinking.get("budget_tokens", 0) or 0)
        floor = budget + 1024 if budget else 1024

        capped = min(requested, self._MAX_TOKENS_PARAM_CAP)

        if not self._enable_1m_context_model:
            try:
                est_input = self._estimate_input_tokens(kwargs)
            except Exception:
                # Estimation is best-effort; never let it block the call.
                est_input = 0
            if est_input > 0:
                headroom = (
                    self._STANDARD_CONTEXT_LIMIT
                    - est_input
                    - self._CONTEXT_LIMIT_SAFETY_HEADROOM
                )
                if headroom >= floor:
                    capped = max(min(capped, headroom), floor)
                else:
                    # Even the floor doesn't fit. Don't shrink below floor —
                    # the request is going to fail and we want the
                    # reduced-thinking-budget fallback to handle it.
                    logger.warning(
                        "max_tokens guard: estimated input %d leaves only "
                        "%d headroom in 200k window (floor=%d). Sending "
                        "anyway; expect API rejection and fallback.",
                        est_input, headroom, floor,
                    )

        if capped < requested:
            logger.info(
                "max_tokens capped: %d -> %d (1M=%s, budget=%d).",
                requested, capped, self._enable_1m_context_model, budget,
            )
            kwargs["max_tokens"] = capped

    @staticmethod
    def _estimate_input_tokens(kwargs: dict[str, Any]) -> int:
        """Cheap char-based token estimate for the request payload.

        4 chars/token is the canonical rule of thumb for English text. We
        only need a rough estimate for the headroom check, so this beats
        round-tripping through ``count_tokens`` (extra API call).
        """
        msgs = kwargs.get("messages") or []
        system = kwargs.get("system") or ""
        tools = kwargs.get("tools") or []
        try:
            chars = (
                len(json.dumps(msgs, default=str))
                + len(json.dumps(system, default=str))
                + len(json.dumps(tools, default=str))
            )
        except (TypeError, ValueError):
            chars = sum(len(str(m)) for m in msgs)
            chars += len(str(system)) + sum(len(str(t)) for t in tools)
        return chars // 4

    @staticmethod
    def _normalise_content(content: Any) -> list[dict[str, Any]]:
        """Convert Anthropic response content (Pydantic objects) to lean plain dicts.

        ``thinking`` blocks (from extended thinking) are preserved so that
        they can be round-tripped back to the API in multi-turn
        conversations — the API requires them when thinking is enabled.
        """
        blocks: list[dict[str, Any]] = []
        for block in content:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else "text")
            if btype == "thinking":
                # Preserve thinking blocks for API round-trip
                if isinstance(block, dict):
                    blocks.append(block)
                elif hasattr(block, "model_dump"):
                    blocks.append(block.model_dump())
            elif btype == "text":
                blocks.append({"type": "text", "text": getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", "")})
            elif btype == "tool_use":
                blocks.append({
                    "type": "tool_use",
                    "id": getattr(block, "id", "") if not isinstance(block, dict) else block.get("id", ""),
                    "name": getattr(block, "name", "") if not isinstance(block, dict) else block.get("name", ""),
                    "input": getattr(block, "input", {}) if not isinstance(block, dict) else block.get("input", {}),
                })
            elif isinstance(block, dict):
                blocks.append(block)
            elif hasattr(block, "model_dump"):
                blocks.append(block.model_dump())
        return blocks

    @staticmethod
    def _extract_usage(usage: Any) -> dict[str, int]:
        input_tok = getattr(usage, "input_tokens", 0) or 0
        output_tok = getattr(usage, "output_tokens", 0) or 0
        # vLLM servers return prompt_tokens / completion_tokens instead of the Anthropic field names.
        if not input_tok and not output_tok:
            input_tok = getattr(usage, "prompt_tokens", 0) or 0
            output_tok = getattr(usage, "completion_tokens", 0) or 0
        return {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        }

    # -- public API ---------------------------------------------------------

    def call(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        force_json: bool = False,
    ) -> LLMResponse:
        saved = self._inject_cache_breakpoint(messages)
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                system=self._system_for_api(system_prompt),
                tools=self._tools_for_api(tools),
                messages=self._strip_metadata(messages),
            )
            tp, extra = self._thinking_params(
                reasoning_effort, model,
                disable_extended_thinking=self._disable_extended_thinking,
            )
            if tp is not None:
                kwargs["thinking"] = tp
                # Budget-based: max_tokens must exceed budget_tokens
                budget = tp.get("budget_tokens", 0)
                if budget and kwargs["max_tokens"] <= budget:
                    kwargs["max_tokens"] = budget + max_tokens
            if extra is not None:
                kwargs["extra_body"] = extra

            self._maybe_attach_1m_context_beta(kwargs, model)
            self._apply_max_tokens_guards(kwargs)

            response = self._client.messages.create(**kwargs)
        finally:
            self._restore_cache_breakpoint(saved)

        u = self._extract_usage(response.usage) if response.usage else {}
        return LLMResponse(
            content_blocks=self._normalise_content(response.content),
            stop_reason=response.stop_reason or "",
            **u,
        )

    def call_streaming(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        on_text_delta: Any | None = None,
        on_tool_input_delta: Any | None = None,
    ) -> LLMResponse:
        saved = self._inject_cache_breakpoint(messages)
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                system=self._system_for_api(system_prompt),
                tools=self._tools_for_api(tools),
                messages=self._strip_metadata(messages),
            )
            tp, extra = self._thinking_params(
                reasoning_effort, model,
                disable_extended_thinking=self._disable_extended_thinking,
            )
            if tp is not None:
                kwargs["thinking"] = tp
                budget = tp.get("budget_tokens", 0)
                if budget and kwargs["max_tokens"] <= budget:
                    kwargs["max_tokens"] = budget + max_tokens
            if extra is not None:
                kwargs["extra_body"] = extra

            self._maybe_attach_1m_context_beta(kwargs, model)
            self._apply_max_tokens_guards(kwargs)

            current_tool: dict[str, Any] | None = None
            current_tool_json = ""
            tool_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []
            # Diagnostic: capture a lightweight summary of every streaming event.
            _event_log: list[dict[str, Any]] = []

            with self._client.messages.stream(**kwargs) as stream:
                for event in stream:
                    # --- diagnostic event capture (lightweight) ---
                    evt_entry: dict[str, Any] = {"type": event.type}
                    if event.type == "content_block_start":
                        cb = getattr(event, "content_block", None)
                        evt_entry["block_type"] = getattr(cb, "type", None)
                        evt_entry["block_id"] = getattr(cb, "id", None)
                        if getattr(event.content_block, "type", None) == "tool_use":
                            current_tool = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": {},
                            }
                            current_tool_json = ""
                    elif event.type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", None)
                        evt_entry["delta_type"] = delta_type
                        if delta_type == "thinking_delta":
                            evt_entry["thinking_chars"] = len(getattr(delta, "thinking", "") or "")
                        if delta_type == "text_delta":
                            text_parts.append(event.delta.text)
                            evt_entry["text_chars"] = len(event.delta.text)
                            if on_text_delta:
                                on_text_delta(event.delta.text)
                        elif delta_type == "input_json_delta":
                            current_tool_json += event.delta.partial_json
                            if on_tool_input_delta and current_tool is not None:
                                on_tool_input_delta(
                                    current_tool["name"],
                                    current_tool["id"],
                                    event.delta.partial_json,
                                )
                    elif event.type == "content_block_stop":
                        if current_tool is not None:
                            try:
                                current_tool["input"] = (
                                    json.loads(current_tool_json) if current_tool_json else {}
                                )
                            except json.JSONDecodeError:
                                current_tool["input"] = {}
                            tool_calls.append(current_tool)
                            current_tool = None
                            current_tool_json = ""
                    elif event.type == "message_delta":
                        md = getattr(event, "delta", None)
                        evt_entry["stop_reason"] = getattr(md, "stop_reason", None)
                        usage = getattr(event, "usage", None)
                        if usage:
                            evt_entry["output_tokens"] = getattr(usage, "output_tokens", None)
                    elif event.type == "message_start":
                        msg = getattr(event, "message", None)
                        if msg:
                            evt_entry["model"] = getattr(msg, "model", None)
                            evt_entry["role"] = getattr(msg, "role", None)
                            u = getattr(msg, "usage", None)
                            if u:
                                evt_entry["input_tokens"] = getattr(u, "input_tokens", None)
                                evt_entry["cache_read"] = getattr(u, "cache_read_input_tokens", None)
                    _event_log.append(evt_entry)

                final_message = stream.get_final_message()
        finally:
            self._restore_cache_breakpoint(saved)

        content_blocks = self._normalise_content(final_message.content)
        u = self._extract_usage(final_message.usage) if final_message.usage else {}

        # Attach raw event log only when the response is empty (diagnostics).
        raw_log = _event_log if not content_blocks else []
        return LLMResponse(
            content_blocks=content_blocks,
            stop_reason=final_message.stop_reason or "",
            _raw_event_log=raw_log,
            **u,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIChatLLMClient(BaseLLMClient):
    """Chat Completions client — wraps ``openai.OpenAI`` with chat.completions.create()."""

    # Default per-request output-token cap.
    _MAX_OUTPUT_TOKENS = 64_000

    def __init__(self, *, api_key: str = "", base_url: str = "", timeout: float | None = None) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = dict(base_url=base_url, api_key=api_key)
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = OpenAI(**kwargs)

    @staticmethod
    def _normalize_model(model: str) -> str:
        """Public OpenAI uses the bare model id (e.g. ``gpt-5``)."""
        return model

    # -- format conversion helpers -----------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Internal (Anthropic-style) tool defs -> OpenAI function-call format."""
        out: list[dict[str, Any]] = []
        for t in tools:
            d = {**t}
            d.pop("cache_control", None)
            out.append({
                "type": "function",
                "function": {
                    "name": d["name"],
                    "description": d.get("description", ""),
                    "parameters": d.get("input_schema", {}),
                },
            })
        return out

    @staticmethod
    def _convert_messages(
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Internal message history -> OpenAI chat-completion messages."""
        out: list[dict[str, Any]] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    tool_results = [
                        b for b in content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    ]
                    if tool_results:
                        collected_images: list[dict[str, Any]] = []
                        for tr in tool_results:
                            text = _extract_text_from_content(tr.get("content", ""))
                            out.append({
                                "role": "tool",
                                "tool_call_id": tr.get("tool_use_id", ""),
                                "content": text,
                            })
                            collected_images.extend(
                                _extract_images_from_content(tr.get("content", ""))
                            )
                        if collected_images:
                            parts: list[dict[str, Any]] = [
                                {"type": "text", "text": "Image returned by tool:"},
                            ]
                            parts.extend(collected_images)
                            out.append({"role": "user", "content": parts})
                    else:
                        oai_parts: list[dict[str, Any]] = []
                        for b in content:
                            if not isinstance(b, dict):
                                continue
                            if b.get("type") == "text":
                                oai_parts.append({"type": "text", "text": b.get("text", "")})
                            else:
                                converted = _anthropic_image_to_openai(b)
                                if converted:
                                    oai_parts.append(converted)
                        if oai_parts:
                            out.append({"role": "user", "content": oai_parts})
                        else:
                            out.append({"role": "user", "content": ""})
                else:
                    out.append({"role": "user", "content": str(content)})

            elif role == "assistant":
                if isinstance(content, str):
                    out.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    blocks = [
                        b.model_dump() if hasattr(b, "model_dump") else b
                        for b in content
                    ]
                    text_parts = []
                    openai_tool_calls = []
                    for b in blocks:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                        elif b.get("type") == "tool_use":
                            openai_tool_calls.append({
                                "id": b.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": b.get("name", ""),
                                    "arguments": json.dumps(b.get("input", {})),
                                },
                            })
                    oai_msg: dict[str, Any] = {"role": "assistant"}
                    oai_msg["content"] = "\n".join(text_parts) if text_parts else None
                    if openai_tool_calls:
                        oai_msg["tool_calls"] = openai_tool_calls
                    out.append(oai_msg)
                else:
                    out.append({"role": "assistant", "content": str(content)})
            else:
                out.append(msg)

        return out

    @staticmethod
    def _response_to_blocks(message: Any) -> list[dict[str, Any]]:
        """Convert an OpenAI ChatCompletionMessage to internal content blocks."""
        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    inp = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    inp = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": inp,
                })
        return blocks

    @staticmethod
    def _map_finish_reason(reason: str | None) -> str:
        mapping = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
        return mapping.get(reason or "", reason or "end_turn")

    # -- request-shaping hooks (overridable by subclasses) ------------------

    def _resolve_max_completion_tokens(
        self,
        max_tokens: int,
        oai_messages: list[dict[str, Any]],
        oai_tools: list[dict[str, Any]] | None,
        system_prompt: str,
    ) -> int:
        """Per-request output-token budget.  Base: the Cortex proxy's 64k cap.

        Subclasses (e.g. self-hosted vLLM) override to clamp dynamically
        against the model's context window.
        """
        return min(max_tokens, self._MAX_OUTPUT_TOKENS)

    def _apply_reasoning_kwargs(
        self,
        kwargs: dict[str, Any],
        reasoning_effort: str | None,
        oai_tools: list[dict[str, Any]] | None,
    ) -> None:
        """Inject reasoning/sampling kwargs into the request.

        Base implementation matches the Cortex OpenAI proxy semantics:
        ``reasoning_effort`` is only accepted when no tools are present.
        Subclasses override to inject provider-specific knobs (e.g. vLLM's
        ``extra_body`` thinking toggle + sampling).
        """
        # Cortex proxy rejects reasoning_effort when tools are present.
        if reasoning_effort and not oai_tools:
            kwargs["reasoning_effort"] = reasoning_effort

    # -- public API ---------------------------------------------------------

    def call(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        force_json: bool = False,
    ) -> LLMResponse:
        oai_messages = self._convert_messages(system_prompt, messages)
        oai_tools = self._convert_tools(tools) if tools else None

        kwargs: dict[str, Any] = dict(
            model=self._normalize_model(model),
            messages=oai_messages,
            max_completion_tokens=self._resolve_max_completion_tokens(
                max_tokens, oai_messages, oai_tools, system_prompt
            ),
        )
        if oai_tools:
            kwargs["tools"] = oai_tools
        self._apply_reasoning_kwargs(kwargs, reasoning_effort, oai_tools)
        if force_json:
            # Force valid-JSON output via vLLM/OpenAI guided decoding. Needed
            # for utility calls (source scorer / content compactor) on
            # reasoning models that emit a reasoning preamble before/around the
            # JSON (e.g. Tongyi-DeepResearch ignores enable_thinking=False) —
            # the grammar constraint prevents any non-JSON prefix.
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        blocks = self._response_to_blocks(choice.message)

        usage = response.usage
        reasoning_tok = 0
        if usage:
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                reasoning_tok = getattr(details, "reasoning_tokens", 0) or 0
        return LLMResponse(
            content_blocks=blocks,
            stop_reason=self._map_finish_reason(choice.finish_reason),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
            reasoning_tokens=reasoning_tok,
        )

    def call_streaming(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        on_text_delta: Any | None = None,
        on_tool_input_delta: Any | None = None,
    ) -> LLMResponse:
        oai_messages = self._convert_messages(system_prompt, messages)
        oai_tools = self._convert_tools(tools) if tools else None

        kwargs: dict[str, Any] = dict(
            model=self._normalize_model(model),
            messages=oai_messages,
            max_completion_tokens=self._resolve_max_completion_tokens(
                max_tokens, oai_messages, oai_tools, system_prompt
            ),
            stream=True,
            stream_options={"include_usage": True},
        )
        if oai_tools:
            kwargs["tools"] = oai_tools
        self._apply_reasoning_kwargs(kwargs, reasoning_effort, oai_tools)

        stream = self._client.chat.completions.create(**kwargs)

        full_text = ""
        tool_calls_by_idx: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0

        for chunk in stream:
            if chunk.usage:
                total_prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                total_completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                details = getattr(chunk.usage, "completion_tokens_details", None)
                if details:
                    total_reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            if delta and delta.content:
                full_text += delta.content
                if on_text_delta:
                    on_text_delta(delta.content)

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_by_idx:
                        tool_calls_by_idx[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    entry = tool_calls_by_idx[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments
                            if on_tool_input_delta and entry["name"]:
                                on_tool_input_delta(
                                    entry["name"],
                                    entry["id"],
                                    tc_delta.function.arguments,
                                )

        blocks: list[dict[str, Any]] = []
        if full_text:
            blocks.append({"type": "text", "text": full_text})
        for _idx in sorted(tool_calls_by_idx):
            entry = tool_calls_by_idx[_idx]
            try:
                inp = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                inp = {}
            blocks.append({
                "type": "tool_use",
                "id": entry["id"],
                "name": entry["name"],
                "input": inp,
            })

        return LLMResponse(
            content_blocks=blocks,
            stop_reason=self._map_finish_reason(finish_reason),
            input_tokens=total_prompt_tokens,
            output_tokens=total_completion_tokens,
            reasoning_tokens=total_reasoning_tokens,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Azure OpenAI Chat Completions implementation
# ---------------------------------------------------------------------------

class AzureOpenAIChatLLMClient(OpenAIChatLLMClient):
    """Azure Chat Completions client — wraps ``openai.AzureOpenAI``."""

    def __init__(
        self,
        *,
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "2025-04-01-preview",
        timeout: float | None = None,
    ) -> None:
        from openai import AzureOpenAI

        kwargs: dict[str, Any] = dict(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = AzureOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Self-hosted vLLM (OpenAI-compatible) implementation — e.g. Qwen3.5
# ---------------------------------------------------------------------------

# Reasoning-effort values that turn Qwen's thinking OFF.  Qwen has no numeric
# effort knob (thinking is a binary chat-template toggle), so we map the
# arcticswarm effort vocabulary onto enable_thinking: anything at or below
# "low" disables thinking; "medium"/"high"/"xhigh"/"max" (and the default)
# enable it.
_VLLM_THINKING_OFF_EFFORTS: frozenset[str] = frozenset({"none", "low", "minimal"})

# Canonical served-model id for the Qwen3.5-27B vLLM deployment.  Friendly
# aliases (e.g. "qwen3.5-27b") are mapped to this before hitting the wire.
_DEFAULT_QWEN_SERVED_MODEL_ID = "Qwen/Qwen3.5-27B"


# Server-reported context window, cached per base_url so we probe /v1/models
# once per process (a swarm builds many clients against the same endpoint).
_VLLM_MAX_LEN_CACHE: dict[str, int | None] = {}
# base_urls we've already logged a cap for, so a swarm's many clients don't
# repeat the same INFO line dozens of times.
_VLLM_CAP_LOGGED: set[str] = set()


def _probe_vllm_max_model_len(base_url: str, served_id: str = "") -> int | None:
    """Return the served model's ``max_model_len`` from ``<base_url>/models``.

    Lets a shared config that hardcodes a larger window for one model (e.g.
    262144 for Qwen3.5) be auto-capped to whatever the *actually-served* model
    supports (e.g. 131072 for Tongyi-DeepResearch-30B-A3B) — exceeding the
    server's ``max_model_len`` hard-errors (HTTP 400). Cached per base_url;
    returns ``None`` (caller keeps its configured value) on any failure.
    """
    if not base_url:
        return None
    if base_url in _VLLM_MAX_LEN_CACHE:
        return _VLLM_MAX_LEN_CACHE[base_url]
    detected: int | None = None
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", []) or []
        chosen = next(
            (m for m in models if served_id and m.get("id") == served_id), None
        ) or (models[0] if models else None)
        if chosen and chosen.get("max_model_len"):
            detected = int(chosen["max_model_len"])
    except Exception as exc:  # network/JSON/missing-field — keep configured
        logger.debug("vLLM max_model_len probe failed for %s: %s", base_url, exc)
    _VLLM_MAX_LEN_CACHE[base_url] = detected
    return detected


class VLLMChatLLMClient(OpenAIChatLLMClient):
    """Chat Completions client for a self-hosted vLLM OpenAI-compatible server.

    Differs from the Cortex-proxy :class:`OpenAIChatLLMClient` in three ways:

    * **Model id** — sends the server's registered id (e.g.
      ``Qwen/Qwen3.5-27B``) instead of adding the Cortex ``openai-`` prefix.
    * **Thinking + sampling** — vLLM/Qwen has no ``reasoning_effort`` knob;
      thinking is a binary ``chat_template_kwargs.enable_thinking`` toggle
      sent via ``extra_body``.  Recommended sampling (temperature/top_p/
      top_k/presence_penalty) is sent explicitly.  The request shape is held
      constant across a run so vLLM's ``--enable-prefix-caching`` stays warm
      (sampling + ``extra_body`` are request metadata, not prompt tokens).
    * **Output budget** — clamped dynamically against the model's context
      window so ``prompt_tokens + max_tokens`` never exceeds ``max_model_len``
      (vLLM hard-errors otherwise).
    """

    # vLLM enforces its own ceiling via max_model_len; this is just a sane
    # upper bound on a single turn's generation (thinking + answer).
    _DEFAULT_MAX_OUTPUT_TOKENS = 32_000
    _OUTPUT_SAFETY_HEADROOM = 8_192
    _OUTPUT_FLOOR = 1_024

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        timeout: float | None = None,
        enable_thinking: bool = True,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 0.0,
        served_model_id: str = "",
        max_model_len: int = 262_144,
        max_output_tokens: int = 0,
    ) -> None:
        # vLLM's OpenAI server requires a non-empty key string even when auth
        # is disabled.
        super().__init__(api_key=api_key or "EMPTY", base_url=base_url, timeout=timeout)
        self._enable_thinking = enable_thinking
        # Tracks whether the most recent request asked for thinking, so
        # _response_to_blocks can distinguish raw (unclosed) chain-of-thought on
        # a tool/runaway turn from a genuine instant-mode answer. Updated in
        # _apply_reasoning_kwargs; safe because arcticswarm uses one client per
        # agent with sequential calls.
        self._last_thinking = enable_thinking
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._presence_penalty = presence_penalty
        self._served_model_id = served_model_id or _DEFAULT_QWEN_SERVED_MODEL_ID
        self._max_model_len = max_model_len
        # Auto-cap to the server's real context window. A shared YAML may
        # hardcode a larger window than the served model supports (e.g. 262144
        # for Qwen3.5 vs 131072 for Tongyi-DeepResearch-30B-A3B); without this,
        # prompt+max_tokens can exceed the server limit and 400 on long cases.
        _detected = _probe_vllm_max_model_len(base_url, self._served_model_id)
        if _detected and _detected < self._max_model_len:
            if base_url not in _VLLM_CAP_LOGGED:
                _VLLM_CAP_LOGGED.add(base_url)
                logger.info(
                    "vLLM server max_model_len=%d for %s; capping configured %d.",
                    _detected, base_url, max_model_len,
                )
            self._max_model_len = _detected
        self._max_output_tokens = max_output_tokens or self._DEFAULT_MAX_OUTPUT_TOKENS

    def _normalize_model(self, model: str) -> str:  # type: ignore[override]
        """Map a friendly alias to the server's registered model id.

        vLLM's ``/v1/chat/completions`` requires the exact served id.  A value
        that already looks like a served id (contains ``/``) is sent
        unchanged; any bare alias (e.g. ``qwen3.5-27b``) maps to
        ``served_model_id``.  Never adds the Cortex ``openai-`` prefix.
        """
        if "/" in model:
            return model
        return self._served_model_id

    def _resolve_thinking(self, reasoning_effort: str | None) -> bool:
        """Map the arcticswarm effort vocabulary onto Qwen's binary thinking toggle.

        Thinking is ON only for an explicit thinking-level effort
        (``medium``/``high``/``xhigh``/``max``).  A missing effort (``None``)
        or a ``none``/``low``/``minimal`` effort means OFF.  This is what keeps
        the *utility* call sites — source scorer, content compactor,
        reflection, interpretation/decomposition — out of thinking mode: they
        all call with ``reasoning_effort=None``, and a ``<think>`` block would
        consume their small token budget and yield empty / truncated JSON.
        The main orchestration + subagents + auditor + reasoning tool pass an
        explicit ``xhigh`` (uniform across the run), so they think and the
        rendered prompt prefix stays cache-stable.
        """
        if not self._enable_thinking:
            return False
        if not reasoning_effort or reasoning_effort.lower() in _VLLM_THINKING_OFF_EFFORTS:
            return False
        return True

    def _apply_reasoning_kwargs(
        self,
        kwargs: dict[str, Any],
        reasoning_effort: str | None,
        oai_tools: list[dict[str, Any]] | None,
    ) -> None:
        # Qwen has no reasoning_effort param — drive thinking via the chat
        # template toggle and send the recommended sampling explicitly.  This
        # holds regardless of tool presence (the whole agent is tool-driven).
        thinking = self._resolve_thinking(reasoning_effort)
        # Remember the mode so _response_to_blocks can tell raw chain-of-thought
        # (drop it) from a genuine no-thinking answer (keep it). Matters when the
        # server runs WITHOUT a reasoning parser (e.g. Tongyi-DeepResearch),
        # where <think> arrives inline in content instead of a separate field.
        self._last_thinking = thinking
        kwargs["temperature"] = self._temperature
        kwargs["top_p"] = self._top_p
        kwargs["presence_penalty"] = self._presence_penalty
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": thinking
            },
            "top_k": self._top_k,
        }

    def _resolve_max_completion_tokens(
        self,
        max_tokens: int,
        oai_messages: list[dict[str, Any]],
        oai_tools: list[dict[str, Any]] | None,
        system_prompt: str,
    ) -> int:
        requested = min(max_tokens, self._max_output_tokens) if max_tokens else self._max_output_tokens
        est_input = self._estimate_input_tokens(oai_messages, system_prompt, oai_tools)
        headroom = self._max_model_len - est_input - self._OUTPUT_SAFETY_HEADROOM
        if headroom < self._OUTPUT_FLOOR:
            # The prompt leaves no usable room for a response. Clamping to the
            # 1024 floor here just yields an empty/truncated turn — and on
            # reasoning models (e.g. Tongyi-DeepResearch) a nudge/retry spiral.
            # Surface it as a context-length error instead so the agent loop's
            # reactive split-and-summarize compaction engages (the wording
            # matches Agent._is_context_too_long).
            raise RuntimeError(
                f"This model's maximum context length is {self._max_model_len} tokens, "
                f"but the request needs more (est_input={est_input}, "
                f"output_headroom={self._OUTPUT_SAFETY_HEADROOM}). Reduce the prompt."
            )
        capped = max(min(requested, headroom), self._OUTPUT_FLOOR)
        if capped < requested:
            logger.info(
                "vLLM max_completion_tokens clamped: %d -> %d "
                "(est_input=%d, max_model_len=%d).",
                requested, capped, est_input, self._max_model_len,
            )
        return capped

    @staticmethod
    def _estimate_input_tokens(
        oai_messages: list[dict[str, Any]],
        system_prompt: str,
        oai_tools: list[dict[str, Any]] | None,
    ) -> int:
        """Cheap 4-chars/token estimate of the prompt size (matches the
        Anthropic client's heuristic).  Used only to keep
        ``prompt + max_tokens`` under ``max_model_len``; over-counting (the
        system prompt also lives inside ``oai_messages``) is safe — it only
        shrinks the output budget."""
        try:
            chars = (
                len(json.dumps(oai_messages, default=str))
                + len(json.dumps(system_prompt, default=str))
                + len(json.dumps(oai_tools or [], default=str))
            )
        except (TypeError, ValueError):
            chars = sum(len(str(m)) for m in oai_messages) + len(str(system_prompt))
        return chars // 4

    # Closing delimiter of an inline chain-of-thought (present when the vLLM
    # server runs WITHOUT a reasoning parser, e.g. Tongyi-DeepResearch served
    # as bare `vllm serve`, so <think>...</think> arrives inside content).
    _THINK_CLOSE = "</think>"

    @staticmethod
    def _server_reasoning(message: Any) -> Any:
        """Return server-separated reasoning if a vLLM reasoning parser is
        active, else falsy.  Qwen3's parser returns it in ``reasoning_content``;
        some builds use ``reasoning``.  Unknown response fields land in the
        SDK's ``model_extra``, so check there too."""
        extra = getattr(message, "model_extra", None) or {}
        return (
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
            or extra.get("reasoning")
            or extra.get("reasoning_content")
        )

    def _response_to_blocks(self, message: Any) -> list[dict[str, Any]]:  # type: ignore[override]
        """Strip an inline chain-of-thought from the assistant text.

        With a vLLM reasoning parser active (``--reasoning-parser qwen3``) the
        server splits CoT out and ``content`` is already clean — detected via
        :meth:`_server_reasoning` — in which case blocks are left as-is (this is
        the Qwen3.5-27B deployment, unchanged).

        Without the reasoning parser, CoT is inline in ``content`` (the
        recommended Tongyi-DeepResearch setup, matching their own bare
        ``vllm serve`` + raw-text ReAct):
          * answer / post-think turn -> ``"...thinking... </think> rest"`` — keep
            only the text after the final ``</think>`` (the hermes tool-call
            parser has already split any ``<tool_call>`` into a tool_use block,
            which we preserve).
          * tool / runaway turn -> ``"...thinking..."`` with no ``</think>`` —
            the whole text is raw CoT, so drop it (a tool_use block, if any, is
            kept; an otherwise-empty turn becomes an empty response the agent
            retries, instead of mistaking CoT for content).

        Keeps answers clean AND keeps CoT out of stored history (never re-sent
        -> bounds context growth), regardless of the server's parser setting.
        """
        blocks = OpenAIChatLLMClient._response_to_blocks(message)
        if self._server_reasoning(message):
            return blocks  # server already separated CoT; content is clean
        out: list[dict[str, Any]] = []
        for b in blocks:
            if b.get("type") == "text":
                txt = b.get("text", "") or ""
                if self._THINK_CLOSE in txt:
                    txt = txt.rsplit(self._THINK_CLOSE, 1)[-1].lstrip()
                elif getattr(self, "_last_thinking", False):
                    txt = ""  # raw, unclosed CoT (tool/runaway turn) — drop it
                b = {**b, "text": txt}
                if not txt.strip():
                    continue
            out.append(b)
        return out


# ---------------------------------------------------------------------------
# OpenAI Responses API implementation
# ---------------------------------------------------------------------------

class OpenAIResponsesLLMClient(BaseLLMClient):
    """Responses API client — uses ``client.responses.create()``.

    Preserves reasoning items across tool-calling turns so the model can
    continue its chain of thought instead of re-reasoning from scratch.
    """

    def __init__(self, *, api_key: str = "", base_url: str = "", timeout: float | None = None) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = dict(base_url=base_url, api_key=api_key)
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = OpenAI(**kwargs)

    # -- format conversion helpers -----------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Internal (Anthropic-style) tool defs -> Responses API format."""
        out: list[dict[str, Any]] = []
        for t in tools:
            d = {**t}
            d.pop("cache_control", None)
            out.append({
                "type": "function",
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": d.get("input_schema", {}),
            })
        return out

    @staticmethod
    def _convert_input(
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert internal message history -> Responses API ``input`` items.

        Returns ``(instructions, input_items)`` where *instructions* is the
        system prompt (passed as the ``instructions`` parameter) and
        *input_items* is the list of typed items for the ``input`` parameter.
        """
        items: list[dict[str, Any]] = []
        valid_call_ids: set[str] = set()

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if isinstance(content, str):
                    items.append({
                        "type": "message",
                        "role": "user",
                        "content": content,
                    })
                elif isinstance(content, list):
                    tool_results = [
                        b for b in content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    ]
                    if tool_results:
                        collected_images: list[dict[str, Any]] = []
                        for tr in tool_results:
                            cid = tr.get("tool_use_id", "")
                            if not cid:
                                continue
                            text = _extract_text_from_content(tr.get("content", ""))
                            items.append({
                                "type": "function_call_output",
                                "call_id": cid,
                                "output": text,
                            })
                            collected_images.extend(
                                _extract_images_from_content(tr.get("content", ""))
                            )
                        if collected_images:
                            parts: list[dict[str, Any]] = [
                                {"type": "input_text", "text": "Image returned by tool:"},
                            ]
                            for img in collected_images:
                                url = img.get("image_url", {}).get("url", "")
                                if url:
                                    parts.append({"type": "input_image", "image_url": url})
                            items.append({
                                "type": "message",
                                "role": "user",
                                "content": parts,
                            })
                    else:
                        oai_parts: list[dict[str, Any]] = []
                        for b in content:
                            if not isinstance(b, dict):
                                continue
                            if b.get("type") == "text":
                                oai_parts.append({"type": "input_text", "text": b.get("text", "")})
                            elif b.get("type") == "image":
                                converted = _anthropic_image_to_openai(b)
                                if converted:
                                    url = converted.get("image_url", {}).get("url", "")
                                    if url:
                                        oai_parts.append({"type": "input_image", "image_url": url})
                        if oai_parts:
                            items.append({
                                "type": "message",
                                "role": "user",
                                "content": oai_parts,
                            })
                else:
                    items.append({
                        "type": "message",
                        "role": "user",
                        "content": str(content),
                    })

            elif role == "assistant":
                if isinstance(content, str):
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    })
                elif isinstance(content, list):
                    blocks = [
                        b.model_dump() if hasattr(b, "model_dump") else b
                        for b in content
                    ]
                    for b in blocks:
                        if not isinstance(b, dict):
                            continue
                        btype = b.get("type", "")
                        if btype == "text":
                            items.append({
                                "type": "message",
                                "role": "assistant",
                                "content": b.get("text", ""),
                            })
                        elif btype == "tool_use":
                            name = b.get("name", "")
                            cid = b.get("id", "")
                            if not name:
                                continue
                            valid_call_ids.add(cid)
                            items.append({
                                "type": "function_call",
                                "call_id": cid,
                                "name": name,
                                "arguments": json.dumps(b.get("input", {})),
                            })
                        elif btype == "reasoning":
                            items.append(b)
                else:
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": str(content),
                    })

        return system_prompt, items

    @staticmethod
    def _response_to_blocks(response: Any) -> list[dict[str, Any]]:
        """Convert a Responses API response ``output`` to internal content blocks.

        Reasoning items are preserved so they flow back through the agent's
        message history and get re-injected on the next call.
        """
        blocks: list[dict[str, Any]] = []
        for item in response.output:
            item_type = getattr(item, "type", "")
            if item_type == "reasoning":
                entry: dict[str, Any] = {
                    "type": "reasoning",
                    "id": getattr(item, "id", ""),
                }
                summary = getattr(item, "summary", None)
                summaries = []
                if summary:
                    for s in summary:
                        summaries.append({
                            "type": getattr(s, "type", "summary_text"),
                            "text": getattr(s, "text", ""),
                        })
                entry["summary"] = summaries
                blocks.append(entry)
            elif item_type == "message":
                for content_item in getattr(item, "content", []):
                    ct = getattr(content_item, "type", "")
                    if ct == "output_text":
                        blocks.append({
                            "type": "text",
                            "text": getattr(content_item, "text", ""),
                        })
            elif item_type == "function_call":
                args_str = getattr(item, "arguments", "{}")
                try:
                    inp = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    inp = {}
                blocks.append({
                    "type": "tool_use",
                    "id": getattr(item, "call_id", ""),
                    "name": getattr(item, "name", ""),
                    "input": inp,
                })
        return blocks

    @staticmethod
    def _map_status(status: str | None) -> str:
        mapping = {"completed": "end_turn", "incomplete": "max_tokens"}
        return mapping.get(status or "", status or "end_turn")

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if not usage:
            return {}
        input_tok = getattr(usage, "input_tokens", 0) or 0
        output_tok = getattr(usage, "output_tokens", 0) or 0
        reasoning_tok = 0
        details = getattr(usage, "output_tokens_details", None)
        if details:
            reasoning_tok = getattr(details, "reasoning_tokens", 0) or 0
        return {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "reasoning_tokens": reasoning_tok,
        }

    # -- public API ---------------------------------------------------------

    def call(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        force_json: bool = False,
    ) -> LLMResponse:
        instructions, input_items = self._convert_input(system_prompt, messages)
        resp_tools = self._convert_tools(tools) if tools else []

        kwargs: dict[str, Any] = dict(
            model=model,
            instructions=instructions,
            input=input_items,
            max_output_tokens=max_tokens,
            store=True,
        )
        if resp_tools:
            kwargs["tools"] = resp_tools
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        response = self._client.responses.create(**kwargs)
        blocks = self._response_to_blocks(response)
        u = self._extract_usage(response)
        has_tool_call = any(b.get("type") == "tool_use" for b in blocks)
        stop = "tool_use" if has_tool_call else self._map_status(getattr(response, "status", None))

        return LLMResponse(
            content_blocks=blocks,
            stop_reason=stop,
            response_id=getattr(response, "id", "") or "",
            **u,
        )

    def call_streaming(
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        on_text_delta: Any | None = None,
        on_tool_input_delta: Any | None = None,
    ) -> LLMResponse:
        instructions, input_items = self._convert_input(system_prompt, messages)
        resp_tools = self._convert_tools(tools) if tools else []

        kwargs: dict[str, Any] = dict(
            model=model,
            instructions=instructions,
            input=input_items,
            max_output_tokens=max_tokens,
            stream=True,
            store=True,
        )
        if resp_tools:
            kwargs["tools"] = resp_tools
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        stream = self._client.responses.create(**kwargs)

        full_text = ""
        tool_calls: dict[str, dict[str, Any]] = {}
        reasoning_items: list[dict[str, Any]] = []
        final_usage: dict[str, int] = {}
        response_id = ""

        for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta":
                delta_text = getattr(event, "delta", "")
                if delta_text:
                    full_text += delta_text
                    if on_text_delta:
                        on_text_delta(delta_text)

            elif event_type == "response.function_call_arguments.delta":
                key = getattr(event, "item_id", "") or getattr(event, "call_id", "")
                delta_args = getattr(event, "delta", "")
                if key not in tool_calls:
                    tool_calls[key] = {"id": key, "call_id": "", "name": "", "arguments": ""}
                tool_calls[key]["arguments"] += delta_args

            elif event_type == "response.function_call_arguments.done":
                key = getattr(event, "item_id", "") or getattr(event, "call_id", "")
                name = getattr(event, "name", "")
                done_call_id = getattr(event, "call_id", "")
                if key in tool_calls:
                    if name:
                        tool_calls[key]["name"] = name
                    if done_call_id:
                        tool_calls[key]["call_id"] = done_call_id
                if on_tool_input_delta and key in tool_calls:
                    entry = tool_calls[key]
                    if entry["name"]:
                        on_tool_input_delta(entry["name"], entry.get("call_id") or key, "")

            elif event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if item and getattr(item, "type", "") == "function_call":
                    item_id = getattr(item, "id", "")
                    cid = getattr(item, "call_id", "")
                    nm = getattr(item, "name", "")
                    key = item_id or cid
                    if key and key not in tool_calls:
                        tool_calls[key] = {"id": key, "call_id": cid, "name": nm, "arguments": ""}
                    elif key:
                        tool_calls[key]["name"] = nm or tool_calls[key]["name"]
                        if cid:
                            tool_calls[key]["call_id"] = cid
                elif item and getattr(item, "type", "") == "reasoning":
                    ri_entry: dict[str, Any] = {
                        "type": "reasoning",
                        "id": getattr(item, "id", ""),
                    }
                    raw_summary = getattr(item, "summary", None)
                    summaries = []
                    if raw_summary:
                        for s in raw_summary:
                            summaries.append({
                                "type": getattr(s, "type", "summary_text"),
                                "text": getattr(s, "text", ""),
                            })
                    ri_entry["summary"] = summaries
                    reasoning_items.append(ri_entry)

            elif event_type == "response.completed":
                resp = getattr(event, "response", None)
                if resp:
                    response_id = getattr(resp, "id", "") or ""
                    final_usage = self._extract_usage(resp)
                    for out_item in getattr(resp, "output", []):
                        if getattr(out_item, "type", "") == "reasoning":
                            rid = getattr(out_item, "id", "")
                            raw_summary = getattr(out_item, "summary", None)
                            summaries: list[dict[str, Any]] = []
                            if raw_summary:
                                for s in raw_summary:
                                    summaries.append({
                                        "type": getattr(s, "type", "summary_text"),
                                        "text": getattr(s, "text", ""),
                                    })
                            existing = next(
                                (ri for ri in reasoning_items if ri.get("id") == rid),
                                None,
                            )
                            if existing:
                                if summaries and not existing.get("summary"):
                                    existing["summary"] = summaries
                            else:
                                reasoning_items.append({
                                    "type": "reasoning",
                                    "id": rid,
                                    "summary": summaries,
                                })

        blocks: list[dict[str, Any]] = []
        blocks.extend(reasoning_items)
        if full_text:
            blocks.append({"type": "text", "text": full_text})
        for key in tool_calls:
            entry = tool_calls[key]
            if not entry["name"]:
                continue
            try:
                inp = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                inp = {}
            tc_id = entry.get("call_id") or entry["id"]
            blocks.append({
                "type": "tool_use",
                "id": tc_id,
                "name": entry["name"],
                "input": inp,
            })

        has_tool_call = any(b.get("type") == "tool_use" for b in blocks)
        stop = "tool_use" if has_tool_call else "end_turn"

        return LLMResponse(
            content_blocks=blocks,
            stop_reason=stop,
            response_id=response_id,
            **final_usage,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Azure OpenAI Responses API implementation
# ---------------------------------------------------------------------------

class AzureOpenAIResponsesLLMClient(OpenAIResponsesLLMClient):
    """Azure Responses API client — wraps ``openai.AzureOpenAI``."""

    def __init__(
        self,
        *,
        api_key: str = "",
        azure_endpoint: str = "",
        api_version: str = "2025-04-01-preview",
        timeout: float | None = None,
    ) -> None:
        from openai import AzureOpenAI

        kwargs: dict[str, Any] = dict(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = AzureOpenAI(**kwargs)

    @staticmethod
    def _strip_provider_prefix(model: str) -> str:
        """Azure deployments use the bare model name; the ``openai-`` prefix
        only applies to the Cortex chat-completions proxy.  Strip it
        defensively so a config edit slip (or HEAD vs working-tree drift)
        doesn't 404 the deployment lookup.
        """
        if model.startswith("openai-"):
            logger.warning(
                "AzureOpenAIResponsesLLMClient: stripping 'openai-' prefix "
                "from model name %r (Azure deployments expect the bare "
                "model name; the prefix is only valid for the Cortex "
                "chat-completions proxy).",
                model,
            )
            return model[len("openai-"):]
        return model

    def call(  # type: ignore[override]
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        force_json: bool = False,
    ) -> LLMResponse:
        return super().call(
            model=self._strip_provider_prefix(model),
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tools=tools,
            messages=messages,
            reasoning_effort=reasoning_effort,
            previous_response_id=previous_response_id,
        )

    def call_streaming(  # type: ignore[override]
        self,
        *,
        model: str,
        max_tokens: int,
        system_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        on_text_delta: Any | None = None,
        on_tool_input_delta: Any | None = None,
    ) -> LLMResponse:
        return super().call_streaming(
            model=self._strip_provider_prefix(model),
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tools=tools,
            messages=messages,
            reasoning_effort=reasoning_effort,
            previous_response_id=previous_response_id,
            on_text_delta=on_text_delta,
            on_tool_input_delta=on_tool_input_delta,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from tool_result content (list of blocks or string)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts) if parts else "(no output)"
    return str(content) if content else "(no output)"


def _anthropic_image_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an Anthropic ``type: image`` block to an OpenAI ``image_url`` block.

    Returns ``None`` if *block* is not a valid Anthropic image block.
    """
    if block.get("type") != "image":
        return None
    src = block.get("source", {})
    if src.get("type") != "base64" or not src.get("data"):
        return None
    media_type = src.get("media_type", "image/png")
    data = src["data"]
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{data}"},
    }


def _extract_images_from_content(content: Any) -> list[dict[str, Any]]:
    """Extract Anthropic image blocks from tool_result content and convert to OpenAI format."""
    if not isinstance(content, list):
        return []
    images: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, dict):
            converted = _anthropic_image_to_openai(item)
            if converted:
                images.append(converted)
    return images


def detect_provider(model: str) -> str:
    """Return ``'vllm'``, ``'openai'``, or ``'anthropic'`` based on model name.

    ``'vllm'`` covers self-hosted, OpenAI-Chat-compatible deployments served
    by vLLM — Qwen3.5 (``"qwen"``) and Alibaba Tongyi DeepResearch
    (``"tongyi"``).  Detection uses the same lowercase substring convention
    relied on by ``system_prompt.py`` and ``run_config.py`` for
    provider-specific prompting/skills.  Tongyi-DeepResearch is
    Qwen3-architecture and is served with the same Qwen tool-call/reasoning
    parsers, so it is treated as the Qwen vLLM family.
    """
    ml = model.lower()
    if "qwen" in ml or "tongyi" in ml:
        return "vllm"
    if model.startswith("openai-") or model.startswith("gpt-"):
        return "openai"
    return "anthropic"


def derive_openai_base_url(base_url: str) -> str:
    """Return an OpenAI-compatible base URL, defaulting to public OpenAI.

    If ``base_url`` already points at an OpenAI-compatible endpoint it is
    used as-is; an Anthropic (or empty) base falls back to public OpenAI.
    """
    if base_url and "anthropic" not in base_url:
        return base_url
    return "https://api.openai.com/v1"


def create_llm_client(
    *,
    model: str,
    api_key: str = "",
    base_url: str = "",
    openai_base_url: str = "",
    openai_api_key: str = "",
    use_azure_openai: bool = False,
    azure_openai_api_key: str = "",
    azure_openai_endpoint: str = "",
    azure_openai_api_version: str = "2025-04-01-preview",
    use_chat_completions: bool = False,
    timeout: float | None = None,
    enable_1m_context_model: bool = False,
    disable_extended_thinking: bool = False,
    # Self-hosted vLLM (Qwen) knobs — only consulted when provider == "vllm".
    vllm_enable_thinking: bool = True,
    vllm_temperature: float = 0.6,
    vllm_top_p: float = 0.95,
    vllm_top_k: int = 20,
    vllm_presence_penalty: float = 0.0,
    vllm_served_model_id: str = "",
    vllm_max_model_len: int = 262_144,
    vllm_max_output_tokens: int = 0,
) -> BaseLLMClient:
    """Factory: return the right client for the model.

    For OpenAI models, defaults to the Responses API
    (``OpenAIResponsesLLMClient``).  Set *use_chat_completions* to ``True``
    to fall back to the legacy Chat Completions API.

    *timeout* sets the per-request HTTP timeout (seconds) for OpenAI clients.
    ``None`` keeps the SDK default (600s).
    """
    provider = detect_provider(model)

    if provider == "vllm":
        # Self-hosted vLLM (OpenAI-compatible).  The endpoint MUST be the vLLM
        # server's /v1 URL — never the Anthropic/Cortex default — or we would
        # silently POST a Qwen model to a closed-model proxy.
        if not base_url or "cortex/anthropic" in base_url or base_url == "https://api.anthropic.com":
            raise ValueError(
                f"vLLM model {model!r} requires the vLLM endpoint to be set "
                "(e.g. llm.agent_model_base_url=http://<host>:<port>/v1); "
                f"got base_url={base_url!r}."
            )
        return VLLMChatLLMClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            enable_thinking=vllm_enable_thinking,
            temperature=vllm_temperature,
            top_p=vllm_top_p,
            top_k=vllm_top_k,
            presence_penalty=vllm_presence_penalty,
            served_model_id=vllm_served_model_id,
            max_model_len=vllm_max_model_len,
            max_output_tokens=vllm_max_output_tokens,
        )

    if provider == "openai":
        if use_azure_openai and azure_openai_endpoint:
            if use_chat_completions:
                return AzureOpenAIChatLLMClient(
                    api_key=azure_openai_api_key or api_key,
                    azure_endpoint=azure_openai_endpoint,
                    api_version=azure_openai_api_version,
                    timeout=timeout,
                )
            return AzureOpenAIResponsesLLMClient(
                api_key=azure_openai_api_key or api_key,
                azure_endpoint=azure_openai_endpoint,
                api_version=azure_openai_api_version,
                timeout=timeout,
            )
        effective_url = openai_base_url or derive_openai_base_url(base_url)
        effective_key = openai_api_key or api_key
        return OpenAIChatLLMClient(api_key=effective_key, base_url=effective_url, timeout=timeout)

    return AnthropicLLMClient(
        api_key=api_key,
        base_url=base_url,
        enable_1m_context_model=enable_1m_context_model,
        disable_extended_thinking=disable_extended_thinking,
    )
