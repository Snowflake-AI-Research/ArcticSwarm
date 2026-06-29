"""Regression tests for Agent fallback provider-filtering helpers.

When the primary model (OpenAI Responses API) returns an empty response
and we fall back to ``claude-4-sonnet``, the stored history contains
OpenAI-specific ``reasoning`` blocks that Anthropic does not accept.
Sending them through unchanged triggered a schema-level 400 on the
Cortex proxy:

    messages.1.content: Field required

``Agent._filter_blocks_for_provider`` / ``Agent._messages_for_provider``
strip those blocks before the fallback request goes out.  These tests
pin that invariant so the bug cannot silently return.
"""
from __future__ import annotations

from arcticswarm.agent import Agent


def test_filter_blocks_for_anthropic_strips_reasoning_blocks():
    blocks = [
        {"type": "reasoning", "id": "rs_abc", "summary": []},
        {"type": "tool_use", "id": "call_1", "name": "load_skill", "input": {}},
    ]
    kept = Agent._filter_blocks_for_provider(blocks, "anthropic")
    assert [b["type"] for b in kept] == ["tool_use"], (
        "reasoning blocks must be stripped when the target provider is "
        "Anthropic — otherwise the Messages API rejects the request."
    )


def test_filter_blocks_for_anthropic_keeps_anthropic_native_types():
    blocks = [
        {"type": "text", "text": "hello"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
        {"type": "tool_use", "id": "call_1", "name": "noop", "input": {}},
        {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
        {"type": "thinking", "thinking": "...", "signature": "sig"},
        {"type": "redacted_thinking", "data": "..."},
    ]
    kept = Agent._filter_blocks_for_provider(blocks, "anthropic")
    assert [b["type"] for b in kept] == [
        "text",
        "image",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
    ], "Anthropic-native block types must pass through unchanged."


def test_filter_blocks_for_openai_is_identity():
    blocks = [
        {"type": "reasoning", "id": "rs_abc", "summary": []},
        {"type": "tool_use", "id": "call_1", "name": "noop", "input": {}},
    ]
    kept = Agent._filter_blocks_for_provider(blocks, "openai")
    assert kept == blocks, (
        "OpenAI targets must preserve reasoning blocks so chain-of-thought "
        "chaining across turns continues to work."
    )


def test_messages_for_anthropic_drops_reasoning_only_assistant_turn():
    """An assistant message containing *only* a reasoning block becomes
    empty after filtering and must be dropped, not sent with content=[]
    (which is what produced the 400)."""
    messages = [
        {"role": "user", "content": "Solve this."},
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
            ],
        },
        {"role": "user", "content": "Please continue."},
    ]
    out = Agent._messages_for_provider(messages, "anthropic")
    for msg in out:
        content = msg.get("content")
        assert content not in (None, "", []), (
            "no message should survive provider filtering with empty content"
        )
        if isinstance(content, list):
            assert all(
                not (isinstance(b, dict) and b.get("type") == "reasoning")
                for b in content
            ), "reasoning blocks must not leak into Anthropic-bound history"


def test_messages_for_anthropic_preserves_tool_use_result_pair():
    """When the assistant turn still has a non-reasoning block after
    filtering, the tool_use -> tool_result pairing must survive so
    Anthropic's structural validation passes."""
    messages = [
        {"role": "user", "content": "Look something up."},
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "web_search",
                    "input": {"query": "x"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "ok",
                    "is_error": False,
                },
            ],
        },
    ]
    out = Agent._messages_for_provider(messages, "anthropic")
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "user"], (
        f"Expected user/assistant/user alternation, got {roles!r}"
    )
    asst_content = out[1]["content"]
    assert isinstance(asst_content, list)
    assert [b["type"] for b in asst_content] == ["tool_use"], (
        "tool_use block must survive the reasoning-strip pass so the "
        "following tool_result isn't orphaned."
    )


def test_fallback_uses_provider_filter():
    """Source-level invariant: ``_fallback_on_empty_response`` must route
    the history through ``_messages_for_provider`` so the reasoning-block
    400 can't silently come back."""
    import inspect

    src = inspect.getsource(Agent._fallback_on_empty_response)
    assert "_messages_for_provider" in src, (
        "Agent._fallback_on_empty_response must call "
        "_messages_for_provider(...) before sending the history to the "
        "fallback model; otherwise OpenAI ``reasoning`` blocks leak to "
        "Anthropic and cause 400 errors."
    )
    assert "detect_provider(fallback_model)" in src, (
        "Provider detection must be driven by the fallback_model so the "
        "correct block types are stripped (or preserved)."
    )
