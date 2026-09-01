"""Unit tests for swarm-side image propagation.

These pin the two pieces that let subagents see the orchestrator's
attached images on turn 0:

- ``_images_of`` extracts ``{"type": "image", ...}`` blocks from a
  multimodal user-content list (and returns ``[]`` for plain strings).
- :class:`SwarmContext` stores ``question_images`` separately from the
  text-only ``question`` so downstream spawn paths can hand the blocks
  to every :class:`SubAgent`.
"""

from __future__ import annotations

from typing import Any

from arcticswarm.swarm.orchestrator import _images_of
from arcticswarm.swarm.tools import SwarmContext


_IMAGE_BLOCK: dict[str, Any] = {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "AAAA",
    },
}


def test_images_of_plain_string() -> None:
    assert _images_of("Describe this.") == []


def test_images_of_returns_only_image_blocks() -> None:
    content: list[dict[str, Any]] = [
        _IMAGE_BLOCK,
        {"type": "text", "text": "What is shown?"},
    ]
    images = _images_of(content)
    assert len(images) == 1
    assert images[0]["type"] == "image"
    assert images[0]["source"]["media_type"] == "image/png"


def test_images_of_preserves_order_across_multiple() -> None:
    second = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "BBBB"},
    }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Image 1:"},
        _IMAGE_BLOCK,
        {"type": "text", "text": "Image 2:"},
        second,
        {"type": "text", "text": "Compare."},
    ]
    images = _images_of(content)
    assert [b["source"]["data"] for b in images] == ["AAAA", "BBBB"]


def test_images_of_returns_copies_not_aliases() -> None:
    """The orchestrator must not alias its list into SubAgent state —
    if a subagent ever mutates a block in-place the sibling subagents
    should not see it."""
    content: list[dict[str, Any]] = [_IMAGE_BLOCK, {"type": "text", "text": "Q"}]
    images = _images_of(content)
    images[0]["tag"] = "mutated"
    assert "tag" not in _IMAGE_BLOCK


def _make_ctx(question_images: list[dict[str, Any]] | None) -> SwarmContext:
    """Bypass the unrelated constructor arguments by using object.__new__
    and manually invoking ``__init__`` with only the fields we care
    about — avoids pulling in SnowflakeClient / ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor
    from arcticswarm.config import ArcticswarmConfig
    from arcticswarm.swarm.task import AgentRegistry, TaskBoard

    return SwarmContext(
        bbs=None,
        task_board=TaskBoard(num_agents=1),
        agent_registry=AgentRegistry(),
        config=ArcticswarmConfig(),
        pool=ThreadPoolExecutor(max_workers=1),
        sf_client=None,
        on_swarm_event=None,
        question="text only",
        question_images=question_images,
    )


def test_swarmcontext_stores_question_images_as_list() -> None:
    ctx = _make_ctx([_IMAGE_BLOCK])
    try:
        assert ctx.question_images == [_IMAGE_BLOCK]
        assert isinstance(ctx.question_images, list)
    finally:
        ctx.pool.shutdown(wait=False)


def test_swarmcontext_defaults_to_empty_list_for_text_only() -> None:
    ctx = _make_ctx(None)
    try:
        assert ctx.question_images == []
    finally:
        ctx.pool.shutdown(wait=False)
