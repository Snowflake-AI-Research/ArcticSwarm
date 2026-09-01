"""Regression tests for :class:`SubAgent` receiving image attachments.

The full subagent construction pulls in an LLM client, tool registry,
skills, BBS, and per-profile system prompts — heavy for a unit test.
These tests therefore verify the two narrow contracts that matter:

1. ``SubAgent.__init__`` accepts ``question_images`` and stores it on a
   public ``self.question_images`` list. Swarm orchestration code
   (see :file:`arcticswarm/swarm/tools.py`) relies on this field name.

2. :meth:`SubAgent._execute_task` wraps the user prompt into a
   multimodal list when ``self.question_images`` is non-empty (and
   leaves it as a plain string otherwise). Instead of running the full
   task pipeline, we read the method source and confirm the wrap branch
   is present in the exact expected shape. This is fragile to source
   drift but stable against business-logic churn — and crucially, it
   would catch a refactor that silently drops the image attachments.
"""

from __future__ import annotations

import inspect
import textwrap

from arcticswarm.swarm.teammate import SubAgent


def test_subagent_init_accepts_question_images() -> None:
    sig = inspect.signature(SubAgent.__init__)
    assert "question_images" in sig.parameters, (
        "SubAgent.__init__ must accept question_images so swarm "
        "orchestration can forward the original question's image "
        "blocks to each subagent's first user message."
    )
    param = sig.parameters["question_images"]
    annotation = str(param.annotation)
    assert "list" in annotation, (
        "question_images should be typed as list[dict] | None."
    )
    assert param.default is None, (
        "question_images must default to None so text-only callers "
        "(every non-HLE-image case) do not have to pass it."
    )


def test_execute_task_wraps_prompt_when_images_present() -> None:
    """Pin the prompt-wrap code block in ``SubAgent._execute_task``.

    If someone rewrites this logic, this test fires and tells them
    exactly which invariants the swarm image-propagation fix relies on:
    image blocks come first, the profile-built text is the final block,
    and the empty-image case stays a plain string.
    """
    src = inspect.getsource(SubAgent._execute_task)
    src_collapsed = textwrap.dedent(src)

    assert "if self.question_images:" in src_collapsed, (
        "_execute_task must branch on self.question_images; without this "
        "branch subagents run blind on image questions."
    )
    assert "*self.question_images" in src_collapsed or (
        "self.question_images" in src_collapsed and "*" in src_collapsed
    ), (
        "_execute_task must unpack self.question_images into the leading "
        "blocks of the multimodal list (Anthropic convention)."
    )
    assert '{"type": "text", "text": prompt_text}' in src_collapsed, (
        "The profile-built text must appear as the final text block so "
        "the LLM sees the task instructions after the image context."
    )
    assert "prompt = prompt_text" in src_collapsed, (
        "The empty-images fallback must keep prompt as the plain "
        "string returned by get_profile_task_prompt — this guards the "
        "text-only fast path that every non-image case takes today."
    )


def test_execute_task_preserves_plain_string_branch_ordering() -> None:
    """The if/else order matters: if we accidentally set ``prompt`` to a
    list *before* checking ``self.question_images`` we would break the
    text-only fast path. Verify the conditional wraps *after* calling
    ``get_profile_task_prompt`` and before handing to the agent."""
    src = inspect.getsource(SubAgent._execute_task)
    gp_idx = src.find("get_profile_task_prompt(")
    wrap_idx = src.find("if self.question_images:")
    run_idx = src.find("self.agent.run_turn_streaming(prompt")

    assert gp_idx != -1 and wrap_idx != -1 and run_idx != -1, (
        "Expected get_profile_task_prompt call, the image-wrap branch, "
        "and the run_turn_streaming call all to exist in execute_task."
    )
    assert gp_idx < wrap_idx < run_idx, (
        "The image-wrap branch must sit between the profile prompt "
        "build and the agent handoff — otherwise the wrapping never "
        "reaches the LLM or never sees the profile text."
    )
