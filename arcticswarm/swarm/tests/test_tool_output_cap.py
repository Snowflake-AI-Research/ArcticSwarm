"""Tests for the tool-output size caps that prevent BrowseComp context overflow.

Two layers enforce the same per-result budget so a single huge page / docling
PDF / uncapped ContentCompactor selection can never push the agent context past
the model window before the (reactive, post-hoc) context budget can compact:

1. ``ContentCompactor._assemble_selected(..., max_chars=...)`` and the
   ``max_output_chars`` ctor arg — bound the compactor's re-assembled selected
   output (the compactor LLM only returns chunk *indices*; the agent-visible
   text is the verbatim re-assembly, which was previously unbounded).
2. ``Agent._cap_tool_output`` — the last-word backstop applied AFTER the
   optional compactor / source scorer, so it bounds *every* path (raw content,
   compactor output, the 2K fallback, cross-agent cache hits), preserving a
   trailing ``[Source Quality: ...]`` annotation.

Token counts use the codebase-wide ~4-chars/token estimate, so a 5k-token cap
is a 20 000-char budget.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from arcticswarm.tools.base import ToolResult
from arcticswarm.tools.content_compactor import (
    ContentCompactor,
    _chunk_text,
)


# ---------------------------------------------------------------------------
# ContentCompactor: _assemble_selected output cap
# ---------------------------------------------------------------------------


def _make_chunks(n: int, chunk_len: int = 1000) -> list[str]:
    """n distinct chunks of ``chunk_len`` chars each (distinct so joins are real)."""
    return [chr(ord("A") + (i % 26)) * chunk_len for i in range(n)]


def test_assemble_selected_caps_output_to_budget():
    chunks = _make_chunks(50, 1000)  # 50 KB of selectable content
    indices = list(range(50))
    cap = 8000  # ~2k tokens

    out = ContentCompactor._assemble_selected(chunks, indices, max_chars=cap)

    # Whole chunks are kept, so we never exceed the budget by more than the
    # trailing marker (joined with "\n\n").
    assert len(out) <= cap + 100, f"output {len(out)} chars exceeds budget {cap}"
    assert "[... selection truncated to output cap]" in out
    # Far smaller than the uncapped 50 KB it would otherwise return.
    assert len(out) < 50_000


def test_assemble_selected_no_cap_returns_everything():
    chunks = _make_chunks(20, 1000)
    indices = list(range(20))

    out = ContentCompactor._assemble_selected(chunks, indices, max_chars=0)

    assert "[... selection truncated to output cap]" not in out
    # All 20 chunks present (joined) → well over any single-chunk size.
    assert len(out) > 18_000


def test_assemble_selected_small_selection_unaffected_by_cap():
    chunks = _make_chunks(10, 500)
    indices = [0, 2]  # non-adjacent → an "[... omitted ...]" gap marker
    cap = 20_000  # far above the tiny selection

    out = ContentCompactor._assemble_selected(chunks, indices, max_chars=cap)

    assert "[... selection truncated to output cap]" not in out
    assert "[... omitted ...]" in out
    assert chunks[0] in out and chunks[2] in out


# ---------------------------------------------------------------------------
# ContentCompactor: ctor arg + end-to-end compact() cap (LLM stubbed)
# ---------------------------------------------------------------------------


def test_ctor_accepts_max_output_chars_without_breaking_super():
    # max_output_chars is keyword-only and must NOT be forwarded to SourceScorer.
    cc = ContentCompactor(max_output_chars=20_000)
    assert cc._max_output_chars == 20_000

    cc_default = ContentCompactor()
    assert cc_default._max_output_chars == 0  # 0 => no cap


def test_compact_end_to_end_respects_output_cap(monkeypatch):
    cap = 20_000  # 5k tokens
    cc = ContentCompactor(max_output_chars=cap)

    # A large page that chunks into many pieces.
    content = ". ".join(f"Sentence number {i} with some filler text" for i in range(4000))
    n_chunks = len(_chunk_text(content))
    assert n_chunks > 20  # sanity: enough chunks to exceed the cap if all selected

    # Stub the compactor LLM to "select every chunk" (worst case for output size).
    def _fake_call_llm(system_prompt, prompt, **kwargs):
        return json.dumps({
            "scores": {"relevance": 9, "answerability": 9, "authority": 9, "data_density": 9},
            "selected_indices": list(range(n_chunks)),
        })

    monkeypatch.setattr(cc, "_call_llm", _fake_call_llm)

    out, scores = cc.compact("what is the answer?", "http://example.com", content)

    assert len(out) <= cap + 100, f"compacted output {len(out)} chars exceeds cap {cap}"
    assert "[... selection truncated to output cap]" in out
    assert scores  # scores still parsed through


# ---------------------------------------------------------------------------
# Agent._cap_tool_output backstop
# ---------------------------------------------------------------------------


def _cap(name: str, result: ToolResult, cap_tokens: int = 5000) -> None:
    """Invoke Agent._cap_tool_output with a minimal stand-in for ``self``."""
    from arcticswarm.agent import Agent

    stub = SimpleNamespace(config=SimpleNamespace(max_tool_output_tokens=cap_tokens))
    Agent._cap_tool_output(stub, name, result)


def test_cap_tool_output_truncates_web_fetch():
    result = ToolResult(output="x" * 500_000)
    _cap("web_fetch", result, cap_tokens=5000)

    assert len(result.output) <= 5000 * 4
    assert "truncated to ~5000 tokens" in result.output


def test_cap_tool_output_preserves_source_quality_annotation():
    annotation = (
        "\n[Source Quality: relevance=9/10, answerability=8/10, "
        "authority=9/10, data_density=8/10 (composite=34/40)]"
    )
    result = ToolResult(output="y" * 500_000 + annotation)
    _cap("pdf_read", result, cap_tokens=5000)

    assert len(result.output) <= 5000 * 4
    assert result.output.endswith(annotation), "trailing [Source Quality] annotation must survive truncation"
    assert "truncated to ~5000 tokens" in result.output


def test_cap_tool_output_ignores_other_tools_and_errors():
    big = "z" * 500_000

    other = ToolResult(output=big)
    _cap("bash", other, cap_tokens=5000)
    assert other.output == big  # non web_fetch/pdf_read untouched

    errored = ToolResult(error="boom", is_error=True, output=big)
    _cap("web_fetch", errored, cap_tokens=5000)
    assert errored.output == big  # error results untouched


def test_cap_tool_output_disabled_when_zero():
    big = "w" * 500_000
    result = ToolResult(output=big)
    _cap("web_fetch", result, cap_tokens=0)
    assert result.output == big  # cap=0 => no-op


def test_cap_tool_output_leaves_small_output_unchanged():
    small = "short content"
    result = ToolResult(output=small)
    _cap("web_fetch", result, cap_tokens=5000)
    assert result.output == small
