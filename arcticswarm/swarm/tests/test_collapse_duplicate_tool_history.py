"""Unit tests for collapse_duplicate_tool_history.

``Agent._collapse_duplicate_tool_results`` rewrites an outbound message slice so
that, for each deduped tool signature issued more than ``dup_history_keep_last``
times, the bulky body of all-but-the-last-N ``tool_result`` blocks is replaced
with a compact stub.  Tool_use blocks are untouched, so tool_use<->tool_result
pairing and role alternation are preserved.  Tested in isolation via a stub so
no full Agent construction is needed.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

from arcticswarm.agent import Agent


def _stub(keep_last: int = 1):
    s = SimpleNamespace(
        config=SimpleNamespace(dup_history_keep_last=keep_last),
        _dup_collapsed_count=0,
    )
    s._dup_signature = types.MethodType(Agent._dup_signature, s)
    return s


def _ws_turn(tid: str, query: str, result_body: str) -> list[dict]:
    return [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": tid, "name": "web_search", "input": {"query": query}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": result_body},
        ]},
    ]


def test_collapses_all_but_last_duplicate_result():
    s = _stub(keep_last=1)
    msgs = (
        _ws_turn("t1", "same query", "BIG RESULT A" * 50)
        + _ws_turn("t2", "same query", "BIG RESULT B" * 50)
        + _ws_turn("t3", "same query", "BIG RESULT C" * 50)
    )
    out = Agent._collapse_duplicate_tool_results(s, msgs)

    def result_body(turn_idx_user_msg):
        return out[turn_idx_user_msg]["content"][0]["content"]

    # user msgs are at indices 1, 3, 5
    assert "collapsed to save context" in result_body(1)   # t1 stubbed
    assert "collapsed to save context" in result_body(3)   # t2 stubbed
    assert result_body(5) == "BIG RESULT C" * 50            # t3 kept in full
    assert s._dup_collapsed_count == 2
    # tool_use blocks are untouched (pairing preserved)
    assert out[0]["content"][0]["type"] == "tool_use"
    assert out[4]["content"][0]["id"] == "t3"


def test_keep_last_n_kept():
    s = _stub(keep_last=2)
    msgs = (
        _ws_turn("t1", "q", "A" * 20)
        + _ws_turn("t2", "q", "B" * 20)
        + _ws_turn("t3", "q", "C" * 20)
    )
    out = Agent._collapse_duplicate_tool_results(s, msgs)
    assert "collapsed to save context" in out[1]["content"][0]["content"]  # t1 stubbed
    assert out[3]["content"][0]["content"] == "B" * 20                     # t2 kept
    assert out[5]["content"][0]["content"] == "C" * 20                     # t3 kept
    assert s._dup_collapsed_count == 1


def test_distinct_queries_not_collapsed():
    s = _stub(keep_last=1)
    msgs = (
        _ws_turn("t1", "first distinct question", "A" * 20)
        + _ws_turn("t2", "second different question", "B" * 20)
    )
    out = Agent._collapse_duplicate_tool_results(s, msgs)
    assert out[1]["content"][0]["content"] == "A" * 20
    assert out[3]["content"][0]["content"] == "B" * 20
    assert s._dup_collapsed_count == 0


def test_returns_input_when_no_tool_signatures():
    s = _stub()
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert Agent._collapse_duplicate_tool_results(s, msgs) is msgs
