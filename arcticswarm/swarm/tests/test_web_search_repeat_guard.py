"""Unit tests for the web_search repeat-query guard.

The guard short-circuits a subagent that re-issues the SAME (normalized) query,
returning the prior result plus a nudge to change strategy instead of re-hitting
the search provider.  This prevents the degenerate query loops observed with
smaller open models (e.g. Qwen firing one exact query 20-40+ times).
"""

from __future__ import annotations

from arcticswarm.tools.base import ToolResult
from arcticswarm.tools.web_search import WebSearchTool


def _tool() -> tuple[WebSearchTool, dict[str, int]]:
    """A WebSearchTool whose provider call is stubbed and counted."""
    tool = WebSearchTool(api_key="dummy")
    calls = {"n": 0}

    def fake_impl(*, query, count=5, country=None, safesearch="moderate", **kw):
        calls["n"] += 1
        return ToolResult(
            output=f"Top 1 result(s) for: {query}\n\n1. Example\n   URL: http://e.com",
            metadata={"search_source": "brave"},
        )

    tool._search_impl = fake_impl  # type: ignore[method-assign]
    return tool, calls


def test_first_query_runs_and_is_memoized():
    tool, calls = _tool()
    res = tool.execute(query="needle in haystack 2026")
    assert not res.is_error
    assert res.metadata.get("search_source") == "brave"
    assert calls["n"] == 1


def test_exact_repeat_is_blocked_with_prior_snippet():
    tool, calls = _tool()
    tool.execute(query="needle in haystack 2026")
    res = tool.execute(query="needle in haystack 2026")
    # Provider was NOT hit a second time.
    assert calls["n"] == 1
    assert res.metadata.get("search_source") == "repeat_guard"
    assert res.metadata.get("repeat_count") == 2
    # The prior result snippet is surfaced back to the model.
    assert "already ran this exact query 2 times" in res.output
    assert "http://e.com" in res.output
    assert "DIFFERENT query" in res.output


def test_repeat_count_increments():
    tool, _ = _tool()
    tool.execute(query="q")
    tool.execute(query="q")
    res = tool.execute(query="q")
    assert res.metadata.get("repeat_count") == 3
    assert tool._repeat_blocked == 2


def test_case_and_whitespace_variants_count_as_repeat():
    tool, calls = _tool()
    tool.execute(query='Karen Chavez "Tri-State" 2026')
    res = tool.execute(query='  karen   chavez "tri-state"   2026 ')
    assert calls["n"] == 1
    assert res.metadata.get("search_source") == "repeat_guard"


def test_different_query_not_blocked():
    tool, calls = _tool()
    tool.execute(query="first query")
    res = tool.execute(query="genuinely different query")
    assert calls["n"] == 2
    assert res.metadata.get("search_source") == "brave"


def test_error_results_not_memoized():
    """A query that errors (no provider hit) must not be cached as a repeat."""
    tool = WebSearchTool(api_key="dummy")
    calls = {"n": 0}

    def err_impl(*, query, count=5, country=None, safesearch="moderate", **kw):
        calls["n"] += 1
        return ToolResult(error="boom", is_error=True)

    tool._search_impl = err_impl  # type: ignore[method-assign]
    tool.execute(query="x")
    res = tool.execute(query="x")
    # Both attempts reach the impl (the first was an error, so not memoized).
    assert calls["n"] == 2
    assert res.is_error


def test_reset_clears_guard_state():
    tool, calls = _tool()
    tool.execute(query="q")
    tool.execute(query="q")
    assert tool._repeat_blocked == 1
    tool.log_and_reset_stats()
    assert tool._repeat_blocked == 0
    assert tool._query_history == {}
    assert tool._family_counts == {}
    # After reset the same query runs fresh again.
    tool.execute(query="q")
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Escalation + hard-stop (forced bail)
# ---------------------------------------------------------------------------


def test_escalation_to_is_error_after_threshold():
    """Soft nudge for the first repeats, then is_error once it keeps repeating."""
    tool, _ = _tool()
    tool.execute(query="q")                 # run (count 1)
    r2 = tool.execute(query="q")            # count 2 -> gentle nudge
    r3 = tool.execute(query="q")            # count 3 -> escalate to is_error
    assert not r2.is_error
    assert r3.is_error
    assert not r3.metadata.get("force_stop")


def test_exact_repeat_force_stop():
    """An exact query hammered to the hard cap emits force_stop metadata."""
    tool, _ = _tool()
    results = [tool.execute(query="stuck query") for _ in range(7)]
    # First runs, then blocks; the 6th *block* (count==6) trips the hard stop.
    forced = [r for r in results if r.metadata.get("force_stop")]
    assert forced, "expected a force_stop once the exact repeat hit the hard cap"
    assert forced[0].is_error
    assert "STOP" in forced[0].output


def test_near_dup_family_force_stop():
    """A runaway REFORMULATION loop (distinct strings, one intent) force-stops.

    Each query is a *different* exact string (year sweep), so exact-match never
    fires — only the near-duplicate family counter catches it.
    """
    tool, calls = _tool()
    forced = None
    for i in range(WebSearchTool._NEARDUP_HARD_STOP + 2):
        r = tool.execute(query=f"palace church destroyed siege {2000 + i}")
        if r.metadata.get("force_stop"):
            forced = r
            break
    assert forced is not None, "near-dup family should force_stop when runaway"
    assert forced.is_error
    # exact-match alone would never have blocked these distinct strings
    assert calls["n"] >= WebSearchTool._NEARDUP_HARD_STOP - 1


def test_legit_year_sweep_below_cap_not_blocked():
    """A modest year-sweep (well under the cap) is genuine search — never blocked."""
    tool, calls = _tool()
    n = 20  # < _NEARDUP_HARD_STOP (40); real legit sweeps reach ~27
    results = [tool.execute(query=f"battle of acre {1100 + i}") for i in range(n)]
    assert calls["n"] == n
    assert not any(r.metadata.get("force_stop") for r in results)
    assert not any(r.metadata.get("search_source") == "repeat_guard" for r in results)


def test_diverse_search_never_blocked():
    """Genuinely different queries always run; the guard stays silent."""
    tool, calls = _tool()
    results = [tool.execute(query=f"distinct topic number {i} alpha beta gamma") for i in range(30)]
    assert calls["n"] == 30
    assert all(r.metadata.get("search_source") == "brave" for r in results)


def test_hard_stop_disabled_never_forces():
    """With hard_stop=False, the guard still blocks/escalates but never bails."""
    tool = WebSearchTool(api_key="dummy", hard_stop=False)
    tool._search_impl = lambda **kw: ToolResult(  # type: ignore[method-assign]
        output="Top 1 result(s) for: x\n\n1. e\n   URL: http://e.com",
        metadata={"search_source": "brave"},
    )
    results = [tool.execute(query="loop me") for _ in range(10)]
    assert not any(r.metadata.get("force_stop") for r in results)
    # still blocks the exact repeats (no force, but is_error escalation present)
    assert any(r.metadata.get("search_source") == "repeat_guard" for r in results)


def test_family_key_collapses_reformulations():
    """The near-dup signature collapses year-swap / reorder / quote-toggle."""
    k = WebSearchTool._family_key
    base = k('"Ding Lei" children born 2020 2021 2022')
    assert k('"Ding Lei" children born 2013 2014') == base          # year-swap
    assert k('born children "Ding Lei"') == base                    # reorder
    assert k('Ding Lei children born') == base                      # quote-toggle
    # genuinely different intent must NOT collapse
    assert k("Steve Jobs Apple founding year") != base


# ---------------------------------------------------------------------------
# Agent turn-loop honors the force_stop bail signal
# ---------------------------------------------------------------------------


def test_agent_turn_terminates_on_force_stop_metadata():
    from arcticswarm.agent import Agent
    tcs = [{"name": "web_search", "id": "t1"}]
    forced = [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
               "metadata": {"search_source": "repeat_guard", "force_stop": True}}]
    not_forced = [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                   "metadata": {"search_source": "repeat_guard", "repeat_count": 3}}]
    assert Agent._tool_batch_terminates_turn(tcs, forced) is True
    assert Agent._tool_batch_terminates_turn(tcs, not_forced) is False
