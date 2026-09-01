"""Tests for the eval-awareness contamination filter in ``Agent._execute_tool``.

The filter lives in ``arcticswarm/agent.py`` and rewrites ``web_search`` /
``web_fetch`` results whose output contains any of the benchmark-leak
keywords in ``_CONTAMINATION_KEYWORDS`` (case-insensitive substring match).

We care about two things:

1. **Coverage** — the canonical benchmark name (``browsecomp``) is present in
   the keyword list.
2. **No over-broad entries** — bare ``"hle"`` would false-positive on common
   English words like ``"athlete"`` or ``"Ashley"``.  The filter must
   therefore only use qualified, high-precision tokens.

Both are enforced at the source / constant level so that any future edit
which adds an over-broad token fails loudly in CI.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Contract tests — the keyword list itself
# ---------------------------------------------------------------------------


def test_contamination_keywords_contain_browsecomp():
    """BrowseComp coverage must never regress."""
    from arcticswarm.agent import _CONTAMINATION_KEYWORDS

    assert "browsecomp" in _CONTAMINATION_KEYWORDS


def test_contamination_keywords_avoid_bare_hle_false_positive():
    """Bare ``"hle"`` would false-positive on ``athlete``, ``Ashley``, etc.

    Any future edit that adds the three-letter token as a top-level entry
    must trip this test.  Use qualified forms instead.
    """
    from arcticswarm.agent import _CONTAMINATION_KEYWORDS

    assert "hle" not in _CONTAMINATION_KEYWORDS, (
        "Bare 'hle' is too broad — it is a substring of common English "
        "words ('athlete', 'Ashley', ...)."
    )

    # Sanity: any keyword containing "hle" must be at least 8 characters so it
    # can never match inside a normal English word.  (No HLE keywords remain,
    # so this loop is vacuously true — it guards against re-introducing one.)
    hle_entries = [k for k in _CONTAMINATION_KEYWORDS if "hle" in k]
    for entry in hle_entries:
        assert len(entry) >= 8, (
            f"HLE keyword {entry!r} is suspiciously short and may "
            "false-positive on common words.  Require ≥ 8 characters."
        )


# ---------------------------------------------------------------------------
# Source contract — the filter call site still uses the keyword list and
# placeholder so the above coverage actually applies at runtime.
# ---------------------------------------------------------------------------


def test_execute_tool_still_applies_contamination_filter():
    """Grep the source of ``Agent._execute_tool`` to confirm the filter is
    still wired up.  We avoid instantiating an ``Agent`` (which needs a full
    config + LLM client) and instead assert the call site directly.
    """
    import inspect

    from arcticswarm.agent import Agent

    src = inspect.getsource(Agent._execute_tool)
    # Must reference the keyword list, the contaminated-tools set, and the
    # placeholder — i.e. the filter is still applied on the hot path.
    assert "_CONTAMINATION_KEYWORDS" in src
    assert "_CONTAMINATED_TOOLS" in src
    assert "_CONTAMINATION_PLACEHOLDER" in src
    assert "contamination_excluded" in src, (
        "Filter must still tag the metadata with 'contamination_excluded' "
        "so the eval harness can report how many results were redacted."
    )
