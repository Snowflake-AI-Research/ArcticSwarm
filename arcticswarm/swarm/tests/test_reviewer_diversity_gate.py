"""Unit tests for the reviewer-diversity gate (web-research swarms).

The gate (``PrepareReportTool._check_reviewer_diversity_gate``) requires a
VERIFIED ``#consensus`` verdict from BOTH a *builder* reviewer (a subagent that
did first-hand web investigation) and a *dedicated* reviewer (a reasoning
auditor reviewing from the BBS) before the final report is unlocked.  Missing
sources are auto-spawned as targeted reviewer tasks; the gate degrades to
advisory once the remediation budget / run deadline is exhausted.

Also covers the canonical verdict detector
``arcticswarm.swarm.bbs.is_verified_consensus_verdict``.
"""
from __future__ import annotations

from arcticswarm.swarm.bbs import BBS, is_verified_consensus_verdict
from arcticswarm.swarm.task import AgentRegistry, TaskBoard
from arcticswarm.swarm.tools import PrepareReportTool, SendReportTool


# ---------------------------------------------------------------------------
# Verdict detector
# ---------------------------------------------------------------------------


class TestIsVerifiedConsensusVerdict:

    def test_affirmative_templates(self):
        for txt in (
            "Reviewed Romanian Statistical Review — all constraints verified with evidence.",
            "VERIFIED - Reviewed candidate X. All 7 constraints satisfied.",
            "Audit Complete. All constraints verified.",
            "VERIFIED: Reviewed [candidate] — all constraints verified with evidence.",
        ):
            assert is_verified_consensus_verdict(txt) is True, txt

    def test_negative_and_deadend_posts_rejected(self):
        for txt in (
            "CONSENSUS: Task is Unsolvable as Stated - No Valid Answer Exists",
            "CONSENSUS: Palace of Holyroodhouse is DISQUALIFIED as the answer",
            "## Withdrawn Consensus - Hemoglobin Paper Rejected",
            "[Candidate] may be wrong because constraint 3 is unverified. CHALLENGE.",
            # Bare "unverified" with no CHALLENGE header — "VERIFIED" is a
            # substring of "UNVERIFIED", so this must still be rejected.
            "Reviewed [candidate] — constraint 3 (birthplace) remains unverified.",
            "NO VERIFIED CANDIDATE FOUND after exhaustive review.",
            "",
        ):
            assert is_verified_consensus_verdict(txt) is False, txt


# ---------------------------------------------------------------------------
# Fakes for the live swarm state the gate inspects
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, web_calls: int = 0):
        self.tool_calls_by_name = {"web_search": web_calls}


class _FakeSubAgent:
    def __init__(self, web_calls: int = 0):
        self.agent = _FakeAgent(web_calls)


class _FakeCtx:
    """Duck-typed stand-in for ``SwarmContext`` used by the gate."""

    def __init__(self, board: TaskBoard, subagent_map: dict):
        self.task_board = board
        self._subagent_map = subagent_map
        self._tid = 0
        self.spawned: list = []  # TaskSpecs passed to spawn_or_assign

    def next_task_id(self) -> str:
        self._tid += 1
        return f"rt-{self._tid}"

    def spawn_or_assign(self, spec) -> str:
        self.spawned.append(spec)
        return f"worker-{spec.profile}"


def _make_tool(
    *,
    bbs: BBS,
    subagent_map: dict,
    min_dedicated: int = 1,
    min_builder: int = 1,
    max_remediations: int = 2,
    has_web_search: bool = True,
) -> tuple[PrepareReportTool, _FakeCtx]:
    board = TaskBoard()
    ctx = _FakeCtx(board, subagent_map)
    tool = PrepareReportTool(
        task_board=board,
        agent_registry=AgentRegistry(),
        report_tool=SendReportTool(has_web_search=has_web_search),
        agent_tools={},
        bbs=bbs,
        swarm_ctx=ctx,
        min_dedicated_reviewers=min_dedicated,
        min_builder_reviewers=min_builder,
        max_reviewer_remediations=max_remediations,
        has_web_search=has_web_search,
    )
    return tool, ctx


def _post_verdict(bbs: BBS, author: str, *, verified: bool = True) -> None:
    content = (
        "Reviewed [candidate] — all constraints verified with evidence."
        if verified
        else "CONSENSUS: Task is Unsolvable as Stated - No Valid Answer Exists"
    )
    bbs.post(channel="consensus", author=author, content=content)


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------


class TestReviewerDiversityGate:

    def test_both_sources_present_passes(self):
        bbs = BBS()
        smap = {"Builder": _FakeSubAgent(web_calls=12), "Auditor": _FakeSubAgent(web_calls=0)}
        _post_verdict(bbs, "Builder")
        _post_verdict(bbs, "Auditor")
        tool, ctx = _make_tool(bbs=bbs, subagent_map=smap)
        assert tool._check_reviewer_diversity_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []

    def test_builder_only_spawns_dedicated(self):
        bbs = BBS()
        smap = {"Builder": _FakeSubAgent(web_calls=20)}
        _post_verdict(bbs, "Builder")
        tool, ctx = _make_tool(bbs=bbs, subagent_map=smap)
        msg = tool._check_reviewer_diversity_gate(force=False, timed_out=False)
        assert msg is not None and "dedicated" in msg
        assert [s.profile for s in ctx.spawned] == ["reasoning"]
        assert ctx.spawned[0].metadata.get("reviewer_kind") == "dedicated"

    def test_dedicated_only_spawns_builder(self):
        bbs = BBS()
        smap = {"Auditor": _FakeSubAgent(web_calls=0)}
        _post_verdict(bbs, "Auditor")
        tool, ctx = _make_tool(bbs=bbs, subagent_map=smap)
        msg = tool._check_reviewer_diversity_gate(force=False, timed_out=False)
        assert msg is not None and "builder" in msg
        assert [s.profile for s in ctx.spawned] == ["browsing"]
        assert ctx.spawned[0].metadata.get("reviewer_kind") == "builder"

    def test_no_reviewer_spawns_both(self):
        bbs = BBS()
        tool, ctx = _make_tool(bbs=bbs, subagent_map={})
        msg = tool._check_reviewer_diversity_gate(force=False, timed_out=False)
        assert msg is not None
        assert sorted(s.profile for s in ctx.spawned) == ["browsing", "reasoning"]
        assert tool._reviewer_remediation_attempts == 1

    def test_negative_consensus_not_counted(self):
        # A builder posts only a negative/dead-end consensus -> not a verdict,
        # so the builder source is still unmet and gets spawned.
        bbs = BBS()
        smap = {"Builder": _FakeSubAgent(web_calls=30), "Auditor": _FakeSubAgent(web_calls=0)}
        _post_verdict(bbs, "Builder", verified=False)
        _post_verdict(bbs, "Auditor", verified=True)
        tool, ctx = _make_tool(bbs=bbs, subagent_map=smap)
        msg = tool._check_reviewer_diversity_gate(force=False, timed_out=False)
        assert msg is not None and "builder" in msg
        assert [s.profile for s in ctx.spawned] == ["browsing"]

    def test_advisory_degrade_after_budget(self):
        bbs = BBS()
        smap = {"Builder": _FakeSubAgent(web_calls=5)}
        _post_verdict(bbs, "Builder")
        tool, ctx = _make_tool(bbs=bbs, subagent_map=smap, max_remediations=2)
        tool._reviewer_remediation_attempts = 2  # budget exhausted
        assert tool._check_reviewer_diversity_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []  # no more spawns
        assert "WARNING" in tool._reviewer_degrade_note
        assert "dedicated" in tool._reviewer_degrade_note

    def test_timed_out_degrades(self):
        bbs = BBS()
        tool, ctx = _make_tool(bbs=bbs, subagent_map={})
        assert tool._check_reviewer_diversity_gate(force=False, timed_out=True) is None
        assert ctx.spawned == []
        assert "WARNING" in tool._reviewer_degrade_note

    def test_disabled_when_min_zero(self):
        bbs = BBS()
        tool, ctx = _make_tool(bbs=bbs, subagent_map={}, min_dedicated=0, min_builder=0)
        assert tool._check_reviewer_diversity_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []

    def test_non_web_run_is_noop(self):
        bbs = BBS()
        tool, ctx = _make_tool(bbs=bbs, subagent_map={}, has_web_search=False)
        assert tool._check_reviewer_diversity_gate(force=False, timed_out=False) is None
        assert ctx.spawned == []

    def test_force_and_timed_out_escape(self):
        bbs = BBS()
        tool, ctx = _make_tool(bbs=bbs, subagent_map={})
        assert tool._check_reviewer_diversity_gate(force=True, timed_out=True) is None
        assert ctx.spawned == []
