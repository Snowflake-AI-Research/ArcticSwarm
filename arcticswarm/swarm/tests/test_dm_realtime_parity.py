"""Focused regression tests for plain DM realtime parity with Duo-style flow."""

from __future__ import annotations

import inspect
from pathlib import Path

from arcticswarm.run_config import RunConfig, SwarmConfig, ToolsConfig
from arcticswarm.swarm.orchestrator import SwarmOrchestrator
from arcticswarm.swarm.profiles import resolve_orchestrator_skill
from arcticswarm.swarm.prompts import (
    build_comm_protocol_inline,
    build_orchestrator_system_prompt,
)
from arcticswarm.swarm.teammate import SubAgent

# ``skills/`` lives alongside ``swarm/`` inside the ``arcticswarm`` package.
# From this test file (``arcticswarm/swarm/tests/``), the package root is
# ``parents[2]``.  Computing the path relatively keeps the test portable.
_SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


def test_dm_realtime_orchestrator_prompt_uses_submit_or_wait():
    prompt = build_orchestrator_system_prompt(
        has_bbs=False,
        has_dm=True,
        orchestrator_realtime=True,
        has_web_search=True,
    )
    assert "I will wait." in prompt, (
        "Plain DM realtime prompt must tell the orchestrator how to wait "
        "without polling."
    )
    assert "do NOT use `prepare_report`" in prompt, (
        "Plain DM realtime prompt must remove the prepare_report barrier "
        "and point the leader at direct report submission."
    )
    assert "`list_tasks` plus incoming DM updates" in prompt, (
        "Plain DM realtime prompt must explicitly frame list_tasks as the "
        "readiness/status check tool."
    )
    assert "do NOT submit from memory alone" in prompt, (
        "Plain DM realtime prompt must prevent orchestrator-only memory "
        "answers on non-trivial benchmark questions."
    )
    assert "created at" in prompt and "`create_task` OR performed" in prompt, (
        "Plain DM realtime prompt must require either delegation or a "
        "concrete verification tool call before final submission."
    )


def test_dm_realtime_followup_light_edit_skips_prepare_report():
    prompt = build_orchestrator_system_prompt(
        has_bbs=False,
        has_dm=True,
        orchestrator_realtime=True,
        is_followup=True,
        turn_number=2,
        has_web_search=True,
    )
    assert (
        "skip task creation and go directly to `send_user_markdown_report`"
        in prompt
    ), "Follow-up light-edit path must not force prepare_report in DM realtime."


def test_dm_realtime_orchestrator_source_wires_direct_report_with_strict_drain():
    src = inspect.getsource(SwarmOrchestrator.run_swarm_turn)
    assert "dm_realtime_direct_report = orchestrator_realtime and has_dm and not has_bbs" in src
    assert 'agent._tools["send_user_markdown_report"] = report_tool' in src, (
        "Plain DM realtime branch must register send_user_markdown_report "
        "directly on the orchestrator toolkit."
    )
    assert "strict_dm_drain=True" in src, (
        "Plain DM realtime direct report path must keep the unread-DM "
        "submission safety barrier."
    )


def test_dm_subagent_inline_protocol_mentions_list_tasks():
    inline = build_comm_protocol_inline(has_bbs=False, has_dm=True)
    assert "`list_tasks`" in inline, (
        "DM subagent inline coordination guidance must mention list_tasks "
        "so workers know they can inspect task status."
    )


def test_dm_subagent_profile_filter_keeps_list_tasks():
    src = inspect.getsource(SubAgent._apply_profile)
    assert '"list_tasks"' in src, (
        "Subagent profile filtering must preserve list_tasks in the swarm "
        "infrastructure tool set."
    )


def test_dm_realtime_skill_doc_matches_direct_report_flow():
    skill = (
        _SKILLS_DIR / "swarm-orchestration-dm-realtime" / "SKILL.md"
    ).read_text()
    assert "send_user_markdown_report" in skill
    assert "Available from the start." in skill, (
        "DM realtime skill doc must describe direct report access."
    )
    assert "`I will wait.`" in skill, (
        "DM realtime skill doc must describe the visible wait line."
    )


def test_exec_enabled_prompt_mode_allows_direct_execution():
    prompt = build_orchestrator_system_prompt(
        has_bbs=False,
        has_dm=True,
        orchestrator_realtime=True,
        has_web_search=True,
        orchestrator_prompt_mode="exec_enabled",
    )
    assert "You MAY execute work directly with your own tools" in prompt, (
        "exec_enabled prompt mode must explicitly permit orchestrator-side "
        "execution."
    )
    assert "All execution MUST be delegated via `create_task`." not in prompt, (
        "exec_enabled prompt mode must replace the delegate-only rule."
    )


def test_run_config_bridges_orchestrator_prompt_mode():
    cfg = RunConfig(swarm=SwarmConfig(orchestrator_prompt_mode="exec_enabled"))
    flat = cfg.to_arcticswarm_config()
    assert flat.orchestrator_prompt_mode == "exec_enabled"


def test_run_config_bridges_orchestrator_skills():
    cfg = RunConfig(
        tools=ToolsConfig(orchestrator_skills=["coding-execution", "web-research"])
    )
    flat = cfg.to_arcticswarm_config()
    assert flat.orchestrator_skills == ["coding-execution", "web-research"]


def test_orchestrator_source_appends_extra_orchestrator_skills():
    src = inspect.getsource(SwarmOrchestrator.run_swarm_turn)
    assert "[orch_skill, *self.config.orchestrator_skills]" in src, (
        "The orchestrator load_skill tool must include the mode-specific skill "
        "plus any extra YAML-configured orchestrator skills."
    )


def test_duo_leader_source_appends_extra_orchestrator_skills():
    src = inspect.getsource(SwarmOrchestrator._run_duo_turn)
    assert "*self.config.orchestrator_skills" in src, (
        "The Duo leader load_skill tool must include extra YAML-configured "
        "orchestrator skills in addition to the duo profile skills."
    )


def test_dm_realtime_resolves_expected_orchestrator_skill():
    # Subagents are always spawned dynamically now, so DM-only realtime
    # resolves to the dynamic-DM orchestration skill.
    skill = resolve_orchestrator_skill(
        has_bbs=False,
        has_web_search=True,
        orchestrator_realtime=True,
    )
    assert skill == "swarm-orchestration-dynamic-dm", (
        "DM-only mode must load the dynamic DM orchestration skill."
    )


# ---------------------------------------------------------------------------
# DM-mode multi-subagent verification contract.
#
# DM realtime mode runs as "multi-subagent Duo": orchestrator delegates
# parallel tasks, then orchestrates a verification round before submitting.
# These tests pin down the prompt/skill contract that drives that pattern,
# so future edits don't silently regress DM mode back into a fan-out-and-
# submit topology with no auditor (which empirically tied with single-agent
# accuracy on HLE).
# ---------------------------------------------------------------------------

_DM_REALTIME_SKILL_PATH = (
    _SKILLS_DIR / "swarm-orchestration-dm-realtime" / "SKILL.md"
)
_DM_COORDINATION_SKILL_PATH = _SKILLS_DIR / "dm-coordination" / "SKILL.md"


def test_dm_realtime_skill_strips_stale_prepare_report_rules():
    """The skill once said the orchestrator MUST call ``prepare_report``
    before ``send_user_markdown_report`` and after timeouts.  ``prepare_report``
    is no longer registered on the DM realtime orchestrator (see
    ``dm_realtime_direct_report`` branch in ``run_swarm_turn``), so the skill
    must not contradict the live system prompt.
    """
    skill = _DM_REALTIME_SKILL_PATH.read_text()
    forbidden = (
        "MUST call `prepare_report`",
        "use `prepare_report(force=true)`",
        "After `prepare_report` returns",
        'After `prepare_report` returns "not ready"',
    )
    for snippet in forbidden:
        assert snippet not in skill, (
            f"Stale prepare_report instruction still present in DM realtime "
            f"skill: {snippet!r}"
        )


def test_dm_realtime_skill_documents_explicit_verification_round():
    """DM mode has no built-in auditor; the orchestrator is responsible for
    running a verification round before submitting.  Pin both the
    peer-DM-challenge path and the verifier-task path so the skill can't
    silently lose them.
    """
    skill = _DM_REALTIME_SKILL_PATH.read_text()
    assert "verification round" in skill.lower(), (
        "DM realtime skill must explicitly require a verification round "
        "before submission (mirrors Duo's auditor)."
    )
    assert "Peer-DM challenge" in skill, (
        "DM realtime skill must describe the cheap peer-DM challenge path "
        "for verification."
    )
    assert "Spawn a verifier task" in skill, (
        "DM realtime skill must describe spawning a verifier task with "
        "depends_on for thorough verification."
    )
    assert "depends_on" in skill, (
        "DM realtime skill must reference depends_on so verifier tasks can "
        "see prior task summaries as context."
    )


def test_dm_realtime_skill_requires_minimum_evidence_gate():
    """The orchestrator may execute directly in ``exec_enabled`` mode, but it
    must not submit HLE-style answers from model memory alone.  Require at
    least one delegation or concrete verification tool call before reporting.
    """
    skill = _DM_REALTIME_SKILL_PATH.read_text()
    assert "Minimum Evidence Gate" in skill, (
        "DM realtime skill must name a minimum evidence gate before final "
        "submission."
    )
    assert "Do NOT answer from memory alone" in skill, (
        "DM realtime skill must explicitly forbid memory-only answers on "
        "non-trivial questions."
    )
    assert "Create at least one `create_task`" in skill, (
        "The gate must allow subagent delegation as the preferred evidence "
        "path."
    )
    assert "perform at least one concrete verification tool call yourself" in skill, (
        "The gate must allow orchestrator-side direct verification for "
        "small checks in exec_enabled mode."
    )


def test_dm_realtime_skill_recommends_diversification_for_first_wave():
    """First-wave parallel tasks must be diversified (different methodology
    or isolated=true) to avoid groupthink, since DM workers do NOT see each
    other's intermediate findings and identical prompts produce identical
    answers.
    """
    skill = _DM_REALTIME_SKILL_PATH.read_text()
    assert "Diversification" in skill, (
        "DM realtime skill must call out diversification of parallel tasks."
    )
    assert "isolated=true" in skill, (
        "DM realtime skill must mention isolated=true for first-wave "
        "browsing tasks."
    )
    assert "distinct methodology hint" in skill, (
        "DM realtime skill must instruct the orchestrator to give each "
        "parallel task a distinct methodology hint."
    )


def test_dm_coordination_skill_replaces_default_nothing_to_flag():
    """The old ``dm-coordination`` text told idle reviewers to default to
    "Nothing to flag" without any independent check, which collapsed
    cross-agent verification.  The new contract requires a quick
    sanity-check before that response.
    """
    skill = _DM_COORDINATION_SKILL_PATH.read_text()
    assert (
        'If everything looks reasonable, respond with "Nothing to flag" and stop.'
        not in skill
    ), (
        "Old default-to-Nothing-to-flag idle behavior must be removed — it "
        "silenced cross-agent verification entirely."
    )
    assert "sanity check" in skill or "sanity-check" in skill, (
        "DM coordination skill must require a quick independent sanity "
        "check before idle reviewers can dismiss a peer's work."
    )


def test_dm_coordination_skill_describes_peer_dm_after_complete_task():
    """Workers' ``complete_task`` summaries are not auto-shared with peers.
    The coordination skill must tell workers when to send a *peer* DM after
    completing, so that non-obvious assumptions and conflicting findings
    surface to teammates, not just to the orchestrator.
    """
    skill = _DM_COORDINATION_SKILL_PATH.read_text()
    assert "After completing a task, send a peer-DM" in skill, (
        "DM coordination skill must include explicit guidance on when to "
        "send a peer-DM after complete_task."
    )
    assert "non-obvious assumption" in skill, (
        "Peer-DM trigger list must include non-obvious assumptions so the "
        "orchestrator can challenge them in the verification round."
    )


def test_dm_subagent_inline_protocol_lowers_peer_dm_bar():
    """The inline DM communication protocol baked into every DM-only
    subagent system prompt must mirror the SKILL.md update: peers do NOT
    automatically receive ``complete_task`` summaries, so the worker is
    expected to send a targeted DM when their conclusion rests on a
    non-obvious assumption / single primary source / conflicts with a peer.
    """
    inline = build_comm_protocol_inline(has_bbs=False, has_dm=True)

    assert "Peers do NOT receive your `complete_task` summary" in inline, (
        "Inline DM protocol must explicitly state that peers don't get "
        "complete_task summaries by default — that's the whole reason for "
        "peer DMs."
    )
    assert "non-obvious assumption" in inline, (
        "Inline DM protocol must list non-obvious assumptions as a peer-DM "
        "trigger."
    )
    assert "sanity-check" in inline or "sanity check" in inline, (
        "Inline idle-review protocol must require an independent sanity "
        "check before responding 'Nothing to flag'."
    )
    assert (
        'When idle and receiving a DM: review for obvious errors only. If OK, say "Nothing to flag."'
        not in inline
    ), (
        "Old default-to-Nothing-to-flag inline guidance must be removed."
    )
