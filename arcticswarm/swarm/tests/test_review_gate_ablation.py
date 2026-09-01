# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the review-gate ablation (BrowseComp-Plus / Qwen3.5-27B).

The ablation runs a cumulative ladder over three "review gates", each of which
is disabled as a design unit = {code enforcement + the prompt/skill text that
drives it}:

    R3  reflection ON  | bbs review ON  | final verify ON   (baseline)
    R2  reflection ON  | bbs review ON  | final verify OFF
    R1  reflection ON  | bbs review OFF | final verify OFF
    R0  reflection OFF | bbs review OFF | final verify OFF

These tests assert, WITHOUT a model/GPU, that:
  * the new ``swarm.disable_final_verification`` + ``swarm.skill_overrides``
    config fields plumb through correctly,
  * each arm resolves to the expected (gate-stripped) SKILL.md variant,
  * each variant's SKILL.md has its confound sections removed while the
    baseline retains them,
  * the four code-built prompt fragments are gated by the arm's flags,
  * R3 (no gate flags) is byte-identical to today's behavior.
"""

import pytest

from arcticswarm.run_config import RunConfig, load_run_config, _dict_to_run_config
from arcticswarm.skill_loader import SkillRegistry, SKILLS_DIR
from arcticswarm.swarm.profiles import (
    resolve_orchestrator_skill,
    resolve_profile_skills,
)
from arcticswarm.swarm.prompts import (
    build_orchestrator_system_prompt,
    build_comm_protocol_inline,
    build_skill_recommendations,
    get_profile_task_prompt,
)
from arcticswarm.swarm.tools import DynamicCreateTaskTool


# --------------------------------------------------------------------------- #
# Arm definitions: the flat-config flags each arm sets, plus its skill remap.
# --------------------------------------------------------------------------- #

ORCH = "swarm-orchestration-dynamic-web"
BBS = "bbs-coordination-web"
WRC = "web-research-corpus"

ARMS = {
    "R3": dict(
        disable_self_reflection=False,
        disable_auditor=False,
        disable_builder_idle=False,
        enforce_alt_task=True,
        disable_final_verification=False,
        skill_overrides={},
    ),
    "R2": dict(
        disable_self_reflection=False,
        disable_auditor=False,
        disable_builder_idle=False,
        enforce_alt_task=False,
        disable_final_verification=True,
        skill_overrides={ORCH: f"{ORCH}-noverify"},
    ),
    "R1": dict(
        disable_self_reflection=False,
        disable_auditor=True,
        disable_builder_idle=True,
        enforce_alt_task=False,
        disable_final_verification=True,
        skill_overrides={ORCH: f"{ORCH}-noverify-noreview", BBS: f"{BBS}-noreview"},
    ),
    "R0": dict(
        disable_self_reflection=True,
        disable_auditor=True,
        disable_builder_idle=True,
        enforce_alt_task=False,
        disable_final_verification=True,
        skill_overrides={
            ORCH: f"{ORCH}-noverify-noreview",
            BBS: f"{BBS}-noreview",
            WRC: f"{WRC}-singlepass",
        },
    ),
}

BROWSING_DOMAIN = ("web-research-corpus", "tool-usage-policy-browsing-corpus")


@pytest.fixture(scope="module")
def registry():
    return SkillRegistry(skills_dir=SKILLS_DIR)


def _arm_ov(arm):
    return ARMS[arm]["skill_overrides"]


# --------------------------------------------------------------------------- #
# 1. Config plumbing
# --------------------------------------------------------------------------- #

class TestConfigPlumbing:
    def test_defaults(self):
        rc = RunConfig()
        assert rc.swarm.disable_final_verification is False
        assert rc.swarm.skill_overrides == {}
        ac = rc.to_arcticswarm_config()
        assert ac.disable_final_verification is False
        assert ac.skill_overrides == {}

    def test_propagates(self):
        rc = RunConfig()
        rc.swarm.disable_final_verification = True
        rc.swarm.skill_overrides = {ORCH: f"{ORCH}-noverify"}
        ac = rc.to_arcticswarm_config()
        assert ac.disable_final_verification is True
        assert ac.skill_overrides == {ORCH: f"{ORCH}-noverify"}

    def test_yaml_dict_loads(self):
        merged = {
            "swarm": {
                "disable_final_verification": True,
                "skill_overrides": {ORCH: f"{ORCH}-noverify", BBS: f"{BBS}-noreview"},
            }
        }
        ac = _dict_to_run_config(merged).to_arcticswarm_config()
        assert ac.disable_final_verification is True
        assert ac.skill_overrides[BBS] == f"{BBS}-noreview"

    def test_overlay_files_merge(self):
        # The committed overlays merge on top of the base config as expected.
        expected = {
            "R2": {ORCH: f"{ORCH}-noverify"},
            "R1": {ORCH: f"{ORCH}-noverify-noreview", BBS: f"{BBS}-noreview"},
            "R0": {
                ORCH: f"{ORCH}-noverify-noreview",
                BBS: f"{BBS}-noreview",
                WRC: f"{WRC}-singlepass",
            },
        }
        for arm, exp in expected.items():
            rc = load_run_config(
                [
                    "conf/bench/browsecomp_plus_qwen.yaml",
                    f"conf/bench/ablation/bcp_qwen_{arm}_skills.yaml",
                ],
                [],
            )
            assert rc.to_arcticswarm_config().skill_overrides == exp, arm


# --------------------------------------------------------------------------- #
# 2. Per-arm skill resolution
# --------------------------------------------------------------------------- #

class TestSkillResolution:
    @pytest.mark.parametrize("arm,expected", [
        ("R3", ORCH),
        ("R2", f"{ORCH}-noverify"),
        ("R1", f"{ORCH}-noverify-noreview"),
        ("R0", f"{ORCH}-noverify-noreview"),
    ])
    def test_orchestrator_skill(self, arm, expected):
        assert resolve_orchestrator_skill(
            has_bbs=True, has_web_search=True, skill_overrides=_arm_ov(arm)
        ) == expected

    @pytest.mark.parametrize("arm,must_have,must_not_have", [
        ("R3", (BBS, WRC), ()),
        ("R2", (BBS, WRC), ()),
        ("R1", (f"{BBS}-noreview",), (BBS,)),
        ("R0", (f"{BBS}-noreview", f"{WRC}-singlepass"), (BBS, WRC)),
    ])
    def test_profile_skills(self, registry, arm, must_have, must_not_have):
        skills = resolve_profile_skills(
            BROWSING_DOMAIN, frozenset({"web_search"}),
            has_bbs=True, has_dm=False, registry=registry,
            skill_overrides=_arm_ov(arm),
        )
        for s in must_have:
            assert s in skills, f"{arm}: {s} missing from {skills}"
        for s in must_not_have:
            assert s not in skills, f"{arm}: {s} should be remapped out of {skills}"


# --------------------------------------------------------------------------- #
# 3. Variant SKILL.md content (confound sections removed / retained)
# --------------------------------------------------------------------------- #

class TestVariantContent:
    def _body(self, registry, name):
        loc = registry.get(name)
        assert loc is not None, f"variant not discovered: {name}"
        return registry.load_skill(name).content

    def test_noverify_strips_final_verification_keeps_review(self, registry):
        b = self._body(registry, f"{ORCH}-noverify")
        for gone in ("Constraint Verification Protocol", "Anti-Fixation Rule", "Best-So-Far"):
            assert gone not in b, gone
        # GATE 2 (review) is still ON in R2 -> auditor text retained.
        assert "auditor agent" in b
        # Generic guidance retained.
        assert "Hard Constraint Violations" in b

    def test_noverify_noreview_strips_both(self, registry):
        b = self._body(registry, f"{ORCH}-noverify-noreview")
        for gone in ("auditor agent", "idle review", "Constraint Verification Protocol",
                     "Anti-Fixation Rule"):
            assert gone not in b, gone
        assert "Hard Constraint Violations" in b  # generic retained

    def test_bbs_noreview_strips_when_idle(self, registry):
        b = self._body(registry, f"{BBS}-noreview")
        assert "When Idle" not in b
        assert "post_to_bbs" in b  # sharing lane retained

    def test_singlepass_strips_reflection_loop(self, registry):
        b = self._body(registry, f"{WRC}-singlepass")
        for gone in ("Iterative Research Workflow", "Completion Criteria", "Self-Assessment"):
            assert gone not in b, gone
        assert "Source Evaluation" in b  # generic retained

    def test_baselines_retain_everything(self, registry):
        orch = self._body(registry, ORCH)
        assert "Constraint Verification Protocol" in orch and "auditor agent" in orch
        assert "When Idle" in self._body(registry, BBS)
        assert "Iterative Research Workflow" in self._body(registry, WRC)


# --------------------------------------------------------------------------- #
# 4. Code-built prompt gates
# --------------------------------------------------------------------------- #

class TestCodeBuiltPromptGates:
    def test_alt_task_rule(self):
        on = build_orchestrator_system_prompt(has_web_search=True, has_bbs=True, enforce_alt_task=True)
        off = build_orchestrator_system_prompt(has_web_search=True, has_bbs=True, enforce_alt_task=False)
        assert "ALTERNATIVE / CONTRARIAN" in on
        assert "ALTERNATIVE / CONTRARIAN" not in off

    def test_create_task_alt_schema(self):
        on = DynamicCreateTaskTool(None, active_profiles=["browsing"],
                                   has_web_search=True, enforce_alt_task=True).parameters_schema()
        off = DynamicCreateTaskTool(None, active_profiles=["browsing"],
                                    has_web_search=True, enforce_alt_task=False).parameters_schema()
        assert "alt" in on["properties"]
        assert "alt" not in off["properties"]

    def test_idle_review_bullet(self):
        on = build_comm_protocol_inline(True, False, disable_idle_review=False)
        off = build_comm_protocol_inline(True, False, disable_idle_review=True)
        assert "When idle reviewing BBS" in on
        assert "When idle reviewing BBS" not in off

    def test_comm_skill_ref_remap(self):
        out = build_comm_protocol_inline(
            True, False, disable_idle_review=True,
            skill_overrides={BBS: f"{BBS}-noreview"},
        )
        assert f"{BBS}-noreview" in out

    def test_skill_recommendations_remap(self):
        out = build_skill_recommendations(
            "browsing", skill_names=BROWSING_DOMAIN,
            skill_overrides={WRC: f"{WRC}-singlepass"},
        )
        assert f"{WRC}-singlepass" in out
        # the un-remapped bare name must not appear as a standalone item
        assert f"`{WRC}`" not in out

    def test_reflection_self_assessment(self):
        t = type("T", (), {"name": "t", "prompt": "p"})()
        on = get_profile_task_prompt("browsing", t, "q", "", "", has_bbs=True,
                                     disable_self_reflection=False)
        off = get_profile_task_prompt("browsing", t, "q", "", "", has_bbs=True,
                                      disable_self_reflection=True)
        assert "self-assessment checklist" in on and "iterative research workflow" in on
        assert "self-assessment checklist" not in off and "iterative research workflow" not in off


# --------------------------------------------------------------------------- #
# 5. R3 byte-identity (regression safety)
# --------------------------------------------------------------------------- #

class TestR3ByteIdentity:
    def test_orchestrator_prompt_defaults_unchanged(self):
        # Calling with the new kwargs omitted (R3) must equal calling them with
        # their baseline values.
        base = build_orchestrator_system_prompt(has_web_search=True, has_bbs=True)
        explicit = build_orchestrator_system_prompt(
            has_web_search=True, has_bbs=True,
            enforce_alt_task=True, skill_overrides=None,
        )
        assert base == explicit
        assert "ALTERNATIVE / CONTRARIAN" in base
        assert f'load_skill("{ORCH}")' in base

    def test_comm_protocol_defaults_unchanged(self):
        base = build_comm_protocol_inline(True, False)
        explicit = build_comm_protocol_inline(
            True, False, disable_idle_review=False, skill_overrides=None,
        )
        assert base == explicit
        assert "When idle reviewing BBS" in base

    def test_task_prompt_defaults_unchanged(self):
        t = type("T", (), {"name": "t", "prompt": "p"})()
        base = get_profile_task_prompt("browsing", t, "q", "", "", has_bbs=True)
        explicit = get_profile_task_prompt("browsing", t, "q", "", "", has_bbs=True,
                                           disable_self_reflection=False)
        assert base == explicit

    def test_create_task_schema_default_has_alt(self):
        schema = DynamicCreateTaskTool(None, active_profiles=["browsing"],
                                       has_web_search=True).parameters_schema()
        assert "alt" in schema["properties"]
