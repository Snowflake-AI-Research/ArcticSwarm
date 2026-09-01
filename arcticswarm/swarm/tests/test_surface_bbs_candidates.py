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

"""Unit test for the BBS candidate-digest (surface_bbs_candidates).

The digest re-surfaces verified consensus verdicts + top key-findings into the
prepare_report unlock message, so a found-but-compacted-away answer reaches the
final-answer step. Default-off must be a no-op.
"""
from arcticswarm.swarm.bbs import BBS
from arcticswarm.swarm.task import TaskBoard
from arcticswarm.swarm.teammate import AgentRegistry
from arcticswarm.swarm.tools import PrepareReportTool, SendReportTool


def _tool(bbs, *, surface):
    board = TaskBoard()
    return PrepareReportTool(
        task_board=board,
        agent_registry=AgentRegistry(),
        report_tool=SendReportTool(has_web_search=True),
        agent_tools={},
        bbs=bbs,
        has_web_search=True,
        surface_bbs_candidates=surface,
    )


def test_digest_surfaces_verified_candidate():
    bbs = BBS()
    bbs.post(channel="consensus", author="Koji",
             content="Reviewed [The Redmond Monument] — all 11 constraints verified with evidence.")
    bbs.post(channel="key-findings", author="James",
             content="Answer: The Redmond Monument (Wexford). Erected 1867, restored 2007.")
    tool = _tool(bbs, surface=True)
    digest = tool._build_candidate_digest()
    assert "Redmond Monument" in digest
    assert "CANDIDATE FINDINGS" in digest
    assert "VERIFIED" in digest


def test_digest_disabled_is_noop():
    bbs = BBS()
    bbs.post(channel="consensus", author="Koji",
             content="Reviewed [X] — all constraints verified with evidence.")
    assert _tool(bbs, surface=False)._build_candidate_digest() == ""


def test_digest_empty_bbs_is_noop():
    assert _tool(BBS(), surface=True)._build_candidate_digest() == ""


if __name__ == "__main__":
    test_digest_surfaces_verified_candidate()
    test_digest_disabled_is_noop()
    test_digest_empty_bbs_is_noop()
    print("all digest tests passed")
