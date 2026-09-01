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

"""Unit tests for the int near-duplicate (reformulation-loop) hard stop.

``search_neardup_hard_stop`` (an int) lets a run lower the force-stop threshold
below the class default (40) so a churning small model is bailed out sooner; a
soft "pivot to a different angle" nudge is prepended in the [hard//3, hard) zone.
``_family_key`` is stubbed to a constant so distinct (non-exact-repeat) queries
share one family, exercising the family counter deterministically.
"""

from __future__ import annotations

from arcticswarm.tools.base import ToolResult
from arcticswarm.tools.web_search import WebSearchTool


def _tool(neardup: int) -> WebSearchTool:
    tool = WebSearchTool(api_key="dummy", neardup_hard_stop=neardup)
    tool._family_key = lambda q: "FAM"  # type: ignore[method-assign]

    def fake_impl(*, query, count=5, country=None, safesearch="moderate", **kw):
        return ToolResult(output=f"Top 1 result(s) for: {query}\n\n1. ok")

    tool._search_impl = fake_impl  # type: ignore[method-assign]
    return tool


def test_neardup_threshold_and_soft_zone_are_configurable():
    tool = _tool(neardup=12)
    assert tool._neardup_hard_stop == 12
    assert tool._neardup_soft == 4  # max(3, 12 // 3)


def test_neardup_hard_stop_forces_stop_at_threshold():
    tool = _tool(neardup=5)
    # 4 distinct (non-repeat) queries below the threshold: real results, no force.
    for i in range(4):
        r = tool.execute(query=f"distinct query number {i}")
        assert not (r.metadata or {}).get("force_stop")
    # 5th issuance hits fam_count >= 5 -> force stop.
    r5 = tool.execute(query="distinct query number 4")
    assert (r5.metadata or {}).get("force_stop") is True


def test_neardup_soft_nudge_below_hard_stop():
    tool = _tool(neardup=9)  # soft zone starts at 3 (= 9 // 3)
    out = ""
    for i in range(3):
        out = tool.execute(query=f"angle {i}").output
    # 3rd issuance is in [3, 9): real results returned, but nudged to pivot.
    assert "pivot to a DIFFERENT angle" in out
    assert "ok" in out  # the real results are still present
