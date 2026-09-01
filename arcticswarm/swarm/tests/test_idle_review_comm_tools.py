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

"""Contract tests for idle-review communication tool selection.

BBS-only subagents should not try to lazily build DM runtime tools
(`send_message`, `read_dm`) during idle review, and DM-only subagents
should not try to lazily build BBS runtime tools (`post_to_bbs`,
`read_bbs`). Those tools are injected directly by the swarm runtime, not
constructed by `ToolFactory`, so including the wrong names in the idle
allowlist creates noisy "unknown tool ... — skipping" warnings and
unnecessary lazy-build attempts.
"""
from __future__ import annotations

import inspect


def test_idle_review_comm_tools_are_conditioned_on_topology():
    """`SubAgent._idle_check()` must gate BBS/DM runtime tools by
    `self._has_bbs` / `self._has_dm`, instead of unconditionally listing
    both families in every mode."""
    from arcticswarm.swarm.teammate import SubAgent

    src = inspect.getsource(SubAgent._idle_check)
    assert "_idle_comm_tools" in src
    assert 'if self._has_bbs:' in src
    assert '{"post_to_bbs", "read_bbs"}' in src
    assert 'if self._has_dm:' in src
    assert '{"send_message", "read_dm"}' in src


def test_idle_review_no_longer_hardcodes_dm_tools_into_bbs_only_research_path():
    """The old regression was a hard-coded mixed allowlist.

    Keep this exact tuple out of the source so BBS-only runs do not try
    to lazy-build `send_message` / `read_dm` during idle review.
    """
    from arcticswarm.swarm.teammate import SubAgent

    src = inspect.getsource(SubAgent._idle_check)
    assert (
        '"post_to_bbs", "read_bbs", "send_message", "read_dm",'
        not in src
    )
