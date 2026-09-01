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

"""Unit test for selective-delete compaction (_prune_certainly_wrong).

Prunes certainly-wrong tool-result CONTENT in the throwaway summarizer input
while preserving every message + tool_result block (pairing untouched).
"""
from arcticswarm.context_management import ContextManagementMixin


def _stub():
    return ContextManagementMixin.__new__(ContextManagementMixin)


def _tr(text, is_error=False):
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": is_error,
         "content": [{"type": "text", "text": text}]}]}


def test_prunes_junk_preserves_real():
    msgs = [
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "web_search", "input": {}}]},
        _tr("(no output)"),
        _tr("SYSTEM SHUTTING DOWN in ~899s. web_search is now DISABLED."),
        _tr("(skipped — max tool calls per turn reached)"),
        _tr("connection error", is_error=True),
        _tr("Found it: The answer is The Redmond Monument, erected 1867 in Wexford, restored 2007. " * 4),
    ]
    n = _stub()._prune_certainly_wrong(msgs)
    assert n == 4, f"expected 4 pruned, got {n}"
    # structure preserved: still 7 messages, each tool_result block intact
    assert len(msgs) == 7
    blocks = [b for m in msgs if isinstance(m["content"], list) for b in m["content"]]
    trs = [b for b in blocks if b.get("type") == "tool_result"]
    assert len(trs) == 5  # all tool_results still present
    # the real finding is untouched
    real = trs[-1]["content"]
    real_text = real if isinstance(real, str) else " ".join(b.get("text", "") for b in real)
    assert "Redmond Monument" in real_text
    # the junk ones are replaced by the placeholder string
    assert isinstance(trs[0]["content"], str) and "pruned" in trs[0]["content"]


def test_no_junk_no_change():
    msgs = [_tr("Real finding with lots of useful content. " * 30)]
    assert _stub()._prune_certainly_wrong(msgs) == 0


if __name__ == "__main__":
    test_prunes_junk_preserves_real()
    test_no_junk_no_change()
    print("all prune tests passed")
