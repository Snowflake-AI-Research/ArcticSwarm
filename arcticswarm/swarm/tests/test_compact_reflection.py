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

"""Unit tests for the compact reflection schema."""

from __future__ import annotations

import json

from arcticswarm.swarm.reflection import (
    ReflectionResult,
    _format_constraints_block,
)


def test_compact_parser_all_exact_high_conf():
    raw = json.dumps({
        "table": {"c1": "E", "c2": "E"},
        "candidate": {"name": "Liberty", "alternatives_seen": 1, "fame_flag": 0},
        "next": None,
    })
    r = ReflectionResult.from_compact_json(raw)
    assert r.is_sufficient is True
    assert r.confidence == "high"
    assert r.knowledge_gaps == []


def test_compact_parser_partial_medium_conf():
    raw = json.dumps({
        "table": {"c1": "E", "c2": "P"},
        "candidate": {"name": "X", "alternatives_seen": 1, "fame_flag": 0},
        "next": None,
    })
    r = ReflectionResult.from_compact_json(raw)
    assert r.is_sufficient is True
    assert r.confidence == "medium"


def test_compact_parser_unknown_low_conf_with_gaps():
    raw = json.dumps({
        "table": {"c1": "U", "c2": "E"},
        "candidate": {"name": "X", "alternatives_seen": 0, "fame_flag": 1},
        "next": "verify c1",
    })
    r = ReflectionResult.from_compact_json(raw)
    assert r.is_sufficient is False
    assert r.confidence == "low"
    assert any("c1" in g for g in r.knowledge_gaps)
    assert any("alternative" in g for g in r.knowledge_gaps)
    assert r.next_queries == ["verify c1"]


def test_compact_parser_no_alternatives_blocks_sufficiency():
    raw = json.dumps({
        "table": {"c1": "E", "c2": "E"},
        "candidate": {"name": "X", "alternatives_seen": 0, "fame_flag": 1},
        "next": None,
    })
    r = ReflectionResult.from_compact_json(raw)
    assert r.is_sufficient is False
    assert r.confidence == "high"


def test_compact_parser_handles_code_fence():
    raw = "```json\n" + json.dumps({
        "table": {"c1": "E"},
        "candidate": {"name": "X", "alternatives_seen": 1},
        "next": None,
    }) + "\n```"
    r = ReflectionResult.from_compact_json(raw)
    assert r.is_sufficient is True


def test_compact_parser_malformed_returns_safe_default():
    r = ReflectionResult.from_compact_json("not json")
    assert r.is_sufficient is False
    assert r.knowledge_gaps == ["parse_error"]


def test_compact_parser_empty_table():
    raw = json.dumps({"table": {}, "candidate": {}, "next": None})
    r = ReflectionResult.from_compact_json(raw)
    assert r.is_sufficient is False
    assert "empty_table" in r.knowledge_gaps


def test_format_constraints_block_with_constraints():
    constraints = [
        {"id": "c1", "text": "born before 1900", "type": "hard"},
        {"id": "c2", "text": "Polish nationality", "type": "hard"},
    ]
    block, template = _format_constraints_block(constraints)
    assert "c1: born before 1900" in block
    assert "c2: Polish nationality" in block
    assert '"c1": "E|P|C|U"' in template


def test_format_constraints_block_fallback():
    block, template = _format_constraints_block(None)
    # Fallback no longer pads to 8 rows; instructs the LLM to enumerate
    # only real constraints to avoid false-unverified noise on short questions.
    assert "do NOT pad" in block
    assert "c1" in template and "c2" in template
    assert "E|P|C|U" in template
