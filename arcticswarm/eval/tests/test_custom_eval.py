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

"""Tests for the config-driven custom evaluation capability.

Covers two pieces that let users evaluate their own dataset + rubric without
editing framework code:

  * :func:`arcticswarm.eval.data_loader._resolve_eval_csv_path` — a present
    custom CSV resolves; a missing explicit ``eval.csv_path`` raises a clear,
    actionable :class:`FileNotFoundError`.
  * :meth:`arcticswarm.eval.judge.LLMJudge.judge_custom` — the custom-rubric
    verdict parser maps both a ``correct: true``-style response and a JSON
    ``{"correct": true}`` response into a correct :class:`QAJudgeResult`.

The LLM call is monkeypatched throughout — no network is hit.
"""

from __future__ import annotations

import pytest

from arcticswarm.eval.data_loader import _resolve_eval_csv_path
from arcticswarm.eval.judge import LLMJudge, QAJudgeResult


# ---------------------------------------------------------------------------
# Custom CSV path resolution
# ---------------------------------------------------------------------------


def test_present_custom_csv_path_resolves(tmp_path):
    """An explicit eval.csv_path that exists resolves to that file."""
    csv_file = tmp_path / "my_dataset.csv"
    csv_file.write_text("TURN_INDEX,CONV_ID\n0,q1\n", encoding="utf-8")

    resolved = _resolve_eval_csv_path(str(csv_file), datasets=["MY_DATASET"])

    assert resolved == csv_file
    assert resolved.exists()


def test_missing_custom_csv_path_raises_clear_error(tmp_path):
    """A missing explicit eval.csv_path raises a clear, actionable error."""
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_eval_csv_path(str(missing), datasets=["MY_DATASET"])

    msg = str(exc_info.value)
    # Names the offending path and points at remediation.
    assert "does_not_exist.csv" in msg
    assert "fetch_datasets.sh" in msg


# ---------------------------------------------------------------------------
# Custom judge verdict parsing
# ---------------------------------------------------------------------------


@pytest.fixture
def custom_judge(tmp_path, monkeypatch):
    """An LLMJudge wired with a custom rubric and a stubbed LLM call.

    The stubbed ``_call_llm`` returns whatever string is stashed on the judge
    via ``judge._stub_output`` — so each test can drive a different verdict
    format without any network access.
    """
    rubric = tmp_path / "rubric.txt"
    rubric.write_text(
        "Q: {question}\nA: {response}\nGold: {correct_answer}\n"
        "End with correct: true or correct: false.\n",
        encoding="utf-8",
    )

    judge = LLMJudge(custom_judge_prompt=str(rubric))

    def _fake_call_llm(prompt, **kwargs):  # noqa: ANN001 - test stub
        # The rubric placeholders must have been filled in.
        assert "{question}" not in prompt
        assert "{response}" not in prompt
        assert "{correct_answer}" not in prompt
        return judge._stub_output

    monkeypatch.setattr(judge, "_call_llm", _fake_call_llm)
    return judge


def test_custom_judge_parses_correct_true_line(custom_judge):
    """A ``correct: true`` verdict maps to a correct QAJudgeResult."""
    custom_judge._stub_output = (
        "The response gives the same year as the gold answer.\ncorrect: true"
    )

    result = custom_judge.judge_custom(
        question="What year?",
        answer="1889",
        expected_answer="1889",
    )

    assert isinstance(result, QAJudgeResult)
    assert result.correct is True


def test_custom_judge_parses_correct_false_line(custom_judge):
    """A ``correct: false`` verdict maps to an incorrect QAJudgeResult."""
    custom_judge._stub_output = "Wrong year.\ncorrect: false"

    result = custom_judge.judge_custom(
        question="What year?",
        answer="1900",
        expected_answer="1889",
    )

    assert result.correct is False


def test_custom_judge_parses_json_object(custom_judge):
    """A JSON ``{"correct": true, ...}`` verdict maps to a correct result."""
    custom_judge._stub_output = (
        '{"correct": true, "reasoning": "matches gold", "confidence": 95}'
    )

    result = custom_judge.judge_custom(
        question="Who wrote Beloved?",
        answer="Toni Morrison",
        expected_answer="Toni Morrison",
    )

    assert result.correct is True
    assert "matches gold" in result.comment
    assert result.judge_confidence == 95.0


def test_custom_judge_parses_json_false(custom_judge):
    """A JSON ``{"correct": false}`` verdict maps to an incorrect result."""
    custom_judge._stub_output = '{"correct": false, "reasoning": "different author"}'

    result = custom_judge.judge_custom(
        question="Who wrote Beloved?",
        answer="Ernest Hemingway",
        expected_answer="Toni Morrison",
    )

    assert result.correct is False
    assert "different author" in result.comment


def test_custom_judge_parses_grade_format(custom_judge):
    """A ``GRADE: INCORRECT`` verdict maps to an incorrect result (and the
    ``correct`` substring inside ``INCORRECT`` is not misread as a pass)."""
    custom_judge._stub_output = "Reasoning: off by a decade.\nGRADE: INCORRECT"

    result = custom_judge.judge_custom(
        question="What year?",
        answer="1879",
        expected_answer="1889",
    )

    assert result.correct is False


def test_custom_judge_empty_answer_short_circuits(custom_judge):
    """An empty agent answer is scored incorrect without calling the LLM."""
    custom_judge._stub_output = "correct: true"  # must NOT be consumed

    result = custom_judge.judge_custom(
        question="What year?",
        answer="",
        expected_answer="1889",
    )

    assert result.correct is False


def test_custom_judge_unparseable_defaults_incorrect(custom_judge):
    """An unparseable verdict defaults to incorrect (conservative)."""
    custom_judge._stub_output = "I am not sure how to grade this."

    result = custom_judge.judge_custom(
        question="What year?",
        answer="1889",
        expected_answer="1889",
    )

    assert result.correct is False


def test_custom_prompt_overrides_browsecomp_judge(custom_judge):
    """With a custom rubric set, even judge_browsecomp routes to the rubric."""
    custom_judge._stub_output = "correct: true"

    result = custom_judge.judge_browsecomp(
        question="What year?",
        answer="1889",
        expected_answer="1889",
    )

    assert result.correct is True


def test_missing_rubric_template_raises(tmp_path):
    """Constructing a judge with a non-existent rubric path fails fast."""
    with pytest.raises(FileNotFoundError):
        LLMJudge(custom_judge_prompt=str(tmp_path / "nope.txt"))
