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

"""Smoke tests for ``Agent.run_turn`` / ``run_turn_streaming`` list content.

Motivation: HLE image cases now attach images as a multimodal content
list (``[{"type":"text",...}, {"type":"image",...}, ...]``) on the
initial user turn rather than embedding a file path into the question
string. The two :class:`Agent` entry points therefore need to accept
``str | list[dict]`` for ``user_message`` and round-trip a list into
``self.messages`` unchanged so the LLM client sees the full multimodal
payload.
"""

from __future__ import annotations

import inspect

from arcticswarm.agent import Agent


def test_run_turn_signature_accepts_list_content() -> None:
    sig = inspect.signature(Agent.run_turn)
    param = sig.parameters["user_message"]
    annotation = str(param.annotation)
    assert "list" in annotation and "str" in annotation, (
        "Agent.run_turn must accept str | list[dict] for user_message so "
        "multimodal HLE cases can attach an image block on turn 0."
    )


def test_run_turn_streaming_signature_accepts_list_content() -> None:
    sig = inspect.signature(Agent.run_turn_streaming)
    param = sig.parameters["user_message"]
    annotation = str(param.annotation)
    assert "list" in annotation and "str" in annotation, (
        "Agent.run_turn_streaming must accept str | list[dict] for "
        "user_message so HLE image cases work in streaming mode too."
    )


def test_append_msg_roundtrips_list_user_content() -> None:
    """The message-append path must preserve list content verbatim so the
    LLM client sees the full multimodal payload (it is
    content-agnostic beyond timestamp stamping)."""
    # Build a minimal Agent without calling __init__ to avoid SF / LLM
    # client setup — we only exercise the message-append contract.
    agent = Agent.__new__(Agent)
    agent.messages = []

    content_list = [
        {"type": "text", "text": "Image 1:"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA",
            },
        },
        {"type": "text", "text": "Describe this image."},
    ]

    Agent._append_msg(agent, {"role": "user", "content": content_list})

    assert len(agent.messages) == 1
    msg = agent.messages[0]
    assert msg["role"] == "user"
    # The list must round-trip unchanged (same object identity is fine —
    # the contract is just that nothing is stringified or dropped).
    assert msg["content"] == content_list
    assert isinstance(msg["content"], list)
    assert msg["content"][1]["source"]["data"] == "AAAA"
    # _append_msg stamps a timestamp; the test should not depend on its
    # value but the field must exist.
    assert "_timestamp" in msg
