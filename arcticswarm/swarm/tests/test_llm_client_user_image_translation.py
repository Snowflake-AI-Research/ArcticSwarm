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

"""Cross-provider guard test: Anthropic-shape user image blocks must be
translated correctly by both OpenAI client message-conversion paths.

Motivation: the eval runner now builds the initial user message as an
Anthropic-shape list (``[{"type":"text"}, {"type":"image", "source": {...}}]``)
for HLE image cases. The arcticswarm LLM client is expected to translate
this list into the appropriate OpenAI shape — ``image_url`` for Chat
Completions, ``input_image`` for the Responses API — so the same
message payload works against Claude, GPT (Chat), GPT (Responses),
Azure OpenAI Chat, and Azure OpenAI Responses without any
eval-runner-side provider branching.

This test locks in that user-message (not just tool_result) images
survive the translation. The tool_result path is already covered
elsewhere.
"""

from __future__ import annotations

from arcticswarm.llm_client import OpenAIChatLLMClient, OpenAIResponsesLLMClient


_USER_CONTENT = [
    {"type": "text", "text": "Image 1:"},
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "AAAA",
        },
    },
    {"type": "text", "text": "Describe this."},
]
_MESSAGES = [{"role": "user", "content": _USER_CONTENT}]


def test_openai_chat_translates_user_image_block_to_image_url() -> None:
    """The Chat Completions vision shape is
    ``{"type":"image_url","image_url":{"url":"data:<mt>;base64,<data>"}}``."""
    converted = OpenAIChatLLMClient._convert_messages("sys", _MESSAGES)

    # First output message is the system prompt; the user message follows.
    user_msgs = [m for m in converted if m.get("role") == "user"]
    assert len(user_msgs) == 1, converted
    user_msg = user_msgs[0]
    assert isinstance(user_msg["content"], list), user_msg

    blocks = user_msg["content"]
    image_blocks = [b for b in blocks if b.get("type") == "image_url"]
    assert len(image_blocks) == 1, blocks
    assert image_blocks[0]["image_url"] == {
        "url": "data:image/png;base64,AAAA",
    }

    # Text blocks (in order) must survive as-is.
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert [b["text"] for b in text_blocks] == ["Image 1:", "Describe this."]


def test_openai_responses_translates_user_image_block_to_input_image() -> None:
    """The Responses API vision shape is
    ``{"type":"input_image","image_url":"data:<mt>;base64,<data>"}``."""
    _instructions, items = OpenAIResponsesLLMClient._convert_input("sys", _MESSAGES)

    user_items = [
        it for it in items
        if it.get("type") == "message" and it.get("role") == "user"
    ]
    assert len(user_items) == 1, items
    content = user_items[0]["content"]
    assert isinstance(content, list), user_items[0]

    image_blocks = [b for b in content if b.get("type") == "input_image"]
    assert len(image_blocks) == 1, content
    assert image_blocks[0]["image_url"] == "data:image/png;base64,AAAA"

    # In the Responses API, user text blocks are tagged ``input_text``.
    text_blocks = [b for b in content if b.get("type") == "input_text"]
    assert [b["text"] for b in text_blocks] == ["Image 1:", "Describe this."]
