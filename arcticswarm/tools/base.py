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

"""Base tool interface for Arcticswarm.

Every tool subclasses :class:`BaseTool` and registers itself via the
module-level :data:`TOOL_REGISTRY` dict.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Uniform result returned by every tool execution."""

    output: str = ""
    error: str | None = None
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional extra content blocks (e.g. image blocks) to include alongside text.
    extra_content: list[dict[str, Any]] = field(default_factory=list)

    def to_content(self) -> list[dict[str, Any]]:
        """Serialise to Anthropic ``tool_result`` content blocks."""
        text = self.error if self.is_error else self.output
        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        blocks.extend(self.extra_content)
        return blocks or [{"type": "text", "text": "(no output)"}]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """Interface every Arcticswarm tool must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in the Anthropic tool-use schema."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-paragraph description shown to the model."""

    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's ``input`` object."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool and return a :class:`ToolResult`."""

    # -- helpers for tool definitions ----------------------------------------

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Return the dict expected by ``anthropic.messages.create(tools=[…])``."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema(),
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """Return the dict expected by ``openai.chat.completions.create(tools=[…])``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> BaseTool:
    """Add *tool* to the global registry (keyed by ``tool.name``)."""
    TOOL_REGISTRY[tool.name] = tool
    return tool


def get_all_tool_definitions() -> list[dict[str, Any]]:
    """Return Anthropic-compatible tool definitions for all registered tools."""
    return [t.to_anthropic_tool() for t in TOOL_REGISTRY.values()]


def execute_tool(name: str, input_data: dict[str, Any]) -> ToolResult:
    """Dispatch a tool call by name."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        available = ", ".join(sorted(TOOL_REGISTRY.keys()))
        return ToolResult(
            error=(
                f"Unknown tool: '{name}'. "
                f"Available tools: {available}. "
                f"Use only the tools listed above."
            ),
            is_error=True,
        )
    try:
        return tool.execute(**input_data)
    except Exception as exc:
        return ToolResult(error=f"Tool '{name}' failed: {exc}", is_error=True)
