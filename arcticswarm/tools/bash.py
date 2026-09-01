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

"""Bash tool — execute shell commands with timeout."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any

from arcticswarm.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # seconds
_MAX_OUTPUT = 100_000  # characters


def _truncate_output(text: str, max_chars: int = _MAX_OUTPUT) -> str:
    """Truncate *text* at a line boundary within *max_chars*.

    Instead of slicing mid-line, this keeps complete lines until the budget
    is exceeded, then appends a summary of the omitted content.
    """
    if len(text) <= max_chars:
        return text

    lines = text.split("\n")
    kept: list[str] = []
    used = 0
    for line in lines:
        # +1 accounts for the newline separator between lines
        needed = len(line) + (1 if kept else 0)
        if used + needed > max_chars and kept:
            break
        kept.append(line)
        used += needed

    omitted_lines = len(lines) - len(kept)
    result = "\n".join(kept)
    result += f"\n... (truncated, {omitted_lines} more lines, {len(text)} total chars)"
    return result


class BashTool(BaseTool):
    name = "bash"
    description = (
        "Execute a bash command in a shell. "
        "Use for git, npm, docker, and other system commands. "
        "Prefer the dedicated file tools (read_file, edit_file) "
        "over shell equivalents (cat, sed).  Unless you pass an "
        "explicit ``working_directory``, commands run in a per-agent scratch "
        "directory so generated files (images, OCR output, downloads) do not "
        "pollute the caller's CWD.  That scratch directory is preserved across "
        "bash calls within a single agent's lifetime, so multi-step workflows "
        "(e.g. download → crop → OCR) still see each other's outputs."
    )

    def __init__(self) -> None:
        # Per-instance persistent scratch directory.  Created lazily on the
        # first ``execute`` call that does not override ``working_directory``,
        # and removed in ``__del__`` / ``close`` when the tool (and therefore
        # the owning ``Agent``) is garbage-collected.  We deliberately do NOT
        # clean up between calls: agents often do multi-step file work and
        # need the previous call's artefacts to still be on disk.
        self._scratch_dir: str | None = None

    def _ensure_scratch_dir(self) -> str:
        """Return the per-instance scratch dir, creating it on first use."""
        if self._scratch_dir is None or not os.path.isdir(self._scratch_dir):
            self._scratch_dir = tempfile.mkdtemp(prefix="arcticswarm_bash_")
            logger.debug("BashTool scratch dir created at %s", self._scratch_dir)
        return self._scratch_dir

    def close(self) -> None:
        """Remove the scratch directory if one was created."""
        if self._scratch_dir is not None:
            shutil.rmtree(self._scratch_dir, ignore_errors=True)
            self._scratch_dir = None

    def __del__(self) -> None:
        # ``__del__`` can run during interpreter shutdown when ``shutil`` has
        # already been torn down, so guard defensively.
        try:
            self.close()
        except Exception:
            pass

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds. Default: {_DEFAULT_TIMEOUT}.",
                },
                "working_directory": {
                    "type": "string",
                    "description": (
                        "Working directory for the command.  If omitted, "
                        "commands run in a per-agent scratch directory "
                        "(isolated from the caller's CWD, persistent across "
                        "bash calls within this agent's lifetime)."
                    ),
                },
            },
            "required": ["command"],
        }

    def execute(
        self,
        *,
        command: str,
        timeout: int = _DEFAULT_TIMEOUT,
        working_directory: str | None = None,
        **_: Any,
    ) -> ToolResult:
        # Default precedence:
        #   1. explicit working_directory= argument (always wins)
        #   2. per-instance scratch dir (default — keeps generated files
        #      out of whatever CWD ``arcticswarm-eval`` was launched from)
        if working_directory:
            effective_cwd = working_directory
        else:
            effective_cwd = self._ensure_scratch_dir()
        try:
            env = os.environ.copy()
            env.pop("MallocStackLogging", None)
            env.pop("MallocStackLoggingNoCompact", None)
            env.setdefault("MPLBACKEND", "Agg")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=effective_cwd,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(error=f"Command timed out after {timeout}s: {command}", is_error=True)
        except Exception as exc:
            return ToolResult(error=f"Command failed: {exc}", is_error=True)

        output_parts = []
        if result.stdout:
            output_parts.append(_truncate_output(result.stdout))
        if result.stderr:
            output_parts.append(f"STDERR:\n{_truncate_output(result.stderr)}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        if result.returncode != 0:
            return ToolResult(
                output=output,
                error=f"Exit code {result.returncode}\n{output}",
                is_error=True,
            )

        return ToolResult(output=output)
