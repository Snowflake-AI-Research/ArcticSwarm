"""Python execution tool — run Python code directly.

A dedicated tool for the coding profile that executes Python code
without shell quoting issues.  Writes code to a temp file, runs it
with ``python3``, and returns stdout/stderr.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from arcticswarm.tools.base import BaseTool, ToolResult

_DEFAULT_TIMEOUT = 120  # seconds
_MAX_OUTPUT = 100_000  # characters

# Prepended to every executed script to suppress GUI pop-ups from PIL.
# PIL.Image.show() launches macOS Preview via `open`, which is not
# controlled by MPLBACKEND.  The try/except is a no-op if PIL is absent.
_PREAMBLE = """\
try:
    import PIL.Image as _PIL_Image
    _PIL_Image.Image.show = lambda self, *a, **kw: None
except ImportError:
    pass
"""


def _truncate(text: str, max_chars: int = _MAX_OUTPUT) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"


class PythonExecuteTool(BaseTool):
    """Execute Python code directly and return stdout/stderr."""

    name = "python_execute"
    description = (
        "Execute Python code directly. Returns stdout and stderr. "
        "Use for data analysis, computation, file processing, and scripting tasks. "
        "The code runs in a subprocess with full access to installed packages."
    )

    def __init__(self) -> None:
        pass

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds. Default: {_DEFAULT_TIMEOUT}.",
                },
            },
            "required": ["code"],
        }

    def execute(
        self,
        *,
        code: str,
        timeout: int = _DEFAULT_TIMEOUT,
        **_: Any,
    ) -> ToolResult:
        if not code.strip():
            return ToolResult(error="Empty code.", is_error=True)

        # Run in a scratch directory so generated files (images, data) don't
        # pollute the repo working directory.
        work_dir = tempfile.mkdtemp(prefix="arcticswarm_py_")
        fd, path = tempfile.mkstemp(suffix=".py", prefix="arcticswarm_", dir=work_dir)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(_PREAMBLE + code)

            env = os.environ.copy()
            env.pop("MallocStackLogging", None)
            env.pop("MallocStackLoggingNoCompact", None)
            env.setdefault("MPLBACKEND", "Agg")
            result = subprocess.run(
                ["python3", path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=work_dir,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                error=f"Python execution timed out after {timeout}s.",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(error=f"Python execution failed: {exc}", is_error=True)
        finally:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(_truncate(result.stdout))
        if result.stderr:
            output_parts.append(f"STDERR:\n{_truncate(result.stderr)}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        if result.returncode != 0:
            return ToolResult(
                output=output,
                error=f"Exit code {result.returncode}\n{output}",
                is_error=True,
            )

        return ToolResult(output=output)
