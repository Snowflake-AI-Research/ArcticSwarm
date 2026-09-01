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

"""Calculator tool — safe evaluation of Python math expressions."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from arcticswarm.tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# Safe expression evaluator
# ---------------------------------------------------------------------------

# Binary operators allowed in expressions
_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Unary operators
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Comparison operators
_CMP_OPS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# Named constants the model can reference directly
_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
    "tau": math.tau,
}

# Whitelisted callable functions
_FUNCTIONS: dict[str, Any] = {
    # builtins
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    # math module
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
    "isnan": math.isnan,
    "isinf": math.isinf,
}


def _safe_eval(node: ast.expr) -> Any:
    """Recursively evaluate an AST node, allowing only safe math operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    # Numeric / boolean literals: 42, 3.14, True
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex, bool)):
            return node.value
        raise ValueError(f"Unsupported literal: {node.value!r}")

    # Named constants: pi, e, inf
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if node.id in _FUNCTIONS:
            # Allow passing function names (e.g. for higher-order use), but
            # in practice the model will call them, handled by ast.Call below.
            return _FUNCTIONS[node.id]
        raise ValueError(
            f"Unknown name: '{node.id}'. "
            f"Available constants: {', '.join(sorted(_CONSTANTS))}. "
            f"Available functions: {', '.join(sorted(_FUNCTIONS))}."
        )

    # Binary operations: a + b, a * b, a ** b
    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))

    # Unary operations: -x, +x
    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))

    # Function calls: sqrt(4), round(3.7, 2)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed (e.g. sqrt(4)).")
        func_name = node.func.id
        if func_name not in _FUNCTIONS:
            raise ValueError(
                f"Unknown function: '{func_name}'. "
                f"Available: {', '.join(sorted(_FUNCTIONS))}."
            )
        args = [_safe_eval(arg) for arg in node.args]
        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")
        return _FUNCTIONS[func_name](*args)

    # Lists and tuples: [1, 2, 3] for sum([1,2,3]), min(1,2,3)
    if isinstance(node, ast.List):
        return [_safe_eval(elt) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(elt) for elt in node.elts)

    # Comparisons: a > b, a == b (returns bool)
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_fn = _CMP_OPS.get(type(op))
            if op_fn is None:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            right = _safe_eval(comparator)
            if not op_fn(left, right):
                return False
            left = right
        return True

    # Ternary: a if condition else b
    if isinstance(node, ast.IfExp):
        return _safe_eval(node.body) if _safe_eval(node.test) else _safe_eval(node.orelse)

    raise ValueError(
        f"Unsupported expression type: {type(node).__name__}. "
        "Only arithmetic, math functions, and comparisons are allowed."
    )


def _format_result(value: Any) -> str:
    """Format a numeric result for display."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        # Show integers without decimal point, otherwise up to 10 significant digits
        if value == int(value) and math.isfinite(value):
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class CalculatorTool(BaseTool):
    """Evaluate math expressions safely without shell or SQL overhead."""

    name = "calculator"
    description = (
        "Evaluate a Python math expression and return the numeric result. "
        "Use for arithmetic, percentages, unit conversions, and quick calculations "
        "instead of guessing or launching a shell. "
        "Supports: +, -, *, /, //, %, ** operators; "
        "functions like sqrt, log, sin, cos, ceil, floor, abs, round, min, max; "
        "constants pi, e, tau, inf. "
        "Examples: '(1200 - 950) / 950 * 100', 'sqrt(3**2 + 4**2)', 'round(pi * 2.5**2, 2)'."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A Python math expression to evaluate. "
                        "Examples: '2 ** 10', 'sqrt(144)', '(revenue - cost) / cost * 100' "
                        "(where revenue and cost are replaced with actual numbers)."
                    ),
                },
            },
            "required": ["expression"],
        }

    def execute(self, *, expression: str, **_: Any) -> ToolResult:
        expression = expression.strip()
        if not expression:
            return ToolResult(error="Empty expression.", is_error=True)

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return ToolResult(error=f"Syntax error: {exc}", is_error=True)

        try:
            result = _safe_eval(tree)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, ArithmeticError) as exc:
            return ToolResult(error=f"Evaluation error: {exc}", is_error=True)

        return ToolResult(output=_format_result(result))
