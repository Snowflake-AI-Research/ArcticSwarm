"""Rule-based (verifiable) unit test evaluation.

Provides helpers to extract JSON from agent responses and run Python
unit-test code against the extracted answer.
"""

from __future__ import annotations

import json
import re


def run_one(test_id: str, test_code: str, answer: dict):
    """Run a unit test with a dict answer."""
    ns = {"__name__": f"test_{test_id}"}  # isolate globals per test
    code_obj = compile(test_code, filename=f"<test:{test_id}>", mode="exec")
    exec(code_obj, ns, ns)
    if "grade" not in ns or not callable(ns["grade"]):
        raise ValueError(f"{test_id}: no callable grade(answer) found")
    return ns["grade"](answer)


def extract_json_from_tags(text: str) -> dict | None:
    """Extract JSON from ``<json>...</json>`` tags and return as dict."""
    pattern = r'<json>\s*([\s\S]*?)\s*</json>'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
        return json.loads(json_str)
    return None


def extract_json_from_markdown(text: str) -> dict | None:
    """Extract JSON from a markdown ````` json```` fenced block."""
    pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(pattern, text)
    if match:
        json_str = match.group(1).strip()
        return json.loads(json_str)
    return None


def extract_raw_json(text: str) -> dict | None:
    """Extract the outermost ``{...}`` JSON object from free-form text."""
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return None

    json_candidate = text[start_idx:end_idx + 1]
    return json.loads(json_candidate)
