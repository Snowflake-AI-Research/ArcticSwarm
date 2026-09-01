"""Shared utilities for external dataset loaders."""

import csv
import json
from pathlib import Path
from typing import Any


def write_unified_eval_csv(
    examples: list[dict[str, Any]],
    output_path: Path,
    dataset_name: str = "CUSTOM",
    eval_mode: str = "QA",
    reference_tools: list[str] | None = None,
) -> None:
    """Write examples in Arcticswarm's Unified_eval CSV format.

    Args:
        examples: List of dicts with keys: 'id', 'question', 'expected_answer'.
            Each example may additionally carry an optional ``'attributes'``
            dict whose keys are merged into the row's ATTRIBUTES JSON (on top
            of ``dataset`` / ``eval_mode`` / ``is_vip``).  Use it to preserve
            per-row benchmark metadata (e.g. WebWalkerQA's ``lang`` /
            ``domain`` / ``difficulty_level`` / ``root_url``).  Callers that
            don't set it are unaffected.
        output_path: Output CSV file path
        dataset_name: Dataset identifier (e.g., "BROWSECOMP_V1")
        eval_mode: "QA" or "INSIGHT"
        reference_tools: Expected tool names (default: ["web_search"])
    """
    if reference_tools is None:
        reference_tools = ["web_search"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header (same as Unified_eval_v227.csv)
        writer.writerow([
            "TURN_INDEX",
            "TURN",
            "PAST_TURNS",
            "TOOL_CHOICE",
            "TOOLS",
            "REFERENCE_MESSAGE",
            "CONV_ID",
            "TOOL_RESOURCES",
            "ATTRIBUTES",
            "TURNS",
            "DATE_OVERRIDE",
            "REFERENCE_TOOLS"
        ])

        for example in examples:
            conv_id = example["id"]
            question = example["question"]
            expected_answer = example["expected_answer"]

            # Build JSON fields
            turns = json.dumps([{
                "Role": "user",
                "Content": [{"Type": "text", "Text": question}]
            }])

            reference = json.dumps({"text": expected_answer})

            attributes = json.dumps({
                "dataset": dataset_name,
                "eval_mode": eval_mode,
                "is_vip": True,
                # Optional per-example metadata (merged last so a benchmark
                # can override defaults if it ever needs to).
                **(example.get("attributes") or {}),
            })

            reference_tools_json = json.dumps(reference_tools)

            # Write row
            writer.writerow([
                "0",                        # TURN_INDEX
                question,                   # TURN
                "[]",                       # PAST_TURNS
                '{"type": "auto"}',        # TOOL_CHOICE
                "[]",                       # TOOLS
                reference,                  # REFERENCE_MESSAGE
                conv_id,                    # CONV_ID
                "{}",                       # TOOL_RESOURCES
                attributes,                 # ATTRIBUTES
                turns,                      # TURNS
                "",                         # DATE_OVERRIDE
                reference_tools_json        # REFERENCE_TOOLS
            ])

    print(f"✅ Wrote {len(examples)} examples to {output_path}")
