"""Load and parse eval cases from unified_eval.csv.

Reads the CSV used by the Go eval pipeline and converts rows into
:class:`EvalCase` dataclass instances, with filtering by dataset,
VIP status, eval mode, and row limit.

Pass the CSV explicitly with ``eval.csv_path=...``; otherwise the bundled
BrowseComp CSV under ``data/`` is used.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The CSV can contain extremely long fields (tool descriptions, etc.)
csv.field_size_limit(sys.maxsize)

_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Default CSV resolution
# ---------------------------------------------------------------------------
def resolve_default_csv() -> Path:
    """Return the bundled default eval CSV (BrowseComp).

    Most runs pass ``eval.csv_path=...`` explicitly; this is only the
    fallback default, resolved from the committed CSVs under ``data/``.
    """
    for name in ("browsecomp_v1.csv", "browsecomp_plus_v1.csv"):
        cand = _DATA_DIR / name
        if cand.exists():
            return cand
    matches = sorted(_DATA_DIR.glob("browsecomp*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No default eval CSV found in {_DATA_DIR}. Pass eval.csv_path explicitly."
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """A single evaluation case parsed from unified_eval.csv."""

    conv_id: str
    turn_index: int
    question: str
    reference_answer: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_resources: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    reference_tools: list[str] = field(default_factory=list)
    date_override: str = ""
    past_turns: list[dict[str, Any]] = field(default_factory=list)
    unit_test: str = ""

    # Absolute paths to image files that should be attached directly to the
    # initial user message as vision blocks (image-before-text). Populated by
    # dataset loaders for multimodal benchmarks. Empty means the
    # question is text-only and the runner sends a plain string as today.
    attached_images: list[str] = field(default_factory=list)

    # Derived convenience properties
    @property
    def dataset(self) -> str:
        return str(self.attributes.get("dataset", ""))

    @property
    def dataset_lineage(self) -> list[str]:
        return self.attributes.get("dataset_lineage", [])

    @property
    def eval_mode(self) -> str:
        return str(self.attributes.get("eval_mode", "QA")).upper()

    @property
    def is_vip(self) -> bool:
        return bool(self.attributes.get("is_vip", False))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _safe_json(value: str) -> Any:
    """Parse a JSON string, returning an empty structure on failure."""
    if not value or not value.strip():
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_question(row: dict[str, str]) -> str:
    """Extract the user question text from TURNS or TURN column."""
    # Prefer TURNS (list of full turn objects)
    turns_raw = _safe_json(row.get("TURNS", ""))
    if isinstance(turns_raw, list) and turns_raw:
        # Get the last user turn
        for turn in reversed(turns_raw):
            if isinstance(turn, dict) and turn.get("Role") == "user":
                content = turn.get("Content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("Type") == "text":
                            return block.get("Text", "")
                elif isinstance(content, str):
                    return content

    # Fallback to TURN column
    turn_raw = _safe_json(row.get("TURN", ""))
    if isinstance(turn_raw, dict):
        content = turn_raw.get("Content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("Type") == "text":
                    return block.get("Text", "")
        elif isinstance(content, str):
            return content

    return ""


def _extract_reference_answer(row: dict[str, str]) -> str:
    """Extract the reference answer text from REFERENCE_MESSAGE."""
    ref = _safe_json(row.get("REFERENCE_MESSAGE", ""))
    if isinstance(ref, dict):
        return ref.get("text", "")
    return ""


def _extract_reference_tools(row: dict[str, str]) -> list[str]:
    """Extract expected tool names from REFERENCE_TOOLS."""
    tools = _safe_json(row.get("REFERENCE_TOOLS", ""))
    if isinstance(tools, list):
        return [str(t) for t in tools if t]
    return []


def _parse_row(row: dict[str, str]) -> EvalCase | None:
    """Convert a CSV row dict into an EvalCase, or None if unparseable."""
    question = _extract_question(row)
    if not question:
        return None

    conv_id = row.get("CONV_ID", "")
    if not conv_id:
        return None

    try:
        turn_index = int(row.get("TURN_INDEX", "0"))
    except (ValueError, TypeError):
        turn_index = 0

    reference_answer = _extract_reference_answer(row)
    tools = _safe_json(row.get("TOOLS", "")) or []
    tool_resources = _safe_json(row.get("TOOL_RESOURCES", "")) or {}
    attributes = _safe_json(row.get("ATTRIBUTES", "")) or {}
    reference_tools = _extract_reference_tools(row)
    date_override = row.get("DATE_OVERRIDE", "")
    past_turns = _safe_json(row.get("PAST_TURNS", "")) or []
    unit_test = row.get("UNIT_TEST", "").strip()

    return EvalCase(
        conv_id=conv_id,
        turn_index=turn_index,
        question=question,
        reference_answer=reference_answer,
        tools=tools if isinstance(tools, list) else [],
        tool_resources=tool_resources if isinstance(tool_resources, dict) else {},
        attributes=attributes if isinstance(attributes, dict) else {},
        reference_tools=reference_tools,
        date_override=date_override,
        past_turns=past_turns if isinstance(past_turns, list) else [],
        unit_test=unit_test,
    )


def _resolve_eval_csv_path(csv_path: str | Path | None, datasets: list[str] | None) -> Path:
    """Resolve the eval CSV path (explicit path, custom datasets, or bundled default)."""
    explicit_csv_path = csv_path
    if csv_path is None:
        if datasets:
            from arcticswarm.eval.custom_datasets import is_custom_dataset, resolve_custom_csv

            all_custom = all(is_custom_dataset(d) for d in datasets)
            if all_custom and len(datasets) == 1:
                custom_path = resolve_custom_csv(datasets[0])
                if custom_path is not None:
                    csv_path = custom_path

        if csv_path is None:
            csv_path = resolve_default_csv()
    path = Path(csv_path)
    if not path.is_absolute():
        if not path.exists():
            candidate = Path.cwd()
            for _ in range(10):
                if (candidate / path).exists():
                    path = candidate / path
                    break
                candidate = candidate.parent

    if not path.exists():
        if explicit_csv_path is not None:
            raise FileNotFoundError(
                f"eval.csv_path points at a CSV that does not exist: {explicit_csv_path!r} "
                f"(resolved to {path}). Check the path in your config, or regenerate the "
                "benchmark CSVs with `bash scripts/fetch_datasets.sh`. "
                "To evaluate your own data, see docs/custom_evaluation.md."
            )
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. Regenerate the benchmark CSVs with "
            "`bash scripts/fetch_datasets.sh`, or point eval.csv_path at your own data "
            "(see docs/custom_evaluation.md)."
        )
    return path


def _load_cases_from_unified_csv(
    path: Path,
    *,
    datasets_upper: list[str] | None,
    vip_only: bool,
    eval_mode_upper: str | None,
    limit: int,
    conv_id: str | None,
    offset: int = 0,
) -> list[EvalCase]:
    """Read unified_eval CSV at *path* and apply the usual filters.

    ``offset`` skips the first N matching cases (after dataset / vip / mode
    filtering) before counting toward ``limit``. The skip is applied stream
    rather than slicing the full list, so it stays cheap on large CSVs.
    """
    logger.info("Loading eval cases from %s", path)
    cases: list[EvalCase] = []
    skipped = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = _parse_row(row)
            if case is None:
                continue

            if conv_id:
                if case.conv_id != conv_id:
                    continue
                cases.append(case)
                break

            if datasets_upper:
                row_dataset = case.dataset.upper()
                row_lineage = [d.upper() for d in case.dataset_lineage]
                if row_dataset not in datasets_upper and not any(
                    d in row_lineage for d in datasets_upper
                ):
                    continue

            if vip_only and not case.is_vip:
                continue

            if eval_mode_upper and case.eval_mode != eval_mode_upper:
                continue

            if skipped < offset:
                skipped += 1
                continue

            cases.append(case)

            if 0 < limit <= len(cases):
                break

    return cases


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_eval_cases(
    csv_path: str | Path | None = None,
    *,
    datasets: list[str] | None = None,
    vip_only: bool = True,
    eval_mode: str | None = None,
    limit: int = 0,
    offset: int = 0,
    conv_id: str | None = None,
) -> list[EvalCase]:
    """Load eval cases from *csv_path*, applying filters.

    Parameters
    ----------
    csv_path:
        Path to the eval CSV (absolute or relative to cwd).
        When ``None`` (the default), the bundled BrowseComp CSV under
        ``data/`` is used.
    datasets:
        If given, only include rows whose ``dataset`` or ``dataset_lineage``
        matches any of the provided names (case-insensitive).
    vip_only:
        If ``True``, only include rows with ``is_vip: true``.
    eval_mode:
        If given (``"QA"`` or ``"INSIGHT"``), filter by eval_mode attribute.
    limit:
        Maximum number of cases to return.  ``0`` means no limit.
    offset:
        Number of matching cases to skip before applying ``limit``.
        ``offset=100, limit=100`` returns cases 100..199 (0-indexed).
        Case ordering is deterministic, so the slice is stable across runs
        and reproducible across modes — useful for extending a previous
        ``limit=N`` run without re-running the first N cases.
    conv_id:
        If given, return only the case with this exact ``conv_id``,
        bypassing dataset, VIP, eval_mode, and limit filters.
    """
    eval_mode_upper = eval_mode.upper() if eval_mode else None
    datasets_upper = [d.upper() for d in datasets] if datasets else None

    # Single conv_id: load the single matching case from the unified CSV.
    if conv_id:
        path = _resolve_eval_csv_path(csv_path, datasets)
        cases = _load_cases_from_unified_csv(
            path,
            datasets_upper=datasets_upper,
            vip_only=vip_only,
            eval_mode_upper=eval_mode_upper,
            limit=limit,
            conv_id=conv_id,
        )
        # ``conv_id`` short-circuits to a single case; offset is meaningless.
        logger.info(
            "Loaded %d eval cases (datasets=%s, vip_only=%s, eval_mode=%s, limit=%d, offset=%d, conv_id=%s)",
            len(cases),
            datasets,
            vip_only,
            eval_mode,
            limit,
            offset,
            conv_id,
        )
        return cases

    merged: list[EvalCase] = []

    if datasets is None:
        path = _resolve_eval_csv_path(csv_path, None)
        merged = _load_cases_from_unified_csv(
            path,
            datasets_upper=None,
            vip_only=vip_only,
            eval_mode_upper=eval_mode_upper,
            limit=limit,
            conv_id=None,
            offset=offset,
        )
        logger.info(
            "Loaded %d eval cases (datasets=%s, vip_only=%s, eval_mode=%s, limit=%d, offset=%d, conv_id=%s)",
            len(merged),
            datasets,
            vip_only,
            eval_mode,
            limit,
            offset,
            conv_id,
        )
        return merged

    path = _resolve_eval_csv_path(csv_path, datasets)
    merged.extend(
        _load_cases_from_unified_csv(
            path,
            datasets_upper=datasets_upper,
            vip_only=vip_only,
            eval_mode_upper=eval_mode_upper,
            limit=0,
            conv_id=None,
        )
    )

    if eval_mode_upper:
        merged = [c for c in merged if c.eval_mode == eval_mode_upper]

    if vip_only:
        merged = [c for c in merged if c.is_vip]

    # In the merged-load path (multiple datasets), apply offset/limit to the
    # unified deterministic ordering after all sources are concatenated and
    # post-filtered.
    if offset > 0:
        merged = merged[offset:]
    if 0 < limit <= len(merged):
        merged = merged[:limit]

    logger.info(
        "Loaded %d eval cases (datasets=%s, vip_only=%s, eval_mode=%s, limit=%d, offset=%d, conv_id=%s)",
        len(merged),
        datasets,
        vip_only,
        eval_mode,
        limit,
        offset,
        conv_id,
    )
    return merged
