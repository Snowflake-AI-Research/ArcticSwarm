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

"""Logging, observability, and diagnostics utilities for Arcticswarm.

Extracted from :mod:`arcticswarm.agent` to keep the core agentic loop focused
on orchestration.  Everything here is side-effect-only (logging, file writes)
and does not affect agent behavior or performance.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from arcticswarm.agent import Agent, TokenUsage
    from arcticswarm.context_management import ContextBudget
    from arcticswarm.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval-awareness: browsing contamination tracking
# ---------------------------------------------------------------------------


class BrowsingContaminationStats:
    """Thread-safe counter for browsing results filtered due to eval contamination."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_browsing_results: int = 0
        self.excluded_results: int = 0

    def record(self, *, excluded: bool) -> None:
        with self._lock:
            self.total_browsing_results += 1
            if excluded:
                self.excluded_results += 1

    def summary(self) -> str:
        with self._lock:
            if self.total_browsing_results == 0:
                return "Eval-awareness: no browsing results observed"
            pct = (self.excluded_results / self.total_browsing_results) * 100
            return (
                f"Eval-awareness: {self.excluded_results}/{self.total_browsing_results} "
                f"browsing results excluded ({pct:.1f}%)"
            )

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            pct = 0.0
            if self.total_browsing_results > 0:
                pct = (self.excluded_results / self.total_browsing_results) * 100
            return {
                "total_browsing_results": self.total_browsing_results,
                "excluded_results": self.excluded_results,
                "excluded_pct": round(pct, 1),
            }


# Module-level shared instance — reset via reset_contamination_stats() between eval runs
_contamination_stats = BrowsingContaminationStats()


def get_contamination_stats() -> BrowsingContaminationStats:
    """Return the current module-level contamination stats."""
    return _contamination_stats


def reset_contamination_stats() -> BrowsingContaminationStats:
    """Reset and return a fresh contamination stats tracker."""
    global _contamination_stats
    _contamination_stats = BrowsingContaminationStats()
    return _contamination_stats


# ---------------------------------------------------------------------------
# Compaction stats logging
# ---------------------------------------------------------------------------


def log_compaction_stats(
    compaction_count: int,
    total_llm_calls: int,
    context_budget: "ContextBudget",
) -> None:
    """Log end-of-turn compaction statistics if any compaction occurred."""
    if compaction_count == 0:
        return
    rate = (
        (compaction_count / total_llm_calls * 100)
        if total_llm_calls > 0 else 0.0
    )
    logger.info(
        "Compaction stats: %d compactions / %d LLM calls (%.1f%%), "
        "peak context utilization %.0f%%",
        compaction_count,
        total_llm_calls,
        rate,
        context_budget.peak_input_tokens / max(context_budget.context_limit, 1) * 100,
    )


# ---------------------------------------------------------------------------
# Git snapshot – captures code state for reproducibility
# ---------------------------------------------------------------------------


def capture_git_snapshot(
    output_dir: Path,
    *,
    prior_trajectory_count: int | None = None,
    resume_flags: dict[str, bool] | None = None,
) -> None:
    """Save git commit info, local diffs, and settings to *output_dir*/code_snapshot.json.

    Captures staged diffs, unstaged diffs, untracked file contents (for
    arcticswarm/ files), and stash state so that the exact code that produced
    the eval run can be fully reconstructed later.  Also embeds a redacted
    copy of the settings file (``config_files.json``) so we can tell which
    API keys / provider settings were active.

    If a ``code_snapshot.json`` already exists in *output_dir* (i.e. this
    is an ``eval.resume`` / ``rerun_*`` re-launch), the new snapshot is
    written to ``code_snapshot_<UTC-timestamp>.json`` instead so the
    original launch's snapshot is preserved.  ``prior_trajectory_count``
    and ``resume_flags`` are recorded in the snapshot so future audits
    can reconstruct what state the directory was in at this launch.
    """

    def _run(cmd: list[str], timeout: int = 30) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            ).stdout.strip()
        except Exception:
            return ""

    repo_root = _run(["git", "rev-parse", "--show-toplevel"])
    if not repo_root:
        logger.warning("Not inside a git repo – skipping code snapshot")
        return

    snapshot: dict[str, Any] = {}

    # Launch metadata (used to disambiguate resumes of the same output_dir).
    from datetime import datetime, timezone
    launched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot["launched_at"] = launched_at
    if prior_trajectory_count is not None:
        snapshot["prior_trajectory_count"] = prior_trajectory_count
    if resume_flags is not None:
        snapshot["resume_flags"] = resume_flags

    # Current branch
    snapshot["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    # HEAD commit
    snapshot["commit"] = _run(["git", "rev-parse", "HEAD"])
    snapshot["commit_short"] = _run(["git", "rev-parse", "--short", "HEAD"])
    snapshot["commit_message"] = _run(["git", "log", "-1", "--pretty=%B"])
    snapshot["commit_author"] = _run(["git", "log", "-1", "--pretty=%an <%ae>"])
    snapshot["commit_date"] = _run(["git", "log", "-1", "--pretty=%ci"])

    # Dirty state — staged + unstaged diffs for tracked files
    staged_diff = _run(["git", "diff", "--cached"])
    unstaged_diff = _run(["git", "diff"])
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"])

    is_dirty = bool(staged_diff or unstaged_diff or untracked)
    snapshot["dirty"] = is_dirty

    snapshot["staged_diff"] = staged_diff or None
    snapshot["unstaged_diff"] = unstaged_diff or None

    # Untracked files: list names AND capture content for arcticswarm/ files
    # so we can reproduce the exact code state (other dirs are skipped to
    # keep the snapshot manageable).
    untracked_list = untracked.splitlines() if untracked else []
    snapshot["untracked_files"] = untracked_list

    untracked_contents: dict[str, str] = {}
    for fpath in untracked_list:
        if not fpath.startswith("arcticswarm/"):
            continue
        full = Path(repo_root) / fpath
        try:
            text = full.read_text(errors="replace")
            if len(text) <= 200_000:  # skip very large files
                untracked_contents[fpath] = text
        except Exception:
            pass
    if untracked_contents:
        snapshot["untracked_file_contents"] = untracked_contents

    # Stash state — record so we can detect stashed-but-not-applied changes
    stash_list = _run(["git", "stash", "list"])
    snapshot["stash_list"] = stash_list or None

    snapshot["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S %z")

    # ----- Settings snapshot (redacted) -----
    from arcticswarm.config import settings_json_path
    settings_path = settings_json_path()
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            # Redact API key values but keep the key names so we know which
            # providers were configured.  Show first-4 / last-4 chars only.
            redacted: dict[str, Any] = {}
            for k, v in settings.items():
                if isinstance(v, str) and ("key" in k.lower() or "token" in k.lower() or "secret" in k.lower()):
                    if len(v) > 12:
                        redacted[k] = v[:4] + "..." + v[-4:]
                    elif v:
                        redacted[k] = "***"
                    else:
                        redacted[k] = ""
                else:
                    redacted[k] = v
            snapshot["config_files"] = redacted
        except Exception as exc:
            snapshot["config_files_error"] = str(exc)

    base_path = output_dir / "code_snapshot.json"
    if base_path.exists():
        # Re-launch into an existing output_dir (eval.resume / rerun_*).
        # Don't overwrite the original; write a timestamped sibling so
        # both launches' code states are preserved on disk.
        ts_for_filename = launched_at.replace(":", "").replace("-", "")
        snapshot_path = output_dir / f"code_snapshot_{ts_for_filename}.json"
    else:
        snapshot_path = base_path
    snapshot_path.write_text(json.dumps(snapshot, indent=2))
    logger.info("Code snapshot saved to %s (commit=%s, dirty=%s)",
                snapshot_path, snapshot["commit_short"], is_dirty)


# ---------------------------------------------------------------------------
# Web fetch instrumentation (used by WebFetchTool)
# ---------------------------------------------------------------------------


class WebFetchInstrumentor:
    """Counters and per-fetch logging for WebFetchTool.

    Extracted here so that ``tools/web_fetch.py`` stays focused on the
    fetch/scrape logic while all observability lives in one place.
    """

    def __init__(self) -> None:
        self.total_fetches: int = 0
        self.grounding_attempts: int = 0
        self.grounding_success: int = 0
        self.jina_success: int = 0
        self.serper_success: int = 0
        self.requests_success: int = 0
        self.cache_hits: int = 0
        self.total_failures: int = 0
        self._fetch_log: list[dict[str, Any]] = []

    def record_fetch(
        self,
        url: str,
        tier: str,
        success: bool,
        error: str | None,
        latency_ms: float,
        content_chars: int,
    ) -> None:
        """Append an entry to the per-fetch log."""
        self._fetch_log.append({
            "url": url,
            "tier": tier,
            "success": success,
            "error": error,
            "latency_ms": round(latency_ms, 1),
            "content_chars": content_chars,
        })

    def drain_fetch_log(self) -> list[dict[str, Any]]:
        """Return and clear the per-fetch log."""
        log_copy = list(self._fetch_log)
        self._fetch_log.clear()
        return log_copy

    def log_and_reset_stats(self) -> None:
        """Log web fetch stats and reset counters."""
        if self.total_fetches > 0:
            if self.grounding_attempts > 0:
                grounding_fail = self.grounding_attempts - self.grounding_success
                grounding_fallback_pct = 100.0 * grounding_fail / self.grounding_attempts
                logger.info(
                    "Web fetch stats: %d total (grounding=%d/%d attempts, %.1f%% fallback, "
                    "jina=%d, serper=%d, requests=%d, cache=%d, failures=%d)",
                    self.total_fetches,
                    self.grounding_success,
                    self.grounding_attempts,
                    grounding_fallback_pct,
                    self.jina_success,
                    self.serper_success,
                    self.requests_success,
                    self.cache_hits,
                    self.total_failures,
                )
            else:
                logger.info(
                    "Web fetch stats: %d total (grounding=%d, jina=%d, serper=%d, requests=%d, cache=%d, failures=%d)",
                    self.total_fetches,
                    self.grounding_success,
                    self.jina_success,
                    self.serper_success,
                    self.requests_success,
                    self.cache_hits,
                    self.total_failures,
                )
        self.total_fetches = 0
        self.grounding_attempts = 0
        self.grounding_success = 0
        self.jina_success = 0
        self.serper_success = 0
        self.requests_success = 0
        self.cache_hits = 0
        self.total_failures = 0


# ---------------------------------------------------------------------------
# Web search stats logging
# ---------------------------------------------------------------------------


def log_web_search_stats(tools: dict[str, "BaseTool"]) -> None:
    """Log aggregated web search stats and reset counters."""
    ws = tools.get("web_search")
    if ws is not None and hasattr(ws, "log_and_reset_stats"):
        ws.log_and_reset_stats()


# ---------------------------------------------------------------------------
# Web fetch stats logging
# ---------------------------------------------------------------------------


def log_web_fetch_stats(tools: dict[str, "BaseTool"]) -> None:
    """Log aggregated web fetch stats and reset counters."""
    wf = tools.get("web_fetch")
    if wf is not None and hasattr(wf, "log_and_reset_stats"):
        wf.log_and_reset_stats()


# ---------------------------------------------------------------------------
# Score-aware truncation (extracted from Agent)
# ---------------------------------------------------------------------------

_SCORE_ANNOTATION_RE: re.Pattern[str] | None = None
_SCORE_ANNOTATION_RE_RELAXED: re.Pattern[str] | None = None


def _get_score_re(relaxed: bool = False) -> re.Pattern[str]:
    """Return compiled regex for [Source Quality: ... composite=X/40] annotations."""
    global _SCORE_ANNOTATION_RE, _SCORE_ANNOTATION_RE_RELAXED
    if relaxed:
        if _SCORE_ANNOTATION_RE_RELAXED is None:
            _SCORE_ANNOTATION_RE_RELAXED = re.compile(
                r"\[Source Quality:.*?composite=([\d.]+)/40\)?\]"
            )
        return _SCORE_ANNOTATION_RE_RELAXED
    if _SCORE_ANNOTATION_RE is None:
        _SCORE_ANNOTATION_RE = re.compile(
            r"\[Source Quality:.*?composite=([\d.]+)/40\]"
        )
    return _SCORE_ANNOTATION_RE


def score_aware_truncate(
    msgs: list[dict[str, Any]],
    high_score_threshold: float = 30.0,
    mid_high_score_threshold: float = 22.0,
    mid_low_score_threshold: float = 12.0,
    low_score_max_chars: int = 500,
    mid_low_score_max_chars: int = 2000,
    mid_high_score_max_chars: int = 4000,
    high_score_max_chars: int = 8000,
    no_score_max_chars: int = 2000,
    relaxed_re: bool = False,
) -> list[dict[str, Any]]:
    """Deep-copy *msgs* and truncate tool_results based on source quality scores.

    Parses the ``[Source Quality: ... composite=X/40]`` annotation.
    Truncation tiers (4-tier system to smooth boundary effects):

    - composite >= 30 (high):      keep *high_score_max_chars* (8000)
    - composite >= 22 (mid-high):  keep *mid_high_score_max_chars* (4000)
    - composite >= 12 (mid-low):   keep *mid_low_score_max_chars* (2000)
    - composite < 12 (low):        keep *low_score_max_chars* (500)
    - no annotation (non-web_fetch): keep *no_score_max_chars* (2000)

    Only applied during compaction — the agent always sees full content
    on first use.
    """
    score_re = _get_score_re(relaxed=relaxed_re)
    out = copy.deepcopy(msgs)
    truncation_stats = {"high": 0, "mid_high": 0, "mid_low": 0, "low": 0, "no_score": 0, "skipped": 0}

    for m in out:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue

            # Extract text from content (may be str or list-of-blocks)
            inner = block.get("content")
            text: str = ""
            text_ref: dict | None = None  # reference to mutate

            if isinstance(inner, str):
                text = inner
            elif isinstance(inner, list):
                for sub in inner:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        text = sub.get("text", "")
                        text_ref = sub
                        break

            if not text or len(text) <= low_score_max_chars:
                continue

            # Parse composite score from annotation
            match = score_re.search(text)
            if match:
                try:
                    composite = float(match.group(1))
                except (ValueError, TypeError):
                    logger.warning(
                        "Malformed composite score in annotation: %r",
                        match.group(1),
                    )
                    composite = 0.0
                if composite >= high_score_threshold:
                    max_chars = high_score_max_chars
                elif composite >= mid_high_score_threshold:
                    max_chars = mid_high_score_max_chars
                elif composite >= mid_low_score_threshold:
                    max_chars = mid_low_score_max_chars
                else:
                    max_chars = low_score_max_chars
            else:
                max_chars = no_score_max_chars

            if len(text) <= max_chars:
                truncation_stats["skipped"] += 1
                continue

            # Track which tier the truncation fell into
            if match:
                if composite >= high_score_threshold:
                    truncation_stats["high"] += 1
                elif composite >= mid_high_score_threshold:
                    truncation_stats["mid_high"] += 1
                elif composite >= mid_low_score_threshold:
                    truncation_stats["mid_low"] += 1
                else:
                    truncation_stats["low"] += 1
            else:
                truncation_stats["no_score"] += 1

            truncated_text = (
                text[:max_chars]
                + f"\n...[truncated from {len(text)} chars — score-aware compaction]"
            )
            if text_ref is not None:
                text_ref["text"] = truncated_text
            elif isinstance(inner, str):
                block["content"] = truncated_text

    total_truncated = sum(v for k, v in truncation_stats.items() if k != "skipped")
    if total_truncated > 0:
        logger.info(
            "Score-aware truncation: %d tool results truncated "
            "(high=%d, mid_high=%d, mid_low=%d, low=%d, no_score=%d), %d skipped (already short)",
            total_truncated,
            truncation_stats["high"],
            truncation_stats["mid_high"],
            truncation_stats["mid_low"],
            truncation_stats["low"],
            truncation_stats["no_score"],
            truncation_stats["skipped"],
        )
    return out


def truncate_tool_results(
    msgs: list[dict[str, Any]], max_chars: int = 2000,
) -> list[dict[str, Any]]:
    """Deep-copy *msgs* and truncate tool_result content blocks.

    Most context bloat comes from large web_fetch / pdf_read / bash
    outputs.  This preserves message structure but caps each
    tool_result's text at *max_chars*.
    """
    out = copy.deepcopy(msgs)
    for m in out:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            if isinstance(inner, str) and len(inner) > max_chars:
                block["content"] = inner[:max_chars] + "\n...[truncated]"
            elif isinstance(inner, list):
                for sub in inner:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        txt = sub.get("text", "")
                        if len(txt) > max_chars:
                            sub["text"] = txt[:max_chars] + "\n...[truncated]"
    return out


# ---------------------------------------------------------------------------
# Empty-response fallback case logging
# ---------------------------------------------------------------------------

_fallback_case_counter: int = 0
_fallback_case_lock = threading.Lock()


def log_empty_fallback(
    *,
    model: str,
    max_tokens: int,
    total_llm_calls: int,
    system_prompt: str | None,
    input_messages: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]],
    last_user_text: str,
    num_messages: int,
    fallback_attempts: list[dict],
    streaming: bool,
    output_dir: str = "",
    primary_raw_event_log: list[dict[str, Any]] | None = None,
    primary_stop_reason: str = "",
    primary_usage: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    context_utilization: float = 0.0,
) -> None:
    """Write a per-case log file for empty-response fallback events.

    Writes a single combined markdown file at
    ``{output_dir}/empty_fallback/case_NNN_TS.md`` containing:
      1. Human-readable prose (timestamp, model, primary response,
         fallback attempts) — top of the file.
      2. A ``## Structured Diagnostic Data`` section with a fenced
         ```json``` block of the full record (raw streaming events,
         fallback-attempt results, etc.) — bottom of the file.

    Tools that need machine-readable data can extract the JSON fence;
    humans see the prose first.

    When *output_dir* is not set, falls back to ``./docs/`` prefix.
    """
    global _fallback_case_counter
    with _fallback_case_lock:
        _fallback_case_counter += 1
        case_num = _fallback_case_counter

    base = output_dir if output_dir else os.path.join(".", "docs")
    out_dir = os.path.join(base, "empty_fallback")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"case_{case_num:03d}_{ts}.md"
    out_filepath = os.path.join(out_dir, out_filename)

    # Extract original question from the first user message
    original_question = ""
    for msg in input_messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                original_question = content
            elif isinstance(content, list):
                original_question = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if original_question.strip():
                break

    # Build message role summary
    role_summary = [msg.get("role", "?") for msg in input_messages]

    # Tool definitions summary
    tool_names = []
    for td in tool_definitions:
        if isinstance(td, dict):
            tool_names.append(td.get("name", "?"))
        else:
            tool_names.append(str(td))

    fallback_models_str = ", ".join(a["model"] for a in fallback_attempts)
    report = (
        f"# Empty Response Fallback — Case {case_num}\n\n"
        f"**Timestamp**: {datetime.now().isoformat()}\n"
        f"**Primary model**: {model}\n"
        f"**Reasoning effort**: {reasoning_effort}\n"
        f"**Fallback models tried**: {fallback_models_str}\n"
        f"**Streaming**: {streaming}\n"
        f"**Conversation length**: {num_messages} messages\n"
        f"**Turn number**: {total_llm_calls}\n"
        f"**Max tokens requested**: {max_tokens}\n"
        f"**Context utilization**: {context_utilization:.1%}\n"
        f"**Primary stop_reason**: {primary_stop_reason!r}\n"
        f"**Primary usage**: {json.dumps(primary_usage or {})}\n\n"
        f"## Original Question (truncated to 2000 chars)\n\n"
        f"```\n{original_question[:2000]}\n```\n\n"
        f"## Last User/Tool Input (truncated to 2000 chars)\n\n"
        f"```\n{last_user_text[:2000]}\n```\n\n"
        f"## Message History Summary\n\n"
        f"**Roles**: {' → '.join(role_summary)}\n\n"
        f"**Tools available** ({len(tool_names)}): {', '.join(tool_names)}\n\n"
        f"## System Prompt (truncated to 3000 chars)\n\n"
        f"```\n{(system_prompt or '')[:3000]}\n```\n\n"
        f"## Full Conversation Messages\n\n"
        f"```json\n{json.dumps(input_messages, indent=2, default=str)[:20000]}\n```\n\n"
        f"## Primary Model Response\n\n"
        f"Empty `content_blocks` — no text or tool_use returned.\n\n"
    )

    # Add raw streaming event log if available
    if primary_raw_event_log:
        report += "## Raw Streaming Events (Primary Model)\n\n"
        report += f"Total events: {len(primary_raw_event_log)}\n\n"
        # Summarise event type counts
        from collections import Counter
        evt_counts = Counter(e.get("type", "?") for e in primary_raw_event_log)
        report += "| Event Type | Count |\n|---|---:|\n"
        for etype, cnt in evt_counts.most_common():
            report += f"| {etype} | {cnt} |\n"
        report += "\n```json\n"
        report += json.dumps(primary_raw_event_log, indent=2, default=str)[:15000]
        report += "\n```\n\n"
    else:
        report += "## Raw Streaming Events (Primary Model)\n\nNot captured (non-streaming or not available).\n\n"

    for i, attempt in enumerate(fallback_attempts, 1):
        fb_model = attempt["model"]
        resp = attempt["response"]
        error = attempt["error"]

        report += f"## Fallback Attempt {i}: {fb_model}\n\n"

        if error:
            report += (
                f"**Status**: FAILED\n\n"
                f"**Error**:\n```\n{error[:2000]}\n```\n\n"
            )
        elif resp and resp.content_blocks:
            parts = []
            for block in resp.content_blocks:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        parts.append(f"[tool_use: {block.get('name', '?')}({block.get('input', {})})]")
            output = "\n".join(parts) if parts else "(empty)"
            report += (
                f"**Status**: SUCCESS\n\n"
                f"**stop_reason**: {resp.stop_reason}\n"
                f"**input_tokens**: {resp.input_tokens}, **output_tokens**: {resp.output_tokens}\n\n"
                f"### Output\n\n"
                f"```\n{output[:5000]}\n```\n\n"
            )
        else:
            resp_details = "(no response object)"
            if resp:
                resp_details = (
                    f"stop_reason={resp.stop_reason!r}, "
                    f"input_tokens={resp.input_tokens}, "
                    f"output_tokens={resp.output_tokens}, "
                    f"content_blocks={resp.content_blocks!r}"
                )
            report += (
                f"**Status**: EMPTY (no content_blocks)\n\n"
                f"**Response details**: {resp_details}\n\n"
            )

    # Build the structured-data record (same fields the legacy JSON
    # sidecar carried) and append it at the end of the markdown file
    # inside a fenced ```json block.  Consumers that need
    # machine-readable data can extract the fence; humans see the
    # prose first.
    json_record: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "case_num": case_num,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "streaming": streaming,
        "turn_number": total_llm_calls,
        "num_messages": num_messages,
        "context_utilization": round(context_utilization, 4),
        "primary_stop_reason": primary_stop_reason,
        "primary_usage": primary_usage or {},
        "primary_raw_event_log": primary_raw_event_log or [],
        "last_user_text": last_user_text[:5000],
        "original_question": original_question[:3000],
        "fallback_attempts": [
            {
                "model": a["model"],
                "error": a.get("error"),
                "success": bool(a.get("response") and a["response"].content_blocks),
                "stop_reason": getattr(a.get("response"), "stop_reason", None),
                "output_tokens": getattr(a.get("response"), "output_tokens", None),
            }
            for a in fallback_attempts
        ],
    }
    report += "## Structured Diagnostic Data\n\n"
    report += "```json\n"
    report += json.dumps(json_record, indent=2, default=str)
    report += "\n```\n"

    try:
        with open(out_filepath, "w") as f:
            f.write(report)
        logger.info("Empty-response fallback logged to %s", out_filepath)
    except Exception:
        logger.debug("Failed to write fallback log to %s", out_filepath, exc_info=True)


# ---------------------------------------------------------------------------
# Content serialization (shared by cli.py and eval/runner.py)
# ---------------------------------------------------------------------------


def serialize_content(content: Any) -> Any:
    """Convert Anthropic ContentBlock objects (Pydantic models) to plain dicts.

    Handles the three message shapes found in ``agent.messages``:
      - ``str`` (plain user text) → returned as-is
      - ``list[ContentBlock]`` (assistant response) → each block serialised
      - ``list[dict]`` (tool_result messages) → returned as-is
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[Any] = []
        for item in content:
            if isinstance(item, dict):
                out.append(item)
            elif hasattr(item, "model_dump"):
                # Anthropic Pydantic model (TextBlock, ToolUseBlock, etc.)
                out.append(item.model_dump())
            else:
                out.append(str(item))
        return out
    if hasattr(content, "model_dump"):
        return content.model_dump()
    return str(content)


def serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a JSON-safe copy of the agent's conversation messages.

    Preserves ``_llm_duration_seconds`` on assistant messages when present
    (injected by :class:`~arcticswarm.swarm.teammate._TimingCollector`),
    and ``_timestamp`` (injected by :meth:`Agent._append_msg`).
    ``_tool_duration_seconds`` inside ``tool_result`` dicts is carried
    through unchanged by :func:`serialize_content`.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {
            "role": msg["role"],
            "content": serialize_content(msg["content"]),
        }
        for key in ("_llm_duration_seconds", "_llm_output_tokens", "_llm_reasoning_tokens", "_timestamp"):
            if key in msg:
                entry[key] = msg[key]
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Noisy logger suppression
# ---------------------------------------------------------------------------

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "snowflake.connector",
    "snowflake.connector.connection",
    "snowflake.connector.cursor",
    "snowflake.connector.network",
    "urllib3",
    "urllib3.connectionpool",
)


def silence_noisy_loggers() -> None:
    """Suppress verbose INFO logs from httpx, snowflake connector, urllib3, etc."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Search/fetch API health probe
# ---------------------------------------------------------------------------
#
# Brave / Serper / Tavily / Jina silently degrade when the account loses
# access — most commonly because the free tier now requires a credit card on
# file, the monthly quota is exhausted, or the key was revoked.  When that
# happens the tools fall back to other providers and the run looks fine but
# produces garbage answers.  This probe runs once at the start of a run,
# makes a tiny request to each *configured* provider, and aborts up-front if
# any one is reachable but rejecting requests.

_UNHEALTHY_BODY_MARKERS: tuple[str, ...] = (
    "credit card",
    "payment required",
    "payment is required",
    "not enough credits",
    "insufficient credits",
    "quota exceeded",
    "out of credits",
    "subscription",
    "billing",
    "free tier",
    "upgrade your plan",
    "account suspended",
    "api key is invalid",
    "invalid api key",
    "unauthorized",
)


class SearchApiUnhealthyError(RuntimeError):
    """A configured search/fetch API is reachable but rejecting requests."""


def _classify_api_response(status_code: int, body: str) -> str | None:
    """Return a short reason if the status/body look unhealthy, else None."""
    if status_code == 401:
        return "401 Unauthorized — API key invalid or revoked"
    if status_code == 402:
        return "402 Payment Required — billing/credit card not on file or quota exhausted"
    body_lc = (body or "").lower()
    if status_code == 403:
        for kw in _UNHEALTHY_BODY_MARKERS:
            if kw in body_lc:
                return f"403 Forbidden — {kw!r} in response (billing/quota issue)"
        return "403 Forbidden — key may lack permission or account is suspended"
    if 400 <= status_code < 500 and status_code != 429:
        for kw in _UNHEALTHY_BODY_MARKERS:
            if kw in body_lc:
                return f"{status_code} — {kw!r} in response (billing/quota issue)"
    return None


def _probe_brave(api_key: str, timeout: int = 10) -> str | None:
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
                "User-Agent": "arcticswarm/0.1.0",
            },
            params={"q": "ping", "count": 1},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return f"network error during probe: {exc}"
    return _classify_api_response(resp.status_code, resp.text)


def _probe_serper(api_key: str, timeout: int = 10) -> str | None:
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": "ping", "num": 1},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return f"network error during probe: {exc}"
    return _classify_api_response(resp.status_code, resp.text)


def _probe_tavily(api_key: str, timeout: int = 10) -> str | None:
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={"api_key": api_key, "query": "ping", "max_results": 1},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return f"network error during probe: {exc}"
    return _classify_api_response(resp.status_code, resp.text)


def _probe_jina(api_key: str, timeout: int = 15) -> str | None:
    try:
        resp = requests.get(
            "https://r.jina.ai/https://example.com",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Engine": "direct",
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return f"network error during probe: {exc}"
    return _classify_api_response(resp.status_code, resp.text)


# Search providers share a fallback chain (web_search tries them in order);
# jina is the fetch provider. A single unhealthy provider must not block a run
# when a working search alternative exists.
_SEARCH_PROVIDERS = frozenset({"brave", "serper", "tavily"})


def check_search_api_health(
    config: Any,
    *,
    raise_on_unhealthy: bool = True,
) -> dict[str, str]:
    """Probe configured Brave / Serper / Tavily / Jina keys before a run.

    Providers are probed only when their API key is non-empty.

    A single unhealthy provider (e.g. Tavily returning ``402 Payment
    Required``) is logged as a warning but does NOT abort the run, as long as
    at least one *search* provider (brave / serper / tavily) is healthy — the
    web_search fallback chain handles the degraded one, and a down fetch
    provider (jina) degrades gracefully. The run is aborted
    (:class:`SearchApiUnhealthyError`, when *raise_on_unhealthy* is True) only
    when every probed search provider is unhealthy, i.e. there is no usable
    web search at all.

    Returns ``{provider: reason}`` for unhealthy providers (whether or not the
    run is aborted), so callers can inspect/report degraded providers.
    """
    probes = (
        ("brave",  (getattr(config, "brave_api_key", "") or "").strip(),  _probe_brave),
        ("serper", (getattr(config, "serper_api_key", "") or "").strip(), _probe_serper),
        ("tavily", (getattr(config, "tavily_api_key", "") or "").strip(), _probe_tavily),
        ("jina",   (getattr(config, "jina_api_key", "") or "").strip(),   _probe_jina),
    )

    failures: dict[str, str] = {}
    probed_search: list[str] = []
    healthy_search: list[str] = []
    probed_any = False
    for name, key, probe in probes:
        if not key:
            continue
        probed_any = True
        if name in _SEARCH_PROVIDERS:
            probed_search.append(name)
        try:
            reason = probe(key)
        except Exception as exc:  # belt-and-braces: never let the probe itself crash the run
            reason = f"probe raised: {exc}"
        if reason:
            failures[name] = reason
        else:
            logger.info("[api-health] %s API ok", name)
            if name in _SEARCH_PROVIDERS:
                healthy_search.append(name)

    if not probed_any:
        logger.debug("[api-health] no Brave/Serper/Tavily/Jina keys configured — skipping probe")
        return failures

    # Fatal only when we probed search providers and NONE are healthy.
    no_search = bool(probed_search) and not healthy_search
    for name, reason in failures.items():
        if no_search:
            logger.error("[api-health] %s API is UNHEALTHY: %s", name, reason)
        else:
            logger.warning(
                "[api-health] %s API unhealthy — continuing (working provider available): %s",
                name, reason,
            )

    if no_search and raise_on_unhealthy:
        details = "; ".join(f"{n} ({r})" for n, r in failures.items())
        raise SearchApiUnhealthyError(
            f"No usable web-search provider — every probed search API is unhealthy: {details}. "
            "Refusing to start the run — fix the credentials/billing or "
            "configure a working search provider before retrying."
        )
    return failures


# ---------------------------------------------------------------------------
# Brave search failure logging
# ---------------------------------------------------------------------------


def enrich_failures_with_next_action(
    failures: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
) -> None:
    """Enrich fallback entries in-place with the agent's next action from the trajectory.

    Walks the serialized trajectory messages, finds ``web_search`` tool_use blocks
    whose query matches a failure entry, then looks ahead for the next tool_use
    block to determine what the agent did after the fallback.
    """
    if not failures or not trajectory:
        return

    # Build a set of failure queries for fast lookup
    failure_queries = {f["query"] for f in failures}
    query_to_failures: dict[str, list[dict[str, Any]]] = {}
    for f in failures:
        query_to_failures.setdefault(f["query"], []).append(f)

    # Get messages from trajectory (handles both single-agent and swarm formats)
    messages: list[dict[str, Any]] = []
    if trajectory and isinstance(trajectory[0], dict):
        traj_entry = trajectory[0]
        if "messages" in traj_entry:
            messages = traj_entry["messages"]
        elif "subagents" in traj_entry:
            # Swarm: combine all subagent messages
            for sa in traj_entry.get("subagents", []):
                messages.extend(sa.get("messages", []))

    # Collect all tool_use blocks in order with their position
    tool_uses: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_uses.append(block)

    # For each web_search tool_use that matches a failure query,
    # find the next tool_use after it
    for i, tu in enumerate(tool_uses):
        if tu.get("name") != "web_search":
            continue
        query = (tu.get("input") or {}).get("query", "")
        if query not in failure_queries:
            continue

        pending = query_to_failures.get(query, [])
        if not pending:
            continue

        # Find the next tool_use (could be in same or later assistant message)
        next_action = None
        if i + 1 < len(tool_uses):
            next_tu = tool_uses[i + 1]
            next_input = next_tu.get("input", {})
            # Summarize input to keep logs compact
            if next_tu.get("name") == "web_search":
                next_action = {"tool": "web_search", "input": {"query": next_input.get("query", "")}}
            elif next_tu.get("name") == "web_fetch":
                next_action = {"tool": "web_fetch", "input": {"url": next_input.get("url", "")}}
            else:
                # Generic: include tool name and truncated input
                summary = {k: str(v)[:100] for k, v in list(next_input.items())[:3]}
                next_action = {"tool": next_tu.get("name", ""), "input": summary}

        # Attach to the first unmatched failure entry for this query
        entry = pending.pop(0)
        if next_action:
            entry["next_agent_action"] = next_action
        else:
            entry["next_agent_action"] = None  # last tool call in conversation


def save_web_search_failures(
    output_dir: Path,
    conv_id: str,
    failures: list[dict[str, Any]],
    trajectory: list[dict[str, Any]] | None = None,
) -> None:
    """Write Brave search failure log for a single eval case.

    Creates ``web_search_failures/{conv_id}.json`` alongside the
    ``trajectories/`` directory.  Only writes if there are failures.
    When there are NO failures, deletes any stale failure file from a
    previous run so that ``--rerun-errors`` does not re-trigger.
    """
    fail_dir = output_dir / "web_search_failures"
    safe_name = conv_id.replace("/", "_").replace("\\", "_")[:200]
    fail_path = fail_dir / f"{safe_name}.json"

    if not failures:
        # Clean up stale failure file from a prior run
        if fail_path.exists():
            fail_path.unlink()
        return

    if trajectory:
        enrich_failures_with_next_action(failures, trajectory)
    fail_dir.mkdir(parents=True, exist_ok=True)
    fail_path.write_text(json.dumps(failures, indent=2))


# ---------------------------------------------------------------------------
# Search score summarization
# ---------------------------------------------------------------------------


def summarize_search_scores(scores: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Condense per-result source scores into a compact summary for the search log.

    Returns a dict with per-result scores and averages, or ``None`` when
    *scores* is empty/falsy.
    """
    if not scores:
        return None
    dims = ("relevance", "answerability", "authority", "data_density")
    per_result = []
    for s in scores:
        entry: dict[str, Any] = {"index": s.get("index", 0)}
        for d in dims:
            entry[d] = s.get(d, 0.0)
        entry["composite"] = s.get("composite", 0.0)
        per_result.append(entry)

    n = len(per_result)
    avg: dict[str, float] = {}
    for d in dims:
        avg[d] = round(sum(e[d] for e in per_result) / n, 2)
    avg["composite"] = round(sum(e["composite"] for e in per_result) / n, 1)

    return {"per_result": per_result, "avg": avg, "n": n}


def save_web_search_log(
    output_dir: Path,
    conv_id: str,
    search_log: list[dict[str, Any]],
) -> None:
    """Write full per-query search log for a single eval case.

    Creates ``web_search_log/{conv_id}.json``.  Records every search call
    with the provider that served the result and query features.
    """
    if not search_log:
        return
    log_dir = output_dir / "web_search_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = conv_id.replace("/", "_").replace("\\", "_")[:200]
    log_path = log_dir / f"{safe_name}.json"
    log_path.write_text(json.dumps(search_log, indent=2))


def save_web_fetch_log(
    output_dir: Path,
    conv_id: str,
    fetch_log: list[dict[str, Any]],
) -> None:
    """Write per-fetch log for a single eval case.

    Creates ``web_fetch_log/{conv_id}.json``.  Records every web_fetch call
    with the tier that served the result, latency, and content size.
    """
    if not fetch_log:
        return
    log_dir = output_dir / "web_fetch_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = conv_id.replace("/", "_").replace("\\", "_")[:200]
    log_path = log_dir / f"{safe_name}.json"
    log_path.write_text(json.dumps(fetch_log, indent=2))


def save_content_filter_log(
    output_dir: Path,
    conv_id: str,
    content_filter_log: list[dict[str, Any]],
) -> None:
    """Write content filter rejection log for a single eval case.

    Creates ``content_filter_log/{conv_id}.json``.  Records each source
    scorer invocation that was blocked by Azure content filters, including
    the query, truncated source content, and the full error message.
    """
    if not content_filter_log:
        return
    log_dir = output_dir / "content_filter_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = conv_id.replace("/", "_").replace("\\", "_")[:200]
    log_path = log_dir / f"{safe_name}.json"
    log_path.write_text(json.dumps(content_filter_log, indent=2))


def save_compactor_log(
    output_dir: Path,
    conv_id: str,
    compactor_log: list[dict[str, Any]],
) -> None:
    """Write per-call ContentCompactor log for a single eval case.

    Creates ``compactor_log/{conv_id}.json``.  Records every ``compact()``
    invocation with source label, input chars, chunk counts, selected
    indices, scores, fallback reason, latency, and the owning subagent.
    Drained from each subagent's compactor by the orchestrator and merged
    in the swarm path; for non-swarm runs it's drained directly from the
    agent's compactor.
    """
    if not compactor_log:
        return
    log_dir = output_dir / "compactor_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = conv_id.replace("/", "_").replace("\\", "_")[:200]
    log_path = log_dir / f"{safe_name}.json"
    log_path.write_text(json.dumps(compactor_log, indent=2, default=str))


def compute_grounding_fetch_stats(
    all_fetch_logs: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate grounding fetch tier stats across all per-case fetch logs.

    Each fetch log entry has shape::

        {"url": str, "tier": str, "success": bool, "error": str|None,
         "latency_ms": float, "content_chars": int}

    A single URL fetch may produce multiple entries: one ``grounding``
    attempt (possibly failed) followed by a fallback tier entry.  We group
    consecutive entries by URL to reconstruct the per-request outcome.

    Returns a dict with::

        grounding_attempts      -- total requests where grounding was tried
        grounding_success       -- grounding succeeded (no fallback needed)
        grounding_fallback      -- grounding tried but failed, fell back
        grounding_fail_rate_pct -- grounding_fallback / grounding_attempts * 100
        fallback_to_jina        -- grounding-failed requests served by Jina
        fallback_to_serper      -- grounding-failed requests served by Serper
        fallback_to_requests    -- grounding-failed requests served by requests
        fallback_to_none        -- grounding-failed AND no fallback succeeded
        total_fetches           -- total fetch entries across all cases
    """
    grounding_attempts = 0
    grounding_success = 0
    fallback_to_jina = 0
    fallback_to_serper = 0
    fallback_to_requests = 0
    fallback_to_none = 0
    total_fetches = 0

    for fetch_log in all_fetch_logs:
        if not fetch_log:
            continue

        # Group consecutive entries by URL to reconstruct per-request outcomes.
        # Each group = one logical web_fetch call (tier 0 attempt + optional fallback).
        i = 0
        while i < len(fetch_log):
            entry = fetch_log[i]
            total_fetches += 1
            tier = entry.get("tier", "")

            # The dedicated grounding fetch tier has been removed; this branch
            # is retained only to keep older fetch logs (which may still carry a
            # ``grounding`` tier label) aggregatable.
            if tier == "grounding":
                grounding_attempts += 1
                if entry.get("success"):
                    grounding_success += 1
                    i += 1
                else:
                    # Grounding failed — look at the next entry for the fallback tier
                    fallback_tier = ""
                    fallback_success = False
                    if i + 1 < len(fetch_log):
                        next_entry = fetch_log[i + 1]
                        fallback_tier = next_entry.get("tier", "")
                        fallback_success = bool(next_entry.get("success"))
                        i += 2
                    else:
                        i += 1

                    if fallback_success:
                        if fallback_tier == "jina":
                            fallback_to_jina += 1
                        elif fallback_tier == "serper":
                            fallback_to_serper += 1
                        elif fallback_tier == "requests":
                            fallback_to_requests += 1
                    else:
                        fallback_to_none += 1
            else:
                i += 1

    grounding_fallback = grounding_attempts - grounding_success
    fail_rate = (100.0 * grounding_fallback / grounding_attempts) if grounding_attempts > 0 else 0.0

    return {
        "grounding_attempts": grounding_attempts,
        "grounding_success": grounding_success,
        "grounding_fallback": grounding_fallback,
        "grounding_fail_rate_pct": round(fail_rate, 1),
        "fallback_to_jina": fallback_to_jina,
        "fallback_to_serper": fallback_to_serper,
        "fallback_to_requests": fallback_to_requests,
        "fallback_to_none": fallback_to_none,
        "total_fetches": total_fetches,
    }


def print_judge_fallback_stats(console: "Any", judge: "Any") -> None:
    """Print content filter fallback and parse failure counts if any."""
    fallbacks = getattr(judge, "content_filter_fallback_count", 0)
    parse_fails = getattr(judge, "_judge_parse_failure_count", 0)
    if fallbacks > 0:
        console.print(
            f"  [yellow]Azure content filter fallbacks: {fallbacks} case(s) judged by claude-4-sonnet instead[/yellow]"
        )
    if parse_fails > 0:
        console.print(
            f"  [yellow]Judge parse failures: {parse_fails} case(s) could not be parsed (counted as incorrect)[/yellow]"
        )


# ---------------------------------------------------------------------------
# Agent-bound convenience wrappers
# ---------------------------------------------------------------------------


def log_compaction_stats_for_agent(agent: "Agent") -> None:
    """End-of-turn compaction stats for an agent."""
    log_compaction_stats(
        agent.compaction_count,
        agent.total_llm_calls,
        agent._context_budget,
    )


def log_web_search_stats_for_agent(agent: "Agent") -> None:
    """Aggregated web search stats for an agent."""
    log_web_search_stats(agent._tools)


def log_web_fetch_stats_for_agent(agent: "Agent") -> None:
    """Aggregated web fetch stats for an agent."""
    log_web_fetch_stats(agent._tools)


def log_empty_fallback_for_agent(
    agent: "Agent",
    *,
    last_user_text: str,
    num_messages: int,
    fallback_attempts: list[dict],
    streaming: bool,
    primary_raw_event_log: list[dict[str, Any]] | None = None,
    primary_stop_reason: str = "",
    primary_usage: dict[str, Any] | None = None,
) -> None:
    """Write a per-case log file for empty-response fallback events on an agent."""
    log_empty_fallback(
        model=agent.config.model,
        max_tokens=agent.config.max_tokens,
        total_llm_calls=agent.total_llm_calls,
        system_prompt=agent.system_prompt,
        input_messages=agent._msgs_for_llm(),
        tool_definitions=agent._get_tool_definitions(),
        last_user_text=last_user_text,
        num_messages=num_messages,
        fallback_attempts=fallback_attempts,
        streaming=streaming,
        output_dir=agent.config.output_dir,
        primary_raw_event_log=primary_raw_event_log,
        primary_stop_reason=primary_stop_reason,
        primary_usage=primary_usage,
        reasoning_effort=agent.config.reasoning_effort,
        context_utilization=agent._context_budget.utilization(),
    )


# ---------------------------------------------------------------------------
# Cross-agent tool-role token aggregation
# ---------------------------------------------------------------------------


def aggregate_tool_role_usage(
    orchestrator_agent: "Agent",
    subagents: Iterable,
) -> tuple[dict[str, "TokenUsage"], dict[str, int]]:
    """Drain compactor / source_scorer token ledgers from every agent.

    Each agent (orchestrator + each subagent) owns its own SourceScorer
    and ContentCompactor instance via its tool factory.  Each ledger is
    keyed by role label ("source_scorer", "compactor", ...).  This
    helper merges them across the swarm into ``{role: TokenUsage}`` and
    also returns ``{role: total_call_count}`` so callers can surface
    roles that fired but failed to record token counts (e.g., when an
    upstream proxy strips the ``usage`` field).
    """
    # Local import to avoid the import cycle: agent -> logging_utils -> agent.
    from arcticswarm.agent import TokenUsage

    totals: dict[str, TokenUsage] = {}
    calls: dict[str, int] = {}

    def _drain(obj: Any) -> None:
        if obj is None or not hasattr(obj, "drain_token_ledger"):
            return
        try:
            ledger = obj.drain_token_ledger()
        except Exception:
            return
        for role, counts in (ledger or {}).items():
            # Per-role guard: a single malformed entry shouldn't lose
            # aggregation for the rest of the roles.
            try:
                bucket = totals.setdefault(role, TokenUsage())
                bucket.input_tokens += int(counts.get("input_tokens", 0) or 0)
                bucket.output_tokens += int(counts.get("output_tokens", 0) or 0)
                bucket.cache_creation_input_tokens += int(
                    counts.get("cache_creation_input_tokens", 0) or 0,
                )
                bucket.cache_read_input_tokens += int(
                    counts.get("cache_read_input_tokens", 0) or 0,
                )
                calls[role] = calls.get(role, 0) + int(counts.get("calls", 0) or 0)
            except Exception:
                continue

    agents = [orchestrator_agent]
    for sa in subagents:
        agent = getattr(sa, "agent", None)
        if agent is not None:
            agents.append(agent)
    # Track which ledger objects we've already drained so we don't
    # double-count when a single SourceScorer / ContentCompactor
    # singleton is referenced from multiple agents (e.g. via
    # ``_ensure_source_scorer`` factory caching).
    drained_ids: set[int] = set()
    for ag in agents:
        # Drain the agent itself (history compactor and any other
        # in-class direct-LLM callers register here).
        if id(ag) not in drained_ids and hasattr(ag, "drain_token_ledger"):
            drained_ids.add(id(ag))
            _drain(ag)
        # Drain every tool that exposes a ledger.  Currently:
        # ReasoningTool ("reasoning_tool" role), and any future
        # tool that adopts the same pattern.  Tools with no
        # ``drain_token_ledger`` method are skipped by ``_drain``.
        for tool in (getattr(ag, "_tools", None) or {}).values():
            if tool is None or id(tool) in drained_ids:
                continue
            drained_ids.add(id(tool))
            _drain(tool)
        # Legacy/explicit drains for helpers that aren't in
        # ``_tools`` (kept as a belt-and-braces — most agents have
        # these as separate attributes too).
        for attr in ("_source_scorer", "_fetch_compactor", "_pdf_compactor"):
            obj = getattr(ag, attr, None)
            if obj is None or id(obj) in drained_ids:
                continue
            drained_ids.add(id(obj))
            _drain(obj)
    return totals, calls
