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

"""Tool execution for the agent loop.

The machinery that runs tool calls and post-processes their results, mixed into
:class:`arcticswarm.agent.Agent` as :class:`ToolExecutionMixin` (behavior-
identical; methods resolve via the MRO). Covers sequential / parallel dispatch,
the eval-awareness contamination filter, source-quality scoring, the per-tool
ContentCompactor handoff, and the per-result output cap.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TYPE_CHECKING

from arcticswarm.tools.base import ToolResult

if TYPE_CHECKING:
    from arcticswarm.agent import ToolCallEnd, ToolCallStart

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval-awareness contamination filter + parallel-safe tool set
# ---------------------------------------------------------------------------

_CONTAMINATION_KEYWORDS = (
    "browsecomp",
)
_CONTAMINATED_TOOLS = frozenset({"web_search", "web_fetch"})
_CONTAMINATION_PLACEHOLDER = "[Result excluded: potential eval contamination]"
_PARALLEL_SAFE_TOOLS = frozenset({
    "web_search", "web_fetch", "pdf_read", "read_file",
    "calculator", "reasoning",
})


# ---------------------------------------------------------------------------
# Tool-execution mixin
# ---------------------------------------------------------------------------


class ToolExecutionMixin:
    """Tool dispatch + result post-processing for :class:`~arcticswarm.agent.Agent`.

    Methods operate on ``Agent`` instance state via ``self`` and call sibling
    mixins (content compactor, source scorer, output cap) through ``self`` as
    well; ``ToolCallStart`` / ``ToolCallEnd`` are imported lazily inside the
    emitting methods to avoid an import cycle with ``arcticswarm.agent``.
    """

    def _execute_tool(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call by name using this agent's tool registry."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys()))
            return ToolResult(
                error=(
                    f"Unknown tool: '{name}'. "
                    f"Available tools: {available}. "
                    f"Use only the tools listed above."
                ),
                is_error=True,
            )
        try:
            result = tool.execute(**input_data)

            # Track tool call in persistent counter (survives clear_history)
            self.tool_calls_by_name[name] = self.tool_calls_by_name.get(name, 0) + 1

            # Eval-awareness: filter browsing results that contain benchmark keywords
            if self.config.web_search_enabled and name in _CONTAMINATED_TOOLS and not result.is_error:
                output_lower = (result.output or "").lower()
                is_contaminated = any(kw in output_lower for kw in _CONTAMINATION_KEYWORDS)
                self.contamination_stats.record(excluded=is_contaminated)
                if is_contaminated:
                    logger.info(
                        "Eval-awareness: excluded %s result (contamination keyword found)",
                        name,
                    )
                    result = ToolResult(
                        output=_CONTAMINATION_PLACEHOLDER,
                        metadata={**result.metadata, "contamination_excluded": True},
                    )

            # Track web_search results for reference extraction
            if name == "web_search" and self.web_source_tracker is not None:
                try:
                    self.web_source_tracker.add_from_tool_result(result)
                except Exception:
                    pass  # Don't fail the tool call if tracking fails

            # Accumulate web_fetch results for batch source scoring
            if name == "web_fetch" and not result.is_error:
                if self._fetch_compactor is not None:
                    # Compactor path: rewrite the result content in-place;
                    # source-scorer accumulation is skipped to avoid double work.
                    self._apply_content_compactor(
                        self._fetch_compactor,
                        input_data.get("url", ""),
                        result,
                    )
                elif self._source_scorer is not None:
                    try:
                        self._pending_sources.append({
                            "url": input_data.get("url", ""),
                            "tool_use_id": "",  # filled by caller after execute
                            "content": (result.output or "")[:2000],
                        })
                    except Exception:
                        pass

            # PDF compaction (no source-scorer fallback path — pdf_read is
            # not accumulated for batch scoring today).
            if (
                name == "pdf_read"
                and not result.is_error
                and self._pdf_compactor is not None
            ):
                self._apply_content_compactor(
                    self._pdf_compactor,
                    input_data.get("source", ""),
                    result,
                )

            # Last-word backstop: bound an oversized web_fetch / pdf_read result
            # (raw, compactor, fallback, or cache) so one turn can't overflow the
            # context window before reactive compaction can run.
            self._cap_tool_output(name, result)

            return result
        except Exception as exc:
            return ToolResult(error=f"Tool '{name}' failed: {exc}", is_error=True)

    def _execute_tools_batch(
        self,
        tool_calls: list[dict[str, Any]],
        on_event: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a batch of tool calls.

        Tool calls run sequentially via :meth:`_execute_tools_sequential`,
        which is the supported default. A concurrent implementation for
        I/O-bound, parallel-safe tools (``_PARALLEL_SAFE_TOOLS``) is kept in
        :meth:`_execute_tools_parallel` for reference but is intentionally not
        dispatched here: parallelizing tool calls within a single turn changed
        result ordering enough to affect answer quality, so sequential
        execution is preferred.
        """
        return self._execute_tools_sequential(tool_calls, on_event)

    def _execute_tools_sequential(
        self,
        tool_calls: list[dict[str, Any]],
        on_event: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """Execute tool calls one at a time (original behaviour)."""
        from arcticswarm.agent import ToolCallEnd, ToolCallStart
        tool_results: list[dict[str, Any]] = []
        for tc in tool_calls:
            if on_event:
                on_event(ToolCallStart(
                    tool_name=tc["name"],
                    tool_input=tc["input"],
                    tool_use_id=tc["id"],
                ))

            result = self._execute_tool(tc["name"], tc["input"])

            if on_event:
                on_event(ToolCallEnd(
                    tool_name=tc["name"],
                    tool_use_id=tc["id"],
                    result=result,
                ))

            entry: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": result.to_content(),
                "is_error": result.is_error,
            }
            if result.metadata:
                entry["metadata"] = result.metadata
            tool_results.append(entry)
        return tool_results

    def _execute_tools_parallel(
        self,
        tool_calls: list[dict[str, Any]],
        on_event: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """Execute I/O-bound tool calls concurrently via threads.

        Only the raw ``tool.execute()`` is parallelised; the lightweight
        post-processing in ``_execute_tool`` (counters, contamination,
        source tracking) runs sequentially afterwards.
        """
        from arcticswarm.agent import ToolCallEnd, ToolCallStart
        # Fire all ToolCallStart events up front.
        if on_event:
            for tc in tool_calls:
                on_event(ToolCallStart(
                    tool_name=tc["name"],
                    tool_input=tc["input"],
                    tool_use_id=tc["id"],
                ))

        # Run raw tool.execute() in parallel — this is the I/O-bound part.
        raw_results: list[ToolResult] = [ToolResult(error="not executed", is_error=True)] * len(tool_calls)

        def _run(idx: int, tc: dict[str, Any]) -> tuple[int, ToolResult]:
            tool = self._tools.get(tc["name"])
            if tool is None:
                available = ", ".join(sorted(self._tools.keys()))
                return idx, ToolResult(
                    error=f"Unknown tool: '{tc['name']}'. Available tools: {available}.",
                    is_error=True,
                )
            try:
                return idx, tool.execute(**tc["input"])
            except Exception as exc:
                return idx, ToolResult(error=f"Tool '{tc['name']}' failed: {exc}", is_error=True)

        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            futures = [pool.submit(_run, i, tc) for i, tc in enumerate(tool_calls)]
            for fut in futures:
                idx, result = fut.result()
                raw_results[idx] = result

        # Sequential post-processing: counters, contamination, source tracking.
        tool_results: list[dict[str, Any]] = []
        for tc, result in zip(tool_calls, raw_results):
            name = tc["name"]

            self.tool_calls_by_name[name] = self.tool_calls_by_name.get(name, 0) + 1

            if self.config.web_search_enabled and name in _CONTAMINATED_TOOLS and not result.is_error:
                output_lower = (result.output or "").lower()
                is_contaminated = any(kw in output_lower for kw in _CONTAMINATION_KEYWORDS)
                self.contamination_stats.record(excluded=is_contaminated)
                if is_contaminated:
                    logger.info("Eval-awareness: excluded %s result (contamination keyword found)", name)
                    result = ToolResult(
                        output=_CONTAMINATION_PLACEHOLDER,
                        metadata={**result.metadata, "contamination_excluded": True},
                    )

            if name == "web_search" and self.web_source_tracker is not None:
                try:
                    self.web_source_tracker.add_from_tool_result(result)
                except Exception:
                    pass

            if name == "web_fetch" and not result.is_error:
                if self._fetch_compactor is not None:
                    self._apply_content_compactor(
                        self._fetch_compactor,
                        tc["input"].get("url", ""),
                        result,
                    )
                elif self._source_scorer is not None:
                    try:
                        self._pending_sources.append({
                            "url": tc["input"].get("url", ""),
                            "tool_use_id": "",
                            "content": (result.output or "")[:2000],
                        })
                    except Exception:
                        pass

            if (
                name == "pdf_read"
                and not result.is_error
                and self._pdf_compactor is not None
            ):
                self._apply_content_compactor(
                    self._pdf_compactor,
                    tc["input"].get("source", ""),
                    result,
                )

            # Last-word backstop (see _execute_tool): bound oversized
            # web_fetch / pdf_read output before it enters context.
            self._cap_tool_output(name, result)

            if on_event:
                on_event(ToolCallEnd(
                    tool_name=name,
                    tool_use_id=tc["id"],
                    result=result,
                ))

            entry: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": result.to_content(),
                "is_error": result.is_error,
            }
            if result.metadata:
                entry["metadata"] = result.metadata
            tool_results.append(entry)

        return tool_results

    def _maybe_score_sources(
        self,
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> None:
        """Batch-score pending web_fetch sources and annotate tool results.

        Called after each tool-call batch in ``run_turn`` / ``run_turn_streaming``.
        Finds ``web_fetch`` entries in the batch, scores all pending sources in
        one LLM call, then appends score annotations directly to each
        ``web_fetch`` tool_result's content.  No documents are removed — scores
        are advisory text, matching DeepSearch's info_evaluator pattern.
        """
        if not self.config.web_search_enabled:
            return
        if not self._pending_sources or self._source_scorer is None:
            return

        sources = list(self._pending_sources)
        self._pending_sources.clear()

        # Determine the query for scoring context
        query = self.source_scoring_query
        if not query:
            query = self._get_user_query()
        if not query:
            return

        try:
            scored, _ = self._source_scorer.evaluate(query, sources)
        except Exception as e:
            logger.warning("Source scoring failed: %s", e)
            return

        if not scored:
            return

        # Build index -> annotation map from scored results
        annotations: dict[int, str] = {}
        for s in scored:
            annotations[s["index"]] = self._source_scorer.format_annotation(s)

        # Match pending sources to web_fetch tool_results by order.
        # Walk through tool_calls to find web_fetch entries and map them
        # to their corresponding tool_results by position.
        source_idx = 0
        for i, tc in enumerate(tool_calls):
            if tc["name"] != "web_fetch" or i >= len(tool_results):
                continue
            if tool_results[i].get("is_error"):
                continue
            if source_idx not in annotations:
                source_idx += 1
                continue

            annotation = annotations[source_idx]
            content = tool_results[i].get("content", "")

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["text"] = block["text"] + annotation
                        break
            elif isinstance(content, str):
                tool_results[i]["content"] = content + annotation

            source_idx += 1

    def _get_user_query(self) -> str:
        """Extract the user's query from conversation messages.

        Handles both plain-string user turns (historical default) and
        multimodal list content (e.g. image cases); for lists, the text
        is recovered by concatenating every
        ``{"type": "text", ...}`` block (image blocks are skipped).
        """
        for msg in self.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content[:500]
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            parts.append(text)
                if parts:
                    return "\n".join(parts)[:500]
        return ""

    def _apply_content_compactor(
        self, compactor: Any, source: str, result: ToolResult,
    ) -> None:
        """Compact a fetched-document tool result in-place via ContentCompactor.

        Replaces ``result.output`` with the LLM-selected chunks (joined by
        ``[... omitted ...]`` separators) and appends a ``[Source Quality:
        ...]`` annotation.  Falls back gracefully on any failure: the
        compactor itself returns the first 2000 chars of the original
        document, so the agent always has something to read.

        Used for both ``web_fetch`` (gated on ``self._fetch_compactor``)
        and ``pdf_read`` (gated on ``self._pdf_compactor``); the caller
        passes whichever ref is non-None.
        """
        if compactor is None:
            return
        query = self.source_scoring_query or self._get_user_query()
        if not query:
            return
        original = result.output or ""
        try:
            new_content, score = compactor.compact(query, source, original)
        except Exception as e:
            logger.warning("Content compactor failed: %s", e)
            return
        result.output = new_content
        if score:
            from arcticswarm.tools.source_scorer import SourceScorer
            composite = round(
                sum(score.get(k, 0) for k in
                    ("relevance", "answerability", "authority", "data_density")),
                1,
            )
            annotation = SourceScorer.format_annotation({**score, "composite": composite})
            result.output = (result.output or "") + annotation

    def _cap_tool_output(self, name: str, result: ToolResult) -> None:
        """Hard-cap an oversized web_fetch / pdf_read result before it enters context.

        A single huge page / docling-converted PDF / uncapped ContentCompactor
        selection can add tens of thousands of tokens in one turn — enough to
        blow past the model window before the (reactive, post-hoc) context
        budget can compact, which is the dominant ``prompt too long`` failure
        mode on BrowseComp. This is the last-word backstop, applied AFTER the
        optional compactor / source scorer, so it bounds *every* path: raw
        content, compactor output, the 2K fallback, and cross-agent cache hits.

        The cap is ``config.max_tool_output_tokens`` tokens, converted to chars
        via the codebase-wide ~4-chars/token estimate (matches
        ``llm_client._estimate_input_tokens`` / ``_estimate_msg_tokens``). 0
        disables. A trailing ``[Source Quality: ...]`` annotation (appended by
        the compactor / source scorer) is preserved across truncation.
        """
        if result.is_error or name not in ("web_fetch", "pdf_read"):
            return
        cap_tokens = getattr(self.config, "max_tool_output_tokens", 0)
        if cap_tokens <= 0:
            return
        cap_chars = cap_tokens * 4
        out = result.output or ""
        if len(out) <= cap_chars:
            return
        # Preserve a trailing one-line [Source Quality: ...] annotation, if any.
        annotation = ""
        m = re.search(r"\n\[Source Quality:[^\n\]]*\]\s*\Z", out)
        if m:
            annotation = out[m.start():]
            out = out[:m.start()]
        notice = (
            f"\n\n[... tool output truncated to ~{cap_tokens} tokens "
            f"to protect the context window]"
        )
        keep = max(0, cap_chars - len(annotation) - len(notice))
        result.output = out[:keep] + notice + annotation

    @staticmethod
    def _tool_batch_terminates_turn(
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> bool:
        """Return True when a tool result should end the current turn.

        Two cases:
        - ``send_user_markdown_report`` is the orchestrator's final delivery
          path. Once it succeeds, asking the LLM for another round in the same
          user turn only creates spurious empty responses and fallback calls.
        - A tool result carrying ``metadata.force_stop`` is a hard bail signal
          (emitted by the web_search repeat-guard when an agent is stuck looping
          the same query). Ending the turn lets a subagent finalize its task
          with the evidence it already has instead of spinning to the turn/time
          budget — a real stop that does not rely on the model complying with a
          nudge.
        """
        return any(
            (tc.get("name") == "send_user_markdown_report" and not tr.get("is_error", False))
            or (tr.get("metadata") or {}).get("force_stop") is True
            for tc, tr in zip(tool_calls, tool_results, strict=False)
        )
