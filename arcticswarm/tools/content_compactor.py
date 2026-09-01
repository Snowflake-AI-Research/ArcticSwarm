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

"""Content compactor — chunks a fetched document, asks an LLM to pick the
chunks relevant to the original task, and returns only those.

Used by ``web_fetch`` (when ``use_fetch_compactor`` is set) and by
``pdf_read`` (when ``use_pdf_compactor`` is set).  Both tools share a
single compactor instance — the pipeline is identical; only the input
source differs.

Replaces ``SourceScorer`` for whichever tool the flag enables.  Search-
result gating (``judge_search_results``) remains on the source scorer.

Pipeline (one call per fetched document, full content — no 2K truncation):

  1. ``_chunk_text(content)`` — sentence-aware split into ~1000-char
     chunks.  If a single sentence exceeds the target, it is hard-split
     at the char boundary so chunk size stays bounded.
  2. ``compact(query, source, content)`` — sends [Chunk i] blocks + the
     research query to the compactor LLM.  Expects a JSON object:

         {"scores": {"relevance": .., "answerability": .., "authority": ..,
                      "data_density": ..},
          "selected_indices": [1, 4, 5, 7]}

  3. Indices are validated, deduped, and sorted ascending; only those
     chunks (separated by "[... omitted ...]" between non-adjacent
     indices) are returned to the agent, with a ``[Source Quality: ...]``
     annotation appended for parity with ``SourceScorer``.
  4. On any failure (parse error, empty selection, transport error), the
     fallback is the same as ``SourceScorer``'s implicit truncation:
     return the first 2000 chars of the original document so the agent
     has something to work with.

The class subclasses :class:`SourceScorer` to inherit the Azure / cortex-
proxy / agent-client routing, ``_call_llm`` / ``_call_agent`` /
``_call_openai`` / ``_strip_markdown_fences`` / ``_scrub_bot_boilerplate``
helpers, and ``format_annotation``.  Only the prompts and the public
``compact()`` method are new.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from arcticswarm.tools.source_scorer import (
    SourceScorer,
    _scrub_bot_boilerplate,
    _strip_markdown_fences,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_CHARS = 2000
_FALLBACK_TRUNCATE_CHARS = 2000  # matches source_scorer's implicit cap
_MAX_PROMPT_CHARS = 600_000  # safety cap on prompt size (chunk text only)
# Fallback window (chars budget is derived from the served model's
# max_model_len at call time; this is only used if the client doesn't expose
# one). Matches the qwen vLLM default so behaviour is unchanged off Tongyi.
_DEFAULT_COMPACTOR_WINDOW = 262_144


# ---------------------------------------------------------------------------
# Sentence-aware chunker
# ---------------------------------------------------------------------------


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _hard_split(s: str, target_chars: int) -> list[str]:
    """Char-boundary split for an oversize sentence."""
    return [s[i : i + target_chars] for i in range(0, len(s), target_chars)]


def _chunk_text(content: str, target_chars: int = _DEFAULT_CHUNK_CHARS) -> list[str]:
    """Split content into ~target_chars chunks on sentence boundaries.

    A single sentence longer than ``target_chars`` is hard-split at the
    char boundary to keep chunks bounded.  Empty pieces are dropped.
    """
    if not content:
        return []
    target_chars = max(target_chars, 1)

    sentences: list[str] = []
    for piece in _SENTENCE_BOUNDARY_RE.split(content):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > target_chars:
            sentences.extend(p for p in _hard_split(piece, target_chars) if p.strip())
        else:
            sentences.append(piece)

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for sent in sentences:
        # +1 for the joining space
        add_len = len(sent) + (1 if buf else 0)
        if buf and buf_len + add_len > target_chars:
            chunks.append(" ".join(buf))
            buf = [sent]
            buf_len = len(sent)
        else:
            buf.append(sent)
            buf_len += add_len
    if buf:
        chunks.append(" ".join(buf))
    return chunks


# ---------------------------------------------------------------------------
# Compactor prompt
# ---------------------------------------------------------------------------

_COMPACTOR_SYSTEM_PROMPT = """\
You are a research-context compactor.  Given a user query and a webpage \
that has been split into numbered chunks, do TWO things:

1. Score the **source as a whole** on four 0-10 dimensions, just like a \
quality reviewer would:
   - relevance: how directly the source addresses the query.
   - answerability: how concretely the source could help answer the query.
   - authority: credibility / trustworthiness of the source.
   - data_density: concentration of facts, dates, numbers, names.

2. Choose the **chunk indices** whose text the downstream agent should \
actually read.  Pick every chunk that contains task-relevant prose, \
data, dates, names, or evidence.  Drop chunks that are pure navigation, \
footer, link soup, repeated boilerplate, or unrelated topics.

Rules:
- Return indices in ASCENDING ORDER.
- Return AT LEAST ONE chunk index — never an empty list.  If nothing is \
clearly relevant, pick the single chunk that comes closest.
- The downstream context budget is LIMITED.  Select only the densest, most \
task-relevant chunks — aim for a COMPACT selection (roughly 5,000 tokens / \
20,000 characters total or less) and drop marginal, redundant, navigational, \
or weakly-related chunks.  Precision of selection matters more than coverage; \
a tight set of high-value chunks is better than a long list.
- Use the integer index shown next to each "[Chunk N]" header.

### Output Format (pure JSON, no markdown fences)

{
  "scores": {"relevance": 9.0, "answerability": 8.5, "authority": 9.0, "data_density": 8.0},
  "selected_indices": [1, 4, 5, 7]
}

IMPORTANT: Return ONLY the JSON object.  No explanations, no markdown."""


# ---------------------------------------------------------------------------
# ContentCompactor
# ---------------------------------------------------------------------------


class ContentCompactor(SourceScorer):
    """Compact a single fetched document (web page or PDF) via an LLM.

    Inherits all routing/parsing helpers from :class:`SourceScorer`; only
    the prompt and :meth:`compact` are new.  Never calls ``evaluate()``
    or ``judge_search_results()`` — those remain on the source scorer
    instance used for the search-result gate.

    The same instance can be shared between ``web_fetch`` and
    ``pdf_read`` consumers; the ``source`` argument to :meth:`compact`
    is just a label included in the prompt for the LLM's situational
    awareness.
    """

    def __init__(self, *args: Any, max_output_chars: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Hard cap (in chars) on the re-assembled selected output. 0 = no cap.
        # Mirrors config.max_tool_output_tokens * 4 (chars/token estimate); see
        # Agent._cap_tool_output for the agent-side backstop that enforces the
        # same budget on every path (raw / fallback / cache). Without this, a
        # dense page where the LLM selects most chunks returns ~the whole page
        # (up to the ~150K-token input cap) into the agent context.
        self._max_output_chars = max_output_chars if max_output_chars > 0 else 0
        # Per-call structured log; drained by the eval runner.
        self._compactor_log: list[dict[str, Any]] = []

    def drain_compactor_log(self) -> list[dict[str, Any]]:
        """Return and clear the per-call compactor log."""
        out = self._compactor_log
        self._compactor_log = []
        return out

    def compact(
        self,
        query: str,
        source: str,
        content: str,
        *,
        chunk_chars: int = _DEFAULT_CHUNK_CHARS,
    ) -> tuple[str, dict[str, Any]]:
        """Score the document and select task-relevant chunks.

        Returns ``(agent_visible_content, score_dict)``.

        - ``agent_visible_content`` — selected chunks joined with
          ``[... omitted ...]`` separators between non-adjacent indices.
          On any failure, the first ``_FALLBACK_TRUNCATE_CHARS`` chars of
          the original document (matching the source scorer's behavior)
          plus a one-line trailer.
        - ``score_dict`` — ``{"relevance":..., "answerability":...,
          "authority":..., "data_density":...}``.  Empty dict on failure.
        """
        t0 = time.monotonic()
        record: dict[str, Any] = {
            "source": source,
            "input_chars": len(content) if content else 0,
            "num_chunks": 0,
            "selected_indices": [],
            "num_selected": 0,
            "scores": {},
            "fallback_reason": "",
            "output_chars": 0,
            "latency_ms": 0.0,
        }

        def _emit(out_text: str) -> None:
            record["output_chars"] = len(out_text) if out_text else 0
            record["latency_ms"] = round((time.monotonic() - t0) * 1000.0, 2)
            self._compactor_log.append(record)

        if not content:
            record["fallback_reason"] = "empty_content"
            _emit("")
            return "", {}
        if self._disabled or not query:
            record["fallback_reason"] = "disabled" if self._disabled else "no_query"
            out = self._fallback_content(content)
            _emit(out)
            return out, {}

        chunks = _chunk_text(content, target_chars=chunk_chars)
        record["num_chunks"] = len(chunks)
        if not chunks:
            record["fallback_reason"] = "empty_chunks"
            out = self._fallback_content(content)
            _emit(out)
            return out, {}

        # Cap the compactor prompt to the SERVED model's window. The static
        # _MAX_PROMPT_CHARS (600K chars ~= 150K tokens) was sized for the Qwen
        # 262K window; on a 131K-window model (Tongyi-DeepResearch) it overflows
        # so the compactor LLM call errors and we fall back to crude truncation.
        # Derive the char budget from the agent client's actual max_model_len,
        # leaving room for the system prompt + the 2K output. (Window rises ->
        # cap rises, so Qwen at 262K is capped at the same 600K as before.)
        _mml = getattr(self._agent_client, "_max_model_len", 0) or _DEFAULT_COMPACTOR_WINDOW
        _cap = max(40_000, min(_MAX_PROMPT_CHARS, int((_mml - 8192 - 2000) * 3.5)))
        prompt = self._build_prompt(query, source, chunks, max_prompt_chars=_cap)
        try:
            raw = self._call_llm(
                _COMPACTOR_SYSTEM_PROMPT, prompt, max_tokens=2000,
                role="compactor",
            )
        except Exception as e:
            error_str = str(e)
            if self._is_permanent_error(e):
                logger.warning("Content compactor disabled — endpoint error: %s", e)
                self._disabled = True
                record["fallback_reason"] = "permanent_error"
            elif "content_filter" in error_str or "content management policy" in error_str:
                self.content_filter_count += 1
                self._content_filter_log.append({
                    "system_prompt": _COMPACTOR_SYSTEM_PROMPT,
                    "user_msg": prompt[:4000],
                    "error": error_str,
                })
                logger.warning(
                    "Content compactor blocked by content filter (count=%d): %s",
                    self.content_filter_count, e,
                )
                record["fallback_reason"] = "content_filter"
            else:
                logger.warning("Content compactor LLM call failed: %s", e)
                record["fallback_reason"] = "llm_error"
            out = self._fallback_content(content)
            _emit(out)
            return out, {}

        scores, indices = self._parse_response(raw, num_chunks=len(chunks))
        record["scores"] = scores
        record["selected_indices"] = indices
        record["num_selected"] = len(indices)
        if not indices:
            logger.warning(
                "Content compactor returned no selectable indices; falling back to truncation. raw=%s",
                raw[:300],
            )
            record["fallback_reason"] = "no_indices"
            out = self._fallback_content(content)
            _emit(out)
            return out, {}

        out = self._assemble_selected(chunks, indices, max_chars=self._max_output_chars)
        _emit(out)
        return out, scores

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(query: str, source: str, chunks: list[str],
                      max_prompt_chars: int = _MAX_PROMPT_CHARS) -> str:
        """Format the user message with [Chunk i] blocks."""
        body_parts: list[str] = []
        running = 0
        for i, chunk in enumerate(chunks):
            scrubbed = _scrub_bot_boilerplate(chunk)
            block = f"[Chunk {i}]\n{scrubbed}"
            if running + len(block) > max_prompt_chars:
                body_parts.append(
                    f"[... {len(chunks) - i} additional chunks omitted from compactor "
                    f"input due to size cap]"
                )
                break
            body_parts.append(block)
            running += len(block) + 2
        joined = "\n\n".join(body_parts)
        return (
            f"Query: {query}\n\n"
            f"Source: {source or '(no source)'}\n\n"
            f"Chunks (total {len(chunks)}):\n\n{joined}"
        )

    @staticmethod
    def _parse_response(
        raw: str, *, num_chunks: int,
    ) -> tuple[dict[str, float], list[int]]:
        """Parse the compactor JSON.  Returns (scores, sorted_indices)."""
        cleaned = _strip_markdown_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            try:
                # Best-effort repair similar to source_scorer's fixups
                repaired = re.sub(r'"(\w+)(?::)', r'"\1":', cleaned)
                repaired = repaired.replace("}}}", "}}")
                parsed = json.loads(repaired)
            except (json.JSONDecodeError, ValueError, Exception):
                logger.warning("Fetch compactor returned non-JSON: %s", raw[:300])
                return {}, []

        if not isinstance(parsed, dict):
            logger.warning("Fetch compactor returned unexpected type: %s", type(parsed))
            return {}, []

        s = parsed.get("scores") or {}
        scores: dict[str, float] = {
            "relevance": float(s.get("relevance", 0)),
            "answerability": float(s.get("answerability", 0)),
            "authority": float(s.get("authority", 0)),
            "data_density": float(s.get("data_density", 0)),
        }

        raw_indices = parsed.get("selected_indices") or []
        if not isinstance(raw_indices, list):
            return scores, []

        seen: set[int] = set()
        valid: list[int] = []
        for v in raw_indices:
            try:
                idx = int(v)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < num_chunks and idx not in seen:
                seen.add(idx)
                valid.append(idx)
        valid.sort()
        return scores, valid

    @staticmethod
    def _assemble_selected(
        chunks: list[str], indices: list[int], *, max_chars: int = 0,
    ) -> str:
        """Join selected chunks; insert "[... omitted ...]" between gaps.

        When ``max_chars`` > 0, stop accumulating once the running length would
        exceed the budget and append a "[... selection truncated to output cap]"
        marker. Whole chunks are kept (the last chunk is never split), so the
        agent-visible output stays bounded without breaking mid-sentence. This
        is the per-tool half of the output cap; ``Agent._cap_tool_output`` is
        the final backstop on the same budget.
        """
        if not indices:
            return ""
        parts: list[str] = []
        running = 0
        prev: int | None = None
        for idx in indices:
            pieces: list[str] = []
            if prev is not None and idx > prev + 1:
                pieces.append("[... omitted ...]")
            pieces.append(chunks[idx])
            # Chars this iteration adds, including the "\n\n" (2-char) joiners.
            joiners = len(pieces) if parts else len(pieces) - 1
            add = sum(len(p) for p in pieces) + 2 * joiners
            if max_chars and parts and running + add > max_chars:
                parts.append("[... selection truncated to output cap]")
                break
            parts.extend(pieces)
            running += add
            prev = idx
        return "\n\n".join(parts)

    @staticmethod
    def _fallback_content(content: str) -> str:
        """First 2000 chars + trailer, matching source scorer's truncation."""
        head = content[:_FALLBACK_TRUNCATE_CHARS]
        if len(content) > _FALLBACK_TRUNCATE_CHARS:
            return (
                head
                + "\n\n[... content truncated — fetch compactor returned no selection]"
            )
        return head
