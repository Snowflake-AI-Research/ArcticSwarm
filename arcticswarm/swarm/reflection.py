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

"""Structured reflection for browsing subagents.

Each subagent runs a Search → Reflect → Search → Summarize loop with hard
structural ceilings.  The reflection is a lightweight, separate LLM call
that evaluates whether gathered information is sufficient and generates
targeted follow-up queries for knowledge gaps.

The three hard ceilings (enforced in Python, not prompts):
  - MAX_SEARCH_PLANS:      outer loop iterations (≈ max_plan_executed_num)
  - MAX_REFLECTION_LOOPS:  inner loop per plan   (≈ max_research_loops)
  - step_budget:           max_turns per search phase (dynamic formula)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from arcticswarm.agent import TokenUsage, _extract_token_usage
from arcticswarm.llm_client import detect_provider

if TYPE_CHECKING:
    from arcticswarm.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


def _strip_think_prefix(raw: str) -> str:
    """Drop a leading ``<think>...</think>`` block before JSON parsing.

    Tongyi-DeepResearch emits a reasoning preamble even when thinking is
    nominally off (it ignores ``enable_thinking=False``); ``force_json`` on the
    vLLM path normally suppresses it, but this is the belt-and-suspenders so a
    stray preamble doesn't turn every reflection verdict into a parse_error.
    """
    text = raw or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()

# ---------------------------------------------------------------------------
# Hard ceilings (structural, not prompt-based)
# ---------------------------------------------------------------------------

MAX_SEARCH_PLANS: int = 2
"""Outer loop iterations — equivalent to OpenJiuwen's max_plan_executed_num."""

MAX_REFLECTION_LOOPS: int = 2
"""Inner loop per plan — equivalent to OpenJiuwen's max_research_loops."""

RESERVE_TURNS_SUMMARIZE: int = 3
"""Turns reserved for the final summarize-and-post phase."""

_MIN_STEP_BUDGET: int = 3
"""Minimum turns per search phase (floor clamp)."""


def compute_step_budget(
    total_max_turns: int,
    max_plans: int = MAX_SEARCH_PLANS,
    max_loops: int = MAX_REFLECTION_LOOPS,
) -> int:
    """Compute per-search-phase max_turns using OpenJiuwen-style budgeting.

    Ensures capacity is reserved for reflection + summarize phases::

        available = total - summarize_reserve - (plans × loops × 1)
        budget    = available ÷ (plans × loops)

    *max_plans* and *max_loops* default to the module-level constants but
    can be overridden by browsing-specific config values.

    Returns at least ``_MIN_STEP_BUDGET``.
    """
    num_phases = max_plans * max_loops
    # Reserve 1 turn worth of overhead per reflection call
    total_reflection_overhead = num_phases
    available = total_max_turns - RESERVE_TURNS_SUMMARIZE - total_reflection_overhead
    budget = max(available // num_phases, _MIN_STEP_BUDGET)
    return budget


# ---------------------------------------------------------------------------
# Reflection result
# ---------------------------------------------------------------------------

@dataclass
class ReflectionResult:
    """Structured output from the supervisor reflection LLM call."""

    is_sufficient: bool = False
    knowledge_gaps: list[str] = field(default_factory=list)
    next_queries: list[str] = field(default_factory=list)
    confidence: str = "low"  # "low", "medium", "high"
    summary_of_findings: str = ""

    @classmethod
    def from_json(cls, raw: str) -> ReflectionResult:
        """Parse LLM output into a ReflectionResult.

        Handles JSON wrapped in markdown code fences, partial output,
        and malformed responses.  On any parse failure, returns a default
        with ``is_sufficient=False`` so the loop continues safely.
        """
        try:
            text = _strip_think_prefix(raw)

            # Strip markdown code fences if present
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0]

            data: dict[str, Any] = json.loads(text.strip())

            # Normalize knowledge_gaps / knowledge_gap (OpenJiuwen uses singular)
            gaps = data.get("knowledge_gaps", [])
            if not gaps:
                gap_str = data.get("knowledge_gap", "")
                gaps = [gap_str] if gap_str else []

            return cls(
                is_sufficient=bool(data.get("is_sufficient", False)),
                knowledge_gaps=gaps,
                next_queries=data.get("next_queries", []),
                confidence=str(data.get("confidence", "low")),
                summary_of_findings=str(data.get("summary_of_findings", "")),
            )
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse reflection JSON (%s), treating as insufficient", exc)
            return cls(is_sufficient=False, knowledge_gaps=["parse_error"])

    @classmethod
    def from_compact_json(cls, raw: str) -> ReflectionResult:
        """Parse the compact reflection schema.

        Schema::

            {
              "table": {"c1": "E"|"P"|"C"|"U", ...},
              "candidate": {"name": str|null, "alternatives_seen": 0|1, "fame_flag": 0|1},
              "next": "≤80 char query or null"
            }

        Termination is mechanical: ``is_sufficient = all hard constraints
        in {E, P} AND alternatives_seen ≥ 1``. ``confidence`` is derived
        from the table density (all-E = high, all-EP = medium, else low).
        Uses the same ``ReflectionResult`` shape so callers don't change
        and ``reflection_stats`` keeps emitting ``confidence`` for the
        downstream confidence detector.
        """
        try:
            text = _strip_think_prefix(raw)
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0]
            data: dict[str, Any] = json.loads(text.strip())
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
            logger.warning(
                "Failed to parse compact reflection JSON (%s), treating as insufficient",
                exc,
            )
            return cls(is_sufficient=False, knowledge_gaps=["parse_error"])

        table = data.get("table", {}) or {}
        candidate = data.get("candidate", {}) or {}
        next_q = data.get("next") or None

        verdicts = [str(v).strip().upper() for v in table.values() if v]
        if not verdicts:
            return cls(is_sufficient=False, knowledge_gaps=["empty_table"])

        all_in_ep = all(v in ("E", "P") for v in verdicts)
        all_e = all(v == "E" for v in verdicts)
        unverified = [c for c, v in table.items() if str(v).strip().upper() not in ("E", "P")]

        alts_seen = int(candidate.get("alternatives_seen", 0) or 0) >= 1
        is_sufficient = all_in_ep and alts_seen
        if all_e:
            confidence = "high"
        elif all_in_ep:
            confidence = "medium"
        else:
            confidence = "low"

        # Build ``knowledge_gaps`` for downstream consumers (followup prompts
        # rely on this list).  Each unverified constraint id becomes a gap.
        gaps: list[str] = [f"constraint {cid} unverified" for cid in unverified]
        if not alts_seen:
            gaps.append("no alternative candidate explored")

        next_queries = [next_q] if isinstance(next_q, str) and next_q.strip() else []

        cand_name = candidate.get("name") or ""
        summary = (
            f"candidate={cand_name or '?'} "
            f"verified={sum(1 for v in verdicts if v in ('E', 'P'))}/{len(verdicts)} "
            f"alternatives_seen={candidate.get('alternatives_seen', 0)}"
        )

        return cls(
            is_sufficient=is_sufficient,
            knowledge_gaps=gaps,
            next_queries=next_queries,
            confidence=confidence,
            summary_of_findings=summary,
        )


# ---------------------------------------------------------------------------
# Reflection prompts
# ---------------------------------------------------------------------------

# --- Compact ---------------------------------------------------------------

COMPACT_REFLECTION_SYSTEM_PROMPT = """\
You fill a constraint table for a research evaluator. Output ONLY valid JSON.
For each constraint id, mark E (EXACT, primary-source quote verifies),
P (PARTIAL, verified by inference), C (CONTRADICTED), or U (UNKNOWN).
UNKNOWN is fine — do not guess. Set fame_flag=1 for very famous candidates;
do not bias toward fame. Cite a quote in `next` for any EXACT verdict you claim."""


COMPACT_REFLECTION_USER_TEMPLATE = """\
## Question
{question}

## Findings (most recent search results)
{findings}

## Constraints to score
{constraints_block}

## Output (ONLY this JSON, no prose)
{{
  "table": {{ {table_template} }},
  "candidate": {{"name": "<best entity or null>", "alternatives_seen": 0|1, "fame_flag": 0|1}},
  "next": "<≤80-char query that targets the most useful UNKNOWN/C constraint, or null>"
}}"""


def _format_constraints_block(constraints: list[dict[str, Any]] | None) -> tuple[str, str]:
    """Return (constraints_block, table_template) strings.

    When *constraints* is None or empty (caller did not pre-extract), we
    fall back to an instruction that asks the LLM to enumerate ONLY real
    constraints (no padding) and key them c1, c2, ...  This avoids the
    failure mode where the LLM pads unused rows with ``U`` for short
    questions and falsely lowers the high_conf_ratio detector signal.
    """
    if not constraints:
        block = (
            "Identify the constraints in the question (typically 2-6).  "
            "Label them c1, c2, ...  ONLY include rows for real constraints; "
            "do NOT pad with extra rows."
        )
        # Show 3-row example template so the model knows the shape but
        # is not forced into 8 rows.
        template = '"c1": "E|P|C|U", "c2": "E|P|C|U", ...'
        return block, template

    rows = [
        f"- {c.get('id', f'c{i+1}')}: {c.get('text', '')}"
        for i, c in enumerate(constraints)
    ]
    block = "\n".join(rows)
    template = ", ".join(
        f'"{c.get("id", f"c{i+1}")}": "E|P|C|U"'
        for i, c in enumerate(constraints)
    )
    return block, template


# --- Legacy (still used when enable_compact_reflection=False) --------------

REFLECTION_SYSTEM_PROMPT = """\
You are a research quality evaluator and adversarial critic. Your job is to:
1. Identify the candidate answer the researcher is converging on.
2. Check EVERY constraint in the original question against that candidate.
3. Actively look for reasons the candidate might be WRONG.
4. Assess whether the gathered information is sufficient.

You MUST respond with valid JSON only. No other text.

CRITICAL: If ANY constraint from the original question has NOT been explicitly \
verified against the current candidate with cited evidence, set \
is_sufficient=false. Matching most constraints is NOT sufficient — ALL must be \
verified. If the findings focus on only ONE entity with no alternatives \
considered, flag this as a knowledge gap."""


REFLECTION_USER_TEMPLATE = """\
## Original Question
{question}

## Task Assignment
{task_prompt}

## Information Gathered So Far
{findings}

## Search Queries Used So Far
{queries_used}

## Assessment Instructions

First, identify the current best candidate answer from the findings. Then \
evaluate it ADVERSARIALLY — check every constraint in the original question \
against this candidate.

Output ONLY valid JSON with this schema:

{{
  "is_sufficient": <true|false>,
  "confidence": "<low|medium|high>",
  "summary_of_findings": "<1-2 sentence summary including the current best candidate and which constraints are verified vs unverified>",
  "knowledge_gaps": ["<specific gap 1>", "<specific gap 2>"],
  "next_queries": ["<targeted query 1>", "<targeted query 2>"]
}}

Rules:
- Identify which entity/answer the findings are converging on. State it in \
summary_of_findings.
- For EACH constraint in the original question, check: is there explicit \
evidence verifying this constraint against the candidate? If not, list it \
as a knowledge_gap (e.g. "Constraint 'born before 1900' is UNVERIFIED for \
candidate John Smith").
- Set is_sufficient=true ONLY if EVERY constraint has been verified with \
cited evidence. 80%% certainty still requires continuation.
- If findings focus on only ONE entity with no alternatives explored, add a \
knowledge_gap: "No alternative candidates investigated — search may have \
tunnel vision."
- If the current best candidate is a highly famous/well-known entity (Nobel \
laureate, Hollywood star, etc.), add a knowledge_gap: "Current candidate is \
very famous — answers are typically obscure. Search for \
lesser-known alternatives matching the constraints."
- next_queries: Prioritize queries that would DISPROVE the current candidate \
or verify unmatched constraints. Also include queries for alternative \
entities. Be specific and different from queries already used. Maximum 3.
- If the last search round returned mostly redundant information AND all \
constraints are verified, set is_sufficient=true."""


# ---------------------------------------------------------------------------
# Run the reflection call
# ---------------------------------------------------------------------------

def run_reflection(
    client: BaseLLMClient,
    model: str,
    question: str,
    task_prompt: str,
    findings: str,
    queries_used: list[str],
    max_tokens: int = 1024,
    compact: bool = False,
    constraints: list[dict[str, Any]] | None = None,
) -> tuple[ReflectionResult, TokenUsage]:
    """Run a lightweight supervisor reflection LLM call.

    Uses a single non-streaming ``client.call()`` with **no tools** — pure
    text evaluation.  Returns ``(ReflectionResult, TokenUsage)`` so callers
    can roll the cost into their own token-usage accumulator (previously
    only ``output_tokens`` was returned, which meant input/cache tokens were
    silently uncounted).

    When *compact* is True, uses the constraint-checklist
    schema instead of the prose JSON schema.  ~60% fewer tokens per call.
    *constraints* is optional; the compact path falls back to a generic
    8-row template when not provided.
    """
    # GPT reasoning models: use low effort for reflection (simple JSON evaluation)
    # and bump token budget since reasoning tokens consume the output budget.
    is_gpt = model.startswith("gpt") or model.startswith("openai-")
    reasoning = "low" if is_gpt else None
    if is_gpt:
        max_tokens = max(max_tokens, 8000)

    if compact:
        constraints_block, table_template = _format_constraints_block(constraints)
        user_msg = COMPACT_REFLECTION_USER_TEMPLATE.format(
            question=question,
            findings=findings if findings else "(no findings yet)",
            constraints_block=constraints_block,
            table_template=table_template,
        )
        system_prompt = COMPACT_REFLECTION_SYSTEM_PROMPT
    else:
        user_msg = REFLECTION_USER_TEMPLATE.format(
            question=question,
            task_prompt=task_prompt,
            findings=findings if findings else "(no findings yet)",
            queries_used="\n".join(f"- {q}" for q in queries_used) if queries_used else "(none)",
        )
        system_prompt = REFLECTION_SYSTEM_PROMPT

    messages = [{"role": "user", "content": user_msg}]

    try:
        response = client.call(
            model=model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tools=[],
            messages=messages,
            reasoning_effort=reasoning,
            force_json=detect_provider(model) == "vllm",
        )
    except Exception:
        logger.exception("Reflection LLM call failed, treating as insufficient")
        return ReflectionResult(is_sufficient=False, knowledge_gaps=["llm_call_error"]), TokenUsage()

    # Extract text from response
    raw_text = ""
    for block in response.content_blocks:
        if block.get("type") == "text":
            raw_text += block.get("text", "")
    usage = _extract_token_usage(response)

    if compact:
        result = ReflectionResult.from_compact_json(raw_text)
    else:
        result = ReflectionResult.from_json(raw_text)
    return result, usage

