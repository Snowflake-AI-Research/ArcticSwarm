"""Empty / refusal answer detection and cheap-win recovery turn.

Extracted from ``arcticswarm.swarm.orchestrator`` so the orchestrator module
stays focused on the main control loop.  This module owns:

- ``extract_answer_from_messages``: bypass fallback that recovers an answer
  from the orchestrator's own assistant turns when ``send_user_markdown_report``
  was never called.
- ``is_empty_or_refusal``: cheap detector for empty / refusal answers, used
  to gate the recovery turn.
- ``run_empty_answer_recovery_turn``: injects a single best-guess recovery
  turn when the orchestrator's first answer is empty or a refusal.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from arcticswarm.agent import StreamEvent
from arcticswarm.swarm.teammate import _TimingCollector, _inject_timings_into_messages

logger = logging.getLogger(__name__)


def extract_answer_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract the orchestrator's text answer from its conversation messages.

    When the orchestrator solves a question directly (e.g., via the reasoning
    tool) without delegating to subagents, ``report_tool.captured_report`` is
    empty because ``prepare_report`` gates on ``task_count > 0``.  This
    fallback recovers the answer from the orchestrator's own assistant turns.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)
    return ""


_REFUSAL_MARKERS: tuple[str, ...] = (
    "unable to determine",
    "insufficient information",
    "cannot determine",
    "cannot answer",
    "i don't have enough",
    "i do not have enough",
    "no answer can be",
    "not enough information",
    "unanswerable",
    "unsolvable",
    "no candidate satisfies",
)


# Broader give-up detector for the FINAL ANSWER line specifically (used by the
# qwen-gated ``reject_refusal_reports`` report bounce). Derived from an earlier
# give-up-detection GIVEUP regex, with the 3rd alternation group extended with
# ``satisf`` so "no candidate satisfies all constraints" — the most common qwen
# give-up phrasing — is caught (it matched NEITHER ``_REFUSAL_MARKERS`` nor the
# original regex). Kept inline here so it ships with the package.
_GIVEUP_RE = re.compile(
    r"no (valid |single |definitive )?(answer|monument|player|person|candidate|"
    r"match|publication|individual|paper|name|location|film|song|book|company|"
    r"title)\b.{0,40}(exist|found|identif|determin|could be|matches|available|satisf)|"
    r"could not (be )?(determin|identif|find|conclusively)|"
    r"unable to (determin|identif|find|conclusively|provide)|"
    r"no definitive answer|insufficient (evidence|information|data)|"
    r"cannot (be )?(determin|conclude|identif)",
    re.IGNORECASE,
)

# Tolerates optional markdown bolding around the marker, e.g. ``**FINAL ANSWER:**``.
_FINAL_ANSWER_RE = re.compile(r"\**\s*FINAL ANSWER\s*\**\s*:?\s*", re.IGNORECASE)


def extract_final_answer(report: str | None) -> str:
    """Return the text after the ``FINAL ANSWER:`` marker (marker→end), stripped.

    The answer may span multiple lines. Returns ``""`` when there is no marker.
    """
    if not report:
        return ""
    m = _FINAL_ANSWER_RE.search(report)
    if not m:
        return ""
    # Strip a trailing ``**`` left by ``**FINAL ANSWER:**`` bolding, plus space.
    return report[m.end():].strip().lstrip("*").strip()


def final_answer_is_giveup(report: str | None) -> bool:
    """True when the report's FINAL ANSWER is a refusal / give-up.

    Used by the qwen-gated ``reject_refusal_reports`` bounce in
    ``SendReportTool.execute``. Operates on the extracted FINAL ANSWER line
    (NOT the whole report — the body legitimately discusses caveats), and
    deliberately does NOT apply ``is_empty_or_refusal``'s ``min_len`` heuristic,
    which would mis-flag short but correct answers ("Tokyo", a full legal name).
    Falls back to scanning the whole report only when no marker is present.
    """
    if not report or not report.strip():
        return True
    m = _FINAL_ANSWER_RE.search(report)
    if m:
        fa = extract_final_answer(report)
        if not fa:
            return True  # marker present but empty -> give-up
        target = fa
    else:
        target = report  # no marker at all -> scan the whole report
    lo = target.lower()
    return any(mk in lo for mk in _REFUSAL_MARKERS) or bool(_GIVEUP_RE.search(target))


def is_empty_or_refusal(answer: str | None, *, min_len: int = 30) -> bool:
    """Return True when *answer* is missing or looks like a refusal.

    Used by the cheap-win recovery turn to detect cases that
    would otherwise skip Layer 4a and slip through as wrong.  Matches the
    empty/refusal wrong-case shape seen during early calibration.
    """
    if not answer or not answer.strip():
        return True
    stripped = answer.strip()
    if len(stripped) < min_len:
        return True
    lo = stripped.lower()
    return any(m in lo for m in _REFUSAL_MARKERS)


_RECOVERY_MSG = (
    "## Recovery Turn\n\n"
    "Your previous response did not produce a final "
    "answer (empty or refusal). Re-evaluate the BBS "
    "evidence and produce your best-guess answer that "
    "fits the most constraints — uncertainty is "
    "acceptable.\n\n"
    "Rules:\n"
    "- The answer ALWAYS exists. Pick the candidate that "
    "best matches the most constraints.\n"
    "- Note any constraints that remain unverified.\n"
    "- Call `prepare_report` then "
    "`send_user_markdown_report`. Do NOT declare the "
    "question unsolvable."
)


def run_empty_answer_recovery_turn(
    *,
    agent: Any,
    answer: str,
    report_tool: Any,
    on_agent_event: Callable[[StreamEvent], None] | None,
) -> str:
    """Inject ONE best-guess recovery turn and return the (possibly new) answer.

    The caller is responsible for gating this on its single-shot flag and on
    ``config.enable_empty_answer_recovery`` / wrap-up state — this function
    just runs the turn unconditionally and returns the better of the two
    answers.  A failure during the turn is logged and the original answer
    is returned unchanged.
    """
    agent._tools.pop("send_user_markdown_report", None)
    msg_start_idx_cw = len(agent.messages)
    orch_collector_cw = _TimingCollector(inner_on_event=on_agent_event)
    orch_collector_cw.start()
    try:
        agent.run_turn_streaming(
            _RECOVERY_MSG,
            on_event=orch_collector_cw.on_event,
        )
        _inject_timings_into_messages(
            agent.messages,
            orch_collector_cw,
            msg_start_idx_cw,
        )
        answer_cw = report_tool.captured_report or ""
        if not answer_cw.strip():
            answer_cw = extract_answer_from_messages(agent.messages)
        if answer_cw and not is_empty_or_refusal(answer_cw):
            logger.info(
                "Cheap-win recovery produced answer (len=%d)", len(answer_cw),
            )
            return answer_cw
    except Exception:
        logger.exception(
            "Cheap-win recovery turn failed — keeping empty answer"
        )
    return answer
