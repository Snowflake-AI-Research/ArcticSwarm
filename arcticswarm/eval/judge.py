"""LLM judge for evaluating agent responses.

Supports these evaluation modes:
  - **QA**: Binary correct/incorrect (matches the Go ``qa_eval.tmpl``).
  - **Answer-Only**: 0/1/2 rating focusing purely on the final answer
    correctness, ignoring methodology and tool usage (matches the Go
    ``answer_only_eval.tmpl``).
  - **BrowseComp / BrowseComp-Plus / EvoBrowseComp**: dataset-specific
    correctness graders.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_JUDGE_MODEL = "claude-4-sonnet"
_MAX_JUDGE_TOKENS = 4096


def _resolve_served_model_id(client: Any, configured: str) -> str:
    """Return a model id the endpoint actually serves.

    A self-hosted judge endpoint may serve a different id than the configured
    ``eval.judge_model`` (e.g. config says ``Qwen/Qwen3-30B-A3B-Instruct-2507``
    but the endpoint serves ``Qwen/Qwen3-32B``). Sending the wrong id 404s on
    every judge call, silently scoring all cases incorrect. If *configured*
    isn't served, fall back to the endpoint's first served id. Returns
    *configured* unchanged on any probe failure.
    """
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as exc:  # noqa: BLE001 — never block judging on a probe
        logger.debug("Could not list judge endpoint models: %s", exc)
        return configured
    if not served or configured in served:
        return configured
    logger.info(
        "Judge model %r not served by endpoint; using served id %r instead.",
        configured, served[0],
    )
    return served[0]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class QAJudgeResult:
    """Result from QA mode evaluation."""

    correct: bool = False
    comment: str = ""
    raw_output: str = ""
    # LLM judge (optional): calibration / extraction
    judge_confidence: float | None = None  # 0–100 from judge output
    extracted_final_answer: str | None = None


@dataclass
class InsightJudgeResult:
    """Result from Insight mode evaluation."""

    rating: int = 0  # 0, 1, or 2
    analysis: str = ""
    reasoning: str = ""
    raw_output: str = ""


@dataclass
class FlexJudgeResult:
    """Result from FLEX mode evaluation (4 independent metrics).

    Mirrors Go's ``QualityScoreResult`` (``llm_judge.go:251-271``).
    """

    # Pre-analysis metadata (extracted from judge reasoning)
    order_sensitivity: str = ""       # STRICT, PARTIAL, NONE
    question_parts: int = 0           # Number of distinct sub-questions
    question_type: str = ""           # FACTUAL, CAUSAL, COMPARATIVE
    acceptable_alternatives: str = "" # Stated alternatives from expected answer

    # Extracted facts (for debugging)
    expected_facts: str = ""
    response_facts: str = ""

    # Industry-standard metrics
    flex_answer_accuracy: int = 0       # 0 or 1
    answer_groundedness: float = 0.0    # 0.00 to 1.00
    answer_relevancy: int = 0           # 0 or 1
    methodology_soundness: int = 0      # 0 or 1

    # Reasoning for each metric
    accuracy_reasoning: str = ""
    groundedness_reasoning: str = ""
    relevancy_reasoning: str = ""
    soundness_reasoning: str = ""
    raw_output: str = ""


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Evaluate agent responses using an LLM judge via the Anthropic API.

    When *use_azure_openai* is ``True``, the judge routes through Azure
    OpenAI instead (requires ``AZURE_OPENAI_API_KEY``,
    ``AZURE_OPENAI_ENDPOINT``, and optionally ``OPENAI_API_VERSION`` env
    vars).
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        model: str = _DEFAULT_JUDGE_MODEL,
        use_azure_openai: bool = False,
        judge_base_url: str = "",
    ) -> None:
        self.model = model
        # ``judge_base_url`` => self-hosted OpenAI-compatible endpoint (vLLM).
        # Reuses the existing Chat Completions path (``_use_openai``) but with a
        # plain ``openai.OpenAI`` client instead of AzureOpenAI.
        self._use_openai = use_azure_openai or bool(judge_base_url)
        self.content_filter_fallback_count = 0
        self._judge_parse_failure_count = 0

        if judge_base_url:
            from openai import OpenAI

            # vLLM requires a non-empty key string even when auth is disabled.
            self._openai_client = OpenAI(base_url=judge_base_url, api_key=api_key or "EMPTY")
            # The endpoint may serve a different id than the configured judge
            # model — resolve to what's actually served so judge calls don't 404.
            self.model = _resolve_served_model_id(self._openai_client, model)
        elif use_azure_openai:
            import os
            from openai import AzureOpenAI
            from arcticswarm.config import load_settings

            settings = load_settings()

            self._openai_client = AzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY") or settings.get("AZURE_OPENAI_API_KEY", ""),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT") or settings.get("AZURE_OPENAI_ENDPOINT", ""),
                api_version=os.environ.get("OPENAI_API_VERSION") or settings.get("OPENAI_API_VERSION", "2025-04-01-preview"),
            )
        else:
            client_kwargs: dict[str, Any] = {}
            is_custom = base_url and base_url != "https://api.anthropic.com"
            if api_key:
                if is_custom:
                    client_kwargs["auth_token"] = api_key
                else:
                    client_kwargs["api_key"] = api_key
            if is_custom:
                client_kwargs["base_url"] = base_url
            self._client = anthropic.Anthropic(**client_kwargs)

        # Load prompt templates
        self._qa_prompt = (_PROMPTS_DIR / "qa_eval.txt").read_text()
        self._answer_only_prompt = (_PROMPTS_DIR / "answer_only_eval.txt").read_text()
        self._browsecomp_prompt = (_PROMPTS_DIR / "browsecomp_eval.txt").read_text()
        self._browsecomp_plus_prompt = (_PROMPTS_DIR / "browsecomp_plus_eval.txt").read_text()
        self._evobrowsecomp_prompt = (_PROMPTS_DIR / "evobrowsecomp_eval.txt").read_text()
        self._seal0_prompt = (_PROMPTS_DIR / "seal0_eval.txt").read_text()

    # ----- QA mode ----------------------------------------------------------

    def judge_qa(
        self,
        question: str,
        answer: str,
        expected_answer: str,
    ) -> QAJudgeResult:
        """Run the QA judge.  Returns a :class:`QAJudgeResult`."""
        if not answer:
            return QAJudgeResult(
                correct=False,
                comment="Agent produced no answer.",
                raw_output="",
            )

        prompt = self._qa_prompt.format(
            question=question,
            expected_answer=expected_answer,
            answer=answer,
        )

        raw = self._call_llm(prompt)
        return self._parse_qa_output(raw)

    @staticmethod
    def _parse_qa_output(raw: str) -> QAJudgeResult:
        """Parse JSON ``{"COMMENT": ..., "EVALUATION": ...}`` from the judge."""
        # Try to extract JSON object from the response
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not json_match:
            logger.warning("QA judge returned non-JSON output: %s", raw[:200])
            return QAJudgeResult(correct=False, comment=raw, raw_output=raw)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("QA judge JSON parse failed: %s", raw[:200])
            return QAJudgeResult(correct=False, comment=raw, raw_output=raw)

        evaluation = str(data.get("EVALUATION", "")).strip().lower()
        comment = str(data.get("COMMENT", ""))
        correct = evaluation == "correct"
        return QAJudgeResult(correct=correct, comment=comment, raw_output=raw)

    # ----- Rating parser (0/1/2, shared by answer-only mode) ----------------

    @staticmethod
    def _parse_insight_output(raw: str) -> InsightJudgeResult:
        """Parse ``ANALYSIS: ... REASONING: ... RATING: N`` from the judge."""
        analysis = ""
        reasoning = ""
        rating = 0

        # Extract ANALYSIS
        analysis_match = re.search(
            r"ANALYSIS:\s*(.*?)(?=REASONING:|$)", raw, re.DOTALL
        )
        if analysis_match:
            analysis = analysis_match.group(1).strip()

        # Extract REASONING
        reasoning_match = re.search(
            r"REASONING:\s*(.*?)(?=RATING:|$)", raw, re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract RATING
        rating_match = re.search(r"RATING:\s*(\d)", raw)
        if rating_match:
            rating = int(rating_match.group(1))
            rating = max(0, min(2, rating))  # clamp to [0, 2]

        return InsightJudgeResult(
            rating=rating,
            analysis=analysis,
            reasoning=reasoning,
            raw_output=raw,
        )

    # ----- Answer-only mode -------------------------------------------------

    def judge_answer_only(
        self,
        question: str,
        answer: str,
        expected_answer: str,
        analysis_date: str = "",
    ) -> InsightJudgeResult:
        """Run the answer-only judge.

        Evaluates only the final answer against the expected answer,
        ignoring methodology, tool usage, and SQL.  Returns an
        :class:`InsightJudgeResult` (same 0/1/2 rating scale as Insight).
        """
        if not answer:
            return InsightJudgeResult(
                rating=0,
                analysis="Agent produced no answer.",
                raw_output="",
            )

        prompt = self._answer_only_prompt.format(
            question=question,
            expected_answer=expected_answer,
            answer=answer,
            analysis_date=analysis_date or "(not specified)",
        )

        raw = self._call_llm(prompt)
        return self._parse_insight_output(raw)

    # ----- BrowseComp mode --------------------------------------------------

    def judge_browsecomp(
        self,
        question: str,
        answer: str,
        expected_answer: str,
    ) -> QAJudgeResult:
        """Run the BrowseComp judge (matches OpenAI's grading logic).

        Returns a :class:`QAJudgeResult`.
        """
        if not answer:
            return QAJudgeResult(
                correct=False,
                comment="Agent produced no answer.",
                raw_output="",
            )

        prompt = self._browsecomp_prompt.format(
            question=question,
            expected_answer=expected_answer,
            answer=answer,
        )

        raw = self._call_llm(prompt)
        return self._parse_browsecomp_output(raw)

    def judge_browsecomp_plus(
        self,
        question: str,
        answer: str,
        expected_answer: str,
    ) -> QAJudgeResult:
        """Run the BrowseComp-Plus judge (official grading prompt from texttron/BrowseComp-Plus)."""
        if not answer:
            return QAJudgeResult(
                correct=False,
                comment="Agent produced no answer.",
                raw_output="",
            )

        prompt = self._browsecomp_plus_prompt.format(
            question=question,
            expected_answer=expected_answer,
            answer=answer,
        )

        raw = self._call_llm(prompt)
        return self._parse_browsecomp_output(raw)

    def _parse_browsecomp_output(self, raw: str) -> QAJudgeResult:
        """Parse BrowseComp judge output.
        The original OpenAI implementation uses regex to find 'correct: (yes|no)'.
        Also handles markdown-formatted output (e.g. **correct:** yes).
        """
        # Strip markdown bold markers so "**correct:** yes" becomes "correct: yes"
        stripped = raw.replace("**", "")
        # Original: match = re.search(r"correct: (yes|no)", grading_response)
        match = re.search(r"correct:\s*(yes|no)", stripped, re.IGNORECASE)
        if match:
            correct = match.group(1).lower() == "yes"
            # Extract reasoning if present
            reasoning_match = re.search(r"reasoning:\s*(.+?)(?=\ncorrect:|\nconfidence:|$)", stripped, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            return QAJudgeResult(correct=correct, comment=reasoning or raw, raw_output=raw)

        # Default to incorrect if we can't parse (matching original: "no" default)
        self._judge_parse_failure_count = getattr(self, "_judge_parse_failure_count", 0) + 1
        logger.warning("BrowseComp judge output could not be parsed: %s", raw[:200])
        return QAJudgeResult(correct=False, comment=raw, raw_output=raw)

    # ----- EvoBrowseComp mode -----------------------------------------------

    def judge_evobrowsecomp(
        self,
        question: str,
        answer: str,
        expected_answer: str,
    ) -> QAJudgeResult:
        """Run the EvoBrowseComp judge (grading prompt from the EvoBrowseComp paper).

        EvoBrowseComp uses a GAIA-style judge that emits ``Final Answer:`` /
        ``Explanation:`` / ``Conclusion: Correct|Incorrect`` rather than
        BrowseComp's ``correct: yes|no``.  Returns a :class:`QAJudgeResult`.
        """
        if not answer:
            return QAJudgeResult(
                correct=False,
                comment="Agent produced no answer.",
                raw_output="",
            )

        prompt = self._evobrowsecomp_prompt.format(
            question=question,
            correct_answer=expected_answer,
            response=answer,
        )

        raw = self._call_llm(prompt)
        return self._parse_evobrowsecomp_output(raw)

    def _parse_evobrowsecomp_output(self, raw: str) -> QAJudgeResult:
        """Parse EvoBrowseComp judge output.

        The verdict lives on the ``Conclusion:`` line as ``Correct`` or
        ``Incorrect``.  We test ``Incorrect`` first (with a word boundary) so
        the substring ``correct`` inside ``Incorrect`` never reads as a pass.
        Also extracts the ``Final Answer:`` and ``Explanation:`` sections.
        """
        # Strip markdown bold markers so "**Conclusion:** Correct" parses.
        stripped = raw.replace("**", "")

        # Scan from the last "Conclusion:" so a stray "Conclusion" inside the
        # explanation can't shadow the real verdict.
        idx = stripped.lower().rfind("conclusion:")
        verdict_region = stripped[idx:] if idx != -1 else ""

        correct: bool | None = None
        if verdict_region:
            if re.search(r"\bincorrect\b", verdict_region, re.IGNORECASE):
                correct = False
            elif re.search(r"\bcorrect\b", verdict_region, re.IGNORECASE):
                correct = True

        # Best-effort extraction of the structured sections for the comment.
        fa_match = re.search(
            r"Final Answer:\s*(.+?)(?=\n\s*Explanation:|\n\s*Conclusion:|$)",
            stripped, re.DOTALL | re.IGNORECASE,
        )
        expl_match = re.search(
            r"Explanation:\s*(.+?)(?=\n\s*Conclusion:|$)",
            stripped, re.DOTALL | re.IGNORECASE,
        )
        final_answer = fa_match.group(1).strip() if fa_match else None
        explanation = expl_match.group(1).strip() if expl_match else ""

        if correct is None:
            # Could not find a parseable verdict — default to incorrect, matching
            # the conservative BrowseComp behavior.
            self._judge_parse_failure_count = getattr(self, "_judge_parse_failure_count", 0) + 1
            logger.warning("EvoBrowseComp judge output could not be parsed: %s", raw[:200])
            return QAJudgeResult(
                correct=False,
                comment=explanation or raw,
                raw_output=raw,
                extracted_final_answer=final_answer,
            )

        return QAJudgeResult(
            correct=correct,
            comment=explanation or raw,
            raw_output=raw,
            extracted_final_answer=final_answer,
        )

    # ----- SEAL-0 mode ------------------------------------------------------
    def judge_seal0(
        self,
        question: str,
        answer: str,
        expected_answer: str,
    ) -> QAJudgeResult:
        """Run the SEAL-0 (SealQA) judge.

        Uses the verbatim grader template from the SealQA authors' grading
        Colab (a SimpleQA-style rubric): the judge emits a single letter
        ``A`` (CORRECT) / ``B`` (INCORRECT) / ``C`` (NOT_ATTEMPTED).  Only
        ``A`` counts as correct — NOT_ATTEMPTED scores as not-correct, matching
        the paper's reported accuracy (fraction of CORRECT).  Returns a
        :class:`QAJudgeResult`.
        """
        if not answer:
            return QAJudgeResult(
                correct=False,
                comment="Agent produced no answer.",
                raw_output="",
            )

        prompt = self._seal0_prompt.format(
            question=question,
            target=expected_answer,
            predicted_answer=answer,
        )

        # Paper grades at temperature 0.0.
        raw = self._call_llm(prompt, temperature=0.0)
        return self._parse_seal0_output(raw)

    @staticmethod
    def _parse_seal0_output(raw: str) -> QAJudgeResult:
        """Parse the SEAL-0 grader's single-letter verdict (A/B/C).

        A = CORRECT, B = INCORRECT, C = NOT_ATTEMPTED.  Only A is correct.
        Scans for the first standalone A/B/C letter (matching the authors'
        ``re.search(r"(A|B|C)")``); if the grader spelled the verdict out
        instead, fall back to the word form (INCORRECT before CORRECT so the
        substring never misreads).  An unparseable verdict defaults to C
        (not-correct), the conservative behavior.
        """
        match = re.search(r"\b([ABC])\b", raw)
        if match:
            letter = match.group(1)
        elif re.search(r"NOT[_\s]*ATTEMPTED", raw, re.IGNORECASE):
            letter = "C"
        elif re.search(r"\bINCORRECT\b", raw, re.IGNORECASE):
            letter = "B"
        elif re.search(r"\bCORRECT\b", raw, re.IGNORECASE):
            letter = "A"
        else:
            m = re.search(r"([ABC])", raw)
            letter = m.group(1) if m else "C"
        grade = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}[letter]
        return QAJudgeResult(
            correct=(letter == "A"),
            comment=grade,
            raw_output=raw,
        )

    # ----- LLM call ---------------------------------------------------------

    def _call_llm(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = _MAX_JUDGE_TOKENS,
    ) -> str:
        """Make a single LLM call and return the text."""
        try:
            if self._use_openai:
                return self._call_openai(prompt, temperature=temperature, max_tokens=max_tokens)
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = self._client.messages.create(**kwargs)
            parts = []
            for block in response.content:
                if block.type == "text":
                    parts.append(block.text)
            return "".join(parts)
        except Exception as exc:
            logger.error("LLM judge call failed: %s", exc, exc_info=True)
            return f"Error: {exc}"

    def _call_openai(self, prompt: str, *, temperature: float | None = None, max_tokens: int = _MAX_JUDGE_TOKENS) -> str:
        """Make a single Azure OpenAI chat completion call with rate-limit retry and content filter fallback."""
        import time
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self._openai_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "rate" in exc_str.lower():
                    wait = 3 * (attempt + 1)
                    logger.warning("Azure OpenAI rate limited (attempt %d/%d), retrying in %ds", attempt + 1, max_retries, wait)
                    time.sleep(wait)
                elif "content_filter" in exc_str or "content management policy" in exc_str:
                    self.content_filter_fallback_count += 1
                    logger.warning("Azure content filter triggered, falling back to Anthropic judge (total fallbacks: %d)", self.content_filter_fallback_count)
                    return self._call_anthropic_fallback(prompt, temperature=temperature, max_tokens=max_tokens)
                else:
                    raise
        # Final attempt, let it raise
        response = self._openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _call_anthropic_fallback(self, prompt: str, *, temperature: float | None = None, max_tokens: int = _MAX_JUDGE_TOKENS) -> str:
        """Fallback to Anthropic claude-4-sonnet for content-filtered cases."""
        fallback = anthropic.Anthropic()
        kwargs: dict[str, Any] = {
            "model": _DEFAULT_JUDGE_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = fallback.messages.create(**kwargs)
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)
