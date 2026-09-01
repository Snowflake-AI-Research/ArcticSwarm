"""Automatic source quality scoring via a secondary GPT model.

Evaluates web sources on four dimensions (relevance, answerability, authority,
data_density) and returns text annotations that are appended to web_fetch tool
results.

Also provides ``judge_search_results()`` for evaluating web *search* results
on the same four dimensions with an accept/reject gate used by
``WebSearchTool`` to decide whether to fall back from Brave to Serper.

This is a **pipeline component**, not an agent-callable tool.  The agent loop
in ``agent.py`` calls ``SourceScorer.evaluate()`` automatically after each
tool-call batch that includes ``web_fetch``, then annotates the tool results
so the browsing agent sees quality scores alongside the fetched content.

Design mirrors DeepSearch's ``info_evaluator``: scores are advisory text
annotations — no documents are dropped programmatically.

Routing follows the same pattern as ``llm_client.py``:
- Default: Cortex proxy via ``openai.OpenAI(base_url=…)``
- Azure: ``openai.AzureOpenAI(…)`` when ``use_azure_openai=True``
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation prompt (adapted from DeepSearch info_evaluator.md)
# ---------------------------------------------------------------------------

_EVALUATOR_SYSTEM_PROMPT = """\
You are a text quality evaluation expert.  Rate the provided query and \
text segments (sources) according to the following dimensions.

### Scoring Dimensions (0-10 scale)

#### Relevance
Measures the direct connection between the content and the **specific topic \
and core concepts** of the query.
- **8-10**: Explicitly addresses the query's specific topic and core concepts.
- **4-7**: Mentions related concepts but does not focus on the specific topic.
- **0-3**: Unrelated or only shares vague keywords.

#### Answerability
How effectively the segment provides **direct, specific information** to \
answer the query.
- **8-10**: Provides concrete details, data, or solutions that directly \
answer part or all of the query.
- **4-7**: Offers background context but does not directly answer the query.
- **0-3**: Fails to provide useful information for answering.

#### Authority
The **credibility** of the information source and **reliability** of the content.
- **8-10**: Authoritative source; content is well-sourced, consistent with \
verified facts, no obvious bias.
- **4-7**: Basic professional qualifications; content is generally reliable \
but may contain slight bias.
- **0-3**: No verifiable credentials; content has errors, extreme bias, or \
is unverifiable.

#### Data Density
Concentration of empirical or quantitative evidence in the content.
- **8-10**: Substantial quantitative evidence (numbers, dates, names, facts) \
central to answering the query.
- **4-7**: Some specific data but limited scope or depth.
- **0-3**: Minimal or no meaningful data, mostly subjective opinions or \
generic descriptions.

### Output Format

Return a JSON object with one key:
- "results": a JSON array where each element is a dict with:
  - "index": the 0-based index of the source in the input list
  - "scores": {"relevance": float, "answerability": float, "authority": float, "data_density": float}

Example (pure JSON, no markdown fences):
{
  "results": [
    {"index": 0, "scores": {"relevance": 9.0, "answerability": 8.5, "authority": 9.0, "data_density": 8.0}},
    {"index": 1, "scores": {"relevance": 7.0, "answerability": 6.5, "authority": 8.0, "data_density": 5.0}}
  ]
}

IMPORTANT: Return ONLY the JSON object.  No explanations, no markdown."""


# ---------------------------------------------------------------------------
# Search result judge prompt (used by judge_search_results)
# ---------------------------------------------------------------------------

_SEARCH_JUDGE_SYSTEM = """\
You are a search result quality evaluator. Given a user query and search \
results, evaluate quality and decide whether the results adequately address \
the query."""

_SEARCH_JUDGE_PROMPT = """\
Evaluate the following search results for the given query.

### Scoring Dimensions (0-10 scale)

1. **Relevance**: Direct connection between results and the query's specific \
topic, core concepts, and sub-questions.
   - High (8-10): Explicitly addresses the query's specific topic and core concepts.
   - Medium (4-7): Mentions general concepts but doesn't focus on the specific topic.
   - Low (0-3): Discusses unrelated topics.

2. **Answerability**: How effectively the results provide direct, specific \
information to answer the query.
   - High (8-10): Provides concrete details, examples, or solutions.
   - Medium (4-7): Offers background context but doesn't directly answer.
   - Low (0-3): Fails to provide helpful information.

3. **Authority**: Authority and reliability of the information sources.
   - High (8-10): Clear authoritative qualifications, consistent with verified facts.
   - Medium (4-7): Basic professional qualifications, generally credible.
   - Low (0-3): No verifiable qualifications, factual errors, or extreme bias.

4. **Data Density**: Concentration of empirical or quantitative evidence.
   - High (8-10): Substantial quantitative evidence central to the analysis.
   - Medium (4-7): Some empirical data but limited scope or depth.
   - Low (0-3): Minimal or no meaningful data, mostly subjective opinions.

Return your evaluation as a JSON object (no markdown, no extra text):
{{
  "scores": {{"relevance": 8.0, "answerability": 7.5, "authority": 6.0, "data_density": 5.0}},
  "doc_time": "2024 Mar"
}}

### Query
{query}

### Search Results
{results}"""


# ---------------------------------------------------------------------------
# JudgeResult — returned by judge_search_results()
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """Outcome of a search content judgment."""

    accept: bool = True
    scores: dict[str, float] = field(default_factory=dict)
    doc_time: str = ""
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return metadata dict suitable for merging into ToolResult.metadata."""
        d: dict[str, Any] = {"search_scores": self.scores}
        if self.doc_time:
            d["doc_time"] = self.doc_time
        d["search_accepted"] = self.accept
        return d


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

_MAX_SOURCES = 10          # max sources per evaluation call
_MAX_CONTENT_CHARS = 2000  # truncate each source's content

# Default scorer model — a cheap GPT model routed through Cortex proxy
_DEFAULT_SCORER_MODEL = "openai-gpt-5-chat"

# Anti-bot / captcha boilerplate from web pages that the Azure prompt-shield
# misclassifies as a jailbreak attempt.  Replace these with a stub so the
# scorer LLM can still see that the source returned nothing useful, without
# the page-level meta-text triggering jailbreak detection.
_BOT_GUARD_RE = re.compile(
    r"(JavaScript is disabled[\s\S]{0,200}?(verify (that )?you'?re not a robot|robot)[\s\S]{0,400}?(reload|cookies|JavaScript)"
    r"|Just a moment[\s\S]{0,200}?Cloudflare"
    r"|Please (verify|confirm) (you are|that you'?re) (a )?human"
    r"|Checking your browser before accessing"
    r"|Enable JavaScript and (then )?reload the page)",
    re.IGNORECASE,
)
_BOT_GUARD_STUB = "[bot-detection page; content unavailable to scorer]"


def _scrub_bot_boilerplate(content: str) -> str:
    """Strip anti-bot/captcha boilerplate that triggers Azure jailbreak filter."""
    if not content:
        return content
    return _BOT_GUARD_RE.sub(_BOT_GUARD_STUB, content)


def _strip_markdown_fences(text: str) -> str:
    """Extract JSON from text that may be wrapped in markdown code fences
    or preceded by preamble text.

    Handles: bare JSON, ```json...```, ```<lang>...```, ```...```,
    trailing text after the closing fence, and preamble text before JSON.
    """
    text = text.strip()
    # Strip markdown fences first
    if "```" in text:
        after_open = text.split("```", 1)[1]
        # Strip the language tag on the first line (e.g. "json", "AI assistant")
        first_newline = after_open.find("\n")
        if first_newline != -1:
            after_open = after_open[first_newline + 1:]
        content = after_open.split("```", 1)[0]
        return content.strip()
    # No fences — try to find the JSON array/object in the text.
    # Skip false positives like "[Source 2]" by requiring '[' to be
    # followed (after whitespace/newlines) by '{' or ']'.
    for i, ch in enumerate(text):
        if ch == "{":
            return text[i:]
        if ch == "[":
            after_bracket = text[i + 1:].lstrip()
            if after_bracket and after_bracket[0] in ("{", "]", "["):
                return text[i:]
            # Not a JSON array — e.g. "[Source 2]", keep scanning
    return text.strip()


# ---------------------------------------------------------------------------
# SourceScorer
# ---------------------------------------------------------------------------


class SourceScorer:
    """Evaluate web sources using a secondary LLM model.

    By default, uses the agent's own LLM client (Claude) as primary and
    falls back to GPT via the Cortex OpenAI proxy on transient errors.

    Routing follows the same pattern as the main agent's LLM client:
    - Agent client — uses the agent's own ``BaseLLMClient.call()``.
    - Cortex proxy — ``openai.OpenAI`` with ``base_url`` derived from the
      Anthropic proxy URL.
    - Azure OpenAI — ``openai.AzureOpenAI`` when ``use_azure_openai=True``.

    Instantiate once per Agent and reuse.  The OpenAI client is created
    lazily on first call (or first fallback attempt).
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        model: str = _DEFAULT_SCORER_MODEL,
        # Separate PAT for OpenAI/GPT via Cortex proxy
        openai_api_key: str = "",
        # Azure-specific (only used when use_azure_openai=True)
        use_azure_openai: bool = False,
        azure_openai_api_key: str = "",
        azure_openai_endpoint: str = "",
        azure_openai_api_version: str = "2025-04-01-preview",
        # When provided, use the agent's own LLM client instead of OpenAI
        agent_client: Any = None,
        agent_model: str = "",
        # When True, NEVER call the OpenAI/GPT path (no closed-model calls).
        # Used for self-hosted vLLM (Qwen) runs: scoring/compaction must run
        # on the agent's own model, with no GPT fallback on transient errors.
        disable_closed_model_fallback: bool = False,
    ) -> None:
        self._api_key = api_key
        self._openai_api_key = openai_api_key
        self._base_url = base_url
        self._model = model
        self._use_azure = use_azure_openai
        self._azure_api_key = azure_openai_api_key
        self._azure_endpoint = azure_openai_endpoint
        self._azure_api_version = azure_openai_api_version
        self._agent_client = agent_client  # BaseLLMClient (Anthropic or OpenAI)
        self._agent_model = agent_model or model
        self._disable_closed_model_fallback = disable_closed_model_fallback
        self._client: Any = None
        self._disabled = False  # set True after first connection failure
        self.content_filter_count: int = 0  # Azure content filter blocks
        self._content_filter_log: list[dict[str, Any]] = []  # detailed logs
        # Per-role token ledger keyed by role label ("source_scorer",
        # "compactor", ...).  Drained by the eval runner / orchestrator to
        # build per-role breakdowns in `swarm_token_usage_breakdown`.
        self._token_ledger: dict[str, dict[str, int]] = {}

    def drain_token_ledger(self) -> dict[str, dict[str, int]]:
        """Return and clear per-role token usage."""
        out = self._token_ledger
        self._token_ledger = {}
        return out

    def _record_usage(
        self,
        role: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        bucket = self._token_ledger.setdefault(role, {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "calls": 0,
        })
        bucket["input_tokens"] += int(input_tokens or 0)
        bucket["output_tokens"] += int(output_tokens or 0)
        bucket["cache_creation_input_tokens"] += int(cache_creation_input_tokens or 0)
        bucket["cache_read_input_tokens"] += int(cache_read_input_tokens or 0)
        bucket["calls"] += 1

    def _get_client(self) -> Any:
        if self._client is None:
            if self._use_azure and self._azure_endpoint:
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    api_key=self._azure_api_key or self._api_key,
                    azure_endpoint=self._azure_endpoint,
                    api_version=self._azure_api_version,
                )
            else:
                from openai import OpenAI
                from arcticswarm.llm_client import derive_openai_base_url
                openai_url = derive_openai_base_url(self._base_url)
                self._client = OpenAI(
                    api_key=self._openai_api_key or self._api_key,
                    base_url=openai_url,
                )
        return self._client

    def _call_agent(
        self, system_prompt: str, user_msg: str, max_tokens: int,
        *, role: str = "source_scorer",
    ) -> str:
        """Call the agent's own LLM client.  Raises if agent_client is None."""
        if self._agent_client is None:
            raise RuntimeError("No agent client available for source scoring")
        # GPT reasoning models: use low effort for scoring (simple classification)
        # and bump token budget since reasoning tokens consume the output budget.
        is_gpt = self._agent_model.startswith("gpt") or self._agent_model.startswith("openai-")
        from arcticswarm.llm_client import detect_provider
        is_vllm = detect_provider(self._agent_model) == "vllm"
        if is_gpt:
            reasoning = "low"
            max_tokens = max(max_tokens, 8000)
        elif is_vllm:
            # Self-hosted vLLM (Qwen): scoring/compaction is a simple
            # structured task. Disable thinking (reasoning_effort="none" -> the
            # vLLM client sends chat_template_kwargs thinking=False / instant
            # mode) so the small token budget isn't consumed by a <think> block,
            # which otherwise produced empty / truncated-JSON responses.
            # Verified on Qwen: ~1929 -> ~254 output tokens, clean JSON.
            reasoning = "none"
            max_tokens = max(max_tokens, 4000)
        else:
            reasoning = None
        response = self._agent_client.call(
            model=self._agent_model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tools=[],
            messages=[{"role": "user", "content": user_msg}],
            reasoning_effort=reasoning,
            # Self-hosted reasoning models (e.g. Tongyi-DeepResearch) emit a
            # reasoning preamble around the JSON even with thinking off, which
            # breaks parsing and forces the truncation fallback. Constrain vLLM
            # output to valid JSON. (Inherited by ContentCompactor too.)
            force_json=is_vllm,
        )
        self._record_usage(
            role,
            input_tokens=getattr(response, "input_tokens", 0),
            output_tokens=getattr(response, "output_tokens", 0),
            cache_creation_input_tokens=getattr(
                response, "cache_creation_input_tokens", 0,
            ),
            cache_read_input_tokens=getattr(
                response, "cache_read_input_tokens", 0,
            ),
        )
        text = ""
        for block in response.content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")
        result = text.strip()
        if not result:
            raise RuntimeError("Agent LLM returned empty response")
        return result

    def _call_openai(
        self, system_prompt: str, user_msg: str, max_tokens: int,
        *, role: str = "source_scorer",
    ) -> str:
        """Call the OpenAI / Azure OpenAI endpoint.  Raises on error."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_completion_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._record_usage(
                role,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        content = response.choices[0].message.content
        result = content.strip() if content else ""
        if not result:
            raise RuntimeError("OpenAI LLM returned empty response")
        return result

    @staticmethod
    def _is_permanent_error(exc: Exception) -> bool:
        """Return True for errors that should permanently disable the scorer."""
        error_str = str(exc)
        return "404" in error_str or "401" in error_str or "403" in error_str

    def _call_llm(
        self, system_prompt: str, user_msg: str, max_tokens: int = 2000,
        *, role: str = "source_scorer",
    ) -> str:
        """Route LLM call with automatic fallback.

        The agent's own model is primary and OpenAI (GPT) is the fallback.

        Permanent errors (404/401/403) are re-raised so the caller can
        disable scoring.  Transient errors trigger a fallback attempt.

        When ``disable_closed_model_fallback`` is set (self-hosted vLLM/Qwen
        runs) the OpenAI/GPT path is never used: the agent's own model is the
        sole provider and a primary failure is re-raised with no fallback.
        """
        if self._disable_closed_model_fallback:
            # No closed-model calls: agent (Qwen) only, no GPT fallback.
            return self._call_agent(system_prompt, user_msg, max_tokens, role=role)

        primary, fallback = self._call_agent, self._call_openai

        try:
            return primary(system_prompt, user_msg, max_tokens, role=role)
        except Exception as e:
            if self._is_permanent_error(e):
                raise
            # Try fallback on transient errors (connection, timeout, etc.)
            try:
                logger.warning("Source scoring primary LLM failed: %s; trying fallback", e)
                return fallback(system_prompt, user_msg, max_tokens, role=role)
            except Exception:
                # Fallback also failed (or unavailable) — raise original
                raise e

    def evaluate(
        self,
        query: str,
        sources: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Score a batch of sources and decide whether to accept them.

        Each input source dict should have ``url`` and ``content``.
        Returns ``(scored_results, accept)`` where each scored dict has
        ``url``, ``relevance``, ``answerability``, ``authority``,
        ``data_density``, ``composite`` and ``accept`` is the overall
        accept/reject decision.

        Returns ``([], True)`` on failure (fail-open — never breaks the
        agent loop).
        """
        if not query or not sources:
            return [], True
        if self._disabled:
            return [], True

        sources = sources[:_MAX_SOURCES]
        truncated = [
            {
                "url": s.get("url", ""),
                "title": s.get("title", s.get("url", "")),
                "content": _scrub_bot_boilerplate(s.get("content", ""))[:_MAX_CONTENT_CHARS],
            }
            for s in sources
        ]

        source_text = "\n\n".join(
            f"[Source {i}]\nURL: {s['url']}\nTitle: {s['title']}\n"
            f"Content: {s['content']}"
            for i, s in enumerate(truncated)
        )
        user_msg = f"Query: {query}\n\nSources to evaluate:\n\n{source_text}"

        try:
            raw = self._call_llm(
                _EVALUATOR_SYSTEM_PROMPT, user_msg, max_tokens=2000,
                role="source_scorer",
            )
        except Exception as e:
            error_str = str(e)
            # Disable permanently on 404 (deployment not found) or auth errors
            if "404" in error_str or "401" in error_str or "403" in error_str:
                logger.warning("Source scoring disabled — endpoint error: %s", e)
                self._disabled = True
            elif "content_filter" in error_str or "content management policy" in error_str:
                self.content_filter_count += 1
                self._content_filter_log.append({
                    "system_prompt": _EVALUATOR_SYSTEM_PROMPT,
                    "user_msg": user_msg,
                    "error": error_str,
                })
                logger.warning("Source scoring blocked by content filter (count=%d): %s", self.content_filter_count, e)
            else:
                logger.warning("Source scoring LLM call failed: %s", e)
            return [], True

        try:
            cleaned = _strip_markdown_fences(raw)
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            # Try to repair common LLM JSON typos
            try:
                repaired = cleaned
                # Safety refusal mid-JSON: truncate at refusal and close the array/object.
                # e.g. `"relevance": I'm sorry...` → salvage entries before the refusal.
                refusal_patterns = [
                    "I'm sorry", "I cannot", "I can't", "I apologize",
                    "I'm not able", "I am sorry", "I am not able",
                ]
                for pat in refusal_patterns:
                    idx = repaired.find(pat)
                    if idx != -1:
                        # Backtrack to the last complete JSON entry
                        before = repaired[:idx]
                        # Find last complete `}` and close the structure
                        last_brace = before.rfind("}")
                        if last_brace != -1:
                            repaired = before[:last_brace + 1] + '], "accept": true}'
                        break
                # Fix extra closing brace: `}}}` → `}}` (LLM adds extra } on last element)
                repaired = repaired.replace("}}}", "}}")
                # Fix missing quotes on keys
                repaired = re.sub(r'"(\w+)(?::)', r'"\1":', repaired)
                # Fix missing "scores" key: `"index": 0, {` → `"index": 0, "scores": {`
                repaired = re.sub(
                    r'("index"\s*:\s*\d+\s*,)\s*\{',
                    r'\1 "scores": {',
                    repaired,
                )
                parsed = json.loads(repaired)
            except (json.JSONDecodeError, ValueError, Exception) as e:
                logger.warning(
                    "Failed to parse source scoring response: %s\nRaw: %s", e, raw,
                )
                return [], True

        # Support both old format (JSON array) and new format (JSON object
        # with "results" and "accept" keys).
        accept = True
        if isinstance(parsed, dict):
            scores_list = parsed.get("results", [])
            accept = bool(parsed.get("accept", True))
        elif isinstance(parsed, list):
            scores_list = parsed
        else:
            logger.warning("Unexpected source scoring response type: %s", type(parsed))
            return [], True

        # Build scored results keyed by index
        scored: list[dict[str, Any]] = []
        # ``scores_list`` may not be a list (some models return a bare string or
        # object under "results"); coerce to an empty list so we skip gracefully.
        if not isinstance(scores_list, list):
            logger.warning(
                "Source scoring 'results' was %s, not a list; skipping",
                type(scores_list).__name__,
            )
            scores_list = []
        for item in scores_list:
            # Tongyi/other vLLM models with forced json_object sometimes emit
            # valid JSON whose result entries are strings or other non-objects
            # (e.g. ["relevant", ...]). Skip anything we can't read as a dict
            # rather than blowing up the whole scoring call.
            if not isinstance(item, dict):
                continue
            idx = item.get("index", 0)
            if not isinstance(idx, int) or idx >= len(truncated) or idx < 0:
                continue
            s_scores = item.get("scores", {})
            if not isinstance(s_scores, dict):
                s_scores = {}

            def _num(v: Any) -> float:
                # Scores may arrive as strings ("8"), "8/10", or non-numeric
                # ("high") when a model ignores the schema. Salvage what we can.
                try:
                    return float(v)
                except (TypeError, ValueError):
                    if isinstance(v, str):
                        m = re.search(r"-?\d+(?:\.\d+)?", v)
                        if m:
                            return float(m.group())
                    return 0.0

            rel = _num(s_scores.get("relevance", 0))
            ans = _num(s_scores.get("answerability", 0))
            auth = _num(s_scores.get("authority", 0))
            dd = _num(s_scores.get("data_density", 0))
            composite = rel + ans + auth + dd

            scored.append({
                "index": idx,
                "url": truncated[idx]["url"],
                "relevance": rel,
                "answerability": ans,
                "authority": auth,
                "data_density": dd,
                "composite": round(composite, 1),
            })

        return scored, accept

    @staticmethod
    def format_annotation(score: dict[str, Any]) -> str:
        """Format a single source's scores as a one-line annotation string."""
        return (
            f"\n[Source Quality: relevance={score['relevance']}/10, "
            f"answerability={score['answerability']}/10, "
            f"authority={score['authority']}/10, "
            f"data_density={score['data_density']}/10 "
            f"(composite={score['composite']}/40)]"
        )

    def drain_content_filter_log(self) -> list[dict[str, Any]]:
        """Return and clear accumulated content filter rejection logs."""
        log = self._content_filter_log
        self._content_filter_log = []
        return log

    # ------------------------------------------------------------------
    # Search result judge (used by WebSearchTool for Brave→Serper gating)
    # ------------------------------------------------------------------

    def judge_search_results(
        self,
        query: str,
        results: str,
        *,
        is_final: bool = False,
    ) -> JudgeResult:
        """Score search results and decide whether to accept them.

        Evaluates on four dimensions (relevance, answerability, authority,
        data_density) plus document freshness (doc_time).  Always requests
        an accept/reject decision; when ``is_final=True`` the caller treats
        the result as accepted regardless (no further fallback available).

        Args:
            query: The original search query.
            results: Formatted search result text (title/URL/snippet list).
            is_final: If True, the result's ``accept`` flag is forced to True
                after scoring (no fallback provider to try).

        Returns:
            A :class:`JudgeResult` with scores, doc_time, and accept flag.
        """
        if self._disabled:
            return JudgeResult(accept=True)

        prompt = _SEARCH_JUDGE_PROMPT.format(
            query=query,
            results=results,
        )

        try:
            raw = self._call_llm(
                _SEARCH_JUDGE_SYSTEM, prompt, max_tokens=1024,
                role="source_scorer",
            )
        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "401" in error_str or "403" in error_str:
                logger.warning("Search judge disabled — endpoint error: %s", e)
                self._disabled = True
            else:
                logger.warning("Search judge LLM call failed: %s", e)
            return JudgeResult(accept=True)  # fail-open

        return self._parse_judge_result(raw, is_final=is_final)

    @staticmethod
    def _parse_judge_result(raw: str, *, is_final: bool) -> JudgeResult:
        """Parse LLM JSON output into a JudgeResult."""
        result = JudgeResult()
        result.raw_output = raw

        if not raw.strip():
            result.accept = True
            return result

        # Strip markdown code fences and trailing text if present
        text = _strip_markdown_fences(raw)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to repair common LLM JSON typos (e.g. missing quotes on keys)
            try:
                repaired = re.sub(r'"(\w+)(?::)', r'"\1":', text)
                data = json.loads(repaired)
            except (json.JSONDecodeError, Exception):
                logger.warning("Search judge returned non-JSON: %s", raw[:200])
                result.accept = True  # fail-open
                return result

        scores = data.get("scores") or {}
        result.scores = {
            "relevance": float(scores.get("relevance", 0)),
            "answerability": float(scores.get("answerability", 0)),
            "authority": float(scores.get("authority", 0)),
            "data_density": float(scores.get("data_density", 0)),
        }
        result.doc_time = str(data.get("doc_time", ""))

        # When is_final, always accept — no fallback provider to try
        result.accept = True if is_final else bool(data.get("accept", True))

        return result
