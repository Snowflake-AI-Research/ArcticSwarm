"""WebSearch tool — Brave Search with Tavily and Serper fallbacks.

Searches the web using Brave Search as the primary provider.  When the
Brave query returns empty, the search falls back to the other configured
providers (order configurable via ``provider_order``; default
Brave → Tavily → Serper).

API docs:
  - Brave Search: https://api.search.brave.com/app/documentation/web-search
  - Tavily Search: https://docs.tavily.com/docs/rest-api/api-reference
  - Google Serper: https://serper.dev/docs
"""

from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

import requests

from arcticswarm.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from arcticswarm.tools.source_scorer import SourceScorer

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Search the public web using configurable search providers."""

    name = "web_search"
    description = (
        "Search the public web. "
        "Returns a short list of results (title, url, snippet). "
        "Use this when you need up-to-date info not available in the local codebase."
    )

    # Class-level flag — shared across all instances so the "disabled" warning
    # is logged only once even when multiple subagents each have their own tool.
    _serper_disabled_globally = False

    # --- repeat-guard thresholds (tuned from BrowseComp loop analysis) ---
    # Exact verbatim repeats are blocked from the 2nd issuance (results are
    # identical).  After this many blocks of one exact query the nudge escalates
    # to an is_error message; after the hard cap it forces the agent to stop.
    _ESCALATE_AFTER_BLOCKS = 3
    _EXACT_HARD_STOP_BLOCKS = 6
    # Near-duplicate families (same intent reworded / year-swapped / reordered)
    # only force a stop when RUNAWAY.  In real runs, legitimate year-sweeps reach
    # ~27 distinct variants, so the cap sits well above that — genuine search is
    # never truncated; only pathological loops (50-300+) trip it.
    _NEARDUP_HARD_STOP = 40
    _STOPWORDS = frozenset(
        "the a an of in on at to for and or by with is was were are be been what "
        "which who whose how when where why did do does that this it its from as "
        "vs s no".split()
    )

    def __init__(
        self,
        api_key: str,
        serper_api_key: str = "",
        tavily_api_key: str = "",
        judge: SourceScorer | None = None,
        rich_callback: bool = False,
        provider_order: list[str] | None = None,
        hard_stop: bool = True,
        neardup_hard_stop: int | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._serper_api_key = (serper_api_key or "").strip()
        self._tavily_api_key = (tavily_api_key or "").strip()
        self._judge = judge
        self._rich_callback = rich_callback
        # Normalize the provider try-order: lowercase, keep only known
        # providers, and append any missing known providers at the end so a
        # partial spec still falls back through everything.  Availability is
        # gated separately by API keys, so reordering never drops a provider.
        _known = ["brave", "tavily", "serper"]
        _default = ["brave", "tavily", "serper"]
        # A bare string (e.g. "serper" or "serper,tavily") must be split on
        # commas — iterating the string would yield single characters, which
        # are all dropped by the known-provider filter below, silently
        # reverting to the default order.
        if isinstance(provider_order, str):
            provider_order = [p for p in provider_order.split(",") if p.strip()]
        _order = [str(p).lower().strip() for p in (provider_order or _default)]
        _order = [p for p in _order if p in _known]
        for _p in _default:
            if _p not in _order:
                _order.append(_p)
        self._provider_order = _order
        self._serper_disabled = WebSearchTool._serper_disabled_globally
        self._total_searches = 0
        self._brave_searches = 0
        self._tavily_searches = 0
        self._serper_searches = 0
        self._fallback_log: list[dict[str, Any]] = []
        self._search_log: list[dict[str, Any]] = []
        self._last_brave_error: str | None = None
        self._last_brave_meta: dict[str, Any] | None = None
        # Repeat-query guard state (per tool instance == per subagent).
        # _query_history maps a normalized query -> {"count", "snippet"} so an
        # exact re-issue is short-circuited with the prior result instead of
        # re-hitting the provider.  _family_counts maps a near-duplicate family
        # signature -> issuance count, so a runaway REFORMULATION loop (the same
        # intent reworded dozens of times) can also be detected.  Smaller open
        # models (e.g. Qwen) otherwise loop one query 20-300+ times and burn the
        # whole turn/token budget.
        self._query_history: dict[str, dict[str, Any]] = {}
        self._family_counts: dict[str, int] = {}
        self._family_snippets: dict[str, str] = {}
        self._repeat_blocked = 0
        self._hard_stop = hard_stop
        # Runaway near-duplicate (reformulation-loop) force-stop threshold.
        # Defaults to the class constant; lower it (e.g. 12-15) to bite a
        # churning small model harder, at the cost of possibly truncating a
        # legit broad sweep.
        self._neardup_hard_stop = (
            int(neardup_hard_stop) if neardup_hard_stop else self._NEARDUP_HARD_STOP
        )
        # Soft near-dup nudge zone: once a family has been reworded this many
        # times (but before the hard stop), append a "pivot to a different
        # angle" hint to the (still-returned) results.
        self._neardup_soft = max(3, self._neardup_hard_stop // 3)

        # Build description based on active providers, following the actual
        # try-order in ``self._provider_order`` so the advertised fallback
        # sequence matches execution.  Each provider is gated by its API key.
        _provider_meta = {
            "brave": ("Brave Search", self._api_key),
            "tavily": ("Tavily Search", self._tavily_api_key),
            "serper": ("Google Serper", self._serper_api_key),
        }
        providers: list[str] = [
            _provider_meta[p][0]
            for p in self._provider_order
            if _provider_meta[p][1]
        ]
        if providers:
            primary = providers[0]
            if len(providers) > 1:
                fallbacks = ", ".join(providers[1:])
                provider_desc = f"{primary} ({fallbacks} as fallback)"
            else:
                provider_desc = primary
            self.description = (
                f"Search the public web using {provider_desc}. "
                "Returns a short list of results (title, url, snippet). "
                "Use this when you need up-to-date info not available in the local codebase."
            )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (max 400 chars).",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return (1-20). Default: 5.",
                },
                "country": {
                    "type": "string",
                    "description": "Optional 2-letter country code to influence results (e.g. 'US').",
                },
                "safesearch": {
                    "type": "string",
                    "description": "SafeSearch level: 'off', 'moderate', or 'strict'. Default: 'moderate'.",
                },
            },
            "required": ["query"],
        }

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize a query for repeat detection.

        Conservative on purpose: lowercase + whitespace-collapse only.  This
        catches verbatim repeats (and trivial case/spacing variants) without
        over-matching genuinely different queries, so a legitimately refined
        search is never blocked.
        """
        return " ".join((query or "").lower().split())

    @classmethod
    def _family_key(cls, query: str) -> str:
        """Near-duplicate signature: collapse reformulations of one intent.

        Lowercase → strip quotes/punctuation → tokenize → drop 4-digit year
        tokens and stopwords → SORTED DISTINCT token set.  Two queries that
        differ only by quoting, word order, year tokens, or stopwords map to the
        same key (validated on real BrowseComp loops: year-sweep, word-reorder,
        quote-toggle).  Returns "" for an empty/degenerate query.
        """
        toks = re.findall(r"[a-z0-9]+", (query or "").lower())
        keep = [
            t for t in toks
            if t not in cls._STOPWORDS and not re.fullmatch(r"(?:19|20)\d{2}", t)
        ]
        return " ".join(sorted(set(keep)))

    @staticmethod
    def _guard_snippet(output: str, limit: int = 600) -> str:
        """Trim a prior result's output to a compact snippet for the guard msg."""
        text = (output or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " …(truncated)"

    def _guard_result(self, snippet: str, n: int, *, force: bool) -> ToolResult:
        """Build the repeat-guard ToolResult with severity graduated by count.

        ``force=True`` attaches ``metadata.force_stop`` which the agent turn loop
        (:meth:`Agent._tool_batch_terminates_turn`) honors to end the agent's
        turn — a real bail that does not depend on the model choosing to comply.
        """
        meta: dict[str, Any] = {"search_source": "repeat_guard", "repeat_count": n}
        if force:
            meta["force_stop"] = True
            msg = (
                f"STOP — you have issued this search {n} times and the results have "
                f"not changed:\n\n{snippet}\n\n"
                "This line of search is exhausted. Do NOT search again. Write your "
                "best answer now from the evidence you already have (complete_task / "
                "prepare_report / send_user_markdown_report). If you cannot fully "
                "answer, give your best candidate and state what is missing."
            )
            return ToolResult(output=msg, is_error=True, metadata=meta)
        if n >= self._ESCALATE_AFTER_BLOCKS:
            msg = (
                f"You have already run this query {n} times — the results are "
                f"unchanged:\n\n{snippet}\n\n"
                "Re-running it will not help. Issue a DIFFERENT query (new "
                "keywords/entities), or stop searching and use what you have."
            )
            return ToolResult(output=msg, is_error=True, metadata=meta)
        msg = (
            f"You already ran this exact query {n} times — the results have not "
            f"changed:\n\n{snippet}\n\n"
            "Try a DIFFERENT query (new keywords/entities/dates), refine or broaden "
            "your terms, web_fetch a specific URL from the results above, or post "
            "what you have and move on."
        )
        return ToolResult(output=msg, metadata=meta)

    def execute(
        self,
        *,
        query: str,
        count: int = 5,
        country: str | None = None,
        safesearch: str = "moderate",
        **kwargs: Any,
    ) -> ToolResult:
        """Repeat-guarded entry point.

        Three tiers, each tuned so that legitimate diverse search / year-sweeps
        are never blocked (only pathological loops are):
          1. EXACT verbatim repeat → short-circuit with the prior result; the
             nudge escalates to is_error after ``_ESCALATE_AFTER_BLOCKS`` and
             forces a stop after ``_EXACT_HARD_STOP_BLOCKS``.
          2. RUNAWAY near-duplicate family (≥ ``_NEARDUP_HARD_STOP`` reworded
             issuances of one intent) → force a stop.
          3. Otherwise run the real search and remember the result.
        """
        norm = self._normalize_query(query)
        fam = self._family_key(query)
        fam_count = 0
        if fam:
            fam_count = self._family_counts.get(fam, 0) + 1
            self._family_counts[fam] = fam_count

        # 1) exact verbatim repeat
        prior = self._query_history.get(norm) if norm else None
        if prior is not None:
            prior["count"] += 1
            n = prior["count"]
            self._repeat_blocked += 1
            force = self._hard_stop and n >= self._EXACT_HARD_STOP_BLOCKS
            logger.info(
                "web_search repeat-guard: blocked exact repeat (x%d)%s of query: %s",
                n, " [FORCE-STOP]" if force else "", (query or "").strip()[:80],
            )
            return self._guard_result(prior["snippet"], n, force=force)

        # 2) runaway near-duplicate family (reformulation loop)
        if self._hard_stop and fam and fam_count >= self._neardup_hard_stop:
            self._repeat_blocked += 1
            logger.info(
                "web_search repeat-guard: runaway near-dup family (x%d) [FORCE-STOP] of query: %s",
                fam_count, (query or "").strip()[:80],
            )
            return self._guard_result(
                self._family_snippets.get(fam, "(no prior result captured)"),
                fam_count, force=True,
            )

        result = self._search_impl(
            query=query, count=count, country=country, safesearch=safesearch, **kwargs
        )

        # Soft near-dup nudge: this intent has been reworded several times but is
        # still below the hard stop — return the real results, but prepend a hint
        # to pivot to a different angle rather than reword again.
        if (
            self._hard_stop and fam and not result.is_error
            and self._neardup_soft <= fam_count < self._neardup_hard_stop
        ):
            result.output = (
                f"(NOTE: you've searched this same question ~{fam_count} different ways "
                "now. If these results still don't answer it, pivot to a DIFFERENT angle "
                "— a different entity/person, source type, or date/number range, or "
                "web_fetch a specific promising URL — rather than rewording again.)\n\n"
                + (result.output or "")
            )

        # Remember successful (non-error) results so a later repeat can surface
        # them.  Errors (bad params, no keys) are not memoized — the agent
        # should be free to retry after fixing the input.
        if norm and not result.is_error:
            snippet = self._guard_snippet(result.output)
            self._query_history[norm] = {"count": 1, "snippet": snippet}
            if fam:
                self._family_snippets.setdefault(fam, snippet)
        return result

    def _search_impl(
        self,
        *,
        query: str,
        count: int = 5,
        country: str | None = None,
        safesearch: str = "moderate",
        **_: Any,
    ) -> ToolResult:
        # --- input validation ---
        if not self._api_key and not self._tavily_api_key and not self._serper_api_key:
            return ToolResult(
                error=(
                    "No search API keys configured. Set brave_api_key, "
                    "tavily_api_key, or serper_api_key in "
                    "config_files.json, "
                    "or export BRAVE_API_KEY / TAVILY_API_KEY / SERPER_API_KEY."
                ),
                is_error=True,
            )

        q = (query or "").strip()
        if not q:
            return ToolResult(error="Missing required parameter: query", is_error=True)

        if len(q) > 400:
            return ToolResult(error=f"Query too long ({len(q)} chars). Max 400 characters.", is_error=True)
        word_count = len(q.split())
        if word_count > 50:
            return ToolResult(error=f"Query too long ({word_count} words). Max 50 words.", is_error=True)

        capped = max(1, min(int(count or 5), 20))
        safe = (safesearch or "moderate").lower()
        if safe not in ("off", "moderate", "strict"):
            safe = "moderate"

        # Provider try-order (default Brave → Tavily → Serper); configurable
        # via ``provider_order``.  The first provider returning usable results
        # wins.  Providers without a configured API key are skipped.
        fallback_entry: dict[str, Any] | None = None

        for provider in self._provider_order:
            if provider == "brave":
                if not self._api_key:
                    continue
                result = self._brave_stage(q, capped, country, safe)
                if result is not None:
                    return result
                logger.info("Brave returned no results for query: %s — trying fallbacks", q[:80])
                if fallback_entry is None:
                    fallback_entry = self._make_fallback_entry(q, self._last_brave_error)
            elif provider == "tavily":
                result = self._try_tavily(q, capped, fallback_entry)
                if result is not None:
                    return result
            elif provider == "serper":
                result = self._serper_stage(q, capped, country, fallback_entry)
                if result is not None:
                    return result

        # All providers failed or returned no results
        self._record_search(q, "none", 0)
        if fallback_entry is not None:
            self._fallback_log.append(fallback_entry)

        return ToolResult(
            output=(
                f"No results for: {q}\n\n"
                "Suggestions: Try rephrasing with different keywords, "
                "remove quoted phrases, or break the query into simpler parts."
            ),
        )

    def log_and_reset_stats(self) -> None:
        """Log web search stats and reset counters."""
        if self._total_searches > 0:
            logger.info(
                "Web search stats: %d total (brave=%d, tavily=%d, serper=%d)",
                self._total_searches,
                self._brave_searches,
                self._tavily_searches,
                self._serper_searches,
            )
        if self._repeat_blocked > 0:
            logger.info(
                "Web search repeat-guard: blocked %d exact-repeat query attempt(s)",
                self._repeat_blocked,
            )
        self._total_searches = 0
        self._brave_searches = 0
        self._tavily_searches = 0
        self._serper_searches = 0
        self._repeat_blocked = 0
        self._query_history.clear()
        self._family_counts.clear()
        self._family_snippets.clear()

    def drain_fallback_log(self) -> list[dict[str, Any]]:
        """Return and clear accumulated Brave fallback events."""
        log = list(self._fallback_log)
        self._fallback_log.clear()
        return log

    def drain_search_log(self) -> list[dict[str, Any]]:
        """Return and clear the full per-query search log (all providers)."""
        log = list(self._search_log)
        self._search_log.clear()
        return log

    def _record_search(
        self,
        query: str,
        source: str,
        result_count: int,
        scores: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append an entry to the full search log."""
        from arcticswarm.logging_utils import summarize_search_scores

        entry: dict[str, Any] = {
            "query": query,
            "source": source,
            "result_count": result_count,
            "query_features": self._query_features(query),
        }
        score_summary = summarize_search_scores(scores)
        if score_summary:
            entry["scores"] = score_summary
        self._search_log.append(entry)

    # ----- Fallback helpers --------------------------------------------------

    @staticmethod
    def _query_features(query: str) -> dict[str, Any]:
        """Extract structural features from a search query for diagnostics."""
        quoted = re.findall(r'"([^"]+)"', query)
        return {
            "num_quoted_phrases": len(quoted),
            "total_quoted_chars": sum(len(p) for p in quoted),
            "has_OR": " OR " in query,
            "has_site": "site:" in query,
            "has_negation": query.startswith("-") or " -" in query,
            "has_range": ".." in query,
            "word_count": len(query.split()),
            "char_length": len(query),
        }

    def _make_fallback_entry(self, query: str, brave_error: str | None = None) -> dict[str, Any]:
        """Create a fresh fallback-tracking dict for *query*.

        Called when the Brave search fails and the query falls through to
        the other providers (Tavily / Serper).
        """
        entry: dict[str, Any] = {
            "query": query,
            # Brave search failed
            "brave_failed": True,
            "brave_error": brave_error,
            # Fallback results
            "tavily_success": False,
            "tavily_result_count": 0,
            "tavily_top_results": [],
            "serper_success": False,
            "serper_result_count": 0,
            "final_source": None,
            "query_features": self._query_features(query),
        }
        if self._last_brave_meta:
            entry["brave_response_meta"] = self._last_brave_meta
        return entry

    def _try_tavily(
        self, q: str, capped: int, fallback_entry: dict[str, Any] | None,
    ) -> ToolResult | None:
        """Attempt Tavily search. Returns ToolResult on success, None to continue."""
        if not self._tavily_api_key:
            return None
        tavily_results = self._search_tavily(q, capped)
        if not tavily_results:
            logger.info("Tavily returned no results for query: %s — trying next fallback", q[:80])
            return None
        self._total_searches += 1
        self._tavily_searches += 1
        scores = self._score_results(q, tavily_results)
        self._record_search(q, "tavily", len(tavily_results), scores)
        if fallback_entry is not None:
            fallback_entry["tavily_result_count"] = len(tavily_results)
            fallback_entry["tavily_top_results"] = [
                {"title": r.get("title", ""), "url": r.get("url", "")}
                for r in tavily_results[:3]
            ]
            fallback_entry["final_source"] = "tavily"
            self._fallback_log.append(fallback_entry)
        return self._format_results(tavily_results, q, capped, source="tavily", scores=scores)

    def _brave_stage(
        self,
        q: str,
        capped: int,
        country: str | None,
        safe: str,
    ) -> ToolResult | None:
        """Run the Brave stage (exact-query search).

        Returns a ToolResult when Brave yields usable results, or None
        when Brave returns nothing (the caller then tries other providers).
        """
        brave_results = self._search_brave(q, capped, country, safe)
        if brave_results:
            # Score for annotation in a single LLM call (no rejection).
            scores = self._score_results(q, brave_results)
            self._total_searches += 1
            self._brave_searches += 1
            self._record_search(q, "brave", len(brave_results), scores)
            return self._format_results(brave_results, q, capped, source="brave", scores=scores)

        return None

    def _serper_stage(
        self,
        q: str,
        capped: int,
        country: str | None,
        fallback_entry: dict[str, Any] | None,
    ) -> ToolResult | None:
        """Attempt Serper search. Returns ToolResult on success, None otherwise."""
        if not self._serper_api_key or self._serper_disabled:
            return None
        serper_results = self._search_serper(q, capped, country)
        if not serper_results:
            return None
        self._total_searches += 1
        self._serper_searches += 1
        scores = self._score_results(q, serper_results)
        self._record_search(q, "serper", len(serper_results), scores)
        if fallback_entry is not None:
            fallback_entry["serper_success"] = True
            fallback_entry["serper_result_count"] = len(serper_results)
            fallback_entry["final_source"] = "serper"
            self._fallback_log.append(fallback_entry)
        return self._format_results(serper_results, q, capped, source="serper", scores=scores)

    # ----- Query quote-relaxation helpers (shared with cortex_search) --------

    @staticmethod
    def _build_or_unquote_rewrite(query: str) -> str | None:
        """Rewrite a multi-quoted query: OR the phrases for recall, append unquoted for ranking.

        ``"A" "B" "C" keywords`` becomes
        ``("A" OR "B" OR "C") A B C keywords``

        Returns ``None`` when the query has fewer than 2 quoted phrases or
        already contains an ``OR`` operator.
        """
        if " OR " in query:
            return None

        spans: list[tuple[int, int, str]] = []
        for m in re.finditer(r'"([^"]+)"', query):
            spans.append((m.start(), m.end(), m.group(1)))

        if len(spans) < 2:
            return None

        # Build the OR group of all quoted phrases
        or_group = "(" + " OR ".join(f'"{phrase}"' for _, _, phrase in spans) + ")"

        # Collect every term unquoted (phrases + any non-quoted keywords)
        unquoted_parts: list[str] = []
        prev_end = 0
        for start, end, phrase in spans:
            between = query[prev_end:start].strip()
            if between:
                unquoted_parts.append(between)
            unquoted_parts.append(phrase)
            prev_end = end
        trailing = query[spans[-1][1]:].strip()
        if trailing:
            unquoted_parts.append(trailing)

        return or_group + " " + " ".join(unquoted_parts)

    @staticmethod
    def _build_unquote_only(query: str) -> str | None:
        """Strip all double-quotes from the query, returning plain keywords.

        ``"A" "B" "C" keywords`` becomes ``A B C keywords``

        Returns ``None`` when the query contains no quoted phrases (nothing
        to strip).
        """
        if '"' not in query:
            return None
        return re.sub(r'"', "", query).strip()

    # ----- Brave Search API -------------------------------------------------

    def _search_brave(
        self, query: str, count: int, country: str | None, safesearch: str,
    ) -> list[dict[str, str]] | None:
        """Call Brave Search API. Returns normalized results or None to signal fallback."""
        params: dict[str, Any] = {
            "q": query,
            "count": count,
            "safesearch": safesearch,
            "enable_rich_callback": self._rich_callback,
        }
        if country:
            params["country"] = str(country).upper()

        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self._api_key,
                    "User-Agent": "arcticswarm/0.1.0",
                },
                params=params,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Brave search failed, will try fallback: %s", exc)
            self._last_brave_error = str(exc)
            return None

        web = data.get("web") or {}
        results = web.get("results") or []
        if not isinstance(results, list) or not results:
            self._last_brave_error = None
            # Capture Brave response metadata for diagnostics
            query_meta = data.get("query") or {}
            self._last_brave_meta = {
                "altered_query": query_meta.get("altered"),
                "bad_results": query_meta.get("bad_results", False),
                "more_results_available": query_meta.get("more_results_available", False),
                "spellcheck_off": query_meta.get("spellcheck_off", False),
                "response_keys": sorted(data.keys()),
                "web_total_results": web.get("totalResults", 0),
            }
            return None
        self._last_brave_meta = None

        # Normalize to common field names
        normalized = []
        for r in results[:count]:
            entry: dict[str, Any] = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", ""),
            }
            normalized.append(entry)
        return normalized

    # ----- Tavily Search API ------------------------------------------------

    def _search_tavily(
        self, query: str, count: int,
    ) -> list[dict[str, str]] | None:
        """Call Tavily Search API. Returns normalized results or None on failure."""
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "api_key": self._tavily_api_key,
                    "query": query,
                    "max_results": count,
                    "search_depth": "advanced",
                    "include_answer": True,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Tavily search failed, will try Serper: %s", exc)
            return None

        results = data.get("results") or []
        if not isinstance(results, list) or not results:
            return None

        normalized = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("content", ""),
                "tavily_score": r.get("score", 0.0),
            }
            for r in results[:count]
        ]

        # Attach AI-generated answer summary if available
        answer = (data.get("answer") or "").strip()
        if answer and normalized:
            normalized[0]["_tavily_answer"] = answer

        return normalized

    # ----- Google Serper API ------------------------------------------------

    def _search_serper(
        self, query: str, count: int, country: str | None,
    ) -> list[dict[str, str]] | None:
        """Call Google Serper API. Returns normalized results or None on failure."""
        if self._serper_disabled:
            return None

        payload: dict[str, Any] = {
            "q": query,
            "num": count,
        }
        if country:
            payload["gl"] = str(country).lower()

        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self._serper_api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=20,
            )
            if resp.status_code >= 400:
                # Check for credit exhaustion or auth errors — disable permanently
                body = ""
                try:
                    body = resp.text
                except Exception:
                    pass
                if any(kw in body.lower() for kw in ("credit", "quota", "limit", "unauthorized")):
                    if not WebSearchTool._serper_disabled_globally:
                        logger.warning("Serper API disabled — %s: %s", resp.status_code, body[:200])
                    self._serper_disabled = True
                    WebSearchTool._serper_disabled_globally = True
                    return None
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Serper search also failed: %s", exc)
            return None

        results = data.get("organic") or []
        if not isinstance(results, list) or not results:
            return None

        # Normalize Serper fields (link/snippet) to Brave fields (url/description)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "description": r.get("snippet", ""),
            }
            for r in results[:count]
        ]

    # ----- Shared formatter -------------------------------------------------

    def _score_results(
        self, query: str, results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score search results for annotation in one LLM call.

        Returns the per-result scores (``[]`` on failure — fail-open).
        Results are never rejected; scoring only annotates them.
        """
        if not self._judge:
            return []
        try:
            sources = [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("description", ""),
                }
                for r in results
            ]
            scored, _ = self._judge.evaluate(query, sources)
            return scored
        except Exception as exc:
            logger.warning("Search result scoring failed: %s", exc)
            return []

    @staticmethod
    def _format_results(
        results: list[dict[str, Any]],
        query: str,
        count: int,
        source: str = "brave",
        scores: list[dict[str, Any]] | None = None,
    ) -> ToolResult:
        """Format normalized search results into the standard numbered output."""
        # Build index → score lookup from scored results
        score_by_idx: dict[int, dict[str, Any]] = {}
        if scores:
            from arcticswarm.tools.source_scorer import SourceScorer
            for s in scores:
                score_by_idx[s["index"]] = s

        lines: list[str] = [f"Top {min(len(results), count)} result(s) for: {query}", ""]

        # Include Tavily AI-generated answer summary if present
        tavily_answer = ""
        if results:
            tavily_answer = str(results[0].get("_tavily_answer") or "").strip()
        if tavily_answer:
            lines.append(f"AI Summary: {tavily_answer}")
            lines.append("")

        for i, r in enumerate(results[:count], start=1):
            title = str(r.get("title") or "").strip()
            url = str(r.get("url") or "").strip()
            desc = str(r.get("description") or "").strip()
            if not title and not url:
                continue
            lines.append(f"{i}. {title}" if title else f"{i}. (no title)")
            if url:
                lines.append(f"   URL: {url}")
            if desc:
                lines.append(f"   Snippet: {desc}")
            tavily_score = r.get("tavily_score")
            if tavily_score and float(tavily_score) > 0:
                lines.append(f"   Relevance: {float(tavily_score):.2f}")
            extra = r.get("extra_snippets")
            if extra and isinstance(extra, list):
                for snippet in extra:
                    s = str(snippet).strip()
                    if s:
                        lines.append(f"   Extra: {s}")
            # Append source quality annotation if available
            sc = score_by_idx.get(i - 1)
            if sc:
                lines.append(f"   {SourceScorer.format_annotation(sc).strip()}")
            lines.append("")

        return ToolResult(
            output="\n".join(lines).rstrip(),
            metadata={"search_source": source},
        )
