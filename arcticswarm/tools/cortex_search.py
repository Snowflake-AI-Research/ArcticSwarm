"""Cortex web-search provider (``web.provider = cortex`` / ``cortex-grounding``).

Routes ``web_search`` (and optionally ``web_fetch``) through the Snowflake
Cortex ``agent:run`` web-search *passthrough* endpoint instead of calling
Brave/Tavily/Serper directly.  Passthrough mode (``UseWebSearchPassthrough``)
bypasses LLM orchestration and executes the search directly, returning Brave
Search (``api_mode="search"``) or Brave Grounding Context
(``api_mode="grounding"``) results.

This is an *optional* backend — the harness still runs end-to-end on the
native (Brave/Tavily/Serper) provider with no Snowflake account.  It is wired
only when ``web.provider`` is set to ``cortex`` / ``cortex-grounding`` (see
``arcticswarm/tools/factory.py``), analogous to the pluggable corpus retriever.

Auth (see :meth:`CortexWebSearchTool._get_host_and_auth`):
  1. an ``sf_client`` session token (preferred for in-cluster runs), else
  2. an explicit ``api_key`` (PAT) + ``cortex_account`` (``web.cortex_account``
     / settings ``cortex_account`` / ``CORTEX_ACCOUNT``).
No account/host is hardcoded.

API endpoint:
  POST https://{host}/api/v2/cortex/agent:run   (Authorization: per above)
SSE response events:
  response.tool_use    -- echoes the query back
  response.tool_result -- contains search results as JSON content

When the passthrough returns empty, two progressively relaxed query rewrites
are attempted (OR-unquote, then pure-unquote) before falling back to Tavily /
Google Serper (if keys are configured) — identical to the native tool's
fallback ladder, reusing :class:`WebSearchTool`'s static helpers.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TYPE_CHECKING

import requests

from arcticswarm.tools.base import BaseTool, ToolResult
from arcticswarm.tools.web_fetch import WebFetchTool
from arcticswarm.tools.web_search import WebSearchTool

if TYPE_CHECKING:
    from arcticswarm.tools.source_scorer import SourceScorer

logger = logging.getLogger(__name__)

# Model is required by the agent:run schema but bypassed by passthrough.
_PASSTHROUGH_MODEL = "claude-sonnet-4-5"


class CortexWebSearchTool(BaseTool):
    """Search the web via the Cortex agent:run passthrough endpoint."""

    name = "web_search"
    description = (
        "Search the public web using Brave Search (Tavily and Google Serper as fallbacks). "
        "Returns a short list of results (title, url, snippet). "
        "Use this when you need up-to-date info not available in the local codebase."
    )

    # Subclasses override these to switch API mode and source-tag prefix.
    _api_mode: str = "search"
    _source_prefix: str = "cortex"

    # Shared across all instances (same as WebSearchTool).
    _serper_disabled_globally = False

    def __init__(
        self,
        *,
        api_key: str = "",
        cortex_account: str = "",
        sf_client: Any = None,
        tavily_api_key: str = "",
        serper_api_key: str = "",
        judge: "SourceScorer | None" = None,
    ) -> None:
        self._api_key = api_key
        self._cortex_account = cortex_account
        self._sf_client = sf_client
        self._tavily_api_key = (tavily_api_key or "").strip()
        self._serper_api_key = (serper_api_key or "").strip()
        self._judge = judge
        self._serper_disabled = CortexWebSearchTool._serper_disabled_globally
        self._total_searches = 0
        self._cortex_searches = 0
        self._tavily_searches = 0
        self._serper_searches = 0
        self._fallback_log: list[dict[str, Any]] = []
        self._search_log: list[dict[str, Any]] = []

    def _get_host_and_auth(self) -> tuple[str, str]:
        """Return (host, authorization_header) using session token or PAT."""
        # Prefer session token from SnowflakeClient (works on the in-cluster host).
        if self._sf_client is not None:
            try:
                host = self._sf_client._get_rest_url()
                token = self._sf_client._get_token()
                return host, f'Snowflake Token="{token}"'
            except Exception:
                pass  # fall through to PAT
        # Fallback: PAT auth with cortex_account
        if self._api_key and self._cortex_account:
            host = f"{self._cortex_account}.snowflakecomputing.com"
            return host, f"Bearer {self._api_key}"
        raise RuntimeError("No auth available: need either sf_client or api_key+cortex_account")

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
            },
            "required": ["query"],
        }

    # ----- Main entry point ---------------------------------------------------

    def execute(
        self,
        *,
        query: str,
        count: int = 5,
        country: str | None = None,
        safesearch: str = "moderate",
        **_: Any,
    ) -> ToolResult:
        q = (query or "").strip()
        if not q:
            return ToolResult(error="Missing required parameter: query", is_error=True)
        if len(q) > 400:
            return ToolResult(error=f"Query too long ({len(q)} chars). Max 400 characters.", is_error=True)
        word_count = len(q.split())
        if word_count > 50:
            return ToolResult(error=f"Query too long ({word_count} words). Max 50 words.", is_error=True)

        capped = max(1, min(int(count or 5), 20))
        fallback_entry: dict[str, Any] | None = None

        # --- Stage 1: Cortex passthrough (original query) ---
        cortex_results = self._search_cortex(q, capped)
        src = self._source_prefix
        if cortex_results:
            # Score for annotation in a single LLM call (no rejection).
            scores = self._score_results(q, cortex_results)
            self._total_searches += 1
            self._cortex_searches += 1
            self._record_search(q, src, len(cortex_results), scores)
            return WebSearchTool._format_results(cortex_results, q, capped, source=src, scores=scores)

        # --- Stage 1b: OR-unquote retry via Cortex ---
        or_results = self._cortex_or_unquote_retry(q, capped)
        if or_results:
            self._total_searches += 1
            self._cortex_searches += 1
            scores = self._score_results(q, or_results)
            src_or = f"{src}_or_rewrite"
            self._record_search(q, src_or, len(or_results), scores)
            return WebSearchTool._format_results(or_results, q, capped, source=src_or, scores=scores)

        # --- Stage 1c: Pure-unquote retry via Cortex ---
        unq_results = self._cortex_unquote_retry(q, capped)
        if unq_results:
            self._total_searches += 1
            self._cortex_searches += 1
            scores = self._score_results(q, unq_results)
            src_unq = f"{src}_unquote"
            self._record_search(q, src_unq, len(unq_results), scores)
            return WebSearchTool._format_results(unq_results, q, capped, source=src_unq, scores=scores)

        logger.info("Cortex (%s) returned no results for query: %s — trying fallbacks", self._api_mode, q[:80])
        fallback_entry = self._make_fallback_entry(q)

        # --- Stage 2: Tavily fallback ---
        result = self._try_tavily(q, capped, fallback_entry)
        if result is not None:
            return result

        # --- Stage 3: Serper fallback ---
        if self._serper_api_key and not self._serper_disabled:
            serper_results = self._search_serper(q, capped, country)
            if serper_results:
                self._total_searches += 1
                self._serper_searches += 1
                scores = self._score_results(q, serper_results)
                self._record_search(q, "serper", len(serper_results), scores)
                if fallback_entry is not None:
                    fallback_entry["serper_success"] = True
                    fallback_entry["serper_result_count"] = len(serper_results)
                    fallback_entry["final_source"] = "serper"
                    self._fallback_log.append(fallback_entry)
                return WebSearchTool._format_results(serper_results, q, capped, source="serper", scores=scores)

        # All providers failed
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

    # ----- Cortex passthrough API --------------------------------------------

    def _search_cortex(
        self, query: str, count: int,
    ) -> list[dict[str, str]] | None:
        """Call the Cortex agent:run passthrough endpoint for web search."""
        try:
            host, auth_header = self._get_host_and_auth()
        except Exception as exc:
            logger.warning("Cortex web search: no auth available: %s", exc)
            self._fallback_log.append({
                "query": query, "cortex_failed": True, "error": str(exc),
            })
            return None

        payload = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "model": _PASSTHROUGH_MODEL,
            "stream": True,
            "origin_application": "coding_agent",
            "tools": [{"tool_spec": {"type": "web_search", "name": "web_search"}}],
            "tool_choice": {"type": "auto"},
            "tool_resources": {
                "web_search": {
                    "api_mode": self._api_mode,
                    "max_results": count,
                },
            },
            "experimental": {
                "CodingAgent": {"UseWebSearchPassthrough": True},
            },
        }

        try:
            resp = requests.post(
                f"https://{host}/api/v2/cortex/agent:run",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Authorization": auth_header,
                    "User-Agent": "arcticswarm/1.0",
                },
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Cortex web search request failed: %s", exc)
            self._fallback_log.append({
                "query": query, "cortex_failed": True, "error": str(exc),
            })
            return None

        return self._parse_sse_response(resp, query, count)

    def _parse_sse_response(
        self, resp: requests.Response, query: str, count: int,
    ) -> list[dict[str, str]] | None:
        """Parse SSE stream to extract search results from response.tool_result event.

        Reads the full response body and splits by SSE event boundaries
        (double newlines) to avoid issues with iter_lines splitting data
        that contains embedded newlines.
        """
        results: list[dict[str, str]] = []

        try:
            raw_body = resp.text
        except Exception as exc:
            logger.warning("Cortex web search: failed to read response: %s", exc)
            self._fallback_log.append({
                "query": query, "cortex_failed": True,
                "error": f"Failed to read response: {exc}",
            })
            return None
        finally:
            resp.close()

        # Split by double-newline (SSE event boundary)
        blocks = raw_body.split("\n\n")
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            event_type = ""
            data_parts: list[str] = []
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_parts.append(line[len("data:"):])

            if not data_parts:
                continue
            raw_data = "\n".join(data_parts)

            if event_type == "response.tool_result":
                parsed = self._extract_search_results(raw_data)
                if parsed:
                    results.extend(parsed)
            elif event_type == "error":
                logger.warning("Cortex web search SSE error: %s", raw_data[:500])
                self._fallback_log.append({
                    "query": query, "cortex_failed": True,
                    "error": f"SSE error event: {raw_data[:500]}",
                })

        if not results:
            return None
        return results[:count]

    @staticmethod
    def _extract_search_results(raw_data: str) -> list[dict[str, str]] | None:
        """Extract normalized search results from a tool_result SSE data payload."""
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.warning("Cortex web search: failed to parse tool_result JSON: %s", raw_data[:300])
            return None

        content_items = data.get("content", [])
        results: list[dict[str, str]] = []

        for item in content_items:
            if not isinstance(item, dict):
                continue
            # JSON content type: {"type": "json", "json": {"search_results": [...]}}
            json_payload = item.get("json") or item.get("data") or {}
            if isinstance(json_payload, dict):
                search_results = json_payload.get("search_results", [])
                for sr in search_results:
                    if isinstance(sr, dict):
                        results.append({
                            "title": sr.get("doc_title", ""),
                            "url": sr.get("doc_id", ""),
                            "description": sr.get("text", ""),
                        })

        return results or None

    # ----- Cortex quote-relaxation retry -------------------------------------

    def _cortex_or_unquote_retry(
        self, query: str, count: int,
    ) -> list[dict[str, str]] | None:
        """Retry with OR-unquoted rewrite via Cortex passthrough."""
        rewritten = WebSearchTool._build_or_unquote_rewrite(query)
        if not rewritten:
            return None
        logger.info("Cortex OR-unquote retry: %s", rewritten[:120])
        return self._search_cortex(rewritten, count)

    def _cortex_unquote_retry(
        self, query: str, count: int,
    ) -> list[dict[str, str]] | None:
        """Retry with all quotes stripped via Cortex passthrough."""
        stripped = WebSearchTool._build_unquote_only(query)
        if not stripped:
            return None
        logger.info("Cortex unquote retry: %s", stripped[:120])
        return self._search_cortex(stripped, count)

    # ----- Tavily fallback ---------------------------------------------------

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
            fallback_entry["tavily_success"] = True
            fallback_entry["tavily_result_count"] = len(tavily_results)
            fallback_entry["tavily_top_results"] = [
                {"title": r.get("title", ""), "url": r.get("url", "")}
                for r in tavily_results[:3]
            ]
            fallback_entry["final_source"] = "tavily"
            self._fallback_log.append(fallback_entry)
        return WebSearchTool._format_results(tavily_results, q, capped, source="tavily", scores=scores)

    def _search_tavily(self, query: str, count: int) -> list[dict[str, str]] | None:
        """Call Tavily Search API. Reuses the native tool's implementation."""
        return WebSearchTool._search_tavily(self, query, count)

    # ----- Google Serper fallback --------------------------------------------

    def _search_serper(
        self, query: str, count: int, country: str | None,
    ) -> list[dict[str, str]] | None:
        """Call Google Serper API. Reuses the native tool's implementation."""
        return WebSearchTool._search_serper(self, query, count, country)

    # ----- Source quality scoring --------------------------------------------

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

    # ----- Fallback & search logging -----------------------------------------

    def _make_fallback_entry(self, query: str) -> dict[str, Any]:
        """Create a fallback-tracking dict after all Cortex stages fail."""
        has_multi_quotes = len(re.findall(r'"[^"]+"', query)) >= 2
        or_eligible = has_multi_quotes and " OR " not in query
        unquote_eligible = '"' in query
        return {
            "query": query,
            "cortex_failed": True,
            "cortex_or_rewrite_eligible": or_eligible,
            "cortex_or_rewrite_failed": or_eligible,
            "cortex_unquote_eligible": unquote_eligible,
            "cortex_unquote_failed": unquote_eligible,
            "tavily_success": False,
            "tavily_result_count": 0,
            "tavily_top_results": [],
            "serper_success": False,
            "serper_result_count": 0,
            "final_source": None,
            "query_features": WebSearchTool._query_features(query),
        }

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
            "query_features": WebSearchTool._query_features(query),
        }
        score_summary = summarize_search_scores(scores)
        if score_summary:
            entry["scores"] = score_summary
        self._search_log.append(entry)

    # ----- Stats (duck-type compat with WebSearchTool) -----------------------

    def log_and_reset_stats(self) -> None:
        """Log web search stats and reset counters."""
        if self._total_searches > 0:
            logger.info(
                "Web search stats: %d total (cortex=%d, tavily=%d, serper=%d)",
                self._total_searches,
                self._cortex_searches,
                self._tavily_searches,
                self._serper_searches,
            )
        self._total_searches = 0
        self._cortex_searches = 0
        self._tavily_searches = 0
        self._serper_searches = 0

    def drain_fallback_log(self) -> list[dict[str, Any]]:
        """Return and clear accumulated fallback events."""
        log = list(self._fallback_log)
        self._fallback_log.clear()
        return log

    def drain_search_log(self) -> list[dict[str, Any]]:
        """Return and clear the full per-query search log (all providers)."""
        log = list(self._search_log)
        self._search_log.clear()
        return log


class CortexGroundingSearchTool(CortexWebSearchTool):
    """Search the web via the Cortex Brave Grounding Context API.

    Subclass of :class:`CortexWebSearchTool` that uses the ``grounding`` API
    mode instead of ``search``.  All retry, fallback, and logging behaviour is
    inherited.  Selected with ``web.provider = cortex-grounding``.
    """

    _api_mode: str = "grounding"
    _source_prefix: str = "cortex_grounding"


class CortexGroundingFetchTool(WebFetchTool):
    """``web_fetch`` with Cortex Grounding as tier 0.

    Subclass of :class:`WebFetchTool` that prepends a Cortex Grounding Context
    API call before the standard Jina -> Serper -> requests chain.  Selected
    with ``web.fetch_backend = cortex-grounding``.  If grounding fails or
    returns empty content, the normal chain runs as fallback.  Grounding
    attempts/successes are tracked via the shared :class:`WebFetchInstrumentor`.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        cortex_account: str = "",
        sf_client: Any = None,
        jina_api_key: str = "",
        serper_api_key: str = "",
        no_js: bool = False,
        content_cache: Any | None = None,
        source_scorer_enabled: bool = True,
        fetch_compactor_enabled: bool = False,
    ) -> None:
        super().__init__(
            jina_api_key=jina_api_key,
            serper_api_key=serper_api_key,
            no_js=no_js,
            content_cache=content_cache,
            source_scorer_enabled=source_scorer_enabled,
            fetch_compactor_enabled=fetch_compactor_enabled,
        )
        self._api_key = api_key
        self._cortex_account = cortex_account
        self._sf_client = sf_client

    # ----- Auth (same as CortexWebSearchTool) --------------------------------

    def _get_host_and_auth(self) -> tuple[str, str]:
        """Return (host, authorization_header) using session token or PAT."""
        if self._sf_client is not None:
            try:
                host = self._sf_client._get_rest_url()
                token = self._sf_client._get_token()
                return host, f'Snowflake Token="{token}"'
            except Exception:
                pass
        if self._api_key and self._cortex_account:
            host = f"{self._cortex_account}.snowflakecomputing.com"
            return host, f"Bearer {self._api_key}"
        raise RuntimeError("No auth available: need either sf_client or api_key+cortex_account")

    # ----- Tier 0: Cortex Grounding Context fetch ----------------------------

    def _fetch_grounding(self, url: str) -> str | None:
        """Fetch URL content via Cortex Grounding Context API.

        Returns extracted markdown content or None on failure.
        """
        try:
            host, auth_header = self._get_host_and_auth()
        except Exception as exc:
            logger.warning("Cortex grounding fetch: no auth available: %s", exc)
            return None

        payload = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": url}]}],
            "model": _PASSTHROUGH_MODEL,
            "stream": True,
            "origin_application": "coding_agent",
            "tools": [{"tool_spec": {"type": "web_search", "name": "web_search"}}],
            "tool_choice": {"type": "auto"},
            "tool_resources": {
                "web_search": {
                    "api_mode": "grounding",
                    "max_results": 1,
                },
            },
            "experimental": {
                "CodingAgent": {"UseWebSearchPassthrough": True},
            },
        }

        try:
            resp = requests.post(
                f"https://{host}/api/v2/cortex/agent:run",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Authorization": auth_header,
                    "User-Agent": "arcticswarm/1.0",
                },
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Cortex grounding fetch request failed: %s", exc)
            return None

        # Parse SSE response for grounding context content
        try:
            raw_body = resp.text
        except Exception:
            return None
        finally:
            resp.close()

        blocks = raw_body.split("\n\n")
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            event_type = ""
            data_parts: list[str] = []
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_parts.append(line[len("data:"):])

            if not data_parts:
                continue
            raw_data = "\n".join(data_parts)

            if event_type == "response.tool_result":
                content = self._extract_grounding_content(raw_data)
                if content:
                    return content

        return None

    @staticmethod
    def _extract_grounding_content(raw_data: str) -> str | None:
        """Extract text content from a grounding tool_result payload."""
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return None

        content_items = data.get("content", [])
        texts: list[str] = []
        for item in content_items:
            if not isinstance(item, dict):
                continue
            # Text content
            if item.get("type") == "text":
                text = item.get("text", "").strip()
                if text:
                    texts.append(text)
            # JSON content with search_results
            json_payload = item.get("json") or item.get("data") or {}
            if isinstance(json_payload, dict):
                for sr in json_payload.get("search_results", []):
                    if isinstance(sr, dict):
                        text = sr.get("text", "").strip()
                        if text:
                            texts.append(text)

        return "\n\n".join(texts) if texts else None

    # ----- Override execute to prepend grounding tier ------------------------

    def execute(self, *, url: str, **kwargs: Any) -> ToolResult:
        url = (url or "").strip()
        if not url:
            return ToolResult(error="Missing required parameter: url", is_error=True)

        t0 = time.monotonic()

        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        if url.startswith("http://"):
            url = "https://" + url[7:]

        # Tier 0: Cortex Grounding
        self._instr.grounding_attempts += 1
        content = self._fetch_grounding(url)
        if content and content.strip():
            self._instr.total_fetches += 1
            self._instr.grounding_success += 1
            self._instr.record_fetch(
                url, "cortex_grounding", True, None,
                (time.monotonic() - t0) * 1000, len(content),
            )
            return ToolResult(
                output=content,
                metadata={"url": url, "via": "Cortex Grounding"},
            )

        logger.info("Cortex grounding returned no content for %s — falling back to native chain", url[:80])

        # Fall back to the standard WebFetchTool chain
        return super().execute(url=url, **kwargs)
