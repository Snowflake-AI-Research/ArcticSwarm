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

"""WebFetch tool — fetch and extract content from web URLs.

Extraction chain (mirrors OpenJiuwen DeepAgent's smart_request.py):
  1. Jina Reader API (primary — browser-rendered, JS-aware extraction)
  2. Serper MCP scrape (fallback — Google-cached page scrape)
  3. requests + MarkItDown (final fallback — raw HTTP + markdown conversion)

Retries the full chain up to 3 times with exponential backoff (4^n seconds).
PDF URLs are auto-detected via Content-Type / magic bytes and delegated to PdfReadTool.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
import warnings
from typing import Any
from urllib.parse import urlsplit

import requests

from arcticswarm.logging_utils import WebFetchInstrumentor
from arcticswarm.tools._security_blocklist import check_blocked
from arcticswarm.tools.base import BaseTool, ToolResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching DeepAgent)
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_JINA_TIMEOUT = 120          # seconds — first attempt
_JINA_EXTENDED_TIMEOUT = 300  # seconds — retry when page not fully loaded
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2       # sleep = 2 * retry_count seconds (2s, 4s, 6s)

# Domains that return garbage or should not be scraped
_RESTRICTED_DOMAINS = {
    "huggingface.co/datasets": (
        "You are trying to scrape a Hugging Face dataset. "
        "Please do not use the scrape tool for this purpose."
    ),
    "huggingface.co/spaces": (
        "You are trying to scrape a Hugging Face Space. "
        "Please do not use the scrape tool for this purpose."
    ),
    "arxiv.org/src": (
        "You are scraping arXiv source files (LaTeX), which are not useful. "
        "Try fetching the abstract page or PDF instead."
    ),
}

# Non-HTML document/data extensions that Jina's browser engine cannot render
# (it returns HTTP 422 "URL may be a file").  These are routed to the
# requests+MarkItDown path first, which handles office docs and raw text.
_DOC_EXTS = {
    "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "csv", "tsv", "json", "xml", "txt", "rtf", "epub", "gz", "zip",
}


def _file_ext(url: str) -> str | None:
    """Return the lowercase file extension of the URL path, or None."""
    try:
        path = urlsplit(url).path.lower()
    except Exception:
        return None
    m = re.search(r"\.([a-z0-9]{1,6})$", path)
    return m.group(1) if m else None


def _is_wayback(url: str) -> bool:
    """True for Internet Archive Wayback URLs (Jina 422s on these heavily)."""
    return "web.archive.org/web/" in url or "web.archive.org/cdx" in url


def _wayback_raw(url: str) -> str:
    """Rewrite a full-timestamp Wayback URL to raw mode (``id_`` suffix).

    ``/web/20230728192724/https://x`` → ``/web/20230728192724id_/https://x``
    Raw mode strips the Wayback toolbar/chrome and returns the original
    archived resource, which the requests path converts far more cleanly.
    Non-matching URLs (e.g. ``/web/2023/…`` or ``/cdx``) are returned as-is.
    """
    return re.sub(r"(/web/\d{14})/", r"\1id_/", url, count=1)


class WebFetchTool(BaseTool):
    """Fetch a web page and return its content as markdown."""

    name = "web_fetch"
    description = (
        "Fetch content from a URL and return it as clean markdown text. "
        "Handles HTML pages and PDF documents. Use after web_search to "
        "read the full content of a page."
    )

    # Class-level flag — shared across all instances (swarm subagents)
    _serper_scrape_disabled_globally = False

    def __init__(
        self, jina_api_key: str = "", serper_api_key: str = "",
        no_js: bool = False,
        content_cache: Any | None = None,
        source_scorer_enabled: bool = True,
        fetch_compactor_enabled: bool = False,
    ) -> None:
        self._jina_api_key = (jina_api_key or "").strip()
        self._serper_api_key = (serper_api_key or "").strip()
        self._no_js = no_js
        self._serper_scrape_disabled = WebFetchTool._serper_scrape_disabled_globally

        # Content cache (shared cross-agent, disk-backed)
        self._content_cache = content_cache
        self._source_scorer_enabled = source_scorer_enabled
        # When the agent post-processes fetches with the chunking compactor,
        # do NOT head-truncate cached content here — the compactor needs the
        # full page to pick relevant chunks.
        self._fetch_compactor_enabled = fetch_compactor_enabled
        # Same-agent dedup: tracks URLs this tool instance has fetched
        self._fetched_urls: set[str] = set()

        # Instrumentation (counters + per-fetch log live in logging_utils)
        self._instr = WebFetchInstrumentor()

    # -- Instrumentation delegates -----------------------------------------

    def log_and_reset_stats(self) -> None:
        """Log web fetch stats and reset counters."""
        self._instr.log_and_reset_stats()

    def drain_fetch_log(self) -> list[dict[str, Any]]:
        """Return and clear the per-fetch log."""
        return self._instr.drain_fetch_log()

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from.",
                },
            },
            "required": ["url"],
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, *, url: str, **_: Any) -> ToolResult:
        url = (url or "").strip()
        if not url:
            return ToolResult(error="Missing required parameter: url", is_error=True)

        t0 = time.monotonic()

        # Auto-add https://
        protocol_note = ""
        if not url.startswith(("http://", "https://")):
            original = url
            url = f"https://{url}"
            protocol_note = (
                f"[NOTE]: Automatically added 'https://' to URL "
                f"'{original}' -> '{url}'\n\n"
            )

        # Upgrade http → https
        if url.startswith("http://"):
            url = "https://" + url[7:]

        # Security blocklist: reject domains flagged by Threat Intel before
        # any DNS / HTTP activity (HEAD, Jina, Serper, requests).
        block_reason = check_blocked(url)
        if block_reason:
            log.warning("web_fetch blocked URL %s — %s", url, block_reason)
            return ToolResult(error=block_reason, is_error=True)

        # Check restricted domains
        for domain_pattern, message in _RESTRICTED_DOMAINS.items():
            if domain_pattern in url:
                return ToolResult(error=message, is_error=True)

        # --- Content cache lookup ---
        from arcticswarm.tools.content_cache import _normalize_url
        norm_url = _normalize_url(url)

        # Same-agent duplicate: content is already in this agent's context
        if norm_url in self._fetched_urls:
            # Still a tool call — count it (+ log as a cache hit) so cached
            # fetches aren't undercounted. Internal only; never shown to model.
            self._instr.total_fetches += 1
            self._instr.cache_hits += 1
            self._instr.record_fetch(url, "cache_same_agent", True, None, (time.monotonic() - t0) * 1000, 0)
            return ToolResult(
                output=(
                    "[NOTE: This URL was already fetched earlier in this conversation. "
                    "The content is already in your context above. "
                    "Please review your earlier results rather than re-fetching.]"
                ),
                metadata={"url": url, "cache_hit": "same_agent"},
            )

        # Cross-agent cache hit: content exists from another agent
        if self._content_cache is not None:
            entry = self._content_cache.get(norm_url)
            if entry is not None:
                self._fetched_urls.add(norm_url)
                if entry.is_error:
                    # Another agent already tried and failed — guide to alternatives
                    self._instr.total_fetches += 1
                    self._instr.cache_hits += 1
                    self._instr.record_fetch(url, "cache", False, "cached failure", (time.monotonic() - t0) * 1000, 0)
                    return ToolResult(
                        output=(
                            "[NOTE: This URL was already attempted by another researcher "
                            "and failed to load. Error: " + entry.content + "\n\n"
                            "Please search for alternative sources rather than retrying this URL.]"
                        ),
                        metadata={"url": url, "cache_hit": "cross_agent_failure"},
                    )
                content = entry.content
                # Transparent cache hit: return cached content EXACTLY as a live
                # fetch would (output = protocol_note + full content, identical
                # metadata shape) so the agent model cannot tell a cache hit
                # from a real fetch. No "retrieved from cache" prefix and no
                # cache-only truncation — both would leak the cache to the model
                # and (for truncation) make hits shorter than live fetches. The
                # downstream source-scorer / compactor then process it the same
                # way they process a live fetch. ``cache_hit`` stays in metadata
                # for instrumentation only (never shown to the model).
                self._instr.total_fetches += 1
                self._instr.cache_hits += 1
                self._instr.record_fetch(url, "cache", True, None, (time.monotonic() - t0) * 1000, len(content))
                return ToolResult(
                    output=protocol_note + content,
                    metadata={"url": url, "via": entry.via or "cache", "cache_hit": "content_cache"},
                )

        # --- File-aware pre-routing -------------------------------------
        # Jina's browser engine returns 422 ("URL may be a file") on ~1k
        # URLs/run, dominated by web.archive.org (~29%) and non-HTML files,
        # and slowly *renders* many .pdf URLs that HEAD content-type missed.
        # Route those URLs to a better extractor BEFORE the Jina-first chain.
        # Every route falls through to the normal chain on failure, so this
        # never strictly regresses.
        ext = _file_ext(url)
        is_wb = _is_wayback(url)
        looks_pdf = (ext == "pdf")

        # HEAD probe only for unknown/extensionless, non-Wayback URLs — catches
        # PDFs served without a .pdf extension (preserves prior behaviour)
        # without adding a HEAD round-trip to URLs we can already classify.
        if ext is None and not is_wb:
            try:
                head_resp = requests.head(
                    url, timeout=10,
                    headers={"User-Agent": _USER_AGENT}, allow_redirects=True,
                )
                if "application/pdf" in head_resp.headers.get("Content-Type", "").lower():
                    looks_pdf = True
            except Exception:
                pass  # HEAD failed — proceed with normal fetch chain

        if looks_pdf:
            result = self._fetch_pdf(url)
            if not result.is_error:
                self._instr.total_fetches += 1
                self._instr.record_fetch(url, "pdf", True, None, (time.monotonic() - t0) * 1000, len(result.output or ""))
                self._fetched_urls.add(norm_url)
                if self._content_cache is not None:
                    self._content_cache.put(norm_url, result.output or "", is_pdf=True, via="pdf")
                return result
            # Not actually a usable PDF — fall through to the normal chain.
            log.info("PDF route failed for %s (%s) — falling back to fetch chain", url, (result.error or "")[:80])
        elif ext in _DOC_EXTS or is_wb:
            target = _wayback_raw(url) if is_wb else url
            content, _err = self._scrape_request(target)
            if content and content.strip():
                if content.strip().startswith("%PDF-"):
                    result = self._fetch_pdf(url)
                    if not result.is_error:
                        self._instr.total_fetches += 1
                        self._instr.record_fetch(url, "pdf", True, None, (time.monotonic() - t0) * 1000, len(result.output or ""))
                        self._fetched_urls.add(norm_url)
                        if self._content_cache is not None:
                            self._content_cache.put(norm_url, result.output or "", is_pdf=True, via="pdf")
                        return result
                else:
                    self._instr.total_fetches += 1
                    self._instr.requests_success += 1
                    self._instr.record_fetch(url, "requests", True, None, (time.monotonic() - t0) * 1000, len(content))
                    self._fetched_urls.add(norm_url)
                    if self._content_cache is not None:
                        self._content_cache.put(norm_url, content, via="requests")
                    via = "requests+MarkItDown (Wayback route)" if is_wb else "requests+MarkItDown (file route)"
                    return ToolResult(output=protocol_note + content, metadata={"url": url, "via": via})
            # Route came up empty — fall through to the normal chain.
            log.info("File/Wayback route empty for %s — falling back to fetch chain", url)

        # --- 3-tier fallback chain with retry ---
        retry_count = 0
        while retry_count < _MAX_RETRIES:
            try:
                error_log = ""

                # Tier 1: Jina Reader API
                content, err = self._scrape_jina(url)
                if err:
                    error_log += f"[Jina] {err}\n"
                elif content and content.strip():
                    self._instr.total_fetches += 1
                    self._instr.jina_success += 1
                    self._instr.record_fetch(url, "jina", True, None, (time.monotonic() - t0) * 1000, len(content))
                    self._fetched_urls.add(norm_url)
                    if self._content_cache is not None:
                        self._content_cache.put(norm_url, content, via="jina")
                    return ToolResult(
                        output=protocol_note + content,
                        metadata={"url": url, "via": "Jina Reader"},
                    )

                # Tier 2: Serper MCP scrape
                content, err = self._scrape_serper(url)
                if err:
                    error_log += f"[Serper] {err}\n"
                elif content and content.strip():
                    self._instr.total_fetches += 1
                    self._instr.serper_success += 1
                    self._instr.record_fetch(url, "serper", True, None, (time.monotonic() - t0) * 1000, len(content))
                    self._fetched_urls.add(norm_url)
                    if self._content_cache is not None:
                        self._content_cache.put(norm_url, content, via="serper")
                    return ToolResult(
                        output=protocol_note + content,
                        metadata={"url": url, "via": "Serper scrape"},
                    )

                # Tier 3: requests + MarkItDown
                content, err = self._scrape_request(url)
                if err:
                    error_log += f"[requests] {err}\n"
                elif content and content.strip():
                    # Check if this is actually a PDF (magic bytes)
                    if content.strip().startswith("%PDF-"):
                        self._instr.total_fetches += 1
                        self._instr.record_fetch(url, "pdf", True, None, (time.monotonic() - t0) * 1000, 0)
                        result = self._fetch_pdf(url)
                        if not result.is_error:
                            self._fetched_urls.add(norm_url)
                            if self._content_cache is not None:
                                self._content_cache.put(
                                    norm_url, result.output or "",
                                    is_pdf=True, via="pdf",
                                )
                        return result
                    self._instr.total_fetches += 1
                    self._instr.requests_success += 1
                    self._instr.record_fetch(url, "requests", True, None, (time.monotonic() - t0) * 1000, len(content))
                    self._fetched_urls.add(norm_url)
                    if self._content_cache is not None:
                        self._content_cache.put(norm_url, content, via="requests")
                    return ToolResult(
                        output=protocol_note + content,
                        metadata={"url": url, "via": "requests+MarkItDown"},
                    )

                # All tiers failed this attempt
                raise RuntimeError(error_log)

            except Exception as exc:
                retry_count += 1
                if retry_count >= _MAX_RETRIES:
                    self._instr.total_fetches += 1
                    self._instr.total_failures += 1
                    self._instr.record_fetch(url, "none", False, str(exc)[:200], (time.monotonic() - t0) * 1000, 0)
                    # Cache the failure so other agents don't retry
                    self._fetched_urls.add(norm_url)
                    if self._content_cache is not None:
                        self._content_cache.put_failure(
                            norm_url,
                            f"All extraction methods failed after {_MAX_RETRIES} attempts.",
                        )
                    return ToolResult(
                        error=(
                            f"All extraction methods failed for {url} "
                            f"after {_MAX_RETRIES} attempts.\n{exc}"
                        ),
                        is_error=True,
                    )
                backoff = _RETRY_BACKOFF_BASE * retry_count
                log.info(
                    "Fetch attempt %d/%d failed for %s, retrying in %ds",
                    retry_count, _MAX_RETRIES, url, backoff,
                )
                time.sleep(backoff)

        # Should not reach here, but safety net
        self._instr.total_fetches += 1
        self._instr.total_failures += 1
        self._instr.record_fetch(url, "none", False, "safety_net", (time.monotonic() - t0) * 1000, 0)
        return ToolResult(
            error=f"Failed to fetch content from {url}", is_error=True
        )

    # ------------------------------------------------------------------
    # Tier 1: Jina Reader API
    # ------------------------------------------------------------------

    def _scrape_jina(self, url: str) -> tuple[str | None, str | None]:
        """Fetch via Jina Reader API with enhanced headers."""
        if not self._jina_api_key:
            return None, "JINA_API_KEY not set."

        jina_headers = {
            "Authorization": f"Bearer {self._jina_api_key}",
            "X-Base": "final",
            "X-With-Generated-Alt": "true",
        }
        if self._no_js:
            jina_headers["X-Engine"] = "direct"
        else:
            jina_headers["X-Engine"] = "browser"
            jina_headers["X-With-Iframe"] = "true"
            jina_headers["X-With-Shadow-Dom"] = "true"
        jina_url = f"https://r.jina.ai/{url}"

        try:
            resp = requests.get(jina_url, headers=jina_headers, timeout=_JINA_TIMEOUT)

            # 422 typically means the URL is a file (not supported by Jina)
            if resp.status_code == 422:
                return None, (
                    "Jina returned 422 — URL may be a file. "
                    "Falling back to other methods."
                )

            resp.raise_for_status()
            content = resp.text

            # If page not fully loaded, retry with longer timeout
            if "Warning: This page maybe not yet fully loaded" in content:
                log.info("Jina partial load for %s — retrying with extended timeout", url)
                resp = requests.get(
                    jina_url, headers=jina_headers, timeout=_JINA_EXTENDED_TIMEOUT
                )
                if resp.status_code == 422:
                    return None, "Jina returned 422 on extended retry."
                resp.raise_for_status()
                content = resp.text

            if not content or not content.strip():
                return None, "Jina returned empty content."

            return content, None

        except Exception as exc:
            return None, f"Jina error: {exc}"

    # ------------------------------------------------------------------
    # Tier 2: Serper MCP scrape
    # ------------------------------------------------------------------

    def _scrape_serper(self, url: str) -> tuple[str | None, str | None]:
        """Scrape via Serper MCP server (npx serper-search-scrape-mcp-server)."""
        if not self._serper_api_key:
            return None, "SERPER_API_KEY not set."
        if self._serper_scrape_disabled:
            return None, "Serper scrape disabled (credit/quota exhaustion)."

        coro = self._scrape_serper_async(url)
        try:
            content = _run_async(coro)
            if not content or not content.strip():
                return None, "Serper returned empty content."
            return content, None
        except Exception as exc:
            coro.close()  # prevent "coroutine was never awaited" warning
            error_msg = str(exc)
            if any(kw in error_msg.lower() for kw in ("credit", "quota", "limit")):
                if not WebFetchTool._serper_scrape_disabled_globally:
                    log.warning("Serper scrape disabled — %s", error_msg[:200])
                self._serper_scrape_disabled = True
                WebFetchTool._serper_scrape_disabled_globally = True
            return None, f"Serper scrape error: {exc}"

    async def _scrape_serper_async(self, url: str) -> str:
        """Async implementation of Serper MCP scrape."""
        import os
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "serper-search-scrape-mcp-server"],
            env={"SERPER_API_KEY": self._serper_api_key},
        )
        # Suppress MCP server stderr (e.g. "Not enough credits" spam)
        devnull = open(os.devnull, "w")
        try:
            async with stdio_client(server_params, errlog=devnull) as (read, write):
                async with ClientSession(read, write, sampling_callback=None) as session:
                    await session.initialize()
                    result = await session.call_tool("scrape", arguments={"url": url})
                    return result.content[-1].text if result.content else ""
        finally:
            devnull.close()

    # ------------------------------------------------------------------
    # Tier 3: requests + MarkItDown
    # ------------------------------------------------------------------

    def _scrape_request(self, url: str) -> tuple[str | None, str | None]:
        """Fetch with requests and convert via MarkItDown."""
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=60,
            )
            resp.raise_for_status()

            # Try MarkItDown conversion first
            try:
                from markitdown import MarkItDown

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    stream = io.BytesIO(resp.content)
                    md = MarkItDown()
                    content = md.convert_stream(stream).text_content
                if content and content.strip():
                    return content, None
            except Exception:
                pass  # MarkItDown failed — fall back to raw text

            # Raw text fallback
            if resp.text and resp.text.strip():
                return resp.text, None

            return None, "requests returned empty content."
        except Exception as exc:
            return None, f"requests error: {exc}"

    # ------------------------------------------------------------------
    # PDF delegation
    # ------------------------------------------------------------------

    def _fetch_pdf(self, url: str) -> ToolResult:
        """Delegate PDF URL to PdfReadTool."""
        try:
            resp = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": _USER_AGENT},
                allow_redirects=True,
            )
            resp.raise_for_status()

            from arcticswarm.tools.pdf_read import PdfReadTool

            return PdfReadTool.extract_from_bytes(resp.content)
        except Exception as exc:
            return ToolResult(
                error=f"Failed to fetch PDF from {url}: {exc}", is_error=True
            )


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from synchronous code.

    Handles the case where an event loop is already running (e.g. in Jupyter
    or inside an async eval runner) by creating a new thread.  Explicitly
    manages event loop lifecycle to avoid ``_ssock`` AttributeError on GC
    in Python 3.11 when loops are created inside thread pool workers.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — run in a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_fresh_loop, coro)
            return future.result(timeout=120)
    else:
        return _run_in_fresh_loop(coro)


def _run_in_fresh_loop(coro):
    """Create a new event loop, run the coroutine, and close cleanly."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            # Shut down async generators and executors before closing
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass
        finally:
            asyncio.set_event_loop(None)
            try:
                loop.close()
            except Exception:
                pass
