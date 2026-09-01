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

"""PdfRead tool — extract text from PDF files (local or URL).

Extraction chain (ordered by speed and quality):
  0. pypdf (fast path — pure Python, ~1s, no crash risk)
  1. opendataloader-pdf (primary — hybrid mode with docling-fast backend,
     routes complex pages to AI for +90% table accuracy)
  2. Jina Reader API (final fallback for URL sources — handles 403s and CAPTCHAs)

Quality detection: if extracted text has abnormally long "words" (no spaces),
the text is likely garbled from a scanned PDF and we escalate to the next tier.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

from arcticswarm.tools._security_blocklist import check_blocked
from arcticswarm.tools.base import BaseTool, ToolResult

log = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_DOWNLOAD_TIMEOUT = 60
_JINA_TIMEOUT = 120
_JINA_EXTENDED_TIMEOUT = 300
_MAX_DOWNLOAD_RETRIES = 3
_RETRY_BACKOFF_BASE = 4  # 4^retry_count seconds

# Quality gating thresholds
_MIN_CHARS_PER_PAGE = 20  # if avg chars < this, try next tier
_GARBLED_WORD_LENGTH = 50  # max word length before declaring garbled

# Wall-clock timeout for opendataloader-pdf hybrid convert() call.
# If hybrid mode exceeds this, we fall back to Java-only mode.
_ODL_HYBRID_FALLBACK_TIMEOUT = 300  # seconds

# ── Thread-safe stderr suppression for opendataloader-pdf ──────────────
# Two layers are needed:
#   1. fd-level: redirect fd 2 → /dev/null (catches JVM subprocess output)
#   2. Python-level: wrap sys.stderr with a filter (catches the library's
#      print() calls which go through Rich/custom stderr wrappers that
#      bypass fd 2)
# Reference counting ensures suppression stays active until ALL concurrent
# convert() calls finish.
_stderr_lock = threading.Lock()
_stderr_refcount = 0
_stderr_saved_fd: int | None = None
_stderr_saved_obj: Any = None

# Patterns printed by opendataloader-pdf's runner.py on conversion failure
_ODL_NOISE = ("Error running opendataloader-pdf", "Return code:")


class _FilteredStderr:
    """Wraps the real sys.stderr and silently drops known ODL noise lines.

    Python's print() calls write() twice: once for the text and once for the
    trailing newline.  We track whether the last write was suppressed so we
    can also swallow the bare ``\\n`` that follows.
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self._suppressed_last = False

    def write(self, s: str) -> int:
        if any(p in s for p in _ODL_NOISE):
            self._suppressed_last = True
            return len(s)
        # Swallow the bare newline that print() emits after a suppressed line
        if self._suppressed_last and s in ("\n", "\r\n"):
            self._suppressed_last = False
            return len(s)
        self._suppressed_last = False
        return self._real.write(s)

    def __getattr__(self, name: str) -> Any:
        # Proxy everything else (flush, fileno, encoding, …) to the real stderr
        return getattr(self._real, name)


def _suppress_stderr() -> None:
    """Redirect fd 2 to /dev/null AND wrap sys.stderr (thread-safe)."""
    global _stderr_refcount, _stderr_saved_fd, _stderr_saved_obj
    with _stderr_lock:
        _stderr_refcount += 1
        if _stderr_refcount == 1:
            # fd-level: catches JVM subprocess output
            _stderr_saved_fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
            # Python-level: catches print(..., file=sys.stderr) via Rich wrappers
            _stderr_saved_obj = sys.stderr
            sys.stderr = _FilteredStderr(_stderr_saved_obj)


def _restore_stderr() -> None:
    """Restore fd 2 and sys.stderr when the last convert() call finishes."""
    global _stderr_refcount, _stderr_saved_fd, _stderr_saved_obj
    with _stderr_lock:
        _stderr_refcount -= 1
        if _stderr_refcount == 0:
            if _stderr_saved_fd is not None:
                os.dup2(_stderr_saved_fd, 2)
                os.close(_stderr_saved_fd)
                _stderr_saved_fd = None
            if _stderr_saved_obj is not None:
                sys.stderr = _stderr_saved_obj
                _stderr_saved_obj = None


def _parse_pages(pages_str: str, total: int) -> list[int]:
    """Parse a page spec like ``"1-5,8,10-12"`` into 0-based indices.

    Handles ranges (``1-5``), single pages (``8``), and mixed.  Pages are
    clamped to ``[0, total)`` and deduplicated in order.
    """
    result: list[int] = []
    seen: set[int] = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if m:
            lo = max(int(m.group(1)) - 1, 0)
            hi = min(int(m.group(2)), total)
            for i in range(lo, hi):
                if i not in seen:
                    result.append(i)
                    seen.add(i)
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < total and idx not in seen:
                result.append(idx)
                seen.add(idx)
    return result


def _is_garbled(text: str) -> bool:
    """Detect garbled text (no spaces between words).

    Returns True if the text contains abnormally long "words" — a sign
    that the PDF text layer lacks word boundaries (common in scanned PDFs).
    Excludes markdown table rows (pipe-separated values) from the check.
    """
    if not text or len(text) < 50:
        return False
    # Filter out markdown table rows which naturally have long "words"
    # like "|5|0.49|0.45|0.49|0.56|0.62|"
    lines = text.split("\n")
    filtered_lines = [line for line in lines if not line.strip().startswith("|")]
    filtered_text = "\n".join(filtered_lines)
    words = filtered_text.split()
    if not words:
        return False
    # Also exclude URLs and markdown links which are naturally long
    words = [w for w in words if not w.startswith(("http://", "https://", "[", "(http"))]
    if not words:
        return False
    max_word = max(len(w) for w in words)
    return max_word > _GARBLED_WORD_LENGTH


def _text_quality_ok(page_texts: dict[int, str], num_pages: int) -> bool:
    """Check if extracted text is sufficient quality to use."""
    combined = "\n".join(page_texts.values())
    if _is_garbled(combined):
        return False
    total_chars = sum(len(t) for t in page_texts.values())
    avg_chars = total_chars / max(num_pages, 1)
    return avg_chars >= _MIN_CHARS_PER_PAGE


class PdfReadTool(BaseTool):
    """Extract text from PDF documents (local files or URLs)."""

    name = "pdf_read"
    description = (
        "Extract text from a PDF document. Accepts a local file path or a URL. "
        "Use the 'pages' parameter to select specific pages (e.g. '1-5', '3,7,10'). "
        "Useful for reading academic papers, reports, or any PDF content."
    )

    def __init__(
        self,
        jina_api_key: str = "",
        serper_api_key: str = "",
        *,
        odl_hybrid: str = "docling-fast",
        odl_hybrid_url: str = "",
        odl_hybrid_timeout: int = 60000,
        odl_hybrid_fallback_timeout: int = _ODL_HYBRID_FALLBACK_TIMEOUT,
        odl_force_ocr: bool = False,
        content_cache: Any = None,
        source_scorer_enabled: bool = True,
        pdf_compactor_enabled: bool = False,
    ) -> None:
        self._jina_api_key = (jina_api_key or "").strip()
        self._serper_api_key = (serper_api_key or "").strip()
        self._failed_urls: dict[str, str] = {}  # url -> error message
        # OpenDataLoader hybrid config
        self._odl_hybrid = odl_hybrid
        self._odl_hybrid_url = odl_hybrid_url  # empty = auto-start server
        self._odl_hybrid_timeout = odl_hybrid_timeout
        self._odl_hybrid_fallback_timeout = odl_hybrid_fallback_timeout
        self._odl_force_ocr = odl_force_ocr
        # Content cache (shared cross-agent, disk-backed)
        self._content_cache = content_cache
        self._source_scorer_enabled = source_scorer_enabled
        # When the agent post-processes PDFs with the chunking compactor,
        # do NOT head-truncate cached content here — the compactor needs
        # the full document to pick relevant chunks.
        self._pdf_compactor_enabled = pdf_compactor_enabled
        # Same-agent dedup: tracks (normalized_url, pages) fetched by this instance
        self._fetched_keys: set[str] = set()

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Local file path or URL to a PDF document."
                    ),
                },
                "pages": {
                    "type": "string",
                    "description": (
                        "Page range to extract, e.g. '1-5', '3,7,10'. "
                        "Default: all pages."
                    ),
                },
            },
            "required": ["source"],
        }

    def execute(
        self,
        *,
        source: str,
        pages: str = "",
        **_: Any,
    ) -> ToolResult:
        source = (source or "").strip()
        if not source:
            return ToolResult(error="Missing required parameter: source", is_error=True)

        # Skip URLs that have already failed in this session
        if source in self._failed_urls:
            return ToolResult(
                error=f"This PDF has already failed: {self._failed_urls[source]}. Try a different source.",
                is_error=True,
            )

        is_url = source.startswith(("http://", "https://"))

        # Security blocklist: reject URLs flagged by Threat Intel before
        # any download or opendataloader-pdf invocation.
        if is_url:
            block_reason = check_blocked(source)
            if block_reason:
                log.warning("pdf_read blocked URL %s — %s", source, block_reason)
                self._failed_urls[source] = block_reason
                return ToolResult(error=block_reason, is_error=True)

        # --- Content cache lookup (URL sources only) ---
        if is_url:
            from arcticswarm.tools.content_cache import _normalize_url
            norm_url = _normalize_url(source)
            cache_key_id = f"{norm_url}|{pages.strip()}"

            # Same-agent duplicate: exact URL+pages already in context
            if cache_key_id in self._fetched_keys:
                return ToolResult(
                    output=(
                        "[NOTE: This PDF (same URL and pages) was already read earlier "
                        "in this conversation. The content is already in your context above. "
                        "Please review your earlier results rather than re-reading.]"
                    ),
                    metadata={"source": source, "cache_hit": "same_agent"},
                )

            # Cross-agent cache hit: exact URL+pages match
            if self._content_cache is not None:
                entry = self._content_cache.get(norm_url, pages=pages)
                if entry is not None:
                    self._fetched_keys.add(cache_key_id)
                    if entry.is_error:
                        # Another agent already tried and failed
                        return ToolResult(
                            output=(
                                "[NOTE: This PDF was already attempted by another researcher "
                                "and failed to load. Error: " + entry.content + "\n\n"
                                "Please search for alternative sources rather than retrying this PDF.]"
                            ),
                            metadata={"source": source, "cache_hit": "cross_agent_failure"},
                        )
                    content = entry.content
                    # Transparent cache hit: return cached text exactly as a
                    # live pdf_read would (no "cached" prefix, no cache-only
                    # truncation) so the model can't tell a hit from a real
                    # read. cache_hit stays in metadata (logging only).
                    return ToolResult(
                        output=content,
                        metadata={"source": source, "via": entry.via or "cache", "cache_hit": "content_cache"},
                    )

                # Shared key space: web_fetch may have fetched this PDF URL
                # (with all pages).  Only applies when pdf_read also requests
                # all pages (no specific pages requested).
                if not pages.strip():
                    wf_entry = self._content_cache.get_any_pages(norm_url)
                    if wf_entry is not None and wf_entry.is_pdf:
                        self._fetched_keys.add(cache_key_id)
                        if wf_entry.is_error:
                            return ToolResult(
                                output=(
                                    "[NOTE: This PDF was already attempted via web_fetch by another "
                                    "researcher and failed to load. Error: " + wf_entry.content + "\n\n"
                                    "Please search for alternative sources rather than retrying this PDF.]"
                                ),
                                metadata={"source": source, "cache_hit": "cross_agent_failure"},
                            )
                        content = wf_entry.content
                        # Transparent cache hit (see note above).
                        return ToolResult(
                            output=content,
                            metadata={"source": source, "via": wf_entry.via or "cache", "cache_hit": "content_cache"},
                        )

        # --- Download with retry (for URLs) ---
        data: bytes | None = None
        download_error: str | None = None

        if is_url:
            for attempt in range(_MAX_DOWNLOAD_RETRIES):
                try:
                    resp = requests.get(
                        source,
                        timeout=_DOWNLOAD_TIMEOUT,
                        headers={"User-Agent": _USER_AGENT},
                        allow_redirects=True,
                    )
                    resp.raise_for_status()
                    data = resp.content
                    break
                except Exception as exc:
                    download_error = str(exc)
                    if attempt < _MAX_DOWNLOAD_RETRIES - 1:
                        time.sleep(_RETRY_BACKOFF_BASE ** (attempt + 1))
        else:
            try:
                with open(source, "rb") as f:
                    data = f.read()
            except Exception as exc:
                return ToolResult(
                    error=f"Failed to read PDF source: {exc}", is_error=True
                )

        # If download failed entirely, try Jina as last resort
        if data is None:
            if is_url:
                result = self._jina_fallback(source, direct_error=download_error or "Unknown error")
                if result.is_error:
                    self._failed_urls[source] = result.error or "download failed"
                elif result.output:
                    # Cache successful Jina fallback result
                    from arcticswarm.tools.content_cache import _normalize_url
                    norm_url = _normalize_url(source)
                    self._fetched_keys.add(f"{norm_url}|{pages.strip()}")
                    if self._content_cache is not None:
                        self._content_cache.put(
                            norm_url, result.output, pages=pages.strip(),
                            is_pdf=True, via="jina",
                        )
                return result
            return ToolResult(error=f"Failed to read PDF source: {download_error}", is_error=True)

        # --- Extract text from bytes ---
        result = self.extract_from_bytes(
            data,
            pages=pages,
            odl_hybrid=self._odl_hybrid,
            odl_hybrid_url=self._resolve_hybrid_url(),
            odl_hybrid_timeout=self._odl_hybrid_timeout,
            odl_hybrid_fallback_timeout=self._odl_hybrid_fallback_timeout,
        )

        # If extraction produced garbled or empty output from a URL, try Jina
        if is_url and result.is_error:
            jina = self._jina_fallback(source)
            if not jina.is_error:
                # Cache successful Jina result
                from arcticswarm.tools.content_cache import _normalize_url
                norm_url = _normalize_url(source)
                self._fetched_keys.add(f"{norm_url}|{pages.strip()}")
                if self._content_cache is not None:
                    self._content_cache.put(
                        norm_url, jina.output or "", pages=pages.strip(),
                        is_pdf=True, via="jina",
                    )
                return jina
        if is_url and result.output and _is_garbled(result.output):
            jina = self._jina_fallback(source)
            if not jina.is_error:
                # Cache successful Jina result
                from arcticswarm.tools.content_cache import _normalize_url
                norm_url = _normalize_url(source)
                self._fetched_keys.add(f"{norm_url}|{pages.strip()}")
                if self._content_cache is not None:
                    self._content_cache.put(
                        norm_url, jina.output or "", pages=pages.strip(),
                        is_pdf=True, via="jina",
                    )
                return jina

        # Cache URL failures so the agent doesn't retry the same broken PDF
        if is_url and result.is_error:
            self._failed_urls[source] = result.error or "extraction failed"
            # Also cache in the shared cross-agent cache
            from arcticswarm.tools.content_cache import _normalize_url
            norm_url = _normalize_url(source)
            self._fetched_keys.add(f"{norm_url}|{pages.strip()}")
            if self._content_cache is not None:
                self._content_cache.put_failure(
                    norm_url, result.error or "PDF extraction failed",
                    pages=pages.strip(), is_pdf=True,
                )

        # Store successful URL results in the shared content cache
        if is_url and not result.is_error and result.output:
            from arcticswarm.tools.content_cache import _normalize_url
            norm_url = _normalize_url(source)
            cache_key_id = f"{norm_url}|{pages.strip()}"
            self._fetched_keys.add(cache_key_id)
            if self._content_cache is not None:
                self._content_cache.put(
                    norm_url, result.output,
                    pages=pages.strip(),
                    is_pdf=True,
                    via=result.metadata.get("via", "pdf"),
                )

        return result

    @classmethod
    def extract_from_bytes(
        cls,
        data: bytes,
        *,
        pages: str = "",
        odl_hybrid: str = "docling-fast",
        odl_hybrid_url: str = "",
        odl_hybrid_timeout: int = 60000,
        odl_hybrid_fallback_timeout: int = _ODL_HYBRID_FALLBACK_TIMEOUT,
    ) -> ToolResult:
        """Extract text from raw PDF bytes.

        Called directly by :class:`WebFetchTool` when a URL serves a PDF.

        Extraction chain:
        0. pypdf (fast, pure Python) -> 1. opendataloader-pdf (hybrid mode)

        If hybrid mode exceeds *odl_hybrid_fallback_timeout* seconds, it is
        abandoned and retried in Java-only mode (``hybrid="off"``).
        """
        # Fast-fail on non-PDF payloads (HTML error pages, redirects, empty
        # downloads). Spec allows the %PDF- magic anywhere in the first 1024
        # bytes, so check that window. Skipping the call to opendataloader's
        # Java backend here avoids a noisy stack trace and lets the caller
        # fall through to the Jina fallback.
        if not data or b"%PDF-" not in data[:1024]:
            preview = data[:80].decode("utf-8", errors="replace") if data else ""
            return ToolResult(
                error=(
                    f"Not a valid PDF (missing %PDF- header in first 1024 bytes; "
                    f"got {len(data)} bytes; preview: {preview!r})"
                ),
                is_error=True,
            )

        # Resolve hybrid URL — auto-start server if needed
        if odl_hybrid and odl_hybrid != "off" and not odl_hybrid_url:
            try:
                from arcticswarm.tools.odl_server import ensure_server
                odl_hybrid_url = ensure_server()
            except Exception as exc:
                log.warning("Could not start hybrid backend: %s — falling back to Java-only", exc)
                odl_hybrid = "off"
        # Determine page count using pypdf (lightweight)
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(data))
            total = len(reader.pages)
        except Exception:
            total = 0
            reader = None

        if total == 0:
            # Can't even parse page count — try opendataloader as hail mary
            hail_mary_text: str | None = None
            if odl_hybrid and odl_hybrid != "off":
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        cls._extract_opendataloader,
                        data, pages_str="",
                        hybrid=odl_hybrid,
                        hybrid_url=odl_hybrid_url,
                        hybrid_timeout=odl_hybrid_timeout,
                    )
                    try:
                        hail_mary_text = fut.result(timeout=odl_hybrid_fallback_timeout)
                    except concurrent.futures.TimeoutError:
                        log.warning("Hybrid hail-mary timed out — trying Java-only")
                if not (hail_mary_text and hail_mary_text.strip()):
                    hail_mary_text = cls._extract_opendataloader(
                        data, pages_str="",
                        hybrid="off", hybrid_url="",
                        hybrid_timeout=odl_hybrid_timeout,
                    )
            else:
                hail_mary_text = cls._extract_opendataloader(
                    data, pages_str="",
                    hybrid=odl_hybrid,
                    hybrid_url=odl_hybrid_url,
                    hybrid_timeout=odl_hybrid_timeout,
                )
            if hail_mary_text and hail_mary_text.strip():
                return ToolResult(
                    output=hail_mary_text,
                    metadata={"via": "opendataloader-pdf"},
                )
            return ToolResult(error="Failed to parse PDF (no pages detected).", is_error=True)

        indices = _parse_pages(pages, total) if pages.strip() else list(range(total))
        if not indices:
            return ToolResult(
                error=f"No valid pages in '{pages}' (PDF has {total} pages).",
                is_error=True,
            )

        # --- Extract metadata (pypdf) ----------------------------------------
        meta_lines: list[str] = []
        if reader and reader.metadata:
            for key, label in [("title", "Title"), ("author", "Author"),
                               ("subject", "Subject"), ("creator", "Creator")]:
                val = getattr(reader.metadata, key, None)
                if val:
                    meta_lines.append(f"{label}: {val}")

        # --- Tier 0: pypdf fast path (pure Python, no crash risk) -------------
        if reader is not None:
            page_texts = cls._extract_pypdf(reader, indices)
            if page_texts and _text_quality_ok(page_texts, len(indices)):
                log.info("pypdf fast-path extraction succeeded (%d chars)",
                         sum(len(t) for t in page_texts.values()))
                return cls._format_output(page_texts, indices, total, meta_lines, via="pypdf")

        # --- Tier 1: opendataloader-pdf (hybrid mode) -------------------------
        # Build 1-based page spec for opendataloader from 0-based indices
        odl_pages = ",".join(str(i + 1) for i in indices)

        is_hybrid = odl_hybrid and odl_hybrid != "off"
        odl_text: str | None = None

        if is_hybrid:
            # Run hybrid extraction in a thread with a timeout so we can
            # fall back to Java-only mode if it hangs.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    cls._extract_opendataloader,
                    data, pages_str=odl_pages,
                    hybrid=odl_hybrid,
                    hybrid_url=odl_hybrid_url,
                    hybrid_timeout=odl_hybrid_timeout,
                )
                try:
                    odl_text = fut.result(timeout=odl_hybrid_fallback_timeout)
                except concurrent.futures.TimeoutError:
                    log.warning(
                        "opendataloader-pdf hybrid timed out after %ds — "
                        "falling back to Java-only mode",
                        odl_hybrid_fallback_timeout,
                    )
                    # Don't cancel — let it finish in the background to avoid
                    # orphaned temp files; just ignore its result.
                    odl_text = None

            # Hybrid timed out or returned nothing usable — retry Java-only
            if odl_text is None or not odl_text.strip() or _is_garbled(odl_text):
                log.info("Retrying opendataloader-pdf in Java-only mode")
                odl_text = cls._extract_opendataloader(
                    data, pages_str=odl_pages,
                    hybrid="off",
                    hybrid_url="",
                    hybrid_timeout=odl_hybrid_timeout,
                )
        else:
            odl_text = cls._extract_opendataloader(
                data, pages_str=odl_pages,
                hybrid=odl_hybrid,
                hybrid_url=odl_hybrid_url,
                hybrid_timeout=odl_hybrid_timeout,
            )
        if odl_text and odl_text.strip() and not _is_garbled(odl_text):
            # Parse opendataloader markdown output into per-page dict
            page_texts = cls._parse_odl_markdown(odl_text, indices)
            if page_texts:
                log.info("opendataloader-pdf extraction succeeded (%d chars)",
                         sum(len(t) for t in page_texts.values()))
                return cls._format_output(page_texts, indices, total, meta_lines, via="opendataloader-pdf")
            # If parsing failed, return as single block
            return ToolResult(
                output=cls._prepend_metadata(odl_text, meta_lines),
                metadata={"total_pages": total, "via": "opendataloader-pdf"},
            )

        # --- Use best available result ----------------------------------------
        # Fall back to pypdf if it got anything at all
        if reader is not None:
            page_texts = cls._extract_pypdf(reader, indices)
            if page_texts and sum(len(t) for t in page_texts.values()) > 0:
                return cls._format_output(page_texts, indices, total, meta_lines, via="pypdf-fallback")

        return ToolResult(error="All PDF extraction methods returned empty content.", is_error=True)

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pypdf(reader: Any, indices: list[int]) -> dict[int, str] | None:
        """Fast extraction using pypdf (already loaded for page count).

        Pure Python — no crash risk, no subprocess needed. ~1s for most PDFs.
        Works well when the PDF has a clean text layer.
        """
        result: dict[int, str] = {}
        try:
            for idx in indices:
                if idx < len(reader.pages):
                    result[idx] = reader.pages[idx].extract_text() or ""
        except Exception as exc:
            log.debug("pypdf extraction failed: %s", exc)
            return None
        return result if result else None

    @staticmethod
    def _extract_opendataloader(
        data: bytes,
        pages_str: str = "",
        *,
        hybrid: str = "docling-fast",
        hybrid_url: str = "http://localhost:5002",
        hybrid_timeout: int = 60000,
    ) -> str | None:
        """Extract text using opendataloader-pdf (Java + hybrid AI backend).

        Writes bytes to a temp file, calls opendataloader_pdf.convert() with
        markdown format, reads the output .md file, and cleans up.

        The JVM runs in its own process — no crash risk to Python. Hybrid mode
        routes complex pages (tables, OCR) to the docling-fast backend for
        +90% table accuracy while keeping simple text pages fast and local.
        """
        try:
            import opendataloader_pdf
        except ImportError:
            log.warning("opendataloader-pdf not installed, skipping")
            return None

        if not data:
            log.debug("opendataloader-pdf: skipping empty input (0 bytes)")
            return None

        tmp_pdf = None
        tmp_dir = None
        try:
            # Write bytes to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(data)
            tmp.flush()
            tmp.close()
            tmp_pdf = tmp.name

            # Create temp output directory
            tmp_dir = tempfile.mkdtemp(prefix="odl_out_")

            # Build convert kwargs
            convert_kwargs: dict[str, Any] = {
                "input_path": [tmp_pdf],
                "output_dir": tmp_dir,
                "format": "markdown",
                "quiet": True,
                "hybrid_fallback": True,
            }
            if hybrid and hybrid != "off":
                convert_kwargs["hybrid"] = hybrid
                convert_kwargs["hybrid_url"] = hybrid_url
                convert_kwargs["hybrid_timeout"] = str(hybrid_timeout)
            if pages_str:
                convert_kwargs["pages"] = pages_str

            mode = f"hybrid={hybrid}" if hybrid and hybrid != "off" else "java-only"
            log.info("PDF extraction via opendataloader-pdf (%s, %d bytes, pages=%s)",
                     mode, len(data), pages_str or "all")

            # Call convert (spawns JVM subprocess).
            # Use reference-counted stderr suppression to hide Java stack traces
            # from malformed PDFs — thread-safe across concurrent eval workers.
            t0 = time.monotonic()
            _suppress_stderr()
            try:
                opendataloader_pdf.convert(**convert_kwargs)
            finally:
                _restore_stderr()
            elapsed = time.monotonic() - t0

            # Find the output .md file
            md_files = list(Path(tmp_dir).rglob("*.md"))
            if not md_files:
                log.warning("opendataloader-pdf produced no markdown output (%.1fs)", elapsed)
                return None

            # Read the first (and usually only) markdown file
            md_text = md_files[0].read_text(encoding="utf-8", errors="replace")
            if md_text:
                log.info("opendataloader-pdf extracted %d chars in %.1fs", len(md_text), elapsed)
            else:
                log.warning("opendataloader-pdf produced empty output (%.1fs)", elapsed)
            return md_text if md_text else None

        except FileNotFoundError:
            log.error("opendataloader-pdf requires Java — 'java' not found in PATH")
            return None
        except subprocess.CalledProcessError as exc:
            log.warning("opendataloader-pdf conversion failed (exit code %s): %s",
                        exc.returncode, str(exc)[:200])
            return None
        except Exception as exc:
            log.warning("opendataloader-pdf extraction failed: %s", exc)
            return None
        finally:
            if tmp_pdf:
                try:
                    os.remove(tmp_pdf)
                except Exception:
                    pass
            if tmp_dir:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    @staticmethod
    def _parse_odl_markdown(md_text: str, indices: list[int]) -> dict[int, str] | None:
        """Parse opendataloader-pdf markdown output into per-page dict.

        opendataloader-pdf uses page separators in its markdown output.
        We split on common page separator patterns and map to requested indices.
        """
        if not md_text:
            return None

        # opendataloader-pdf default page separator: "---" or "\n---\n"
        # Also handle "<!-- page N -->" style separators
        parts = re.split(r"\n-{3,}\n|\n<!-- *page \d+ *-->\n", md_text)

        # If no splits, the whole text is one page
        if len(parts) <= 1:
            if len(indices) == 1:
                return {indices[0]: md_text.strip()}
            # Can't split into pages — return as single block for first page
            return {indices[0]: md_text.strip()} if indices else None

        result: dict[int, str] = {}
        for i, part in enumerate(parts):
            if i < len(indices):
                result[indices[i]] = part.strip()
        return result if result else None

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @classmethod
    def _format_output(
        cls,
        page_texts: dict[int, str],
        indices: list[int],
        total: int,
        meta_lines: list[str],
        via: str = "",
    ) -> ToolResult:
        """Format page texts into final output string."""
        parts: list[str] = []

        if meta_lines:
            parts.append("\n".join(meta_lines) + "\n")

        for idx in indices:
            text = page_texts.get(idx, "")
            parts.append(f"--- Page {idx + 1} ---\n{text}\n")

        output = "\n".join(parts).rstrip()
        return ToolResult(
            output=output,
            metadata={"total_pages": total, "pages_extracted": len(indices), **({"via": via} if via else {})},
        )

    @staticmethod
    def _prepend_metadata(text: str, meta_lines: list[str]) -> str:
        """Prepend metadata lines to text if available."""
        if meta_lines:
            return "\n".join(meta_lines) + "\n\n" + text
        return text

    # ------------------------------------------------------------------
    # Hybrid server management
    # ------------------------------------------------------------------

    def _resolve_hybrid_url(self) -> str:
        """Resolve the hybrid backend URL, auto-starting a server if needed.

        If ``odl_hybrid_url`` is set explicitly, use it as-is (external server).
        If empty and hybrid mode is enabled, auto-start a managed server.
        """
        if self._odl_hybrid_url:
            return self._odl_hybrid_url
        if not self._odl_hybrid or self._odl_hybrid == "off":
            return ""
        try:
            from arcticswarm.tools.odl_server import ensure_server
            url = ensure_server(force_ocr=self._odl_force_ocr)
            self._odl_hybrid_url = url  # cache for subsequent calls
            return url
        except Exception as exc:
            log.warning("Could not start hybrid backend: %s — falling back to Java-only", exc)
            return ""

    # ------------------------------------------------------------------
    # Jina fallback (for URLs blocked by CAPTCHA/403)
    # ------------------------------------------------------------------

    def _jina_fallback(
        self, url: str, direct_error: str = ""
    ) -> ToolResult:
        """Last-resort fetch via Jina Reader API for URLs that block direct download."""
        if not self._jina_api_key:
            msg = f"Failed to read PDF from {url}."
            if direct_error:
                msg = f"Download failed ({direct_error}) and no Jina API key configured."
            return ToolResult(error=msg, is_error=True)

        try:
            jina_headers = {
                "Authorization": f"Bearer {self._jina_api_key}",
                "X-Base": "final",
                "X-With-Generated-Alt": "true",
                "X-Engine": "browser",
                "X-With-Iframe": "true",
                "X-With-Shadow-Dom": "true",
            }

            resp = requests.get(
                f"https://r.jina.ai/{url}",
                headers=jina_headers,
                timeout=_JINA_TIMEOUT,
            )

            # Retry with extended timeout if page not fully loaded
            if resp.ok and "Warning: This page maybe not yet fully loaded" in resp.text:
                log.info("Jina partial load for %s — retrying with extended timeout", url)
                resp = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=jina_headers,
                    timeout=_JINA_EXTENDED_TIMEOUT,
                )

            resp.raise_for_status()
            text = resp.text
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (402, 429):
                err = f"Jina API credit/rate limit reached (HTTP {status}). URL: {url}"
                if direct_error:
                    err = f"Direct download failed ({direct_error}). {err}"
                return ToolResult(error=err, is_error=True)
            err = f"Jina Reader failed for {url}: {exc}"
            if direct_error:
                err = f"Direct download failed ({direct_error}). {err}"
            return ToolResult(error=err, is_error=True)
        except Exception as exc:
            err = f"Jina Reader failed for {url}: {exc}"
            if direct_error:
                err = f"Direct download failed ({direct_error}). {err}"
            return ToolResult(error=err, is_error=True)

        if not text or not text.strip():
            return ToolResult(
                error=f"All extraction methods returned empty content for {url}.",
                is_error=True,
            )

        return ToolResult(
            output=text,
            metadata={"url": url, "via": "Jina Reader"},
        )
