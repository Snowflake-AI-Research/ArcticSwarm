"""Thread-safe disk-backed content cache for web_fetch and pdf_read.

Shared across all agents working on the same question. Uses the output_dir
as the question-level isolation boundary.  Cache entries are stored as JSON
files named by SHA-256 hash of the normalized URL (+pages for pdf_read).

Design:
  - FULL content is always stored (never truncated).
  - Truncation for delivery is handled by the tool at read time based on
    whether the source scorer is enabled.
  - Thread safety via a single threading.Lock (low contention since file I/O
    is fast for small JSON files).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlparse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    """Normalize a URL for cache key purposes.

    - Strip fragment (#...)
    - Lowercase scheme and host
    - Strip trailing slash from path (unless path is just '/')
    """
    url = (url or "").strip()
    url, _ = urldefrag(url)  # strip fragment
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=path,
    )
    return normalized.geturl()


def _cache_key(url: str, pages: str = "") -> str:
    """Compute a filesystem-safe cache key from URL + optional pages.

    Returns a 32-char hex hash.  If pages is empty, the key is just the
    URL hash.  If pages is non-empty, the key includes the pages suffix
    to distinguish different page extractions of the same PDF.
    """
    norm = _normalize_url(url)
    if pages.strip():
        raw = f"{norm}|pages={pages.strip()}"
    else:
        raw = norm
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single cached content entry."""

    url: str  # normalized URL
    content: str  # FULL content (never truncated), or error message if is_error
    pages: str = ""  # pages param (pdf_read only, empty = all pages)
    is_pdf: bool = False  # True if content came from a PDF
    is_error: bool = False  # True if this entry records a fetch failure
    via: str = ""  # extraction method (jina, serper, pypdf, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ContentCache:
    """Thread-safe, disk-backed content cache for a single question.

    Cache directory: ``{cache_dir}/cache/content/``

    Each entry is a JSON file named by its 32-char hex cache key.

    Thread safety: uses ``threading.Lock`` for all read/write operations.
    The lock is fine-grained enough for the expected concurrency (2-16
    subagents, each making sequential tool calls).
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
        case_id: str = "",
    ) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._cache_dir: Path | None = None

        if cache_dir and enabled:
            base = Path(cache_dir) / "cache" / "content"
            if case_id:
                base = base / case_id
            self._cache_dir = base
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._cache_dir is not None

    def get(self, url: str, pages: str = "") -> CacheEntry | None:
        """Look up a cache entry by URL and pages.  Returns None on miss.

        Consults the per-question file cache, which may also hold cached
        failures for within-run dedup.
        """
        if not self.enabled:
            return None

        key = _cache_key(url, pages)

        if self._cache_dir is None:
            return None
        with self._lock:
            path = self._cache_dir / f"{key}.json"  # type: ignore[operator]
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return CacheEntry(**data)
            except Exception as exc:
                log.warning("Cache read error for key %s: %s", key, exc)
                return None

    def get_any_pages(self, url: str) -> CacheEntry | None:
        """Look up a cache entry for a URL regardless of pages.

        Checks the no-pages key first (web_fetch cache entries store PDFs
        without a pages parameter).  Used by pdf_read to find content cached
        by web_fetch.
        """
        if not self.enabled:
            return None
        # Try with no pages (how web_fetch stores PDF content)
        return self.get(url, pages="")

    def put(
        self,
        url: str,
        content: str,
        *,
        pages: str = "",
        is_pdf: bool = False,
        via: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store content in the cache.  Only stores non-empty successful results."""
        if not self.enabled or not content:
            return

        key = _cache_key(url, pages)
        entry = CacheEntry(
            url=_normalize_url(url),
            content=content,
            pages=pages.strip(),
            is_pdf=is_pdf,
            via=via,
            metadata=metadata or {},
        )

        if self._cache_dir is None:
            return
        with self._lock:
            path = self._cache_dir / f"{key}.json"  # type: ignore[operator]
            try:
                # Recreate the case dir right before writing in case it was
                # removed since __init__ (idempotent, microseconds).
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(asdict(entry), ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                log.warning("Cache write error for key %s: %s", key, exc)

    def put_failure(
        self,
        url: str,
        error_message: str,
        *,
        pages: str = "",
        is_pdf: bool = False,
    ) -> None:
        """Cache a fetch failure so other agents don't retry the same broken URL.

        Failures are recorded in the per-question file cache only (within-run
        dedup).
        """
        if not self.enabled or not error_message:
            return
        if self._cache_dir is None:
            return

        key = _cache_key(url, pages)
        entry = CacheEntry(
            url=_normalize_url(url),
            content=error_message,
            pages=pages.strip(),
            is_pdf=is_pdf,
            is_error=True,
            via="",
            metadata={},
        )

        with self._lock:
            path = self._cache_dir / f"{key}.json"  # type: ignore[operator]
            # Don't overwrite a successful cache entry with a failure
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if not existing.get("is_error", False):
                        return  # keep the successful entry
                except Exception:
                    pass
            try:
                # Recreate the case dir in case it was removed since __init__
                # (see put()).
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(asdict(entry), ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                log.warning("Cache failure write error for key %s: %s", key, exc)

    def has(self, url: str, pages: str = "") -> bool:
        """Check if a URL (with optional pages) is in the cache."""
        if not self.enabled:
            return False
        key = _cache_key(url, pages)
        if self._cache_dir is None:
            return False
        with self._lock:
            return (self._cache_dir / f"{key}.json").exists()  # type: ignore[operator]
