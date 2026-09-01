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
import sqlite3
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
# Global, cross-run SQLite cache store
# ---------------------------------------------------------------------------
#
# A single SQLite file shared by EVERY run on a machine, so a URL fetched once
# is never re-fetched.  Keyed by the same ``_cache_key`` the per-question
# file cache uses (url[+pages] hash), so it is seeded directly from historical
# ``cache/content`` dirs (see scripts/build_fetch_cache.py).  Layered UNDER the
# existing ``ContentCache`` API: success entries are written through to it on
# every fetch; failures are NOT (a transient network failure must never poison
# the shared cache — those URLs get refetched live).  On key conflict the
# longer content wins.

_GLOBAL_STORES: dict[str, "_GlobalSqliteStore | None"] = {}
_GLOBAL_STORES_LOCK = threading.Lock()


def get_global_store(db_path: str | Path | None) -> "_GlobalSqliteStore | None":
    """Return the process-wide store for ``db_path`` (one per resolved path).

    Returns ``None`` when ``db_path`` is empty or the store can't be opened
    (e.g. the directory isn't writable) — callers then fall back to the
    per-question file cache only, so a missing global cache never breaks a run.
    """
    if not db_path:
        return None
    key = str(Path(db_path).expanduser())
    with _GLOBAL_STORES_LOCK:
        if key in _GLOBAL_STORES:
            return _GLOBAL_STORES[key]
        try:
            # The cluster sweeps /data to S3; if the cache file is gone, restore
            # the shared cache root from S3 before opening (no-op when already
            # present / off-cluster / aws absent). Syncs the PARENT dir
            # (the shared cache root), so one `aws s3 sync` restores the fetch
            # cache; keyed once per dir per process.
            from arcticswarm.tools.cache_restore import ensure_cache_restored
            ensure_cache_restored(Path(key).parent, present_path=Path(key))
        except Exception as exc:  # noqa: BLE001 - restore is best-effort
            log.warning("Global fetch cache: S3 restore check failed: %s", exc)
        try:
            store: _GlobalSqliteStore | None = _GlobalSqliteStore(Path(key))
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            log.warning("Global fetch cache unavailable at %s: %s", key, exc)
            store = None
        _GLOBAL_STORES[key] = store
        return store


class _GlobalSqliteStore:
    """Thread-safe SQLite key/value store of :class:`CacheEntry` rows.

    One connection (``check_same_thread=False`` + WAL + a coarse lock) is shared
    across the run's worker threads — fine for the modest, mostly-read fetch
    workload.  Only successful entries are stored.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "  key TEXT PRIMARY KEY,"
            "  url TEXT NOT NULL,"
            "  content TEXT NOT NULL,"
            "  pages TEXT NOT NULL DEFAULT '',"
            "  is_pdf INTEGER NOT NULL DEFAULT 0,"
            "  via TEXT NOT NULL DEFAULT '',"
            "  metadata TEXT NOT NULL DEFAULT '{}'"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> "CacheEntry | None":
        with self._lock:
            row = self._conn.execute(
                "SELECT url, content, pages, is_pdf, via, metadata FROM entries WHERE key=?",
                (key,),
            ).fetchone()
        if not row:
            return None
        try:
            meta = json.loads(row[5]) if row[5] else {}
        except Exception:
            meta = {}
        return CacheEntry(
            url=row[0], content=row[1], pages=row[2],
            is_pdf=bool(row[3]), is_error=False, via=row[4], metadata=meta,
        )

    def has(self, key: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM entries WHERE key=? LIMIT 1", (key,)
            ).fetchone() is not None

    def put(self, key: str, entry: "CacheEntry") -> None:
        """Insert a success entry; on key conflict the LONGER content wins."""
        if entry.is_error or not entry.content:
            return
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO entries(key, url, content, pages, is_pdf, via, metadata)"
                    " VALUES(?,?,?,?,?,?,?)"
                    " ON CONFLICT(key) DO UPDATE SET"
                    "   content=excluded.content, url=excluded.url, pages=excluded.pages,"
                    "   is_pdf=excluded.is_pdf, via=excluded.via, metadata=excluded.metadata"
                    " WHERE length(excluded.content) > length(entries.content)",
                    (
                        key, entry.url, entry.content, entry.pages,
                        int(entry.is_pdf), entry.via,
                        json.dumps(entry.metadata or {}, ensure_ascii=False),
                    ),
                )
                self._conn.commit()
            except Exception as exc:  # noqa: BLE001
                log.warning("Global fetch cache write error for key %s: %s", key, exc)


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
        global_db_path: str | Path | None = None,
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

        # Global, cross-run SQLite store (shared by every run). Layered UNDER
        # the per-question file cache: read global-first, write successes
        # through. None when disabled / unavailable.
        self._global = get_global_store(global_db_path) if enabled else None

    @property
    def enabled(self) -> bool:
        return self._enabled and (self._cache_dir is not None or self._global is not None)

    def get(self, url: str, pages: str = "") -> CacheEntry | None:
        """Look up a cache entry by URL and pages.  Returns None on miss.

        The global (cross-run) store is consulted first; on a miss the
        per-question file cache is checked (which may also hold cached
        failures for within-run dedup).
        """
        if not self.enabled:
            return None

        key = _cache_key(url, pages)

        if self._global is not None:
            entry = self._global.get(key)
            if entry is not None:
                return entry

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

        # Write through to the shared global store (success only).
        if self._global is not None:
            self._global.put(key, entry)

        if self._cache_dir is None:
            return
        with self._lock:
            path = self._cache_dir / f"{key}.json"  # type: ignore[operator]
            try:
                # The cluster sweeps /data to S3 and deletes locally mid-run, so
                # the case dir created in __init__ may be gone by write time.
                # Recreate it right before writing (idempotent, microseconds).
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

        Failures are recorded ONLY in the per-question file cache (within-run
        dedup) — never in the shared global store, so a transient network
        failure never poisons future runs.
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
                # Recreate the case dir in case the /data->S3 sweep removed it
                # since __init__ (see put()).
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
        if self._global is not None and self._global.has(key):
            return True
        if self._cache_dir is None:
            return False
        with self._lock:
            return (self._cache_dir / f"{key}.json").exists()  # type: ignore[operator]
