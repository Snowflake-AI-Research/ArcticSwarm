"""On-disk SQLite search-result cache for ArcticSwarm web_search.

Serves previously-fetched search results (from ``live_search_documents``,
loaded via ``scripts/build_search_cache.py``) so repeated / cross-run queries
bypass the live provider call.  Design goals:

- **Seamless**: the cache returns *raw* provider results (``{title, url,
  description, ...}``) at the same point the live provider would, so the
  downstream source-scorer (judge) and formatter run unchanged.  The model
  cannot tell a hit from a live call.
- **Never reuse stored judge scores**: only ``title/url/snippet`` (+ benign
  provider extras like Tavily's score/answer) are stored; the source scorer is
  always re-run on read.
- **Fallback preserved**: lookups are per-provider, so a provider miss returns
  ``None`` and the existing provider-order loop falls through to the next one,
  exactly like a live miss.
- **Self-populating**: ``put`` write-through on a live miss-fill, so the next
  lookup hits.

Storage: one row per (provider, query-key, url) in a single SQLite DB, opened
once per process (``get_search_cache``) and shared across all agents/workers.
WAL + busy_timeout make concurrent reader + write-through safe.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fields stored verbatim in their own columns; everything else (e.g.
# ``tavily_score``, ``_tavily_answer``) goes into the JSON ``extra`` column.
# Judge/score fields are NEVER stored (the builder strips them and live results
# don't carry them at the raw stage).
_CORE_FIELDS = ("title", "url", "description")
_FIELD_SEP = "\x1f"


def _norm_query(query: str) -> str:
    """Lowercase + collapse whitespace (operators/quotes preserved)."""
    return " ".join((query or "").lower().split())


def _qkey(query: str, count: int, country: str | None) -> str:
    """Cache key: normalized query + count + country (per design decision)."""
    return _FIELD_SEP.join((_norm_query(query), str(int(count)), (country or "").lower()))


class SearchCache:
    """SQLite-backed, thread-safe per-provider search-result store."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False + our own lock; WAL for concurrent read/write.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        # Monotonic insertion counter for stable result ordering. Seed from the
        # current max so builder + write-through share one ordering space.
        row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM docs").fetchone()
        self._seq = int(row[0]) if row else 0

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                provider TEXT NOT NULL,
                qkey     TEXT NOT NULL,
                url      TEXT NOT NULL,
                title    TEXT,
                snippet  TEXT,
                extra    TEXT,
                seq      INTEGER NOT NULL,
                PRIMARY KEY (provider, qkey, url)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_docs_lookup ON docs (provider, qkey, seq)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(
        self, provider: str, query: str, count: int, country: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Return cached raw results for (provider, query, count, country), or
        ``None`` if there are no rows (treated as a miss → live fallback)."""
        qk = _qkey(query, count, country)
        with self._lock:
            rows = self._conn.execute(
                "SELECT title, url, snippet, extra FROM docs "
                "WHERE provider=? AND qkey=? ORDER BY seq",
                (provider, qk),
            ).fetchall()
        if not rows:
            return None
        out: list[dict[str, Any]] = []
        for title, url, snippet, extra in rows:
            entry: dict[str, Any] = {
                "title": title or "",
                "url": url or "",
                "description": snippet or "",
            }
            if extra:
                try:
                    entry.update(json.loads(extra))
                except (json.JSONDecodeError, ValueError):
                    pass
            out.append(entry)
        # Honor the requested count (the key already pins count, but results may
        # have been merged across fills — return at most ``count``).
        return out[: max(1, int(count))] if count else out

    # ------------------------------------------------------------------
    # Write (builder + live miss-fill write-through)
    # ------------------------------------------------------------------

    def put(
        self,
        provider: str,
        query: str,
        count: int,
        country: str | None,
        results: list[dict[str, Any]],
        replace: bool = False,
    ) -> int:
        """Insert raw results (dedup by url). Strips judge/score-style fields;
        keeps title/url/snippet + benign provider extras. Returns rows added.

        When ``replace`` is True, existing rows for (provider, qkey) are deleted
        first so the key reflects the latest live result (refresh / write-only
        mode). Otherwise insert is additive (INSERT OR IGNORE miss-fill)."""
        if not results:
            return 0
        qk = _qkey(query, count, country)
        rows = []
        with self._lock:
            if replace:
                self._conn.execute(
                    "DELETE FROM docs WHERE provider=? AND qkey=?", (provider, qk)
                )
            for r in results:
                url = str(r.get("url") or "").strip()
                if not url:
                    continue
                extra = {
                    k: v for k, v in r.items()
                    if k not in _CORE_FIELDS and not _is_score_field(k)
                }
                self._seq += 1
                rows.append((
                    provider, qk, url,
                    str(r.get("title") or ""),
                    str(r.get("description") or r.get("snippet") or ""),
                    json.dumps(extra, ensure_ascii=False) if extra else None,
                    self._seq,
                ))
            if not rows:
                return 0
            self._conn.executemany(
                "INSERT OR IGNORE INTO docs "
                "(provider, qkey, url, title, snippet, extra, seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
            return self._conn.total_changes  # cumulative; caller treats as best-effort

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


def _is_score_field(key: str) -> bool:
    """True for source-scorer / judge fields we must never reuse."""
    k = key.lower()
    return any(s in k for s in ("score", "judge", "rating", "relevance", "quality")) \
        and k != "tavily_score"  # tavily_score is a provider relevance signal, not our judge


# ---------------------------------------------------------------------------
# Process-global singletons keyed by db path (shared across agents/workers).
# ---------------------------------------------------------------------------

_STORES: dict[str, SearchCache] = {}
_STORES_LOCK = threading.Lock()


def get_search_cache(db_path: str | Path | None) -> SearchCache | None:
    """Return the shared ``SearchCache`` for ``db_path`` (created once), or
    ``None`` if no path is given."""
    if not db_path:
        return None
    key = str(db_path)
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            try:
                # If an S3 mirror is configured and the search cache is gone,
                # restore the shared cache root from S3 before opening (no-op
                # by default / when present / when aws is absent). Syncs the
                # PARENT dir, so one `aws s3 sync` restores both the search and
                # fetch caches that share a root; keyed once per dir per process.
                from arcticswarm.tools.cache_restore import ensure_cache_restored
                ensure_cache_restored(Path(key).parent, present_path=key)
            except Exception as exc:  # best-effort restore
                logger.warning("SearchCache: S3 restore check failed: %s", exc)
            try:
                store = SearchCache(key)
                _STORES[key] = store
            except Exception as exc:
                logger.warning("SearchCache: failed to open %s: %s", key, exc)
                return None
        return store
