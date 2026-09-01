#!/usr/bin/env python3
"""Build (or refresh) the global web_fetch / pdf_read SQLite cache.

Mines every historical ``cache/content`` directory (e.g. the per-question
caches written under any run's ``results/*/cache/content/**``) into a single
SQLite file that ArcticSwarm's :class:`~arcticswarm.tools.content_cache.ContentCache`
reads as its global, cross-run cache. So a URL fetched (or a PDF page read)
once in any past run is served from disk instead of the network.

Why this works with zero re-keying: the cache key is derived from the *same*
``_normalize_url`` / ``_cache_key``, and each cache file is named by that key.
So the filename stem IS the cache key and the JSON body already matches
``CacheEntry``. We just dedupe and load them.

Rules:
  * Only SUCCESSFUL entries are imported (``is_error`` rows are skipped) — a
    past transient failure must never block a future live fetch.
  * On key conflict across runs the LONGER content wins (best extraction).
  * Idempotent: re-running merges new sources into an existing DB.

Schema matches ``_GlobalSqliteStore``:
  entries(key, url, content, pages, is_pdf, via, metadata)

Usage:
  python scripts/build_fetch_cache.py \
      --source '/path/to/results/*/cache/content' \
      --db /path/to/cache/fetch_cache.sqlite \
      --workers 12

  # --source is repeatable (multiple globs allowed). --db defaults to the
  # configured fetch_cache_path when omitted.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from multiprocessing import Pool

_KEY_RE = re.compile(r"^[0-9a-f]{32}$")

_SCHEMA = (
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

# Upsert that keeps the longer content on conflict.
_UPSERT = (
    "INSERT INTO entries(key, url, content, pages, is_pdf, via, metadata)"
    " VALUES(?,?,?,?,?,?,?)"
    " ON CONFLICT(key) DO UPDATE SET"
    "   content=excluded.content, url=excluded.url, pages=excluded.pages,"
    "   is_pdf=excluded.is_pdf, via=excluded.via, metadata=excluded.metadata"
    " WHERE length(excluded.content) > length(entries.content)"
)


def _fast_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=120.0)
    # One-time, rebuildable build artifact -> trade durability for speed.
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(_SCHEMA)
    return conn


def _discover_dirs(source_globs: list[str]) -> list[str]:
    """Return every directory (under the source globs) that contains *.json."""
    roots: list[str] = []
    for g in source_globs:
        roots.extend(glob.glob(g))
    leaf_dirs: set[str] = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if any(fn.endswith(".json") for fn in filenames):
                leaf_dirs.add(dirpath)
    return sorted(leaf_dirs)


def _process_chunk(args: tuple[int, list[str], str]) -> tuple[str, int, int, int]:
    """Worker: import all *.json under the given dirs into a shard DB.

    Returns (shard_path, n_files_seen, n_stored, n_errors_skipped).
    """
    worker_id, dirs, shard_path = args
    conn = _fast_connect(shard_path)
    seen = stored = errs = 0
    batch = 0
    for d in dirs:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".json"):
                continue
            stem = name[:-5]
            if not _KEY_RE.match(stem):
                continue
            seen += 1
            fpath = os.path.join(d, name)
            try:
                with open(fpath, "rb") as fh:
                    obj = json.loads(fh.read())
            except Exception:
                continue
            if obj.get("is_error"):
                errs += 1
                continue
            content = obj.get("content") or ""
            if not content:
                continue
            url = obj.get("url") or ""
            meta = obj.get("metadata") or {}
            try:
                meta_s = json.dumps(meta, ensure_ascii=False)
            except Exception:
                meta_s = "{}"
            try:
                conn.execute(_UPSERT, (
                    stem, url, content, obj.get("pages") or "",
                    int(bool(obj.get("is_pdf"))), obj.get("via") or "", meta_s,
                ))
                stored += 1
                batch += 1
                if batch >= 5000:
                    conn.commit()
                    batch = 0
            except Exception:
                continue
    conn.commit()
    conn.close()
    return shard_path, seen, stored, errs


def _merge_shards(db_path: str, shard_paths: list[str]) -> int:
    """Merge shard DBs into the master, longest-content-wins. Returns row count."""
    conn = _fast_connect(db_path)
    for sp in shard_paths:
        if not os.path.exists(sp):
            continue
        conn.execute("ATTACH DATABASE ? AS shard", (sp,))
        conn.execute(
            "INSERT INTO entries(key, url, content, pages, is_pdf, via, metadata)"
            " SELECT key, url, content, pages, is_pdf, via, metadata FROM shard.entries WHERE true"
            " ON CONFLICT(key) DO UPDATE SET"
            "   content=excluded.content, url=excluded.url, pages=excluded.pages,"
            "   is_pdf=excluded.is_pdf, via=excluded.via, metadata=excluded.metadata"
            " WHERE length(excluded.content) > length(entries.content)"
        )
        conn.commit()
        conn.execute("DETACH DATABASE shard")
    n = conn.execute("SELECT count(*) FROM entries").fetchone()[0]
    conn.execute("PRAGMA optimize")
    conn.close()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--source", action="append", default=[],
        help="Glob of cache/content roots to mine (repeatable), e.g. "
             "'/path/to/results/*/cache/content'. Required.",
    )
    ap.add_argument("--db", default="", help="Output SQLite path (default: configured fetch_cache_path).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--keep-shards", action="store_true", help="Don't delete shard DBs after merge.")
    args = ap.parse_args()

    sources = args.source
    if not sources:
        print("ERROR: --source is required (glob of cache/content roots to mine).", file=sys.stderr)
        return 2

    db_path = args.db
    if not db_path:
        try:
            from arcticswarm.config import ArcticswarmConfig
            db_path = ArcticswarmConfig.resolve().fetch_cache_path
        except Exception:
            db_path = ""
    if not db_path:
        print("ERROR: no --db and no configured fetch_cache_path.", file=sys.stderr)
        return 2
    db_path = os.path.expanduser(db_path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    t0 = time.monotonic()
    print(f"[build_fetch_cache] sources={sources}", flush=True)
    print(f"[build_fetch_cache] db={db_path} workers={args.workers}", flush=True)
    print("[build_fetch_cache] discovering cache dirs ...", flush=True)
    dirs = _discover_dirs(sources)
    print(f"[build_fetch_cache] found {len(dirs)} cache dir(s) in {time.monotonic()-t0:.0f}s", flush=True)
    if not dirs:
        print("[build_fetch_cache] nothing to do.", flush=True)
        return 0

    workers = max(1, min(args.workers, len(dirs)))
    chunks: list[list[str]] = [[] for _ in range(workers)]
    for i, d in enumerate(dirs):
        chunks[i % workers].append(d)

    shard_dir = tempfile.mkdtemp(prefix="fetchcache_shards_", dir=os.path.dirname(db_path) or ".")
    tasks = [
        (wid, chunk, os.path.join(shard_dir, f"shard_{wid}.sqlite"))
        for wid, chunk in enumerate(chunks) if chunk
    ]

    print(f"[build_fetch_cache] importing across {len(tasks)} worker(s) ...", flush=True)
    seen = stored = errs = 0
    shard_paths: list[str] = []
    with Pool(processes=len(tasks)) as pool:
        for sp, s, st, er in pool.imap_unordered(_process_chunk, tasks):
            shard_paths.append(sp)
            seen += s; stored += st; errs += er
            print(f"[build_fetch_cache]   shard done: +{st} stored (running: seen={seen} stored={stored} err_skipped={errs}, {time.monotonic()-t0:.0f}s)", flush=True)

    print(f"[build_fetch_cache] merging {len(shard_paths)} shard(s) ...", flush=True)
    total = _merge_shards(db_path, shard_paths)

    if not args.keep_shards:
        for sp in shard_paths:
            try:
                os.remove(sp)
            except OSError:
                pass
        try:
            os.rmdir(shard_dir)
        except OSError:
            pass

    size_gb = os.path.getsize(db_path) / 1e9 if os.path.exists(db_path) else 0.0
    print(
        f"[build_fetch_cache] DONE: {total} unique entries -> {db_path} "
        f"({size_gb:.1f} GB) | seen={seen} stored={stored} errors_skipped={errs} "
        f"in {time.monotonic()-t0:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
