#!/usr/bin/env python3
"""Merge node-local cache mirror(s) back into the shared /data masters.

The eval auto-syncs deltas during a run (web.cache_local_mirror), but use this
to consolidate manually — e.g. after a multi-host run, or if a process exited
before its final sync. Does a FULL merge (every local row): longest-content
wins for the fetch cache, first-wins for the search cache. Serialized across
hosts via flock + rollback-journal mode (safe on Lustre — no WAL/-shm mmap).

Usage:
  python snowflake/scripts/sync_cache.py \
      --local-dir /data-fast/soyoon/cache \
      --fetch-master  /data/soyoon/cache/fetch_cache.sqlite \
      --search-master /data/soyoon/cache/search_cache.sqlite
"""

from __future__ import annotations

import argparse
import os

from arcticswarm.tools.cache_sync import (
    _Cache,
    _FETCH_COLS,
    _FETCH_CONFLICT,
    _SEARCH_COLS,
)


def _merge(master: str, local: str, table: str, cols, conflict: str, order_col: str) -> None:
    if not os.path.exists(local):
        print(f"skip {table}: no local cache at {local}")
        return
    if not os.path.exists(master):
        print(f"skip {table}: no master at {master}")
        return
    c = _Cache(master, local, table=table, cols=cols, conflict=conflict, order_col=order_col)
    c.watermark = 0  # full merge (every local row)
    n = c.sync()
    print(f"merged {table}: ~{n} rows  {local} -> {master}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dir", default="/data-fast/soyoon/cache")
    ap.add_argument("--fetch-master", default="/data/soyoon/cache/fetch_cache.sqlite")
    ap.add_argument("--search-master", default="/data/soyoon/cache/search_cache.sqlite")
    a = ap.parse_args()
    _merge(a.fetch_master, os.path.join(a.local_dir, "fetch_cache.sqlite"),
           "entries", _FETCH_COLS, _FETCH_CONFLICT, "rowid")
    _merge(a.search_master, os.path.join(a.local_dir, "search_cache.sqlite"),
           "docs", _SEARCH_COLS, " ON CONFLICT(provider, qkey, url) DO NOTHING", "seq")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
