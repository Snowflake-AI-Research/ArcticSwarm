#!/usr/bin/env python3
"""Build the ArcticSwarm SearchCache SQLite DB from a ``live_search_documents``
corpus.

Corpus layout (per provider sub-dir ``brave/ serper/ tavily/ cortex/ misc/``):
  - ``documents_*.json`` : list of ``{uid, title, url, snippet}``
  - ``metadata.json``    : ``uid -> {query, count, country, ...}``  (1:1 with docs)

We join doc.uid -> metadata to compute the cache key (normalized query + count +
country) and insert ``title/url/snippet`` into the SQLite ``docs`` table. Judge /
source-scorer score fields are NEVER stored — the source scorer is always re-run
at read time.

Streaming: uses ``ijson`` if installed (bounded RAM over the multi-GB files);
otherwise falls back to ``json.load`` per file (needs several GB RAM transiently
— ``pip install ijson`` recommended).

Usage:
  python scripts/build_search_cache.py \
      --src /home/soyoon/cortex/snowswarm/live_search_documents \
      --db  /home/soyoon/ArcticSwarm/cache/search_cache.sqlite \
      [--providers brave serper tavily cortex misc]

Idempotent: re-running merges (INSERT OR IGNORE on (provider, qkey, url)).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Reuse the exact key normalization + score-field filter the cache uses at read.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcticswarm.tools.search_cache import _qkey, _is_score_field, _CORE_FIELDS  # noqa: E402

try:
    import ijson  # type: ignore
    _HAVE_IJSON = True
except Exception:
    _HAVE_IJSON = False

_DEFAULT_PROVIDERS = ["brave", "serper", "tavily", "cortex", "misc"]
_BATCH = 20000


def _open_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")      # build-time speed; fine for a rebuild
    conn.execute("PRAGMA cache_size=-200000")   # ~200MB page cache
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
            provider TEXT NOT NULL, qkey TEXT NOT NULL, url TEXT NOT NULL,
            title TEXT, snippet TEXT, extra TEXT, seq INTEGER NOT NULL,
            PRIMARY KEY (provider, qkey, url)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_lookup ON docs (provider, qkey, seq)")
    conn.commit()
    return conn


def _iter_metadata(path: str):
    """Yield (uid, meta_dict) from a metadata.json (dict keyed by uid)."""
    if _HAVE_IJSON:
        with open(path, "rb") as f:
            for uid, meta in ijson.kvitems(f, ""):
                yield uid, meta
    else:
        with open(path, encoding="utf-8") as f:
            for uid, meta in json.load(f).items():
                yield uid, meta


def _iter_documents(path: str):
    """Yield each {uid,title,url,snippet} from a documents_*.json (list)."""
    if _HAVE_IJSON:
        with open(path, "rb") as f:
            for item in ijson.items(f, "item"):
                yield item
    else:
        with open(path, encoding="utf-8") as f:
            for item in json.load(f):
                yield item


def build_provider(conn: sqlite3.Connection, provider_dir: Path, provider: str,
                    seq_start: int) -> int:
    meta_path = provider_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  [{provider}] no metadata.json — skipping")
        return seq_start

    t0 = time.time()
    # uid -> qkey  (compact: we only need the key, not the full meta)
    print(f"  [{provider}] loading metadata ...", flush=True)
    uid2key: dict[str, str] = {}
    for uid, m in _iter_metadata(str(meta_path)):
        try:
            uid2key[uid] = _qkey(m.get("query", ""), m.get("count", 0) or 0, m.get("country"))
        except Exception:
            continue
    print(f"  [{provider}] {len(uid2key):,} metadata entries ({time.time()-t0:.0f}s)", flush=True)

    seq = seq_start
    inserted = 0
    batch: list[tuple] = []
    doc_files = sorted(glob.glob(str(provider_dir / "documents_*.json")))
    for df in doc_files:
        print(f"  [{provider}] streaming {os.path.basename(df)} ...", flush=True)
        for d in _iter_documents(df):
            uid = d.get("uid")
            qk = uid2key.get(uid)
            if qk is None:
                continue
            url = str(d.get("url") or "").strip()
            if not url:
                continue
            # benign extras only (never score/judge fields)
            extra = {k: v for k, v in d.items()
                     if k not in ("uid", *_CORE_FIELDS, "snippet") and not _is_score_field(k)}
            seq += 1
            batch.append((
                provider, qk, url,
                str(d.get("title") or ""),
                str(d.get("snippet") or d.get("description") or ""),
                json.dumps(extra, ensure_ascii=False) if extra else None,
                seq,
            ))
            if len(batch) >= _BATCH:
                conn.executemany(
                    "INSERT OR IGNORE INTO docs (provider,qkey,url,title,snippet,extra,seq) "
                    "VALUES (?,?,?,?,?,?,?)", batch)
                conn.commit(); inserted += len(batch); batch.clear()
    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO docs (provider,qkey,url,title,snippet,extra,seq) "
            "VALUES (?,?,?,?,?,?,?)", batch)
        conn.commit(); inserted += len(batch)
    print(f"  [{provider}] inserted ~{inserted:,} doc-rows ({time.time()-t0:.0f}s)", flush=True)
    return seq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="live_search_documents root dir")
    ap.add_argument("--db", required=True, help="output SQLite path")
    ap.add_argument("--providers", nargs="*", default=_DEFAULT_PROVIDERS)
    args = ap.parse_args()

    if not _HAVE_IJSON:
        print("WARNING: ijson not installed — falling back to json.load per file "
              "(needs several GB RAM). Recommend: pip install ijson", flush=True)

    src = Path(args.src)
    conn = _open_db(args.db)
    row = conn.execute("SELECT COALESCE(MAX(seq),0) FROM docs").fetchone()
    seq = int(row[0]) if row else 0

    for prov in args.providers:
        pdir = src / prov
        if not pdir.is_dir():
            print(f"  [{prov}] dir not found — skipping")
            continue
        seq = build_provider(conn, pdir, prov, seq)

    print("Finalizing (ANALYZE) ...", flush=True)
    conn.execute("ANALYZE")
    total = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    keys = conn.execute("SELECT COUNT(DISTINCT provider||qkey) FROM docs").fetchone()[0]
    conn.commit(); conn.close()
    print(f"DONE: {total:,} doc-rows across {keys:,} (provider,query) keys -> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
