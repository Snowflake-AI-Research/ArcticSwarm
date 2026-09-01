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

"""Tests for node-local cache mirror + delta-sync to a shared master.

Covers the fetch (entries, longest-wins) merge, the rowid delta watermark, and
CacheMirrorManager repointing the config to the local mirror. No network /
flock contention (single process, tmp files).
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from arcticswarm.tools import cache_sync as cs


def _make_fetch_db(path, rows):
    c = sqlite3.connect(str(path))
    c.execute("CREATE TABLE entries (key TEXT PRIMARY KEY, url TEXT NOT NULL, content TEXT NOT NULL,"
              " pages TEXT NOT NULL DEFAULT '', is_pdf INTEGER NOT NULL DEFAULT 0,"
              " via TEXT NOT NULL DEFAULT '', metadata TEXT NOT NULL DEFAULT '{}')")
    c.executemany("INSERT INTO entries(key,url,content) VALUES(?,?,?)", rows)
    c.commit(); c.close()


def test_fetch_full_merge_longest_wins(tmp_path):
    master = tmp_path / "m.sqlite"
    local = tmp_path / "l.sqlite"
    _make_fetch_db(master, [("A", "http://a", "short")])
    _make_fetch_db(local, [("A", "http://a", "a much longer body wins"), ("B", "http://b", "new row")])
    c = cs._Cache(str(master), str(local), table="entries", cols=cs._FETCH_COLS,
                  conflict=cs._FETCH_CONFLICT, order_col="rowid")
    c.watermark = 0  # full merge
    c.sync()
    m = sqlite3.connect(str(master))
    got = dict(m.execute("SELECT key, content FROM entries").fetchall())
    assert got["A"] == "a much longer body wins"   # longer won
    assert got["B"] == "new row"                    # new merged in


def test_fetch_delta_only(tmp_path):
    master = tmp_path / "m.sqlite"
    local = tmp_path / "l.sqlite"
    _make_fetch_db(master, [("A", "http://a", "a")])
    # local = copy of master (rowid 1) + one new row (rowid 2)
    _make_fetch_db(local, [("A", "http://a", "a")])
    lc = sqlite3.connect(str(local)); lc.execute("INSERT INTO entries(key,url,content) VALUES('B','http://b','b')"); lc.commit(); lc.close()
    c = cs._Cache(str(master), str(local), table="entries", cols=cs._FETCH_COLS,
                  conflict=cs._FETCH_CONFLICT, order_col="rowid")
    c.watermark = 1  # only rows with rowid > 1 (the new B)
    n = c.sync()
    m = sqlite3.connect(str(master))
    keys = {r[0] for r in m.execute("SELECT key FROM entries").fetchall()}
    assert keys == {"A", "B"} and n == 1


def test_manager_setup_repoints_config(tmp_path):
    master = tmp_path / "data" / "fetch_cache.sqlite"
    master.parent.mkdir(parents=True)
    _make_fetch_db(master, [("A", "http://a", "x")])
    local_dir = tmp_path / "local"
    cfg = SimpleNamespace(
        fetch_cache_path=str(master),
        cache_local_dir=str(local_dir), cache_sync_every=5,
    )
    mgr = cs.CacheMirrorManager(cfg, output_dir=str(tmp_path / "out"))
    mgr.setup()
    # config now points at the node-local mirror, which is a real copy of master
    assert cfg.fetch_cache_path == str(local_dir / "fetch_cache.sqlite")
    assert (local_dir / "fetch_cache.sqlite").exists()
    c = sqlite3.connect(cfg.fetch_cache_path)
    assert c.execute("SELECT count(*) FROM entries").fetchone()[0] == 1


def test_storage_present(tmp_path):
    # an existing dir (and any descendant path under it) is "present"
    assert cs.storage_present(str(tmp_path))
    assert cs.storage_present(str(tmp_path / "does" / "not" / "exist" / "yet"))
    # a path whose mount is absent is not present (no real ancestor below root)
    assert not cs.storage_present("/this-mount-should-not-exist-12345/x/y")


def test_should_auto_mirror_gating(tmp_path):
    base = dict(cache_local_mirror=True, cache_local_dir=str(tmp_path),
                fetch_cache_path=str(tmp_path / "f.sqlite"))
    # allowed + caching active + storage present -> engage
    assert cs.should_auto_mirror(SimpleNamespace(**base))
    # explicit off -> never
    assert not cs.should_auto_mirror(SimpleNamespace(**{**base, "cache_local_mirror": False}))
    # no cache configured at all -> nothing to mirror
    assert not cs.should_auto_mirror(SimpleNamespace(**{**base, "fetch_cache_path": ""}))
    # caching on but node-local storage absent -> skip (fall back to /data)
    assert not cs.should_auto_mirror(SimpleNamespace(**{
        **base, "cache_local_dir": "/no-such-mount-98765/cache"}))
