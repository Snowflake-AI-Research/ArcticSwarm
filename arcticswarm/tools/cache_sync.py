"""Node-local cache mirror + periodic delta-sync to a shared (Lustre) master.

WHY: SQLite WAL mode mmaps a ``-shm`` index that is host-local; on a network
filesystem (Lustre) shared across hosts, a writer on another host invalidates
the mapping and any access raises **SIGBUS** (or deadlocks on cross-host
``flock``s). So when an eval runs on multiple hosts with caches enabled,
pointing every node at one writable WAL SQLite on a shared filesystem crashes.

FIX (this module): when ``web.cache_local_mirror`` is on, each node:
  1. **mirrors** the shared master cache (mostly-read) to a node-local working
     copy on a fast local disk (``cache_local_dir``, xfs) at startup. All reads
     and write-throughs hit the LOCAL WAL DB — safe (single host).
  2. periodically **syncs deltas** (only rows added since the last sync) from
     the local copy back into the shared master, in rollback-journal (DELETE)
     mode under an exclusive ``flock`` — so the master is never written with
     WAL/``-shm`` mmap and never by two hosts at once (no SIGBUS).

A background daemon thread triggers the sync every ``cache_sync_every``
completed cases (counted from the run's ``trajectories/`` dir), plus a final
sync at process exit. The master must be in DELETE journal mode (checkpoint it
once: ``PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE``).

Deltas auto-sync back during the eval run (see ``CacheMirrorManager``). See
ENVIRONMENT.md for the cache env vars (paths, optional S3 bucket/prefix).
"""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Per-cache merge spec: table, conflict clause, and the monotonic column used
# to select "new since last sync" rows.
#   fetch entries: PK(key); longer content wins; ordered by implicit rowid.
_FETCH_COLS = ("key", "url", "content", "pages", "is_pdf", "via", "metadata")
_FETCH_CONFLICT = (
    " ON CONFLICT(key) DO UPDATE SET"
    "   content=excluded.content, url=excluded.url, pages=excluded.pages,"
    "   is_pdf=excluded.is_pdf, via=excluded.via, metadata=excluded.metadata"
    " WHERE length(excluded.content) > length(entries.content)"
)


def storage_present(path: str) -> bool:
    """True if some existing ancestor directory of ``path`` exists below root.

    Used to decide whether there is real node-local fast storage (e.g.
    ``/data-fast``) to mirror into. The leaf cache dir need not exist yet — it's
    enough that its fast-disk mount does. Returns False on a dev box / CPU pod
    where no such mount is present (so the caller falls back to the shared
    master directly instead of erroring on a missing mount).
    """
    p = os.path.abspath(path or "/")
    while p and p != os.path.dirname(p):
        if os.path.isdir(p):
            return True
        p = os.path.dirname(p)
    return False


def should_auto_mirror(config) -> bool:
    """Decide whether to engage the node-local cache mirror for this run.

    True iff the mirror is allowed (``web.cache_local_mirror``, default True),
    caching is active (a fetch cache path is set), and node-local fast storage
    (an ancestor of ``cache_local_dir``) actually exists. This keeps the mirror
    automatic on cluster nodes while making it a silent no-op on machines
    without a fast-disk mount.
    """
    if not getattr(config, "cache_local_mirror", True):
        return False
    caching_active = bool((getattr(config, "fetch_cache_path", "") or "").strip())
    if not caching_active:
        return False
    local_dir = (getattr(config, "cache_local_dir", "") or "cache").strip()
    return storage_present(local_dir)


def _flock(path: Path, exclusive: bool):
    """Context manager: flock a lockfile (LOCK_EX for writes, LOCK_SH for reads).

    Lustre honors flock (mount option), so this serializes master access across
    hosts. Degrades to a no-op lock object if the lockfile can't be opened.
    """
    class _L:
        def __enter__(self_):
            self_.f = open(path, "w")
            fcntl.flock(self_.f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            return self_

        def __exit__(self_, *a):
            try:
                fcntl.flock(self_.f, fcntl.LOCK_UN)
                self_.f.close()
            except Exception:
                pass
    return _L()


def _max_order(db: Path, table: str, order_col: str) -> int:
    try:
        c = sqlite3.connect(str(db), timeout=60.0)
        try:
            row = c.execute(f"SELECT COALESCE(MAX({order_col}),0) FROM {table}").fetchone()
            return int(row[0]) if row else 0
        finally:
            c.close()
    except Exception:
        return 0


class _Cache:
    """One mirrored cache (the fetch cache)."""

    def __init__(self, master: str, local: str, *, table: str, cols, conflict: str, order_col: str):
        self.master = Path(master).expanduser()
        self.local = Path(local).expanduser()
        self.table = table
        self.cols = cols
        self.conflict = conflict
        self.order_col = order_col
        self.lockpath = self.master.with_suffix(self.master.suffix + ".synclock")
        self.watermark = 0
        self._sync_lock = threading.Lock()  # serialize syncs within this process

    def mirror(self) -> str:
        """Copy master -> local if local is missing/older. Returns local path."""
        self.local.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.master.exists():
                need = (not self.local.exists()) or (
                    self.master.stat().st_mtime > self.local.stat().st_mtime + 1
                )
                if need:
                    log.info("cache mirror: copying %s -> %s (one-time per node)", self.master, self.local)
                    tmp = str(self.local) + ".tmp"
                    # shared lock: many nodes may copy concurrently; a merge (LOCK_EX) waits.
                    with _flock(self.lockpath, exclusive=False):
                        shutil.copyfile(str(self.master), tmp)
                    os.replace(tmp, str(self.local))
                    # a fresh copy => drop any stale local WAL/shm so the local opens clean
                    for sfx in ("-wal", "-shm"):
                        try:
                            os.remove(str(self.local) + sfx)
                        except OSError:
                            pass
        except Exception as exc:  # noqa: BLE001
            log.warning("cache mirror: copy of %s failed (%s); using local as-is/empty", self.master, exc)
        self.watermark = _max_order(self.local, self.table, self.order_col)
        return str(self.local)

    def sync(self) -> int:
        """Merge local rows added since the last sync into the master. Returns rows seen."""
        with self._sync_lock:
            if not self.local.exists():
                return 0
            new_wm = _max_order(self.local, self.table, self.order_col)
            if new_wm <= self.watermark:
                return 0
            collist = ", ".join(self.cols)
            sql = (
                f"INSERT INTO main.{self.table}({collist}) "
                f"SELECT {collist} FROM loc.{self.table} WHERE {self.order_col} > ?"
                + self.conflict
            )
            try:
                with _flock(self.lockpath, exclusive=True):
                    m = sqlite3.connect(str(self.master), timeout=120.0)
                    try:
                        m.execute("PRAGMA journal_mode=DELETE")   # NO wal/-shm on the shared master
                        m.execute("PRAGMA synchronous=NORMAL")
                        m.execute("PRAGMA busy_timeout=120000")
                        m.execute("ATTACH DATABASE ? AS loc", (str(self.local),))
                        m.execute(sql, (self.watermark,))
                        m.commit()
                        m.execute("DETACH DATABASE loc")
                    finally:
                        m.close()
                n = new_wm - self.watermark
                self.watermark = new_wm
                log.info("cache sync: merged ~%d new %s rows into %s", n, self.table, self.master)
                return n
            except Exception as exc:  # noqa: BLE001 - never let sync break the run
                log.warning("cache sync: merge into %s failed: %s", self.master, exc)
                return 0


class CacheMirrorManager:
    """Mirror enabled caches to node-local disk and sync deltas back periodically.

    Use: build from the resolved config, call :meth:`setup` (mutates the config
    so the cache layer uses the local mirrors), then :meth:`start` to launch the
    background sync thread + register a final sync at exit.
    """

    def __init__(self, config, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.local_dir = (getattr(config, "cache_local_dir", "") or "cache").strip()
        self.sync_every = max(1, int(getattr(config, "cache_sync_every", 5) or 5))
        self.caches: list[_Cache] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def setup(self) -> None:
        """Mirror each enabled cache to node-local disk and repoint the config."""
        Path(self.local_dir).mkdir(parents=True, exist_ok=True)
        fetch_master = (getattr(self.config, "fetch_cache_path", "") or "").strip()
        if fetch_master:
            local = os.path.join(self.local_dir, "fetch_cache.sqlite")
            c = _Cache(fetch_master, local, table="entries", cols=_FETCH_COLS,
                       conflict=_FETCH_CONFLICT, order_col="rowid")
            c.mirror()
            self.config.fetch_cache_path = local  # cache layer now uses the local copy
            self.caches.append(c)
        log.info(
            "cache mirror: %d cache(s) mirrored to %s; delta-sync every %d cases",
            len(self.caches), self.local_dir, self.sync_every,
        )

    def _traj_count(self) -> int:
        try:
            return len(os.listdir(os.path.join(self.output_dir, "trajectories")))
        except OSError:
            return 0

    def _loop(self) -> None:
        last_synced_at = 0
        while not self._stop.is_set():
            self._stop.wait(20)
            done = self._traj_count()
            if done - last_synced_at >= self.sync_every:
                last_synced_at = done
                for c in self.caches:
                    c.sync()

    def start(self) -> None:
        if not self.caches:
            return
        self._thread = threading.Thread(target=self._loop, name="cache-sync", daemon=True)
        self._thread.start()
        atexit.register(self.final_sync)

    def final_sync(self) -> None:
        self._stop.set()
        for c in self.caches:
            c.sync()

    def setup_and_start(self) -> None:
        self.setup()
        self.start()
