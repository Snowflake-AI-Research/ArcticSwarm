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

"""One-time restore of cache directories from S3 when they're missing.

General mechanism: when a cache the run depends on (the fetch cache) is absent
at process start, and an S3 mirror is configured, restore it with a single
``aws s3 sync`` before opening. Disabled by default — a no-op unless
``ARCTICSWARM_CACHE_S3_BUCKET`` is set.

The S3 layout mirrors the absolute local path under a fixed bucket/prefix, so
caches kept under one root restore with a single sync:

    <cache-root>  <->  s3://<bucket>/<prefix><cache-root>

Each cache calls :func:`ensure_cache_restored` with its PARENT dir as the sync
target, so caches sharing a root resolve to the same sync and it runs at most
once per process.

This NEVER raises and is a no-op when: the cache is already present (e.g. on a
workstation where the path is a symlink to a persistent disk), it was already
attempted this process, the ``aws`` CLI is missing, the bucket is unset
(default), or the sync fails — the cache then simply starts empty/disabled,
exactly as before this fallback existed. Enable / override via env:
``ARCTICSWARM_CACHE_S3_BUCKET`` (empty disables) and
``ARCTICSWARM_CACHE_S3_PREFIX``. See ENVIRONMENT.md for the cache env vars
(bucket/prefix/paths).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_BUCKET = ""
_DEFAULT_PREFIX = ""
_SYNC_TIMEOUT = 3600  # seconds — caches can be tens of GB over the cluster net

_lock = threading.Lock()
_attempted: set[str] = set()


def _bucket(override: str | None) -> str:
    if override is not None:
        return override
    return os.environ.get("ARCTICSWARM_CACHE_S3_BUCKET", _DEFAULT_BUCKET)


def _prefix(override: str | None) -> str:
    if override is not None:
        return override
    return os.environ.get("ARCTICSWARM_CACHE_S3_PREFIX", _DEFAULT_PREFIX)


def build_s3_uri(sync_target: str | Path, *, bucket: str | None = None, prefix: str | None = None) -> str:
    """s3 URI mirroring the absolute local ``sync_target`` path."""
    b = _bucket(bucket)
    p = _prefix(prefix).strip("/")
    rel = str(Path(sync_target)).lstrip("/")
    head = f"{p}/" if p else ""
    return f"s3://{b}/{head}{rel}"


def _present(present_path: Path) -> bool:
    """True if the cache resource exists (a file, or a non-empty dir)."""
    try:
        if present_path.is_file():
            return True
        if present_path.is_dir():
            return any(present_path.iterdir())
    except OSError:
        pass
    return False


def ensure_cache_restored(
    sync_target: str | Path,
    *,
    present_path: str | Path | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
) -> bool:
    """If the cache at ``present_path`` is missing, restore ``sync_target`` from S3.

    ``sync_target`` is the local path passed to ``aws s3 sync`` (and mirrored to
    S3); ``present_path`` is the file/dir whose absence triggers the restore
    (defaults to ``sync_target``). Returns whether the cache is present after.
    Runs at most once per ``sync_target`` per process.
    """
    sync_target = Path(sync_target).expanduser()
    present = Path(present_path).expanduser() if present_path is not None else sync_target
    key = str(sync_target)

    with _lock:
        if key in _attempted:
            return _present(present)
        if _present(present):
            _attempted.add(key)
            return True
        b = _bucket(bucket)
        if not b:
            _attempted.add(key)
            return False
        if shutil.which("aws") is None:
            log.warning("cache restore: aws CLI not found; cannot restore %s", sync_target)
            _attempted.add(key)
            return False

        uri = build_s3_uri(sync_target, bucket=bucket, prefix=prefix)
        # Make the parent exist so aws can create the target; do NOT pre-create
        # sync_target itself (it may be a file path, e.g. the fetch cache DB).
        try:
            sync_target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        log.info("cache restore: %s absent — aws s3 sync %s -> %s", present, uri, sync_target)
        try:
            subprocess.run(
                ["aws", "s3", "sync", uri, str(sync_target)],
                check=True, timeout=_SYNC_TIMEOUT,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            log.warning("cache restore: aws s3 sync failed (%s): %s", uri, exc)
            _attempted.add(key)
            return _present(present)

        _attempted.add(key)
        ok = _present(present)
        log.info("cache restore: %s after S3 sync of %s", "present" if ok else "still absent", sync_target)
        return ok
