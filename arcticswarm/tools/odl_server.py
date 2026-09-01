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

"""Auto-managed opendataloader-pdf hybrid backend server.

Starts a ``opendataloader-pdf-hybrid`` server on a random free port as a
subprocess.  The server is shared across all PdfReadTool instances within
the same process and is cleaned up on exit (atexit, SIGINT, SIGTERM).

Cross-process reuse: a port file at ``/tmp/arcticswarm_odl_hybrid.json``
records the running server's PID and port so that subsequent runs (or
parallel processes) reuse the existing server instead of spawning new
Java/Tika processes that exhaust system memory.
"""

from __future__ import annotations

import atexit
import json as _json
import logging
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

# How long to wait for the backend to become ready (model loading).
_STARTUP_TIMEOUT = 120  # seconds
_HEALTH_CHECK_INTERVAL = 2  # seconds

# Port file for cross-process server discovery / reuse.
_PORT_FILE = os.path.join(tempfile.gettempdir(), "arcticswarm_odl_hybrid.json")


def _find_free_port() -> int:
    """Find a free TCP port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_server_ready(url: str) -> bool:
    """Check if the hybrid backend is responding."""
    try:
        resp = requests.get(f"{url}/health", timeout=3)
        return resp.ok
    except Exception:
        pass
    # Some versions don't have /health — try a simple TCP connect
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        with socket.create_connection((parsed.hostname, parsed.port), timeout=3):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Port file helpers — cross-process server discovery
# ---------------------------------------------------------------------------

def _write_port_file(pid: int, port: int) -> None:
    """Record the running server's PID and port for cross-process reuse."""
    try:
        with open(_PORT_FILE, "w") as f:
            _json.dump({"pid": pid, "port": port}, f)
    except OSError:
        pass  # non-fatal


def _remove_port_file() -> None:
    """Remove the port file (best-effort)."""
    try:
        os.remove(_PORT_FILE)
    except OSError:
        pass


def _try_reuse_existing() -> str | None:
    """Check if an ODL server from a previous run is still alive.

    Reads the port file, verifies the PID is alive, then health-checks
    the HTTP endpoint.  Returns the base URL if reusable, else ``None``.
    """
    try:
        with open(_PORT_FILE) as f:
            info = _json.load(f)
        pid = info.get("pid")
        port = info.get("port")
        if not pid or not port:
            return None
    except (FileNotFoundError, _json.JSONDecodeError, KeyError):
        return None

    # Check if the process is still alive (signal 0 = existence check)
    try:
        os.kill(pid, 0)
    except OSError:
        # Process is dead — clean up stale port file
        _remove_port_file()
        return None

    # Process alive — health-check the HTTP endpoint
    url = f"http://127.0.0.1:{port}"
    if _is_server_ready(url):
        log.info(
            "Reusing existing opendataloader-pdf-hybrid on port %d (pid %d)",
            port, pid,
        )
        return url

    # PID alive but server not responding (could be a recycled PID).
    _remove_port_file()
    return None


class ODLHybridServer:
    """Manages a single opendataloader-pdf-hybrid subprocess.

    When ``owned=True`` (default), the server is stopped on cleanup.
    When ``owned=False`` (reused from another process), we leave the
    process running so other sessions can keep using it.
    """

    def __init__(
        self,
        port: int,
        proc: subprocess.Popen | None,
        url: str,
        *,
        owned: bool = True,
    ) -> None:
        self.port = port
        self.proc = proc
        self.url = url
        self.owned = owned

    def is_alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        # Reused server — check via health endpoint
        return _is_server_ready(self.url)

    def stop(self) -> None:
        """Gracefully stop the server (only if we own it)."""
        if not self.owned or self.proc is None:
            return
        if not self.is_alive():
            return
        log.info("Stopping opendataloader-pdf-hybrid server (port %d, pid %d)", self.port, self.proc.pid)
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("Hybrid server did not stop gracefully, killing (pid %d)", self.proc.pid)
            self.proc.kill()
            self.proc.wait(timeout=5)
        _remove_port_file()


# Module-level singleton: one server per process.
_active_server: ODLHybridServer | None = None
_cleanup_registered = False
_server_lock = threading.Lock()


def _cleanup_server() -> None:
    """atexit / signal handler — stop the server if running."""
    global _active_server
    if _active_server is not None:
        _active_server.stop()
        _active_server = None


def invalidate_server() -> None:
    """Forget the cached server handle so the next ensure_server() re-resolves.

    Call this when a conversion request fails with a connection error — the
    cached URL may point at a process that has died or been recycled.
    """
    global _active_server
    with _server_lock:
        if _active_server is not None and _active_server.owned:
            try:
                _active_server.stop()
            except Exception:
                pass
        _active_server = None


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle SIGINT/SIGTERM by stopping the server, then re-raising."""
    _cleanup_server()
    # Re-raise the signal with default handler so the process exits normally
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)


def ensure_server(
    *,
    force_ocr: bool = False,
    ocr_lang: str = "",
) -> str:
    """Ensure a hybrid backend server is running. Returns its base URL.

    Resolution order:
    1. In-process singleton (``_active_server``) — fastest.
    2. Cross-process reuse via port file — avoids spawning duplicate Java
       processes that exhaust system memory on laptops.
    3. Start a new server on a random free port.

    The server is automatically stopped on process exit or Ctrl+C.
    """
    global _active_server, _cleanup_registered

    with _server_lock:
        # 1. Already running in this process? Trust the cached handle.
        # For an owned server, a cheap proc.poll() check; for a reused server
        # (proc=None), trust it — the caller invokes invalidate_server() if a
        # real conversion request fails. This avoids a /health round-trip on
        # every PDF.
        if _active_server is not None:
            if _active_server.proc is not None:
                if _active_server.is_alive():
                    return _active_server.url
                # Owned server died — fall through to re-create.
                _active_server = None
            else:
                return _active_server.url

        # 2. Reuse a server started by a previous / parallel process?
        existing_url = _try_reuse_existing()
        if existing_url is not None:
            from urllib.parse import urlparse
            parsed = urlparse(existing_url)
            _active_server = ODLHybridServer(
                port=parsed.port,
                proc=None,
                url=existing_url,
                owned=False,
            )
            return existing_url

        # 3. Start a new server.
        # Check that the command exists
        if shutil.which("opendataloader-pdf-hybrid") is None:
            log.warning("opendataloader-pdf-hybrid not found in PATH — hybrid mode unavailable")
            raise FileNotFoundError("opendataloader-pdf-hybrid not found in PATH")

        port = _find_free_port()
        url = f"http://127.0.0.1:{port}"

        cmd = ["opendataloader-pdf-hybrid", "--port", str(port)]
        if force_ocr:
            cmd.append("--force-ocr")
        if ocr_lang:
            cmd.extend(["--ocr-lang", ocr_lang])

        log.info("Starting opendataloader-pdf-hybrid on port %d (cmd: %s)", port, " ".join(cmd))

        env = os.environ.copy()
        env.pop("MallocStackLogging", None)
        env.pop("MallocStackLoggingNoCompact", None)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            # Start in its own process group so it doesn't get SIGINT from the terminal
            # (we handle cleanup ourselves).
            preexec_fn=lambda: __import__("os").setpgrp(),
        )

        # Register cleanup handlers (once per process)
        if not _cleanup_registered:
            atexit.register(_cleanup_server)
            # Signal handlers only work from the main thread; eval workers call
            # from pool threads so we guard with a check.
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, _signal_handler)
                signal.signal(signal.SIGTERM, _signal_handler)
            _cleanup_registered = True

        # Wait for server to become ready
        log.info("Waiting for hybrid backend to load models (up to %ds)...", _STARTUP_TIMEOUT)
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        ready = False
        elapsed_logged = 0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr_output = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                log.error("Hybrid backend crashed during startup (exit code %s): %s",
                          proc.returncode, stderr_output[:500])
                raise RuntimeError(
                    f"opendataloader-pdf-hybrid exited with code {proc.returncode} "
                    f"during startup. stderr: {stderr_output[:500]}"
                )
            if _is_server_ready(url):
                ready = True
                break
            elapsed = int(time.monotonic() - (deadline - _STARTUP_TIMEOUT))
            if elapsed >= elapsed_logged + 10:
                log.info("Still waiting for hybrid backend... (%ds elapsed)", elapsed)
                elapsed_logged = elapsed
            time.sleep(_HEALTH_CHECK_INTERVAL)

        if not ready:
            proc.kill()
            proc.wait()
            log.error("Hybrid backend did not become ready within %ds on port %d", _STARTUP_TIMEOUT, port)
            raise TimeoutError(
                f"opendataloader-pdf-hybrid did not become ready within {_STARTUP_TIMEOUT}s on port {port}"
            )

        log.info("opendataloader-pdf-hybrid ready on port %d (pid %d)", port, proc.pid)
        _active_server = ODLHybridServer(port=port, proc=proc, url=url)

        # Write port file so other processes can reuse this server
        _write_port_file(proc.pid, port)

        return url


def stop_server() -> None:
    """Explicitly stop the managed server (e.g. at end of eval run)."""
    _cleanup_server()
