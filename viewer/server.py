"""Arcticswarm BBS Trajectory Viewer - FastAPI server."""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import json
import re
import shutil
import subprocess
import threading
import webbrowser
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from parser import QuestionSummary, TimelineData, build_timeline, load_report

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Arcticswarm Viewer")

# Will be set in main()
RUN_DIR: Path = Path(".")
QUESTIONS: list[QuestionSummary] = []
QUESTIONS_BY_ID: dict[str, QuestionSummary] = {}

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def _dc_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts for JSON serialization."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dc_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_dc_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _dc_to_dict(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/questions")
def get_questions():
    """Return question list from report.json."""
    result = []
    for q in QUESTIONS:
        result.append({
            "conv_id": q.conv_id,
            "dataset": q.dataset,
            "question": q.question[:300],
            "duration_seconds": q.duration_seconds,
            "total_tokens": q.total_tokens,
            "judge_correct": q.judge_correct,
            "score": q.score,
            "has_error": q.has_error,
            "error": q.error,
            "had_timeout": q.had_timeout,
            "swarm_teammates_spawned": q.swarm_teammates_spawned,
            "swarm_bbs_message_count": q.swarm_bbs_message_count,
            "reference_answer": q.reference_answer,
        })
    return result


@lru_cache(maxsize=5)
def _cached_timeline(conv_id: str) -> TimelineData | None:
    return build_timeline(RUN_DIR, conv_id)


@app.get("/api/questions/{conv_id}/timeline")
def get_timeline(conv_id: str):
    """Return full parsed timeline + BBS snapshots for one question."""
    if conv_id not in QUESTIONS_BY_ID:
        raise HTTPException(404, f"Question {conv_id} not found")

    timeline = _cached_timeline(conv_id)
    if timeline is None:
        raise HTTPException(404, f"Trajectory not found for {conv_id}")

    # Enrich with report data
    q = QUESTIONS_BY_ID[conv_id]
    timeline.response_text = q.response_text

    # Serialize - convert bbs_snapshots keys from int to str for JSON
    data = _dc_to_dict(timeline)

    # Add judge/answer data from report
    data["reference_answer"] = q.reference_answer
    data["judge_correct"] = q.judge_correct
    data["judge_comment"] = q.judge_comment
    data["judge_raw_output"] = q.judge_raw_output
    data["score"] = q.score
    data["total_tokens"] = q.total_tokens
    data["duration_seconds"] = q.duration_seconds
    data["has_error"] = q.has_error
    data["had_timeout"] = q.had_timeout
    data["error"] = q.error

    # Trim large fields to keep response manageable
    # Keep tool_result_text but truncate very large ones
    for event in data["events"]:
        if event.get("tool_result_text") and len(event["tool_result_text"]) > 5000:
            event["tool_result_text"] = event["tool_result_text"][:5000] + "... (truncated)"
        if event.get("tool_input") and len(json.dumps(event["tool_input"], default=str)) > 5000:
            event["tool_input"] = {"_truncated": True, "summary": event.get("tool_input_summary", "")}

    return JSONResponse(data)


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Cloudflare quick tunnel
# ---------------------------------------------------------------------------


_TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def _start_cloudflare_tunnel(port: int) -> subprocess.Popen | None:
    """Start a Cloudflare quick tunnel for the given local port.

    Prints the public https://*.trycloudflare.com URL to stdout once it
    appears in cloudflared's logs. Returns the subprocess (or None if
    cloudflared is not installed). Cleans up on interpreter exit.
    """
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        print(
            "[tunnel] cloudflared not found on PATH — skipping public tunnel.\n"
            "[tunnel] Install with `brew install cloudflared` to share a public URL."
        )
        return None

    print(f"[tunnel] Starting Cloudflare quick tunnel for http://localhost:{port} ...")
    proc = subprocess.Popen(
        [cloudflared, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _reader() -> None:
        url_printed = False
        assert proc.stdout is not None
        for line in proc.stdout:
            if not url_printed:
                m = _TRYCLOUDFLARE_RE.search(line)
                if m:
                    url = m.group(0)
                    bar = "=" * 72
                    print(f"\n{bar}\n  PUBLIC URL: {url}\n  Share this with others to view the viewer remotely.\n{bar}\n", flush=True)
                    url_printed = True

    threading.Thread(target=_reader, daemon=True).start()

    def _cleanup() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_cleanup)
    return proc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    global RUN_DIR, QUESTIONS, QUESTIONS_BY_ID

    parser = argparse.ArgumentParser(description="Arcticswarm BBS Trajectory Viewer")
    parser.add_argument("run_dir", type=Path, help="Path to results directory")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Disable the Cloudflare quick tunnel (public URL).",
    )
    args = parser.parse_args()

    RUN_DIR = args.run_dir.resolve()
    if not (RUN_DIR / "report.json").exists():
        print(f"Error: {RUN_DIR / 'report.json'} not found")
        return

    print(f"Loading report from {RUN_DIR}...")
    QUESTIONS = load_report(RUN_DIR)
    QUESTIONS_BY_ID = {q.conv_id: q for q in QUESTIONS}
    print(f"Loaded {len(QUESTIONS)} questions")

    # Mount static files (after API routes are defined)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    import uvicorn

    if not args.no_tunnel:
        _start_cloudflare_tunnel(args.port)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
