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

"""Bulletin Board System (BBS) — shared communication channel for swarm agents.

The BBS is the central communication primitive for multi-agent swarms.
Discoveries are broadly useful, consensus needs visibility, and
observability is first-class.

Thread safety: writes are serialised via ``threading.Lock``; reads return
immutable snapshots so agents never see partial updates.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BBS channels
# ---------------------------------------------------------------------------

CHANNEL_DISCOVERIES = "discoveries"
CHANNEL_KEY_FINDINGS = "key-findings"
CHANNEL_CONSENSUS = "consensus"
CHANNEL_TASKS = "tasks"
CHANNEL_DISCUSSION = "discussion"

ALL_CHANNELS = frozenset({
    CHANNEL_DISCOVERIES,
    CHANNEL_KEY_FINDINGS,
    CHANNEL_CONSENSUS,
    CHANNEL_TASKS,
    CHANNEL_DISCUSSION,
})


# ---------------------------------------------------------------------------
# Verdict semantics
# ---------------------------------------------------------------------------

# Affirmative-verification markers — a ``#consensus`` post carrying one of
# these (and no overriding negative marker) is treated as a VERIFIED verdict.
_VERDICT_POS_MARKERS = (
    "VERIFIED",
    "CONFIRMED",
    "ALL CONSTRAINTS",
    "AUDIT COMPLETE",
    "REVIEWED",
)

# Negative / non-affirming consensus language. A reviewer that posts a
# challenge, disqualification, or dead-end to ``#consensus`` (instead of the
# affirmative template) does NOT count as a VERIFIED verdict.
_VERDICT_NEG_MARKERS = (
    "CHALLENGE",
    "DISQUALIF",
    "CANNOT BE SATISFIED",
    "CANNOT BE VERIFIED",
    "NOT VERIFIED",
    "NO VERIFIED",
    "NO CANDIDATE",
    "NO KNOWN",
    "UNABLE TO",
    "DESPITE VERIFICATION GAPS",
    # Catches bare "unverified" too — note "VERIFIED" is a substring of
    # "UNVERIFIED", so without this a negative post ("constraint 3 is
    # unverified") would match the positive marker and wrongly count as a
    # VERIFIED verdict. Subsumes the earlier "REMAINS UNVERIFIED".
    "UNVERIFIED",
    "IS WRONG",
    "NO ANSWER",
    "UNSOLVABLE",
    "WITHDRAWN",
    "REJECTED",
)


def is_verified_consensus_verdict(content: str) -> bool:
    """True if a ``#consensus`` post is an affirming VERIFIED verdict.

    Reviewers are instructed (see ``IDLE_REVIEW_MESSAGE_RESEARCH_ADVERSARIAL``
    in :mod:`arcticswarm.swarm.prompts`) to post affirmations to ``#consensus``
    ("Reviewed [candidate] — all constraints verified with evidence") and to
    route CHALLENGE / ALTERNATIVE verdicts to ``#discussion``. In practice a
    few negative / dead-end consensus posts ("Task is Unsolvable",
    "DISQUALIFIED", "Withdrawn") still appear; this filter keeps only the
    affirmative ones so the reviewer-diversity gate counts only true verdicts.
    """
    if not content:
        return False
    u = content.upper()
    if not any(m in u for m in _VERDICT_POS_MARKERS):
        return False
    if not any(m in u for m in _VERDICT_NEG_MARKERS):
        return True
    # Both positive and negative language present. An explicit challenge /
    # disqualification header wins; a strong "all constraints verified"
    # affirmation is trusted; otherwise treat as a non-verdict (conservative).
    head = u[:80]
    if "CHALLENGE" in head or "DISQUALIF" in head or "UNSOLVABLE" in head:
        return False
    if "ALL CONSTRAINTS VERIFIED" in u or "ALL CONSTRAINTS ARE VERIFIED" in u:
        return True
    return False


# ---------------------------------------------------------------------------
# BBS message
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBSMessage:
    """A single BBS post.  Immutable once created."""

    id: str
    channel: str
    author: str
    timestamp: float
    content: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    in_reply_to: str | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d

    def summary(self, max_len: int = 120) -> str:
        """One-line human-readable summary for the viewer."""
        text = self.content.replace("\n", " ")
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        return f"[{self.channel}] {self.author}: {text}"


# ---------------------------------------------------------------------------
# BBS
# ---------------------------------------------------------------------------


class BBS:
    """Thread-safe shared bulletin board for swarm agents.

    All writes go through :meth:`post`, which acquires the lock and appends.
    Reads return a snapshot list (copy), so callers can iterate safely.
    """

    def __init__(
        self,
        on_post: Callable[[BBSMessage], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._messages: list[BBSMessage] = []
        # Monotonic counter for fast "since" queries
        self._next_seq: int = 0
        # Optional observer fired AFTER each post commits.  Used by the
        # candidate-emergence disagreement gate.  Wrapped
        # in try/except at call time so observers can never break BBS writes.
        self._on_post: Callable[[BBSMessage], None] | None = on_post

    def set_on_post(self, on_post: Callable[[BBSMessage], None] | None) -> None:
        """Register or clear the post-commit observer.  Idempotent."""
        self._on_post = on_post

    # -- writing -------------------------------------------------------------

    def post(
        self,
        *,
        channel: str,
        author: str,
        content: str,
        structured_data: dict[str, Any] | None = None,
        in_reply_to: str | None = None,
        tags: list[str] | None = None,
    ) -> BBSMessage:
        """Post a new message.  Returns the created message."""
        sd = dict(structured_data) if structured_data else {}

        msg = BBSMessage(
            id=uuid.uuid4().hex[:12],
            channel=channel,
            author=author,
            timestamp=time.monotonic(),
            content=content,
            structured_data=sd,
            in_reply_to=in_reply_to,
            tags=tuple(tags or []),
        )
        with self._lock:
            self._messages.append(msg)
            self._next_seq += 1
        # Fire the observer outside the lock so callbacks can take their
        # time without blocking other writers.  Errors are isolated.
        if self._on_post is not None:
            try:
                self._on_post(msg)
            except Exception:
                logger.exception("BBS on_post observer raised; ignored")
        return msg

    # -- reading (snapshot-based) --------------------------------------------

    def read(
        self,
        *,
        channel: str | None = None,
        tags: list[str] | None = None,
        since_id: str | None = None,
        limit: int = 50,
    ) -> list[BBSMessage]:
        """Read messages, optionally filtered.  Returns a snapshot (copy)."""
        with self._lock:
            msgs = list(self._messages)

        # Filter by since_id (messages posted after that ID)
        if since_id:
            found = False
            filtered: list[BBSMessage] = []
            for m in msgs:
                if found:
                    filtered.append(m)
                elif m.id == since_id:
                    found = True
            msgs = filtered

        if channel:
            msgs = [m for m in msgs if m.channel == channel]

        if tags:
            tag_set = set(tags)
            msgs = [m for m in msgs if tag_set.intersection(m.tags)]

        # Return latest N
        return msgs[-limit:]

    def read_all(self) -> list[BBSMessage]:
        """Return a snapshot of every message."""
        with self._lock:
            return list(self._messages)

    @property
    def message_count(self) -> int:
        with self._lock:
            return len(self._messages)

    # -- serialisation -------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the full BBS history as JSON."""
        with self._lock:
            msgs = list(self._messages)
        return json.dumps([m.to_dict() for m in msgs], indent=2, default=str)

    def export(self, path: Path) -> Path:
        """Write BBS history to a JSON file.  Returns the file path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    # -- text rendering (for LLM consumption) --------------------------------

    def render_for_llm(
        self,
        *,
        channel: str | None = None,
        limit: int = 50,
    ) -> str:
        """Render BBS messages as a text block suitable for LLM context."""
        msgs = self.read(channel=channel, limit=limit)
        if not msgs:
            return "(no messages on the BBS yet)"

        lines: list[str] = []
        for m in msgs:
            header = f"[{m.channel}] {m.author}"
            if m.in_reply_to:
                header += f" (re: {m.in_reply_to})"
            if m.tags:
                header += f" tags={list(m.tags)}"
            lines.append(header)
            lines.append(m.content)
            if m.structured_data:
                lines.append(f"  data: {json.dumps(m.structured_data, default=str)}")
            lines.append("")
        return "\n".join(lines)
