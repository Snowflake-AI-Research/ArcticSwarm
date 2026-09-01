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

"""Answer verification: candidate-emergence rival sweep.

Extracted from :mod:`arcticswarm.swarm.orchestrator` to keep the orchestrator
focused on the main control loop.  This module owns the candidate-emergence
hook:

- ``wire_candidate_emergence_hook``: BBS observer that posts a rival-sweep
  task the first time a candidate emerges (single-shot per turn).

The ``orch`` parameter is the ``SwarmOrchestrator`` instance — kept as an
explicit handle so we can read/write its single-shot ``_rival_sweep_fired``
flag without converting it to a module global.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, TYPE_CHECKING

from arcticswarm.swarm.bbs import BBS

if TYPE_CHECKING:
    from arcticswarm.swarm.orchestrator import SwarmOrchestrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate-emergence BBS hook
# ---------------------------------------------------------------------------


def wire_candidate_emergence_hook(
    orch: "SwarmOrchestrator",
    *,
    bbs: BBS,
    task_board: Any,
    question_text: str,
) -> None:
    """Register the BBS observer that fires the rival sweep on first
    qualifying candidate post.

    Single-shot per swarm turn.  Posts a high-priority rival-sweep
    task to the board for an existing browsing subagent to pick up.
    Robust to no-pickup (the task simply sits unclaimed).

    Thread-safety: BBS posts can arrive concurrently from multiple
    subagents.  We use a lock around the check-and-set so two near-
    simultaneous candidate posts don't both fire a rival sweep.
    """
    from arcticswarm.swarm.task import TaskSpec

    candidate_channels = {
        "discoveries", "key-findings", "consensus",
    }
    min_chars = int(getattr(
        orch.config, "candidate_emergence_min_chars", 60,
    ))
    max_turns = int(getattr(
        orch.config, "candidate_emergence_max_turns", 8,
    ))
    gate_lock = threading.Lock()

    def _on_post(msg: Any) -> None:
        # Cheap pre-checks outside the lock.
        if msg.channel not in candidate_channels:
            return
        if (msg.author or "").startswith("rival-sweep"):
            return
        if not msg.content or len(msg.content.strip()) < min_chars:
            return
        # Atomic check-and-set so concurrent posts can't double-fire.
        with gate_lock:
            if orch._rival_sweep_fired:
                return
            orch._rival_sweep_fired = True
        try:
            sweep_prompt = (
                "## Rival-Candidate Sweep\n\n"
                "An initial candidate just emerged on BBS.  Before the "
                "swarm commits, find at least 3 plausible alternative "
                "candidates that match the question's hard constraints. "
                "Do NOT prefer fame; obscure answers are common.\n\n"
                f"## Question\n{question_text[:1500]}\n\n"
                f"## Already-surfaced candidate (do not duplicate)\n"
                f"{(msg.content or '')[:600]}\n\n"
                "## Instructions\n"
                "1. Identify 1-2 hard constraints that most narrowly "
                "distinguish the answer (specific dates, places, "
                "unusual biographical details).\n"
                "2. Run web searches that EXCLUDE the surfaced "
                "candidate's name; bias queries toward less-famous "
                "matches.\n"
                "3. For each rival you find, post ONE BBS message to "
                "#discoveries in this format: "
                "`RIVAL: <name> | obscurity_hint: <wikipedia/news mention level> | "
                "matches: <which constraints>`.\n"
                f"4. Cap web searches at {max_turns}. Do NOT commit a "
                "final answer.  When done, complete the task with a "
                "one-sentence summary."
            )
            spec = TaskSpec(
                id=f"rival-sweep-{uuid.uuid4().hex[:8]}",
                name="alternative-candidate-sweep",
                prompt=sweep_prompt,
                profile="browsing",
                # ``alt=True`` so this contrarian rival sweep is detected as an
                # alternative task by ``task_is_alt`` — it satisfies the
                # premature-commitment gate (so no redundant auto-spawn) and is
                # counted as an alt task in trajectories.
                metadata={"source": "candidate_emergence_gate", "alt": True},
            )
            task_board.add_task(spec)
            # alt_task_force_dispatch: add_task alone is PASSIVE — it relies
            # on a browsing worker going idle and pulling the task, which often
            # never happens on a saturated vLLM (workers stay busy until shutdown),
            # so the rival sweep dies pending with 0 tool_uses (~40% of cases).
            # When enabled, also actively dispatch via ctx.spawn_or_assign (assigns
            # an idle worker or spawns a new one up to max_subagents, else queues),
            # mirroring the working enforce_alt_task gate (_spawn_contrarian_task).
            dispatched = None
            if getattr(orch.config, "alt_task_force_dispatch", False):
                ctx = getattr(orch, "_ctx", None)
                if ctx is not None:
                    try:
                        dispatched = ctx.spawn_or_assign(spec)
                    except Exception:
                        logger.exception("rival-sweep dispatch failed; left on board")
            logger.info(
                "Disagreement gate fired (channel=%s author=%s) "
                "— rival-sweep task posted (id=%s, dispatched=%s)",
                msg.channel, msg.author, spec.id, dispatched,
            )
        except Exception:
            logger.exception("Rival-sweep task post failed; ignored")
            # Reset under lock so a future post can retry.
            with gate_lock:
                orch._rival_sweep_fired = False

    bbs.set_on_post(_on_post)
