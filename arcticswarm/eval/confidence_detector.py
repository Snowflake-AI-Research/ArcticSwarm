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

"""Confidence scorer for the gated-retry wrapper.

Plan-literal scorer.  Higher score = more
likely correct.  ``T = -0.38`` is the recommended retry threshold.

The two dominant features are ``layer4a_present`` (+0.795) and
``reflection_high_conf_ratio`` (+0.742); the rest are small corrections.
This module reads only fields persisted on ``EvalResult`` -- it does not
depend on logged JSON paths.
"""

from __future__ import annotations

import re
from typing import Any


_REFUSAL_RE = re.compile(
    r"\b(unable to determine|insufficient information|cannot determine|"
    r"cannot answer|no answer can be|not enough information|"
    r"unanswerable|unsolvable)\b",
    re.IGNORECASE,
)


def _reflection_high_conf_ratio(stats: dict[str, Any]) -> float:
    """Fraction of reflection calls that ended at high confidence."""
    if not stats:
        return 0.0
    dist = stats.get("confidence_distribution") or stats.get("confidence_dist") or {}
    high = float(dist.get("high", 0))
    total = sum(float(v) for v in dist.values()) if dist else 0.0
    if total <= 0:
        # Fall back to the older shape sometimes emitted as flat counts.
        total_calls = float(stats.get("total_calls", 0) or 0)
        return high / total_calls if total_calls > 0 else 0.0
    return high / total


def _reflection_sufficient_ratio(stats: dict[str, Any]) -> float:
    if not stats:
        return 0.0
    suff = float(stats.get("sufficient", 0) or 0)
    total = float(stats.get("total_calls", 0) or 0)
    return suff / total if total > 0 else 0.0


def compute_confidence_score(result: Any) -> float:
    """Return the calibrated confidence score for *result*.

    Mirrors the plan-literal scorer with weights from
    ``calibration_artifacts/logreg_weights.json``.  Negative score = low
    confidence (retry candidate).
    """
    layer4a = getattr(result, "swarm_layer4a", {}) or {}
    rival = getattr(result, "swarm_rival_audit", {}) or {}
    refl = getattr(result, "swarm_reflection_stats", {}) or {}
    response_text = getattr(result, "response_text", "") or ""

    layer4a_present = 1.0 if layer4a.get("fired") else 0.0
    layer4a_clean = 1.0 if layer4a.get("clean") else 0.0
    high_conf_ratio = _reflection_high_conf_ratio(refl)
    suff_ratio = _reflection_sufficient_ratio(refl)
    rival_keep = 1.0 if rival.get("recommendation") == "keep_leader" else 0.0
    rival_high_conf = 1.0 if rival.get("confidence") == "high" else 0.0

    # Negative-direction features.
    response_len_chars = len(response_text)
    long_response_pen = max(0.0, (response_len_chars - 80000.0) / 100000.0)
    refusal_hit = 1.0 if _REFUSAL_RE.search(response_text) else 0.0
    no_rival_seen = 1.0 if not (rival.get("candidates_considered") or []) else 0.0

    # Plan-literal weights (1, 1, 1, 1, 0.5, 1, 0.5) for positives and
    # the negative penalties scaled to keep the threshold near -0.38.
    score = (
        1.0 * layer4a_present
        + 1.0 * high_conf_ratio
        + 1.0 * layer4a_clean
        + 1.0 * suff_ratio
        + 0.5 * rival_keep
        + 1.0 * rival_high_conf
        - 0.5 * long_response_pen
        - 1.0 * refusal_hit
        - 0.3 * no_rival_seen
    )
    # Centre to roughly match calibration's scale.  The baseline expected
    # score for a clean correct case is ~0; wrong cases land below -0.4.
    score -= 1.5
    return float(score)


def pick_better(result_a: Any, result_b: Any) -> Any:
    """Pick the result with the higher confidence score.

    Tie-break: prefer the one whose Layer 4a fired clean.  Final tie-break:
    keep ``result_a`` (the original).  Used by the gated-retry wrapper.
    """
    score_a = compute_confidence_score(result_a)
    score_b = compute_confidence_score(result_b)
    if score_b > score_a:
        return result_b
    if score_b == score_a:
        clean_a = bool((getattr(result_a, "swarm_layer4a", {}) or {}).get("clean"))
        clean_b = bool((getattr(result_b, "swarm_layer4a", {}) or {}).get("clean"))
        if clean_b and not clean_a:
            return result_b
    return result_a
