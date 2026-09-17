#!/usr/bin/env bash
# Overnight driver for the first LoHoSearch run on Qwen3.5-27B.
#
# Sequence:
#   1. Wait for the in-flight 3-case smoke run to finish.
#   2. Gate on it: the smoke run must have produced a report.json with judged
#      results. If it didn't, STOP and leave a diagnosis — do not burn a night
#      of GPU on a broken pipeline.
#   3. Launch the full 544q run.
#   4. If that exits without a report.json (crash, OOM, endpoint blip), resume
#      it with RESUME=1 — up to MAX_RESUMES times. Resume reuses the same
#      eval.output and rebuilds from trajectories, so finished cases aren't
#      redone.
#
# POD REQUIREMENT: run this on a GPU pod (2 TB / 184 CPU), NOT on
# soyoung-cpu-in. That pod's container cgroup caps memory at 7.6 GiB, and 3
# concurrent LoHoSearch cases x 16 subagents blew straight through it — the
# eval was SIGKILLed mid-case with cgroup memory.events oom_kill=8 and
# memory.peak above memory.max. Its 2 CPUs were also the reason a single case
# took ~45 min.
#
# Run detached on the pod:
#   cd /code/users/soyoung/ArcticSwarm_lohosearch
#   nohup bash scripts/drive_lohosearch_0917.sh > /data/soyoung/important/arcticswarm/_logs/lohosearch_driver.log 2>&1 &
set -uo pipefail

ROOT=/data/soyoung/important/arcticswarm
LOGS=$ROOT/_logs
RUN_NAME=0917_lohosearch_qwen
SMOKE_DIR=$ROOT/${RUN_NAME}_smoke3
FULL_DIR=$ROOT/$RUN_NAME
VENV=/data-fast/soyoung/venvs/lohosearch
MAX_RESUMES=${MAX_RESUMES:-6}

mkdir -p "$LOGS"
say() { echo "[driver $(date -u +%H:%M:%SZ)] $*"; }

# --- 1. wait for the smoke run --------------------------------------------
# Match THIS benchmark's eval only. A bare "arcticswarm-eval" pattern also
# matches unrelated evals sharing the pod (a BrowseComp-Plus run at parallel=20
# was live here), so the driver waited on someone else's job and never
# proceeded. The bracket keeps the pattern from matching this script's own
# command line.
EVAL_PAT="conf/bench/lohosearch[_]qwen.yaml"
say "waiting for smoke run to finish"
while pgrep -f "$EVAL_PAT" >/dev/null 2>&1; do sleep 60; done
say "no lohosearch eval process running"

# --- 2. gate on the smoke result ------------------------------------------
if [[ ! -f "$SMOKE_DIR/report.json" ]]; then
  say "ABORT: smoke run produced no report.json in $SMOKE_DIR"
  say "last 30 lines of $LOGS/lohosearch_smoke.log:"
  tail -30 "$LOGS/lohosearch_smoke.log" 2>/dev/null
  exit 1
fi

# The judge must have actually labelled the cases; a report with zero judged
# results means the pipeline ran but scoring is broken, which is exactly the
# failure a smoke test exists to catch.
JUDGED=$("$VENV/bin/python" - "$SMOKE_DIR/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
# The per-case verdict field is `judge_correct` (NOT `correct` — an earlier
# version of this gate looked for the latter, found nothing, and wrongly
# aborted a healthy run). Count cases that carry a non-null verdict.
cases = r.get("per_case") or []
if isinstance(cases, dict):
    cases = list(cases.values())
print(sum(1 for c in cases if isinstance(c, dict) and c.get("judge_correct") is not None))
PY
)
say "smoke report.json found; judged verdicts = ${JUDGED:-0}"
if [[ "${JUDGED:-0}" -lt 1 ]]; then
  say "ABORT: smoke report has no judged verdicts — scoring is broken, not launching the full run"
  exit 1
fi

# --- 3 + 4. full run, with bounded resume ---------------------------------
cd /code/users/soyoung/ArcticSwarm_lohosearch || exit 1

for attempt in $(seq 0 "$MAX_RESUMES"); do
  if [[ -f "$FULL_DIR/report.json" ]]; then
    say "full run COMPLETE — report.json present after $attempt attempt(s)"
    break
  fi

  if [[ "$attempt" -eq 0 ]]; then
    say "launching FULL 544q run -> $FULL_DIR"
    RESUME=0 VENV="$VENV" RUN_NAME="$RUN_NAME" \
      bash scripts/launch_lohosearch_qwen.sh >> "$LOGS/lohosearch_full.log" 2>&1
  else
    say "full run exited without report.json — resume attempt $attempt/$MAX_RESUMES"
    sleep 60
    RESUME=1 VENV="$VENV" RUN_NAME="$RUN_NAME" \
      bash scripts/launch_lohosearch_qwen.sh >> "$LOGS/lohosearch_full.log" 2>&1
  fi
  say "attempt $attempt returned rc=$?"
done

if [[ -f "$FULL_DIR/report.json" ]]; then
  say "DONE. report: $FULL_DIR/report.json"
else
  say "GAVE UP after $MAX_RESUMES resumes; see $LOGS/lohosearch_full.log"
fi
