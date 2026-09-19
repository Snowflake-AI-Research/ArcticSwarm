#!/usr/bin/env bash
# Recovery pass for the LoHoSearch 544q run: re-execute the cases that errored
# or hit the eval.timeout deadline, once the first full pass is complete.
#
# WHY: the main run went at parallel=48 across 3 endpoints. That quadrupled
# throughput but inflated per-case duration from ~85 min (parallel=9 baseline)
# to ~162 min, and ~45 cases crossed the 20000s eval.timeout. Those did NOT
# fail — ArcticSwarm's force-report path made them commit an answer from
# whatever the swarm had mid-investigation, and the judge scored it (nearly all
# wrong). They are indistinguishable from genuine wrong answers in the final
# number, yet they are a concurrency artifact, not a capability result. At the
# time of writing the forced set (~45) is about the size of the correct set
# (~45), so recovering even a third of it moves the headline materially.
#
# WHAT: RESUME=1 sets eval.resume + eval.rebuild_from_trajectories +
# eval.rerun_errors + eval.rerun_timeouts. cli.py then preserves every complete
# case and re-executes only those whose duration >= eval.timeout (plus errors).
#
# PARALLEL: 24 = 8 concurrent cases per endpoint. The parallel=9 baseline that
# produced ZERO forced cases and an 85 min mean was 9 per endpoint, so 8 is
# strictly lighter — fast (~3h for ~50 cases) while staying under the load
# level that caused the problem.
#
# Run detached:
#   cd /code/users/soyoung/ArcticSwarm_lohosearch
#   nohup bash scripts/rerun_lohosearch_recovery.sh > /data/soyoung/important/arcticswarm/_logs/lohosearch_rerun.log 2>&1 &
set -uo pipefail

ROOT=/data/soyoung/important/arcticswarm
RUN_NAME=${RUN_NAME:-0917_lohosearch_qwen}
OUT_DIR=$ROOT/$RUN_NAME
LOGS=$ROOT/_logs
VENV=${VENV:-/data-fast/soyoung/venvs/lohosearch}
ENDPOINT=${ENDPOINT:-http://soyoung-rebuttal-1:7777/v1,http://soyoung-glm:7777/v1,http://soyoung-rebuttal:7777/v1}
PARALLEL=${PARALLEL:-24}
EXPECTED=${EXPECTED:-544}
EVAL_PAT="conf/bench/lohosearch[_]qwen.yaml"

say() { echo "[rerun $(date -u +%H:%M:%SZ)] $*"; }

# Completion test: report.json doubles as a mid-run CHECKPOINT, so its presence
# proves nothing. A finished report has no "checkpoint" key and per_case
# covering all 544.
is_complete() {
  "$VENV/bin/python" - "$OUT_DIR/report.json" "$EXPECTED" <<'PY' 2>/dev/null
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
if "checkpoint" in r:
    sys.exit(1)
sys.exit(0 if len(r.get("per_case") or []) >= int(sys.argv[2]) else 1)
PY
}

summarize() {  # $1 = report path, $2 = label
  "$VENV/bin/python" - "$1" "$2" <<'PY'
import json, sys, statistics as s
r = json.load(open(sys.argv[1])); c = r.get("per_case") or []
d = [(x.get("duration_seconds") or 0) for x in c]
ok = sum(1 for x in c if x.get("judge_correct") is True)
judged = sum(1 for x in c if x.get("judge_correct") is not None)
forced = sum(1 for x in d if x >= 20000)
print(f"{sys.argv[2]}: {ok}/{judged} correct = {100.0*ok/max(judged,1):.2f}%  "
      f"| cases={len(c)} forced={forced} errors={r.get('total_errors')} "
      f"mean={s.mean(d)/60:.0f}min" if d else f"{sys.argv[2]}: empty")
PY
}

say "waiting for the first 544-case pass to complete"
while true; do
  if ! pgrep -f "$EVAL_PAT" >/dev/null 2>&1 && is_complete; then break; fi
  sleep 300
done
say "first pass COMPLETE"
summarize "$OUT_DIR/report.json" "PRE-RERUN (as measured at parallel=48)"

# Snapshot the as-measured result BEFORE the rerun overwrites report.json —
# otherwise the parallel=48 baseline is lost and the recovery delta can't be
# quantified. Trajectories are preserved by the resume, but the aggregate
# report is rewritten in place.
SNAP="$OUT_DIR/report_pre_rerun.json"
cp -n "$OUT_DIR/report.json" "$SNAP" && say "snapshotted baseline -> $SNAP"

say "launching recovery pass: parallel=$PARALLEL, rerun_errors + rerun_timeouts"
cd /code/users/soyoung/ArcticSwarm_lohosearch || exit 1
RESUME=1 VENV="$VENV" ENDPOINT="$ENDPOINT" PARALLEL="$PARALLEL" RUN_NAME="$RUN_NAME" \
  bash scripts/launch_lohosearch_qwen.sh >> "$LOGS/lohosearch_full.log" 2>&1
say "recovery pass returned rc=$?"

# Let any lingering process settle, then report the before/after.
while pgrep -f "$EVAL_PAT" >/dev/null 2>&1; do sleep 120; done
say "=== RESULT ==="
summarize "$SNAP" "PRE-RERUN "
summarize "$OUT_DIR/report.json" "POST-RERUN"
