#!/usr/bin/env bash
# Wait for the main LoHoSearch run to finish, then run the isolation-off /
# review-off ablation on the same 3-endpoint pool.
#
# Ordering matters: both runs share the same three vLLM endpoints, so they must
# not overlap or they would contend and neither number would be comparable to
# the other. This chains them.
#
# COMPLETION TEST for the main run: report.json doubles as a mid-run CHECKPOINT,
# so its presence proves nothing. A finished report has no "checkpoint" key and
# per_case covering all 544.
#
# Run detached:
#   cd /code/users/soyoung/ArcticSwarm_lohosearch
#   nohup bash scripts/chain_lohosearch_ablation.sh > /data/soyoung/important/arcticswarm/_logs/lohosearch_ablation.log 2>&1 &
set -uo pipefail

ROOT=/data/soyoung/important/arcticswarm
MAIN_RUN=${MAIN_RUN:-0917_lohosearch_qwen}
MAIN_DIR=$ROOT/$MAIN_RUN
ABL_RUN=${ABL_RUN:-0919_lohosearch_isoreview_off}
LOGS=$ROOT/_logs
VENV=${VENV:-/data-fast/soyoung/venvs/lohosearch}
ENDPOINT=${ENDPOINT:-http://soyoung-rebuttal-1:7777/v1,http://soyoung-glm:7777/v1,http://soyoung-rebuttal:7777/v1}
PARALLEL=${PARALLEL:-30}
EXPECTED=${EXPECTED:-544}
EVAL_PAT="conf/bench/lohosearch[_]qwen.yaml"

say() { echo "[chain $(date -u +%H:%M:%SZ)] $*"; }

is_complete() {
  "$VENV/bin/python" - "$MAIN_DIR/report.json" "$EXPECTED" <<'PY' 2>/dev/null
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

summarize() {
  "$VENV/bin/python" - "$1" "$2" <<'PY'
import json, sys, statistics as s
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    print(f"{sys.argv[2]}: (no report)"); raise SystemExit
c = r.get("per_case") or []
d = [(x.get("duration_seconds") or 0) for x in c]
ok = sum(1 for x in c if x.get("judge_correct") is True)
judged = sum(1 for x in c if x.get("judge_correct") is not None)
forced = sum(1 for x in d if x >= 20000)
print(f"{sys.argv[2]}: {ok}/{judged} = {100.0*ok/max(judged,1):.2f}%  "
      f"cases={len(c)} forced={forced} errors={r.get('total_errors')} "
      f"mean={s.mean(d)/60 if d else 0:.0f}min")
PY
}

say "waiting for main run $MAIN_RUN to complete"
while true; do
  if ! pgrep -f "$EVAL_PAT" >/dev/null 2>&1 && is_complete; then break; fi
  sleep 300
done
say "main run COMPLETE"
summarize "$MAIN_DIR/report.json" "MAIN (iso+review ON)"

# The main run's own watchdog exits on completion; make sure nothing lingers
# before we start competing for the same endpoints.
while pgrep -f "$EVAL_PAT" >/dev/null 2>&1; do sleep 60; done

say "launching ablation $ABL_RUN (isolation off / review off), parallel=$PARALLEL"
cd /code/users/soyoung/ArcticSwarm_lohosearch || exit 1
RUN_NAME="$ABL_RUN" VENV="$VENV" ENDPOINT="$ENDPOINT" PARALLEL="$PARALLEL" \
  bash scripts/launch_lohosearch_isoreview_off.sh >> "$LOGS/lohosearch_ablation_run.log" 2>&1
say "ablation returned rc=$?"

while pgrep -f "$EVAL_PAT" >/dev/null 2>&1; do sleep 120; done
say "=== ABLATION COMPARISON ==="
summarize "$MAIN_DIR/report.json"       "MAIN     (iso+review ON )"
summarize "$ROOT/$ABL_RUN/report.json"  "ABLATION (iso+review OFF)"
