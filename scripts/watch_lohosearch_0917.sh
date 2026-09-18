#!/usr/bin/env bash
# Resume-watchdog for the in-flight LoHoSearch 544q run.
#
# Replaces drive_lohosearch_0917.sh, whose job (gate on the smoke, then launch)
# is already done. This one only keeps the full run alive: if the eval dies for
# any reason short of completion, resume it.
#
# Why it exists: the original run was launched on a pod named
# "soyoung-rebuttal-temp-oneday", which was reclaimed after ~1 day and took the
# eval, its driver, and its vLLM endpoint with it. /code and /data are shared
# Lustre so the trajectories survived and RESUME=1 recovered everything — but
# nothing was left watching. A watchdog on a *different* pod from the endpoint
# would be better still; this one at least survives eval crashes and endpoint
# blips.
#
# COMPLETION TEST: report.json is written mid-run as a checkpoint too, so its
# mere presence means nothing. A finished run's report has NO "checkpoint" key
# and per_case covering all 544. That is what this checks.
#
# Run detached:
#   cd /code/users/soyoung/ArcticSwarm_lohosearch
#   nohup bash scripts/watch_lohosearch_0917.sh > /data/soyoung/important/arcticswarm/_logs/lohosearch_watchdog.log 2>&1 &
set -uo pipefail

ROOT=/data/soyoung/important/arcticswarm
RUN_NAME=${RUN_NAME:-0917_lohosearch_qwen}
OUT_DIR=$ROOT/$RUN_NAME
LOGS=$ROOT/_logs
VENV=${VENV:-/data-fast/soyoung/venvs/lohosearch}
# May be a COMMA-SEPARATED list: runner.py's _parse_endpoint_spec splits it and
# builds an EndpointPool that spreads cases least-connections across them.
ENDPOINT=${ENDPOINT:-http://soyoung-rebuttal-1:7777/v1,http://soyoung-glm:7777/v1}
EXPECTED=${EXPECTED:-544}
MAX_RESUMES=${MAX_RESUMES:-12}
# Match THIS benchmark's eval only — other evals share these pods.
EVAL_PAT="conf/bench/lohosearch[_]qwen.yaml"

say() { echo "[watchdog $(date -u +%H:%M:%SZ)] $*"; }

is_complete() {
  "$VENV/bin/python" - "$OUT_DIR/report.json" "$EXPECTED" <<'PY' 2>/dev/null
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
# A checkpointed (mid-run) report carries a "checkpoint" key; a final one does not.
if "checkpoint" in r:
    sys.exit(1)
cases = r.get("per_case") or []
sys.exit(0 if len(cases) >= int(sys.argv[2]) else 1)
PY
}

say "watching $OUT_DIR (expect $EXPECTED cases, endpoint $ENDPOINT)"

for attempt in $(seq 0 "$MAX_RESUMES"); do
  # Let the current eval run to its end before judging the situation.
  while pgrep -f "$EVAL_PAT" >/dev/null 2>&1; do sleep 120; done

  if is_complete; then
    say "run COMPLETE — final report.json has >= $EXPECTED cases"
    "$VENV/bin/python" - "$OUT_DIR/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
c = r.get("per_case") or []
ok = sum(1 for x in c if x.get("judge_correct") is True)
judged = sum(1 for x in c if x.get("judge_correct") is not None)
print(f"FINAL: {ok}/{judged} judged correct = {100.0*ok/max(judged,1):.1f}%  (cases={len(c)}, errors={r.get('total_errors')})")
PY
    exit 0
  fi

  if [[ "$attempt" -eq "$MAX_RESUMES" ]]; then
    say "GAVE UP after $MAX_RESUMES resumes"
    exit 1
  fi

  # Endpoints may have died with their pods; wait for at least one rather than
  # burning retries. ENDPOINT can be a comma-separated list, so probe each.
  for _ in $(seq 1 60); do
    live=0
    for u in ${ENDPOINT//,/ }; do
      curl -s -m 10 "${u}/models" >/dev/null 2>&1 && live=$((live+1))
    done
    [[ "$live" -gt 0 ]] && { say "$live endpoint(s) reachable"; break; }
    say "no endpoint of ${ENDPOINT} reachable — waiting"
    sleep 60
  done

  say "eval not running and run incomplete — resume attempt $((attempt+1))/$MAX_RESUMES"
  cd /code/users/soyoung/ArcticSwarm_lohosearch || exit 1
  RESUME=1 VENV="$VENV" ENDPOINT="$ENDPOINT" RUN_NAME="$RUN_NAME" \
    bash scripts/launch_lohosearch_qwen.sh >> "$LOGS/lohosearch_full.log" 2>&1
  say "resume attempt $((attempt+1)) returned rc=$?"
  sleep 30
done
