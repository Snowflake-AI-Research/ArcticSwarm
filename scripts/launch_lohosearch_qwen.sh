#!/usr/bin/env bash
# LoHoSearch (544q) on self-hosted Qwen3.5-27B — launch script.
#
# Mirrors the 0729 BrowseComp-Plus "ours again 3rd" command flag-for-flag, with
# three deliberate differences:
#   1. LIVE WEB instead of the Cortex corpus. LoHoSearch has no static corpus,
#      so web.corpus_backend=cortex / web.provider=corpus are DROPPED.
#   2. Search providers pinned to BRAVE ONLY (web.search_provider_order).
#      Tavily and Serper are out of credit, so leaving them in the chain only
#      buys failed requests + latency on every Brave miss.
#   3. eval.csv_path / datasets / judge point at LOHOSEARCH_V1.
#
# Judge: Azure GPT-4.1 only (the BrowseComp grading prompt = LoHoSearch judge
# #1 of 2). We do NOT run the paper's second Qwen2.5-32B SimpleQA judge, so our
# number is the GPT-4.1 component rather than the published 2-judge mean.
#
# Run from a pod labeled dss=true so in-cluster DNS resolves the vLLM endpoint:
#   kontrol in connect soyoung-cpu-in
#   bash scripts/launch_lohosearch_qwen.sh              # full 544q
#   SMOKE=3 bash scripts/launch_lohosearch_qwen.sh      # 3-case smoke test
set -euo pipefail

RUN_NAME="${RUN_NAME:-0917_lohosearch_qwen}"
OUT_DIR="${OUT_DIR:-/data/soyoung/important/arcticswarm/${RUN_NAME}}"
ENDPOINT="${ENDPOINT:-http://soyoung-rebuttal-temp-oneday:7777/v1}"
SETTINGS="${SETTINGS:-/code/users/soyoung/snowswarm_settings_cortex.json}"
PARALLEL="${PARALLEL:-9}"
SMOKE="${SMOKE:-0}"

export ARCTICSWARM_SETTINGS_PATH="$SETTINGS"
export SF_SKIP_WARNING_FOR_READ_PERMISSIONS_ON_CONFIG_FILE=true
source /code/users/soyoung/activate_snowswarm.sh

# A smoke test wants a fresh, tiny, throwaway run — not the real output dir.
EXTRA=()
if [[ "$SMOKE" != "0" ]]; then
  OUT_DIR="${OUT_DIR}_smoke${SMOKE}"
  PARALLEL="$SMOKE"
  EXTRA+=("eval.limit=${SMOKE}")
  echo "### SMOKE MODE: ${SMOKE} cases -> ${OUT_DIR}"
fi

echo "### LoHoSearch run: ${RUN_NAME}"
echo "### endpoint: ${ENDPOINT}"
echo "### output:   ${OUT_DIR}"

arcticswarm-eval \
  --config conf/bench/lohosearch_qwen.yaml \
  eval.csv_path=arcticswarm/eval/data/lohosearch_v1.csv \
  eval.output="${OUT_DIR}" \
  llm.agent_model_base_url="${ENDPOINT}" \
  eval.parallel="${PARALLEL}" \
  azure.enabled=true \
  eval.judge_model=gpt-4-1-dev \
  llm.compact_tokens=180000 \
  eval.timeout=20000 \
  reject_refusal_reports=true \
  compaction_prune_junk=true \
  surface_bbs_candidates=true \
  enable_empty_answer_recovery=true \
  llm.orchestrator_max_tool_calls_per_turn=0 \
  swarm.max_subagents=16 \
  llm.subagent_max_turns=200 \
  swarm.min_dedicated_reviewers=1 \
  swarm.min_builder_reviewers=1 \
  swarm.max_reviewer_remediations=2 \
  swarm.enforce_alt_task=true \
  swarm.context_reset=false \
  enable_candidate_emergence_sweep=true \
  web.collapse_duplicate_tool_history=true \
  web.dup_history_keep_last=1 \
  web.search_neardup_hard_stop=12 \
  web.browsing_max_reflection_loops=2 \
  llm.reasoning_effort="xhigh" \
  llm.subagent_reasoning_effort="xhigh" \
  web.enable_search_cache=false \
  web.search_cache_read=false \
  web.fetch_cache_path="off" \
  web.search_provider_order='["brave"]' \
  llm.model=qwen3.5-27b \
  llm.vllm_served_model_id=Qwen/Qwen3.5-27B \
  llm.disable_closed_model_fallback=true \
  eval.rebuild_from_trajectories=true \
  eval.rerun_errors=true \
  eval.rerun_timeouts=true \
  "${EXTRA[@]}"
