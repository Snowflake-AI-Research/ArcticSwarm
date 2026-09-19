#!/usr/bin/env bash
# LoHoSearch ablation: isolation OFF + review gates OFF, on Qwen3.5-27B.
#
# Mirrors the 0918 SEAL-0 isoreview-off command, retargeted to LoHoSearch. Every
# non-ablation setting is held identical to the main 0917_lohosearch_qwen run so
# the two are directly comparable — same 3-endpoint pool, same parallel, same
# caches-off, same Brave-only + OR-unquote retry, same judge.
#
# WHAT IS ABLATED (vs the main run):
#   review gates   swarm.min_dedicated_reviewers 1 -> 0
#                  swarm.min_builder_reviewers   1 -> 0
#                  swarm.max_reviewer_remediations 2 -> 0
#                  swarm.disable_auditor           -> true
#                  swarm.disable_final_verification -> true
#                  swarm.enforce_alt_task     true -> false
#                  swarm.disable_builder_idle      -> true
#   isolation      web.disable_bbs_isolation       -> true
#   reflection     web.browsing_max_reflection_loops 2 -> 0
#                  web.disable_self_reflection     -> true
#   prompts        two stripped SKILL variants, so the prompts stop *asking* for
#                  review/verification that the code no longer enforces —
#                  without these the ablation is prompt-inconsistent and the
#                  agent still burns turns soliciting reviews that never come.
#
# Held identical: 3 endpoints, parallel, compact_tokens=180000, timeout=20000,
# max_subagents=16, subagent_max_turns=200, context_reset=false,
# candidate_emergence_sweep=true, reject_refusal_reports, compaction_prune_junk,
# surface_bbs_candidates, empty_answer_recovery, dup-history collapse,
# neardup_hard_stop=12, xhigh/xhigh, all global caches off, Brave-only with
# disable_brave_or_fallback=false, Azure GPT-4.1 judge.
#
# NOTE on web.search_provider_order=brave: written bare to match the SEAL-0
# ablation command, but it is cosmetic — WebSearchTool expands both `brave` and
# `["brave"]` to ['brave','tavily','serper'] because it re-appends any omitted
# provider. Brave-only actually comes from SETTINGS pointing at
# snowswarm_settings_brave_only.json, which has the dead Tavily/Serper keys
# stripped. Do not drop that or the dead providers come back.
#
# Expected to run substantially faster than the main run (the review/alt-task
# gates and reflection loops are what make a case spend hours), which is the
# point: it isolates how much of the accuracy those gates actually buy.
#
# Usage:
#   PARALLEL=48 bash scripts/launch_lohosearch_isoreview_off.sh
set -euo pipefail

RUN_NAME="${RUN_NAME:-0919_lohosearch_isoreview_off}"
OUT_DIR="${OUT_DIR:-/data/soyoung/important/arcticswarm/${RUN_NAME}}"
ENDPOINT="${ENDPOINT:-http://soyoung-rebuttal-1:7777/v1,http://soyoung-glm:7777/v1,http://soyoung-rebuttal:7777/v1}"
SETTINGS="${SETTINGS:-/code/users/soyoung/snowswarm_settings_brave_only.json}"
PARALLEL="${PARALLEL:-48}"
VENV="${VENV:-/data-fast/soyoung/venvs/lohosearch}"
RESUME="${RESUME:-0}"

export ARCTICSWARM_SETTINGS_PATH="$SETTINGS"
export SF_SKIP_WARNING_FOR_READ_PERMISSIONS_ON_CONFIG_FILE=true
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

EXTRA=()
if [[ "$RESUME" != "0" ]]; then
  # eval.resume is REQUIRED with rebuild_from_trajectories, else cli.py skips the
  # case load and then crashes on `len(cases)`.
  EXTRA+=("eval.resume=true" "eval.rebuild_from_trajectories=true"
          "eval.rerun_errors=true" "eval.rerun_timeouts=true")
  echo "### RESUME MODE"
fi

echo "### LoHoSearch ABLATION (isolation off / review off): ${RUN_NAME}"
echo "### endpoints: ${ENDPOINT}"
echo "### parallel:  ${PARALLEL}"
echo "### output:    ${OUT_DIR}"

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
  swarm.context_reset=false \
  enable_candidate_emergence_sweep=true \
  web.collapse_duplicate_tool_history=true \
  web.dup_history_keep_last=1 \
  web.search_neardup_hard_stop=12 \
  llm.reasoning_effort="xhigh" \
  llm.subagent_reasoning_effort="xhigh" \
  web.enable_search_cache=false \
  web.search_cache_read=false \
  web.fetch_cache_path="off" \
  web.search_provider_order=brave \
  web.disable_brave_or_fallback=false \
  llm.model=qwen3.5-27b \
  llm.vllm_served_model_id=Qwen/Qwen3.5-27B \
  llm.disable_closed_model_fallback=true \
  \
  swarm.min_dedicated_reviewers=0 \
  swarm.min_builder_reviewers=0 \
  swarm.max_reviewer_remediations=0 \
  swarm.disable_auditor=true \
  swarm.enforce_alt_task=false \
  swarm.disable_final_verification=true \
  swarm.disable_builder_idle=true \
  web.disable_bbs_isolation=true \
  web.browsing_max_reflection_loops=0 \
  web.disable_self_reflection=true \
  swarm.skill_overrides.swarm-orchestration-dynamic-web=swarm-orchestration-dynamic-web-noverify-noreview \
  swarm.skill_overrides.bbs-coordination-web=bbs-coordination-web-noreview \
  "${EXTRA[@]}" \
  "$@"
