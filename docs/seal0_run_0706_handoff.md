# SEAL-0 (SealQA) — Run Handoff

**Run name:** `0706_seal` · **Date:** 2026-07-06 22:01:23 → 2026-07-07 04:00:29 UTC (**~6.0 h wall-clock**)
**Model:** self-hosted **Qwen 3.5-27B** (text) · **Benchmark:** SEAL-0, the 111-question hardest subset of SealQA (arXiv 2506.01062)
**Branch:** `seal0-qwen27b` (off `internal`) · **Status:** complete, exit 0 · **Headline: accuracy 0.514 (57/111)**

---

## 1. TL;DR

Integrated SEAL-0 into ArcticSwarm (external loader → unified CSV → verbatim SimpleQA A/B/C judge) and ran the full "winning swarm" config on Qwen 3.5-27B with fetch caching disabled and search caching in write-only mode. Final **rubric-free binary accuracy 0.514** (57 CORRECT / 51 INCORRECT / 3 NOT_ATTEMPTED), 0 case errors, 0 refusals. For context, SEAL-0 is designed so GPT-4.1-chat scores ~near-zero and strong search-augmented systems land ~20–40% — so 0.514 is a strong result.

---

## 2. Accuracy

| Metric | Value |
|---|---|
| Cases graded | 111 / 111 |
| **Accuracy (CORRECT / total)** | **57 / 111 = 0.514** |
| Grade breakdown | 57 CORRECT · 51 INCORRECT · **3 NOT_ATTEMPTED** |
| Refusal final-answers | 0 |
| Case-level errors (`num_errors`) | 0 |
| Judge coverage | 111/111 (every case has a raw A/B/C verdict) |

**By freshness** (SEAL-0's design axis): slow-changing 25/42 = **0.60** › fast-changing 18/35 = 0.51 › never-changing 14/34 = **0.41** (hardest — search yields conflicting/unhelpful results by design).
**By question type:** false-premise 5/8 = 0.62 · advanced-reasoning 45/87 = 0.52 · entity/event-disambiguation 32/63 = 0.51 · temporal-tracking 5/12 = 0.42.
**By topic:** Sports 11/16 = 0.69 · Sci&Tech 17/33 = 0.52 · History&Geo 2/4 = 0.50 · Entertainment 15/31 = 0.48 · Politics 5/11 = 0.45 · Others 7/16 = 0.44.
**By effective year:** 2024 11/14 = 0.79 (n=14) · 2026 14/26 = 0.54 · before-2024 24/49 = 0.49 · 2025 8/22 = 0.36.

Note: **0 NOT_ATTEMPTED would be ideal**; the 3 that landed there are cases the judge scored as not containing the gold answer. The `reject_refusal_reports` + empty-answer-recovery levers kept refusals at 0.

---

## 3. Latency & throughput

| Metric | Value |
|---|---|
| Wall-clock (end-to-end) | **~6.0 h** (22:01:23 → 04:00:29 UTC) |
| Cumulative case-seconds (`total_elapsed_seconds`) | 244,740 s ≈ 68.0 h (summed across 12 parallel workers) |
| Avg case duration | **2,205 s ≈ 36.7 min** |
| Median (p50) / p90 case duration | 2,010 s (33.5 min) / 3,333 s (55.6 min) |
| Min / max case duration | 12.5 min / 134.1 min |
| Avg turns per case (`avg_steps`) | 285 |
| Judge latency (Azure GPT-4.1, per case) | avg 2.3 s (max 5.6 s) |
| Parallelism | 12 concurrent cases |

Under 12-way parallelism against a single vLLM endpoint, the endpoint was contended: intermittent transient `Connection error`s (self-healed via 5→10→20→40 s retry, **0 case failures**). Case duration rose from ~16 min early to ~37 min avg under load. All cases finished far under the 9,000 s per-case cap.

---

## 4. Token economics

| Metric | Value |
|---|---|
| Avg **e2e** tokens / case | **27.3 M** (`avg_total_token_e2e`) |
| Avg billed tokens / case | 20.4 M (`avg_total_tokens`) |
| Avg **input** tokens / case | 20.3 M |
| Avg **output** tokens / case | 96.8 K |
| Total e2e tokens (run) | **~3.03 B** |

Input dominates (~20 M/case) because each case is a deep swarm (orchestrator + subagents + tool LLMs) running ~285 turns with long web/reasoning context. Output is comparatively tiny (~97 K/case). This is the cost profile of an `xhigh`-thinking swarm on a 256 K window, not a per-answer generation cost.

---

## 5. Model & vLLM serving config

During the SEAL-0 run the agent endpoint served (verified live at run time):

- **Served model id:** `Qwen/Qwen3.5-27B` (text; config alias `qwen3.5-27b` auto-maps to the served id via `_normalize_model`)
- **Endpoint:** `http://vboonsanong-8-h200-docai:7777/v1` (in-cluster)
- **Context window:** `max_model_len = 262144` (native 256 K)
- **Hardware:** 8 × H200 (~131 GB used of 143 GB each), tensor-parallel across all 8

> ⚠️ **Serving-recipe caveat.** The pod's vLLM server has since been **re-provisioned** to `Qwen/Qwen3.6-35B-A3B` for a K-BrowseComp run, so exact launch flags could not be captured from the SEAL-0-era server. The pod's **standard serving recipe** (captured post-hoc from the current server; representative, model name differs) is:
> ```
> vllm serve --served-model-name <model> --host 0.0.0.0 --port 7777 \
>   --tensor-parallel-size 8 --max-model-len 262144 --gpu-memory-utilization 0.90 \
>   --max-num-seqs 512 --max-num-batched-tokens 16384 --attention-backend FLASHINFER \
>   --enable-prefix-caching --async-scheduling --language-model-only \
>   --reasoning-parser <…> --enable-auto-tool-choice --tool-call-parser <…> \
>   --trust-remote-code --disable-custom-all-reduce
> ```

**Sampling / thinking (from config, sent via `extra_body`):** `enable_thinking: true` (xhigh), `temperature 0.6`, `top_p 0.95`, `top_k 20`, `presence_penalty 0.0` (Qwen model-card thinking-mode sampling).

**Judge model:** Azure **GPT-4.1** (`gpt-4-1-dev`); routed to Azure because `azure.enabled=true` and `judge_model.startswith("gpt")`. Post-hoc only — never in the agent trajectory.

---

## 6. Agent / swarm hyperparameters (`conf/bench/seal0_qwen.yaml` + CLI overrides)

**Cloned from `browsecomp_qwen.yaml` (full winning swarm).**

Tools — agent: `[read_file, calculator, web_search, web_fetch, pdf_read, python_execute, load_skill]`; orchestrator: `[reasoning, load_skill]`; profiles: `browsing` / `reasoning`.

| Knob | Value |
|---|---|
| `prompt_style` | web |
| `swarm.enabled` | true; profiles `[browsing, reasoning]` |
| `swarm.auditor_reasoning_effort` | xhigh |
| reviewer gates | `min_dedicated_reviewers=1`, `min_builder_reviewers=1`, `max_reviewer_remediations=2` |
| `swarm.enforce_alt_task` | true (premature-commitment guard) |
| `enable_candidate_emergence_sweep` | true |
| **anti-give-up levers** | `reject_refusal_reports=true`, `enable_empty_answer_recovery=true`, `compaction_prune_junk=true`, `surface_bbs_candidates=true` |
| `llm.reasoning_effort` / `subagent_reasoning_effort` | xhigh / xhigh |
| `llm.max_tokens` | 32,000 |
| `llm.compact_tokens` | 180,000 (proactive compaction trigger) |
| `llm.max_tool_calls_per_turn` | 1 (browsing subagents) |
| `llm.orchestrator_max_tool_calls_per_turn` | 0 (unlimited — avoids fan-out truncation) |
| `llm.always_execute_tools_per_turn` | `[post_to_bbs]` |
| `llm.subagent_max_turns` | 200 |
| `llm.timeout` / `max_turns` | 1200 / 1200 |
| `llm.disable_closed_model_fallback` | true (no Claude/GPT in agent loop) |
| `web.max_tool_output_tokens` | 5,000 (cap per fetch/pdf result) |
| `eval.parallel` | 12 |
| `eval.timeout` (per case) | 9,000 s |
| `eval.checkpoint_interval` | 1 |

---

## 7. Caching (3 layers — the SEAL-0-specific requirement)

| Layer | Setting | State |
|---|---|---|
| Global cross-run **fetch** cache | `web.fetch_cache_path=off` | **OFF** (56 GB shared `fetch_cache.sqlite` untouched) |
| Global **search** cache | `web.enable_search_cache=true` + `web.search_cache_read=false` | **write-only / refresh** (always live, never served; overwrites keys) |
| → search cache DB | `web.search_cache_db=/data/soyoon/cache/seal0_search.sqlite` | isolated file; persisted **91,496 rows** |
| Per-question **content** cache | `swarm.enable_content_cache` (default true) | **ON** — the "question-wise local cache" that was to be kept |

`cache_sync` mirrors the search cache to the node-local `/data-fast/cache` and delta-merges to master every 5 cases. **Gotcha fixed live:** the merge assumes the master DB already has the `docs` table; a brand-new master path (as here) starts empty, so every merge failed (`no such table: main.docs`) until the `docs` table + `idx_docs_lookup` index were manually created in the master. After that, merges succeeded and the master populated.

---

## 8. Exact reproduction command

`scripts/launch_0706_seal.sh` on pod `vboonsanong-8-h200-docai-0-vww6h` (ns `mltraining-dev`), tmux session `seal0_0706`:

```bash
export ARCTICSWARM_SETTINGS_PATH=/code/users/soyoung/snowswarm_settings.json
source /code/users/soyoung/activate_snowswarm.sh
cd /code/users/soyoung/ArcticSwarm

arcticswarm-eval --config conf/bench/seal0_qwen.yaml \
  eval.output=/data/soyoung/important/arcticswarm/0706_seal \
  llm.agent_model_base_url="http://vboonsanong-8-h200-docai:7777/v1" \
  eval.parallel=12 azure.enabled=true eval.judge_model=gpt-4-1-dev \
  llm.compact_tokens=180000 eval.timeout=9000 \
  reject_refusal_reports=true compaction_prune_junk=true surface_bbs_candidates=true \
  enable_empty_answer_recovery=true llm.orchestrator_max_tool_calls_per_turn=0 \
  llm.subagent_max_turns=200 \
  web.enable_search_cache=true web.search_cache_read=false \
  web.search_cache_db="/data/soyoon/cache/seal0_search.sqlite" web.fetch_cache_path="off"
```

> The endpoint now serves Qwen3.6-35B-A3B — to reproduce, re-serve `Qwen/Qwen3.5-27B` first (8×H200, recipe in §5).

---

## 9. Integration (code changes on `seal0-qwen27b`)

| File | Change |
|---|---|
| `arcticswarm/eval/data/external/seal0.py` | Loader (`load_seal0`/`convert_seal0_to_csv`) — HF `vtllms/sealqa` config `seal_0` split `test`, plaintext (no decrypt); keeps topic/freshness/urls/etc, drops bulky `*_docs` |
| `arcticswarm/eval/data/external/__init__.py` | Export loader |
| `arcticswarm/eval/data/seal0_v1.csv` | Generated, 111 cases, tag `SEAL0_V1` |
| `arcticswarm/eval/prompts/seal0_eval.txt` | **Verbatim** SimpleQA-style A/B/C grader (from the SealQA authors' grading Colab) |
| `arcticswarm/eval/judge.py` | `judge_seal0` + `_parse_seal0_output` (A=CORRECT only; temp 0.0) |
| `arcticswarm/eval/runner.py` | `SEAL0_V1`→`judge_seal0` dispatch + added to `CANONICAL_METRIC_ONLY_DATASETS` |
| `conf/bench/seal0_qwen.yaml` | Run config (§6) |

---

## 10. Known issues / gotchas

1. **67 `prompt_too_long` events** (= 67 compaction events) — handled by reactive compaction, **0 case failures**. Indicates real context pressure from deep swarms on the 256 K window. To reduce next time: lower `swarm.max_subagents`, tighten `compact_tokens`, or reduce `max_tool_output_tokens`.
2. **Endpoint saturation at `parallel=12`** on a single vLLM endpoint → intermittent connection errors (self-healing, no case loss) and inflated per-case latency (~37 min). For a cleaner/faster run, use `parallel=8` and/or a second endpoint.
3. **`cache_sync` fresh-master bug** — write-only search cache to a new master path fails to merge until the `docs` schema is pre-created (see §7). Fix upstream: have `cache_sync` create the target table if absent.
4. **Config `qwen3.5-27b` → served `Qwen/Qwen3.5-27B`** auto-maps (`_normalize_model`); a raw curl with the short alias 404s (expected).
5. **Branch not committed/pushed** — `seal0-qwen27b` changes live in the local worktree and as on-pod working-tree edits; nothing is committed or pushed.

---

## 11. Artifacts & locations

- **Output dir (pod):** `/data/soyoung/important/arcticswarm/0706_seal/` (`report.json`, `trajectories/`, `resolved_config.yaml`, `code_snapshot.json`)
- **Console log:** `/data/soyoung/important/arcticswarm/0706_seal.console.log`
- **Search cache (write-only):** `/data/soyoon/cache/seal0_search.sqlite` (91,496 rows)
- **Launcher:** `scripts/launch_0706_seal.sh` (pod checkout)
- **Code:** branch `seal0-qwen27b`, files in §9
