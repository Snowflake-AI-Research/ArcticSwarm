# 0629 — batch winner + full-slice launch

## Batch (dev-120, same-case A/B vs baseline_120 = 0.558)

Both configs built on the **v3 base** (`llm.subagent_max_turns=200 swarm.max_subagents=10
swarm.disable_builder_idle=true enable_empty_answer_recovery=true surface_bbs_candidates=true
compaction_prune_junk=true`) PLUS the new stack (`llm.orchestrator_max_tool_calls_per_turn=0`
[role-aware orchestrator fan-out, no longer truncated to 1] + `reject_refusal_reports=true`
[anti-give-up commit] + the auto-gated post→complete sequencing for subagents at cap==1).

| Config | n | acc (`score`) | base (same-case) | flips | regress | NET |
|---|---|---|---|---|---|---|
| **newA_stack** (no reframe) | 120 | **0.625** (75/120) | 0.558 | 18 | 10 | **+8** ✅ |
| newB_reframe (= A + `reframe_prompt`) | 120 | **0.008** (1/120) | 0.558 | 1 | 67 | −66 ❌ |

**Winner = newA_stack = 0.625** — clears 61% on the dev set, beats baseline (+6.7 pts) and beats
the prior best v3 (0.583). Verified real: `score1=75, score0=45, scoreNone=0`; all 120 produced
non-empty `response_text`; judge ran on all (`judge_raw_output` 120/120). Connection-error noise
was actually *higher* on newA (608) than newB (129), so newB's collapse is **not** infra.

### newB_reframe collapse — `reframe_prompt` is harmful with `reject_refusal`
newB produced non-empty answers for all 120 but **systematically wrong** (1/120 correct, 3 refusals).
Mechanism: `reframe_prompt`'s ANTI_ANCHOR_BLOCK pushes browsing agents to consider/commit
*alternative interpretations* and drop the working frame; combined with `reject_refusal_reports`
(forces committing *an* answer rather than "no answer"), the swarm commits to **reframed-wrong**
candidates ~99% of the time. NOTE this contradicts the earlier `recallB_reframe` partial (+3) —
that run was reframe on the OLD base WITHOUT the new stack. The interaction (reframe × reject_refusal
× post→complete sequencing) is the killer. **Keep `reframe_prompt` OFF.** Do not ship it.

## Full run — 0629_slice600 (launched 16:40 UTC)
- Config: **newA_stack** (exact winner; NO reframe).
- Data: `browsecomp_slice600_0629.csv` = first 600 of the 1146-case complement (built from
  `browsecomp_complement_part{1..5}of5.csv`). **600/600 have a 0625 baseline score** (`CONV_ID`
  keyed). **Slice baseline (0625 same-case) = 0.583** (first-600 runs a touch below the full
  complement's 0.603).
- Endpoints (9 live qwen3.5-27B): 0621, b200, b200-2, h2, h4, h5, sunonly-temp, vboonsanong-8-h200,
  vboonsanong-glm52-serve (glm52-serve repurposed to Qwen). parallel=200 (~22/endpoint).
- Same params as the winning batch: `eval.timeout=9000 eval.rerun_timeouts=true
  llm.compact_tokens=180000 judge=gpt-4-1-dev`.
- Output: `/data/soyoung/important/arcticswarm/0629_slice600`; launcher `/tmp/run_full.sh`;
  A/B via `/tmp/cmp_full.py slice600` (vs 0625_1..5). Monitor `b9o1cbyat`.
- **Target:** beat slice baseline 0.583 same-case and clear 0.61 absolute. newA's +6.7pt dev lift,
  if it transfers, lands ~0.65.

## Infra notes (0629)
- `b200-3`/`b200-4` were torn down (replaced by `h4`/`h5`); the first newA launch was hitting those
  dead endpoints (0 useful cases) and was killed + relaunched on 6 live endpoints. Always probe
  `/v1/models` per endpoint before allocating.
- `content_cache.py` fix (recreate case dir before each write; /data→S3 sweep deletes it mid-run)
  committed `e959c0a`, pushed to origin/main, and synced to the cluster editable checkout
  (`/code/users/soyoung/ArcticSwarm`, verified lines 332/383). Cluster install is editable.
