# 03 — Ablation Results, Best Config, Efficiency

Diagnostic subset: `browsecomp_diag28_0628.csv` — 28 cases drawn from 0625_1..5
(complement set; representative-120 held out for validation), enriched for the target
failure modes + a regression guard:
- 8 answer-loss (found-but-lost), 6 thrash-timeout, 3 give-up, 4 recall-miss, **7 clean-win**.
- **Historical baseline on these 28 = 7/28 correct** (only the clean-wins). So any arm that
  recovers wrong→right while keeping the 7 wins is a real gain. (Caveat: historical baseline
  ran at 9000s+rerun on other endpoints; arms run at 6000s/no-rerun on the dedicated B200 —
  so cross-compare directionally; the clean head-to-head is on representative-120 later.)

All arms: base config `browsecomp_qwen_noneff_retrieval.yaml`, judge `gpt-4-1-dev`,
dedicated node `soyoung-vllm-sunonly-temp` (8×B200), `eval.parallel=8` (24 total),
`eval.timeout=6000 rerun_timeouts=false compact_tokens=180000`.

## ROUND 1 — config-only hypothesis ladder (launched 2026-06-28 10:00Z)
All three carry the **thrash caps** (the H11/H2 root-enabler fix):
`llm.subagent_max_turns=70 swarm.max_subagents=6 web.search_neardup_hard_stop=12 swarm.disable_builder_idle=true`.

| Arm | Adds on top of thrash caps | Tests |
|---|---|---|
| **R1a_thrash** | (nothing — thrash caps only) | H2/H11 thrash |
| **R1b_commit** | `enable_empty_answer_recovery=true swarm.min_dedicated_reviewers=0 enable_candidate_emergence_sweep=false` | H1/H5/H10 commit & verify |
| **R1c_effmax** | R1b + `web.browsing_max_reflection_loops=1 swarm.max_subagent_tasks=2` | aggressive efficiency |

### Round 1 results — _(filling in as arms complete)_
| Arm | N done | correct | acc | notes |
|---|---|---|---|---|
| R1a_thrash | 2 | 1 | (killed) | KILLED early: cases BLOCKED on wait_for_tasks (kept the gates) |
| R1b_commit | 13 | 8 | 0.615 | thrash caps + commit knobs (config-only) |
| R1c_effmax | 10 | 6 | 0.600 | + aggressive efficiency (reflection_loops=1, subtasks=2) |

**Round-1 BY CATEGORY (R1b @ n=13, config-only, OLD code):**
| category | R1b correct/total |
|---|---|
| clean_win (regression guard) | **7/7** (100% preserved — no regression) |
| giveup | **1/1 FLIP** (0→1) |
| answer_loss | **0/4** (023/370/787/558 — all found the answer, all ran 408-870 turns → hit 6000s timeout) |
| thrash_timeout | 0/1 |

(R1b/R1c killed at n=13/10 to free the dedicated node for the 120-validation; signal sufficient.)

**EARLY findings (small n, partial):**
- **Thrash caps work (efficiency):** completed cases run 138-365 turns / ~750-2600s vs
  baseline 344 turns / 6931s — and EASY cases finish ~3× faster.
- **Gate-removal fixes the deadline-block (Pattern A):** R1a (kept `min_dedicated=1`,
  `enforce_alt_task`, emergence on) had its in-flight cases BLOCKED on `wait_for_tasks`
  (alternative-candidate-sweep / verify-*), so it crawled (n=2) — **killed it** to free
  throughput. R1b/R1c (`min_dedicated=0`, emergence off, empty-recovery on) do NOT block and
  finish fast. So the commit-knob bundle directly removes the blocking that caused found-but-
  -not-committed timeouts.
- **A give-up flipped:** browsecomp_343 (give-up under baseline) → correct in R1b.
- **Caveat:** at temp 0.6 / n=1, individual clean-wins flip both ways (372, 391 regressed in
  some arms) — that's sampling noise; judge on the aggregate, and validate on the 120 set.

Key things to read off Round 1:
1. Did thrash caps cut median web_search (target: 221→~109) and **compaction events** (target → 0)?
2. Did the commit/verify knobs flip answer-loss + give-up cases wrong→right?
3. Any regression on the 7 clean-wins? (must stay correct)

### ROUND 1 INTERIM CONCLUSION (n≈11 R1b / 9 R1c)
- **Clean-wins preserved**: 8/8 (R1b), 6/6 (R1c) kept correct — NO real regression.
- **Give-ups flip**: 343 → correct (commit knobs work).
- **HARD answer-loss / thrash / recall cases still WRONG and STILL hit the timeout**
  (023 answer-loss ran 870 orchestrator turns → killed at 6000s; 370, 325 same). The thrash
  caps bound *subagents*, but the **orchestrator keeps spawning verify-* tasks and blocking on
  `wait_for_tasks`** until the wall — and it times out BEFORE reaching `prepare_report`, so
  `enable_empty_answer_recovery` / `surface_bbs_candidates` (report-time hooks) never fire.
- **Implication:** config-alone ≈ "preserve + flip give-ups" (modest). To convert the
  found-but-blocked cases (023/370), need a **hard cap on ORCHESTRATOR turns** (force it to
  `prepare_report` early, where surface_bbs_candidates commits the candidate) — `llm.max_turns`
  is currently 1200. → add `llm.max_turns≈500-600` to the 120 config (NEW lever for v3).
- Diagnostic's 6000s timeout is too aggressive; the 120-validation uses 9000s+rerun.

## ROUND 2 — 120-validation (representative-120, judge gpt-4-1-dev, timeout 9000 + rerun)

Compared on the SAME cases against baseline_120 = **0.558**.

### v2 = "aggressive efficiency" + 3 code changes — RESULT: REGRESSION (key honest finding)
Config: `subagent_max_turns=70, max_subagents=6, disable_builder_idle, min_dedicated_reviewers=0,
enable_candidate_emergence_sweep=false, enable_empty_answer_recovery`, + `surface_bbs_candidates`
+ `compaction_prune_junk`.
On the first **72 completed cases** (same-case A/B vs baseline_120):
- baseline **0.667** (48/72) vs **v2 0.597** (43/72) → **NET −5**: **4 flips** (base-wrong→v2-right)
  vs **9 regressions** (base-right→v2-wrong).
- **Interpretation:** the aggressive thrash caps + verification-removal CUT necessary
  search/verification on medium cases baseline solved. The "thrash is causal waste"
  hypothesis is **NOT validated** — much of the search was difficulty-driven, not waste.
  Net: ~3× faster but ~tie-to-worse accuracy (a Pareto *speed* win, NOT a ceiling break).
  (Some of −5 is temp=0.6 single-run noise, but it is clearly not the +0.19 the early
  clean-win-skewed reading suggested.) **Killed at n=72** to repurpose nodes for v3.

### v3 = verification KEPT + gentle caps + answer-retention code — RUNNING
Corrected config: KEEP `min_dedicated_reviewers=1` + `enable_candidate_emergence_sweep=true`
(the verification v2 removed), gentle `subagent_max_turns=200, max_subagents=10`,
`disable_builder_idle`, + the 3 retention flags (`surface_bbs_candidates`,
`compaction_prune_junk`, `enable_empty_answer_recovery`). Isolates "do the retention code
fixes help baseline WITHOUT harmful search cuts?" Same-case A/B vs baseline_120 as it lands.
_(Result pending — may be partial by wake given throughput.)_

| Run | set | acc (same-cases) | vs baseline | verdict |
|---|---|---|---|---|
| baseline_120 | rep-120 | 0.558 (67/120) | — | reference |
| **v2** aggressive+code | rep-120 (72 done) | **0.597** vs base 0.667 | **−0.07 (NET −5)** | regression — efficiency cuts hurt |
| **v3b** verify-kept+gentle+code | rep-120 (120/120) | **0.583** vs base 0.558 | **+0.025 (NET +3: 14 flip, 11 regress)** | small positive, within noise |
| **v4** v3b + alt_task_force_dispatch | rep-120 (running) | tracking ≈ base / slightly − | NET ~−1 @ n=47 | dispatch fix runs but distracts easy cases |

## EFFICIENCY (token / wall-clock)
Baseline median wrong-case = 9305s / 221 searches / 441 LLM calls / 66K out-tokens. v2's
thrash caps cut cases to ~150-320 turns / ~1-2.6k s (~3× faster) — but at an accuracy cost
(above). Track

> THROUGHPUT FINDING (from the dedicated B200 vLLM logs under Round-1 load): the wall-clock
> ceiling is **generation throughput ≈ 1500 tok/s aggregate (~75 tok/s/stream)**, NOT KV
> cache (only 2-5% used) and NOT queueing (`Waiting: 0`). Prefill is fast (12-32k tok/s,
> 86% prefix-cache hit). Implications: (1) parallelism past ~20 concurrent cases does not
> help — the node is generation-saturated; (2) the ONLY way to speed up eval wall-clock is
> to cut **output tokens per case** — thrash caps (fewer turns) AND lowering
> `subagent_reasoning_effort` from `xhigh` (search subagents burn huge thinking budgets that
> may not need xhigh). The latter is an untested Round-2 efficiency lever (accuracy-neutral
> hypothesis). This also explains why the full 1266-set runs take days. → for throughput,
> POOL nodes (each adds ~1500 tok/s) rather than over-parallelizing one.
