# BrowseComp Ceiling-Breaker Investigation (0628)

**Goal:** Break the ~61% accuracy ceiling on the full BrowseComp set (1266) with the
Qwen3.5-27B dynamic swarm. 61% is the Qwen provider's *single-agent* number; a swarm
should beat it. Baseline on the 120 representative subset = **55.8%** (67/120).

**Owner of this doc:** autonomous overnight session 2026-06-28. This directory is the
living plan + memory + results. Updated continuously.

---

## STATUS (last update: 2026-06-28 ~12:10Z)

| Phase | State |
|---|---|
| Multi-hypothesis quantitative analysis (10 hyp, exact numbers) | ✅ `02_hypotheses.md` |
| One-by-one trajectory classification (24-case) | ✅ `05_case_by_case.md` |
| Code-lever mapping | ✅ done |
| Code changes (3, qwen-gated, default-off, tested, pushed `main@d2a114e`, on cluster) | ✅ `code_changes.md` |
| Ablation Round 1 (config-only, diag28) | ✅ done (killed at n=13/10; signal captured) |
| **Round-1 result** | clean-wins 13/13 preserved; give-up flips; **answer-loss 0/6 (config-only) — needs the force-report fix in v2** |
| **120-validation (v1 config / v2 config+code)** | 🔄 RUNNING (launched 12:08Z, pooled across 7 endpoints) |
| Endpoint monitors | ✅ readiness `bc3gfzuax`, idle `be61u5kt9`, 120-accuracy `b2p5fxajh` |

**Headline pending:** `0628_eff120_v1` (config-only) and `0628_eff120_v2` (config + 3 code
changes incl. candidate-led timeout force-report) on representative-120, judge gpt-4-1-dev,
timeout 9000 + rerun, vs baseline **55.8%** (67/120). v1 isolates config gain; v2 adds the
answer-retention code. ETA ~3-4h (throughput-bound + pooled).

---

## HEADLINE FINDINGS (evidence in 02_hypotheses.md)

1. **ANSWER SELECTION is the dominant failure, not search recall.** In **≥31% of wrong
   cases** the correct answer is present in the swarm's own task summaries / subagent
   messages / BBS but does NOT appear in the final answer. The swarm *finds it and loses
   it*. (Lower bound — substring matcher has false negatives.)
2. **Thrash is monotonically lethal.** Accuracy by web_search calls: 0-50 → **0.76**,
   100-200 → 0.55, 200-400 → **0.28**, 400+ → **0.17**. By LLM calls: 100-300 → 0.71,
   500-800 → 0.23, 800+ → **0.07**. Median wrong case: **221 searches / 441 LLM calls /
   66K out-tokens**; median correct: **107 / 236 / 33K**.
3. **Wall-clock timeouts are a SYMPTOM of thrash, not a root cause** (user already
   mitigates with `rerun_timeouts=true`). Cases finishing ≤9000s score 0.74-0.87;
   the ~200 that hit the 9305s wall score **0.20**. The median *wrong* case ran to the
   wall (9305s); median correct = 6788s.
4. The two prime suspects for answer-loss (per user + data): **history compaction
   dropping the finding**, and **a rival/verifier overturning a correct candidate**
   (smoking gun: Redmond Monument — found + 11/11 criteria verified by two agents, one
   verifier rejected it, swarm gave up).

## STRATEGY

Attack answer-selection + thrash, not "search harder". Levers, in priority order:
- **Commit the best-supported candidate; never emit "no answer".** (enable_force_submit,
  extend empty_answer_recovery to give-ups, best-candidate selection in report.)
- **Protect verified findings from being overturned** by a single rival/verifier
  (treat single-verifier rejection as advisory; keep majority-verified candidate).
- **Protect verified findings through compaction** — investigate whether compaction
  evicts the answer; try *selective delete of certainly-wrong paths* instead of pure
  LLM-summarize (user's idea).
- **Cut thrash**: cap searches/reformulation per task, fewer concurrent subagents,
  trim idle re-spawning + rival sweeps — so budget is spent on the answer, not burying it.

All code changes **gated to the qwen browsecomp run** (config flag default-off, enabled
only in the new YAML) so other models' runs (sonnet 4.5 etc.) on the same checkout are
unaffected.

## RESOURCES

- Dedicated ablation node: `http://soyoung-vllm-sunonly-temp:7777/v1` (8×GPU, mine; was
  still loading model at session start — allocator monitor watches readiness). ≤24 parallel/node.
- Shared (in-flight 0625 + evobrowsecomp), use only spare headroom: each already at
  parallel=20, so +4 each available: `soyoung-vllm-qwen35-0621`, `-b200` (ending soon),
  `-b200-2`, `-h2`, `vboonsanong-8-h200`, `vboonsanong-glm52-serve` — all serve Qwen3.5-27B.
- Data + run host: pod `soyoung-dev-cpu-0-nhzn6` (ns mltraining-dev), data under
  `/data/soyoung/important/arcticswarm/`. Drive non-interactively via `kubectl exec`.
- Judge: Azure `gpt-4-1-dev` (matches user's reported numbers).

## RUN TEMPLATE

```
export ARCTICSWARM_SETTINGS_PATH=/code/users/soyoung/snowswarm_settings.json && \
source /code/users/soyoung/activate_snowswarm.sh && \
arcticswarm-eval --config conf/bench/<config>.yaml \
  eval.csv_path=<subset.csv> eval.output=/data/soyoung/important/arcticswarm/0628_<name> \
  llm.agent_model_base_url="<endpoints>" eval.parallel=<N> \
  odl.hybrid_url=http://soyoung-dataloader-cpu-1:5002 azure.enabled=true \
  eval.judge_model=gpt-4-1-dev llm.compact_tokens=180000 \
  eval.rerun_timeouts=true eval.timeout=9000
```

See `03_results.md` for ablation runs and `04_errors_and_mitigations.md` for the running
log of problems + fixes + todos.

**`06_timing_race_and_selection.md` (added 2026-06-29)** — decomposes the "found-but-not-selected"
family (≥31% of wrong cases) into 4 distinct mechanisms via a 500-case population scan, and isolates
the new **TIMING-RACE hypothesis (H12)**: a subagent finds the answer but posts it *after* the
orchestrator commits at the `prepare_report` timeout wall. Canonical cases browsecomp_1049 (timing
race) and browsecomp_1253 (present-but-overturned), code-grounded + adversarially verified. Answers
"why did the reviewer CONFIRM the wrong candidate" and "would a longer timeout help" (no).

**`07_present_not_selected.md` (added 2026-06-29)** — decomposes the ~20% PRESENT_NOT_SELECTED
bucket (28-case classification). Key correction: **>57% is NOT a selection bug** (36% judge/format
scoring artifacts where the swarm got it right; 21% recall-misses mis-binned by substring). The
genuine selection bug (~9% of wrong cases) is **dominated by P2 "verification-veto → give-up"**
(canonical Redmond_012): a correct candidate is elevated, one mis-measured/contested constraint
vetoes it, swarm emits "no answer." Code-grounded fix (A1): close the `bbs.py:108` /
`_build_candidate_digest` give-up hole that silently disables `surface_bbs_candidates` for exactly
these cases. P0 judge-normalization is the cheapest win overall (lives in the eval harness).
