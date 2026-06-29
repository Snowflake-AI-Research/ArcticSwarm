# 02 — Root-Cause Hypotheses (exact numbers)

Data: per-case `report.json` + full trajectory + content-compactor/web logs for
`0625_1..5` + `0625_baseline_120` (all Qwen3.5-27B, identical noneff_retrieval config,
only endpoint/data-shard/parallel differ — verified: every endpoint serves
`Qwen/Qwen3.5-27B`, every run requested it; the `kimi`/`glm52`/`8-h200` endpoint names
are repurposed pods, NOT different models). Tooling: `.invest_0628/` scripts.

### Method & caveats (read first)
- **Trajectory-metrics sample = 560 cases** (cases with a full trajectory file). Its raw
  acc is **0.496**, biased LOW vs the true ~0.60 because checkpoint-resumed cases (the
  377 `dur=0/steps=0` records, ~65% correct) are quick wins under-represented here. So
  treat **absolute** acc in this sample as a floor; the **relative/bucketed** comparisons
  below are robust.
- **Reference-answer matcher**: normalized substring of the reference (≤70 chars). Validated
  against correct cases: it matches the ref in the orchestrator's *late context* in
  **253/265 = 95%** of correct cases (reliable presence detector) but in the terse *final
  answer* only **47/265 = 18%** (final answers paraphrase the entity, so `ref_in_final`
  has a high false-negative rate and is used only directionally). ⇒ "found somewhere" and
  "lost before decision" are reliable; counts are **lower bounds**.

---

## RANKED FINDINGS

### H1 (PRIMARY) — Answer SELECTION/RETENTION, not just recall
Among matched-ref **wrong** cases (n=271):
- **69.4% (188)** never surfaced the answer anywhere → recall/reasoning miss (upper bound;
  paraphrase misses inflate this).
- **30.6% (83)** found the correct answer in the swarm's own findings but lost it. Split:
  - **lost BEFORE the decision** (not in last 30 orchestrator msgs): **35 = 42% of found-wrong** → compaction / BBS-surfacing.
  - **present at decision but NOT selected** (in late context, wrong final answer): **48 = 58% of found-wrong** → overturn / bad selection.

**Why it's the priority despite being the minority:** the ≥31% "found-but-lost" bucket is
the *cheapest* to recover (answer already in hand). Recovering even half ≈ **+5-7 pts**.

### H2 (PRIMARY) — Thrash is monotonically lethal
Accuracy vs work done (560-case sample):
| web_search calls | acc | | LLM calls | acc | | compaction events | acc |
|---|---|---|---|---|---|---|---|
| 0-50 | **0.76** | | 100-300 | 0.71 | | **0** | **0.673** |
| 50-100 | 0.74 | | 300-500 | 0.37 | | 1-3 | 0.228 |
| 100-200 | 0.55 | | 500-800 | 0.23 | | 3-6 | 0.221 |
| 200-400 | **0.28** | | 800+ | **0.07** | | 6-12 | 0.091 |
| 400+ | **0.17** | | | | | 12+ | 0.000 |

Median **wrong** case: **221 searches / 441 LLM calls / 66K out-tokens / 1 compaction**;
median **correct**: **109 / 237 / 34K / 0 compaction**.

### H3 — Timeouts are a SYMPTOM of thrash (not a root cause)
Clean duration buckets (dur>0): 1-3k s → 0.87, 6-8k → 0.76, 8-9k → 0.74, **9000-9300 →
0.94**, **>9305 (hit the 9000s wall) → 0.20** (40/200). Median wrong-case duration =
**9305s** (ran to the wall); median correct = 6788s. The cliff is the wall-clock cutoff,
not difficulty. User already mitigates via `rerun_timeouts=true`; the real fix is removing
the *cause* (thrash) — see H11.

### H4 — Compaction is a real answer-loss amplifier (code-confirmed)
Mechanism (code map, `context_management.py`): TWO paths. **Proactive** (fires at
`compact_tokens`/0.9·window) uses a candidate-preserving structured prompt. **Reactive**
(fires on prompt-too-long *error*) uses a GENERIC prompt and, in fallback, **truncates
tool-results to 2000 chars and drops the oldest 2/3 of messages** (`:846`). Both then
`self.messages.clear()` → one summary msg; **nothing is pinned**. `0625_1`: **348
prompt-too-long events / 185 cases**. Correlation: any compaction → acc collapses
0.67→0.22. `lost_before_decision` cases have median 1 compaction vs 0 for
`present_late`. ⇒ reduce thrash so reactive rarely fires; make proactive fire earlier;
re-inject/pin findings.

### H5 — Reviewer-diversity gate overturns majority-verified correct candidates (code-confirmed)
The `prepare_report` reviewer gate (`tools.py:2604`) requires BOTH a *builder* AND a
*dedicated* (reasoning-auditor) VERIFIED `#consensus` verdict. The auditor is prompted to
CHALLENGE ("partial matches are the #1 source of wrong answers", `prompts.py:1179`).
**Smoking gun (browsecomp_012):** two builders independently found "Redmond Monument" with
11/11 criteria; the dedicated auditor posted a CHALLENGE → gate unsatisfied → swarm
emitted "no monument matches all constraints." `enable_force_submit=false`, so no escape.
⇒ majority-override / treat single dedicated dissent as advisory / pin the verified
candidate on degrade.

### H6 — BBS findings can vanish from the orchestrator's working context (code-confirmed)
`check_new_messages` auto-injects new BBS posts with **default limit=50**; `read` does
`since_id` filter then `msgs[-limit:]` — if >50 posts accrue between injections (high-volume
cases), older posts are silently dropped and the cursor advances past them. After a
compaction wipes injected posts from `self.messages`, they are **not re-injected** unless
the agent voluntarily re-reads. ⇒ raise inject limit; re-inject a #key-findings/#consensus
digest into the `prepare_report` prompt.

### H7 — "Tool errors" are mostly a timeout symptom, not an independent cause
Median tool-errors: correct=8, **wrong=28**; 20+ errors → 0.26 acc. But decoding the error
strings (scanned 127 trajectories): ~**1093** are `"SYSTEM SHUTTING DOWN… web_search
DISABLED"` (the 9000s wall forcing submission — a thrash/timeout symptom), **438** benign
`"max tool calls per turn reached"` (agent fighting `max_tool_calls_per_turn=1`), **1025**
`"(no output)"` empty results. Genuine bugs are rare: ~38 tool-hallucinations
(`Unknown tool: web_search` from a wrong-profile agent), and a real `post_to_bbs` crash
(`dictionary update sequence element #0 has length 1`, 3×). ⇒ folds into H2/H11; minor
cleanup items only.

### H8 — Content compactor / fetch truncation can drop the answer at ingest
`content_compactor.py`: scores (relevance/answerability/authority/data_density) are parsed
but **never used to gate selection** (`:341`); `_assemble_selected` truncates kept chunks
to the `max_tool_output_tokens` (5000) cap **in document order**, so an answer-bearing
chunk late in a long page can be crowded out. Sidecar logs were largely swept to S3
(only 2-6 cases left locally) so population numbers are thin; mechanism is real but lower
priority than H1/H2. ⇒ raise cap to 10-12K and/or priority-order chunk assembly.

### H9 — Compute/Python need: REJECTED
**0 / 560** cases used `python_execute` (browsing profile lacks it). `calculator` used in 74
cases: acc **0.500 with vs 0.496 without** — no difference. BrowseComp failures are
search/selection-bound, not compute-bound. ⇒ **do NOT add python_execute** (matches user's
"only if analysis shows need").

### H10 — Explicit give-up ("no answer exists"): real but smaller than it looks
Final-answer give-ups = **18/282 wrong = 6.4%** (stricter than the 34% figure that also
scanned intermediate text). Of these only 2 had the ref in findings. ⇒ `enable_empty_answer_recovery`
(currently OFF for qwen) + a commitment prompt covers it cheaply; not the main lever.

### H11 (ROOT ENABLER) — `subagent_max_turns=0` is a no-op → browsing subagents are ~unbounded
`run_config.py:48 subagent_max_turns=0` falls back to `max_turns=1200`. With
`browsing_max_search_plans=2 × browsing_max_reflection_loops=2`, a flailing browsing
subagent gets **~298 turns PER search phase** before any structural stop, and there is **NO
swarm-wide search/turn budget anywhere** (`max_subagents=16 × max_subagent_tasks=3` = up to
48 task-executions/question). This is the structural cause of the 221-vs-109 search gap in
H2. ⇒ THE dominant lever: set `llm.subagent_max_turns≈70`, `swarm.max_subagents≈6`,
`web.search_neardup_hard_stop≈12`.

---

## Failure decomposition (matched-ref wrong, n=271) — what each lever can recover
| Bucket | Share | Lever family |
|---|---|---|
| Never found (recall/reasoning) | ~69% | thrash-cut frees budget for real search (H2/H11); content-compactor (H8) |
| Present at decision, not selected | ~18% | commit/verify: force-submit, reviewer override, empty-answer-recovery (H1/H5/H10) |
| Lost before decision | ~13% | compaction fix + BBS re-inject (H4/H6) |

> NOTE on "never found": inflated by the paraphrase-matcher; true recall-miss is lower and
> some of these are actually thrash-buried findings. Cutting thrash (H2/H11) is expected to
> help this bucket too, not just the found-but-lost ones.
