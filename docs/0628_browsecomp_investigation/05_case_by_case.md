# 05 — One-by-one trajectory findings

Per-case deep read of 24 stratified trajectories (8 answer-loss, 4 recall-miss, 3 give-up,
5 clean-win, + extras), via digest analysis cross-checked against `raw/master_*.jsonl`.
This sample is ENRICHED for failure modes, so the histogram is mechanism-confirmation, not a
population estimate (population %s are in `02_hypotheses.md`).

## Mechanism histogram (24 cases)
| primary mechanism | n | recommended lever |
|---|---|---|
| recall_miss_never_found | 11 | better_search_recall (HARD) |
| clean_success | 5 | — |
| found_lost_to_compaction / not-committed | 4 | force_commit / surface_bbs_candidates |
| thrash_ran_out_of_budget | 3 | thrash_caps + force_commit |
| found_wrong_candidate_selected | 1 | surface_bbs_candidates |

`found_correct_answer_internally`: of the 19 non-clean cases, **7 found the answer internally
but lost it** (cheap to recover), **12 never found it** (hard). Matches the population split.

## Refined patterns (the actionable part)

### Pattern A — Found, then BLOCKED on `wait_for_tasks` until the deadline (RECOVERABLE)
Cases 023 (Stranger Things), 558 (New Zealand), 370 (Washington St.), 403.
The orchestrator zeroes in on the right lead, spawns a verification subtask, and its **terminal
action is a `wait_for_tasks` tool call** — it blocks on an unfinished task until the 9000s wall
kills the run, never committing the candidate it already holds. `final_answer_tail` is literally
a `{"task_names":[...],"timeout":180}` payload, not prose.
→ **Two-part fix that COMBINES:** (1) thrash caps make the case reach a normal `prepare_report`
before the wall instead of dying mid-wait; (2) `surface_bbs_candidates` + `enable_empty_answer_recovery`
force commitment of the best candidate there. Neither alone suffices — together they convert this
whole bucket. Also implies the timeout force-report path should emit a best-candidate answer, not
a raw dump / leave a dangling wait (documented as a further code lever, not yet shipped).

### Pattern B — Found inside one subagent but NEVER elevated to BBS / a verification task
Cases 834 (Whitesnake surfaced only in a raw agent message; all 15 task summaries name *rejected*
candidates), 097 (Laura in a subagent's context, never propagated to summary/orchestrator).
→ `surface_bbs_candidates` only helps if the finding reached the BBS. The deeper fix is to get
subagents to POST candidates to #key-findings (prompt/behavior) — flagged as a follow-up.

### Pattern C — Reference-string / judge artifacts (some scored-0 cases are actually correct)
Case 405: the swarm found "Masato Kato (加藤 正人)" matching every clue, but the reference is
"Masata Kato" (transliteration typo) → substring-matcher misses AND the judge may mark it wrong.
→ A few points of the "ceiling" are label/judge noise, not model error. Worth a manual judge
re-grade pass on borderline cases (won't fix via config).

### Pattern D — Recall miss = wrong-framing anchoring (the HARD ~63% of wrong cases)
Cases 274 (built a self-consistent Zwack Unicum narrative instead of Union Carbide), 349 (anchored
on "fruit-juice business", answer was a restaurant Dal Pescatore), 352 (wrong literal title), 831
(chased a "match" instead of a player+century), 550 (never found the name in 510 searches).
The swarm locks onto ONE interpretation and all subagents pursue it; it reformulates *searches*
but never reframes the *hypothesis*.
→ Needs hypothesis DIVERSITY, not more searches. The `enforce_alt_task` / `candidate_emergence_sweep`
are meant to do this but — observed repeatedly — the **`alternative-candidate-sweep` task is spawned
yet stays PENDING with 0 tool_uses** (021, 323, 349, 403): the diversity mechanism often never runs.
This is the biggest open lever for the hard bucket and a likely real bug (alt task spawned too late
/ no worker claims it). Flagged for follow-up; not safely fixable unattended tonight.

## Implications for the config sweep
- Thrash caps + commit/surface levers should convert Pattern A (and help B) — Round 1
  (R1b/R1c) + Round 2 (code flags) test this. (Empirically: v2 regressed, v3 ≈ baseline —
  see `03_results.md`.)
- Pattern D (recall/anchoring, ~63% of wrong) is largely untouched by tonight's levers — it
  caps how far config/retention changes can go. The alt-task-never-runs bug + subagent-posting
  are the next frontier.

## QUANTIFIED: the diversity mechanism (alt-candidate sweep) silently fails ~40% of the time
Scanned 596 cases across 0625_1-5 (`.invest_0628/alttask_scan.py`):
- **100%** of cases spawn an alt/rival/contrarian task (`enforce_alt_task` +
  `enable_candidate_emergence_sweep` always fire).
- **40.3% (240/596): the alt task NEVER RUNS** — spawned, left `pending`, `tool_use_count=0`.
  No worker ever executes it. (544 alt-task instances are `pending` vs 350 `completed`.)
- Accuracy: cases where the alt task **ran = 0.444** vs **never-ran = 0.404** (+4 pts when it
  works; confounded by difficulty, but the 40% never-run rate is a clean bug).
- **Likely cause:** the alt task is spawned LATE (at prepare_report / candidate-emergence time)
  and the run reaches its report / timeout before an idle worker claims+executes it; or no
  worker is free. → **#1 recommended fix:** ensure the contrarian/alt task is spawned EARLY and
  is actually claimed+run (block report until it runs, or reserve a worker). This targets the
  recall/anchoring bucket — the real ceiling — far more than the efficiency/retention levers did.
