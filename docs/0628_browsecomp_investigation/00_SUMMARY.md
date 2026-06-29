# 0628 BrowseComp Ceiling — Executive Summary (read me first)

_Status as of ~13:00Z 2026-06-28. Live numbers land in `03_results.md`; this is the
narrative. Headline run still in flight — see "RESULTS" below, updated as it completes._

## The ask
Break the ~61% BrowseComp ceiling (Qwen3.5-27B swarm). 61% is the provider's SINGLE-agent
number; a swarm should beat it. Proxy target: beat baseline_120 = **55.8%** on the
representative-120 (and the 300-set baseline = **0.6033**).

## What I found (full detail: `02_hypotheses.md`, `05_case_by_case.md`)
The ceiling is NOT one thing. From per-case + trajectory analysis of 0625_1-5 + baseline_120
(≈900 cases) and a one-by-one read of 24 trajectories:

1. **Answer SELECTION/RETENTION, not just recall** — in **≥31% of wrong cases the correct
   answer is already in the swarm's own findings** (task summaries / BBS / subagent messages)
   but never reaches the final answer. The swarm finds it and loses it.
2. **Thrash is monotonically lethal** — >200 web_searches → <0.28 acc (vs 0.76 at <50);
   >800 LLM calls → 0.07. Root enabler: **`subagent_max_turns=0` (a no-op → 1200)**, so
   browsing subagents are ~unbounded.
3. **Timeouts are a SYMPTOM of thrash** — cases that hit the 9305s wall score 0.20 (≈22% of
   the full set). Already mitigated by `rerun_timeouts`, but the real fix is removing the cause.
4. **Compaction amplifies loss** — ANY compaction → acc 0.67→0.22; the reactive (prompt-too-
   long) path is lossy (drops 2/3 of messages).
5. **The Redmond mechanism** — the reviewer gate needs BOTH a builder AND a dedicated-auditor
   verdict; the auditor is prompted to challenge, so it overturns majority-found correct
   candidates → swarm gives up.
6. **The killer concrete bug** — found-but-blocked cases: the orchestrator finds the answer,
   spawns a `verify-*` task, then BLOCKS on `wait_for_tasks` until the wall; the timeout
   force-report then dumps raw BBS posts (not a committed answer) → judge can't extract → 0.
7. NOT the problem: compute/python (0 uses, calculator no-diff). Some "ceiling" is judge/
   reference-string noise (e.g. found "Masato Kato 加藤正人", reference typo'd "Masata Kato").

## What I changed (config + code, all qwen-gated, default-off — `code_changes.md`)
**Config (the `_eff` lever bundle):** `subagent_max_turns=70`, `max_subagents=6`,
`disable_builder_idle=true`, `min_dedicated_reviewers=0` (drop the auditor veto),
`enable_candidate_emergence_sweep=false`, `enable_empty_answer_recovery=true`.
**Code (3 flags, unit-tested, pushed to `main`):**
- `surface_bbs_candidates` — prepare_report appends a verified-candidate digest so a
  compacted-away answer is re-surfaced before the final answer; AND the **timeout
  force-report now commits the verified candidate** instead of dumping raw posts (fixes #6).
- `compaction_prune_junk` — selective-delete of certainly-wrong paths from the compaction
  summarizer input (your idea), implemented safely (summarizer-input copy only).

## Round-1 diagnostic (28 hard cases, config-only, OLD code)
Clean-wins **13/13 preserved** (no regression), give-ups **flip**, BUT answer-loss **0/6**
and thrash **0/3** — they still ran to the timeout, and the OLD force-report scored them 0.
**This is exactly why the v2 force-report fix matters** (it ran without it).
Also: thrash caps cut cases to ~150-320 turns / ~1-2.6k s (vs 344 turns / 6931s) — ~3× faster.

## RESULTS (headline) — _updating live_
| Run | set | config | acc | vs baseline | notes |
|---|---|---|---|---|---|
| baseline_120 | rep-120 | noneff (orig) | **0.558** | — | reference |
| 0625 combined | complement | noneff (orig) | **0.603** | — | reference (full-set-ish) |
| **v2** aggressive eff + code | rep-120 (72/120) | thrash caps + min_ded=0 + code | **0.597** | base 0.667 on same 72 → **NET −5** | **REGRESSION** — see below |
| **v3** verify-kept + gentle + code | rep-120 | gentle caps, verify ON, + retention code | _running_ | _pending_ | the corrected bet |

## ⚠️ BOTTOM LINE (honest, as of ~16:25Z — v3 still running)
Two configs tested on rep-120, same-case A/B vs baseline_120:

**v2 (aggressive efficiency: thrash caps + verification REMOVED + code):** REGRESSION —
0.597 vs base 0.667 on 72 same cases (9 regress, 4 flip, NET −5). The thrash caps +
verification removal cut search/verification that medium cases needed. **My "thrash is causal
waste" hypothesis is falsified** — most of that search was difficulty-driven. v2 = ~3× faster
but worse accuracy.

**v3 (gentle caps + verification KEPT + answer-retention code):** SMALL NET POSITIVE.
Partial (n≈87/120): **NET +2 (9 flips, 7 regress)**, v3b 0.632 vs base 0.609 on same cases.
The NET wobbled around 0 through the easy/medium cases, then turned **positive as the HARD
tail completed** (the last batches added flips with few regressions) — because the retention
code (esp. the candidate-led timeout force-report) targets exactly the found-but-blocked-at-
timeout cases, which are slow and finish last. So the code's intended benefit shows up late.
Net read: a **small but real improvement** over baseline (~+0.02-0.05), NOT a dramatic break.
_(n≈87, ~+3 flips net; final lands in `03_results.md`. The gain is within ~1-2σ of single-run
noise, so treat as "promising small win, needs multi-seed confirmation.")_

### Honest conclusion
**No dramatic ceiling break tonight, but a coherent, useful result:** the aggressive
efficiency approach (v2) REGRESSES; keeping search+verification and adding the answer-retention
code (v3) gives a **small net gain**, driven by recovering found-but-blocked-at-timeout cases.
The thorough investigation is the main deliverable.

### Takeaways
1. **Don't cut search/verification to "reduce thrash"** — it regresses (v2). The 221-vs-109
   search gap was mostly difficulty, not waste (hypothesis falsified).
2. **The answer-retention code (3 default-off flags) is a small net-positive lever.** FINAL v3b
   (120/120): **0.583 vs 0.558 baseline, NET +3 (14 flip, 11 regress) = +2.5pts** — within ~1σ
   of single-run noise but positive; gains came on the hard tail (the candidate-led timeout
   force-report recovering found-but-blocked cases). Keep baseline search+verification, add the
   flags; **confirm with a multi-seed / full-set run** before trusting it.
3. **The alt-task bug is real and I FIXED it — but forcing the sweep to run is NOT a net win.**
   Quantified: the diversity sweep is spawned 100% but **never runs in 40.3%** (it only
   `add_task`s, never dispatches a worker). I shipped `alt_task_force_dispatch` (default-off) so
   it actually dispatches (confirmed live: `dispatched=Erin/Veronica`). BUT v4 (= v3 + the fix)
   tracks **≈ baseline / slightly negative** (NET ~−1 @ n=47): *running* the sweep everywhere
   surfaces rival candidates that **distract easy cases** (regressions) about as much as they
   recover anchored hard ones. **Lesson: the fix is mechanically correct but must be SELECTIVE**
   (fire only on low-confidence / anchoring signals), not forced on every case. This + the
   subagent-not-posting gap is the next real lever — needs the selective trigger, not just the
   dispatch.




## What's NOT done / next (your call)
- **`07_present_not_selected.md` (NEW, 2026-06-29) — the ~20% present-not-selected bucket decomposed
  (28-case classification + code-grounded fixes).** COURSE-CORRECTION: **>57% of it is NOT a
  selection bug** — 36% are P0 judge/format scoring artifacts (swarm got it right: "Richard Todd" vs
  "Richard Andrew Palethorpe-Todd"; "Bridgewater State University" vs "Virtual Commons - …"), 21% are
  P1 recall-misses mis-binned by substring. True selection bug ≈ 9% of wrong cases, **dominated by P2
  "verification-veto → give-up"** (canonical Redmond_012: 10/11 verified, one mis-geocoded distance →
  emits "no monument satisfies all constraints"). Top fix (A1, small): close the `bbs.py:108` give-up
  hole in `_build_candidate_digest` — `is_verified_consensus_verdict` filters out CHALLENGE/DISQUALIF/
  UNSOLVABLE posts, so `surface_bbs_candidates` re-surfaces NOTHING in exactly the give-up cases.
  Cheapest win overall = entity-aware judge/scoring normalization (P0, ~7% of wrong cases, harness-
  level, zero model risk). Ties to H5 (reviewer overturn) + H10 (give-up).
- **`06_timing_race_and_selection.md` (2026-06-29) — the found-but-lost family decomposed.**
  500-case population scan splits the ≥31% "found-but-lost" bucket into: NEVER_FOUND ~70%,
  **PRESENT_NOT_SELECTED ~20%** (answer in orch context, overridden — case browsecomp_1253),
  **NEVER_POSTED ~7%** (Pattern B), and the newly-isolated **TIMING_RACE ~1% (H12)** — subagent
  finds the answer but posts it *after* the orchestrator commits at the `prepare_report` timeout wall
  (canonical browsecomp_1049; replicated in 192/574/834). Code-grounded root cause: **there is no
  answer-selection mechanism** (`answer = captured_report or last-assistant-text`,
  orchestrator.py:1633). Key consequence: **increasing the timeout does NOT fix it** (symptom, not
  cause; contradicts v2). Top fixes: commit-delay/report-reopen on in-flight findings (flips 1049),
  a quote/attribute-disambiguation sub-task (right-entity-wrong-substring), and a hard-vs-soft
  constraint adjudicator (flips 1253). NB: prior H1 stats *missed long-quote answers* (matcher gated
  `len≤70`).
- **RECALL/ANCHORING (~63% of wrong) — now being tested.** Re-investigation of 19 never-found
  cases: **17/19 anchored** on a wrong framing (median ~249 searches / ~13 fetches). Two runs
  live (`03_results.md`): **Run A** read-deeper (`max_tool_output_tokens=12000`) and **Run B**
  `reframe_prompt` (anti-anchor browsing prompt — the primary bet). Results pending.
- **Subagents don't always post candidates to the BBS** (834: found "Whitesnake", only in a
  raw agent msg) — surface_bbs_candidates can't recover what never reached the board.
- **alt_task_force_dispatch should be SELECTIVE** (fire on anchoring/low-confidence, not every
  case) — forcing everywhere was net-neutral (v4).
- Full 1266-set run of the best config (couldn't fit overnight at available throughput).
- Multi-seed confirmation — v3's +2.5pt is within single-run noise.
