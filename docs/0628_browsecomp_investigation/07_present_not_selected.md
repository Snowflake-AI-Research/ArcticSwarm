# 07 — "Present-not-selected" decomposed + fix design (the ~20% bucket is mostly NOT a selection bug)

_Added 2026-06-29. Follows `06_timing_race_and_selection.md`. Method: 500-case population scan
(`race_scan.py`) → 28 reliable PRESENT_NOT_SELECTED cases (ref_len≥18) → per-case decision digests
(`gen_decision_digest.py`, run on pod) → 28-way sub-mechanism classification + code-grounded fix
synthesis (workflow `wf_a1f13852-52d`). Code claims verified against `arcticswarm/swarm/`._

## TL;DR (the course-correction)
Doc 06 / H1 flagged **PRESENT_NOT_SELECTED ≈ 20% of wrong cases** as the big "found-but-lost"
sub-bucket. Classifying 28 of them shows **>57% of that bucket is NOT a selection bug at all**:

| Sub-mechanism | n/28 | Real selection bug? |
|---|---|---|
| **P0 — scoring/judge artifact** | **10 (36%)** | ❌ swarm got it right; lost on name-form |
| **P1 — incidental (recall miss, mis-binned by substring)** | **6 (21%)** | ❌ ref only in a snippet/answer-key |
| **P2 — verification-veto → give-up** | **7 (25%)** | ✅ **dominant genuine bug** |
| **P3 — anchor's hard-constraint rationalized away** | **3 (11%)** | ✅ (this is case 1253) |
| **P4 — wrong candidate better-supported, no disambiguation** | **2 (7%)** | ✅ |

**True selection-failure rate ≈ 12/28 = 43% of the bucket ≈ 9% of all wrong cases** (not 20%). And
the genuine bug is **dominated by P2 give-up (7/12 = 58%)** — *not* the 1253-style rationalization
(P3) that doc 06 spotlighted (only 3/28). So the single most-leveraged swarm fix targets **give-up /
verification-veto**, and the single cheapest win overall is **judge/scoring normalization (P0)**,
which lives in the eval harness, not the swarm.

### Per-case classification
- **P0 (10):** 181, 226, 409, 456, 628, 786, 951, 961, 968, 166 — e.g. emitted "Richard Todd" vs
  ref "Richard Andrew Palethorpe-Todd"; "Bridgewater State University" vs "Virtual Commons -
  Bridgewater State University"; "The Future of Media" vs "The Future of Media Conference". Same
  entity, judged wrong on surface form.
- **P1 (6):** 019, 064, 184, 749, 993, 1218 — ref appears only inside a search snippet, a failed
  query, or even the answer-key metadata; never elevated as a candidate. (749/184 hit only the
  REFERENCE/judge blocks → these mis-inflate the population denominator; see fix #5.)
- **P2 (7):** 012, 209, 424, 520, 537, 1016, 741.
- **P3 (3):** 1053, 1217, 1253.
- **P4 (2):** 220, 1058.

---

## The dominant genuine mechanism — P2: verification-veto → give-up

Identical shape across all 7: a **correct candidate is elevated** (named in task summaries +
#consensus with a constraint tally), then a **single verification subtask flags one constraint as
failing**, and the swarm **flips consensus to "no valid candidate exists"** (or chases an
alternative). The vetoing check is almost always one of:
- **mis-measured** — `browsecomp_1016` Fort Columbia: AZ distance rejected at 1087 mi *great-circle*
  vs the question's *Google-Maps driving* method; `012` Redmond: NDLS distance "~392 m" from a
  wrong geocode.
- **contested/soft boundary** — `012` Redmond: "town founded in the 900s" hinged on a 9th-vs-10th
  century cutoff (one source said early 900s).
- **recall-miss dressed as disconfirmation** — `741` William Fry (3/5): "could not FIND article on
  New South growth" treated as "constraint FAILS"; `537` Practical Mechanics.

**Canonical case — browsecomp_012 (The Redmond Monument).** James + Marina elevated it ("All 11
Criteria Verified"); Chase called it "the strongest candidate, 7 constraints definitively verified."
Then a distance subtask measured NDLS at ~392 m (vs 60–80 m) and a date subtask contested "900s," so
#consensus posted "Redmond Monument is NOT the Valid Answer" and the swarm emitted *"No monument has
been definitively identified that satisfies all 11 constraints."* Score 0. This is the H5 smoking
gun, now shown to be a *recurring* mechanism, not a one-off.

**Root cause is structural, and code-grounded:**
1. **The auditor is built to veto on partial matches and cannot re-check.** `IDLE_REVIEW_MESSAGE_
   RESEARCH_ADVERSARIAL` (`prompts.py:1179`): *"If a candidate matches most but not all constraints,
   that is a RED FLAG — partial matches are the #1 source of wrong answers"* — and the auditor "does
   NOT have web_search or web_fetch" and is capped at 3 reasoning calls. So a mis-measured/not-found
   constraint becomes a hard veto with no recourse.
2. **The candidate-resurfacing safety net is silently disabled in exactly these cases.**
   `is_verified_consensus_verdict` (`bbs.py:85`) filters OUT any post whose head contains
   `CHALLENGE`/`DISQUALIF`/`UNSOLVABLE` (neg-markers `bbs.py:61-79`, exclusion `:108`).
   `_build_candidate_digest` (`tools.py:2612`, gated on `surface_bbs_candidates`) harvests only
   *affirmative* verdicts — so when consensus flips to a give-up, **the digest harvests nothing** and
   re-surfaces no candidate. `surface_bbs_candidates` therefore cannot rescue the P2 cases it was
   meant to.
3. **No force-commit / recovery fires.** `enable_force_submit=False` (`config.py:347`) and
   `enable_empty_answer_recovery=False` for qwen (`run_config.py:432`); and the recovery refusal
   markers (`empty_answer_recovery.py:52`) don't catch "no monument…/no valid candidate exists."

---

## Fix design (code-grounded)

### (a) P2 give-up fix — HIGHEST LEVERAGE (7/12 true failures)
Principle: *a single negative constraint must not override a strong multi-criteria positive match;
never emit "no answer" while a candidate clears the hard chain.*

- **A1 (small, low-risk, do first): close the digest give-up hole.** In `_build_candidate_digest`
  (`tools.py:2612`), when `is_verified_consensus_verdict` yields nothing, **fall back to harvesting
  the best-elevated candidate** from `#key-findings`/`#discoveries`/task summaries (the post that
  named a candidate with the highest verified-constraint tally), *ignoring* the later disqualifying
  flip. Strengthen the digest header (`tools.py:2649`): *"A single failing constraint — especially a
  distance measurement, a date-range boundary, or a 'could not find' result — is NOT grounds to emit
  'no answer'. Treat 'could not find evidence for X' as MISSING evidence, not X being FALSE. If a
  candidate clears the defining chain and ≥70% of constraints, emit it."*
- **A2: a give-up adjudication gate.** Add `_check_giveup_gate(force, timed_out)` beside
  `_check_alt_task_gate` (`tools.py:2583`): if the latest #consensus is a give-up AND an earlier
  post elevated a named candidate with a high tally, refuse once and spawn a `reasoning`
  reconciliation task — *"Re-examine ONLY the failing constraint C: is it a hard chain constraint or
  a soft/contested one (distance method, decade boundary, name-collision)? Was C disconfirmed or
  merely not-found? If soft/contested or not-found, re-instate the candidate."* New flag
  `enable_giveup_reconciliation` (default off, qwen YAML only); reuse the advisory-degrade machinery.
- **A3: widen empty-answer recovery.** Add `"no valid candidate"`, generic `"no .* (exists|found|
  satisfies)"` to `_REFUSAL_MARKERS` (`empty_answer_recovery.py:52`) and inject the A1 digest into
  `_RECOVERY_MSG` so the recovery turn sees the best candidate verbatim.
- **A4 (cheap): soften the auditor veto.** Append to `prompts.py:1179`: *"A partial match is a flag
  to re-examine the missing constraint, NOT to conclude 'no answer'. Distinguish disconfirmed
  (evidence says FALSE) from unverified (no evidence found)."*

*Existing-flag coverage:* `surface_bbs_candidates` was supposed to cover this but is **defeated by
the bbs.py:108 exclusion** (A1 closes it). `enable_empty_answer_recovery` covers only literal-empty
refusals (A3 widens it).

### (b) P3 — hard-vs-soft constraint adjudicator (3 cases incl. 1253)
- **B1 (cheap): ban the "question error" override.** In `1253` the consensus wrote "all constraints
  verified EXCEPT 'got married' which appears to be a question error." Add to `prompts.py:1179` +
  the report header: *"You may NOT dismiss a failing constraint as a 'question error'. If your
  selected candidate FAILS a HARD constraint (married / born-in-year / located-in-region) while a
  different present candidate SATISFIES it, select the candidate that satisfies it."*
- **B2: hard-constraint conflict gate** — when the selected candidate has a rationalized-away hard
  failure and a co-present rival has it VERIFIED, refuse once and spawn a `reasoning` adjudication
  that ranks candidates **hard-constraints-first** (dates/places/identity/marital = HARD;
  fame/obscurity/"advertised-vs-mentioned" = SOFT).

### (c) P4 — disambiguation among co-qualifying candidates (2 cases)
- **C1:** when ≥2 candidates each satisfy the hard constraints (220 two opium books; 1058 two valid
  "Untold" episodes), post a **disambiguation task** keyed to the question's narrowest distinguishing
  phrase, or hedge across them — don't commit to the first-found. Add as a second mode of
  `wire_candidate_emergence_hook` (`answer_verification.py:35`).
- **C2:** keep `alt_task_force_dispatch` **selective** — force-dispatch only when ≥2 candidates are
  co-present (not every case; forcing-everywhere was net-neutral per `00_SUMMARY.md`).

---

## Priority order (by population leverage)
1. **P2 give-up fix (A1 first, then A2/A3/A4)** — ~5% of all wrong cases; A1 is a tiny patch.
2. **Judge/scoring normalization for P0** — ~7% of wrong cases, **more than all selection fixes
   combined, at zero model-behavior risk.** Not a swarm change — make the eval scorer entity-aware
   (strip middle names/suffixes/venue qualifiers; accept superset/subset of the reference). **Flag
   to the harness owner; do in parallel.**
3. **P3 adjudicator (B1 cheap, B2)** — ~2% of wrong cases.
4. **P4 disambiguation (C1/C2)** — ~1.5%.
5. **Fix the scan binning** — re-gate the population "present" flag on a non-empty ref-context
   extract so answer-key-only hits (749/184) stop inflating the denominator (`scripts/alttask_scan.py`).

## Honest caveats
- **N=28; only 12 are true selection failures** → P3 (3) and P4 (2) are individually too small to
  trust. The ordering **P2 ≫ P3 ≈ P4 is robust**; exact P3/P4 split is noise.
- **P0 (10/28) rests on the single judge's own comments** ("same person, omits middle name"). This
  is the **biggest swing factor**: an entity-aware scorer reclassifies all 10 P0s to *correct* and
  shrinks the whole bucket by a third — making P2 the overwhelming residual. If instead the real
  BrowseComp scorer is stricter than these judge comments suggest, some P0s are legitimate failures.
- **The safe headline:** ≥57% of "present-not-selected" is not a selection bug; among genuine ones,
  P2 give-up dominates, and the A1 digest-hole patch (`bbs.py:108` / `tools.py:2612`) is worth
  shipping regardless of how the P0 scoring question resolves.

## Relationship to prior docs
- **H1** named "present at decision, not selected" but didn't decompose it — this doc shows the
  majority is judge-noise/recall, not selection.
- **H5** (reviewer-gate overturn, Redmond_012) is exactly the **P2** mechanism — now quantified as
  the dominant genuine sub-bug and given a concrete fix beyond `min_dedicated_reviewers`.
- **H10** (give-up) overlaps P2 but undercounted it (it scanned only literal final-answer give-ups;
  P2 includes "no candidate satisfies all constraints" non-empty reports).
- **Doc 06**'s case 1253 is **P3**, a minority (3/28) — useful but not the main lever.
