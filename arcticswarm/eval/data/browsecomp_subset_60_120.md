# BrowseComp representative subsets (60 / 120)

`browsecomp_subset_60.csv` (60 questions) and `browsecomp_subset_120.csv` (120
questions) are **disjoint, accuracy-matched, topic-diverse** subsets of the full
`browsecomp_v1.csv` pool (1266 questions). They exist so we can iterate on
BrowseComp quickly and cheaply while still getting numbers that track the full
set: the 60 is the fast smoke / ablation subset, the 120 is the higher-precision
subset for promising changes.

Both files are stored via Git LFS (see `arcticswarm/.gitattributes`: `*.csv ->
lfs`). They use the standard eval CSV schema (`CONV_ID`, `TURN`, … — same columns
as `browsecomp_v1.csv`), so they are drop-in for `eval.csv_path`:

```yaml
# conf/bench/browsecomp.yaml
eval:
  csv_path: arcticswarm/arcticswarm/eval/data/browsecomp_subset_60.csv   # or _120
```

## How the subsets were selected

The selection is **stratified + constrained**, not a plain random sample, so the
subsets reproduce full-set behavior by construction. Pipeline (run on the
`syoon_bcp60_eval` branch; the builder scripts themselves are intentionally not
vendored here — see "Reproducing" below):

1. **Per-question master table.** For every one of the 1266 questions we recorded
   correctness from multiple full eval passes of three models — **Sonnet 4.6**
   (`s46`), **GPT‑5** (`gpt5`), and **Sonnet 4.5** run twice (`s45a`, `s45b`) —
   plus two flags: `long_context` (question needs a large fetch/PDF context) and
   `outlier` (degenerate/known-bad cases). Topics were assigned from a fixed
   taxonomy.

2. **Stratify the clean pool.** Drop outliers, then bucket the remaining
   questions by the joint cell `(s46_correct, gpt5_correct)` — four cells:
   both‑right, s46‑only, gpt5‑only, both‑wrong.

3. **Fix per-cell counts to match marginal accuracy.** Per-cell integer counts
   are chosen so each subset's **marginal accuracy for BOTH Sonnet 4.6 and GPT‑5
   matches the full set to within <1% by construction**:
   - 60: `{(1,1):35, (1,0):8, (0,1):8, (0,0):9}` → 43/60 correct for each model (~71.7%).
   - 120: `{(1,1):70, (1,0):16, (0,1):16, (0,0):18}` → 86/120 correct for each model (~71.7%).

4. **Seeded search over which questions fill those counts.** A deterministic
   randomized search (`random.Random(12345)`, 100k restarts) picks the actual
   questions to minimize a weighted objective:
   - **topic diversity** — subset topic distribution close to the full-set
     distribution, with a coverage penalty for any taxonomy topic left at zero;
   - **Sonnet 4.5 representativeness** — keep `s45a`/`s45b` accuracy close to
     full-set (weighted 3×), so the subsets are not over-fit to just the two
     stratification models;
   - **long-context budget** — the 60 is capped at **≤5/60 (<10%)** long-context
     questions so it stays cheap to run yet still exercises the fetch/PDF
     compactors; the 120 targets **~26/120 (~21.6%)**, matching the clean-pool
     long-context fraction for cost representativeness.

5. **Disjoint, exhaustive partition.** `subset_60`, `subset_120`, and the
   remaining `rest` are mutually disjoint and together cover all 1266 questions
   (asserted in the builder). The 60 is **not** a subset of the 120 — they are
   independent draws under the same constraints.

Net result: running the 60 (or 120) gives Sonnet‑4.6 / GPT‑5 accuracy within
~1% of the full 1266-question set, with a topic mix and long-context cost profile
representative of the whole benchmark — at roughly 5% / 9% of the runtime.

## Reproducing / regenerating

The builder (`ablation_analysis/build_master.py` → `select_subsets.py`, which
emits `subsets.json` with the chosen `CONV_ID`s) lives on the `syoon_bcp60_eval`
branch and depends on local `results/<run>/report.json` eval artifacts, so it is
**not** committed here. To regenerate, check out that branch, reproduce the
master eval passes, and re-run `select_subsets.py` (seed `12345`). Only the
materialized CSVs and this note are carried forward.
