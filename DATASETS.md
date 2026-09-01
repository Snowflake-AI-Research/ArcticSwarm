# Datasets

ArcticSwarm evaluates on public web-research benchmarks. **We do not redistribute
benchmark questions or gold answers.** Instead the repo ships:

- **loaders** that download + decrypt each benchmark from its official source, and
- **`arcticswarm/eval/data/subset_specs/`** — lists of `CONV_ID`s (no questions, no
  answers) that select the published subsets, plus the BrowseComp-Plus index map.

One command regenerates every CSV used in the paper, locally:

```bash
bash scripts/fetch_datasets.sh            # BrowseComp + BrowseComp-Plus + all subsets
```

This writes the following under `arcticswarm/eval/data/` (git-ignored — regenerate,
don't commit):

| File | Rows | How it's produced |
|------|------|-------------------|
| `browsecomp_v1.csv` | 1266 | downloaded + decrypted from OpenAI's public blob |
| `browsecomp_plus_v1.csv` | 830 | **derived from BrowseComp** (see below) |
| `browsecomp_subset_*.csv`, `browsecomp_complement_*.csv` | varies | selected from BrowseComp by `subset_specs/*.ids` |
| `browsecomp_plus_subset_*.csv` | varies | selected from BrowseComp-Plus by `subset_specs/*.ids` |

## Sources, licenses & citations

Each benchmark is governed by **its own license** — the Apache-2.0 `LICENSE` in this
repo covers ArcticSwarm's code only and does **not** relicense any dataset. Review the
upstream terms before redistributing anything you generate.

### BrowseComp (OpenAI)
- Source: <https://openai.com/index/browsecomp/> · loader pulls the official
  `simple-evals` public blob and decrypts it (per-row canary + SHA256/XOR).
- License/terms: per OpenAI's release. The encrypted distribution + canary exists to
  prevent verbatim redistribution and training contamination — which is exactly why we
  ship the loader, not the plaintext.
- Cite: Wei, Sun, Papay, et al., *BrowseComp: A Simple Yet Challenging Benchmark for
  Browsing Agents* (OpenAI, 2025).

### BrowseComp-Plus
- **Questions**: BrowseComp-Plus reuses BrowseComp's questions — a curated 830-question
  subset. So `browsecomp_plus_v1.csv` is derived directly from `browsecomp_v1.csv`
  (each row is its BrowseComp source row with `CONV_ID` `browsecomp_<i>` →
  `browsecomp_plus_<i+1>` and `dataset` = `BROWSECOMP_PLUS_V1`). The ordered source-id
  map is `subset_specs/browsecomp_plus_v1.srcids`. No separate question/answer download
  is needed.
- **Retrieval corpus** (separate, large, optional): BrowseComp-Plus is a *corpus-grounded*
  benchmark — agents retrieve from a fixed corpus rather than the live web. The corpus +
  prebuilt indexes are published on HuggingFace under
  [`Tevatron/browsecomp-plus-corpus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)
  (and `Tevatron/browsecomp-plus-indexes`). Point the corpus retriever at it via the
  `corpus_backend` block in `conf/bench/browsecomp_plus*.yaml`. The paper uses Arctic
  Embed L v2.0 for hybrid retrieval; published baselines typically use Qwen3-Embed-8B
  (read the numbers with that retriever-side difference in mind). See the dataset cards
  for licenses and the source paper.

## Subsets

The subset specs under `subset_specs/` are plain `CONV_ID` lists in selection order
(e.g. `browsecomp_subset_representative.ids`, the 120-question representative slice used
throughout the paper). `scripts/build_subsets.py` filters the regenerated base CSVs by
these specs — so subsets are reproduced exactly without storing any question text.

## Custom datasets

To evaluate on your own data, drop a CSV in the unified format (see
`arcticswarm/eval/data/external/utils.py::write_unified_eval_csv`) and point a config at
it with `eval.csv_path` + `eval.datasets`. See the **Custom evaluation** section of the
README for an end-to-end example, including a custom judge rubric.
