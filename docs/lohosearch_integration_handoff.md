# LoHoSearch integration + first Qwen3.5-27B run — handoff

Branch: `lohosearch-qwen27b` (off `internal`)

## What LoHoSearch is

[LoHoSearch](https://huggingface.co/datasets/meituan-longcat/LoHoSearch)
(arXiv [2606.12837](https://arxiv.org/pdf/2606.12837), Meituan LongCat, MIT) is a
544-question benchmark for **long-horizon** search. Where BrowseComp is
human-authored (and now largely saturated — top models >90%), LoHoSearch is
generated from a knowledge graph over 7M+ English Wikipedia entities with
Wikidata type annotations, which lets the authors systematically maximize search
space size and constraint depth rather than being bounded by what an annotator
can hold in their head.

Each item is a multi-constraint entity-identification puzzle: an instruction
("Identify the horse that satisfies all of the following conditions:") plus a
numbered list of obfuscated constraints that chain several hops through
placeholder entities (`Municipality A`, `Horse C`, `Person E`) with deliberately
vague anchors ("early 1960s", "approximately 2.3 square kilometres").

Measured on the decrypted data: mean 1041 chars/question (max 2266), mean 2.9
numbered constraints (range 1–4), mean 3.1 words/answer. Answers are short
KG-verified-unique Wikipedia-title-style strings, often carrying the
disambiguator: `Sky High (horse)`, `Vogorno`, `Alloway, Queensland`.

**Published baselines** (their 2-judge mean): GPT-5.5 **34.74%** (best) ·
DeepSeek-V4-Pro 15.99 · Claude-Opus-4.6 15.62 · Kimi-K2.6 15.53 ·
Gemini-3.1-Pro 13.32 · GLM-5.1 12.77 · MiniMax-M2.7 2.48. Expect low absolute
numbers.

### Two claims worth testing against our own findings

1. The paper reports context-management strategies buy **at most +6.8%** on
   LoHoSearch — far less than on prior benchmarks. That is in direct tension
   with our `web.force_bbs_isolation` result (+2.8pt proven on BCP) and the
   answer-retention cluster. Good adversarial test of whether our levers
   transfer to a benchmark built to resist them.
2. The dataset ships **no domain/category/difficulty column** despite the card
   advertising 11 domains (and shipping `assets/domain_distribution.png`). So
   the per-subset breakdowns our BCP ablations lean on are **not available**
   without deriving them ourselves.

## Files added/changed

| File | What |
|:--|:--|
| `arcticswarm/eval/data/external/lohosearch.py` | Loader + decryption |
| `arcticswarm/eval/data/lohosearch_v1.csv` | All 544 rows, unified eval format |
| `arcticswarm/eval/data/external/__init__.py` | Register loader |
| `arcticswarm/eval/judge.py` | `judge_lohosearch` |
| `arcticswarm/eval/runner.py` | `LOHOSEARCH_V1` dispatch + `CANONICAL_METRIC_ONLY_DATASETS` |
| `conf/bench/lohosearch_qwen.yaml` | Run config |
| `scripts/launch_lohosearch_qwen.sh` | Launcher |
| `scripts/make_brave_only_settings.py` | Strip dead provider keys |

### Decryption

Rows are obfuscated (to keep them out of crawler training data) but **not
secret**: each row carries its own `canary` column, and each field is
`base64(XOR(plaintext, SHA256(canary)-derived keystream))`. The canary is the
password and it ships in-band, so no external secret is needed. The loader
reimplements the authors' `derive_key`/XOR rather than shelling out to their
`decrypt.py`.

All 544 rows share one canary. **The canary is used for decryption only and is
deliberately NOT persisted into row metadata** — only `source_index` is kept.

`csv.field_size_limit(sys.maxsize)` is required; the base64 ciphertext of a
2.3K-char question trips csv's default 128K field cap on some rows.

### Judge — read this before comparing to the paper

LoHoSearch's protocol **averages two gradings** per question:

1. BrowseComp grading prompt, GPT-4.1 as judge
2. SimpleQA grading prompt, Qwen2.5-32B as judge

(their stated rationale: averaging complementary judges cancels each one's
over-strictness/over-leniency).

**We score only #1** — the BrowseComp prompt on Azure GPT-4.1. So our number is
the GPT-4.1 component, **not** the published 2-judge mean, and is not
apples-to-apples with their table. Adding the Qwen2.5-32B SimpleQA pass is the
obvious follow-up if we want a directly comparable figure.

Reusing the BrowseComp grader is sound here — LoHoSearch answers are short
unique entity names, exactly its design target. Verified on a live GPT-4.1 call
that it accepts `Sky High` for gold `Sky High (horse)` (strips the
disambiguator), rejects `Phar Lap`, and returns incorrect for an empty answer.

## Two traps hit while wiring this up

### 1. `rebuild_from_trajectories` is a RESUME flag

The 0729 BCP command carried `eval.rebuild_from_trajectories=true
eval.rerun_errors=true eval.rerun_timeouts=true` because it was a *3rd resume
pass* over an existing output dir. Copied verbatim onto a fresh run they abort
at startup:

```
FileNotFoundError: No trajectories/ directory found in <output_dir>
```

They are now opt-in via `RESUME=1` in the launcher. First pass fresh, later
passes over the same `eval.output` with `RESUME=1`.

### 2. `web.search_provider_order` cannot restrict providers, only reorder them

Tavily and Serper are out of credit. Setting
`web.search_provider_order=["brave"]` looks like it should pin to Brave, but
`WebSearchTool` **re-appends every known provider the order omits**
(`arcticswarm/tools/web_search.py`, "Availability is gated separately by API
keys, so reordering never drops a provider"). The first smoke run duly burned a
Tavily 402 plus a Serper 400 "not enough credits" on *every* Brave miss —
roughly 2 doomed requests per miss, on a benchmark whose vague anchors invite a
lot of near-duplicate querying.

Availability is gated on **API-key presence**, so the only config-level fix is a
settings file with those keys removed:

```bash
python scripts/make_brave_only_settings.py \
  /code/users/soyoung/snowswarm_settings_cortex.json \
  /code/users/soyoung/snowswarm_settings_brave_only.json
```

The launcher defaults `SETTINGS` to that file. Verified: the Brave-only run logs
zero Tavily/Serper lines and the health check probes only Brave + Jina. Note the
keys are also read from `TAVILY_API_KEY` / `SERPER_API_KEY` env vars, which must
stay unset. `web.search_provider_order` is still set in the config, to record
intent.

Consequence for PDFs: `pdf_read`'s Serper fallback is dead too, so PDF
extraction rides **Jina Reader** plus the local OpenDataLoader/docling path.
`jina_api_key` is present and health-checks OK.

## Cluster setup

**Pod: `soyoung-rebuttal-temp-oneday-1-0-f2vpn`** (ns `mltraining-dev`).

The first attempt ran on `soyoung-dev-cpu-in` as originally asked, and **that pod
cannot host this eval**: its container cgroup caps memory at **7.6 GiB**
(`memory.max` 8137830400) with 2 CPUs. Three concurrent cases × 16 subagents
exceeded it and the eval was SIGKILLed mid-case — `memory.events oom_kill 8`,
`memory.peak` 8144232448 > `memory.max`. Its 2 CPUs also explain the ~45 min per
case. The GPU pods have **2.04 TB / 184 CPUs**, so the run moved there; the eval
client is I/O bound, so co-locating it with a serving vLLM is fine (that pod was
already serving at load 10/184).

Check before choosing a pod:
```bash
kubectl get pods -n mltraining-dev -o custom-columns=\
'NAME:.metadata.name,MEM:.spec.containers[0].resources.limits.memory,CPU:.spec.containers[0].resources.limits.cpu'
```

Two things that made this OOM hard to read:

- **It masquerades as a scoring failure.** `kubectl get pod` reports
  `restarts=0` and an empty `lastState.terminated.reason` — the *container*
  never died, only the process inside it. The evidence is `Killed` in the
  launcher's stdout plus the cgroup counters.
- **`report.json` is also written mid-run as a checkpoint**, not only at the
  end, so its presence does not mean the run finished. A completion check must
  also confirm no eval process is alive. The first driver aborted on exactly
  this: it read the checkpointed report of an OOM-killed run, found no verdicts,
  and reported "scoring is broken" when scoring had simply never been reached.
  (It also looked for a key named `correct`; the real field is
  `per_case[].judge_correct`.) Both fixed.

`/code` is shared Lustre across pods, so the worktree is visible from any of
them — only the `/data-fast` venv has to be rebuilt per pod.

Endpoint verified from that pod — `GET http://soyoung-rebuttal-temp-oneday:7777/v1/models`
returns served id `Qwen/Qwen3.5-27B`, `max_model_len` 262144, matching the
config exactly. Port 7777 resolves fine from in-cluster (no dss policy patch
needed here).

Per the served-id trap: `llm.model` stays **bare** (`qwen3.5-27b`) and the
slashed id goes in `llm.vllm_served_model_id` (`Qwen/Qwen3.5-27B`). A slash in
`llm.model` silently bypasses the override → 404.

### Why a separate worktree

The primary checkout `/code/users/soyoung/ArcticSwarm` sits at an old commit
(`ae9e3ce`) and carries **uncommitted local edits to
`arcticswarm/eval/cli.py` and `arcticswarm/eval/recovery.py`** (68 + 47 lines).
Those were left alone. Instead:

```
/code/users/soyoung/ArcticSwarm_lohosearch     # git worktree, branch lohosearch-qwen27b
/data-fast/soyoung/venvs/lohosearch            # its own uv env
```

`activate_snowswarm.sh` hardcodes `REPO=/code/users/soyoung/ArcticSwarm` and
uv-syncs from it, so sourcing it from a worktree would import the *primary*
checkout's code. Hence `VENV=` in the launcher, which activates the worktree's
own env and skips that script. `/data-fast` is ephemeral node-local NVMe — after
a pod restart, re-run `uv sync` in the worktree with
`UV_PROJECT_ENVIRONMENT=/data-fast/soyoung/venvs/lohosearch`.

## Running it

```bash
kontrol in connect soyoung-cpu-in
cd /code/users/soyoung/ArcticSwarm_lohosearch

# 3-case smoke
SMOKE=3 VENV=/data-fast/soyoung/venvs/lohosearch bash scripts/launch_lohosearch_qwen.sh

# full 544q
VENV=/data-fast/soyoung/venvs/lohosearch bash scripts/launch_lohosearch_qwen.sh

# resume a partial run (same OUT_DIR)
RESUME=1 VENV=/data-fast/soyoung/venvs/lohosearch bash scripts/launch_lohosearch_qwen.sh
```

Config is the 0729 BCP swarm settings (16 subagents, xhigh/xhigh, both reviewer
gates + `enforce_alt_task`, `surface_bbs_candidates`,
`enable_empty_answer_recovery`, `compaction_prune_junk`,
`reject_refusal_reports`, near-dup hard stop 12, all global caches off,
`compact_tokens=180000`, `parallel=9`) minus the corpus backend.

`enforce_alt_task` should be especially load-bearing here: on a multi-constraint
chain a partially-satisfied candidate looks convincing but violates a later
constraint, and the alt sweep is what surfaces the rival entity.

`parallel=9` (not 18) is deliberate — 16 subagents × parallel cases saturates a
single 27B endpoint, and the resulting timeouts cost more accuracy than the lost
concurrency buys.

## Open follow-ups

- Add the Qwen2.5-32B SimpleQA judge pass for a number directly comparable to
  the paper's 2-judge mean.
- Derive domain labels if we want per-subset breakdowns (not in the dataset).
- `train.csv` (2000 rows, pipeline-generated, **not** human-verified) is
  available via `--file train.csv` if we ever want training/dev data.
- Consider whether `web.search_provider_order` should be able to genuinely
  restrict the chain (currently reorder-only), rather than needing a stripped
  settings file.
