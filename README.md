# ArcticSwarm

**A state-of-the-art, fully open-source multi-agent framework for long-horizon web research — and the harness that reproduces it.**

ArcticSwarm runs a *swarm* of LLM agents that search the web, read pages, reason, and **cross-examine each other's findings** before committing to an answer. Its central idea is to **defer early consensus**: instead of sampling N correlated rollouts and majority-voting (which converges too fast on a plausible-but-wrong hypothesis), ArcticSwarm keeps diversity alive in *two* places —

1. **Diverse generation.** Subagents run with distinct profiles, tools, context budgets, and *private* search trajectories in **isolation mode** (they may post findings to a gated bulletin board but cannot read it), so you get N rollouts from N priors rather than N samples from one posterior.
2. **Diverse review.** When search completes, agents enter collaboration mode and peer-review findings, issuing structured **Challenge / Alternative / Verified** verdicts that feed back into the board and reshape ongoing search — turning selection into an active control signal instead of a post-hoc vote.

The result is the **best published accuracy on BrowseComp and BrowseComp-Plus**, ahead of the model providers' own deep-research systems and the strongest open-source swarms — and it transfers across model families (GPT-5, Claude, and a fully **self-hosted open Qwen 3.5**).

## Headline results

| Benchmark | Model | Model-provider baseline | Best open swarm | **ArcticSwarm** |
|---|---|---|---|---|
| **BrowseComp** (full, 1266 q) | GPT-5 | 54.9% (OpenAI Deep Research) | 63.4% (MiroFlow) | **73.6%** |
| **BrowseComp** (full, 1266 q) | Sonnet 4.5 | 43.8% (provider) | — | **50.2%** |
| **BrowseComp-Plus** (full, 830 q) | GPT-5 | 72.9% (provider) | 66% (MiroFlow) | **88.3%** |
| **EvoBrowseComp** | Qwen 3.5-27B *(open, self-hosted)* | 25% (official reported) | — | **42.2%** |

<sub>Numbers from the ArcticSwarm paper (Table 1) and additional runs. Baselines: OpenAI Deep Research, MiroFlow, KARL (68.3% on BC-Plus), and the Opus 4.5 system card. Every number is reproducible from a shipped config in `conf/bench/` with the GPT-4.1 judge — see [§5](#5-run-the-benchmarks). The same orchestration drives gains across model families: with a fully open, self-hosted **Qwen 3.5-27B** it reaches **79.6% on BrowseComp-Plus** and **42.2% on EvoBrowseComp** (vs 25% officially reported) — evidence that the orchestration, not just model scale, is the dominant factor.</sub>

---

## 1. Install

```bash
pip install -r requirements.txt   # pinned, validated versions
pip install -e . --no-deps        # the arcticswarm + arcticswarm-eval CLIs
```

(`pip install -e .` alone also works, using the looser ranges in `pyproject.toml`.) This installs two CLIs: `arcticswarm` (interactive agent) and `arcticswarm-eval` (the benchmark runner).

**PDF reading** (used by `web_fetch` on PDF results) needs **Java 11+**: `brew install openjdk` (macOS) or `apt install default-jdk` (Linux). The PDF backend auto-starts on a free port; no manual setup. Full setup notes: **[ENVIRONMENT.md](ENVIRONMENT.md)**.

## 2. Configure (credentials & the config file)

All secrets and endpoints live in a single JSON **settings file**. Copy the template and fill in what you have:

```bash
cp config_files.template.json config_files.json   # then edit config_files.json
```

- **Default location:** `./config_files.json` (repo root). It is git-ignored, so your real keys are never committed — only the template is tracked.
- **Custom path:** point `ARCTICSWARM_SETTINGS_PATH` at any file: `export ARCTICSWARM_SETTINGS_PATH=/etc/arcticswarm/prod.json`.
- The template's `__help__` block documents every key. Minimal example:

```json
{
  "api_key": "sk-ant-...",                       
  "base_url": "https://api.anthropic.com",       
  "openai_api_key": "sk-...",                    
  "openai_base_url": "https://api.openai.com/v1",
  "brave_api_key": "", "serper_api_key": "", "tavily_api_key": "",
  "jina_api_key": ""
}
```

| Key | Needed for |
|-----|-----------|
| `api_key` + `base_url` | **Claude** agent and/or judge (public Anthropic API by default) |
| `openai_api_key` + `openai_base_url` | **GPT-5** agent and the default **`openai-gpt-4.1` judge** |
| `brave_api_key` / `serper_api_key` / `tavily_api_key` | **Live web search** (any one; Brave primary, Serper/Tavily fallbacks) |
| `jina_api_key` | **Web/PDF fetch** via Jina Reader (optional — fetch falls back to Serper → `requests`) |

> Two kinds of "config" — don't confuse them: the **settings file** above (`config_files.json`, secrets/endpoints) vs. the **run configs** you pass with `--config conf/bench/*.yaml` (which benchmark/model/tools to run). The settings file is resolved once from `ARCTICSWARM_SETTINGS_PATH` or `./config_files.json`; the run configs are explicit CLI args.

> **The eval judge** runs *after* each question and never touches the agent trajectory. The shipped configs default to the public **`openai-gpt-4.1`** judge (needs `openai_api_key`). To use a Claude judge instead: `eval.judge_model=claude-sonnet-4-5`. To grade on a self-hosted judge: `eval.judge_model=<served-id> eval.judge_model_base_url=<url>`. (An Azure GPT-4.1 deployment also works: `azure.enabled=true eval.judge_model=<your-deployment>` with `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT` set.)

## 3. Get the datasets

ArcticSwarm does **not** redistribute benchmark questions or gold answers. One command downloads + decrypts them from their official sources and rebuilds every subset locally:

```bash
bash scripts/fetch_datasets.sh              # BrowseComp + EvoBrowseComp + BrowseComp-Plus + all subsets
bash scripts/fetch_datasets.sh --skip-evo   # skip EvoBrowseComp (which needs HuggingFace)
```

This writes the CSVs under `arcticswarm/eval/data/` (git-ignored — regenerate, don't commit). See **[DATASETS.md](DATASETS.md)** for per-dataset sources, licenses, citations, and the BrowseComp-Plus retrieval-corpus setup.

## 4. Smoke test

```bash
arcticswarm-eval -c conf/bench/browsecomp.yaml eval.limit=1 eval.output=/tmp/smoke
```

## 5. Run the benchmarks

Each preset in `conf/bench/` is a complete run config; override any key on the CLI with dotted paths (`eval.output=...`, `llm.model=...`). All three model backends below use the **same orchestration** — only the model changes.

### BrowseComp (live web)

```bash
# Claude Sonnet 4.5
arcticswarm-eval -c conf/bench/browsecomp.yaml llm.model=claude-sonnet-4-5 eval.output=results/bc_sonnet45

# OpenAI GPT-5  (requires openai_api_key)
arcticswarm-eval -c conf/bench/browsecomp.yaml llm.model=gpt-5 llm.openai_base_url=https://api.openai.com/v1 eval.output=results/bc_gpt5

# Open Qwen 3.5-27B on a self-hosted vLLM endpoint
arcticswarm-eval -c conf/bench/browsecomp_qwen.yaml llm.agent_model_base_url=http://<your-vllm-host>:7777/v1 eval.output=results/bc_qwen35
```

Models whose name contains `qwen`/`tongyi` route to the vLLM backend. We served **Qwen 3.5-27B on a single 8×H200 node** (bf16 weights, fp8 KV cache) — the exact launch that produced the Qwen numbers above (point `HF_HOME` at your own HuggingFace cache):

```bash
export HF_HOME=/path/to/hf-cache
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export TORCH_NCCL_ENABLE_MONITORING=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve Qwen/Qwen3.5-27B \
  --served-model-name Qwen/Qwen3.5-27B --host 0.0.0.0 --port 7777 \
  --data-parallel-size 1 --tensor-parallel-size 8 --kv_cache_dtype fp8 \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --max-num-seqs 512 --max-num-batched-tokens 16384 \
  --mamba-cache-mode align --mamba-block-size 8 \
  --attention-backend FLASHINFER --enable-prefix-caching --async-scheduling \
  --language-model-only --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --trust-remote-code --seed 0 --disable-custom-all-reduce
```

### BrowseComp-Plus (corpus retrieval)

BrowseComp-Plus replaces live search with retrieval over a fixed corpus. The retrieval backend is **pluggable** via `web.corpus_backend`, so the harness runs out of the box:

| `web.corpus_backend` | What it does | Setup |
|---|---|---|
| `stub` (default) | No real retrieval — pipeline runs end-to-end for smoke tests; scores will be low | none |
| `local` | Retrieves from a local corpus JSONL with a built-in scorer — template for your own retriever (BM25/embeddings) | a corpus `.jsonl` at `web.corpus_local_path`; see `arcticswarm/tools/corpus_retriever.py` |
| `cortex` | Retrieves from a Snowflake **Cortex Search** service (the paper's backend) | a Snowflake account + Cortex Search service over the BC-Plus corpus + a PAT |

```bash
arcticswarm-eval -c conf/bench/browsecomp_plus.yaml llm.model=gpt-5 llm.openai_base_url=https://api.openai.com/v1 eval.output=results/bcp_gpt5
```

The BrowseComp-Plus corpus (`Tevatron/browsecomp-plus-corpus`) and how to wire each backend are documented in **[DATASETS.md](DATASETS.md)**. The `corpus_backend: cortex` coordinates live (commented, with placeholders) in each `conf/bench/browsecomp_plus*.yaml`.

### EvoBrowseComp

```bash
# GPT-5
arcticswarm-eval -c conf/bench/evobrowsecomp.yaml llm.model=gpt-5 llm.openai_base_url=https://api.openai.com/v1 eval.output=results/evo_gpt5
```

Open **Qwen 3.5-27B** — reproduces the **42.2%** in the table above (serve Qwen with the vLLM command in [§5 BrowseComp](#5-run-the-benchmarks), then point the endpoints at your own hosts):

```bash
export ARCTICSWARM_SETTINGS_PATH=/path/to/config_files.json   # optional; defaults to ./config_files.json
arcticswarm-eval --config conf/bench/browsecomp_qwen.yaml \
  eval.csv_path=arcticswarm/eval/data/evobrowsecomp_v1.csv \
  'eval.datasets=[EVOBROWSECOMP_V1]' \
  eval.output=results/evobrowsecomp_qwen \
  llm.agent_model_base_url="http://<your-vllm-host>:7777/v1" \
  eval.parallel=24 eval.judge_model=openai-gpt-4.1
```

(`odl.hybrid_url` is optional — omit it to auto-start the PDF backend locally.)

**Common knobs:** `eval.limit=20` (cap cases) · `eval.parallel=8` (concurrency) · `eval.csv_path=...` (swap the question set) · `eval.repeat=3` (multi-run with stddev).

## 6. Custom evaluation

Evaluate on **your own dataset and your own judge rubric without editing any framework code** — point a config at your CSV and (optionally) a judge-prompt template:

```yaml
eval:
  csv_path: path/to/my_dataset.csv
  datasets: [MY_DATASET]
  custom_judge_prompt: path/to/my_rubric.txt   # optional; {question}/{response}/{correct_answer}
```

```bash
arcticswarm-eval -c conf/bench/custom_example.yaml eval.output=results/my_eval
```

Full walkthrough (CSV format, rubric contract, example): **[docs/custom_evaluation.md](docs/custom_evaluation.md)**.

## 7. Results

Each run writes `results/<name>/` — per-case trajectories plus a `report.json` with accuracy and detailed metrics. Browse a run interactively by passing its result path:

```bash
python -m viewer.server PATH_TO_RESULT_FILE   # local-only by default; pass --tunnel to expose a public URL
```

## 8. Content cache (within-run dedup)

`web_fetch`/`pdf_read` are slow, and the same URL is often requested more than once **within a single question** (32–43% of fetches are cross-agent duplicates in a swarm). A per-question, disk-backed cache deduplicates these so a URL fetched once during a question returns from disk for the rest of that question. **Every search and fetch is otherwise live — there is no cross-run cache.**

- **Location:** `{eval.output}/cache/content/<case_id>/` (one subdir per question). Enabled by default; `enable_content_cache=false` disables it.
- **Scope:** per question only — never shared across questions or runs, so a new run always re-fetches live. Hits are byte-identical to live fetches.

## Models at a glance

| Backend | `llm.model` | Provider | Extra settings |
|---|---|---|---|
| Claude Sonnet 4.5 | `claude-sonnet-4-5` | Anthropic (public) | `api_key` |
| OpenAI GPT-5 | `gpt-5` (or a variant you have) | OpenAI (public) | `openai_api_key` + `llm.openai_base_url` |
| Qwen 3.5-27B | `qwen3.5-27b` | self-hosted vLLM | `llm.agent_model_base_url` |

## Citation

If you use ArcticSwarm, please cite the paper:

```bibtex
@inproceedings{arcticswarm,
  title     = {ArcticSwarm: Deferring Early Consensus in Long-Horizon Multi-Agent Research},
  author    = {Yoon, Soyoung and Liu, Boyi and Wang, Yite and Wu, Ruofan and Xu, Canwen and Kuang, Nikki Lijing and Yao, Zhewei and Hwang, Seung-won and He, Yuxiong},
  booktitle = {Under review},
  year      = {2026}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)) — covers ArcticSwarm's code only. Benchmark datasets retain their own upstream licenses; see [DATASETS.md](DATASETS.md).
