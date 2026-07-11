# Evaluation harness

`arcticswarm-eval` runs the agent over a benchmark dataset and scores the
results. The first open-source release supports these benchmarks:

- **BrowseComp** — open-web research (live search via Brave/Serper/Tavily).
- **BrowseComp-Plus** — corpus research over a fixed document set, via the
  pluggable corpus retriever (`web.corpus_backend` = `stub` | `cortex` |
  `local`).
- **EvoBrowseComp** — open-web research over *evolving* knowledge
  (arXiv:2606.13120). English subset: 400 complex questions.

For end-to-end run commands on all three model backends (Claude, GPT-5,
self-hosted Qwen 3.5), see the top-level [`README.md`](../../README.md).

---

## Quick start

```bash
arcticswarm-eval -c conf/bench/browsecomp.yaml \
  llm.model=claude-sonnet-4-5 \
  azure.enabled=true \
  eval.judge_model=gpt-4-1-dev \
  eval.output=results/bc_sonnet45
```

Credentials live in a repo-root `config_files.json` (copy
`config_files.template.json` and fill it in, or point `ARCTICSWARM_SETTINGS_PATH`
at another file). See the top-level README for the key layout. Public
Anthropic / OpenAI endpoints are the default.

> The eval judge runs after each question and never sees or alters the agent
> trajectory. We standardize on the Azure-hosted GPT-4.1 deployment
> (`azure.enabled=true eval.judge_model=gpt-4-1-dev`), which needs
> `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` in `config_files.json`. A
> Claude or self-hosted judge also works (`azure.enabled=false
> eval.judge_model=claude-sonnet-4-5`).

---

## Datasets

| Dataset id            | Benchmark        | Default CSV |
|-----------------------|------------------|-------------|
| `BROWSECOMP_V1`       | BrowseComp       | `eval/data/browsecomp_v1.csv` (+ subset CSVs) |
| `BROWSECOMP_PLUS_V1`  | BrowseComp-Plus  | `eval/data/browsecomp_plus_v1.csv` (+ subsets) |
| `EVOBROWSECOMP_V1`    | EvoBrowseComp    | `eval/data/evobrowsecomp_v1.csv` (English, 400) |
| `SEAL0_V1`            | SEAL-0 (SealQA)  | `eval/data/seal0_v1.csv` (111) |
| `KBROWSECOMP_V1`      | K-BrowseComp (Korean) | `eval/data/kbrowsecomp_v1.csv` (300 verified) |

Select the question set with `eval.csv_path=...` and cap cases with
`eval.limit=N`. Each dataset is scored by its dedicated LLM judge
(`eval.judge_model`, run after the agent completes — it never sees or alters
the agent trajectory).

The EvoBrowseComp CSV is regenerated from HuggingFace with:

```bash
python -m arcticswarm.eval.data.external.evobrowsecomp \
  --output arcticswarm/eval/data/evobrowsecomp_v1.csv   # --language en (default)
```

---

## Common knobs

| Flag | Effect |
|------|--------|
| `eval.parallel=N`          | concurrent cases |
| `eval.timeout=S`           | per-case wall-clock budget |
| `eval.max_retries=N`       | retries on transient errors |
| `eval.checkpoint_interval` | flush partial results every N cases (resumable) |
| `eval.judge_model`         | judge model id (standard: `gpt-4-1-dev` with `azure.enabled=true`; also `claude-sonnet-4-5`, `gpt-5`) |
| `eval.judge_model_base_url`| judge endpoint (e.g. a self-hosted vLLM judge) |

Self-hosted judges/agents (Qwen via vLLM) are configured with
`llm.agent_model_base_url` / `eval.judge_model_base_url`; see the top-level
README and `conf/bench/browsecomp_qwen.yaml`.

---

## Output

Each run writes `results/<name>/` with per-case trajectories and a
`report.json` (accuracy + token/latency metrics). Browse runs interactively
with the bundled viewer:

```bash
python -m viewer.server   # open the printed localhost URL
```
