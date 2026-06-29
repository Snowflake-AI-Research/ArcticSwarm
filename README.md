# Arcticswarm

A clean-room, pure-Python multi-agent **web-research agent** and evaluation harness. It runs a swarm of LLM agents that search the web, read pages, reason, and cross-verify answers — and scores them on the **BrowseComp** and **BrowseComp-Plus** benchmarks.

This README covers exactly one thing: **how to run BrowseComp and BrowseComp-Plus on three model backends — Anthropic Claude Sonnet 4.5, OpenAI GPT-5, and a self-hosted Qwen 3.5.**

> Settings live in a repo-root `config_files.json` (copy `config_files.template.json` and fill it in). A Snowflake connection profile under `~/.snowflake/connections.toml` is read only by the optional Cortex corpus backend, which uses the `snowflake-connector-python` client for authentication. Nothing else about the harness depends on any hosted provider.

---

## 1. Install

```bash
pip install -r requirements.txt   # pinned, validated versions
pip install -e . --no-deps        # the arcticswarm + arcticswarm-eval CLIs
```

(`pip install -e .` alone also works, using the looser ranges in `pyproject.toml`.)

This installs two CLIs: `arcticswarm` (interactive agent) and `arcticswarm-eval` (the benchmark runner).

**PDF reading** (used by `web_fetch` on PDF results) needs **Java 11+**: `brew install openjdk` (macOS) or `apt install default-jdk` (Linux). The PDF backend auto-starts on a free port; no manual setup.

For full environment replication (credentials, the optional global fetch cache), see **[ENVIRONMENT.md](ENVIRONMENT.md)**. Snowflake-cluster specifics (shared env, S3 cache restore, exact paths/scripts, our vLLM command) live in **[snowflake/snowflake_specific.md](snowflake/snowflake_specific.md)**.

---

## 2. Credentials

Copy the template and fill in your keys (`ARCTICSWARM_SETTINGS_PATH` still overrides the location):

```bash
cp config_files.template.json config_files.json   # then edit config_files.json
```

`config_files.json` is git-ignored, so your real secrets are never committed (only the template is tracked). The template's `__help__` block documents every key. Minimal example:

```json
{
  "api_key": "sk-ant-...",
  "base_url": "https://api.anthropic.com",

  "openai_api_key": "sk-...",
  "openai_base_url": "https://api.openai.com/v1",

  "serper_api_key": "",
  "brave_api_key": "",
  "tavily_api_key": "",

  "jina_api_key": ""
}
```

| Key | Needed for |
|-----|-----------|
| `api_key` + `base_url` | **Claude** (agent and/or judge). Public Anthropic API by default. |
| `openai_api_key` + `openai_base_url` | **GPT-5** (set `openai_base_url` to `https://api.openai.com/v1`). |
| `serper_api_key` / `brave_api_key` / `tavily_api_key` | **Live web search** for BrowseComp (any one works; Brave is primary, Serper/Tavily are fallbacks). |
| `jina_api_key` | **Web/PDF content fetch** via the Jina Reader API (used by `web_fetch` / `pdf_read`). Optional but recommended — fetch falls back to Serper → `requests` without it. Settings key `jina_api_key` or env var `JINA_API_KEY`. |

> **The eval judge** runs *after* each question completes and never touches the agent trajectory. We standardize on the Azure-hosted **GPT-4.1** deployment (`azure.enabled=true eval.judge_model=gpt-4-1-dev`), which needs `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` in `config_files.json`. (A Claude or self-hosted judge also works — e.g. `azure.enabled=false eval.judge_model=claude-sonnet-4-5`.)

---

## 3. Run BrowseComp

BrowseComp uses **live web search**, so it works with public API keys + a search key. Base preset: `conf/bench/browsecomp.yaml` (dynamic swarm: browsing + reasoning profiles).

### Claude Sonnet 4.5
```bash
arcticswarm-eval -c conf/bench/browsecomp.yaml \
  llm.model=claude-sonnet-4-5 \
  azure.enabled=true \
  eval.judge_model=gpt-4-1-dev \
  eval.output=results/bc_sonnet45
```

### OpenAI GPT-5
```bash
arcticswarm-eval -c conf/bench/browsecomp.yaml \
  llm.model=gpt-5 \
  llm.openai_base_url=https://api.openai.com/v1 \
  azure.enabled=true \
  eval.judge_model=gpt-4-1-dev \
  eval.output=results/bc_gpt5
```
(Requires `openai_api_key` in settings. Swap `gpt-5` for the GPT-5 variant you have access to, e.g. `gpt-5.4`.)

### Qwen 3.5 (self-hosted vLLM)
Serve Qwen 3.5 on an OpenAI-compatible vLLM endpoint, then point the agent at it. A general launch (tune flags to your hardware):
```bash
vllm serve Qwen/Qwen3.5-27B \
  --served-model-name Qwen/Qwen3.5-27B --host 0.0.0.0 --port 7777 \
  --tensor-parallel-size 8 --kv_cache_dtype fp8 \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --trust-remote-code
```
(The exact cluster command we used — with our HF cache / venv paths — is in [snowflake/snowflake_specific.md](snowflake/snowflake_specific.md).) Then point the agent at the served port:
```bash
arcticswarm-eval -c conf/bench/browsecomp.yaml \
  llm.model=qwen3.5-27b \
  llm.agent_model_base_url=http://<your-vllm-host>:7777/v1 \
  azure.enabled=true \
  eval.judge_model=gpt-4-1-dev \
  eval.output=results/bc_qwen35
```
Models whose name contains `qwen`/`tongyi` route to the vLLM backend. Qwen thinking-mode knobs (`llm.enable_thinking`, `llm.vllm_temperature`, …) can be overridden; defaults follow the model card. To keep the judge self-hosted too, set `eval.judge_model` + `eval.judge_model_base_url` to your judge endpoint instead of the Azure judge above.

**Common knobs:** `eval.limit=20` (cap cases), `eval.parallel=8` (concurrency), `eval.csv_path=...` (swap the question subset; default is `browsecomp_subset_representative.csv`).

**Cortex web-search provider (optional).** By default `web_search`/`web_fetch` use the native providers (Brave/Tavily/Serper for search; Jina/Serper for fetch). To route web search through a Snowflake **Cortex `agent:run` passthrough** instead, set `web.provider`:

| `web.provider` | Tool wired | What it calls |
|---|---|---|
| `native` (default) | `WebSearchTool` | Brave → Serper → Tavily directly |
| `cortex` | `CortexWebSearchTool` | Cortex passthrough (Brave Search), Tavily/Serper as fallbacks |
| `cortex-grounding` | `CortexGroundingSearchTool` | Cortex passthrough (Brave **Grounding Context**) |

Optionally set `web.fetch_backend=cortex-grounding` to prepend a Cortex Grounding fetch (tier 0) before the native Jina→Serper→requests chain. Auth is taken from the live Snowflake session (`~/.snowflake/connections.toml`) when available, else a PAT via `web.cortex_account` (or settings `cortex_account` / `CORTEX_ACCOUNT`). These are opt-in — the harness still runs on the native provider with no Snowflake account.

**Azure GPT-4.1-dev judge (standard).** Grading uses the Azure-hosted GPT-4.1 deployment by default: `azure.enabled=true eval.judge_model=gpt-4-1-dev` (needs `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` in `config_files.json` or env). When the judge model is a `gpt-*` deployment and Azure is enabled, the Azure path takes precedence over any preset `eval.judge_model_base_url`, so the two flags are enough even on presets that bake in a self-hosted (e.g. Qwen) judge URL. To use a Claude judge instead, pass `azure.enabled=false eval.judge_model=claude-sonnet-4-5`.

---

## 4. Run BrowseComp-Plus

BrowseComp-Plus replaces live web search with retrieval over a fixed ~100K-document corpus. The retrieval backend is **pluggable** (`web.corpus_backend`), so the harness runs out of the box and you choose how documents are fetched:

| `web.corpus_backend` | What it does | Setup |
|---|---|---|
| `stub` (default) | No real retrieval — every `web_search`/`web_fetch` returns a "configure a corpus backend" notice. The pipeline runs end-to-end (useful for smoke tests), but scores will be low. | none |
| `cortex` | Retrieves from a **Snowflake Cortex Search service** over a REST API — the backend used for the numbers in the paper. | a Snowflake account + a Cortex Search service over the BCP corpus + a PAT |
| `local` | Retrieves from a **local corpus JSONL** with a built-in token-overlap scorer — a reference template to plug in your own retriever (BM25, embeddings, or your search service). See `arcticswarm/tools/corpus_retriever.py`. | a corpus `.jsonl` (`{"id","title","text"}` per line) at `web.corpus_local_path` |

The three `browsecomp_plus*.yaml` presets default to `corpus_backend: stub`. The exact Cortex Search service coordinates used for the paper are shown (commented, with placeholders) in each preset:

```yaml
# corpus_backend: cortex
# corpus_account: <your-account>
# corpus_db: <db>
# corpus_schema: <schema>
# corpus_chunked_service: <chunked-service>   # -> web_search
# corpus_service: <service>                    # -> web_fetch
```

- **To use the Cortex backend:** load the BCP corpus into a search service in your own Snowflake account, put a PAT in `~/.snowflake/connections.toml`, and either uncomment + edit the block above or pass the coordinates on the CLI (`web.corpus_backend=cortex web.corpus_account=… web.corpus_db=… web.corpus_schema=… web.corpus_chunked_service=… web.corpus_service=…`).
- **To use a local retriever:** set `web.corpus_backend=local web.corpus_local_path=/path/to/corpus.jsonl` (optionally swap the scorer in `LocalCorpusRetriever` for BM25/embeddings).

The commands below run the preset (stub by default — add the corpus settings above to retrieve for real):

### Claude Sonnet 4.5
```bash
arcticswarm-eval -c conf/bench/browsecomp_plus.yaml \
  llm.model=claude-sonnet-4-5 \
  azure.enabled=true \
  eval.judge_model=gpt-4-1-dev \
  eval.output=results/bcp_sonnet45
```

### OpenAI GPT-5
```bash
arcticswarm-eval -c conf/bench/browsecomp_plus.yaml \
  llm.model=gpt-5 \
  llm.openai_base_url=https://api.openai.com/v1 \
  azure.enabled=true \
  eval.judge_model=gpt-4-1-dev \
  eval.output=results/bcp_gpt5
```

### Qwen 3.5 (self-hosted vLLM)
```bash
arcticswarm-eval -c conf/bench/browsecomp_plus.yaml \
  llm.model=qwen3.5-27b \
  llm.agent_model_base_url=http://<your-vllm-host>:7777/v1 \
  azure.enabled=true \
  eval.judge_model=gpt-4-1-dev \
  eval.output=results/bcp_qwen35
```

BrowseComp-Plus is scored by its dedicated judge. The `provider: corpus` / `fetch_backend: corpus` settings in the preset route `web_search`/`web_fetch` to the corpus retriever selected by `web.corpus_backend` (above).

---

## 5. Results

Each run writes to `results/<name>/` — per-case trajectories plus a `report.json` with accuracy. To browse runs interactively:

```bash
python -m viewer.server        # then open the printed localhost URL
```

---

## 6. Caches (skip re-fetching / re-searching)

`web_fetch` / `pdf_read` and `web_search` are slow, and the same URLs/queries
recur across runs. Two optional local SQLite caches make any URL fetched (or
query run) **once** — in any past or current run — return from disk instead of
the network. They are plain SQLite files at a **configurable path**; the example
root below is `/data/cache/`:

```
/data/cache/
  ├─ fetch_cache.sqlite    # web_fetch + pdf_read content (global, cross-run)
  └─ search_cache.sqlite   # web_search raw results (per-provider)
```

- **Defaults:** `fetch_cache_path=/data/cache/fetch_cache.sqlite`,
  `search_cache_db=/data/cache/search_cache.sqlite` (override via
  settings / env `ARCTICSWARM_FETCH_CACHE` / `web.fetch_cache_path` /
  `web.search_cache_db`; `web.fetch_cache_path=off` disables the fetch cache).
  Each opens lazily and silently disables itself if its path isn't writable, so
  the defaults are always safe even when the path doesn't exist.
- **Fetch cache layering:** read global-first, then the per-question cache.
  Successful fetches write through to the global store incrementally; failures
  stay per-question (never poison the shared cache); longer extraction wins on
  conflict. Covers both `web_fetch` and `pdf_read` (same normalized-URL+pages
  hash). Cache hits are byte-identical to live fetches — the model can't tell.

**Seed the fetch cache from historical runs** (one-time; idempotent):

```bash
python scripts/build_fetch_cache.py \
  --source '/path/to/results/*/cache/content' \
  --db /data/cache/fetch_cache.sqlite --workers 14
```

If the cache path is on ephemeral storage, point it at a persistent disk via
`fetch_cache_path` / `ARCTICSWARM_FETCH_CACHE` (or symlink the root).

> **Snowflake-cluster specifics** (S3 auto-restore, the node-local mirror, our
> exact paths, the seeding/ship scripts) live in
> [snowflake/snowflake_specific.md](snowflake/snowflake_specific.md).

---

## Models at a glance

| Backend | `llm.model` | Provider | Extra settings |
|---|---|---|---|
| Claude Sonnet 4.5 | `claude-sonnet-4-5` | Anthropic (public) | `api_key` |
| OpenAI GPT-5 | `gpt-5` (or `gpt-5.4`) | OpenAI (public) | `openai_api_key` + `llm.openai_base_url` |
| Qwen 3.5 | `qwen3.5-27b` | self-hosted vLLM | `llm.agent_model_base_url` |
