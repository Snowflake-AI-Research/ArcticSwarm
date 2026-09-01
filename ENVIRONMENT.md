# Environment setup / replication

How to build the ArcticSwarm eval environment so anyone can reproduce
BrowseComp / BrowseComp-Plus runs.

## Prerequisites

- **Python 3.11+** (validated on 3.12).
- **Java 11+** — required only for PDF reading (`web_fetch` on PDFs and
  `pdf_read`), used by the `opendataloader-pdf` hybrid backend.
  - Linux: `apt install default-jdk` · macOS: `brew install openjdk`
- **`opendataloader-pdf[hybrid]`** — the hybrid PDF extractor (pinned in
  `requirements.txt`; looser range in `pyproject.toml`). The `[hybrid]` extra
  installs the `opendataloader-pdf-hybrid` server binary on `PATH`; ArcticSwarm
  auto-starts it on a free port (no manual server needed) and falls back to the
  `pypdf` fast-path → Jina Reader when it is unavailable.

## Fresh install (local or a new pod)

```bash
python -m venv .venv && source .venv/bin/activate      # or conda create -n arcticswarm python=3.12
pip install -r requirements.txt                        # pinned, validated versions
pip install -e . --no-deps                             # install the arcticswarm + arcticswarm-eval CLIs
```

`pip install -e .` (without `requirements.txt`) also works and uses the looser
ranges in `pyproject.toml`; `requirements.txt` exists for reproducible pins.

> **Verify the editable install points at the repo you're editing** —
> `pip show arcticswarm` should report `Editable project location:
> <this repo>`. If it points elsewhere, `arcticswarm-eval` will run stale code;
> fix with `pip install -e <repo> --no-deps`.

Sanity check:

```bash
arcticswarm-eval --help
arcticswarm-eval -c conf/bench/browsecomp.yaml eval.limit=1 eval.output=/tmp/smoke   # 1-case smoke run
```

## Credentials & the config file

ArcticSwarm reads all secrets/endpoints from one JSON **settings file**.

```bash
cp config_files.template.json config_files.json   # then edit config_files.json
```

**Where the settings file is read from** (in order):

1. `$ARCTICSWARM_SETTINGS_PATH`, if set — an absolute or `~`-expanded path to any
   JSON file. Use this to keep secrets outside the repo or to switch profiles:
   `export ARCTICSWARM_SETTINGS_PATH=/etc/arcticswarm/prod.json`.
2. Otherwise `./config_files.json` (relative to the current working directory —
   normally the repo root).

`config_files.json` is git-ignored, so real secrets are never committed — only
`config_files.template.json` is tracked, and its `__help__` block documents
every key. See README §2 for the key table.

> **Settings file vs. run config — two different things.** The settings file
> above holds *credentials/endpoints*. The benchmark/model/tool selection is a
> separate **run config** passed explicitly with `--config conf/bench/*.yaml`
> (composable left-to-right, with `dotted.key=value` overrides). The settings
> file path is never passed with `--config`.

Common keys:

```json
{
  "api_key": "sk-ant-...",
  "openai_api_key": "sk-...",
  "openai_base_url": "https://api.openai.com/v1",
  "brave_api_key": "", "serper_api_key": "", "tavily_api_key": "",
  "jina_api_key": "",
  "cortex_account": "",
  "use_azure_openai": true,
  "AZURE_OPENAI_API_KEY": "", "AZURE_OPENAI_ENDPOINT": ""
}
```

- Cortex corpus / Cortex web-search providers also read a PAT from
  `~/.snowflake/connections.toml` (or use the live Snowflake session token).
- `jina_api_key` (or env `JINA_API_KEY`) enables the Jina Reader API as the
  primary `web_fetch` / `pdf_read` extractor; optional — fetch degrades to
  Serper → `requests` when it is unset.
- The default eval judge is the public `openai-gpt-4.1` (needs `openai_api_key`).
  For an Azure GPT-4.1 deployment instead, set `azure.enabled=true
  eval.judge_model=<your-deployment>` with `AZURE_OPENAI_API_KEY` /
  `AZURE_OPENAI_ENDPOINT`. (`azure.enabled=true` is the run-config override for
  the settings-file key `use_azure_openai: true` shown above — set either one.)
  A Claude judge works too: `eval.judge_model=claude-sonnet-4-5`.

## Caching

**Within-run content cache (on by default).** `web_fetch` / `pdf_read` use a
per-question, disk-backed content cache so a URL fetched once during a question
is not re-fetched for the rest of that question (by the same agent or by sibling
swarm agents) — transparently (the model can't tell a hit from a live call).

- **Location:** `{eval.output}/cache/content/<case_id>/`. Enabled by default;
  set `enable_content_cache=false` to disable.
- **Scope:** per question only — entries are never shared across questions or
  runs.

**Cross-run fetch cache (optional, off by default).** For cheaper re-runs
against a static benchmark you can enable a disk cache that persists `web_fetch`
/ `pdf_read` content across runs. It is **off by default** — every search and
fetch is live unless you turn it on — and cache hits are byte-identical to live
results (the fetched corpus is cached, not any answer or judgment).

- **Fetch cache:** set `web.fetch_cache_path=<path.sqlite>` (or env
  `ARCTICSWARM_FETCH_CACHE`) to enable; empty disables. Seed one from prior runs
  with `python scripts/build_fetch_cache.py`.
- **Optional S3 restore:** set `ARCTICSWARM_CACHE_S3_BUCKET` /
  `ARCTICSWARM_CACHE_S3_PREFIX` to restore a cache root on first use (empty
  disables). Point these only at buckets you control.
- **Multi-host runs:** `web.cache_local_dir` mirrors the shared cache to
  node-local fast disk and syncs deltas back (avoids SIGBUS from a shared WAL
  SQLite on a network filesystem); off unless set.

See the README FAQ ("Is anything cached across runs?") for the short version.
