# Environment setup / replication

How to build the ArcticSwarm eval environment so anyone can reproduce
BrowseComp / BrowseComp-Plus runs.

> Running on our Snowflake cluster? The shared prebuilt env, S3 cache restore,
> and exact paths/scripts live in
> [snowflake/snowflake_specific.md](snowflake/snowflake_specific.md). The
> fresh-install path below is everything else.

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

## Credentials

Copy the template to a repo-root `config_files.json` and fill in your keys (or
point `ARCTICSWARM_SETTINGS_PATH` at another file). See README §2 for the full
table. `config_files.json` is git-ignored — only the template is tracked.

```bash
cp config_files.template.json config_files.json   # then edit config_files.json
```

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
  "AZURE_OPENAI_API_KEY": "", "AZURE_OPENAI_ENDPOINT": "",
  "fetch_cache_path": "/data/cache/fetch_cache.sqlite"
}
```

- Cortex corpus / Cortex web-search providers also read a PAT from
  `~/.snowflake/connections.toml` (or use the live Snowflake session token).
- `jina_api_key` (or env `JINA_API_KEY`) enables the Jina Reader API as the
  primary `web_fetch` / `pdf_read` extractor; optional — fetch degrades to
  Serper → `requests` when it is unset.
- The Azure GPT-4.1-dev judge needs `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT`
  (`azure.enabled=true eval.judge_model=gpt-4-1-dev`).

## Caches (optional but recommended)

`web_fetch`/`pdf_read` and `web_search` share optional cross-run SQLite caches
so a URL fetched (or query run) once is never repeated — transparently (the
model can't tell a hit from a live call). They are plain local SQLite files at a
configurable path; the example root below is **`/data/cache/`**:
`fetch_cache.sqlite` + `search_cache.sqlite` (override via `fetch_cache_path` /
`search_cache_db` settings, `ARCTICSWARM_FETCH_CACHE` env, or `web.*`). Each
opens lazily and disables itself if the path isn't writable, so the defaults are
safe even when absent.

- **Seed the fetch cache:** `python scripts/build_fetch_cache.py --source '<results>/*/cache/content' --db /data/cache/fetch_cache.sqlite`
- If the path is ephemeral, point `fetch_cache_path` / `ARCTICSWARM_FETCH_CACHE`
  at a persistent disk (or symlink the root).

Snowflake-cluster specifics (S3 auto-restore, the node-local mirror, our exact
paths, the seeding/ship scripts) live in
[snowflake/snowflake_specific.md](snowflake/snowflake_specific.md). See
README §6 for more on how the caches work.
