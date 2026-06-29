# Snowflake-cluster specifics

Everything in this file is specific to our Snowflake pod/cluster. The rest of
the repo runs without any of it (given correct config/paths). Keep
company/cluster-specific knowledge HERE; the general code reads it via env vars
and config so `./snowflake/` can be deleted for an open-source checkout.

---

## 1. Cluster environment (prebuilt shared env)

On the cluster the eval environment is already built. Activate the shared env,
which puts `arcticswarm` / `arcticswarm-eval` on `PATH` with the validated,
pinned dependency set (same versions as `requirements.txt`):

```bash
source /code/users/soyoung/activate_snowswarm.sh
```

This is the in-cluster equivalent of the fresh-install path in
`ENVIRONMENT.md` (Option B). Use it instead of building a venv on the pod.

---

## 2. Caches on our pod/cluster

Two cross-run SQLite caches make web fetches and searches near-free across runs:

- `fetch_cache.sqlite` — `web_fetch` + `pdf_read` content (global, cross-run)
- `search_cache.sqlite` — `web_search` raw results (per provider)

### Real cache root: `/data/soyoon/cache/`

On our pod the master caches live under one root so a single sync moves the
whole set:

```
/data/soyoon/cache/
  ├─ fetch_cache.sqlite
  └─ search_cache.sqlite
```

### Node-local mirror: `/data-fast/soyoon/cache` (the `cache_local_dir` optimization)

`/data` is a shared (Lustre) network filesystem. A WAL-mode SQLite mmaps a
host-local `-shm` index; if several GPU hosts open one WAL DB on `/data`, a
write on another host invalidates the mapping and access raises **SIGBUS** (or
deadlocks on cross-host `flock`s). To avoid that, each node mirrors the master
caches to its **node-local fast disk** `/data-fast/soyoon/cache` (xfs) at
startup, reads/writes there, and periodically syncs only the new rows back to
the `/data` master under an exclusive `flock` in rollback-journal (DELETE) mode
(never WAL on the shared master). See `arcticswarm/tools/cache_sync.py`:

- `storage_present(path)` — returns True iff a node-local fast-disk mount
  (an ancestor of `cache_local_dir`, e.g. `/data-fast`) exists. False on a dev
  box / CPU pod that has no such mount.
- `should_auto_mirror(config)` — engages the mirror iff `web.cache_local_mirror`
  is on (default True), caching is active, AND `storage_present()` is True. So
  the mirror is automatic on GPU nodes and a silent no-op everywhere else.
- `CacheMirrorManager.setup()` — copies each master cache to
  `cache_local_dir/<name>.sqlite` and repoints `config.fetch_cache_path` /
  `config.search_cache_db` at the local copies.
- `CacheMirrorManager.start()` — launches a background daemon that triggers a
  delta-sync back to `/data` every `cache_sync_every` finished cases (counted
  from the run's `trajectories/` dir), plus a final sync at process exit.
  This is the "save upstream every ~5 questions finished" design.

The eval CLI wires this automatically (`arcticswarm/eval/cli.py`); you only need
to set `web.cache_local_dir` so the general code points at our fast disk.

### S3 restore-on-demand (the cluster sweeps `/data` and purges)

The cluster periodically sweeps `/data` to S3 and deletes it locally, so a
cache the run depends on can be gone at process start. When a cache is absent
and an S3 mirror is configured, `arcticswarm/tools/cache_restore.py` restores
the whole cache root with one `aws s3 sync` before opening (no-op when present
/ off-cluster / `aws` missing / bucket unset; failures degrade to an empty
cache). The restore command for our pod:

```bash
aws s3 sync s3://ml-dev-sfc-or-dev-misc1-k8s/snowflake_research/data/soyoon/cache /data/soyoon/cache
```

Enable it for the general code by setting the bucket/prefix env vars (below).
`cache_restore.py` keys the restore by the cache's PARENT dir and runs it at
most once per dir per process, so one sync restores both caches.

### Operational scripts (moved here from `scripts/`)

- `snowflake/scripts/push_fetch_cache.sh` — one-time push of a prebuilt
  `fetch_cache.sqlite` to a pod (streams gzip → pod → gunzip; auto-resolves the
  suffixed pod name; verifies the row count on the pod). Hardcodes our pod
  name / namespace / `/data/soyoon/cache` paths.
- `snowflake/scripts/sync_cache.py` — standalone full merge of the node-local
  mirror(s) back into the `/data` masters (use after a multi-host run or if a
  process exited before its final sync). Imports the merge logic from
  `arcticswarm.tools.cache_sync`; run it from the repo root.

The general seeding utility `scripts/build_fetch_cache.py` stays in `scripts/`
(it has no company defaults — pass `--source`/`--db`).

### Reproduce our caching on the cluster (exact env exports)

```bash
export ARCTICSWARM_FETCH_CACHE=/data/soyoon/cache/fetch_cache.sqlite
export ARCTICSWARM_CACHE_S3_BUCKET=ml-dev-sfc-or-dev-misc1-k8s
export ARCTICSWARM_CACHE_S3_PREFIX=snowflake_research
```

and on the CLI (or in your YAML `web:` block):

```
web.search_cache_db=/data/soyoon/cache/search_cache.sqlite
web.cache_local_dir=/data-fast/soyoon/cache
```

---

## 3. Env vars to point the GENERAL cache code at our paths/bucket

| Variable / override | Our value |
|---|---|
| `ARCTICSWARM_FETCH_CACHE` (env) | `/data/soyoon/cache/fetch_cache.sqlite` |
| `web.search_cache_db` (config) | `/data/soyoon/cache/search_cache.sqlite` |
| `web.cache_local_dir` (config) | `/data-fast/soyoon/cache` |
| `ARCTICSWARM_CACHE_S3_BUCKET` (env) | `ml-dev-sfc-or-dev-misc1-k8s` |
| `ARCTICSWARM_CACHE_S3_PREFIX` (env) | `snowflake_research` |

`ARCTICSWARM_CACHE_S3_BUCKET` empty (the OSS default) disables the S3 restore;
empty `web.cache_local_dir` (the OSS default) leaves the node-local mirror off
unless a fast-disk mount exists.

---

## 4. Exact vLLM launch command (Qwen3.5-27B, verbatim)

The exact command we used on the cluster. Note the cluster-specific paths
`/checkpoint/huggingface` and `/data-fast/vllm-venv`:

```bash
set -euo pipefail
export HF_HOME=/checkpoint/huggingface
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /data-fast/vllm-venv/bin/vllm serve Qwen/Qwen3.5-27B \
  --served-model-name Qwen/Qwen3.5-27B --host 0.0.0.0 --port 7777 \
  --data-parallel-size 1 --tensor-parallel-size 8 --kv_cache_dtype fp8 \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --max-num-seqs 512 --max-num-batched-tokens 16384 \
  --attention-backend FLASHINFER --enable-prefix-caching --async-scheduling \
  --language-model-only --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --trust-remote-code --seed 0 --disable-custom-all-reduce
```

Point the agent at the served port with `llm.agent_model_base_url=http://<host>:7777/v1`.
A general (placeholder) version of this command is in the README's Qwen section.
