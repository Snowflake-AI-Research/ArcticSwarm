---
name: india-cluster-eval-ops
description: >
  Operational runbook for running ArcticSwarm BrowseComp-Plus evals on the
  Snowflake india k8s cluster: launching self-hosted Qwen3.5-27B on vLLM,
  wiring cross-pod `dss` networking, the Cortex corpus PAT, and how to report
  a long multi-arm ablation honestly while it is still in flight.
  Written 2026-09-16/17 from the 0916 review-gate ablation (6 concurrent runs).
---

# India-cluster eval ops (BrowseComp-Plus + self-hosted Qwen)

Hard-won specifics for `sfc-in-dev-meta-k8s-1` / namespace `mltraining-dev`.
Companion doc: `/home/soyoon/prime-rl/TODO/soyoon/india_servers.md` (dss recipe,
DeepSeek/GLM serve configs). This file covers what that doc does **not**: the
ArcticSwarm eval side, the Qwen3.5 serve stack, and reporting discipline.

---

## 1. Filesystem layout — the paths in your Oregon commands are wrong here

Commands copied from Oregon reference `/code/users/soyoung/...`. On india the
real files live in **`/code/soyoung/`**; `/code/users/soyoung/` only holds
symlinks. `activate_snowswarm.sh` *also* hardcodes `/code/users/soyoung`, so the
fix is to create the symlinks once rather than rewrite commands:

```bash
cd /code/users/soyoung
for f in ArcticSwarm activate_snowswarm.sh snowswarm_settings_cortex.json \
         .local .snowflake connections.toml; do
  ln -sfn /code/soyoung/$f $f
done
```

Mount semantics — this distinction causes most surprises:

| Path | Kind | Survives pod restart? |
|---|---|---|
| `/code`, `/data`, `/checkpoint` | Lustre (shared, all pods) | **yes** |
| `/data-fast` | node-local NVMe | **NO — wiped, and it's per-node** |

So: model weights → `/checkpoint/huggingface` (download once, every pod sees it).
Eval outputs → `/data/...` (safe). vLLM venv + serve scripts → `/data-fast`
(must be rebuilt on every pod restart).

`activate_snowswarm.sh` hard-errors without `/data-fast` ("GPU pods only"), so
**the eval client must run on a GPU pod**, not `soyoung-dev-cpu-in`. Simplest
layout: run each eval on the same GPU pod that serves its model.

---

## 2. `dss` networking — two independent things, both required

A client can reach a server only if **both** hold:
1. the **caller** pod carries `dss=true`;
2. the **callee** job has an `allow-dss` ingress policy **on that exact port**.

Two traps, both of which fail as a *silent timeout*, not an error:

- **Existing `allow-dss` policies may only open port 8000.** `soyoung-traj-gen-1`
  and `soyoung-glm` each had one, but scoped to 8000 only — serving on **7777**
  timed out cross-pod until patched. Check before trusting it:
  ```bash
  kubectl get networkpolicy -n mltraining-dev <job>-allow-dss -o jsonpath='{.spec.ingress}'
  ```
  Patch additively (never edit the kueue-owned `<job>` policy):
  ```bash
  kubectl patch networkpolicy -n mltraining-dev <job>-allow-dss --type=merge -p \
    '{"spec":{"ingress":[{"from":[{"podSelector":{"matchLabels":{"dss":"true"}}}],
      "ports":[{"port":8000,"protocol":"TCP"},{"port":7777,"protocol":"TCP"}]}]}}'
  ```
- **A brand-new job name has no `allow-dss` policy at all.** Create one (see
  `india_servers.md` §5 for the manifest). Policies select by `job-name`, so they
  **persist across pod restarts** — create once per job, never again.

Labels are **per-pod** and lost on restart; re-apply after every recreation:
```bash
kubectl label pod -n mltraining-dev $P dss=true --overwrite
```

The `soyoung-*` jobs get an auto-created **headless Service** named after the job.
It lists no ports, but headless DNS resolves to the pod IP, so **any** port works
— no Service needs creating. Endpoint form: `http://<job>:7777/v1`.

Diagnostic: from a `dss=true` client, **connection refused in milliseconds** =
policy fine, server not up yet. **Timeout** = missing label or missing policy.

---

## 3. Serving Qwen3.5-27B on vLLM

Weights were **not** in `/checkpoint/huggingface` (it had Qwen3.**8**-27B and
Qwen3.5-{397B,9B,4B}). Download once into shared storage:

```bash
HF_HOME=/checkpoint/huggingface \
  /home/yak/miniconda3/envs/dev/bin/hf download Qwen/Qwen3.5-27B
```

`Qwen3.5-27B` is `Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5`, a
**hybrid** model (`layer_types`) with vision preprocessor configs. Consequences:
- needs **`causal_conv1d`** in the vLLM venv (for `--mamba-cache-mode align`);
  a missing `causal_conv1d` is a per-node gotcha — takes ~2.5 min to build;
- pass **`--language-model-only`** (text-only eval).

### Venv stack — pin it, and keep it identical across arms

```bash
UV=/home/yak/miniconda3/envs/dev/bin/uv
export UV_LINK_MODE=copy
$UV venv /data-fast/vllm-venv --python 3.12
$UV pip install --python /data-fast/vllm-venv/bin/python "vllm==0.28.0" hf_transfer
# FlashInfer python + jit-cache MUST be the same version, else vLLM JIT-compiles
# and cc1plus segfaults; a mismatched-but-newer pair breaks trtllm-gen kernels
# on standard Qwen models ("Mismatched number of arguments when calling trtllm_*").
$UV pip install --python /data-fast/vllm-venv/bin/python \
  "flashinfer-python==0.6.16.post3" "flashinfer-jit-cache==0.6.16.post3+cu130" \
  --extra-index-url https://flashinfer.ai/whl/cu130/
$UV pip install --python /data-fast/vllm-venv/bin/python causal_conv1d --no-build-isolation
```

vllm 0.29.0 + FlashInfer 0.6.18 (a *matched* newer pair) also serves this model
fine. But **for an ablation, standardize one stack across every arm** — a vLLM
version difference between arms confounds the comparison you are trying to make.
If a user hands you a different pinned wheel, say so and prefer comparability.

### Serve script

Keep it in a file (`/data-fast/serve_qwen35_27b.sh`) and run under tmux, so a
dropped `kubectl exec` never kills the server:

```bash
export HF_HOME=/checkpoint/huggingface
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_DISABLE_VERSION_CHECK=1   # benign cubin/python skew
export TORCH_NCCL_ENABLE_MONITORING=0
source /data-fast/vllm-venv/bin/activate
exec vllm serve Qwen/Qwen3.5-27B --served-model-name Qwen/Qwen3.5-27B \
  --host 0.0.0.0 --port 7777 --tensor-parallel-size 8 --kv_cache_dtype fp8 \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --max-num-seqs 512 --max-num-batched-tokens 16384 \
  --mamba-cache-mode align --mamba-block-size 8 \
  --attention-backend FLASHINFER --enable-prefix-caching --async-scheduling \
  --language-model-only --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --trust-remote-code --seed 0 --disable-custom-all-reduce
```

```bash
tmux new-session -d -s q35 "bash /data-fast/serve_qwen35_27b.sh > /data-fast/qwen35.log 2>&1"
# ready when: curl -s localhost:7777/v1/models | grep -q Qwen3.5-27B   (~5 min: weights + FlashInfer autotune + 83 CUDA graphs)
```

### The served-model-id trap

Repointing a run at self-hosted vLLM needs **both**, and a slash in `llm.model`
silently bypasses the override → 404:

```
llm.model=qwen3.5-27b                      # bare, NO slash
llm.vllm_served_model_id=Qwen/Qwen3.5-27B  # must equal --served-model-name exactly
```

---

## 4. BrowseComp-Plus / Cortex corpus

`conf/bench/browsecomp_plus_qwen.yaml` uses `web.corpus_backend: cortex` —
**Snowflake Cortex Search (remote)**, account `AKB73862`. There is no local
retriever, so a local embedding server is *not* a dependency (it can be killed
to free GPUs). Auth is a PAT from a `connections.toml` profile named by
`web.corpus_pat_connection` (`ml_data`).

**On india, `~/.snowflake/connections.toml` does NOT contain `[ml_data]`** (only
`[preprod8]`/`[preprod8_2]`). The `[ml_data]` profile is at
`/code/soyoung/connections.toml`. Don't edit the creds file — point at it:

```bash
export SNOWFLAKE_CONNECTIONS_FILE=/code/users/soyoung/connections.toml
```
`arcticswarm/tools/corpus_retriever.py` honors `$SNOWFLAKE_CONNECTIONS_FILE`
first, then `$SNOWFLAKE_HOME/connections.toml`, then `~/.snowflake/`.

Judge: `azure.enabled=true eval.judge_model=gpt-4-1-dev`, keys from
`ARCTICSWARM_SETTINGS_PATH` (a `config_files.json`-shaped file). Smoke-test it
**before** a long run rather than discovering it hours in — one call, status only:

```bash
python3 -c "import json,urllib.request; d=json.load(open('<settings>.json'));
u=d['AZURE_OPENAI_ENDPOINT'].rstrip('/')+'/openai/deployments/gpt-4-1-dev/chat/completions?api-version='+d['OPENAI_API_VERSION']
r=urllib.request.urlopen(urllib.request.Request(u,data=json.dumps({'messages':[{'role':'user','content':'say OK'}],'max_tokens':5}).encode(),
headers={'Content-Type':'application/json','api-key':d['AZURE_OPENAI_API_KEY']}),timeout=45); print(r.status)"
```

### Known judge hole: Azure content filter → unjudged case

A BrowseComp answer can trip Azure's content policy (seen: `violence` severity
`medium` → HTTP 400 `ResponsibleAIPolicyViolation`). The harness falls back to an
**Anthropic** judge, but if `api_key` is empty in the settings file that fallback
dies (`Could not resolve authentication method`) and the case ends up
`judge_correct: None`, `judge_mode: None` — silently ungraded.

- **`report.json`'s `num_content_filter_questions` stays 0**, so you cannot detect
  this from the report. Grep the log: `grep -c "Azure content filter triggered"`.
- Rate observed: ~1 in ~1300 judge calls.
- **Do not add an Anthropic key mid-experiment.** Those cases would then be graded
  by a *different* judge than every other case — judge drift has moved BCP numbers
  by ~5 points before. Tally the affected cases and re-judge with the same judge
  afterwards, or exclude them.

---

## 5. Always verify `resolved_config.yaml`, never trust the CLI string

A mistyped flag or skill-override name does **not** error — it silently no-ops,
turning an aggressive ablation into a mild one. After every launch:

```bash
grep -nE "agent_model_base_url|vllm_served_model_id|^  parallel:|corpus_backend|judge_model:|\
disable_final_verification|disable_auditor|disable_builder_idle|disable_self_reflection|\
browsing_max_reflection_loops|min_dedicated_reviewers|min_builder_reviewers|max_reviewer_remediations|\
reject_refusal_reports|surface_bbs_candidates|enable_empty_answer_recovery" <outdir>/resolved_config.yaml
grep -A5 skill_overrides <outdir>/resolved_config.yaml
```

Also confirm the override *targets* exist before launching:
`ls -d arcticswarm/skills/<name>` for each `swarm.skill_overrides.X=Y`.

Behavioural confirmation is even better than config: e.g. seeing
`answer_verification: Disagreement gate fired — rival-sweep task posted` in the
log proves the answer-retention cluster is actually live.

---

## 6. Operational gotchas that cost real time

- **`pkill -f "vllm serve"` via `kubectl exec` kills your own exec** (the pattern
  matches your `bash -lc` command line) → exit 143. Use a bracket:
  `pkill -f "vllm[ ]serve"`. Same trick for `pgrep` counts.
- **Never `tmux kill-server`** on these pods — other people's sessions live there.
  Kill named sessions only.
- **Pods get recreated AND renamed.** Kueue suspends a job whose pod exceeds the
  `PodsReady` timeout (slow image pull) and creates a fresh pod:
  `Job suspended` → `Exceeded the PodsReady timeout` → new pod name. Anything
  holding the old pod name gets `NotFound`, and `/data-fast` (venv + scripts) is
  gone. **Resolve pods by job every time:**
  ```bash
  P=$(kubectl get pods -n mltraining-dev -o name | grep "<job>-0" | sed 's|pod/||')
  ```
  Recovery = re-label `dss=true` → re-copy scripts → rebuild venv → serve → re-run
  the eval script (the harness resumes from its checkpoint). The `allow-dss`
  policy does **not** need recreating.
- Nodes named `*-temp-oneday*` may be short-lived. A full 830-question arm takes
  16–30 h, so confirm node lifetime before starting, or plan for a checkpoint
  resume on a replacement node.
- Long agentic load produces steady client-side `Connection error` warnings while
  the server logs pure `200 OK`. That's endpoint saturation; the harness retries
  and `num_errors` stays 0. Check the *server* log before treating it as a fault.

---

## 7. Reporting a long run honestly

`report.json` → `['overall']` gives `num_cases`, `num_errors`,
`qa_llm_accuracy`, `avg_duration_seconds`, `p90_duration_seconds`;
`['per_case']` has `judge_correct` / `judge_mode` per case.

**Report cases as `N/830` plus a remaining-time ETA**, not just a percentage —
`cases/elapsed` gives the rate; expect it to drift down because the hard tail
comes last.

Five things to get right, learned by nearly getting them wrong:

1. **A mid-run accuracy is optimistically biased.** Short/easy cases finish first,
   so the running mean starts high and decays (observed: 0.87 → 0.79 → 0.75 over
   one night, on *both* replicates simultaneously). Say so every time you quote it.
   Only the final number, or a case-ID-matched subset, is a result.
2. **Never compare two arms at different completion levels.** An arm at 16% and an
   arm at 65% have different *sets* of completed cases; the less-finished one looks
   unfairly bad. Compare on the intersection of completed case IDs.
3. **Replicate agreement is the signal worth reporting.** Two independent runs of
   the same config on different nodes converging (e.g. 0.7992 vs 0.7989) is strong
   evidence the estimate is stable; a 4-point early gap that narrows to 1 point as
   n grows was sampling noise, and saying so up front prevents over-reading it.
4. **Filter error greps precisely.** `grep -c "401"` matches retry-delay floats
   like `0.452401 seconds` — it will invent errors that don't exist. Use
   `grep -cE 'Traceback|HTTP/1.1" (401|429)|AuthenticationError|RateLimitError'`.
   Verify any nonzero count by reading the actual lines before escalating.
5. **Compare per-case duration, not throughput, when `eval.parallel` differs.**
   Arms run at parallel 15/18/20 are not throughput-comparable. Note also that two
   replicates of the *same* config can diverge a lot in speed (observed 0.39 vs
   0.53 cases/min, avg duration 1977 s vs 1546 s) without any accuracy difference —
   report it as a scheduling/case-mix artifact, and flag that their finish times
   will differ by many hours.

Per-check template that has worked: a table of run × (tmux alive, vLLM HTTP code,
cases/830, remaining ETA, failed cases, accuracy, content-filter count), then only
the *deltas* and anything genuinely new. Keep a cumulative
"N check-ins, M interventions" line — an unbroken zero is itself information, and
it makes a real intervention impossible to miss.

**Escalate (push a notification) only for unrecoverable breakage.** Restart vLLM
or resume an eval silently; a saturation warning or one ungraded case is a line in
the next report, not an interrupt.
