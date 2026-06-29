# 04 — Errors, Mitigations & Ongoing TODOs

Running log of problems hit during the overnight session and how they were handled.
(User explicitly asked for this.)

## ERRORS / INCIDENTS

### E1 — `tmux kill-server` killed the dedicated node's vLLM server (SELF-INFLICTED) — RESOLVED
- **When:** 2026-06-28 ~09:52Z, right after launching Round-1 arms.
- **What:** To launch the 3 eval arms I ran `tmux kill-server` on pod
  `soyoung-vllm-sunonly-temp-0-xt5zg`. The node's **vLLM server was itself running in a
  tmux session**, so kill-server terminated it. GPU memory dropped 167GB→0 on all 8 GPUs;
  nothing listening on :7777; the 3 eval arms threw `Connection error` retries.
- **Mitigation:** (1) Recovered the exact serve command from `/home/yak/.bash_history`
  (8×TP, FLASHINFER, mamba-hybrid flags, port 7777) and restarted it in a **named** tmux
  session `vllm` via `/tmp/serve_vllm.sh`. (2) Killed the 3 failing eval arms with
  **targeted** `tmux kill-session -t <name>` (NEVER kill-server again). (3) Waiting on the
  endpoint monitor for readiness (~10 min model load), then relaunching arms on clean
  output dirs.
- **Lesson / rule:** On shared GPU pods, NEVER `tmux kill-server`. Always
  `tmux kill-session -t <name>`. The vLLM serve process lives in tmux here.
- **Serve command (for re-use):** see `/tmp/serve_vllm.sh` on the pod; key flags:
  `vllm serve Qwen/Qwen3.5-27B --port 7777 --tensor-parallel-size 8 --kv_cache_dtype fp8
  --max-model-len 262144 --gpu-memory-utilization 0.90 --max-num-seqs 512
  --max-num-batched-tokens 16384 --mamba-cache-mode align --mamba-block-size 8
  --attention-backend FLASHINFER --enable-prefix-caching --async-scheduling
  --language-model-only --reasoning-parser qwen3 --enable-auto-tool-choice
  --tool-call-parser qwen3_xml --trust-remote-code --seed 0 --disable-custom-all-reduce`

### E2 — tmux `kill-server` then immediate `new-session` race — RESOLVED
- First launch of arm R1a was lost to a "server exited unexpectedly" race after
  kill-server. Relaunched individually. (Now moot — no more kill-server.)

### E3 — cluster `git pull` broken (custom SSH host alias) — RESOLVED via direct copy
- **What:** the cluster checkout's `origin` uses an SSH host alias
  `github-snowflake-ai` that doesn't resolve from the pod, so `git pull` fails
  (`Could not resolve hostname github-snowflake-ai`). I had `rm`'d the untracked
  diag28 CSV pre-pull, so it briefly vanished (Round 1 unaffected — it loads the
  CSV into memory at launch).
- **Mitigation:** pushed the commit to `main` from the workspace (gh HTTPS
  workaround — works locally), then **directly copied** the 5 changed code files +
  2 tests + the CSV to `/code/users/soyoung/ArcticSwarm` via `tar | kubectl exec`.
  Verified flags parse on the cluster (`surface_bbs_candidates=True
  compaction_prune_junk=True`). Canonical version is on `main` (commit 9f65209);
  the cluster has the same files as uncommitted working-tree edits.
- **Note for user:** to get a clean cluster tree, fix the cluster remote to HTTPS
  (or restore the SSH alias) and `git checkout -- .` then `git pull` — my files
  match `main@9f65209`.

### E4 — eval throughput: least-connections pool UNDER-uses a free node + ~12 cases/hr/node
- **What:** the 120-validation pooled across 7 endpoints crawled (2-3 cases / 32 min). The
  dedicated B200 sat at `Running: 1 req` while the shared endpoints (busy with 0625) absorbed
  my load — the `agent_model_base_url` least-connections balancer did NOT concentrate my load
  on the free node. Also: each node only does ~12 thrash-capped xhigh cases/hr, so 120 on one
  node ≈ many hours.
- **Mitigation:** pointed v2 (headline) at the **dedicated node ALONE at parallel 20** →
  `Running: 24, gen ~1500 tok/s` (full utilization). Killed v1 to focus resources + relieve
  the shared pool (helps 0625 finish → frees nodes). Rule: to load N free nodes, set
  `eval.parallel ≈ N×20-24`; per-node parallel ~20-24 is the sweet spot (KV stays <5%, gen-
  throughput-bound). The 120/300 need MULTIPLE free nodes — gated on the 0625 shards finishing.

## RUN HYGIENE RULES (for the rest of the session)
- vLLM serve lives in tmux session `vllm` on each GPU pod. Do not touch it except to
  restart if down.
- Before relaunching an arm on an existing output dir, `rm -rf` the dir (cases that errored
  during E1 would otherwise resume from a dirty checkpoint).
- Endpoint allocator monitor (`bc3gfzuax`) emits on UP/DOWN transitions of all 7 endpoints.

## ONGOING TODOs
- [ ] Wait for `soyoung-vllm-sunonly-temp` ready → relaunch Round-1 arms (R1a/b/c) clean.
- [ ] While Round 1 runs: implement qwen-gated code changes (see below) + commit + pull.
- [ ] Round 1 analysis → pick best config arm.
- [ ] Round 2 = best config + code changes; launch.
- [ ] Validate winner on representative-120 (+ same-node baseline) at full 9000s/rerun.
- [ ] Keep `03_results.md` updated with each run's accuracy + efficiency.

## CODE CHANGES PLANNED (qwen-gated, default-off; see 02_hypotheses.md for rationale)
1. `reactive_use_structured_compaction` — route the prompt-too-long reactive path through
   the candidate-preserving structured prompt instead of the generic lossy one. (H4)
2. `bbs_report_findings_digest` — append a deterministic #key-findings/#consensus candidate
   digest into the prepare_report output so the final-answer step always sees candidates. (H1/H6)
3. `reviewer_builder_verified_unlocks` — when ≥1 builder VERIFIED verdict exists, don't let
   a missing dedicated verdict hard-block; surface/pin the builder-verified candidate. (H5)
4. broaden `empty_answer_recovery` give-up detection to the FINAL-ANSWER line + inline top
   BBS candidates into the recovery turn. (H1/H10)
5. selective-delete compaction (user request) — in proactive compaction, deterministically
   drop certainly-wrong content (`(no output)` searches, rejected-candidate results) and
   preserve verified findings verbatim. (H4)

### E5 — dedicated-node vLLM crashed under v3 load (15:49Z) — RECOVERED
- **What:** the dedicated `sunonly-temp` vLLM engine shut down (`MPClient: start timeout=0s`)
  ~10 min into v3 (parallel 40, `subagent_max_turns=200` → much larger contexts than v2's 70).
  Likely memory/engine pressure from many concurrent long-context cases. v3's dedicated-node
  cases errored (conn=286), contaminating its report (showed v=0.048 — all errors).
- **Mitigation:** restarted vLLM via the clean tmux method (NOT setsid/pkill — my `pkill -f
  serve_vllm.sh` self-matched the exec command and SIGKILLed it, exit 137). Recovered in
  ~3 min. Killed the contaminated v3, relaunched **v3b at parallel 24** (halved concurrent
  big-context load) on [dedicated + b200], fresh dir. Monitor `bzwpil4eu` now also alerts on
  vLLM health!=200.
- **Lesson:** (1) never `pkill -f <pattern>` where the pattern matches your own exec command;
  (2) high parallel × large `subagent_max_turns` can OOM/crash a node — keep parallel ≤24 for
  big-context configs; (3) the dedicated node is flaky under load — b200 has been stable.
