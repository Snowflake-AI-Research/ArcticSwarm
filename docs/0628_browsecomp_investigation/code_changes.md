# Code changes (0628) — qwen-gated, default-off, unit-tested

All changes follow the in-tree convention: a flag declared on the YAML-facing dataclass
(`run_config.py`) + the flat `ArcticswarmConfig` (`config.py`), bridged in
`to_arcticswarm_config()`, read at the use site via `getattr(self.config, …, default)`.
**Defaults preserve current behavior**, so sonnet/gpt runs sharing this checkout are
byte-for-byte unchanged. Enabled only in the qwen browsecomp run.

## 1. `surface_bbs_candidates` (default False) — answer retention
**Targets:** H1 (≥31% of wrong cases found the answer but lost it) + H6 (BBS findings vanish
from the leader's working context after compaction / burst-truncation).
**What:** `PrepareReportTool._build_candidate_digest()` harvests VERIFIED `#consensus`
verdicts + top `#key-findings`/`#discoveries` posts from the BBS and appends a "CANDIDATE
FINDINGS THE TEAM CONVERGED ON" block to the report-unlock message, instructing the leader
to commit the best-supported candidate and never answer "no answer found". Re-surfaces a
finding even if it scrolled out of the compacted history.
**Files:** `config.py`, `run_config.py` (+bridge), `swarm/orchestrator.py` (construction),
`swarm/tools.py` (`PrepareReportTool.__init__` param + `_build_candidate_digest`, appended in
the execute success path). Test: `swarm/tests/test_surface_bbs_candidates.py` (3 cases, pass).
**Risk:** could re-introduce distractor candidates if the board holds many wrong ones — A/B
in Round 2.

## 2. `compaction_prune_junk` (default False) — selective-delete compaction (USER REQUEST)
**Targets:** H4 (compaction is an answer-loss amplifier: any compaction → acc 0.67→0.22; the
reactive path is lossy). User asked: "remove histories/paths that are certainly wrong —
selective delete" instead of pure LLM-summarize.
**What:** In `_compact_context_structured`, BEFORE the summarizer LLM call,
`_prune_certainly_wrong()` replaces certainly-junk tool-result CONTENT — empty `(no output)`,
`SYSTEM SHUTTING DOWN`/`is now DISABLED` timeout notices, `skipped — max tool calls` stubs,
`is_error` results, `(no results)` — with a 1-line placeholder, **only on the throwaway
`truncated_msgs` copy fed to the summarizer** (never the live history). So the model spends
its summary budget on real findings and certainly-wrong paths are dropped, not lossily
summarized.
**Why this (safe) shape:** modifying live `self.messages` structure risks breaking the
tool_use/tool_result pairing the API requires (→ all cases 400-error). Operating on the
disposable summarizer-input copy gets the user's intent with zero pairing risk. Junk pruning
is bounded to results <600 chars so a long result merely *containing* an error string is
preserved.
**Files:** `config.py`, `run_config.py` (+bridge), `context_management.py`
(`_prune_certainly_wrong` + call site + `_JUNK_RESULT_MARKERS`). Test:
`swarm/tests/test_compaction_prune_junk.py` (2 cases, pass).
**Deferred (documented, NOT shipped):** physically deleting message pairs from the live
history (true structural selective-delete) is higher-risk and not safe to deploy unattended;
the thrash caps (Round 1) are the first-line mitigation that should drive compaction events
toward 0 (measured in Round 1). If Round 1 shows compactions persist AND hurt, escalate to
structural deletion with careful pairing-preservation.

## 3. `_force_report` candidate-led timeout fallback (under `surface_bbs_candidates`)
**Targets:** the diagnostic's biggest residual failure — answer-loss/thrash cases that find
the answer (it's on the BBS) but never reach `send_user_markdown_report`: they hit the soft
timeout, the LLM keeps blocking on `wait_for_tasks`, and the force-report timer
(`timeout+200s`) fired the OLD fallback = a raw dump of the last 20 BBS posts → judge can't
extract the answer → scored 0.
**What:** when `surface_bbs_candidates` is on, `_force_report` (orchestrator.py ~1507) now
builds a COMMITTED answer that LEADS with VERIFIED `#consensus` verdicts + top
`#key-findings`/`#discoveries`, framed "commit the single best-supported candidate; a definite
answer exists". Deterministic (no LLM call in the timer thread → no hang risk); on any
exception falls back to the legacy raw dump. Default-off via the same flag.
**Why it matters:** the report-time hooks (`enable_empty_answer_recovery`, the prepare_report
digest) don't fire for these cases because the LLM never reaches the report step — the
timeout fallback is the ONLY place to commit. This is the lever for the recoverable bucket
on both the 120 and (larger payoff) the full set's ~22% timeout cases.

## 4. `alt_task_force_dispatch` (default False) — make the diversity sweep actually RUN
**Targets:** the recall/anchoring bucket (~63% of wrong cases) — the #1 ceiling lever.
**Root cause (workflow `wjutwlkp1`):** the candidate-emergence rival sweep
(`answer_verification.py` `wire_candidate_emergence_hook`) only calls `task_board.add_task(spec)`
— it NEVER dispatches a worker. On a saturated vLLM no browsing worker goes idle to pull it, so
it dies PENDING with 0 tool_uses; then `ctx.shutdown` ends all run_loops. Measured: spawned in
100% of cases, **never runs in 40.3%**; cases where it runs score +4 pts.
**What:** when the flag is on, the hook ALSO calls `orch._ctx.spawn_or_assign(spec)` (idle
worker / new worker up to `max_subagents` / else queue) — mirroring the working
`_spawn_contrarian_task` (the late gate). So the contrarian/diversity task is actually
dispatched + run. Confirmed live in v4: `dispatched=Erin/Veronica` (was always board-only before).
**Files:** `config.py`, `run_config.py` (+bridge), `swarm/answer_verification.py`. Test:
`swarm/tests/test_alt_task_force_dispatch.py` (on→dispatch, off→board-only; pass; alt-gate suite green).
**Risk:** one extra dispatched task per case (bounded by max_subagents); low. Requires
`enable_candidate_emergence_sweep=true` (already set in the qwen config) for the hook to wire.

## 5. `reframe_prompt` (default False) — anti-anchoring browsing prompt
**Targets:** the dominant recall ceiling. Recall re-investigation (workflow over 19
never-found cases): **17/19 = anchoring** — the swarm locks onto one interpretation/candidate
and reformulates searches WITHIN that frame (median ~249 searches vs ~13 fetches = snippet-
scanning, not reading; the frame-breaking alt-sweep often pending). Config knobs that add
searches search MORE within the wrong frame — they can't fix this.
**What:** when `reframe_prompt=true`, append `ANTI_ANCHOR_BLOCK` (prompts.py) to the BROWSING
agent system prompt only (teammate.py:600, gated): decompose into hard constraints; consider
2-3 different interpretations (entity type/era/domain/language); search each constraint
separately; DROP + reframe on disconfirm (don't reformulate the failed frame); read pages
(web_fetch) not just snippets. Default-off → other models/runs unaffected (per the
"route prompt changes to qwen only" constraint).
**Files:** `swarm/prompts.py` (ANTI_ANCHOR_BLOCK), `config.py`, `run_config.py` (+bridge),
`swarm/teammate.py` (gated append to browsing profile). 204 swarm tests pass.
**Tested live:** Run B (`recallB_reframe`) — see `03_results.md`.

## 6. `orchestrator_max_tool_calls_per_turn` (default -1=inherit) — role-aware leader budget (0629)
**Targets:** the browsecomp_1098 pathology — the orchestrator inherits `llm.max_tool_calls_per_turn: 1`
(it's built as a plain `Agent(self.config)`, orchestrator.py:1039, and drives turns via the same
`run_turn_streaming` every subagent uses). When Qwen emits N tool calls in one message, calls `[1:]`
are sliced off and replaced with `(skipped — max tool calls per turn reached)` is_error stubs
(agent.py streaming + non-streaming sites); **the dropped intent is never re-queued** (no retry path
anywhere). For a coordinator whose job is fan-out, this silently drops batched `create_task` calls and
`wait_for_tasks` polls. Empirical (`.invest_0628/raw/master_all.jsonl`, 1098/run 0625_5):
`create_task=15` but `n_tasks=10` → ~5 task-creations lost; 10 workers spawned (well under the 16 cap,
so capacity was NOT the bottleneck). Compounded by `wait_for_tasks` blocking up to 1500s
(tools.py:1928), which starves dispatch of the already-created tasks → near-serial fan-out.
**What:** `max_tc=1` is a deliberate BROWSING-SUBAGENT discipline (one-step search→read→reason on a
saturated 27B), so the fix is **role-aware**, not a global loosen. Added a per-agent override:
`Agent.max_tool_calls_per_turn_override` (None=inherit), resolved at both turn-loop enforcement sites
via the pure helper `_resolve_max_tool_calls(override, config_value)` (agent.py). The orchestrator
construction path (orchestrator.py:1039) sets the override from the new config flag — gated on `>= 0`
so the default `-1` is a strict no-op — and **never mutates the shared config** the subagents read.
`-1`=inherit (default), `0`=unlimited orchestrator, `>=1`=explicit orchestrator cap. Enabled in
`browsecomp_qwen.yaml` as `llm.orchestrator_max_tool_calls_per_turn: 0` (orchestrator unlimited;
subagents stay at 1).
**Files:** `config.py` (flat field), `run_config.py` (LLMConfig field + bridge), `agent.py`
(`_resolve_max_tool_calls` helper + `max_tool_calls_per_turn_override` attr + both enforcement sites),
`swarm/orchestrator.py` (per-agent override, gated), `conf/bench/browsecomp_qwen.yaml` (enable).
Test: `swarm/tests/test_orchestrator_max_tool_calls.py` (6 cases: resolver semantics, config default -1,
bridge, Agent attr, turn loops use the resolver, orchestrator gates on `>=0` and doesn't mutate config;
pass). Full swarm suite green (210).
**Default-safety:** Yes — flat default `-1` inherits current behavior, so every other config/model
(browsecomp.yaml, gpt5_duo, etc.) is byte-for-byte unchanged (verified: they load as -1). Only the
orchestrator agent is affected, and only when the flag is explicitly set ≥ 0.
**Note:** the reviewer-idle "No candidates to review" is NOT a runaway loop in the default config — it's
a prompt-template echo and the auditor runs the capped path (no config enables the uncapping flags);
not addressed here. The dropped-call salvage (re-queue) and the `wait_for_tasks` pre-dispatch drain
remain as follow-ups.

## 7. Viewer "errored" token rendering (0629)
**Targets:** the viewer (`viewer/`) showed `-` for tokens on ~half the cases, making errored/timed-out
cases (which legitimately captured no usage) look identical to clean zero-token runs. Root cause is
UPSTREAM-faithful (errored/timeout cases never populated `result.token_usage`; see runner.py token
harvest + the dead-code `total_input_tokens`/`total_output_tokens` reads at runner.py:1525/1530 — a
separate latent bug, NOT fixed here), so the viewer fix is display-only.
**What:** `viewer/server.py` `get_timeline` now passes `has_error`/`had_timeout`/`error` (already in
`get_questions`). `static/app.js` adds `tokenCell(q)` — when tokens are 0/absent AND the case errored,
render `err`/`timeout` (error string on hover) instead of `-`; `formatTokens` hardened against
null/undefined. `static/detail.js` mirrors it with `tokenLabel(data)` (`errored` / `errored (timeout)`).
The separate Error/Timeout columns and the "Unknown" judge badge already cover the missing-judge-label
case (checkpoint-before-judge race — an upstream issue, not the viewer).
**Files:** `viewer/server.py`, `viewer/static/app.js`, `viewer/static/detail.js`.
**Default-safety:** Yes — display-only; no eval/model paths touched.

## 8. `always_execute_tools_per_turn` + post→complete sequencing (default []) — subagent finding-preservation (0629)
**Targets:** the subagent `max_tc=1` tail-loss path. The final-summary prompt
(`teammate.py` `_build_summarize_prompt`) presented "post to BBS" (step 2) and "call complete_task"
(step 3) together, inviting Qwen to batch them. With `max_tc=1` only the first survives: usual order
(post first) lands the post and re-issues complete (fine, +1 turn); reverse order drops the post, and a
model that deems itself "done" may not re-post → feeds the documented never-posted bucket
(`browsecomp-found-but-not-selected`). Tail risk, but a real one for answer-retention.
**What (two parts):**
1. **Privileged always-execute tools** — new config `always_execute_tools_per_turn: list[str]` (default
   `[]`). At the per-turn cap sites, instead of a plain `tool_calls[:max_tc]` slice, the new pure helper
   `_split_capped_tool_calls(tool_calls, max_tc, privileged)` keeps the first `max_tc` calls **plus** any
   call whose name is privileged (regardless of position), dropping the rest. So
   `web_search + post_to_bbs + list_tasks` at cap 1 runs `web_search` AND `post_to_bbs`, drops
   `list_tasks`; `complete_task + post_to_bbs` runs **both** (post bypasses the cap) → a posted finding
   **always lands**, eliminating the loss path regardless of ordering. Reads from `self.config` so it
   applies to any capped agent (subagents/auditor); the orchestrator is unlimited so the block is skipped.
2. **Prompt sequencing** — when the resolved per-turn cap is 1, `_build_summarize_prompt` now tells the
   subagent to make `post_to_bbs` its ONLY call that turn and call `complete_task`/`update_task_summary`
   in a SEPARATE turn ("Never call `post_to_bbs` and `complete_task` in the same turn"). Gated on the
   actual cap (`single_tool_call`), not the model — so uncapped runs are unchanged. Reduces the wasted
   dropped+re-issued `complete_task` (latency) on top of the hard guarantee in (1).
**Files:** `config.py`, `run_config.py` (LLMConfig field + bridge), `agent.py`
(`_split_capped_tool_calls` + both enforcement sites), `swarm/teammate.py` (`_build_summarize_prompt`
sequencing + call site passes `single_tool_call`), `conf/bench/browsecomp_qwen.yaml`
(`always_execute_tools_per_turn: ["post_to_bbs"]`). Test:
`swarm/tests/test_orchestrator_max_tool_calls.py` (extended: split keeps first-N+privileged, post-first
unchanged, post-after-complete still executes, no-privileged=plain truncation, unlimited keeps all,
config default empty / qwen enables post_to_bbs, turn loops use the split, prompt sequences post→complete;
14 cases). Full swarm suite green (218).
**Default-safety:** Yes — `always_execute_tools_per_turn` defaults `[]` (strict cap, no bypass) and the
prompt sequencing is gated on cap==1, so other models/runs are byte-for-byte unchanged; enabled only in
the qwen config.

## Gating verification
