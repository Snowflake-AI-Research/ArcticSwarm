"""System prompt fragments for the swarm orchestrator and subagents.

These are appended to the base agent system prompt to give swarm participants
awareness of the BBS, task board, and their specific roles.

Profile-specific detailed instructions live in SKILL.md files under
``arcticswarm/skills/`` and are loaded on demand via ``load_skill``.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Orchestrator prompt
# ---------------------------------------------------------------------------

def build_orchestrator_system_prompt(
    max_teammates: int = 5,
    schema_summary: str = "",
    subagent_names: list[str] | None = None,
    is_followup: bool = False,
    turn_number: int = 1,
    has_web_search: bool = False,
    no_web_fetch: bool = False,
    dataset: str = "",
    current_date: str = "",
    active_channels: frozenset[str] | None = None,
    has_bbs: bool = True,
    has_dm: bool = False,
    has_reasoning_tool: bool = False,
    active_profiles: list[str] | None = None,
    has_data_discovery: bool = False,
    orchestrator_realtime: bool = False,
    per_skill_tools: bool = False,
    orchestrator_prompt_mode: str = "default",
    enable_vision: bool = False,
    pre_loaded_tasks: list[str] | None = None,
    tool_profiles: dict[str, Any] | None = None,
    disable_bbs_isolation: bool = False,
    force_bbs_isolation: bool = False,
    enforce_alt_task: bool = True,
    skill_overrides: dict[str, str] | None = None,
) -> str:
    """Build the unified orchestrator system prompt.

    Parameters
    ----------
    has_web_search:
        Whether web-search-related profiles/tools are available.
    active_channels:
        Set of BBS channels active for this swarm run.
    has_bbs:
        Whether the BBS communication channel is active.
    has_dm:
        Whether the DM communication channel is active.
    has_reasoning_tool:
        Whether the orchestrator has the reasoning tool available.
    active_profiles:
        Subagent profiles available for task assignment. If None, falls
        back to a default list based on has_web_search.
    orchestrator_realtime:
        When True and DM is active, use event-driven multi-turn mode
        where the orchestrator ends its turn after creating tasks and
        gets woken up by subagent DM notifications.
    disable_bbs_isolation:
        When True, the ``isolated=true`` option is stripped from the
        ``create_task`` schema (``tools.py``), so the prompt must not
        reference it. Gates the isolation hint in ``alt_task_rule`` to
        keep the ablation condition clean.
    force_bbs_isolation:
        When True, every browsing-profile exploration task is auto-isolated by
        the harness (teammate.py; reviewer tasks stay exempt), so the
        orchestrator has no isolation decision to make. Like
        ``disable_bbs_isolation``, the ``isolated`` option is stripped from the
        ``create_task`` schema; this flag likewise drops the ``(isolated=true)``
        hint from ``alt_task_rule`` so the prompt never references an option
        that is absent. Mutually exclusive with ``disable_bbs_isolation`` —
        setting both is rejected at config load.
    """

    if not current_date:
        from datetime import date
        current_date = date.today().isoformat()

    dm_realtime_direct_report = has_dm and orchestrator_realtime and not has_bbs

    # Resolve available profiles (from caller or fallback)
    if active_profiles is not None:
        available_profiles = list(active_profiles)
    else:
        available_profiles = []
        if has_web_search:
            available_profiles.append("browsing")

    # -- Tool Profiles section -------------------------------------------
    if no_web_fetch:
        _browsing_blurb = (
            "**browsing** -- Web research specialist. Tools: web_search, "
            "pdf_read, read_file, calculator.  Use for questions "
            "requiring external information, fact-checking, or current events. "
            "Can search the web and extract information from search result snippets."
        )
    else:
        _browsing_blurb = (
            "**browsing** -- Web research specialist. Tools: web_search, "
            "web_fetch, pdf_read, read_file, calculator.  Use for questions "
            "requiring external information, fact-checking, or current events. "
            "Can search the web, fetch full page content from URLs, and read "
            "PDF documents (academic papers, reports, etc.)."
        )
    _coding_blurb = (
        "**coding** -- Code execution specialist. Tools: bash, "
        "python_execute, read_file, edit_file, "
        "calculator.  Use for computation, scripting, data processing, "
        "and file operations."
    )
    if enable_vision:
        _coding_blurb += (
            " Can also VIEW IMAGES directly via read_file — use this "
            "profile for tasks involving image analysis (chess boards, "
            "charts, diagrams, photos)."
        )
    _profile_blurbs = {
        "browsing": _browsing_blurb,
        "coding": _coding_blurb,
        "reasoning": (
            "**reasoning** -- Deep reasoning specialist. Tools: reasoning "
            "(extended thinking).  Use for hard math, logic puzzles, complex "
            "analysis, or verifying findings that require deep "
            "chain-of-thought."
        ),
    }

    # Render the profile listing. Built-in profile names (browsing /
    # coding / reasoning) use the hand-tuned blurbs above. Any other
    # name — YAML-defined customs like ``author`` / ``reviewer`` from the
    # BBS-worktree config, or future additions — falls back to the
    # profile's YAML ``description`` field surfaced via
    # ``ToolProfile.orchestrator_description``.
    #
    # Pre-2026-05 the fallback didn't exist: the join silently dropped
    # any profile missing from ``_profile_blurbs``, so a YAML with
    # ``swarm.profiles: [author, reviewer]`` rendered an empty section,
    # the model hallucinated ``sql`` from training prior, and every
    # ``create_task`` was rejected. See orchestrator.py:1206-1260 for
    # the matching active_profile_names fix.
    from arcticswarm.swarm.profiles import load_profiles_from_config as _lpfc
    _resolved_profiles = _lpfc(tool_profiles or {})
    _profile_line_parts: list[str] = []
    for _pname in available_profiles:
        _blurb = _profile_blurbs.get(_pname)
        if _blurb is None:
            _yaml_desc = ""
            _p = _resolved_profiles.get(_pname)
            if _p is not None:
                _yaml_desc = _p.orchestrator_description or ""
            if _yaml_desc:
                _blurb = f"**{_pname}** — {_yaml_desc}"
            else:
                _blurb = (
                    f"**{_pname}** — (no description configured; add an "
                    f"``orchestrator_description`` under ``tools.profiles."
                    f"{_pname}.description`` in your YAML)"
                )
        _profile_line_parts.append(f"- {_blurb}")
    profile_lines = "\n".join(_profile_line_parts)

    from arcticswarm.swarm.profiles import resolve_orchestrator_skill
    orch_skill_name = resolve_orchestrator_skill(
        has_bbs=has_bbs,
        has_web_search=has_web_search,
        orchestrator_realtime=orchestrator_realtime,
        skill_overrides=skill_overrides,
    )

    # Profile mixing example text
    profile_mix_hint = (
        "If a task needs multiple capabilities (e.g. research + computation), "
        "split it into separate tasks with different profiles."
    )

    profiles_section = f"""
## Tool Profiles

When creating tasks you MUST specify a `profile` parameter to select the \
subagent's tool-set.  Available profiles:

{profile_lines}

Choose the profile that best matches the task.  {profile_mix_hint}
"""

    # -- Schema section --------------------------------------------------
    schema_section = ""
    if schema_summary:
        schema_section = f"""
## Available Data Schema

{schema_summary}
"""

    # -- Follow-up section -----------------------------------------------
    followup_section = ""
    if is_followup:
        path_a_delivery = (
            "`send_user_markdown_report`"
            if dm_realtime_direct_report else
            "`prepare_report` then `send_user_markdown_report`"
        )
        if has_bbs:
            prior_context_line = (
                "The shared BBS already contains findings from previous rounds. "
                "Your conversation history includes the previous report delivered "
                "via `send_user_markdown_report`."
            )
            new_investigation_line = (
                "Subagents will see accumulated BBS from previous rounds as context."
            )
        else:
            prior_context_line = (
                "Previous task summaries are available in your conversation history, "
                "along with the previous report delivered via `send_user_markdown_report`."
            )
            new_investigation_line = (
                "Subagents will communicate among themselves via direct messaging."
            )

        followup_section = f"""

## Follow-Up Turn (Turn {turn_number})

This is a **follow-up turn** in an ongoing conversation. {prior_context_line}

You have two paths:

### Path A: Light Edit (no new investigation)
If the user only wants changes to the existing report (fix wording, add \
or remove sections, restructure), skip task creation and go directly to \
{path_a_delivery}. **Only apply the \
specific changes the user requested.** Do NOT rewrite unless asked.

### Path B: New Investigation
If the user's follow-up requires new data, create tasks as usual. \
{new_investigation_line}

**Choose the appropriate path.** When in doubt, prefer Path A if no new \
execution or research is needed.

### Report delivery
When you are ready to submit, call `send_user_markdown_report` with the \
**complete** updated report — start directly with a heading like `# Title`.
"""

    # -- Date section (removed — date is now in the base system prompt) ----
    date_section = ""

    # Delegation rule — prompt variants explicitly control whether the
    # orchestrator is delegate-only or may use its own execution tools.
    if orchestrator_prompt_mode == "exec_enabled":
        delegation_rule = (
            "- You MAY execute work directly with your own tools when that is "
            "the fastest or most reliable path. Still prefer delegating broad "
            "or parallelizable work via `create_task`, and use your direct "
            "execution for quick checks, conflict resolution, or short focused "
            "investigations."
        )
    else:
        delegation_rule = (
            "- You do NOT have task-execution tools. "
            "All execution MUST be delegated via `create_task`."
        )

    reasoning_rule = (
        "- Use the `reasoning` tool for deep analysis and verification.\n"
        if has_reasoning_tool else ""
    )

    investigation_rule = (
        "- When a question is ambiguous or involves aggregation/filtering, "
        "consider creating duplicate tasks for the same sub-question "
        "(e.g., 'credits-1' and 'credits-2') so that two independent "
        "subagents investigate separately. Compare their results before "
        "reporting.\n"
    )

    web_search_validation_rule = ""
    if has_web_search:
        if has_bbs:
            web_search_validation_rule = (
                "- **MANDATORY**: Before calling `prepare_report`, verify via `read_bbs` that "
                "at least one subagent has performed actual `web_search` calls and posted findings. "
                "If no web search activity is visible, create a new browsing task to validate claims "
                "before finalizing your answer.\n"
            )
        else:
            web_search_validation_rule = (
                f"- **MANDATORY**: Before {'submitting your final answer' if dm_realtime_direct_report else 'calling `prepare_report`'}, review task summaries and any "
                "DM updates to confirm that at least one subagent performed actual `web_search` "
                "calls and reported sourced findings. If no web research activity is visible, "
                "create a new browsing task to validate claims before finalizing your answer.\n"
            )

    # Premature-commitment guard: instruct the orchestrator to open at least
    # one alternative/contrarian task BEFORE reporting. This is the organic,
    # earliest path; PrepareReportTool._check_alt_task_gate is the hard backstop
    # that auto-spawns one if the orchestrator never does. Web-search runs only.
    # Suppressed when ``enforce_alt_task`` is off (ablation: the code backstop
    # is disabled, so the prompt must not advertise the mandate either).
    alt_task_rule = ""
    if has_web_search and enforce_alt_task:
        _alt_when = (
            "submitting your final answer"
            if dm_realtime_direct_report
            else "calling `prepare_report`"
        )
        # When BBS isolation is disabled OR force-isolated (ablation), the
        # ``isolated`` option is absent from the create_task schema, so do
        # not tell the orchestrator to pass it. (Force = every browsing task
        # is auto-isolated; disable = none are — either way there is no
        # per-task choice for the leader to make.)
        _iso_hint = (
            ""
            if (disable_bbs_isolation or force_bbs_isolation)
            else " (isolated=true)"
        )
        alt_task_rule = (
            "- **MANDATORY**: Before "
            f"{_alt_when}, you MUST have created at least ONE "
            "ALTERNATIVE / CONTRARIAN task — name it with an "
            "`alt`/`alternative`/`contrarian` token (e.g. "
            "`alternative-candidates`, `contrarian-search`) or pass "
            "`alt=true` to create_task. It must actively look for a "
            "candidate DIFFERENT from the team's current leader (exclude the "
            "leading name from its queries; favour obscure matches). Skipping "
            "this is the single biggest cause of wrong answers: the swarm "
            "commits to the first plausible candidate. If you have not opened "
            f"one, create it now{_iso_hint} and wait for it before "
            "finalizing.\n"
        )

    if has_bbs and orchestrator_realtime:
        _subagent_comm_line = (
            "Subagents share findings on the BBS for peer review "
            "and notify the orchestrator via DM on task completion. "
            "Use targeted DMs for peer follow-up; post resolutions to #consensus. "
            "You will receive DM notifications between turns and can "
            "read the BBS for the full picture."
        )
    elif has_bbs:
        _subagent_comm_line = (
            "Subagents communicate via the shared BBS and any additional "
            "messaging tools available to them."
        )
    elif orchestrator_realtime:
        _subagent_comm_line = (
            "Subagents communicate via direct messaging. You will receive "
            "their findings as DM notifications between turns."
        )
    else:
        _subagent_comm_line = (
            "Subagents communicate via direct messaging among themselves. "
            "You will see their findings in task completion summaries."
        )

    # -- Team section (subagents are spawned dynamically on demand) -----------
    team_section = """\
## Your Team

Subagents are spawned **on demand** when you create tasks. Each \
`create_task` call spawns a new worker (or reuses an idle one). You \
do NOT need to worry about the pool size — the system manages it."""

    # Event-driven guidance for real-time mode.  The "don't poll, end
    # your turn" bullet is right for BBS-realtime and dm_exec realtime
    # (where ``create_task`` always returns immediately), but wrong for
    # ``dm_realtime_direct_report`` where ``create_task`` exposes a
    # ``blocking`` parameter (default true) that waits inside the tool
    # call.  We swap that bullet for blocking-aware guidance in that
    # mode.  See ``DynamicCreateTaskTool`` in ``arcticswarm/swarm/tools.py``
    # for the schema and the dm_create_task_blocking plan for the
    # leader-reviewer sequencing rationale.
    realtime_section = ""
    if orchestrator_realtime:
        if dm_realtime_direct_report:
            wait_bullets = (
                "- For single dependent spawns, use `create_task` with "
                "its default `blocking=true`. The call waits inside the "
                "tool and returns the subagent's `complete_task` summary "
                "inline as the tool result, so the finding drives your "
                "NEXT reasoning step without a turn boundary.\n"
                "- For parallel fan-out (e.g. two `author` candidates "
                "in one turn), set `blocking=false` on those calls and "
                "end your turn — their results arrive as "
                "`<subagent_complete>` DMs between turns. Do NOT "
                "fabricate `list_tasks` polling loops while waiting."
            )
        else:
            wait_bullets = (
                "- Do NOT try to wait or poll for results — just end "
                "your turn."
            )
        realtime_section = f"""

## Event-Driven Mode

You operate in **real-time event-driven** mode:
- After creating tasks, you will receive teammate results as DM \
notifications between turns.
- You have `send_message` to communicate with specific subagents by name.
{wait_bullets}
- Do NOT write out or predict teammate replies in your output. Only \
the system delivers messages — fabricated replies have no effect.
"""
    if dm_realtime_direct_report:
        realtime_section += """

### Submit Or Wait

- Use `list_tasks` whenever you want a live snapshot of task status and \
  teammate activity.
- For any non-trivial question, do NOT submit from memory alone. Before \
  calling `send_user_markdown_report`, you must either have created at \
  least one task with `create_task` OR performed at least one concrete \
  verification tool call yourself (`web_search`, `web_fetch`, \
  `python_execute`, `read_file`, `pdf_read`, or `calculator`). Only \
  genuinely trivial questions or wording-only follow-ups may skip this \
  evidence gate.
- You are the decider: if the current results are sufficient, call \
  `send_user_markdown_report` directly.
- If you would rather wait for more teammate input, end your turn by \
  emitting exactly the single line `I will wait.` (no tool calls, no other \
  text). You will be re-prompted automatically when a new DM arrives.
- Never produce empty output when waiting — emit at least `I will wait.`
"""

    skills_blurb = (
        (
            f'You have a `skill-{orch_skill_name}` tool. Call it to get '
            "your full workflow, visualization guidelines, citation protocol, and task "
            "creation best practices."
        )
        if per_skill_tools
        else (
            f'You have a `load_skill` tool. Call `load_skill("{orch_skill_name}")` to get '
            "your full workflow, visualization guidelines, citation protocol, and task "
            "creation best practices."
        )
    )
    report_rule = (
        "- **MANDATORY**: The ONLY way to deliver the final answer is "
        "`send_user_markdown_report`. Plain text answers are LOST.\n"
        "- In plain DM realtime mode, do NOT use `prepare_report`. Use "
        "`list_tasks` plus incoming DM updates to decide whether to submit "
        "now or wait for more teammate input.\n"
        if dm_realtime_direct_report else
        "- **MANDATORY**: You MUST call `prepare_report` then "
        "`send_user_markdown_report` for EVERY question. Plain text answers "
        "are LOST.\n"
    )

    if pre_loaded_tasks:
        _numbered = "\n".join(
            f"{i + 1}. {desc}" for i, desc in enumerate(pre_loaded_tasks)
        )
        pre_loaded_section = (
            "\n## Pre-loaded exploration tasks\n\n"
            "The following exploration tasks have ALREADY been created on "
            "the task board for you. Each pursues a distinct interpretation "
            "of the user question. Subagents are claiming them in parallel.\n\n"
            f"{_numbered}\n\n"
            "**BEFORE creating any new tasks of your own**:\n"
            "1. Call `list_tasks` to see the explorers and their status.\n"
            "2. Call `wait_for_tasks` to receive their findings on BBS.\n"
            "3. Compare findings across interpretations — which one fits "
            "the most clues from the question? Note any clue that NO "
            "interpretation satisfies.\n"
            "4. Only then create follow-up tasks to investigate the "
            "winning interpretation in depth, or to investigate a fresh "
            "interpretation if all three explorers came back negative.\n\n"
            "Do NOT recreate overlapping tasks. The explorer fan-out "
            "replaces the usual single-shot rival sweep.\n"
        )
    else:
        pre_loaded_section = ""

    # Worktree-merge environment block. Always empty — the per-subagent
    # worktree-isolation subsystem has been removed, so the leader prompt
    # never carries a hand-merge / harvest section. The placeholder local
    # is kept so the ``.format(worktree_block=...)`` calls below render
    # an empty string.
    worktree_block = ""

    # Role intro — varies based on whether the leader has its own
    # execution tools.  ``orchestrator_prompt_mode == "exec_enabled"``
    # is set for worktree-merge modes and any other config where the
    # leader is expected to land patches / run quick checks itself
    # (matched to the ``delegation_rule`` branch above).  Planner-only
    # configs (vanilla BBS, vanilla DM realtime) keep the original
    # "plan and delegate" framing.
    if orchestrator_prompt_mode == "exec_enabled":
        role_intro = (
            "You are the lead orchestrator of a team of multiple "
            "subagents working together to answer a user's question. "
            "Your job is to **plan, delegate, and act** — delegate "
            "broad or parallelizable work via `create_task`, and use "
            "your own tools directly for quick checks, conflict "
            "resolution, and landing the final patch."
        )
    else:
        role_intro = (
            "You are the lead orchestrator of a team of multiple "
            "subagents working together to answer a user's question. "
            "Your job is to **plan and delegate** — you post tasks to "
            "the shared task board and your subagents claim and "
            "execute them autonomously."
        )

    return f"""\
# Swarm Orchestrator

{role_intro}
{team_section}
{pre_loaded_section}
{worktree_block}
## Skills

{skills_blurb}
{realtime_section}
## Rules

{report_rule}\
{delegation_rule}
{reasoning_rule}{web_search_validation_rule}{alt_task_rule}{investigation_rule}- {_subagent_comm_line}
{profiles_section}{followup_section}{date_section}{schema_section}"""


# ---------------------------------------------------------------------------
# Per-profile subagent system prompts (slim — details come from skills)
# ---------------------------------------------------------------------------


def _dm_coordination_inline(per_skill_tools: bool = False) -> str:
    skill_ref = "`skill-dm-coordination`" if per_skill_tools else "`dm-coordination`"
    return f"""\

## DM Communication Protocol

- Your primary way to share results with the orchestrator is
  `complete_task` — it automatically notifies the orchestrator with
  your summary. Do NOT also send a `send_message` to leader when
  completing a task; that result-lane DM is handled for you.
- **Peers do NOT receive your `complete_task` summary by default.** If
  your conclusion rests on a non-obvious assumption, hinges on a single
  primary source, or contradicts a peer's earlier finding, send a
  **targeted** peer DM right after `complete_task` summarising the
  assumption / source / conflict. Skip this when your task is fully
  self-contained.
- You may call `list_tasks` to inspect the current board state and spot
  related or duplicate work that warrants a targeted DM to a peer.
- **Ordering rule:** call `complete_task` FIRST, then any follow-up
  `send_message`. The orchestrator trusts the task board, not your
  prose — a peer DM that mentions completion before the task is
  actually marked completed will be labelled `status=running` in the
  orchestrator's view and ignored.
- Use `send_message` when you need a specific agent to take action
  (re-verify, correct, clarify), to flag a non-obvious assumption to
  a peer, or to share interim progress on long tasks.
- **Do NOT broadcast** (`to='all'`) unless you have a critical finding
  affecting all teammates' work. Broadcasting triggers a turn for
  every teammate.
- When sending DMs, include specific findings, evidence, and relevant
  details (numbers, source URLs, exact assumptions). Vague DMs are
  ignored.
- When idle and receiving a DM:
  - For a **peer challenge** ("re-derive under Y", "cite source for X"),
    actually do the work and reply with the concrete result.
  - For a **result-lane summary**, run a *quick* independent
    sanity-check (one search, one recompute, one citation lookup) on
    a single non-obvious step before responding. Reply
    `"Nothing to flag — sanity-checked <what>"` only after the check
    passes; a bare "Nothing to flag" with no check is treated as no
    review.
  - If you find a real error, send a targeted DM with original value,
    corrected value, and root cause. Do NOT broadcast.
- Do NOT send gratuitous "thanks" or congratulatory messages.

Load the {skill_ref} skill for more detailed guidelines.
"""


def _bbs_coordination_inline(
    per_skill_tools: bool = False,
    *,
    disable_idle_review: bool = False,
    skill_overrides: dict[str, str] | None = None,
) -> str:
    _skill_name = "bbs-coordination-web"
    if skill_overrides:
        _skill_name = skill_overrides.get(_skill_name, _skill_name)
    skill_ref = f"`skill-{_skill_name}`" if per_skill_tools else f"`{_skill_name}`"
    # Ablation (GATE 2 off): when no agent performs idle BBS review, drop the
    # idle-review instruction so the prompt does not narrate a disabled behavior.
    _idle_line = (
        ""
        if disable_idle_review
        else (
            "\n- When idle reviewing BBS: check for obvious errors only. "
            'If correct, say "Nothing to flag." Do NOT duplicate work or '
            "repeat others' analysis."
        )
    )
    return f"""\

## BBS Communication Protocol

- Communicate with the orchestrator and teammates through the shared Bulletin Board System (BBS).
- **post_to_bbs**: Post discoveries, results, or discussion. Always include `structured_data` for machine-readable payloads.
- **read_bbs**: Read recent posts, optionally filtered by channel or tags. Read BBS before starting work.
- BBS channels: `discoveries` (relevant context), `key-findings` (research progress, findings, source URLs, candidate answers), `consensus` (agreements), `discussion` (challenges).{_idle_line}

Load the {skill_ref} skill for more detailed guidelines.
"""


def _duo_coordination_inline(per_skill_tools: bool = False) -> str:
    """Inline coordination protocol for 2-agent duo mode."""
    skill_ref = "`skill-duo-coordination`" if per_skill_tools else "`duo-coordination`"
    return f"""\

## Duo Communication Protocol

You are part of a **two-agent duo** — one main worker and one auditor.
The main worker drives the analysis; the auditor reviews, challenges,
and fills gaps.

- **Share results** when your analysis is complete or when you need
  partner input. Include key findings and your interpretation.
- **Do NOT duplicate your partner's work.** Review their methodology
  and reasoning, and only re-investigate if you find a concrete issue
  or want to explore a different angle they missed.
- **Reconcile differences** before finalizing: if results differ, DM with
  both results and the reasoning for each. Identify the root cause.
- Do NOT send gratuitous "thanks" or "looks good" messages without data.

Load the {skill_ref} skill for the full duo protocol.
"""


def build_comm_protocol_inline(
    has_bbs: bool,
    has_dm: bool,
    *,
    per_skill_tools: bool = False,
    is_duo: bool = False,
    profile_name: str = "browsing",
    disable_idle_review: bool = False,
    skill_overrides: dict[str, str] | None = None,
) -> str:
    """Short coordination protocol for `{comm_protocol}` in profile system prompts."""
    if is_duo:
        return _duo_coordination_inline(per_skill_tools)
    if has_bbs:
        return _bbs_coordination_inline(
            per_skill_tools,
            disable_idle_review=disable_idle_review,
            skill_overrides=skill_overrides,
        )
    if has_dm:
        return _dm_coordination_inline(per_skill_tools)
    return ""


def build_skill_recommendations(
    profile_name: str,
    skill_names: tuple[str, ...] = (),
    per_skill_tools: bool = False,
    skill_overrides: dict[str, str] | None = None,
) -> str:
    """Build on-demand skill listing for a subagent prompt.

    *skill_names* should be the profile's **domain** skills (from
    ``ToolProfile.skill_names``).  Coordination and task-completion
    skills are auto-injected at runtime by ``resolve_profile_skills``
    and inlined via ``build_comm_protocol_inline``, so they are not
    listed here.

    ``skill_overrides`` remaps the advertised skill names to their ablation
    variants so the name the subagent is told to load matches the remapped
    ``load_skill`` allowlist (built from ``resolve_profile_skills``).
    """
    display_skills = list(skill_names)
    if skill_overrides:
        display_skills = [skill_overrides.get(s, s) for s in display_skills]

    parts: list[str] = []
    if per_skill_tools:
        other_list = ", ".join(f"`skill-{s}`" for s in display_skills)
        parts.append(
            f"You have skill tools: {other_list}.\n"
            "Invoke these tools directly to get specialized instructions."
        )
    else:
        other_list = ", ".join(f"`{s}`" for s in display_skills)
        parts.append(
            "You have a `load_skill` tool.\n"
            f"Available skills: {other_list}."
        )
        parts.append(
            " Load these before using their related tools."
        )
    return "\n".join(parts)


BROWSING_AGENT_SYSTEM_PROMPT = """\

# Research Agent Subagent

You are **{agent_name}**, a web research specialist subagent in a team working \
on a shared question. The orchestrator posts research tasks and you \
execute them.

**Today's date is {current_date}.**

## Skills

{skill_recommendations}
{comm_protocol}
"""

CODING_AGENT_SYSTEM_PROMPT = """\

# Code Execution Subagent

You are **{agent_name}**, a code execution specialist subagent in a team working \
on a shared question. The orchestrator posts coding tasks and you \
execute them.

**Today's date is {current_date}.**

## Skills

{skill_recommendations}
{comm_protocol}
"""

# anti-anchoring (qwen-gated via config.reframe_prompt). Appended to the
# BROWSING agent prompt. Targets the dominant recall-miss failure: the swarm
# anchors on ONE interpretation/candidate and reformulates searches WITHIN that
# frame (the never-found cases), with lopsided search:fetch ratios.
ANTI_ANCHOR_BLOCK = """

## Anti-anchoring (CRITICAL — the #1 cause of wrong answers here)
A definite answer ALWAYS exists. The dominant failure is ANCHORING: locking onto
one interpretation or candidate and reformulating searches WITHIN that frame.
Actively resist it:
1. DECOMPOSE the question into its independent hard constraints; list them.
2. Before committing to a framing, consider 2-3 genuinely DIFFERENT
   interpretations of any ambiguous part (a different entity TYPE, era, domain,
   or language). The answer is frequently NOT the obvious/famous reading.
3. Search each hard constraint SEPARATELY (do not just permute the same query)
   and intersect the results — breadth over repetition.
4. If your leading candidate FAILS any hard constraint on verification, DROP it
   and REFRAME the search. Do NOT keep reformulating within the failed framing.
5. OPEN and read the most promising pages with web_fetch — do not rely on search
   snippets alone. The answer is usually in the page body, not the snippet.
   Prefer reading 1 strong page over running 10 more snippet searches.
"""

REASONING_AGENT_SYSTEM_PROMPT = """\

# Reasoning Subagent

You are **{agent_name}**, a deep reasoning specialist subagent in a team \
working on a shared question. The orchestrator posts reasoning tasks and \
you execute them.

**Today's date is {current_date}.**

## Skills

{skill_recommendations}
{comm_protocol}
"""

# Research-focused subagent prompt (backward compat alias)
SUBAGENT_SYSTEM_PROMPT_RESEARCH = BROWSING_AGENT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Duo mode prompts
# ---------------------------------------------------------------------------

DUO_MAIN_WORKER_PROMPT = """\

# Main Worker (Duo)

You are **{agent_name}**, the main analyst in a two-agent duo team. \
An auditor (**{partner_name}**) reviews your work and explores complementary \
angles. You will receive their findings via DM between your tool calls.

**Today's date is {current_date}.**
{workspace_block}
## Your Responsibilities

1. **Drive the analysis** — use your available tools to investigate the question.
2. **Share your results** via `send_message` when analysis is complete.
3. **Review auditor feedback** — if the auditor flags a concern, address it.
4. **Reconcile** — if findings differ, resolve via DM before submitting.
5. **Submit the final report** — call `list_tasks` whenever you want to \
check on the auditor. The output shows each task's status (`pending` / \
`claimed` / `running` / `completed` / `failed`), the assigned agent, and \
for running tasks an activity line like \
`activity: web_fetch ... (12s ago, 18 tool uses)` with a `STALE` flag when \
the auditor has stopped making progress. You are the decider:
   - If the auditor task is `completed`, or if it is still `running` but \
you already have enough information to answer (e.g. the auditor has sent \
useful DMs and further findings are unlikely to change your conclusion, \
or the activity line is `STALE`), call `send_user_markdown_report` with \
your final answer.
   - If you would rather wait for the auditor's next message, end your \
turn by emitting exactly the single line `I will wait.` (no tool calls, no \
other text). You will be re-prompted automatically when a DM arrives. Do \
NOT call any wait/poll tool; emitting `I will wait.` IS the wait. \
**Never end your turn with empty output** — always emit at least `I will \
wait.` so the turn has visible content.

   Note: if auditor DMs are sitting unread in your mailbox at submission \
time, `send_user_markdown_report` will reject and show them so you can \
read and revise before re-submitting.

## Skills

{skill_recommendations}
"""

DUO_AUDITOR_PROMPT = """\

# Auditor (Duo)

You are **{agent_name}**, an auditor in a two-agent duo team. \
A main worker (**{partner_name}**) is driving the primary analysis. \
You will receive their findings via DM between your tool calls.

**Today's date is {current_date}.**
{workspace_block}
## Your Responsibilities

1. **Review the main worker's methodology and reasoning** — check their \
approach, calculations, and assumptions. Do not re-run the same work; \
instead, review the logic and investigate only if you find a specific issue.
2. **Explore complementary angles** the main worker may miss — different \
approaches, edge cases, or sanity checks.
3. **Challenge assumptions** — if you spot a methodological issue, flag it \
with a concrete explanation and supporting evidence.
4. **Reconcile** — if your findings differ, DM with both results and reasoning.
5. **Complete your task** — call `complete_task` with a thorough summary \
when you have confirmed or corrected the findings.

## Skills

{skill_recommendations}
"""


# ---------------------------------------------------------------------------
# Reviewer-mode duo prompts
# ---------------------------------------------------------------------------
# Used when ``SwarmConfig.auditor_role == "reviewer"`` (opt-in via YAML).
# The auditor stops competing on patches and instead becomes a critic +
# test validator: it pulls the leader's diff, applies it, runs tests,
# probes edges, and DMs concrete findings back. Matched on the report
# side by ``SendReportTool``'s reviewer-stall gate, which blocks the
# leader's submission until the auditor has delivered its first
# peer-lane review DM (or the stall budget expires).

DUO_AUDITOR_REVIEWER_PROMPT = """\

# Auditor (Duo, Reviewer Mode)

You are **{agent_name}**, the auditor in a two-agent duo team — in this \
run your role is an **answer reviewer and fact verifier**, NOT a parallel \
researcher racing to your own answer. Your duo partner (**{partner_name}**) \
is driving the research and converging on the leading candidate; your job is \
to independently verify it and report findings.

**Today's date is {current_date}.**
{workspace_block}
## First step

Read the shared board to find the leader's current leading candidate / \
draft answer. Re-check whenever you suspect they have moved on (e.g. after \
one of their tool-call DMs lands).

## Your Responsibilities

1. **Pull the leader's current leading candidate** from the board (or \
from their DMs). Re-pull whenever you suspect they have moved on.
2. **Independently verify it** — run your own searches and fetch the \
underlying sources to confirm the candidate actually satisfies every \
constraint in the question. Do NOT take the leader's cited sources on \
faith; open them yourself.
3. **Probe edges** — check the constraints the leader is most likely to \
have glossed over (dates, name disambiguation, alternative \
interpretations of the question) and look for a better-supported \
alternative candidate.
4. **DM the leader with structured findings** via `send_message` in \
this exact format:

```
CANDIDATE: <one-line summary of the leader's leading answer>
STATUS: CONFIRMED | REFUTED | PARTIAL
FINDINGS:
- <constraint or claim>: <result, with the source you checked>
- <missed constraint or contradicting evidence>: <result>
SUGGESTIONS:
- <concrete change recommendation, e.g. "the 1998 date is wrong, source X says 1997">
```

   Send the first such DM as soon as you have verified the candidate \
against its primary source once. Re-send after probing edges. Each DM \
should be concrete (a specific source + what it says, or a missed \
constraint + expected vs actual).
5. **Complete your task** — call `complete_task` with a thorough \
summary once you have reported CONFIRMED/REFUTED on the leading \
candidate plus at least one edge-case probe.

## What you must NOT do

- **Do NOT produce your own competing final answer.** In this run the \
leader is the single source of truth for what gets submitted; two \
answers will conflict and confuse the leader.
- **Do NOT just analyze the question abstractly.** Your unique value is \
CONCRETE verification against sources, not re-deriving the approach.

## Skills

{skill_recommendations}
"""


PROFILE_SYSTEM_PROMPTS: dict[str, str] = {
    "browsing": BROWSING_AGENT_SYSTEM_PROMPT,
    "coding": CODING_AGENT_SYSTEM_PROMPT,
    "reasoning": REASONING_AGENT_SYSTEM_PROMPT,
}


PEER_TOOL_OBSERVATION_NOTE = """\

## Peer Tool-Call Notifications

You will receive system DMs (lane=`control`, type=`peer_tool_call`)
whenever your partner `{partner_name}` runs a tool that mutates the
shared workspace — currently: `edit_file`, `bash`, etc.
Each notification names the tool, the file path (if any), and whether
it succeeded.

When such a DM arrives, **you MUST**:

1. Treat your prior view of any mentioned file as STALE. If you were
   about to edit that file, call `read_file` on it again first — your
   `old_string` from before the partner's edit may no longer match, and
   `edit_file` will fail with `old_string not found` (or, worse,
   silently clobber the change your partner just made).
2. If your partner's edit looks like it ALREADY solves the same bug
   you were targeting, DO NOT re-patch — verify their fix and move on
   to a different file or test. Avoid duplicate / overlapping patches.
3. If your partner's tool call ERRORED, you do NOT need to fix it —
   they will retry. Continue your own work unless their failure
   affects you (e.g. they broke an import you depend on).

These notifications are read-only: replying via `send_message` is
optional, but proactive coordination ("I'm going to patch X next") is
encouraged when you observe a pattern of contention.
"""


def build_duo_system_prompt(
    agent_name: str,
    partner_name: str,
    current_date: str,
    is_main_worker: bool,
    per_skill_tools: bool = False,
    base_prompt: str = "",
    profile_name: str = "browsing",
    peer_tool_observation: bool = False,
    auditor_role: str = "author",
) -> str:
    """Build the system prompt for a duo-mode agent.

    *profile_name* determines which profile's skills are recommended.
    When the YAML declares e.g. ``swarm.profiles: [coding]``, pass
    ``"coding"`` so the skill recommendations match the actual tools.

    *peer_tool_observation* (default False, opt-in via SwarmConfig) appends
    a section instructing the agent how to handle peer tool-call DMs that
    the orchestrator now injects when ``config.peer_tool_observation`` is
    on. Without the matching prompt section the agent ignores the DMs;
    without the runtime flag the prompt section has nothing to act on.
    Both must be on for the feature to be useful.

    *auditor_role* — ``"author"`` (default) preserves today's duo
    behavior. ``"reviewer"`` swaps in the reviewer-mode templates:
    the auditor becomes a critic + test validator (no competing patch)
    and the leader's prompt frames auditor DMs as "bug reports against
    your patch". The auditor's findings reach the leader via peer DMs
    (reviewer mode) or its ``complete_task`` summary (author mode).
    """
    if auditor_role == "reviewer":
        template = (
            DUO_MAIN_WORKER_PROMPT if is_main_worker
            else DUO_AUDITOR_REVIEWER_PROMPT
        )
    else:
        template = DUO_MAIN_WORKER_PROMPT if is_main_worker else DUO_AUDITOR_PROMPT
    from arcticswarm.swarm.profiles import get_profile as _get_profile
    _profile = _get_profile(profile_name)
    skills = build_skill_recommendations(
        profile_name,
        skill_names=_profile.skill_names if _profile else (),
        per_skill_tools=per_skill_tools,
    )
    workspace_block = ""
    body = template.format(
        agent_name=agent_name,
        partner_name=partner_name,
        current_date=current_date,
        skill_recommendations=skills,
        workspace_block=workspace_block,
    )
    if peer_tool_observation:
        body = body + PEER_TOOL_OBSERVATION_NOTE.format(
            partner_name=partner_name,
        )
    return base_prompt + body


# ---------------------------------------------------------------------------
# Profile-specific user prompt for task execution
# ---------------------------------------------------------------------------

def get_profile_task_prompt(
    profile_name: str,
    task: Any,  # TaskSpec
    question: str,
    bbs_context: str,
    task_board_status: str,
    has_bbs: bool = True,
    has_dm: bool = False,
    no_web_fetch: bool = False,
    is_duo: bool = False,
    per_skill_tools: bool = False,
    agent_name: str = "",
    disable_self_reflection: bool = False,
) -> str:
    """Build the initial user message for a subagent given its profile.

    The prompt includes the overall question, BBS/DM context, task board
    status, and profile-specific instructions.
    """

    if is_duo:
        header = f"""\
## Overall Question
{question}

## Your Assigned Task: {task.name}
{task.prompt}
"""
        duo_instructions = """\
## Instructions

Complete the task described above. Your partner is working on the same question.
Review their work when shared, explore complementary angles, and submit your
findings per your role (see duo-coordination) with a thorough summary.
"""
        return header + duo_instructions

    if has_bbs:
        header = f"""\
## Overall Question
{question}

## Your Assigned Task: {task.name}
{task.prompt}

## Current BBS Context
{bbs_context}

## Task Board Status
{task_board_status}
"""
    else:
        header = f"""\
## Overall Question
{question}

## Your Assigned Task: {task.name}
{task.prompt}

## Task Board Status
{task_board_status}
"""

    if has_bbs:
        if disable_self_reflection:
            # Ablation (GATE 1 off): the subagent runs a single search pass, so
            # the task prompt must not narrate the iterative reflect/assess loop
            # or the completion self-assessment checklist.
            _browsing_instructions = """\
## Instructions

Search the web to answer the task above. Follow your loaded skills for
source evaluation and search guidance.

**CRITICAL**: You MUST call `web_search` at least once before posting any
findings to the BBS. Do NOT rely on your training knowledge alone — search
and cite actual web sources for every claim.

### Completion
1. Post ALL findings with source URLs to the appropriate BBS channel.
2. Call `complete_task` with a summary of what you found.
"""
        else:
            _browsing_instructions = """\
## Instructions

Search the web to answer the task above. Follow your loaded skills
(`web-research`, `tool-usage-policy-browsing`, `task-completion-web`) for
the iterative research workflow, source evaluation, and completion checklist.

**CRITICAL**: You MUST call `web_search` at least once before posting any
findings to the BBS. Do NOT rely on your training knowledge alone — search
and cite actual web sources for every claim.

### Completion
1. Post ALL findings with source URLs to the appropriate BBS channel.
2. Run the completion self-assessment checklist before finishing.
3. Call `complete_task` with a summary of what you found.
"""
        _profile_instructions = {
            "browsing": _browsing_instructions,
            "coding": """\
## Instructions

Write and execute code to accomplish the task above.
After producing results:
1. Post your code and output to the appropriate BBS channel.
2. Call `complete_task` with a summary of what you accomplished.
""",
            "reasoning": """\
## Instructions

Analyse this problem thoroughly using deep chain-of-thought reasoning.
- Read all BBS context carefully before reasoning.
- Use the `reasoning` tool with the full context.
- Check for hidden assumptions, mistakes, and alternative interpretations.
After completing your analysis:
1. Post your reasoning and conclusions to the appropriate BBS channel.
2. Call `complete_task` with a summary of your analysis.
""",
        }
        default_instructions = """\
## Instructions

Complete the task described above. Post your findings to the appropriate \
BBS channel, then call `complete_task` with a summary.
"""
    else:
        _profile_instructions = {
            "browsing": """\
## Instructions

Search the web to answer the task above. Include source URLs for every claim.
After gathering sufficient information:
1. Call `complete_task` with a thorough summary of what you found, including source URLs.
   The orchestrator is automatically notified — do NOT also call `send_message` to leader.
2. Only broadcast via `send_message to='all'` if you found a critical issue that
   affects other agents' active work (e.g., a data quality problem).
""",
            "coding": """\
## Instructions

Write and execute code to accomplish the task above.
After producing results:
1. Call `complete_task` with a thorough summary of what you accomplished, including code and output.
   The orchestrator is automatically notified — do NOT also call `send_message` to leader.
2. Only broadcast via `send_message to='all'` if you found a critical issue that
   affects other agents' active work (e.g., a data quality problem).
""",
            "reasoning": """\
## Instructions

Analyse this problem thoroughly using deep chain-of-thought reasoning.
- Use the `reasoning` tool with the full context.
- Check for hidden assumptions, mistakes, and alternative interpretations.
After completing your analysis:
1. Call `complete_task` with a thorough summary of your analysis.
   The orchestrator is automatically notified — do NOT also call `send_message` to leader.
2. Only broadcast via `send_message to='all'` if you found a critical issue that
   affects other agents' active work (e.g., a data quality problem).
""",
        }
        default_instructions = """\
## Instructions

Complete the task described above, then call `complete_task` with a thorough summary. \
The orchestrator is automatically notified — do NOT also call `send_message` to leader. \
Only broadcast via `send_message to='all'` if you found a critical issue affecting other agents.
"""

    instructions = _profile_instructions.get(profile_name, default_instructions)

    return header + instructions


# ---------------------------------------------------------------------------
# Idle review messages (injected as user messages by SubAgent._idle_check)
# ---------------------------------------------------------------------------

IDLE_REVIEW_MESSAGE_RESEARCH = """\
## Original Question
{question}

## New BBS Messages From Other Agents

{new_messages}

Review these messages. If you disagree with a finding or spot something that \
looks wrong, post to #discussion. \
If everything looks fine, just say "Nothing to flag." \
Do NOT try to duplicate work or verify results yourself.
"""

IDLE_REVIEW_MESSAGE_RESEARCH_ADVERSARIAL = """\
## Original Question
{question}

## New BBS Messages From Other Agents

{new_messages}

## Adversarial Review Protocol

You are reviewing findings from other agents. Your job is to CHALLENGE, not \
rubber-stamp. You do NOT have web_search or web_fetch tools — you cannot run \
your own searches. However, you DO have the `reasoning` tool for deep \
chain-of-thought analysis.

**IMPORTANT**: You may call the `reasoning` tool AT MOST 3 TIMES per review. \
Use it to rigorously evaluate whether the evidence presented actually supports \
the proposed candidate. Do NOT call it more than needed — use it judiciously \
for deep analysis.

1. **Identify** any candidate answer proposed in the messages above.
2. **Audit constraints**: For each candidate, check which constraints from the \
original question are verified with evidence and which are NOT. If a candidate \
matches most but not all constraints, that is a RED FLAG — partial matches are \
the #1 source of wrong answers.
3. **Reason deeply**: Use the `reasoning` tool (but no more than 3) to analyze whether the \
evidence chain is logically sound. Look for gaps, assumptions, or leaps in reasoning.
4. Share your verdict:
   - **CHALLENGE**: Post to #discussion: "[Candidate] \
may be wrong because [constraint X] is unverified/contradicted. Please \
re-verify [specific thing]."
   - **ALTERNATIVE**: Post to #discussion: "Consider \
[alternative entity] which also matches because [reason]."
   - **VERIFIED**: Post to #consensus: "Reviewed [candidate] — all constraints \
verified with evidence."

If there are no candidate answers in the messages, say "No candidates to review." \
Do NOT try to duplicate work or re-run searches yourself. \
If you have already reviewed this candidate, say "Already reviewed — nothing new to add."
"""