The purpose of the freeform branch (baseline 1 - B1) is to answer two different questions. 
1. Does the CodeEvolver harness with its memory and controls and hooks offer any benefit over a simple loop (B0)?
2. Does exploratory population search add value over continuous improvement without explore.

The name of the freeform branch signifies that it is allowed to act how it normally would, so there was no exploit versus exploration guidance. The prompt stays identical otherwise.

## Git
The cleanest design is to make git allowed but not mentioned to agent.

Before ending the iteration, commit all changes you intend to submit. Leave the repository checked out at the commit representing your final result. The current HEAD at iteration close will be treated as your submitted state.

## Finalize
The loop should register the following states automatically:
- close_sha = git rev-parse HEAD
- current branch/ref, if one exists
- standard full branch metrics
- dirty-state check, ideally requiring clean working tree (auto-commit if not committed)

The final candidate is the branch left at the head of the last closed iteration.. The closed record for freeform does not need to be primary or was primary or promotional evidence or claimed improvement mode. These would not be submitted by the agent and do not need to be registered. However, if the closed record is the easiest way to maintain branch tracking, then what we could do is use primary in the record to describe the branch that was submitted at close, even though the agent didn't provide it. That's a design option up to you.

All branches require the prefix codeevolver- (per the close iteration prompt). If a non codeevolver- branch is emitted, the close process should add the codeevolver- prefix as a rename. 

If the agent submits a detached head, we should create a new branch with a codeevolver- prefix and whatever branch it inherits from. Use that parent branch name and then add -detached-{sha}. so codeevolver-main-detached-{sha}. This is unlikely, but it's most likely on the freeform agent. If you determine that this closing loop attribute is safe to run on all branches, then you may do it, but evolution needs a parent and EE registers its final branch, so unlikely on those branches.

The closed iteration loop mode: you can add a new one for free form, and then improvement_mode should not be surfaced to the agent. It does not receive any exploit versus explore hints.

Constraint-gate:
- If a constraint fires, then we force-revert to parent_sha as usual. Note: we need to make sure our solution is robust. We don't want to lose state based on one bad round.

## Prompt
NOWHERE should the agent ever be guided to exploit or explore. It is freeform. Improvement as normal. Telling the agent to exploit would guide it down a search path that does not pursue big ideas, and that's not the point of this branch.

Asking "was this exploration or exploitation?" injects the concept the arm is supposed to lack. claimed_explore_fraction becomes None, and enable_explore_guard=False so _MUST_EXPLORE_SECTION never appends.

```
Your task is to improve the program in the repository at `{workspace_path}`. 

## Guidance
1. Study {memory_dir}/memory.md and {memory_dir}/iteration_change_summaries.md to learn about past architectures attempted.
2. Run a baseline evaluation on the training dataset using the `run_eval` tool. `run_eval` scores the program on a random {subsample_size}-row sample of the training set. Note, the run_eval tool fixes the training subsample of rows to use for this iteration. (using eval_sample="train")
3. Make your improvement(s) as you see fit. (Do not violate any constraints from the additional instructions below).
4. **Call `run_eval(eval_sample="train")` again** to measure the result. Any call after the baseline automatically re-uses the same sample, so the before/after numbers are directly comparable (a new seed is drawn for each iteration - your work spans one iteration).

You must run the baseline eval, and it is recommended to do this before making edits. The system will force an eval run if you do not run one before your run ends. Each run writes trace files you can inspect with the inspect_traces tool or with Bash / Read / Grep. Traces often contain valuable examples to learn from on how a program executes. Pass a specific, substantive query — the inspector is your eyes into the trace file.

    {traces_dir}/iteration_<N>_parent.jsonl   (baseline eval traces)
    {traces_dir}/iteration_<N>_child.jsonl    (post-change eval traces)

## Making changes

Use `coding_agent(mode="execute", change_request=...)` for substantive changes, or edit files directly and commit with Bash for small edits. Either way, the baseline eval must come first.

Measure before and after - at least once per iteration.

## Directories
You have access to a few directories:
Workspace path: {workspace_path}
Memory directory: {memory_dir}
Traces directory: {traces_dir}
History directory: {history_dir}

## Finishing an iteration

Once your planned improvement(s) are complete along with any follow-on changes you want to make, it is time to finalize the iteration. Call the close_iteration skill and follow those instructions.

{additional_section}

{program_section}
{subsample_note}
{budget_note}

"""
```

Given this prompt has edits in multiple places, define it in its own Architect Exploit prompt as opposed to having an unreadable Frankenstein object shared among ex/ee/evo, but the base prompt is really useful for Explorer Void versus Evo because at least there the swaps are pretty parallel.

### Close Iteration Skill

Before ending the iteration, commit all changes you intend to submit. Leave the repository checked out at the commit representing your final result. The current HEAD at iteration close will be treated as your submitted state.

```
# Close Iteration Skill
Once your planned improvement(s) are complete along with any follow-on changes you want to make, it is time to finalize the iteration. 

1. Make sure the program is committed at the point you want to finalize. If you did not like your changes, `git reset --hard` to where you want to leave the branch. The loop continues improving from wherever you leave HEAD, and auto-commits any dirty branch states.
2. Before exiting, **update memory** at the {memory_dir}/memory.md file with key metrics and learnings (See "Saving to Memory" below).
3. Finally, call the close_iteration tool to finalize the iteration. 

Do not rewrite `{workspace_path}/codeevolver.md` yourself — a post-iteration agent updates it to describe the program on your branch after you close.

## Close Iteration Tool
Call the close_iteration tool to finalize the iteration, and one field:
- `memory_updated`: true / false

## Saving to Memory

Reflect on the iteration and record in `/memory/memory.md` for future agents to benefit.

Document the memories you find the most important. Generally speaking it is useful to share the approach you took, the overall progression on evaluations, observations, and possibly ideas. If there are concerns, note them down and how to avoid.

Keep memories concise and to the point. A future agent can inspect the branch for details. Leave room for other iterations.

Note: **Delete or tidy-up stale content** - Move unhelpful old memories to old_memory.md. The file gets really big after ~15+ iterations! It should never exceed 150k characters

## Notes

- Memory location: add your reflection to `{memory_dir}/memory.md`. memory.md functions as cumulative memory across iterations.
- Git branches: your branch must contain the codeevolver- prefix. DO NOT MODIFY BRANCHES THAT DON'T HAVE THE PREFIX `codeevolver-`. 
```

## Misc.
Remove the Greedy folder entirely as this is a completely different implementation. Implement this alongside the freeform implementation (Or in a second implementation stage if needed).
Across all branches, we should add a hook that, if the close iteration skill is not used, then we recall the agent session, tell it to use the close iteration skill, and resubmit the close. Implement this alongside the freeform implementation.
- Updates skills_to_stage to omit exploration. otherwise on par w/ ee
- Memory template should exclude the macro approaches section.
- The memory template should remove the macro approaches since that is an exploration activity, and it should also remove the candidate pool section. Having the seed branch and seed architecture in there is okay.
- The sub sample note should be updated. Do not provide any guidance on the comparison set usage. We would remove the comment that it is reserved for comparing branches apples to apples.
- Coding agent is a control across all groups. Keep it.
- Prompt drift: we do not need a test to enforce prompt drift. This will be done manually.
- Include the any closing metrics you need like close_SHA. Less variables changing is better.
- Hooks should be the same unless you find any hook that conflicts with the research plan.
- Generally speaking, all variables not mentioned should be controlled unless you find anything that conflicts with the research plan. 
- The freeform agent is allowed to try branching. It is also able to make use of the comparison set, so it is not blocked from using the comparison set. It is able to discover the comparison set through the Run eval tool description. We choose to avoid describing the comparison set in the prompt so that we don't guide it toward spending budget unnecessarily (assuming it runs one primary branch trajectory). 

---
## Agent Notes
Agents: Do not edit above the Agent Notes section unless directly requested.

Package: `src/engine/optimizer/v3/freeform/` (entry
`optimization_entry.py`, method `freeform`; `FREEFORM_STRATEGY` in
`freeform_optimizer.py`). It runs on the shared `v3/optimizer_loop.py`
driver; the delta from EE is entirely in `OptimizerStrategy` seams:

- **Prompt:** `prompts.FREEFORM_ARCHITECT_SYSTEM_PROMPT` — the text above,
  as its own literal (not spliced from `ARCHITECT_SYSTEM_PROMPT_BASE`).
  The literal is `FREEFORM_ARCHITECT_SYSTEM_PROMPT_BASE` with two slots
  (`{git_branching_section}{merge_note}`, empty for this arm), because
  `pareto_evolution` renders the same base with its `## Git Branch
  Background` section (see `specs/engine3.md`).
  `build_freeform_architect_system_prompt` never appends
  `_MUST_EXPLORE_SECTION`; `_subsample_note(..., include_comparison_note=False)`
  drops the comparison-set sentence. Drift vs. EE is checked by hand.
- **Skills:** `skills_to_stage=("close_iteration",)` (no `exploration`; no
  arm stages `prompt_engineering` / `tracing`); `skill_overrides["close_iteration"]` →
  `freeform/skills/close_iteration_replacement.md` (the skill above, plus a
  `{final_iteration_note}` slot). New seam `OptimizerStrategy.
  skill_template_values(ctx)` → `ArchitectConfig.skill_template_values` →
  `stage_codeevolver_skills_to(extra_values=)`; skills are re-staged per
  session, so the note is per-iteration. Trigger
  (`freeform_optimizer.final_iteration_note`): remaining eval rows ≤ 5 ×
  `subsample_size`; if the subsample size is unknown, ≥ 90% of the budget;
  never on an unbudgeted run.
- **Close (`close_iteration_mode="head"`):** the tool is
  `close_iteration(memory_updated)` (`agent_tools.close_iteration_head_tool`,
  schema `_CLOSE_ITERATION_HEAD_SPEC`). `v3/head_close.py::register_head`
  auto-commits a dirty tree, renames an unprefixed branch to
  `codeevolver-<name>` (`-<sha8>` on collision), names a detached HEAD
  `codeevolver-<start-branch-minus-prefix>-detached-<sha8>`, and synthesizes
  `RegisteredBranch{is_primary=True, was_primary=(branch == primary at
  start), promotion_evidence=None}`; `claimed_improvement_mode` is None. It
  runs inside the tool (so checks 3/4/4b/5c run on a real branch) and again
  post-session (`optimizer_loop._head_close`, after the drift commit): HEAD
  at session end is the submitted state, and when the tool was never called
  this is the fallback close (`closed_by_fallback=True`, memory claim from
  the memory.md hash). A git failure degrades to `_fallback_close`.
- **Primary follows HEAD.** No new record fields: the synthesized entry
  drives `resolve_primary` → `CEState.set_primary` → `_final_candidate`, so
  the final program is the HEAD of the last closed iteration.
  `freeform_on_iteration_closed` re-pins `primary_branch_after` to the
  closed branch after the constraint-violation override (the branch was
  reset to the parent's content; the loop continues from HEAD either way).
  Attempted trees stay anchored under `refs/codeevolver/evals/<N>/<i>`.
- **Guard / metrics:** `enable_explore_guard=False`; `improvement_mode` is
  None on every record (`claimed_explore_fraction` reads 0.0 — meaningless
  for this arm, as are the primary-relative `run_metrics`; the lineage graph
  from `derive_iteration_parent` + `branches_seen` is the behavioral record).
- **Memory template:** `build_memory_md_template(method="freeform")` →
  `MEMORY_MD_TEMPLATE_NO_MACRO` (seed branch / architecture, Overall
  Progress, Observations).
- **Close-skill watchdog (all arms):** `IterationArchitect.
  _run_close_skill_watchdog_if_needed` (after the close-missing watchdog,
  before the drift commit) resumes the session once when `close_iteration`
  was called but no `skill`/`Skill` tool call named `close_iteration`.
  Detection uses `raw_provider_metadata["tool_calls"]` (OpenCode's stdout
  parser emits it; the Claude harness emits a names+inputs
  list from its `ToolUseBlock`s), accumulated on
  `IterationArchitect._session_tool_calls` across resumes. Fail-open when a
  harness reports no `tool_calls` (e.g. the test fakes).
- **Wiring.** `freeform` is an arm of `orchestrator/matrix.main.yaml` and is
  registered in `local_launcher._ENTRY_SCRIPT_REL` and
  `tests/functional/test_e2e.py::_ENTRY_SCRIPTS`. `CE_ARM=freeform` →
  `STRATEGY` → the launcher resolves it with no runner change. There is no
  greedy arm.
- **Tests:** `tests/unit/test_freeform_prompt_and_skills.py`,
  `tests/controls/test_freeform_close.py` (real git),
  `tests/controls/test_close_skill_watchdog.py`. E2E:
  `local_test_app.py freeform_e2e` (`scripts/run_tests.sh
  --include-freeform-e2e`, local-only, billable).
- Controls unchanged from EE: coding_agent, inspect_traces, run_eval +
  budget gate, constraint gate (force-revert to parent_sha), analysis agent
  + `codeevolver.md`, no-change / close-missing / revert-check / memory-diff
  watchdogs, clean-slate / run-integrity / meta-reflection auditors.
