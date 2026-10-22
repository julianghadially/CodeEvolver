# Simple Loop
The name of this arm is simple_loop. See specs/research_plan.md.

The harness will be OpenCode (We thought we might switch to Mini-SWE-Agent later, but OpenCode allows us to make stronger causal links around the CodeEvolver Harness. The strength of Mini-SWE Agent is that its the closest to a raw agent coding in a loop). 

## Basic Program
This is a general directional picture of what the basic program would look like. You may decide to refactor, but keep it simple and as clean as possible.

```
initialize_repository(seed_commit)
initialize_budget(max_eval_rows)

while eval_rows_used < max_eval_rows:
    if wall_clock_exhausted() or optimizer_budget_exhausted():
        break

    agent = create_fresh_pinned_agent(
        model=optimizer_model,
        environment=budgeted_environment,
        config=pinned_config,
    )

    agent.run(FIXED_OPTIMIZATION_PROMPT)
    commit_session_end()

    run_allowed = run_constraint_checker()
    if not pass:
        revert_to_parent()          # + empty "Revert iteration N" marker commit

    if too_many_sessions_without_evaluation():
        terminate_as_stalled()
        
final_tree = register_final_worktree()
evaluate_seed_and_final_on_hidden_test(seed_tree, final_tree)
run_integrity_agent()
meta_reflection()
```

Notes: 
- Final tree corresponds to the final git state that the loop ends at. Whether evaluated or not. Must pass the constraint agent of course. Does not include Meta reflection or run integrity agent, which should not be changing code, but may as well grab the state before those execute. Noise from untested evals is fine if it is on the branch. Don't worry about that. 
- Post-loop test evals must not increment max_eval_rows

### Runner / Orchestrator
Orchestrator puts a clean-state benchmark into the runner.

### Initialize repo
Repo is armed with traces, and the loop has a method of collecting open telemetry traces through each eval. I.e., the same level of traces is available. The agent does not receive an inspect_traces tool, but the prompt tells it where the optimizer collector emits the traces (i.e., .codeevolver/traces/)

The simple loop will use one branch across the full run, which should be the codeevolver-{timestamp}-main. There is no protection of the main branch.

### Agent definition
Includes
- Run_eval: 

### Eval
Same run_eval tool. It allows for the sub-sample and the training comparison set using the same exact settings (control). MCP for opencode.

Note: if we change to using mini-SWE-agent, then we need to express the eval as bash. (i.e., run_eval --eval-sample train)

Agent controls:
- when to evaluate
- train subsample versus comparison set
- whether to inspect available traces
- what code changes to make

Experiment controls:
- eligible rows
- rows per call
- sampling policy
- feedback fields
- trace collection
- trace exposure policy
- total evaluation-row budget

The sampling policy for the eval is the same. The first train call per second per session locks the rows, which later calls reuse, and so the should describe this if it doesn't already.

### Agent.run / Prompt
The prompt keeps the run_eval dynamics the same as the other arms, since that's our budget control.

The prompt still needs the additional section for the additional instructions, which contain the constraints. The program section for the program entry point is valuable. For the prompt, the budget note is useful. Let's keep it. The sample size comments are not something to keep.

Wording of the budget note: a single line, `You have used N% of your evaluation budget; R of M rows remain.` — no per-eval-type costs, no "room for ~K more train evals", no before/after endgame warning. Those stay on the shared v3 builder for the treatment arms only. Empty when no budget is configured.

We do need to tell it where to find the traces if those are outside of the repo.

The prompt currently is stale in two places:
1. It says the system will block edits until you have run a baseline. That is removed, and so that sentence should be removed.
2. It tells the agent to use Coding Agent. That should be removed as well. Drop Coding Agent from the simple loop and from the prompt.

### Run constraint checker
Constraint agent: The same constraint agent must fire so that we do not end up with solutions that cheat, as this is decently common. This includes the permissions that keep the deny paths out of reach. Also keeps the no-hardcoding Q-A pairs constraint


The test repos always send additional instructions in their config with a constraint field (the trigger for enabling the gate).

Revert the branch state to the parent and leave an empty `Revert iteration N` marker commit quoting the violated constraint (never the judge's reason); the next session's kickoff message says the previous session was reverted by the constraint checker (see `specs/engine3.md`). Assume Claude Sonnet 5 as a constraint checker. (More reliable and benefit of different model)

### Iteration
Any uncommitted files should be added and committed at commit_session_end. 

`too_many_sessions_without_evaluation` The loop should count consecutive sessions with zero eval rows or with errors and use that to error out. Meanwhile, wall clock time control is controlled by a global timeout for the loop. Use the same timeout as CodeEvolver. Meanwhile, at the orchestrator level, I think it should be able to detect across any of these strategies if there has been a stalled iteration. Essentially, if 4 hours have passed with no iteration progress, it should log a warning. That's it.

### Loop 
The per-session record should include iteration number and {start_sha, end_sha, rows_used, constraint_violation}. Optionally, we can include {subsample_baseline, subsample_best, and comparison_best} if it is low overhead - Would it be less code to just fetch the iteration_{n}_eval.json file to get the sub-sample baseline and sub-sample best and comparison best? 

The iteration history for the simple loop can be boiled down to a status update that reveals the most recent iteration and current budget. The main consumer of this is the orchestrator.

### Optimize config
We need a better definition for this section, but here are some ad hoc comments. 

The caller will pass the same optimized configuration as before. This includes:
- the training set path
- the test set path
- the program path
- the metric input keys
- reflection language model decoding harness will be fixed
- that open code harness model will be passed as well
- max metric calls
- number of threads
- additional instructions
- initial branch
- sub-sample size
- Comparison set size
- etc.  
- Assume enable meta reflection is true. We don't have to capture that one.



### Integrity Agent and Meta Reflection
Ideally, we collect architect traces and then run the same integrity agent somewhere. 

Then we can run the integrity agent to make sure that the implementation is clean, and the meta reflection agent also helps to make sure that the implementation is clean. The meta reflection uses a prompt focused on making improvements to any gaps in functionality.

Meta-reflection's two defenses still apply. Architect traces are a memory artifact; keep (1) traces dir outside the workspace and (2) the _architect.jsonl deny fragment, and keep the research-purity clause in the reflection prompt (otherwise it recommends building the treatment arm every run). Capture gating is enable_meta_reflection OR enable_run_integrity_agent.

Consider whether meta-reflections and integrity agent should be refactored into a meta/ folder. This would serve run all strategies, and so we would have a main meta-reflection and integrity agent and then a separate simple_meta_reflection for simple_loop with its own meta-reflection prompt.

(Note: If using Mini-SWE, Does Mini-SWE allow for its own traces to be emitted separately from the run eval traces?)

## Variable Decisions
| Topic                                      |      Kept? | Revised form                                                                     |
| --------------------------------------------- | ---------: | -------------------------------------------------------------------------------- |
| Prior final state becomes next starting state |        Yes | One persistent linear worktree; no parent selection or branch pool               |
| Evaluation rows are the primary budget        |        Yes | Enforce centrally and atomically (RunEvalBinding._enforce_eval_budget) |
| Equal optimization-data access                |        Yes | Give harness the same total optimization pool available to the CodeEvolver arms (train dataset is now trainval dataset) |
| No separate validation-set spending           |        Yes | harness accepts the train dataset which includes subsample and comparison evaluator mechanisms through the run_eval tool. The optimized_config also passes the holdout testset path.      |
| No `inspect_traces` tool                      |        Yes | harness uses ordinary Bash and evaluator output                                 |
| Held-out final evaluation                     |        Yes | Evaluate the seed and final program outside the optimization process             |
| Fresh session boundaries                      |        Yes | Instantiate a fresh agent after each completed session                  |
| Safety limits                                 |        Yes | Wall-clock, optimizer-token/cost, cancellation, and repeated-failure limits      |
| Holdout protection                            | Absolutely | Test data and evaluation code should be inaccessible from Bash, but that is enforced in the runner, not the loop  |
| Evaluation provenance                         |        Yes | Record the exact Git tree associated with every evaluation                       |

- Last-not-best algorithm: Dropping the commit-regression watchdog means the final tree can be worse than the best tree seen. Ultimately, this is the agent's decision.
- subsample_size is a control across arms (incl. the RAGQA deviation)

Not included. B0 is not the CE harness
* Not included: Architect, coding agent, skills, memory, inspect_traces, select_candidate, close_iteration, CE prompt, CE hooks
* Commit-regression watchdog: This is a CodeEvolver harness element. Not part of a simple loop.
* There is no trainval or valset or any deny path. The Simple Loop should be completely oblivious to deny paths. The Simple Loop should also be oblivious to the holdout data set because the runner is responsible for depriving privileges.
* Bootstrap suppression. stage_project_skills=False, no /s bootstrap, no /update-specs finalization. 
* Reset repo is not used because runs execute in the runner/orchestrator instead (applies to all arms).

## Guidance
This simple loop needs clean definitions and clean code. 

- Do not attempt patch fixes that complicate the program. Instead, document issues.
- Each stage should be clearly defined. 
- To keep the codebase clean, we should consider moving the integrity agent and the meta-reflection agent outside the simple_loop folder. Maybe it's an external service that gets called on the artifacts of the run. For simple loop, please keep documentation at the top of files and functions. Avoid inline comments. Minimize line count per file.
- If there are any checks, there should be a check stage specified at the top level. For example, the constraint checker is required to prevent cheating that is called out at the top level.
- The simple loop is only needed for research purposes. This allows us to define repo-level inputs, if needed.
- Config settings should be defined globally. If we need context, we should define an IterationContext class and keep it simple. 
-  the loop's only obligation is to spawn the agent through the harness path that honors agent_user() and never as privileged

## CodeEvolver components
**Using CodeEvolver Classes:** It is okay to import schemas from the main optimizer that do not bring behavior alongside them, but *bringing in behavior-bearing classes is not okay, barring the exceptions below.* 

Behavior-bearing classes are not allowed except these:
    - Declared controls such as `run_eval` 
    - Controls that do not impact code and offer shared integrity across all runs. `constraint_checker`. While constraint checker is an agent, it does not perform any improvement task and is standardized across all arms. 

Schemas that do not import behavior and are not agent-facing are okay:
- coding_harness_config
- iteration_history.IterationHistory (Invisible to the model)

Shared config settings
- optimizer_loop.TIMEOUT_BUFFER_SECONDS
- optimize_config

## Misc.
- We are using OpenCode, not mini-swe-agent. Ignore all references to mini-swe.

---
## Agent Notes
Agents: Do not edit above the Agent Notes section unless directly requested.

Package: `src/engine/optimizer/simple_loop/` (entry `optimization_entry.py`, method
`simple_loop`).

**Files.** `loop.py` — the stages, in order: `initialize_repository` (fetch, checkout
`initial_branch`, `codeevolver-{ts}-main`, `validate_sandbox_environment`, seed record 0) →
per session: `run_session` (`session.py`: `RunEvalBinding.from_agent_config` + OpenCode harness via
`create_coding_harness`, registry `{"run_eval"}`, `HarnessRequest` with the system prompt in
`raw_provider_metadata`) → `commit_session_end` (`git_add_and_commit`) →
`check_session_constraints` (`constraint_check.py`: `evaluate_change_constraints(start_sha,
end_sha)`; on `fail` → `git reset --hard start_sha` + the revert marker commit, so `close_sha`
is the marker; skipped when the session diff is empty)
→ `session_record` (`records.py`) + `callbacks.update_progress` → `check_stalled` → after the
loop `finalize_on_holdout` (seed + final on `suffix="test"` and an uncounted
`suffix="final_comparison"`; final skipped when `final_sha == seed_sha`) → `set_completed` →
`_run_auditors`. `state.py` (`SimpleLoopState`: `total_num_evals`,
`final_comparison_evals`) is the whole state; `prompt.py`/`prompt.md` the system prompt and
kickoff message. No `select_parent`, child branches, `coding_agent`, `inspect_traces`,
`close_iteration`, watchdogs, `ctx.finalize()` attribution, or deny-path construction.

**Budget.** Enforced by `RunEvalBinding._enforce_eval_budget`; the loop's `while` only checks
between sessions, so the session that crosses the line keeps its one post-budget comparison
allowance. Finalization evals never touch the state counter.

**ArchitectConfig.** `agent_managed_git=False` (MCP bridge exposes exactly `run_eval`),
`stage_project_skills=False` (no skills, `/s`, `/update-specs`), `subsample_timeout` /
`comparison_timeout` threaded from the config,
`capture_architect_traces = enable_meta_reflection OR enable_run_integrity_agent` (trace
capture gate), `additional_deny_paths=list(config.additional_deny_paths or [])`. The harness
also applies its own fence from `evaluator.testset_path` + `_architect.jsonl`
(`opencode/harness.py::_architect_holdout_deny_paths`), but that fence covers the holdout only —
it is NOT a substitute for the client's declared extras. Because simple_loop has no
`coding_agent` tool (the architect session IS the coding agent), passing the extras here is
what enforces `additional_deny_paths` on this arm. Note the residual documented in
`specs/experiment-docs/experiment-fleet.md` (contamination control 0): extras keep the name fence only (they are
not relocated to `/opt/holdout`), and `bash` carries no deny rules on any arm. The agent process
is spawned by `run_architect`, which honors `agent_user()`.

**Per-session record** (from `ctx.eval_records`, no trace-file parsing): `git_branch`,
`parent_sha`=start, `close_sha`=end (the revert marker after a gate revert), `changes_landed`
(False on a gate revert even though the sha moved), `eval_rows_used` (new whitelisted field
in `ITERATION_RECORD_FIELDS` + `IterationHistoryRecord`), `initial_subsample_score`,
`best_subsample_score`, `comparison_best_score`, `score`=best train,
`constraint_violation`, `skipped` (session raised), `subsample_error`; `test_score` on record 0
and the last record; `integrity_*` stamped by `stamp_integrity_verdict`.
**Best train = engine 3's rule:** the max over `train` evals whose `sample_key`
equals the baseline's (`ctx.baseline_sample_key`; fail-open when either is None, e.g. a
ceiling-rejected baseline) — `single` evals and redrawn samples are excluded. **A session the
constraint gate reverted has `score=None`**, so a reverted session's reading can never
become the run's "best train"; `best_subsample_score` / `comparison_best_score` keep
the measured readings beside `constraint_violation`, and `score` — the only field consumed as a
headline (`candidate_scores`, the progress line, the meta-reflection run table) — is nulled.
`prompt.md` documents `eval_sample="single"` (1-row smoke check, spends a row, not a
keep/revert signal) and `_budget_note` names it as spending rows; `single` evals never count
as the session score. **No sha is
attributed** to either score (`best_subsample_sha` / `comparison_best_sha` are not emitted):
the simple loop never reconciles dirty-tree evals, so an eval's `sha` is HEAD at eval time —
the *parent* commit whenever the tree was dirty — so such a field would credit a dirty-tree
reading to the parent commit. The exact
tree of every eval is in `iteration_{N}_evals.jsonl` and in each trace directory's
`provenance.json`.

**Stall / safety.** `check_stalled` raises `SimpleLoopStalledError` after
`MAX_CONSECUTIVE_ARCHITECT_FAILURES` (5) consecutive sessions that raised or spent zero rows;
wall clock = `job_timeout_seconds - TIMEOUT_BUFFER_SECONDS` (same as CE); cancellation via
`callbacks.check_cancelled`. Fleet-level: `runner/entrypoint.py::write_progress` writes
`/artifacts/progress.json` on every poll tick (status.json is end-of-run only) and
`orchestrator/controller.py::check_stalled` logs `STALLED WARNING <run_id>` once per episode
when `(current_iteration, total_metric_calls)` has not moved for `STALL_WARN_SECONDS` (4h).
Arm-agnostic, warning only.

**Post-run auditors** live in `src/engine/optimizer/meta/`: `meta_reflection.py`,
`run_integrity.py` (all arms) and `simple_meta_reflection.py` (the simple-loop
prompt: research-purity clause, inline run table, points at `simple_loop/` source).
`v3/clean_slate.py` stays in v3 (pre-run, abort-capable).

**Config / wiring.** simple_loop consumes each benchmark's `OPTIMIZE_CONFIG_LOCAL` unchanged
(trainval `trainset_path`, `comparison_set_size`, CE `subsample_size`, `additional_instructions`
— the constraint-gate trigger; without it the gate logs that it is DISABLED). Wired in
`local_launcher._ENTRY_SCRIPT_REL`, `orchestrator/matrix.main.yaml` (one of the five arms),
`local_test_app.py simple_loop_e2e`, `tests/functional/test_e2e.py`. Research-only. Tests:
`tests/unit/test_simple_loop.py`, `tests/unit/test_simple_meta_reflection.py`, stall tests in
`tests/unit/test_experiment_fleet.py`.

**Import boundary (the control's credibility).** The simple loop must not *be*
CodeEvolver, so what it imports from `optimizer/v3/` is a narrow allowlist,
enforced by `tests/unit/test_simple_loop_import_boundary.py`: config + controls
(`optimization_config`, `constraint_gate`, `lm_config`, `coding_harness_config`),
the shared eval instrumentation (`run_eval_tool`), and the entry shim (`_entry`).
The per-iteration record schema (`IterationHistory` / `ITERATION_RECORD_FIELDS`)
lives in `src/engine/schemas/iteration_history.py` and is imported from there; the
EE change-summary I/O is `v3/change_summary.py` and is not imported. The
behavior-bearing classes are NOT imported. `IterationContext` and `ArchitectConfig`
are split along the schema/logic seam into `src/engine/schemas/iteration_context.py`
(`BaseIterationContext` + `EvalRecord`: the eval log, subsample lock, budget
snapshot, token usage, and the two append-only `record_*` methods) and
`src/engine/schemas/agent_config.py` (`BaseAgentConfig`: the harness + eval-tool
settings). Engine 3's `IterationContext(BaseIterationContext)` and
`ArchitectConfig(BaseAgentConfig)` extend these with the closing protocol and
its logic (`finalize`, `reconcile_eval_shas`, `score_for_content`, branch
linkage), pinned by the v3 contract tests. `RunEvalBinding` and both coding harnesses type against the base schemas,
so the simple loop constructs `BaseIterationContext` / `BaseAgentConfig` and
shares only instrumentation with the treatment arms. The harness capture flag is
`capture_architect_traces`, set from the public `OptimizationConfig`
knobs `enable_meta_reflection` / `enable_run_integrity_agent`. `utc_run_timestamp`, the two loop constants, and the budget
note are the simple loop's own copies (no shared prompt/loop code with EE).

**Known issues (documented, not patched).** The harness-inherited testset deny fence cannot be
switched off without a harness flag; `status.json` remains end-of-run only (live status comes
from `progress.json`).
