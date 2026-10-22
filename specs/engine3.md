# Engine 3

Engine 3 is CodeEvolver's optimizer. One architect agent session runs per iteration against a client repository; the agent reads memory, measures the program with `run_eval`, changes code (directly or through a coding sub-agent), records what it learned in `memory.md`, and ends the iteration with a `close_iteration` tool call. After the session, the loop closes the iteration deterministically from git state: it derives the parent, decides whether the program changed, runs a constraint gate, records a closing record, and anchors every evaluated tree. All arms are budgeted in evaluation rows, so they can be compared at equal cost. Five optimization methods share this machinery. Four of them run on one driver (`src/engine/optimizer/v3/optimizer_loop.py`) and differ only through an `OptimizerStrategy`. The fifth, the simple loop, has its own small loop and reuses the evaluator, the budget gate, the constraint gate and the post-run auditors.

Related specs: `specs/requirements.md` (product requirements), `specs/research_plan.md` (the research matrix and its variable table), `specs/v3_freeform.md`, `specs/v3_simple_loop.md`, `specs/archive-public/v3_asa.md`, `specs/archive-public/comparison_credits.md`, `specs/archive-public/sampling_variance.md`, `specs/overfitting101.md`, and `specs/experiment-docs/` (Docker runner, fleet and local deployment).

---

## 1. Overview

### 1.1 Arms

| `method` | Research role | Who chooses the parent | Who sets the primary / final program | Package |
|---|---|---|---|---|
| `simple_loop` | Baseline 0: a plain coding agent in a loop with one `run_eval` tool | Always the previous session's HEAD (one linear branch) | The last session's HEAD | `src/engine/optimizer/simple_loop/` |
| `freeform` | Baseline 1: the full CodeEvolver harness (memory, skills, coding agent, trace inspector, watchdogs) with no search-strategy guidance | The agent, through git | Follows HEAD: the branch the agent closes on | `v3/freeform/` |
| `ee` | Treatment A: agent-directed population search (exploit + explore) | The agent, through git; the loop derives the parent afterwards | The agent, through `close_iteration(register_branch.is_primary)` | `v3/ee/` |
| `pareto_evolution` | Treatment B: fixed evolutionary control on the freeform prompt | The loop: Pareto-proportional sample or exploit-phase best | The loop: best comparison-set score in the population | `v3/pareto_evolution/` |
| `asa` | Treatment C: EE plus a requirement to traverse the population tree. Implemented, but not part of the paper's results. | The agent, constrained by rule-driven requirements | The agent (as in EE), plus loop-run comparison evals | `v3/asa/` |

The method string is the wire identifier (`OptimizationConfig` entry, launcher dispatch in `src/launchers/local_launcher.py::_ENTRY_SCRIPT_REL`, the `method` field on job and iteration records). An unknown method fails at launch.

`pareto_evolution` uses the freeform system prompt, so freeform → pareto_evolution isolates one variable: fixed evolutionary control (loop-owned parent selection, admission by comparison eval, population-best primary). EE differs from freeform by the explore/exploit guidance and agent-managed population.

### 1.2 Eval-row budgeting (all arms)

- The budget is `max_metric_calls`: the number of evaluation rows the run may spend.
- Counted: every training-subsample eval (`train`, `single`), every comparison-set eval (agent- or loop-initiated), and every `run_variance_estimate` run.
- Not counted: the seed step's environment sanity eval (5 trainset rows) and every finalization eval (held-out test draws and the final comparison readings).
- Provider-error row retries inside the evaluator are not charged to the budget.
- The loop checks the budget between iterations, and `run_eval` enforces it within an iteration (§5.4).

### 1.3 Other stop conditions

- `job_timeout_seconds`. The loop stops `TIMEOUT_BUFFER_SECONDS` (10 hours) before the hard timeout, which leaves time for finalization and the post-run agents.
- Cancellation, polled through the job callbacks.
- `debug_max_iterations`.
- A safety cap of 200 iterations (`DEFAULT_MAX_ITERATIONS`).
- `MAX_CONSECUTIVE_ARCHITECT_FAILURES` (5): this many consecutive architect sessions that raised, with no successful iteration between them, halt the run with an error.

---

## 2. Code map

All paths are under `src/engine/` unless noted.

### 2.1 Driver and strategy

| Module | Role |
|---|---|
| `optimizer/v3/optimizer_loop.py` | `run_optimization` (dataset resolution, seed step, strategy hooks, loop, finalization, result dict, post-run agents) and `_run_optimizer_loop` (per-iteration narrative). Holds `TIMEOUT_BUFFER_SECONDS`, `MAX_CONSECUTIVE_ARCHITECT_FAILURES`, `DEFAULT_MAX_ITERATIONS`. |
| `optimizer/v3/strategy.py` | `OptimizerStrategy` (per-arm policy) and `LoopServices` (workspace, evaluator config, train/comparison ids, comparison timeout, run timestamp, the run's single `rng`, `max_eval_rows`, the `OptimizationConfig`). |
| `optimizer/v3/{ee,freeform,pareto_evolution,asa}/` | One strategy object per arm plus an `optimization_entry.py` shim. |
| `optimizer/v3/_entry.py` | Shared entry body: argparse, `.env` loading, config → kwargs, the `OPTIMIZATION_RESULT:{json}` stdout marker. For OpenCode runs it sets `CODEEVOLVER_TRACE_MAX_LINE_CHARS`. |
| `optimizer/v3/optimization_config.py` | `OptimizationConfig`, the config-JSON contract (§11). |

`OptimizerStrategy` fields (defaults are EE's):

| Field | Meaning |
|---|---|
| `prompt_builder` | System-prompt builder in `prompts.py`. |
| `skills_to_stage`, `skill_overrides`, `skill_template_values` | Which skills are staged, per-skill `SKILL.md` replacements (staged under the canonical directory name), and per-iteration `{placeholder}` values. |
| `enable_explore_guard` | Explore-frequency guard (§3.5). |
| `loop_owns_parent` | The loop picks and checks out the parent in `on_iteration_start`; post-session parent derivation is skipped. |
| `close_iteration_mode` | `register_branch` (EE, ASA), `loop_branch` (pareto_evolution) or `head` (freeform). |
| `on_iteration_start`, `on_seed_complete`, `on_iteration_closed` | Loop hooks. |
| `run_eval_gate`, `extra_tools` | ASA's baseline gate and `branch_comparison` tool. |
| `commit_before_comparison` | `run_eval` auto-commits a dirty tree before a comparison eval (pareto_evolution, ASA). |
| `comparison_ack_note` | Arm-specific comparison acknowledgement text. |
| `comparison_closes_iteration` | After the agent's comparison eval, every further `run_eval` is refused (pareto_evolution). |

### 2.2 Sibling modules in `optimizer/v3/`

| Module | Role |
|---|---|
| `seed.py` | Seed step (§3.1). |
| `iteration_architect.py` | `ArchitectConfig`, `IterationArchitect.run()` (launch the session, watchdogs, drift commit, reconciliation). |
| `architect_tool_registry.py` | Builds the per-iteration tool table; holdout deny-path fragments; `INSPECT_TRACE_TARGETS`. |
| `architect_watchdogs.py` | One-shot post-session watchdogs (§3.6). |
| `architect_hooks.py` | Claude architect holdout guard hook and the `apply_patch` holdout fence. |
| `agent_tools.py` | `coding_agent`, `inspect_traces`, and the three `close_iteration` variants, plus the shared close checks. |
| `run_eval_tool.py` | `RunEvalBinding`, the `run_eval` tool. Owns every effectful eval step. |
| `eval_budget.py` | Pure budget, final-comparison and credit policy plus refusal texts. |
| `eval_filters.py` | Train-subsample filter normalization, matching and locking. |
| `eval_log.py` | Eval ids, `iteration_{N}_evals.jsonl`, row-score lines, sampling notes, `variance` suffixes. |
| `loop_eval.py` | `run_loop_comparison_eval`, used by loop-initiated comparison evals. |
| `variance_estimate.py` | `run_variance_estimate` (§5.6). |
| `iteration_context.py` | `IterationContext`: per-iteration state, eval records, `reconcile_eval_shas`, `finalize`, `comparison_record_for_tree`. |
| `closing.py` | `derive_iteration_parent`, `resolve_primary`, `build_close_record`. |
| `head_close.py` | `register_head` (freeform's HEAD registration). |
| `close_effects.py` | Git side effects of closing: dirty-workspace safety net, `head_close`, `fallback_close`, `force_revert_to_parent`, `revert_to_comparison_point`, `checkout_registered_branch`, `commit_codeevolver_md_if_modified`, `write_attempted_diff`, `anchor_eval_trees`, `anchor_iteration_state`. |
| `program_paths.py` | Program-equivalence predicate (§4.2). |
| `constraint_gate.py` | Constraint gate and revert marker (§7.2). The judge lives in `agent/constraint_checker.py`. |
| `analysis_agent.py`, `change_summary.py`, `codeevolver_md.py` | Per-iteration change summary, the combined summaries file, and `codeevolver.md` generation and parsing. |
| `ce_state.py`, `records.py` | `CEState` (run state, schema version 7) and typed record shapes (`CloseRecord`, `BranchEntry`, `ComparisonEvalEntry`, `RegisteredBranch`). |
| `select_parent.py` | `fork_child_from` and `utc_run_timestamp`. |
| `pareto_front.py` | Instance-front computation and GEPA dominated-candidate pruning (§6). |
| `run_metrics.py` | Run-level exploration metrics and the explore guard. |
| `run_reporting.py` | ce_state payload, `final_candidate`, history stamping, progress callbacks. |
| `finalization.py` | `score_seed_and_final` and the post-run agent wrappers (§8). |
| `clean_slate.py` | Start-of-run clean-slate inspector (§7.3). |
| `architect_trace.py`, `architect_trace_markdown.py` | Architect session capture and rendered digests (§9). |
| `prompts.py` | Every system prompt, memory templates, budget and sampling notes, trace-inspection and analysis-agent prompts. |
| `lm_config.py`, `coding_harness_config.py` | LiteLLM callables (including the constraint judge) and harness selection (`opencode` by default, or `claude`). |
| `skills/` | Canonical skills: `close_iteration`, `exploration`, and `prompt_engineering` / `tracing` (kept in-tree; never staged for an architect). |

### 2.3 Shared outside `v3/`

| Path | Role |
|---|---|
| `schemas/iteration_context.py`, `schemas/agent_config.py`, `schemas/iteration_history.py` | `BaseIterationContext`, `BaseAgentConfig` and `IterationHistory` / `ITERATION_RECORD_FIELDS`. These are arm-agnostic and imported by every arm, including the simple loop. |
| `agent/harnesses/` | Claude Agent SDK and OpenCode harnesses. `architect_tools.py` holds the LLM-facing tool specs. The MCP tool server is named `ce` (`mcp__ce__run_eval` on Claude, `ce_run_eval` on OpenCode). |
| `evaluator/` | `dataset_config.py` (train/test resolution, comparison-set draw), `sandbox_client.py` (`run_eval`), `holdout.py` (test-set draws), `row_scores.py`, `mounted/evaluate.py` (the in-workspace evaluator, row retries). |
| `telemetry/` | `trace_layout.py` (eval trace directories), `trace_json.py` (line-capped JSON), `converter.py` (OpenInference span conversion). |
| `optimizer/meta/` | `meta_reflection.py` (v3 arms), `simple_meta_reflection.py` (simple loop), `run_integrity.py` (all arms). |
| `optimizer/simple_loop/` | Baseline 0 (see `specs/v3_simple_loop.md`). |
| `paths.py` | Per-job directories (`traces_dir`, `memory_dir`, `history_dir`, `architect_traces_dir`, `logs_dir`), `agent_user()`, evolution knobs. |
| `scripts/run_meta_reflection.py`, `scripts/inspect_run_integrity.py` (repo root) | Offline meta-reflection replay and mid-run integrity inspection. |

---

## 3. Iteration lifecycle

### 3.1 Seed step (`seed.py`)

1. If enabled, the clean-slate inspector runs (§7.3). It runs before the run branch exists and before any agent spend, so an abort costs nothing.
2. The loop creates the run-main branch `codeevolver-{ts}-main` from `initial_branch`. This branch is the initial primary.
3. Engine artifacts (`.codeevolver/`, `.codeevolver.lock`, `.claude/skills/`, `.opencode/skills/`, `__pycache__/`, `*.pyc`) are appended to `.git/info/exclude`, never to the client's `.gitignore`. They therefore can never change a content-identity tree.
4. A read-only repo-analysis agent writes `codeevolver.md` (with a fallback body), and the loop commits it.
5. `memory.md` is created from the arm's template (§3.4). `iteration_change_summaries.md` is created with the seed entry.
6. An environment sanity eval runs on 5 trainset rows (uncounted). A broken environment aborts the run here.
7. The loop builds `CEState` (budget counter at 0), writes `{history_dir}/datasets.json` (dataset paths, sizes, hashes, comparison ids), and pre-populates `IterationHistory[0]` as the seed record.

After the seed step, `run_optimization` grants the initial comparison credits (§5.5) and calls `strategy.on_seed_complete`. pareto_evolution and ASA run a counted seed comparison eval there.

### 3.2 Per-iteration order (`_run_optimizer_loop`)

1. Budget, timeout and cancellation check.
2. Safety net: commit (and push) any drift left in the workspace (`cleanup_dirty_workspace_if_needed`).
3. Explore guard (EE and ASA only).
4. Grant per-iteration comparison credits and draw this iteration's 2-digit acknowledgement code from the run rng.
5. Build a locked `IterationContext`: budget snapshot, primary at start, memory.md hash, variance estimate, and whether the previous iteration was constraint-reverted.
6. `strategy.on_iteration_start`. pareto_evolution forks its child branch here; ASA computes requirements.
7. Snapshot `start_branch` / `start_sha`, build `ArchitectConfig`, and run `IterationArchitect.run()` (§3.3).
8. Deterministic close: `head_close` (freeform) or `fallback_close` if the agent never called the tool. The loop then resolves `close_sha` and `close_tree` and records every `codeevolver-*` branch in `branches_seen`.
9. Reconcile eval shas (§5.1). Derive the parent (§4.3), or mark it `preselected` when the loop owns the parent.
10. `context.finalize()` (§4.2). `skipped` is True only when no branch was registered.
11. Constraint gate (§7.2). On a violation, the branch is force-reverted to the parent.
12. Anchor evaluated trees (`refs/codeevolver/evals/<N>/<i>`).
13. `build_close_record`, then `strategy.on_iteration_closed`. This hook may run a counted comparison eval, and it decides admission and the primary for pareto_evolution and the primary for freeform.
14. For any non-skipped iteration, check out the registered branch, run the analysis agent, and commit its `codeevolver.md` update. If that commit moved the tip, `close_sha` is re-resolved.
15. Stamp the history entry, `state.record_iteration` (branch registry and primary transition), anchor `refs/codeevolver/iterations/<N>` to `close_sha`, log, render the architect trace markdown (§9), push the registered branch (only `codeevolver-*` branches can be pushed), and post progress.

If the architect raises, the iteration is recorded as a skipped stub (`skipped=True`, `changes_landed=False`, no `close_sha`). Its evaluated trees are still anchored, and its trace is still rendered.

### 3.3 The architect session

`IterationArchitect.run()` launches one harness session with the arm's system prompt, the tool registry, staged skills, and the harness's own file and shell tools (Read / Grep / Glob / Edit / Write / Bash). After the session exits, it runs, in order:

1. Pre-commit watchdogs: `NO_CHANGE`, `CLOSE_MISSING`, `CLOSE_SKILL`.
2. Workstation-committed check: auto-commit any uncommitted work (`iter N: uncommitted work at close (auto-committed)`).
3. Eval-sha reconciliation.
4. Post-reconcile watchdogs: `REVERT_CHECK`, `MEMORY_DIFF`.

Each watchdog fires at most once per iteration. A firing watchdog resumes the same session with a prompt, and later checks re-read live state.

**Tools** (built per iteration by `architect_tool_registry.build_tool_registry`):

| Tool | Arms | Notes |
|---|---|---|
| `run_eval(eval_sample, filter?, acknowledge_message?)` | all | §5. `eval_sample` is `train`, `single` or `comparison`. |
| `coding_agent(mode, change_request)` | v3 arms | Spawns the coding sub-agent (50 turns) with the holdout deny paths. Its changes are committed (`Code mutation`). `mode="tracing"` is not implemented and returns an error. |
| `inspect_traces(query, trace)` | v3 arms | Read-only sub-agent (15 turns) over one of this iteration's eval trace directories. `trace` is `parent` (baseline train eval), `child` (latest post-change train eval) or `comparison` (latest comparison eval). A missing directory returns a structured error without spawning a session. |
| `close_iteration(...)` | v3 arms | §4.1. The description begins with "Load the `close_iteration` skill BEFORE calling this tool". |
| `run_variance_estimate()` | v3 arms | Registered only while the run has no estimate (§5.6). |
| `branch_comparison` | `asa` | Read-only lineage comparison table (`specs/archive-public/v3_asa.md`). |

The agent may also edit files and run git directly. Edits made before the baseline are allowed. The per-iteration baseline requirement is enforced at close (check 5c), not at the tool layer, because a tool-layer ordering rule can only be applied by pattern-matching edit tools, leaks through Bash, and adds friction that varies by model.

**Skills.** Each arm stages an explicit list; the only inter-arm skill difference is `exploration`:

| Arm | Staged | Overrides |
|---|---|---|
| `ee` | `close_iteration`, `exploration` | none |
| `asa` | `close_iteration`, `exploration` | both replaced (`asa/skills/`) |
| `freeform` | `close_iteration` | replaced; renders `{final_iteration_note}` when the remaining rows are ≤ 5 subsamples |
| `pareto_evolution` | `close_iteration` | replaced (the comparison eval is the close step) |

Skill templates are rendered at staging time (`{memory_dir}`, `{history_dir}`, `{traces_dir}`, `{workspace_path}`, plus strategy values).

**System prompts** (`prompts.py`):
- EE and ASA use `ARCHITECT_SYSTEM_PROMPT_BASE`: explore/exploit guidance, the `## Git Branching Guidance / Candidate Pool` section (primary branch named), and `## THIS ITERATION MUST EXPLORE` when the guard fires. ASA appends its rule sections.
- freeform uses `FREEFORM_ARCHITECT_SYSTEM_PROMPT`: numbered measure-change-measure guidance with no explore/exploit wording and no comparison-set sentence.
- pareto_evolution uses the freeform base plus `## Git Branch Background` ("you are on `{child_branch}` … do not switch branches") and, only in exploit-phase iterations, `EVO_MERGE_NOTE` ("Combine Approaches").
- Every prompt carries the client's `additional_instructions`, the program entry point, `### Sampling Guidance` (including the variance paragraph, §5.6), and `### Eval-row budget` (rows remaining of max at iteration start, per-eval-type costs, and a warning when fewer than `2 * subsample_size` rows remain).
- When the previous iteration was reverted by the constraint gate, the kickoff message appends `PREVIOUS_ITERATION_REVERTED_NOTE` (§7.2).

### 3.4 Memory and shared documents

| File | Location | Owner | Notes |
|---|---|---|---|
| `memory.md` | `{memory_dir}` (outside the repo) | The architect | Written directly at close per the skill. Because it is outside the repo, memory updates never change `close_tree`. |
| `old_memory.md` | `{memory_dir}` | The architect | Overflow for stale memory. The skill asks that `memory.md` stay under 150k characters; nothing enforces this. |
| `iteration_{N}_change_summary.md`, `iteration_change_summaries.md` | `{memory_dir}` | Analysis agent / loop | Per-iteration summary and the combined file the architect reads. The combined file is regenerated after every iteration. |
| `codeevolver.md` | Repo root | Analysis agent | Architecture description (`## ARCHITECTURE TITLE/SUMMARY/DESCRIPTION`). The architect is told not to rewrite it. |

Memory templates: `ee` and `asa` get `MEMORY_MD_TEMPLATE` (Overall Progress, Macro Approaches, Observations, plus an agent-managed `## Candidate Pool`). `freeform` and `pareto_evolution` get `MEMORY_MD_TEMPLATE_NO_MACRO` (Overall Progress, Observations). pareto_evolution's population is loop-internal, so its memory has no pool section.

**Analysis agent** (`analysis_agent.py`). Runs after every non-skipped iteration, whether or not a change landed. It receives the iteration's facts inline (branch, shas, scores, change requests, eval series) and writes `iteration_{N}_change_summary.md`. Iterations where nothing landed are tagged `[REVERTED — nothing landed]`, and the agent receives `{history_dir}/iteration_{N}_attempted.diff` (parent tree → best evaluated tree) because there may be no commit to diff. On iterations where a change landed, it rewrites `codeevolver.md` in place. The loop commits only `codeevolver.md` (`iter N: update codeevolver.md`), discards stray edits to any other file, and warns when a landing iteration left the file untouched. The agent is read-only apart from these files and is fenced from the holdout.

### 3.5 Explore guard (EE, ASA)

`run_metrics.explore_guard_should_fire`: once at least 4 iterations have closed (all non-skipped records, including reverted attempts), the guard fires if the claimed exploration share (`claimed_improvement_mode == "exploration"`) is below 30%. When it fires, the prompt carries `## THIS ITERATION MUST EXPLORE`. `derive_run_metrics` also reports `branches_explored`, `iterations_closed` / `iterations_landed` / `land_rate`, `iterations_on_non_primary`, `efforts_on_non_primary_lineages`, and the claimed and derived explore fractions. On freeform and pareto_evolution, `claimed_improvement_mode` is always None.

### 3.6 Watchdogs (`architect_watchdogs.py`)

| Watchdog | Fires when | Resume prompt asks for |
|---|---|---|
| `NO_CHANGE` | No `close_iteration`, no `coding_agent` call, clean tree, HEAD still at `start_sha` | Make the change, or record the decision not to and close explicitly |
| `CLOSE_MISSING` | Session ended without `close_iteration` | Load the skill, commit, write memory, close |
| `CLOSE_SKILL` | `close_iteration` was called but the harness-reported tool calls contain no skill load naming `close_iteration` (fails open when the harness reports no tool calls) | Load the skill, follow it, re-call the tool (last call wins) |
| `REVERT_CHECK` | No comparison eval ran this iteration, at least 2 comparable train evals exist (`single` excluded), the closing content's train score (matched by sha or program-equivalent tree) is below the iteration's best by more than the run-to-run std (strict before a variance estimate exists), and the best is not the closing content | Leave the branch in the best state; revert in git if desired; re-close. Advisory only: no forced revert. |
| `MEMORY_DIFF` | `memory.md` hash unchanged while real work happened (at least 2 `train` evals, or the registered branch moved) | Write the reflection and re-close with `memory_updated=true` |

If a session still has not closed after the resume, the loop's deterministic close takes over (§4.1).

---

## 4. `close_iteration` and closing semantics

### 4.1 Tool variants

| Mode (arms) | LLM-facing signature | What it registers |
|---|---|---|
| `register_branch` (ee, asa) | `close_iteration(register_branch, improvement_mode, memory_updated)` | `register_branch` is a flat object `{"branch", "was_primary", "is_primary", "promotion_evidence"}`. Lists are rejected; a legacy single-key nested object is unwrapped. |
| `loop_branch` (pareto_evolution) | `close_iteration(memory_updated)` | The loop-forked `ctx.child_branch`. Refused if HEAD is on any other branch. Never primary. |
| `head` (freeform) | `close_iteration(memory_updated)` | `register_head`: auto-commits a dirty tree, renames a branch lacking the `codeevolver-` prefix, names a detached HEAD `codeevolver-<parent>-detached-<sha8>`, and registers it with `is_primary=True`. |

The tool is re-callable; the last successful call wins. Validation is synchronous and raises `CloseIterationError` (a structured tool error the model can fix):

1. Shape (`register_branch` only). The branch must be a non-empty string starting with `codeevolver-`.
2. Check 3: the branch exists.
3. Check 4: the working tree is clean.
4. Check 4b: eval shas are reconciled onto their commits (§5.1).
5. Check 5c: if HEAD moved from `start_sha`, at least one non-`single` train eval ran this iteration. This check is waived when no train eval is reachable: fewer than `2 * subsample_size` rows remain (the same threshold as the prompt's budget note), or the pareto_evolution comparison latch is engaged. A deliberate no-op (HEAD unmoved) is exempt.
6. Check 6 (`register_branch` only): `improvement_mode` is `exploration` or `exploitation`.
7. Check 7 (`register_branch` only): promoting a new branch to primary requires `promotion_evidence`.

There is deliberately no check that the closing commit itself was evaluated. A post-eval formatter run or doc edit would otherwise force a redundant eval. The close record never claims the closing commit was measured, the constraint gate diffs commits rather than evals, and finalization measures the final program regardless.

**Session ended without a close.** For `register_branch` and `loop_branch`, `fallback_close` registers `ctx.child_branch` if the loop owns it. Otherwise it registers the current `codeevolver-` branch, then the start branch, then the primary. It carries the primary forward (`is_primary = was_primary`), derives `memory_updated` from the memory.md hash, and sets `closed_by_fallback`. For `head`, `head_close` always re-runs `register_head` after the session, so the submitted state is HEAD at session end, and it doubles as the fallback close. A half-finished branch is therefore inherited by the next iteration rather than lost.

### 4.2 Content-identity closing

An iteration's end state is `close_sha`, the registered branch's tip after the session (and after the loop's `codeevolver.md` commit). Its identity is that commit's tree, `close_tree`.

- **Program equivalence** (`program_paths.py`): two trees are program-equivalent when they differ only in `codeevolver.md`, `.gitignore` or root-level `README*.md` / `CHANGELOG*.md` / `CONTRIBUTING*.md` / `LICENSE*`. Nested markdown is program content, and so is any other root markdown (a program may read `PROMPT.md` at runtime). The list is fixed and not client-configurable. When git cannot answer, the trees are treated as different.
- **`changes_landed`**: the parent tree and the close tree are not program-equivalent. If a tree is unreadable, the loop falls back to sha inequality.
- **`skipped`**: the iteration produced nothing (the architect raised, or no branch was registered). An iteration that tested a change and reverted has `skipped=False, changes_landed=False`. It counts in run metrics, is a valid parent, and gets a change summary.
- **Progress readings, not head certificates.** `finalize()` attributes no score to the closing content. It records:
  - `subsample_best` (+ `subsample_best_sha`): the best train eval on the baseline's locked sample (`sample_key` match; `single` and redrawn samples excluded);
  - `comparison_best` (+ `_sha`): the best comparison eval on the registered branch.

  The `*_sha` fields are the holding commit or the `refs/codeevolver/evals/<N>/<i>` anchor when the tree was never committed. The history `score` equals `subsample_best`.
- `subsample_baseline` is the first accepted train eval. `baseline_ran_on_modified_tree` and `baseline_dirty_paths` record when that baseline measured the parent plus edits already made (observability only).

**Durable end states.** A recorded sha can become unreachable when a later iteration resets its branch, after which `git gc` would prune it. The loop therefore writes `refs/codeevolver/iterations/<N>` → `close_sha` and wraps every evaluated tree (including never-committed ones) in a commit under `refs/codeevolver/evals/<N>/<i>`. `refs/codeevolver/*` is outside `refs/heads/*`, so it is neither pushed nor shown in the architect's branch listing.

### 4.3 Parent derivation (`closing.derive_iteration_parent`)

When the agent owns the parent (EE, ASA, freeform), past records are walked from newest to oldest, ignoring only error stubs with no `close_sha`:

1. The first past `close_sha` that equals, or is an ancestor of, this `close_sha` is the parent (`via="ancestor"`).
2. Otherwise, the first non-null `merge-base(close_sha, past_close_sha)` is the parent sha, attributed to that iteration (`via="merge_base"`). For a fresh fork from an old commit, this attributes the parent to the most recent iteration with a common ancestor; the fork point is the more precise answer.
3. Otherwise, the seed (`via="seed"`).

Reverted iterations are not filtered out, because their tip is what the next iteration starts from. `parent_is_primary` compares the parent branch with the primary at iteration start. `new_lineage` is true when the registered branch has never been seen before. pareto_evolution records `parent_derivation="preselected"` and `new_lineage=True`.

### 4.4 Branch registry and primary

- `CEState.branches` holds one `BranchEntry` per branch that appeared in a closing record (`last_sha` = latest `close_sha`, created and last iteration). `CEState.primary_branch` is always set; it starts as the run-main branch.
- `resolve_primary`: `is_primary=True` makes the registered branch primary. Demoting the current primary without promoting a replacement keeps the prior primary and sets `primary_fallback=True`. The run is never left without a primary.
- Per arm:
  - EE and ASA: the registration decides the primary.
  - freeform: `on_iteration_closed` makes the closed branch primary, including after a constraint revert.
  - pareto_evolution: the primary is the population's best comparison score after every iteration.
  - A constraint violation keeps the start-of-iteration primary on the agent-directed arms.
- The run's final program (`run_reporting.final_candidate`) is the primary branch's `last_sha`.

### 4.5 The close record (`records.CloseRecord`)

`build_close_record` produces one record per iteration; `CEState.record_iteration` appends it and applies the branch upsert and primary transition. Error stubs carry only `iteration`, `skipped`, `changes_landed` and `error`.

| Group | Fields |
|---|---|
| Identity | `iteration`, `skipped`, `changes_landed`, `branch`, `close_sha`, `close_tree`, `branches_seen`, `registered_branch` |
| Parent | `parent_sha`, `parent_tree`, `parent_branch`, `parent_iteration`, `parent_is_primary`, `parent_derivation` (`ancestor` / `merge_base` / `seed` / `preselected`), `new_lineage` |
| Readings | `subsample_baseline`, `subsample_best`, `subsample_best_sha`, `comparison_best`, `comparison_best_sha`, `comparison_scores` (every comparison eval this iteration, without row scores), `baseline_ran_on_modified_tree`, `baseline_dirty_paths`, `variance_estimate` (only on the iteration that took it) |
| Claims and primary | `claimed_improvement_mode`, `memory_updated`, `primary_branch_at_start`, `primary_branch_after`, `primary_fallback`, `promotion_evidence`, `was_primary`, `is_primary` (the outcome of the transition, not the claim) |
| Process | `closed_by_fallback`, `watchdog_fired`, `constraint_violation`, `comparison_credits_after` |
| pareto_evolution | `candidate_admitted`, `admission_reason`, `selected_parent_iteration`, `selection_policy`, `comparison_set_score`, `comparison_state`, `comparison_eval_id_at_close`, `comparison_score_at_close` |
| asa | `asa_*` keys (`specs/archive-public/v3_asa.md`) |

A new record field must also be added to `ITERATION_RECORD_FIELDS` and, when it is copied onto history, to `CLOSE_RECORD_HISTORY_KEYS`. `CEState` carries a schema version; a state written under a different version is refused rather than migrated.

---

## 5. Evaluation

### 5.1 Eval identity is the git tree

The evaluator imports client code from the working tree, so an eval of uncommitted edits would be stamped with the parent's sha if HEAD were used as its identity. Every eval record therefore carries:

- `tree`: `subprocess_service.git_tree_signature`, a tree hash computed through a throwaway index, which does not modify the repo. It equals the committed tree hash for the same content.
- `sha_at_eval`: HEAD at eval time (audit trail).
- `committed`: whether HEAD's tree equals the evaluated tree.

`IterationContext.reconcile_eval_shas` re-points each record's `sha` onto the newest commit (since `start_sha`, on the registered branch) whose tree matches. Newest-wins is deliberate: for content that reappears after a revert, it picks the commit the agent is standing on. Reconciliation runs at every dirty→committed boundary: inside `close_iteration` (check 4b), after the post-session drift commit, and in the loop once `close_sha` is known. Elsewhere, ancestry-bearing shas stay canonical: parent derivation, the constraint gate's `git diff`, and revert suggestions all use real commits.

Before a comparison eval on pareto_evolution and ASA, and before a variance estimate, `run_eval` auto-commits a dirty tree on a named branch. The response then carries `auto_committed: true` and an explanatory note.

### 5.2 Train subsample, `single`, and the comparison set

- **`train`**: the first call draws `subsample_size` rows uniformly from the trainset (or a filter-narrowed pool) using the run rng and locks them for the iteration. Later `train` calls reuse the same rows, so before/after deltas are fair. Trace suffix: `parent` for the baseline, `child` afterwards.
  - Optional `filter`: column → scalar, or `{op: value}` with `eq` / `neq` / `in` / `contains`. Columns are ANDed. The filter applies to the first train call and is locked after that; a different filter on a later call raises an error.
  - Ceiling handling: if the baseline scores ≥ `max_score`, the loop redraws up to 3 times (each redraw requires a pool of at least 1.5× `subsample_size`, or 2× for the third), then falls back once to the full filtered pool. If the result is still at ceiling, the response returns `baseline_accepted: false`, the lock is cleared, and the next train call becomes the baseline.
- **`single`**: the first trainset row (1 row, counted). It is a smoke check and never counts as a baseline, a best reading, a revert-check reading or a check-5c satisfier.
- **`comparison`**: the fixed comparison set, `comparison_set_size` rows drawn once from the trainset with the constant seed 42 (`DatasetConfig`). The seed is constant so that the set is identical across runs of the same dataset. The set overlaps the training data by design. Engine 3 has no validation set, so all labelled non-test data serves both training and selection, and the comparison set exists only to compare branches on the same rows. When `comparison_set_size` is unset or 0, comparison evals are refused; pareto_evolution requires a comparison set. Every comparison eval appends a line (`ts, iteration, branch, sha, tree, score, n_rows, eval_id, row_scores`) to `{history_dir}/comparison_set_history.jsonl`, which the agent may read.

The held-out test set is never reachable from `run_eval`.

**Response** (summary only; per-row data stays on disk):
- Always: `eval_id`, `eval_sample`, `suffix`, `sha`, `tree`, `primary_score`, `scores_by_metric`, `n_rows`, `zero_or_failed_count` (rows that scored 0 or failed, which on a low-scoring task is mostly wrong answers), `infra_failed` (failures matching the provider-error retry classifier), `score_reliable` (`infra_failed == 0`), `trace_dir`, `trace_index_file`, `trace_pretty_file`, `evals_log_file`, `sampling_note`, and the budget fields.
- When applicable: `row_ids` (filtered train evals only), `baseline_accepted` / `baseline_note`, `gate_note` (ASA), `auto_committed`, `comparison_credits_remaining`, and `comparison_history_path` on comparison evals.

### 5.3 Eval logs and per-row scores

- `iteration_{N}_evals.jsonl`, written to both `{traces_dir}` and `{history_dir}`, holds one line per eval: `eval_id` (12 hex), `ts`, `iteration`, `eval_sample`, `suffix`, `n_rows`, `sample_key`, `filter`, `score`, `scores_by_metric`, `branch`, `sha_at_eval`, `tree`, `committed`, `auto_committed`, `by_loop`, and `credits_balance_before` on comparison evals.
- `{history_dir}/row_scores.jsonl` holds per-row scores and failed rows for every eval of every split, including finalization draws and final comparison readings.
- `provenance.json` is written into each agent-visible trace directory with the same payload. It lets the agent tell which content a directory describes, since directories are rewritten per eval (§5.7).

### 5.4 `run_eval` hard-enforces the eval budget

The loop only regains control between sessions, so the budget gate lives in `run_eval` (`eval_budget.enforce_eval_budget`). It reads the live `state.total_num_evals`, not the start-of-iteration snapshot.

- **Under budget**: any eval is allowed, however large. An eval is judged by the budget when it starts, so the last one may overshoot.
- **At or over budget**: `train`, `single` and variance runs are refused with a `RunEvalError` that states the counts and what is still allowed.
- **Final comparison allowance**: exactly one comparison eval remains available after the budget is spent (`FINAL_COMPARISON_ALLOWANCE = 1`, tracked on `CEState.final_comparison_evals`, marked spent only after the eval returns a score). It lets the agent check the branch it will leave as primary. Runs with no comparison set have no allowance.
- **Budget fields in every response**: `eval_rows_remaining` and `budget_note` (`"40 rows remaining."`). Once rows are spent, the note says how many final comparison runs remain. The allowance is not mentioned while rows remain, so it cannot read as a spendable extra.
- The rule is identical on every deployment; the engine never branches on research mode.

Loop-initiated comparison evals (`run_comparison_for_loop`) bypass the agent-facing refusal and never consume the allowance. They are still counted.

### 5.5 Comparison-set credits

The comparison set costs several training subsamples per run. Unpaced, it can consume most of a run's budget and starve the iteration count that population arms need, so comparison runs are paced by a credit ledger. Full design: `specs/archive-public/comparison_credits.md`.

- **Grants**: `comparison_credits_initial` (default 1) before the seed hook, and `comparison_credits_per_iteration` (default 1) at each iteration start. Setting the per-iteration rate to `None`, or having no comparison set, disables the ledger and the acknowledgement. There is no cap, so banking credits is legitimate.
- **Agent-initiated comparison evals** are refused at balance ≤ 0 before any rows are spent. Otherwise they debit 1 after the eval returns.
- **Loop-initiated comparison evals** (pareto_evolution fallback, ASA seed / promotion / stale primary) debit 1 and may overdraw; a fixed rule is never blocked.
- The post-budget final comparison is exempt from the balance check and the debit. Finalization evals are never counted.
- **Acknowledgement**: each iteration has a fresh 2-digit code drawn from the run rng. The first comparison call of an iteration without `acknowledge_message=<code>` returns a structured non-error refusal (`ok: false, refused: "acknowledgement_required"`) with the arm's note, the code, the balance and the rows remaining; no rows are spent. A correct code latches for the rest of the iteration. Every comparison eval needs the acknowledgement, including the final one.
- Arm notes: the standard note (cost framing, credit schedule) for EE, freeform and simple_loop; `EVO_COMPARISON_ACK_NOTE` for pareto_evolution (the comparison is the close step, one run, at the end, later changes are reverted); `ASA_COMPARISON_ACK_NOTE` for ASA.
- Records carry `comparison_credits_after`; the balance lives on `CEState.comparison_credits_balance`.

### 5.6 `run_variance_estimate`

Agents on every arm eventually need to know how noisy the training subsample is, and they otherwise buy that knowledge late by re-measuring the same tree. This tool provides one principled reading per run. Full design: `specs/archive-public/sampling_variance.md`.

- Available on every v3 arm (not the simple loop) until the run has an estimate. A second call in the same iteration is a structured refusal that spends nothing. On pareto_evolution it is refused after the iteration's comparison eval.
- It auto-commits, draws one fresh random `subsample_size` sample from the full trainset (not the locked subsample), and evaluates the current program on it `variance_estimate_samples` times (default 5). Each run is counted and judged against the budget when it starts. If at least 2 runs complete, the estimate is recorded; otherwise it is not.
- It returns the per-run scores, `within_item_variance` = `mean_i(s_i²)` (with R−1 denominators), and `run_to_run_std` = `sqrt(within_item_variance / N)` over rows scored in every run.
- The runs are recorded as `eval_sample="variance"` (suffixes `variance`, `variance_2`, …), so they never become a baseline, `subsample_best`, a revert-check reading, promotion evidence, a pareto admission input or a check-5c satisfier.
- The result is stored on `CEState.variance_estimate` and stamped on the record of the iteration that took it. It is rendered into every later prompt's Sampling Guidance; before it exists, that paragraph recommends calling the tool at the end of iterations 1–3 rather than on the seed. It also becomes the revert-check tolerance. If all runs produced byte-identical outputs, a warning says a response cache is probably active.

### 5.7 Trace directories

Each eval writes `{traces_dir}/iteration_{N}_{suffix}/` (`telemetry/trace_layout.py`):

```
index.jsonl                 one short line per example (example_id, score, failed, feedback, error_summary, row_file)
index_pretty_print.txt      first 40 rows, lines capped at 2000 chars
rows/{example_id}/row.json  everything but span bodies, plus a `spans` table
rows/{example_id}/span_NNN.json   one converted span each
provenance.json             the eval's log line (agent-visible evals only)
eval_stderr.log             the eval subprocess's stderr, when non-empty (last 2 MB)
```

- Directories are built in `{dir}.tmp` and atomically replaced, so a directory is rewritten by every eval with the same suffix.
- Readers go through `trace_layout.read_index` / `iter_rows` / `load_row`. EVAL_RESULT and the `run_eval` response carry `trace_dir` and `trace_index_file`.
- Held-out draws use suffixes `holdout`, `holdout_2`, …, so every held-out directory contains the fence fragment `_holdout`.

**OpenCode line cap.** The agents read traces with their harness's file tools, and both harnesses have hard read limits:
- OpenCode's `read` and `grep` truncate every line at 2,000 characters. This is hardcoded (`MAX_LINE_LENGTH` in OpenCode's `tool/read.ts`) and not configurable. A single `read` also returns at most ~50 KB.
- Claude Code's `Read` errors above 25,000 tokens per call.

A trace that stores a row on one line, or a 20 KB prompt as one JSON string, is therefore unreadable past its first 2,000 characters. Every trace file is pretty-printed through `telemetry/trace_json.py::dump_trace_json` and split per span, which keeps files small. Pretty-printing alone does not shorten a long string, so on OpenCode runs `v3/_entry.py::run_entry` sets `CODEEVOLVER_TRACE_MAX_LINE_CHARS` to `paths.OPENCODE_TRACE_MAX_LINE_CHARS` (1,900). The eval subprocess inherits it, and any string whose JSON-escaped width exceeds it is written as `{"__chunked_string__": [chunk, …]}` (`split_string` prefers to break on whitespace). Readers rejoin chunks with `join_chunked_strings` / `load_trace_json`. Claude runs write plain JSON. The architect-trace markdown (§9) applies the same rule with `MD_MAX_LINE_CHARS` (1,800), and `index_pretty_print.txt` truncates its lines at 2,000.

**Telemetry converter** (`telemetry/converter.py`). The evaluator captures the client program's OpenTelemetry spans, which follow OpenInference conventions, and converts each span into one trace entry:
- The flattened `llm.input_messages.<i>.message.{role,content}` attributes are rebuilt into `inputs["messages"]`, and `llm.output_messages.*` into `output`.
- Everything promoted that way (`input.value`, `output.value`, the flattened keys, `gen_ai.*`) is dropped from `metadata`. Without this, every LM prompt is stored three times and metadata dominates the row.
- Spans from CodeEvolver's own `ce.*` decorators pass through unchanged.

The client is responsible for instrumentation: the program must emit spans (for example through `DSPyInstrumentor` plus a span on its retriever), or its traces hold only a root span.

---

## 6. Pareto evolution (`pareto_evolution`)

The loop owns candidate selection, retention and the primary. The agent works on the branch it is handed and decides whether to measure its result on the comparison set.

**Population** (`pareto_evolution/population.py`). A loop-internal `{history_dir}/population.json` holds iteration-keyed `Candidate`s: `branch`, `sha` (the recorded `close_sha`), `tree`, `parent_iteration`, per-row comparison scores, `comparison_set_score`, and informational `train_baseline` / `train_best`. It is never agent-facing. `on_seed_complete` runs one counted comparison eval on the seed and adds candidate 0. `get_best_program()` returns the candidate with the highest `comparison_set_score`; ties go to the lowest iteration.

**Parent selection** (`select_candidate.evo_select_parent`, run in `on_iteration_start`):
1. One draw decides the exploit phase, with probability `p = min(0.5, 0.5·(1 + tanh(k·(budget_fraction − threshold))))`. `budget_fraction` is the share of `max_metric_calls` spent, `k` = `CODEEVOLVER_EVO_EXPLOIT_K` (default 7.0) and `threshold` = `CODEEVOLVER_EVO_EXPLOIT_THRESHOLD` (default 0.75). Without a budget, `p = 0`.
2. In the exploit phase, the parent is the best candidate, and the same flag adds `EVO_MERGE_NOTE` to the prompt, so the agent ports ideas from other branches with git. There is no crossover operator.
3. Otherwise, the parent is drawn Pareto-proportionally: candidates are weighted by the number of comparison rows they win on the pruned instance front. The fallbacks are a uniform pick among frontier members, then the best candidate, then the seed.
4. `select_parent.fork_child_from` checks out the candidate's recorded `close_sha` (not its branch tip, which may have moved) and forks `codeevolver-{ts}-iter{N}`, resetting it if it already exists. The record carries `selected_parent_iteration` and `selection_policy` (`best` or `pareto_sample`).

**Instance front and pruning** (`pareto_front.py`, shared with ASA's `branch_comparison`):
- `pareto_front` maps each row to the set of candidates tied at that row's maximum. Ties credit every winner.
- `prune_dominated` implements GEPA's `remove_dominated_programs`. A candidate is dominated when every row it wins is also won by another surviving candidate (a union cover). Candidates are examined in ascending aggregate score (ties broken by ascending key) and removed one at a time until nothing changes. A union-covered candidate is pruned even if it has the top aggregate score. Every non-empty row keeps at least one winner.
- The loop logs `pareto: pruned front holds K of N admitted candidates` each iteration.

**Close contract**:
- The comparison eval is the close step. `commit_before_comparison=True` makes the run point a commit. After the agent's comparison eval, every further `run_eval` is refused (`COMPARISON_CLOSED_NOTE`), which also limits the agent to one comparison per iteration.
- If the closing tree is not program-equivalent to the comparison eval's tree, the loop resets the branch to the comparison-run commit (`close_effects.revert_to_comparison_point`) rather than paying for a rerun. If that reset fails, it falls back to a loop rerun.

**Admission** (`pareto_optimizer.evo_on_iteration_closed`):
1. Structural exclusions come first: `skipped`, `constraint_violation`, and `no_change` (the closing content is program-equivalent to the parent and already represented).
2. Otherwise, the iteration is admitted if and only if its closing content has a comparison-set eval. `comparison_state` maps to `admission_reason`:
   - `EXACT` or `PROGRAM_EQUIVALENT` → `agent_comparison`;
   - `REVERTED_TO_COMPARISON_POINT` → `reverted_to_comparison_point`;
   - `RERUN_BY_LOOP` → `loop_comparison`.
   A train tie or regression is not a barrier.
3. Fallback (GEPA-inspired): when the agent ran no comparison and the train subsample strictly progressed (`subsample_best > subsample_baseline`), the loop runs one comparison eval at close. It is counted, overdraws credits, and is tracked in `CEState.loop_comparison_reruns` / `loop_comparison_rows`. An iteration with neither a comparison nor strict progression is `no_comparison` and spends nothing. At most one comparison runs per iteration.
4. After every iteration, `primary_branch_after` is the population best. The record carries `candidate_admitted`, `admission_reason`, `comparison_set_score`, `comparison_state`, `comparison_eval_id_at_close` and `comparison_score_at_close`.

Because `memory.md` is outside the repo and `codeevolver.md` is non-program, a memory or summary edit after the comparison eval never changes what was admitted.

---

## 7. Safety and integrity

### 7.1 Holdout fence

- **Deny paths (all agents except the run-integrity inspector).** Substring fragments: the test-set path and basename, `_holdout` (every held-out trace directory), `constraint_diffs` (CodeEvolver's own constraint-checker fixtures), `clean_slate_report`, `_architect_trace` (sub-agents only), and any client `additional_deny_paths` (appended, never replacing the built-ins).
  - On the Claude architect, `architect_hooks._make_holdout_guard_hook` blocks Read / Edit / Write / Grep / Glob / Bash calls whose arguments mention a fragment.
  - On OpenCode, they become permission deny patterns. Because `apply_patch` hides its targets inside `patchText`, a plugin fences `apply_patch` specifically.
  - Coding, inspection, analysis and seed agents receive the list as `deny_paths`.
  - These fences match patterns. They are policy fences, and an agent with a shell can work around them.
- **OS-level fence (Docker research runner).** `paths.agent_user()` reads `CODEEVOLVER_AGENT_USER`. When it is set, every agent subprocess (architect, coding, inspection, MCP preflight, session export) is spawned as that unprivileged user, and a failed user switch raises rather than silently dropping the fence. The evaluator and git helpers stay privileged. The runner relocates the test set to a root-owned mode-600 path outside the workspace and evicts its blob from the seed commit before the run. Only the run-integrity inspector runs privileged, because it must read the test set to detect hard-coding. Mechanism and accepted residuals: `specs/experiment-docs/experiment-fleet.md`.
- The architect is told to run training and comparison rows only through `run_eval`. This is a prompt rule. Budget bypass is audited after the fact (§7.4); policy is that every task-LM call must be budgeted, while re-scoring cached outputs is allowed.

### 7.2 Constraint gate (`constraint_gate.py`)

- **When it runs**: only when `additional_instructions` is non-empty. With no client constraints, the gate is a no-op.
- **What it judges**: `git diff <parent_sha> <close_sha>`, so both coding-agent and direct edits are covered. The agent's change requests are supplied as interpretive context. The judge prompt (`agent/constraint_checker.py`) also carries a global constraint against hard-coding question→answer mappings from held-out splits.
- **Judge model**: fixed at `lm_config.CONSTRAINT_CHECK_LM` (`anthropic/claude-sonnet-5`). It falls back to `reflection_lm` with a warning when `ANTHROPIC_API_KEY` is unset.
- **Fail-open**: a diff error, empty diff or missing verdict returns no violation. Every judged diff logs a pass or FAIL line.
- **On FAIL** (`close_effects.force_revert_to_parent`):
  - the registered branch is `git reset --hard` to the parent;
  - an empty marker commit is added: `Revert iteration N: the constraint checker found this change violated the constraints in the additional instructions.` plus `Violated constraint: <verbatim quote>`;
  - `finalize()` re-runs, so `changes_landed=False`;
  - the record carries `constraint_violation`, and the primary does not move on the agent-directed arms.
- **Next iteration**: the kickoff message appends `PREVIOUS_ITERATION_REVERTED_NOTE`, which states only the fact of the revert.
- **What the agent never sees**: the judge's `reason` is never written anywhere agent-readable, because it would teach the agent how to argue past the checker. Only the constraint text, which the agent already holds, is quoted.
- The simple loop applies the same gate per session (`simple_loop/constraint_check.py`).

### 7.3 Clean-slate inspector (`clean_slate.py`)

- **Flag**: `enable_clean_slate_inspector`, default False because a `contaminated` verdict aborts the run.
- **When**: in the seed step, before the run branch exists.
- **What it verifies**:
  1. git history holds no prior solutions beyond the provided program;
  2. every holdout file is covered by the deny paths (names only);
  3. no prior-run leftovers (data, memory, notes, `codeevolver-` branches) are reachable.
- **Tools**: it is the only pre-run agent with a Bash tool, because git history and the reflog cannot be read otherwise. It is non-writable and runs as the agent user. `_snapshot_repo` fingerprints HEAD, the working-tree content hash, refs, the reflog tip and `status --porcelain --ignored` before and after the session; any difference raises `CleanSlateMutationError` on every exit path.
- **Verdicts**: `contaminated` raises `CleanSlateContaminationError` before any eval spend. `uncertain`, an agent error or an unparseable verdict fails open with a warning. The status is stamped on the seed record (`clean_slate_status`).
- **Report**: written to `<logs_dir>/<job_id>_clean_slate_report.md`, not the memory dir, because it exists before the architect's first turn.

### 7.4 Run-integrity auditor (`meta/run_integrity.py`)

- **Flag**: `enable_run_integrity_agent`. It runs after meta-reflection and is advisory only; it never stops or fails a run.
- **What it classifies**:
  - cheating: `no_cheating`, `non_impactful_cheating` or `impactful_cheating` (cheating = using the held-out set to inform the program; a data index counts only if it indexes question→answer pairs);
  - the share of functioning iterations;
  - `budget_bypass` (training or comparison rows run outside `run_eval`, with an estimated row count).
- **Access**: it may read the test set, it holds a read-only Bash tool, and it is the one privileged inspection session.
- **Output**: the verdict (the final fenced JSON block) is stamped onto the latest iteration record (`integrity_*` keys) and re-flushed through the progress callback. The report goes to `{memory_dir}/run_integrity.md`.
- **Mid-run safe mode**: `scripts/inspect_run_integrity.py` runs the inspector against a live run with a do-not-disrupt preamble and writes to an operator path, never into the memory dir.

### 7.5 Egress

Research runs execute in a Docker runner whose outbound traffic passes through an allow-listed proxy. A preflight verifies, before any spend, that the agent user exists and can run git, that `rg` is on PATH, and that required hosts are reachable. An egress audit joins proxy logs against each run's eval windows to flag out-of-band task-LM calls. The details live in `specs/experiment-docs/experiment-fleet.md`; the engine itself has no egress logic.

---

## 8. Finalization

**Scoring** (`finalization.score_seed_and_final`, none of it counted against the budget):
- The final program is the primary branch at the end of the loop (§4.4).
- The seed is scored at `seed_sha` with a detached, forced checkout, never by checking out the seed branch. The run-main branch's tip is usually the final program, so scoring by branch name would measure the final program twice.
- Each program is drawn on the held-out test set: the seed `seed_test_samples` times (default 1) and the final program `test_samples` times (default 3). The seed's readings already average across runs that share it, which is why one draw is enough. A failed draw is logged and skipped. The results are reported as per-draw lists (`seed_test_scores`, `final_test_scores`) with their means (`seed_test_score`, `final_test_score`).
- Each program is also scored once on the comparison set (`seed_train_score`, `final_train_score`, trace suffix `final_comparison`). These readings are not appended to the agent-facing comparison history.
- If the final sha equals the seed sha and no more draws are needed than the seed took, the seed's readings are reused.
- The mean and the per-draw list (`test_score`, `test_scores`) are stamped on the seed record and on the latest record of the final branch.
- A scoring failure is reported as `finalization_error`; the job still completes and the final branch is restored.
- **Response caching**: repeated draws carry information only if the client program's LM calls are not served from a response cache. The client repo owns its cache settings, and the engine configures none. When every draw's per-row outputs are byte-identical, the engine logs a warning that a response cache may be active.

**Result.** The result dict and `set_completed` carry `final_candidate`, the seed and final scores above, `run_metrics`, `total_metric_calls`, `num_branches`, the `ce_state` payload, and per-model token usage.

**Post-run agents** (opt-in and advisory; they run after completion is posted, so they cannot affect run success):
- **Meta-reflection** (`meta/meta_reflection.py`, flag `enable_meta_reflection`): a read-only agent with Bash audits the run (traces, memory, change summaries, the engine source and this spec) and writes `{memory_dir}/meta_reflection.md`. It has no deny paths.
- **Run integrity**: §7.4.
- Both go through `run_reflection_agent`, which retries a session that produced no output (3 attempts, backoff 5 s / 15 s) and logs `NO VERDICT` at ERROR when an auditor returns nothing.
- Architect trace capture is on whenever either flag is on.
- On OpenCode, every inspection session gets its own data and state directory (`XDG_DATA_HOME` / `XDG_STATE_HOME`), while the cache stays shared, so sessions running as different users never share one SQLite store. An OpenCode error event is never mistaken for report text.

**Offline replay.** A cancelled run never reaches finalization. `scripts/run_meta_reflection.py --run-dir <collected run> --workspace <clone>` replays the arm's auditor (v3 or simple loop) against a collected run directory, using the run's own harness, model, budget and history. The workspace is materialized from `repo.bundle`, including `refs/codeevolver/*`. Held-out scores are reported as N/A. The replay spawns a live, billable session.

---

## 9. Traces and reporting

**Architect traces** (`architect_trace.py`) are captured into `{architect_traces_dir}/iteration_{N}_architect_trace/`:
- `index.jsonl` has one line per turn: `turn`, `role`, `ts`, `kinds`, `tools`, `is_error`, `chars`, `file`, `captured`, and a ≤200-character `summary` showing tool calls with their identifying argument, a one-line `run_eval` result, and text previews.
- Each turn is stored in `turn_{i:04d}.json`.
- On Claude, a turn file is appended per stream message, and the initial prompt and every watchdog resume prompt are recorded as user turns (`meta.prompt_kind`).
- On OpenCode, `opencode export` runs after each invocation (initial and resumes) and the directory is rewritten. Turns carry the export's own timestamps, errored tool parts keep their error text, and patch parts appear as `[patch] <files>`.
- On OpenCode, multi-session captures (initial plus watchdog resumes) merge their result turns into one trailing turn with summed usage (`_merge_result_turns`).
- A failed capture leaves a `captured:false` stub turn, and the capture error is logged.
- `read_architect_trace(N)` is the programmatic reader. The directory name contains `ARCHITECT_TRACE_DENY_FRAGMENT` (`_architect_trace`), which keeps the architect from reading its own past sessions.
- Before every OpenCode architect session, an MCP preflight spawns the tool server exactly as OpenCode will and completes a `tools/list` handshake. The session aborts if any registered tool is missing.

**Rendered digests** (`architect_trace_markdown.py`). After each iteration is closed (and after its last watchdog resume, since OpenCode rewrites the directory per export), the loop renders, deterministically and without an LLM:
- `digest.md`: the session without thinking. A header (branch, close sha, changes landed, train baseline → best, comparison best, rows by sample, watchdog resumes, counts, elapsed time, harness and model, tokens), then every turn: prompts and watchdog resumes verbatim, assistant text in full, tool calls as `name(args)`, tool results capped at 1,000 characters, and `run_eval` results as one structured line.
- `transcript.md`: everything, including thinking, with results capped at 2,000 characters.
- `header.json`: the merged header.

Rendering happens only after the close for two reasons: OpenCode rewrites the trace directory on every export, and the close facts in the header exist only after the close. On Claude, `append_session_prompt_turn` records the initial and resume prompts (`meta.prompt_kind`), so both harnesses render the same `Prompt` / `Watchdog resume` headings. Per-turn timing (`+12.0s`) is shown only when the turn timestamps differ.

`{architect_traces_dir}/README.md` has one row per iteration with links and is regenerated on every close. Capped payloads end with `→ full: turn_NNNN.json`, and every line fits OpenCode's read cap. A render failure only logs. The auditors' shared reading guide is: README → every `digest.md` → `transcript.md` for the one or two iterations that matter → individual turn files only for a capped payload.

**History and progress.**
- `IterationHistory` holds one record per iteration (field whitelist: `schemas/iteration_history.py::ITERATION_RECORD_FIELDS`). Closing facts are copied from the close record (`run_reporting.CLOSE_RECORD_HISTORY_KEYS`, which includes the per-arm and `asa_*` keys).
- After each iteration, `stamp_and_report_iteration` reads the architecture title and summary from `codeevolver.md`, stamps the change summary, regenerates `iteration_change_summaries.md`, logs the budget and time position, and posts progress (current iteration, `final_candidate`, `num_branches`, `total_metric_calls`, the `ce_state` payload, and a memory.md snapshot) through the job callbacks.
- Per-iteration scores are read from `ce_state.iteration_history`.
- `eval_rows_used` is stamped only by the simple loop; on the v3 arms, per-iteration spend is in `iteration_{N}_evals.jsonl`.

**Collected artifacts** in `{history_dir}`: `comparison_set_history.jsonl`, `iteration_{N}_evals.jsonl`, `row_scores.jsonl`, `datasets.json`, `population.json` (pareto_evolution), `branch_comparison_iter{N}.md` (ASA), and `iteration_{N}_attempted.diff`.

---

## 10. Task-LM provider fallback chain

CodeEvolver never configures the language model that runs the client's program; that LM belongs to the client repository. The benchmark seed programs build their task LM through a small infrastructure module that sits outside the evolvable package (outside the module tree the optimization config names as the parent module), so the architect never edits it as part of the program. The module defines:
- one pinned model, served by a ranked list of providers;
- a default provider equal to the head of that list;
- a per-call divert to the next provider on provider errors only. Errors that would fail identically elsewhere, such as context-window overflow and unsupported parameters, are not diverted, so they stay visible to the architect;
- a process-wide circuit breaker, so a sustained outage costs a constant number of failed calls rather than one per call.

The client repositories originally pinned a single provider (GMI), so an intermittent provider error failed the call and scored the row 0. They now use this fallback chain instead. The fallback belongs to the client because the client owns its language model; CodeEvolver does not. In the benchmark repositories the ranked order is GMI → DeepSeek → DeepInfra, so GMI is the default primary.

Two environment variables override routing for a run without rebuilding the seed program: `LM_PROVIDER` (the primary) and `LM_FALLBACK` (the fallback order). The eval subprocess inherits the environment. Diverted calls are logged to stderr, which the engine saves as `eval_stderr.log` in the eval's trace directory.

The engine has no mechanism that knows about this module. Its protection comes from each benchmark's `additional_instructions`, which name the module as infrastructure that must not be edited, bypassed or replaced and require every LM call to go through it, enforced by the constraint gate (§7.2). `additional_deny_paths` is deliberately not used for this, because it also denies reads, and an agent that cannot read the builder's signature tends to write a direct LM call instead.

The engine's own row-level retry (`mounted/evaluate.py`, retryable provider signatures, capped at 5% of rows, not charged to the budget) sits outside this per-call layer.

---

## 11. Optimization config (`OptimizationConfig`)

The config JSON is flat; the job and auth fields (`repo_url`, `github_token`, `callback_url`, `jwt_token`, `job_id`, `initial_branch`) sit at the top level and are rebuilt into a `JobContext`.

| Field | Default | Meaning |
|---|---|---|
| `program`, `metric` | required | Program entry point (the parent module) and metric. |
| `workspace` | `/workspace` | Client repo path. |
| `trainset_json` / `trainset_path`, `testset_json` / `testset_path` | — | Datasets. The test set is used only at finalization. |
| `comparison_set_size` | None | Fixed comparison set size, drawn from the trainset with seed 42. None or 0 means no comparison set (required by pareto_evolution). |
| `comparison_credits_initial`, `comparison_credits_per_iteration` | 1, 1 | Credit ledger (§5.5). A per-iteration value of `None` disables it. |
| `metric_names`, `primary_metric`, `input_keys`, `num_threads`, `max_score` | `["accuracy"]`, first name, —, 1, 1.0 | Evaluator settings. `max_score` drives baseline-ceiling handling. |
| `reflection_lm` | `anthropic/claude-opus-4-7` | On the Claude harness it selects the agent model. On OpenCode it serves the LiteLLM side calls (seed analysis fallback, constraint-judge fallback). |
| `coding_harness`, `coding_harness_model` | `opencode`, — | Harness for every agent, and the OpenCode model id. |
| `seed` | 0 | Run rng: subsample draws, acknowledgement codes, pareto sampling. |
| `subsample_size` | 20 | Train subsample rows. |
| `max_metric_calls` | None | Eval-row budget. The legacy key `max_eval_rows` is read as a fallback. |
| `job_timeout_seconds` | None | Hard timeout; the loop stops 10 h earlier. |
| `comparison_eval_timeout`, `subsample_eval_timeout` | 14400, 7200 | Per-eval backstops (seconds). The first also covers finalization draws. |
| `test_samples`, `seed_test_samples` | 3, 1 | Held-out draws for the final program and the seed (each ≥ 1). |
| `variance_estimate_samples` | 5 | Runs per variance estimate (≥ 2). Internal; the agent never chooses it. |
| `debug_max_iterations` | None | Iteration cap for tests. |
| `asa_behavioral_contribution_constant`, `asa_behavioral_contribution_threshold_cap`, `asa_primary_stale_landed_iterations` | 0.10, 0.05, 4 | ASA knobs (`specs/archive-public/v3_asa.md`). |
| `additional_instructions` | None | Client constraints. Shown to the architect; turns on the constraint gate. |
| `enable_meta_reflection`, `enable_run_integrity_agent`, `enable_clean_slate_inspector` | False | Auditors (§7, §8). The Docker research runner turns all three on and pins `test_samples=3`, `seed_test_samples=1`, `variance_estimate_samples=5` and both credit knobs (`runner/entrypoint.py::RESEARCH_DEFAULTS`). An explicit override always wins. |
| `additional_deny_paths` | None | Extra substring fences, appended to the built-in holdout fences. |

**Rejected keys.** `valset_json`, `valset_path`, `valset_bar` and `max_valset_size` raise `ValueError` (engine 3 has no validation set; fold those rows into the trainset and use `comparison_set_size`). Other unknown keys are ignored. Engine-only knobs that are not config fields: `CODEEVOLVER_EVO_EXPLOIT_K` / `CODEEVOLVER_EVO_EXPLOIT_THRESHOLD` (pareto exploit curve), `CODEEVOLVER_AGENT_USER` (OS fence), and `CODEEVOLVER_TRACE_MAX_LINE_CHARS` (set automatically for OpenCode).

---

## 12. Tests

| Area | Where |
|---|---|
| Close tool checks, fallback close, parent derivation | `tests/controls/test_agent_tools.py`, `test_close_and_parent_derivation.py`, `test_baseline_requirement.py`, `test_freeform_close.py`, `test_evo_close.py`; `tests/unit/test_iteration_context.py` |
| Content identity and anchors | `tests/controls/test_eval_identity_tree.py`, `test_replay_recorded_run.py`, `test_close_side_effects.py`; `tests/unit/test_git_tree_signature.py`, `test_program_paths.py` |
| Watchdogs | `tests/controls/test_no_change_watchdog.py`, `test_close_skill_watchdog.py`, `test_revert_check.py`, `test_reflection_watchdog.py` |
| `run_eval`, budget, credits, variance | `tests/unit/test_run_eval_budget.py`, `test_run_eval_filter.py`, `test_run_eval_response_shape.py`, `test_budget_note.py`, `test_comparison_credits.py`, `test_variance_estimate.py`, `test_inspect_traces_target.py`; `tests/controls/test_credits_loop_wiring.py` |
| Pareto evolution | `tests/unit/test_population.py`, `test_pareto_select.py`, `test_pareto_front.py`, `test_evo_prompt_and_skills.py`; `tests/controls/test_evo_admission.py` |
| Prompts, skills, memory templates | `tests/unit/test_freeform_prompt_and_skills.py`, `test_skill_staging_policy.py`, `test_memory_template_and_prompt_notes.py`, `test_architect_prompt_paths.py` |
| Constraint gate, holdout fences, clean slate | `tests/unit/test_constraint_gate.py`, `test_agent_user_fence.py`, `test_clean_slate.py`; `tests/controls/test_testset_and_lock.py` |
| Finalization, traces, auditors | `tests/unit/test_holdout_sampling.py`, `test_trace_layout.py`, `test_trace_json.py`, `test_architect_trace.py`, `test_architect_trace_markdown.py`, `test_meta_reflection.py`, `test_run_integrity.py` |
| ASA | `tests/unit/test_asa_rules.py`, `test_asa_prompt_and_skills.py`, `test_branch_comparison.py`; `tests/controls/test_asa_gate.py`, `test_asa_close.py` |
| Live (billable) agent behaviour | `tests/functional/` (`test_e2e.py`, `test_close_iteration.py`, `test_session_resume.py`, `test_comparison_credits.py`, `test_variance_tool.py`, `test_test_sampling.py`, `test_auditors_back_to_back.py`), driven by `scripts/run_tests.sh` |

Offline tests that touch git use real temporary repositories. Fixtures for agent-facing tool results must look like real outputs, because a fixture that announces itself as a stub changes model behavior.

---

## 13. Past issues and their guards

Moved to `specs/engine3_past_issues.md`: each problem we have hit, its cause, and the rule that now prevents it.

---

## Appendix: simple loop and ASA in one paragraph each

**simple_loop** (`specs/v3_simple_loop.md`). One linear branch and one fresh OpenCode session per iteration, with a single `run_eval` tool: no skills, no memory template, no coding or trace sub-agents, no watchdogs, and no `run_variance_estimate`. It uses the same `RunEvalBinding` (budget gate, credits, acknowledgement), the same constraint gate (revert marker and kickoff note), the same held-out sampling helper (`evaluator/holdout.py`), the same run-integrity auditor, and its own meta-reflection prompt (`meta/simple_meta_reflection.py`). The session's best train reading excludes `single` and off-sample evals and is None when the constraint gate reverted the session. Its budget note states only the rows used and remaining.

**asa** (`specs/archive-public/v3_asa.md`). EE's prompt, close tool, explore guard, watchdogs and auditors, plus:
- rule-driven requirements computed in `on_iteration_start` (eligible non-primary lineages by behavioral contribution);
- a `run_eval_gate` at the baseline that warns while the violation allowance `ceil(closed/10)` holds and refuses otherwise;
- the read-only `branch_comparison` tool;
- the seed registered as its own non-primary lineage (`codeevolver-{ts}-seed`), with one counted seed comparison eval;
- loop comparison evals at promotion without a tip-matching record, or when the primary's record is stale by more than `asa_primary_stale_landed_iterations` landed iterations. These are skipped when the agent ran a comparison that iteration.
