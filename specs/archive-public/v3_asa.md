# Artificial Selection Algorithm v3
The purpose of the ASA branch is to combine the explore/exploit harness with a requirement to traverse the population tree (as opposed to an unenforced ability). 

## Guiding principles: 
1. Growing an evolutionary tree requires different rates of vertical versus horizontal growth across the exploration path. Early on, exploitation is easy and valuable, but it also can lead to local optima. Early on, it is also difficult to require branching from non-primary branches because they don't exist yet. Toward the end, it might make sense to exploit one or two branches
2. Non-primary branch work doesn't really matter if you work on a lineage that has no meaningful positive behavioral contribution.
3. Eligible non-primary branches (A non-primary branch with greater than d% positive behavioral contribution) take a while to establish and may never be discovered
4. Iterations are inherently quantized, which makes smoothing functions relatively meaningless unless they reflect probabilities.
5. Our algorithm must work on our iteration budget for the research as well as larger iteration settings. We can achieve this through the right window definitions, and window definitions should combine iteration counts with percentages so they are robust to lower iteration or higher iteration settings.

## Mechanism
The ASA Population Management Mechanism utilizes two threshold-based requirements - `eligible_non_primary_parent` and `new_lineage` requirement - to dictate when behaviors are required. The advisory mechanism is a prompt injection in the main prompt with the requirement. The enforcement mechanism is a threshold. If the agent ignores the requirement When it is below the threshold, we don't reject on the first offense. We permit a certain number of violations `allowed_violation_count = min(default_allowed_violation_count, floor(10%*n))`. We use the `run_eval` tool at the baseline run to determine if the agent is actually complying with the new_lineage or eligible_non_primary_parent requirements, and enforce through an error on run_eval, with a note.  

We introduce a `branch_comparison` tool. ASA creates a `branch_comparison` tool for computing positive behavioral contribution between branches and for emitting Pareto rows per branch (see section below).

### Rule set
This branch comparison tool serves a second purpose as the load-bearing enforcement mechanism.If no positive behavioral contribution is greater than contribution threshold relative to the primary branch, then there are no eligible non-primary parents. The rules are as follows
1. During the phase one window, the agent must create new lineages in 40% of iterations. Over 10 iterations the minimum would be 4. Prompt injects, enforcements, and allowed violations begin on iteration 4, whenever the current rate is below 40%.
2. If eligible_non_primary_branch exists:
    - enforce `> 30% × n_iter_in_window` iterations include eligible_non_primary_parent = True. prompt inject / enforcement begins on the 3rd eligible iteration
    - eligible_non_primary_parent(s) are the parents where positive_behavioral_contribution > bc_threshold % of score (Use min(5%, 10% of primary branch's score gap to max_score). This is an optimize config element with a system default). I.e., at 96% this would be 0.4%; at 70% it would be 3%; at 30% it would be 5%.
- One iteration can count for both the new branch and the new non-primary parent

#### Rule 1 — new lineages
Inject when new lineage creation rate is below 40%:
```
## THIS ITERATION MUST EXPLORE AND ADD A NEW LINEAGE
  The run has created fewer new lineages than its pacing floor. Build this iteration's change on a new `codeevolver-` branch and register it at close. You choose the parent. See the `exploration` skill.
```
And in the user message:
```
new_lineage_note = (
    "This iteration MUST be add a new lineage and explore — see the "
    "`exploration` skill. "
    if self.context.must_add_new_lineage
    else ""
)
```

Lineage creation is enforced via run_eval baseline. Exploration is prompted but not enforced (same as EE - if EE Enforces please correct).
run_eval uses the note:
> You must create a new branch / lineach for this iteration. Please select that branch before running the baseline and then call the run_eval tool again.

#### Rule 2 - _MUST_BRANCH_FROM_ELIGIBLE_NONPRIMARY
Inject when current rate falls below 30% during Eligible Non-Primary Branch Window:
```
_MUST_BRANCH_FROM_ELIGIBLE_NONPRIMARY = """
## THIS ITERATION MUST SELECT AN ELIGIBLE NON-PRIMARY BRANCH AS ITS PARENT
A non-primary branch has emerged that contains meaningfully positive behavioral contribution relative to the primary branch, and the share of iterations that have improved off of it (either in a new-branch exploration or an exploitation) has fallen below the 30% quota. See the `exploration` skill for context and use the `branch_comparison` tool to identify eligible non-primary branches.
"""
```
And in the user message:
```
must_branch_from_eligible_nonprimary_note = (
    "This iteration MUST select an eligible non-primary branch as its parent — see the "
    "candidate pool section of the `exploration` skill. You may exploit or explore here."
    if self.context.must_branch_from_eligible_nonprimary
    else ""
)
```
Enforced via run_eval baseline using the note:
> You must use an eligible non-primary branch as the parent for this iteration. Please select that branch before running the baseline and then call the run_eval tool again.

#### Rule 3 - Must explore
The Must Explore section stays with a 30% quota.

```
_MUST_EXPLORE_SECTION = """
## THIS ITERATION MUST EXPLORE
The share of iterations that have reported exploration has fallen below the 30% quota.
"""
```
```
must_explore_note = (
    "This iteration MUST be an exploration iteration (the run's "
    "exploration share has fallen below quota) — see the "
    "`exploration` skill. "
    if self.context.must_explore
    else ""
)
```
### Rules omit during free tail
- Begins when `evals_remaining ≤ max(2 × 5 × subsample_size, 0.10 × max_eval_rows)`.
- In the tail no requirement fires, no rule-based prompts are injected, and no close-time ASA check runs. Counters are still recorded. The `branch_comparison` tool remains available.

### Seed
The seed branch is often a legitimate evolutionary branch, especially considering it is usually the simplest. The ASA algorithm should run the comparison set on the seed branch.

### Eligibility
- **positive_behavioral_contribution is the distance(B, R) | Bi>Ri** = Σ over comparison-set rows of `max(0, score_B(row) −  score_R(row))`. Reported alongside `n_rows_ahead` (rows where B > R).
- **Normalized distance** = `positive_behavioral_distance / comparison_set_size`, i.e. mean
  per-row advantage, in the same units as a score. All thresholds are on the normalized value.
- **estimated_max_score**: the evaluator's estimated maximum attainable score, calculated using the max row score * n rows.
- **behavioral_contribution_threshold** = `min(behavioral_contribution_threshold_cap, behavioral_contribution_gap_fraction × (estimated_max_score − primary_comparison_score))`, optimize-config elements with system defaults `behavioral_contribution_threshold_cap = 0.05`, `behavioral_contribution_gap_fraction = 0.10`. (when primary branch scores 96% on estimated max of 100%, 0.96 → 0.004; at 0.70 → 0.03; at 0.30 → 0.05.) Principle 3's "d%" is this value.
- **Eligibility:** A live non-primary lineage is *eligible* if its normalized distance against the primary's latest comparison record is `> bc_threshold`.
- **Which comparison record represents a lineage:** the newest comparison eval whose content is the lineage tip (exact tree or program-equivalent, per `comparison_record_for_tree`); failing that, the newest comparison eval that ran on that branch. A lineage with no comparison eval at all is **not eligible** (it is unknown, not different). 
    - Comparison set records need commits or lineage is unreliable!
- **Staleness:** a record is stale when its tree differs from the branch's current tip tree. Being a stale record is okay! It still permits eligibility [DECISION — the alternative, requiring fresh records, makes eligibility depend on comparison spend the agent-driven arm is meant to avoid]. The tool flags staleness so the agent can choose to refresh.
- **No primary comparison record → no eligible lineages → rule 2 is inert.** 
- Primary comparison set check: Without a check, if the agent never runs a comparison eval on primary, then rule 2 won't activate. Create a prompt injection that, if budget exceeds 33%, it prompts that you must exploit on the primary branch on this iteration and run a comparison set at the end. The closed iteration tool checks for the comparison set and runs comparison set on primary if none is run. May require switching branches

### Comparison Set Eval Itegrity: Commit!
Similar to the Pareto_evolution mechanism, we need to force commit inside run_eval when eval_sample == "comparison" and the tree is dirty (git_add_and_commit with a message like iter N: auto-commit before comparison eval), before the eval runs, so sha, committed=True, the JSONL line, and CloseRecord.comparison_scores are all correct at write time — no reconciliation dependence at all

ASA requires enough comparison sets to be run at the right time in order to have a mechanism to enforce the eligible nonprimary rule (rule 2). However, it is difficult for the loop to know when the right time is to run the comparison set.

Unlike the Pareto evolution branch, we do not require comparison sets to run at the end of an iteration UNLESS a branch is being promoted and it hasn't run the comparison set. 
- The mechanism then is a combination of 30% new branches, and eventually one of those branches are likely to be promoted. If discovery isn't fruitful, rule two doesn't add value.
- In addition to this mechanism, the agent will naturally run the comparison set when it considers promotion and when it contributes new branches to the pool (via prompt)

### Window
- Phase 1 window lasts until budget consumption exceeds 50% at the start of budget and a minimum of 10 iterations.
- Eligible Non-Primary Branch Window lasts while eligible_non_primary_branch = True
    - Enforcement during this window counts the denominator as the number of iterations in that window; i.e., `n_iter_in_window`
- Free tail occurs at the end and lasts from 90% onwards or at least two iterations. I.e., `max(2 estimated iterations, 10%*budget)`. Use estimated_n iterations using 5x sub-sample size window here.

#### Progress units
- The run's length is governed by `max_eval_rows`, not an iteration count. Every ASA window is
  therefore defined in **closed iterations** for the front of the run and in **remaining eval
  budget** for the tail.
- Size windows based past iteration counts and current budget progress. Don't estimate total iteration count except for estimating remaining iterations for the final window.
- **Counting iterations.** Window denominators count *closed* iterations exactly as `run_metrics._closed` does: an iteration that tested a change and reverted counts; an  architect-error stub (`skipped`) does not; a deliberate no-op (HEAD unmoved, no eval) counts. Rationale: a reverted attempt consumed the iteration's opportunity to branch.

### Healthy Tree
Our guideline for a healthy tree is that, in phase one, I would hope for 30% - 50% of new branches (ideally claiming exploration). Once a branch emerges that contains positive behavioral advantage, we should work on those lineages 30% - 50% of the time. While the loop is setting these boundaries, the agent gets to decide what parent to select. 

A healthy tree also contains pure exploitation. I would hope to see 30%-50% exploitation on the primary parent across all phases. and up to 70% of activity occuring on the primary branch

In the final 10% of the budget (phase 3), all rules return to 0%+, to focus on full flexibility.

### Prompts
The prompt should be the same as EE. The exploration skill should contain the differences. This preserves the principle of not distracting the agent when it is improving as usual versus when it is improving using exploration. 

Delta from EE:
1. Tool list: `branch_comparison` added with description.
2. `exploration` skill (see below): Candidate Pool section replaced as drafted; adds Non-primary branches and Branch comparison tool subsections.
3. `close_iteration` skill (see below)
4. Injected blocks (system prompt + user-message note, same dual mechanism as EE), multiple can fire: `_MUST_CREATE_NEW_LINEAGE`, `_MUST_BRANCH_FROM_ELIGIBLE_NONPRIMARY`, and MUST_EXPLORE. 

#### Exploration skill
Most of the exploration skill is the same. However, I have added changes to the candidate pool section.

```
## The Candidate Pool
If your new macro approach is sufficiently different from the existing methods, you should build it on its own branch and add it to your candidate pool, which you can track in the memory.md file. Use a new git branch when an approach is sufficiently new, and name it based on the macro approach, with the prefix `codeevolver-`. DO NOT TOUCH BRANCHES THAT DON'T HAVE THE PREFIX `codeevolver-`. 

At any given time, there is one branch that is the "primary" / best branch. Before your run, you were provided a primary branch. If a new branch should be promoted to primary, you may update it at the end of your run by passing the appropriate branch as primary when you call the close_iteration tool. The primary branch is important because future exploit-focused agents will focus on improving the primary branch whenever they decide to exploit the best program. It's also important because the final program is the latest primary. Do NOT set a non-best program as primary.

Branches you do not plan to continue working on can be removed from the candidate pool that you're tracking. (This doesn't mean deleting the branch just removing them from the candidate pool that you're tracking in the memory.md file). If a branch is useful for future experiments, keep it in the pool!

Source of truth: the memory.md file is the source of truth for organizing which branches are still in the candidate pool. The close_iteration tool is the sole place to set the primary branch for a future agent.

### Non-primary branches
A non-primary branches is only valuable if it offers a positive behavioral advantage that the primary branch does not offer. You can measure this behavioral advantage using the `branch_comparison` tool (see below).

Once there is a non-primary branch with sufficient positive behavioral advantage, it becomes worthwhile to explore improvements to it, especially when that branch takes a meaningfully different approach.

For the period of time where non-primary branches have meaningfully positive behavioral advantage, you should consider dedicating iteration(s) to them some fraction of the time, even if their overall score is lower than the primary branch.

### Branch comparison tool
The `branch_comparison` tool computes positive behavioral contribution between branches and emits Pareto scores per branch.
- Input arguments: pass a reference branch (usually the primary branch). The tool will compute behavioral contributions against this branch.
- Positive Behavioral contribution is the summation of positive row-wise score differences on the comparison set between two branches. It does not include rows where the branch is inferior - You can use the overall comparison set scores for that.
- Candidates are entire branch lineages and utilize the most recent comparison set taken for that branch.
- The tool will note whenever the comparison set is out of date for a branch. That is okay.

### Promoting a branch to the primary branch
[same]

```
#### User Message
The user message will be the same, except when we hit our threshold requirements. We should append the rule 1 and rule 2 injections to the system prompt and user message as a dual mechanism to inject prompts. (similar to how EE does it).

NOTE: _MUST_EXPLORE_SECTION still fires, as guidance only. It is perfectly ok for the Must Explore section to fire at the same time as a New Lineage section.

If, however, we have an eligible non-primary parent and our threshold has fired, then we should also add _MUST_BRANCH_FROM_ELIGIBLE_NONPRIMARY (see above)

#### Close Iteration Skill
Comparison evals are committed automatically, So we need to let the agent know to commit deliberately first - Similar to the Pareto evolution mechanism.

Replacement section 1
```
1. Make sure the program is **committed** at the point you want to finalize. If you did not like your changes, `git reset --hard` to where you want to leave the branch. The loop continues improving from wherever you leave HEAD.
2. If your changes made meaningful progress and this branch is likely to stay or get added to the candidate pool, you should consider running the comparison set eval. See when to run the comparison set below.  
3. Before exiting, **update memory** at the {memory_dir}/memory.md file with key metrics and learnings (See "Saving to Memory" below) including **adding your branch to the candidate pool** if you want future agents to have access to it.
4. Finally, call the close_iteration tool to finalize the iteration. 
```

Second replacement section
```
### Promoting a branch to the primary branch
If you believe your new candidate might outperform the primary candidate, we need to run a larger, fixed comparison set where we can evaluate branches apples-to-apples and see if a branch should be promoted as the primary candidate. This only applies when your baseline was not taken from the primary candidate. **IMPORTANT: when comparing two branches, given the high variance between different training sub-samples, we cannot compare scores from different training samples; we should use a fixed comparison set instead.** Before promoting a branch to be the new primary candidate you need an apples-to-apples comparison. If the baseline train set was not the primary branch, then you should use the comparison set to compare (use `run_eval` on the new branch with eval_sample = "comparison"). To compare to the 'incumbent' primary branch, check out if the latest version of the primary candidate branch has a comparison set score in the {history_dir}/comparison_set_history.jsonl file. Otherwise, you will need to run the incumbent primary branch on the comparison set (eval_sample = "comparison") as well before declaring a new primary candidate.

Note: While the score in the comparison set provides guidance for promotion, do also use logic as to which one to keep as primary (e.g., ask which will generalize better). And note that our comparison set is drawn from the same set as the training set, which means one branch could have a slight advantage if it trained on more of the comparison set rows than the other. 

Run the comparison set sparingly, since it consumes budget!

### When to run the comparison set
The comparison set is what allows you to make comparisons against this branch in the future (via the branch_comparison tool). Those comparisons include showing you how many rows this branch has an advantage relative to other branches. The comparison set consumes budget, so omit it if progress was minimal. 

When to run the comparison set
- When a new branch should enter the candidate pool (i.e, the shortlist of candidates that you plan to work from in the future). Run the comparison set without second thoughts.
- When an existing branch in the candidate pool has meaningful progression against the evaluation
- When considering a branch for promotion
- A stale reading is okay if there has not been meaningful eval progress. 
- Careful not to run too many comparison sets on the primary branch - this could get costly, especially if a series of exploit iterations keep running comparison sets on it. Hence, a stale reading is ok if there has not been meaningful eval progress.

Always commit before running the comparison set and run it using `run_eval(eval_sample="comparison")`. 
```

## Branch Comparison Tool
The branch comparison tool should be exposed to the agent with:
- Input arguments: reference branch. The agent passes the reference branch which it wants behavioral contributions computed against.
- Positive Behavioral contribution is the summation of row-wise score differences on the comparison set between two branches where the branch exceeds the reference branch.
- Candidates are entire branch lineages and utilize the most recent comparison set taken for that branch.
- Tool output notes whenever the comparison set is out of date

The Branch Comparison tool should also run at the beginning of each iteration across every registered branch pair, using whatever comparison set is available and skipping those that do not have comparison sets. 

### `branch_comparison` contract
- Read-only; costs no eval rows; may be called any number of times.
- Argument `reference` (branch name; default = current primary). Errors if the reference has no
  comparison record, naming the branch and suggesting `run_eval(subsample="comparison")`.
- One row per lineage (no retiring)
- keys / columns: `branch, comparison_score, positive_behavioral_distance, normalized_distance, n_rows_ahead, pareto_score, up_to_date, eligible`, `is_primary`.
- Include reference row against itself and add (reference).
- Rows are compared by `example_id`; a lineage whose record covers a different row set (changed `comparison_set_size`) is listed with non-compatible rows as 0 and a note. This shouldn't happen. I don't think the agent has access to comparison set size.
- Loop-side branch_comparison tool run at iteration start is internal. it computes the eligible set and the fired requirement, and writes the full table to `{history_dir}/branch_comparison_iter{N}.md`. It is **not** placed in the prompt or user message.
- Pareto's score is f(c), described below.

### Example output:
I have expressed the output in CSV format, but we should use the format that is easiest to read for the agent.
```
Behavioral contribution is expressed relative to {branch_name} branch. The positive_behavioral_contribution includes the row-wise differences where the branch has an advantage over the reference branch.:

branch, positive_behavioral_distance, total_samples, pareto_score, comparison set up-to-date, eligible, is_primary
codeevolver-react (reference), 0, 50, 4, false, false, true,
codeevolver-reasoning-workflow, 2, 50, 1, true, true, false
codeevolver-one-shot, 0, 50, 0, true, false, false
Other branches were not run if they did not have comparison sets run on them
```

## Definitions (Glossary)
All rule counters use loop-derived facts from the closing record, never the agent's claims.
- **new_lineage** (derived): the branch registered at close did not exist in the branch registry at iteration start (`ce_state.known_branches()`), AND it was registered via `register_branch`. A git branch that is created but not registered is not a lineage.
- **parent_branch** (derived): the branch of the newest past closing record whose `close_sha` is an ancestor of this iteration's `close_sha` (merge-base fallback, seed as last resort — the existing `derive_iteration_parent` rule).
- **eligible_non_primary_parent** (derived, per iteration): `parent_branch` was a member of the eligible set computed at *this iteration's start* (see Eligibility). Eligibility is frozen at iteration start so the agent's own comparison runs during the iteration cannot change what it is being held to.
- **Live lineage:** a registered branch that has not been retired (see Retirement).

### Pareto Calculation
This should be the same as with Pareto Evolution and is inspired by GEPA.

For every comparison/validation row \(i\), GEPA finds the best score achieved by any candidate:

> s*(i) = max_over_all_candidates_c( score_c(i) )
- c: Candidate c
- i: row i
- s: score
- s*: Frontier Score

It then forms the set of all candidates tied at that best score:

> P*(i) = {c : score_c(i) == s*(i)}
- P*: pareto frontier set, before pruning

So if A and B both score 1.0 on row 7, both are initially on the frontier for row 7. It is not restricted to a unique winner. 

Then we remove redundant/dominated candidates and computes

f(c) = ∑ ( ​1[c ∈ P^∗(i)] )
- f(c): pareto frontier score after pruning
- 1[]: 1 if the terms in the brackets are true, 0 otherwise.
- ∈ A member of the set
- P^*: The Pareto Frontier after pruning

## Misc.
- Add this enforcement along with your ASA implementation: Observing if any training or comparison set rows were ran outside the eval (e.g., bash) - the constraint checker agent enforce do this for all harnesses. Every prompt says to never run rows outside of the run_eval. This is cheating vis-a-vis the budget. running a single row at a time for testing purposes is okay, but not if the purpose is to bypass the run_eval_tool
- Lineage Retirement: The source of truth for the candidate pool is memory.md. If the agent drops a candidate but it still has positive behavioral contribution, the loop will still recognize it as an eligible branch, and that is okay. We cannot allow the agent to retire branches because this would bypass the entire branching mechanism. The agent will figure out that it can explore a completely new approach and rewrite that entire branch if it needs to.

## Implementation notes (2026-08-30 — ASA landed as the `asa` arm)

Code: `src/engine/optimizer/v3/asa/` (`asa_optimizer.py` hooks + `ASA_STRATEGY`,
`rules.py` pure rule engine, `branch_comparison.py` pure contribution / eligibility /
table, `prompt_sections.py` injected text, `skills/*_replacement.md`,
`optimization_entry.py`). Shared: `v3/pareto_front.py` (front + GEPA pruning, used by
Treatment B too). Tests: `tests/unit/test_asa_rules.py`, `test_branch_comparison.py`,
`test_pareto_front.py`, `test_asa_prompt_and_skills.py`; `tests/controls/test_asa_gate.py`,
`test_asa_close.py` (real git). E2E: `local_test_app.py asa_e2e` / `run_tests.sh
--include-asa-e2e` (local-only, **not yet run** — billable).

Decisions that refine or override the text above (the text above is kept as the design
narrative; where they differ, this section is what runs):

- **Violation allowance** = `ceil(closed_iterations / 10) − violations_so_far` (1 for
  iterations 1–10, 2 for 11–20, …). There is no `default_allowed_violation_count`.
  **Warn-then-refuse:** while the allowance is > 0 the baseline proceeds with a warning
  (`gate_note` on the run_eval response) and the allowance is held for the whole
  iteration; at 0 `run_eval` refuses **before any rows are spent**. The durable violation
  count is derived at close (`asa_violation` = a fired requirement not met by the derived
  facts), not from the warning.
- **The gate applies to the iteration's baseline only** (the first train/single eval);
  later train evals are ungated and comparison evals are never gated. Rule 1 therefore
  means **fork first, then baseline** — a baseline on a known branch is refused. Rule 2
  compliance = current branch is eligible, or its last close is an *ancestor* of HEAD
  (`derive_iteration_parent` via `"ancestor"` only — the merge-base fallback would
  attribute a fork from the primary to whichever eligible branch closed last).
- **Rule 1 switches off entirely after phase 1** (budget used > 50% AND ≥ 10 closed).
- **Primary refresh is loop-side only, no prompt injection.** Trigger: the primary's
  comparison record is older than `asa_primary_stale_landed_iterations` (default 4)
  *landed* iterations on the primary branch. `asa_on_iteration_closed` runs the counted
  comparison eval itself (`asa_loop_comparison_reason="stale_primary"`), never in the
  free tail and never when the close promoted a new primary.
- **Promotion** (judged on `primary_branch_after != primary_branch_at_start`, which the
  constraint gate resets on a force-revert — `is_primary` is not) with no record whose
  tree matches the new primary tip (exact or program-equivalent) → one counted
  comparison eval (`"promotion"`); `comparison_scores` is re-projected afterwards.
- **Seed = two lineages, one eval.** `asa_on_seed_complete` creates
  `codeevolver-{ts}-seed` from the seed commit, registers it non-primary beside the
  `-main` primary (`CEState.register_lineage`), runs ONE counted comparison eval on the
  seed content, and pushes the branch. Both lineages resolve to that record by tree.
- **Pareto score = GEPA Algorithm 2 with dominated pruning** (`pareto_front.prune_dominated`,
  GEPA `remove_dominated_programs` semantics since 2026-09-11: a lineage is off the front when
  every row it wins is also won by another surviving lineage — a union cover, no unique row —
  examined lowest comparison score first, one removal at a time, to a fixpoint; of two lineages
  with equal winning sets the lower scorer goes). Before 2026-09-11 the rule only dropped a
  lineage whose winning set was a strict subset of ONE other lineage's, which is far weaker.
  Lineages that resolve to the same record (same tree — the seed twins,
  program-equivalent branches) are ONE candidate for pruning: `build_table` prunes over one
  representative per record (primary first) and every twin carries its score, so the primary
  is never shown as dominated by its own identical twin. Applied to Treatment B's
  `Population.get_pareto_front` as well, so the two arms score identically. ASA's table shows pruned lineages as
  `n/a - fully dominated by other programs`.
- **Eligibility units.** `positive_behavioral_contribution` = Σ max(0, branch − reference)
  over shared rows (score differences, not a row count); `n_rows_ahead` is separate;
  normalized = contribution / comparison_set_size; threshold =
  `min(asa_behavioral_contribution_threshold_cap, asa_behavioral_contribution_constant ×
  (max_score − primary_comparison_score))` (defaults 0.05, 0.10; `max_score` from the
  optimize config). Eligibility is always vs the **primary** regardless of the tool's
  `reference`. Stale records still count; a lineage with no record is never eligible.
- **Lineage representation** always uses program-equivalence
  (`program_paths.diff_is_non_program_only`), never exact tree equality alone — the
  post-close `codeevolver.md` commit leaves every record one non-program commit behind
  its tip. `comparison_set_history.jsonl` lines now carry `tree` for this.
- **Force-commit before comparison evals** is ASA-only (`OptimizerStrategy.
  commit_before_comparison`); pareto_evolution does not do this (it maps by tree). Skipped
  on a detached HEAD.
- **Rule 3 (must explore)** is EE's explore guard unchanged; it may fire alongside rules
  1/2 (spec §User Message).
- **No lineage retirement**, per §Misc: the registry (not memory.md) is what the loop
  reads; dropping a branch from the memory.md pool does not retire it.
- **Budget-bypass check** lives in the run_integrity agent (all arms, flag-only):
  verdict keys `budget_bypass`, `budget_bypass_estimated_rows`, `budget_bypass_rationale`
  → record fields `integrity_budget_bypass*`.

Seams added for ASA (`v3/strategy.py`; all default to "off" so EE/pareto/freeform are
unchanged): `on_iteration_start` (renamed from `select_parent` — pareto forks the child
there, ASA computes the eligible set + fired requirements and writes
`{history_dir}/branch_comparison_iter{N}.md`), `run_eval_gate`, `extra_tools`
(builds `branch_comparison`; the tool is in `ARCHITECT_TOOL_SPECS` /
`ordered_architect_tool_names()` but both harnesses expose only registry members),
`commit_before_comparison`, and `LoopServices.config`. Prompt/user-message injection is
keyed on `IterationContext.asa_requirements` (only ASA's start hook sets it), with the text
in `asa/prompt_sections.py`. Records: `asa_requirements`, `asa_satisfied`, `asa_violation`,
`asa_eligible_at_start`, `asa_parent_is_eligible`,
`asa_parent_positive_contribution_to_primary_normalized`, `asa_allowance_left`,
`asa_free_tail`, `asa_phase1`, `asa_loop_comparison_reason`. Config:
`asa_behavioral_contribution_constant`, `asa_behavioral_contribution_threshold_cap`,
`asa_primary_stale_landed_iterations`. Fleet: `orchestrator/matrix.yaml` is 3 × 5 × 3 =
45 runs on five hosts, one arm per host — `asa` has `ce-e` to itself. Adding the
fifth host retired the per-seed `rotation` map and its list-valued entries in favour
of a flat `assignment`.

Note: records with `method="asa"` dated before 2026-08-27 are the old EE arm (renamed).
