# Sampling Variance Spec — the `run_variance_estimate` tool

Implemented 2026-08-31. See `specs/engine3.md` §"Run-variance estimate tool" for the cross-reference.

## Motivation
For the majority of our runs the architect ALWAYS finds out that it needs a better estimate of noise, and it usually tries to compute noise itself — but way too late, and limited in the budget it can spend on it (re-running the same tree on the 200-row comparison set, arguing over one-row deltas). To save the agent that trouble, we provide a one-time tool credit to calculate noise:

> CodeEvolver estimates stochastic evaluation variance using a fixed randomized subsample of training evaluation items, with the subsample set to the training subsample size (usually 50). Each is evaluated independently five times (250 total evaluations). We estimate within-item stochastic variance across repeated executions and use these variance components to estimate the expected run-to-run variance of the aggregate evaluation metric. This metric is provided as a heuristic. It assumes row-to-row independence, item-level homoskedasticity, and representativeness of the calibration sample. It knowingly does not hold up to candidate variance differences, or variance changes as the program progresses. Evaluator instability gets bundled into the estimate if an LLM judge is used.

The estimate is agent-invoked, not a seed-phase step: the seed program often scores ~40 and the first iteration hill-climbs into the 70s, so noise measured on the seed would describe a program that no longer exists. The agent decides when (the prompt recommends the end of iterations 1–3).

## Terms
* **Within-item variance:** the variance of one row's score across repeated executions of the same program on the same input (pooled over the sample's rows).
* **Run-to-run variance:** the variance of the aggregate subsample metric when the same sample is run multiple times. This excludes sampling variance (the variance from drawing different rows).

## Calculation
R = number of repeated runs (`variance_estimate_samples`, default 5), N = rows scored in every run.

Within-item variance for row i, and the pooled estimate:

    s_i² = (1 / (R − 1)) · Σ_r (Y_ir − mean_r(Y_i))²
    within_item_variance = mean_i(s_i²)

Run-to-run variance and standard deviation of the N-item metric:

    Var(mean(Y)) = within_item_variance / N
    run_to_run_std = sqrt(within_item_variance / N)

The R aggregate scores and their empirical sample std (`aggregate_std_empirical`) are recorded alongside. Rows that failed in any run are excluded (N counts only rows scored in every run).

## Feature
An architect tool `run_variance_estimate` (no arguments), available to every engine-3 strategy (ee, freeform, pareto_evolution, asa) but **not** the simple loop.

When called it:
1. auto-commits any uncommitted changes (so the recorded sha names the content measured);
2. draws ONE fresh random training sample of `subsample_size` rows — a new draw from the full trainset, not the iteration's locked subsample;
3. evaluates the current program on those same rows R times;
4. returns the score of each run, the within-item variance and the run-to-run standard deviation, plus the interpretation caveats above.

The runs consume eval budget (R × subsample_size rows). The tool may be used **once per run**: a second call in the same iteration is a structured refusal carrying the existing estimate (no rows spent), and the tool is absent from every later iteration's tool list. The result is saved to the run state and the iteration record so that all future system prompts receive it.

Decision (2026-08-31): no config-supplied variance override. It is not measured at seed, and the run should decide when to use the tool.

### Sub-sample note (system prompt)
The middle paragraph of the `### Sampling Guidance` section (`prompts._subsample_note`, rendered by `variance_estimate.variance_note`) has two variants.

Before the tool has run:
```
Subsamples when small (e.g., < 100 rows) can be quite noisy depending on the application, even when the seed is fixed - but especially when the seed differs between iterations. Given the variance above, use the training data to see examples and run quick tests, but be careful not to overfit. Ultimately, the application is judged by its ability to generalize beyond the training set.

You can and should estimate the run-to-run variance at least once, using the run_variance_estimate tool. This will run the program {variance_estimate_samples} times with a fixed sample of {subsample_size}, and the result is available to all future iterations. You will receive the within-item variance, the run-to-run standard deviation and the program score for each run. The best time to estimate run-to-run variance is after you have run a few improvement cycles. Usually at the end of iterations 1-3 would be a good time to run this tool. Running it on the seed program immediately is not recommended because if you then make quick improvements, the variance may differ on a program that scores 2x higher.
```

After:
```
Subsamples when small (e.g., < 100 rows) can be quite noisy depending on the application, even when the seed is fixed - but especially when the seed differs between iterations.

On this codebase, the program at iteration {iteration} and sha {sha} was run {n_runs} times with a fixed sample of size {subsample_size}.
The within-item variance was {within_item_variance}
The run-to-run standard deviation for the {n_rows}-item evaluation metric was {run_to_run_std}
The program scored [{aggregate_scores}]

Given the variance above, use the training data to see examples and run quick tests, but be careful not to overfit. Ultimately, the application is judged by its ability to generalize beyond the training set.
```
When every run produced byte-identical row outputs (the `holdout.samples_are_identical` cache detector), a warning line is added: an LLM response cache is probably active and the estimate understates the true variance.

## Revert-Check
For the revert check (`architect_watchdogs.decide_revert_check`), the threshold is the best prior score minus the run-to-run standard deviation: the watchdog fires only when the closing sha's train score is below `best − run_to_run_std`. Before the run has taken its estimate the check is unchanged (strict).

## Implementation notes
- **Module:** `src/engine/optimizer/v3/variance_estimate.py` — `estimate_variance` (pure statistics), `VarianceEstimate` (the recorded shape), `run_variance_estimate_tool` (the driver), `variance_note` (the two prompt paragraphs). The eval-sample constant and suffix helper live in `eval_log.py` (`VARIANCE_EVAL_SAMPLE = "variance"`, `variance_suffix(k)` → `variance`, `variance_2`, …) so `run_eval_tool` can import them without a cycle.
- **Eval identity:** the R runs go through `RunEvalBinding.run_variance_samples` → `_run_once(eval_sample="variance")`: counted against the budget, logged to `iteration_{N}_evals.jsonl` with per-row scores kept on the `EvalRecord`, trace dirs `iteration_{N}_variance[_k]/` (trainset split; agent-readable). Because nothing keyed on `TRAIN_EVAL_SAMPLES` or `"comparison"` reads them, they can never become the iteration's baseline, `subsample_best`, a revert-check reading, promotion evidence, a pareto-admission input or a close-check-5c satisfier; the memory-diff watchdog's "real work" count ignores them too. ASA's `run_eval_gate` fires only on the baseline train eval, so a variance run before the baseline bypasses it (it is not a baseline).
- **Budget:** each run is judged against the budget when it starts (`eval_budget.enforce_eval_budget("variance", …)` refuses at/over budget with its own message). A refusal or evaluator failure before the first run propagates; after that the completed runs are kept and the estimate is marked `truncated` (at least two runs are required, else no estimate is recorded — the rows already run are still counted).
- **Auto-commit:** `RunEvalBinding._auto_commit_dirty_tree("variance")` (the generalised ASA pre-comparison commit; skipped on a detached HEAD). On pareto_evolution the call is refused after the iteration's comparison run (`COMPARISON_CLOSED_NOTE`) — the comparison is the close step there and the auto-commit would otherwise move HEAD past the comparison point.
- **State / records:** `CEState.variance_estimate` (schema v6 → v7; `record_variance_estimate` refuses a second write). The loop stamps it onto every `IterationContext.variance_estimate` at construction (with `variance_estimate_samples` for the prompt text); `variance_estimate_ran` marks the iteration that took it, and only that iteration's close record / history record (`ITERATION_RECORD_FIELDS`, `IterationHistoryRecord.variance_estimate`, `CLOSE_RECORD_HISTORY_KEYS`) carries the dict.
- **Tool surface:** `architect_tools.ARCHITECT_TOOL_SPECS["run_variance_estimate"]` (empty schema), listed in `ordered_architect_tool_names()`; `architect_tool_registry.build_tool_registry` registers it only while `state.variance_estimate is None`. Both harnesses expose exactly the registry's members, so the tool disappears from later iterations without any harness change. The simple loop passes its own `{"run_eval": …}` registry and never sees it.
- **Knob:** `variance_estimate_samples` (default 5, `>= 2`) on `OptimizationConfig`, `OptimizeRequest`, `src/api/main.py`, `local_launcher`, `BaseAgentConfig` → `ArchitectConfig`, and pinned in `runner/entrypoint.py::RESEARCH_DEFAULTS` (5 × 50 = 250 rows = 5% of a 5000-row research budget). Internal — the agent never chooses it; e2e tests set 2.
- **Analysis agent:** its `eval_series` line explains `variance=` entries as noise readings, not before/after measurements.
- **Tests:** `tests/unit/test_variance_estimate.py`, `tests/controls/test_revert_check.py` (tolerance cases); functional `tests/functional/test_variance_tool.py` (`local_test_app.py variance_tool`, swept per harness by `run_tests.sh`): a real architect session against the FactChecker fixture is told "this is a test — call `run_variance_estimate` once"; `subsample_size=2`, `variance_estimate_samples=2` (four rows), real registry with the tool wrapped in a counter, no close tool / skills / watchdogs; asserts one successful call, the run state, two `variance` records + four counted rows, both trace dirs, and that the next iteration's prompt renders the numbers.
