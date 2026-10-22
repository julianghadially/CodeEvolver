# Best-of-population audit

An offline, per-researcher audit that asks how much performance is spread
across the candidate programs a search run produced but is absent from the
program the run finally returned. For each benchmark and arm (by default
`freeform` and `pareto_evolution`) it audits one completed run, shortlists five
of that run's candidate programs, picks a winner per difficult row from fresh
evaluations, and then re-evaluates those frozen winners independently to
estimate:

- **B**: verified best-of-population score on the difficult set,
- **F**: the final program's score on the same rows,
- **G = B - F**: the paired exploration gap,

each with a standard error, plus the evolution-minus-freeform difference in B.

The tool is not part of the engine. It never runs inside an optimization run,
never modifies candidate programs, and never starts a new search. It only
reads collected run directories and evaluates already-committed programs.

A larger gap does not by itself mean an arm explored better: it describes
performance distributed across candidates but not captured by the final
program. The audit covers one selected run per arm (the best recorded final
score), so it is a best-final-score-run analysis, not an average over runs.

## Inputs

- **Collected runs** under `$CE_RUNS_DIR` (default `~/ce-runs`); override or
  add roots with `--runs-root` (repeatable; the first root that has a run wins,
  later roots can only supply a missing bundle or trace directory). Each run
  directory provides `status.json`, `settings.json`, `candidates.jsonl`,
  `repo.bundle` and the job history under
  `.codeevolver/history/<job_id>/` (`row_scores.jsonl`, `datasets.json`).
  Run ids follow `<arm>-<bench>-s<seed>[-<label>]`. Directories that
  `scripts/results_tables.py` marks as superseded for a cell are never read.
- **Benchmark specs**: the client checkout, experiment directory and
  gitignored assets come from `orchestrator/benchmarks.yaml`; program factory,
  metric, dataset paths, input keys, thread count and eval timeouts come from
  `experiments/<experiment_dir>/input.py` (the `OPTIMIZE_CONFIG_LOCAL` dict by
  default, `--config-name` to choose another). The held-out test file is
  staged from the local client checkout.
- **API keys**: the evaluation stages run the client programs, so the
  environment must hold the provider keys the audited runs used. Preflight
  compares the environment against `settings.json["provider_keys"]` of each
  selected run and exits 2 if any is missing. Keys can be loaded from a file
  with `--dotenv PATH`.

## Stages

Every stage prints what it did. Frozen artifacts (selection, difficult set,
candidates, portfolio, winners) are loaded when present and only recomputed
with `--refreeze`.

### 0. Run selection (`select-run`, `select_run.py`)

Per arm, the completed run with a `repo.bundle` and the highest recorded final
held-out score: `status.json.final_test_score`, or, when that is missing, a
`posthoc_holdout_rescore.json` beside the run whose `program_sha` matches the
run's final sha. Ties go to the lower seed. The choice and its score source
are written to `selection/<bench>.json` before any evaluation.

### 1. Difficult set H (`difficult`, `difficult.py`)

Pools the seed program's held-out per-row scores (`row_scores.jsonl` lines
with `iteration == 0` and `split == "test"`) across every collected run of the
benchmark, all arms. Only compatible runs are pooled: the majority
(`datasets.json.testset_sha256`, `settings.json.models`) key is the reference
and other runs are excluded with a reason. Rows the evaluator flagged as
failed are recorded as failed observations and excluded from the estimate.

- If every observed value is 0 or a single positive maximum, the score is
  binary and each row gets a success probability; otherwise each row gets a
  mean normalized expected score.
- H = rows whose estimate is strictly below `--threshold` (default 0.5).
- One H per benchmark is shared by both arms. Each row's estimate and
  observation count are saved; sparse rows are flagged and rows with no seed
  observation are reported, not classified. An empty H marks the benchmark as
  not applicable.

Historical seed data only define H; they are never used as confirmation data.

### 2a. Materialize candidates (`materialize`, `candidates.py`, `materialize.py`, `environment.py`)

The candidate pool, with the same rule for both arms, is the `close_sha` of
every closed, non-skipped iteration (deduplicated by git tree) plus the final
program. For `pareto_evolution` this includes iterations that were never
admitted and population members that were later pruned. `--candidates-file`
(`{"<arm>": ["<sha>", ...]}`) supplies an explicit list instead.

Each run's `repo.bundle` is fetched into an audit-owned bare repo
`<out>/repos/<bench>.git` under `refs/jobs/<job_id>/*`. Candidates are checked
out detached into a pool of worktrees (one per worker). One `uv` venv is built
per distinct `requirements.txt` and symlinked into each worktree. The staged
held-out file's sha256 must equal the frozen H's `testset_sha256` (row ids are
positional) or the stage refuses to run; the hash is checked again after each
evaluation phase.

### 2b. Screening and top-5 portfolio (`screen`, `portfolio.py`)

Each candidate is evaluated once on H (draw 0). The observed Pareto front keeps,
per row, the candidates tied at the row maximum; it is then pruned with the
engine's `prune_dominated` (the same union-cover rule the `pareto_evolution`
arm uses during search), ordered by aggregate H mean. Observed dominance from a
single draw is a screening heuristic, not proof of dominance.

From the survivors, a greedy top-k (default 5, `--top-k`): start from the top
mean scorer, then repeatedly add the candidate with the largest additional
row-wise gain `sum_r max(0, s_c[r] - max_{p in portfolio} s_p[r])`, ties to the
higher mean, then the tree hash. If the final program is not selected it
replaces the last candidate added. If fewer than k survive, the portfolio is
backfilled from the pruned candidates by aggregate H mean. A candidate whose
evaluation still misses rows after one retry is logged as unstable and dropped.
The portfolio and the rule text are saved to `portfolio/<bench>-<arm>.json`.

### 3. Per-row winner selection (`select`, `winners.py`)

Every portfolio member gets `--reps-select` (default 5) fresh draws on H,
distinct from the screening draw. For each row, the winner is the candidate
with the highest selection mean; ties go to the higher all-of-H selection
mean, then the tree hash. Different rows can choose different programs. The
row-to-candidate mapping and each winner's selection mean are frozen in
`winners/<bench>-<arm>.json` before confirmation starts.

### 4. Confirmation (`confirm`, `execute.py`)

Each winner is re-evaluated `--reps-confirm` (default 5) more times on the rows
it won, grouped into one evaluation call per (winner, round), giving rounds with
exactly one observation per row of H. The final program is evaluated in the same
rounds on the rows it did not win. On rows it did win, its confirmation
observation is the winner's own, so those rows contribute exactly zero to the
paired gap. Selection draws are never reused and confirmation never changes
the winners.

### 5. Metrics and report (`report`, `metrics.py`, `report.py`)

For confirmation round r:

- `B_r` = mean over H of each row's frozen winner,
- `F_r` = mean over H of the final program,
- `D_r = B_r - F_r`.

`B`, `F` and `G` are the means over rounds. `SE(B)` and `SE(G)` are the sample
standard deviations of `B_r` and `D_r` (denominator R - 1) divided by sqrt(R);
the gap uses the paired `D_r`, not `SE(B)`. Approximate 95% intervals use the
t critical value for R - 1 degrees of freedom (2.776 at R = 5). Zero-variance
estimates are flagged. Sums (`|H| * B`, `|H| * F`) are reported beside the
means. Across arms, evolution minus freeform in B uses
`SE = sqrt(SE_evo^2 + SE_ff^2)`, treating the arms' draws as independent with
no round pairing. A selection-to-confirmation drop (mean selection score of
the chosen winners minus B) is reported as a diagnostic of selection optimism,
not as a correction.

The standard errors are conditional on the fixed rows, run, portfolio and
winners. They do not include run-to-run variability, shortlisting or selection
error, or uncertainty in the seed difficulty classification. The final
program's historical held-out score is reported separately and labelled with
its source.

## Execution details

- **Draw namespaces** (`planner.py`): screen = draw 0, select = draws
  1..reps_select, confirm and final = the following reps_confirm draws (round r
  is draw reps_select + r). Each (phase, draw) gets its own trace directory and
  its own `DSPY_CACHEDIR`.
- **Failures** (`execute.py`): a row the evaluator flagged as failed keeps the
  evaluator's score and is counted, never dropped. A whole-evaluation failure
  (timeout, crash) is retried once. A fill pass re-runs failed rows once under
  `<phase>_fill` and replaces only observations that were failed. Failed-row
  counts per phase appear in the results. A candidate still missing rows is
  dropped in screening and is a hard error (exit 3) in later phases. Rerun the
  stage to resume.
- **Identical-draw guard**: consecutive draws within a phase, and the
  boundaries between screen/select and select/confirm, are compared with the
  engine's `holdout.samples_are_identical` to detect response caching. It only
  warns. Findings are stored in the winners file.
- **Workers**: `--workers N` runs N spawned processes, one worktree slot each.
  Concurrent provider calls scale with workers times the benchmark's thread
  count.
- **Timeout**: `--eval-timeout` is in seconds per evaluation call and defaults
  to the benchmark's `comparison_eval_timeout`.

## Resumable row store (`store.py`)

`scores/<bench>/rows.jsonl`, `evals.jsonl` and `attempts.jsonl` hold every
observation keyed by `(job_id, tree, phase, row_id, draw)`. Planners return
only the keys that are missing, and the store is appended after every
evaluation call. An interrupted stage resumes where it stopped, and a stored
observation can never be counted again as a new repetition.

## Plan mode (`plan`, `estimate.py`)

`plan` runs no evaluations. It reports, per arm, the row evaluations still
needed:

```
|H| * (N_candidates + reps_select * k + reps_confirm) + reps_confirm * (|H| - rows the final wins)
```

It also reports a wall-clock estimate, serial and at `--workers`, with a
verdict of <= 1, <= 3, <= 5 or > 5 days. Seconds per row come from the store's
measured median once anything has run. Before that they come from the median
spacing of the audited runs' own final held-out draws. The plan is written to
`plan/<bench>.json`. Every evaluation stage also accepts `--dry-run`.

## Usage

Run from the repository root with the project's Python environment.
Evaluation stages (`screen`, `select`, `confirm`, `all`) are billable because
they call the client programs' model providers.

```bash
python scripts/best_of_population/run.py --help
python scripts/best_of_population/run.py <stage> --help

# Step by step (the --out default is ~/ce-analysis/bop)
python scripts/best_of_population/run.py select-run  --bench hover
python scripts/best_of_population/run.py difficult   --bench hover
python scripts/best_of_population/run.py materialize --bench hover --workers 3
python scripts/best_of_population/run.py plan        --bench hover --workers 3
python scripts/best_of_population/run.py screen      --bench hover --workers 3 --dry-run
python scripts/best_of_population/run.py screen      --bench hover --workers 3
python scripts/best_of_population/run.py select      --bench hover --workers 3
python scripts/best_of_population/run.py confirm     --bench hover --workers 3
python scripts/best_of_population/run.py report

# All stages in order; stops at the first non-zero exit
python scripts/best_of_population/run.py all --bench hover --workers 3 --dotenv path/to/.env

# Merge audit roots produced separately (e.g. different benchmarks on different machines)
python scripts/best_of_population/run.py aggregate --inputs ROOT_A ROOT_B --out MERGED
```

Common flags: `--out`, `--bench`, `--runs-root` (repeatable), `--arms`
(comma-separated), `--workers`, `--dotenv`, `--config-name`, `--reps-select`,
`--reps-confirm`, `--top-k`, `--threshold`, `--eval-timeout`,
`--include-void-seed` (also pool superseded runs into the seed estimate),
`--refreeze`, `--candidates-file`. Flags are persisted in `<out>/config.json`,
so later stages inherit them. Exit codes: 2 = preflight failure (missing keys
or assets), 3 = incomplete phase (rerun to resume).

Run `materialize --workers N` with the same N you will evaluate with, so all
worktree slots exist before parallel workers start.

`aggregate` copies the frozen artifacts and per-arm results from each input
root into the merged root, refuses on conflicting files, and regenerates the
summary.

### Workbook (`bop_sheet.py`)

```bash
python scripts/best_of_population/bop_sheet.py --results <out>/results --out <out>/results/bop_sheet.xlsx
```

Builds an Excel workbook only from the raw `results/<bench>-<arm>.json` files
(`--results` is repeatable). The sheets are `Rows` (per-row winner, confirmation
and final-program observations, the only raw numbers), `Rounds` and `BoP`
(every metric is a formula), `Inputs`, `Provenance` (input file hashes, run and
job ids, testset hash) and `Check`. Before writing, it recomputes every metric
in Python and exits 2 if any differs from the stored metrics or an observation
is missing. `--recalc PATH/recalc.py` with `--verify`, or `--verify-file` with
a recalculated copy, compares formula results against Python. Requires
`openpyxl`.

## Outputs

All outputs go under `--out` (default `~/ce-analysis/bop`). They are never
committed.

```
config.json                         persisted CLI configuration
selection/<bench>.json              stage 0: audited run per arm
difficult/<bench>.json              stage 1: H, per-row seed estimate and n, excluded runs
candidates/<bench>-<arm>.json       candidate pool
materialize/<bench>.json            materialization record (venvs, testset hash)
plan/<bench>.json                   evaluation counts and duration verdict
portfolio/<bench>-<arm>.json        screening scores, front, portfolio, rule text
winners/<bench>-<arm>.json          frozen row winners, selection means, guard findings
scores/<bench>/                     row store (rows / evals / attempts .jsonl)
traces/<bench>/                     per-draw evaluation traces
results/<bench>-<arm>.json          identities, every raw repetition, metrics, historical final score
results/<bench>-<arm>-rows.csv      row-level table for spreadsheets
results/summary.{csv,md,json}       one row per (bench, arm) with evolution - freeform in B
repos/, venvs/, worktrees/, logs/   working state
```

## Tests

Synthetic fixtures with no evaluator in `tests/unit/test_bop_*.py`. They cover
row winners, isolation of confirmation from selection, zero paired
contribution when the final program is the winner, sums versus means, SE and
CI, no systematic selection optimism under equal-probability candidates,
idempotent resume, fill semantics, run selection, H construction, portfolio
pruning, forcing and backfill, catalog and materialization.
