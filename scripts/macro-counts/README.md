# macro-counts: macro approaches per run, counted against a curated catalog

Measures how much each arm explores. For each benchmark it counts how many
distinct macro approaches each run landed, against a catalog of macro approaches
that a human fixed in advance for that benchmark.

- **v5** (`catalog.py`) is the current method.
- **v4**, a pairwise ranking (`macro_counts.py`), is kept for reproducibility.
- **Prompt lessons** (`lessons.py`, `prompt_text.py`) reuse the same run
  pairing, workspaces and session code to count distinct lessons in LM-visible
  prompt text, which the macro definition treats as incremental.

All three are offline analyses over collected runs. They are not part of the
engine and are read-only against the run directories.

## Runs and pairing (`runs.py`)

Each run is a directory `<arm>-<bench>-s<seed>[-redo]` with at least
`history.json` and `repo.bundle`. `status.json` and `.codeevolver/memory/` are
used when present. Run directories are found under
`<root>/ce-run*-archive/ce-runs/<run>`, `<root>/ce-runs/<run>` or
`<root>/<run>`.

The two arms are `--arm1` / `--arm2`, by default `pareto_evolution` /
`freeform`. A `-redo` run supersedes the original of the same arm, benchmark and
seed. Runs that match `--void REGEX` are listed as unpaired and not judged. The
default regex is `^pareto_evolution-<bench>-s[123]$`, the original pareto runs
superseded by `-redo` runs; pass `--void ''` for none. `runs.py` knows the
program file of `hover`, `phantomwiki`, `ragqa`, `paperreview`, `arcagi2`,
`appworld` and `alfworld`. `--program-file BENCH=PATH` names it for another
benchmark.

## v5: curated catalog (`catalog.py`)

### Definition

A **macro approach** is a distinct, non-incremental change in how the program
solves the task, relative to the seed program. Score and diff size do not
affect the classification. `macro_definition.md` holds:

- the definition,
- a table of generic incremental changes (guards, checks, critics, LLM checking
  judges, tie-breaks, prompt and parameter changes, ...),
- a table of generic macro patterns.

The discover prompt includes all of it. The count prompt stops before the macro
patterns, so only catalog entries can be counted. The generic incremental table
is the only macro/incremental boundary: catalogs carry no per-benchmark
incremental examples.

A run's **count** is the number of catalog macro approaches the run *landed*. An
approach landed if its mechanism is in the closing commit of an iteration whose
`history.json` record has `changes_landed: true`, on any branch. Each approach
counts at most once per run, and one mechanism matches at most one catalog entry
(the most specific one). These never count:

- incremental changes,
- anything the seed program already does,
- non-incremental approaches missing from the catalog. These go in a separate
  `uncatalogued` list for the next curation pass.

The headline is each arm's mean ± sd over its seeds.

In `pareto_evolution` nearly every iteration lands on its own branch, whether
or not the loop later admits it to the population. In `freeform`, reverted
iterations do not land. "Any branch" is therefore more permissive for
`pareto_evolution` by construction.

### Steps (per benchmark)

0. **Fill** (`catalog.py fill --bench B`, before `count`). It writes two files
   under `inputs/<bench>/<run>/`:
   - `raw_change_summaries.md`: the run's `iteration_change_summaries.md`,
     verbatim.
   - `processed_change_summaries.md`: adds a section for every iteration the raw
     file has no heading for.

   An added section is the run's `iteration_<N>_change_summary.md` when one
   exists. Otherwise one OpenCode session per run writes the missing summaries
   from each iteration's diff (parent to closing commit, `fill_prompt.md`). The
   validator requires exactly one summary per missing iteration. Runs with no
   gaps get no session. `fill.json` records the source of each added section.
   `inputs/` is gitignored; `--force` rewrites it. `count` refuses a run without
   a processed file.
1. **Discover** (`catalog.py discover --bench B`). One OpenCode session reads
   all paired runs of the benchmark. Its prompt carries every run's change
   summaries. Through the workspace and bash it reaches the seed program, each
   run's `memory.md` and per-iteration summaries, and a git clone of each run.
   It proposes every macro approach it confirmed in the code, each with a
   reason it is macro.
   - **No provenance.** The output must not mention any run, arm, seed, job,
     branch, commit or iteration, or how many runs used an approach. The human
     curator therefore does not know where an approach came from.
     `validate_discovery` enforces this with patterns. A hit fails the attempt
     and starts a fresh session; softer wording becomes a warning in the draft.
   - Output: `drafts/<bench>.md` (review form, entries `M1..`) and
     `drafts/<bench>.json`.
   - `--salvage SESSION_DIR...` re-validates stored replies of finished sessions
     before starting a new one.
2. **Curate** (human). Copy the draft to `catalogs/<bench>.md` and edit it
   (merge, split, delete, promote or demote entries, renumber).
   `catalog.py check --bench B` validates it. The catalog is a method input like
   the prompts, so it is committed.
3. **Count** (`catalog.py count --bench B`). One OpenCode session per run,
   `--jobs` in parallel. Judging each run separately removes order and
   attention effects between runs. The prompt carries only that run's
   summaries and states its iteration count. It also lists the iterations with
   no summary heading, and the session must read the code of each one that
   landed. It returns:
   - `catalog_hits`: catalog ids the run landed, each with landed iterations
     and code evidence,
   - `uncatalogued`: landed non-incremental approaches that match no entry.

   `validate_count` retries in a fresh session when an id is unknown or
   repeated, or a hit lists no iterations. `apply_landed_check` then drops,
   deterministically from `history.json`, any hit that cites no landed
   iteration. Dropped hits are shown in the review file. Results are cached per
   run, and a report over every counted benchmark follows unless
   `--no-report` is given.

Neither step is blinded: the sessions may read scores in the run memories. Both
prompts state that score is irrelevant to whether an approach is macro.

### Catalog format (`catalogs/<bench>.md`)

```
# Macro-approach catalog: arcagi2

## Macro approaches

### M1: Program synthesis with demonstration verification
Description: The model writes a transform function that is executed and must reproduce every
demonstration pair before its output is used.
```

Only `## Macro approaches` is parsed. Each entry needs a `Description:`, which
may continue on the following lines. The count session sees only ids, names and
descriptions. An optional `Why macro:` line is accepted but never shown to it.
An `## Incremental examples` section or an `I<n>` entry is a `check` error. An
empty `## Macro approaches` section is valid, and every run then counts 0. Other
sections (`## Rejected`, `## Notes`) are ignored.

### Usage

```bash
python scripts/macro-counts/catalog.py fill     --bench arcagi2
python scripts/macro-counts/catalog.py discover --bench arcagi2 --dry-run   # workspace and prompt only
python scripts/macro-counts/catalog.py discover --bench arcagi2             # one session -> drafts/arcagi2.md
cp ~/ce-analysis/macro-counts/catalog-v5/drafts/arcagi2.md scripts/macro-counts/catalogs/arcagi2.md   # then curate
python scripts/macro-counts/catalog.py check    --bench arcagi2
python scripts/macro-counts/catalog.py count    --bench arcagi2             # one session per run, then the report
python scripts/macro-counts/catalog.py report                               # all counted benchmarks
python scripts/macro-counts/export_csv.py                                   # CSV export of the count results
```

`--bench` repeats. `--jobs N` (default 2) runs N sessions in parallel:
benchmarks for `discover`, runs for `count`. A cached result is reused while the
model, the full prompt, the workspace inputs (RUNS.md, seed, memory copies,
bundle hashes) and, for counts, the catalog are unchanged. `--force` re-runs.

Other flags:

- `--archive-root`: default `$CE_RUNS_DIR`, else `~/ce-runs`.
- `--out`: default `~/ce-analysis/macro-counts/catalog-v5`.
- `--catalog-dir`.
- `--seed N`: fill and count only; repeatable.
- `--arm1` / `--arm2`, `--void`, `--program-file`.
- `--model`: default `anthropic/claude-sonnet-5-5`.
- `--max-turns` (default 200), `--timeout` (seconds per session, hard kill),
  `--retries`, `--dry-run`, `--no-report`, `--salvage`.

`export_csv.py` reads `<results>/count/<bench>/*.json` (`--results`, default the
v5 `--out`) and writes `macro_counts_runs.csv` and
`macro_counts_approaches.csv` to `--out-dir` (default
`scripts/macro-counts/output/`, gitignored). It applies the same landed check.

Requirements: `opencode` >= 1.17.11, `rg` and `git` on `PATH`, and the judge
model's provider key (`ANTHROPIC_API_KEY` for the default) in the environment
or in a `.env` at the repo root or in this folder. Run from the repo root;
`judge.py` imports the engine's OpenCode harness.

### Outputs (`--out`)

| path | what |
|---|---|
| `COUNTS.md` | per benchmark: each arm's count mean ± sd and per seed, uncatalogued mean, and a run x catalog matrix (with landed iterations) |
| `counts.csv` | one row per run: count, counted / dropped ids, uncatalogued count, held-out scores, judge metadata |
| `review/<bench>.md` | per run: session metadata, counted hits with iterations and evidence, dropped hits, uncatalogued approaches, review flags |
| `drafts/<bench>.md`, `.json` | the discovery draft (review form) and the raw session result |
| `count/<bench>/<run_id>.json` | one run's raw count result, the catalog it used, hashes, usage, tool calls |
| `*.transcript.json` | exported OpenCode sessions (best effort) |
| `prompts/<bench>.discover.md`, `prompts/<bench>.count.<run_id>.md` | the exact prompts sent |
| `workspaces/<bench>/` | what the sessions could read: `RUNS.md`, `seed/`, `repos/<run>/`, `memory/<run>/` |

### Files

| file | role |
|---|---|
| `catalog.py` | fill / discover / check / count / report; prompts, validators, landed check, reports |
| `bench_workspace.py` | builds the per-benchmark workspace (`RUNS.md` iteration tables, clones, memory copies) |
| `runs.py` | run discovery, pairing (redo / void rules), git helpers; shared with v4 |
| `judge.py` | the OpenCode inspection session (`run_session`, `run_json_session`: fresh session per attempt, fenced-JSON parse, validate, retry) |
| `macro_definition.md`, `discover_prompt.md`, `count_prompt.md`, `fill_prompt.md` | definition and examples, and the task prompts |
| `catalogs/<bench>.md` | the curated catalogs |
| `export_csv.py` | CSV export of the v5 count results |

## Prompt lessons (`lessons.py`, `prompt_text.py`)

Macro counts classify every prompt edit as incremental, so they cannot see
experience that a run accumulates in its prompts. Prompt lessons measure that
layer. Two numbers per run: lessons over the **whole run** (present in any
landed program or the final program) and lessons in the **final program**.
Their difference is the number of lessons lost.

### Definition (`lesson_definition.md`)

A **prompt lesson** is one task-specific belief or decision rule, stated in text
the task LM sees, that the seed's prompts do not already contain.

- **LM-visible text** includes signature and instruction strings, field
  descriptions, docstrings of tools handed to an LM agent, and few-shot demos.
  Helper docstrings, comments, log strings and code logic are not prompt text.
  A lesson implemented only in code is not counted.
- **Not lessons**: generic advice, a step's role or plumbing, and pure style.
  Force and wording are irrelevant.
- **Unit**: one belief or one decision rule. Examples, the positive and negative
  form of a rule, copies across prompts, emphasis, thresholds and refinements
  merge into one lesson. Contradictory variants are separate lessons.
- **Grouping**: catalogs group lessons into themes.

### Pipeline

0. **Extraction** (`prompt_text.py`, deterministic). It parses the program
   package's Python files at a commit with `ast` and emits every string literal
   that could be prompt text, labelled with its location (file, enclosing
   class/def, assignment target or call keyword, or docstring). It is
   deliberately a superset, and the judges decide what is LM-visible. It writes
   these into the workspace:
   - `prompt_text/seed.md`,
   - `prompt_text/<run>/final.md` (final task file plus files changed since the
     seed),
   - `prompt_text/<run>/iterations.md` (per landed iteration: strings added,
     removed and changed relative to its parent).
1. **Discover** (`lessons.py discover`). By default this has two stages:
   - **Extract**: one tool-less session per run gets that run's text inline,
     with the run id replaced by "this run", and lists the run's lessons
     (`lesson_extract_prompt.md`).
   - **Consolidate**: one tool-less session merges the per-run lists, shuffled
     and labelled anonymously, into a theme-grouped draft with no provenance
     (`lesson_consolidate_prompt.md`).

   `--single-session` uses the earlier one-session discovery over all runs
   (`lesson_discover_prompt.md`).
2. **Curate** (human). Edit the draft into `lesson_catalogs/<bench>.md`
   (`## Lessons`, then `### T<n>` themes, then `#### L<n>` entries with `Label:`
   and `Description:`). `lessons.py check` validates it.
3. **Count** (`lessons.py count`). Each distinct program gets one blind session:
   the seed, every landed iteration's closing commit, and the final program.
   Programs with identical prompt text are judged once. The program's prompt
   text and the catalog are inline (`lesson_count_prompt.md`). The session
   returns a present/absent verdict for every lesson, with a verbatim quote when
   present. Missing ids fail validation; a quote not found verbatim is a
   warning. `--finals-only` judges only the seed and the final programs.
4. **Report** (`lessons.py report`). Per run, it computes whole-run and
   final-program lessons and excludes seed lessons. Counts are
   quote-reconciled: a lesson is also present in a program when a quote a judge
   gave for it in the same run appears verbatim in that program. Flicker
   measures reliability: the raw verdict is present in a parent and absent in
   the child although the parent's quote is still in the child. Parents are
   matched by `parent_iteration`.

Outputs: `LESSONS.md`, `lesson_counts.csv`, `lesson_matrix.csv`, drafts,
prompts and cached verdicts under `--out` (default
`~/ce-analysis/prompt-lessons`).

```bash
python scripts/macro-counts/lessons.py discover --bench hover
cp ~/ce-analysis/prompt-lessons/drafts/hover.md scripts/macro-counts/lesson_catalogs/hover.md   # then curate
python scripts/macro-counts/lessons.py check  --bench hover
python scripts/macro-counts/lessons.py count  --bench hover --finals-only   # seed + finals only
python scripts/macro-counts/lessons.py count  --bench hover --timeout 1200
python scripts/macro-counts/lessons.py report --bench hover                 # re-render from cached verdicts
```

Flags mirror `catalog.py`. `--archive-root` defaults to `$CE_RUNS_DIR`, else
`~/ce-runs`. `--jobs` defaults to 4, and `--dry-run` builds the workspace and
prompts and prints how many programs need judging. Extractions and verdicts are
cached by content hash, and editing a catalog invalidates its verdicts. The
engine harness leaves a per-session OpenCode config directory in the temp
directory. `lessons.py` points its temp directory at `<out>/tmp` and sweeps
stale config directories after each session.

## v4: pairwise ranked macro approaches (`macro_counts.py`)

Kept for reproducibility. For each benchmark and seed, v4 asks how many
distinct, far-reaching approaches the `pareto_evolution` run tried that the
`freeform` run did not, and vice versa.

### Definition

A macro approach is a distinct, non-incremental change in the solution relative
to previously attempted solutions. An LLM judge identifies each distinct
approach in each run, consolidating repeats within a run. It then ranks the
approaches of both runs together by how far they depart from the seed program,
leaving out any approach both runs implemented. `judge_prompt.md` holds the
instructions and two sets of examples:

- incremental changes (guards, format checks, critics, LLM checking judges).
  These end each list, because the judge stops listing at the first change it
  considers incremental.
- macro approaches in lower- and higher-departure bands, which calibrate the
  ranking.

### How one pair is judged

One read-only OpenCode session (`judge.py`) runs per pair per order, over a
workspace that `macro_counts.py` materializes
(`workspaces/<bench>-s<seed>__<order>/`):

```
INDEX.md                          the map: benchmark, seed, program file, iteration counts, every file listed
seed/<program file>               full seed source
A/iterations/NN/meta.md           parent iteration, landed or not, skipped
A/iterations/NN/summary.md        the run's own change summary (blinded) + commit-message headlines
A/iterations/NN/diff.patch        parent -> closing commit diff (program source only, never truncated)
A/iterations/NN/attempted.patch   for a reverted iteration: parent -> last eval snapshot (did NOT land)
A/iterations/NN/<program file>    full program source at that iteration's closing commit
A/final/<program file>            final program
B/...                             same
```

The session has read/grep/glob/list only: no bash, no edits, no skills, no
memory. It ends with one fenced JSON block holding:

- `program_a_approaches` and `program_b_approaches`: each program's approaches
  in rank order, ending at the incremental cutoff, at most `--max-per-program`
  (15) each,
- `combined_ranking`: both programs' approaches in one order, with shared
  approaches left out.

**The count** is how many of a run's unique approaches appear in the combined
ranking (`--top-k 0`, the default, uses the whole ranking). `--top-k N` counts
only the top N. The code computes the count; the model never reports one.

- **Both orders.** Every pair is judged twice: in `ab`, `pareto_evolution` is
  Program A, and in `ba`, `freeform` is. The headline is each arm's mean ± sd
  over seeds x orders. The table also shows means as Program A versus as
  Program B, and how often the orders disagree, so position bias is visible.
- **Blinding.** The judge sees only "Program A" and "Program B". The workspace
  carries no arm name, run id, branch, lineage or score. Search vocabulary in
  summaries and commit messages is replaced by `[redacted]` (`BLIND_WORDS`);
  code and patches are never altered. Raw run repos are never exposed.
- **Repeatability.** Each session gets a fresh OpenCode config and data
  directory, a pinned model and read-only tools. Tool calls and the exported
  transcript are stored, so the judge's reading path can be inspected.
  `compare_judgments.py` measures the effect of a re-judge. A session with no
  fenced JSON or an inconsistent judgment is retried in a fresh session
  (`--retries`, default 3).

### Usage

```bash
python scripts/macro-counts/macro_counts.py --archive-root ~/ce-runs --dry-run           # workspaces and sizes only
python scripts/macro-counts/macro_counts.py --archive-root ~/ce-runs --jobs 2            # judge every pair, both orders
python scripts/macro-counts/macro_counts.py --archive-root ~/ce-runs --runs 'alfworld-s2$'
python scripts/macro-counts/judge.py <out>/workspaces/alfworld-s2__ab --out <out>/judge/alfworld-s2__ab.json
python scripts/macro-counts/compare_judgments.py <out>/judge other/judge
```

`--archive-root` defaults to the parent directory of this repository, so pass
it explicitly for collected runs. `--out` defaults to
`~/ce-analysis/macro-counts`. Other flags:

- `--runs REGEX`, `--bench NAME`, `--arm1` / `--arm2`, `--program-file`,
  `--void`,
- `--top-k` (default 0), `--max-per-program` (default 15),
- `--stage build|judge|report|all`,
- `--model` (default `deepinfra/zai-org/GLM-5.2`; keep it fixed so counts are
  comparable), `--max-turns` (default 80), `--timeout` (default 7200 s, hard
  kill), `--retries` (default 3), `--jobs` (default 2), `--force`, `--dry-run`.

A cached judgment is reused only if its workspace hash, prompt hash, model and
`--max-per-program` all match. `--top-k` affects only the reports. The exit code
is non-zero if any judgment failed. The default model needs
`DEEPINFRA_API_KEY`; another `--model` needs its provider's key.

Outputs (`--out`):

- `MACRO_COUNTS.md`: per-arm headline counts, the per-seed table and the
  position check.
- `macro_counts.csv`: one row per pair and order.
- `review/<pair>.md`: both ranked lists, the combined ranking, the rationale,
  flags, and both runs' own iteration summaries.
- `judge/<pair>__<order>.json` and `.transcript.json`.
- `workspaces/`, plus `bundles/` and `opencode/` (caches).

Results are never committed.
