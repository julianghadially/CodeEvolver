# CodeEvolver Tests

Tests are organized into **section folders**. `pyproject.toml` sets
`pythonpath = ["."]`, so `from tests.<section>.X import ...` and
`from tests._test_env import ...` resolve from any subfolder (no `__init__.py`
needed). `conftest.py`, `_test_env.py`, and `test_data/` stay at the `tests/`
root.

| Section | Folder | Contents |
|---|---|---|
| Unit | `tests/unit/` | Fast, deterministic, no SDK. |
| Hook & control | `tests/controls/` | Control/guard tests via `FakeCodingHarness` driving the real `IterationArchitect` (`_arch_helpers.py`). |
| Coding harness | `tests/coding_harness/` | Harness *component* tests (config/contract/factory/skill-staging/claude). |
| OpenCode | `tests/coding_harness/opencode/` | OpenCode-specific harness tests. |
| Functional agent | `tests/functional/` | Real-agent tests: close_iteration, session_resume, analysis_agent, inspect_traces, architect_trace capture, meta_reflection, holdout_guard, variance_tool, test_sampling, constraint_checker, live smoke, ee_e2e / pareto_e2e / freeform_e2e / simple_loop_e2e. |
| API | `tests/api/` | FastAPI endpoint / auth / lifecycle (most `integration`-marked). Hosted-service only: not in the public release; add it to the offline command where it exists. |

Markers: `unit`, `integration`, `live`, `functional`. The deterministic offline
suite is selected with `-m "not live and not integration"`.

## Running

**Offline suite (deterministic, no API keys, no spend):**

```bash
pytest tests/unit tests/controls tests/coding_harness -m "not live and not integration"
```

**Everything through one runner:**

```bash
scripts/run_tests.sh [--harness claude|opencode|both]
                     [--phase all|deterministic|functional]
                     [--include-e2e | --include-ee-e2e | --include-pareto-e2e
                      | --include-freeform-e2e | --include-simple-e2e
                      | --include-asa-e2e]
```

- Phase 1 (offline): the pytest command above. `--phase deterministic` runs
  only this phase.
- Phase 2 (functional, **billable**): constraint-checker and `test_sampling`
  (harness-independent: finalization draws a one-row held-out set 3x for the
  final program and 1x for the seed on the FactChecker fixture, needs
  `OPENAI_API_KEY`), then `close_iteration`, `session_resume`,
  `analysis_agent`, `inspect_traces`, `architect_trace`, `meta_reflection`,
  and `holdout_guard` (inspection + architect), swept over `claude` then
  `opencode` via `local_test_app.py`. Requires `ANTHROPIC_API_KEY` (claude) /
  `OPENAI_API_KEY` + `CODING_HARNESS_MODEL` (opencode); a harness pass is
  skipped when its creds are absent. `--phase functional` skips Phase 1.
- Phase 3 (e2e, opt-in per arm, **billable**): `--include-ee-e2e`,
  `--include-pareto-e2e`, `--include-freeform-e2e`, `--include-simple-e2e`
  and `--include-asa-e2e` select one arm each and combine; `--include-e2e` is
  shorthand for all five. Every arm is a real optimizer run against a real
  agent, so pick deliberately — e.g. `--phase functional --include-simple-e2e`
  runs the functional sweep plus the cheapest arm and nothing else. Arms you
  did not select still appear in the summary as SKIP with the flag that would
  enable them. `simple_loop_e2e` is bounded by `--max-metric-calls`, since
  that loop ignores `debug_max_iterations`.

The functional, e2e and `live` suites call real model APIs and cost money; only
the offline suite is free.

**A single functional test (real SDK; needs API keys):**

```bash
python local_test_app.py inspect_traces --coding-harness claude
python local_test_app.py test_sampling          # no harness; 4 single-row evals
python local_test_app.py variance_tool --coding-harness claude   # real session calls run_variance_estimate; 4 rows
python local_test_app.py meta_reflection --coding-harness opencode \
  --coding-harness-model openai/gpt-5.3-codex-spark
```

E2E (long-running, manual):

```bash
python local_test_app.py ee_e2e
python local_test_app.py pareto_e2e        # pareto_evolution arm
python local_test_app.py freeform_e2e      # freeform (Baseline 1) arm
python local_test_app.py asa_e2e           # ASA arm

# Simple loop (research Baseline 0) — bounded by eval rows, not iterations.
python local_test_app.py simple_loop_e2e --coding-harness opencode --max-metric-calls 40
```

### E2E seed branch and artifacts

All arms share one driver, `tests/functional/test_e2e.py::run_e2e`, and
seed from FactChecker's **`clean-slate`** branch — a single-commit orphan branch
holding only the seed program, the trainset and the holdout. `main` git-tracks
prior optimization output (`results/optimization/optimized_program_*.json`,
graded `results/evaluation_*.json`) which stays recoverable through history —
real contamination of the run's own results, so the branch choice stands
regardless of which inspectors are armed. The cached clone under
`~/.codeevolver/test-repos/` is swept before each run (stale `codeevolver-*`
branches deleted, reflog expired, `gc --prune=now`) for the same reason.

Each run arms two of the three inspectors a research run arms
(`runner/entrypoint.py::RESEARCH_DEFAULTS`): meta-reflection and run-integrity.
Pass `--no-meta-reflection` / `--no-run-integrity` for a cheaper one-off. The
run FAILS when an armed auditor breaks (no architect traces, no
`meta_reflection.md`, no integrity verdict) or on any constraint-gate fire. A
cheating classification from the run-integrity inspector is a **WARNING**, not a
failure: the local holdout fence is deny-paths only — bash is
allowed, the `ce-agent` uid fence exists only in the Docker runner — so an
architect that `head`s the holdout is model behaviour the e2e cannot prevent.
Pass `--fail-on-cheating` to make any non-`no_cheating` verdict fail the run.

**Clean-slate is OFF by default locally** (`--clean-slate` to arm it). A
research run gets a fresh container, so leftovers there mean something; a local
run shares one cached clone and keeps every past run's memory / traces / logs
under `~/.codeevolver/e2e/`, all reachable by the inspector's shell — it would
be reporting on *other e2e runs*, and a `contaminated` verdict aborts before any
eval spend. The Docker runner arms it (fresh container per run).
When it is armed, any status other than `clean` fails the run; an *absent*
status is reported as `n/a`, not a failure — `simple_loop` has its own
repository init and does not run that inspector.

Every run gets a `test_id` (override with `--test-id`) that becomes the
optimizer's `job_id`, so artifacts survive the test under
`~/.codeevolver/e2e/`:

```
logs/{test_id}.log                      full optimizer stdout
logs/{test_id}_detailed_agent_log.log   OpenCode architect tool calls + reasoning
logs/{test_id}_clean_slate_report.md
memory/{test_id}/                       memory.md, meta_reflection.md,
                                        run_integrity.md, iteration_history_*.md
traces/{test_id}/                       client OTel traces (iteration_{N}_{suffix}/ dirs)
architect_traces/{test_id}/             iteration_{N}_architect_trace/ (index.jsonl + turn_NNNN.json)
```

Nothing prunes that tree — it is the operator's to clean up.

## See also

`specs/engine3.md` (engine behaviour under test) and `specs/research_plan.md`
(the arms the e2e drivers exercise).
