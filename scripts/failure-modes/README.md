# failure-modes: failure modes per iteration

Counts how often an architect agent makes predefined mistakes in handling its
own results (score claims, comparisons, memory, rules) within one iteration of
a CodeEvolver run. Each iteration is a unit. A read-only judge session reads one
iteration's evidence against fixed definitions and answers in JSON. Programs
check every quote the judge gives, join run, model and arm back in, and write
the tables. The session code (`judge.run_json_session`, the OpenCode harness,
`.env` key loading) is shared with `../macro-counts/judge.py`.

The counting pass is fm-v3 (`fm_v3.py`):

| script | definitions | prompts | judge decides | Python decides |
|---|---|---|---|---|
| `fm_v3.py` | `fm_definitions_v3.md` | `fm_v3_prompt.md`, `fm_v3_rethink_prompt.md` | MS, WR, FN (via a claim list), UR, BC | KR, UL, BO (`fm_facts.py`) |

Supporting modules: `fm_count.py` (shared helpers: text wrapping, git diffs,
harness routing, permutation tests, session environment and caching; its own
CLI is retired), `fm_runs.py` (shared run readers), `fm_facts.py` (harness
facts and the Python-decided modes of fm-v3), `anthropic_api.py` (a direct
Messages API call and the USD price table used to report session cost).

## Study design (`fm_runs.py`)

The runs are fixed in `fm_runs.core_runs()`: two architect models (`Qwen3`,
`GLM-5.2`), two arms (`freeform`, `pareto` = `pareto_evolution`), two
benchmarks (`hover`, `phantomwiki`) and seeds 1-3, giving 24 runs. Run folder
names are `<arm>-<bench>-s<seed>-qwen3c` for Qwen3, and `<arm>-<bench>-s<seed>`
for GLM-5.2 (with a `-redo` suffix for pareto). The module also holds the
study constants: comparison-set and train-subsample sizes, the noise thresholds
tau per benchmark (the SD of the difference between two evaluations of
identical code) and model-specific train SDs.

`fm_runs.py` reads only the raw run folders (`history.json`,
`history/<job>/*.jsonl`, logs, architect traces, `repo.bundle`) with the
standard library. It provides:

- a persistent `git cat-file --batch` reader with two tree identities: file
  content, and a normalized Python AST,
- a per-iteration log parser and an architect-transcript reader,
- memory.md classification and replay,
- `RunData`, a lazily loaded view of one run.

The only thing it writes is a scratch bare git repo per run under the private
directory. Unit ids are opaque, `U-` plus 8 hex digits of
`sha1("<run>|<iteration>")`. They are deterministic, so they keep run, model
and arm names out of the judge's workspace but are not secret.

## Common properties

- **Blinding.** No model, arm or run name is written into a judge workspace.
  The mapping from unit to run lives only in the private output directory and
  is joined in at the `report` stage.
- **Read-only judge.** The judge runs in a fresh OpenCode session per unit with
  `bash` denied by default. Its working directory is the unit's workspace.
- **Quote validation.** Every instance must quote the agent's own words. A reply
  is rejected, and a fresh session is started (up to `--retries`), when its
  shape is wrong or a quote is not found verbatim (whitespace aside) in the
  cited turn's assistant text or tool-call input, or in `memory_after.md` for
  memory quotes. Rejected attempts are kept.
- **At-the-time rule.** Each statement is judged only against what the agent
  had seen before it. Tool results, harness messages and file contents the agent
  merely read are not the agent's words. See the definitions files.
- **Caching.** Results are cached by model, prompt hash and workspace hash. A
  cache miss draws a new session and `--force` redraws.
- **Run-level statistics.** Iterations of one run are not independent, so the
  model comparison uses run-level shares and an exact two-sided permutation
  test on the difference of means (all 924 splits of 6 runs vs 6), per arm.
- **Judge model.** The default is `anthropic/claude-sonnet-5-5` (`--model`).
  A different judge is a different measurement.

## fm-v3: `fm_v3.py`

fm-v3 splits the modes between Python and the judge.

**Python-decided modes** (`fm_facts.py`). These follow from the eval logs, the
code identities and the memory tool calls, so they are exactly reproducible:

- **KR (kept a measured regression).** The closing code changed and the
  constraint checker did not revert it. The agent scored the closing code on
  the comparison set and saw the result. The mean of those scores is more than
  one comparison SD below the start code's comparison mean.
- **UL (unmeasured landing).** The closing code changed and was not reverted,
  the agent ran no comparison eval on it, and comparison credit was left at
  the end of the iteration.
- **BO (blind memory overwrite).** From iteration 2 on, with a non-empty
  `memory_before`, a full-file write of `memory.md` happened before the agent
  had read it, and at least half of its substantive lines are gone at the end.

For the judge modes, `fm_facts.py` computes candidates and context only, never
verdicts. These are the failed versions UR is about, the memory lines FN is
about, the rule text BC is about, and every score-like number in the agent's
visible words.

**Filtered trace.** Each iteration's architect trace passes through one section
filter. These are kept verbatim:

- every block the agent wrote visibly,
- every result of a harness tool (`ce_*`),
- every tool result of at most 1,000 characters.

Other tool results and code-edit bodies become a one-line stub naming the turn,
tool, size and a file under `turns/` with the full text.

**Pass A** (default, all iterations). The filtered trace goes inline in the
prompt of one read-only session per iteration (`fm_v3_prompt.md` +
`fm_definitions_v3.md`). The judge may open stubs (read) or search the full
turns (grep). It lists every score claim with a verdict (`ok`, `MS`, `WR`, `FN`
or `corrected`) and maps every score-like number Python found (`N<k>`) to a
claim or to `-`. A reply that skips a number or quotes words not found verbatim
is replaced by a fresh session. UR and BC are decided in the same session, with
BC only when the constraint checker fired. The agent's private reasoning is not
evidence in pass A. FN is judged but not reported, because it overlaps with MS
and WR.

**Pass B** (`--pass B`, needs pass A, only iterations that have reasoning
blocks). A second session gets pass A's claims, the tables and every assistant
turn with its reasoning and visible text inline (`fm_v3_rethink_prompt.md`). It
re-judges each claim. Instances remain claims in the visible output: the
reasoning can only change how a visible claim is read. A mistake that appears
only in the reasoning is recorded as `thinking_only` for information. Pass B
measures whether seeing the reasoning changes pass-A verdicts. It is a
sensitivity check, not part of the main counts. It runs as a plain API call
with no tools, because its evidence is inline.

| stage | output |
|---|---|
| `build` | `<out>/ws/<unit>/` (`turns/`, `memory_before.md`, `memory_after.md`, `code_diff.patch`, `attempted.diff`), `<out>/tables|inline|reasoning/<unit>.md` (prompt evidence), `<out>/meta/<unit>.json` (facts, quotable text, number ids; outside the workspace) |
| `count` | `<out>/count/` (pass A) or `<out>/rethink/` (pass B); with `--api-judge M`, `count_<M>/` and `rethink_<M>/` |
| `report` | `<results>/TABLE.md` (pass A: uncaught [all] per model x arm with run-level permutation tests; pass B: the same with re-judged verdicts and the A to B changes), `units.csv`, `instances.csv`, `claims.csv` |
| `export` | `<results>/judged_items.csv` (one row per judged item of every judge on disk, with a `current` flag for the prompt version) and `counted_instances.csv` |

```bash
python scripts/failure-modes/fm_v3.py build
python scripts/failure-modes/fm_v3.py count --dry-run
python scripts/failure-modes/fm_v3.py count --jobs 4 [--only U-xxxxxxxx,...]
python scripts/failure-modes/fm_v3.py count --pass B --jobs 4
python scripts/failure-modes/fm_v3.py report [--partial] [--f2 <dir>]
python scripts/failure-modes/fm_v3.py export
```

Flags: `--runs-root` (default `$CE_RUNS_DIR`, else `~/ce-runs`), `--out`
(default `~/ce-analysis/fm-v3`), `--results` (default `<out>/results`),
`--model`, `--only`, `--limit`, `--jobs` (default 4), `--max-turns` (default
100), `--timeout` (default 3600), `--retries` (default 3), `--allow-bash`
(gives the judge bash; off by default because bash can read outside the
workspace and break blinding), `--pass A|B`, `--api-judge MODEL` (runs both
passes as one plain API call to that model: OpenAI chat completions for `gpt*`
models, the Anthropic Messages API otherwise), `--force`, `--dry-run`,
`--partial` (report with iterations not yet counted).

fm-v3 puts the prompt on the command line (up to about 800 KB), so it needs
macOS or Linux.

## `anthropic_api.py`

`call()` makes one Messages API request with adaptive thinking, an optional
cached system block, and retry with backoff on transport errors, 408/409/429,
5xx and 529. `cost()` converts a usage dict to USD with the module's price
table. The fm-v3 report uses `cost()` to report session spend. The
key comes from the environment or the first `.env` (repo root, then
`scripts/macro-counts`) that has it, and its value is never printed.

## Requirements

- Python 3.11+ (the tools use the standard library) and the repo's engine on
  the import path.
- `opencode` >= 1.17.11 and `git` on `PATH`.
- `ANTHROPIC_API_KEY` in the environment or a `.env` at the repo root or in
  `scripts/macro-counts`. `OPENAI_API_KEY` is needed only for `fm_v3.py
  --api-judge gpt-...`.
- `scripts/macro-counts/judge.py`, imported for the session harness.

## Outputs

Everything goes under `~/ce-analysis/` by default and is never committed:
`fm-v3/` (workspaces, meta, judge records, transcripts, rejected attempts,
and scratch bare repos under `<out>/private`), and the results directories above. Workspaces and
meta files contain run evidence. Review them before sharing.
