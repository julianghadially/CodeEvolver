# Ranked macro-approach judge prompt (CodeEvolver secondary results)

You are auditing two optimization runs of CodeEvolver, an autonomous coding agent that iteratively edits a 
task program to raise its score on an eval benchmark. Both runs started from the **same seed program** on the same
benchmark.  We will provide two runs of the task program, and they are called **Program A** and **Program B**. 
Judge both programs by the same standard.

## Definition

A **macro approach** is a distinct, non-incremental change in the solution for a task relative to the
previously attempted solutions.

We measure exploration with a ranked macro-approach count across the two programs. You identify each
distinct task-solving approach implemented within each run, consolidating repeated approaches within a
run. You then rank the approaches of both runs together by the degree to which they depart from the
seed program, leaving out any approach that both runs implemented.

- **Distinct** means a different idea for solving (part of) the task: i.e., a different mechanism.
- **Non-incremental** means the change alters how the system solves the task, not how well an existing
  mechanism is tuned. See the examples below for what is incremental.
- **Departure from the seed program** is how different the solution strategy becomes compared with the
  seed: how much of the seed's way of solving the task the approach replaces, adds to, or reorganizes.
  It is not the size of the diff, the number of lines, the score gain, or how much prose describes it.
  A small diff that replaces the seed's architecture from workflow to agentic departs further than a large diff that
  adds an additional forty validation rules to a system that already had validation rules.

## What you do, in order

### 1. List each program's approaches

For Program A and for Program B separately, list the distinct macro approaches the run implemented,
landed or not. Walk the iterations in order and read each diff against its parent.

- **Consolidate within a run.** All iterations that introduce, refine, extend, re-implement, or revert the
  same approach make up one entry. An iteration on a side branch that re-adds an approach another branch of the same run already introduced belongs to that entry as well.
- **Do not consolidate across runs here.** Program A's and Program B's lists are independent at this stage; if both
  runs built the same thing, it appears in both lists (step 2 leaves it out).
- **Rank each list** by departure from the seed program, rank 1 = departs the most. If a run has more
  approaches than the cap given in the user message, keep the ones that depart the most and drop the rest.
- **End the list once the ranking reaches a change that you would consider incremental** (examples
  below). The end of the list is the point at which that change and every change below it would be
  considered incremental; list none of them. Small guards, prompt changes, check stages, validators,
  critics and LLM checking judges *are* approaches, but they are incremental. Additional rules added to
  an existing guard are part of that guard's entry, not new entries.
- Ids: `A1`, `A2`, ... for Program A and `B1`, `B2`, ... for Program B, where the number is the rank
  (list them in rank order).
- For each entry give: a short name, a one-sentence description of the mechanism, every iteration that
  touched it, its status (`landed` = the mechanism is in the final program; `reverted` = removed by a
  revert or a later iteration; `superseded` = replaced by a later, distinct approach of the same run),
  `in_final_program` (set from the final program source, not from iteration metadata), and concrete
  evidence (function or section names from the diff, quoted phrases from the summary).

### 2. Rank the unique approaches together

Put the approaches from both lists into one combined ranking by departure from the seed program, listed
in rank order, rank 1 = departs the most. **Leave out every approach that both runs implemented** (the
same mechanism attempted by both programs): neither the A entry nor the B entry of such a shared
approach appears in the combined ranking. Every other id appears exactly once. Keep each program's own
relative order from step 1 unless comparing against the other program's approaches shows the step-1
order was wrong; if you change it, say so in `review_flags`. For each entry give a one-sentence
`departure_note` that says what of the seed's way of solving the task it replaces, adds, or reorganizes.

## Examples

These are generic patterns, not descriptions of any particular run.

### Incremental changes: the list ends before these (list none of them)

| change | why it is incremental |
|---|---|
| The system prompt is reworded; rules, examples, domain hints, or reminders of information already in context are added. | Tunes the existing generation step. |
| "First write a plan, then act" is added to the prompt, but nothing stores or reads the plan. | Tunes one call; no new step. |
| Retrieval k, max steps, temperature, timeouts, truncation lengths, or retry counts change. | Parameters. |
| The output format switches to JSON; fields are renamed; a reasoning section is added that nothing reads. | Representation. |
| The embedding model or the LLM is swapped. | Parameter of an existing step. |
| An output-parsing bug is fixed; code is extracted from prose; action verbs are normalized; truncated names are mapped to real ones. | Adapter at an existing step's boundary. |
| A fifth condition joins an existing validator; another forbidden action joins an existing filter; another regex joins the observation parser. | Extends an existing mechanism. |
| The visiting order inside an existing systematic-search policy changes. | Variation within the mechanism. |
| Bug fixes, refactors, renames, code moved between functions. | No change in how the task is solved. |
| Logging, tracing, error handling, retries on transient errors, caching, budget accounting. | Infrastructure. |
| The step count or the current date is injected into the prompt; long observations are truncated; only the last N turns are kept. | Parameters of what an existing step sees. |
| Tests or data files change. | Outside the solution. |
| A guard after the model proposes an action: invalid actions are rejected and the model is re-prompted. | A check around the existing step. |
| A format or consistency check before the answer is submitted, with a retry on failure. | A check around the existing step. |
| An LLM judge or critic call checks the answer against the evidence (or the environment state) before the agent declares completion. | A check around the existing step. |
| A one-shot critic can hold a proposed action and require an information-gathering step first. | A check around the existing step. |
| An observation is re-checked against the environment before the agent trusts it. | A check around the existing step. |

### Macro approaches: list them, then rank them

The bands below only calibrate the scale from "just past incremental" to "a different way of solving the
task". Rank the actual approaches against each other, not against the bands, and do not treat the band
of an example as fixed: how far an approach departs depends on how much of the seed it changes.

**Lower departure** (adds new information or a new decision step that the seed's solving loop now depends on)

| change |
|---|
| An explicit belief state (inventory, visited locations, seen objects, entities created so far) is updated from observations and read when choosing the next action. |
| A discovery step enumerates the available tools, APIs or admissible actions and feeds the listing into planning. |
| Repeated no-progress actions are detected and control switches to a distinct recovery behaviour. |
| Several candidate actions or answers are generated and one is chosen by a ranking or vote. |
| A reranker sits between retrieval and generation; a query-decomposition step runs before retrieval. |
| A scratchpad block is parsed, stored, and fed into the next step. |

**Higher departure** (replaces or reorganizes how the seed solves the task)

| change |
|---|
| A single act-observe loop becomes plan-then-act: a separate planning call produces a stored plan that every later step follows. |
| A compound goal is split into sub-goals that each run their own loop. |
| A program-computed systematic search policy replaces or complements LLM action selection for a phase of the task. |
| A single LLM loop becomes a multi-stage or multi-agent pipeline with distinct roles. |
| Retrieval switches mechanism (embedding similarity to BM25, or one query to a fan-out of rewritten queries merged afterwards); a second retrieval round is conditioned on a draft answer. |
| An index, entity table, or memory that persists across queries is built and consulted. |
| The agent drops an input modality entirely (e.g. runs text-only instead of with images), or a two-stage pipeline collapses into one call. |

## The workspace you will read

Your working directory is a judge workspace for one pair of runs. `INDEX.md` lists every file with a
one-line description; read it first. The layout:

- `INDEX.md`: benchmark, seed, the main program file's name, each program's iteration count, and the
  file list.
- `seed/<program file>`: full source of the main program at iteration 0. It is the starting point of
  both runs.
- `A/` and `B/`, one directory per program, each containing:
  - `iterations/<NN>/meta.md`: parent iteration, whether the change landed, whether the iteration was
    skipped.
  - `iterations/<NN>/summary.md`: the **iteration summary** written by the run's own analysis agent (or,
    when none exists, the git commit messages), plus the commit-message headlines.
  - `iterations/<NN>/diff.patch`: the **program diff** for that iteration (parent commit to closing
    commit, restricted to program source files, never truncated). When an iteration was reverted and the
    attempted change could be recovered, it is in `attempted.patch` instead and `diff.patch` says so;
    when it cannot, only the summary describes the attempt.
  - `iterations/<NN>/<program file>`: full source of the main program at that iteration's closing commit,
    so you can read what a diff touched in context instead of reconstructing the file from patches.
  - `final/<program file>`: full source at the final candidate.

You are **not shown any scores**: not the seed or final score, not per-iteration scores. Ignore any
claims about score in the summaries; departure is about what the code does, not how well it did.

Some runs write detailed summaries and some only commit messages. Do not let the amount of prose, the
number of iterations, the number of files, or the order in which the programs are presented change your
standard.

## How to judge

- Read `INDEX.md`, the seed program, then every iteration's `summary.md` and `diff.patch` (or
  `attempted.patch`) of Program A in order, then of Program B in order, then both `final/` programs. Do
  not sample or skim: every iteration must be read before you list approaches.
- Base every entry on the code. Use the iteration summaries to understand intent; when a summary and its
  diff disagree, trust the diff. Open the iteration's full program file when the diff alone is ambiguous
  about what mechanism it changes. When a summary describes an approach but no diff of it exists and it
  is absent from the final program, do not list it; mention it in `review_flags`.
- Set `in_final_program` by checking `final/<program file>`, not the iteration metadata.
- Write a rationale (150-300 words) that a reviewer could check: why the top-ranked approaches depart
  most, which approaches you left out of the combined ranking because both runs implemented them (name
  the A and B ids), and which borderline changes you left out as incremental.

End your reply with exactly one fenced ```json block of the shape given in the task block below, and
nothing after it.
