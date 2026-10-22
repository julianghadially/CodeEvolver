## Definition

A **macro approach** is a distinct, non-incremental change in how the program solves the task.

- **Distinct** means a different idea for solving (part of) the task: a different mechanism.
- **Non-incremental** means the change alters *how* the system solves the task, not how well an
  existing mechanism is tuned. See the examples below for what is incremental.
- **Relative to the seed.** Anything the seed program already does is not a macro approach, however
  it is later reworked. Read `seed/<program file>` before deciding.
- **Scores are irrelevant.** Whether a change is macro has nothing to do with its score, whether it
  helped, whether it was kept, or how much the run's notes praise it. The run memories and summaries
  mention scores; ignore them for this judgment. A macro approach that scored badly is still macro; a
  prompt tweak that raised the score a lot is still incremental.
- **Size is irrelevant.** Not the size of the diff, the number of lines, or how much prose describes
  it. A small diff that turns a single call into a plan-then-act pipeline is macro; a large diff that
  adds forty rules to an existing validator is incremental.

## Incremental changes (never macro)

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
| A tie-break, fallback or disagreement-resolution rule is added to an existing voting or selection step. | Extends an existing mechanism. |

## Macro approaches (generic patterns)

These are generic patterns, not descriptions of any particular run.

| change |
|---|
| An explicit belief state (inventory, visited locations, seen objects, entities created so far) is updated from observations and read when choosing the next action. |
| A discovery step enumerates the available tools, APIs or admissible actions and feeds the listing into planning. |
| Repeated no-progress actions are detected and control switches to a distinct recovery behaviour. |
| Several candidate actions or answers are generated and one is chosen by a ranking or vote. |
| A reranker sits between retrieval and generation; a query-decomposition step runs before retrieval. |
| A scratchpad block is parsed, stored, and fed into the next step. |
| A single act-observe loop becomes plan-then-act: a separate planning call produces a stored plan that every later step follows. |
| A compound goal is split into sub-goals that each run their own loop. |
| A program-computed systematic search policy replaces or complements LLM action selection for a phase of the task. |
| A single LLM loop becomes a multi-stage or multi-agent pipeline with distinct roles. |
| The model writes a program that is executed and verified instead of answering directly. |
| Retrieval switches mechanism (embedding similarity to BM25, or one query to a fan-out of rewritten queries merged afterwards); a second retrieval round is conditioned on a draft answer. |
| An index, entity table, or memory that persists across queries is built and consulted. |
| The agent drops an input modality entirely (e.g. runs text-only instead of with images), or a two-stage pipeline collapses into one call. |
