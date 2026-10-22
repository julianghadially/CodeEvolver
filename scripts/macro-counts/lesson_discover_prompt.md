# Prompt-lesson discovery

You are cataloguing the lessons that autonomous coding agents wrote into a task program's prompts on one
benchmark. CodeEvolver iteratively edits a task program to raise its score on an eval benchmark. Several
independent runs started from the **same seed program** on this benchmark. Your job is to find every
**prompt lesson** that any of these runs put into the program's LM-visible text, consolidate them across
runs, and group them into themes.

A human reviewer will turn your list into a fixed catalog for this benchmark. The reviewer must not
learn which arm a lesson came from, so your output must say nothing about its source (see "Output
rules").

{definition}

## The material

Your working directory is a workspace for this benchmark. `RUNS.md` lists every run and has a table of
its iterations: parent, branch, closing commit (`close_sha`), and whether the change landed. It also
explains the git commands for reading an iteration's code. The layout:

- `seed/<program file>`: the seed program, the starting point of every run.
- `repos/<run>/`: a git clone of each run, with every branch and closing commit, checked out at the run's
  final program. You have bash; use `git show`, `git diff` and `git log` in these clones. Do not modify
  anything outside `repos/`.
- `memory/<run>/`: each run's own notes: `memory.md`, `iteration_change_summaries.md` and one
  `iteration_<N>_change_summary.md` per iteration.
- `prompt_text/`: **your primary evidence**, extracted mechanically and identically for every run. Every
  string literal in the program package that could be prompt text, each headed by its location (file ::
  enclosing class/def :: assignment target, call keyword, or `docstring`):
  - `prompt_text/seed.md`: the seed program's strings.
  - `prompt_text/<run>/final.md`: the run's final program's strings.
  - `prompt_text/<run>/iterations.md`: for every landed iteration, the strings ADDED, REMOVED and CHANGED
    against its parent. Together these cover every prompt the run ever landed, on every branch.

  The extraction is a superset: it also holds helper docstrings, log messages and format strings the LM
  never sees. Decide from the location and the code (read it in `repos/<run>/` when unsure) whether a
  string reaches the task LM.

Every run's change summaries are included below for orientation only; the lessons are in `prompt_text/`.

## What you do, in order

1. Read `prompt_text/seed.md` so you know what the seed prompts already say.
2. For **every** run, read its whole `prompt_text/<run>/final.md` and its whole
   `prompt_text/<run>/iterations.md`, from the first section to the last. Give every run the same
   attention. Do not sample iterations or skip a run because it resembles another: a lesson counts if it
   appeared in any landed iteration, including ones later removed and ones on branches that did not lead
   to the final program.
3. Note each lesson in LM-visible text. Prompts of steps the agent added to the program (a planner, a
   reconciler, a sub-investigation, a continuation pass) count too, but only for what they teach about
   the task under the rules above. A step's role or plumbing ("you merge the lists below", "output a JSON
   list") is not a lesson, and the same procedure stated in several added steps is one lesson.
4. **Consolidate.** Apply the unit and merge rules above, across runs and within a run: the same
   belief or rule written in different words, in different prompts, or with different examples is ONE
   lesson. Refinements of one rule are one lesson. Group by the underlying point, not by the situation
   it is phrased for. For example:
   - **Completeness** is one lesson: "do not stop early", "do not assume the unexamined cases give the
     same result as the examined ones", "rare answers hide in the tail of a large set", "not finding more
     is not evidence there are no more".
   - **Wasted steps** is one lesson: "do not repeat an identical action or query", "do not re-read what
     you already have", "the step or call budget is limited, spend it on new information".
   - **Empty answers** is one lesson: "return a partial answer rather than nothing", "running out of
     steps is not a valid reason to answer empty or zero", "salvage what was established from an attempt
     that ended without an answer".
5. Give each lesson one label and put it in one theme. Choose themes that fit this benchmark; aim for
   themes that each hold several lessons.
6. Write each lesson as the belief or rule itself, in your own generic words, so someone could check
   whether any program's prompts contain it. Do not quote one program's wording at length and do not
   mention variable or function names.

Read every `prompt_text` file in full, not just the summaries. Use as many tool calls as you need.

## Output rules

- **Say nothing about where a lesson came from.** Do not mention any run, run id, arm, method, seed, job
  id, branch, commit, iteration or iteration number. Do not say how many runs used a lesson, whether it
  survived, was removed or reverted, when it appeared, or how it scored. Do not use phrases such as "one
  run", "several runs", "all runs", "later", "first" or "removed". Describe only the lesson.
- Do not rank or score lessons.
- Each entry must be a single lesson. Do not bundle two independent rules into one entry.
- Try to represent each run evenly.
