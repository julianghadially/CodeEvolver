# Macro-approach discovery 

You are cataloguing the solution approaches that autonomous coding agents tried on one benchmark.
CodeEvolver iteratively edits a task program to raise its score on an eval benchmark. Several
independent runs started from the **same seed program** on this benchmark. Your job is to find every
**macro approach** that any of these runs implemented, and to explain why each one is macro.

A human reviewer will turn your list into a fixed catalog for this benchmark. The reviewer must not
learn which arm an approach came from, so your output must say nothing about its source (see "Output
rules"). 

{definition}

## The material

Your working directory is a workspace for this benchmark. `RUNS.md` lists every run and has a table of
its iterations: parent, branch, closing commit (`close_sha`), and whether the change landed. It also
explains the git commands for reading an iteration's code. The layout:

- `seed/<program file>`: the seed program, the starting point of every run.
- `repos/<run>/`: a git clone of each run, with every branch and closing commit. You have bash; use
  `git show`, `git diff` and `git checkout` in these clones. Do not modify anything outside `repos/`.
- `memory/<run>/`: each run's own notes: `memory.md` (the agent's running memory),
  `iteration_change_summaries.md` and one `iteration_<N>_change_summary.md` per iteration.

The full `iteration_change_summaries.md` of every run is included below, before your task.

## What you do, in order

1. Read the seed program so you know what is already there.
2. Read every run's change summaries (below), in order, and note each candidate approach that looks
   non-incremental. Also note borderline changes that look new but are incremental.
3. **Confirm each candidate in the code.** For each one, find an iteration that implements it, and
   read the code at that iteration's closing commit, or the diff against its parent, in the run's repo
   clone. Summaries can overstate or misdescribe a change; when the summary and the code disagree,
   trust the code. Consult `memory.md` and the per-iteration summaries when the change summaries are
   unclear. Drop any candidate you cannot find in the code.
4. **Consolidate across runs.** When different runs implemented the same mechanism, possibly under
   different names, it is **one** entry. Also consolidate within a run:
   refinements, re-implementations and extensions of one mechanism are one entry.
5. Decide for each confirmed candidate: macro (goes in `macro_approaches`) or incremental (left out).
   Use the generic incremental table above: new guards, critics, checks, voting tweaks and fallbacks
   are incremental however new they look.
6. Name and describe each macro approach by its **mechanism**, generically enough that someone could
   check whether any program implements it. Say how it changes the way the task is solved. Do not
   mention function names or wording specific to one implementation.

Read the code, not just the summaries. Use as many tool calls as you need.

## Output rules

- **Say nothing about where an approach came from.** Do not mention any run, run id, arm, method,
  seed, job id, branch, commit, iteration or iteration number. Do not say how many runs used an
  approach, whether it landed or was reverted, when it appeared, or how it scored. Do not use phrases
  such as "one run", "several runs", "all runs", "later", "first" or "reverted". Describe only the
  mechanism and why it is macro. Any mention of provenance makes the whole answer invalid.
- Do not rank or score approaches.
- Each entry must be a single mechanism. Do not bundle two independent mechanisms into one entry.
- Try to represent each arm evenly. 

End your reply with exactly one fenced ```json block of the shape given below, and nothing after it.
