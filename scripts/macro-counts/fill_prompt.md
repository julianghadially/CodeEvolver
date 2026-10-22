# Fill missing change summaries

You are documenting one optimization run of CodeEvolver on one benchmark. CodeEvolver is an autonomous
coding agent that iteratively edits a task program to raise its score on an eval benchmark. After each
iteration the run writes a change summary, but some iterations have none. Your job is to write the
missing change summaries from the code.

## The material

Your working directory is a workspace for this benchmark. Use only the run named in your task: its
section of `RUNS.md` and its repo clone `repos/<run>/`; ignore the other runs. You have bash; use
`git show` and `git diff` in the clone. Do not modify anything.

Below are the run's existing change summaries (for context and style), then, for each iteration that
lacks one, the diff from its parent commit to its closing commit. A diff may be truncated; when it is,
read the rest with `git -C repos/<run> diff <parent> <close> -- <file>`.

## What you write

For each missing iteration, one change summary in the style of the existing ones:
- Describe what the change does to **how the program works**: which steps, calls, data structures,
  rules or prompts it adds, removes or changes, and how they interact. Name the functions or classes.
- Describe only what the diff shows. Do not guess at intent beyond the code, and do not describe
  earlier or later iterations.
- Say nothing about scores, whether the change helped, or whether it is a big or small change.
- If the diff is empty or only touches notes, docs or tests, say so in one sentence.
- Three to eight sentences.
