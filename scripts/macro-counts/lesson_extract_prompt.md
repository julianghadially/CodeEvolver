# Prompt-lesson extraction: one run

You are listing the lessons that an autonomous coding agent wrote into a task program's prompts during
ONE optimization run on one benchmark. CodeEvolver iteratively edits a task program to raise its score
on an eval benchmark; the run started from the seed program. Your job is to find every **prompt lesson**
this run put into the program's LM-visible text in any landed iteration. Your list will later be merged
with the lists of other runs into a catalog, so it must say nothing about where a lesson came from.

{definition}

## The material

Below the instructions are three texts, in full: the seed program's prompt-like strings, the run's final
program's prompt-like strings, and, for every landed iteration of the run, the strings ADDED, REMOVED and
CHANGED against its parent. Together these cover every prompt the run ever landed, on every branch.

The texts are extracted mechanically. Every string literal that could be prompt text is headed by its
location (file :: enclosing class/def :: assignment target, call keyword, or `docstring`). The
extraction is a superset: it also holds helper docstrings, log messages and format strings the LM never
sees. Decide from the location whether a string reaches the task LM.

## What you do, in order

1. Read the seed text so you know what the seed prompts already say.
2. Read the whole final-program text and the whole iteration history, from the first section to the
   last. A lesson counts if it appeared in any landed iteration, including ones later removed and ones on
   branches that did not lead to the final program.
3. Note each lesson in LM-visible text. Prompts of steps the agent added to the program (a planner, a
   reconciler, a sub-investigation, a continuation pass) count too, but only for what they teach about
   the task under the rules above. A step's role or plumbing ("you merge the lists below", "output a JSON
   list") is not a lesson, and the same procedure stated in several added steps is one lesson.
4. **Consolidate within the run.** Apply the unit and merge rules above: the same belief or rule written
   in different words, in different prompts, in different iterations or with different examples is ONE
   lesson. Refinements of one rule are one lesson. Group by the underlying point, not by the situation it
   is phrased for. For example:
   - **Completeness** is one lesson: "do not stop early", "do not assume the unexamined cases give the
     same result as the examined ones", "rare answers hide in the tail of a large set", "not finding more
     is not evidence there are no more".
   - **Wasted steps** is one lesson: "do not repeat an identical action or query", "do not re-read what
     you already have", "the step or call budget is limited, spend it on new information".
   - **Empty answers** is one lesson: "return a partial answer rather than nothing", "running out of
     steps is not a valid reason to answer empty or zero", "salvage what was established from an attempt
     that ended without an answer".
   Opposite variants of a rule (for example "X only" versus "X or Y") are separate lessons.
5. Give each lesson one label.
6. Write each lesson as the belief or rule itself, in your own generic words, so someone could check
   whether any program's prompts contain it. Do not quote the program's wording at length and do not
   mention variable or function names.

## Output rules

- **Say nothing about where a lesson came from.** Do not mention the run, its id, arm, method, seed, job
  id, branch, commit, iteration or iteration number, whether a lesson survived, was removed or reverted,
  when it appeared, or how it scored. Describe only the lesson.
- Do not rank or score lessons.
- Each entry must be a single lesson. Do not bundle two independent rules into one entry.
