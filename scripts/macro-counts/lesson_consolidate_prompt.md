# Prompt-lesson consolidation

Several independent optimization runs on one benchmark each produced a list of the **prompt lessons**
they wrote into a task program's LM-visible text. The lists are below, in random order, labelled only
by number. Your job is to merge them into ONE catalog of distinct lessons for this benchmark, grouped
into themes. A human reviewer will curate your catalog, so it must say nothing about which list a lesson
came from.

{definition}

## What you do, in order

1. Read every list in full.
2. **Merge.** Apply the unit and merge rules above across lists: the same belief or rule described in
   different words in different lists is ONE lesson. Refinements of one rule are one lesson. Group by the
   underlying point, not by the situation it is phrased for. For example:
   - **Completeness** is one lesson: "do not stop early", "do not assume the unexamined cases give the
     same result as the examined ones", "rare answers hide in the tail of a large set", "not finding more
     is not evidence there are no more".
   - **Wasted steps** is one lesson: "do not repeat an identical action or query", "do not re-read what
     you already have", "the step or call budget is limited, spend it on new information".
   - **Empty answers** is one lesson: "return a partial answer rather than nothing", "running out of
     steps is not a valid reason to answer empty or zero", "salvage what was established from an attempt
     that ended without an answer".
   Opposite variants of a rule (for example "X only" versus "X or Y") stay separate lessons. Do not merge
   two lessons only because they share a topic.
3. Drop an entry only if it is not a lesson under the definition (generic advice, plumbing, pure style).
   Do not drop a lesson because only one list has it.
4. Give each lesson one label and put it in one theme. Choose themes that fit this benchmark; aim for
   themes that each hold several lessons.
5. Write each lesson as the belief or rule itself, in generic words, so someone could check whether any
   program's prompts contain it.

You do not need any tool; everything is below.

## Output rules

- **Say nothing about where a lesson came from.** Do not mention list numbers, runs, arms, seeds,
  iterations, how many lists contain a lesson, whether it was removed, or how it scored. Describe only
  the lesson.
- Do not rank or score lessons.
- Each entry must be a single lesson. Do not bundle two independent rules into one entry.
