## Definition

A **prompt lesson** is one task-specific belief or decision rule, stated in text the task LM sees, that
the seed program's prompts do not already contain.

- **Text the task LM sees** means: signature / instruction strings, input and output field
  descriptions, the docstrings or descriptions of tools handed to an LM agent (e.g. the tools of a
  `dspy.ReAct`), few-shot demonstrations, and any other string the program sends to the LM. Python
  comments, docstrings of helper functions that are never shown to the LM, log messages and code logic
  are NOT prompt text, even when they encode the same knowledge. A lesson implemented only in code is
  not a prompt lesson.
- **Task-specific** means it says something about this task, its data, its tools or how it is scored.
  Generic advice that would fit any task ("think step by step", "be careful", "be efficient",
  "double-check your answer") is not a lesson.
- **Relative to the seed.** Anything the seed program's prompts already say is not a lesson.
- **Scores are irrelevant.** Whether a lesson helped, was kept, or is praised in the run notes has
  nothing to do with whether it is a lesson.
- **Force and wording are irrelevant.** A rule ("MUST"), a guideline ("prefer"), a hint ("consider") and
  a warning ("a common mistake is ...") are all lessons on equal footing. Capital letters, "CRITICAL" and
  repetition do not make something a lesson or a separate one.

## The unit: one lesson = one belief or one decision rule

Two statements are the **same lesson** when following either one leads to the same behaviour. These
never create a separate lesson; they belong to the lesson they illustrate or repeat:

| merged into the lesson it belongs to | example |
|---|---|
| Examples and worked instances | "e.g. 'Volvo S40', not 'Did the Volvo S40 share its chassis?'" illustrates "write a keyword phrase, not a question" |
| The positive and negative form of one rule | "do not combine several entities in one query" = "target one entity per query" |
| The same text in several prompts, fields or modules | one rule copied into a planner prompt and an executor prompt |
| Rewording, emphasis, restating later | "CRITICAL:", capitals, a closing reminder of an earlier rule |
| Thresholds and fine-tuning of one rule | "after 2 searches, alternate" refining "interleave searching and tracing" |

**Different lessons** are about different situations or different decisions: what to search for vs
when to stop searching; how a tool behaves vs what the answer must contain.

**Definition tables.** A list of plain definitions of the domain's vocabulary counts once, as one lesson
for the whole table ("compound relations are derived by chaining these first-degree relations"). A single
definition counts as its own lesson only when it is stated as a contrast with a wrong reading ("X, NOT
Y", "ONLY X", "X is a different relation from Y"): the contrast marks a specific misunderstanding the
prompt rules out.

**Output and format.** A statement about the form of the answer is a lesson when it is about what
scores: what must or must not be in the answer, how values must be written to match, partial vs empty
answers. Pure style ("be concise", renaming a field, "answer in JSON" with no scoring reason) is not.

## Labels

Every lesson gets one label. The label is a review aid; it does not decide whether something is a lesson.

| label | covers |
|---|---|
| **Task** | what is actually being asked and how it is scored (e.g. "this is retrieval, not verification"; "most questions have several correct answers") |
| **Domain** | facts about the world or the data (e.g. "pages list only first-degree relations"; "claims may contain factual errors") |
| **Tool** | how a tool or retriever behaves (e.g. "the search returns only the top few pages"; "the retriever matches tokens") |
| **Procedure** | steps or the order of work (e.g. "derive compound relations one hop at a time"; "interleave enumeration with tracing") |
| **Decision** | what to pick, include, exclude, or when to stop (e.g. a priority order for the next search target; a stopping rule) |
| **Output** | what the answer must contain or how it must be written to score |

## Themes

Group lessons into **themes**: the benchmark-specific topic a set of lessons is about (for example
"kinship derivation", "enumeration completeness", "retriever behaviour"). A theme can hold lessons of
any label. Themes are the coarse counting level; lessons are the fine one.
