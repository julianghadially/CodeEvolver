# Prompt-lesson check: one program

Below is the prompt-like text of ONE task program, and a fixed catalog of **prompt lessons** for its
benchmark. Your job: for every lesson in the catalog, decide whether this program's LM-visible text
contains it. You see one program only; judge it on its own.

{definition}

## How to judge

- **Present** means the program's LM-visible text states the lesson's belief or rule, in any wording, in
  any prompt, field description or tool description, as the catalog description says it. A lesson stated
  only partly counts if its core point is there; a passing mention of the same topic that does not state
  the rule does not.
- **Contradictory variants.** Some catalog lessons are opposite variants of each other (for example "X
  only" versus "X or Y"). Mark the variant the text actually states; a text can state both.
- **LM-visible only.** The text below is extracted mechanically from the program's source and is a
  superset: each block is headed by its location (file :: enclosing class/def :: assignment target, call
  keyword, or `docstring`). Helper-function docstrings, log and error messages, URLs and format strings
  that are never sent to the LM do not count. Tool docstrings of functions handed to an LM agent, signature
  instructions, field descriptions and prompt templates do count.
- **Quote evidence.** For every lesson you mark present, copy a short quote (one sentence or less)
  verbatim from the text that states it. The quote is checked against the text.
- **Go theme by theme.** Give a verdict for EVERY lesson in the catalog, in catalog order, under its
  theme. Do not skip any lesson id.
- **Uncatalogued.** List any lesson in the text (by the definition above) that matches no catalog entry.
  These are not counted; they inform the next curation pass.
- Scores, quality and whether the lesson is correct are irrelevant: a wrong lesson stated in the text is
  still present.

## The catalog

{catalog}

## The program's prompt-like text

{program_text}
