# Prompt-lesson catalog: arcagi2

21 lessons in 5 themes. By label: Task 1, Domain 3, Tool 1, Procedure 7, Decision 5, Output 4.

## Lessons

### T1: Rule inference from demonstrations

#### L1: One shared rule governs all demonstrations
Label: Domain
Description: All demonstration input/output pairs are produced by a single common transformation rule. The inferred rule must explain every pair, not just some of them.

#### L2: Grid and color encoding
Label: Domain
Description: Each grid is a 2D array of single-digit integers, where each digit 0-9 is a color. Grids are at most 30x30.

#### L3: Inventory the objects first
Label: Procedure
Description: Before stating the rule, identify the objects in each demonstration pair, with their colors, positions, sizes and counts.

#### L4: Prefer the simplest consistent rule
Label: Decision
Description: Among candidate transformation rules, choose the simplest one that is consistent with every demonstration pair.

#### L5: Avoid proxy shortcuts
Label: Decision
Description: Do not rely on features that merely correlate with the demonstration outputs, such as the highest color value, grid dimensions or cell counts. Find the actual transformation logic, because a shortcut that fits the demos may fail on the test input.

#### L6: Rule must reproduce every demonstration exactly
Label: Task
Description: A correct rule reproduces every demonstration output cell for cell. This holds whether the rule is applied by hand or implemented as code. Before applying it to the test input, check it against each demo by actually re-deriving the output, not by copying the shown one. If any demo fails, the rule is wrong and must be revised. Close but inexact reproduction counts as a failure.

#### L7: State the rule briefly before coding
Label: Procedure
Description: Put the inferred rule into one short sentence first, then write the implementation or output that follows from it.

### T2: Output grid shape and exact-match scoring

#### L8: Output size may differ from input size and must be derived first
Label: Domain
Description: The output grid can have different height and width than the input. Work out how each demo's output size relates to its input, fix the test output size from the rule, and then fill every cell of a grid of that size.

#### L9: Exact match on dimensions and every cell
Label: Output
Description: An answer is correct only if the output grid has exactly the right number of rows and columns and every cell has the right color. A grid of the wrong size cannot be right.

#### L10: Returned grid must be a valid grid
Label: Output
Description: The result must be a rectangular 2D list of integers in the range 0-9. Anything else, such as ragged rows or non-integer colors, is rejected as a non-grid.

#### L11: Always give the final grid as a JSON array
Label: Output
Description: Always end with the output grid for the test input as a JSON array of arrays of ints, even when code is also written. This grid is the scored final answer.

#### L12: Compact run-length format for large grids
Label: Output
Description: When the output grid is large and might not fit in the output budget, it may be written compactly. Each row is a list of [color, count] pairs, and every row must expand to the same length.

#### L13: Keep visible reasoning short so the grid is not truncated
Label: Procedure
Description: Long reasoning can use up the whole output budget and cut off or omit the grid, which scores nothing. Keep reasoning short and emit the complete grid promptly. After an empty or failed attempt, state the rule in one sentence and go straight to code and grid without tracing every cell.

### T3: Solving by writing a program

#### L14: Express the rule as a Python function
Label: Procedure
Description: Write the inferred transformation as a Python function that maps an input grid (list of lists of ints) to an output grid, then use it to produce the answer. This avoids cell-by-cell mistakes that come from computing a large grid by hand.

#### L15: Generalize, do not hardcode
Label: Decision
Description: The function must compute the result from the input and work for any grid that follows the rule. Do not hardcode demonstration outputs, positions, sizes or values. Do not branch on input dimensions to return a memorized grid.

#### L16: Sandbox restrictions on generated code
Label: Tool
Description: Generated code runs in a restricted environment. Imports are limited or blocked, and the permitted names are provided directly. There is no numpy, no file access or I/O, no printing, and no exec, eval or input. The code should be efficient.

#### L17: Offer several genuinely different candidate programs when unsure
Label: Procedure
Description: When the exact rule is uncertain, give up to a few candidate functions in separate code blocks, each using a genuinely different approach. Each is verified independently and the most consistent verified one is used. This reduces overfitting to the demos.

### T4: Repairing code from execution feedback

#### L18: Correct the rule from reported mismatches
Label: Procedure
Description: If the code ran but did not reproduce the demos, use the reported details to fix the rule. These are the failing demo, the produced versus expected grid, the differing cells (got vs wanted) and the count of correct demos. Check typical bug classes such as wrong dimensions, off-by-one, wrong color, wrong condition and mis-counted objects. If the dimensions are wrong, re-derive them from the rule and the input size.

#### L19: Fix crashes and timeouts
Label: Procedure
Description: If the code crashed or failed to compile, rewrite it so it runs and still reproduces every demo, without relying on blocked imports. A timeout probably means an infinite loop, so make sure every loop terminates.

### T5: Reviewing and choosing among candidate outputs

#### L20: Check a proposed output against the rule and correct it if wrong
Label: Decision
Description: When given a candidate test output, check whether it follows the rule deduced from the demonstrations. Return it unchanged if it does, and return a corrected grid if it does not.

#### L21: Reject candidates with wrong dimensions
Label: Decision
Description: When choosing between candidate outputs, one whose dimensions do not match the size derived from the demos cannot be correct and must not be chosen. If all candidates are wrong, derive the correct grid directly with the right dimensions.

## Notes from discovery (ignored by the parser)

- Several lessons merge hand-applied and code-based variants of one rule. This applies to 'Rule must reproduce every demonstration exactly' (checking a stated rule, regenerating demo outputs, and verifying code against demos), 'Express the rule as a Python function' and 'Generalize, do not hardcode'.
- The sandbox lesson merges differing statements about imports. Some say all imports are blocked and names are provided directly. Others say a small whitelist of pure standard-library modules is allowed. The common belief is that the code environment is restricted, with no numpy or I/O. A reviewer may want to check the exact import policy per program.
- Crash/timeout repair and mismatch-based repair are kept separate because they respond to different failure situations. One is a runtime error and the other is wrong output.
- The brevity lessons are merged into one rule: keep reasoning short so a complete grid is emitted. This covers both the general case and the case after an empty or failed attempt.
- Generic items were not counted as lessons, for example a lighter variant of reproducing only one demo to keep the answer short. That variant is treated as a threshold of the demo-reproduction lesson.
- (validator) possible provenance wording 'a run' in: Crash/timeout repair and mismatch-based repair are kept separate because they respond to different failure situations. O
