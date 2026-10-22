# Macro-approach catalog: arcagi2

## Macro approaches

### M1: LLM Sampling or Voting Approach
Description: Multiple independent LLM samples are drawn per test input using the same prompt; the parsed output grids are combined by majority vote (with early-exit when a quorum agrees). This replaces the seed's single-call-per-test-input with a multi-sample voting step.

### M2: Demo-reproduction verification
Description: The model is asked to reproduce every demonstration output alongside the test answer; Python checks each reproduced grid against the known demonstration outputs. Samples whose reproductions all match are deemed 'verified' (correct rule) and their test outputs are preferred over unverified ones, or a provably-wrong majority can be overridden by a verified minority.

### M3: Hold-out-demonstration verification
Description: One demonstration's output is withheld; the model re-derives the rule from the remaining demos only and is asked to reproduce the withheld output and solve the test. If the withheld reproduction exactly matches the known output, the rule is certified and the freshly-derived test answer is adopted; otherwise the original draft is kept.

### M4: Program synthesis and/or sandboxed execution
Description: The LLM itself writes a Python function (e.g., `transform(grid)`) implementing the rule it inferred; the function is executed in a sandbox (restricted builtins, import whitelist, timeout). Often for verifying against demonstration pairs. Often using a fallback if verification fails or the code crashes.

### M5: Multi-path ensemble with reconciliation
Description: Two or more solve paths (e.g., a direct-JSON-grid path and a code-generation path) are run independently for each test case; their outputs are reconciled by a computational agreement gate (prefer one path only when it corroborates the other on dimensions and cell-level similarity) or by an LLM judge that picks between disagreeing candidates.

### M6: Deterministic function reuse across cases 
Description: A solution artifact derived while solving one test case of a task, typically a synthesized Python program, is cached and re-executed directly on the task's remaining test inputs without fresh LLM inference for those cases. As an example, once a Python `transform(grid)` function is verified against all demonstration pairs on one test case of a task, it is cached and re-applied directly to the task's remaining test inputs with no new LLM call, keeping the inferred rule consistent across cases and eliminating per-case variance.
