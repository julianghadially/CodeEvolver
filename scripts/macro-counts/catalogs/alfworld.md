# Macro-approach catalog: alfworld

## Macro approaches

### M1: Reasoning steps - Breaking the workflow into separate LM calls for different reasoning steps (e.g., analysis) 
Description: A new stage upstream of the main step whose output feeds its reasoning logic. For example, running an analysis before producing an action. Calls that check, critique, repair, retry or tie-break an existing output don't count; a call covered by a more specific entry counts only there.

### M2: Symbolic state tracking (belief state)
Description: Observations are parsed each step into a structured belief state — held inventory, visited receptacles and their contents, object heat/cool/clean states, failed or no-op actions, current location, and goal-condition progress — which is stored across steps and injected into the prompt as a state block that the LLM reads when choosing the next action.

### M3: Rules for program-chosen actions
Description: A program-computed action that replaces the model's choice, derived from the task or tracked state. E.g., when the LLM fails to produce a parseable admissible action, or repeated no-progress is detected (e.g. three or more consecutive parse failures, a dead-end receptacle with no target, or a no-op on a processing action), the program overrides the passive fallback and selects a recovery action (e.g., based on the tracked state — navigating to an unvisited or unsearched receptacle, opening a closed container to search, placing a held object at the known destination, or retrying a processing action at a different appliance instance).

### M4: Added Tool Use
Description: adding LLM-visible tools that an LLM decides when to use for interacting with the environment. Whereas simply adding additional functions to the adapter layer would be considered incremental. 
