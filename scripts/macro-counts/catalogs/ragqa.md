# Macro-approach catalog: ragqa

## Macro approaches

### M1: Reasoning steps - Breaking the workflow into separate LM calls for different reasoning steps (e.g., gap analysis) 
Description: A new stage upstream of the main step whose output feeds its reasoning logic. For example, running an analysis before generating a query. Calls that check, critique, repair, retry or tie-break an existing output don't count; a call covered by a more specific entry counts only there.

### M2: Multi-query retrieval fan-out
Description:  A hop runs two or more queries (LLM rewrites, the raw question, or both) and merges the results before synthesis (instead of a single query).

### M3: LLM passage reranker (e.g., between retrieval and synthesis)
Description: Example: after the multi-hop retrieval/dedup loop and before answer synthesis, a separate LLM call ranks the deduplicated passage pool by usefulness to the specific question, re-ordering (and optionally dropping) passages so the most relevant evidence leads the context fed to the synthesizer.

### M4: Deterministic rules-based method for query formulation or writing answers
Description: For example, using parsing rules to generate queries or writing answers. Deterministic rules for other portions of the system are most likely incremental. This approach excludes using the question verbatim as a query.

### M5: Use of executable sandbox coding environment for aiding in tech Q&A
Description: Language model tests answers by running them in a sandbox coding environment. 
