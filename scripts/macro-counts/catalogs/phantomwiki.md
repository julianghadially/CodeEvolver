# Macro-approach catalog: phantomwiki

## Macro approaches
### M1: Sequential reasoning steps - breaking the workflow into separate LM calls for reasoning steps that follow each other (e.g., analysis) 
Description: A new stage upstream of the main step whose output feeds its reasoning logic. For example, running an analysis before generating a query. Calls that check, critique, repair, retry or tie-break an existing output don't count; a call covered by a more specific entry counts only there.

### M2: Parallel reasoning steps - sub-agent pipeline for breaking down parallel reasponing paths (e.g., sub-agents per-entity)
Description: Reasoning threads are broken up by sub-agents. Example: A compound attribute-anchored question is decomposed: a gather/discovery step enumerates all matching entities, then a fresh-context reasoning sub-agent runs the relation chain for each entity (in parallel), and the per-entity answers are unioned (sometimes alongside a full-question run) into the final set, with a final aggregation stage.

### M3: Data index for efficient retrieval lookups
Description: A local index over the corpus (date traversal, attribute-value-to-names) is built once from the data files and consulted by retrieval tools, giving exact, complete lookups instead of remote semantic retrieval (which caps results and buries exact attribute matches among near-misses).

### M4: Scratchpad for plan 
Description: A separate planning call (chain-of-thought) produces a search or relation-chain plan that is stored and fed into the ReAct loop (as a signature input or prepended to the first step), shaping the agent's strategy across the trajectory.

### M5: Scratchpad or workspace for persistent findings
Description: An explicit findings workspace accumulates verified answers as the agent confirms them during the loop, and a finalization step reads the workspace and unions it with the loop's final answer to produce the output.

### M6: Deterministic query generation
Description: A tool or function issues query formulations - for example multiple queries of one value like a date or attribute value. In some cases, the tool may merge the deduplicated passages before returning them.

### M7: Multiple agent sampling techniques (multiple calls on the same question)
Description: Multiple independent ReAct runs of the same question are executed (in parallel) and their answers unioned; a continuation run is conditioned on the already-found answers (its prompt lists them and asks for new ones) to discover additional values.

### M8: LM Reranker
Description: Reranker prioritizes results on some objective. 
