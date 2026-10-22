# Macro-approach catalog: paperreview

## Macro approaches

### M1: Reasoning steps - Breaking workflow into separate LM calls for different reasoning steps  
Description: For example, Breaking up an analysis step into a separate model call before generating a verdict. Or a separate model specialized weighing different paper questions, such as evidence sufficiency. Calls that check, critique, repair, retry or tie-break an existing output don't count; a call covered by a more specific entry counts only there.

### M2: LLM Sampling or Voting Approach
Description: Multiple independent LLM samples are drawn and their accept/reject verdicts are aggregated by majority vote or by thresholding the median of a numeric score. Includes approaches that use conditional LLM sampling, if the sample functions as another vote.

### M3: Role Play Judge Approach 
Description: Different judges play different roles and use role-play to judge the paper. For example an advocate versus skeptic judge share both sides of the argument. 

### M4: Creation of targeted skills (separate from prompt)
Description: Creating separate skills.md files to break up prompts into specialized categories.
