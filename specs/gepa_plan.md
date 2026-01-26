# GEPA System Analysis and Plan

## Overview
CodeEvolver offers autonomous coding agents for high reliability AI systems. It uses GEPA optimization to evolve your AI system code until it performs optimally for a given dataset and outcome metric. See specs/CodeEvolver_analysis.md

This repository (CodeEvolver) integrates the GEPA (Genetic-Pareto) Optimizer. The GEPA package lives in a sibling repository (GEPA-CodeEvolver), which forks gepa-ai/gepa. CodeEvolver implements the adapter, tool-augmented reflection, code mutation integration, and the /optimize API.

**License:** MIT. All forked code must retain: `Copyright © 2025 Lakshya A Agrawal and the GEPA contributors`

## Analysis of GEPA
See github.com/gepa-ai/gepa or see local copy in "GEPA-CodeEvolver" folder, in the same parent folder as this CodeEvolver project.

## Key GEPA Components

### GEPAState
**program_candidates:** [{"module_1":{\<program_json\>}}]
- Defined by adapter
- For DSPY: each module is a DSPY program JSON
- For Lakshya's full program optimization, each module is a string
- For CodeEvolver:
	- the adapter for CodeEvolver should find a git branch in the candidate. {"git_branch": "codeevolver-3x67c", "module_1.predict":"instruction text...", etc.} 
	- inside the prompt key, we will use an adapter based on the package type. If the package is DSPY, the prompt JSON will look like the DSPY program JSON. We will start with only one adapter, and that adapter will be for DSPY.
**parent_program_for_candidates:** [[None], [prog_3], [prog_5, prog_1], [prog_3]] stores direct parent programs, and two programs for merge case
**prog_candidate_val_subscores**: provides metric output for each eval row
[
    {val_id_1: 0.8, val_id_2: 0.9, val_id_3: 0.7}, # Program 0
    {val_id_1: 0.85, val_id_2: 0.95}               # Program 1
]
**prog_candidate_objective_scores**: provides aggregate objective score for each program
[
	{"accuracy": 0.85, "latency": 0.2, "cost": 0.1},  # Program 0
    {"accuracy": 0.90, "latency": 0.15, "cost": 0.12} # Program 1
]
**pareto_front_valset**:
*When in instance mode for FrontierType:*
{
      "val_0": 0.85,   # Best score for validation instance 0
      "val_1": 0.92,   # Best score for validation instance 1
}
**program_at_pareto_front_valset**:
{
      "val_0": {2, 5},      # Programs 2 and 5 both achieved score 0.85 for val_0
      "val_1": {3},         # Program 3 achieved score 0.92 for val_1
}
**full_program_trace:** iteration metadata (iteration number, selected program ID, subsample IDs, aggregate scores),
*best_outputs_valset:* Optional. Stores the actual outputs

### Trace Usage
Reflective LM uses the reflective_dataset (processed DSPy traces).

adapter.make_reflective_dataset() creates a reflective_dataset that includes Program Inputs, Program Outputs, and Program Trace

The tracing is provided through the GEPA adapter. See below.

### GEPAAdapter
This is the single integration point between external systems and the GPA optimization engine.

Three inputs:
- DataInst: User-defined type of input data to the program under optimization.
- Trajectory: User-defined type of trajectory data, which typically captures the different steps of the program candidate execution.
- RolloutOutput: User-defined type of output data from the program candidate.

Key functions:
- **make_reflective_dataset:** uses EvaluationBatch (trajectory, outputs, scores), and produces a JSON data set. Only does so for the components you want to update for this round
- **Program and evaluation orchestration (evaluate):** For DSPY, ultimately imports DSPY Evaluate() to run the evaluation.
- **propose_new_texts (Optional):** uses ProposalFn to modify instruction_proposal.py, Which can be used to modify how the reflective LM works. Could be a useful function to modify, except that it is limited to prompts / str. (Note adapter implements propose_new_texts if it wants to delegate to a custom implementation)

### InstructionProposalSignature

**default_prompt_template:** 
I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks.

## Proposed Plan for CodeEvolver

### Two Repositories

| Repository | Role |
|------------|------|
| **GEPA** (github.com/gepa-ai/gepa) | Optimization algorithm and execution orchestration, State tracking, adapter protocol |
| **CodeEvolver** | Execution service, CodeEvolverAdapter implementation, Reflection agent with code read tool implementations, sandbox, git, etc. |

### Optimization jobs to be done 
Implements gepa.optimizer and runs in /optimize at API in CE. 

#### Jobs
- /optimize endpoint - CE
- main GEPA run program and evaluate orchestation - CE imports gepa
    - Initialize GEPAState with seed_candidate
- LOOP:
    - Select candidate from Pareto frontier - GEPA
    - Sample minibatch from trainset - GEPA
    - Run program and evaluate x10 examples x10 sandboxes (start w seed) - CE
        - `adapter.evaluate` - CE Adapter
    - `adapter.make_reflective_dataset` (process traces → JSON) - CE adapter in GEPA
    - Reflective LM x1 (agent) -> change request - CE
        - `adapter.propose_new_texts` - CE Adapter
    - Create new candidate by calling edit code or editing prompt directly
    - edit code - CE
    - Run_program And evaluate - CE
        -  `adapter.evaluate`
    - accept/reject. retry edit if necessary
    - Update GEPAState tracking - GEPA


*CE = CodeEvolver*
*GEPA = gepa-ai/gepa, no fork no mod
*GEPAmod = GEPA-CodeEvolver*


## Plan:

1. Create a CodeEvolverAdapter that handles the "code" mutation path separately - CE
    - Update evaluate() 
    - Can have different adapters for different AI frameworks e.g. A CodeEvolverDSPYAdapter vs opik, etc. 
2. Add Reflective Agent with tools (via adapter) - CE
    - **Modify proposer, ReflectiveMutationProposer, with tools?:** ToolSet and workspace_context that interacts with codebase. wraps the Claude Agents SDK with full codebase access
3. **Modify candidate tracking?** No need to change the GPA package for candidates. See candidate structure below, which is compatible.
4. Use GEPAState

#### Candidate structure
This candidate structure is compatible with GEPA. It references, the instruction text directly, and the adapter converts it to the dspy program format, using DspyAdapter.build_program()
candidate = {
    "git_branch": "codeevolver-3x67c",  # ← still a string!
    "module_1.predict": "instruction text...",
    "module_2.predict": "instruction text...",
}

#### CodeEvolverAdapter
- Adapter already delegates to propose_new_texts if it exists
- Add a propose_new_texts function, which Build a prompt and execute the reflection agent:
    - builds a reflection prompt using self._build_reflection_prompt(candidate, reflective_dataset, components_to_update)
    - response = await claude_agent_sdk.query(prompt, ClaudeAgentOptions(cwd = self.workspace_path, allowed_tools = ["Read", "Grep", "Glob"], permission_mode="acceptEdits") 
    - return self._parse_proposed_texts(response)
- Instead of inheriting From GEPAAdapter, the adapter conforms via duck typing

#### evaluate()
1. Checkout git_branch from candidate
2. Load DSPy program from program.json, apply candidate prompt texts
3. Run program on batch (in sandbox)
4. Return EvaluationBatch(outputs, scores, trajectories if capture_traces)

### Optimization Loop (runs in CodeEvolver)
Creates a CodeEvolver.GEPA optimize manager class with CodeEvolverAdapter -> _build_seed_candidate -> gepa.optimize.compile

A dedicated long-running Modal function (1hr timeout) calls `gepa.optimize()` synchronously. State is persisted to MongoDB each iteration via a `StopperProtocol` callback.

Follows the DSPy GEPA pattern from `dspy/teleprompt/gepa/gepa.py`

#### CodeEvolverGEPA interface (mirrors dspy.GEPA)
The GEPA optimization interface
1. **__init__**: 
   - Store config (reflection_lm model, budget, etc.)
   - metric - User provides a metric script path in their repo (e.g., `eval/metric.py`) and function name. 
2. **_build_seed_candidate(student_module)**:
   - Extract initial instructions from DSPy module predictors
   - Add `git_branch` key pointing to initial branch
   - Return `dict[str, str]` seed candidate
3. **compile(student, trainset, valset)**:
   - Create `CodeEvolverAdapter` (or `CodeEvolverDSPYAdapter`)
   - Call `_build_seed_candidate(student)` → seed_candidate
   - Call `gepa.optimize(seed_candidate, trainset, valset, adapter, ...)`
   - Return optimized program from `adapter.build_program(result.best_candidate)`





