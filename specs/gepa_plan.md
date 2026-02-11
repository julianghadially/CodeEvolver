# GEPA System Analysis and Plan
Copyright © 2026 440 Labs LLC

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
This is the single integration point between external systems and the GEPA optimization engine.

Three inputs:
- DataInst: User-defined type of input data to the program under optimization.
- Trajectory: User-defined type of trajectory data, which typically captures the different steps of the program candidate execution.
- RolloutOutput: User-defined type of output data from the program candidate.

Key functions:
- **make_reflective_dataset:** uses EvaluationBatch (trajectory - aka traces, outputs, scores), and produces a JSON data set. Only does so for the components you want to update for this round. 
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

### CodeEvolver Component Selectors (src/optimizer/component_selector.py)

**CodeFrequencyComponentSelector** controls code vs prompt mutation frequency with exponential decay.

The ratio is expressed as **prompts per code change**:
- 0 = only code changes (no prompts)
- 1 = 1 prompt per code (alternating: code, prompt, code, prompt, ...)
- 2 = 2 prompts per code (code, prompt, prompt, code, prompt, prompt, ...)
- 4 = 4 prompts per code (code, prompt, prompt, prompt, prompt, code, ...)

Parameters:
- `initial`: Starting prompts per code (default: 1)
- `decay_rate`: Iterations between each multiplier step (default: 25)
- `decay_factor`: Multiplier applied at each decay step (default: 2)
- `code_cutoff_step`: Stop code mutations after this iteration

Decay formula: `prompts_per_code = initial * (decay_factor ** (iteration // decay_rate))`

With defaults (initial=1, decay_rate=25, decay_factor=2):
- Iterations 0-24: prompts_per_code = 1 (1:1 ratio)
- Iterations 25-49: prompts_per_code = 2 (1:2 ratio)
- Iterations 50-74: prompts_per_code = 4 (1:4 ratio)
- Iterations 75-99: prompts_per_code = 8 (1:8 ratio)

```python
# Default: start 1:1, decay every 25 iterations
selector = CodeFrequencyComponentSelector()

# Code only (no prompts)
selector = CodeFrequencyComponentSelector(initial=0)

# Or via API:
{"initial": 1, "decay_rate": 25, "decay_factor": 2, "code_cutoff_step": 100}
```

### ReflectiveMutationProposer.propose_new_texts()
It appears that ReflectiveMutationProposer.propose_new_texts is the function inside GEPA that cycles through all the components to update, and makes a proposed change. In the default mode module_selector="round_robin", only 1 component is mutated per iteration. Additionally, the component selection is purely algorithmic. The LLM is only invoked in propose_new_texts() after the components have been selected by the strategy. The components themselves are exactly the keys in the seed_candidate dictionary ("module_1.predict").

*(Side note, mutating one component at a time makes it easier to attribute performance changes to specific modifications and provides cleaner evolutionary signals.)*

See https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/reflective_mutation/reflective_mutation.py

```python
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.strategies.instruction_proposal import InstructionProposalSignature
class ReflectiveMutationProposer(ProposeNewCandidate[DataId])
def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

        if self.reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")
        new_texts: dict[str, str] = {}
        for name in components_to_update:
            # Gracefully handle cases where a selected component has no data in reflective_dataset
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(
                    f"Component '{name}' is not in reflective dataset. Skipping."
                )
                continue

            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]
            new_texts[name] = InstructionProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                    "prompt_template": self.reflection_prompt_template,
                },
            )["new_instruction"]
        return new_texts
```

### Capturing Traces
Traces, a.k.a. trajectories, are captured by the adapter. The DSPY adapter capture traces as shown below. If we're capturing traces, We use dspy...bootstrap_trace_data instead of dspy.evaluate

```python
from dspy.evaluate import Evaluate
def evaluate(self, batch, candidate, capture_traces=False):
    program = self.build_program(candidate)
    outputs: list[Prediction] = []
    scores: list[float] = []
    subscores: list[dict[str, float]] = []
    trajs: list[TraceData] | None = None
    # If we're capturing traces, We use dspy...bootstrap_trace_data instead of dspy.evaluate
    if capture_traces:
            # bootstrap_trace_data-like flow with trace capture
            from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data

            trajs = bootstrap_trace_data()
            #traj Is a dictionary that Includes a key for predictions and scores
    else:
        evaluator = Evaluate()
```

## Proposed GEPA Plan for CodeEvolver

### Two Repositories

| Repository | Role |
|------------|------|
| **GEPA** (github.com/gepa-ai/gepa) | Optimization algorithm and execution orchestration, State tracking, adapter protocol |
| **CodeEvolver** | Execution service, CodeEvolverAdapter implementation, Reflection agent with code read tool implementations, sandbox, git, etc. |

### Optimization jobs to be done 
Implements gepa.optimizer and runs in /optimize at API in CE. 

#### GEPA Jobs
- /optimize endpoint - CE
- `run_optimization` Create a MODAL function, which runs `run_gepa_optimization`. This is the main GEPA orchestator, Which will perform the following:
    - Create the trainset (looks like dspy.Example) - CE
    - Initialize adapter
    - Build seed candidate
    - start the GEPA loop
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

### GEPA Changes:

1. Create a CodeEvolverAdapter that handles the "code" mutations and prompt mutations - CE
    - Update Adapter.evaluate() 
    - Can have different adapters for different AI frameworks e.g. A CodeEvolverDSPYAdapter vs opik, etc. 
2. Add Reflective Agent with tools (via adapter) - CE
    - **Modify proposer, ReflectiveMutationProposer, with tools?:** ToolSet and workspace_context that interacts with codebase. wraps the Claude Agents SDK with full codebase access
3. **Modify candidate tracking?** No need to change the GPA package for candidates. See candidate structure below, which is compatible.
4. Use GEPAState
5. **additional_instructions**: A client-provided string that guides GEPA optimization. The string is included in all reflection LM prompts (both code and prompt mutations). Contents may include:
   - **Constraints**: Changes that are completely off-limits (e.g., "Do not modify the authentication module")
   - **Services**: External services available with API keys already in the environment (e.g., "Firecrawl API is available for web scraping")
   - **Ideas**: Suggestions for optimization approaches

#### Branch Naming Convention
All branches created during a GEPA run share a common timestamp prefix for traceability:
- **Run main branch**: `codeevolver-{YYYYMMDDHHmmss}-main` — Created at the start of each optimization run from the initial branch
- **Mutation branches**: `codeevolver-{YYYYMMDDHHmmss}-{uuid}` — Created from parent branches during code mutations

#### Initial Branch Selection
Users can specify which branch to use as the starting point for optimization via the `initial_branch` parameter (defaults to "main"). This allows:
- Starting optimization from a feature branch
- Continuing optimization from a previous codeevolver branch
- Working with repos that use different default branches (e.g., "master", "develop")

The initial branch is cloned directly using `git clone --branch {initial_branch}`, and the CodeEvolver main branch (`codeevolver-{timestamp}-main`) is created from it.

The run main branch contains:
- `codeevolver.md` — LM-generated architecture summary of the program being optimized

#### Candidate structure
This candidate structure is compatible with GEPA. It references the instruction text directly, and the adapter converts it to the dspy program format, using DspyAdapter.build_program()
```python
candidate = {
    "_code": json.dumps({
        "git_branch": "codeevolver-yyyymmddhhmm-main",  # or codeevolver-20260202163000-a1b2c3
        "parent_module_path": "src.factchecker.FactCheckerPipeline",  # top-most module to evaluate
        "change_request": "The change that was just executed",
        "last_change_summary": "Agent output summary"
    }),
    "module_1.predict": "instruction text...",
    "module_2.predict": "instruction text...",
}
```
Architecture is stored in `codeevolver.md` (not in `_code`) so it stays in sync with code changes. Both the reflection agent and coding agent read/update this file directly.

**codeevolver.md format:**
```
PARENT_MODULE_PATH: src.factchecker.FactCheckerPipeline
METRIC_MODULE_PATH: eval.evaluate.metric

# Architecture Summary
...
```
The `parent_module_path` in `_code` is parsed from `codeevolver.md` after each code mutation, allowing the coding agent to change the entry point (e.g., creating a pipeline wrapper).

The `_code` component participates in GEPA's round-robin selection. When selected, a two-phase mutation occurs: (1) reflection agent proposes a change, (2) coding agent executes it on a new branch.

#### CodeEvolverAdapter
- Adapter already delegates to propose_new_texts if it exists
- **Initialization flow (build_seed_candidate):**
    1. Create run main branch (`codeevolver-{timestamp}-main`) from initial branch
    2. Use reflection LM to analyze program code and generate architecture summary
    3. Save architecture summary to `codeevolver.md` and commit
    4. Return seed candidate with the run main branch as git_branch
- Add a propose_new_texts function that routes by component type:
    - **Prompt components:** Use GEPA's InstructionProposalSignature with reflection_lm
    - **`_code` component:** Two-phase mutation:
        1. Reflection agent (Read/Grep/Glob tools) analyzes feedback and proposes a change
        2. Coding agent (Read/Write/Edit/Bash tools) executes the change on a new branch
    - When `_code` is mutated, also updates `git_branch` to the new branch name
- Branch handling: checkout parent branch → create new branch (`codeevolver-{timestamp}-{uuid}`) → execute mutation
- Instead of inheriting from GEPAAdapter, the adapter conforms via duck typing

#### evaluate()
DSPy-native evaluation (no sandbox for prompt-only v1):
1. Import and instantiate DSPy module (from dotted import path), load program.json, apply candidate prompt texts
2. Run each example through the program in-process
3. Score with user's metric function
4. Optionally capture DSPy traces for reflection
5. Return EvaluationBatch(outputs, scores, trajectories if capture_traces)

#### Optimization Loop (runs in CodeEvolver)
Creates a CodeEvolver.GEPA optimize manager class with CodeEvolverAdapter -> _build_seed_candidate -> gepa.optimize.compile

A dedicated long-running Modal function (1hr timeout) calls `gepa.optimize()` synchronously. State is persisted to MongoDB each iteration via a `StopperProtocol` callback.

Follows the DSPy GEPA pattern from `dspy/teleprompt/gepa/gepa.py`

#### CodeEvolverGEPA interface (mirrors dspy.GEPA)
The GEPA optimization interface
1. **__init__**: 
   - Store config (reflection_lm model, budget, etc.)
   - metric - User provides a dotted import path to their metric function (e.g., `eval.evaluate.metric`).
2. **_build_seed_candidate(student_module)**:
   - Extract initial instructions from DSPy module predictors
   - Add `git_branch` key pointing to initial branch
   - Add `_code` component with architecture from requirements.md
   - Return `dict[str, str]` seed candidate
3. **compile(student, trainset, valset)**:
   - Create `CodeEvolverAdapter` (or `CodeEvolverDSPYAdapter`)
   - Call `_build_seed_candidate(student)` → seed_candidate
   - Call `gepa.optimize(seed_candidate, trainset, valset, adapter, ...)`
   - Return optimized program from `adapter.build_program(result.best_candidate)`

#### Traces
In adapter.evaluate(), We sent to the sandbox via GEPASandbox.exec_prebuilt() With command "evaluate," Which then triggers _evaluate_with_traces if capture_traces = true.

### Sandbox Coordination
The GEPA optimizer process is managed by a long-running MODAL function that manages the code evolution / optimization. It creates a Client sandbox to run client code inside of. See specs/requirements.md for ClientSandbox architecture.

`GEPASandbox` manages the client sandbox environment (long-running), in service of a long-running Modal function that runs the optimization loop.

`GEPASandbox` inherits from `ClientSandbox` (in `src/sandbox/client_sandbox.py`). The client sandbox mimics the client's AI application environment — it installs the client's repository and `requirements.txt`. To run GEPA without any DSPy dependencies in the orchestrator, evaluation commands are delegated to prebuilt scripts inside the sandbox.

**Key Methods:**
- `GEPASandbox.exec_prebuilt(command)` — Sends JSON command to `master_script.py`, returns JSON result
- `GEPASandbox.exec_agent(change_request)` — Execute coding agent for code mutations
- `GEPASandbox.exec_reflection_agent(prompt)` — Execute reflection agent with read-only tools (Read/Grep/Glob)
- `ClientSandbox.start()` — Clones repo, installs client requirements.txt
- `ClientSandbox.exec_bash(command)` — Run arbitrary bash in sandbox
- `ClientSandbox.stop()` — Terminate sandbox

#### Autonomous Execution Design

The coding agent runs via Claude Agent SDK with `permission_mode="bypassPermissions"` plus a **user proxy callback** (`can_use_tool`) that auto-approves plan mode and auto-answers questions.

**The Problem:** Even with `bypassPermissions`, Claude Code can enter "plan mode" via `EnterPlanMode` tool. Once in plan mode, `ExitPlanMode` requires explicit user approval that never comes in an autonomous context.

**The Solution:** Use the `can_use_tool` callback as a user proxy:
```python
from claude_agent_sdk.types import PermissionResultAllow, ToolPermissionContext

async def user_proxy(tool_name: str, input_data: dict, context: ToolPermissionContext):
    if tool_name == "ExitPlanMode":
        # Auto-approve plan mode exit
        return PermissionResultAllow(updated_input=input_data)
    if tool_name == "AskUserQuestion":
        # Auto-respond with first option for each question
        questions = input_data.get("questions", [])
        answers = {q["question"]: q["options"][0]["label"] for q in questions}
        return PermissionResultAllow(updated_input={"questions": questions, "answers": answers})
    return PermissionResultAllow(updated_input=input_data)

# Usage:
ClaudeAgentOptions(permission_mode="bypassPermissions", can_use_tool=user_proxy)
```

This allows Claude to use plan mode for complex multi-file changes while running autonomously.
See `specs/ralph_claude_code.md` for analysis of alternative CLI-based approaches.


## Remaining
- [x] Reflection LLM agent
- [ ] End-to-end testing of GEPA with code mutations + evaluate
- [ ] Performance results and tuning
- [ ] Compatability with non-DSPy (clients organize prompt json?)
- [ ] Speed: Consider multiple parallel workers, with workers expanding as code branches increase, and prompt optimization "depth" focused on each individual branch / worker.
- [ ] change user to @codeevolver.dev (not .ai)
- [ ] How should we provide the current prompts to the coding agent? Should this be a separate tool call?
