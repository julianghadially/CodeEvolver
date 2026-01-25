# GEPA Integration Requirements

## Overview

CodeEvolver integrates GEPA (Genetic-Pareto) for optimizing AI workflows. We will **fork** gepa-ai/gepa and extend it with tool-augmented reflection and code mutation support.

**License:** MIT. All forked code must retain: `Copyright © 2025 Lakshya A Agrawal and the GEPA contributors`

---

## Architecture Decision: Fork

### Why Fork (not Copy)

1. GEPA's adapter pattern cleanly separates optimization logic from execution
2. Our extensions (tool-augmented reflection, code mutations) could benefit the broader GEPA ecosystem
3. Maintains path for upstream contributions and potential collaboration with Lakshya / Gepa team
4. GEPA is a library - CodeEvolver imports and runs it

### Two Repositories

| Repository | Role |
|------------|------|
| **GEPA Fork** (gepa-ai/gepa fork) | Optimization algorithm, adapter protocol, tool interface |
| **CodeEvolver** | Execution service, adapter implementation, tool implementations, sandbox |

---

## Division of Labor

### GEPA Fork Provides:

1. **Core Optimization Algorithm**
   - `GEPAEngine` - main optimization loop
   - `GEPAState` - Pareto frontier tracking, caching
   - `ReflectiveMutationProposer` - mutation proposal logic
   - `MergeProposer` - crossover/merge logic
   - Selection strategies, batch samplers

2. **Adapter Protocol** (`GEPAAdapter`)
   - Interface that CodeEvolver implements
   - `evaluate(batch, candidate) -> EvaluationBatch`
   - `make_reflective_dataset(...) -> traces`

3. **Tool Interface Protocol** (NEW - our fork extension)
   - Defines what tools the reflective agent can use
   - `ReadTool`, `GrepTool`, `GlobTool` protocol definitions
   - The reflection LM becomes an agent that can call these tools

4. **Extended Candidate Format** (NEW - our fork extension)
   - Support for code change requests alongside prompt mutations
   ```python
   candidate = {
       "component.predict": "new prompt text",  # prompt mutation
       "__code_change__": {                      # code mutation (optional)
           "description": "Add retry logic",
           "target_files": ["src/agent.py"]
       }
   }
   ```

5. **Reflective Agent Mode** (NEW - our fork extension)
   - `ReflectiveAgentProposer` that uses tools during reflection
   - Takes a `tool_provider` that supplies tool implementations
   - GEPA provides the agent loop, expects tools to be injected

### CodeEvolver Provides:

1. **CodeEvolverAdapter** (implements `GEPAAdapter`)
   ```python
   class CodeEvolverAdapter(GEPAAdapter):
       def __init__(self, sandbox_manager, git_service, run_path, ...):
           self.sandbox = sandbox_manager
           self.git = git_service
           self.run_path = run_path
       
       def evaluate(self, batch, candidate, capture_traces=False):
           # 1. Apply mutations (prompt and/or code)
           # 2. Run program in Modal sandbox via run_path
           # 3. Return outputs, scores, traces
       
       def make_reflective_dataset(self, ...):
           # Extract traces from DSPy/execution for reflection
   ```

2. **Tool Implementations** (implements GEPA's tool protocol)
   ```python
   class CodeEvolverToolProvider:
       """Provides tool implementations that work in the cloned repo."""
       
       def __init__(self, workspace_path: str):
           self.workspace = workspace_path  # cloned repo location
       
       def read_file(self, path: str) -> str:
           # Read file from cloned repo
       
       def grep(self, pattern: str, path: str = ".") -> list[Match]:
           # Search in cloned repo
       
       def glob(self, pattern: str) -> list[str]:
           # Find files in cloned repo
   ```

3. **Git Services**
   - Clone repository (via GitHub App)
   - Create worktrees for parallel mutations
   - Commit changes after code mutations
   - These live in CodeEvolver because they interact with Modal sandbox filesystem

4. **Modal Sandbox Execution**
   - Spin up sandboxes for program execution
   - Run user's `run_path` script
   - Collect outputs and traces

5. **Coding Agent** (for code mutations)
   - When GEPA proposes a code change request, CodeEvolver's coding agent executes it
   - Uses Claude Agent SDK to make actual code edits
   - Commits changes to git branch

6. **API Endpoints**
   - `POST /optimizer/create_job` - Start optimization
   - `GET /optimizer/job/{job_id}` - Get status/results

---

## Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CodeEvolver API                                  │
│  POST /optimizer/create_job                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CodeEvolver Orchestrator                              │
│  1. Clone repo via GitService                                            │
│  2. Initialize CodeEvolverAdapter                                        │
│  3. Initialize CodeEvolverToolProvider (pointed at cloned repo)          │
│  4. Start GEPA optimization                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         GEPA Engine (from fork)                          │
│                                                                          │
│  Main Loop:                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ 1. Select candidate from Pareto frontier                           │ │
│  │ 2. Sample minibatch from trainset                                  │ │
│  │ 3. Call adapter.evaluate(batch, candidate, capture_traces=True)    │ │
│  │         │                                                          │ │
│  │         ▼  ┌──────────────────────────────────────────────────┐   │ │
│  │            │ CodeEvolverAdapter (in CodeEvolver)               │   │ │
│  │            │ - Apply prompt mutation to program.json           │   │ │
│  │            │ - Run program in Modal sandbox                    │   │ │
│  │            │ - Return scores + traces                          │   │ │
│  │            └──────────────────────────────────────────────────┘   │ │
│  │ 4. Call reflective_proposer.propose(traces, tools)                 │ │
│  │         │                                                          │ │
│  │         ▼  ┌──────────────────────────────────────────────────┐   │ │
│  │            │ ReflectiveAgentProposer (in GEPA fork)            │   │ │
│  │            │ - Reflection LM analyzes traces                   │   │ │
│  │            │ - Calls tools.read(), tools.grep() as needed      │   │ │
│  │            │         │                                         │   │ │
│  │            │         ▼  ┌────────────────────────────────┐    │   │ │
│  │            │            │ CodeEvolverToolProvider        │    │   │ │
│  │            │            │ (in CodeEvolver)               │    │   │ │
│  │            │            │ - Reads from cloned repo       │    │   │ │
│  │            │            └────────────────────────────────┘    │   │ │
│  │            │ - Proposes new prompts AND/OR code changes        │   │ │
│  │            └──────────────────────────────────────────────────┘   │ │
│  │ 5. If code change proposed:                                        │ │
│  │         │                                                          │ │
│  │         ▼  ┌──────────────────────────────────────────────────┐   │ │
│  │            │ CodeEvolver Coding Agent                          │   │ │
│  │            │ - Receives change request from GEPA               │   │ │
│  │            │ - Uses Claude Agent SDK to edit code              │   │ │
│  │            │ - Commits to git branch                           │   │ │
│  │            └──────────────────────────────────────────────────┘   │ │
│  │ 6. Evaluate new candidate, update Pareto frontier                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## GEPA Fork Modifications

### 1. Tool Interface Protocol

```python
# In GEPA fork: gepa/core/tools.py

from typing import Protocol

class ReadTool(Protocol):
    def __call__(self, path: str) -> str:
        """Read file contents."""
        ...

class GrepTool(Protocol):
    def __call__(self, pattern: str, path: str = ".") -> list[dict]:
        """Search for pattern, return matches with file/line info."""
        ...

class GlobTool(Protocol):
    def __call__(self, pattern: str) -> list[str]:
        """Find files matching pattern."""
        ...

class ToolProvider(Protocol):
    """Interface that CodeEvolver implements."""
    read: ReadTool
    grep: GrepTool
    glob: GlobTool
```

### 2. Reflective Agent Proposer

```python
# In GEPA fork: gepa/proposer/reflective_agent.py

class ReflectiveAgentProposer(ProposeNewCandidate):
    """
    Extended proposer that uses tools during reflection.
    The reflection LM becomes an agent that can inspect code.
    """
    
    def __init__(
        self,
        reflection_lm: LanguageModel,
        tool_provider: ToolProvider,  # Injected by CodeEvolver
        allow_code_mutations: bool = False,
        ...
    ):
        self.reflection_lm = reflection_lm
        self.tools = tool_provider
        self.allow_code_mutations = allow_code_mutations
    
    def propose(self, state: GEPAState) -> CandidateProposal | None:
        # 1. Get traces from current evaluation
        # 2. Run reflection LM as agent with tool access
        # 3. LM can call tools.read(), tools.grep() to inspect code
        # 4. LM proposes prompt mutations AND/OR code change requests
        # 5. Return extended CandidateProposal
```

### 3. Extended Candidate Format

```python
# Current GEPA candidate (prompt-only):
candidate = {
    "judge.predict": "You are a fact-checking judge...",
    "summarizer.predict": "Summarize the following..."
}

# Extended candidate (prompt + code):
candidate = {
    "judge.predict": "You are a fact-checking judge...",
    "__code_mutation__": {
        "type": "change_request",
        "description": "Add retry logic with exponential backoff to API calls in src/api_client.py",
        "rationale": "Traces show frequent timeout errors on API calls",
        "target_files": ["src/api_client.py"]
    }
}
```

---

## CodeEvolver Implementation

### CodeEvolverAdapter

```python
# In CodeEvolver: src/adapters/gepa_adapter.py

from gepa.core.adapter import GEPAAdapter, EvaluationBatch

class CodeEvolverAdapter(GEPAAdapter):
    def __init__(
        self,
        sandbox_manager: SandboxManager,
        git_service: GitService,
        run_path: str,
        program_json_path: str | None = None,
    ):
        self.sandbox = sandbox_manager
        self.git = git_service
        self.run_path = run_path
        self.program_json_path = program_json_path
    
    def evaluate(
        self,
        batch: list[Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        # 1. Check for code mutation
        if "__code_mutation__" in candidate:
            self._apply_code_mutation(candidate["__code_mutation__"])
        
        # 2. Apply prompt mutations to program.json
        prompt_candidate = {k: v for k, v in candidate.items() if not k.startswith("__")}
        if self.program_json_path and prompt_candidate:
            self._apply_prompt_mutation(prompt_candidate)
        
        # 3. Run program in sandbox
        results = []
        for example in batch:
            result = self.sandbox.run(
                script_path=self.run_path,
                example=example,
                capture_traces=capture_traces
            )
            results.append(result)
        
        return EvaluationBatch(
            outputs=[r["output"] for r in results],
            scores=[r["score"] for r in results],
            trajectories=[r.get("traces") for r in results] if capture_traces else None,
        )
    
    def _apply_code_mutation(self, mutation: dict):
        """Delegate to coding agent to execute code change."""
        from src.core.agent import run_coding_agent
        run_coding_agent(
            workspace=self.sandbox.workspace_path,
            change_request=mutation["description"],
            target_files=mutation.get("target_files"),
        )
        self.git.commit(message=f"GEPA code mutation: {mutation['description'][:50]}")
```

### CodeEvolverToolProvider

```python
# In CodeEvolver: src/adapters/tool_provider.py

from gepa.core.tools import ToolProvider
import subprocess

class CodeEvolverToolProvider(ToolProvider):
    """
    Tool implementations that work on the cloned repo.
    Injected into GEPA's ReflectiveAgentProposer.
    """
    
    def __init__(self, workspace_path: str):
        self.workspace = workspace_path
    
    def read(self, path: str) -> str:
        full_path = os.path.join(self.workspace, path)
        with open(full_path) as f:
            return f.read()
    
    def grep(self, pattern: str, path: str = ".") -> list[dict]:
        result = subprocess.run(
            ["rg", "--json", pattern, path],
            cwd=self.workspace,
            capture_output=True, text=True
        )
        # Parse ripgrep JSON output
        return self._parse_rg_output(result.stdout)
    
    def glob(self, pattern: str) -> list[str]:
        import glob as globlib
        return globlib.glob(
            os.path.join(self.workspace, pattern),
            recursive=True
        )
```

---

## API

### POST /optimizer/create_job

```python
@app.post("/optimizer/create_job")
async def create_job(request: CreateJobRequest) -> CreateJobResponse:
    config = json.loads(request.config_json)
    examples_location = save_examples(request.examples)
    job_id = create_job_record(config, examples_location)
    
    asyncio.create_task(run_optimization(job_id))
    return CreateJobResponse(job_id=job_id, status="started")

async def run_optimization(job_id: str):
    """Runs GEPA optimization with CodeEvolver adapter."""
    job = get_job(job_id)
    
    # 1. Clone repo
    git_service = GitService()
    workspace = await git_service.clone(job.repo_url, job.installation_id)
    
    # 2. Create adapter and tool provider
    sandbox = SandboxManager(workspace)
    adapter = CodeEvolverAdapter(sandbox, git_service, job.run_path)
    tool_provider = CodeEvolverToolProvider(workspace)
    
    # 3. Initialize GEPA with our adapter and tools
    from gepa import GEPAEngine, ReflectiveAgentProposer
    
    proposer = ReflectiveAgentProposer(
        reflection_lm=get_reflection_lm(),
        tool_provider=tool_provider,  # Our tools injected here
        allow_code_mutations=job.config.get("allow_code_mutations", False),
    )
    
    engine = GEPAEngine(
        adapter=adapter,
        reflective_proposer=proposer,
        **job.gepa_settings
    )
    
    # 4. Run optimization
    result = engine.run()
    save_result(job_id, result)
```

### Config JSON

```python
config_json = {
    "repo_url": "https://github.com/user/project",
    "run_path": "optimize/run_and_eval.py",
    "program_json_path": "program.json",  # Optional, for DSPy
    "allow_code_mutations": True,
    "gepa_settings": {
        "max_metric_calls": 500,
        "perfect_score": 1.0,
        "seed": 42,
        "reflection_lm": "anthropic/claude-sonnet-4-20250514",
    }
}
```

### Examples (JSONL)

```jsonl
{"query": "What is the capital of France?", "label": "Paris"}
{"query": "What is the capital of Texas?", "label": "Austin"}
```

---

## User's Run Script

User provides `run_and_eval.py` in their repo:

```python
# User's optimize/run_and_eval.py
from my_program import MyAgent
from schemas.example import Example

def run_program(example: Example) -> dict:
    """Execute program on single example."""
    agent = MyAgent()
    return {"prediction": agent.run(example.query)}

def calculate_metric(prediction: dict, example: Example) -> float:
    """Calculate reward score (0-1)."""
    return 1.0 if prediction["prediction"] == example.label else 0.0
```

---

## Tracing

For MVP, assume users have DSPy:
- DSPy captures traces automatically
- Traces passed to GEPA's reflective agent for analysis
- MLflow integration optional

---

## Summary: Who Provides What

| Component | Repository | Notes |
|-----------|------------|-------|
| GEPAEngine, GEPAState | GEPA Fork | Core algorithm unchanged |
| GEPAAdapter protocol | GEPA Fork | Interface definition |
| ToolProvider protocol | GEPA Fork | NEW - tool interface |
| ReflectiveAgentProposer | GEPA Fork | NEW - agent with tools |
| CodeEvolverAdapter | CodeEvolver | Implements GEPAAdapter |
| CodeEvolverToolProvider | CodeEvolver | Implements ToolProvider |
| Git clone/worktree | CodeEvolver | Works with sandbox filesystem |
| Modal sandbox execution | CodeEvolver | Runs user's program |
| Coding agent | CodeEvolver | Executes code mutations |
| API endpoints | CodeEvolver | Job management |

---

## Contribution Path to Upstream GEPA

The following fork modifications could be contributed back:

1. **Tool Interface Protocol** - Generally valuable for any GEPA user wanting richer reflection
2. **ReflectiveAgentProposer** - Novel extension, likely accepted
3. **Extended candidate format for code** - requires discussion with gepa maintainers

The `CodeEvolverAdapter` and `CodeEvolverToolProvider` are specific to our service and stay in CodeEvolver.
