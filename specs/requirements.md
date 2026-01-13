# CodeEvolver Agents Requirements

## Overview
An open source remote service for executing evolutionary code changes. Users' code flows only to: our server → Claude → execution environment. Privacy and speed are paramount. 

## Technology Stack
- **Language**: Python
- **API Framework**: FastAPI
- **Database**: MongoDB (flexible, but preferred)
- **Execution Environment**: Rapid-scale option (Modal, Fly.io, or similar) - must spin up in <10 seconds, support 20+ concurrent environments per user

## Components

### API (FastAPI)

#### POST /execute_step
Executes one optimization step: applies a mutation, runs the program, returns output for GEPA reward calculation.

**Request Payload**:
```json
{
  "client_id": "string",
  "program_id": "string (new program id for this mutation)",
  "parent_program_id": "string (parent program id)",
  "mutation_type": "prompt | code",

  // Program location (path from project root):
  "program_json_path": "path/to/program.json",
  "entry_point": "module.ClassName",  // DSPy module to instantiate

  // For prompt mutations - GEPA's candidate format:
  "candidate": {
    "component_name": "new instruction text",
    ...
  },

  // For code mutations - natural language:
  "change_request": "string (natural language description)",
  "change_location": "string (module path, optional)",

  // Test data to run after mutation:
  "test_examples": [
    {"input_field": "value", ...},
    ...
  ],
  "capture_traces": false
}
```

**Field Details**:
- `program_json_path`: Path to program.json from project root (we have the code via /connect-git)
- `entry_point`: DSPy module class to instantiate and run (e.g., `"fire.FIREJudge"`)
- `candidate`: For prompt mutations - `dict[str, str]` mapping component names to new instruction text
- `change_request`: For code mutations - natural language description for Claude agent
- `test_examples`: DSPy Examples for running the mutated program
- `capture_traces`: Whether to return execution traces for GEPA reflection

**Response**:
```json
{
  "program_id": "string",
  "status": "success | failed",
  "pipeline_outputs": [
    {"example_id": 0, "output": <any DSPy forward() return value>},
    ...
  ],
  "traces": [...],  // If capture_traces=true
  "branch_name": "string (for code mutations)",
  "program_json": {...},  // Updated program state after mutation
  "error": "string (if failed)"
}
```

- `pipeline_outputs`: Raw outputs from running the mutated program on each test example
- GEPA computes scores client-side using outputs + ground truth labels

#### POST /connect-git
Registers a client repository. Minimal payload - paths provided per-request in `/execute_step`.

**Request**:
```json
{
  "repo_url": "https://github.com/user/project"
}
```

**Response**:
```json
{
  "client_id": "client_abc123",
  "status": "connected"
}
```

- Public repos initially, private repo support later
- Clones repo to server storage
- Returns `client_id` for future requests

#### GET /program/{program_id}
Retrieves program details and `program_json`.

### Program Database (MongoDB)
Stores all program versions and their optimized prompts.

| Field | Description |
|-------|-------------|
| `client_id` | Internal client identifier |
| `program_id` | Unique program version identifier |
| `parent_program_id` | Parent program(s) - 1 for mutation, 2 for crossover |
| `program_json` | DSPy optimized program JSON (see `specs/example_program.json`) |
| `branch_name` | Git branch for this program version |
| `created_at` | Timestamp |
| `status` | pending / in_progress / completed / failed |

**Purpose**: Centralizes prompt changes for direct editing by external optimizer (GEPA).

### DSPy Program JSON Structure
The `program_json` is the serialized output of DSPy's `program.save()`. Structure:

```json
{
  "module_path.predict": {
    "traces": [],
    "train": [],
    "demos": [],
    "signature": {
      "instructions": "The system prompt / task description",
      "fields": [
        {"prefix": "Input:", "description": "field description"},
        {"prefix": "Output:", "description": "field description"}
      ]
    },
    "lm": null
  },
  "another_module.predict": { ... },
  "metadata": {
    "dependency_versions": {"python": "3.11", "dspy": "3.0.4", ...}
  }
}
```

Key mutation targets:
- `signature.instructions`: The main prompt text (primary target for prompt mutations)
- `signature.fields[].description`: Field descriptions
- `demos`: Few-shot examples (can be added/modified)
- Module structure itself (code mutations only)

### Auto-coder Agent
Executes the actual code/prompt changes using Claude Agents SDK.

**SDK Details** (from research):
- **Requires files on disk** - cannot work with in-memory files
- **API Access**: Use `query()` or `ClaudeSDKClient` from `claude_agent_sdk`
- **Concurrent execution**: Each agent needs its own `cwd` (working directory)
- **Permission mode**: `acceptEdits` for autonomous code modifications

**Implementation Pattern**:
```python
from claude_agent_sdk import query, ClaudeAgentOptions

async def run_mutation(workspace_path: str, change_request: str):
    async for message in query(
        prompt=change_request,
        options=ClaudeAgentOptions(
            cwd=workspace_path,  # Isolated checkout for this branch
            allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits"
        )
    ):
        yield message
```

**Capabilities**:
- **Prompt mutations**: Modify `program_json` (still needs code checkout to run program)
- **Code mutations**: Edit actual Python DSPy module code in git repo
- **Constraints**: Future support for constrained edits

### Execution Environment
Each agent runs in an isolated environment with its own filesystem.

**Git Worktree Strategy**:
Instead of multiple clones, use git worktree for efficient parallel branch access:
```bash
# Single clone per client
git clone <repo> /workspace/{client_id}/main

# Create worktree per program/mutation
git worktree add /workspace/{client_id}/prog_042 branch_prog_042
git worktree add /workspace/{client_id}/prog_043 branch_prog_043
```

Benefits:
- Single clone, multiple working directories
- Shared `.git` metadata = faster than multiple clones
- Each worktree has different branch checked out
- Changes (fetch) visible across all worktrees

**Requirements**:
- Spin-up time: <10 seconds (worktree add is near-instant)
- Concurrency: 20+ simultaneous worktrees per client
- Isolation: Each worktree = one branch, separate directory
- Contains: Worktree checkout, Claude Agents SDK runtime, Python/DSPy

**Execution Platform: Modal**

Decision: Use [Modal](https://modal.com) for execution environments.

Rationale:
- ~1-2s cold start (meets <10s requirement)
- Python-native SDK
- Apache-2.0 friendly (no license conflicts)
- Good concurrency support
- Built-in container caching

```python
import modal

app = modal.App("codeevolver-agents")

@app.function(
    image=modal.Image.debian_slim().pip_install("dspy", "anthropic"),
    timeout=300,
)
async def execute_mutation(
    workspace_path: str,
    candidate: dict[str, str],
    test_examples: list[dict],
) -> dict:
    # Clone/worktree setup, apply mutation, run program
    ...
```

Future options if needed:
- Daytona (faster, but requires commercial license for MIT/Apache compatibility)
- Self-hosted (maximum privacy)

### Run Program
Executes the mutated DSPy program and returns output.

- Loads program from `program_json`
- Runs as fixed DSPy Adapter
- Returns `pipeline_output` to GEPA for reward calculation

### Git Branching
Handles version control for code mutations.

- Creates branch `program_{program_id}` from parent branch
- For 2-parent crossover: merge strategy TBD
- Can be handled by auto-coder agent or separate bot

## External Integration

### GEPA Integration (gepa-ai/gepa)
GEPA orchestrates the evolutionary optimization. Currently runs on client machine, will eventually run fully on this service.

**GEPA Architecture** (from codebase analysis):
- Uses `GEPAAdapter` protocol for program execution
- **Candidate format**: `dict[str, str]` mapping component names → instruction text
- DSPy programs wrapped via `DspyAdapter` which calls `named_predictors()`
- Pareto frontier tracks best programs per validation instance
- Mutations selected from frontier programs (may span different branches)

**Integration Pattern - CodeEvolverAdapter**:
GEPA will use a custom adapter that calls our API:

```python
class CodeEvolverAdapter(GEPAAdapter):
    def __init__(self, api_endpoint: str, client_id: str,
                 program_json_path: str, entry_point: str, metric_fn):
        self.api_endpoint = api_endpoint
        self.client_id = client_id
        self.program_json_path = program_json_path
        self.entry_point = entry_point
        self.metric_fn = metric_fn  # Scoring done client-side

    def evaluate(
        self,
        batch: list[Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        response = requests.post(
            f"{self.api_endpoint}/execute_step",
            json={
                "client_id": self.client_id,
                "program_id": generate_program_id(),
                "parent_program_id": self.current_parent_id,
                "mutation_type": "prompt",
                "program_json_path": self.program_json_path,
                "entry_point": self.entry_point,
                "candidate": candidate,
                "test_examples": [ex.toDict() for ex in batch],
                "capture_traces": capture_traces
            }
        )
        result = response.json()

        # Scores computed client-side by GEPA
        outputs = result["pipeline_outputs"]
        scores = [self.metric_fn(ex, out) for ex, out in zip(batch, outputs)]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=result.get("traces")
        )
```

**Communication Flow**:
1. GEPA runs locally (for now), selects candidate from Pareto frontier
2. GEPA calls `/execute_step` with candidate + test examples
3. CodeEvolver service: spins up environment, applies mutation, runs program
4. Returns `pipeline_outputs` → GEPA computes reward with ground truth labels
5. GEPA updates Pareto frontier, selects next mutation

**Future**: GEPA runs entirely on CodeEvolver service (once mutation logic is finalized)

## Mutation Types

### Prompt Mutations (most common)
- Edit `signature.instructions` in `program_json` for specific components
- GEPA sends candidate as `dict[str, str]`: `{"component_name": "new instruction text"}`
- Still requires code checkout to run the program after mutation
- Example: `{"fire_judge.judge.predict": "You are a fact-checking judge. Be more conservative..."}`

### Code Mutations (less common)
- Edit Python DSPy module code in repository (add modules, change structure)
- Requires Claude agent to understand and modify code
- Creates new branch, commits changes
- Example: "Add a sub-module that searches company websites before classification"

## Performance Requirements
- Optimization runs are slow, so parallelization is critical
- Target: 20 mutations tested simultaneously per user
- Environment spin-up: <10 seconds
- Prompt mutations should be near-instant (DB-only path)

## Privacy Considerations
- Minimize data hops: User → Our Server → Claude → Execution Env
- Future: Option for users to run execution environment on their own infrastructure
- No unnecessary third-party services

## Other Requirements
- Our License will be either MIT or Apache, so cannot incorporate any GNU GPL or Affero licenses

---

## Implementation Status

### Completed (v0.1.0)
- [x] POST /connect-git - Clone repository and register client
- [x] POST /execute_step - Apply mutations (prompt only), run program (placeholder)
- [x] GET /program/{program_id} - Retrieve program details
- [x] GET /health - Health check
- [x] MongoDB integration with Motor (async)
- [x] Git worktree management for parallel branches
- [x] Pydantic schemas for all request/response types

### Pending
- [ ] DSPy runtime integration (ProgramRunner.run_program returns placeholder)
- [ ] Claude Agents SDK integration for code mutations
- [ ] Modal integration for execution environments
- [ ] Private repository authentication

### Implementation Notes
- **Git Worktrees**: Using GitPython's `git.worktree` commands. Each program gets its own worktree directory at `{workspace_root}/{client_id}/{program_id}/`
- **Prompt Mutations**: Directly edit `signature.instructions` in program.json, then commit
- **Code Mutations**: Return 501 Not Implemented until Claude Agents SDK is integrated
- **Program Execution**: Returns placeholder outputs - DSPy runtime integration needed