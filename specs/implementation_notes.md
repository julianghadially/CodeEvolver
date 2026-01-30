
--------------------------------------------------------------------------------

# Ongoing notes on requirements

Below this line is an ongoing, workspace for Claude / AI coding agents. Do not edit text above the line unless it is fully incorrect.

## Components

### API (FastAPI served by Modal app)

### POST /optimize API (CodeEvolver endpoint)

**Request:**
```json
{
  "repo_url": "https://github.com/user/project",
  "program": "src.module.ClassName",
  "metric": "eval.evaluate.metric",
  "trainset_path": "data/train.jsonl",
  "reflection_lm": "openai/gpt-5-mini",
  "max_metric_calls": 1000,
  "num_threads": 1
}
```

- `program`: Dotted import path to DSPy program class. Last component is class name.
- `metric`: Dotted import path to metric function. Last component is function name.
- `trainset_path`: Path to training data file in repo (.json, .jsonl, or .csv). Alternative: send inline `trainset` list.
- `valset_path`: Optional path to validation data file. Alternative: send inline `valset` list.
- `saved_program_json_path`: Optional path to saved DSPy program state (program.json).

**Response:**
```json
{
  "status": "completed",
  "best_candidate": {"git_branch": "...", "module_1.predict": "..."},
  "best_score": 0.92,
  "logs_url": "..."
}
```

#### POST /execute_step
Receives a change request from GEPA optimization as input, and executes that single optimization step: applies a mutation, runs the program, returns output for GEPA reward calculation.

**Request Payload**:
```json
{
  "client_id": "string",
  "program_id": "string (new program id for this mutation)",
  "parent_program_id": "string (parent program id)",
  "mutation_type": "prompt | code",

  // DSPy program module (dotted import path):
  "program": "module.ClassName",  // DSPy module class to import and instantiate
  "saved_program_json_path": "path/to/program.json",  // Optional: saved program state

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
- `program`: Dotted import path to DSPy program class to import and instantiate (e.g., `"fire.FIREJudge"`)
- `saved_program_json_path`: Optional path to saved program.json from project root (if provided, loads saved state)
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
  "repo_url": "https://github.com/user/project",
  "installation_id": 12345  // Optional: GitHub App installation ID for private repos
}
```

**Response**:
```json
{
  "client_id": "client_abc123",
  "status": "connected"
}
```

- Supports both public and private repositories
- Private repos require `installation_id` from GitHub App installation
- Uses `GitHubAppService` for token-based authentication
- Clones repo to server storage
- Returns `client_id` for future requests

#### POST /change_request
Execute a code change via the Claude coding agent using GEPASandbox.

**Request**:
```json
{
  "repo_url": "https://github.com/user/project",
  "change_request": "Add a docstring to the main function",
  "change_location": "src/core/agent.py",  // Optional hint
  "branch_name": "feature-branch",         // Optional, auto-generated if not provided
  "push_to_remote": true,
  "installation_id": 12345                 // Optional: GitHub App installation ID
}
```

**Response**:
```json
{
  "success": true,
  "branch_name": "feature-branch",
  "error": null,
  "output": "Agent execution logs..."
}
```

- Uses `GEPASandbox.exec_agent()` for code mutations
- Creates venv isolation for client dependencies
- Commits changes automatically after successful mutation
- Optionally pushes to remote if `push_to_remote=true`

**Deprecated**: `/execute_sandbox` endpoint still works but redirects to `/change_request`

#### GET /job/{job_id}
Retrieves job status, progress, and current best candidate.

**Response**:
```json
{
  "job_id": "string",
  "status": "running | completed | failed | paused",
  "current_iteration": 42,
  "total_iterations": 100,
  "best_candidate": {"git_branch": "...", "module_1.predict": "..."},
  "best_score": 0.87,
  "created_at": "2026-01-26T...",
  "updated_at": "2026-01-26T..."
}
```

### Job Database (MongoDB)
Stores optimization jobs with GEPAState for resumption and coordination. Individual program candidates live inside `gepa_state.program_candidates`.

| Field | Description |
|-------|-------------|
| `job_id` | Unique optimization job identifier |
| `client_id` | Client identifier |
| `repo_url` | Repository being optimized |
| `config` | Job configuration (iterations, batch_size, mutation_type, etc.) |
| `status` | `pending` / `running` / `completed` / `failed` / `paused` |
| `gepa_state` | Full GEPAState object (for resumption) |
| `current_iteration` | Progress tracking |
| `best_candidate` | Current best candidate (denormalized for quick access) |
| `best_score` | Current best score |
| `created_at` | Job creation timestamp |
| `updated_at` | Last state update timestamp |

**State Persistence**: GEPAState saved every iteration to enable job resumption after failures.

**Purpose**: Enables long-running optimization jobs to survive failures and coordinate across servers. Final GEPAState returned to client on completion.

### DSPy Program JSON Structure
The `program_json` is the serialized output of DSPy's `program.save()`. It is created from the GEPA state candidates by using adapter.build_program(). Program JSON Format:

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
- **Requires Claude Code CLI** - The Python SDK (`claude-agent-sdk`) is a wrapper that spawns the Claude Code CLI (`@anthropic-ai/claude-code`) as a subprocess. Both must be installed.
- **Requires files on disk** - cannot work with in-memory files
- **API Access**: Use `query()` or `ClaudeSDKClient` from `claude_agent_sdk`
- **Concurrent execution**: Each agent needs its own `cwd` (working directory)
- **Permission mode**: `acceptEdits` for autonomous code modifications (no human prompts)
- **Tools available**: Read, Write, Edit, Bash, Glob, Grep (Write for new files, Edit for existing)

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

**Execution Platform: Modal Sandbox**

Decision: Use [Modal Sandbox](https://modal.com/docs/guide/sandboxes) for execution environments.

Architecture inspired by [Modal Vibe](https://github.com/modal-labs/modal-vibe).

**Key Insight:** The Claude Agent SDK runs **inside** the Modal Sandbox, so its native tools (Bash, Grep, Glob, Read, Edit) work directly via subprocess. No custom tool wrappers needed.

Rationale:
- Modal Sandbox provides persistent filesystem while alive
- Dynamic pip install (user's requirements.txt with unknown packages)
- Full bash/python access for Claude Agent
- Apache-2.0 friendly (no license conflicts)
- Client isolation via separate sandboxes

**Architecture:**
See sandbox architecture above

**Implementation Pattern (src/core/sandbox.py):**

```python
import modal
from ..services.github_app import GitHubAppService

class SandboxApp:
    """Manages a Modal sandbox for executing code mutations."""

    @staticmethod
    async def create(
        app: modal.App,
        client_id: str,
        program_id: str,
        repo_url: str,
        installation_id: int | None = None,  # For private repos
        secrets: dict[str, str] | None = None,
    ) -> "SandboxApp":
        # Handle private repo authentication via GitHubAppService
        authenticated_url = repo_url
        if installation_id:
            token = GitHubAppService.get_installation_token(installation_id)
            authenticated_url = GitHubAppService.get_authenticated_repo_url(repo_url, token)

        sandbox = modal.Sandbox.create(app=app, image=get_sandbox_image(), timeout=600)
        sandbox_app = SandboxApp(sandbox, metadata)

        await sandbox_app._clone_repo(authenticated_url)
        await sandbox_app._install_deps()
        if secrets:
            await sandbox_app._inject_secrets(secrets)

        return sandbox_app

    async def apply_code_mutation(self, change_request: str) -> MutationResult:
        # Generate agent script that runs Claude SDK inside sandbox
        script = generate_agent_script(self._workspace, change_request)
        self.sandbox.exec("bash", "-c", f"cat > /tmp/agent.py << 'EOF'\n{script}\nEOF").wait()
        p = self.sandbox.exec("python", "/tmp/agent.py")
        p.wait()
        return parse_agent_output(p.stdout.read(), p.stderr.read(), p.returncode)

# Main entry point
async def execute_mutation(app, client_id, program_id, repo_url, ..., installation_id=None):
    sandbox_app = await SandboxApp.create(app, client_id, program_id, repo_url, installation_id)
    try:
        mutation_result = await sandbox_app.apply_code_mutation(change_request)
        run_result = await sandbox_app.run_program(...)
        return ExecutionResult(...)
    finally:
        sandbox_app.terminate()
```

**Why Sandbox over Modal Function:**

| Feature | Modal Function | Modal Sandbox |
|---------|---------------|---------------|
| Dependencies | Baked at deploy time | Dynamic pip install ✅ |
| Filesystem | Ephemeral per call | Persistent while alive ✅ |
| Bash access | Limited | Full interactive shell ✅ |
| User code | Can't adapt to requirements.txt | Installs anything ✅ |

Future options if needed:
- Self-hosted Docker (maximum privacy, same pattern)
- User's own infrastructure (enterprise)

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

## GEPA Integration
GEPA orchestrates the evolutionary optimization. Runs on this service via /optimize.

**GEPA Plan**: see specs/gepa_plan.md

### Mutation Types

#### Prompt Mutations (most common)
- Edit `signature.instructions` in `program_json` for specific components
- GEPA sends candidate as `dict[str, str]`: `{"component_name": "new instruction text"}`
- Still requires code checkout to run the correct program branch after mutation

#### Code Mutations
- Edit Python DSPy module code in repository (add modules, change structure)
- Requires Claude agent to understand and modify code
- Creates new branch, commits changes
- Example: "Add a sub-module that searches company websites before classification"

### GEPA Performance Requirements
- Optimization runs are slow, so parallelization is critical
- Target: 10 mutations tested simultaneously per user
- Environment spin-up: <10 seconds
- Prompt mutations should be near-instant (DB-only path)


## Other Requirements
- Our License will be either MIT or Apache, so cannot incorporate any GNU GPL or Affero licenses.

---

## Implementation Status

### Completed (v0.1.0)
- [x] POST /connect-git - Clone repository and register client
- [x] POST /execute_step - Apply mutations (prompt only), run program (placeholder)
- [ ] GET /job/{job_id} - Retrieve job status and best candidate
- [x] GET /health - Health check
- [x] MongoDB integration with Motor (async)
- [x] Git worktree management for parallel branches
- [x] Pydantic schemas for all request/response types
- [x] Modal app structure with FastAPI web endpoint
- [x] Private repository authentication (GitHub App)

### Completed (v0.2.0)
- [x] Restructured codebase with `src/core/` module (inspired by modal-vibe)
- [x] SandboxApp class for managing Modal sandbox lifecycle
- [x] Agent module for Claude SDK integration (generates agent scripts)
- [x] Program runner module for DSPy execution (generates runner scripts)
- [x] System prompt module for code mutation prompts
- [x] Removed old mutation_service.py and sandbox_executor.py (duplicates)
- [x] Integrated GitHubAppService into sandbox.py for private repo authentication
- [x] Successfully tested single code mutations

### Completed (v0.3.0 - GEPA Integration)
- [x] CodeEvolverDSPyAdapter implementing GEPAAdapter protocol (src/gepa/adapter.py)
- [x] CallbackProgressTracker for per-iteration state persistence via HTTP (src/gepa/callback.py)
- [x] GEPA optimization orchestrator (src/gepa/optimizer.py)
- [x] POST /optimize endpoint - starts GEPA optimization job
- [x] GET /job/{job_id} endpoint - returns job status and progress
- [x] Job schemas (OptimizeRequest, JobStatusResponse, JobRecord)
- [x] Modal function for long-running GEPA optimization (run_optimization)
- [x] Process-level isolation: ClientSandbox base class + GEPASandbox (no dspy in orchestrator)
- [x] Sandbox prebuilt scripts for DSPy evaluation (src/core/sandbox_scripts/)
- [x] End-to-end testing of GEPA optimization (/optimize)

### Completed (v0.3.3 - Coding Agent Migration)
- [x] Migrated coding agent to GEPASandbox architecture with venv isolation
- [x] New `/change_request` endpoint using `GEPASandbox.exec_agent()`
- [x] Simplified request/response schemas (ChangeRequest, ChangeResponse)
- [x] `execute_change_request` Modal function replaces `execute_in_sandbox`
- [x] Deprecated `/execute_sandbox` endpoint (redirects to `/change_request`)

### Pending
- [ ] Add code change request to GEPA optimization loop
- [ ] End-to-end testing of GEPA with code mutations + evaluate
- [ ] Reflection LLM agent
- [ ] CodeEvolver service
- [ ] Proxy To safeguard (mask) our anthropic API key and unify into one key
- [ ] Code mutations in GEPA optimization (v2: Claude Agent SDK reflection)


### Implementation Notes (v0.2.1)
**Claude Agent SDK Architecture:**
The Claude Agent SDK is a Python wrapper around the Claude Code CLI. The SDK spawns the CLI as a subprocess, which connects to the Anthropic API and manages the agentic loop.

```
Your Code (Python) → claude-agent-sdk (pip) → Claude Code CLI (npm) → Anthropic API
```

**Required installations in sandbox:**
1. `npm install -g @anthropic-ai/claude-code` - The actual coding agent
2. `pip install claude-agent-sdk` - Python wrapper for programmatic access

**Permission modes for autonomous operation:**
- `acceptEdits` - Auto-accepts file edits, logs actions (recommended)
- `bypassPermissions` - Full autonomy, bypasses all checks (use with caution)

### Implementation Notes

**Architecture (v0.3.2 - Process Isolation):**
```
src/
  core/                    # Core logic (inspired by modal-vibe)
    __init__.py
    sandbox.py             # SandboxApp class, execute_mutation() (short-lived, per-mutation)
    client_sandbox.py      # ClientSandbox base class (long-lived, for optimizers)
    agent.py               # Claude agent script generation
    program_runner.py      # DSPy program execution
    system_prompt.py       # Prompts for coding agent
    sandbox_scripts/       # Scripts that run inside client sandboxes
      __init__.py
      master.py            # Main dispatcher (entry point)
      dspy/                # DSPy-specific handlers
        __init__.py        # Shared utilities
        build_seed_candidate.py
        evaluate.py
        make_reflective_dataset.py
  gepa/                    # GEPA optimization integration
    __init__.py
    adapter.py             # CodeEvolverDSPyAdapter (RPC proxy to sandbox)
    gepa_sandbox.py        # GEPASandbox (extends ClientSandbox)
    optimizer.py           # run_gepa_optimization() orchestrator
    callback.py            # CallbackProgressTracker (StopperProtocol via HTTP)
    utils.py               # Dataset loading utilities (no dspy)
  services/                # Supporting services
    git_service.py         # Git operations (clone, worktree, commit)
    github_app.py          # GitHub App authentication
  schemas/                 # Pydantic models
    job_schemas.py         # OptimizeRequest, JobStatusResponse, JobRecord
  db/                      # MongoDB integration
  config.py
  main.py                  # FastAPI endpoints
modal_app.py               # Modal app entrypoint (includes run_optimization)
```

- **Modal Architecture**: FastAPI runs as Modal web endpoint. Mutations execute via `execute_in_sandbox()` which calls `src.core.execute_mutation()`.
- **SandboxApp Pattern**: Similar to modal-vibe's `SandboxApp`, manages sandbox lifecycle: create -> clone -> install -> mutate -> run -> terminate.
- **Private Repo Authentication**: `SandboxApp.create()` accepts optional `installation_id` parameter. When provided, uses `GitHubAppService.get_installation_token()` and `get_authenticated_repo_url()` to clone private repositories.
- **Agent Scripts**: Code mutations generate Python scripts that run Claude Agent SDK inside the sandbox, where native tools work via subprocess.
- **Git Worktrees**: Using GitPython's `git.worktree` commands. Each program gets its own worktree directory at `{workspace_root}/{client_id}/{program_id}/`
- **Prompt Mutations**: Directly edit `signature.instructions` in program.json via `apply_prompt_mutation()`, then commit
- **Code Mutations**: Generate and execute agent scripts in sandbox (requires Modal deployment)
- **Program Execution**: Imports and instantiates DSPy modules directly (no runner scripts needed)
- **Local Development**: Use `modal serve modal_app.py` for dev, `modal deploy modal_app.py` for production

**Service Integration:**
- `src/core/sandbox.py` imports and uses `GitHubAppService` from `src/services/github_app.py`
- `src/services/git_service.py` handles local git operations (clone, worktree, commit) for the FastAPI layer
- Both services share the same authentication pattern for private repos via GitHub App tokens

### Implementation Notes (v0.3.0 - GEPA Integration)

**GEPA Optimization Architecture:**
- POST /optimize creates a job in MongoDB and spawns a Modal function via `.spawn()` (fire-and-forget)
- The `run_optimization` Modal function runs `gepa.optimize()` synchronously (blocking)
- `CodeEvolverDSPyAdapter` implements the `GEPAAdapter` protocol (structural typing, no inheritance)
- `CallbackProgressTracker` implements `StopperProtocol` — called each iteration to persist state via HTTP callbacks
- FastAPI endpoints use `motor` (async); optimization callbacks use `httpx` (sync)



**GEPA Package:**
- Installed from PyPI: `pip install gepa>=0.0.26`

**V1 Scope (prompt-only):**
- Candidates are `dict[str, str]` mapping predictor names to instruction text
- No git branching — instructions applied in-memory via `pred.signature.with_instructions()`
- Reflection uses GEPA's default `InstructionProposalSignature` with litellm
- No Claude Agent SDK for reflection (deferred to v2 with code mutations)

**Metric Function:**
Users provide a metric as a single dotted import path (e.g., `eval.evaluate.metric`). Last component is the function name. The function signature must be `metric(example: dspy.Example, prediction: dspy.Prediction) -> float`.

For detailed GEPA architecture, see `specs/gepa_plan.md`.

### Implementation Notes (v0.3.1 - Unified Calling Pattern)

**Unified Candidate Structure:**
All candidates now include a `git_branch` key for consistent tracking across prompt-only and code mutations:

```python
candidate = {
    "git_branch": "main",  # or "codeevolver-abc123" for mutations
    "claim_extractor.predict": "Extract factual claims...",
    "fire_judge.predict": "Evaluate the claim...",
    "aggregator.predict": "Aggregate verdicts...",
}
```

**Evaluation: Sandbox-Isolated (v0.3.2)**
The adapter delegates all DSPy operations to `GEPASandbox` via `exec_prebuilt()`. The sandbox runs prebuilt scripts (`sandbox_scripts/dspy/evaluate.py`) that:
1. Import the user's DSPy module
2. Apply candidate prompt instructions in-memory
3. Run each example through the program
4. Score with the user's metric function
5. Optionally capture traces for reflection
6. Return results as JSON via stdout

This process isolation prevents dependency conflicts between the orchestrator and client code.

**User Repository Requirements:**
Users provide:

| What | Format | Example |
|------|--------|---------|
| DSPy program | `program` dotted path | `src.factchecker.FactCheckerPipeline` |
| Metric function | `metric` dotted path | `eval.evaluate.metric` |
| Training data | `trainset_path` file in repo | `data/train.jsonl` |
| Program state | `saved_program_json_path` (optional) | `program.json` |

No eval script, no runner script. Template files in `specs/user_templates/`.

**Dataset Loading:**
Server loads data from `trainset_path` after cloning. Supports `.json`, `.jsonl`, `.csv`.
Alternatively, send inline `trainset` in the API request.

### Implementation Notes (v0.3.3 - Coding Agent Migration)

**Endpoint Rename:**
`/execute_sandbox` → `/change_request` with simplified schema:

| Old Field | New Field | Notes |
|-----------|-----------|-------|
| `client_id`, `program_id` | (removed) | Not needed for standalone code changes |
| `mutation_type` | (removed) | Always "code" |
| `program_json_path`, `entry_point` | (removed) | DSPy-specific, not needed |
| `candidate`, `test_examples`, `capture_traces` | (removed) | DSPy-specific |
| `skip_program_run` | (removed) | Always true for code-only |
| `change_request` | `change_request` | Required |
| `repo_url` | `repo_url` | Required |
| `change_location` | `change_location` | Optional hint |
| `branch_name` | `branch_name` | Optional |
| `push_to_remote` | `push_to_remote` | Optional |
| `installation_id` | `installation_id` | Optional |

**Response Schema:**
Changed from `status: "success" | "failed"` to `success: bool` for clarity.
Added `output` field for agent execution logs.

**Architecture:**
```
POST /change_request
    │
    ▼
execute_change_request (Modal Function, sandbox_image)
    │
    ├── GEPASandbox.start(use_venv=True)
    │       └── Creates sandbox with venv for client deps
    │
    ├── SandboxGitService.checkout(branch_name, create=True)
    │       └── Creates branch if specified
    │
    ├── GEPASandbox.exec_agent(change_request, ...)
    │       ├── generate_agent_script()
    │       ├── Execute with system Python (claude-agent-sdk)
    │       └── parse_agent_output()
    │
    ├── SandboxGitService.push(branch_name)  [if push_to_remote]
    │
    └── GEPASandbox.stop()
```

**Key Changes:**
- Uses `GEPASandbox` (extends `ClientSandbox`) instead of deprecated `SandboxApp`
- Agent runs with system Python (has claude-agent-sdk)
- Client code runs in venv (`/workspace/.venv`)
- Git operations via `SandboxGitService` (runs bash commands in sandbox)
- Uses `sandbox_image` which has claude-agent-sdk, git, httpx, pyjwt
```