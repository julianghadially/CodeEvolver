# Requirements

## Overview
CodeEvolver offers autonomous coding agents for high reliability AI systems. It uses GEPA optimization to evolve your AI system code until it performs optimally for a given dataset and outcome metric.

This combines several mechanisms:
- **Optimizer algorithm:** GEPA is a reflective language model algorithm that makes point mutations to the code base, over many iterations, and the best solution is selected, based on a dataset and a reward metric.
- **Coding agents**: Autonomous agents execute code changes that are requested by the optimizer. 
- **Git branching:** A git process manages evolving code across many git worktrees  
- **Sandboxing for security:** Coding agents are a big cyber risk without sandboxing, network policies, etc. 

### Optimizer
The optimizer is handled by a separate repository, which will later be loaded into this repository, as defined by specs/gepa_plan.md. This repository will create a /optimize endpoint to run GEPA optimization orchestration, and Will interface with that package as defined in gepa_plan.md. 

### Coding Agents
CodeEvolver agents uses Claude Agents SDK in a fully autonomous, dangerously-skip-permissions mode, which uses a Modal sandbox execution environment for modifying code, running code, and executing bash / grep / glob. After code changes are made, the app needs to run a mutated version of the code, and return the output. 

Code changes will be made in the context of GEPA optimization - i.e., an evolutionary, 100+ step process. Speed and parallel execution of coding changes is important. The AI worfklow code needs to be edited over 100 times. Each mutation is small, but costs will add up. Do not worry about cost right now.

### Git branching
Users Connect their code with our service by adding our GitHub app, which adds our organization as a contributor to their GitHub.

### Security
Security should be designed for from day one, because autonomous coding agents introduce the trifecta of security risk: 
1. Untrusted inputs, including prompt injection embedded into popular sites
2. Network access
3. Access to user data (RAG databases, code, secrets, and possibly PII). 

See security architecture below.

## V1 outcomes (current goal):
- Connect a GitHub repository [complete]
- Execute a change request [complete]
- GEPA optimizer runs in CodeEvolver [WIP]
- Complete v1 of security: (see for v1 below)
- API / sandbox deployed to Modal App

## Technology Stack and Architecture
- **Language**: Python
- **API Framework**: Modal Sandbox App serving FastAPI
- **Database**: MongoDB (flexible, but preferred)
- **Execution Environment**: Modal Sandbox. (must spin up in <10 seconds, support 20+ concurrent environments per user)

**Sandbox Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Modal Web Endpoint (FastAPI)                                    │
│  - Receives HTTP requests                                        │
│  - Creates Modal sandbox for each change request                 │
│  - Manages MongoDB connections                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Creates Sandbox
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Modal Sandbox (per mutation)                                    │
│  ───────────────────────────────────────────────────────────── │
│  /workspace/                                                     │
│    ├── .git/                                                     │
│    ├── requirements.txt  ← pip install -r this                  │
│    ├── src/                                                      │
│    └── program.json                                              │
│                                                                  │
│  Claude Agent SDK runs HERE:                                     │
│  - Native Bash tool → subprocess.run() ✅                        │
│  - Native Grep tool → subprocess.run("grep") ✅                  │
│  - Native Read/Edit → file operations ✅                         │
│  - pip install → works dynamically ✅                            │
│                                                                  │
│  Lifecycle:                                                      │
│  1. Sandbox.create() → container starts                          │
│  2. git clone repo, pip install dependencies                     │
│  3. Run Claude Agent with full capabilities                      │
│  4. sandbox.terminate() → container destroyed                    │
└─────────────────────────────────────────────────────────────────┘
```

## Security Architecture
- **Client-specific isolation (v2):** Execution of code will be isolated in v2. Each client should be in a separate container (e.g., client could have malicious code to steal other clients' data or secrets)
- **Network Egress Control and whitelists:** Limit urls to allowed domains and ips set by our best practices and by the user (e.g., api.firecrawl.dev)
- **Secrets management (v2)**: use env file for v1
- **Monitoring and detection:** omit for v1


```
┌─────────────────────────────────────────────────────────────────┐
│                       CodeEvolver Service (v2)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      API Gateway                          │   │
│  │  - Authentication                                         │   │
│  │  - Rate limiting                                          │   │
│  │  - Request validation                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Secrets Manager                        │   │
│  │  - Per-client encrypted secrets                           │   │
│  │  - Never exposed to agent                                 │   │
│  │  - Injected via proxy                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Client A    │  │  Client B    │  │  Client C    │          │
│  │  Sandbox     │  │  Sandbox     │  │  Sandbox     │          │
│  │  ──────────  │  │  ──────────  │  │  ──────────  │          │
│  │  - Isolated  │  │  - Isolated  │  │  - Isolated  │          │
│  │  - Own net   │  │  - Own net   │  │  - Own net   │          │
│  │  - Egress    │  │  - Egress    │  │  - Egress    │          │
│  │    proxy     │  │    proxy     │  │    proxy     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Egress Proxy                          │   │
│  │  - Whitelist domains                                      │   │
│  │  - Inject secrets as headers                              │   │
│  │  - Log all outbound traffic                               │   │
│  │  - Block unauthorized destinations                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│                    ┌───────────────────┐                         │
│                    │ Allowed APIs      │                         │
│                    │ - Claude          │                         │
│                    │ - OpenAI          │                         │
│                    │ - Our whitelist   │                         │
│                    │ - Users whitelist │                         │
│                    └───────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Security needs for v1:** 
- Keep workers specific to each client
- Make API requests direct to modal / sandbox app (Omit separate api gateway)
- No egress proxy (temp)
- Use env for secrets (temp)

### Privacy
- Future: Option for users to run execution environment on their own private cloud
- No unnecessary third-party services. Necessary third parties include modal.


-------------------------------------------------------------------------


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

  // AI Workflow program location (path from project root):
  "program_json_path": "path/to/program.json",
  "entry_point": "module.ClassName",  // DSPy module to instantiate < is this needed?

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
- [x] MongoDBProgressTracker for per-iteration state persistence (src/gepa/progress.py)
- [x] GEPA optimization orchestrator (src/gepa/optimizer.py)
- [x] POST /optimize endpoint - starts GEPA optimization job
- [x] GET /job/{job_id} endpoint - returns job status and progress
- [x] Job schemas (OptimizeRequest, JobStatusResponse, JobRecord)
- [x] Modal function for long-running GEPA optimization (run_optimization)
- [x] GEPA image with DSPy, litellm, pymongo dependencies

### Pending
- [ ] End-to-end testing of GEPA optimization (/optimize)
- [ ] Implementation of run_program and evaluate (standalone)
- [ ] End-to-end testing of code mutations + run_program and evaluate
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

**Architecture (v0.3.0 - GEPA integration):**
```
src/
  core/                    # Core logic (inspired by modal-vibe)
    __init__.py
    sandbox.py             # SandboxApp class, execute_mutation()
    agent.py               # Claude agent script generation
    program_runner.py      # DSPy program execution
    system_prompt.py       # Prompts for coding agent
  gepa/                    # GEPA optimization integration
    __init__.py
    adapter.py             # CodeEvolverDSPyAdapter (GEPAAdapter protocol)
    optimizer.py            # run_gepa_optimization() orchestrator
    progress.py            # MongoDBProgressTracker (StopperProtocol)
  services/                # Supporting services
    git_service.py         # Git operations (clone, worktree, commit)
    github_app.py          # GitHub App authentication (used by sandbox.py)
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
- **Program Execution**: Generates runner scripts for DSPy execution (placeholder until DSPy integration)
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
- `MongoDBProgressTracker` implements `StopperProtocol` — called each iteration to persist state and check cancellation
- Uses `pymongo` (sync) inside the optimization loop; FastAPI endpoints use `motor` (async)

**Sync/Async Strategy:**
GEPA's `optimize()` is synchronous. The Modal function is sync. The adapter uses DSPy's `Evaluate` (sync) and `bootstrap_trace_data` (sync). MongoDB operations inside the optimization loop use `pymongo` (sync driver), not `motor`.

**GEPA Package:**
- Development: installed from local GEPA-CodeEvolver dir via `add_local_dir()` + `pip install -e`
- Production: will switch to `pip install gepa @ git+https://github.com/<org>/GEPA-CodeEvolver.git`

**V1 Scope (prompt-only):**
- Candidates are `dict[str, str]` mapping predictor names to instruction text
- No git branching — instructions applied in-memory via `pred.signature.with_instructions()`
- Reflection uses GEPA's default `InstructionProposalSignature` with litellm
- No Claude Agent SDK for reflection (deferred to v2 with code mutations)

**Metric Function:**
Users provide a metric as a single dotted import path (e.g., `eval.evaluate.metric`). Last component is the function name. The function signature must be `metric(example: dspy.Example, prediction: dspy.Prediction) -> float`.

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

**Evaluation: DSPy-native (v0.3.2)**
The adapter evaluates directly using DSPy — no eval script needed for prompt-only optimization.
The adapter imports the user's DSPy module, applies candidate prompt instructions in-memory,
runs each example, and scores with the user's metric function.

Sandbox mode with eval scripts is deferred to v2 (code mutations).

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
```