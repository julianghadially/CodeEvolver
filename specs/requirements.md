# Requirements
Copyright © 2026 440 Labs LLC
See specs/implementation_notes.md for additional detail.

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


### Optimizer Details (incl. ClientSandbox)

CodeEvolver Optimizers must execute client code in a sandbox that Matches the clients environment and provides a security layer. (we are executing client code as well as AI generated code).

Meanwhile, The optimizer process is a long-running process that manages the code evolution / optimization.

Our modal function controls for security risks. We are downloading and reading code from clients, and a coding agent is writing code to the client repo. This Function should be scoped to the client data only and not to other clients - i.e., we take away CodeEvolver database access. Instead, it is restricted to pre-defined CodeEvolver mongodb endpoints. That is, it cannot access other clients data. (Uses JWT tokens).

#### ClientSandbox class is a base class for optimizer sandboxes. It provides:
- `start()` — Create Modal sandbox, clone repo, install client's `requirements.txt`
- `stop()` — Terminate sandbox
- `reinstall_deps()` — Re-run pip install if requirements change
- `exec_bash(command)` — Execute arbitrary bash commands
- `exec_prebuilt(command)` — Abstract method for subclasses to implement prebuilt script executions for each optimizer

Subclasses (e.g., `GEPASandbox`) implements `exec_prebuilt()` to run domain-specific scripts inside the sandbox. This enables process-level isolation — the orchestrator (Modal Function) can run without client dependencies installed.

#### Optimizer Architecture
The GEPA orchestrator (Modal Function) Is an isolated process that runs without `dspy` installed. Client code (DSPy programs, metrics) runs in a separate `GEPASandbox`. Communication happens via JSON IPC through prebuilt scripts.

```
Modal Function (gepa_image — no dspy)         Modal Sandbox (client's env)
┌─────────────────────────────────────┐       ┌──────────────────────────────────┐
│ run_optimization()                  │       │ /workspace/ (cloned from GitHub) │
│                                     │       │ pip install -r requirements.txt  │
│ GEPA loop:                          │       │                                  │
│   adapter.build_seed_candidate()    │       │ sandbox/mounted/master_script.py │
│     → sandbox.exec_prebuilt()     ──┼──────>│   dispatches to dspy/*.py        │
│     ← JSON result                 <─┼───────│                                  │
│   adapter.evaluate(batch, cand)     │       │ EVAL_RESULT:{json} on stdout     │
│   adapter.make_reflective_dataset() │       │                                  │
│   GEPA reflection (litellm)         │       │                                  │
└─────────────────────────────────────┘       └──────────────────────────────────┘
```

This avoids dependency conflicts between GEPA's orchestration deps and the client's dspy version.

**Prebuilt Scripts** (`src/sandbox/mounted/`, `src/ai_frameworks/mounted/`):
Scripts copied into the sandbox image and executed via `sandbox.exec()`. They have full access to the client's Python environment. The `master_script.py` dispatcher routes commands to appropriate handlers.

For GEPA-specific sandbox details, see `specs/gepa_plan.md`.

### Coding Agent Details
In the GEPA context, the coding agent is most powerful if it can run code in the client's environment. 

However, we have to control for the following factors:
- **Client environment:** Client code must run in a dedicated environment. --> create venv
- **Anthropic API key:** needs to be protected from malicious actors posing as clients. --> Use a Modal Proxy Function with rate limiting (future case)
- **Python version:** Claude agents SDK requires 3.10+. Clients with earlier version of Python will have to wait extra time for our sandbox to download client python into the VENV 
- **Image startup time:** This is a bit slower, but sand boxes are long running across the whole optimization. 

#### Coding Agent Architecture

```
  Modal Function                              Client Sandbox                                                                                                                                 
  ┌──────────────────┐                     ┌─────────────────────────────┐ 
  │ GEPA Optimizer   │───exec_prebuilt()───│ venv                        │
  │ (no agent)       │                     │ - client deps (Incl DSPy)
  └──────────────────┘                     │ 
                                           │ sys
                                           │  Node.js + Claude Code CLI
                                           │  Claude Agent SDK
                                           │ (reasoning + execution)     │                                  
                                           └─────────────────────────────┘   
```

**For v1**:
- Omit proxy codeevolver secret deployment. Client brings Anthropic key



#### Coding Agent Discussion
The coding agent's Read, grep, glob, and edit tools don't need python. They just need the file system. 
The bash tools can access the VENV environment by running `python` commands with the VENV python in the PATH.

This architecture is preferable to the alternatives. 
- Rejected alternative A: Run 2 sandboxes and push commits between them. This makes it impossible for an agent to test python / run bash commands in the client's environment. 
- Rejected alternative B: Host the Claude agents SDK in the MODAL function (with git clone repo), and rebuild the SDK tools so they can execute inside the sandbox. This would require constant git commit pushing and is not standard.

### Privacy
- Future: Option for users to run execution environment on their own private cloud
- No unnecessary third-party services. Necessary third parties include modal.

### Technology Stack and Architecture
- **Language**: Python
- **API Framework**: Modal Sandbox App serving FastAPI
- **Database**: MongoDB (flexible, but preferred)
- **Execution Environment**: Modal Sandbox. (must spin up in <10 seconds, support 20+ concurrent environments per user)

## V1 outcomes (current goal):
- Connect a GitHub repository [complete]
- Execute a change request [complete]
- GEPA optimizer runs in CodeEvolver [WIP]
- Complete v1 of security: (see for v1 below)
- API / sandbox deployed to Modal App
- **Security needs for v1:**
  - Keep workers specific to each client
  - Make API requests direct to modal / sandbox app (Omit separate api gateway)
  - No egress proxy (temp)
  - Use env for secrets (temp)


## Flexible, Reliable Sandboxes Work for Every Client
We want the sandbox environment to work with every code base. This is difficult to achieve, but for v1, We can start with a evaluation testing script that quickly let the user know whether there is a failure.

The sandbox validation is implemented in `src/sandbox/verify_environment.py` and called via `GEPASandbox.validate_environment()`:

After seed candidate is built, before GEPA optimization loop starts, we run evaluation on the first (up to) 15 rows of training data. It captures execution traces for detailed error diagnostics

Fails if system errors occur on >10% of rows OR if accuracy is 0%. Returns comprehensive error information, including:
  - Row indices that failed
  - Inputs, outputs, and scores for failed rows
  - Execution trace information (trace length, failed steps)
  - Failed step details with signature and error messages
  - First 5 error examples (to avoid overwhelming output)

## Security Architecture
- **Client-specific isolation (v2):** Execution of code will be isolated in v2. Each client should be in a separate container (e.g., client could have malicious code to steal other clients' data or secrets)
- **Network Egress Control and whitelists:** Limit urls to allowed domains and ips set by our best practices and by the user (e.g., api.firecrawl.dev)
- **Secrets management (v2)**: use env file for v1
- **Monitoring and detection:** omit for v1
- Ensure that sandboxes have limited database access
- Ensure client environments cannot access the github app private key. However, they can access the decoded installation token, which is scoped for one user/repository, according to Claude


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
