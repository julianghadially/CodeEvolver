# CodeEvolver Agents
Remote service for executing evolutionary code changes with CodeEvolver

## Architecture Components

- **API Framework** - FastAPI (Python)
- **Database** - MongoDB with Motor (async driver)
- **Version Control** - GitPython (for repository cloning and git worktree management)
- **Execution Environment** - Modal (serverless Python runtime, ~1-2s cold start)
- **Auto-coder Agent** - Claude Agents SDK (Anthropic) for code mutations
- **Program Runtime** - DSPy (Stanford NLP) for executing optimized programs
- **Data Validation** - Pydantic (request/response schemas)
- **External Optimizer** - GEPA (evolutionary algorithm client, gepa-ai/gepa)

### Implementation Status
- **Implemented**: FastAPI, MongoDB/Motor, GitPython, Pydantic
- **Pending**: Modal, Claude Agents SDK, DSPy runtime integration

WIP