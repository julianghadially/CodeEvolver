# CodeEvolver
Remote service for executing evolutionary code changes with CodeEvolver

This repo is **WORK IN PROGRESS!!!**

## Architecture Components

- **API Framework** - FastAPI (Python)
- **Database** - MongoDB with Motor (async driver)
- **Version Control** - GitPython (for repository cloning and git worktree management)
- **Execution Environment** - Modal (serverless Python runtime, ~1-2s cold start)
- **Auto-coder Agent** - Claude Agents SDK (Anthropic) for code mutations
- **Optimizer (External)** - GEPA (evolutionary algorithm client, gepa-ai/gepa)

### Implementation Status
- **Implemented**: FastAPI, MongoDB/Motor, GitPython, Pydantic
- **Pending**: Modal, Claude Agents SDK, DSPy runtime integration

