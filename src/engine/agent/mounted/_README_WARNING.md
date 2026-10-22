# Agent Folder
The agent/mounted folder contains definitions for coding and reflection agents. They are launched as subprocesses via `src/engine/services/subprocess_service.py`; the JSON config they consume is built by `src/engine/agent/agent_config.py`.

## Sandbox Warning
All code from this folder is mounted into the client sandbox, and thus needs to calibrate its package dependencies with the sandbox image. Given the sandbox creates a separate internal venv environment for the client's code vs the agent code, adding requirements is permitted, but should be done carefully.