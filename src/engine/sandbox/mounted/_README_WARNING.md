# AI Frameworks
The sandbox/mounted folder orchestrates the loading of scripts into the client sandbox, via the master_script.py.

## Sandbox Warning
All code from this folder is mounted into the client sandbox, and thus needs to calibrate its package dependencies with the sandbox image. Given the sandbox creates a separate internal venv environment for the client's code vs the agent code, adding requirements is permitted, but should be done carefully.