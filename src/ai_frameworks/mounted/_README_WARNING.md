# AI Frameworks
The ai_frameworks/mounted folder contains scripts for integrating with different AI frameworks.

## Sandbox Warning
All code from this folder is mounted into the client sandbox, and thus needs to calibrate its package dependencies with the sandbox image. Given the sandbox creates a separate internal venv environment for the client's code vs the agent code, adding requirements is permitted, but should be done carefully.