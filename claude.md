# Claude.md for CodeEvolver Agents

CodeEvolver Agents Is a remote service for executing evolutionary code changes Based on the CodeEvolver algorithm(s). The first CodeEvolver algorithm is a modified GEPA algorithm that makes code change requests for the agent service to execute. The agent service receives change requests, and execute them remotely.

Always refer to specs/*.md for detailed requirements, and update them after a session with any technology specifics or decisions made. 

## Key Resources
- **Auto coders**: 
    - **Claude Code:** If 
    - **Other**: Auto-Claude is missing a reproducible license. Gas Town is written in Go. anomalyco/opencode is written in typescript
- **DSPy docs**: https://dspy.ai/

## DSPy Assumption
In version 1, all programs will be provided as DSPy programs

## Additional rules
- Never change the language model unless explicitly asked by user. GPT-5 and GPT-5-mini do in fact exist - This project began January 2026, and your knowledge training cutoff may be up to mid-2025!