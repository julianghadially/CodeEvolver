# Service Options
Service method is driven by security for enterprise and convenience / cost / agent environment for developers. Agent environment is tricky because good agents require sandboxing. You cannot safely run dangerously-skip-permissions on your computer (for now).

## Fully-hosted optimization (agent power + parallelized speed)
- Charge \$/M Tokens, public 30% upcharge on selected claude
- Runs as a sandboxed modal app

## Enterprise - Private deployment
- **Keep code and data in the 4 walls**
- Deploy CodeEvolver’s specialized coding agents privately through your private cloud (to be hosted on AWS marketplace, Google Cloud marketplace, etc.)
- Target open source models on your cloud. 
- Charge licensing and support fees, similar to CodeRabbit and Claude Code

## Alternative options

### Self-hosted sandbox
- User sets up own sandbox locally or through third party provider
- Agent can run with full permissions
- Code can execute 
- Hard for code to escape container
- Docker Setup: 
    - Copy paste docker container file, add your project root, and add proxies for specific sites (Claude, AI model provider, services, etc.). Consider gVisor for higher security
- Cloud Setup:
    - Deploy on your own cloud provider, share your secrets with them
- Drawback: have to set up docker with proxies or at that point, may as well host on a cloud service

### Self-hosted, dangerously without sandbox (Strongly NOT recommended)
- Claude Agents SDK sends queries, where Anthropic arms Claude with its own environment to run as a tool. Does not apply to running bash, python commands, or executing the AI workflow itself in the user's environment. The latter requires secrets. 
- Drawback: coding agent is weak, and good results are unlikely. 
- Drawback: Bad security running mutations locally. You are trusting that Claude will never inject bad code into your python project (i.e., succumb to prompt injection).

### Hosted open source model
- Coding agents sdk query goes to our hosted open source model
- Cheaper than claude; charge $/M Tokens
- Drawbacks: More prone to prompt injection (less security training)

## Old options
- Hosted Claude Agent, no secrets, for security (Not needed: Doesn't move the needle) ...Gives agent bashing, pycompile, and execution environment, etc. on your repo. However, no secrets, so can’t execute optimization run. Also Same drawbacks as self-hosted dangerously without sandbox, but slightly more powerful agent.
