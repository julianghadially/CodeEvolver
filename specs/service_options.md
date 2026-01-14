# Service Options
Service method is driven by security for enterprise and convenience / cost / agent environment for developers. Agent environment is tricky because good agents require sandboxing. You cannot safely run dangerously-skip-permissions on your computer (for now).


## Self-hosted, docker container or cloud
- Agent can run with full permissions
- Code can execute 
- Hard for code to escape container
- Docker Setup: 
    - Copy paste docker container file, add your project root, and add proxies for specific sites (Claude, AI model provider, services, etc.). Consider gVisor for higher security
- Cloud Setup:
    - Deploy on your own cloud provider, share your secrets with them
- Drawback: have to set up docker with proxies or at that point, may as well host on a cloud service

## Fully-hosted optimization (agent power + parallelized speed)
- \$/M Tokens, public 30% upcharge on selected claude
- Drawback: must manage secrets with a new vendor (CodeEvolver) for any services in AI workflow pipeline


## Enterprise - keep code in the 4 walls
- CodeEvolver’s specialized coding agent, with local open source model deployment. 
- Licensing and support fees
- Drawback: expensive and sales process

# Other Service Ideas

## Self-hosted, "raw-dog" (Strongly NOT recommended)
- Claude Agents SDK sends queries, where Anthropic arms Claude with its own environment to run as a tool. Does not apply to running bash, python commands, or executing the AI workflow itself in the user's environment. The latter requires secrets. 
- Drawback: coding agent is weak, and good results are unlikely. 
- Drawback: Bad security running mutations locally. You are trusting that Claude will never inject bad code into your python project (i.e., succumb to prompt injection).

## Hosted Claude Agent, no secrets, for security (Not needed: Doesn't move the needle)
- \$/M Tokens + 30\% upcharge on selected claude models 
- Gives agent bashing, pycompile, and execution environment, etc. on your repo. However, no secrets, so can’t execute optimization run
- Drawbacks: Same drawbacks as self-hosted "raw-dog", but slightly more powerful agent.

## Hosted open source model
- Local agents sdk; query goes to the hosted open source model
- Cheaper than claude; charge $/M Tokens
- Drawbacks: More prone to prompt injection (less security training)
