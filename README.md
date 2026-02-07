# CodeEvolver
CodeEvolver offers autonomous coding agents for high reliability AI systems. 

AI engineering with LLMs is a messy cycle of writing 1,000 word prompts, manually inspecting them, and continously tweaking them every time you have a new customer, new AI model, or new use case.

CodeEvolver replaces 90% of that manual work. Given a dataset, CodeEvolver optimizes the codebase against a reward function. Code changes include the prompts, context pipeline, tools, and AI system architecture. 

This repo is opening for beta testing very soon. Contact us on Linkedin at /in/julianghadially.

## Pick a service

Optimizing code requires making many mutations to your code base, tracking each with git branches, and evaluating (running) each one on an objective function. 

To do this safely, you need a secure, sandboxed environment to run AI generated code. This environment also needs to run your code, which means it needs access to whatever secrets and databases are required for running your AI system.

You can use our secure, sandboxed environments, or host on your own private cloud \- coming soon with our enterprise plans.  We do NOT recommend running 100s of AI-generated code mutations on your own machine without the security protections in this repo.

## Fully-hosted optimization

In our hosted solution, we orchestrate coding agents, git branching, optimization jobs, and most importantly…

### We handle security for you: 

1. **Sandboxed execution environments:**  AI-generated code executes in an isolated environment, protecting your files.  
2. **Network protection:** No data can leave the sandbox, except to white-listed domains.  
3. **Secrets management via infisical:** Secrets stored securely and accessed by the sandbox at runtime.  
4. **Zero persistent storage:** program tracking is stored for three months with jwt token for access control. Code is cloned at runtime and dies after the worker is finished.
5. **Transparent Third Parties:** Modal for cloud services and sandboxes, Anthropic for Claude Agents SDK, Mongodb Atlas service, and infisical for keys

### How to get started

**Step 1: Create account**

1. Connect coding agent to your Github repository  
2. Connect to a secrets manager (powered by infisical or link to your own)

**Step 2: Prepare your project**   
You will need the following (See guide):

1. *Training dataset*, including inputs and any fields used by your objective metric (e.g.,  ground truth label)   
2. *Objective metric*  
3. *AI system* (Currently limited to DSPy). Looking for contributors\! 

**Step 3: Create Job**

1. Send job to /optimize  
2. See results directly on your GitHub, on the branch, codeevolver-best-program (check status with /job\_id/check\_job

## Private Cloud

Please contact us on linkedin: /in/julianghadially


## Gratitude and References
This work is built on GEPA. We would like to extend a special thank you to Lakshya Agrawal and the team behind GEPA for the powerful algorithm that drives CodeEvolver.
``` 
Agrawal, L. A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M. J., Jiang, M., Potts, C., Sen, K., Dimakis, A. G., Stoica, I., Klein, D., Zaharia, M., & Khattab, O. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning [Computer software]
```

This work is also built on DSPy. A special thank you goes out to Omar Khottab and the DSPy community that made self-improving prompts a thing, and showcased broad interest and support in self-improving AI systems and the idea of not having to write prompts!