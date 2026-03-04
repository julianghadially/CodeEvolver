# CodeEvolver
CodeEvolver is a coding agent that reflects on the errors in your AI application, and makes changes to the prompts and code to improve your system. 

AI engineering with LLMs is a messy cycle of writing 1,000 word prompts, manually inspecting them, and continously tweaking them every time you have a new customer, new AI model, or new use case.

CodeEvolver replaces 90% of that manual work. Given a dataset, CodeEvolver optimizes the codebase against a reward function. Code changes include the prompts, context pipeline, tools, and AI system architecture. 

## Status: Early Access

**THIS CODEBASE IS NOT READY FOR SELF-SERVICE USE**

We are currently serving researchers and AI-native companies through an early access program. To use this codebase, please request access via codeevolver.ai, and we will contact you.

## Service Options

Optimizing code requires making many mutations to your code base, tracking each with git branches, and evaluating (running) each one on an objective function. 

To do this safely, you need a secure, sandboxed environment to run AI generated code. This environment also needs to run your code, which means it needs access to whatever secrets and databases are required for running your AI system.

For now, you can use our secure, sandboxed environments.  We do NOT recommend running 100s of AI-generated code mutations on your own machine without security protections. 

## Fully-hosted optimization

In our hosted solution, we orchestrate coding agents, git branching, optimization jobs, and most importantly…

### We handle security for you: 

1. **Sandboxed execution environments:**  AI-generated code executes in an isolated environment, protecting your files.  
2. **Network protection:** No data can leave the sandbox, except to white-listed domains - *coming soon*.
3. **Secrets management:** Secrets stored securely and accessed by the sandbox at runtime - *coming soon*.
4. **Zero persistent storage:** program tracking such as optimization improvement trajectories are stored for three months. Access is controlled via jwt tokens. Your AI program is cloned at runtime in a modal sandbox and dies after the worker is finished.
5. **Transparent Third Parties:** Modal for cloud services and sandboxes, Anthropic for Claude Agents SDK, Mongodb Atlas service, and infisical for keys.

### How to get started

**Step 1: Join Early Access Program**
Visit codeevolver.ai to join our early access program, as this repository is not yet ready for self-service.

**Step 2: Prepare your project**   
You will need the following:

1. *Training dataset*, including inputs and any fields used by your objective metric (e.g.,  ground truth label)   
2. *Objective metric*  
3. *AI system* (Currently limited to DSPy). Looking for contributors\! 

**Step 3: Create Job**

1. Send job to /optimize  
2. See results directly on your GitHub, on the branch, codeevolver-best-program (check status with /job\_id/check\_job

## Private Cloud

Please visit codeevolver.ai.


## Gratitude and References
This work is built on GEPA. We would like to extend a special thank you to Lakshya Agrawal and the team behind GEPA for the powerful algorithm that drives CodeEvolver.
``` 
Agrawal, L. A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M. J., Jiang, M., Potts, C., Sen, K., Dimakis, A. G., Stoica, I., Klein, D., Zaharia, M., & Khattab, O. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning [Computer software]
```

This work is also built on DSPy. A special thank you goes out to Omar Khottab and the DSPy community that made self-improving prompts a thing, and showcased broad interest and support in self-improving AI systems and the idea of not having to write prompts!