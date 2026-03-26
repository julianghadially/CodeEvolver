# CodeEvolver
CodeEvolver is a system of self-improving code for AI applications. Our agent reflects on the errors in your AI application, and makes changes to the prompts and code to improve your system. 

AI engineering with LLMs is a messy cycle of writing 1,000 word prompts, manually inspecting them, and continously tweaking them every time you have a new customer, new AI model, or new use case.

CodeEvolver replaces 90% of that manual work. Given a dataset, CodeEvolver optimizes the codebase against a reward function. Code changes include the prompts, context pipeline, tools, and AI system architecture. 

## Website
Please visit codeevolver.ai.

### We're in Early Access Mode

**THIS CODEBASE IS NOT READY FOR SELF-SERVICE USE**

We are currently serving researchers and AI-native companies through an early access program. To use this codebase, please request access via codeevolver.ai, and we will contact you.

## Stealth mode

We are working on some big changes to CodeEvolver. Those changes are being made privately until the time comes to do a dedicated open-source launch. 

**This repo serves to publically demonstrate the first iteration of our product and is many versions behind!**

## Gratitude and References
This work is built on GEPA. We would like to extend a special thank you to Lakshya Agrawal and the team behind GEPA for the powerful algorithm that drives CodeEvolver.
``` 
Agrawal, L. A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M. J., Jiang, M., Potts, C., Sen, K., Dimakis, A. G., Stoica, I., Klein, D., Zaharia, M., & Khattab, O. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning [Computer software]
```

This work is also built on DSPy. A special thank you goes out to Omar Khottab and the DSPy community that made self-improving prompts a thing, and showcased broad interest and support in self-improving AI systems and the idea of not having to write prompts!