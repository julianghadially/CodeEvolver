---
name: prompt-engineering
description: Analyze and improves prompts using research-backed frameworks. Use when you need to improve, rewrite, structure, or engineer a prompt. Recommends the right framework based on intent (create, transform, reason, critique, recover, clarify, agentic), asks targeted questions, and delivers a structured, high-quality result.
compatibility: Requires no external dependencies. Works with any Agent Skills compatible tool.
metadata:
  author: adapted from ckelsoe
  reference: https://github.com/ckelsoe/prompt-architect
---

# Prompt Architect

You are an expert in prompt engineering and AI system design. Help improve prompts into well-structured, effective prompts through analysis of past runs, your own reasoning, and framework application as needed.

If prompt changes are needed, your task is to apply the practical methods in this document and test them as part of your broader experimentation. If needed, there are multiple frameworks provided in the skill, which are optional and for reference only. Remember to balance the number of experiments you run on code changes with the number you run on prompt changes.

Sometimes the best prompts are simple.

## Core Prompt Engineering Process

### 1. Initial Assessment

How is the system performing today?
- **Eval and Feedback**: What is the score and what feedback is being provided by the evaluation metric?
- **Traces**: How is data processed through the system? Which module is responsible for the lower than expected score? Is this a prompt issue?

**Is this a prompt issue** Remember we are making changes over multiple rounds of experimentation and in the context of editing code (Context pipelines, retrieval, creation of new modules). Make sure to balance your experiments appropriately between code changes and prompt changes (And sometimes both in tandem). 

If there is a prompt issue, identify the module(s) where a prompt improvement would be useful

Beyond your observations in the traces, signs that you have a bad prompt include issues across the following dimensions:
- **Clarity**: Is the goal clear and unambiguous?
- **Specificity**: Are requirements detailed enough?
- **Context**: Is necessary background provided?
- **Constraints**: Are limitations specified?
- **Output Format**: Is desired format clear?

### 2. Prompt Engineering Approaches

**A. Practical methods**: General prompting principles can be very helpful, like in context learning examples, instruction sets, etc.
**B. Framework based**: Prompting frameworks often have research showcasing value; however, some of this research is based on earlier language models. These frameworks may or may not provide benefit on newer models.

### 3. Identify The Challenge the System or Module is Solving

A prompt engineer must be aware of the underlying challenge behind a system and what that system or module is solving. What makes this problem hard? Is the difficulty due to reasoning? retrieval? etc.

Prompting frameworks generally lend themselves to specific problem types. Experiment with prompt engineering methodologies that work well within the applicable categories.

**A. REASONING SYSTEMS** — Solving a reasoning or calculation problem

| Signal | Framework |
|--------|-----------|
| Numerical/calculation, zero-shot | **Plan-and-Solve (PS+)** |
| Multi-hop with ordered dependencies | **Least-to-Most** |
| Needs first-principles before answering | **Step-Back** |
| Multiple distinct approaches to compare | **Tree of Thought** |
| Verify reasoning didn't overlook conditions | **RCoT** |
| Linear step-by-step reasoning | **Chain of Thought** |

---
**B. RETRIEVAL SYSTEMS** — Systems with retrieval or web search

---

**C. JUDGE / CRITIQUE** — Stress-testing, judging, or verifying output

| Signal | Framework |
|--------|-----------|
| General quality improvement | **Self-Refine** |
| Align to explicit principle/standard | **CAI Critique-Revise** |
| Find the strongest opposing argument | **Devil's Advocate** |
| Identify failure modes before they happen | **Pre-Mortem** |
| Verify reasoning didn't miss conditions | **RCoT** |

*Self-Refine = any quality. CAI = principle compliance. Devil's Advocate = opposing arguments. Pre-Mortem = failure analysis. RCoT = condition verification.*


---
### 4. Practical Approaches Quick Reference
1. In-context learning examples
2. Instructions / step by step
3. Chain of thought reasoning
4. Decompose the problem or task
5. Specify decision boundaries and required details.
6. Simplification
7. Address failure
8. Be concise. Context rot is real
9. Clear output format
10. Experimentation: Different models perform differently on different prompts methods. Experiment with what works!


### 5. Framework Quick Reference

One-line per framework (load `references/frameworks/` for full detail):

**Simple:** APE | RTF | CTF
**Medium:** RACE | CARE | BAB | BROKE | CRISPE
**Comprehensive:** CO-STAR | RISEN | TIDD-EC
**Data:** RISE-IE | RISE-IX
**Reasoning:** Plan-and-Solve | Chain of Thought | Least-to-Most | Step-Back | Tree of Thought | RCoT
**Structure/Iteration:** Skeleton of Thought | Chain of Density
**Critique/Quality:** Self-Refine | CAI Critique-Revise | Devil's Advocate | Pre-Mortem
**Meta/Reverse:** RPEF | Reverse Role Prompting
**Agentic:** ReAct

### 6. Apply Method

1. Determine if the prompt change should be a practical change or should leverage a framework.
2. If a framework is used, load the appropriate template from `assets/templates/`
3. Map trace learnings and learnings from memory into framework components. Fill any missing elements with reasonable defaults but avoid frameworks with too much unknown

### 7. Present Improvements

Present your final prompt as a change request to the coding agent, with the following components:
- **The Language model call or module the prompt is being applied to**: Specify which prompt is being revised.
- **The revised prompt**: Present as a clean, flat-text block inside triple backticks
- **No framework details** You can think through the framework details, but you do not need to provide them in your change request.

## Practical methods
### 1. In-context learning examples
Providing examples in the prompt that cover a broad range of potential common questions is one of the highest leverage prompting techniques.

Include ~3-5 examples of what good and bad look like. 
Consider including up to ten examples of edge cases and failure cases.

Be careful not to overfit. The example should be included not only because it failed in a past run, but because we expect similar patterns in the future.

### 2. Instructions / step by step

Use explicit steps when the task has multiple transformations.
- Use numbered lists for detailed steps or sometimes “First… Then…” is enough
- Keep steps concise
- Ensure broadly applicable steps
- Avoid over-specifying trivial operations

### 3. Chain of thought reasoning

Use step-by-step reasoning when errors come from skipped logic.

- Use prompts like "Take a step back" and "Think step by step"
- Describe the reasoning process after decomposing the problem or task (see below)
- There is a strong interplay between reasoning steps and the modules in a system. Think about this prompt from the perspective of the system as a whole. 
- Prefer internal reasoning so you keep the final output clean
- Skip for simpler tasks

### 4. Decompose the problem or task

Break complex tasks into smaller parts. What is at the core of the problem this system is trying to solve? Break down the problem first and design the prompt around it.

Decompose at two levels
- Either within the prompt
- or across modules
- If a prompt is doing too much, consider splitting it. However, consider that a new submodule introduces many considerations: separate context, often runs every time, added benefit must be worth the added latency.

### 5. Specify decision boundaries and required details

Eliminate ambiguity.
- Define the boundary of decision-making within the task.
- Think about main border line cases.
- Be explicit on definitions
- Be explicit about what to do
- Define what to do when unsure (abstain, list options, etc.)

### 6. Simplification

Simpler prompts often perform better. Language models are already trained on vast amounts of data. If it already has knowledge of a certain task, sometimes a clear definition of the ask is better than conflating the prompt with instructions that carry predefined assumptions that seemed good for a few examples, but did not generalize.

- Remove redundancy and conflicting instructions
- Keep only high-signal guidance
- If performance drops, simplify before adding complexity
- Consider stripping a prompt down to the bare bones request. Stay specific.

### 7. Address failure

Fix recurring failure patterns directly.

- Use evals and traces to identify patterns of failure
- Provide guidance or examples for classes of errors
- Don’t simply patch individual examples

### 8. Be concise. Context rot is real

Long prompts degrade performance.

- Keep prompts tight and readable
- Remove stale or low-value context
- Continuously prune as you iterate

### 9. Clear output format

Make outputs deterministic.

- Specify exact format (e.g., JSON, fields, ordering)
- Disallow extra text
- Enables reliable downstream use and evaluation

### 10. Experimentation

Prompting is empirical.

- Different models respond differently
- Test variations
- Use evals to guide decisions


## Framework References

Detailed framework docs in `references/frameworks/`:
- `co-star.md` - Context, Objective, Style, Tone, Audience, Response
- `risen.md` - Role, Instructions, Steps, End goal, Narrowing
- `rise.md` - **Dual variant support**: RISE-IE (Input-Expectation) & RISE-IX (Instructions-Examples)
- `tidd-ec.md` - Task type, Instructions, Do, Don't, Examples, Context
- `ctf.md` - Context, Task, Format
- `rtf.md` - Role, Task, Format
- `ape.md` - Action, Purpose, Expectation (ultra-minimal)
- `bab.md` - Before, After, Bridge (transformation/rewrite tasks)
- `race.md` - Role, Action, Context, Expectation (medium complexity)
- `crispe.md` - Capacity+Role, Insight, Instructions, Personality, Experiment
- `broke.md` - Background, Role, Objective, Key Results, Evolve
- `care.md` - Context, Ask, Rules, Examples (constraint-driven)
- `tree-of-thought.md` - Branching exploration of multiple solution paths
- `react.md` - Reasoning + Acting (agentic tool-use cycles)
- `skeleton-of-thought.md` - Skeleton-first then expand (parallel generation)
- `step-back.md` - Abstract to principles first, then answer (Google DeepMind)
- `least-to-most.md` - Decompose into ordered subproblems, solve sequentially
- `plan-and-solve.md` - Zero-shot: plan + extract variables + calculate (PS+)
- `chain-of-thought.md` - Step-by-step reasoning techniques
- `chain-of-density.md` - Iterative refinement through compression
- `self-refine.md` - Generate → Feedback → Refine loop (NeurIPS 2023)
- `cai-critique-revise.md` - Principle-based critique + revision (Anthropic)
- `devils-advocate.md` - Strongest opposing argument generation (ACM IUI 2024)
- `pre-mortem.md` - Assume failure, identify causes + warning signs (Gary Klein)
- `rcot.md` - Reverse Chain-of-Thought: verify by reconstructing the question
- `rpef.md` - Reverse Prompt Engineering: recover prompt from output (EMNLP 2025)
- `reverse-role.md` - AI-Led Interview: AI asks you questions first (FATA)

Load these when applying specific frameworks for detailed component guidance, selection criteria, and examples.

## Templates

Framework templates in `assets/templates/` provide structure:
- `co-star_template.txt` - Full CO-STAR structure
- `risen_template.txt` - Full RISEN structure
- `rise-ie_template.txt` - RISE-IE structure (Input-Expectation for data tasks)
- `rise-ix_template.txt` - RISE-IX structure (Instructions-Examples for creative tasks)
- `tidd-ec_template.txt` - TIDD-EC structure (Task, Instructions, Do, Don't, Examples, Context)
- `ctf_template.txt` - CTF structure (Context-Task-Format for situational prompts)
- `rtf_template.txt` - Full RTF structure
- `ape_template.txt` - APE structure (Action-Purpose-Expectation ultra-minimal)
- `bab_template.txt` - BAB structure (Before-After-Bridge for transformations)
- `race_template.txt` - RACE structure (Role-Action-Context-Expectation)
- `crispe_template.txt` - CRISPE structure (with Experiment/variants)
- `broke_template.txt` - BROKE structure (with Key Results + Evolve)
- `care_template.txt` - CARE structure (with Rules + Examples)
- `tree-of-thought_template.txt` - Tree of Thought branching exploration structure
- `react_template.txt` - ReAct Thought-Action-Observation cycle structure
- `skeleton-of-thought_template.txt` - Skeleton + expand structure
- `step-back_template.txt` - Step-back question + principle application
- `least-to-most_template.txt` - Decompose + sequential solving
- `plan-and-solve_template.txt` - PS+ trigger phrase structure
- `chain-of-thought_template.txt` - Step-by-step reasoning with verification
- `chain-of-density_template.txt` - Iterative compression with stopping criterion
- `self-refine_template.txt` - Generate → Feedback → Refine structure
- `cai-critique-revise_template.txt` - Principle → Critique → Revision structure
- `devils-advocate_template.txt` - Position attack with severity ranking
- `pre-mortem_template.txt` - Failure assumption + cause analysis
- `rcot_template.txt` - 4-step backward verification structure
- `rpef_template.txt` - Output analysis + recovered prompt template
- `reverse-role_template.txt` - Intent + interview trigger structure
- `hybrid_template.txt` - Combined framework approach

## Key Principles

1. **Show Your Work** - Think through analysis, show framework mapping, Why this framework? Why these changes? (provide in your thinking, but not in the change request itself)
2. **Use Traces and Memory** - Take advantage of the past system runs to observe where existing prompt(s) failed.
3. **Avoid Overfitting** - It is okay to use in-context examples, but your principles and guidance should generalize beyond the current training data.
4. - **Avoid removal of domain expertise**: Unless the domain expertise is wrong, many prompts may include domain-specific expertise from a client (e.g., legal guidance). We are allowed to edit, but we should exercise caution not to remove domain expertise without a good reason.

## When NOT to Use Frameworks

Applying prompt frameworks takes time away from making improvements to the code. Skip prompting or prompt frameworks when:

- **The prompt is performing well**: Lack of evidence for errors related to prompts and prompts are well defined with clear goals, full context, defined formats.
- **Purely factual lookups**: "What is the capital of France?" — no framework needed. Prompts should never guide factual answers. Instead, prompts should focus on other components of factual answering systems, such as retrieval or reasoning steps (if applicable).

---

## Guidance on specific AI frameworks
Many code bases will have their prompts defined in some prompt folder of prompt generator function that is provided directly to the AI framework. Here are notes for specific AI frameworks.

### DSPy
In DSPy, the final prompt is a combination of the signature class docstring and the descriptions on any InputField and OutputField. To edit prompts in DSPy, you should edit the signature class docstring directly.
```python
def MySignature(dspy.Signature):
    '''prompt goes here'''
    some_field: str = InputField(desc="This field description goes into the prompt as well")
    some_other_field: str = OutputField(desc="This field description goes into the prompt as well")    
```


## Usage Notes

- Always start by analyzing the traces and the original prompts that produced them
- Not everything needs a framework. Often, the simpler, direct prompts perform very well.
- Identify framework(s) in your reasoning
- Make use of templates only if helpful
- Load framework references only when needed for detailed guidance
- Remember, prompt changes will be made in the context of multiple rounds of experimentation and in the context of editing code (Context pipelines, retrieval, creation of new modules). Make sure to balance your time appropriately between code changes and prompt changes. 