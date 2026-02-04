"""Prompts for GEPA reflection agent and architecture generation.

Separates prompt construction from adapter logic for better maintainability.
"""

import json
from datetime import datetime


# DSPy guidelines appended to codeevolver.md for coding agent reference
DSPY_GUIDELINES = """
## DSPy Patterns and Guidelines

DSPy is an AI framework for defining a compound AI system across multiple modules. Instead of writing prompts, we define signatures. Signatures define the inputs and outputs to a module in an AI system, along with the purpose of the module in the docstring. DSPy leverages a prompt optimizer to convert the signature into an optimized prompt, which is stored as a JSON, and is loaded when compiling the program.

**DSPy docs**: https://dspy.ai/api/

Stick to DSPy for any AI modules you create, unless the client codebase does otherwise.

Defining signatures as classes is recommended. For example:

```python
class WebQueryGenerator(dspy.Signature):
    \"\"\"Generate a query for searching the web.\"\"\"
    question: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query for searching the web")
```

Next, modules are used as nodes in the project, either as a single line:

```python
predict = dspy.Predict(WebQueryGenerator)
```

Or as a class:

```python
class WebQueryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_generator = dspy.Predict(WebQueryGenerator)

    def forward(self, question: str):
        return self.query_generator(question=question)
```

A module can represent a single module, or the module can act as a pipeline that calls a sequence of sub-modules inside `def forward`.

Common prebuilt modules include:
- `dspy.Predict`: for simple language model calls
- `dspy.ChainOfThought`: for reasoning first, followed by a response
- `dspy.ReAct`: for tool calling
- `dspy.ProgramOfThought`: for getting the LM to output code, whose execution results will dictate the response
"""


def build_architecture_prompt(
    program_path: str,
    metric_path: str,
) -> str:
    """Build prompt for architecture summarization.

    Used during build_seed_candidate to generate an initial architecture
    summary that gets saved to codeevolver.md.

    Args:
        program_path: Dotted import path to program (e.g., "src.factchecker.FactCheckerPipeline").
        metric_path: Dotted import path to metric function.

    Returns:
        Formatted prompt string for the reflection agent.
    """
    # Convert dotted path to file path
    file_path = program_path.replace(".", "/") + ".py"

    return f"""You are analyzing a codebase to generate an architecture summary.

## Program Being Optimized
- **Entry Point**: `{program_path}`
- **Metric**: `{metric_path}`

## Your Task
1. Use the Read tool to examine the program entry point file: `{file_path}`
2. Use Glob to find related Python files in the same directory
3. If there's a README.md, read it for additional context

Then generate an architecture summary of the program that starts with these exact lines:
```
PARENT_MODULE_PATH: {program_path}
METRIC_MODULE_PATH: {metric_path}
```

Followed by 500-2500 characters that includes:
1. What this program does (high-level purpose)
2. Key modules and their responsibilities
3. Data flow through the system
4. The metric being optimized

The PARENT_MODULE_PATH is the top-most module that will be evaluated. If the program has a pipeline wrapper, use that path instead.

Provide the summary as a single markdown-style output."""


def build_architecture_fallback(
    program_path: str,
    metric_path: str,
) -> str:
    """Build fallback architecture summary when reflection agent fails.

    Args:
        program_path: Dotted import path to program.
        metric_path: Dotted import path to metric function.

    Returns:
        Basic architecture summary string with DSPy guidelines appended.
    """
    return f"""PARENT_MODULE_PATH: {program_path}
METRIC_MODULE_PATH: {metric_path}

# Architecture Summary

## Program
- **Entry Point**: `{program_path}`
- **Metric**: `{metric_path}`

## Overview
This is a DSPy program being optimized by CodeEvolver.

*Generated automatically by CodeEvolver at {datetime.now().isoformat()}*
{DSPY_GUIDELINES}"""


def append_dspy_guidelines(architecture: str) -> str:
    """Append DSPy guidelines to an architecture summary.

    Used after the reflection agent generates the architecture content.

    Args:
        architecture: The architecture summary from reflection agent.

    Returns:
        Architecture with DSPy guidelines appended.
    """
    return architecture.rstrip() + "\n" + DSPY_GUIDELINES


def build_code_reflection_prompt(
    feedback: list[dict],
    parent_branch: str,
    additional_instructions: str | None = None,
    attempted_changes: list[str] | None = None,
) -> str:
    """Build prompt for the reflective LM to analyze feedback and propose a change.

    Used during code mutation to analyze evaluation results and propose
    a targeted code change.

    Args:
        feedback: List of feedback items from evaluation.
        parent_branch: The branch this mutation will spawn from.
        additional_instructions: Optional client-provided guidance for optimization.
        attempted_changes: List of changes already tried from this branch (to avoid repeating).

    Returns:
        Formatted prompt string for the reflection agent.
    """
    # Limit to 10 examples to avoid token limits
    feedback_str = json.dumps(feedback[:10], indent=2)

    # Build additional instructions section if provided
    additional_section = ""
    if additional_instructions:
        additional_section = f"""
## Additional Instructions from Client
{additional_instructions}
"""

    # Build attempted changes section - only show changes attempted FROM this branch
    # This allows parallel branches to independently discover the same mutation
    attempted_section = ""
    if attempted_changes:
        attempted_list = "\n".join(f"- {change}" for change in attempted_changes[-10:])
        attempted_section = f"""
## Previously Attempted Changes (DO NOT REPEAT)
The following changes have already been tried. Propose something different:
{attempted_list}
"""

    return f"""You are analyzing the performance of an AI system to propose a single change to the AI system code (not the prompts).

Unless otherwise specified in the additional instructions, the changes should be related to:
- Context pipeline
- Memory
- Language model modules
- Module inputs and outputs
- AI workflow architecture (e.g., How each module connects to each other)
    - sub-modules
    - dynamic prompts

Change should NOT be related to any of the following:
- Prompts
- DSPy docstrings
- Logging
- Client database structure
- Code that does not pertain to the AI workflow
- Any Constraints provided by the client in the additional instructions section

## Your Task
1. First, read codeevolver.md to understand the system architecture
2. Analyze the evaluation feedback below
3. Propose ONE specific code change that would most improve performance

{additional_section}{attempted_section}

## Evaluation Feedback
Each item shows an example input, the system output, and the score (1.0 = perfect).
Items may also include exceptions if the code failed.

{feedback_str}

## General Guidelines
- If there are code failures (exceptions), prioritize fixing those
- If scores are consistently low for certain input patterns, propose changes to handle those cases
- Be specific: mention file paths and what to change
- Do NOT propose changes that have already been attempted (see above)
- Change can involve significant change to the AI system, as long as it is one concept or feature.
- Follow DSPy patterns and guidelines, below.

Respond with a specific, actionable change request that a coding agent can execute.

## Additional Guidelines

### DSPy Guidelines
- DSPy assumes a "compound AI system"
- A compound AI system can involve one single module, or multiple modules that are wrapped by a parent module for the whole workflow
- The top-most parent module is the "entry point" to the AI system and dictates the flow all the way to the final output
"""
