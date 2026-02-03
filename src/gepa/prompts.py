"""Prompts for GEPA reflection agent and architecture generation.

Separates prompt construction from adapter logic for better maintainability.
"""

import json
from datetime import datetime


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

Then generate an architecture summary (500-2500 characters) that includes:
1. What this program does (high-level purpose)
2. Key modules and their responsibilities
3. Data flow through the system
4. The metric being optimized

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
        Basic architecture summary string.
    """
    return f"""# Architecture Summary

## Program
- **Entry Point**: `{program_path}`
- **Metric**: `{metric_path}`

## Overview
This is a DSPy program being optimized by CodeEvolver.

*Generated automatically by CodeEvolver at {datetime.now().isoformat()}*
"""


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

    return f"""You are analyzing the performance of an AI system to propose a single targeted change to the AI system code (not the prompts).

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
3. Propose ONE specific, targeted code change that would most improve performance

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

Respond with a specific, actionable change request that a coding agent can execute."""
