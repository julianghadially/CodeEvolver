"""System prompts for the CodeEvolver coding agent.

The coding agent receives change requests and executes them autonomously
using Claude's native tools (Bash, Read, Edit, Glob, Grep).
"""


def get_code_mutation_prompt(
    change_request: str,
    change_location: str | None = None,
    workspace_path: str = "/workspace",
    program_path: str | None = None,
) -> str:
    """
    Generate a system prompt for code mutation tasks.

    Args:
        change_request: Natural language description of the code change
        change_location: Optional hint about which module/file to focus on
        workspace_path: Path to the workspace directory
        program_path: Dotted import path to the parent module class

    Returns:
        Formatted prompt for the Claude agent
    """
    location_hint = ""
    if change_location:
        location_hint = f"\n\nFocus on: {change_location}"

    parent_module_constraint = ""
    if program_path:
        parent_module_constraint = f"""

CRITICAL CONSTRAINT:
All changes must be within the top-most parent module class: `{program_path}`.
You may create new sub-modules and Signature classes, but they must be used by this parent module's forward() method.
Do NOT create new pipeline/wrapper classes. The evaluation system calls `{program_path}` directly."""

    return f"""You are an autonomous coding agent executing a code modification request.

WORKSPACE: {workspace_path}

CHANGE REQUEST:
{change_request}{location_hint}{parent_module_constraint}

AUTONOMOUS EXECUTION CONTEXT:
You are running in an automated pipeline with a user proxy that auto-approves your plans and answers questions. This means:
- If you use plan mode (EnterPlanMode), your plan will be automatically approved
- If you ask questions (AskUserQuestion), the first option will be automatically selected
- You can use plan mode for complex multi-file changes - it will proceed without delay
- Work as if you have full autonomy - approvals are instant

INSTRUCTIONS:
1. Read codeevolver.md to understand the system architecture
2. Explore the codebase to understand its structure
3. For complex changes affecting multiple files, you MAY use plan mode - it will be auto-approved
4. Make the requested changes based on best engineering practices
5. Ensure the changes maintain code quality and don't break existing functionality
6. If uncertain about implementation details, choose the most conservative/safe approach
7. After making changes, verify they are syntactically correct
8. Update codeevolver.md if your changes affect the architecture

CONSTRAINTS:
- Only modify files directly related to the change request and AI workflow
- Do NOT modify databases, data files, or unrelated functionality
- Do NOT modify the metric
- Make targeted, minimal changes - avoid unnecessary refactoring
- Preserve existing code style and conventions
- Do NOT modify configuration files unless explicitly requested

Codeevolver.md Details:
- Ensure the parent module is correct in the format PARENT_MODULE_PATH:"", with dot notation for the path, from the project root (see metric path for comparison)
- 500-2500 characters describing what this program does, key modules, data flow, services

The changes will be evaluated automatically. Work efficiently."""


def get_prompt_mutation_prompt(
    component_name: str,
    current_instruction: str,
    new_instruction: str,
    program_json_path: str,
) -> str:
    """
    Generate a prompt for updating a DSPy component's instruction.

    This is typically not needed since prompt mutations are applied directly
    to program.json, but can be used for more complex prompt refactoring.

    Args:
        component_name: Name of the component to update
        current_instruction: Current instruction text
        new_instruction: New instruction text to apply
        program_json_path: Path to the program.json file

    Returns:
        Formatted prompt for the Claude agent
    """
    return f"""You are updating a DSPy program's instruction.

FILE: {program_json_path}
COMPONENT: {component_name}

CURRENT INSTRUCTION:
{current_instruction}

NEW INSTRUCTION:
{new_instruction}

Update the program.json file to replace the instruction for the specified component.
The instruction is located at: {component_name} -> signature -> instructions"""


def get_program_execution_prompt(
    entry_point: str,
    program_json_path: str,
    test_examples: list[dict],
) -> str:
    """
    Generate a prompt for running a DSPy program.

    Args:
        entry_point: DSPy module class to instantiate (e.g., 'fire.FIREJudge')
        program_json_path: Path to program.json
        test_examples: Test examples to run

    Returns:
        Formatted prompt for program execution
    """
    examples_str = "\n".join(f"  - {ex}" for ex in test_examples[:3])
    if len(test_examples) > 3:
        examples_str += f"\n  ... and {len(test_examples) - 3} more"

    return f"""Run the DSPy program and capture outputs.

ENTRY POINT: {entry_point}
PROGRAM JSON: {program_json_path}
TEST EXAMPLES ({len(test_examples)} total):
{examples_str}

Steps:
1. Load the program from program.json
2. Instantiate the entry point module
3. Run forward() on each test example
4. Return the outputs"""
