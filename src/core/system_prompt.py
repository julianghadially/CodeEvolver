"""System prompts for the CodeEvolver coding agent.

The coding agent receives change requests and executes them autonomously
using Claude's native tools (Bash, Read, Edit, Glob, Grep).
"""


def get_code_mutation_prompt(
    change_request: str,
    change_location: str | None = None,
    workspace_path: str = "/workspace",
) -> str:
    """
    Generate a system prompt for code mutation tasks.

    Args:
        change_request: Natural language description of the code change
        change_location: Optional hint about which module/file to focus on
        workspace_path: Path to the workspace directory

    Returns:
        Formatted prompt for the Claude agent
    """
    location_hint = ""
    if change_location:
        location_hint = f"\n\nFocus on: {change_location}"

    return f"""You are a coding agent executing a code modification request.

WORKSPACE: {workspace_path}

CHANGE REQUEST:
{change_request}{location_hint}

INSTRUCTIONS:
1. First, explore the codebase to understand its structure
2. Identify the files that need to be modified
3. Make the requested changes carefully
4. Ensure the changes maintain code quality and don't break existing functionality
5. After making changes, verify they are syntactically correct

CONSTRAINTS:
- Only modify files directly related to the change request and AI workflow.
- Do not modify databases, data files, or any functionality not related to the AI workflow.
- Make only one change at a time. 
- Preserve existing code style and conventions
- Do not modify configuration files unless explicitly requested

When complete, the changes should be ready to commit."""


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
