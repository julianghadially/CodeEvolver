"""Service for applying mutations and executing DSPy programs."""

import json
from pathlib import Path
from typing import Any


class MutationService:
    """Service for handling prompt and code mutations."""

    @staticmethod
    def load_program_json(worktree_path: Path, program_json_path: str) -> dict[str, Any]:
        """
        Load program.json from the worktree.

        Args:
            worktree_path: Path to the git worktree
            program_json_path: Relative path to program.json

        Returns:
            Parsed program JSON

        Raises:
            FileNotFoundError: If program.json doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        full_path = worktree_path / program_json_path
        if not full_path.exists():
            raise FileNotFoundError(f"Program JSON not found: {full_path}")

        with open(full_path) as f:
            return json.load(f)

    @staticmethod
    def save_program_json(
        worktree_path: Path,
        program_json_path: str,
        program_json: dict[str, Any],
    ) -> None:
        """
        Save program.json to the worktree.

        Args:
            worktree_path: Path to the git worktree
            program_json_path: Relative path to program.json
            program_json: Program data to save
        """
        full_path = worktree_path / program_json_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w") as f:
            json.dump(program_json, f, indent=2)

    @staticmethod
    def apply_prompt_mutation(
        program_json: dict[str, Any],
        candidate: dict[str, str],
    ) -> dict[str, Any]:
        """
        Apply prompt mutation to program.json.

        Updates signature.instructions for the specified components.

        Args:
            program_json: Original program JSON
            candidate: Dict mapping component_name -> new instruction text

        Returns:
            Modified program JSON

        Raises:
            KeyError: If component not found in program
        """
        modified = program_json.copy()

        for component_name, new_instruction in candidate.items():
            if component_name not in modified:
                raise KeyError(f"Component not found in program: {component_name}")

            component = modified[component_name]
            if "signature" not in component:
                raise KeyError(f"Component {component_name} has no signature")

            component["signature"]["instructions"] = new_instruction

        return modified

    @staticmethod
    async def apply_code_mutation(
        worktree_path: Path,
        change_request: str,
        change_location: str | None = None,
    ) -> None:
        """
        Apply code mutation using Claude agent.

        Note: This is a placeholder. Full implementation requires Claude Agents SDK.

        Args:
            worktree_path: Path to the git worktree
            change_request: Natural language description of code change
            change_location: Optional module path hint
        """
        # TODO: Implement with Claude Agents SDK
        # from claude_agent_sdk import query, ClaudeAgentOptions
        #
        # async for message in query(
        #     prompt=change_request,
        #     options=ClaudeAgentOptions(
        #         cwd=str(worktree_path),
        #         allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
        #         permission_mode="acceptEdits"
        #     )
        # ):
        #     pass

        raise NotImplementedError(
            "Code mutations require Claude Agents SDK integration. "
            "Use prompt mutations for now."
        )


class ProgramRunner:
    """Service for loading and executing DSPy programs."""

    @staticmethod
    async def run_program(
        worktree_path: Path,
        program_json_path: str,
        entry_point: str,
        test_examples: list[dict[str, Any]],
        capture_traces: bool = False,
    ) -> tuple[list[dict[str, Any]], list[Any] | None]:
        """
        Run a DSPy program on test examples.

        Note: This is a placeholder. Full implementation requires DSPy runtime.

        Args:
            worktree_path: Path to the git worktree
            program_json_path: Relative path to program.json
            entry_point: DSPy module class (e.g., 'fire.FIREJudge')
            test_examples: List of example dicts to run
            capture_traces: Whether to capture execution traces

        Returns:
            Tuple of (pipeline_outputs, traces or None)

        Raises:
            NotImplementedError: DSPy runtime not yet integrated
        """
        # TODO: Implement DSPy runtime execution
        # This requires:
        # 1. Setting up Python path to include worktree
        # 2. Importing the entry_point module dynamically
        # 3. Loading the program state from program.json
        # 4. Running program.forward() on each example
        # 5. Optionally capturing traces

        # For now, return placeholder outputs
        outputs = []
        for i, example in enumerate(test_examples):
            outputs.append({
                "example_id": i,
                "output": {
                    "_placeholder": True,
                    "_note": "DSPy runtime not yet integrated",
                    "input": example,
                },
            })

        traces = [] if capture_traces else None

        return outputs, traces
