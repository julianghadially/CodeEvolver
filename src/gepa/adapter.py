"""CodeEvolver DSPy adapter for GEPA optimization.

Implements the GEPAAdapter protocol via structural typing (no inheritance).
All DSPy operations are delegated to a GEPASandbox via JSON IPC,
providing process-level isolation between the GEPA orchestrator and client code.

Copyright notice for GEPA-derived patterns:
Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
"""

import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Callable

import litellm

from gepa.core.adapter import EvaluationBatch

logger = logging.getLogger(__name__)

# Reserved key for code component (underscore prefix distinguishes from DSPy predictor names)
# git_branch is stored INSIDE _code to prevent GEPA from treating it as a mutable component
CODE_COMPONENT_KEY = "_code"


class CodeEvolverDSPyAdapter:
    """GEPAAdapter implementation for DSPy programs in CodeEvolver.

    All evaluation, seed candidate building, and reflective dataset
    construction are delegated to the eval sandbox via JSON RPC.
    No dspy is imported in this module.

    Args:
        sandbox_manager: GEPASandbox instance (already started).
        program: DSPy module class path (e.g., "src.factchecker.FactCheckerPipeline").
        metric: Dotted import path to metric function (e.g., "eval.metric").
        saved_program_json_path: Relative path to program.json within the repo (optional).
        failure_score: Score to assign on evaluation failure.
        num_threads: Number of threads for parallel evaluation.
        input_keys: Field names to mark as inputs on dspy.Example.
        initial_branch: Initial git branch name for seed candidate.
    """

    def __init__(
        self,
        sandbox_manager: Any,
        program: str,
        metric: str,
        saved_program_json_path: str | None = None,
        failure_score: float = 0.0,
        num_threads: int = 1,
        input_keys: list[str] | None = None,
        initial_branch: str = "main",
        program_lm: str = "openai/gpt-5-mini",
        reflection_lm: str = "openai/gpt-5-mini",
        reflection_prompt_template: str | None = None,
        additional_instructions: str | None = None,
    ):
        self._sandbox = sandbox_manager
        self.program_path = program
        self.metric_path = metric
        self.saved_program_json_path = saved_program_json_path
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.input_keys = input_keys or []
        self.initial_branch = initial_branch
        self.program_lm = program_lm
        self.reflection_lm = reflection_lm
        self.reflection_prompt_template = reflection_prompt_template
        self.additional_instructions = additional_instructions
        # Track attempted changes per parent branch to avoid repeating the same ideas.
        # Key: parent branch name, Value: list of changes attempted FROM that branch.
        # This allows parallel branches to independently discover the same mutation.
        self._attempted_changes_by_branch: dict[str, list[str]] = {}
        # Run timestamp for consistent branch naming (YYYYMMDDHHMinMin format)
        self._run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # The codeevolver main branch for this run (set during build_seed_candidate)
        self._ce_main_branch: str | None = None
        # Cached LM callable for prompt mutations
        self._reflection_lm_callable: Callable[[str], str] | None = None

    def _get_reflection_lm_callable(self) -> Callable[[str], str]:
        """Get a callable function that invokes the reflection LM.

        GEPA's InstructionProposalSignature.run() expects an LM callable,
        not a model name string. This method wraps the model name into
        a callable using LiteLLM.

        Returns:
            A function that takes a prompt string and returns a response string.
        """
        if self._reflection_lm_callable is not None:
            return self._reflection_lm_callable

        model_name = self.reflection_lm

        def lm_fn(prompt: str) -> str:
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        self._reflection_lm_callable = lm_fn
        return lm_fn

    def _get_git_branch_from_candidate(self, candidate: dict[str, str]) -> str:
        """Extract git_branch from the _code component.

        Args:
            candidate: Candidate dict with _code component.

        Returns:
            The git branch name, or the run main branch as fallback.
        """
        code_data = json.loads(candidate.get(CODE_COMPONENT_KEY, "{}"))
        return code_data.get("git_branch", self._ce_main_branch or "main")

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose new texts for specified components.

        Routes _code component to code mutation and others to prompt mutation.
        When _code is mutated, git_branch is updated inside the _code JSON.

        Args:
            candidate: Current candidate dict with _code and predictor texts.
            reflective_dataset: Feedback data per component from evaluation.
            components_to_update: List of component names to update.

        Returns:
            Dict mapping component names to new text values.
        """
        new_texts = {}

        for component_name in components_to_update:
            if component_name == CODE_COMPONENT_KEY:
                # Code mutation returns _code with git_branch inside
                new_texts[CODE_COMPONENT_KEY] = self._propose_code_mutation(
                    candidate, reflective_dataset
                )
            else:
                new_texts[component_name] = self._propose_prompt_mutation(
                    component_name, candidate, reflective_dataset
                )

        return new_texts

    def build_seed_candidate(self) -> dict[str, str]:
        """Extract initial instructions from the DSPy program via sandbox.

        Creates the run's main branch (codeevolver-{timestamp}-main), generates
        an architecture summary using the reflection LM, saves it to codeevolver.md,
        and commits it to the branch.

        Returns:
            Dict with 'git_branch', '_code' key and predictor instruction texts.
        """
        print(f"[ADAPTER] build_seed_candidate() called: program={self.program_path}", flush=True)
        result = self._sandbox.exec_prebuilt({
            "command": "build_seed_candidate",
            "program": self.program_path,
            "saved_program_json_path": self.saved_program_json_path,
        })

        print(f"[ADAPTER] build_seed_candidate result: success={result.get('success')}", flush=True)
        if not result.get("success", False):
            print(f"[ADAPTER] build_seed_candidate failed: {result.get('error', 'unknown')}", flush=True)
            if result.get("traceback"):
                print(f"[ADAPTER] Traceback:\n{result.get('traceback')}", flush=True)
            raise RuntimeError(
                f"build_seed_candidate failed: {result.get('error', 'unknown')}"
            )

        # Create the run's main branch from initial_branch (usually "main")
        self._ce_main_branch = f"codeevolver-{self._run_timestamp}-main"
        self._create_ce_main_branch()

        # Set commit message on sandbox for all future mutations
        self._sandbox.commit_message = f"codeevolver mutation. Date: {self._run_timestamp}"

        # Generate architecture summary and save to codeevolver.md
        # Architecture lives in the file (not _code JSON) so it stays in sync with code
        architecture = self._generate_architecture_summary()
        self._save_architecture_to_file(architecture)

        candidate = result["candidate"]
        # git_branch is stored INSIDE _code to prevent GEPA from treating it as a mutable component
        candidate[CODE_COMPONENT_KEY] = self._build_initial_code_component(self._ce_main_branch)
        print(f"[ADAPTER] Seed candidate has {len(candidate)} keys", flush=True)
        return candidate

    def _create_ce_main_branch(self) -> None:
        """Create the run's main branch from the initial branch.

        Creates branch named codeevolver-{timestamp}-main from self.initial_branch.

        Raises:
            RuntimeError: If branch creation fails.
        """
        # Checkout initial branch (usually "main")
        checkout_result = self._sandbox.exec_bash(f"git checkout {self.initial_branch}")
        if checkout_result.get("returncode") != 0:
            raise RuntimeError(
                f"Failed to checkout initial branch {self.initial_branch}: "
                f"{checkout_result.get('stderr')}"
            )

        # Create and checkout run's main branch
        create_result = self._sandbox.exec_bash(f"git checkout -b {self._ce_main_branch}")
        if create_result.get("returncode") != 0:
            raise RuntimeError(
                f"Failed to create run main branch {self._ce_main_branch}: "
                f"{create_result.get('stderr')}"
            )

        print(f"[ADAPTER] Created run main branch {self._ce_main_branch} from {self.initial_branch}", flush=True)

    def _generate_architecture_summary(self) -> str:
        """Generate architecture summary using the reflection agent.

        Uses the reflection agent (read-only: Read, Grep, Glob tools) to
        analyze the program's code and produce a comprehensive architecture summary.

        Returns:
            Architecture summary string.
        """
        # Build the prompt for architecture summarization
        prompt = f"""You are analyzing a codebase to generate an architecture summary.

## Program Being Optimized
- **Entry Point**: `{self.program_path}`
- **Metric**: `{self.metric_path}`

## Your Task
1. Use the Read tool to examine the program entry point file: `{self.program_path.replace(".", "/")}.py`
2. Use Glob to find related Python files in the same directory
3. If there's a README.md, read it for additional context

Then generate an architecture summary (500-2500 characters) that includes:
1. What this program does (high-level purpose)
2. Key modules and their responsibilities
3. Data flow through the system
4. The metric being optimized

Provide the summary as a single markdown-style output."""

        # Call reflection agent with structured output
        result = self._sandbox.exec_reflection_agent(prompt, output_type="architecture")

        summary = result.get("proposed_change", "")
        if not summary or summary == "No change proposed":
            # Fallback to a basic summary
            summary = f"""# Architecture Summary

## Program
- **Entry Point**: `{self.program_path}`
- **Metric**: `{self.metric_path}`

## Overview
This is a DSPy program being optimized by CodeEvolver.

*Generated automatically by CodeEvolver at {datetime.now().isoformat()}*
"""

        return summary

    def _save_architecture_to_file(self, architecture: str) -> None:
        """Save architecture summary to codeevolver.md, commit, and push.

        Args:
            architecture: The architecture summary to save.

        Raises:
            RuntimeError: If file operations fail.
        """
        # Write the architecture summary to codeevolver.md
        # Use heredoc to safely write multi-line content
        write_result = self._sandbox.exec_bash(
            f"cat > codeevolver.md << 'CODEEVOLVER_EOF'\n{architecture}\nCODEEVOLVER_EOF"
        )
        if write_result.get("returncode") != 0:
            print(f"[ADAPTER] Warning: Failed to write codeevolver.md: {write_result.get('stderr')}", flush=True)
            return

        # Configure git and commit the file
        self._sandbox.exec_bash("git config user.email 'codeevolver@codeevolver.ai'")
        self._sandbox.exec_bash("git config user.name 'CodeEvolver'")
        self._sandbox.exec_bash("git add codeevolver.md")

        commit_msg = f"codeevolver mutation. Date: {self._run_timestamp}"
        commit_result = self._sandbox.exec_bash(f'git commit -m "{commit_msg}"')
        if commit_result.get("returncode") != 0:
            print(f"[ADAPTER] Warning: Failed to commit codeevolver.md: {commit_result.get('stderr')}", flush=True)
            return

        print(f"[ADAPTER] Committed codeevolver.md to {self._ce_main_branch}", flush=True)

        # Push the branch to remote (with fresh authentication)
        push_result = self._sandbox.push_authenticated(self._ce_main_branch)
        if not push_result.get("success"):
            raise RuntimeError(
                f"Failed to push {self._ce_main_branch}: {push_result.get('stderr')}"
            )
        print(f"[ADAPTER] Pushed {self._ce_main_branch} to origin", flush=True)

    def _build_initial_code_component(self, git_branch: str) -> str:
        """Build initial _code component as JSON-encoded dict.

        Architecture is stored in codeevolver.md (not in _code) so it stays
        in sync with code changes and can be updated by the coding agent.

        Args:
            git_branch: The git branch for this candidate.

        Returns:
            JSON string with git_branch, change_request, and last_change_summary.
        """
        return json.dumps({
            "git_branch": git_branch,
            "change_request": "",
            "last_change_summary": "Initial state"
        })

    def _get_prompt_texts(self, candidate: dict[str, str]) -> dict[str, str]:
        """Extract prompt texts from candidate, excluding reserved keys."""
        return {k: v for k, v in candidate.items() if k != CODE_COMPONENT_KEY}

    def evaluate(
        self,
        batch: list,
        candidate: dict[str, str],
        capture_traces: bool = True,
    ) -> EvaluationBatch:
        """Run evaluation via sandbox.

        Args:
            batch: List of dicts (raw examples from GEPA).
            candidate: Dict with _code (containing git_branch) and predictor instructions.
            capture_traces: Whether to capture DSPy execution traces.

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories.
        """
        print(f"[ADAPTER] evaluate() called: batch_size={len(batch)}, capture_traces={capture_traces}", flush=True)
        prompt_texts = self._get_prompt_texts(candidate)
        git_branch = self._get_git_branch_from_candidate(candidate)

        # Convert batch items to plain dicts if they aren't already
        batch_json = []
        for ex in batch:
            if isinstance(ex, dict):
                batch_json.append(ex)
            else:
                batch_json.append(dict(ex))

        result = self._sandbox.exec_prebuilt({
            "command": "evaluate",
            "program": self.program_path,
            "metric": self.metric_path,
            "saved_program_json_path": self.saved_program_json_path,
            "candidate": prompt_texts,
            "batch": batch_json,
            "capture_traces": capture_traces,
            "num_threads": self.num_threads,
            "input_keys": self.input_keys,
            "failure_score": self.failure_score,
            "program_lm": self.program_lm,
            "git_branch": git_branch,
        })

        print(f"[ADAPTER] evaluate result: success={result.get('success')}, error={result.get('error', 'none')[:200] if result.get('error') else 'none'}", flush=True)

        # Print logs from sandbox
        if result.get("logs"):
            print(f"[ADAPTER] Sandbox logs:\n" + "\n".join(result["logs"]), flush=True)

        if not result.get("success", False):
            print(f"[ADAPTER] Evaluation failed: {result.get('error', 'unknown')}", flush=True)
            if result.get("traceback"):
                print(f"[ADAPTER] Traceback:\n{result.get('traceback')}", flush=True)
            return EvaluationBatch(
                outputs=[None] * len(batch),
                scores=[self.failure_score] * len(batch),
                trajectories=None,
            )

        return EvaluationBatch(
            outputs=result.get("outputs", []),
            scores=result.get("scores", []),
            trajectories=result.get("trajectories"),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build reflective dataset from DSPy traces via sandbox.

        Delegates to sandbox scripts which use signature_key fingerprinting
        to match serialized trace entries to predictors.
        """
        prompt_texts = self._get_prompt_texts(candidate)

        result = self._sandbox.exec_prebuilt({
            "command": "make_reflective_dataset",
            "program": self.program_path,
            "saved_program_json_path": self.saved_program_json_path,
            "candidate": prompt_texts,
            "trajectories": eval_batch.trajectories or [],
            "scores": eval_batch.scores,
            "components_to_update": components_to_update,
            "failure_score": self.failure_score,
        })

        if not result.get("success", False):
            raise Exception(
                result.get("error", "No valid predictions found for any module.")
            )

        return result["reflective_dataset"]

    def _extract_score(self, score_obj: Any) -> float:
        """Extract a float score from various score formats."""
        if score_obj is None:
            return self.failure_score
        if isinstance(score_obj, dict):
            val = score_obj.get("score")
            if val is not None:
                return float(val)
            return self.failure_score
        if hasattr(score_obj, "score"):
            val = getattr(score_obj, "score", None)
            if val is not None:
                return float(val)
            return self.failure_score
        try:
            return float(score_obj)
        except (TypeError, ValueError):
            return self.failure_score

    def apply_code_mutation(
        self,
        change_request: str,
        change_location: str | None = None,
    ) -> dict:
        """Apply a code mutation via the coding agent.

        Called by GEPA optimizer when code changes are needed.
        The sandbox must have been started with use_venv=True for proper
        isolation between agent SDK and client dependencies.

        Args:
            change_request: Natural language description of the code change.
            change_location: Optional module path hint (e.g., "src/core/agent.py").

        Returns:
            Dict with 'success', 'error', 'output' keys.
        """
        print(f"[ADAPTER] apply_code_mutation() called: {change_request[:100]}...", flush=True)
        return self._sandbox.exec_agent(change_request, change_location)

    def _propose_code_mutation(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> str:
        """Propose and execute a code mutation based on evaluation feedback.

        Two-phase process:
        1. Reflection Phase: Reflective LM reads codeevolver.md, analyzes feedback, proposes change
        2. Execution Phase: Coding agent executes the change and updates codeevolver.md if needed

        Args:
            candidate: Current candidate with _code component (containing git_branch).
            reflective_dataset: Feedback from evaluation.

        Returns:
            JSON-encoded _code string with git_branch, change_request, and last_change_summary.

        Raises:
            RuntimeError: If code mutation fails.
        """
        current_code_data = json.loads(candidate.get(CODE_COMPONENT_KEY, "{}"))
        code_feedback = list(reflective_dataset.get(CODE_COMPONENT_KEY, []))
        parent_branch = current_code_data.get("git_branch", self._ce_main_branch or "main")

        # Phase 1: Reflective LM proposes what to change (reads codeevolver.md for architecture)
        proposed_change = self._reflect_on_code(code_feedback, parent_branch)

        if not proposed_change or proposed_change == "No change proposed":
            # Return unchanged if reflection couldn't propose anything
            return json.dumps({
                "git_branch": parent_branch,  # Stay on same branch
                "change_request": "",
                "last_change_summary": "No change proposed"
            })

        # Phase 2: Create new branch from parent and execute mutation
        new_branch = self._create_mutation_branch(parent_branch)

        # Execute coding agent (commits and pushes changes automatically)
        # Agent also updates codeevolver.md if architectural changes are made
        result = self._sandbox.exec_agent(
            proposed_change,
            push_branch=new_branch,
        )

        if not result.get("success"):
            raise RuntimeError(f"Code mutation failed: {result.get('error')}")

        # Track this change AFTER successful execution to avoid re-attempting from same parent
        if parent_branch not in self._attempted_changes_by_branch:
            self._attempted_changes_by_branch[parent_branch] = []
        self._attempted_changes_by_branch[parent_branch].append(proposed_change[:200])

        # Return updated _code with new git_branch inside
        return json.dumps({
            "git_branch": new_branch,
            "change_request": proposed_change,
            "last_change_summary": result.get("output", "Change applied")[:500]
        })

    def _create_mutation_branch(self, parent_branch: str) -> str:
        """Checkout parent branch and create a new branch for mutation.

        Branch naming uses the run's timestamp for consistency:
        codeevolver-{YYYYMMDDHHMinMin}-{uuid}

        Args:
            parent_branch: Branch to create from.

        Returns:
            Name of the new branch.

        Raises:
            RuntimeError: If branch operations fail.
        """
        # Generate unique branch name with run timestamp
        short_id = uuid.uuid4().hex[:6]
        new_branch = f"codeevolver-{self._run_timestamp}-{short_id}"

        # Checkout parent branch first
        checkout_result = self._sandbox.exec_bash(f"git checkout {parent_branch}")
        if checkout_result.get("returncode") != 0:
            raise RuntimeError(
                f"Failed to checkout parent branch {parent_branch}: "
                f"{checkout_result.get('stderr')}"
            )

        # Create and checkout new branch
        create_result = self._sandbox.exec_bash(f"git checkout -b {new_branch}")
        if create_result.get("returncode") != 0:
            raise RuntimeError(
                f"Failed to create branch {new_branch}: "
                f"{create_result.get('stderr')}"
            )

        print(f"[ADAPTER] Created mutation branch {new_branch} from {parent_branch}", flush=True)
        return new_branch

    def _reflect_on_code(
        self,
        code_feedback: list[dict],
        parent_branch: str,
    ) -> str:
        """Use reflective LM to analyze feedback and propose a targeted change.

        The LM reads codeevolver.md for architecture context, then analyzes
        feedback to prioritize:
        - Code failures (exceptions, crashes)
        - Accuracy issues (low scores)
        - Performance patterns

        Args:
            code_feedback: List of feedback items with input, output, score, exception.
            parent_branch: The branch this mutation will spawn from (for tracking).

        Returns:
            Proposed change as a natural language string.
        """
        reflection_prompt = self._build_code_reflection_prompt(
            feedback=code_feedback,
            parent_branch=parent_branch,
        )

        # Call reflection agent with structured output
        result = self._sandbox.exec_reflection_agent(
            reflection_prompt, output_type="change_request"
        )

        return result.get("proposed_change", "No change proposed")

    def _build_code_reflection_prompt(
        self,
        feedback: list[dict],
        parent_branch: str,
    ) -> str:
        """Build prompt for the reflective LM to analyze and propose a change.

        Args:
            feedback: List of feedback items from evaluation.
            parent_branch: The branch this mutation will spawn from.

        Returns:
            Formatted prompt string.
        """
        # Limit to 10 examples to avoid token limits
        feedback_str = json.dumps(feedback[:10], indent=2)

        # Build additional instructions section if provided
        additional_section = ""
        if self.additional_instructions:
            additional_section = f"""
## Additional Instructions from Client
{self.additional_instructions}
"""

        # Build attempted changes section - only show changes attempted FROM this branch
        # This allows parallel branches to independently discover the same mutation
        attempted_section = ""
        branch_attempts = self._attempted_changes_by_branch.get(parent_branch, [])
        if branch_attempts:
            attempted_list = "\n".join(f"- {change}" for change in branch_attempts[-10:])
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

    def _propose_prompt_mutation(
        self,
        component_name: str,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> str:
        """Propose a new prompt instruction using GEPA's InstructionProposalSignature.

        Args:
            component_name: Name of the predictor component to update.
            candidate: Current candidate dict.
            reflective_dataset: Feedback from evaluation.

        Returns:
            New instruction text for the component.
        """
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        current_instruction = candidate.get(component_name, "")
        component_feedback = list(reflective_dataset.get(component_name, []))

        # GEPA expects an LM callable, not a model name string
        lm_callable = self._get_reflection_lm_callable()

        result = InstructionProposalSignature.run(
            lm=lm_callable,
            input_dict={
                "current_instruction_doc": current_instruction,
                "dataset_with_feedback": component_feedback,
                "prompt_template": self.reflection_prompt_template,
            },
        )

        return result.get("new_instruction", current_instruction)
