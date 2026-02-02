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
from typing import Any

from gepa.core.adapter import EvaluationBatch

logger = logging.getLogger(__name__)

# Reserved key for git branch in candidate dict
GIT_BRANCH_KEY = "git_branch"
# Reserved key for code component (underscore prefix distinguishes from DSPy predictor names)
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

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose new texts for specified components.

        Routes _code component to code mutation and others to prompt mutation.
        When _code is mutated, also updates git_branch to the new branch.

        Args:
            candidate: Current candidate dict with git_branch, _code, and predictor texts.
            reflective_dataset: Feedback data per component from evaluation.
            components_to_update: List of component names to update.

        Returns:
            Dict mapping component names to new text values.
            May include GIT_BRANCH_KEY if code was mutated.
        """
        new_texts = {}

        for component_name in components_to_update:
            if component_name == CODE_COMPONENT_KEY:
                # Code mutation returns both _code and git_branch
                code_result = self._propose_code_mutation(
                    candidate, reflective_dataset
                )
                new_texts[CODE_COMPONENT_KEY] = code_result[CODE_COMPONENT_KEY]
                new_texts[GIT_BRANCH_KEY] = code_result[GIT_BRANCH_KEY]
            else:
                new_texts[component_name] = self._propose_prompt_mutation(
                    component_name, candidate, reflective_dataset
                )

        return new_texts

    def build_seed_candidate(self) -> dict[str, str]:
        """Extract initial instructions from the DSPy program via sandbox.

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

        candidate = result["candidate"]
        candidate[GIT_BRANCH_KEY] = self.initial_branch
        candidate[CODE_COMPONENT_KEY] = self._build_initial_code_component()
        print(f"[ADAPTER] Seed candidate has {len(candidate)} keys", flush=True)
        return candidate

    def _build_initial_code_component(self) -> str:
        """Build initial _code component as JSON-encoded dict.

        Returns:
            JSON string with architecture, change_request, and last_change_summary.
        """
        architecture = self._load_architecture()

        return json.dumps({
            "architecture": architecture,
            "change_request": "",
            "last_change_summary": "Initial state"
        })

    def _load_architecture(self) -> str:
        """Load architecture description from client repo.

        Tries requirements.md first, then README.md, otherwise generates default.

        Returns:
            Architecture description string.
        """
        result = self._sandbox.exec_bash(
            "cat requirements.md 2>/dev/null || cat README.md 2>/dev/null || echo ''"
        )
        content = result.get("stdout", "").strip()

        if not content:
            content = f"""# System Architecture
Program: {self.program_path}
Metric: {self.metric_path}

(Auto-generated. Add a requirements.md to your repository for better context.)
"""
        return content

    def _get_prompt_texts(self, candidate: dict[str, str]) -> dict[str, str]:
        """Extract prompt texts from candidate, excluding reserved keys."""
        return {k: v for k, v in candidate.items()
                if k not in (GIT_BRANCH_KEY, CODE_COMPONENT_KEY)}

    def evaluate(
        self,
        batch: list,
        candidate: dict[str, str],
        capture_traces: bool = True,
    ) -> EvaluationBatch:
        """Run evaluation via sandbox.

        Args:
            batch: List of dicts (raw examples from GEPA).
            candidate: Dict with 'git_branch' and predictor instructions.
            capture_traces: Whether to capture DSPy execution traces.

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories.
        """
        print(f"[ADAPTER] evaluate() called: batch_size={len(batch)}, capture_traces={capture_traces}", flush=True)
        prompt_texts = self._get_prompt_texts(candidate)

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
    ) -> dict[str, str]:
        """Propose and execute a code mutation based on evaluation feedback.

        Two-phase process:
        1. Reflection Phase: Reflective LM analyzes feedback and proposes a change
        2. Execution Phase: Coding agent executes the proposed change on a new branch

        Args:
            candidate: Current candidate with _code component and git_branch.
            reflective_dataset: Feedback from evaluation.

        Returns:
            Dict with CODE_COMPONENT_KEY (JSON-encoded code data) and
            GIT_BRANCH_KEY (new branch name).

        Raises:
            RuntimeError: If code mutation fails.
        """
        current_code_data = json.loads(candidate.get(CODE_COMPONENT_KEY, "{}"))
        code_feedback = list(reflective_dataset.get(CODE_COMPONENT_KEY, []))
        parent_branch = candidate.get(GIT_BRANCH_KEY, "main")

        # Phase 1: Reflective LM proposes what to change
        proposed_change = self._reflect_on_code(current_code_data, code_feedback)

        if not proposed_change or proposed_change == "No change proposed":
            # Return unchanged if reflection couldn't propose anything
            return {
                CODE_COMPONENT_KEY: json.dumps({
                    "architecture": current_code_data.get("architecture", ""),
                    "change_request": "",
                    "last_change_summary": "No change proposed"
                }),
                GIT_BRANCH_KEY: parent_branch,  # Stay on same branch
            }

        # Phase 2: Create new branch from parent and execute mutation
        new_branch = self._create_mutation_branch(parent_branch)

        # Execute coding agent (commits changes automatically)
        result = self._sandbox.exec_agent(proposed_change)

        if not result.get("success"):
            raise RuntimeError(f"Code mutation failed: {result.get('error')}")

        # Return updated code data and new branch
        return {
            CODE_COMPONENT_KEY: json.dumps({
                "architecture": current_code_data.get("architecture", ""),
                "change_request": proposed_change,
                "last_change_summary": result.get("output", "Change applied")[:500]
            }),
            GIT_BRANCH_KEY: new_branch,
        }

    def _create_mutation_branch(self, parent_branch: str) -> str:
        """Checkout parent branch and create a new branch for mutation.

        Args:
            parent_branch: Branch to create from.

        Returns:
            Name of the new branch.

        Raises:
            RuntimeError: If branch operations fail.
        """
        # Generate unique branch name
        short_id = uuid.uuid4().hex[:6]
        new_branch = f"codeevolver-{short_id}"

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
        current_code_data: dict,
        code_feedback: list[dict],
    ) -> str:
        """Use reflective LM to analyze feedback and propose a targeted change.

        The LM has agency to prioritize:
        - Code failures (exceptions, crashes)
        - Accuracy issues (low scores)
        - Performance patterns

        Args:
            current_code_data: Dict with architecture, pending_change_request, etc.
            code_feedback: List of feedback items with input, output, score, exception.

        Returns:
            Proposed change as a natural language string.
        """
        reflection_prompt = self._build_code_reflection_prompt(
            architecture=current_code_data.get("architecture", ""),
            feedback=code_feedback
        )

        # Call reflection agent (read-only tools, no edits)
        result = self._sandbox.exec_reflection_agent(reflection_prompt)

        return result.get("proposed_change", "No change proposed")

    def _build_code_reflection_prompt(
        self,
        architecture: str,
        feedback: list[dict],
    ) -> str:
        """Build prompt for the reflective LM to analyze and propose a change.

        Args:
            architecture: Description of the system architecture.
            feedback: List of feedback items from evaluation.

        Returns:
            Formatted prompt string.
        """
        # Limit to 10 examples to avoid token limits
        feedback_str = json.dumps(feedback[:10], indent=2)

        return f"""You are analyzing the performance of an AI system to propose a single targeted code change.

## System Architecture
{architecture}

## Evaluation Feedback
Each item shows an example input, the system output, and the score (1.0 = perfect).
Items may also include exceptions if the code failed.

{feedback_str}

## Your Task
Analyze the feedback and propose ONE specific, targeted code change that would most improve performance.
- If there are code failures (exceptions), prioritize fixing those
- If scores are consistently low for certain input patterns, propose changes to handle those cases
- Be specific: mention file paths and what to change

Respond with a clear, actionable change request that a coding agent can execute."""

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

        result = InstructionProposalSignature.run(
            lm=self.reflection_lm,
            input_dict={
                "current_instruction_doc": current_instruction,
                "dataset_with_feedback": component_feedback,
                "prompt_template": self.reflection_prompt_template,
            },
        )

        return result.get("new_instruction", current_instruction)
