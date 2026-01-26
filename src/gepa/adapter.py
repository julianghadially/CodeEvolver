"""CodeEvolver DSPy adapter for GEPA optimization.

Implements the GEPAAdapter protocol via structural typing (no inheritance).
Follows the pattern from gepa.adapters.dspy_adapter.DspyAdapter.

Evaluation runs in Modal sandbox using user's eval/evaluate.py script.

Copyright notice for GEPA-derived patterns:
Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
"""

import importlib
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import dspy
from dspy.primitives import Example, Prediction

from gepa.core.adapter import EvaluationBatch


# Reserved key for git branch in candidate dict
GIT_BRANCH_KEY = "git_branch"

# Default path to user's evaluation script
DEFAULT_EVAL_SCRIPT = "eval/evaluate.py"


class CodeEvolverDSPyAdapter:
    """GEPAAdapter implementation for DSPy programs in CodeEvolver.

    Conforms to GEPAAdapter[Example, TraceData, Prediction] protocol.

    Evaluation runs in Modal sandbox by calling user's eval script:
        python eval/evaluate.py --candidate X --batch Y --output Z

    Args:
        workspace_path: Path to the cloned user repository.
        entry_point: DSPy module class path (e.g., "src.factchecker.FactCheckerPipeline").
        metric_fn: Callable metric(example, prediction) -> float.
        program_json_path: Relative path to program.json within the repo (optional).
        eval_script: Path to user's evaluation script (default: eval/evaluate.py).
        task_lm: DSPy LM model string (e.g., "openai/gpt-5-mini").
        failure_score: Score to assign on evaluation failure.
        num_threads: Number of threads for DSPy Evaluate.
        initial_branch: Initial git branch name for seed candidate.
    """

    def __init__(
        self,
        workspace_path: str,
        entry_point: str,
        metric_fn: Callable,
        program_json_path: str | None = None,
        eval_script: str = DEFAULT_EVAL_SCRIPT,
        task_lm: str = "openai/gpt-5-mini",
        failure_score: float = 0.0,
        num_threads: int = 1,
        initial_branch: str = "main",
    ):
        self.workspace_path = Path(workspace_path)
        self.entry_point = entry_point
        self.metric_fn = metric_fn
        self.program_json_path = program_json_path
        self.eval_script = eval_script
        self.task_lm = task_lm
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.initial_branch = initial_branch
        self.rng = random.Random(0)

        # Sandbox manager is injected by the optimizer
        self._sandbox_manager = None

        # Ensure workspace is on sys.path for user code imports
        ws_str = str(self.workspace_path)
        if ws_str not in sys.path:
            sys.path.insert(0, ws_str)

        # Configure DSPy with task LM
        dspy.configure(lm=dspy.LM(task_lm))

        # Import the user's DSPy module class
        module_path, class_name = entry_point.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        self.student_class = getattr(mod, class_name)

        # Load a seed instance to discover predictor names
        seed_instance = self.student_class()
        if self.program_json_path:
            full_path = self.workspace_path / self.program_json_path
            if full_path.exists():
                seed_instance.load(str(full_path))
        self._named_predictor_names = [name for name, _ in seed_instance.named_predictors()]

    def set_sandbox_manager(self, manager: Any) -> None:
        """Inject sandbox manager from optimizer."""
        self._sandbox_manager = manager

    # Use GEPA's default InstructionProposalSignature for reflection
    propose_new_texts = None

    def build_seed_candidate(self) -> dict[str, str]:
        """Extract initial instructions from the DSPy program to form seed candidate.

        Returns:
            Dict with 'git_branch' key and predictor instruction texts.
            Example: {"git_branch": "main", "predictor.predict": "instruction..."}
        """
        program = self.student_class()
        if self.program_json_path:
            full_path = self.workspace_path / self.program_json_path
            if full_path.exists():
                program.load(str(full_path))

        # Start with git_branch key for tracking
        seed = {GIT_BRANCH_KEY: self.initial_branch}

        # Extract instructions from each predictor
        for name, pred in program.named_predictors():
            seed[name] = pred.signature.instructions

        return seed

    def _get_prompt_texts(self, candidate: dict[str, str]) -> dict[str, str]:
        """Extract prompt texts from candidate, excluding git_branch."""
        return {k: v for k, v in candidate.items() if k != GIT_BRANCH_KEY}

    def _build_program(self, candidate: dict[str, str]) -> dspy.Module:
        """Instantiate DSPy module, load program.json state, apply candidate instructions."""
        program = self.student_class()
        if self.program_json_path:
            full_path = self.workspace_path / self.program_json_path
            if full_path.exists():
                program.load(str(full_path))

        # Apply candidate instruction texts to matching predictors
        prompt_texts = self._get_prompt_texts(candidate)
        for name, pred in program.named_predictors():
            if name in prompt_texts:
                pred.signature = pred.signature.with_instructions(prompt_texts[name])
        return program

    def evaluate(
        self,
        batch: list[Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run evaluation in sandbox using user's eval script.

        The sandbox:
        1. Checks out candidate's git_branch
        2. Writes candidate.json and batch.json
        3. Runs: python {eval_script} --candidate X --batch Y --output Z
        4. Reads and returns results

        Args:
            batch: List of DSPy Examples to evaluate.
            candidate: Dict with 'git_branch' and predictor instructions.
            capture_traces: Whether to request traces (passed to user script).

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories.
        """
        if self._sandbox_manager is None:
            raise RuntimeError(
                "Sandbox manager not set. Call set_sandbox_manager() first, "
                "or use run_gepa_optimization() which handles this automatically."
            )

        git_branch = candidate.get(GIT_BRANCH_KEY, self.initial_branch)
        prompt_texts = self._get_prompt_texts(candidate)

        # Convert batch to JSON-serializable format
        batch_json = [dict(ex) for ex in batch]

        # Run evaluation in sandbox
        result = self._sandbox_manager.run_evaluation(
            git_branch=git_branch,
            eval_script=self.eval_script,
            candidate_json=prompt_texts,
            batch_json=batch_json,
            capture_traces=capture_traces,
        )

        if not result.get("success", False):
            # Return failure scores for all examples
            return EvaluationBatch(
                outputs=[None] * len(batch),
                scores=[self.failure_score] * len(batch),
                trajectories=None,
            )

        return EvaluationBatch(
            outputs=result.get("outputs", []),
            scores=result.get("scores", []),
            trajectories=result.get("traces"),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build reflective dataset from DSPy traces for the given components.

        Simplified version of DspyAdapter.make_reflective_dataset: uses score
        and ground truth labels as feedback instead of custom feedback functions.
        """
        from dspy.teleprompt.bootstrap_trace import FailedPrediction

        program = self._build_program(candidate)

        ret_d: dict[str, list[dict[str, Any]]] = {}
        for pred_name in components_to_update:
            # Find the predictor module for this component
            module = None
            for name, m in program.named_predictors():
                if name == pred_name:
                    module = m
                    break
            if module is None:
                continue

            items: list[dict[str, Any]] = []
            for data in eval_batch.trajectories or []:
                trace = data.get("trace", [])
                example = data.get("example")
                prediction = data.get("prediction")
                score = self._extract_score(data.get("score"))

                # Find trace instances matching this predictor's signature
                trace_instances = [t for t in trace if t[0].signature.equals(module.signature)]
                trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]
                if len(trace_instances) == 0:
                    # Check for failed predictions
                    failed_instances = [
                        t for t in trace
                        if t[0].signature.equals(module.signature) and isinstance(t[2], FailedPrediction)
                    ]
                    if failed_instances:
                        selected = failed_instances[0]
                    else:
                        continue
                else:
                    if isinstance(prediction, FailedPrediction):
                        continue
                    selected = self.rng.choice(trace_instances)

                inputs = selected[1]
                pred_output = selected[2]

                # Build input dict
                new_inputs = {k: str(v) for k, v in inputs.items()}

                # Build output dict
                if isinstance(pred_output, FailedPrediction):
                    new_outputs = (
                        "Couldn't parse the output as per the expected format. "
                        f"Raw response:\n```\n{pred_output.completion_text}\n```"
                    )
                else:
                    new_outputs = {k: str(v) for k, v in pred_output.items()}

                # Build feedback from score and ground truth
                feedback_parts = [f"Score: {score}"]
                if example is not None:
                    input_keys = set(inputs.keys())
                    for k, v in example.items():
                        if k not in input_keys:
                            feedback_parts.append(f"Expected {k}: {v}")

                d: dict[str, Any] = {
                    "Inputs": new_inputs,
                    "Generated Outputs": new_outputs,
                    "Feedback": "\n".join(feedback_parts),
                }
                items.append(d)

            if items:
                ret_d[pred_name] = items

        if not ret_d:
            raise Exception("No valid predictions found for any module.")

        return ret_d

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


def parse_eval_output(result_json: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate output from user's evaluation script.

    Expected format:
    {
        "scores": [1.0, 0.0, ...],
        "outputs": [{"verdict": "..."}, ...],
        "traces": [...]  // optional
    }

    Returns:
        Dict with 'success', 'scores', 'outputs', 'traces'
    """
    if not isinstance(result_json, dict):
        return {
            "success": False,
            "error": "Result is not a dict",
            "scores": [],
            "outputs": [],
            "traces": None,
        }

    scores = result_json.get("scores", [])
    if not isinstance(scores, list):
        return {
            "success": False,
            "error": "scores is not a list",
            "scores": [],
            "outputs": [],
            "traces": None,
        }

    return {
        "success": True,
        "scores": scores,
        "outputs": result_json.get("outputs", []),
        "traces": result_json.get("traces"),
    }
