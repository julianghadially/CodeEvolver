"""CodeEvolver DSPy adapter for GEPA optimization.

Implements the GEPAAdapter protocol via structural typing (no inheritance).
All DSPy operations are delegated to a GEPAEvalSandbox via JSON IPC,
providing process-level isolation between the GEPA orchestrator and client code.

Copyright notice for GEPA-derived patterns:
Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.core.adapter import EvaluationBatch

logger = logging.getLogger(__name__)

# Reserved key for git branch in candidate dict
GIT_BRANCH_KEY = "git_branch"


class CodeEvolverDSPyAdapter:
    """GEPAAdapter implementation for DSPy programs in CodeEvolver.

    All evaluation, seed candidate building, and reflective dataset
    construction are delegated to the eval sandbox via JSON RPC.
    No dspy is imported in this module.

    Args:
        sandbox_manager: GEPAEvalSandbox instance (already started).
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
    ):
        self._sandbox = sandbox_manager
        self.program_path = program
        self.metric_path = metric
        self.saved_program_json_path = saved_program_json_path
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.input_keys = input_keys or []
        self.initial_branch = initial_branch

    # Use GEPA's default InstructionProposalSignature for reflection
    propose_new_texts = None

    def build_seed_candidate(self) -> dict[str, str]:
        """Extract initial instructions from the DSPy program via sandbox.

        Returns:
            Dict with 'git_branch' key and predictor instruction texts.
        """
        result = self._sandbox.exec_command({
            "command": "build_seed_candidate",
            "program": self.program_path,
            "saved_program_json_path": self.saved_program_json_path,
        })

        if not result.get("success", False):
            raise RuntimeError(
                f"build_seed_candidate failed: {result.get('error', 'unknown')}"
            )

        candidate = result["candidate"]
        candidate[GIT_BRANCH_KEY] = self.initial_branch
        return candidate

    def _get_prompt_texts(self, candidate: dict[str, str]) -> dict[str, str]:
        """Extract prompt texts from candidate, excluding git_branch."""
        return {k: v for k, v in candidate.items() if k != GIT_BRANCH_KEY}

    def evaluate(
        self,
        batch: list,
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run evaluation via sandbox.

        Args:
            batch: List of dicts (raw examples from GEPA).
            candidate: Dict with 'git_branch' and predictor instructions.
            capture_traces: Whether to capture DSPy execution traces.

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories.
        """
        prompt_texts = self._get_prompt_texts(candidate)

        # Convert batch items to plain dicts if they aren't already
        batch_json = []
        for ex in batch:
            if isinstance(ex, dict):
                batch_json.append(ex)
            else:
                batch_json.append(dict(ex))

        result = self._sandbox.exec_command({
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
        })

        if not result.get("success", False):
            logger.warning(f"Evaluation failed: {result.get('error', 'unknown')}")
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

        Delegates to eval_worker which uses signature_key fingerprinting
        to match serialized trace entries to predictors.
        """
        prompt_texts = self._get_prompt_texts(candidate)

        result = self._sandbox.exec_command({
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
