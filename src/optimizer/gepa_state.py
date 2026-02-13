"""
Copyright © 2026 440 Labs LLC

GEPA state management for CodeEvolver optimization.

This module provides state containers for tracking GEPA optimization progress
and serializing state for callbacks and persistence.
"""

from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState


class GEPAStateRecord:
    """Serializable record of GEPA optimization state.

    This class provides a clean interface for capturing and serializing GEPA state
    for callbacks, persistence, and progress tracking. It separates state representation
    from serialization logic.

    Attributes:
        program_candidates: List of all candidate programs evaluated.
        candidate_scores: Aggregate validation scores for each candidate.
        parent_programs: Parent program indices for each candidate.
        num_iterations: Number of iterations completed.
        total_evals: Total number of metric evaluations performed.
    """

    def __init__(
        self,
        program_candidates: list[dict[str, str]],
        candidate_scores: list[float],
        parent_programs: list[list[int | None]],
        num_iterations: int,
        total_evals: int,
    ):
        self.program_candidates = program_candidates
        self.candidate_scores = candidate_scores
        self.parent_programs = parent_programs
        self.num_iterations = num_iterations
        self.total_evals = total_evals

    @classmethod
    def from_gepa_state(cls, gepa_state: GEPAState) -> "GEPAStateRecord":
        """Create a GEPAStateRecord from a live GEPAState instance.

        This is used during optimization to capture state at each iteration.

        Args:
            gepa_state: Live GEPA state from the optimization loop.

        Returns:
            GEPAStateRecord with current state snapshot.
        """
        parent_programs=getattr(gepa_state, "parent_program_for_candidate",  getattr(gepa_state, "parent_program_for_candidates", []))
        return cls(
            program_candidates=gepa_state.program_candidates,
            candidate_scores=gepa_state.program_full_scores_val_set,
            parent_programs=parent_programs,
            num_iterations=gepa_state.i + 1,
            total_evals=gepa_state.total_num_evals,
        )

    @classmethod
    def from_gepa_result(cls, result: GEPAResult) -> "GEPAStateRecord":
        """Create a GEPAStateRecord from a completed GEPAResult.

        This is used after optimization completes to capture the final state.

        Args:
            result: GEPAResult from the completed optimization run.

        Returns:
            GEPAStateRecord with final state.
        """
        return cls(
            program_candidates=result.candidates,
            candidate_scores=result.val_aggregate_scores,
            parent_programs=result.parents,
            num_iterations=result.num_candidates,
            total_evals=result.total_metric_calls or 0,
        )

    def to_dict(self) -> dict:
        """Serialize state to a dictionary for JSON serialization.

        Returns:
            Dictionary with all state fields, ready for JSON serialization.
        """
        return {
            "program_candidates": self.program_candidates,
            "candidate_scores": self.candidate_scores,
            "parent_programs": self.parent_programs,
            "num_iterations": self.num_iterations,
            "total_evals": self.total_evals,
        }
    def create_progress_payload(self, best_score: float, best_candidate: dict) -> dict:
        return {
            "current_iteration": self.num_iterations,
            "best_score": best_score,
            "best_candidate": best_candidate,
            "total_metric_calls": self.total_evals,
            "num_candidates": len(self.program_candidates),
            "gepa_state": self.to_dict(),
        }