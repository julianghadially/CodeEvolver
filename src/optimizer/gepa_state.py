"""
Copyright © 2026 440 Labs LLC

GEPA state management for CodeEvolver optimization.

This module provides state containers for tracking GEPA optimization progress
and serializing state for callbacks and persistence.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    @staticmethod
    def _parse_code_field(candidate: dict[str, str]) -> dict:
        """Parse the _code JSON field from a candidate. Returns empty dict on failure."""
        code_str = candidate.get("_code", "")
        if not code_str:
            return {}
        try:
            return json.loads(code_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _detect_change(
        self, idx: int, candidate: dict[str, str]
    ) -> tuple[str, str, str | None]:
        """Detect what changed between a candidate and its parent.

        Returns (change_type, change_description, git_branch).

        change_type is one of: "seed", "code", "prompt".
        change_description explains what changed this iteration.
        """
        code_data = self._parse_code_field(candidate)
        git_branch = code_data.get("git_branch")

        # Seed candidate (no parent)
        parents = self.parent_programs[idx] if idx < len(self.parent_programs) else None
        if not parents or parents[0] is None or idx == 0:
            return "seed", "Initial seed candidate", git_branch

        parent_idx = parents[0]
        if parent_idx >= len(self.program_candidates):
            return "unknown", "Parent not found", git_branch

        parent = self.program_candidates[parent_idx]
        parent_code_data = self._parse_code_field(parent)

        # Code change: _code.git_branch differs from parent's
        parent_branch = parent_code_data.get("git_branch")
        if git_branch and parent_branch and git_branch != parent_branch:
            change_request = code_data.get("change_request", "Code change")
            return "code", change_request or "Code change", git_branch
        else:
            # Prompt change: find which prompt components differ
            changed_components = []
            for key in candidate:
                if key == "_code":
                    continue
                if candidate[key] != parent.get(key, ""):
                    changed_components.append(key)

            if changed_components:
                desc = f"Prompt change at {', '.join(changed_components)}"
                return "prompt", desc, git_branch

        return "unknown", "No detectable change", git_branch

    def to_history_dict(self) -> dict[str, dict]:
        """Return per-candidate iteration records for state history tracking.

        Only candidates in program_candidates are tracked — GEPA only adds
        candidates that pass subsample evaluation (losers are skipped and
        never enter program_candidates).

        Returns a dict keyed by candidate index (as string for MongoDB compatibility).
        Each value contains the candidate, score, parent info, change type/description,
        and git_branch.
        """
        history: dict[str, dict] = {}
        for idx, candidate in enumerate(self.program_candidates):
            change_type, change_description, git_branch = self._detect_change(idx, candidate)
            parent_candidates = (
                self.parent_programs[idx]
                if idx < len(self.parent_programs)
                else None
            )
            history[str(idx)] = {
                "candidate": candidate,
                "score": self.candidate_scores[idx] if idx < len(self.candidate_scores) else None,
                "parent_candidates": parent_candidates,
                "change_type": change_type,
                "change_description": change_description,
                "git_branch": git_branch,
            }
        return history

    def create_progress_payload(self, best_score: float, best_candidate: dict) -> dict:
        return {
            "current_iteration": self.num_iterations,
            "best_score": best_score,
            "best_candidate": best_candidate,
            "total_metric_calls": self.total_evals,
            "num_candidates": len(self.program_candidates),
            "gepa_state": self.to_dict(),
        }