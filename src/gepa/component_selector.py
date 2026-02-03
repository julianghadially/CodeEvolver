"""Copyright © 2026 440 Labs LLC
Custom component selector for CodeEvolver GEPA optimization.

Controls when code mutations vs prompt mutations are performed.

Copyright notice for GEPA-derived patterns:
Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
"""

from gepa.core.adapter import Trajectory
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector

# Must match the key used in adapter.py
CODE_COMPONENT_KEY = "_code"


class CodeFrequencyComponentSelector(ReflectionComponentSelector):
    """Component selector that controls code mutation frequency.

    Args:
        code_frequency: Number of code iterations per prompt iteration.
            - 0: Prompt only (never do code mutations)
            - 1: Alternating code/prompt (50% each)
            - 2: 2 code per 1 prompt (67% code)
            - 3: 3 code per 1 prompt (75% code)
            Default is 1 (alternating).

        code_cutoff_step: After this iteration, no more code mutations.
            Set to None for no cutoff (default).

    Pattern examples:
        code_frequency=0: prompt, prompt, prompt, ...
        code_frequency=1: code, prompt, code, prompt, ...
        code_frequency=2: code, code, prompt, code, code, prompt, ...
        code_frequency=3: code, code, code, prompt, ...

    Example usage:
        # 3 code per prompt, stop code after iteration 100
        selector = CodeFrequencyComponentSelector(code_frequency=3, code_cutoff_step=100)

        # Prompt only
        selector = CodeFrequencyComponentSelector(code_frequency=0)
    """

    def __init__(
        self,
        code_frequency: int = 1,
        code_cutoff_step: int | None = None,
    ):
        if code_frequency < 0:
            raise ValueError("code_frequency must be >= 0")
        if code_cutoff_step is not None and code_cutoff_step < 0:
            raise ValueError("code_cutoff_step must be >= 0 or None")

        self.code_frequency = code_frequency
        self.code_cutoff_step = code_cutoff_step

    def _is_code_iteration(self, iteration: int) -> bool:
        """Determine if this iteration should be a code mutation."""
        if self.code_frequency == 0:
            return False

        if self.code_cutoff_step is not None and iteration > self.code_cutoff_step:
            return False

        # Cycle: N code iterations then 1 prompt
        cycle_length = self.code_frequency + 1
        position_in_cycle = iteration % cycle_length
        return position_in_cycle < self.code_frequency

    def _get_prompt_components(self, candidate: dict[str, str]) -> list[str]:
        """Get list of prompt components (excluding code component)."""
        return [k for k in candidate.keys() if k != CODE_COMPONENT_KEY]

    def __call__(
        self,
        state: GEPAState,
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        """Select which components to update for this iteration."""
        iteration = state.i + 1

        if CODE_COMPONENT_KEY in candidate and self._is_code_iteration(iteration):
            print(f"[COMPONENT SELECTOR] selected code component for candidate {candidate_idx}", flush=True)
            return [CODE_COMPONENT_KEY]

        prompt_components = self._get_prompt_components(candidate)
        if not prompt_components:
            if CODE_COMPONENT_KEY in candidate:
                return [CODE_COMPONENT_KEY]
            return []

        # Round-robin on prompt components
        pid = state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx]

        prompt_idx = 0
        current_idx = 0
        for name in state.list_of_named_predictors:
            if name == CODE_COMPONENT_KEY:
                if current_idx == pid:
                    pid = (pid + 1) % len(state.list_of_named_predictors)
                current_idx += 1
                continue
            if current_idx == pid:
                break
            prompt_idx += 1
            current_idx += 1

        component_name = prompt_components[prompt_idx % len(prompt_components)]

        next_pid = (pid + 1) % len(state.list_of_named_predictors)
        if state.list_of_named_predictors[next_pid] == CODE_COMPONENT_KEY:
            next_pid = (next_pid + 1) % len(state.list_of_named_predictors)
        state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = next_pid
        print(f"[COMPONENT SELECTOR] selected {component_name} for candidate {candidate_idx}", flush=True)
        return [component_name]
