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
    """Component selector that controls code mutation frequency with exponential decay.

    The ratio is expressed as **prompts per code change**:
        - 0 = only code changes (no prompts)
        - 1 = 1 prompt per code (alternating: code, prompt, code, prompt, ...)
        - 2 = 2 prompts per code (code, prompt, prompt, code, prompt, prompt, ...)
        - 4 = 4 prompts per code (code, prompt, prompt, prompt, prompt, code, ...)

    The ratio increases over time via exponential decay:
        prompts_per_code = initial * (decay_factor ** (iteration // decay_rate))

    With defaults (initial=1, decay_rate=25, decay_factor=2):
        - Iterations 0-24: prompts_per_code = 1 (1:1 ratio)
        - Iterations 25-49: prompts_per_code = 2 (1:2 ratio)
        - Iterations 50-74: prompts_per_code = 4 (1:4 ratio)
        - Iterations 75-99: prompts_per_code = 8 (1:8 ratio)

    Args:
        initial: Starting prompts per code (default: 1).
        decay_rate: Iterations between each multiplier step (default: 25).
        decay_factor: Multiplier applied at each decay step (default: 2).
        code_cutoff_step: Stop code mutations after this iteration (default: None).

    Example usage:
        # Default: start 1:1, decay every 25 iterations
        selector = CodeFrequencyComponentSelector()

        # Code only (no prompts)
        selector = CodeFrequencyComponentSelector(initial=0)

        # Start with 2:1 prompt:code, double every 50 iterations
        selector = CodeFrequencyComponentSelector(initial=2, decay_rate=50)
    """

    def __init__(
        self,
        initial: int = 1,
        decay_rate: int = 25,
        decay_factor: int = 2,
        code_cutoff_step: int | None = None,
    ):
        if initial < 0:
            raise ValueError("initial must be >= 0")
        if decay_rate < 0:
            raise ValueError("decay_rate must be >= 0")
        if decay_factor < 1:
            raise ValueError("decay_factor must be >= 1")
        if code_cutoff_step is not None and code_cutoff_step < 0:
            raise ValueError("code_cutoff_step must be >= 0 or None")

        self.initial = initial
        self.decay_rate = decay_rate
        self.decay_factor = decay_factor
        self.code_cutoff_step = code_cutoff_step

    def _get_prompts_per_code(self, iteration: int) -> int:
        """Calculate prompts per code at this iteration using decay function."""
        if self.decay_rate <= 0:
            return self.initial
        decay_steps = iteration // self.decay_rate
        return self.initial * (self.decay_factor ** decay_steps)

    def _is_code_iteration(self, iteration: int) -> bool:
        """Determine if this iteration should be a code mutation."""
        prompts_per_code = self._get_prompts_per_code(iteration)

        if prompts_per_code == 0:
            # Only code changes, no prompts
            return True

        if self.code_cutoff_step is not None and iteration > self.code_cutoff_step:
            return False

        # Cycle: 1 code then N prompts
        cycle_length = 1 + prompts_per_code
        position_in_cycle = iteration % cycle_length
        return position_in_cycle == 0

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
