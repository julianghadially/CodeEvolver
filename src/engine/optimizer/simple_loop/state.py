"""Copyright © 2026 440 Labs LLC

Run state for the simple loop: the eval-row counter, the comparison-credit
ledger, and the comparison-history append `run_eval` looks up by duck typing.

This is the whole interface `RunEvalBinding` needs from a state object. The
budget is enforced inside `run_eval` (`_enforce_eval_budget`), so the loop
only reads `total_num_evals` between sessions. Deliberately imports nothing
from `optimizer/v3/` (see tests/unit/test_simple_loop_import_boundary.py) —
`record_comparison_eval` mirrors the CEState line shape with stdlib only, so
the file `run_eval`'s messaging promises actually exists on this arm too.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass
class SimpleLoopState:
    """Eval-row accounting shared by every session of one run."""

    total_num_evals: int = 0
    final_comparison_evals: int = 0
    # Comparison-credit ledger (specs/archive-public/comparison_credits.md). The simple loop
    # runs in one process with no resume, so in-memory is durable enough.
    comparison_credits_balance: int = 0
    # Agent-readable comparison-eval log; unset disables the append.
    comparison_history_path: str | None = None

    def increment_evals(self, count: int) -> None:
        """Count rows spent by an agent-initiated `run_eval`."""
        self.total_num_evals += int(count or 0)

    def note_final_comparison_eval(self) -> None:
        """Mark the post-budget comparison allowance as spent."""
        self.final_comparison_evals += 1

    def grant_comparison_credits(self, n: int) -> None:
        self.comparison_credits_balance += int(n)

    def debit_comparison_credit(self) -> None:
        self.comparison_credits_balance -= 1

    def record_comparison_eval(
        self,
        *,
        iteration: int,
        branch: str | None,
        sha: str | None,
        score: float,
        n_rows: int,
        scores_by_metric: Mapping[str, float] | None = None,
        row_scores: Mapping[str, float] | None = None,
        eval_id: str | None = None,
        tree: str | None = None,
    ) -> None:
        """Append one comparison-set eval to the agent-readable JSONL log.

        Called synchronously from the run_eval binding at eval time so the
        branch/sha recorded are the ones actually evaluated. Same line shape
        as `CEState.record_comparison_eval`; posting the file is ALL this
        does — no other CodeEvolver state semantics.
        """
        path = self.comparison_history_path
        if not path:
            return
        line: dict[str, Any] = {
            "ts": time.time(),
            "iteration": iteration,
            "branch": branch,
            "sha": sha,
            "score": score,
            "n_rows": n_rows,
        }
        if eval_id:
            line["eval_id"] = eval_id
        if tree:
            line["tree"] = tree
        if scores_by_metric:
            line["scores_by_metric"] = dict(scores_by_metric)
        if row_scores:
            line["row_scores"] = dict(row_scores)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:
            logger.warning(
                "failed to append comparison history to %s", path, exc_info=True
            )


__all__ = ["SimpleLoopState"]
