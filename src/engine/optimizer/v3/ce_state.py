"""Copyright © 2026 440 Labs LLC

CEState — the engine 3 run state for agent-managed-git optimization.

Engine 3 has no candidate pool and no Pareto frontier: the architect manages
its own lineage branches (tracked in memory.md) and declares one primary
branch per repo through the `close_iteration` tool. CEState is the
orchestrator-side ledger of that process:

- `branches`: registry of every branch that has appeared in a closing record
  (branch -> last sha / primary flag / iteration bookkeeping).
- `primary_branch`: the current primary — the source of truth set by the last
  `close_iteration` call (with a deterministic fallback when a close leaves
  no primary). The final program of the run is the last primary.
- `iterations`: append-only closing records (see `closing.build_close_record`).
- `total_num_evals`: the eval-row budget counter. Starts at 0 — the seed
  stage's environment sanity eval and the finalization test-set evals are
  deliberately NOT counted; train and comparison evals are.
- `final_comparison_evals`: how many comparison evals the architect has run
  AFTER the eval-row budget was reached. In research mode `run_eval` allows
  exactly one (see `run_eval_tool.RunEvalBinding._enforce_eval_budget`);
  it is a counter rather than a bool so the ledger says how many were taken,
  not just whether the allowance was used.
- `comparison_set_history.jsonl`: an agent-readable JSONL log of every
  comparison-set eval ({ts, iteration, branch, sha, score, n_rows}), appended
  synchronously at eval time so branch/sha are captured before the agent can
  switch branches.

The loop is the single writer of `iterations`/`branches`/`primary_branch`
(post-session); the run_eval binding appends comparison history and bumps the
eval counter synchronously. Plain-JSON persistence keeps resume simple.

Pareto-frontier semantics live in `pareto_evolution/population.py`, not here.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Mapping

from .records import BranchEntry, CloseRecord, ComparisonEvalEntry

logger = logging.getLogger(__name__)

# 4: content-identity closing. `skipped` narrowed to "the iteration produced
# nothing"; `changes_landed` added for "did the program move"; close records
# gained close_tree / parent_tree / final_score_source / subsample_best /
# branches_seen and dropped subsample_close.
# 5: no score is attributed to the closing content —
# `subsample_final` / `final_score_source` / `final_sha` are dropped; records
# carry `subsample_best(+_sha)` / `comparison_best(+_sha)` as progress
# readings, and pareto_evolution's `comparison_state` /
# `comparison_eval_id_at_close` / `comparison_score_at_close` for the closing
# content's loop-mapped fitness. There is no back-compat, so an older
# file is refused rather than migrated.
# 6: comparison-credit ledger (`comparison_credits_balance`,
# specs/archive-public/comparison_credits.md). Persisted so a resume cannot re-earn credits.
# 7: `variance_estimate` — the run's one `run_variance_estimate`
# result (specs/archive-public/sampling_variance.md). Persisted so a resume cannot re-run it
# and every later iteration's prompt keeps rendering it.
_SCHEMA_VERSION = 7


class CEState:
    """Run state for an engine 3 (agent-managed-git) optimization.

    The full schema is the class-level annotations below (every attribute is
    assigned in ``__init__``; nothing else is added later). Record shapes
    (`CloseRecord`, `BranchEntry`, `ComparisonEvalEntry`) live in
    `v3/records.py`.
    """

    seed_branch: str
    seed_sha: str
    # Invariant: always set. The close_iteration registration is the source
    # of truth; the seed branch is the initial primary.
    primary_branch: str
    # Branch registry: every branch that has appeared in a closing record.
    branches: dict[str, BranchEntry]
    # Append-only closing records, one per iteration (see records.CloseRecord).
    iterations: list[CloseRecord]
    # Eval-row budget counter. Starts at 0 (seed sanity + finalization test
    # evals are uncounted; train + comparison evals are counted).
    total_num_evals: int
    # Comparison evals run after the eval-row budget was reached. Research
    # mode allows exactly one such "final" comparison so the architect can
    # verify the branch it is about to leave as primary.
    final_comparison_evals: int
    # pareto_evolution budget observability: how many times the LOOP had to
    # re-run the comparison set because the closing content matched no
    # comparison eval by tree (`comparison_state == "RERUN_BY_LOOP"`), and the
    # rows that cost. Counted budget either way; this is what makes a
    # systematic leak visible per run.
    loop_comparison_reruns: int
    loop_comparison_rows: int
    # Comparison-credit ledger (specs/archive-public/comparison_credits.md): balance of
    # comparison-set runs the run may still spend. Granted by the loop
    # (initial grant before the seed hook, +N at each iteration start);
    # debited by every comparison eval except the post-budget final
    # allowance. Loop-initiated runs may overdraw (negative balance) — a
    # fixed rule is never blocked — but the agent-facing gate refuses at
    # balance <= 0.
    comparison_credits_balance: int
    # The run's one `run_variance_estimate` result (`variance_estimate.
    # VarianceEstimate.to_dict()`), or None until the architect calls the
    # tool. Run-level: rendered into every later iteration's system prompt
    # and read by the revert-check watchdog as its tolerance.
    variance_estimate: dict[str, Any] | None
    # Agent-readable JSONL of every comparison eval (ComparisonEvalEntry lines).
    comparison_history_path: str | None

    def __init__(
        self,
        *,
        seed_branch: str,
        seed_sha: str,
        comparison_history_path: str | None = None,
    ) -> None:
        self.seed_branch = seed_branch
        self.seed_sha = seed_sha
        self.primary_branch = seed_branch
        self.branches = {
            seed_branch: BranchEntry(
                last_sha=seed_sha,
                is_primary=True,
                created_iteration=0,
                last_iteration=0,
            )
        }
        self.iterations = []
        self.total_num_evals = 0
        self.final_comparison_evals = 0
        self.loop_comparison_reruns = 0
        self.loop_comparison_rows = 0
        self.comparison_credits_balance = 0
        self.variance_estimate = None
        self.comparison_history_path = comparison_history_path

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------
    def increment_evals(self, count: int) -> None:
        self.total_num_evals += int(count)

    def note_final_comparison_eval(self) -> None:
        """Record that a post-budget ("final") comparison eval was spent.

        Called by the `run_eval` tool immediately after such an eval returns,
        so the allowance is consumed by the eval that actually ran.
        """
        self.final_comparison_evals += 1

    def note_loop_comparison_rerun(self, n_rows: int) -> None:
        """Record one loop-initiated comparison re-run (pareto_evolution)."""
        self.loop_comparison_reruns += 1
        self.loop_comparison_rows += int(n_rows or 0)

    # ------------------------------------------------------------------
    # Run-variance estimate (specs/archive-public/sampling_variance.md)
    # ------------------------------------------------------------------
    def record_variance_estimate(self, estimate: Mapping[str, Any]) -> None:
        """Store the run's one variance estimate. Refuses a second write —
        the tool is once-per-run by contract, and the registry hides it once
        this is set."""
        if self.variance_estimate is not None:
            raise ValueError("variance_estimate is already recorded for this run")
        self.variance_estimate = dict(estimate)

    # ------------------------------------------------------------------
    # Comparison-credit ledger (specs/archive-public/comparison_credits.md)
    # ------------------------------------------------------------------
    def grant_comparison_credits(self, n: int) -> None:
        self.comparison_credits_balance += int(n)

    def debit_comparison_credit(self) -> None:
        """Spend one comparison credit. May go negative: loop-initiated
        runs overdraw rather than being blocked (the agent-facing gate in
        `run_eval` is what refuses at balance <= 0)."""
        self.comparison_credits_balance -= 1

    # ------------------------------------------------------------------
    # Iteration records
    # ------------------------------------------------------------------
    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    def record_iteration(self, record: CloseRecord | Mapping[str, Any]) -> None:
        """Append one closing record and apply its branch/primary transition.

        The record is `closing.build_close_record`'s output (or a minimal
        {"iteration", "skipped"/"error"} stub for skipped/errored iterations).
        Branch upsert and the primary transition only happen when the record
        names a branch.
        """
        rec: CloseRecord = dict(record)  # type: ignore[assignment]
        self.iterations.append(rec)

        branch = rec.get("branch")
        if not branch:
            return
        iteration = rec.get("iteration")
        entry = self.branches.setdefault(
            branch,
            BranchEntry(
                last_sha=None,
                is_primary=False,
                created_iteration=iteration,
                last_iteration=iteration,
            ),
        )
        # close_sha is the registered branch's actual tip (finalize() may
        # have fallen back to the parent sha without a git revert).
        entry["last_sha"] = rec.get("close_sha") or entry["last_sha"]
        entry["last_iteration"] = iteration

        new_primary = rec.get("primary_branch_after")
        if new_primary and new_primary != self.primary_branch:
            logger.info(
                "primary branch transition at iteration %s: %s -> %s",
                iteration, self.primary_branch, new_primary,
            )
            self.set_primary(new_primary)
        elif new_primary:
            # Same primary; make sure the registry flag is consistent.
            self.set_primary(new_primary)

    def set_primary(self, branch: str) -> None:
        """Mark `branch` as the single primary in the registry."""
        self.branches.setdefault(
            branch,
            BranchEntry(
                last_sha=None,
                is_primary=False,
                created_iteration=None,
                last_iteration=None,
            ),
        )
        for name, entry in self.branches.items():
            entry["is_primary"] = name == branch
        self.primary_branch = branch

    def known_branches(self) -> set[str]:
        return set(self.branches)

    def register_lineage(
        self,
        branch: str,
        *,
        sha: str | None,
        iteration: int,
        is_primary: bool = False,
    ) -> None:
        """Add a branch to the registry outside a closing record.

        Used by ASA's seed hook to register the seed as its own non-primary
        lineage. `is_primary=True` routes through `set_primary` so the
        single-primary invariant holds.
        """
        if branch not in self.branches:
            self.branches[branch] = BranchEntry(
                last_sha=sha,
                is_primary=False,
                created_iteration=iteration,
                last_iteration=iteration,
            )
        if is_primary:
            self.set_primary(branch)

    # ------------------------------------------------------------------
    # Comparison-set history
    # ------------------------------------------------------------------
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
        branch/sha recorded are the ones actually evaluated — immune to the
        agent switching branches afterwards.
        """
        line: dict[str, Any] = {
            "ts": time.time(),
            "iteration": iteration,
            # ComparisonEvalEntry fields (records.py):
            "branch": branch,
            "sha": sha,
            "score": score,
            "n_rows": n_rows,
        }
        if eval_id:
            line["eval_id"] = eval_id
        if tree:
            # Content identity of what was scored — lets a later reader match
            # a record to a branch tip by tree (ASA's lineage representation)
            # instead of by sha, which the post-close codeevolver.md commit
            # moves past.
            line["tree"] = tree
        if scores_by_metric:
            line["scores_by_metric"] = dict(scores_by_metric)
        if row_scores:
            line["row_scores"] = dict(row_scores)
        path = self.comparison_history_path
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:
            logger.warning("failed to append comparison history to %s", path, exc_info=True)

    def latest_comparison_score(self, branch: str) -> float | None:
        """Latest comparison score recorded for `branch` in this run's records."""
        for rec in reversed(self.iterations):
            for comp in reversed(rec.get("comparison_scores") or []):
                if comp.get("branch") == branch:
                    return comp.get("score")
        return None

    # ------------------------------------------------------------------
    # Persistence (plain JSON; resume-safe)
    # ------------------------------------------------------------------
    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "seed_branch": self.seed_branch,
            "seed_sha": self.seed_sha,
            "primary_branch": self.primary_branch,
            "branches": self.branches,
            "iterations": self.iterations,
            "total_num_evals": self.total_num_evals,
            "final_comparison_evals": self.final_comparison_evals,
            "loop_comparison_reruns": self.loop_comparison_reruns,
            "loop_comparison_rows": self.loop_comparison_rows,
            "comparison_credits_balance": self.comparison_credits_balance,
            "variance_estimate": self.variance_estimate,
            "comparison_history_path": self.comparison_history_path,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CEState":
        with open(path) as f:
            d = json.load(f)
        version = d.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"CEState schema_version {version!r} != {_SCHEMA_VERSION} — refusing to load "
                "(engine 2 state files are not compatible)"
            )
        state = cls(
            seed_branch=d["seed_branch"],
            seed_sha=d["seed_sha"],
            comparison_history_path=d.get("comparison_history_path"),
        )
        state.primary_branch = d["primary_branch"]
        state.branches = d["branches"]
        state.iterations = d["iterations"]
        state.total_num_evals = d["total_num_evals"]
        # .get: added inside schema v4, so a state file written mid-run by an
        # earlier build has no key. 0 = allowance untouched.
        state.final_comparison_evals = int(d.get("final_comparison_evals", 0) or 0)
        state.loop_comparison_reruns = int(d.get("loop_comparison_reruns", 0) or 0)
        state.loop_comparison_rows = int(d.get("loop_comparison_rows", 0) or 0)
        state.comparison_credits_balance = int(d["comparison_credits_balance"])
        state.variance_estimate = d.get("variance_estimate")
        return state


__all__ = ["CEState"]
