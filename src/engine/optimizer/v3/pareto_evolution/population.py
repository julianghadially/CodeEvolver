"""Copyright © 2026 440 Labs LLC

Loop-owned candidate population for the pareto_evolution arm.

Engine 3 has no valset and no candidate indexes: a candidate is a closed
iteration (branch + close sha) whose fitness is its per-row score vector on
the run's FIXED comparison set (seed 42, identical rows on every branch —
the one sample where per-row scores are comparable across candidates).

The population is keyed by iteration number (0 = seed). It is loop-internal
state: persisted as JSON under `history_dir()` and never referenced by any
prompt or skill — the research plan's Variable Table makes the population
log loop-owned for this arm, and the agent's picture of other branches comes
from the same surfaces EE reads (memory.md, iteration_change_summaries.md,
comparison_set_history.jsonl).

Frontier semantics follow GEPA (Algorithm 2): per comparison row, the set of
candidates tied at the row's maximum score, with dominated candidates pruned
per GEPA's `remove_dominated_programs` (a candidate every one of whose
winning rows is also won by another surviving candidate is off the front;
lowest comparison score pruned first, to a fixpoint — `v3/pareto_front.py`,
shared with ASA); a candidate's selection weight is the number of rows it is
on the pruned front for (`select_candidate._instance_front_counts`). `Population` duck-types the
surface `select_candidate._pareto_sample` reads — `get_pareto_front`,
`get_frontier_members`, `get_best_program` — with keys meaning iteration
numbers instead of list indexes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Mapping, TypedDict

from src.engine.paths import history_dir

from ..pareto_front import pareto_front, prune_dominated

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class Candidate(TypedDict, total=False):
    """One admitted program in the population."""

    iteration: int              # 0 = seed
    branch: str
    sha: str                    # the iteration's close_sha — canonical identity
    tree: str | None
    parent_iteration: int | None
    # str(example_id) -> primary score on the comparison set.
    row_scores: dict[str, float]
    # Aggregate comparison-set score (the evaluator's mean over rows).
    comparison_set_score: float
    # Informational train evidence from the admitting iteration (baseline and
    # best on the same locked subsample). NOT an admission criterion —
    # `train_best <= train_baseline` is valid here; admission is
    # the existence of a comparison eval. Not a score of `sha`. None on seed.
    train_baseline: float | None
    train_best: float | None
    admitted_at: float


class Population:
    """Iteration-keyed candidate population with an on-demand instance front."""

    def __init__(self, candidates: Mapping[int, Candidate] | None = None) -> None:
        self.candidates: dict[int, Candidate] = {
            int(k): dict(v) for k, v in (candidates or {}).items()  # type: ignore[misc]
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @classmethod
    def path(cls) -> str:
        return f"{history_dir()}/population.json"

    @classmethod
    def load_or_none(cls, path: str | None = None) -> "Population | None":
        path = path or cls.path()
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        version = data.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"population.json schema_version={version!r}; expected "
                f"{_SCHEMA_VERSION}"
            )
        return cls({int(k): v for k, v in data.get("candidates", {}).items()})

    def save(self, path: str | None = None) -> None:
        path = path or self.path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "candidates": {str(k): v for k, v in sorted(self.candidates.items())},
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(self, cand: Candidate) -> None:
        it = int(cand["iteration"])
        if it in self.candidates:
            raise ValueError(f"population already holds iteration {it}")
        rec: Candidate = dict(cand)  # type: ignore[assignment]
        rec.setdefault("admitted_at", time.time())
        self.candidates[it] = rec

    def __contains__(self, iteration: object) -> bool:
        return iteration in self.candidates

    def __len__(self) -> int:
        return len(self.candidates)

    # ------------------------------------------------------------------
    # Frontier / selection surface (duck-typed for select_candidate)
    # ------------------------------------------------------------------
    def get_pareto_front(self) -> dict[str, set[int]]:
        """`example_id -> {iterations tied at the row's max score}`, with
        dominated candidates pruned.

        Rows a candidate never scored on do not count against it (it simply
        is not on that row's front). Ties credit every winner, as in GEPA's
        instance-front update; then a candidate with no row it wins uniquely
        among the survivors is removed, lowest `comparison_set_score` first,
        to a fixpoint (GEPA `remove_dominated_programs`,
        `pareto_front.prune_dominated`).
        """
        front = pareto_front(
            {it: cand.get("row_scores") or {} for it, cand in self.candidates.items()}
        )
        pruned_front, _ = prune_dominated(
            front,
            scores={it: cand.get("comparison_set_score") for it, cand in self.candidates.items()},
        )
        return pruned_front

    def get_frontier_members(self) -> set[int]:
        members: set[int] = set()
        for winners in self.get_pareto_front().values():
            members |= winners
        return members

    def is_on_frontier(self, iteration: int) -> bool:
        return iteration in self.get_frontier_members()

    def get_best_program(self) -> int:
        """Iteration with the highest comparison-set score (tie -> lowest
        iteration number, i.e. the older candidate)."""
        if not self.candidates:
            raise ValueError("population is empty")
        return min(
            self.candidates,
            key=lambda it: (
                -float(self.candidates[it].get("comparison_set_score") or 0.0),
                it,
            ),
        )

    def best(self) -> Candidate:
        return self.candidates[self.get_best_program()]

    def get(self, iteration: int) -> Candidate:
        return self.candidates[int(iteration)]

    def summary(self) -> list[dict[str, Any]]:
        """Compact per-candidate view for logs / run metrics."""
        front = self.get_frontier_members()
        best = self.get_best_program() if self.candidates else None
        return [
            {
                "iteration": it,
                "branch": c.get("branch"),
                "sha": (c.get("sha") or "")[:8],
                "comparison_set_score": c.get("comparison_set_score"),
                "on_frontier": it in front,
                "is_best": it == best,
            }
            for it, c in sorted(self.candidates.items())
        ]


__all__ = ["Candidate", "Population"]
