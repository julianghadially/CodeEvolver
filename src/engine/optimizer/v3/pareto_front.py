"""Copyright © 2026 440 Labs LLC

Instance-front (Pareto) computation over per-row comparison-set scores.

Shared by the pareto_evolution population (`pareto_evolution/population.py`,
selection weights) and ASA's `branch_comparison` tool (per-lineage Pareto
score). Per `specs/archive-public/v3_asa.md` §Pareto Calculation (GEPA
Algorithm 2):

1. `pareto_front`: for every comparison row, the set of candidates tied at
   the row's maximum score. Ties credit every winner; rows a candidate never
   scored on do not count against it.
2. `prune_dominated`: GEPA's `remove_dominated_programs` — a candidate is
   removed when EVERY row it wins is also won by some other SURVIVING
   candidate (a union cover; no row it wins uniquely). Candidates are
   examined lowest aggregate score first, removed one at a time, to a
   fixpoint, so of two candidates with identical winning sets the lower
   scorer goes (on a score tie, the lower key). A union-covered candidate is
   pruned even when it has the highest aggregate score. Every non-empty row
   keeps at least one winner. A candidate on no row at all is dominated by
   any candidate on at least one row.
3. `pareto_counts`: f(c) = number of rows whose (pruned) winner set contains c.
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Mapping, TypeVar

K = TypeVar("K", bound=Hashable)


def pareto_front(
    row_scores_by_key: Mapping[K, Mapping[str, float | None]],
) -> dict[str, set[K]]:
    """`row_id -> {keys tied at the row's max score}` (unpruned)."""
    front: dict[str, set[K]] = {}
    best: dict[str, float] = {}
    for key, rows in row_scores_by_key.items():
        for rid, score in (rows or {}).items():
            if score is None:
                continue
            if rid not in best or score > best[rid]:
                best[rid] = score
                front[rid] = {key}
            elif score == best[rid]:
                front[rid].add(key)
    return front


def winning_rows(front: Mapping[str, set[K]]) -> dict[K, frozenset[str]]:
    """`key -> rows it is on the front for` (keys on no row are absent)."""
    rows: dict[K, set[str]] = {}
    for rid, winners in front.items():
        for key in winners:
            rows.setdefault(key, set()).add(rid)
    return {k: frozenset(v) for k, v in rows.items()}


def prune_dominated(
    front: Mapping[str, set[K]],
    *,
    keys: Mapping[K, object] | set[K] | None = None,
    scores: Mapping[K, float | None] | None = None,
) -> tuple[dict[str, set[K]], set[K]]:
    """Drop dominated candidates from every row's winner set (GEPA
    `remove_dominated_programs` semantics).

    A candidate is dominated when every row it is on the front for also has
    another surviving candidate on its front. Candidates are considered in
    ascending `scores` order (missing/None -> 0.0; ties -> ascending key) and
    removed one at a time until no further removal is possible, so lower
    scorers are pruned first and the survivors form a cover of the rows
    biased toward high aggregate scorers.

    `keys` (optional) names every candidate under consideration so those on
    no row at all are reported as pruned too. Returns `(pruned_front,
    pruned_keys)`.
    """
    wins = winning_rows(front)
    scores = scores or {}

    def _order(k: K) -> tuple[float, K]:
        return (float(scores.get(k) or 0.0), k)

    programs = sorted(wins, key=_order)
    dominated: set[K] = set()
    found = True
    while found:
        found = False
        for y in programs:
            if y in dominated:
                continue
            covered = all(
                any(o != y and o not in dominated for o in front[rid])
                for rid in wins[y]
            )
            if covered:
                dominated.add(y)
                found = True
                break

    pruned: set[K] = set(dominated)
    if keys is not None and wins:
        # Candidates on no row at all are dominated by anyone on a row.
        pruned |= {k for k in keys if k not in wins}

    out: dict[str, set[K]] = {}
    for rid, winners in front.items():
        kept = {k for k in winners if k not in pruned}
        if kept:
            out[rid] = kept
    return out, pruned


def pareto_counts(front: Mapping[str, set[K]]) -> Counter[K]:
    """f(c): rows whose winner set contains c."""
    counts: Counter[K] = Counter()
    for winners in front.values():
        for key in winners:
            counts[key] += 1
    return counts


__all__ = ["pareto_counts", "pareto_front", "prune_dominated", "winning_rows"]
