"""Copyright © 2026 440 Labs LLC

Pure planners: which `EvalTask`s a phase still needs, given the store.

Draw namespaces are global so `EvalTask.suffix` (and with it the trace dir
and the per-draw DSPy cache dir) is unique per (phase, draw):

    screen   draw 0
    select   draws 1 .. reps_select
    confirm  draws reps_select+1 .. reps_select+reps_confirm  (round r <-> draw reps_select+r)
    final    same draws as confirm, rows H - W_final only

Every planner returns only the rows the store is missing for that key, so a
resumed stage never re-runs what it has and a cached observation is never
counted as a new repetition.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .evaluate import EvalTask, task_for
from .store import AuditStore

PHASE_SCREEN, PHASE_SELECT, PHASE_CONFIRM, PHASE_FINAL = "screen", "select", "confirm", "final"


def select_draws(reps_select: int) -> list[int]:
    return list(range(1, reps_select + 1))


def confirm_draws(reps_select: int, reps_confirm: int) -> list[int]:
    return list(range(reps_select + 1, reps_select + reps_confirm + 1))


def plan_screen(cands: Iterable[dict], H: list[int], store: AuditStore) -> list[EvalTask]:
    tasks = []
    for c in cands:
        if not c.get("tree") or c.get("unstable"):
            continue
        missing = store.missing_rows(c["job_id"], c["tree"], PHASE_SCREEN, H, 0)
        if missing:
            tasks.append(task_for(c, missing, 0, PHASE_SCREEN))
    return tasks


def plan_select(portfolio: Iterable[dict], H: list[int], store: AuditStore, reps_select: int) -> list[EvalTask]:
    tasks = []
    for c in portfolio:
        for d in select_draws(reps_select):
            missing = store.missing_rows(c["job_id"], c["tree"], PHASE_SELECT, H, d)
            if missing:
                tasks.append(task_for(c, missing, d, PHASE_SELECT))
    return tasks


def plan_confirm(portfolio_by_tree: Mapping[str, dict], rows_by_winner: Mapping[str, list[int]],
                 store: AuditStore, reps_select: int, reps_confirm: int) -> list[EvalTask]:
    """One task per (winner program, round) over the rows it won."""
    tasks = []
    for tree, rows in rows_by_winner.items():
        c = portfolio_by_tree[tree]
        for d in confirm_draws(reps_select, reps_confirm):
            missing = store.missing_rows(c["job_id"], tree, PHASE_CONFIRM, sorted(rows), d)
            if missing:
                tasks.append(task_for(c, missing, d, PHASE_CONFIRM))
    return tasks


def plan_final(final: dict, H: list[int], rows_won_by_final: Iterable[int], store: AuditStore,
               reps_select: int, reps_confirm: int) -> list[EvalTask]:
    """The final program's fresh rounds on H minus the rows it won (those are
    read from the confirm phase)."""
    won = set(rows_won_by_final)
    rows = [r for r in H if r not in won]
    tasks = []
    for d in confirm_draws(reps_select, reps_confirm):
        missing = store.missing_rows(final["job_id"], final["tree"], PHASE_FINAL, rows, d)
        if missing:
            tasks.append(task_for(final, missing, d, PHASE_FINAL))
    return tasks


def phase_matrix(store: AuditStore, job_id: str, tree: str, phase: str, rows: Iterable[int],
                 draws: Iterable[int]) -> dict[int, list[float]]:
    """`row -> [score per draw]` (draw order) with failed rows at the
    evaluator's recorded score (0.0 when none); missing keys are omitted."""
    out: dict[int, list[float]] = {}
    for r in rows:
        vals = []
        for d in draws:
            rec = store.record(job_id, tree, phase, r, d)
            if rec is None:
                break
            s = rec.get("score")
            vals.append(float(s) if s is not None else 0.0)
        if vals:
            out[r] = vals
    return out


def count_rows(tasks: Iterable[EvalTask]) -> dict[str, Any]:
    tasks = list(tasks)
    return {"calls": len(tasks), "rows": sum(len(t.row_ids) for t in tasks)}


__all__ = [
    "PHASE_CONFIRM", "PHASE_FINAL", "PHASE_SCREEN", "PHASE_SELECT", "confirm_draws", "count_rows",
    "phase_matrix", "plan_confirm", "plan_final", "plan_screen", "plan_select", "select_draws",
]
