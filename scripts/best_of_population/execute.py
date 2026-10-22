"""Copyright © 2026 440 Labs LLC

Phase executor: plan -> run -> replan once (the retry) -> fill failed rows.

Evaluator policies are preserved: a row whose program call raised is a
`failed` row scored by the evaluator (0.0) and is COUNTED, never dropped; a
whole-eval failure (timeout, crash) yields no rows, so the replan retries it
once; a candidate still missing rows after that is `unstable` (screening) or
a hard error (later phases). A fill pass re-runs only the failed rows once,
under a distinct `<phase>_fill` suffix, and `AuditStore.append_fill` swaps in
the new observation only where the stored one was failed. Byte-identical
repeat draws (an LLM response cache) are detected within a phase by
`evaluate._check_identical` and across phases by `cross_phase_guard`.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .config import AuditConfig, BenchSpec
from .evaluate import CALL_OVERHEAD_SECONDS, DEFAULT_SECONDS_PER_ROW, EvalTask, execute_tasks, task_for
from .store import AuditStore


@dataclass
class PhaseReport:
    phase: str
    planned_calls: int = 0
    planned_rows: int = 0
    attempts: int = 0
    executed_calls: int = 0
    errors: list[str] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)     # {tree, draw, rows} still missing
    filled_rows: int = 0
    still_failed: dict[str, int] = field(default_factory=dict)  # tree -> rows failed after the fill
    identical_pairs: list[tuple[str, str]] = field(default_factory=list)
    dry_run: bool = False
    estimate: dict[str, Any] = field(default_factory=dict)


def eval_spec(cfg: AuditConfig, spec: BenchSpec) -> BenchSpec:
    """Every audit eval uses the run's held-out cap."""
    t = cfg.eval_timeout or spec.comparison_eval_timeout
    return dataclasses.replace(spec, subsample_eval_timeout=t, comparison_eval_timeout=t)


def phase_estimate(cfg: AuditConfig, bench: str, tasks: list[EvalTask], store: AuditStore) -> dict[str, Any]:
    """Wall-clock estimate for a task list: the store's measured s/row once
    anything ran, else the audited runs' historical held-out draw rate."""
    from .estimate import historical_seconds_per_row

    rows = sum(len(t.row_ids) for t in tasks)
    spr = store.per_row_latency()
    source = "store"
    if not spr:
        spr, n = historical_seconds_per_row(cfg, bench)
        source = f"historical holdout draws (n={n})" if spr else "default"
        spr = spr or DEFAULT_SECONDS_PER_ROW
    seconds = rows * spr + len(tasks) * CALL_OVERHEAD_SECONDS
    return {"calls": len(tasks), "rows": rows, "seconds_per_row": spr, "seconds_per_row_source": source,
            "hours_serial": seconds / 3600.0, "hours_at_workers": seconds / 3600.0 / max(1, cfg.workers)}


def run_phase(
    cfg: AuditConfig, bench: str, spec: BenchSpec, mat: dict[str, Any], plan_fn: Callable[[AuditStore], list[EvalTask]],
    store: AuditStore, phase: str, *, dry_run: bool = False, frozen_testset_sha256: str | None = None, log=print,
) -> PhaseReport:
    rep = PhaseReport(phase=phase, dry_run=dry_run)
    ecfg = cfg.evaluator_cfg()
    espec = eval_spec(cfg, spec)
    tasks = plan_fn(store)
    rep.planned_calls, rep.planned_rows = len(tasks), sum(len(t.row_ids) for t in tasks)
    rep.estimate = phase_estimate(cfg, bench, tasks, store)
    log(f"[{bench}] {phase}: {rep.planned_calls} run_eval call(s), {rep.planned_rows} row-evals to go "
        f"(~{rep.estimate['hours_serial']:.1f} h serial at {rep.estimate['seconds_per_row']:.1f} s/row wall, "
        f"{rep.estimate['seconds_per_row_source']})")
    if dry_run or not tasks:
        return rep
    for attempt in (1, 2):
        if not tasks:
            break
        rep.attempts = attempt
        r = execute_tasks(ecfg, bench, espec, mat, tasks, store, log=log)
        rep.executed_calls += r.tasks
        rep.identical_pairs.extend(r.identical_pairs)
        for err in r.errors:
            rep.errors.append(f"attempt {attempt}: {err}")
        for t in tasks:
            store.log_attempt(phase=phase, tree=t.tree, draw=t.draw, attempt=attempt, rows=len(t.row_ids),
                              ok=not store.missing_rows(t.job_id, t.tree, phase, t.row_ids, t.draw))
        tasks = plan_fn(store)
        if tasks and attempt == 1:
            log(f"[{bench}] {phase}: retrying {len(tasks)} call(s) that returned no rows")
    for t in tasks:
        rep.unresolved.append({"tree": t.tree, "draw": t.draw, "rows": list(t.row_ids), "job_id": t.job_id})
        log(f"  UNRESOLVED {phase} {t.tree[:8]} d{t.draw}: {len(t.row_ids)} rows after 2 attempts")
    _fill_failed_rows(cfg, ecfg, bench, espec, mat, plan_fn, store, phase, rep, log)
    if frozen_testset_sha256:
        bad = {h for h in store.testset_hashes() if h != frozen_testset_sha256}
        if bad:
            raise RuntimeError(f"{bench}: evals recorded testset sha256 {sorted(bad)} != frozen {frozen_testset_sha256[:12]}")
    return rep


def _fill_failed_rows(cfg, ecfg, bench, espec, mat, plan_fn, store: AuditStore, phase: str, rep: PhaseReport, log) -> None:
    """One extra run of the rows the evaluator flagged failed in this phase."""
    by_tree = {c["tree"]: c for cs in mat["candidates"].values() for c in cs if c.get("tree")}
    fill_purpose = f"{phase}_fill"
    fill_tasks: list[EvalTask] = []
    keys = {(r["job_id"], r["tree"], r["draw"]) for r in store.rows() if r.get("purpose") == phase}
    for job_id, tree, draw in sorted(keys):
        failed = store.failed_rows(job_id, tree, phase, draw)
        # only rows never filled before (a resume must not re-spend on the same failure)
        failed = [r for r in failed if store.record(job_id, tree, fill_purpose, r, draw) is None]
        if failed and tree in by_tree:
            fill_tasks.append(task_for(by_tree[tree], failed, draw, fill_purpose))
    if fill_tasks:
        n = sum(len(t.row_ids) for t in fill_tasks)
        log(f"[{bench}] {phase}: fill pass — re-running {n} failed row(s) in {len(fill_tasks)} call(s)")
        execute_tasks(ecfg, bench, espec, mat, fill_tasks, store, log=log)
        for t in fill_tasks:
            rows = []
            for r in t.row_ids:
                rec = store.record(t.job_id, t.tree, fill_purpose, r, t.draw)
                if rec is not None:
                    rows.append({**rec, "purpose": phase})
            if rows:
                rep.filled_rows += store.append_fill({"eval_id": f"fill_{t.tree[:8]}_d{t.draw}", "purpose": phase,
                                                      "fill": True, "job_id": t.job_id, "tree": t.tree, "draw": t.draw,
                                                      "n_rows": len(rows), "ts": time.time()}, rows)
    for job_id, tree, draw in sorted(keys):
        still = store.failed_rows(job_id, tree, phase, draw)
        if still:
            rep.still_failed[tree] = rep.still_failed.get(tree, 0) + len(still)
            log(f"  WARNING {phase} {tree[:8]} d{draw}: {len(still)} row(s) still failed after the fill; "
                "kept at the evaluator's score (0.0) and counted")


def cross_phase_guard(store: AuditStore, pairs: Iterable[tuple[str, str, str, int, str, int]], *,
                      log=print) -> list[dict]:
    """`pairs` = (job_id, tree, phase_a, draw_a, phase_b, draw_b); compares the
    two evals' trace dirs. Warn-only: findings are returned for the winners
    file, never raised (caching has always been off in the task repos)."""
    from src.engine.evaluator.holdout import samples_are_identical

    evals = store.evals()

    def _trace(job_id, tree, phase, draw):
        for e in reversed(evals):
            if e.get("job_id") == job_id and e.get("tree") == tree and e.get("purpose") == phase and e.get("draw") == draw and e.get("trace_dir"):
                return e["trace_dir"]
        return None

    findings = []
    for job_id, tree, pa, da, pb, db in pairs:
        a, b = _trace(job_id, tree, pa, da), _trace(job_id, tree, pb, db)
        if not a or not b:
            continue
        try:
            same = samples_are_identical([a, b])
        except Exception as exc:  # noqa: BLE001
            findings.append({"tree": tree, "pair": f"{pa} d{da} / {pb} d{db}", "error": str(exc)[:120]})
            continue
        if same:
            findings.append({"tree": tree, "pair": f"{pa} d{da} / {pb} d{db}", "identical": True})
            log(f"  WARNING {tree[:8]}: {pa} d{da} and {pb} d{db} have byte-identical outputs — "
                "expected by chance on tiny label-output draws; a response cache only if it repeats across many rows")
    return findings


__all__ = ["PhaseReport", "cross_phase_guard", "eval_spec", "phase_estimate", "run_phase"]
