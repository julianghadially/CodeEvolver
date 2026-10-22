"""Copyright © 2026 440 Labs LLC

Plan mode — evaluation counts and a wall-clock verdict before spending.

Rows per arm = |H| * (N_candidates + reps_select * k + reps_confirm)
             + reps_confirm * (|H| - |rows the final wins|)   (upper bound: |H|)
Seconds per row: the store's measured median once any eval has run; before
that the median wall time of the audited runs' own final held-out draws
(`row_scores.jsonl` ts deltas between `holdout`, `holdout_2`, ... of the
same run, divided by n_rows — the only latency record a run leaves).
"""

from __future__ import annotations

from typing import Any

from .catalog import RunRecord, discover_runs
from .config import AuditConfig
from .evaluate import CALL_OVERHEAD_SECONDS, DEFAULT_SECONDS_PER_ROW
from .records import final_holdout_lines, holdout_draw_index, void_reason
from .store import AuditStore

DAY = 86400.0


def historical_seconds_per_row(cfg: AuditConfig, bench: str) -> tuple[float | None, int]:
    """Median (s/row wall) over consecutive final held-out draws of every
    non-void run of the bench; (None, 0) when no run has two draws."""
    samples: list[float] = []
    for run in discover_runs(cfg):
        if run.bench != bench or void_reason(run.run_id):
            continue
        draws = sorted(
            ((holdout_draw_index(l.get("suffix")), l) for l in final_holdout_lines(run)
             if holdout_draw_index(l.get("suffix")) is not None), key=lambda x: x[0]
        )
        for (i, a), (j, b) in zip(draws, draws[1:]):
            if j == i + 1 and a.get("ts") and b.get("ts") and b.get("n_rows"):
                dt = float(b["ts"]) - float(a["ts"])
                if dt > 0:
                    samples.append(dt / float(b["n_rows"]))
    if not samples:
        return None, 0
    samples.sort()
    return samples[len(samples) // 2], len(samples)


def plan_counts(n_H: int, n_candidates: int, *, k: int, reps_select: int, reps_confirm: int,
                n_final_won: int | None = None, done_rows: int = 0) -> dict[str, int]:
    final_rows = n_H if n_final_won is None else max(0, n_H - n_final_won)
    screen = n_H * n_candidates
    select = n_H * k * reps_select
    confirm = n_H * reps_confirm
    final = final_rows * reps_confirm
    total = screen + select + confirm + final
    return {"screen": screen, "select": select, "confirm": confirm, "final": final,
            "total": total, "done": done_rows, "remaining": max(0, total - done_rows),
            "calls": n_candidates + k * reps_select + k * reps_confirm + reps_confirm}


def verdict(hours: float) -> str:
    days = hours / 24.0
    if days <= 1:
        return "<= 1 day"
    if days <= 3:
        return "<= 3 days"
    if days <= 5:
        return "<= 5 days"
    return f"> 5 days ({days:.1f} d)"


def estimate_bench(cfg: AuditConfig, bench: str, difficult: dict[str, Any] | None, per_arm_candidates: dict[str, int],
                   store: AuditStore | None, seed_test_score: float | None, testset_size: int | None,
                   n_final_won: dict[str, int] | None = None, num_threads: int = 1) -> dict[str, Any]:
    assumed = difficult is None or difficult.get("not_applicable")
    if difficult and not difficult.get("not_applicable"):
        n_H = int(difficult["n_H"])
    elif seed_test_score is not None and testset_size:
        n_H = int(round((1.0 - float(seed_test_score)) * testset_size))
    else:
        n_H = 0
    spr_store = store.per_row_latency() if store else None
    spr_hist, n_hist = historical_seconds_per_row(cfg, bench)
    spr = spr_store or spr_hist or DEFAULT_SECONDS_PER_ROW
    spr_source = "store" if spr_store else ("historical holdout draws" if spr_hist else "default")
    arms = {}
    total_rows = 0
    total_calls = 0
    for arm, n_c in per_arm_candidates.items():
        done = 0
        if store:
            done = sum(1 for r in store.rows() if r.get("arm") == arm and r.get("purpose") in ("screen", "select", "confirm", "final"))
        counts = plan_counts(n_H, n_c, k=cfg.top_k, reps_select=cfg.reps_select, reps_confirm=cfg.reps_confirm,
                             n_final_won=(n_final_won or {}).get(arm), done_rows=done)
        arms[arm] = counts
        total_rows += counts["remaining"]
        total_calls += counts["calls"]
    # spr is WALL seconds per row at the run's thread count (the historical draws ran at spec.num_threads)
    serial_h = (total_rows * spr + total_calls * CALL_OVERHEAD_SECONDS) / 3600.0
    workers = max(1, cfg.workers)
    return {
        "bench": bench, "n_H": n_H, "n_H_assumed": assumed, "seconds_per_row": spr, "seconds_per_row_source": spr_source,
        "historical_samples": n_hist, "arms": arms, "rows_remaining": total_rows, "calls": total_calls,
        "hours_serial": serial_h, "workers": workers, "hours_at_workers": serial_h / workers,
        "verdict_serial": verdict(serial_h), "verdict_at_workers": verdict(serial_h / workers),
        "concurrent_provider_calls": workers * num_threads,
    }


def format_estimate(e: dict[str, Any]) -> str:
    lines = [f"[{e['bench']}] |H| = {e['n_H']}{' (ASSUMED from the seed score — run `difficult`)' if e['n_H_assumed'] else ''}"]
    for arm, c in e["arms"].items():
        lines.append(f"  {arm:18s} screen {c['screen']:6d}  select {c['select']:6d}  confirm {c['confirm']:5d}  "
                     f"final <= {c['final']:5d}  | total {c['total']:6d} rows, done {c['done']}, remaining {c['remaining']}")
    lines.append(f"  {e['rows_remaining']} row-evals in ~{e['calls']} run_eval calls; {e['seconds_per_row']:.1f} s/row wall "
                 f"({e['seconds_per_row_source']}, n={e['historical_samples']})")
    lines.append(f"  serial: {e['hours_serial']:.1f} h -> {e['verdict_serial']};  at {e['workers']} worker(s): "
                 f"{e['hours_at_workers']:.1f} h -> {e['verdict_at_workers']}")
    lines.append(f"  caveat: {e['workers']} worker(s) x num_threads = ~{e['concurrent_provider_calls']} concurrent provider "
                 "calls per benchmark; latency rises with load, so the worker estimate is optimistic")
    return "\n".join(lines)


__all__ = ["estimate_bench", "format_estimate", "historical_seconds_per_row", "plan_counts", "verdict"]
