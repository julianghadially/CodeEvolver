"""Copyright © 2026 440 Labs LLC

Stage 2b — the frozen top-5 portfolio from the single screening draw.

Observed Pareto set: per difficult row the candidates tied at the row max
(`pareto_front`), then GEPA's union-cover pruning (`prune_dominated`, the same
rule the pareto_evolution arm uses in-run) with the aggregate H mean as the
tie order. Observed dominance is a screening heuristic, not proof of true
dominance under stochastic evaluation. Then the greedy top-k, the
final program forced in (replacing the last added), and a backfill by
aggregate score when fewer than k survive. Everything is written to
`portfolio/<bench>-<arm>.json` and never recomputed unless `--refreeze`.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.engine.optimizer.v3.pareto_front import pareto_front, prune_dominated

from . import metrics as M
from .config import AuditConfig, arm_path
from .planner import PHASE_SCREEN
from .store import AuditStore

SCREEN_RULE = (
    "one screening draw per candidate on H; observed Pareto front = per-row ties at the row max; "
    "prune_dominated (GEPA union cover, ascending aggregate H mean); greedy top-k: top mean first, then "
    "argmax additional row-wise gain sum(max(0, s_c - portfolio_max)), ties -> higher mean, then tree; "
    "final program forced in by replacing the last added; backfill by aggregate H mean when < k survive"
)


def build_portfolio(cfg: AuditConfig, bench: str, arm: str, cand_doc: dict[str, Any], H: list[int],
                    store: AuditStore, *, log=print) -> dict[str, Any]:
    job_id = cand_doc["job_id"]
    final_tree = cand_doc["final_tree"]
    scores: dict[str, dict[int, float]] = {}
    unstable: list[dict] = [{"tree": c.get("tree"), "sha": c.get("sha"), "reason": c["unstable"]}
                            for c in cand_doc["candidates"] if c.get("unstable")]
    n_failed: dict[str, int] = {}
    for c in cand_doc["candidates"]:
        if not c.get("tree") or c.get("unstable"):
            continue
        s = store.scores(job_id, c["tree"], PHASE_SCREEN, 0)
        missing = [r for r in H if r not in s]
        if missing:
            reason = f"{len(missing)} of {len(H)} difficult rows missing after 2 attempts (infrastructure failure)"
            if c["tree"] == final_tree:
                raise RuntimeError(f"{bench}/{arm}: the final program is unstable: {reason}")
            unstable.append({"tree": c["tree"], "sha": c["sha"], "reason": reason})
            log(f"  DROPPED {arm} {c['tree'][:8]} (iterations {c.get('iterations')}): {reason}")
            continue
        scores[c["tree"]] = {r: s[r] for r in H}
        n_failed[c["tree"]] = len(store.failed_rows(job_id, c["tree"], PHASE_SCREEN, 0))
    if final_tree not in scores:
        raise RuntimeError(f"{bench}/{arm}: the final program has no screening scores")
    means = M.aggregate_means(scores, H)
    front = pareto_front({t: {str(r): v for r, v in s.items()} for t, s in scores.items()})
    _, pruned = prune_dominated(front, keys=set(scores), scores=means)
    survivors = sorted(set(scores) - pruned)
    k = cfg.top_k
    chosen, steps = M.greedy_top_k(scores, survivors, H, k)
    backfilled: list[str] = []
    if len(chosen) < k:
        chosen, backfilled = M.backfill(chosen, scores, sorted(pruned), H, k)
    chosen, replaced = M.force_final(chosen, final_tree)
    by_tree = {c["tree"]: c for c in cand_doc["candidates"] if c.get("tree")}
    portfolio = [{"tree": t, "sha": by_tree[t]["sha"], "iterations": by_tree[t].get("iterations", [by_tree[t].get("iteration")]),
                  "is_final": t == final_tree, "reqhash": by_tree[t].get("reqhash"), "job_id": job_id, "arm": arm,
                  "screen_mean": means[t], "on_front": t not in pruned, "n_failed_rows_screen": n_failed.get(t, 0)}
                 for t in chosen]
    doc = {
        "bench": bench, "arm": arm, "run_id": cand_doc["run_id"], "job_id": job_id, "frozen_at": time.time(),
        "H": list(H), "n_H": len(H), "k": k, "rule": SCREEN_RULE,
        "n_screened": len(scores), "screen": {t: {str(r): v for r, v in s.items()} for t, s in scores.items()},
        "screen_means": means, "front_rows": {r: sorted(ts) for r, ts in front.items()},
        "pruned": sorted(pruned), "survivors": survivors, "greedy_steps": steps,
        "portfolio": portfolio, "final_tree": final_tree, "final_forced": replaced is not None,
        "replaced": replaced, "backfilled": backfilled, "unstable": unstable,
        "n_failed_rows_screen": n_failed,
    }
    log(f"  {bench}/{arm}: screened {len(scores)} candidates, {len(survivors)} on the observed front, "
        f"portfolio = {[t[:8] for t in chosen]}" + (f" (final forced in, replaced {replaced[:8]})" if replaced else "")
        + (f" (backfilled {len(backfilled)})" if backfilled else ""))
    return doc


def load_portfolio(cfg: AuditConfig, bench: str, arm: str) -> dict[str, Any]:
    path = arm_path(cfg, "portfolio", bench, arm)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `screen --bench {bench}` first")
    return json.loads(path.read_text())


def write_portfolio(cfg: AuditConfig, bench: str, arm: str, doc: dict[str, Any]) -> None:
    path = arm_path(cfg, "portfolio", bench, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))


__all__ = ["SCREEN_RULE", "build_portfolio", "load_portfolio", "write_portfolio"]
