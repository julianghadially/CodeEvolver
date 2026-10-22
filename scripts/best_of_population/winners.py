"""Copyright © 2026 440 Labs LLC

Stage 3 — frozen row-to-candidate mapping from the selection draws.

Every portfolio candidate has `reps_select` fresh draws on H (distinct from
the screening draw). Per row the winner is the candidate with the highest
selection mean; ties -> the higher overall (all-of-H) selection mean, then
ascending tree. The mapping is written to `winners/<bench>-<arm>.json` before
confirmation starts and confirmation results never change it.
"""

from __future__ import annotations

import json
import time
from typing import Any

from . import metrics as M
from .config import AuditConfig, arm_path
from .planner import PHASE_SELECT, phase_matrix, select_draws
from .store import AuditStore


def build_winners(cfg: AuditConfig, bench: str, arm: str, portfolio_doc: dict[str, Any], store: AuditStore,
                  *, log=print) -> dict[str, Any]:
    H = list(portfolio_doc["H"])
    job_id = portfolio_doc["job_id"]
    draws = select_draws(cfg.reps_select)
    select: dict[str, dict[int, list[float]]] = {}
    n_failed: dict[str, int] = {}
    for c in portfolio_doc["portfolio"]:
        mat = phase_matrix(store, job_id, c["tree"], PHASE_SELECT, H, draws)
        short = [r for r in H if len(mat.get(r, [])) < cfg.reps_select]
        if short:
            raise RuntimeError(f"{bench}/{arm}: candidate {c['tree'][:8]} has fewer than {cfg.reps_select} "
                               f"selection draws on {len(short)} row(s); rerun `select` to resume")
        select[c["tree"]] = {r: mat[r] for r in H}
        n_failed[c["tree"]] = sum(len(store.failed_rows(job_id, c["tree"], PHASE_SELECT, d)) for d in draws)
    winners, select_means, overall = M.row_winners(select, H)
    rows_by_winner: dict[str, list[int]] = {}
    for r in H:
        rows_by_winner.setdefault(winners[r]["tree"], []).append(r)
    doc = {
        "bench": bench, "arm": arm, "run_id": portfolio_doc["run_id"], "job_id": job_id, "frozen_at": time.time(),
        "H": H, "reps_select": cfg.reps_select, "select_draws": draws,
        "portfolio": portfolio_doc["portfolio"], "final_tree": portfolio_doc["final_tree"],
        "select_scores": {t: {str(r): v for r, v in s.items()} for t, s in select.items()},
        "select_means": {t: {str(r): v for r, v in s.items()} for t, s in select_means.items()},
        "overall_means": overall,
        "winners": {str(r): winners[r] for r in H},
        "rows_by_winner": rows_by_winner,
        "n_rows_final_is_winner": len(rows_by_winner.get(portfolio_doc["final_tree"], [])),
        "n_tied_rows": sum(1 for r in H if winners[r]["tied"]),
        "n_failed_rows_select": n_failed,
        "rule": "per row: highest selection mean; ties -> higher overall selection mean, then ascending tree",
    }
    log(f"  {bench}/{arm}: winners frozen — " + ", ".join(f"{t[:8]}:{len(rs)} rows" for t, rs in rows_by_winner.items())
        + f"; final wins {doc['n_rows_final_is_winner']}/{len(H)}; {doc['n_tied_rows']} tied rows")
    return doc


def load_winners(cfg: AuditConfig, bench: str, arm: str) -> dict[str, Any]:
    path = arm_path(cfg, "winners", bench, arm)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `select --bench {bench}` first")
    return json.loads(path.read_text())


def write_winners(cfg: AuditConfig, bench: str, arm: str, doc: dict[str, Any]) -> None:
    path = arm_path(cfg, "winners", bench, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))


__all__ = ["build_winners", "load_winners", "write_winners"]
