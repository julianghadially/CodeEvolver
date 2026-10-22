"""Copyright © 2026 440 Labs LLC

Stage 5 — metrics and outputs.

Per (bench, arm): `results/<bench>-<arm>.json` (identities, H, portfolio,
winners, every raw repetition with phase / draw / eval id / trace dir /
failed / fill flags, the metrics, and the HISTORICAL final score clearly
labelled) and `results/<bench>-<arm>-rows.csv` for Excel. Across
benchmarks: `results/summary.{csv,md,json}` with one row per (bench, arm)
and the evolution-minus-freeform difference in B.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from . import metrics as M
from .config import AuditConfig, arm_path, audit_paths
from .planner import PHASE_CONFIRM, PHASE_FINAL, PHASE_SELECT, confirm_draws, phase_matrix, select_draws
from .store import AuditStore

PHASES = ("screen", "select", "confirm", "final")


def _raw_repetitions(store: AuditStore, job_id: str, trees: list[str]) -> list[dict]:
    out = []
    trace_by_eval = {e.get("eval_id"): e.get("trace_dir") for e in store.evals()}
    for r in store.rows():
        if r.get("job_id") != job_id or r.get("tree") not in trees or r.get("purpose") not in PHASES:
            continue
        out.append({"tree": r["tree"], "phase": r["purpose"], "draw": r["draw"], "row": r["row_id"], "score": r.get("score"),
                    "failed": bool(r.get("failed")), "fill": bool(r.get("fill")), "eval_id": r.get("eval_id"),
                    "trace_dir": trace_by_eval.get(r.get("eval_id")), "ts": r.get("ts")})
    out.sort(key=lambda x: (x["phase"], x["tree"], x["draw"], x["row"]))
    return out


def build_results(cfg: AuditConfig, bench: str, arm: str, selection: dict[str, Any], difficult: dict[str, Any],
                  portfolio: dict[str, Any], winners: dict[str, Any], store: AuditStore, *, log=print) -> dict[str, Any]:
    H = list(winners["H"])
    job_id = winners["job_id"]
    final_tree = winners["final_tree"]
    cdraws = confirm_draws(cfg.reps_select, cfg.reps_confirm)
    win = {r: winners["winners"][str(r)]["tree"] for r in H}
    confirm: dict[int, list[float]] = {}
    for tree, rows in winners["rows_by_winner"].items():
        mat = phase_matrix(store, job_id, tree, PHASE_CONFIRM, rows, cdraws)
        for r in rows:
            confirm[r] = mat.get(r, [])
    final_rows = [r for r in H if win[r] != final_tree]
    final = phase_matrix(store, job_id, final_tree, PHASE_FINAL, final_rows, cdraws)
    select_means = {t: {int(r): v for r, v in s.items()} for t, s in winners["select_means"].items()}
    m = M.confirmation_metrics(H, win, confirm, final, final_tree, select_means, cfg.reps_confirm)
    sel_arm = selection["arms"][arm]
    n_failed = {ph: sum(len(store.failed_rows(job_id, c["tree"], ph, d))
                        for c in portfolio["portfolio"]
                        for d in ({"select": select_draws(cfg.reps_select), "confirm": cdraws, "final": cdraws, "screen": [0]}[ph]))
                for ph in PHASES}
    trees = [c["tree"] for c in portfolio["portfolio"]]
    res = {
        "bench": bench, "arm": arm, "run_id": winners["run_id"], "job_id": job_id, "reported_at": time.time(),
        "historical": {"final_test_score": sel_arm.get("final_test_score"), "final_test_scores": sel_arm.get("final_test_scores"),
                       "score_source": sel_arm.get("score_source"), "seed_test_score": sel_arm.get("seed_test_score"),
                       "label": "historical run record, not an audit measurement"},
        "difficult": {"n_H": len(H), "label": difficult.get("label"), "score_type": difficult.get("score_type"),
                      "testset_sha256": difficult.get("testset_sha256"), "n_seed_observations": difficult.get("n_seed_observations")},
        "portfolio": portfolio["portfolio"], "n_candidates_screened": portfolio.get("n_screened"),
        "unstable": portfolio.get("unstable", []), "final_tree": final_tree, "final_forced": portfolio.get("final_forced"),
        "candidate_H_means_selection": {t: winners["overall_means"][t] for t in trees},
        "winners": winners["winners"], "rows_by_winner": winners["rows_by_winner"],
        "metrics": m, "n_failed_rows_by_phase": n_failed, "reps_select": cfg.reps_select, "reps_confirm": cfg.reps_confirm,
        "raw": _raw_repetitions(store, job_id, trees),
    }
    log(f"  {bench}/{arm}: B={m['B']:.4f} (SE {m['SE_B']:.4f})  F={m['F']:.4f}  G={m['G']:+.4f} (SE {m['SE_G']:.4f})  "
        f"drop={m.get('selection_to_confirmation_drop', float('nan')):+.4f}  |H|={len(H)}")
    return res


def write_results(cfg: AuditConfig, bench: str, arm: str, res: dict[str, Any], difficult: dict[str, Any],
                  winners: dict[str, Any], store: AuditStore) -> tuple[Path, Path]:
    path = arm_path(cfg, "results", bench, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    csv_path = arm_path(cfg, "rows_csv", bench, arm)
    _write_rows_csv(csv_path, cfg, res, difficult, winners, store)
    return path, csv_path


def _write_rows_csv(path: Path, cfg: AuditConfig, res: dict[str, Any], difficult: dict[str, Any],
                    winners: dict[str, Any], store: AuditStore) -> None:
    H = list(winners["H"])
    trees = [c["tree"] for c in res["portfolio"]]
    job_id = res["job_id"]
    final_tree = res["final_tree"]
    cdraws = confirm_draws(cfg.reps_select, cfg.reps_confirm)
    R = cfg.reps_confirm
    header = (["row", "seed_estimate", "n_seed_obs"] + [f"sel_mean_c{i + 1}" for i in range(len(trees))]
              + ["winner", "winner_sel_mean"] + [f"conf_r{k + 1}" for k in range(R)] + [f"final_r{k + 1}" for k in range(R)]
              + ["final_is_winner", "n_failed_obs"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# {res['bench']} / {res['arm']} / {res['run_id']}; candidates: "
                    + "; ".join(f"c{i + 1}={c['tree'][:12]} sha {c['sha'][:8]} iters {c['iterations']}{' FINAL' if c['is_final'] else ''}"
                                for i, c in enumerate(res["portfolio"]))])
        w.writerow(header)
        for r in H:
            wn = winners["winners"][str(r)]
            row_info = difficult["rows"].get(str(r), {})
            conf = phase_matrix(store, job_id, wn["tree"], PHASE_CONFIRM, [r], cdraws).get(r, [])
            if wn["tree"] == final_tree:
                fin = conf
            else:
                fin = phase_matrix(store, job_id, final_tree, PHASE_FINAL, [r], cdraws).get(r, [])
            nf = sum(1 for x in store.rows() if x.get("job_id") == job_id and x.get("row_id") == r
                     and x.get("tree") in trees and x.get("purpose") in PHASES and x.get("failed"))
            w.writerow([r, row_info.get("mean"), row_info.get("n")]
                       + [winners["select_means"][t].get(str(r)) for t in trees]
                       + [f"c{trees.index(wn['tree']) + 1}", wn["mean"]]
                       + list(conf) + [None] * (R - len(conf)) + list(fin) + [None] * (R - len(fin))
                       + [int(wn["tree"] == final_tree), nf])


def load_results(cfg: AuditConfig) -> list[dict[str, Any]]:
    d = audit_paths(cfg, "_")["results_dir"]
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        if p.name.startswith("summary"):
            continue
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


SUMMARY_COLUMNS = ["bench", "arm", "run_id", "n_candidates_screened", "n_unstable", "n_H", "label",
                   "F_final_on_H", "sel_mean_c1", "sel_mean_c2", "sel_mean_c3", "sel_mean_c4", "sel_mean_c5",
                   "B_best_of_population", "SE_B", "G_gap", "SE_G", "CI95_G_lo", "CI95_G_hi", "sum_B", "sum_F",
                   "selection_to_confirmation_drop", "historical_final_test_score", "historical_score_source",
                   "evo_minus_freeform_B", "SE_diff", "zero_variance_flag", "n_failed_rows", "not_applicable"]


def summary_rows(results: list[dict[str, Any]], not_applicable: dict[str, str] | None = None) -> list[dict[str, Any]]:
    by = {(r["bench"], r["arm"]): r for r in results}
    rows = []
    for bench in sorted({b for b, _ in by} | set(not_applicable or {})):
        if not_applicable and bench in not_applicable and not any(b == bench for b, _ in by):
            rows.append({"bench": bench, "arm": "-", "not_applicable": not_applicable[bench]})
            continue
        evo, ff = by.get((bench, "pareto_evolution")), by.get((bench, "freeform"))
        diff = M.cross_arm(evo["metrics"], ff["metrics"]) if evo and ff else None
        for arm in ("freeform", "pareto_evolution"):
            r = by.get((bench, arm))
            if not r:
                continue
            m = r["metrics"]
            sel = [r["candidate_H_means_selection"][c["tree"]] for c in r["portfolio"]]
            row = {"bench": bench, "arm": arm, "run_id": r["run_id"], "n_candidates_screened": r["n_candidates_screened"],
                   "n_unstable": len(r.get("unstable", [])), "n_H": m["n_H"], "label": r["difficult"]["label"],
                   "F_final_on_H": m["F"], "B_best_of_population": m["B"], "SE_B": m["SE_B"], "G_gap": m["G"],
                   "SE_G": m["SE_G"], "CI95_G_lo": m["CI95_G"][0], "CI95_G_hi": m["CI95_G"][1],
                   "sum_B": m["sum_B"], "sum_F": m["sum_F"],
                   "selection_to_confirmation_drop": m.get("selection_to_confirmation_drop"),
                   "historical_final_test_score": r["historical"]["final_test_score"],
                   "historical_score_source": r["historical"]["score_source"],
                   "evo_minus_freeform_B": diff["evo_minus_freeform_B"] if diff else None,
                   "SE_diff": diff["SE"] if diff else None,
                   "zero_variance_flag": int(m["degenerate_zero_variance_B"] or m["degenerate_zero_variance_G"]),
                   "n_failed_rows": sum(r["n_failed_rows_by_phase"].values()), "not_applicable": ""}
            for i in range(5):
                row[f"sel_mean_c{i + 1}"] = sel[i] if i < len(sel) else None
            rows.append(row)
    return rows


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_summary(cfg: AuditConfig, rows: list[dict[str, Any]]) -> dict[str, Path]:
    p = audit_paths(cfg, "_")
    p["results_dir"].mkdir(parents=True, exist_ok=True)
    with open(p["summary_csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in SUMMARY_COLUMNS})
    p["summary_json"].write_text(json.dumps(rows, indent=2))
    md = ["# Best-of-population audit — summary", "",
          "Independently verified best-of-population performance on the frozen difficult set H per benchmark. "
          "B = mean over 5 confirmation rounds of each row's frozen winner; F = final program on the same rows; "
          "G = B - F (paired); SE from the 5 rounds (n-1 = 4); evolution - freeform in B with independent SEs. "
          "Standard errors are conditional on the fixed rows, run, portfolio and winners. "
          "The selection-to-confirmation drop is a noisy diagnostic of selection optimism, not a correction.", "",
          "| bench | arm | run | cands | |H| | label | F | c1 | c2 | c3 | c4 | c5 | B | SE(B) | G | SE(G) | 95% CI(G) | sum B | sum F | drop | hist. final | evo-ff B | SE |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("not_applicable"):
            md.append(f"| {r['bench']} | - | not applicable: {r['not_applicable']} |" + " |" * 20)
            continue
        md.append("| " + " | ".join([
            r["bench"], r["arm"], r["run_id"], str(r["n_candidates_screened"]), str(r["n_H"]), r["label"], _fmt(r["F_final_on_H"]),
            *[_fmt(r.get(f"sel_mean_c{i + 1}")) for i in range(5)], _fmt(r["B_best_of_population"]), _fmt(r["SE_B"]),
            _fmt(r["G_gap"]), _fmt(r["SE_G"]), f"[{_fmt(r['CI95_G_lo'])}, {_fmt(r['CI95_G_hi'])}]", _fmt(r["sum_B"]), _fmt(r["sum_F"]),
            _fmt(r["selection_to_confirmation_drop"]), f"{_fmt(r['historical_final_test_score'])} ({r['historical_score_source']})",
            _fmt(r["evo_minus_freeform_B"]), _fmt(r["SE_diff"]),
        ]) + " |")
    flagged = [r for r in rows if r.get("zero_variance_flag")]
    if flagged:
        md += ["", "Zero-variance (degenerate) confirmation estimates: " + ", ".join(f"{r['bench']}/{r['arm']}" for r in flagged)]
    p["summary_md"].write_text("\n".join(md) + "\n")
    return {"csv": p["summary_csv"], "md": p["summary_md"], "json": p["summary_json"]}


__all__ = ["SUMMARY_COLUMNS", "build_results", "load_results", "summary_rows", "write_results", "write_summary"]
