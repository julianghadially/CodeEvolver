"""Copyright © 2026 440 Labs LLC

Stage 1 — the frozen difficult set H per benchmark.

Pools the SEED program's held-out per-row scores across every collected run
of the benchmark (all arms, ~3 seeds x 5 arms) whose settings are compatible:
same `testset_sha256` (row ids are positional) and the same `settings.json`
`models` block. Void directories are excluded unless `--include-void-seed`.
Binary benchmarks (every value 0 or one positive m — hover's 0.998) estimate
a success PROBABILITY per row; fractional benchmarks a mean normalized
EXPECTED SCORE. H = rows strictly below 0.5. Rows the evaluator flagged as
failed are recorded as failed observations and excluded from the estimate.
Historical seed data determine H only; they are never confirmation data.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import metrics as M
from .catalog import RunRecord, discover_runs
from .config import AuditConfig, audit_paths
from .records import read_json, seed_holdout_line, settings_models, void_reason

EXPECTED_SEED_OBSERVATIONS = 15
SPARSE_N = 5
LARGE_TESTSET = 500


def _manifest(run: RunRecord) -> dict:
    if not run.history_dir:
        return {}
    return read_json(Path(run.history_dir) / "datasets.json") or {}


def build_difficult(cfg: AuditConfig, bench: str, *, log=print) -> dict[str, Any]:
    runs = sorted((r for r in discover_runs(cfg) if r.bench == bench), key=lambda r: r.run_id)
    used: list[dict] = []
    excluded: list[dict] = []
    obs_raw: dict[int, list[tuple[str, float]]] = {}
    failed_obs: dict[int, list[str]] = {}
    keys: Counter = Counter()
    per_run: list[tuple[RunRecord, dict, tuple, dict]] = []
    for run in runs:
        void = void_reason(run.run_id)
        if void and not cfg.include_void_seed:
            excluded.append({"run_id": run.run_id, "reason": void})
            continue
        line = seed_holdout_line(run)
        if line is None:
            excluded.append({"run_id": run.run_id, "reason": "no seed held-out row_scores line"})
            continue
        man = _manifest(run)
        tsha = man.get("testset_sha256") or run.testset_sha256
        key = (tsha, json.dumps(settings_models(run), sort_keys=True))
        keys[key] += 1
        per_run.append((run, line, key, man))
    if not per_run:
        return {"bench": bench, "not_applicable": True, "reason": "no seed held-out observations found",
                "runs_used": [], "runs_excluded": excluded, "H": [], "rows": {}}
    ref_key, _ = keys.most_common(1)[0]
    testset_size = None
    for run, line, key, man in per_run:
        if key != ref_key:
            excluded.append({"run_id": run.run_id,
                             "reason": f"incompatible settings (testset_sha256/models) {key[0][:8] if key[0] else None} vs reference {ref_key[0][:8] if ref_key[0] else None}"})
            continue
        testset_size = testset_size or man.get("testset_size")
        if man.get("testset_size") and line.get("n_rows") and line["n_rows"] != man["testset_size"]:
            log(f"  WARNING {run.run_id}: seed holdout line has n_rows={line['n_rows']} but testset_size={man['testset_size']}")
        failed = {str(x) for x in (line.get("failed_rows") or [])}
        for rid_s, score in (line.get("row_scores") or {}).items():
            rid = int(rid_s)
            if rid_s in failed or score is None:
                failed_obs.setdefault(rid, []).append(run.run_id)
                continue
            obs_raw.setdefault(rid, []).append((run.run_id, float(score)))
        used.append({"run_id": run.run_id, "job_id": run.job_id, "arm": run.arm, "seed": run.seed,
                     "eval_id": line.get("eval_id"), "ts": line.get("ts"), "n_rows": line.get("n_rows"),
                     "score": line.get("score"), "n_failed": len(failed)})
    all_values = [s for obs in obs_raw.values() for _, s in obs]
    score_type, max_score = M.classify_score_type(all_values)
    if score_type == "fractional" and max_score > 1.0 + 1e-9:
        raise RuntimeError(f"{bench}: fractional scores exceed 1.0 (max {max_score}); confirm normalization")
    label = "probability" if score_type == "binary" else "expected_score"
    norm: dict[int, list[float]] = {
        rid: [M.normalize_observation(s, score_type, max_score) for _, s in obs] for rid, obs in obs_raw.items()
    }
    H = M.difficult_set(norm, cfg.threshold)
    n_rows_total = testset_size or (max(list(obs_raw) + list(failed_obs)) + 1 if (obs_raw or failed_obs) else 0)
    rows = {}
    for rid in range(n_rows_total):
        o = norm.get(rid, [])
        rows[str(rid)] = {"mean": M.mean(o) if o else None, "n": len(o), "n_failed": len(failed_obs.get(rid, [])),
                          "obs": o, "in_H": rid in H}
    missing = [rid for rid in range(n_rows_total) if not norm.get(rid)]
    sparse = [rid for rid in range(n_rows_total) if 0 < len(norm.get(rid, [])) < SPARSE_N]
    out = {
        "bench": bench, "frozen_at": time.time(), "testset_sha256": ref_key[0], "testset_size": n_rows_total,
        "models": json.loads(ref_key[1]) if ref_key[1] else {},
        "score_type": score_type, "label": label, "max_score_observed": max_score, "threshold": cfg.threshold,
        "n_seed_observations": len(used), "expected_seed_observations": EXPECTED_SEED_OBSERVATIONS,
        "runs_used": used, "runs_excluded": excluded,
        "rows": rows, "H": H, "n_H": len(H), "missing_rows": missing, "sparse_rows": sparse,
        "not_applicable": len(H) == 0, "over_500_rows": n_rows_total > LARGE_TESTSET,
        "seed_mean_over_rows": M.mean(v["mean"] for v in rows.values() if v["mean"] is not None) if norm else None,
    }
    log(f"  {bench}: {len(used)} seed observations (expected ~{EXPECTED_SEED_OBSERVATIONS}), "
        f"{len(excluded)} runs excluded; testset {n_rows_total} rows, {score_type} ({label}); "
        f"|H| = {len(H)}; {len(missing)} rows unclassifiable, {len(sparse)} sparse (n<{SPARSE_N})"
        + (f"; WARNING testset > {LARGE_TESTSET} rows" if out["over_500_rows"] else ""))
    return out


def load_difficult(cfg: AuditConfig, bench: str) -> dict[str, Any]:
    path = audit_paths(cfg, bench)["difficult"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `difficult --bench {bench}` first")
    return json.loads(path.read_text())


def write_difficult(cfg: AuditConfig, bench: str, data: dict[str, Any]) -> Path:
    path = audit_paths(cfg, bench)["difficult"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


__all__ = ["build_difficult", "load_difficult", "write_difficult"]
