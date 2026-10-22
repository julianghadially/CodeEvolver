"""Copyright © 2026 440 Labs LLC

Stage 0 — freeze the audited run per (benchmark, arm): the completed run with
the highest RECORDED final held-out score, using existing records only.

Record of a final score, in order: `status.json.final_test_score`; when that
is None (held-out draws timed out), `posthoc_holdout_rescore.json` beside the
run (an existing record written by a post-hoc rescore; its `program_sha`
must match the run's final sha). Void directories (results_tables'
replacement table) are never read. Ties -> the lower seed. The choice is
written to `selection/<bench>.json` BEFORE any evaluation and never
recomputed unless `--refreeze`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .catalog import RunRecord, discover_runs
from .config import AuditConfig, audit_paths
from .records import parse_run_id, posthoc_rescore, void_reason


def _score_of(run: RunRecord) -> tuple[float | None, str, list[float]]:
    if run.final_test_score is not None:
        return float(run.final_test_score), "status.json", list(run.final_test_scores)
    post = posthoc_rescore(run)
    if post and post.get("final_test_score") is not None:
        if post.get("program_sha") and run.final_sha and post["program_sha"] != run.final_sha:
            return None, f"posthoc_holdout_rescore.json program_sha {post['program_sha'][:8]} != final {run.final_sha[:8]}", []
        return float(post["final_test_score"]), "posthoc_holdout_rescore.json", list(post.get("final_test_scores") or [])
    return None, "no final_test_score recorded", []


def select_runs(cfg: AuditConfig, bench: str, *, log=print) -> dict[str, Any]:
    runs = [r for r in discover_runs(cfg) if r.bench == bench]
    out: dict[str, Any] = {"bench": bench, "frozen_at": time.time(), "runs_roots": list(cfg.runs_roots),
                           "arms": {}}
    for arm in cfg.arms:
        considered = []
        eligible = []
        for run in sorted(runs, key=lambda r: r.run_id):
            if run.arm != arm:
                continue
            parts = parse_run_id(run.run_id) or {}
            void = void_reason(run.run_id)
            score, source, draws = _score_of(run)
            reason = void
            if reason is None and run.status != "completed":
                reason = f"status {run.status}"
            elif reason is None and not run.bundle_path:
                reason = "no repo.bundle"
            elif reason is None and not run.final_sha:
                reason = "no final sha"
            elif reason is None and score is None:
                reason = source
            considered.append({"run_id": run.run_id, "job_id": run.job_id, "seed": parts.get("seed"),
                               "status": run.status, "score": score, "score_source": source,
                               "eligible": reason is None, "reason": reason})
            if reason is None:
                eligible.append((score, parts.get("seed", 10**6), run, source, draws))
        if not eligible:
            out["arms"][arm] = {"run_id": None, "considered": considered,
                                "note": "no eligible completed run with a recorded final score"}
            log(f"  {bench}/{arm}: NO eligible run")
            continue
        eligible.sort(key=lambda e: (-e[0], e[1]))
        score, _, run, source, draws = eligible[0]
        out["arms"][arm] = {
            "run_id": run.run_id, "job_id": run.job_id, "run_dir": run.artifacts_dir,
            "final_sha": run.final_sha, "final_branch": run.final_branch,
            "final_test_score": score, "final_test_scores": draws, "score_source": source,
            "seed_test_score": run.seed_test_score, "history_dir": run.history_dir,
            "bundle_path": run.bundle_path, "testset_sha256": run.testset_sha256,
            "considered": considered, "run": asdict(run),
        }
        log(f"  {bench}/{arm}: {run.run_id} final_test={score:.4f} ({source}); "
            f"{sum(c['eligible'] for c in considered)} eligible of {len(considered)} considered")
    return out


def load_selection(cfg: AuditConfig, bench: str) -> dict[str, Any]:
    path = audit_paths(cfg, bench)["selection"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `select-run --bench {bench}` first")
    return json.loads(path.read_text())


def write_selection(cfg: AuditConfig, bench: str, sel: dict[str, Any]) -> Path:
    path = audit_paths(cfg, bench)["selection"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sel, indent=2))
    return path


def selection_table(sel: dict[str, Any]) -> str:
    lines = [f"[{sel['bench']}] audited run per arm (best recorded final held-out score)"]
    for arm, entry in sel["arms"].items():
        lines.append(f"  {arm}:")
        for c in entry.get("considered", []):
            mark = "*" if entry.get("run_id") == c["run_id"] else " "
            sc = f"{c['score']:.4f}" if c["score"] is not None else "-"
            note = "" if c["eligible"] else f"  ({c['reason']})"
            lines.append(f"   {mark} {c['run_id']:40s} {sc:>8s}  {c['score_source']}{note}")
        if not entry.get("run_id"):
            lines.append(f"    -> {entry.get('note')}")
    return "\n".join(lines)


__all__ = ["load_selection", "select_runs", "selection_table", "write_selection"]
