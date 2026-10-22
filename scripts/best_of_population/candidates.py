"""Copyright © 2026 440 Labs LLC

Stage 2a — candidate programs of the audited run, materialized.

Candidate pool (same rule for both arms): the `close_sha` of every closed,
non-skipped iteration of the run — deduped by git tree — plus the final
program. For pareto_evolution this includes iterations the controller never
admitted and members it later pruned. Each run's `repo.bundle` is fetched
into `<out>/repos/<bench>.git` under `refs/jobs/<job_id>/*`; one uv venv is
built per distinct `requirements.txt`; the worktree pool is primed and the
held-out testset staged from the researcher's checkout. Its sha256 must equal
the frozen H's `testset_sha256` (row ids are positional) or nothing runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .catalog import RunRecord, candidates_from_run
from .config import AuditConfig, BenchSpec, arm_path, audit_paths
from .environment import ensure_venv, requirements_hash
from .materialize import (
    WorktreePool, anchor_ref, blob_sha256, ensure_bare_repo, fetch_run, file_at, resolve_sha, resolve_tree,
)


def _resolve_candidates(bare: Path, run: RunRecord, spec: BenchSpec, cands: list[dict], *, log) -> list[dict]:
    seed_ref = f"refs/jobs/{run.job_id}/heads/seed"
    seed_train = blob_sha256(bare, seed_ref, spec.trainset_path)
    seed_test = blob_sha256(bare, seed_ref, spec.testset_path) if spec.testset_path else None
    by_tree: dict[str, dict] = {}
    unresolved: list[dict] = []
    for c in cands:
        ref = anchor_ref(run.job_id, c["sha"])
        try:
            c["sha"] = resolve_sha(bare, ref)
            c["tree"] = resolve_tree(bare, ref)
        except RuntimeError as exc:
            c["tree"] = None
            c["unstable"] = f"unresolvable in bundle: {str(exc)[:160]}"
            unresolved.append(c)
            log(f"  WARNING {run.run_id} iteration {c['iteration']} sha {c['sha'][:8]}: {c['unstable']}")
            continue
        req = file_at(bare, c["sha"], "requirements.txt")
        c["requirements"] = req
        c["reqhash"] = requirements_hash(req)
        th = blob_sha256(bare, c["sha"], spec.trainset_path)
        c["trainset_drift"] = bool(seed_train and th and th != seed_train)
        tsh = blob_sha256(bare, c["sha"], spec.testset_path) if spec.testset_path else None
        c["testset_in_tree"] = bool(tsh)                       # the testset is amended out of every bundle
        c["testset_drift"] = bool(tsh and seed_test and tsh != seed_test)
        prev = by_tree.get(c["tree"])
        if prev is None:
            c["iterations"] = [c["iteration"]]
            by_tree[c["tree"]] = c
        else:
            prev["is_final"] = prev["is_final"] or c["is_final"]
            prev["iterations"].append(c["iteration"])
            prev.setdefault("duplicates", []).append({"iteration": c["iteration"], "sha": c["sha"]})
    return list(by_tree.values()) + unresolved


def materialize_arm(cfg: AuditConfig, bench: str, spec: BenchSpec, arm: str, sel_arm: dict[str, Any],
                    bare: Path, venvs: dict[str, str], *, explicit_shas: list[str] | None = None, log=print) -> dict[str, Any]:
    run = RunRecord(**sel_arm["run"])
    if not run.bundle_path:
        raise RuntimeError(f"{run.run_id}: no repo.bundle")
    n_refs = fetch_run(bare, Path(run.bundle_path), run.job_id)
    if explicit_shas:
        cands = [{"job_id": run.job_id, "bench": bench, "arm": arm, "iteration": -1, "branch": None, "sha": s,
                  "source": "explicit", "is_final": (s == run.final_sha), "is_seed": False} for s in explicit_shas]
        if run.final_sha and run.final_sha not in explicit_shas:
            cands.append({"job_id": run.job_id, "bench": bench, "arm": arm, "iteration": -1, "branch": None,
                          "sha": run.final_sha, "source": "final", "is_final": True, "is_seed": False})
    else:
        cands = [asdict(c) for c in candidates_from_run(cfg, run)]
    cands = _resolve_candidates(bare, run, spec, cands, log=log)
    paths = audit_paths(cfg, bench)
    for c in cands:
        if not c.get("tree"):
            continue
        if c["reqhash"] not in venvs:
            try:
                venvs[c["reqhash"]] = str(ensure_venv(paths["venvs"], bench, c["reqhash"], c.get("requirements"),
                                                      python_version=spec.python_version))
            except Exception as exc:  # noqa: BLE001
                c["unstable"] = f"venv build failed: {str(exc)[:160]}"
                log(f"  WARNING {run.run_id} tree {c['tree'][:8]}: {c['unstable']}")
        c.pop("requirements", None)
    finals = [c for c in cands if c.get("is_final") and c.get("tree")]
    if not finals:
        raise RuntimeError(f"{run.run_id}: the final program {run.final_sha} could not be resolved in the bundle")
    if finals[0].get("unstable"):
        raise RuntimeError(f"{run.run_id}: the final program is unusable: {finals[0]['unstable']}")
    doc = {
        "bench": bench, "arm": arm, "run_id": run.run_id, "job_id": run.job_id, "fetched_refs": n_refs,
        "final_sha": finals[0]["sha"], "final_tree": finals[0]["tree"],
        "pool_rule": "every closed non-skipped iteration's close_sha, deduped by git tree, plus the final program"
                     if not explicit_shas else "explicit --candidates-file list plus the final program",
        "n_candidates": sum(1 for c in cands if c.get("tree") and not c.get("unstable")),
        "candidates": cands, "materialized_at": time.time(),
    }
    path = arm_path(cfg, "candidates", bench, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    log(f"  {bench}/{arm}: {run.run_id} -> {doc['n_candidates']} candidate trees "
        f"({len([c for c in cands if c.get('unstable')])} unusable), final tree {doc['final_tree'][:8]}")
    return doc


def materialize_bench(cfg: AuditConfig, bench: str, spec: BenchSpec, selection: dict[str, Any],
                      difficult: dict[str, Any], *, explicit: dict[str, list[str]] | None = None, log=print) -> dict[str, Any]:
    paths = audit_paths(cfg, bench)
    bare = ensure_bare_repo(paths["bare_repo"])
    venvs: dict[str, str] = {}
    docs: dict[str, dict] = {}
    for arm in cfg.arms:
        sel_arm = selection["arms"].get(arm) or {}
        if not sel_arm.get("run_id"):
            log(f"  {bench}/{arm}: no audited run — skipped")
            continue
        docs[arm] = materialize_arm(cfg, bench, spec, arm, sel_arm, bare, venvs,
                                    explicit_shas=(explicit or {}).get(arm), log=log)
    if not docs:
        raise RuntimeError(f"{bench}: nothing to materialize")
    first = next(c["sha"] for d in docs.values() for c in d["candidates"] if c.get("tree"))
    pool = WorktreePool(bare, paths["worktrees"], spec, cfg.workers)
    pool.ensure(first)
    testset_hash = pool.testset_sha256(pool.slots[0])
    frozen = difficult.get("testset_sha256")
    if frozen and testset_hash != frozen:
        raise RuntimeError(
            f"{bench}: the staged held-out testset ({spec.repo_path}/{spec.testset_path}) hashes to "
            f"{testset_hash[:12]} but the frozen difficult set was built on {frozen[:12]}; row ids are "
            "positional, so the audit refuses to run on a different file"
        )
    mat = {
        "bench": bench, "bare_repo": str(bare),
        "candidates": {d["job_id"]: d["candidates"] for d in docs.values()},
        "arms": {arm: {"job_id": d["job_id"], "run_id": d["run_id"], "final_tree": d["final_tree"]} for arm, d in docs.items()},
        "venvs": venvs, "worktrees": [str(s.path) for s in pool.slots],
        "testset_sha256": testset_hash, "spec": spec.to_dict(), "materialized_at": time.time(),
    }
    paths["materialize"].parent.mkdir(parents=True, exist_ok=True)
    paths["materialize"].write_text(json.dumps(mat, indent=2))
    return mat


def load_candidates(cfg: AuditConfig, bench: str, arm: str) -> dict[str, Any]:
    path = arm_path(cfg, "candidates", bench, arm)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `materialize --bench {bench}` first")
    return json.loads(path.read_text())


__all__ = ["load_candidates", "materialize_arm", "materialize_bench"]
