"""Copyright © 2026 440 Labs LLC

Evaluation driver: executes a list of `EvalTask`s (planned by `planner.py`
over what the store is missing) through the engine's evaluator on detached
worktrees, and appends per-row scores to the store.

One `run_eval` call per (candidate, split, draw) with every needed row, so
the client program's start-up is amortized. Workers are PROCESSES, not
threads: the evaluator resolves `CODEEVOLVER_TRACES_DIR` from the process
environment, so each worker owns one worktree slot and its own env.
"""

from __future__ import annotations

import csv
import io
import json
import multiprocessing as mp
import os
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import AnalysisConfig, BenchSpec, analysis_paths, env_preamble
from .store import ScoreStore, row_line

DEFAULT_SECONDS_PER_ROW = 12.0
CALL_OVERHEAD_SECONDS = 20.0


@dataclass
class EvalTask:
    job_id: str
    arm: str
    sha: str
    tree: str
    reqhash: str | None
    split: str            # "test" | "train"
    row_ids: list[int]
    draw: int
    purpose: str          # screen | select | confirm | final (+ "<phase>_fill")
    label: str = ""

    @property
    def suffix(self) -> str:
        base = f"{self.purpose}_{self.tree[:8]}_d{self.draw}"
        return f"holdout_{base}" if self.split == "test" else base


def dataset_len(path: Path) -> int:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return sum(1 for line in text.splitlines() if line.strip())
    if path.suffix == ".csv":
        return sum(1 for _ in csv.DictReader(io.StringIO(text)))
    data = json.loads(text)
    return len(data)


# ---------------------------------------------------------------------------
# Single-process evaluator (one worktree slot)
# ---------------------------------------------------------------------------

class SlotEvaluator:
    def __init__(self, cfg: AnalysisConfig, bench: str, spec: BenchSpec, mat: dict[str, Any], slot_index: int) -> None:
        from .materialize import WorktreePool

        self.cfg, self.bench, self.spec, self.mat = cfg, bench, spec, mat
        self.paths = analysis_paths(cfg, bench)
        self.pool = WorktreePool(Path(mat["bare_repo"]), self.paths["worktrees"], spec, cfg.workers)
        self.slot = self.pool.slots[slot_index]
        first = (mat.get("baseline") or {}).get("sha") or next(
            c["sha"] for cs in mat["candidates"].values() for c in cs if c.get("tree")
        )
        self.pool.ensure(first)
        self.evaluator_config = spec.evaluator_config()

    def run(self, task: EvalTask) -> tuple[dict, list[dict]]:
        from src.engine.evaluator.row_scores import row_scores_from_trace_meta
        from src.engine.evaluator.sandbox_client import run_eval

        venv = Path(self.mat["venvs"][task.reqhash]) if task.reqhash in self.mat["venvs"] else None
        self.pool.checkout(self.slot, task.sha, venv)
        traces_dir = self.paths["traces"] / task.job_id / task.tree[:12]
        traces_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CODEEVOLVER_TRACES_DIR"] = str(traces_dir)
        os.environ.pop("CODEEVOLVER_JOB_ID", None)
        # A DSPy disk cache shared across draws would make repeat draws
        # byte-identical for any candidate that left `dspy.LM(cache=True)` on.
        # The engine only detects that (`holdout.samples_are_identical`); the
        # audit isolates it: one cache dir per (program, phase, draw).
        os.environ["DSPY_CACHEDIR"] = str(traces_dir / f"dspy_cache_{task.suffix}")
        # Same cap as the run's held-out evals — never the subsample cap.
        timeout = self.spec.comparison_eval_timeout
        t0 = time.time()
        result = run_eval(
            workspace=str(self.slot.path),
            config=self.evaluator_config,
            row_ids=list(task.row_ids),
            iteration_idx=0,
            suffix=task.suffix,
            capture_traces=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        eval_id = uuid.uuid4().hex[:12]
        meta = result.trace_meta or []
        scores = row_scores_from_trace_meta(meta)
        failed = {str(r.get("example_id")) for r in meta if r.get("failed")}
        testset_hash = self.pool.testset_sha256(self.slot)
        trainset_hash = self.pool.trainset_sha256(self.slot)
        rows = []
        for rid in task.row_ids:
            key = str(rid)
            rows.append(row_line(
                bench=self.bench, job_id=task.job_id, arm=task.arm, tree=task.tree, sha=task.sha,
                split=task.split, row_id=rid, draw=task.draw,
                score=scores.get(key), failed=(key in failed or key not in scores),
                purpose=task.purpose, eval_id=eval_id,
                testset_sha256=testset_hash, trainset_sha256=trainset_hash,
            ))
        eval_line = {
            "eval_id": eval_id, "bench": self.bench, "job_id": task.job_id, "arm": task.arm,
            "tree": task.tree, "sha": task.sha, "split": task.split, "purpose": task.purpose,
            "draw": task.draw, "row_ids": list(task.row_ids), "n_rows": len(task.row_ids),
            "score": result.primary_score, "trace_dir": result.trace_dir,
            "elapsed_s": elapsed, "worker": self.slot.index, "ts": time.time(),
            "testset_sha256": testset_hash, "trainset_sha256": trainset_hash,
            "n_failed": len([r for r in rows if r["failed"]]),
        }
        return eval_line, rows


def _worker_main(slot_index: int, cfg_dict: dict, bench: str, spec_dict: dict, mat: dict,
                 task_q: "mp.Queue", result_q: "mp.Queue") -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    cfg = AnalysisConfig(**cfg_dict)
    env_preamble(cfg)
    spec = BenchSpec(**{**spec_dict, "assets": tuple(spec_dict["assets"]),
                        "input_keys": tuple(spec_dict["input_keys"]) if spec_dict.get("input_keys") else None})
    ev = SlotEvaluator(cfg, bench, spec, mat, slot_index)
    while True:
        item = task_q.get()
        if item is None:
            return
        task = EvalTask(**item)
        try:
            eval_line, rows = ev.run(task)
            result_q.put(("ok", item, eval_line, rows))
        except Exception:  # noqa: BLE001
            result_q.put(("error", item, traceback.format_exc(), None))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@dataclass
class ExecutionReport:
    tasks: int = 0
    rows: int = 0
    failed_tasks: int = 0
    identical_pairs: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def execute_tasks(
    cfg: AnalysisConfig, bench: str, spec: BenchSpec, mat: dict[str, Any],
    tasks: list[EvalTask], store: ScoreStore, *, log=print,
) -> ExecutionReport:
    rep = ExecutionReport()
    if not tasks:
        return rep

    def _on_result(task: EvalTask, eval_line: dict, rows: list[dict]) -> None:
        added = store.append_eval(eval_line, rows)
        rep.tasks += 1
        rep.rows += added
        log(f"  [{rep.tasks}/{len(tasks)}] {task.purpose} {task.arm} {task.tree[:8]} "
            f"{task.split} d{task.draw} rows={len(task.row_ids)} score={eval_line['score']} "
            f"failed={eval_line['n_failed']} {eval_line['elapsed_s']:.0f}s")
        if task.draw >= 1:
            _check_identical(store, eval_line, task, rep, log)

    if cfg.workers <= 1:
        env_preamble(cfg)
        ev = SlotEvaluator(cfg, bench, spec, mat, 0)
        for task in tasks:
            try:
                eval_line, rows = ev.run(task)
            except Exception as exc:  # noqa: BLE001
                rep.failed_tasks += 1
                rep.errors.append(f"{task.label or task.tree[:8]}: {exc}")
                log(f"  ERROR {task.purpose} {task.tree[:8]} d{task.draw}: {exc}")
                continue
            _on_result(task, eval_line, rows)
        return rep

    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()
    n = min(cfg.workers, len(tasks))
    procs = [
        ctx.Process(target=_worker_main, args=(i, asdict(cfg), bench, spec.to_dict(), mat, task_q, result_q), daemon=True)
        for i in range(n)
    ]
    for p in procs:
        p.start()
    for t in tasks:
        task_q.put(asdict(t))
    for _ in procs:
        task_q.put(None)
    pending = len(tasks)
    while pending:
        status, item, payload, rows = result_q.get()
        pending -= 1
        task = EvalTask(**item)
        if status == "ok":
            _on_result(task, payload, rows)
        else:
            rep.failed_tasks += 1
            rep.errors.append(f"{task.label or task.tree[:8]}: {payload.splitlines()[-1] if payload else '?'}")
            log(f"  ERROR {task.purpose} {task.tree[:8]} d{task.draw}:\n{payload}")
    for p in procs:
        p.join(timeout=60)
    return rep


def _check_identical(store: ScoreStore, eval_line: dict, task: EvalTask, rep: ExecutionReport, log) -> None:
    """Warn-only cache check. The task repos have always run with caching
    off, and the detector compares only each row's final `output`, so a
    1-2 row draw of a label-output benchmark (paperreview "accept") repeats
    by chance. Findings are recorded in the report / winners file; nothing
    is ever aborted over them."""
    from src.engine.evaluator.holdout import samples_are_identical

    prev = [e for e in store.evals()
            if e["job_id"] == task.job_id and e["tree"] == task.tree and e.get("split") == task.split
            and e["purpose"] == task.purpose and e["draw"] == task.draw - 1 and e.get("trace_dir")]
    if not prev or not eval_line.get("trace_dir"):
        return
    if samples_are_identical([prev[-1]["trace_dir"], eval_line["trace_dir"]]):
        rep.identical_pairs.append((task.tree, f"d{task.draw - 1}/d{task.draw}"))
        log(f"  WARNING: draws d{task.draw - 1} and d{task.draw} of {task.tree[:8]} have byte-identical "
            f"outputs over {eval_line.get('n_rows', '?')} row(s) — expected by chance on tiny label-output "
            "draws; a response cache only if it repeats across many rows")


def task_for(c: dict, rows: list[int], draw: int, purpose: str, *, split: str = "test") -> EvalTask:
    """One task for candidate dict `c` (a `candidates/<bench>-<arm>.json` entry)."""
    return EvalTask(job_id=c["job_id"], arm=c.get("arm", "?"), sha=c["sha"], tree=c["tree"],
                    reqhash=c.get("reqhash"), split=split, row_ids=list(rows), draw=draw, purpose=purpose,
                    label=f"{c.get('arm', '?')}:{c['sha'][:8]}")


def cost_summary(tasks: Iterable[EvalTask], store: ScoreStore, *, num_threads: int) -> dict[str, Any]:
    tasks = list(tasks)
    rows = sum(len(t.row_ids) for t in tasks)
    spr = store.per_row_latency() or DEFAULT_SECONDS_PER_ROW
    seconds = sum(len(t.row_ids) / max(1, num_threads) * spr + CALL_OVERHEAD_SECONDS for t in tasks)
    return {"calls": len(tasks), "rows": rows, "est_seconds_serial": round(seconds),
            "seconds_per_row_assumed": spr,
            "by_purpose": {p: sum(len(t.row_ids) for t in tasks if t.purpose == p)
                           for p in sorted({t.purpose for t in tasks})}}


__all__ = [
    "CALL_OVERHEAD_SECONDS", "DEFAULT_SECONDS_PER_ROW", "EvalTask", "ExecutionReport",
    "SlotEvaluator", "cost_summary", "dataset_len", "execute_tasks", "task_for",
]
