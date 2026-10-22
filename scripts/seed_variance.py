"""Copyright © 2026 440 Labs LLC

Standalone seed-program variance estimate — the run_variance_estimate tool's
math, outside an optimization.

The in-run tool (src/engine/optimizer/v3/variance_estimate.py) is deliberately
architect-facing: it spends the run's eval budget, records onto CEState, and
is refused on the seed program. That leaves no way to answer the budgeting
question "how noisy is a subsample reading of the SEED program?" without
burning an optimization run. This script answers it directly: draw ONE fixed
random `subsample_size` sample from the trainset, evaluate the seed program on
those same rows R times through the same evaluator path the optimizer uses
(`sandbox_client.run_eval`), and report the same statistics via the same
`estimate_variance` function — within-item variance, run-to-run std, empirical
aggregate std, and the R aggregate scores. Everything (including per-row
scores per run) is saved as JSON so the reading can be re-derived offline.

No optimizer, no budget ledger, no state: the caveats in
specs/archive-public/sampling_variance.md apply, plus one more — this measures the SEED
program, and variance can differ once the program improves. That is exactly
the reading wanted for sizing subsample/comparison budgets before a run.

Runs inside a runner container via `entrypoint.py --seed-variance` (which
prepares the workspace, venv, egress and simulators exactly as a real run
would, then calls `run_seed_variance`), or directly against a local checkout:

    python -m scripts.seed_variance --workspace /workspace \
        --program AppWorld --config OPTIMIZE_CONFIG_LOCAL \
        --runs 5 --output /artifacts/seed_variance.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

APP_ROOT = Path(os.environ.get("CE_APP_ROOT", Path(__file__).resolve().parents[1]))

# The default draw seed for the calibration sample. Deliberately NOT 42:
# the comparison set is drawn from the same trainset with seed 42
# (dataset_config.COMPARISON_SET_SEED), and a variance sample that equals the
# comparison set would measure a different thing than a fresh subsample.
SAMPLE_SEED = 4242


def run_seed_variance(
    *,
    workspace: str,
    config: dict,
    n_runs: int = 5,
    subsample_size: int | None = None,
    sample_seed: int = SAMPLE_SEED,
    output_path: str | None = None,
    log=print,
) -> dict[str, Any]:
    """Evaluate the seed program `n_runs` times on one fixed subsample.

    `config` is the fully resolved run config (the dict `experiments.run.
    resolve_config` returns — entrypoint.py's `effective_config`). Returns the
    result payload; writes it to `output_path` when given. Raises if fewer
    than two runs complete (no estimate is possible).
    """
    sys.path.insert(0, str(APP_ROOT))
    from src.engine.evaluator.config import EvaluatorConfig
    from src.engine.evaluator.dataset_config import DatasetConfig
    from src.engine.evaluator.holdout import samples_are_identical
    from src.engine.evaluator.row_scores import row_scores_from_trace_meta
    from src.engine.evaluator.sandbox_client import run_eval
    from src.engine.optimizer.v3.eval_log import sample_key_for_rows, variance_suffix
    from src.engine.optimizer.v3.variance_estimate import (
        HEURISTIC_NOTE,
        IDENTICAL_OUTPUTS_NOTE,
        estimate_variance,
    )
    from src.engine.services.subprocess_service import git_get_sha

    # Trainset only: the variance sample never touches the (possibly already
    # sequestered) holdout, and no comparison set is needed here.
    datasets = DatasetConfig.prepare(
        workspace,
        trainset_json=config.get("trainset_json"),
        trainset_path=config.get("trainset_path"),
        comparison_set_size=None,
        testset_json=None,
        testset_path=None,
    )

    metric_names = (config.get("metric_names") or ["accuracy"])[:1]
    primary_metric = config.get("primary_metric") or metric_names[0]
    evaluator_config = EvaluatorConfig(
        program_factory=config["program"],
        metric_fns=[config["metric"]],
        metric_names=metric_names,
        primary_metric=primary_metric,
        num_threads=int(config.get("num_threads") or 1),
        trainset_path=datasets.trainset_rel_path,
        testset_path=None,
        input_keys=config.get("input_keys"),
    )

    size = int(subsample_size or config.get("subsample_size") or 0)
    if size <= 0:
        raise ValueError(f"subsample_size must be positive, got {size}")
    if n_runs < 2:
        raise ValueError(f"at least 2 runs are needed, got {n_runs}")
    if not datasets.trainset_ids:
        raise ValueError("trainset is empty; cannot draw a variance sample")

    k = min(size, len(datasets.trainset_ids))
    row_ids = sorted(random.Random(sample_seed).sample(list(datasets.trainset_ids), k))
    timeout = config.get("subsample_eval_timeout")
    timeout = int(timeout) if timeout else None
    log(
        f"seed_variance: {n_runs} runs x {k} rows "
        f"(sample_seed={sample_seed}, timeout={timeout}s) ids={row_ids}"
    )

    runs: list[dict[str, Any]] = []
    per_run_row_scores: list[dict[str, float]] = []
    aggregate_scores: list[float] = []
    trace_dirs: list[str] = []
    for run_idx in range(n_runs):
        started = time.time()
        try:
            result = run_eval(
                workspace=workspace,
                config=evaluator_config,
                row_ids=list(row_ids),
                iteration_idx=0,
                suffix=variance_suffix(run_idx),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            log(
                f"seed_variance: run {run_idx + 1}/{n_runs} FAILED "
                f"({type(exc).__name__}: {exc}); keeping {len(runs)} completed run(s)"
            )
            runs.append({
                "run": run_idx + 1,
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 1),
            })
            continue
        row_scores = row_scores_from_trace_meta(result.trace_meta)
        per_run_row_scores.append(row_scores)
        aggregate_scores.append(float(result.primary_score))
        trace_dirs.append(result.trace_dir)
        runs.append({
            "run": run_idx + 1,
            "primary_score": float(result.primary_score),
            "scores_by_metric": dict(result.scores_by_metric),
            "row_scores": row_scores,
            "trace_dir": result.trace_dir,
            "elapsed_seconds": round(time.time() - started, 1),
        })
        log(
            f"seed_variance: run {run_idx + 1}/{n_runs} -> "
            f"{result.primary_score:.4f} ({runs[-1]['elapsed_seconds']}s)"
        )

    if len(per_run_row_scores) < 2:
        raise RuntimeError(
            f"only {len(per_run_row_scores)} of {n_runs} runs completed; at "
            "least two are needed to estimate variance"
        )

    stats = estimate_variance(per_run_row_scores, aggregate_scores)
    identical = samples_are_identical(trace_dirs)
    try:
        sha = git_get_sha(workspace)
    except Exception:  # noqa: BLE001
        sha = None

    notes = [HEURISTIC_NOTE]
    if identical:
        notes.insert(0, IDENTICAL_OUTPUTS_NOTE)
    notes.append(
        "Measured on the SEED program via scripts/seed_variance.py — the "
        "in-run tool measures an improved program a few iterations in, so "
        "the two readings are not interchangeable."
    )

    payload: dict[str, Any] = {
        "kind": "seed_variance",
        "program": config.get("program"),
        "seed_sha": sha,
        "trainset_path": config.get("trainset_path"),
        "trainset_size": datasets.trainset_size,
        "subsample_size": k,
        "sample_seed": sample_seed,
        "sample_key": sample_key_for_rows(row_ids),
        "row_ids": list(row_ids),
        "n_runs_requested": n_runs,
        "n_runs_completed": stats.n_runs,
        "n_rows_common": stats.n_rows,
        "aggregate_scores": aggregate_scores,
        "aggregate_mean": sum(aggregate_scores) / len(aggregate_scores),
        "within_item_variance": stats.within_item_variance,
        "run_to_run_std": stats.run_to_run_std,
        "aggregate_std_empirical": stats.aggregate_std_empirical,
        "identical_outputs_warning": bool(identical),
        "num_threads": evaluator_config.num_threads,
        "runs": runs,
        "note": " ".join(notes),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(payload, indent=2))
        log(f"seed_variance: wrote {output_path}")
    log(
        "seed_variance: scores="
        + str([round(s, 4) for s in aggregate_scores])
        + f" mean={payload['aggregate_mean']:.4f}"
        + f" run_to_run_std={stats.run_to_run_std:.4f}"
        + f" empirical_std={stats.aggregate_std_empirical:.4f}"
        + f" within_item_variance={stats.within_item_variance:.4f}"
        + (" IDENTICAL-OUTPUTS" if identical else "")
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--workspace", required=True, help="client repo checkout")
    parser.add_argument("--program", required=True,
                        help="experiments/ directory name, e.g. AppWorld")
    parser.add_argument("--config", default="OPTIMIZE_CONFIG_LOCAL",
                        help="named config in the experiment's input.py")
    parser.add_argument("--set", action="append", default=[], dest="sets",
                        metavar="KEY=VALUE", help="config override (repeatable)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--subsample-size", type=int, default=None,
                        help="override the config's subsample_size")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(APP_ROOT))
    os.environ.setdefault("APP_MODE", "local")
    from experiments.run import resolve_config  # noqa: PLC0415

    module = importlib.import_module(f"experiments.{args.program}.input")
    overrides = [f"repo_url={args.workspace}", *args.sets]
    _, config = resolve_config(module, args.program, args.config, overrides)

    run_seed_variance(
        workspace=args.workspace,
        config=config,
        n_runs=args.runs,
        subsample_size=args.subsample_size,
        sample_seed=args.sample_seed,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
