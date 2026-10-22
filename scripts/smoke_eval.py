"""Copyright © 2026 440 Labs LLC

Run a handful of rows of a benchmark's eval on the LOCAL client checkout and
show the traces — no optimization run, no Docker, no API server.

The full eval path is already host-runnable: `sandbox_client.run_eval` shells
out to `master_script.py` with `cwd=<workspace>`, so nothing here reimplements
evaluation semantics. What did not exist was an ergonomic front door:

* `scripts/seed_variance.py` needs `--runs >= 2`, draws a RANDOM sample, and
  reports statistics rather than trace paths.
* `scripts/best_of_population/run.py` needs a select-run/materialize pipeline over
  finished runs and worktree pools.
* `local_test_app.py eval_trace` clones FactChecker and runs the seed step.
* `verify_environment.py::validate_sandbox_environment` is a GATE — it raises
  on a 0.0 aggregate, so it cannot show you a failing program.

This script is a viewer, not a gate: it never raises on a low score, and a
benchmark that blows up is reported and skipped so the sweep continues.

The benchmark's workspace, program factory, metric, dataset paths, input keys
and thread count all come from `load_bench_spec` (orchestrator/benchmarks.yaml
+ experiments/<dir>/input.py::OPTIMIZE_CONFIG_LOCAL) — the same two sources a
research run reads. The checkout is scored AS-IS: nothing is checked out or
stashed, and HEAD's branch + sha are printed so which program ran is never
ambiguous (arcagi2 and paperreview normally sit on a seed-program branch).

Usage:

    python -m scripts.smoke_eval --bench hover --n 5
    python -m scripts.smoke_eval --bench all --n 5 --show 2 --out ~/ce-smoke
    python -m scripts.smoke_eval --bench phantomwiki --split test --n 3

Environment (checked at runtime, reported per benchmark):

* Program LM keys: GMI_API_KEY / DEEPINFRA_API_KEY / DEEPSEEK_API_KEY.
* OPENAI_API_KEY — the ragqa judge.
* ragqa ONLY needs its retriever sidecar up first (default
  http://localhost:8894, override with RAGQA_RETRIEVER_URL):
      cd ~/Documents/code/ragqa-retriever-server && . ragqa-server.sh
      python -m ragqa_retriever
  hover's ColBERT is a hosted Modal endpoint, so it needs no local sidecar.

Pass `--dotenv PATH` to load a .env file before running (client repos here keep
theirs at `.env.local`, which nothing loads automatically).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

APP_ROOT = Path(os.environ.get("CE_APP_ROOT", Path(__file__).resolve().parents[1]))

# Benchmarks whose client checkout lives on this kind of machine and whose eval
# is runnable without fleet infrastructure. `--bench all` expands to this.
DEFAULT_BENCHES = ("hover", "ragqa", "phantomwiki", "arcagi2", "paperreview")

# Trace suffix. `holdout_*` is what routes the mounted evaluator to the testset
# (evaluate.py::_load_dataset -> is_holdout_suffix), and it also puts the
# `_holdout` fragment in the trace dir name, so the standard agent deny-path
# fence still covers these traces if one is ever pointed at them.
TRAIN_SUFFIX = "smoke"
TEST_SUFFIX = "holdout_smoke"

RAGQA_DEFAULT_RETRIEVER = "http://localhost:8894/api/search"
# A cold dense-retriever query measured ~8.4s on this Mac, so a short probe
# timeout reads a warm, healthy server as down. Generous here, and a timeout is
# treated as "listening but slow" rather than a blocker — only a refused
# connection means nothing is serving.
RAGQA_PROBE_TIMEOUT = 30

# Truncation for the `--show` span dump. Long enough to read a prompt's shape,
# short enough that five rows stay scrollable.
SHOW_VALUE_CHARS = 400


def emit(*args: Any, **kwargs: Any) -> None:
    """`print` that flushes.

    A full sweep runs benchmarks sequentially and takes tens of minutes, so it
    is usually backgrounded or piped — where Python block-buffers stdout and
    nothing appears until the process exits. Every line here is progress
    someone is waiting on, so pay the flush.
    """
    print(*args, flush=True, **kwargs)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _git_head(workspace: str) -> str:
    """`branch@sha` for the checkout, or a reason it could not be read."""
    from src.engine.services.subprocess_service import _git_run, git_get_sha

    try:
        sha = git_get_sha(workspace)[:12]
    except Exception as exc:  # noqa: BLE001
        return f"(no git sha: {exc})"
    branch = _git_run(workspace, "rev-parse", "--abbrev-ref", "HEAD")
    name = branch.stdout.strip() if branch.returncode == 0 else "?"
    dirty = _git_run(workspace, "status", "--porcelain")
    suffix = " +dirty" if dirty.returncode == 0 and dirty.stdout.strip() else ""
    return f"{name}@{sha}{suffix}"


def _ragqa_retriever_up() -> tuple[bool, str]:
    """Probe the ragqa retriever sidecar. Returns (ok, detail).

    Only a refused/failed connection blocks. A slow answer still means the
    server is serving — the eval's own timeout is the right place to give up
    on that, not a preflight check.
    """
    import requests

    url = os.environ.get("RAGQA_RETRIEVER_URL", RAGQA_DEFAULT_RETRIEVER)
    try:
        requests.get(url, params={"query": "ping", "k": 1},
                     timeout=RAGQA_PROBE_TIMEOUT)
    except requests.exceptions.Timeout:
        return True, f"{url} (listening but slow to answer a probe query)"
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"retriever sidecar unreachable at {url} ({type(exc).__name__}). "
            "Start it with: cd ~/Documents/code/ragqa-retriever-server && "
            ". ragqa-server.sh && python -m ragqa_retriever"
        )
    return True, url


def preflight(slug: str, spec, split: str) -> list[str]:
    """Blocking problems for this benchmark. Empty list == good to run."""
    problems: list[str] = []
    workspace = Path(spec.repo_path)
    if not workspace.exists():
        problems.append(
            f"client checkout not found at {workspace} "
            f"(set CE_REPO_{slug.upper()} or CE_REPO_ROOT)"
        )
        return problems  # everything below needs the checkout

    # `subprocess_service._build_env` prepends `<workspace>/env/bin` to PATH;
    # without it the evaluator runs in whatever python happens to be first and
    # fails on the client's imports.
    from src.engine.paths import repo_venv_bin, repo_venv_dirname

    venv = workspace / repo_venv_dirname() / repo_venv_bin()
    if not venv.exists():
        problems.append(f"no client venv at {venv}")

    rel = spec.testset_path if split == "test" else spec.trainset_path
    if not rel:
        problems.append(f"no {split}set path configured for {slug}")
    elif not (workspace / rel).exists():
        problems.append(f"{split}set file missing: {workspace / rel}")

    for asset in spec.assets:
        if not (workspace / asset).exists():
            problems.append(f"gitignored asset missing: {workspace / asset}")

    if slug == "ragqa":
        ok, detail = _ragqa_retriever_up()
        if not ok:
            problems.append(detail)

    return problems


# ---------------------------------------------------------------------------
# Trace display
# ---------------------------------------------------------------------------

def _clip(value: Any, limit: int = SHOW_VALUE_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


def show_traces(trace_dir: str, n_rows: int) -> None:
    """Print the span chain of the first `n_rows` rows of a trace directory."""
    from src.engine.telemetry.trace_layout import iter_rows

    for i, row in enumerate(iter_rows(trace_dir, with_spans=True)):
        if i >= n_rows:
            break
        emit(f"\n  --- row {row.get('example_id')} "
              f"score={row.get('score')} failed={bool(row.get('failed'))} ---")
        spans = row.get("trace") or []
        if not spans:
            emit("    (no spans captured — is the client repo OTel-instrumented?)")
        for span in spans:
            key = span.get("signature_key") or "?"
            parent = span.get("parent_key")
            emit(f"    [{key}]" + (f" <- {parent}" if parent else "  (root)"))
            for name, val in (span.get("inputs") or {}).items():
                emit(f"        in.{name}: {_clip(val)}")
            if span.get("output") is not None:
                emit(f"        out: {_clip(span['output'])}")


# ---------------------------------------------------------------------------
# One benchmark
# ---------------------------------------------------------------------------

def smoke_one(
    slug: str,
    *,
    n: int,
    split: str,
    rows_mode: str,
    seed: int,
    out_dir: Path,
    show: int,
    num_threads: int | None,
    log=emit,
) -> dict[str, Any]:
    """Evaluate `n` rows of one benchmark on its local checkout. Never raises."""
    from scripts.best_of_population.config import load_bench_spec
    from src.engine.evaluator.dataset_config import DatasetConfig
    from src.engine.evaluator.row_scores import (
        failed_rows_from_trace_meta,
        row_scores_from_trace_meta,
    )
    from src.engine.evaluator.sandbox_client import run_eval

    report: dict[str, Any] = {"bench": slug, "split": split}
    log(f"\n{'=' * 72}\n{slug}\n{'=' * 72}")

    try:
        spec = load_bench_spec(slug)
    except Exception as exc:  # noqa: BLE001
        log(f"  SKIP: cannot resolve bench spec: {type(exc).__name__}: {exc}")
        return {**report, "status": "skipped", "error": str(exc)}

    workspace = str(spec.repo_path)
    report["workspace"] = workspace

    problems = preflight(slug, spec, split)
    if problems:
        log("  SKIP:")
        for p in problems:
            log(f"    - {p}")
        return {**report, "status": "skipped", "problems": problems}

    head = _git_head(workspace)
    report["head"] = head
    log(f"  workspace: {workspace}")
    log(f"  HEAD:      {head}   (scored as-is; nothing checked out)")
    log(f"  program:   {spec.program_factory}")
    log(f"  metric:    {spec.metric_fn}")

    # Only hand the evaluator the dataset it is allowed to touch: on a train
    # smoke run the testset path stays None, so the holdout file is never even
    # named in the payload.
    try:
        datasets = DatasetConfig.prepare(
            workspace,
            trainset_json=None,
            trainset_path=spec.trainset_path,
            comparison_set_size=None,
            testset_json=None,
            testset_path=spec.testset_path if split == "test" else None,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  SKIP: dataset resolution failed: {type(exc).__name__}: {exc}")
        return {**report, "status": "skipped", "error": str(exc)}

    pool = datasets.testset_ids if split == "test" else datasets.trainset_ids
    if not pool:
        log(f"  SKIP: {split}set is empty")
        return {**report, "status": "skipped", "error": f"{split}set is empty"}

    k = min(n, len(pool))
    if rows_mode == "random":
        row_ids = sorted(random.Random(seed).sample(list(pool), k))
    else:
        row_ids = list(pool[:k])
    report["row_ids"] = row_ids
    log(f"  {split}set: {len(pool)} rows; evaluating {k} ({rows_mode}) -> {row_ids}")

    evaluator_config = spec.evaluator_config()
    if split != "test":
        # `evaluator_config()` carries both paths; drop the holdout on a train
        # run so it cannot be reached even by a mistyped suffix.
        evaluator_config.testset_path = None
    if num_threads is not None:
        evaluator_config.num_threads = num_threads
    log(f"  num_threads: {evaluator_config.num_threads}")

    # Per-bench trace root. Without this every run lands in the shared
    # ~/.codeevolver/traces and successive runs overwrite each other.
    traces_root = out_dir / slug
    traces_root.mkdir(parents=True, exist_ok=True)
    os.environ["CODEEVOLVER_TRACES_DIR"] = str(traces_root)
    os.environ.pop("CODEEVOLVER_JOB_ID", None)

    suffix = TEST_SUFFIX if split == "test" else TRAIN_SUFFIX
    timeout = spec.subsample_eval_timeout
    started = time.time()
    try:
        result = run_eval(
            workspace=workspace,
            config=evaluator_config,
            row_ids=row_ids,
            iteration_idx=0,
            suffix=suffix,
            capture_traces=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.time() - started, 1)
        log(f"  FAILED after {elapsed}s: {type(exc).__name__}: {exc}")
        log(f"  stderr, if the evaluator got far enough to write it: "
            f"{traces_root}/iteration_0_{suffix}/eval_stderr.log")
        return {**report, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": elapsed}

    elapsed = round(time.time() - started, 1)
    scores = row_scores_from_trace_meta(result.trace_meta)
    failed = set(failed_rows_from_trace_meta(result.trace_meta))

    log(f"\n  primary ({evaluator_config.primary_metric}): "
        f"{result.primary_score:.4f}   [{elapsed}s]")
    log(f"  {'row':>8}  {'score':>8}  status")
    for rid in row_ids:
        key = str(rid)
        score = scores.get(key)
        status = "FAILED" if key in failed else ("no score" if score is None else "ok")
        log(f"  {rid:>8}  {('-' if score is None else f'{score:.4f}'):>8}  {status}")
    if failed:
        log(f"  {len(failed)} row(s) raised — see the trace rows below for the error.")

    log(f"\n  traces: {result.trace_dir}")
    log(f"  pretty: {os.path.join(result.trace_dir, 'index_pretty_print.txt')}")
    if show:
        show_traces(result.trace_dir, show)

    return {
        **report,
        "status": "ok",
        "primary_metric": evaluator_config.primary_metric,
        "primary_score": float(result.primary_score),
        "scores_by_metric": dict(result.scores_by_metric),
        "row_scores": scores,
        "failed_rows": sorted(failed),
        "trace_dir": result.trace_dir,
        "num_threads": evaluator_config.num_threads,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run N rows of a benchmark's eval locally and show the traces."
    )
    parser.add_argument("--bench", nargs="+", default=["all"],
                        help=f"benchmark slugs, or 'all' ({', '.join(DEFAULT_BENCHES)})")
    parser.add_argument("--n", type=int, default=5, help="rows to evaluate (default 5)")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="train (default) or the held-out test set")
    parser.add_argument("--rows", choices=["first", "random"], default="first",
                        help="take the first N row ids, or a random draw")
    parser.add_argument("--seed", type=int, default=0, help="seed for --rows random")
    parser.add_argument("--num-threads", type=int, default=None,
                        help="override the benchmark's configured eval concurrency")
    parser.add_argument("--show", type=int, default=0, metavar="K",
                        help="pretty-print the span chain of the first K trace rows")
    parser.add_argument("--out", default=None,
                        help="output root (default ~/ce-smoke/<timestamp>)")
    parser.add_argument("--dotenv", default=None, help=".env file to load first")
    args = parser.parse_args(argv)

    if args.n <= 0:
        parser.error(f"--n must be positive, got {args.n}")

    sys.path.insert(0, str(APP_ROOT))
    from scripts.best_of_population.config import AnalysisConfig, env_preamble

    out_dir = Path(args.out).expanduser() if args.out else (
        Path.home() / "ce-smoke" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    env_preamble(AnalysisConfig(out=str(out_dir), dotenv=args.dotenv))

    benches = list(DEFAULT_BENCHES) if "all" in args.bench else list(args.bench)
    emit(f"smoke_eval: {len(benches)} benchmark(s), {args.n} {args.split} row(s) each")
    emit(f"output: {out_dir}")

    reports = [
        smoke_one(
            slug,
            n=args.n,
            split=args.split,
            rows_mode=args.rows,
            seed=args.seed,
            out_dir=out_dir,
            show=args.show,
            num_threads=args.num_threads,
        )
        for slug in benches
    ]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "n": args.n, "split": args.split, "rows": args.rows,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": reports,
    }, indent=2))

    emit(f"\n{'=' * 72}\nsummary\n{'=' * 72}")
    for rep in reports:
        if rep["status"] == "ok":
            detail = (f"{rep['primary_score']:.4f} "
                      f"({len(rep['failed_rows'])} failed, {rep['elapsed_seconds']}s)")
        else:
            detail = rep.get("error") or "; ".join(rep.get("problems") or [])
        emit(f"  {rep['bench']:<14} {rep['status']:<8} {detail}")
    emit(f"\nwrote {summary_path}")

    # Non-zero only when nothing ran at all — a low score is a result, not an
    # error, and one dead sidecar should not fail the whole sweep.
    return 0 if any(r["status"] == "ok" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
