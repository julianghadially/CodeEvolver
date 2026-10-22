#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Safe-mode run-integrity check against a LIVE (or finished) local run.

Spawns the read-only run-integrity inspector (cheating classification +
functioning-iteration %) in SAFE MODE — the prompt forbids branch switches,
writes, and program/eval execution — and writes the verdict where an external
operator can pick it up:

  - ``<--output>``               the parsed JSON verdict
  - ``<--output dir>/run_integrity.md``   the full markdown report

It never touches the optimizer's iteration history (the closing-loop
invocation inside ``optimizer_loop.run_optimization`` owns those keys), and
it never writes into ``memory_dir()`` — a live run's architect can reach that
tree. The fleet runner (``runner/entrypoint.py``) triggers this at ~20% of the
eval budget with ``--output /artifacts/integrity.json``; the orchestrator
folds the verdict into its status sweep and logs an INTEGRITY WARNING when it
is not clean. The verdict is informational only — it never stops the run.

Usage:
    python scripts/inspect_run_integrity.py \
        --job-id job_abc123 --workspace /workspace \
        [--output /artifacts/integrity.json] \
        [--testset-path data/bench_test.json] \
        [--coding-harness opencode] [--model provider/model] [--max-turns 40]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--job-id", required=True, help="Live/finished job id.")
    parser.add_argument(
        "--workspace", required=True, help="Absolute path to the client repo."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the JSON verdict (default: <workspace>/../integrity.json). "
        "The markdown report is written next to it.",
    )
    parser.add_argument(
        "--testset-path",
        default=None,
        help="Test-set path for the hard-coding check (repo-relative or "
        "absolute). Falls back to the workspace's "
        ".codeevolver/optimization_config.json when omitted.",
    )
    parser.add_argument("--coding-harness", default="opencode",
                        choices=["opencode", "claude"])
    parser.add_argument("--model", default=None,
                        help="Inspection agent model override.")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument(
        "--history-root",
        default=None,
        help="JSONL history root (default ~/.codeevolver/history).",
    )
    args = parser.parse_args()

    # Environment preamble BEFORE engine imports: safe-mode inspection targets
    # the local research deploy, and the paths accessors must resolve the
    # live run's per-job tree (~/.codeevolver/*/{job_id}).
    os.environ.setdefault("APP_MODE", "local")
    os.environ.setdefault("SANDBOX_MODE", "local")
    os.environ["CODEEVOLVER_JOB_ID"] = args.job_id
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.export_history import reconstruct
    from src.engine.optimizer.meta.run_integrity import run_integrity_inspection

    workspace = os.path.abspath(args.workspace)
    output = os.path.abspath(
        args.output or os.path.join(workspace, os.pardir, "integrity.json")
    )

    history_root = (
        Path(args.history_root)
        if args.history_root
        else Path.home() / ".codeevolver" / "history"
    )
    replay = reconstruct(args.job_id, history_root)
    iteration_indices = sorted(
        int(k) for k in replay["iteration_history"] if str(k).lstrip("-").isdigit()
    )
    progress = replay.get("last_progress") or {}

    testset_path = args.testset_path
    if not testset_path:
        cfg_path = os.path.join(workspace, ".codeevolver", "optimization_config.json")
        try:
            with open(cfg_path) as f:
                testset_path = json.load(f).get("testset_path")
        except (OSError, ValueError):
            testset_path = None

    verdict = run_integrity_inspection(
        workspace=workspace,
        method=str(progress.get("method") or "ee"),
        num_iterations=len(iteration_indices),
        best_valset_score=progress.get("final_train_score"),
        seed_test_score=progress.get("seed_test_score"),
        best_candidate_test_score=progress.get("final_test_score"),
        iteration_indices=iteration_indices,
        testset_path=testset_path,
        safe_mode=True,
        output_json_path=output,
        agent_model=args.model,
        coding_harness=args.coding_harness,
        coding_harness_model=args.model,
        max_turns=args.max_turns,
    )
    if verdict is None:
        print("integrity check produced no verdict (see logs)", file=sys.stderr)
        return 1
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    raise SystemExit(main())
