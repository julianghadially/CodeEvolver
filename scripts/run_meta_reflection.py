#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Offline post-run meta-reflection against a COLLECTED run directory.

The in-run auditor (``optimizer/meta/meta_reflection.py`` for the v3 arms,
``meta/simple_meta_reflection.py`` for the simple loop) runs inside
finalization — so a run that was cancelled, or died before finalization,
never gets one. This script replays that call against the artifacts the
fleet runner leaves behind in ``~/ce-data/runs/<run_id>/`` (the container's
HOME): ``status.json`` / ``settings.json`` / ``history.json`` plus the
``.codeevolver/{memory,architect_traces,traces,history}/<job_id>/`` trees
the auditor's prompt points at.

The client repo must be materialised separately — ``repo.bundle`` carries
every ``codeevolver-*`` branch and the ``refs/codeevolver/*`` anchors::

    git clone <run>/repo.bundle <ws>
    git -C <ws> fetch <run>/repo.bundle 'refs/codeevolver/*:refs/codeevolver/*'
    git -C <ws> checkout <final_candidate.git_branch>

Same prompt, same harness/model as the run (from ``settings.json``), same
``max_turns`` — the output is what finalization would have written to
``{memory_dir()}/meta_reflection.md``; a copy lands at ``--output``
(default ``<run-dir>/meta_reflection.md``, beside ``run_integrity.md``).

Billable: this spawns one live inspection-agent session.

Usage:
    python scripts/run_meta_reflection.py \
        --run-dir ~/ce-data/runs/asa-hover-s1 --workspace /path/to/clone \
        [--coding-harness opencode] [--model provider/model] [--max-turns 40] \
        [--output <run-dir>/meta_reflection.md]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path

logger = logging.getLogger("run_meta_reflection")


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _program_from_workspace_tarball(run_dir: Path) -> str | None:
    """``program`` (dotted entry point) from the archived optimization config.

    Only the simple-loop prompt needs it; the archive is read in memory and
    nothing else from that file (it also holds the job JWT) is touched.
    """
    tarball = run_dir / "workspace.tar.gz"
    if not tarball.exists():
        return None
    try:
        with tarfile.open(tarball) as tf:
            for name in ("./.codeevolver/optimization_config.json",
                         ".codeevolver/optimization_config.json"):
                try:
                    member = tf.extractfile(name)
                except KeyError:
                    continue
                if member is not None:
                    return json.load(member).get("program")
    except (OSError, tarfile.TarError, ValueError) as exc:
        logger.warning("could not read program from %s: %s", tarball, exc)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True,
                        help="Collected run directory (holds status.json, "
                             "history.json, settings.json, .codeevolver/).")
    parser.add_argument("--workspace", required=True,
                        help="Absolute path to the materialised client repo.")
    parser.add_argument("--coding-harness", default=None,
                        choices=["opencode", "claude"],
                        help="Default: settings.json models.coding_harness.")
    parser.add_argument("--model", default=None,
                        help="Harness model. Default: settings.json "
                             "models.coding_harness_model.")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--output", default=None,
                        help="Where to copy the markdown "
                             "(default <run-dir>/meta_reflection.md).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    workspace = os.path.abspath(os.path.expanduser(args.workspace))
    status = _load_json(run_dir / "status.json")
    settings = _load_json(run_dir / "settings.json")
    history_blob = _load_json(run_dir / "history.json")

    job_id = status["job_id"]
    arm = status.get("arm") or settings.get("arm")
    models = settings.get("models") or {}
    coding_harness = args.coding_harness or models.get("coding_harness") or "opencode"
    coding_harness_model = args.model or models.get("coding_harness_model")
    budget = settings.get("budget") or {}

    # Environment preamble BEFORE any engine import: the paths accessors must
    # resolve to this run's artifact trees, job-scoped exactly as in-run.
    ce_dir = run_dir / ".codeevolver"
    os.environ.setdefault("APP_MODE", "local")
    os.environ.setdefault("SANDBOX_MODE", "local")
    os.environ["CODEEVOLVER_JOB_ID"] = job_id
    os.environ["CODEEVOLVER_MEMORY_DIR"] = str(ce_dir / "memory")
    os.environ["CODEEVOLVER_ARCHITECT_TRACES_DIR"] = str(ce_dir / "architect_traces")
    os.environ["CODEEVOLVER_TRACES_DIR"] = str(ce_dir / "traces")
    os.environ["CODEEVOLVER_HISTORY_DIR"] = str(ce_dir / "history")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.engine.paths import architect_traces_dir, memory_dir

    if not os.path.isdir(architect_traces_dir()):
        logger.error("no architect traces at %s", architect_traces_dir())
        return 2
    if not os.path.isdir(os.path.join(workspace, ".git")):
        logger.error("%s is not a git checkout", workspace)
        return 2

    history = {
        int(k): (v or {})
        for k, v in (history_blob.get("iteration_history") or {}).items()
        if str(k).lstrip("-").isdigit()
    }
    final = status.get("final_candidate") or {}
    logger.info(
        "run %s (%s, %s): %d history records, status=%s, harness=%s model=%s",
        status.get("run_id"), arm, job_id, len(history), status.get("status"),
        coding_harness, coding_harness_model,
    )

    if arm == "simple_loop":
        from src.engine.optimizer.meta.simple_meta_reflection import (
            run_simple_loop_meta_reflection,
        )

        markdown = run_simple_loop_meta_reflection(
            workspace=workspace,
            history=history,
            branch=final.get("git_branch"),
            program_path=_program_from_workspace_tarball(run_dir),
            seed_test_score=status.get("seed_test_score"),
            final_test_score=status.get("final_test_score"),
            subsample_size=budget.get("subsample_size"),
            max_eval_rows=budget.get("max_metric_calls"),
            total_evals=status.get("total_metric_calls"),
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
        )
    else:
        from src.engine.optimizer.meta.meta_reflection import run_meta_reflection

        markdown = run_meta_reflection(
            workspace=workspace,
            method=arm,
            num_iterations=len(history),
            seed_test_score=status.get("seed_test_score"),
            best_candidate_test_score=status.get("final_test_score"),
            iteration_indices=sorted(history),
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
        )

    if not markdown:
        print("meta-reflection produced no output (see logs)", file=sys.stderr)
        return 1

    written = os.path.join(memory_dir(), "meta_reflection.md")
    output = os.path.abspath(args.output or str(run_dir / "meta_reflection.md"))
    if os.path.exists(written) and os.path.abspath(written) != output:
        shutil.copyfile(written, output)
    else:
        with open(output, "w") as f:
            f.write(markdown)
    print(f"meta-reflection written to {written}\ncopied to {output}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    raise SystemExit(main())
