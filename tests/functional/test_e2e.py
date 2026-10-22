"""Shared end-to-end optimizer + eval-trace test logic.

This module is strategy-agnostic: ``ee``, ``pareto_evolution`` and
``simple_loop`` all run through the same driver and differ only by which
``optimization_entry.py`` gets exec'd. The per-strategy names live on the
`local_test_app.py` subcommands (``ee_e2e`` / ``pareto_e2e`` /
``simple_loop_e2e``), not here.

Two functions live here:

* ``run_e2e(env, ...)`` — the full optimizer loop test. Sets up a
  client repo (cached by `LocalClientRepoEnv`), writes an optimization
  config, exec's
  ``optimization_entry.py`` with ``debug_max_iterations`` to bound run
  time, and parses the ``OPTIMIZATION_RESULT:`` marker.

* ``run_eval_trace(env, ...)`` — wraps ``run_e2e`` with
  ``debug_max_iterations=0`` (seed-only) and reads
  ``/traces/iteration_0_validate_env.jsonl`` from the env (the
  environment-validation smoke eval is the seed-time trace producer, fixed
  at 5 training rows). Tests OTel
  trace collection without running the optimization loop.

Both functions take a ``ClientRepoTestEnvironment`` (see
``tests/_test_env.py``); the entrypoints in ``local_test_app.py`` call them.
"""
from __future__ import annotations

import json
import os

from tests._test_env import ClientRepoTestEnvironment, parse_marker

# The e2e seeds from a dedicated orphan branch on the fixture repo rather than
# `main`. `main` git-tracks prior optimization output (`results/optimization/
# optimized_program_*.json`, graded `results/evaluation_*.json`) which stays
# recoverable through history — that is real contamination of the run's own
# results, not merely something an inspector objects to, so the branch choice
# stands whether or not the clean-slate inspector is armed (it is off by
# default locally; see `run_e2e`). `clean-slate` is a single-commit orphan
# branch holding only the seed program, the trainset and the fenced holdout.
DEFAULT_SEED_BRANCH = "clean-slate"

# Passed so the holdout lands in `holdout_deny_paths` (v3/seed.py) — that is
# what makes it legible to the clean-slate inspector as *fenced* rather than
# as unguarded test data — and so the run reports real held-out scores.
# The NORMALIZED holdout (`statement` / SUPPORTED|REFUTED — the seed program's
# entry point takes `statement`). The raw `data/FacTool_QA_test.jsonl` is keyed
# `claim` and every row raises `missing ... 'statement'` → seed test = 0.0.
DEFAULT_TESTSET_PATH = "data/FacTool_QA_test_normalized.jsonl"


def run_e2e(
    env: ClientRepoTestEnvironment,
    *,
    repo_url: str = "https://github.com/julianghadially/FactChecker",
    program: str = "src.factchecker.simple.factchecker_program:FactCheckerProgram",
    metric: str = "src.codeevolver.metric:metric",
    trainset_path: str = "data/FacTool_QA_train_10examples_normalized.jsonl",
    testset_path: str | None = DEFAULT_TESTSET_PATH,
    initial_branch: str = DEFAULT_SEED_BRANCH,
    max_iterations: int = 2,
    comparison_set_size: int | None = None,
    primary_metric: str = "accuracy",
    num_threads: int = 1,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
    github_token: str | None = None,
    enable_meta_reflection: bool = True,
    enable_run_integrity_agent: bool = True,
    enable_clean_slate_inspector: bool = False,
    method: str = "ee",
    max_metric_calls: int | None = None,
) -> dict:
    """Run the optimizer end-to-end against ``repo_url`` and return parsed result.

    The result dict carries the optimizer's ``OPTIMIZATION_RESULT`` payload
    plus two extra fields the trace-collection test depends on:

    - ``trace_file_path`` — the seed environment-validation trace file
      (suffix ``validate_env``, iteration 0) resolved against
      ``env.traces_dir`` (the durable per-run dir locally).
    - ``trace_file_content`` — the JSONL contents of that file, or ``""``.

    Meta-reflection and run-integrity default ON, matching the flags
    ``runner/entrypoint.py::RESEARCH_DEFAULTS`` pins for a real research run —
    an e2e that leaves them off never exercises the paths a real run always
    takes. The clean-slate inspector defaults OFF: this driver also runs on a
    developer's host, where the cached clone and the artifacts of every past
    e2e run are shared and reachable, so the inspector would report on other
    runs (a `contaminated` verdict aborts the run before any eval spend).
    ``local_test_app.py`` passes the value explicitly and leaves it off
    behind a ``--clean-slate`` opt-in. Outcomes are summarised into the
    payload by ``_summarize_post_run_agents``; the caller decides what is fatal.
    """
    env.prepare_workspace(
        repo_url=repo_url, branch=initial_branch, github_token=github_token,
        coding_harness=coding_harness,
    )

    try:
        config = {
            "workspace": env.workspace_dir,
            "program": program,
            "metric": metric,
            "trainset_path": trainset_path,
            # When set, the optimizer scores the seed + final branch on this
            # held-out test set at finalization (seed_test_score /
            # final_test_score in the result) and locks it from agents.
            "testset_path": testset_path,
            "reflection_lm": "anthropic/claude-opus-4-7",
            "coding_harness": coding_harness,
            "coding_harness_model": coding_harness_model,
            "num_threads": num_threads,
            "seed": 0,
            # Engine 3: fixed comparison set drawn from the trainset (seed 42),
            # used for apples-to-apples branch comparison. None disables it.
            "comparison_set_size": comparison_set_size,
            "debug_max_iterations": max_iterations,
            "comparison_eval_timeout": 3600,
            "metric_names": [primary_metric],
            "primary_metric": primary_metric,
            # Capture architect-session traces during the real loop and run the
            # post-finalization meta-reflection agent.
            "enable_meta_reflection": enable_meta_reflection,
            # Post-run integrity inspection (advisory everywhere: the local
            # entrypoint reports its verdict and fails only on a MISSING
            # verdict, or on cheating under `--fail-on-cheating`) and the
            # start-of-run clean-slate inspector, which ABORTS the run on a
            # contaminated verdict.
            "enable_run_integrity_agent": enable_run_integrity_agent,
            "enable_clean_slate_inspector": enable_clean_slate_inspector,
        }
        # Scopes every artifact path to this run: `v3/_entry.py` exports it as
        # CODEEVOLVER_JOB_ID and `paths._job_subdir()` appends it to the
        # memory / traces / architect-traces dirs, while `logs_dir()` uses it
        # as a filename prefix. `LocalClientRepoEnv` sets the identical value
        # on this process so the test resolves the same paths back.
        if env.test_id:
            config["job_id"] = env.test_id
        if max_metric_calls is not None:
            # simple_loop is bounded by eval rows, not iterations.
            config["max_metric_calls"] = max_metric_calls
        config_json = json.dumps(config)
        config_path = "/tmp/optimization_config.json"
        env.write_file(config_path, config_json)

        # Container paths: /app/src/...; LocalClientRepoEnv rewrites the /app/ prefix
        # to the auto-detected engine root from `src.engine.paths`.
        _ENTRY_SCRIPTS = {
            "ee": "/app/src/engine/optimizer/v3/ee/optimization_entry.py",
            "pareto_evolution": "/app/src/engine/optimizer/v3/pareto_evolution/optimization_entry.py",
            "freeform": "/app/src/engine/optimizer/v3/freeform/optimization_entry.py",
            "asa": "/app/src/engine/optimizer/v3/asa/optimization_entry.py",
            "simple_loop": "/app/src/engine/optimizer/simple_loop/optimization_entry.py",
        }
        if method not in _ENTRY_SCRIPTS:
            raise ValueError(
                f"unknown method {method!r}; expected one of {sorted(_ENTRY_SCRIPTS)}"
            )
        entry_script = _ENTRY_SCRIPTS[method]

        result_payload: dict | None = None
        stdout_lines: list[str] = []
        for line in env.exec_optimizer(entry_script, config_path):
            print(line, flush=True)
            stdout_lines.append(line)

        result_payload = parse_marker(stdout_lines, "OPTIMIZATION_RESULT")
        if result_payload is None:
            result_payload = {
                "success": False,
                "error": (
                    f"No OPTIMIZATION_RESULT marker on stdout "
                    f"(test_id={env.test_id}). Last 40 lines:\n"
                    + "\n".join(stdout_lines[-40:])
                ),
            }

        # Engine 3: the seed valset eval is gone; the seed-time trace producer
        # is the environment-validation smoke eval (verify_environment.py,
        # suffix "validate_env", iteration 0, first 5 training rows).
        trace_path = os.path.join(env.traces_dir, "iteration_0_validate_env")
        result_payload["trace_file_path"] = trace_path
        result_payload["trace_file_content"] = _trace_dir_as_jsonl(trace_path)

        _summarize_post_run_agents(
            env,
            result_payload,
            enable_meta_reflection=enable_meta_reflection,
            enable_run_integrity_agent=enable_run_integrity_agent,
            enable_clean_slate_inspector=enable_clean_slate_inspector,
        )
        result_payload["test_id"] = env.test_id
        result_payload["artifact_root"] = env.artifact_root
        result_payload["run_log_path"] = getattr(env, "run_log_path", None)
        return result_payload
    finally:
        env.cleanup()


# ---------------------------------------------------------------------------
# Post-run agent outcomes
# ---------------------------------------------------------------------------


def _summarize_post_run_agents(
    env: ClientRepoTestEnvironment,
    payload: dict,
    *,
    enable_meta_reflection: bool,
    enable_run_integrity_agent: bool,
    enable_clean_slate_inspector: bool,
) -> None:
    """Fold the run's own self-checks into ``payload``, in place.

    Nothing here needs engine plumbing: the optimizer already stamps every
    signal onto the iteration history it returns — `clean_slate_status` on the
    seed record (`v3/seed.py`), `constraint_violation` per iteration
    (`v3/optimizer_loop.py`, `simple_loop/records.py`), and the `integrity_*`
    keys onto the last record (`meta/run_integrity.py::stamp_integrity_verdict`).
    We read them back out and locate the human-readable reports on disk.
    """
    history = payload.get("history") or {}
    # The marker payload round-trips through JSON, so iteration indices arrive
    # as strings; sort numerically where the key allows it.
    records = sorted(
        ((str(k), v) for k, v in history.items() if isinstance(v, dict)),
        key=lambda kv: int(kv[0]) if kv[0].lstrip("-").isdigit() else 0,
    )

    if enable_meta_reflection:
        # Meta-reflection writes {memory_dir}/meta_reflection.md at the very
        # end (after finalization). Capture happens during the real iterations
        # into {architect_traces_dir}/iteration_{N}_architect_trace/.
        mr_path = os.path.join(env.memory_dir, "meta_reflection.md")
        mr_content = env.read_file(mr_path) or ""
        payload["meta_reflection_path"] = mr_path
        payload["meta_reflection_written"] = bool(mr_content.strip())
        payload["meta_reflection_len"] = len(mr_content)
        # One trace per iteration that ran. Bound the scan by the highest
        # iteration actually recorded (+2 for the seed and finalization
        # records) rather than by the requested iteration count — the
        # eval-budgeted arms run as many iterations as their rows allow.
        highest = max(
            (int(k) for k, _ in records if k.lstrip("-").isdigit()), default=0
        )
        payload["architect_traces_captured"] = sum(
            1
            for idx in range(0, highest + 2)
            if (
                env.read_file(
                    os.path.join(
                        env.architect_traces_dir,
                        f"iteration_{idx}_architect_trace",
                        "index.jsonl",
                    )
                )
                or ""
            ).strip()
        )

    if enable_run_integrity_agent:
        # The verdict is stamped onto the LAST iteration record.
        last = records[-1][1] if records else {}
        payload["integrity_verdict"] = {
            key: last.get(key)
            for key in (
                "integrity_cheating_category",
                "integrity_cheating_rationale",
                "integrity_functioning_iterations",
                "integrity_pct",
            )
        }
        ri_path = os.path.join(env.memory_dir, "run_integrity.md")
        payload["run_integrity_path"] = ri_path
        payload["run_integrity_written"] = bool((env.read_file(ri_path) or "").strip())

    if enable_clean_slate_inspector:
        # Stamped on the seed record. Absent for arms that do not run the
        # inspector (simple_loop has its own repository init and skips it) —
        # `None` here means "not run", which is not a failure.
        seed_record = records[0][1] if records else {}
        payload["clean_slate_status"] = seed_record.get("clean_slate_status")
        cs_path = os.path.join(
            env.logs_dir, f"{env.test_id}_clean_slate_report.md"
        ) if env.test_id else None
        payload["clean_slate_report_path"] = cs_path
        payload["clean_slate_report_written"] = bool(
            cs_path and (env.read_file(cs_path) or "").strip()
        )

    # Constraint gates fire mid-run and force-revert the iteration; every arm
    # records the reason on the iteration it fired for.
    payload["constraint_violations"] = [
        {"iteration": key, "reason": rec.get("constraint_violation")}
        for key, rec in records
        if rec.get("constraint_violation")
    ]


def run_eval_trace(
    env: ClientRepoTestEnvironment,
    *,
    repo_url: str = "https://github.com/julianghadially/FactChecker",
    program: str = "src.factchecker.simple.factchecker_program:FactCheckerProgram",
    trainset_path: str = "data/FacTool_QA_train_10examples_normalized.jsonl",
    num_threads: int = 1,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
    initial_branch: str = DEFAULT_SEED_BRANCH,
    enable_clean_slate_inspector: bool = False,
) -> dict:
    """Seed-only run; surfaces ``trace_file_content`` for OTel assertions.

    Returns the ``run_e2e`` payload with ``trace_file_content`` populated.
    The caller asserts on pipeline-root spans / judge-like signatures.

    The trace under test is the environment-validation smoke eval (suffix
    ``validate_env``, iteration 0). Its row count is fixed by the seed step
    (first 5 training rows); there is no ``max_rows`` knob.

    ``num_threads`` controls evaluator concurrency for the validation eval —
    pass > 1 to validate per-row trace partitioning under threads against the
    real client program.

    ``coding_harness`` is threaded through for signature symmetry with
    ``run_e2e``; the seed-only path does not run the architect, so the
    harness is never actually exercised here.

    The post-run agents are OFF: this path exists to check trace collection
    cheaply, and meta-reflection / run-integrity have nothing to reflect on
    when no iteration ran. Clean-slate is opt-in — useful for validating a
    fresh seed branch before a real run, and skipped otherwise.
    """
    return run_e2e(
        env,
        repo_url=repo_url,
        program=program,
        trainset_path=trainset_path,
        initial_branch=initial_branch,
        max_iterations=0,
        num_threads=num_threads,
        coding_harness=coding_harness,
        coding_harness_model=coding_harness_model,
        enable_meta_reflection=False,
        enable_run_integrity_agent=False,
        enable_clean_slate_inspector=enable_clean_slate_inspector,
    )


# ---------------------------------------------------------------------------
# Trace-collection assertions (used by the local entrypoints)
# ---------------------------------------------------------------------------


def _trace_dir_as_jsonl(trace_dir: str) -> str:
    """Flatten an eval trace directory into one JSONL row per example.

    An eval writes `index.jsonl` plus `rows/{example_id}/span_{k}.json`
    (`src/engine/evaluator/mounted/evaluate.py`). Each output line is
    `{"example_id": ..., "trace": [span, ...]}`, the shape
    `evaluate_trace_collection` reads. Returns "" when the directory is absent.
    """
    index = os.path.join(trace_dir, "index.jsonl")
    if not os.path.isfile(index):
        return ""
    lines = []
    with open(index, encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            entry = json.loads(raw)
            row_dir = os.path.join(
                trace_dir, os.path.dirname(entry.get("row_file") or f"rows/{entry['example_id']}/row.json")
            )
            spans = []
            if os.path.isdir(row_dir):
                for name in sorted(os.listdir(row_dir)):
                    if name.startswith("span_") and name.endswith(".json"):
                        with open(os.path.join(row_dir, name), encoding="utf-8") as sf:
                            spans.append(json.load(sf))
            lines.append(json.dumps({"example_id": entry.get("example_id"), "trace": spans}))
    return "\n".join(lines)


def evaluate_trace_collection(trace_content: str) -> dict:
    """Inspect a trace JSONL string; return a pass/fail report.

    Counts pipeline-root spans and judge-like signatures; flags missing or
    partial instrumentation. Returns a dict with ``failures`` (list of
    human-readable messages), ``pass`` (bool), and a few stats fields.
    """
    rows: list[dict] = []
    for line in trace_content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    pipeline_roots = 0
    judge_entries: list[dict] = []
    distinct_keys: set[str] = set()
    for row in rows:
        for entry in (row.get("trace") or []):
            sig = entry.get("signature_key") or ""
            distinct_keys.add(sig)
            if sig == "pipeline" and entry.get("parent_key") is None:
                pipeline_roots += 1
            if "judge" in sig.lower():
                judge_entries.append(entry)

    def _instrumented(entry: dict) -> bool:
        inputs = entry.get("inputs") or {}
        output = entry.get("output")
        inputs_ok = any(
            isinstance(v, str) and len(v.strip()) >= 10 for v in inputs.values()
        )
        output_ok = isinstance(output, str) and len(output.strip()) >= 10
        return inputs_ok and output_ok

    instrumented_judge = [e for e in judge_entries if _instrumented(e)]

    failures: list[str] = []
    if not rows:
        failures.append("no trace rows found in file")
    elif pipeline_roots != len(rows):
        failures.append(
            f"expected {len(rows)} pipeline root spans, got {pipeline_roots}"
        )
    if not judge_entries:
        failures.append(
            "no span with 'judge' in signature_key — client repo is likely "
            "not OTel-instrumented."
        )
    if judge_entries and not instrumented_judge:
        failures.append(
            "judge span exists but inputs/output are empty or <10 chars — "
            "instrumentation is partial."
        )

    return {
        "pass": not failures,
        "failures": failures,
        "row_count": len(rows),
        "pipeline_roots": pipeline_roots,
        "judge_entries": len(judge_entries),
        "instrumented_judge": len(instrumented_judge),
        "distinct_signature_keys": sorted(distinct_keys),
    }
