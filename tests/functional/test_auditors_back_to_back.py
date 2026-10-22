"""Functional test: meta-reflection then run-integrity, back-to-back, one process.

The production shape at finalization (`optimizer_loop.py`: `run_post_run_meta_reflection`
immediately followed by `run_post_run_integrity`, same process, no delay). The
existing G5 / H4 tests each run their auditor ALONE, so they cannot catch a
failure that only appears when the two share a process (e.g. a shared
OpenCode SQLite store; see specs/engine3.md). This test asserts both sessions
produce output when run in sequence. Outside the Docker runner there is no root /
ce-agent split, so it exercises the per-session data-dir + retry plumbing —
the DAC case itself is verified in the Docker runner.

Expensive; run manually (billable agent calls — two sessions).

To run::

    python local_test_app.py auditors_back_to_back --coding-harness opencode
"""

from __future__ import annotations

import os
import textwrap

from tests._test_env import SyntheticTestEnvironment, parse_marker, seed_architect_trace_dir
from tests.functional.test_meta_reflection_agent import (
    COMBINED_CHANGE_SUMMARIES,
    INITIAL_MEMORY_MD,
    ITERATION_INDICES,
    _build_clean_architect_trace,
    _build_nugget_architect_trace,
    _provider_env_keys,
)

MARKER = "AUDITORS_BACK_TO_BACK_RESULT"


def _build_driver_script(
    *,
    engine_root: str,
    workspace: str,
    coding_harness: str,
    coding_harness_model: str | None,
) -> str:
    return textwrap.dedent(
        f"""\
        import json, os, sys, traceback
        sys.path.insert(0, {engine_root!r})
        os.chdir({engine_root!r})
        os.environ.setdefault("IS_SANDBOX", "1")

        from src.engine.optimizer.meta.meta_reflection import run_meta_reflection
        from src.engine.optimizer.meta.run_integrity import run_integrity_inspection
        from src.engine.paths import memory_dir

        harness = dict(
            agent_model=None,
            coding_harness={coding_harness!r},
            coding_harness_model={coding_harness_model!r},
        )
        payload = {{"meta_reflection": None, "run_integrity": None, "run_error": None}}
        try:
            # Exactly the finalization order: meta-reflection, then integrity,
            # with nothing in between.
            meta = run_meta_reflection(
                workspace={workspace!r}, method="ee", num_iterations=3,
                seed_test_score=None, best_candidate_test_score=None,
                iteration_indices={ITERATION_INDICES!r}, **harness,
            )
            verdict = run_integrity_inspection(
                workspace={workspace!r}, method="ee", num_iterations=3,
                best_valset_score=None, seed_test_score=None,
                best_candidate_test_score=None,
                iteration_indices={ITERATION_INDICES!r}, testset_path=None,
                safe_mode=False, **harness,
            )
            payload["meta_reflection"] = bool(meta and meta.strip())
            payload["meta_reflection_head"] = (meta or "")[:160]
            payload["run_integrity"] = verdict
        except Exception as exc:  # noqa: BLE001
            payload["run_error"] = f"{{type(exc).__name__}}: {{exc}}"
            traceback.print_exc()
        payload["meta_reflection_file"] = os.path.exists(
            os.path.join(memory_dir(), "meta_reflection.md"))
        payload["run_integrity_file"] = os.path.exists(
            os.path.join(memory_dir(), "run_integrity.md"))
        print({MARKER!r} + ":" + json.dumps(payload))
        """
    )


def run_auditors_back_to_back(
    env: SyntheticTestEnvironment,
    *,
    coding_harness: str = "opencode",
    coding_harness_model: str | None = None,
) -> dict:
    provider_keys = _provider_env_keys()
    try:
        env.seed_file(
            f"{env.workspace_dir}/.env",
            "".join(f"{k}={v}\n" for k, v in provider_keys.items()),
        )
        env.seed_file(
            f"{env.workspace_dir}/codeevolver.md",
            "# Architecture: Baseline FactChecker\nSmall retrieval + synthesis pipeline.\n",
        )
        env.seed_file(f"{env.memory_dir}/memory.md", INITIAL_MEMORY_MD)
        env.seed_file(
            f"{env.memory_dir}/iteration_change_summaries.md", COMBINED_CHANGE_SUMMARIES,
        )
        seed_architect_trace_dir(
            env, f"{env.architect_traces_dir}/iteration_1_architect_trace",
            _build_clean_architect_trace(1),
        )
        seed_architect_trace_dir(
            env, f"{env.architect_traces_dir}/iteration_2_architect_trace",
            _build_nugget_architect_trace(),
        )
        driver = _build_driver_script(
            engine_root=env.engine_root, workspace=env.workspace_dir,
            coding_harness=coding_harness, coding_harness_model=coding_harness_model,
        )
        stdout_lines: list[str] = []
        for line in env.exec_python_driver(driver, env_vars=provider_keys):
            print(line, flush=True)
            stdout_lines.append(line)
        payload = parse_marker(stdout_lines, MARKER)
        if payload is None:
            tail = "\n".join(stdout_lines[-40:])
            return {"error": f"No {MARKER} marker on stdout. Last lines:\n{tail}"}
        payload.setdefault("error", payload.get("run_error"))
        return payload
    finally:
        env.cleanup()


def report_auditors_back_to_back(result: dict) -> int:
    print("\n=== auditors_back_to_back ===")
    for key in ("meta_reflection", "meta_reflection_head", "meta_reflection_file",
                "run_integrity_file", "error"):
        print(f"{key:24s} {result.get(key)}")
    verdict = result.get("run_integrity")
    print(f"{'run_integrity':24s} "
          f"{verdict.get('cheating_category') if isinstance(verdict, dict) else verdict}")
    failures = []
    if not result.get("meta_reflection"):
        failures.append("meta-reflection produced no output")
    if not isinstance(verdict, dict):
        failures.append("run-integrity produced no verdict when run right after meta-reflection")
    elif verdict.get("cheating_category") in (None, "unparsed"):
        failures.append(
            "run-integrity verdict was not parseable "
            f"({verdict.get('cheating_rationale', '')[:160]!r}) -- the session "
            "likely failed and its output was not a report"
        )
    if result.get("error"):
        failures.append(str(result["error"]))
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS: both auditors produced output back-to-back in one process")
    return 0
