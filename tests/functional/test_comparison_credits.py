"""Functional test: a REAL architect completes the comparison-credit
acknowledgement round-trip through the REAL `run_eval` tool.

Every deterministic test of the ledger (`tests/unit/test_comparison_credits.py`)
proves the mechanism: the first comparison call returns a structured refusal
carrying a per-iteration verification code, and a re-call with
`acknowledge_message=<code>` runs the eval and debits a credit. What no
offline test can prove is that a real model, reading only the tool
description and that refusal, actually re-calls with the code instead of
giving up or re-calling without it. That is the model-behavior risk the
code-based acknowledgement was chosen to manage (over free-text Y/N), and it
is what this test measures — on whichever harness/model it is run with.

Nothing in the tool chain is replaced. The architect is the production
`IterationArchitect` with its production `build_tool_registry` (so `run_eval`
is the real `RunEvalBinding`, `close_iteration` the real close tool, and the
production watchdogs run). Only `run_eval_tool._run_eval` — the evaluator
subprocess — is patched, so no client program executes. The ledger is seeded
with exactly ONE credit and the iteration's ack code is fixed.

Protocol (via additional_instructions; the code is NOT mentioned anywhere the
agent can read — it only ever appears in the tool's refusal):
  1. `run_eval(eval_sample="train")` once (baseline).
  2. Run the comparison set and obtain its score.
  3. No code changes; update memory.md briefly; close the iteration.

Success checks (see ``comparison_credits_failures``):
  - the acknowledgement refusal was returned at least once (the model did
    not somehow know the code up front);
  - the comparison eval EXECUTED exactly once (one patched `_run_eval` call
    with the comparison suffix), i.e. the model completed the round-trip;
  - `ctx.comparison_ack_given` — only a correct code sets it;
  - the ledger balance went 1 -> 0 and the response carried it;
  - the real close tool validated the close.

To run::

    python local_test_app.py comparison_credits [--coding-harness opencode ...]
"""

from __future__ import annotations

import os
import textwrap

from tests._test_env import (
    SyntheticTestEnvironment,
    git_fixture_snippet,
    parse_marker,
)

_PROVIDER_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPINFRA_API_KEY")


def _provider_env_keys() -> dict[str, str]:
    return {k: os.environ[k] for k in _PROVIDER_ENV_VARS if os.environ.get(k)}


# --- Shared fixtures ----------------------------------------------------

ITERATION_NUM = 3
CREDITS_BRANCH = "codeevolver-20260831000000-credits-test"
# Fixed for reproducibility of the assertions; production draws it from the
# run rng at iteration start.
ACK_CODE = "73"

INITIAL_MEMORY_MD = f"""\
# CodeEvolver Memory

Seed branch: {CREDITS_BRANCH}
Seed architecture: Baseline FactChecker

This file is maintained by YOU, the architect. Update it at the end of every
iteration (see the `close_iteration` skill).

## Candidate Pool
- {CREDITS_BRANCH} (PRIMARY): seed program

## Overall Progress
- Iteration 0 (Seed Program): baseline established.

## Observations
No observations yet.
"""

STUB_CODEEVOLVER_MD = (
    "# Architecture: Baseline FactChecker\n"
    "\n"
    "Stub file for the comparison-credits functional test.\n"
)


def _seed_synthetic_fixtures(env: SyntheticTestEnvironment, provider_keys: dict) -> None:
    env.seed_file(
        f"{env.workspace_dir}/.env",
        "".join(f"{k}={v}\n" for k, v in provider_keys.items()),
    )
    env.seed_file(
        f"{env.workspace_dir}/.gitignore",
        ".claude/\n.env\n.opencode/\n",
    )
    env.seed_file(f"{env.workspace_dir}/codeevolver.md", STUB_CODEEVOLVER_MD)
    env.seed_file(f"{env.memory_dir}/memory.md", INITIAL_MEMORY_MD)


# --- additional_instructions --------------------------------------------
# Deliberately says nothing about credits, codes or acknowledgement: the
# only place the model can learn the mechanism is the tool description and
# the tool's own refusal. That is the condition under test.

ADDITIONAL_INSTRUCTIONS = f"""\
This is an automated integration test of the `run_eval` tool. The evaluator
is stubbed (scores are synthetic). Follow this protocol EXACTLY:

1. Call `run_eval(eval_sample="train")` once, as the baseline.
2. Run the comparison set with `run_eval(eval_sample="comparison")` and
   obtain its primary_score. Read every tool response carefully and do
   whatever the tool tells you is needed to get the comparison score. Do not
   give up until you either have a comparison primary_score or the tool
   refuses with an error saying no further comparison runs are possible.
3. Do NOT call `coding_agent`, do NOT call `inspect_traces`, do NOT edit or
   commit any program files, and stay on the current git branch.
4. Update memory.md with a two-line note recording the baseline and
   comparison scores you observed, then load the `close_iteration` skill and
   call the `close_iteration` tool with:
   register_branch = {{"branch": "{CREDITS_BRANCH}", "was_primary": true,
   "is_primary": true, "promotion_evidence": "n/a"}},
   improvement_mode="exploitation", memory_updated=true.
"""


# --- Driver script (runs inside the environment) ------------------------

def _build_driver_script(
    *,
    engine_root: str,
    workspace: str,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> str:
    """In-environment driver: real git fixture, PRODUCTION IterationArchitect
    with its production tool registry, `_run_eval` patched, one credit."""
    git_fixture = git_fixture_snippet(
        branch=CREDITS_BRANCH,
        message='seed: comparison-credits test fixture',
    )
    return textwrap.dedent(
        f"""\
        import json, os, subprocess, sys
        sys.path.insert(0, {engine_root!r})
        os.chdir({engine_root!r})
        os.environ.setdefault("IS_SANDBOX", "1")

        from src.engine.optimizer.v3 import eval_budget, run_eval_tool
        from src.engine.optimizer.v3.ce_state import CEState
        from src.engine.optimizer.v3.iteration_architect import (
            ArchitectConfig, IterationArchitect, hash_file, memory_md_path,
        )
        from src.engine.optimizer.v3.iteration_context import IterationContext
        from src.engine.evaluator.config import EvaluatorConfig
        from src.engine.evaluator.sandbox_client import EvalResult
        from src.engine.paths import history_dir
        from src.engine.services.subprocess_service import read_file

        ITERATION_NUM = {ITERATION_NUM}
        ACK_CODE = {ACK_CODE!r}
        CREDITS_BRANCH = {CREDITS_BRANCH!r}
        ADDITIONAL_INSTRUCTIONS = {ADDITIONAL_INSTRUCTIONS!r}
        WORKSPACE = {workspace!r}

        # --- Tiny REAL git repo: initial commit + codeevolver- branch ---
        {git_fixture}

        # --- The ONLY patch: the evaluator subprocess. Everything the
        # architect touches (RunEvalBinding, close tool, watchdogs) is real.
        eval_calls = []

        def _fake_run_eval(*, workspace, config, row_ids, iteration_idx,
                           suffix, capture_traces=True, timeout=None):
            eval_calls.append({{"suffix": suffix, "n_rows": len(row_ids)}})
            score = 0.30 if suffix == "parent" else 0.55
            return EvalResult(
                scores_by_metric={{"accuracy": score}},
                primary_score=score,
                trace_meta=[
                    {{"example_id": r, "score": score, "failed": False,
                      "feedback": None}}
                    for r in row_ids
                ],
                trace_dir=f"{{WORKSPACE}}/.codeevolver/traces/iteration_{{iteration_idx}}_{{suffix}}",
                trace_index_file=f"{{WORKSPACE}}/.codeevolver/traces/iteration_{{iteration_idx}}_{{suffix}}/index.jsonl",
                row_ids=list(row_ids),
                raw={{}},
            )

        run_eval_tool._run_eval = _fake_run_eval

        # Count refusals without altering them: wrap the pure helper that
        # builds the acknowledgement response.
        refusals = []
        _real_refusal = eval_budget.acknowledgement_refusal

        def _counting_refusal(**kw):
            resp = _real_refusal(**kw)
            refusals.append({{"code": resp["verification_code"],
                              "balance": resp["comparison_credits_remaining"]}})
            return resp

        eval_budget.acknowledgement_refusal = _counting_refusal

        state = CEState(
            seed_branch=CREDITS_BRANCH,
            seed_sha=START_SHA,
            comparison_history_path=os.path.join(
                history_dir(), "comparison_set_history.jsonl"
            ),
        )
        state.grant_comparison_credits(1)   # exactly one credit

        ctx = IterationContext(iteration_idx=ITERATION_NUM)
        ctx.parent_locked = True
        ctx.start_branch = CREDITS_BRANCH
        ctx.start_sha = START_SHA
        ctx.primary_branch_at_start = CREDITS_BRANCH
        ctx.memory_md_hash_at_start = hash_file(memory_md_path())
        ctx.subsample_size = 3
        ctx.comparison_set_size = 3
        ctx.max_eval_rows = 100
        ctx.comparison_ack_code = ACK_CODE

        evaluator = EvaluatorConfig(
            program_factory="stub_pkg.stub_mod:stub_factory",
            metric_fns=["stub_pkg.stub_mod:stub_metric"],
            metric_names=["accuracy"],
            primary_metric="accuracy",
            trainset_path=f"{{WORKSPACE}}/.codeevolver/trainset.json",
        )

        cfg = ArchitectConfig(
            run_timestamp="20260831000000",
            additional_instructions=ADDITIONAL_INSTRUCTIONS,
            program_path=None,
            evaluator=evaluator,
            trainset_ids=[0, 1, 2],
            comparison_ids=[0, 1, 2],
            subsample_size=3,
            max_turns=60,
            coding_harness={coding_harness!r},
            coding_harness_model={coding_harness_model!r},
        )

        # PRODUCTION architect + PRODUCTION registry — nothing subclassed.
        architect = IterationArchitect(
            context=ctx,
            state=state,
            workspace=WORKSPACE,
            config=cfg,
        )
        import traceback as _traceback
        run_error = None
        try:
            architect.run()
        except Exception as exc:  # noqa: BLE001
            run_error = f"{{type(exc).__name__}}: {{exc}}"
            _traceback.print_exc()

        tool_calls = architect._session_tool_calls or []
        comparison_calls = [
            (c.get("input") or {{}})
            for c in tool_calls
            if str(c.get("name", "")).endswith("run_eval")
            and (c.get("input") or {{}}).get("eval_sample") == "comparison"
        ]
        memory_md = read_file(memory_md_path()) or ""

        payload = {{
            "refusals": refusals,
            "comparison_evals_run": sum(
                1 for c in eval_calls if c["suffix"] == "comparison"
            ),
            "train_evals_run": sum(
                1 for c in eval_calls if c["suffix"] != "comparison"
            ),
            "comparison_records": sum(
                1 for r in ctx.eval_records if r.get("eval_sample") == "comparison"
            ),
            "ack_given": bool(ctx.comparison_ack_given),
            "credits_balance": state.comparison_credits_balance,
            "comparison_tool_calls": comparison_calls,
            "tool_calls_captured": architect._session_tool_calls is not None,
            "close_called": bool(ctx.close_called),
            "watchdog_fired": bool(getattr(ctx, "watchdog_fired", False)),
            "memory_changed": (
                hash_file(memory_md_path()) != ctx.memory_md_hash_at_start
            ),
            "memory_md": memory_md,
            "run_error": run_error,
        }}
        print("CREDITS_ACK_RESULT:" + json.dumps(payload))
        """
    )


# --- Pass/fail criteria -------------------------------------------------

# Model-driven; a model may occasionally abandon the comparison after the
# refusal. Retry a bounded number of times before calling it a failure.
CREDITS_MAX_ATTEMPTS = 3


def comparison_credits_failures(result: dict) -> list[str]:
    """Return the list of assertion failures for one run (empty == pass)."""
    failures: list[str] = []
    if result.get("run_error"):
        failures.append(f"architect.run() raised: {result['run_error']}")
    if result.get("train_evals_run", 0) < 1:
        failures.append("no baseline train eval ran — protocol step 1 missed.")
    if not result.get("refusals"):
        failures.append(
            "the acknowledgement refusal was never returned — the model "
            "either never attempted the comparison set, or supplied the code "
            "up front (impossible unless leaked)."
        )
    n_cmp = result.get("comparison_evals_run", 0)
    if n_cmp != 1:
        failures.append(
            f"comparison eval executed {n_cmp} times (expected exactly 1) — "
            + ("the model never completed the acknowledgement round-trip."
               if n_cmp == 0 else "a second run should have been refused.")
        )
    if not result.get("ack_given"):
        failures.append(
            "ctx.comparison_ack_given is False — the model never re-called "
            "run_eval with the correct acknowledge_message."
        )
    if result.get("credits_balance") != 0:
        failures.append(
            f"ledger balance is {result.get('credits_balance')} (expected 0: "
            "one credit granted, one debited)."
        )
    if result.get("comparison_records", 0) != 1:
        failures.append(
            f"{result.get('comparison_records')} comparison records on the "
            "context (expected 1)."
        )
    if result.get("tool_calls_captured"):
        calls = result.get("comparison_tool_calls") or []
        if not any(str(c.get("acknowledge_message", "")).strip() == ACK_CODE for c in calls):
            failures.append(
                "no run_eval(comparison) tool call carried the verification "
                f"code {ACK_CODE!r} — the model did not pass the code back."
            )
    if not result.get("close_called"):
        failures.append("close_iteration was never (successfully) called (retryable).")
    return failures


def summarize_result(result: dict) -> str:
    return (
        f"refusals={len(result.get('refusals') or [])} "
        f"comparison_evals_run={result.get('comparison_evals_run')} "
        f"ack_given={result.get('ack_given')} "
        f"credits_balance={result.get('credits_balance')} "
        f"close_called={result.get('close_called')} "
        f"error={result.get('error') or result.get('run_error')}"
    )


# --- Runner ------------------------------------------------------------

def run_comparison_credits(
    env: SyntheticTestEnvironment,
    *,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> dict:
    """Run the comparison-credits acknowledgement test in `env`."""
    provider_keys = _provider_env_keys()

    try:
        _seed_synthetic_fixtures(env, provider_keys)

        driver = _build_driver_script(
            engine_root=env.engine_root,
            workspace=env.workspace_dir,
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
        )
        stdout_lines: list[str] = []
        for line in env.exec_python_driver(driver, env_vars=provider_keys):
            print(line, flush=True)
            stdout_lines.append(line)

        payload = parse_marker(stdout_lines, "CREDITS_ACK_RESULT")

        if payload is None:
            tail = "\n".join(stdout_lines[-40:])
            return {
                "refusals": [],
                "comparison_evals_run": 0,
                "train_evals_run": 0,
                "comparison_records": 0,
                "ack_given": False,
                "credits_balance": None,
                "comparison_tool_calls": [],
                "tool_calls_captured": False,
                "close_called": False,
                "watchdog_fired": False,
                "memory_changed": False,
                "memory_md": "",
                "run_error": None,
                "error": (
                    f"No CREDITS_ACK_RESULT marker on stdout. "
                    f"Last stdout lines:\n{tail}"
                ),
            }

        payload.setdefault("error", payload.get("run_error"))
        return payload
    finally:
        env.cleanup()
