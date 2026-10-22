"""Local integration-test entrypoints.

Runs the functional and e2e test bodies on the host machine via the
``Local*Env`` implementations in ``tests/_test_env.py``. Commands (engine 3):

* ``ee_e2e``             — full optimizer loop against FactChecker
* ``pareto_e2e``         — full PARETO_EVOLUTION (fixed evolution) loop against FactChecker
* ``freeform_e2e``       — full FREEFORM (Baseline 1) loop against FactChecker
* ``simple_loop_e2e``    — full SIMPLE LOOP (Baseline 0) run against FactChecker (local only)
* ``eval_trace``         — seed-only run + OTel trace assertion
                           (environment-validation trace)
* ``inspect_traces``  — architect → inspect_traces sub-agent
* ``analysis_agent``  — analysis agent writes iteration_{N}_change_summary.md
                           (and leaves memory.md alone)
* ``close_iteration`` — architect writes its memory.md reflection and calls
                           the real close_iteration tool
* ``architect_trace`` — architect-session trace capture
* ``meta_reflection`` — post-run meta-reflection golden nuggets
* ``holdout_guard``   — armed agent cannot read the held-out test set
* ``test_sampling``   — finalization draws a one-row test set 3x (final), 1x (seed)

The functional commands accept ``--coding-harness {claude,opencode}`` (and
``--coding-harness-model`` for OpenCode) so the run_tests.sh runner sweeps both.

The e2e commands seed from FactChecker's ``clean-slate`` branch (a single-commit
orphan branch holding only the seed program, the trainset and the fenced
holdout — ``main`` git-tracks prior optimization output, which would contaminate
the run's own results), and they arm two of the three inspectors a research run
does: meta-reflection and run-integrity (each with a ``--no-<name>`` form for a
deliberately cheaper one-off). The clean-slate inspector is OFF here and opt-in
via ``--clean-slate``: a local run reuses one cached clone and keeps every past
run's artifacts under ``~/.codeevolver/e2e/``, so what the inspector finds is
other e2e runs rather than anything about this one.

Every e2e run gets a ``test_id`` (override with ``--test-id``), which becomes the
optimizer's ``job_id``, so its artifacts survive under::

    ~/.codeevolver/e2e/logs/{test_id}.log                       full run stdout
    ~/.codeevolver/e2e/logs/{test_id}_detailed_agent_log.log    OpenCode agent detail
    ~/.codeevolver/e2e/logs/{test_id}_clean_slate_report.md
    ~/.codeevolver/e2e/memory/{test_id}/                        memory.md, meta_reflection.md,
                                                                run_integrity.md, iteration history
    ~/.codeevolver/e2e/traces/{test_id}/
    ~/.codeevolver/e2e/architect_traces/{test_id}/

Prerequisites on the host (one-time):

  brew install python@3.11 node deno
  npm install -g @anthropic-ai/claude-code
  brew tap mongodb/brew && brew install mongodb-community@7.0
  brew services start mongodb-community@7.0
  git clone https://github.com/<your>/CodeEvolver ~/.codeevolver/app
  cd ~/.codeevolver/app && python3.11 -m venv env && env/bin/pip install -e .[dev]

Required env vars per command (the runner fails fast if missing):

  ee_e2e / eval_trace  : ANTHROPIC_API_KEY, OPENAI_API_KEY
  inspect_traces    : ANTHROPIC_API_KEY
  analysis_agent    : ANTHROPIC_API_KEY
  close_iteration   : ANTHROPIC_API_KEY
  session_resume    : ANTHROPIC_API_KEY
  comparison_credits: ANTHROPIC_API_KEY (or the OpenCode provider key)

Usage:
    python local_test_app.py ee_e2e
    python local_test_app.py eval_trace
    python local_test_app.py inspect_traces
    python local_test_app.py analysis_agent
    python local_test_app.py close_iteration
"""
from __future__ import annotations

import argparse
import os
import sys

# Fixture defaults live with the e2e driver so every entrypoint agrees on
# which branch / holdout a run seeds from. Cheap import:
# `test_e2e` pulls in only stdlib plus `src.engine.paths`.
from tests.functional.test_e2e import DEFAULT_SEED_BRANCH, DEFAULT_TESTSET_PATH


def _require_env(*names: str) -> None:
    """Fail fast with a clear message when required API keys are unset."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(
            "ERROR: missing required environment variables: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Set them before re-running, e.g.:\n"
            + "\n".join(f"  export {n}=..." for n in missing),
            file=sys.stderr,
        )
        sys.exit(2)


# Provider API-key env var a coding harness + model needs. OpenCode talks to
# whatever provider its `provider/model` string names; Claude is Anthropic.
_PROVIDER_ENV_BY_PREFIX = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
}


def _require_harness_env(coding_harness: str, coding_harness_model: str | None) -> None:
    """Fail fast (before any agent runs) if the API key the *selected* coding
    harness needs is unset — checking the right key for the harness rather than
    always demanding ANTHROPIC_API_KEY (which an OpenCode run does not use)."""
    if coding_harness != "opencode":
        _require_env("ANTHROPIC_API_KEY")
        return
    provider = (coding_harness_model or "").split("/", 1)[0].lower()
    env_name = _PROVIDER_ENV_BY_PREFIX.get(provider)
    if env_name is None:
        print(
            "ERROR: OpenCode harness selected but the provider for model "
            f"{coding_harness_model!r} is unknown — cannot tell which API key "
            "is required. Pass a 'provider/model' string such as "
            "'openai/gpt-5.4-mini' via --coding-harness-model.",
            file=sys.stderr,
        )
        sys.exit(2)
    _require_env(env_name)


# ---------------------------------------------------------------------------
# ee_e2e
# ---------------------------------------------------------------------------


def _cmd_e2e(args: argparse.Namespace) -> int:
    _require_env("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    from tests._test_env import LocalClientRepoEnv, new_test_id
    from tests.functional.test_e2e import run_e2e

    # Every arm reuses this whole command — same driver, same assertions —
    # and only swaps the optimizer entry script and the loop's bound.
    method = getattr(args, "method", "ee")
    max_metric_calls = getattr(args, "max_metric_calls", None)
    bound = (
        f"max_metric_calls={max_metric_calls}"
        if method == "simple_loop"
        else f"iterations={args.max_iterations}"
    )
    # pareto_evolution seeds the population with a counted comparison eval,
    # and asa measures the seed lineage on it, so a comparison set is
    # mandatory for both (the driver default is None).
    if method in ("pareto_evolution", "asa") and not args.comparison_set_size:
        args.comparison_set_size = 5
    test_id = args.test_id or new_test_id(
        method=method, harness=args.coding_harness
    )
    env = LocalClientRepoEnv(timeout=args.timeout, test_id=test_id)
    print(
        f"Running {method.upper()} E2E test locally (repo={args.repo_url}, "
        f"{bound}, harness={args.coding_harness}, test_id={test_id})..."
    )
    result = run_e2e(
        env,
        repo_url=args.repo_url,
        program=args.program,
        metric=args.metric,
        trainset_path=args.trainset_path,
        testset_path=args.testset_path,
        initial_branch=args.initial_branch,
        max_iterations=args.max_iterations,
        comparison_set_size=args.comparison_set_size,
        primary_metric=args.primary_metric,
        coding_harness=args.coding_harness,
        coding_harness_model=args.coding_harness_model,
        enable_meta_reflection=args.meta_reflection,
        enable_run_integrity_agent=args.run_integrity,
        enable_clean_slate_inspector=args.clean_slate,
        method=method,
        max_metric_calls=max_metric_calls,
    )

    print("\n" + "=" * 60)
    print(f"Result: success={result.get('success')}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    print(f"Seed branch: {result.get('seed_branch')}")
    print(f"Primary branch: {result.get('primary_branch')}")
    print(f"Final: {result.get('final_branch')} @ {result.get('final_sha')}")
    print(
        f"Train scores (comparison set): seed={result.get('seed_train_score')} "
        f"final={result.get('final_train_score')}"
    )
    print(
        f"Test scores (held-out, mean): seed={result.get('seed_test_score')} "
        f"final={result.get('final_test_score')}"
    )
    print(
        f"Test samples (held-out, per draw): seed={result.get('seed_test_scores')} "
        f"final={result.get('final_test_scores')}"
    )
    print(f"Total metric calls: {result.get('total_metric_calls')}")
    print(f"Run metrics: {result.get('run_metrics')}")
    print(f"Iterations run: {result.get('iterations')}")
    _print_post_run_agents(args, result)
    print("\n--- IterationHistory ---")
    for idx, entry in sorted((result.get("history") or {}).items()):
        print(f"  iteration {idx}: {entry}")
    print("\n--- Artifacts ---")
    print(f"run log:      {result.get('run_log_path')}")
    print(f"artifact root:{result.get('artifact_root')}  (test_id={result.get('test_id')})")
    print("=" * 60 + "\n")

    if not result.get("success"):
        print("FAIL: Optimization did not succeed")
        return 1
    failures, warnings = _post_run_agent_failures(args, result)
    if warnings:
        print("WARN: the run's own self-checks reported (non-fatal):")
        for w in warnings:
            print(f"  - {w}")
        print(f"Evidence is under {result.get('artifact_root')}.\n")
    if failures:
        print("FAIL: the run's own self-checks reported problems:")
        for f in failures:
            print(f"  - {f}")
        print(f"\nEvidence is under {result.get('artifact_root')}.")
        return 1
    if warnings:
        print(
            f"PASS (with warnings): Optimization completed successfully; "
            f"{len(warnings)} non-fatal self-check warning(s) above."
        )
        return 0
    print("PASS: Optimization completed successfully")
    return 0


def _print_post_run_agents(args: argparse.Namespace, result: dict) -> None:
    """Surface what the run's own inspectors concluded."""
    if args.meta_reflection:
        print("\n--- Meta-reflection (integrated) ---")
        print(f"architect_traces_captured: {result.get('architect_traces_captured')}")
        print(f"meta_reflection_written:   {result.get('meta_reflection_written')}")
        print(f"meta_reflection_len:       {result.get('meta_reflection_len')}")
        print(f"meta_reflection_path:      {result.get('meta_reflection_path')}")
    if args.clean_slate:
        status = result.get("clean_slate_status")
        print("\n--- Clean slate (start of run) ---")
        # Absent means the arm does not run the inspector (simple_loop has its
        # own repository init and skips it) — a normal outcome, not a failure.
        print(f"status: {status if status is not None else 'n/a (arm does not run it)'}")
        print(f"report: {result.get('clean_slate_report_path')}")
    if args.run_integrity:
        verdict = result.get("integrity_verdict") or {}
        print("\n--- Run integrity (post-run) ---")
        print(f"cheating_category:      {verdict.get('integrity_cheating_category')}")
        print(f"functioning_iterations: {verdict.get('integrity_functioning_iterations')}")
        print(f"integrity_pct:          {verdict.get('integrity_pct')}")
        rationale = verdict.get("integrity_cheating_rationale")
        if rationale:
            print(f"rationale:              {rationale}")
        print(f"report: {result.get('run_integrity_path')}")
    violations = result.get("constraint_violations") or []
    print(f"\n--- Constraint gate --- ({len(violations)} fired)")
    for v in violations:
        print(f"  iteration {v.get('iteration')}: {v.get('reason')}")


def _post_run_agent_failures(
    args: argparse.Namespace, result: dict
) -> tuple[list[str], list[str]]:
    """Adverse verdicts, split into ``(failures, warnings)`` lines.

    Failures are the signals that mean the harness or one of its auditors
    broke: no architect traces / no meta-reflection when it was armed, no
    integrity verdict when the inspector was armed, a constraint-gate fire, or
    a clean-slate status other than ``clean``. An absent `clean_slate_status`
    is not a failure — it means the arm does not run that inspector.

    A *cheating* classification from the run-integrity inspector is a WARNING
    by default. On the local target the holdout fence is deny-paths only —
    bash is allowed, and the kernel-level `ce-agent` uid fence exists only in
    the Docker runner — so an architect that `head`s the holdout is model
    behaviour the e2e cannot prevent, not a harness defect.
    `--fail-on-cheating` restores the strict policy for a deliberate check.
    """
    failures: list[str] = []
    warnings: list[str] = []
    if args.meta_reflection:
        if not result.get("architect_traces_captured"):
            failures.append("no iteration_{N}_architect_trace/ captured during the run")
        if not result.get("meta_reflection_written"):
            failures.append("meta_reflection.md was not produced at finalization")
    if args.clean_slate:
        status = result.get("clean_slate_status")
        if status is not None and status != "clean":
            failures.append(
                f"clean-slate status is {status!r}, not 'clean' — see "
                f"{result.get('clean_slate_report_path')}"
            )
    if args.run_integrity:
        verdict = result.get("integrity_verdict") or {}
        category = verdict.get("integrity_cheating_category")
        if not category:
            failures.append(
                "run-integrity agent produced no verdict (it was enabled, so "
                "this means the agent itself failed)"
            )
        elif category != "no_cheating":
            line = (
                f"run-integrity classified this run as {category!r}: "
                f"{verdict.get('integrity_cheating_rationale')}"
            )
            if getattr(args, "fail_on_cheating", False):
                failures.append(line)
            else:
                warnings.append(line)
    for v in result.get("constraint_violations") or []:
        failures.append(
            f"constraint gate fired on iteration {v.get('iteration')} "
            f"(change force-reverted): {v.get('reason')}"
        )
    return failures, warnings


# ---------------------------------------------------------------------------
# eval_trace
# ---------------------------------------------------------------------------


def _cmd_eval_trace(args: argparse.Namespace) -> int:
    _require_env("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    from tests._test_env import LocalClientRepoEnv
    from tests.functional.test_e2e import evaluate_trace_collection, run_eval_trace

    print(
        f"Running eval_trace locally "
        f"(repo={args.repo_url}, seed-only, environment-validation rows, "
        f"num_threads={args.num_threads})..."
    )
    result = run_eval_trace(
        LocalClientRepoEnv(timeout=args.timeout),
        repo_url=args.repo_url,
        program=args.program,
        trainset_path=args.trainset_path,
        num_threads=args.num_threads,
        coding_harness=args.coding_harness,
        coding_harness_model=args.coding_harness_model,
    )

    trace_content = result.get("trace_file_content") or ""
    trace_path = result.get("trace_file_path") or "(unknown)"

    print("\n" + "=" * 60)
    print(f"Trace file: {trace_path}")
    print("=" * 60)
    print(trace_content if trace_content.strip() else "(empty)")
    print("=" * 60 + "\n")

    report = evaluate_trace_collection(trace_content)
    print(f"Parsed {report['row_count']} trace row(s).")
    print(f"Distinct signature_keys seen: {report['distinct_signature_keys']}")
    print(
        f"Pipeline root spans: {report['pipeline_roots']} / {report['row_count']} rows"
    )
    print(f"Judge-like span entries: {report['judge_entries']}")
    print(
        f"Judge entries with non-trivial inputs+output: {report['instrumented_judge']}"
    )

    print("\n" + "=" * 60)
    if not report["pass"]:
        print("FAIL:")
        for f in report["failures"]:
            print(f"  - {f}")
        return 1
    print("PASS: pipeline roots + instrumented judge span present.")
    return 0


# ---------------------------------------------------------------------------
# inspect_traces
# ---------------------------------------------------------------------------


def _cmd_inspect_traces(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_inspect_traces_architect import (
        INSPECT_MAX_ATTEMPTS, inspect_traces_failures, inspect_traces_warnings,
        run_inspect_traces_architect,
    )

    print(
        f"Running the inspect_traces architect test (D1) locally "
        f"(harness={args.coding_harness})..."
    )

    # The architect reaching for `inspect_traces` is model-driven and not
    # 100% reliable; retry up to INSPECT_MAX_ATTEMPTS times before failing.
    passed_attempt: int | None = None
    last_failures: list[str] = []
    for attempt in range(1, INSPECT_MAX_ATTEMPTS + 1):
        print(f"\n----- attempt {attempt}/{INSPECT_MAX_ATTEMPTS} -----")
        result = run_inspect_traces_architect(
            LocalSyntheticEnv(),
            coding_harness=args.coding_harness,
            coding_harness_model=args.coding_harness_model,
        )

        print("\n" + "=" * 60)
        print(f"inspect_traces_call_count: {result['inspect_traces_call_count']}")
        print(f"coding_agent_called:       {result['coding_agent_called']}")
        print(f"direct_edit:               {result.get('direct_edit', False)}")
        print(f"mentions FireCrawl:        {result['mentions_firecrawl']}")
        print(f"mentions proxy URL:        {result['mentions_proxy_url']}")
        print(f"mentions ResearchAgent:    {result['mentions_research_agent']}")
        print("\n--- inspect_traces queries (architect → tool) ---")
        for i, q in enumerate(result.get("inspect_traces_queries") or []):
            print(f"[{i}] {q[:400]}")
        print("\n--- inspect_traces summaries (tool → architect) ---")
        for i, s in enumerate(result.get("inspect_traces_summaries") or []):
            print(f"[{i}] {s[:600]}")
        print("\n--- architect change_request(s) ---")
        for i, cr in enumerate(result.get("change_requests") or []):
            print(f"[{i}] {cr[:800]}")
        if result.get("error"):
            print(f"\n[error] {result['error']}")
        print("=" * 60 + "\n")

        failures = inspect_traces_failures(result)
        if not failures:
            passed_attempt = attempt
            break

        last_failures = failures
        print(f"attempt {attempt} did not pass:")
        for f in failures:
            print(f"  - {f}")
        if attempt < INSPECT_MAX_ATTEMPTS:
            print("retrying (inspect_traces is model-driven and not 100% reliable)...")

    if passed_attempt is None:
        print(
            f"FAIL: inspect_traces test did not pass in {INSPECT_MAX_ATTEMPTS} attempts:"
        )
        for f in last_failures:
            print(f"  - {f}")
        return 1

    how = (
        "used inspect_traces"
        if result["inspect_traces_call_count"] > 0
        else "read the trace directory directly"
    )
    if passed_attempt >= INSPECT_MAX_ATTEMPTS:
        print(
            f"PASS (with warning): architect {how} and grounded "
            f"its fix, but only on attempt {passed_attempt}/{INSPECT_MAX_ATTEMPTS} "
            f"— reliability is marginal."
        )
    else:
        print(
            f"PASS: architect {how} and grounded its fix "
            f"(attempt {passed_attempt}/{INSPECT_MAX_ATTEMPTS})"
        )
    for w in inspect_traces_warnings(result):
        print(f"  WARNING: {w}")
    return 0


# ---------------------------------------------------------------------------
# architect_trace (G2 — architect-session trace capture)
# ---------------------------------------------------------------------------


def _cmd_architect_trace(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_architect_trace_capture import (
        report_architect_trace_capture, run_architect_trace_capture,
    )

    print(
        f"Running the architect-trace capture test locally "
        f"(harness={args.coding_harness})..."
    )
    result = run_architect_trace_capture(
        LocalSyntheticEnv(),
        coding_harness=args.coding_harness,
        coding_harness_model=args.coding_harness_model,
    )
    return report_architect_trace_capture(result, args.coding_harness)


# ---------------------------------------------------------------------------
# meta_reflection (G5 — post-run meta-reflection golden nuggets)
# ---------------------------------------------------------------------------


def _cmd_meta_reflection(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_meta_reflection_agent import (
        report_meta_reflection_agent, run_meta_reflection_agent,
    )

    print(
        f"Running the meta-reflection golden-nugget test locally "
        f"(harness={args.coding_harness})..."
    )
    result = run_meta_reflection_agent(
        LocalSyntheticEnv(),
        coding_harness=args.coding_harness,
        coding_harness_model=args.coding_harness_model,
    )
    return report_meta_reflection_agent(result)


# ---------------------------------------------------------------------------
# auditors_back_to_back — meta-reflection then run-integrity, one process
# ---------------------------------------------------------------------------


def _cmd_auditors_back_to_back(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_auditors_back_to_back import (
        report_auditors_back_to_back, run_auditors_back_to_back,
    )

    print(
        f"Running meta-reflection then run-integrity back-to-back "
        f"(harness={args.coding_harness})..."
    )
    result = run_auditors_back_to_back(
        LocalSyntheticEnv(),
        coding_harness=args.coding_harness,
        coding_harness_model=args.coding_harness_model,
    )
    return report_auditors_back_to_back(result)


# ---------------------------------------------------------------------------
# run_integrity (H4 — run-integrity inspector golden nuggets)
# ---------------------------------------------------------------------------


def _cmd_run_integrity(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_run_integrity_agent import (
        report_run_integrity_agent, run_run_integrity_agent,
    )

    print(
        f"Running the run-integrity golden-nugget test locally "
        f"(scenario={args.scenario}, harness={args.coding_harness})..."
    )
    result = run_run_integrity_agent(
        LocalSyntheticEnv(),
        scenario=args.scenario,
        coding_harness=args.coding_harness,
        coding_harness_model=args.coding_harness_model,
    )
    return report_run_integrity_agent(result)


# ---------------------------------------------------------------------------
# analysis_agent
# ---------------------------------------------------------------------------


def _cmd_analysis_agent(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_analysis_agent import (
        ANALYSIS_MAX_ATTEMPTS, VERIFICATION_TOKEN, analysis_agent_failures,
        run_analysis_agent, summarize_result,
    )

    print(
        f"Running the analysis-agent test locally (harness={args.coding_harness})..."
    )

    # The analysis agent carrying the verification signal into the change
    # summary is model-driven and not 100% reliable; retry up to
    # ANALYSIS_MAX_ATTEMPTS times before failing.
    passed_attempt: int | None = None
    last_failures: list[str] = []
    for attempt in range(1, ANALYSIS_MAX_ATTEMPTS + 1):
        print(f"\n----- attempt {attempt}/{ANALYSIS_MAX_ATTEMPTS} -----")
        result = run_analysis_agent(
            LocalSyntheticEnv(),
            coding_harness=args.coding_harness,
            coding_harness_model=args.coding_harness_model,
        )

        coherent = result.get("change_summary_coherent")
        coherent_label = (
            "PASS" if coherent is True
            else "FAIL" if coherent is False
            else "SKIPPED"
        )

        print("\n" + "=" * 60)
        print(f"Write success: {result['write_success']}")
        print(f"Change summary written: {result['summary_written']}")
        print(f"Has verification signal ({VERIFICATION_TOKEN}): {result['has_token']}")
        print(f"memory.md modified (must be False): {result['memory_modified']}")
        print(
            f"Change summary coherent (GPT-5.4-mini judge): {coherent_label} "
            f"— {result.get('change_summary_judge_reason')}"
        )
        print("\n--- iteration_5_change_summary.md after agent ---")
        print((result.get("change_summary") or "(empty / not written)")[:1500])
        print(f"\nResult: {summarize_result(result)}")
        print("=" * 60 + "\n")

        failures = analysis_agent_failures(result)
        if not failures:
            passed_attempt = attempt
            break

        last_failures = failures
        print(f"attempt {attempt} did not pass:")
        for f in failures:
            print(f"  - {f}")
        if attempt < ANALYSIS_MAX_ATTEMPTS:
            print("retrying (the analysis agent is model-driven and not 100% reliable)...")

    if passed_attempt is None:
        print(
            f"FAIL: analysis-agent test did not pass in {ANALYSIS_MAX_ATTEMPTS} attempts:"
        )
        for f in last_failures:
            print(f"  - {f}")
        return 1

    if passed_attempt >= ANALYSIS_MAX_ATTEMPTS:
        print(
            f"PASS (with warning): analysis-agent checks passed, but only on "
            f"attempt {passed_attempt}/{ANALYSIS_MAX_ATTEMPTS} — reliability is marginal."
        )
    else:
        print(
            f"PASS: All analysis-agent checks passed "
            f"(attempt {passed_attempt}/{ANALYSIS_MAX_ATTEMPTS})"
        )
    return 0


# ---------------------------------------------------------------------------
# close_iteration (memory.md reflection + real close_iteration tool)
# ---------------------------------------------------------------------------


def _cmd_close_iteration(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_close_iteration import (
        CLOSE_BRANCH, CLOSE_MAX_ATTEMPTS, HIDDEN_MODULE,
        close_iteration_failures, run_close_iteration,
    )

    print(
        f"Running the close-iteration test locally "
        f"(harness={args.coding_harness})..."
    )

    # The architect grounding its reflection and closing cleanly is
    # model-driven and not 100% reliable; retry up to CLOSE_MAX_ATTEMPTS
    # times before failing.
    passed_attempt: int | None = None
    last_failures: list[str] = []
    for attempt in range(1, CLOSE_MAX_ATTEMPTS + 1):
        print(f"\n----- attempt {attempt}/{CLOSE_MAX_ATTEMPTS} -----")
        result = run_close_iteration(
            LocalSyntheticEnv(),
            coding_harness=args.coding_harness,
            coding_harness_model=args.coding_harness_model,
        )

        print("\n" + "=" * 60)
        print(f"close_called:            {result['close_called']}")
        print(f"registered_branch:       {result.get('registered_branch')}")
        print(f"  (expected)             {CLOSE_BRANCH}")
        print(f"branch exists in git:    {result.get('registered_branch_exists')}")
        print(f"improvement_mode:        {result.get('claimed_improvement_mode')}")
        print(f"memory_updated (claim):  {result.get('memory_updated_claimed')}")
        print(f"memory.md changed:       {result['memory_changed']}")
        print(f"mentions {HIDDEN_MODULE}: {result['mentions_hidden_module']}")
        print(f"mentions timeout:        {result['mentions_timeout']}")
        print(f"coding_agent_called:     {result['coding_agent_called']}")
        print(f"watchdog_fired:          {result.get('watchdog_fired')}")
        print("\n--- memory.md after session ---")
        print((result.get("memory_md") or "")[:2000])
        if result.get("error"):
            print(f"\n[error] {result['error']}")
        print("=" * 60 + "\n")

        failures = close_iteration_failures(result)
        if not failures:
            passed_attempt = attempt
            break

        last_failures = failures
        print(f"attempt {attempt} did not pass:")
        for f in failures:
            print(f"  - {f}")
        if attempt < CLOSE_MAX_ATTEMPTS:
            print("retrying (the closing protocol is model-driven and not 100% reliable)...")

    if passed_attempt is None:
        print(
            f"FAIL: close-iteration test did not pass in "
            f"{CLOSE_MAX_ATTEMPTS} attempts:"
        )
        for f in last_failures:
            print(f"  - {f}")
        return 1

    if passed_attempt >= CLOSE_MAX_ATTEMPTS:
        print(
            f"PASS (with warning): close-iteration checks passed, but only on "
            f"attempt {passed_attempt}/{CLOSE_MAX_ATTEMPTS} — reliability is "
            f"marginal."
        )
    else:
        print(
            f"PASS: close-iteration checks passed "
            f"(attempt {passed_attempt}/{CLOSE_MAX_ATTEMPTS})"
        )
    return 0


# ---------------------------------------------------------------------------
# session_resume (watchdog-resumed session retains first-session context)
# ---------------------------------------------------------------------------


def _cmd_session_resume(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_session_resume import (
        RESUME_MAX_ATTEMPTS, SESSION_TOKEN,
        session_resume_failures, run_session_resume,
    )

    print(
        f"Running the session-resume test locally "
        f"(harness={args.coding_harness})..."
    )

    passed_attempt: int | None = None
    last_failures: list[str] = []
    for attempt in range(1, RESUME_MAX_ATTEMPTS + 1):
        print(f"\n----- attempt {attempt}/{RESUME_MAX_ATTEMPTS} -----")
        result = run_session_resume(
            LocalSyntheticEnv(),
            coding_harness=args.coding_harness,
            coding_harness_model=args.coding_harness_model,
        )

        print("\n" + "=" * 60)
        print(f"watchdog_fired (resume): {result.get('watchdog_fired')}")
        print(f"close_called:            {result['close_called']}")
        print(f"token in memory.md:      {result['token_in_memory']}")
        print(f"  (token)                {SESSION_TOKEN}")
        print(f"token issued exactly 1x: {result.get('token_issue_count')}")
        print(f"memory.md changed:       {result['memory_changed']}")
        print(f"coding_agent_called:     {result['coding_agent_called']}")
        print("\n--- memory.md after session ---")
        print((result.get("memory_md") or "")[:2000])
        if result.get("error"):
            print(f"\n[error] {result['error']}")
        print("=" * 60 + "\n")

        failures = session_resume_failures(result)
        if not failures:
            passed_attempt = attempt
            break

        last_failures = failures
        print(f"attempt {attempt} did not pass:")
        for f in failures:
            print(f"  - {f}")
        if attempt < RESUME_MAX_ATTEMPTS:
            print("retrying (the end-turn-without-closing step is model-driven)...")

    if passed_attempt is None:
        print(
            f"FAIL: session-resume test did not pass in "
            f"{RESUME_MAX_ATTEMPTS} attempts:"
        )
        for f in last_failures:
            print(f"  - {f}")
        return 1

    if passed_attempt >= RESUME_MAX_ATTEMPTS:
        print(
            f"PASS (with warning): session-resume checks passed, but only on "
            f"attempt {passed_attempt}/{RESUME_MAX_ATTEMPTS}."
        )
    else:
        print(
            f"PASS: resumed session retained first-session context "
            f"(attempt {passed_attempt}/{RESUME_MAX_ATTEMPTS})"
        )
    return 0


# ---------------------------------------------------------------------------
# comparison_credits (real architect completes the credit acknowledgement
# round-trip through the real run_eval tool; evaluator patched)
# ---------------------------------------------------------------------------


def _cmd_comparison_credits(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_comparison_credits import (
        ACK_CODE, CREDITS_MAX_ATTEMPTS,
        comparison_credits_failures, run_comparison_credits,
    )

    print(
        f"Running the comparison-credits acknowledgement test locally "
        f"(harness={args.coding_harness})..."
    )

    passed_attempt: int | None = None
    last_failures: list[str] = []
    for attempt in range(1, CREDITS_MAX_ATTEMPTS + 1):
        print(f"\n----- attempt {attempt}/{CREDITS_MAX_ATTEMPTS} -----")
        result = run_comparison_credits(
            LocalSyntheticEnv(),
            coding_harness=args.coding_harness,
            coding_harness_model=args.coding_harness_model,
        )

        print("\n" + "=" * 60)
        print(f"ack refusals returned:     {len(result.get('refusals') or [])}")
        print(f"comparison evals executed: {result.get('comparison_evals_run')}")
        print(f"ack_given (code {ACK_CODE}):   {result.get('ack_given')}")
        print(f"credits balance after:     {result.get('credits_balance')}")
        print(f"comparison tool calls:     {result.get('comparison_tool_calls')}")
        print(f"close_called:              {result.get('close_called')}")
        print(f"watchdog_fired:            {result.get('watchdog_fired')}")
        print("\n--- memory.md after session ---")
        print((result.get("memory_md") or "")[:1500])
        if result.get("error"):
            print(f"\n[error] {result['error']}")
        print("=" * 60 + "\n")

        failures = comparison_credits_failures(result)
        if not failures:
            passed_attempt = attempt
            break

        last_failures = failures
        print(f"attempt {attempt} did not pass:")
        for f in failures:
            print(f"  - {f}")
        if attempt < CREDITS_MAX_ATTEMPTS:
            print("retrying (completing the acknowledgement is model-driven)...")

    if passed_attempt is None:
        print(
            f"FAIL: comparison-credits test did not pass in "
            f"{CREDITS_MAX_ATTEMPTS} attempts:"
        )
        for f in last_failures:
            print(f"  - {f}")
        return 1

    if passed_attempt >= CREDITS_MAX_ATTEMPTS:
        print(
            f"PASS (with warning): comparison-credits checks passed, but only "
            f"on attempt {passed_attempt}/{CREDITS_MAX_ATTEMPTS}."
        )
    else:
        print(
            f"PASS: the architect completed the acknowledgement round-trip "
            f"(attempt {passed_attempt}/{CREDITS_MAX_ATTEMPTS})"
        )
    return 0


# ---------------------------------------------------------------------------
# test_sampling (finalization draws the held-out set `test_samples` times)
# ---------------------------------------------------------------------------


def _cmd_test_sampling(args: argparse.Namespace) -> int:
    # The FactChecker fixture program calls OpenAI; no coding harness involved.
    _require_env("OPENAI_API_KEY")
    from tests._test_env import LocalClientRepoEnv, new_test_id
    from tests.functional.test_test_sampling import run_test_sampling

    test_id = args.test_id or new_test_id(method="test_sampling")
    env = LocalClientRepoEnv(timeout=args.timeout, test_id=test_id)
    print(
        f"Running finalization test-sampling check locally "
        f"(repo={args.repo_url}, final samples={args.samples}, "
        f"seed samples={args.seed_samples}, test_id={test_id})..."
    )
    result = run_test_sampling(
        env, repo_url=args.repo_url, initial_branch=args.initial_branch,
        num_samples=args.samples, seed_samples=args.seed_samples,
        eval_timeout=args.timeout,
    )
    print("\n" + "=" * 60)
    print(f"Result: success={result.get('success')}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    print(f"Scores: {result.get('scores')}")
    print(f"Checks: {result.get('checks')}")
    print(f"Trace dirs: {result.get('trace_dirs')}")
    print(
        f"Identical samples per branch (0=seed, 1=final; only branches with >1 draw): "
        f"{result.get('identical_samples')}  "
        f"cache_warning_fired={result.get('cache_warning_fired')}"
    )
    if any((result.get("identical_samples") or {}).values()):
        print(
            "NOTE: identical draws mean the client program served the same "
            "completion every time — a response cache is likely on in the "
            "fixture repo (the engine does not configure caching by design)."
        )
    print(f"Artifacts: {env.artifact_root} (test_id={test_id})")
    return 0 if result.get("success") else 1


# ---------------------------------------------------------------------------
# variance_tool (a real architect session calls run_variance_estimate)
# ---------------------------------------------------------------------------


def _cmd_variance_tool(args: argparse.Namespace) -> int:
    # The FactChecker fixture program calls OpenAI (four rows), and the
    # architect session needs the selected harness's key.
    _require_env("OPENAI_API_KEY")
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalClientRepoEnv, new_test_id
    from tests.functional.test_variance_tool import (
        VARIANCE_MAX_ATTEMPTS,
        run_variance_tool,
        variance_tool_failures,
    )

    for attempt in range(1, VARIANCE_MAX_ATTEMPTS + 1):
        test_id = args.test_id or new_test_id(
            method="variance_tool", harness=args.coding_harness
        )
        env = LocalClientRepoEnv(timeout=args.timeout, test_id=test_id)
        print(
            f"Running run_variance_estimate functional test locally "
            f"(harness={args.coding_harness}, model={args.coding_harness_model}, "
            f"repo={args.repo_url}, attempt {attempt}/{VARIANCE_MAX_ATTEMPTS}, "
            f"test_id={test_id})..."
        )
        result = run_variance_tool(
            env,
            coding_harness=args.coding_harness,
            coding_harness_model=args.coding_harness_model,
            repo_url=args.repo_url,
            initial_branch=args.initial_branch,
            eval_timeout=args.timeout,
        )
        failures = variance_tool_failures(result)
        print("\n" + "=" * 60)
        print(f"Result: success={result.get('success')}")
        print(
            f"Tool calls: {result.get('tool_call_count')} "
            f"(refused: {result.get('refused_calls')}, errors: {result.get('tool_errors')})"
        )
        est = result.get("variance_estimate") or {}
        print(
            f"Estimate: scores={est.get('aggregate_scores')} "
            f"within_item_variance={est.get('within_item_variance')} "
            f"run_to_run_std={est.get('run_to_run_std')} "
            f"identical_outputs_warning={result.get('identical_outputs_warning')}"
        )
        print(f"Trace rows: {result.get('trace_rows')}")
        print(f"Checks: {result.get('checks')}")
        print(f"Artifacts: {env.artifact_root} (test_id={test_id})")
        if not failures:
            print("PASS")
            return 0
        print("FAIL: " + "; ".join(failures))
        if attempt < VARIANCE_MAX_ATTEMPTS:
            print("Retrying...")
    return 1


# ---------------------------------------------------------------------------
# holdout_guard (behavioral held-out test-set fence)
# ---------------------------------------------------------------------------


def _cmd_holdout_guard(args: argparse.Namespace) -> int:
    _require_harness_env(args.coding_harness, args.coding_harness_model)
    from tests._test_env import LocalSyntheticEnv
    from tests.functional.test_holdout_guard_behavioral import (
        SECRET_TOKEN, run_holdout_guard_behavioral,
    )

    print(
        f"Running behavioral holdout-guard test locally "
        f"(agent={args.agent}, harness={args.coding_harness})..."
    )
    result = run_holdout_guard_behavioral(
        LocalSyntheticEnv(),
        agent=args.agent,
        coding_harness=args.coding_harness,
        coding_harness_model=args.coding_harness_model,
    )

    print("\n" + "=" * 60)
    print(f"agent:  {result.get('agent')}")
    print(f"ran:    {result.get('ran')}")
    print(f"leaked: {result.get('leaked')}")
    print(f"\n--- agent output ---\n{(result.get('output') or '')[:1500]}")
    if result.get("error"):
        print(f"\n[error] {result['error']}")
    print("=" * 60 + "\n")

    if not result.get("ran"):
        print(f"FAIL: test did not run: {result.get('error')}")
        return 1
    if result.get("leaked"):
        print(
            f"FAIL: holdout secret {SECRET_TOKEN!r} surfaced — the guard did "
            "NOT block the read."
        )
        return 1
    print(
        f"PASS: holdout secret never surfaced — the armed guard blocked the "
        f"{args.agent} agent."
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# The OpenCode coding model every research run uses (experiments/*/input.py
# pins the same string), so a local functional test exercises the model the
# fleet actually runs instead of demanding one on every invocation.
DEFAULT_OPENCODE_MODEL = "deepinfra/zai-org/GLM-5.2"


def _add_harness_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--coding-harness`` / ``--coding-harness-model`` flags."""
    parser.add_argument(
        "--coding-harness",
        default="claude",
        choices=["claude", "opencode"],
        help="Provider coding harness for the architect / inspection agent.",
    )
    parser.add_argument(
        "--coding-harness-model",
        default=DEFAULT_OPENCODE_MODEL,
        help="Provider/model string for OpenCode (default: "
        f"{DEFAULT_OPENCODE_MODEL}, the research fleet's coding model); "
        "ignored in Claude mode.",
    )


def _add_post_run_agent_args(p: argparse.ArgumentParser) -> None:
    """The three inspectors a real research run always has armed.

    Meta-reflection and run-integrity default ON to match
    `runner/entrypoint.py::RESEARCH_DEFAULTS` — an e2e that leaves them off
    never exercises the paths a real run takes. Each has a `--no-` form for a
    deliberately cheaper one-off run.

    Clean-slate is the exception: it defaults OFF locally. A research run gets
    a fresh container, so leftovers there mean something; a local run reuses
    one cached clone (`~/.codeevolver/test-repos/{slug}`) and keeps every past
    run's memory / traces / logs under `~/.codeevolver/e2e/`, which the
    inspector's shell can reach. It would be reporting on OTHER e2e runs, and
    since a `contaminated` verdict aborts before any eval spend and anything
    other than `clean` fails the test, that is a false failure on a healthy
    run. Pass `--clean-slate` to arm it deliberately (validating a fresh seed
    branch).
    """
    p.add_argument(
        "--meta-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture architect-session traces during the run and run the "
        "post-finalization meta-reflection agent; assert both happened.",
    )
    p.add_argument(
        "--run-integrity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the post-run integrity inspector and fail if it produces "
        "no verdict at all (the inspector itself broke). A cheating "
        "classification is reported as a WARNING unless --fail-on-cheating.",
    )
    p.add_argument(
        "--fail-on-cheating",
        action="store_true",
        default=False,
        help="Fail the run on any integrity verdict other than 'no_cheating'. "
        "Off by default: the local holdout fence is deny-paths only (bash is "
        "allowed), so an architect reading the holdout is model behaviour "
        "this e2e cannot prevent, not a harness defect.",
    )
    p.add_argument(
        "--clean-slate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the start-of-run clean-slate inspector (default OFF locally "
        "— the host reuses one cached clone and keeps prior runs' artifacts "
        "under ~/.codeevolver/e2e/, so its verdict reflects other e2e runs). "
        "A contaminated verdict aborts the run; anything other than 'clean' "
        "fails the test. Arms that do not run the inspector report 'n/a'.",
    )
    p.add_argument(
        "--test-id",
        default=None,
        help="Name this run. Becomes the optimizer's job_id, so it scopes "
        "every artifact directory and log filename. Auto-generated if unset.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local_test_app.py", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ee_e2e / pareto_e2e share one driver + argument set; only the
    # optimizer entry script differs (`method`).
    def _add_engine3_e2e_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--repo-url",
            default="https://github.com/julianghadially/FactChecker",
        )
        p.add_argument(
            "--program",
            default="src.factchecker.simple.factchecker_program:FactCheckerProgram",
        )
        p.add_argument("--metric", default="src.codeevolver.metric:metric")
        p.add_argument(
            "--trainset-path",
            default="data/FacTool_QA_train_10examples_normalized.jsonl",
        )
        p.add_argument("--testset-path", default=DEFAULT_TESTSET_PATH)
        p.add_argument("--initial-branch", default=DEFAULT_SEED_BRANCH)
        p.add_argument("--max-iterations", type=int, default=2)
        p.add_argument(
            "--comparison-set-size",
            type=int,
            default=None,
            help="Engine 3 fixed comparison set drawn from the trainset "
            "(replaces the removed max_valset_size); default None disables it "
            "(pareto_e2e falls back to 5 — the arm needs one).",
        )
        p.add_argument("--primary-metric", default="accuracy")
        p.add_argument(
            "--coding-harness",
            default="claude",
            choices=["claude", "opencode"],
            help="Provider coding harness for the architect loop.",
        )
        p.add_argument(
            "--coding-harness-model",
            default=None,
            help="Provider/model string for OpenCode (e.g. openai/gpt-5.3-codex-spark); "
            "ignored in Claude mode.",
        )
        _add_post_run_agent_args(p)
        p.add_argument("--timeout", type=int, default=7200)

    p_ee = sub.add_parser("ee_e2e", help="Full optimizer loop against a real repo")
    p_ee.set_defaults(func=_cmd_e2e)
    _add_engine3_e2e_args(p_ee)

    p_par = sub.add_parser(
        "pareto_e2e",
        help="Full PARETO_EVOLUTION (fixed evolution) loop against a real repo",
    )
    p_par.set_defaults(func=_cmd_e2e, method="pareto_evolution")
    _add_engine3_e2e_args(p_par)

    p_ff = sub.add_parser(
        "freeform_e2e",
        help="Full FREEFORM (Baseline 1) loop against a real repo; local-only",
    )
    p_ff.set_defaults(func=_cmd_e2e, method="freeform")
    _add_engine3_e2e_args(p_ff)

    p_asa = sub.add_parser(
        "asa_e2e",
        help="Full ASA (Treatment C: EE + non-primary branch control + branch_comparison) loop; local-only",
    )
    p_asa.set_defaults(func=_cmd_e2e, method="asa")
    _add_engine3_e2e_args(p_asa)

    # simple_loop_e2e — same driver as ee_e2e, different entry script.
    _add_eval_budget_e2e_parser(
        sub, "simple_loop_e2e", "simple_loop",
        "Full SIMPLE LOOP (Baseline 0) run against a real repo; local-only",
    )

    # eval_trace
    p_et = sub.add_parser(
        "eval_trace", help="Seed-only run; assert OTel trace collection"
    )
    p_et.set_defaults(func=_cmd_eval_trace)
    p_et.add_argument(
        "--repo-url",
        default="https://github.com/julianghadially/FactChecker",
    )
    p_et.add_argument(
        "--program",
        default="src.factchecker.simple.factchecker_program:FactCheckerProgram",
    )
    p_et.add_argument(
        "--trainset-path",
        default="data/FacTool_QA_train_10examples_normalized.jsonl",
    )
    p_et.add_argument("--num-threads", type=int, default=1)
    p_et.add_argument(
        "--coding-harness",
        default="claude",
        choices=["claude", "opencode"],
        help="Provider coding harness (unused on the seed-only path; accepted "
        "for parity with ee_e2e).",
    )
    p_et.add_argument(
        "--coding-harness-model",
        default=None,
        help="Provider/model string for OpenCode; ignored in Claude mode.",
    )
    p_et.add_argument("--timeout", type=int, default=3600)

    # inspect_traces
    p_it = sub.add_parser(
        "inspect_traces",
        help="Architect → inspect_traces sub-agent (synthetic workspace)",
    )
    p_it.set_defaults(func=_cmd_inspect_traces)
    _add_harness_args(p_it)

    # architect_trace (G2 — architect-session trace capture)
    p_at = sub.add_parser(
        "architect_trace",
        help="Real architect session collects iteration_{N}_architect_trace/ (synthetic)",
    )
    p_at.set_defaults(func=_cmd_architect_trace)
    _add_harness_args(p_at)

    # meta_reflection (G5 — post-run meta-reflection golden nuggets)
    p_mr = sub.add_parser(
        "meta_reflection",
        help="Meta-reflection surfaces a buried nugget from seeded architect traces",
    )
    p_mr.set_defaults(func=_cmd_meta_reflection)
    _add_harness_args(p_mr)

    p_ri = sub.add_parser(
        "run_integrity",
        help="Run-integrity inspector classifies a hard-coded-testset workspace as cheating",
    )
    p_ri.add_argument(
        "--scenario", choices=["cheat", "clean"], default="cheat",
        help="cheat: pipeline hard-codes testset answers (expect non-no_cheating); "
        "clean: same pipeline without the lookup (expect no_cheating).",
    )
    p_ri.set_defaults(func=_cmd_run_integrity)
    _add_harness_args(p_ri)

    p_ab = sub.add_parser(
        "auditors_back_to_back",
        help="Meta-reflection then run-integrity in one process (the finalization shape)",
    )
    p_ab.set_defaults(func=_cmd_auditors_back_to_back)
    _add_harness_args(p_ab)

    # analysis_agent
    p_aa = sub.add_parser(
        "analysis_agent",
        help="Analysis agent writes iteration_{N}_change_summary.md and "
        "leaves memory.md alone (synthetic workspace)",
    )
    p_aa.set_defaults(func=_cmd_analysis_agent)
    _add_harness_args(p_aa)

    # close_iteration
    p_ci = sub.add_parser(
        "close_iteration",
        help="Architect writes its memory.md reflection and calls the real "
        "close_iteration tool (synthetic workspace + tiny git repo)",
    )
    p_ci.set_defaults(func=_cmd_close_iteration)
    _add_harness_args(p_ci)

    # session_resume
    p_sr = sub.add_parser(
        "session_resume",
        help="Watchdog-resumed architect session retains the first session's "
        "context (secret token delivered only in session 1's transcript)",
    )
    p_sr.set_defaults(func=_cmd_session_resume)
    _add_harness_args(p_sr)

    # comparison_credits
    p_cc = sub.add_parser(
        "comparison_credits",
        help="A real architect completes the comparison-credit acknowledgement "
        "round-trip through the real run_eval tool (evaluator patched; one credit)",
    )
    p_cc.set_defaults(func=_cmd_comparison_credits)
    _add_harness_args(p_cc)

    # test_sampling (finalization draws the held-out set N times)
    p_ts = sub.add_parser(
        "test_sampling",
        help="Finalization samples a one-row held-out set 3x for the final "
        "program and 1x for the seed; reports every draw and the means",
    )
    p_ts.set_defaults(func=_cmd_test_sampling)
    p_ts.add_argument(
        "--repo-url", default="https://github.com/julianghadially/FactChecker",
    )
    p_ts.add_argument("--initial-branch", default=DEFAULT_SEED_BRANCH)
    p_ts.add_argument("--samples", type=int, default=3)
    p_ts.add_argument("--seed-samples", type=int, default=1)
    p_ts.add_argument("--timeout", type=int, default=1800)
    p_ts.add_argument("--test-id", default=None)

    # variance_tool (real architect session calls run_variance_estimate once)
    p_vt = sub.add_parser(
        "variance_tool",
        help="A real architect session calls the zero-argument "
        "run_variance_estimate tool once against the FactChecker fixture "
        "(subsample 2 x 2 runs = 4 rows); asserts state, records, trace dirs",
    )
    p_vt.set_defaults(func=_cmd_variance_tool)
    _add_harness_args(p_vt)
    p_vt.add_argument(
        "--repo-url", default="https://github.com/julianghadially/FactChecker",
    )
    p_vt.add_argument("--initial-branch", default=DEFAULT_SEED_BRANCH)
    p_vt.add_argument("--timeout", type=int, default=1800)
    p_vt.add_argument("--test-id", default=None)

    # holdout_guard (behavioral held-out test-set fence)
    p_hg = sub.add_parser(
        "holdout_guard",
        help="Behavioral held-out test-set guard — armed agent cannot read holdout",
    )
    p_hg.set_defaults(func=_cmd_holdout_guard)
    p_hg.add_argument(
        "--agent",
        default="inspection",
        choices=["inspection", "architect"],
        help="Which armed agent to probe.",
    )
    _add_harness_args(p_hg)

    return parser


def _add_eval_budget_e2e_parser(sub, name: str, method: str, help_text: str) -> None:
    """Parser for the eval-row-budgeted arm (simple_loop)."""
    p_van = sub.add_parser(name, help=help_text)
    p_van.set_defaults(func=_cmd_e2e, method=method)
    p_van.add_argument(
        "--repo-url",
        default="https://github.com/julianghadially/FactChecker",
    )
    p_van.add_argument(
        "--program",
        default="src.factchecker.simple.factchecker_program:FactCheckerProgram",
    )
    p_van.add_argument("--metric", default="src.codeevolver.metric:metric")
    p_van.add_argument(
        "--trainset-path",
        default="data/FacTool_QA_train_10examples_normalized.jsonl",
    )
    p_van.add_argument("--testset-path", default=DEFAULT_TESTSET_PATH)
    p_van.add_argument("--initial-branch", default=DEFAULT_SEED_BRANCH)
    # The eval-row budget is what bounds this arm's run.
    p_van.add_argument("--max-metric-calls", type=int, default=40)
    p_van.add_argument("--max-iterations", type=int, default=2)
    p_van.add_argument("--comparison-set-size", type=int, default=None)
    p_van.add_argument("--primary-metric", default="accuracy")
    p_van.add_argument(
        "--coding-harness",
        default="claude",
        choices=["claude", "opencode"],
        help="Provider coding harness for the architect loop.",
    )
    p_van.add_argument(
        "--coding-harness-model",
        default=None,
        help="Provider/model string for OpenCode; ignored in Claude mode.",
    )
    _add_post_run_agent_args(p_van)
    p_van.add_argument("--timeout", type=int, default=7200)



def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
