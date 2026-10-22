"""Functional test for architect-session trace collection (Group G2).

Makes a REAL call to the IterationArchitect through the selected coding harness
with ``capture_architect_traces=True`` (the ``enable_meta_reflection``
config knob), then asserts the architect's own session
trace was collected to ``{architect_traces_dir()}/iteration_{N}_architect_trace/``.

This is the *capture* half of the meta-reflection pipeline (the *consumption*
half is ``test_meta_reflection_agent.py``). It runs the deployed capture path
end-to-end: a real Claude Agent SDK (or OpenCode) architect session whose
message stream is serialized by ``claude.py``/``architect_trace.py``. The
eval/coding/inspect tools are stubbed (as in the D1 ``inspect_traces`` test) so
the session is short and deterministic — capture happens in the harness message
loop regardless of which tools the architect calls.

Assertions differ by harness:
- **claude**: the trace file exists with real captured content — a tool_use, a
  tool_result, a result line, AND ≥1 non-empty ``thinking`` block (the latter
  proves the SDK message-stream capture + ``display:"summarized"`` path works on
  the live model).
- **opencode**: rich capture runs via post-invocation ``opencode export``
  serialization (``output.py::serialize_opencode_export``); we require the same
  tool_use/tool_result nuggets and a result line as claude, but ``thinking`` is
  reported informationally only (reasoning availability is provider/model-
  dependent — e.g. OpenAI exports encrypted reasoning with empty text). The
  ``captured:false`` stub must be ABSENT on a successful session. If the
  OpenCode session itself fails (binary missing, model unreachable), the
  entrypoints SKIP and log — they do not fail the suite.

Expensive; run manually.

To run::

    python local_test_app.py architect_trace --coding-harness claude
"""

from __future__ import annotations

import json
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


ITERATION_NUM = 4
SEED_BRANCH = "codeevolver-20260419000000-seed"

STUB_CODEEVOLVER_MD = (
    "# Architecture: Baseline FactChecker\n"
    "\n"
    "Stub file for the architect-trace capture integration test. Only exists\n"
    "so the architect's Phase 1 read of codeevolver.md succeeds.\n"
    "\n"
    "parent_module_path: stub_pkg.stub_mod:StubProgram\n"
    "metric_module_path: stub_pkg.stub_mod:stub_metric\n"
)

INITIAL_MEMORY_MD = """\
# CodeEvolver Memory

## Candidate Pool
- codeevolver-20260419000000-seed (PRIMARY): baseline pipeline

## Observations
- Iteration 0: Baseline pipeline scored 0.62.
- Iteration 2: A re-ranking experiment showed promise but was not finished.
"""

# --- Capture-FIDELITY nuggets ------------------------------------------
#
# This test verifies the capture pipeline RECORDS what really flowed through the
# live SDK session — a different property than meta_reflection, which tests
# autonomous DISCOVERY (and therefore must NOT instruct the agent). Here we
# control exact tokens and assert they were captured verbatim:
#
#   EMITTED_NUGGET      — the architect is told to put this exact string in its
#                         change_request; it must land in a captured assistant
#                         `tool_use` block (proves assistant-output capture).
#   TOOLRESULT_NUGGET   — returned by the stubbed run_eval; it can ONLY reach the
#                         trace via the real SDK UserMessage -> ToolResultBlock
#                         capture path (the branch added in claude.py). This is
#                         the headline nugget — analogous to the FireCrawl token.
EMITTED_NUGGET = "ZEPHYR_QUASAR_42"
TOOLRESULT_NUGGET = "CAPTURE_PROBE_TOOLRESULT_9QXZ"

# Reasoning-inducing instructions: forming a hypothesis reliably engages
# extended thinking, so the captured trace contains a thinking block. The
# stubbed tools resolve instantly, keeping the session to a few turns. The
# architect is told to echo EMITTED_NUGGET verbatim so we can confirm its real
# output was captured.
_ADDITIONAL_INSTRUCTIONS = f"""\
This is an automated integration test verifying that your own session trace is
captured. Keep it brief — a few turns total. Steps:

1. Briefly reason about ONE concrete hypothesis for improving the pipeline
   (think it through before acting).
2. Call run_eval(eval_sample="train") to get a baseline (it returns a fixed
   stub score).
3. Call coding_agent(mode="execute", change_request=...) with one specific change
   that matches your hypothesis. The change_request MUST include the exact
   verification code {EMITTED_NUGGET} somewhere in its text (this is required by
   the test harness so it can confirm your output was captured).
4. Call close_iteration(register_branch={{"branch":
   "codeevolver-20260419000000-seed", "was_primary": true, "is_primary": true}},
   improvement_mode="exploitation", memory_updated=false) — it is stubbed.
5. EXIT. You do NOT need to update memory.md for this test.
"""


def _build_driver_script(
    *,
    engine_root: str,
    workspace: str,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> str:
    git_fixture = git_fixture_snippet(
        branch=SEED_BRANCH,
        message="seed: architect-trace test fixture",
        stage_all=False,
        ignore=(".env",),
    )
    return textwrap.dedent(
        f"""\
        import json, os, subprocess, sys, traceback
        sys.path.insert(0, {engine_root!r})
        os.chdir({engine_root!r})
        os.environ.setdefault("IS_SANDBOX", "1")

        from src.engine.optimizer.v3.iteration_architect import (
            ArchitectConfig, IterationArchitect,
        )
        from src.engine.optimizer.v3.iteration_context import IterationContext
        from src.engine.optimizer.v3.architect_trace import architect_trace_path, read_architect_trace
        from src.engine.evaluator.config import EvaluatorConfig

        ITERATION_NUM = {ITERATION_NUM}
        ADDITIONAL_INSTRUCTIONS = {_ADDITIONAL_INSTRUCTIONS!r}
        SEED_BRANCH = {SEED_BRANCH!r}
        WORKSPACE = {workspace!r}

        # Real repo so the close-time git hooks have something to read.
        # `.env` holds live provider keys and stays gitignored.
        {git_fixture}

        class _StubState:
            primary_branch = SEED_BRANCH

        class _TestArchitect(IterationArchitect):
            \"\"\"All tools stubbed; the real harness session + capture run.\"\"\"

            def build_tool_registry(self, *, agent_harness):
                ctx = self.context

                def run_eval(eval_sample, filter=None):
                    if ctx.initial_subsample_score is None:
                        ctx.initial_subsample_score = 0.40
                        ctx.subsample_ids = [0, 1, 2]
                    ctx.record_eval(START_SHA, 0.40)
                    # The nugget rides back in the tool's return value. It can
                    # only appear in the captured trace via the real SDK
                    # UserMessage -> ToolResultBlock capture path.
                    return {{
                        "scores_by_metric": {{"accuracy": 0.40}},
                        "primary_score": 0.40,
                        "capture_probe": {TOOLRESULT_NUGGET!r},
                        "trace_meta": [
                            {{"example_id": i, "score": 0.40, "failed": False,
                              "feedback": {TOOLRESULT_NUGGET!r}, "error_summary": None}}
                            for i in range(3)
                        ],
                    }}

                def coding_agent(mode, change_request):
                    ctx.coding_agent_called = True
                    ctx.change_requests.append(change_request)
                    return {{
                        "ok": True, "success": True,
                        "summary": "stub coding agent applied a no-op.",
                    }}

                def inspect_traces(query, trace="child"):
                    return {{"ok": True, "summary": "stub inspect_traces: nothing notable."}}

                def close_iteration(register_branch, improvement_mode,
                                    memory_updated):
                    ctx.registered_branch = (
                        register_branch if isinstance(register_branch, dict)
                        else {{}}
                    )
                    ctx.claimed_improvement_mode = improvement_mode
                    ctx.close_called = True
                    ctx.child_branch = SEED_BRANCH
                    return {{"ok": True}}

                return {{
                    "run_eval": run_eval,
                    "coding_agent": coding_agent,
                    "inspect_traces": inspect_traces,
                    "close_iteration": close_iteration,
                }}

        # Engine 3 iteration-start snapshot (the loop locks the context at
        # construction; there is no select_candidate).
        ctx = IterationContext(iteration_idx=ITERATION_NUM)
        ctx.parent_locked = True
        ctx.start_branch = SEED_BRANCH
        ctx.start_sha = START_SHA
        ctx.primary_branch_at_start = SEED_BRANCH

        evaluator = EvaluatorConfig(
            program_factory="stub_pkg.stub_mod:stub_factory",
            metric_fns=["stub_pkg.stub_mod:stub_metric"],
            metric_names=["accuracy"],
            primary_metric="accuracy",
            trainset_path="{workspace}/.codeevolver/trainset.json",
        )
        cfg = ArchitectConfig(
            run_timestamp="20260419000000",
            additional_instructions=ADDITIONAL_INSTRUCTIONS,
            program_path=None,
            evaluator=evaluator,
            trainset_ids=[0, 1, 2],
            max_turns=30,
            coding_harness={coding_harness!r},
            coding_harness_model={coding_harness_model!r},
            capture_architect_traces=True,   # <-- turns on architect-session capture
        )
        architect = _TestArchitect(
            context=ctx, state=_StubState(), workspace={workspace!r}, config=cfg,
        )

        run_error = None
        try:
            architect.run()
        except Exception as exc:  # noqa: BLE001
            run_error = f"{{type(exc).__name__}}: {{exc}}"
            traceback.print_exc()

        # Analyze the captured architect trace (written during the session by
        # the harness, regardless of post-session watchdog behavior).
        tp = architect_trace_path(ITERATION_NUM)
        try:
            lines = read_architect_trace(ITERATION_NUM)
        except Exception:
            lines = []

        def _blocks():
            for ln in lines:
                for b in ln.get("blocks", []) or []:
                    yield b

        has_thinking = any(
            b.get("kind") == "thinking" and (b.get("text") or "").strip()
            for b in _blocks()
        )
        has_tool_use = any(b.get("kind") == "tool_use" for b in _blocks())
        has_tool_result = any(b.get("kind") == "tool_result" for b in _blocks())
        has_result_line = any(ln.get("role") == "result" for ln in lines)
        opencode_stub = any(
            (ln.get("meta") or {{}}).get("captured") is False for ln in lines
        )

        EMITTED_NUGGET = {EMITTED_NUGGET!r}
        TOOLRESULT_NUGGET = {TOOLRESULT_NUGGET!r}
        # Capture-fidelity: the tool-result nugget can ONLY be present if the
        # real SDK UserMessage->ToolResultBlock was captured; the emitted nugget
        # proves the architect's own assistant output (tool_use/text) was captured.
        captured_toolresult_nugget = any(
            b.get("kind") == "tool_result" and TOOLRESULT_NUGGET in str(b.get("content", ""))
            for b in _blocks()
        )
        captured_emitted_nugget = any(
            (b.get("kind") == "tool_use" and EMITTED_NUGGET in json.dumps(b.get("input", {{}})))
            or (b.get("kind") in ("text", "thinking") and EMITTED_NUGGET in (b.get("text") or ""))
            for b in _blocks()
        )

        payload = {{
            "session_ok": run_error is None,
            "trace_dir": tp,
            "trace_file_exists": bool(lines),
            "num_lines": len(lines),
            "has_thinking": has_thinking,
            "has_tool_use": has_tool_use,
            "has_tool_result": has_tool_result,
            "has_result_line": has_result_line,
            "captured_toolresult_nugget": captured_toolresult_nugget,
            "captured_emitted_nugget": captured_emitted_nugget,
            "opencode_stub": opencode_stub,
            "run_error": run_error,
        }}
        print("ARCHITECT_TRACE_RESULT:" + json.dumps(payload))
        """
    )


def run_architect_trace_capture(
    env: SyntheticTestEnvironment,
    *,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> dict:
    """Run the architect-trace-capture test against a `SyntheticTestEnvironment`.

    Returns a dict for pass/fail/skip reporting by the entrypoints. The
    harness-specific pass/skip logic (claude requires has_thinking; opencode
    skips on session failure) lives in the entrypoint in
    ``local_test_app.py``.
    """
    provider_keys = _provider_env_keys()
    try:
        env.seed_file(
            f"{env.workspace_dir}/.env",
            "".join(f"{k}={v}\n" for k, v in provider_keys.items()),
        )
        env.seed_file(f"{env.workspace_dir}/codeevolver.md", STUB_CODEEVOLVER_MD)
        env.seed_file(f"{env.memory_dir}/memory.md", INITIAL_MEMORY_MD)

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

        payload = parse_marker(stdout_lines, "ARCHITECT_TRACE_RESULT")
        if payload is None:
            tail = "\n".join(stdout_lines[-40:])
            return {
                "session_ok": False,
                "trace_file_exists": False,
                "num_lines": 0,
                "has_thinking": False,
                "has_tool_use": False,
                "has_tool_result": False,
                "has_result_line": False,
                "captured_toolresult_nugget": False,
                "captured_emitted_nugget": False,
                "opencode_stub": False,
                "error": f"No ARCHITECT_TRACE_RESULT marker on stdout. Last lines:\n{tail}",
            }
        payload.setdefault("error", payload.get("run_error"))
        return payload
    finally:
        env.cleanup()


def report_architect_trace_capture(result: dict, coding_harness: str) -> int:
    """Pass/skip/fail reporting for the local entrypoint.

    Returns a process exit code (0 pass/skip, 1 fail). Claude requires faithful
    capture: the tool-result nugget (proves the real SDK
    UserMessage->ToolResultBlock was recorded), the emitted nugget (proves the
    architect's own output was recorded), a result line, and a non-empty
    thinking block. OpenCode skips-and-logs on session failure; on success it
    requires the same nuggets and result line via the `opencode export`
    capture path, with thinking reported informationally only and the
    captured:false stub absent.
    """
    print("\n" + "=" * 60)
    for k in (
        "session_ok", "trace_file_exists", "num_lines", "has_thinking",
        "has_tool_use", "has_tool_result", "has_result_line",
        "captured_toolresult_nugget", "captured_emitted_nugget", "opencode_stub",
    ):
        print(f"{k:26}: {result.get(k)}")
    if result.get("error"):
        print(f"[error] {result['error']}")
    print("=" * 60 + "\n")

    if coding_harness == "opencode":
        if not result.get("session_ok") or not result.get("trace_file_exists"):
            print(
                "SKIP: OpenCode architect session did not complete / no trace "
                f"written (error={result.get('error')}). Nothing to assert "
                "about capture fidelity."
            )
            return 0
        failures = []
        if result.get("opencode_stub"):
            failures.append(
                "the captured:false stub is present — the session succeeded, so "
                "the `opencode export` capture path should have produced a rich "
                "trace instead"
            )
        if not result.get("has_result_line"):
            failures.append("no result line captured in the architect trace")
        if not result.get("captured_toolresult_nugget"):
            failures.append(
                f"the tool-result nugget {TOOLRESULT_NUGGET!r} (returned by the "
                "stubbed run_eval) was NOT found in any captured tool_result — "
                "the export tool-state capture path is broken"
            )
        if not result.get("captured_emitted_nugget"):
            failures.append(
                f"the emitted nugget {EMITTED_NUGGET!r} was NOT found in any "
                "captured tool_use/text — assistant-output capture is broken "
                "(or the model did not comply)"
            )
        if failures:
            print("FAIL:")
            for f in failures:
                print(f"  - {f}")
            return 1
        # Thinking is informational on opencode: reasoning-part availability is
        # provider/model-dependent (encrypted reasoning exports empty text).
        print(
            "PASS: OpenCode architect session captured via export — tool-result "
            "nugget + emitted nugget + result line present "
            f"(has_thinking={result.get('has_thinking')}, informational)."
        )
        return 0

    failures: list[str] = []
    if not result.get("session_ok"):
        failures.append(f"architect session raised: {result.get('error')}")
    if not result.get("trace_file_exists"):
        failures.append("no iteration_{N}_architect_trace/ was written")
    if not result.get("has_result_line"):
        failures.append("no result line captured in the architect trace")
    if not result.get("has_thinking"):
        failures.append(
            "no non-empty thinking block captured — the SDK message-stream "
            "thinking capture (display:summarized) path did not yield text"
        )
    if not result.get("captured_toolresult_nugget"):
        failures.append(
            f"the tool-result nugget {TOOLRESULT_NUGGET!r} (returned by the "
            "stubbed run_eval) was NOT found in any captured tool_result — the "
            "real SDK UserMessage->ToolResultBlock capture path is broken"
        )
    if not result.get("captured_emitted_nugget"):
        failures.append(
            f"the emitted nugget {EMITTED_NUGGET!r} (the architect was told to "
            "put it in its change_request) was NOT found in any captured "
            "tool_use/text — assistant-output capture is broken (or the model "
            "did not comply)"
        )
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "PASS: architect session captured faithfully — tool-result nugget + "
        "emitted nugget + thinking all present in the trace."
    )
    return 0
