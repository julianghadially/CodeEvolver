"""Integration test: a watchdog-resumed architect session truly retains the
first session's context.

Every other resume test only proves the resume is *routed* with the captured
session id. This test proves the resumed agent actually carries the earlier
conversation: a secret token is delivered ONLY inside session 1's transcript
(a stubbed `run_eval` tool result — deliberately NOT the system prompt, which
is re-sent on resume, and NOT the resume prompt), and the agent is told to
end its turn without closing. The production close-missing watchdog then
resumes the session with its standard prompt (which never mentions the
token). The agent can only write the token into memory.md if the resumed
session genuinely has the first session's context.

Session 1 protocol (via additional_instructions — token NOT included there):
  1. `run_eval(eval_sample="train")` — the stubbed baseline result carries
     `session_token` (first call only; later calls never re-reveal it).
  2. `coding_agent(...)` once (stub) — so the no-change watchdog stays quiet
     and the close-missing watchdog is the single production resume vehicle.
  3. End the turn WITHOUT updating memory.md and WITHOUT calling
     `close_iteration`.

Resume (production `_run_close_missing_watchdog_if_needed` prompt): follow
the close_iteration skill — update memory.md with the outcome reflection
(which, per the standing instructions, must include the session token from
the earlier run_eval result) and call the `close_iteration` tool.

Success checks (see ``session_resume_failures``):
  - ``watchdog_fired`` — the resume path was actually exercised (an agent
    that closes in session 1 never resumes; that run is a retryable miss).
  - ``close_called`` — the real close_iteration tool validated the close.
  - memory.md contains the secret token — only possible with retained
    session context.

To run::

    python local_test_app.py session_resume
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

ITERATION_NUM = 5

RESUME_BRANCH = "codeevolver-20260420000000-resume-test"

# Delivered ONLY in the stubbed baseline run_eval tool result (session 1's
# transcript). Never present in the system prompt, the resume prompt, or any
# seeded file — its later presence in memory.md is proof of retained context.
SESSION_TOKEN = "AMBER-FALCON-0427"


INITIAL_MEMORY_MD = f"""\
# CodeEvolver Memory

Seed branch: {RESUME_BRANCH}
Seed architecture: Baseline FactChecker

This file is maintained by YOU, the architect. Update it at the end of every
iteration (see the `close_iteration` skill).

## Candidate Pool
- {RESUME_BRANCH} (PRIMARY): seed program

## Overall Progress
- Iteration 0 (Seed Program): baseline established.

## Observations
No observations yet.
"""

STUB_CODEEVOLVER_MD = (
    "# Architecture: Baseline FactChecker\n"
    "\n"
    "Stub file for the session-resume integration test.\n"
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
# NOTE: the token itself must NOT appear here — these instructions land in
# the system prompt, which is re-sent on resume.

ADDITIONAL_INSTRUCTIONS = f"""\
This is an automated integration test of session resumption. `run_eval` and
`coding_agent` are stubbed. Follow this protocol EXACTLY:

1. Call `run_eval(eval_sample="train")` once. Its response includes a field
   `session_token`. Remember that token — you will need it later. Do NOT
   write it to any file yet, and do NOT repeat it in your visible replies.
2. Call `coding_agent(mode="execute", change_request="add retry to research
   call")` once.
3. Then END YOUR TURN IMMEDIATELY. Do NOT update memory.md. Do NOT call the
   `close_iteration` tool. (Ending without closing is intentional here — the
   system will re-prompt you.)
4. When you are LATER re-prompted to close the iteration: update memory.md
   with a short outcome reflection that MUST include the exact session_token
   you were given in step 1, then call the `close_iteration` tool with:
   register_branch = {{"branch": "{RESUME_BRANCH}", "was_primary": true,
   "is_primary": true, "promotion_evidence": "n/a"}},
   improvement_mode="exploitation", memory_updated=true.
5. Stay on the current git branch; do not create branches or commit.

Do not re-call run_eval after step 1 — the token is only issued once.
"""


# --- Driver script (runs inside the environment) ------------------------

def _build_driver_script(
    *,
    engine_root: str,
    workspace: str,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> str:
    """In-environment driver: real git fixture, stub run_eval/coding_agent,
    REAL close_iteration binding, one architect iteration (whose run()
    includes the production watchdogs), then a RESUME_RESULT marker."""
    git_fixture = git_fixture_snippet(
        branch=RESUME_BRANCH,
        message='seed: session-resume test fixture',
    )
    return textwrap.dedent(
        f"""\
        import json, os, subprocess, sys
        sys.path.insert(0, {engine_root!r})
        os.chdir({engine_root!r})
        os.environ.setdefault("IS_SANDBOX", "1")

        from src.engine.optimizer.v3.iteration_architect import (
            ArchitectConfig, IterationArchitect, hash_file, memory_md_path,
        )
        from src.engine.optimizer.v3.iteration_context import IterationContext
        from src.engine.optimizer.v3.agent_tools import close_iteration_tool
        from src.engine.evaluator.config import EvaluatorConfig
        from src.engine.services.subprocess_service import (
            git_branch_exists, git_get_sha, read_file,
        )

        ITERATION_NUM = {ITERATION_NUM}
        SESSION_TOKEN = {SESSION_TOKEN!r}
        RESUME_BRANCH = {RESUME_BRANCH!r}
        ADDITIONAL_INSTRUCTIONS = {ADDITIONAL_INSTRUCTIONS!r}
        WORKSPACE = {workspace!r}

        # --- Tiny REAL git repo: initial commit + codeevolver- branch ---
        {git_fixture}

        ctx = IterationContext(iteration_idx=ITERATION_NUM)
        ctx.parent_locked = True
        ctx.start_branch = RESUME_BRANCH
        ctx.start_sha = START_SHA
        ctx.primary_branch_at_start = RESUME_BRANCH
        ctx.memory_md_hash_at_start = hash_file(memory_md_path())
        ctx.subsample_size = 3

        class _StubState:
            primary_branch = RESUME_BRANCH

        token_issue_count = 0

        class _TestArchitect(IterationArchitect):
            \"\"\"run_eval/coding_agent stubbed; close_iteration REAL.

            The baseline run_eval result carries SESSION_TOKEN exactly once —
            the ONLY channel the token reaches the agent through, so its
            later appearance in memory.md proves the resumed session kept
            the first session's transcript.
            \"\"\"

            def build_tool_registry(self, *, agent_harness):
                ctx = self.context
                workspace = self.workspace

                def run_eval(eval_sample, filter=None):
                    global token_issue_count
                    sha = git_get_sha(workspace, "HEAD")
                    response = {{
                        "scores_by_metric": {{"accuracy": 0.30}},
                        "primary_score": 0.30,
                        "trace_meta": [
                            {{"example_id": i, "score": 0.3,
                              "failed": False, "feedback": None}}
                            for i in range(3)
                        ],
                    }}
                    if ctx.initial_subsample_score is None:
                        ctx.initial_subsample_score = 0.30
                        ctx.subsample_ids = [0, 1, 2]
                        token_issue_count += 1
                        response["session_token"] = SESSION_TOKEN
                        response["note"] = (
                            "Remember session_token — you will be asked to "
                            "record it later. It will not be shown again."
                        )
                    ctx.record_eval(
                        sha, 0.30, eval_sample="train",
                        branch=RESUME_BRANCH, n_rows=3,
                    )
                    return response

                def coding_agent(mode, change_request):
                    ctx.coding_agent_called = True
                    ctx.change_requests.append(change_request)
                    return {{
                        "ok": True,
                        "success": True,
                        "summary": "stub coding agent applied no-op change",
                    }}

                def close_iteration(register_branch, improvement_mode,
                                    memory_updated):
                    return close_iteration_tool(
                        ctx=ctx,
                        workspace=workspace,
                        register_branch=register_branch,
                        improvement_mode=improvement_mode,
                        memory_updated=memory_updated,
                    )

                return {{
                    "run_eval": run_eval,
                    "coding_agent": coding_agent,
                    "close_iteration": close_iteration,
                }}

        evaluator = EvaluatorConfig(
            program_factory="stub_pkg.stub_mod:stub_factory",
            metric_fns=["stub_pkg.stub_mod:stub_metric"],
            metric_names=["accuracy"],
            primary_metric="accuracy",
            trainset_path=f"{{WORKSPACE}}/.codeevolver/trainset.json",
        )

        cfg = ArchitectConfig(
            run_timestamp="20260420000000",
            additional_instructions=ADDITIONAL_INSTRUCTIONS,
            program_path=None,
            evaluator=evaluator,
            trainset_ids=[0, 1, 2],
            subsample_size=3,
            max_turns=60,
            coding_harness={coding_harness!r},
            coding_harness_model={coding_harness_model!r},
        )

        architect = _TestArchitect(
            context=ctx,
            state=_StubState(),
            workspace=WORKSPACE,
            config=cfg,
        )
        import traceback as _traceback
        run_error = None
        try:
            architect.run()  # includes the production watchdog resumes
        except Exception as exc:  # noqa: BLE001
            run_error = f"{{type(exc).__name__}}: {{exc}}"
            _traceback.print_exc()

        memory_md = read_file(memory_md_path()) or ""

        payload = {{
            "close_called": bool(ctx.close_called),
            "watchdog_fired": bool(getattr(ctx, "watchdog_fired", False)),
            "coding_agent_called": bool(ctx.coding_agent_called),
            "memory_changed": (
                hash_file(memory_md_path()) != ctx.memory_md_hash_at_start
            ),
            "token_in_memory": SESSION_TOKEN in memory_md,
            "token_issue_count": token_issue_count,
            "memory_md": memory_md,
            "run_error": run_error,
        }}
        print("RESUME_RESULT:" + json.dumps(payload))
        """
    )


# --- Pass/fail criteria -------------------------------------------------

# Model-driven and not 100% reliable (the agent may close early instead of
# ending its turn, which skips the resume path). Retry up to this many times.
RESUME_MAX_ATTEMPTS = 3


def session_resume_failures(result: dict) -> list[str]:
    """Return the list of assertion failures for one run (empty == pass)."""
    failures: list[str] = []
    if not result["coding_agent_called"]:
        failures.append("coding_agent was never invoked — protocol step 2 missed.")
    if not result["watchdog_fired"]:
        failures.append(
            "no watchdog resume fired — the agent likely closed in session 1, "
            "so the resume path was never exercised (retryable)."
        )
    if not result["close_called"]:
        failures.append("close_iteration was never (successfully) called.")
    if not result["memory_changed"]:
        failures.append("memory.md was not modified on the resumed turn.")
    if result.get("token_issue_count") != 1:
        failures.append(
            f"session_token was issued {result.get('token_issue_count')} times "
            "(expected exactly 1) — a re-issued token would invalidate the "
            "context-retention proof."
        )
    if not result["token_in_memory"]:
        failures.append(
            f"memory.md does not contain the session token {SESSION_TOKEN!r} — "
            "the resumed session did not retain the first session's context."
        )
    return failures


def summarize_result(result: dict) -> str:
    return (
        f"close_called={result.get('close_called')} "
        f"watchdog_fired={result.get('watchdog_fired')} "
        f"token_in_memory={result.get('token_in_memory')} "
        f"memory_changed={result.get('memory_changed')} "
        f"error={result.get('error') or result.get('run_error')}"
    )


# --- Runner ------------------------------------------------------------

def run_session_resume(
    env: SyntheticTestEnvironment,
    *,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> dict:
    """Run the session-resume integration test in `env`."""
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

        payload = parse_marker(stdout_lines, "RESUME_RESULT")

        if payload is None:
            tail = "\n".join(stdout_lines[-40:])
            return {
                "close_called": False,
                "watchdog_fired": False,
                "coding_agent_called": False,
                "memory_changed": False,
                "token_in_memory": False,
                "token_issue_count": 0,
                "memory_md": "",
                "run_error": None,
                "error": (
                    f"No RESUME_RESULT marker on stdout. "
                    f"Last stdout lines:\n{tail}"
                ),
            }

        payload.setdefault("error", payload.get("run_error"))
        return payload
    finally:
        env.cleanup()
