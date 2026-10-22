"""Integration test for the engine 3 iteration-closing path (memory.md
reflection + the `close_iteration` tool).

There is no ``outcome_reflection`` skill or reflection-file watchdog: the
architect writes its outcome reflection directly into
``{memory_dir}/memory.md`` (per the ``close_iteration`` skill) and finalizes
the iteration by calling the ``close_iteration`` tool with a
``register_branch`` payload.

Setup:

- Seeded ``/workspace/.env``, ``/workspace/codeevolver.md``,
  ``/memory/memory.md``, and a ``/traces/iteration_{N}_child/`` trace directory whose
  ``trace[]`` entries carry a golden-nugget token (ResearchAgent timeout)
  that the compact trace meta does NOT reveal.
- The workspace is initialized as a REAL tiny git repo inside the driver
  (initial commit + a ``codeevolver-`` branch) so the real
  ``agent_tools.close_iteration_tool`` binding can run its git validations
  (``git_branch_exists`` / ``git_is_dirty`` / ``git_get_sha``) unmodified.
- A stub-tool subclass of ``IterationArchitect``: ``run_eval`` and
  ``coding_agent`` are stubbed (no evaluator subprocess, no real coding
  agent); ``inspect_traces`` stays REAL
  (a production grounding path the reflection may take); ``close_iteration``
  is the REAL ``agent_tools.close_iteration_tool`` bound to the fixture repo.
- The loop's iteration-start snapshot is replicated on the context
  (``parent_locked=True``, ``start_branch`` / ``start_sha`` /
  ``primary_branch_at_start`` / ``memory_md_hash_at_start``), matching how
  `optimizer_loop` locks the context at construction in engine 3.
- A real Claude Agent SDK (or OpenCode) session drives the run end-to-end.

Success checks (see ``close_iteration_failures``):

- ``ctx.close_called`` is True — the architect called the real tool and its
  validations passed.
- ``ctx.registered_branch`` carries the fixture's ``codeevolver-`` branch.
- ``memory.md`` was updated with a reflection containing the planted golden
  nugget from the trace (the hidden module or its timeout error).

To run::

    python local_test_app.py close_iteration
"""

from __future__ import annotations

import json
import os
import textwrap

from tests._test_env import (
    seed_eval_trace_dir,
    SyntheticTestEnvironment,
    git_fixture_snippet,
    parse_marker,
)


# Provider API keys forwarded into the driver process / .env. ANTHROPIC for the
# claude harness, OPENAI for the opencode pass. Only keys present in the
# environment are forwarded.
_PROVIDER_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPINFRA_API_KEY")


def _provider_env_keys() -> dict[str, str]:
    return {k: os.environ[k] for k in _PROVIDER_ENV_VARS if os.environ.get(k)}


# --- Shared fixtures ----------------------------------------------------

ITERATION_NUM = 5

# The codeevolver- branch the driver creates in the fixture repo. It is both
# the iteration-start branch and the primary at start, so registering it with
# is_primary=true needs no promotion evidence.
CLOSE_BRANCH = "codeevolver-20260419000000-close-test"

# Buried in per-module trace entries of the child JSONL. Compact trace meta
# (delivered inline with ``run_eval``) and the seeded index file omit this —
# the reflection can only surface it by genuinely grounding in the trace:
# Reading the trace file directly OR calling the REAL ``inspect_traces``
# sub-agent, both valid production paths. Surfacing the token in the memory.md
# reflection is the golden-nugget assertion.
HIDDEN_MODULE = "ResearchAgent"
HIDDEN_ERROR = (
    "Connection timed out after 30s waiting for external research API"
)


# Engine 3 memory.md seed — matches the MEMORY_MD_TEMPLATE shape (Candidate
# Pool section; no "memory last updated" checkpoint marker). Deliberately does
# NOT mention the hidden module or timeouts, so their later presence in
# memory.md can only come from the architect's own reflection.
INITIAL_MEMORY_MD = f"""\
# CodeEvolver Memory

Run id: test-close-iteration
Seed branch: {CLOSE_BRANCH}
Seed architecture: Baseline FactChecker

This file is maintained by YOU, the architect. Update it at the end of every
iteration (see the `close_iteration` skill).

## Candidate Pool
- {CLOSE_BRANCH} (PRIMARY): seed program

## Overall Progress
- Iteration 0 (Seed Program): baseline established.

## Macro Approaches
(none yet)

## Observations
No observations yet.
"""


# Stub codeevolver.md so the architect system prompt's architecture read does
# not spin. The real seed step produces a much richer file; the test only
# needs a non-empty placeholder so the Read succeeds and the agent moves on
# to the tool cycle.
STUB_CODEEVOLVER_MD = (
    "# Architecture: Baseline FactChecker\n"
    "\n"
    "Stub file for the close-iteration integration test. The real seed\n"
    "step writes a full architecture description here; this placeholder\n"
    "exists only so the architect's initial read succeeds.\n"
    "\n"
    "parent_module_path: src.factchecker.simple.factchecker_program:FactCheckerProgram\n"
    "metric_module_path: src.codeevolver.metric:metric\n"
)


def _build_child_trace_rows() -> list[dict]:
    """Rows of ``/traces/iteration_{N}_child/`` with HIDDEN_ERROR in ResearchAgent."""
    statements = [
        "The Earth orbits the Sun",
        "Water boils at 100 degrees Celsius at sea level",
        "The Great Wall of China is visible from space",
    ]
    rows = []
    for i, stmt in enumerate(statements):
        rows.append(
            {
                "example_id": i,
                "score": 0.1,
                "feedback": None,
                "failed": False,
                "input": {"statement": stmt},
                "output": {"verdict": "insufficient evidence"},
                "trace": [
                    {
                        "signature_key": "ClaimExtractor",
                        "inputs": {"statement": stmt},
                        "output": {"claims": [stmt]},
                    },
                    {
                        "signature_key": HIDDEN_MODULE,
                        "inputs": {"claims": [stmt]},
                        "output": {
                            "__failed__": True,
                            "error": HIDDEN_ERROR,
                        },
                    },
                    {
                        "signature_key": "FireJudge",
                        "inputs": {
                            "claim": stmt,
                            "evidence": "No evidence available",
                        },
                        "output": {"judgment": "insufficient"},
                    },
                ],
            }
        )
    return rows



def _seed_synthetic_fixtures(env: SyntheticTestEnvironment, provider_keys: dict) -> None:
    """Seed the fixtures the test needs into the synthetic environment."""
    env.seed_file(
        f"{env.workspace_dir}/.env",
        "".join(f"{k}={v}\n" for k, v in provider_keys.items()),
    )
    # Keep the harness-staged skills and the .env out of git so the real
    # close_iteration tool's dirty-tree check passes on a clean close.
    env.seed_file(
        f"{env.workspace_dir}/.gitignore",
        ".claude/\n.env\n.opencode/\n",
    )
    env.seed_file(f"{env.workspace_dir}/codeevolver.md", STUB_CODEEVOLVER_MD)
    env.seed_file(f"{env.memory_dir}/memory.md", INITIAL_MEMORY_MD)
    seed_eval_trace_dir(
        env,
        f"{env.traces_dir}/iteration_{ITERATION_NUM}_child",
        _build_child_trace_rows(),
    )


# --- additional_instructions --------------------------------------------

def _build_instructions() -> str:
    """Instructions for a minimal, deterministic close cycle.

    The change itself is stubbed; the features under test are (a) the
    outcome reflection written into memory.md and (b) the real
    `close_iteration` tool call. The trace-file pointer is intentional — it
    gives the architect a concrete place to ground the reflection so it
    cites the hidden token instead of a generic rubber-stamp narrative.
    """
    return f"""\
This is an automated integration test. `run_eval` and `coding_agent` are
stubbed; run exactly one minimal cycle:

1. `run_eval(eval_sample="train")` — baseline, one call.
2. `coding_agent(mode="execute", change_request="add retry to research call")` —
   one call.
3. `run_eval(eval_sample="train")` — post-change eval, one call.
4. Stay on the current git branch `{CLOSE_BRANCH}` — do NOT create new
   branches and do NOT commit anything yourself (the coding agent is a stub).
5. BEFORE EXITING, follow the `close_iteration` skill:
   a. Update the memory file (memory.md in the memory directory) with your
      outcome reflection. When writing it, open the child trace directory
      in the traces directory (`iteration_{ITERATION_NUM}_child/`) or use
      `inspect_traces`, and be specific about which module failed and why —
      that failure signal is the primary thing the reflection must carry
      forward.
   b. Call the `close_iteration` tool with:
      register_branch = {{"branch": "{CLOSE_BRANCH}", "was_primary": true,
      "is_primary": true, "promotion_evidence": "n/a"}},
      improvement_mode="exploitation", memory_updated=true.

Do NOT skip the memory.md update or the close_iteration call — those are the
features under test.
"""


# --- Driver script (runs inside the environment) ------------------------

def _build_driver_script(
    additional_instructions: str,
    *,
    engine_root: str,
    workspace: str,
    traces_dir: str,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> str:
    """Build the in-environment Python driver.

    Initializes the workspace as a real git repo with a ``codeevolver-``
    branch, replicates the engine 3 loop's iteration-start context snapshot,
    stubs ``run_eval`` / ``coding_agent``, binds the REAL
    ``close_iteration_tool``, runs one architect iteration, and prints
    ``CLOSE_RESULT:{json}`` on stdout for the outer harness.
    """
    git_fixture = git_fixture_snippet(
        branch=CLOSE_BRANCH,
        message='seed: close-iteration test fixture',
    )
    return textwrap.dedent(
        f"""\
        import json, os, subprocess, sys
        sys.path.insert(0, {engine_root!r})
        os.chdir({engine_root!r})
        # Claude CLI refuses `--dangerously-skip-permissions` under
        # root/sudo unless IS_SANDBOX=1 is set. Production's
        # optimization_entry.py sets this; our driver bypasses that path,
        # so we must replicate it here or the SDK session won't boot.
        os.environ.setdefault("IS_SANDBOX", "1")

        from src.engine.optimizer.v3.iteration_architect import (
            ArchitectConfig, IterationArchitect, hash_file, memory_md_path,
        )
        from src.engine.optimizer.v3.iteration_context import IterationContext
        from src.engine.optimizer.v3.agent_tools import (
            close_iteration_tool, inspect_traces_tool,
        )
        from src.engine.optimizer.v3.prompts import build_trace_inspection_prompt
        from src.engine.evaluator.config import EvaluatorConfig
        from src.engine.services.subprocess_service import (
            git_branch_exists, git_get_sha, read_file,
        )

        ITERATION_NUM = {ITERATION_NUM}
        HIDDEN_MODULE = {HIDDEN_MODULE!r}
        CLOSE_BRANCH = {CLOSE_BRANCH!r}
        ADDITIONAL_INSTRUCTIONS = {additional_instructions!r}
        TRACES_DIR = {traces_dir!r}
        WORKSPACE = {workspace!r}

        # --- Tiny REAL git repo: initial commit + codeevolver- branch ---
        {git_fixture}

        # --- Iteration-start snapshot (what the engine 3 loop stamps) ---
        ctx = IterationContext(iteration_idx=ITERATION_NUM)
        ctx.parent_locked = True
        ctx.start_branch = CLOSE_BRANCH
        ctx.start_sha = START_SHA
        ctx.primary_branch_at_start = CLOSE_BRANCH
        ctx.memory_md_hash_at_start = hash_file(memory_md_path())
        ctx.subsample_size = 3

        class _StubState:
            primary_branch = CLOSE_BRANCH

        class _TestArchitect(IterationArchitect):
            \"\"\"run_eval/coding_agent stubbed; inspect_traces + close_iteration REAL.\"\"\"

            def build_tool_registry(self, *, agent_harness):
                ctx = self.context
                workspace = self.workspace

                # --- run_eval: stub (no evaluator subprocess). Records the
                # REAL current HEAD sha so the close_iteration tool's
                # "registered HEAD was evaluated" validation exercises the
                # production check against genuine git state.
                def run_eval(eval_sample, filter=None):
                    sha = git_get_sha(workspace, "HEAD")
                    if ctx.initial_subsample_score is None:
                        ctx.initial_subsample_score = 0.30
                        ctx.subsample_ids = [0, 1, 2]
                        score = 0.30
                    else:
                        score = 0.45
                    ctx.record_eval(
                        sha, score, eval_sample="train",
                        branch=CLOSE_BRANCH, n_rows=3,
                    )
                    return {{
                        "scores_by_metric": {{"accuracy": score}},
                        "primary_score": score,
                        "trace_dir": (
                            f"{{TRACES_DIR}}/iteration_{{ITERATION_NUM}}_child"
                        ),
                        "trace_index_file": (
                            f"{{TRACES_DIR}}/iteration_{{ITERATION_NUM}}_child/index.jsonl"
                        ),
                        "trace_meta": [
                            {{"example_id": i, "score": 0.1,
                              "failed": False, "feedback": None}}
                            for i in range(3)
                        ],
                    }}

                # --- coding_agent: stub (captures the change_request; makes
                # no commit, so HEAD stays at the evaluated start sha).
                def coding_agent(mode, change_request):
                    ctx.coding_agent_called = True
                    ctx.change_requests.append(change_request)
                    return {{
                        "ok": True,
                        "success": True,
                        "summary": "stub coding agent applied no-op change",
                    }}

                # --- inspect_traces: REAL (mirrors D1) ---
                # A real read-only inspection sub-agent over the seeded child
                # trace — grounding via inspect_traces is a production path
                # the reflection may legitimately take; a canned response
                # would hand the golden nugget over without any trace read.
                def inspect_traces(query, trace="child"):
                    full_prompt = build_trace_inspection_prompt(
                        query=query,
                        iteration_num=ctx.iteration_idx,
                        suffix="child",
                    )
                    return inspect_traces_tool(
                        ctx=ctx,
                        query=full_prompt,
                        workspace=workspace,
                        agent_harness=agent_harness,
                    )

                # --- close_iteration: REAL production binding ---
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
                    "inspect_traces": inspect_traces,
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
            run_timestamp="20260419000000",
            additional_instructions=ADDITIONAL_INSTRUCTIONS,
            program_path=None,
            evaluator=evaluator,
            trainset_ids=[0, 1, 2],
            subsample_size=3,
            max_turns=80,
            coding_harness={coding_harness!r},
            coding_harness_model={coding_harness_model!r},
        )

        architect = _TestArchitect(
            context=ctx,
            state=_StubState(),
            workspace=WORKSPACE,
            config=cfg,
        )
        # Wrap architect.run() so the marker always prints even if the SDK
        # session raises. Without this, any exception (e.g. max_turns hit,
        # SDK boot failure) exits the driver with code 1 and the outer
        # harness reports "No CLOSE_RESULT marker" with no clue why.
        import traceback as _traceback
        run_error = None
        try:
            architect.run()
        except Exception as exc:  # noqa: BLE001
            run_error = f"{{type(exc).__name__}}: {{exc}}"
            _traceback.print_exc()

        memory_md = read_file(memory_md_path()) or ""
        memory_lower = memory_md.lower()

        registered = ctx.registered_branch or {{}}
        registered_branch = registered.get("branch")

        payload = {{
            "close_called": bool(ctx.close_called),
            "registered_branch_entry": registered,
            "registered_branch": registered_branch,
            "registered_branch_exists": bool(
                registered_branch
                and git_branch_exists(WORKSPACE, registered_branch)
            ),
            "claimed_improvement_mode": ctx.claimed_improvement_mode,
            "memory_updated_claimed": bool(ctx.memory_updated_claimed),
            "coding_agent_called": bool(ctx.coding_agent_called),
            "watchdog_fired": bool(getattr(ctx, "watchdog_fired", False)),
            "memory_changed": (
                hash_file(memory_md_path()) != ctx.memory_md_hash_at_start
            ),
            "memory_md": memory_md,
            "mentions_hidden_module": HIDDEN_MODULE.lower() in memory_lower,
            "mentions_timeout": (
                "timeout" in memory_lower or "timed out" in memory_lower
            ),
            "change_requests": list(ctx.change_requests),
            "run_error": run_error,
        }}
        print("CLOSE_RESULT:" + json.dumps(payload))
        """
    )


# --- Pass/fail criteria (shared by the entrypoints) ---------------------

# The architect writing the reflection and closing cleanly is model-driven and
# not 100% reliable: a weaker model may paraphrase the trace signal away or
# fumble the register_branch payload. The entrypoints retry up to this many
# times: a single miss is tolerated, and we only warn when it takes the final
# attempt. Mirrors INSPECT_MAX_ATTEMPTS / ANALYSIS_MAX_ATTEMPTS.
CLOSE_MAX_ATTEMPTS = 3


def close_iteration_failures(result: dict) -> list[str]:
    """Return the list of assertion failures for one run (empty == pass)."""
    failures: list[str] = []
    if not result["coding_agent_called"]:
        failures.append(
            "coding_agent was never invoked — stub cycle did not complete."
        )
    if not result["close_called"]:
        failures.append(
            "close_iteration was never (successfully) called — "
            "ctx.close_called is False."
        )
    if result["close_called"] and result.get("registered_branch") != CLOSE_BRANCH:
        failures.append(
            f"registered_branch does not carry {CLOSE_BRANCH!r} "
            f"(got {result.get('registered_branch')!r})."
        )
    if result["close_called"] and not result.get("registered_branch_exists"):
        failures.append(
            "the registered branch does not exist in the fixture git repo — "
            "the real close_iteration git validation should have prevented this."
        )
    if not result["memory_changed"]:
        failures.append(
            "memory.md was not modified — no outcome reflection was written."
        )
    # Trace-grounding: a reflection that exists but cites neither the hidden
    # module nor a timeout means the agent never grounded itself in the child
    # trace — neither by Reading the trace file nor via inspect_traces.
    if result["memory_changed"] and not (
        result["mentions_hidden_module"] or result["mentions_timeout"]
    ):
        failures.append(
            f"memory.md mentions neither {HIDDEN_MODULE!r} nor 'timeout' — "
            "the reflection was written but did not ground itself in the "
            "child trace (via Read or inspect_traces)."
        )
    return failures


# --- Runner ------------------------------------------------------------

def run_close_iteration(
    env: SyntheticTestEnvironment,
    *,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> dict:
    """Run the close-iteration integration test in `env`.

    Runs against any ``SyntheticTestEnvironment`` (locally,
    ``LocalSyntheticEnv``).
    ``coding_harness`` selects the provider harness; the runner sweeps both.
    """
    provider_keys = _provider_env_keys()

    try:
        _seed_synthetic_fixtures(env, provider_keys)

        driver = _build_driver_script(
            _build_instructions(),
            engine_root=env.engine_root,
            workspace=env.workspace_dir,
            traces_dir=env.traces_dir,
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
        )
        stdout_lines: list[str] = []
        for line in env.exec_python_driver(driver, env_vars=provider_keys):
            print(line, flush=True)
            stdout_lines.append(line)

        payload = parse_marker(stdout_lines, "CLOSE_RESULT")

        if payload is None:
            tail = "\n".join(stdout_lines[-40:])
            return {
                "close_called": False,
                "registered_branch_entry": {},
                "registered_branch": None,
                "registered_branch_exists": False,
                "claimed_improvement_mode": None,
                "memory_updated_claimed": False,
                "coding_agent_called": False,
                "watchdog_fired": False,
                "memory_changed": False,
                "memory_md": "",
                "mentions_hidden_module": False,
                "mentions_timeout": False,
                "change_requests": [],
                "run_error": None,
                "error": (
                    f"No CLOSE_RESULT marker on stdout. "
                    f"Last stdout lines:\n{tail}"
                ),
            }

        payload.setdefault("error", payload.get("run_error"))
        return payload
    finally:
        env.cleanup()
