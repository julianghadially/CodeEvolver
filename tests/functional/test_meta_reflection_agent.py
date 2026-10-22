"""Functional test for the post-run meta-reflection agent (Group G5).

Plants unique "golden nuggets" inside seeded architect-session traces, runs the
REAL deployed ``run_meta_reflection()`` (which spawns a real read-only inspection
agent over those traces + memory + the CodeEvolver source), and asserts the
written ``meta_reflection.md`` surfaces the planted signal.

This is the *consumption* half of the meta-reflection pipeline (the *capture*
half is ``test_architect_trace_capture.py``). It is the direct analogue of the D1
``inspect_traces`` test: a deterministic signal is buried where only an agent
that actually reads the architect traces will find it, and we assert that signal
appears in the output.

The story planted in ``iteration_2_architect_trace/``: the architect IGNORED a
glaring tool error (``WIDGET_SECRET_MISSING_7F3A``) and wasted the iteration
re-chasing a dead-end it had already ruled out (``XQZ_RERANK_DEADEND``). The
meta-reflection agent should flag the ignored error in its proposals.

Expensive; run manually.

To run::

    python local_test_app.py meta_reflection --coding-harness claude
"""

from __future__ import annotations

import json
import os
import textwrap

from tests._test_env import SyntheticTestEnvironment, parse_marker, seed_architect_trace_dir

_PROVIDER_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPINFRA_API_KEY")


def _provider_env_keys() -> dict[str, str]:
    return {k: os.environ[k] for k in _PROVIDER_ENV_VARS if os.environ.get(k)}


# --- Golden nuggets ----------------------------------------------------

# Buried in a tool_result with is_error:true that the architect ignored.
# Unique + unguessable, so a match in the output proves the agent read the trace.
IGNORED_ERROR_TOKEN = "WIDGET_SECRET_MISSING_7F3A"
# In a thinking block where the architect chases a ruled-out dead-end.
WASTED_HYPOTHESIS_TOKEN = "XQZ_RERANK_DEADEND"

ITERATION_INDICES = [0, 1, 2]
NUGGET_ITERATION = 2


# --- Memory fixtures ---------------------------------------------------

INITIAL_MEMORY_MD = """\
# CodeEvolver Memory

## Candidate Pool
- codeevolver-20260614000000-seed (PRIMARY): baseline pipeline

## Overall Progress
- Iteration 0: baseline pipeline scored 0.62.
- Iteration 1: a re-ranking experiment regressed to 0.55 and was abandoned.
- Iteration 2: scored 0.40 (a regression). See the iteration history.

## Macro Approaches
- Re-ranking: tried in iteration 1, regressed, abandoned as unpromising.
"""


# Engine 3 removed the per-iteration iteration_history_{N}.md files; the
# meta agent's on-disk inputs are memory.md + iteration_change_summaries.md
# + the architect session traces. The combined change summaries deliberately
# do NOT reveal the tool error — the agent must read the architect trace to
# find it (mirrors D1's compact-index-without-error).
COMBINED_CHANGE_SUMMARIES = textwrap.dedent(
    """\
    # Change Description Iteration 0 (seed)
    Seed program: Baseline FactChecker (subsample 0.62).

    # Change Description Iteration 1 (codeevolver-20260614000000-rerank @ a1111111)
    Added a retrieval re-ranking stage; subsample regressed 0.62 -> 0.55 and
    the approach was abandoned.

    # Change Description Iteration 2 (codeevolver-20260614000000-fused @ a2222222)
    Fused the research and judging calls into one prompt; subsample regressed
    to 0.40. Root cause not summarized here — see the trace.
    """
)


def _architect_turn(role: str, blocks: list[dict], **extra) -> dict:
    entry = {"ts": "2026-06-14T00:00:00+00:00", "iteration": NUGGET_ITERATION,
             "role": role, "blocks": blocks}
    entry.update(extra)
    return entry


def _build_nugget_architect_trace() -> str:
    """`iteration_2_architect_trace/` — the architect ignores a glaring error."""
    lines = [
        _architect_turn("assistant", [
            {"kind": "thinking", "text": "Score is low. I'll baseline on the "
             "primary branch, then try a change.", "signature_present": True},
            {"kind": "tool_use", "name": "mcp__ce__run_eval",
             "id": "toolu_1", "input": {"eval_sample": "train"}},
        ]),
        _architect_turn("user", [
            {"kind": "tool_result", "tool_use_id": "toolu_1", "is_error": False,
             "content": "{\"primary_score\": 0.62}",
             "truncated": False},
        ]),
        _architect_turn("assistant", [
            {"kind": "tool_use", "name": "mcp__ce__coding_agent", "id": "toolu_2",
             "input": {"mode": "execute",
                       "change_request": "Add a WidgetClient enrichment step."}},
        ]),
        # The glaring, ignored error.
        _architect_turn("user", [
            {"kind": "tool_result", "tool_use_id": "toolu_2", "is_error": True,
             "content": (
                 f"ERROR: {IGNORED_ERROR_TOKEN} — the WidgetClient could not "
                 "authenticate; the widget API secret is missing from the "
                 "environment. The enrichment step produced no output and every "
                 "row scored 0. Set WIDGET_API_KEY to enable it."
             ),
             "truncated": False},
        ]),
        # The architect ignores the error and re-chases a ruled-out dead-end.
        _architect_turn("assistant", [
            {"kind": "thinking", "text": (
                f"I'll ignore that and pursue {WASTED_HYPOTHESIS_TOKEN}: add the "
                "re-ranking step again. (Memory says iteration 1 already showed "
                "re-ranking regresses, but I'll try it anyway.)"
            ), "signature_present": True},
            {"kind": "tool_use", "name": "mcp__ce__run_eval", "id": "toolu_3",
             "input": {"eval_sample": "train"}},
        ]),
        _architect_turn("user", [
            {"kind": "tool_result", "tool_use_id": "toolu_3", "is_error": False,
             "content": "{\"primary_score\": 0.40}", "truncated": False},
        ]),
        _architect_turn("result", [], usage={"input_tokens": 5000, "output_tokens": 800},
                        meta={"subtype": "success", "num_turns": 6,
                              "duration_ms": 42000, "total_cost_usd": 0.12,
                              "session_id": "sess-iter2"}),
    ]
    return lines


def _build_clean_architect_trace(iteration: int) -> str:
    """A clean, uneventful architect trace for a non-nugget iteration."""
    lines = [
        {"ts": "2026-06-14T00:00:00+00:00", "iteration": iteration, "role": "assistant",
         "blocks": [{"kind": "thinking", "text": "Routine iteration; small prompt tweak.",
                     "signature_present": True},
                    {"kind": "tool_use", "name": "mcp__ce__run_eval", "id": "t1",
                     "input": {"eval_sample": "train"}}]},
        {"ts": "2026-06-14T00:00:00+00:00", "iteration": iteration, "role": "user",
         "blocks": [{"kind": "tool_result", "tool_use_id": "t1", "is_error": False,
                     "content": "{\"primary_score\": 0.55}", "truncated": False}]},
        {"ts": "2026-06-14T00:00:00+00:00", "iteration": iteration, "role": "result",
         "blocks": [], "usage": {"output_tokens": 200},
         "meta": {"subtype": "success", "num_turns": 2}},
    ]
    return lines


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
        from src.engine.paths import memory_dir

        IGNORED_ERROR_TOKEN = {IGNORED_ERROR_TOKEN!r}
        WASTED_HYPOTHESIS_TOKEN = {WASTED_HYPOTHESIS_TOKEN!r}

        run_error = None
        out = None
        try:
            out = run_meta_reflection(
                workspace={workspace!r},
                method="ee",
                num_iterations=3,
                seed_test_score=None,
                best_candidate_test_score=None,
                iteration_indices={ITERATION_INDICES!r},
                agent_model=None,
                coding_harness={coding_harness!r},
                coding_harness_model={coding_harness_model!r},
            )
        except Exception as exc:  # noqa: BLE001
            run_error = f"{{type(exc).__name__}}: {{exc}}"
            traceback.print_exc()

        out_path = os.path.join(memory_dir(), "meta_reflection.md")
        content = ""
        if os.path.exists(out_path):
            with open(out_path) as f:
                content = f.read()
        # Prefer the file; fall back to the returned string.
        text = content or (out or "")

        payload = {{
            "produced": bool(out),
            "file_written": os.path.exists(out_path),
            "content_len": len(text),
            "mentions_ignored_error": IGNORED_ERROR_TOKEN in text,
            "mentions_wasted_hypothesis": WASTED_HYPOTHESIS_TOKEN in text,
            "cites_iteration_2": ("iteration 2" in text.lower()
                                  or "iteration_2" in text.lower()),
            "run_error": run_error,
        }}
        print("META_REFLECTION_RESULT:" + json.dumps(payload))
        """
    )


def run_meta_reflection_agent(
    env: SyntheticTestEnvironment,
    *,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> dict:
    """Run the meta-reflection golden-nugget test against a `SyntheticTestEnvironment`.

    Returns a dict for pass/fail reporting. Primary signal:
    ``mentions_ignored_error`` — the agent surfaced the unique error token that
    is ONLY present in the seeded architect trace, proving it read the trace.
    """
    provider_keys = _provider_env_keys()
    try:
        # Workspace + memory fixtures.
        env.seed_file(
            f"{env.workspace_dir}/.env",
            "".join(f"{k}={v}\n" for k, v in provider_keys.items()),
        )
        env.seed_file(
            f"{env.workspace_dir}/codeevolver.md",
            "# Architecture: Baseline FactChecker\nStub for the meta-reflection test.\n",
        )
        env.seed_file(f"{env.memory_dir}/memory.md", INITIAL_MEMORY_MD)
        env.seed_file(
            f"{env.memory_dir}/iteration_change_summaries.md",
            COMBINED_CHANGE_SUMMARIES,
        )
        # Architect traces — the nugget lives in iteration 2; iteration 1 is clean.
        seed_architect_trace_dir(
            env,
            f"{env.architect_traces_dir}/iteration_1_architect_trace",
            _build_clean_architect_trace(1),
        )
        seed_architect_trace_dir(
            env,
            f"{env.architect_traces_dir}/iteration_2_architect_trace",
            _build_nugget_architect_trace(),
        )

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

        payload = parse_marker(stdout_lines, "META_REFLECTION_RESULT")
        if payload is None:
            tail = "\n".join(stdout_lines[-40:])
            return {
                "produced": False,
                "file_written": False,
                "content_len": 0,
                "mentions_ignored_error": False,
                "mentions_wasted_hypothesis": False,
                "cites_iteration_2": False,
                "error": f"No META_REFLECTION_RESULT marker on stdout. Last lines:\n{tail}",
            }
        payload.setdefault("error", payload.get("run_error"))
        return payload
    finally:
        env.cleanup()


def report_meta_reflection_agent(result: dict) -> int:
    """Pass/fail reporting for the local entrypoint.

    Primary assertion: the output surfaces ``IGNORED_ERROR_TOKEN``, which is
    present ONLY in the seeded architect trace — proving the agent read it.
    The other signals are logged but not required (D1-style robustness).
    """
    print("\n" + "=" * 60)
    for k in (
        "produced", "file_written", "content_len", "mentions_ignored_error",
        "mentions_wasted_hypothesis", "cites_iteration_2",
    ):
        print(f"{k:27}: {result.get(k)}")
    if result.get("error"):
        print(f"[error] {result['error']}")
    print("=" * 60 + "\n")

    failures: list[str] = []
    if not (result.get("produced") and result.get("file_written")):
        failures.append("meta_reflection.md was not produced")
    if not result.get("mentions_ignored_error"):
        failures.append(
            f"output does not surface the buried error token {IGNORED_ERROR_TOKEN!r} "
            "— the agent likely did not read the seeded architect trace"
        )
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "PASS: meta-reflection read the architect trace and surfaced the buried "
        f"error (soft signals: wasted_hypothesis="
        f"{result.get('mentions_wasted_hypothesis')}, "
        f"cites_iter2={result.get('cites_iteration_2')})."
    )
    return 0
