"""Copyright © 2026 440 Labs LLC

Unit tests for post-run meta-reflection (offline — no Claude Agent SDK).

Covers the prompt builder (structured-proposals format, the five surfaces,
holdout guardrails) and the `run_meta_reflection` driver's contracts: it writes
the agent output to a CodeEvolver-owned file, runs with NO deny paths (engine 3:
it runs post-finalization, after the run is over), and is strictly non-fatal.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.engine.optimizer.meta import meta_reflection as mr


def test_prompt_names_all_five_surfaces_and_guardrails():
    prompt = mr.build_meta_reflection_prompt(
        method="ee",
        num_iterations=5,
        seed_test_score=0.60,
        best_candidate_test_score=0.71,
        workspace="/workspace/repo",
        iteration_indices=[0, 1, 2, 3, 4],
    )
    for surface in (
        "A. Architect system prompt",
        "B. Skills",
        "C. Tool contracts",
        "D. Iteration controls",
        "E. Memory schema",
    ):
        assert surface in prompt
    # Advisory + holdout guardrails are stated.
    assert "ADVISORY" in prompt
    assert "holdout" in prompt.lower() or "held-out" in prompt.lower()
    # Points at the architect meta-trace files and memory.
    assert "_architect_trace" in prompt
    assert "memory.md" in prompt
    assert "iteration_history" in prompt


def test_prompt_embeds_codeevolver_context_and_repo_location():
    """The agent must be told to study CodeEvolver's own specs/source FIRST,
    and where the CodeEvolver repo lives (so it can actually read it)."""
    from src.engine.paths import engine_root

    prompt = mr.build_meta_reflection_prompt(
        method="ee",
        num_iterations=2,
        seed_test_score=None,
        best_candidate_test_score=None,
        workspace="/workspace/client",
        iteration_indices=[0, 1],
    )
    # Hard-coded /s-style instruction (verbatim core).
    assert (
        "Study specs/requirements.md and specs/engine3.md to get context"
        in prompt
    )
    # Repo root + the real meta-agent source the agent should ground proposals in.
    assert engine_root() in prompt
    assert "src/engine/optimizer/v3" in prompt
    assert "prompts.py" in prompt and "architect_hooks.py" in prompt


def test_prompt_names_only_skills_that_exist_in_engine_3():
    """The prompt points the meta-agent at real skills.

    Engine 3 deleted the `outcome_reflection` skill (folded into
    `close_iteration`) and stopped referencing `prompt_engineering`. Naming
    either sends the meta-agent hunting for files that are not there and
    invites proposals against surfaces that no longer exist.
    """
    from src.engine.paths import engine_root

    prompt = mr.build_meta_reflection_prompt(
        method="ee",
        num_iterations=2,
        seed_test_score=None,
        best_candidate_test_score=None,
        workspace="/workspace/client",
        iteration_indices=[0, 1],
    )
    assert "outcome_reflection" not in prompt
    assert "prompt_engineering" not in prompt
    assert "prompt-engineering" not in prompt
    assert "close_iteration" in prompt and "exploration" in prompt

    skills_dir = Path(engine_root()) / "src" / "engine" / "optimizer" / "v3" / "skills"
    if skills_dir.is_dir():
        assert not (skills_dir / "outcome_reflection").exists()
        assert (skills_dir / "close_iteration").is_dir()
        assert (skills_dir / "exploration").is_dir()


def test_prompt_graceful_when_specs_absent(tmp_path, monkeypatch):
    """When specs/ is not mounted the prompt tells the agent to skip specs."""
    # Point engine_root at a dir with no specs/ subtree.
    monkeypatch.setattr(mr, "engine_root", lambda: str(tmp_path))
    prompt = mr.build_meta_reflection_prompt(
        method="ee",
        num_iterations=1,
        seed_test_score=None,
        best_candidate_test_score=None,
        workspace="/w",
        iteration_indices=[0],
    )
    assert "NOT present in this environment" in prompt
    # Still points at the engine source.
    assert "src/engine/optimizer/v3" in prompt


def test_prompt_handles_missing_scores():
    prompt = mr.build_meta_reflection_prompt(
        method="ee",
        num_iterations=0,
        seed_test_score=None,
        best_candidate_test_score=None,
        workspace="/w",
        iteration_indices=[],
    )
    assert "N/A" in prompt  # None scores render as N/A, not a crash


def test_run_meta_reflection_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)

    captured = {}

    def fake_run_inspection_agent(**kwargs):
        captured["deny_paths"] = kwargs.get("deny_paths")
        captured["writable"] = kwargs.get("writable")
        return {"success": True, "proposed_change": "## Proposals\n- A: tweak prompt"}

    import src.engine.services.subprocess_service as svc

    monkeypatch.setattr(svc, "run_inspection_agent", fake_run_inspection_agent)

    out = mr.run_meta_reflection(
        workspace="/workspace",
        method="ee",
        num_iterations=3,
        seed_test_score=0.5,
        best_candidate_test_score=0.6,
        iteration_indices=[0, 1, 2],
        agent_model="claude-opus-4-7",
        coding_harness="claude",
        coding_harness_model=None,
    )
    assert out is not None and "Proposals" in out
    written = os.path.join(str(tmp_path), "meta_reflection.md")
    assert os.path.exists(written)
    with open(written) as f:
        assert "Proposals" in f.read()
    # Read-only; engine 3 runs post-finalization with NO deny paths, so the
    # agent may read architect traces (its primary input).
    assert captured["writable"] is False
    assert captured["deny_paths"] == []


def test_run_meta_reflection_non_fatal_on_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))

    def boom(**kwargs):
        raise RuntimeError("inspection agent exploded")

    import src.engine.services.subprocess_service as svc

    monkeypatch.setattr(svc, "run_inspection_agent", boom)

    out = mr.run_meta_reflection(
        workspace="/workspace",
        method="ee",
        num_iterations=1,
        seed_test_score=None,
        best_candidate_test_score=None,
        iteration_indices=[0],
        agent_model=None,
        coding_harness="claude",
        coding_harness_model=None,
    )
    assert out is None  # swallowed, never raised


def test_run_meta_reflection_returns_none_on_empty_output(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))

    import src.engine.services.subprocess_service as svc

    calls = []
    sleeps = []
    monkeypatch.setattr(mr, "_sleep", sleeps.append)
    monkeypatch.setattr(
        svc,
        "run_inspection_agent",
        lambda **k: (
            calls.append(k)
            or {"success": False, "proposed_change": "", "error": "no output"}
        ),
    )

    out = mr.run_meta_reflection(
        workspace="/workspace",
        method="ee",
        num_iterations=1,
        seed_test_score=None,
        best_candidate_test_score=None,
        iteration_indices=[0],
        agent_model=None,
        coding_harness="claude",
        coding_harness_model=None,
    )
    assert out is None
    assert not os.path.exists(os.path.join(str(tmp_path), "meta_reflection.md"))
    # No-output exits are retried with backoff, then given up on.
    assert len(calls) == mr.REFLECTION_AGENT_ATTEMPTS
    assert sleeps == list(mr.REFLECTION_AGENT_BACKOFF_SECONDS)[: mr.REFLECTION_AGENT_ATTEMPTS - 1]


def test_reflection_agent_retries_a_no_output_exit_then_succeeds(tmp_path, monkeypatch):
    """A known production failure shape: the harness exits non-zero with no
    output. One retry later it works; the second attempt's markdown is what
    gets written, and the wait is the first backoff step."""
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)

    import src.engine.services.subprocess_service as svc

    attempts = []
    sleeps = []
    monkeypatch.setattr(mr, "_sleep", sleeps.append)

    def flaky(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return {
                "success": False, "proposed_change": "",
                "error": "OpenCode exited with code 1: Failed query: PRAGMA wal_checkpoint(PASSIVE)",
            }
        return {"success": True, "proposed_change": "## second try"}

    monkeypatch.setattr(svc, "run_inspection_agent", flaky)

    out = mr.run_reflection_agent(
        workspace="/workspace", prompt="p", label="run-integrity (ee)",
        num_iterations=1, coding_harness="opencode",
    )
    assert out == "## second try"
    assert len(attempts) == 2
    assert sleeps == [mr.REFLECTION_AGENT_BACKOFF_SECONDS[0]]
    # Every attempt is the same read-only session.
    assert all(a["writable"] is False for a in attempts)
    with open(os.path.join(str(tmp_path), "meta_reflection.md")) as f:
        assert f.read() == "## second try"


def test_reflection_agent_accepts_nonzero_exit_with_output_without_retrying(
    tmp_path, monkeypatch
):
    """A non-zero exit that still produced markdown is accepted on the first
    attempt (the pre-existing behavior) -- the retry keys on empty output."""
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)

    import src.engine.services.subprocess_service as svc

    attempts = []
    monkeypatch.setattr(mr, "_sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    monkeypatch.setattr(
        svc, "run_inspection_agent",
        lambda **k: attempts.append(k) or {
            "success": False, "proposed_change": "partial report", "error": "exit 1",
        },
    )
    out = mr.run_reflection_agent(
        workspace="/workspace", prompt="p", label="ee", num_iterations=1,
    )
    assert out == "partial report"
    assert len(attempts) == 1


def test_shared_trace_guide_is_verbatim_in_both_prompts():
    """Pins the extraction in BOTH directions.

    The architect-trace reading guide is the one prompt fragment that is
    genuinely identical for engine 3 and the simple loop, and it is
    load-bearing JSONL schema documentation. If someone edits it for the simple
    loop's benefit, this fails on the engine 3 side; if someone re-inlines it
    in engine 3, it fails on the simple-loop side.
    """
    from src.engine.optimizer.meta.simple_meta_reflection import (
        build_simple_loop_meta_reflection_prompt,
    )

    guide = mr.architect_trace_reading_guide([1, 2])
    v3_prompt = mr.build_meta_reflection_prompt(
        method="ee",
        num_iterations=2,
        seed_test_score=0.4,
        best_candidate_test_score=0.6,
        workspace="/w",
        iteration_indices=[1, 2],
    )
    vanilla_prompt = build_simple_loop_meta_reflection_prompt(
        workspace="/w",
        history={1: {}, 2: {}},
        iteration_indices=[1, 2],
        branch="codeevolver-main",
        program_path="p.py",
        seed_test_score=0.4,
        final_test_score=0.6,
        subsample_size=10,
        max_eval_rows=100,
        total_evals=50,
    )
    assert guide in v3_prompt
    assert guide in vanilla_prompt
    # Pin the numbering and indent that engine 3's byte-identity turns on.
    assert f"2. {guide}" in v3_prompt


def test_trace_guide_names_rendered_files():
    """The auditors are steered README → digest → transcript → turn file."""
    from src.engine.optimizer.meta.meta_reflection import architect_trace_reading_guide

    guide = architect_trace_reading_guide([1, 2])
    for needle in ("README.md", "digest.md", "transcript.md", "→ full: turn_NNNN.json", "for N in: 1, 2"):
        assert needle in guide, needle
    assert guide.index("digest.md") < guide.index("transcript.md") < guide.index("turn_NNNN.json` only")
