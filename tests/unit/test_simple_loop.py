"""Copyright © 2026 440 Labs LLC

Unit tests for the simple loop (research Baseline 0) — offline, no agents.

Pins the properties that make the arm a clean control (`specs/v3_simple_loop.md`):
  - the agent surface is exactly `run_eval` with no skills / bootstrap,
  - the loop's ArchitectConfig carries the flags and timeouts the spec requires,
  - eval-row budget: the state object satisfies the shared `run_eval` gate,
  - sessions are auto-committed, constraint failures revert to the start sha,
  - the per-session record is derived from the in-process eval log,
  - stall halting, and finalization staying outside the budget.
"""

from __future__ import annotations

import json
import random
import subprocess
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from src.engine.agent.harnesses import AgentRole, HarnessRequest, HarnessResult
from src.engine.agent.harnesses.fake import FakeCodingHarness
from src.engine.agent.harnesses.opencode.architect_mcp import OpenCodeArchitectToolBridge
from src.engine.agent.harnesses.opencode.harness import OpenCodeHarness
from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.sandbox_client import EvalResult
from src.engine.optimizer.simple_loop import constraint_check, loop, session
from src.engine.optimizer.simple_loop.prompt import build_simple_loop_prompt
from src.engine.optimizer.simple_loop.records import session_record
from src.engine.optimizer.simple_loop.state import SimpleLoopState
from src.engine.optimizer.v3.coding_harness_config import CodingHarnessConfig
from src.engine.optimizer.v3.iteration_architect import ArchitectConfig
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.schemas.iteration_history import ITERATION_RECORD_FIELDS
from src.engine.optimizer.v3.optimization_config import OptimizationConfig
from src.engine.optimizer.v3.run_eval_tool import RunEvalBinding, RunEvalError


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "program.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "seed")
    return tmp_path


def _evaluator() -> EvaluatorConfig:
    return EvaluatorConfig(
        program_factory="p:P", metric_fns=["m:m"], metric_names=["accuracy"],
        primary_metric="accuracy", trainset_path="train.json",
    )


def _cfg(**kw) -> ArchitectConfig:
    base = dict(
        run_timestamp="t", additional_instructions=None, program_path="p:P",
        evaluator=_evaluator(), trainset_ids=list(range(20)), comparison_ids=[0, 1, 2],
        max_turns=10, agent_managed_git=False, stage_project_skills=False,
        system_prompt_builder=lambda **k: "system prompt",
    )
    base.update(kw)
    return ArchitectConfig(**base)


def _ctx(**kw) -> IterationContext:
    base = dict(iteration_idx=1, parent_locked=True, max_eval_rows=100, evals_used=0,
                subsample_size=5)
    base.update(kw)
    return IterationContext(**base)


# --- agent surface ------------------------------------------------------------


def test_tool_surface_is_run_eval_only():
    bridge = OpenCodeArchitectToolBridge({"run_eval": lambda **k: {}}, agent_managed_git=False)
    assert [t["name"] for t in bridge.list_tools()] == ["run_eval"]


def test_session_spec_has_no_skills_or_bootstrap():
    harness = OpenCodeHarness(architect_config=_cfg())
    spec = harness.build_session_spec(HarnessRequest(
        role=AgentRole.ARCHITECT, workspace="/tmp", prompt="hi", max_turns=5,
        raw_provider_metadata={"system_prompt": "sp", "commit_changes": True},
    ))
    assert spec.bootstrap_commands == [] and spec.finalization_commands == []
    assert not spec.prompt.startswith("/s")


def test_run_session_registers_only_run_eval_and_records_usage(monkeypatch):
    fake = FakeCodingHarness(architect_results=[HarnessResult.successful(
        output="done", session_id="s1",
        raw_provider_metadata={"model": "openai/gpt-5.5", "usage": {"input_tokens": 3, "output_tokens": 4}},
    )])
    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(session, "create_coding_harness", fake_factory)
    ctx = _ctx()
    session.run_session(ctx=ctx, state=SimpleLoopState(), workspace="/w",
                        config=_cfg(), rng=random.Random(0))
    assert list(captured["architect_tool_registry"]) == ["run_eval"]
    assert isinstance(captured["architect_tool_registry"]["run_eval"], RunEvalBinding)
    assert captured["architect_context"] is ctx
    assert fake.calls[0].role is AgentRole.ARCHITECT
    assert "session 1" in fake.calls[0].prompt
    assert ctx.token_usage_by_model


# --- prompt -------------------------------------------------------------------


def test_prompt_has_no_stale_surfaces():
    prompt = build_simple_loop_prompt(
        additional_instructions="#### Constraints:\n- no web search",
        program_path="pkg.prog:P", workspace_path="/w", context=_ctx(max_eval_rows=200),
    )
    assert "coding_agent" not in prompt
    assert "block edits" not in prompt
    assert "iteration_<N>_parent/" in prompt
    assert "## Additional instructions" in prompt and "no web search" in prompt
    assert "pkg.prog:P" in prompt


# --- budget -------------------------------------------------------------------


def _eval_result(row_ids):
    return EvalResult(scores_by_metric={"accuracy": 0.5}, primary_score=0.5,
                      trace_meta=[{"row_id": r} for r in row_ids], trace_dir="t",
                      trace_index_file="ti", row_ids=list(row_ids), raw={})


def _stubbed_binding(tmp_path: Path, used: int, max_eval_rows: int):
    (tmp_path / "train.json").write_text(json.dumps([{"id": i} for i in range(20)]))
    state = SimpleLoopState()
    state.increment_evals(used)
    binding = session.build_run_eval(
        ctx=_ctx(max_eval_rows=max_eval_rows, evals_used=used), state=state,
        workspace=str(tmp_path), config=_cfg(subsample_size=5), rng=random.Random(0),
    )
    stub = patch.multiple(
        "src.engine.optimizer.v3.run_eval_tool",
        _run_eval=lambda **kw: _eval_result(kw.get("row_ids") or []),
        git_get_sha=lambda _w: "sha", git_tree_signature=lambda _w: "tree",
        git_get_current_branch=lambda _w: "codeevolver-t-main", git_dirty_paths=lambda _w: [],
    )
    return binding, state, stub


def test_state_satisfies_the_shared_budget_gate(tmp_path):
    binding, state, stub = _stubbed_binding(tmp_path, used=100, max_eval_rows=100)
    with stub:
        with pytest.raises(RunEvalError):
            binding("train")
        binding("comparison")
        assert state.final_comparison_evals == 1
        with pytest.raises(RunEvalError):
            binding("comparison")
    assert state.total_num_evals == 103


# --- git stages ---------------------------------------------------------------


def test_commit_session_end_commits_dirty_files(tmp_path):
    repo = _repo(tmp_path)
    start = _git(repo, "rev-parse", "HEAD")
    (repo / "program.py").write_text("x = 2\n")
    end = loop.commit_session_end(str(repo), 1)
    assert end != start and _git(repo, "status", "--porcelain") == ""
    assert loop.commit_session_end(str(repo), 2) == end


def test_constraint_failure_reverts_to_start_sha_and_leaves_a_marker(tmp_path, monkeypatch):
    """The violating content is reset away; an empty `Revert iteration N`
    commit sits on top so the revert is visible in `git log`. The message
    quotes the constraint, never the judge's reason."""
    repo = _repo(tmp_path)
    start = _git(repo, "rev-parse", "HEAD")
    (repo / "program.py").write_text("answers = {...}\n")
    end = loop.commit_session_end(str(repo), 1)
    verdict = types.SimpleNamespace(
        status="fail", violated_constraint="no hard-coding", reason="judge rationale"
    )
    monkeypatch.setattr(constraint_check, "evaluate_change_constraints", lambda **k: verdict)
    reason = constraint_check.check_session_constraints(
        workspace=str(repo), start_sha=start, end_sha=end,
        additional_instructions="c", lm_callable=lambda p: "", iteration_idx=1,
    )
    assert reason == "no hard-coding"
    head = _git(repo, "rev-parse", "HEAD")
    assert head not in (start, end)
    assert _git(repo, "rev-parse", "HEAD~1") == start
    assert _git(repo, "rev-parse", "HEAD^{tree}") == _git(repo, "rev-parse", f"{start}^{{tree}}")
    assert (repo / "program.py").read_text() == "x = 1\n"
    body = _git(repo, "log", "-1", "--format=%B")
    assert "Revert iteration 1" in body and "Violated constraint: no hard-coding" in body
    assert "judge rationale" not in body


def test_constraint_pass_keeps_head(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    start = _git(repo, "rev-parse", "HEAD")
    (repo / "program.py").write_text("x = 3\n")
    end = loop.commit_session_end(str(repo), 1)
    monkeypatch.setattr(constraint_check, "evaluate_change_constraints", lambda **k: None)
    assert constraint_check.check_session_constraints(
        workspace=str(repo), start_sha=start, end_sha=end,
        additional_instructions="c", lm_callable=lambda p: "", iteration_idx=1,
    ) is None
    assert _git(repo, "rev-parse", "HEAD") == end


def test_gate_disabled_without_instructions_warns(caplog):
    with caplog.at_level("WARNING"):
        assert constraint_check.build_constraint_checker(
            additional_instructions=None, reflection_lm="anthropic/x") is None
    assert "DISABLED" in caplog.text


# --- record + stall -----------------------------------------------------------


def test_session_record_derives_from_eval_log():
    ctx = _ctx()
    ctx.initial_subsample_score = 0.4
    ctx.baseline_sample_key = "5:locked"
    ctx.record_eval("a", 0.4, eval_sample="train", n_rows=5, sample_key="5:locked")
    ctx.record_eval("b", 0.7, eval_sample="train", n_rows=5, sample_key="5:locked")
    ctx.record_eval("b", 0.55, eval_sample="comparison", n_rows=3)
    ctx.record_eval("c", 0.6, eval_sample="train", n_rows=5, sample_key="5:locked")
    rec = session_record(ctx, branch="br", start_sha="a", end_sha="c", rows_used=18,
                         constraint_violation=None, skipped=False)
    assert rec["best_subsample_score"] == 0.7
    assert rec["comparison_best_score"] == 0.55
    assert rec["score"] == 0.7 and rec["initial_subsample_score"] == 0.4
    assert rec["changes_landed"] is True and rec["eval_rows_used"] == 18
    # No sha is attributed: the simple loop never reconciles dirty-tree evals,
    # so an eval's sha is HEAD-at-eval-time (the parent for a dirty tree).
    assert "best_subsample_sha" not in rec and "comparison_best_sha" not in rec
    assert set(rec) <= set(ITERATION_RECORD_FIELDS)


def test_session_record_best_train_ignores_single_and_other_samples():
    """A 1-row `single` eval must not become the session score. Mirror
    engine 3: only train evals on the baseline's locked rows compete."""
    ctx = _ctx()
    ctx.initial_subsample_score = 0.8755
    ctx.baseline_sample_key = "50:locked"
    ctx.record_eval("p", 0.8755, eval_sample="train", n_rows=50, sample_key="50:locked")
    ctx.record_eval("p", 0.974, eval_sample="single", n_rows=1, sample_key="1:one")
    ctx.record_eval("p", 0.99, eval_sample="train", n_rows=50, sample_key="50:redrawn")
    ctx.record_eval("p", 0.8751, eval_sample="train", n_rows=50, sample_key="50:locked")
    ctx.record_eval("q", 0.836, eval_sample="train", n_rows=50, sample_key="50:locked")
    rec = session_record(ctx, branch="br", start_sha="p", end_sha="q", rows_used=201,
                         constraint_violation=None, skipped=False)
    assert rec["score"] == rec["best_subsample_score"] == 0.8755
    assert rec["comparison_best_score"] is None


def test_session_record_best_train_fails_open_without_a_locked_key():
    """A rejected (ceiling) baseline leaves `baseline_sample_key` None — every
    non-single train eval competes then, exactly as in engine 3."""
    ctx = _ctx()
    ctx.record_eval("p", 0.5, eval_sample="train", n_rows=5, sample_key="5:a")
    ctx.record_eval("p", 0.9, eval_sample="single", n_rows=1, sample_key="1:x")
    ctx.record_eval("p", 0.7, eval_sample="train", n_rows=5, sample_key="5:b")
    ctx.record_eval("p", 0.6, eval_sample="train", n_rows=5)
    rec = session_record(ctx, branch="br", start_sha="p", end_sha="p", rows_used=16,
                         constraint_violation=None, skipped=False)
    assert rec["score"] == 0.7


def test_session_record_nulls_headline_score_when_constraint_reverted():
    """A tree the constraint gate reverted must not become the run's "best
    train". The headline `score` is None for a reverted session; the measured
    readings stay beside `constraint_violation` for analysis."""
    ctx = _ctx()
    ctx.initial_subsample_score = 0.9256
    ctx.baseline_sample_key = "50:locked"
    ctx.record_eval("p", 0.9256, eval_sample="train", n_rows=50, sample_key="50:locked")
    ctx.record_eval("p", 0.9652, eval_sample="train", n_rows=50, sample_key="50:locked")
    ctx.record_eval("p", 0.8811, eval_sample="comparison", n_rows=200)
    rec = session_record(ctx, branch="br", start_sha="p", end_sha="p", rows_used=300,
                         constraint_violation="hard-coded three titles", skipped=False)
    assert rec["score"] is None
    assert rec["best_subsample_score"] == 0.9652
    assert rec["comparison_best_score"] == 0.8811
    assert rec["changes_landed"] is False
    assert rec["constraint_violation"] == "hard-coded three titles"


def test_session_record_reverted_session_lands_nothing_even_though_head_moved():
    """After the gate, `close_sha` is the revert marker — a new sha with the
    parent's content — so sha inequality alone must not read as 'landed'."""
    ctx = _ctx()
    ctx.record_eval("p", 0.9, eval_sample="train", n_rows=5)
    rec = session_record(ctx, branch="br", start_sha="p", end_sha="revert-marker", rows_used=5,
                         constraint_violation="no hard-coding", skipped=False)
    assert rec["close_sha"] == "revert-marker"
    assert rec["changes_landed"] is False and rec["score"] is None


def test_kickoff_message_carries_revert_note_only_after_a_gate_revert():
    from src.engine.optimizer.simple_loop.prompt import build_kickoff_message
    from src.engine.optimizer.v3.constraint_gate import PREVIOUS_ITERATION_REVERTED_NOTE

    plain = build_kickoff_message(_ctx(iteration_idx=4))
    assert plain == "You are on session 4. Improve the program on the current branch."
    noted = build_kickoff_message(_ctx(iteration_idx=4, previous_iteration_constraint_reverted=True))
    assert noted.startswith(plain) and PREVIOUS_ITERATION_REVERTED_NOTE in noted
    assert "git log" not in noted


def test_eval_rows_used_is_whitelisted():
    from src.db.db_schemas.db_schemas import IterationHistoryRecord

    assert "eval_rows_used" in ITERATION_RECORD_FIELDS
    assert "eval_rows_used" in IterationHistoryRecord.model_fields


def test_check_stalled_threshold():
    loop.check_stalled(4, 4)
    with pytest.raises(loop.SimpleLoopStalledError):
        loop.check_stalled(5, 5)


# --- finalization -------------------------------------------------------------


def test_finalization_stays_outside_the_budget(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    seed = _git(repo, "rev-parse", "HEAD")
    (repo / "program.py").write_text("x = 9\n")
    final = loop.commit_session_end(str(repo), 1)
    _git(repo, "checkout", "-q", "-b", "codeevolver-t-main")
    calls = []

    def fake_run_eval(**kw):
        calls.append((kw["suffix"], kw["iteration_idx"]))
        return _eval_result(kw["row_ids"])

    monkeypatch.setattr(loop, "run_eval", fake_run_eval)
    monkeypatch.setattr(loop, "clean_python_bytecode", lambda _w: None)
    datasets = types.SimpleNamespace(has_testset=True, testset_ids=[1, 2], comparison_ids=[3])
    scores = loop.finalize_on_holdout(
        workspace=str(repo), evaluator_config=_evaluator(), datasets=datasets,
        branch="codeevolver-t-main", seed_sha=seed, final_sha=final, timeout=10,
    )
    assert calls == [("holdout", 0), ("final_comparison", 0), ("holdout", 1), ("final_comparison", 1)]
    assert scores["seed_test_score"] == 0.5 and scores["final_train_score"] == 0.5
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "codeevolver-t-main"
    assert scores["finalization_error"] is None


def test_finalization_failure_sets_error_and_restores_branch(tmp_path, monkeypatch):
    """Mirror of the v3 finalization rule: a holdout scoring failure surfaces as
    `finalization_error` in the scores dict (which is splatted into
    `set_completed`) instead of vanishing into a warning log."""
    repo = _repo(tmp_path)
    seed = _git(repo, "rev-parse", "HEAD")
    (repo / "program.py").write_text("x = 9\n")
    final = loop.commit_session_end(str(repo), 1)
    _git(repo, "checkout", "-q", "-b", "codeevolver-t-main")

    def fake_run_eval(**kw):
        if kw["suffix"] == "final_comparison":
            raise RuntimeError("provider outage")
        return _eval_result(kw["row_ids"])

    monkeypatch.setattr(loop, "run_eval", fake_run_eval)
    monkeypatch.setattr(loop, "clean_python_bytecode", lambda _w: None)
    datasets = types.SimpleNamespace(has_testset=True, testset_ids=[1, 2], comparison_ids=[3])
    scores = loop.finalize_on_holdout(
        workspace=str(repo), evaluator_config=_evaluator(), datasets=datasets,
        branch="codeevolver-t-main", seed_sha=seed, final_sha=final, timeout=10,
    )
    assert scores["finalization_error"] is not None
    assert "provider outage" in scores["finalization_error"]
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "codeevolver-t-main"


# --- the loop, end to end with a fake session ----------------------------------


def test_run_simple_loop_wires_config_and_records(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    seed = _git(repo, "rev-parse", "HEAD")
    captured: dict = {}

    def fake_session(*, ctx, state, workspace, config, rng):
        captured["config"] = config
        state.increment_evals(10)
        ctx.initial_subsample_score = 0.4
        ctx.record_eval(seed, 0.4, eval_sample="train", n_rows=5)
        ctx.record_eval(seed, 0.8, eval_sample="train", n_rows=5)
        (Path(workspace) / "program.py").write_text("x = 42\n")
        return HarnessResult.successful(output="ok")

    gate_calls = []

    def fake_gate(**kw):
        gate_calls.append(kw)
        return None

    monkeypatch.setattr(loop, "initialize_repository", lambda **k: ("codeevolver-t-main", seed))
    monkeypatch.setattr(loop, "run_session", fake_session)
    monkeypatch.setattr(loop, "check_session_constraints", fake_gate)
    monkeypatch.setattr(loop, "finalize_on_holdout", lambda **k: {
        "seed_test_score": 0.3, "final_test_score": 0.6,
        "seed_train_score": None, "final_train_score": None})
    monkeypatch.setattr(loop, "_run_auditors", lambda **k: captured.setdefault("audited", True))
    progress = []
    completed = {}
    callbacks = types.SimpleNamespace(
        check_cancelled=lambda: False,
        update_progress=lambda **k: progress.append(k),
        set_completed=lambda **k: completed.update(k),
    )
    config = OptimizationConfig(
        program="p:P", metric="m:m", workspace=str(repo), trainset_path="train.json",
        max_metric_calls=10, subsample_size=5, additional_instructions="#### Constraints:\n- x",
        enable_meta_reflection=False, enable_run_integrity_agent=True,
        additional_deny_paths=["pairwise_lfrqa_examples.json"],
    )
    datasets = types.SimpleNamespace(
        trainset_ids=list(range(20)), comparison_ids=[0, 1], has_testset=False, testset_ids=[])
    with patch.object(constraint_check, "build_constraint_lm_callable", lambda lm: (lambda p: "")):
        result = loop.run_simple_loop(
            config=config, workspace=str(repo), datasets=datasets, evaluator_config=_evaluator(),
            callbacks=callbacks, coding_harness_config=CodingHarnessConfig("opencode", "openai/gpt-5.5"),
            agent_model=None,
        )

    cfg = captured["config"]
    assert cfg.agent_managed_git is False and cfg.stage_project_skills is False
    assert cfg.capture_architect_traces is True
    assert cfg.subsample_timeout == config.subsample_eval_timeout
    assert cfg.comparison_timeout == config.comparison_eval_timeout
    # The client's declared deny paths reach the session. simple_loop's
    # architect session IS the coding agent, so this list is the only source of
    # client-declared fences, and `additional_deny_paths` must be enforced here
    # like on every other arm.
    assert cfg.additional_deny_paths == ["pairwise_lfrqa_examples.json"]
    assert cfg.comparison_ids == [0, 1]

    rec = result["history"][1]
    assert rec["parent_sha"] == seed and rec["close_sha"] == result["final_sha"] != seed
    assert rec["eval_rows_used"] == 10 and rec["score"] == 0.8 and rec["changes_landed"]
    assert rec["constraint_violation"] is None and rec["skipped"] is False
    assert gate_calls[0]["start_sha"] == seed and gate_calls[0]["end_sha"] == result["final_sha"]
    assert result["history"][0]["test_score"] == 0.3 and rec["test_score"] == 0.6
    assert result["method"] == "simple_loop" and result["total_metric_calls"] == 10
    assert completed["total_metric_calls"] == 10 and completed["final_test_score"] == 0.6
    assert progress[0]["current_iteration"] == 1 and progress[0]["total_metric_calls"] == 10
    assert captured["audited"] is True
    assert _git(repo, "status", "--porcelain") == ""


def test_run_simple_loop_reverted_session_is_visible_to_the_next_one(tmp_path, monkeypatch):
    """Real gate, judge forced to fail once: session 1 is reset with a marker
    commit, its record closes on the marker (= session 2's parent), and
    session 2's context carries the revert flag that the kickoff renders."""
    repo = _repo(tmp_path)
    seed = _git(repo, "rev-parse", "HEAD")
    seen: list = []

    def fake_session(*, ctx, state, workspace, config, rng):
        seen.append(ctx)
        state.increment_evals(10)
        ctx.record_eval(seed, 0.4, eval_sample="train", n_rows=5)
        (Path(workspace) / "program.py").write_text(f"x = {len(seen)}00\n")
        return HarnessResult.successful(output="ok")

    verdicts = iter([
        types.SimpleNamespace(status="fail", violated_constraint="- x", reason="secret"),
        None,
    ])
    monkeypatch.setattr(constraint_check, "evaluate_change_constraints", lambda **k: next(verdicts))
    monkeypatch.setattr(loop, "initialize_repository", lambda **k: ("codeevolver-t-main", seed))
    monkeypatch.setattr(loop, "run_session", fake_session)
    monkeypatch.setattr(loop, "finalize_on_holdout", lambda **k: {
        "seed_test_score": None, "final_test_score": None,
        "seed_train_score": None, "final_train_score": None})
    monkeypatch.setattr(loop, "_run_auditors", lambda **k: None)
    callbacks = types.SimpleNamespace(
        check_cancelled=lambda: False, update_progress=lambda **k: None, set_completed=lambda **k: None)
    config = OptimizationConfig(
        program="p:P", metric="m:m", workspace=str(repo), trainset_path="train.json",
        max_metric_calls=20, subsample_size=5, additional_instructions="#### Constraints:\n- x",
        enable_meta_reflection=False, enable_run_integrity_agent=False,
    )
    datasets = types.SimpleNamespace(trainset_ids=list(range(20)), comparison_ids=[], has_testset=False, testset_ids=[])
    with patch.object(constraint_check, "build_constraint_lm_callable", lambda lm: (lambda p: "")):
        result = loop.run_simple_loop(
            config=config, workspace=str(repo), datasets=datasets, evaluator_config=_evaluator(),
            callbacks=callbacks, coding_harness_config=CodingHarnessConfig("opencode", None),
            agent_model=None,
        )

    one, two = result["history"][1], result["history"][2]
    assert one["constraint_violation"] == "- x" and one["changes_landed"] is False
    assert one["close_sha"] != seed and _git(repo, "rev-parse", f"{one['close_sha']}~1") == seed
    assert _git(repo, "log", "-1", "--format=%s", one["close_sha"]).startswith("Revert iteration 1")
    assert "secret" not in _git(repo, "log", "-1", "--format=%B", one["close_sha"])
    assert two["parent_sha"] == one["close_sha"] and two["changes_landed"] is True
    assert seen[0].previous_iteration_constraint_reverted is False
    assert seen[1].previous_iteration_constraint_reverted is True


def test_run_simple_loop_halts_when_stalled(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    seed = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(loop, "initialize_repository", lambda **k: ("codeevolver-t-main", seed))
    monkeypatch.setattr(loop, "run_session", lambda **k: HarnessResult.successful(output="noop"))
    monkeypatch.setattr(loop, "check_session_constraints", lambda **k: None)
    callbacks = types.SimpleNamespace(
        check_cancelled=lambda: False, update_progress=lambda **k: None, set_completed=lambda **k: None)
    config = OptimizationConfig(program="p:P", metric="m:m", workspace=str(repo),
                                trainset_path="train.json", max_metric_calls=10)
    datasets = types.SimpleNamespace(trainset_ids=[0], comparison_ids=[], has_testset=False, testset_ids=[])
    with pytest.raises(loop.SimpleLoopStalledError):
        loop.run_simple_loop(
            config=config, workspace=str(repo), datasets=datasets, evaluator_config=_evaluator(),
            callbacks=callbacks, coding_harness_config=CodingHarnessConfig("opencode", None),
            agent_model=None,
        )


# --- comparison history + git identity ----------------------------------------


def test_state_records_comparison_eval_to_history_file(tmp_path):
    """`run_eval` looks up `record_comparison_eval` by duck typing and
    its messaging tells the agent the eval was appended to
    comparison_set_history.jsonl — the simple loop's state must actually
    write it. Minimal contract: post the file, nothing else."""
    path = tmp_path / "history" / "comparison_set_history.jsonl"
    state = SimpleLoopState(comparison_history_path=str(path))
    state.record_comparison_eval(
        iteration=2, branch="codeevolver-t-main", sha="abc123", score=0.75,
        n_rows=20, scores_by_metric={"accuracy": 0.75},
        row_scores={"0": 1.0, "1": 0.5}, eval_id="ev1", tree="tree1",
    )
    state.record_comparison_eval(
        iteration=3, branch="codeevolver-t-main", sha="def456", score=0.8, n_rows=20,
    )
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["iteration"] == 2 and lines[0]["sha"] == "abc123"
    assert lines[0]["score"] == 0.75 and lines[0]["n_rows"] == 20
    assert lines[0]["eval_id"] == "ev1" and lines[0]["tree"] == "tree1"
    assert lines[1]["branch"] == "codeevolver-t-main"
    # The duck-typed lookup in run_eval_tool finds it.
    assert callable(getattr(state, "record_comparison_eval", None))


def test_state_without_history_path_is_a_silent_noop(tmp_path):
    state = SimpleLoopState()
    state.record_comparison_eval(
        iteration=1, branch="b", sha="s", score=0.5, n_rows=3,
    )  # must not raise, must not write anywhere


def test_git_set_identity_lets_a_bare_commit_succeed(tmp_path):
    """Without an identity, the agent's own `git commit` dies with "Author
    identity unknown" on every fresh container."""
    from src.engine.services.subprocess_service import git_set_identity

    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "f.txt").write_text("v1")
    git_set_identity(str(tmp_path))
    assert _git(tmp_path, "config", "user.email") == "ce@codeevolver.ai"
    assert _git(tmp_path, "config", "user.name") == "CodeEvolver"
    # An agent-style bare commit (no -c identity, no --global fallback).
    _git(tmp_path, "add", "-A")
    out = subprocess.run(
        ["git", "commit", "-m", "agent commit"],
        cwd=tmp_path, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "nohome"),
             "GIT_CONFIG_NOSYSTEM": "1"},
    )
    assert out.returncode == 0, out.stderr
    assert "CodeEvolver" in _git(tmp_path, "log", "-1", "--format=%an")
