"""Copyright © 2026 440 Labs LLC

Auditor tools, the close schema, and eval provenance:

- the post-run auditors (meta-reflection, run-integrity) get a Bash tool;
- `close_iteration`'s `register_branch` is a FLAT object, not nested under a
  codebase folder key;
- `run_eval` mints an eval id and logs one line per eval to
  `{traces_dir}/iteration_{N}_evals.jsonl`;
- the seed step excludes engine-owned in-workspace artifacts from git so they
  can never count as program content.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


# --- auditors get Bash -------------------------------------------------------


def _capture_inspection(monkeypatch, tmp_path):
    from src.engine.services import subprocess_service as svc

    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"success": True, "proposed_change": "## report\n", "error": None}

    monkeypatch.setattr(svc, "run_inspection_agent", fake)
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path / "memory"))
    return captured


def test_meta_reflection_runs_with_bash(monkeypatch, tmp_path):
    from src.engine.optimizer.meta import meta_reflection as mr

    captured = _capture_inspection(monkeypatch, tmp_path)
    mr.run_meta_reflection(
        workspace=str(tmp_path), method="ee", num_iterations=2,
        seed_test_score=0.1, best_candidate_test_score=0.2, iteration_indices=[1, 2],
    )
    assert captured["allow_bash"] is True
    assert captured["writable"] is False
    assert captured["run_as_agent_user"] is True


def test_run_integrity_runs_with_bash_and_privilege(monkeypatch, tmp_path):
    from src.engine.optimizer.meta import run_integrity as ri

    captured = _capture_inspection(monkeypatch, tmp_path)
    ri.run_integrity_inspection(
        workspace=str(tmp_path), method="ee", num_iterations=2,
        best_valset_score=None, seed_test_score=0.1, best_candidate_test_score=0.2,
        iteration_indices=[1, 2], testset_path=None,
    )
    assert captured["allow_bash"] is True
    assert captured["run_as_agent_user"] is False
    assert captured["writable"] is False


def test_reflection_agent_defaults_to_no_bash(monkeypatch, tmp_path):
    from src.engine.optimizer.meta import meta_reflection as mr

    captured = _capture_inspection(monkeypatch, tmp_path)
    mr.run_reflection_agent(
        workspace=str(tmp_path), prompt="p", label="x", num_iterations=1,
    )
    assert captured["allow_bash"] is False


def test_auditor_prompts_mention_bash():
    from src.engine.optimizer.meta.meta_reflection import architect_trace_reading_guide
    from src.engine.optimizer.meta.run_integrity import build_run_integrity_prompt

    assert "Bash tool for read-only inspection" in architect_trace_reading_guide([1])
    safe = build_run_integrity_prompt(
        method="ee", num_iterations=1, best_valset_score=None, seed_test_score=None,
        best_candidate_test_score=None, workspace="/ws", iteration_indices=[1],
        testset_path=None, safe_mode=True,
    )
    assert "Bash for read-only" in safe


# --- close_iteration payload shape -------------------------------------------


def test_skill_staging_renders_the_flat_register_branch(tmp_path):
    from src.engine.agent.harnesses.skills import stage_codeevolver_skills_to

    ws = tmp_path / "LangProBe-hover"
    ws.mkdir()
    dest = stage_codeevolver_skills_to(
        tmp_path / "skills", workspace=ws, skills=["close_iteration"]
    )
    text = (dest / "close_iteration" / "SKILL.md").read_text()
    assert "codebase folder" not in text
    assert "{codebase_folder}" not in text
    assert "insert repo name" not in text
    assert '"branch": "insert branch name"' in text
    assert "Do not rewrite `" + str(ws) + "/codeevolver.md` yourself" in text


def test_close_iteration_description_and_schema_are_flat():
    from src.engine.agent.harnesses.architect_tools import (
        architect_tool_spec,
        mcp_architect_tool_definitions,
    )

    spec = architect_tool_spec("close_iteration")
    assert "codebase folder" not in spec.description
    assert '{"branch": "codeevolver-..."' in spec.description
    assert "<repo>" not in spec.description
    # The entry's own fields are declared, not hidden behind a wildcard
    # additionalProperties map keyed by repo name.
    entry = spec.input_schema["properties"]["register_branch"]
    assert set(entry["properties"]) == {
        "branch", "was_primary", "is_primary", "promotion_evidence"
    }
    assert entry["required"] == ["branch", "was_primary", "is_primary"]
    assert entry["additionalProperties"] is False
    # Loop-branch mode has no register_branch at all.
    loop = architect_tool_spec("close_iteration", close_iteration_mode="loop_branch")
    assert "register_branch" not in loop.input_schema["properties"]

    defs = mcp_architect_tool_definitions()
    close = next(d for d in defs if d["name"] == "close_iteration")
    assert "codebase folder" not in close["description"]


def test_system_prompt_does_not_name_a_codebase_folder():
    from src.engine.optimizer.v3.iteration_context import IterationContext
    from src.engine.optimizer.v3.prompts import build_architect_system_prompt

    prompt = build_architect_system_prompt(
        additional_instructions=None, program_path=None,
        workspace_path="/home/x/ce-runs/ee-hover-s1/workspace",
        context=IterationContext(iteration_idx=1),
    )
    assert "Codebase folder name" not in prompt
    assert "/home/x/ce-runs/ee-hover-s1/workspace" in prompt


# --- eval ids + evals.jsonl --------------------------------------------------


def test_sample_key_is_order_independent():
    from src.engine.optimizer.v3.run_eval_tool import sample_key_for_rows

    assert sample_key_for_rows([3, 1, 2]) == sample_key_for_rows([1, 2, 3])
    assert sample_key_for_rows([1, 2]) != sample_key_for_rows([1, 2, 3])
    assert sample_key_for_rows([1, 2, 3]).startswith("3:")


def test_run_eval_logs_one_line_per_eval_with_an_id(monkeypatch, tmp_path):
    from src.engine.evaluator.sandbox_client import EvalResult
    from src.engine.optimizer.v3 import run_eval_tool as ret
    from src.engine.optimizer.v3.iteration_context import IterationContext

    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.setattr(ret, "git_get_sha", lambda ws, ref="HEAD": "headsha")
    monkeypatch.setattr(ret, "git_tree_signature", lambda ws, ref=None: "tree1")
    monkeypatch.setattr(ret, "git_get_current_branch", lambda ws: "codeevolver-x")
    monkeypatch.setattr(ret, "git_dirty_paths", lambda ws: [])

    def fake_eval(*, row_ids, **kwargs):
        return EvalResult(
            scores_by_metric={"acc": 0.5}, primary_score=0.5,
            trace_meta=[{"example_id": r, "score": 0.5} for r in row_ids],
            trace_dir="/t", trace_index_file="/ti", row_ids=list(row_ids), raw={},
        )

    monkeypatch.setattr(ret, "_run_eval", fake_eval)

    ctx = IterationContext(iteration_idx=7, parent_locked=True)
    binding = ret.RunEvalBinding(
        ctx=ctx, workspace=str(tmp_path), config=None,  # type: ignore[arg-type]
        trainset_ids=list(range(10)), comparison_ids=[0, 1], subsample_size=3,
    )
    first = binding("train")
    second = binding("comparison")

    assert first["eval_id"] and second["eval_id"] and first["eval_id"] != second["eval_id"]
    # `/t` is not a real directory here, so the provenance stamp is skipped
    # without failing the eval (best-effort by contract).
    assert not (tmp_path / "t").exists()
    assert ctx.eval_records[0]["eval_id"] == first["eval_id"]
    assert ctx.eval_records[0]["sample_key"] == ctx.baseline_sample_key
    assert ctx.eval_records[0]["committed"] is True  # clean tree: HEAD holds it
    assert "sample_key" not in ctx.eval_records[1]

    log = tmp_path / "traces" / "iteration_7_evals.jsonl"
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["eval_sample"] for l in lines] == ["train", "comparison"]
    assert lines[0]["eval_id"] == first["eval_id"]
    assert lines[0]["sha_at_eval"] == "headsha" and lines[0]["tree"] == "tree1"
    assert lines[0]["n_rows"] == 3 and lines[1]["n_rows"] == 2
    assert lines[1]["by_loop"] is False


def test_run_eval_stamps_provenance_into_the_trace_dir(monkeypatch, tmp_path):
    """Trace directories are rewritten by every eval with the same suffix, so
    without provenance an agent can analyse traces of a program that was
    already reverted. `provenance.json` says which eval (id / sha / tree)
    produced the directory."""
    from src.engine.evaluator.sandbox_client import EvalResult
    from src.engine.optimizer.v3 import run_eval_tool as ret
    from src.engine.optimizer.v3.iteration_context import IterationContext
    from src.engine.telemetry.trace_layout import read_provenance

    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.setattr(ret, "git_get_sha", lambda ws, ref="HEAD": "headsha")
    monkeypatch.setattr(ret, "git_tree_signature", lambda ws, ref=None: "tree1")
    monkeypatch.setattr(ret, "git_get_current_branch", lambda ws: "codeevolver-x")
    monkeypatch.setattr(ret, "git_dirty_paths", lambda ws: [])

    def fake_eval(*, row_ids, suffix, iteration_idx, **kwargs):
        d = tmp_path / "traces" / f"iteration_{iteration_idx}_{suffix}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.jsonl").write_text("")
        return EvalResult(
            scores_by_metric={"acc": 0.5}, primary_score=0.5,
            trace_meta=[{"example_id": r, "score": 0.5} for r in row_ids],
            trace_dir=str(d), trace_index_file=str(d / "index.jsonl"),
            row_ids=list(row_ids), raw={},
        )

    monkeypatch.setattr(ret, "_run_eval", fake_eval)
    ctx = IterationContext(iteration_idx=7, parent_locked=True)
    binding = ret.RunEvalBinding(
        ctx=ctx, workspace=str(tmp_path), config=None,  # type: ignore[arg-type]
        trainset_ids=list(range(10)), comparison_ids=[0, 1], subsample_size=3,
    )
    first = binding("train")
    stamp = read_provenance(str(tmp_path / "traces" / "iteration_7_parent"))
    assert stamp is not None
    assert stamp["eval_id"] == first["eval_id"] == ctx.eval_records[0]["eval_id"]
    assert stamp["sha_at_eval"] == "headsha" and stamp["tree"] == "tree1"
    assert stamp["committed"] is True and stamp["eval_sample"] == "train"
    assert stamp["suffix"] == "parent" and stamp["score"] == 0.5
    assert stamp["trace_dir"].endswith("iteration_7_parent")
    # The stamp is the evals.jsonl line (plus trace_dir): one source of truth.
    log = tmp_path / "traces" / "iteration_7_evals.jsonl"
    line = json.loads(log.read_text().splitlines()[0])
    assert {k: v for k, v in stamp.items() if k != "trace_dir"} == line


def test_loop_comparison_is_marked_by_loop(monkeypatch, tmp_path):
    from src.engine.evaluator.sandbox_client import EvalResult
    from src.engine.optimizer.v3 import run_eval_tool as ret
    from src.engine.optimizer.v3.iteration_context import IterationContext

    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.setattr(ret, "git_get_sha", lambda ws, ref="HEAD": "headsha")
    monkeypatch.setattr(ret, "git_tree_signature", lambda ws, ref=None: "tree1")
    monkeypatch.setattr(ret, "git_get_current_branch", lambda ws: "b")
    monkeypatch.setattr(
        ret, "_run_eval",
        lambda *, row_ids, **kw: EvalResult(
            scores_by_metric={"acc": 1.0}, primary_score=1.0,
            trace_meta=[{"example_id": r, "score": 1.0} for r in row_ids],
            trace_dir="/t", trace_index_file="/ti", row_ids=list(row_ids), raw={},
        ),
    )
    ctx = IterationContext(iteration_idx=2, parent_locked=True)
    binding = ret.RunEvalBinding(
        ctx=ctx, workspace=str(tmp_path), config=None,  # type: ignore[arg-type]
        trainset_ids=[0], comparison_ids=[0, 1],
    )
    binding.run_comparison_for_loop()
    assert ctx.eval_records[-1]["by_loop"] is True
    assert ctx.eval_records[-1]["eval_id"]


# --- seed: engine artifacts excluded from git --------------------------------


def test_seed_excludes_engine_artifacts_from_the_tree(tmp_path):
    from src.engine.optimizer.v3.seed import _exclude_engine_artifacts_from_git
    from src.engine.services.subprocess_service import git_tree_signature

    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=ws, check=True)
    (ws / "program.py").write_text("K = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=ws, check=True)
    before = git_tree_signature(str(ws))

    _exclude_engine_artifacts_from_git(str(ws))
    _exclude_engine_artifacts_from_git(str(ws))  # idempotent
    (ws / ".codeevolver").mkdir()
    (ws / ".codeevolver" / "state.json").write_text("{}")
    (ws / ".codeevolver.lock").write_text("")
    (ws / ".opencode" / "skills").mkdir(parents=True)
    (ws / ".opencode" / "skills" / "SKILL.md").write_text("x")

    exclude = (ws / ".git" / "info" / "exclude").read_text()
    assert exclude.count(".codeevolver/") == 1
    assert git_tree_signature(str(ws)) == before, "engine artifacts are not program content"
