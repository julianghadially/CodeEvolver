"""Copyright © 2026 440 Labs LLC

Unit tests for the simple loop's post-run meta-reflection (offline — no SDK).

Two things carry most of the weight here:

* Anti-contamination. The prompt must not point the agent at artifacts a
  simple-loop run never produces (`memory.md`, `iteration_change_summaries.md`,
  skills, `inspect_traces`, `codeevolver.md`).
* Research purity. The prompt must state that this is the control arm and
  that proposing to add memory / skills / trace tooling is out of scope.

Assertions key on surface headers and read-paths, never on bare identifiers:
the purity clause legitimately names memory and skills in order to forbid them.
"""

from __future__ import annotations

import os

from src.engine.optimizer.meta import simple_meta_reflection as smr
from src.engine.paths import memory_dir


def _history() -> dict[int, dict]:
    return {
        0: {"git_branch": "codeevolver-20260827-main", "close_sha": "abcdef1234567890", "skipped": False},
        1: {
            "git_branch": "codeevolver-20260827-main",
            "parent_sha": "abcdef1234567890", "close_sha": "1111222233334444",
            "eval_rows_used": 100, "initial_subsample_score": 0.4000,
            "best_subsample_score": 0.5500, "comparison_best_score": 0.5100,
            "changes_landed": True, "constraint_violation": "no new dependencies",
        },
        2: {
            "git_branch": "codeevolver-20260827-main",
            "parent_sha": "1111222233334444", "close_sha": "1111222233334444",
            "eval_rows_used": 0, "initial_subsample_score": 0.5500,
            "best_subsample_score": None, "changes_landed": False,
            "subsample_error": "eval timed out after 600s",
        },
    }


def _prompt(**overrides) -> str:
    kwargs = dict(
        workspace="/workspace/repo", history=_history(), iteration_indices=[1, 2],
        branch="codeevolver-20260827-main", program_path="program.py",
        seed_test_score=0.4200, final_test_score=0.5100, subsample_size=20,
        max_eval_rows=600, total_evals=340,
    )
    kwargs.update(overrides)
    return smr.build_simple_loop_meta_reflection_prompt(**kwargs)


def test_prompt_names_all_five_surfaces():
    prompt = _prompt()
    for surface in ("A. Simple-loop system prompt", "B. Tool contracts", "C. Loop controls",
                    "D. Budget guidance", "E. Diagnostics"):
        assert surface in prompt
    assert "ADVISORY" in prompt
    assert "held-out" in prompt.lower()


def test_prompt_omits_codeevolver_only_surfaces():
    prompt = _prompt()
    assert "B. Skills" not in prompt
    assert "E. Memory schema" not in prompt
    assert f"{memory_dir()}/memory.md" not in prompt
    assert "iteration_change_summaries" not in prompt
    assert "codeevolver.md" not in prompt
    assert "coding_agent" not in prompt


def test_prompt_states_research_purity_constraint():
    prompt = _prompt()
    assert "simple loop" in prompt
    assert "is NOT to change" in prompt
    assert "out of scope" in prompt


def test_prompt_points_at_simple_loop_source_and_specs():
    from src.engine.paths import engine_root

    prompt = _prompt()
    assert engine_root() in prompt
    assert os.path.join("src", "engine", "optimizer", "simple_loop") in prompt
    for name in ("prompt.md", "constraint_check.py", "session.py", "loop.py", "v3_simple_loop.md"):
        assert name in prompt
    assert os.path.join("src", "engine", "optimizer", "v3", "prompts.py") not in prompt
    assert "commit_watchdog" not in prompt


def test_prompt_graceful_when_specs_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(smr, "engine_root", lambda: str(tmp_path))
    prompt = _prompt()
    assert "NOT present in this environment" in prompt
    assert os.path.join(str(tmp_path), "src", "engine", "optimizer", "simple_loop") in prompt


def test_prompt_embeds_run_table_facts():
    prompt = _prompt()
    assert "11112222" in prompt
    assert "0.4000" in prompt and "0.5500" in prompt and "0.5100" in prompt
    assert "no new dependencies" in prompt
    assert "eval timed out" in prompt
    row2 = [ln for ln in prompt.splitlines() if ln.startswith("| 2 ")][0]
    assert "| no |" in row2
    row1 = [ln for ln in prompt.splitlines() if ln.startswith("| 1 ")][0]
    assert "| 100 |" in row1 and "| yes |" in row1


def test_run_table_shape():
    table = smr._run_table(_history())
    lines = table.splitlines()
    assert all(ln.startswith("|") for ln in lines)
    assert len(lines) == len(_history()) + 2
    assert smr._run_table({}) == "(no sessions recorded)"


def test_seed_excluded_from_trace_indices(monkeypatch):
    captured = {}
    monkeypatch.setattr(smr, "run_reflection_agent",
                        lambda **kw: captured.update(prompt=kw["prompt"]) or "ok")
    smr.run_simple_loop_meta_reflection(workspace="/w", history=_history())
    assert "for N in: 1, 2" in captured["prompt"]
    assert "for N in: 0, 1, 2" not in captured["prompt"]


def test_subsample_score_labelled_in_sample():
    prompt = _prompt()
    assert "IN-SAMPLE" in prompt and "NOT a holdout" in prompt


def test_prompt_handles_empty_history_and_missing_scores():
    prompt = _prompt(history={}, iteration_indices=[], seed_test_score=None,
                     final_test_score=None, subsample_size=None, max_eval_rows=None,
                     total_evals=None)
    assert "N/A" in prompt and "(none)" in prompt and "(no sessions recorded)" in prompt


def test_run_simple_loop_meta_reflection_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    captured = {}

    def fake_run_inspection_agent(**kwargs):
        captured["deny_paths"] = kwargs.get("deny_paths")
        captured["writable"] = kwargs.get("writable")
        return {"success": True, "proposed_change": "## Proposals\n- A: tighten prompt"}

    import src.engine.services.subprocess_service as svc

    monkeypatch.setattr(svc, "run_inspection_agent", fake_run_inspection_agent)
    out = smr.run_simple_loop_meta_reflection(
        workspace="/workspace", history=_history(), branch="codeevolver-main",
        program_path="program.py", coding_harness="opencode",
    )
    assert out is not None and "Proposals" in out
    with open(os.path.join(str(tmp_path), "meta_reflection.md")) as f:
        assert "Proposals" in f.read()
    assert captured["writable"] is False and captured["deny_paths"] == []


def test_run_simple_loop_meta_reflection_non_fatal_on_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    import src.engine.services.subprocess_service as svc

    def boom(**kwargs):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(svc, "run_inspection_agent", boom)
    assert smr.run_simple_loop_meta_reflection(workspace="/w", history=_history()) is None
    assert not os.path.exists(os.path.join(str(tmp_path), "meta_reflection.md"))
