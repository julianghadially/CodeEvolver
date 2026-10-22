"""Copyright © 2026 440 Labs LLC

pareto_evolution surface pins (freeform-based): the evo
prompt is the FREEFORM prompt plus `## Git Branch Background` and the
exploit-phase merge note — never any explore/exploit guidance; only the
`close_iteration` skill is staged (replacement); the loop-branch
close_iteration tool has one field; the memory template is the macro-free
one; plus per-row comparison scores in run_eval.

Together these pin the research plan's Variable Table: freeform and
pareto_evolution share prompt, skills and tools, and differ ONLY in the
loop-owned population mechanics.
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.prompts import (
    ARCHITECT_SYSTEM_PROMPT,
    EE_GIT_BRANCHING_SECTION,
    EVO_GIT_BRANCHING_SECTION,
    EVO_MERGE_NOTE,
    _MUST_EXPLORE_SECTION,
    build_architect_system_prompt,
    build_evo_architect_system_prompt,
    build_freeform_architect_system_prompt,
    build_memory_md_template,
)

_EE_HEADING = "## Git Branching Guidance / Candidate Pool"
_EVO_HEADING = "## Git Branch Background"
_SLOTS = ("{git_branching_section}", "{merge_note}", "{exploit_branch_line}", "{child_branch}", "{primary_branch}")
# Words that must never reach the freeform-based arms (mirrors
# tests/unit/test_freeform_prompt_and_skills.py::_FORBIDDEN).
_FORBIDDEN = (
    "explor", "exploit", "candidate pool", "register_branch",
    "improvement_mode", "macro approach", "primary branch",
)


def _ctx(**kw) -> IterationContext:
    base = dict(
        iteration_idx=3,
        primary_branch_at_start="codeevolver-20260601120000-main",
        child_branch="codeevolver-20260601120000-iter3",
    )
    base.update(kw)
    return IterationContext(**base)


def _build(builder, ctx):
    return builder(
        additional_instructions=None,
        program_path="my_pkg.program:make_program",
        workspace_path="/workspace",
        context=ctx,
    )


# --- prompt sections ---------------------------------------------------------


def test_evo_prompt_is_the_freeform_prompt_plus_the_git_section():
    ctx = _ctx()
    evo = _build(build_evo_architect_system_prompt, ctx)
    ee = _build(build_architect_system_prompt, ctx)
    ff = _build(build_freeform_architect_system_prompt, ctx)

    assert _EVO_HEADING in evo and _EE_HEADING not in evo
    assert "codeevolver-20260601120000-iter3" in evo
    assert "pareto frontier" in evo
    # Exactly one delta from freeform when the merge draw did not fire.
    section = EVO_GIT_BRANCHING_SECTION.format(
        child_branch="codeevolver-20260601120000-iter3"
    )
    assert evo.replace(section + "\n", "") == ff
    assert EVO_MERGE_NOTE not in evo

    assert _EE_HEADING in ee and _EVO_HEADING not in ee
    assert "codeevolver-20260601120000-main" in ee
    for slot in _SLOTS:
        assert slot not in evo, slot
        assert slot not in ee, slot
    assert not re.search(r"\{[a-z_]+\}", evo)


def test_evo_prompt_never_carries_search_guidance_even_when_guards_fire():
    """The arm has no explore guard, but pin the builder itself: with every
    flag that could add guidance set, nothing explore/exploit-flavoured
    renders — only the merge note (Treatment B's merging half)."""
    ctx = _ctx(must_explore=True, exploit_phase=True)
    evo = _build(build_evo_architect_system_prompt, ctx)
    assert EVO_MERGE_NOTE in evo
    assert _MUST_EXPLORE_SECTION not in evo
    low = evo.replace(EVO_MERGE_NOTE, "").lower()
    for word in _FORBIDDEN:
        assert word not in low, word
    # And the merge note itself names no search mode.
    assert "explor" not in EVO_MERGE_NOTE.lower()
    assert "exploit" not in EVO_MERGE_NOTE.lower()


def test_ee_prompt_template_is_the_ee_rendered_base():
    assert EE_GIT_BRANCHING_SECTION.strip() in ARCHITECT_SYSTEM_PROMPT
    assert EVO_GIT_BRANCHING_SECTION.strip() not in ARCHITECT_SYSTEM_PROMPT
    assert "{git_branching_section}" not in ARCHITECT_SYSTEM_PROMPT


def test_must_explore_section_appends_for_ee_only():
    ctx = _ctx(must_explore=True)
    assert _MUST_EXPLORE_SECTION in _build(build_architect_system_prompt, ctx)
    assert _MUST_EXPLORE_SECTION not in _build(build_evo_architect_system_prompt, ctx)
    assert "primary branch" not in _MUST_EXPLORE_SECTION


def test_evo_prompt_falls_back_to_start_branch_when_child_unset():
    ctx = _ctx(child_branch=None, start_branch="codeevolver-x-iter9")
    assert "codeevolver-x-iter9" in _build(build_evo_architect_system_prompt, ctx)


# --- memory template ---------------------------------------------------------


def test_memory_template_variants():
    ee = build_memory_md_template(seed_branch="s", architecture_title="t", method="ee")
    evo = build_memory_md_template(
        seed_branch="s", architecture_title="t", method="pareto_evolution"
    )
    freeform = build_memory_md_template(seed_branch="s", architecture_title="t", method="freeform")
    assert "## Candidate Pool" in ee and "## Macro Approaches" in ee
    # evo is macro-free like freeform (its population is loop-owned).
    assert evo == freeform
    assert "## Candidate Pool" not in freeform and "## Macro Approaches" not in freeform
    for body in (ee, evo, freeform):
        assert "{pool_section}" not in body and "{seed_branch}" not in body


# --- strategy + skill overrides ----------------------------------------------


def test_pareto_strategy_override_files_exist_and_keep_canonical_names():
    from src.engine.optimizer.v3.pareto_evolution import PARETO_STRATEGY

    assert PARETO_STRATEGY.loop_owns_parent is True
    assert PARETO_STRATEGY.close_iteration_mode == "loop_branch"
    assert PARETO_STRATEGY.enable_explore_guard is False
    assert PARETO_STRATEGY.skills_to_stage == ("close_iteration",)
    assert set(PARETO_STRATEGY.skill_overrides) == {"close_iteration"}
    assert PARETO_STRATEGY.skill_template_values is None
    for name, path in PARETO_STRATEGY.skill_overrides.items():
        text = Path(path).read_text()
        assert Path(path).exists()
        assert f"name: {name}" in text.splitlines()[1]
        low = text.lower()
        for word in _FORBIDDEN:
            assert word not in low, word
        assert "Promoting a branch" not in text
        assert "is_primary" not in text
    close_text = Path(PARETO_STRATEGY.skill_overrides["close_iteration"]).read_text()
    assert 'run_eval(eval_sample="comparison")' in close_text
    assert "Do NOT modify program files after it" in close_text
    assert "acknowledge_message" in close_text
    assert "including your branch name and its comparison-set score" in close_text
    assert "Do not rewrite `{workspace_path}/codeevolver.md` yourself" in close_text
    assert "## Saving to Memory" in close_text
    assert "- `memory_updated`: true / false" in close_text
    assert "stay on the branch you were given" in close_text.lower()


def test_strategy_requires_on_iteration_start_when_loop_owns_parent():
    from src.engine.optimizer.v3.strategy import OptimizerStrategy

    with pytest.raises(ValueError, match="on_iteration_start"):
        OptimizerStrategy(method="x", prompt_builder=lambda **k: "", loop_owns_parent=True)


def test_skill_override_is_staged_under_canonical_name_and_rendered(tmp_path):
    from src.engine.agent.harnesses.skills import stage_codeevolver_skills_to
    from src.engine.paths import memory_dir

    repl = tmp_path / "repl.md"
    repl.write_text("---\nname: close_iteration\n---\nWrite to {memory_dir}/memory.md\n")
    staged = stage_codeevolver_skills_to(
        tmp_path / "skills", workspace=tmp_path, overrides={"close_iteration": repl}
    )
    text = (staged / "close_iteration" / "SKILL.md").read_text()
    assert text.startswith("---\nname: close_iteration")
    assert "{memory_dir}" not in text and memory_dir() in text
    assert "register_branch" not in text
    # Other skills untouched.
    assert "register_branch" in (staged / "exploration" / "SKILL.md").read_text() or (
        "Candidate Pool" in (staged / "exploration" / "SKILL.md").read_text()
    )


def test_skill_override_unknown_skill_raises(tmp_path):
    from src.engine.agent.harnesses.skills import stage_codeevolver_skills_to

    repl = tmp_path / "repl.md"
    repl.write_text("x")
    with pytest.raises(ValueError, match="unknown skill"):
        stage_codeevolver_skills_to(tmp_path / "skills", overrides={"nope": repl})


def test_pareto_strategy_overrides_stage_cleanly(tmp_path):
    from src.engine.agent.harnesses.skills import stage_codeevolver_skills_to
    from src.engine.optimizer.v3.pareto_evolution import PARETO_STRATEGY

    staged = stage_codeevolver_skills_to(
        tmp_path / "skills",
        workspace=tmp_path,
        skills=PARETO_STRATEGY.skills_to_stage,
        overrides=PARETO_STRATEGY.skill_overrides,
    )
    names = sorted(p.name for p in staged.iterdir() if p.is_dir())
    assert names == ["close_iteration"]
    text = (staged / "close_iteration" / "SKILL.md").read_text()
    assert "{memory_dir}" not in text and "{history_dir}" not in text
    assert "{workspace_path}" not in text
    assert not re.search(r"\{[a-z_]+\}", text)
    low = text.lower()
    for word in _FORBIDDEN:
        assert word not in low, word


# --- close_iteration tool schema --------------------------------------------


def test_loop_branch_close_iteration_schema_has_one_field():
    from src.engine.agent.harnesses.architect_tools import (
        ARCHITECT_TOOL_SPECS,
        architect_tool_spec,
        mcp_architect_tool_definitions,
    )

    loop = architect_tool_spec("close_iteration", close_iteration_mode="loop_branch")
    assert loop.name == "close_iteration"
    assert loop.input_schema["required"] == ["memory_updated"]
    assert list(loop.input_schema["properties"]) == ["memory_updated"]
    assert "do not switch branches" in loop.description
    assert "improvement_mode" not in loop.description
    assert "explor" not in loop.description.lower()

    default = architect_tool_spec("close_iteration")
    assert default is ARCHITECT_TOOL_SPECS["close_iteration"]
    assert "register_branch" in default.input_schema["required"]

    defs = {d["name"]: d for d in mcp_architect_tool_definitions(close_iteration_mode="loop_branch")}
    assert defs["close_iteration"]["inputSchema"]["required"] == ["memory_updated"]
    defs = {d["name"]: d for d in mcp_architect_tool_definitions()}
    assert "register_branch" in defs["close_iteration"]["inputSchema"]["required"]


# --- run_eval per-row scores + loop comparison ---------------------------------


def test_row_scores_from_trace_meta_skips_bad_rows():
    from src.engine.optimizer.v3.run_eval_tool import row_scores_from_trace_meta

    meta = [
        {"example_id": 1, "score": 1.0},
        {"example_id": "q7", "score": 0},
        {"example_id": None, "score": 1.0},
        {"example_id": 3, "score": None},
        {"example_id": 4, "score": "bad"},
    ]
    assert row_scores_from_trace_meta(meta) == {"1": 1.0, "q7": 0.0}


def test_comparison_eval_records_row_scores_and_loop_variant_bypasses_budget(tmp_path):
    from src.engine.evaluator.config import EvaluatorConfig
    from src.engine.evaluator.sandbox_client import EvalResult
    from src.engine.optimizer.v3.ce_state import CEState
    from src.engine.optimizer.v3.run_eval_tool import RunEvalBinding, RunEvalError

    def fake_run_eval(**kwargs):
        rows = kwargs["row_ids"]
        return EvalResult(
            scores_by_metric={"accuracy": 0.5},
            primary_score=0.5,
            trace_meta=[{"example_id": r, "score": float(r % 2)} for r in rows],
            trace_dir="/t.jsonl",
            trace_index_file="/t_index.jsonl",
            row_ids=list(rows),
            raw={},
        )

    hist = tmp_path / "comparison_set_history.jsonl"
    state = CEState(seed_branch="seed", seed_sha="sha", comparison_history_path=str(hist))
    state.increment_evals(100)  # budget already exhausted
    ctx = IterationContext(iteration_idx=1, parent_locked=True, max_eval_rows=100, evals_used=100)
    (tmp_path / "train.json").write_text(json.dumps([{"id": i} for i in range(6)]))
    binding = RunEvalBinding(
        ctx=ctx,
        workspace=str(tmp_path),
        config=EvaluatorConfig(
            program_factory="p:m", metric_fns=["m:a"], metric_names=["accuracy"],
            primary_metric="accuracy", trainset_path="train.json",
        ),
        trainset_ids=list(range(6)),
        comparison_ids=[0, 1, 2, 3],
        rng=random.Random(0),
        state=state,
    )
    with patch.multiple(
        "src.engine.optimizer.v3.run_eval_tool",
        _run_eval=fake_run_eval,
        git_get_sha=lambda _ws: "sha1",
        git_tree_signature=lambda _ws: "tree1",
        git_get_current_branch=lambda _ws: "codeevolver-x-iter1",
        git_dirty_paths=lambda _ws: [],
    ):
        # Agent-facing: over budget, train refused.
        with pytest.raises(RunEvalError):
            binding("train")
        # Loop variant: runs, counts, records per-row scores, never touches
        # the final-comparison allowance.
        result = binding.run_comparison_for_loop()
    assert result.primary_score == 0.5
    assert state.total_num_evals == 104
    assert state.final_comparison_evals == 0
    rec = ctx.eval_records[-1]
    assert rec["eval_sample"] == "comparison"
    assert rec["row_scores"] == {"0": 0.0, "1": 1.0, "2": 0.0, "3": 1.0}
    line = json.loads(hist.read_text().strip().splitlines()[-1])
    assert line["row_scores"] == rec["row_scores"]
    assert line["branch"] == "codeevolver-x-iter1"
    # comparison_record_for_tree matches by exact tree / equivalence.
    ctx.close_tree = "tree1"
    assert ctx.comparison_record_for_tree(lambda a, b: False) is rec
    ctx.close_tree = "other"
    assert ctx.comparison_record_for_tree(lambda a, b: False) is None
    assert ctx.comparison_record_for_tree(lambda a, b: True) is rec


def test_every_close_iteration_description_says_load_the_skill_first():
    from src.engine.agent.harnesses.architect_tools import (
        LOAD_CLOSE_SKILL_FIRST,
        architect_tool_spec,
    )

    for mode in (None, "loop_branch", "head"):
        spec = architect_tool_spec("close_iteration", close_iteration_mode=mode)
        assert spec.description.startswith(LOAD_CLOSE_SKILL_FIRST)
        assert "BEFORE calling this tool" in spec.description
