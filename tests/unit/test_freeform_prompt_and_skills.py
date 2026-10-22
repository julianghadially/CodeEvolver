"""Copyright © 2026 440 Labs LLC

Freeform (Baseline 1) surface pins — `specs/v3_freeform.md`:

- the prompt carries no explore/exploit, primary, candidate-pool or
  comparison-set guidance, never appends the must-explore block, and renders
  every slot;
- `exploration` is not staged; the close-skill override keeps the canonical
  name, mentions none of the population concepts, and renders
  `{final_iteration_note}`;
- the `head` close tool schema has exactly `memory_updated`;
- the final-iteration note trigger;
- the strategy flags and the memory template.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.engine.agent.harnesses.architect_tools import (
    architect_tool_spec,
    mcp_architect_tool_definitions,
)
from src.engine.agent.harnesses.skills import stage_codeevolver_skills_to
from src.engine.optimizer.v3.freeform.freeform_optimizer import (
    FINAL_ITERATION_NOTE,
    FREEFORM_STRATEGY,
    final_iteration_note,
    freeform_skill_template_values,
)
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.prompts import (
    FREEFORM_ARCHITECT_SYSTEM_PROMPT,
    _MUST_EXPLORE_SECTION,
    build_architect_system_prompt,
    build_freeform_architect_system_prompt,
    build_memory_md_template,
)

_FORBIDDEN = (
    "explor", "exploit", "primary", "candidate pool",
    "register_branch", "improvement_mode", "macro approach",
)


def _build(builder, **kw):
    ctx = IterationContext(iteration_idx=3, subsample_size=20, max_eval_rows=1000, evals_used=100, **kw)
    return builder(
        additional_instructions="Do not hard-code answers.",
        program_path="my_pkg.program:make_program",
        workspace_path="/ws/repo",
        context=ctx,
    )


def test_freeform_prompt_has_no_search_guidance_and_renders_every_slot():
    prompt = _build(build_freeform_architect_system_prompt)
    low = prompt.lower()
    for word in _FORBIDDEN:
        assert word not in low, word
    assert not re.search(r"\{[a-z_]+\}", prompt)
    assert "/ws/repo" in prompt
    assert "iteration_change_summaries.md" in prompt
    assert "Do not hard-code answers." in prompt
    assert "my_pkg.program:make_program" in prompt
    assert "close_iteration skill" in prompt
    assert "### Sampling Guidance" in prompt
    assert "### Eval-row budget (run-wide)" in prompt


def test_freeform_prompt_never_appends_must_explore():
    prompt = _build(build_freeform_architect_system_prompt, must_explore=True)
    assert _MUST_EXPLORE_SECTION not in prompt
    assert _MUST_EXPLORE_SECTION in _build(build_architect_system_prompt, must_explore=True)


def test_ee_keeps_the_comparison_sentence_freeform_drops_it():
    ee = _build(build_architect_system_prompt)
    ff = _build(build_freeform_architect_system_prompt)
    assert 'run_eval(eval_sample="comparison")' in ee
    assert 'run_eval(eval_sample="comparison")' not in ff
    assert "apples-to-apples" not in ff
    assert FREEFORM_ARCHITECT_SYSTEM_PROMPT.count("{subsample_note}") == 1


# --- strategy + skills -------------------------------------------------------


def test_strategy_flags():
    assert FREEFORM_STRATEGY.method == "freeform"
    assert FREEFORM_STRATEGY.close_iteration_mode == "head"
    assert FREEFORM_STRATEGY.enable_explore_guard is False
    assert FREEFORM_STRATEGY.loop_owns_parent is False
    assert FREEFORM_STRATEGY.on_iteration_start is None
    assert "exploration" not in FREEFORM_STRATEGY.skills_to_stage
    assert "close_iteration" in FREEFORM_STRATEGY.skills_to_stage
    assert set(FREEFORM_STRATEGY.skill_overrides) == {"close_iteration"}
    assert FREEFORM_STRATEGY.skill_template_values is freeform_skill_template_values


def test_close_skill_override_keeps_canonical_name_and_no_population_concepts():
    path = Path(FREEFORM_STRATEGY.skill_overrides["close_iteration"])
    text = path.read_text()
    assert "name: close_iteration" in text.splitlines()[1]
    low = text.lower()
    for word in _FORBIDDEN:
        assert word not in low, word
    assert "{final_iteration_note}" in text
    assert "memory_updated" in text


def test_staging_omits_exploration_and_renders_the_final_note(tmp_path):
    dest = stage_codeevolver_skills_to(
        tmp_path / "skills",
        workspace=tmp_path,
        skills=FREEFORM_STRATEGY.skills_to_stage,
        overrides=FREEFORM_STRATEGY.skill_overrides,
        extra_values={"final_iteration_note": FINAL_ITERATION_NOTE},
    )
    assert not (dest / "exploration").exists()
    close = (dest / "close_iteration" / "SKILL.md").read_text()
    assert FINAL_ITERATION_NOTE in close
    assert "{final_iteration_note}" not in close
    assert str(tmp_path) in close and "{memory_dir}" not in close

    # Early in the run the slot renders empty, never leaks.
    dest2 = stage_codeevolver_skills_to(
        tmp_path / "skills2",
        workspace=tmp_path,
        skills=FREEFORM_STRATEGY.skills_to_stage,
        overrides=FREEFORM_STRATEGY.skill_overrides,
        extra_values={"final_iteration_note": ""},
    )
    close2 = (dest2 / "close_iteration" / "SKILL.md").read_text()
    assert "{final_iteration_note}" not in close2 and "Budget exhaustion" not in close2


# --- final-iteration note ----------------------------------------------------


def test_final_iteration_note_trigger():
    def ctx(**kw):
        return IterationContext(iteration_idx=1, **kw)

    assert final_iteration_note(ctx(max_eval_rows=None, evals_used=0, subsample_size=50)) == ""
    assert final_iteration_note(ctx(max_eval_rows=1000, evals_used=0, subsample_size=50)) == ""
    assert final_iteration_note(ctx(max_eval_rows=1000, evals_used=749, subsample_size=50)) == ""
    assert final_iteration_note(ctx(max_eval_rows=1000, evals_used=750, subsample_size=50)) == FINAL_ITERATION_NOTE
    # No subsample size known: fall back to the 90% budget fraction.
    assert final_iteration_note(ctx(max_eval_rows=1000, evals_used=899, subsample_size=0)) == ""
    assert final_iteration_note(ctx(max_eval_rows=1000, evals_used=900, subsample_size=0)) == FINAL_ITERATION_NOTE
    assert freeform_skill_template_values(ctx(max_eval_rows=1000, evals_used=990, subsample_size=5)) == {
        "final_iteration_note": FINAL_ITERATION_NOTE
    }


# --- close tool schema -------------------------------------------------------


def test_head_close_iteration_schema_has_memory_updated_only():
    spec = architect_tool_spec("close_iteration", close_iteration_mode="head")
    assert spec.name == "close_iteration"
    assert spec.input_schema["required"] == ["memory_updated"]
    assert list(spec.input_schema["properties"]) == ["memory_updated"]
    assert "improvement_mode" not in spec.description
    defs = {d["name"]: d for d in mcp_architect_tool_definitions(close_iteration_mode="head")}
    assert defs["close_iteration"]["inputSchema"]["required"] == ["memory_updated"]


# --- memory template ---------------------------------------------------------


def test_freeform_memory_template_keeps_seed_only():
    body = build_memory_md_template(seed_branch="codeevolver-x-main", architecture_title="T", method="freeform")
    assert "Seed branch: codeevolver-x-main" in body and "Seed architecture: T" in body
    assert "## Candidate Pool" not in body and "## Macro Approaches" not in body
    assert "## Overall Progress" in body
