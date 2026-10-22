"""ASA surface pins: EE prompt untouched unless a requirement fired, rule
sections/notes per flag, skill overrides, tool exposure, memory template,
config round-trip, strategy wiring (specs/archive-public/v3_asa.md §Prompts)."""

from pathlib import Path

import pytest

from src.engine.agent.harnesses.architect_tools import (
    ARCHITECT_TOOL_SPECS,
    mcp_architect_tool_definitions,
    ordered_architect_tool_names,
)
from src.engine.optimizer.v3.asa.asa_optimizer import ASA_STRATEGY, seed_lineage_name
from src.engine.optimizer.v3.asa.prompt_sections import (
    REQ_ELIGIBLE_NONPRIMARY,
    REQ_NEW_LINEAGE,
    system_prompt_sections,
    user_message_notes,
)
from src.engine.optimizer.v3.ee.ee_optimizer import EE_STRATEGY
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.prompts import (
    build_architect_system_prompt,
    build_memory_md_template,
)


def _ctx(**kw) -> IterationContext:
    base = dict(iteration_idx=3, subsample_size=10, primary_branch_at_start="codeevolver-x-main",
                start_branch="codeevolver-x-main", parent_locked=True)
    base.update(kw)
    return IterationContext(**base)


def _prompt(ctx):
    return build_architect_system_prompt(
        additional_instructions=None, program_path="p:m", workspace_path="/ws", context=ctx,
    )


def test_asa_prompt_is_ee_prompt_when_nothing_fired():
    assert ASA_STRATEGY.prompt_builder is EE_STRATEGY.prompt_builder
    assert _prompt(_ctx()) == _prompt(_ctx(asa_requirements=[]))
    assert "MUST EXPLORE AND ADD A NEW LINEAGE" not in _prompt(_ctx())
    assert "ELIGIBLE NON-PRIMARY" not in _prompt(_ctx())


def test_rule_sections_append_per_requirement_and_list_eligible():
    ctx = _ctx(asa_requirements=[REQ_NEW_LINEAGE])
    p = _prompt(ctx)
    assert "## THIS ITERATION MUST EXPLORE AND ADD A NEW LINEAGE" in p
    assert "ELIGIBLE NON-PRIMARY" not in p
    ctx = _ctx(asa_requirements=[REQ_NEW_LINEAGE, REQ_ELIGIBLE_NONPRIMARY],
               asa_eligible_at_start=["codeevolver-react"],
               asa_eligible_contributions_at_start={"codeevolver-react": 0.0625})
    p = _prompt(ctx)
    assert "MUST EXPLORE AND ADD A NEW LINEAGE" in p
    assert "## THIS ITERATION MUST SELECT AN ELIGIBLE NON-PRIMARY BRANCH AS ITS PARENT" in p
    assert "`codeevolver-react`: 0.0625" in p
    # EE base still precedes the injected sections
    assert p.index("MUST EXPLORE AND ADD") > p.index("Candidate Pool")


def test_must_explore_and_asa_sections_coexist():
    ctx = _ctx(must_explore=True, asa_requirements=[REQ_NEW_LINEAGE])
    p = _prompt(ctx)
    assert "## THIS ITERATION MUST EXPLORE\n" in p and "ADD A NEW LINEAGE" in p


def test_user_message_notes_and_sections_are_empty_without_requirements():
    assert user_message_notes(_ctx()) == "" and system_prompt_sections(_ctx()) == ""
    notes = user_message_notes(_ctx(asa_requirements=[REQ_ELIGIBLE_NONPRIMARY]))
    assert "eligible non-primary branch as its parent" in notes


def test_branch_comparison_spec_exists_but_only_registry_members_are_exposed():
    assert "branch_comparison" in ARCHITECT_TOOL_SPECS
    spec = ARCHITECT_TOOL_SPECS["branch_comparison"]
    assert spec.input_schema["required"] == []  # `reference` optional
    assert "branch_comparison" in ordered_architect_tool_names()
    ee_names = {"run_eval", "coding_agent", "inspect_traces", "close_iteration"}
    exposed = {d["name"] for d in mcp_architect_tool_definitions(registered_names=ee_names)}
    assert "branch_comparison" not in exposed
    exposed = {d["name"] for d in mcp_architect_tool_definitions(registered_names=ee_names | {"branch_comparison"})}
    assert "branch_comparison" in exposed


def test_strategy_wiring():
    assert ASA_STRATEGY.method == "asa"
    assert ASA_STRATEGY.enable_explore_guard and not ASA_STRATEGY.loop_owns_parent
    assert ASA_STRATEGY.close_iteration_mode == "register_branch"
    assert ASA_STRATEGY.commit_before_comparison and not EE_STRATEGY.commit_before_comparison
    assert ASA_STRATEGY.run_eval_gate is not None and EE_STRATEGY.run_eval_gate is None
    assert ASA_STRATEGY.extra_tools is not None and EE_STRATEGY.extra_tools is None
    assert set(ASA_STRATEGY.skill_overrides) == {"close_iteration", "exploration"}
    for p in ASA_STRATEGY.skill_overrides.values():
        assert Path(p).exists()


def test_skill_overrides_carry_spec_sections_and_stage_under_canonical_names(tmp_path):
    from src.engine.agent.harnesses.skills import stage_codeevolver_skills_to

    explore = Path(ASA_STRATEGY.skill_overrides["exploration"]).read_text()
    assert "### Non-primary branches" in explore and "### Branch comparison tool" in explore
    assert "NEVER run the metric" in explore  # run_eval-only rule kept
    close = Path(ASA_STRATEGY.skill_overrides["close_iteration"]).read_text()
    assert "### Promoting a branch to the primary branch / Comparison Set Usage" in close
    assert "2. If your changes made meaningful progress" in close
    assert "dropping a branch from the pool does not retire it" in close
    dest = stage_codeevolver_skills_to(tmp_path, overrides=ASA_STRATEGY.skill_overrides)
    staged = (Path(dest) / "exploration" / "SKILL.md").read_text()
    assert "name: exploration" in staged and "branch_comparison" in staged
    assert "{memory_dir}" not in staged


def test_memory_template_keeps_candidate_pool_for_asa():
    t = build_memory_md_template(seed_branch="codeevolver-x-main", architecture_title="t", method="asa")
    assert "Candidate Pool" in t and "Macro Approaches" in t


def test_seed_lineage_name():
    assert seed_lineage_name("codeevolver-20260601-main") == "codeevolver-20260601-seed"
    assert seed_lineage_name("codeevolver-other") == "codeevolver-other-seed"


def test_config_round_trip_of_asa_knobs():
    from src.engine.optimizer.v3.optimization_config import OptimizationConfig

    d = {"program": "p", "metric": "m", "asa_behavioral_contribution_constant": 0.2,
         "asa_behavioral_contribution_threshold_cap": 0.03, "asa_primary_stale_landed_iterations": 6}
    cfg = OptimizationConfig.from_dict(d)
    assert cfg.asa_behavioral_contribution_constant == 0.2
    assert cfg.asa_behavioral_contribution_threshold_cap == 0.03
    assert cfg.asa_primary_stale_landed_iterations == 6
    back = cfg.to_dict()
    assert OptimizationConfig.from_dict(back).to_dict() == back
    defaults = OptimizationConfig.from_dict({"program": "p", "metric": "m"})
    assert (defaults.asa_behavioral_contribution_constant, defaults.asa_behavioral_contribution_threshold_cap,
            defaults.asa_primary_stale_landed_iterations) == (0.10, 0.05, 4)


def test_asa_record_fields_are_whitelisted_everywhere():
    from src.db.db_schemas.db_schemas import IterationHistoryRecord
    from src.engine.schemas.iteration_history import ITERATION_RECORD_FIELDS

    keys = ["asa_requirements", "asa_satisfied", "asa_violation", "asa_eligible_at_start",
            "asa_parent_is_eligible", "asa_parent_positive_contribution_to_primary_normalized",
            "asa_allowance_left", "asa_free_tail", "asa_phase1", "asa_loop_comparison_reason",
            "integrity_budget_bypass", "integrity_budget_bypass_estimated_rows",
            "integrity_budget_bypass_rationale"]
    for k in keys:
        assert k in ITERATION_RECORD_FIELDS, k
        assert k in IterationHistoryRecord.model_fields, k
