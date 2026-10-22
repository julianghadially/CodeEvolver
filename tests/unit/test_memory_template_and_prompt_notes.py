"""Copyright © 2026 440 Labs LLC

Engine 3 review-fix regressions: the memory template carries no Run id line,
and the architect initial prompt only carries the forced-explore note when
the explore-frequency guard actually fired (never on early iterations).
"""

from __future__ import annotations

import random

from src.engine.optimizer.v3.iteration_architect import IterationArchitect
from src.engine.optimizer.v3.prompts import build_memory_md_template
from tests.controls._arch_helpers import make_architect


def test_memory_template_has_no_run_id_line():
    body = build_memory_md_template(
        seed_branch="codeevolver-20260601120000-main",
        architecture_title="Seed program",
        method="ee",
    )
    assert "Run id" not in body
    assert "Seed branch: codeevolver-20260601120000-main" in body
    assert "## Candidate Pool" in body
    assert "## Macro Approaches" in body


def test_initial_prompt_has_no_explore_note_by_default():
    arch, ctx, _state = make_architect(iteration_idx=1)
    assert ctx.must_explore is False
    prompt = arch._build_initial_prompt()
    assert "MUST be an exploration iteration" not in prompt


def test_initial_prompt_carries_explore_note_only_when_guard_fired():
    arch, ctx, _state = make_architect(iteration_idx=6)
    ctx.must_explore = True
    prompt = arch._build_initial_prompt()
    assert "MUST be an exploration iteration" in prompt


def test_final_comparison_suffix_reads_trainset():
    from src.engine.evaluator.mounted.evaluate import _load_dataset
    import json, os, tempfile

    with tempfile.TemporaryDirectory() as ws:
        rel = "data/train.json"
        os.makedirs(os.path.join(ws, "data"))
        with open(os.path.join(ws, rel), "w") as f:
            json.dump([{"q": "a"}, {"q": "b"}], f)
        extra = {"trainset_path": rel, "testset_path": None}
        assert _load_dataset(ws, extra, "final_comparison") == [{"q": "a"}, {"q": "b"}]
        assert _load_dataset(ws, extra, "comparison") == [{"q": "a"}, {"q": "b"}]
        assert _load_dataset(ws, extra, "holdout") == []


def test_initial_prompt_carries_constraint_revert_note_only_when_flag_set():
    from src.engine.optimizer.v3.constraint_gate import PREVIOUS_ITERATION_REVERTED_NOTE

    arch, ctx, _state = make_architect(iteration_idx=3)
    assert ctx.previous_iteration_constraint_reverted is False
    assert PREVIOUS_ITERATION_REVERTED_NOTE not in arch._build_initial_prompt()
    ctx.previous_iteration_constraint_reverted = True
    prompt = arch._build_initial_prompt()
    assert PREVIOUS_ITERATION_REVERTED_NOTE in prompt
    # The fact only — no pointer to inspect git, no judge rationale.
    assert "git log" not in prompt
