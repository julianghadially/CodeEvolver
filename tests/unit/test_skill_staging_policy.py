"""Every arm names its staged skills explicitly (spec §29).

`prompt_engineering` and `tracing` stay in the canonical skills directory for
the engine's own use but are never staged for an architect; the only
inter-arm skill delta is `exploration` (absent on freeform and on
pareto_evolution, which uses the freeform prompt).
"""

from __future__ import annotations

import pytest

from src.engine.optimizer.v3.asa.asa_optimizer import ASA_STRATEGY
from src.engine.optimizer.v3.ee.ee_optimizer import EE_STRATEGY
from src.engine.optimizer.v3.freeform.freeform_optimizer import FREEFORM_STRATEGY
from src.engine.optimizer.v3.pareto_evolution.pareto_optimizer import PARETO_STRATEGY

NEVER_STAGED = {"prompt_engineering", "tracing"}


@pytest.mark.parametrize(
    "strategy", [EE_STRATEGY, ASA_STRATEGY, PARETO_STRATEGY, FREEFORM_STRATEGY],
    ids=lambda s: s.method,
)
def test_every_arm_stages_an_explicit_list_without_prompt_engineering_or_tracing(strategy):
    assert strategy.skills_to_stage is not None, strategy.method
    assert not NEVER_STAGED & set(strategy.skills_to_stage), strategy.method
    assert "close_iteration" in strategy.skills_to_stage


def test_exploration_is_the_only_inter_arm_skill_delta():
    for strategy in (EE_STRATEGY, ASA_STRATEGY):
        assert set(strategy.skills_to_stage) == {"close_iteration", "exploration"}
    for strategy in (FREEFORM_STRATEGY, PARETO_STRATEGY):
        assert set(strategy.skills_to_stage) == {"close_iteration"}
