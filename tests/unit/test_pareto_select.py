"""Copyright © 2026 440 Labs LLC

Unit tests for the Pareto-Evolution parent-selection policy
(`v3/pareto_evolution/select_candidate.py`) — GEPA-style instance-front
proportional sampling over the loop-owned `Population`, the fallback chain,
rng determinism, the exploit-and-merge draw, and the `evo_select_parent`
wiring into `select_parent.fork_child_from`.
"""

from __future__ import annotations

import random
from collections import Counter
from unittest.mock import patch

from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.pareto_evolution.population import Population
from src.engine.optimizer.v3.pareto_evolution.select_candidate import (
    _instance_front_counts,
    _pareto_sample,
    _weighted_choice,
    choose_parent_iteration,
    evo_select_parent,
)


def _cand(it: int, branch: str, sha: str, rows: dict[str, float]) -> dict:
    return {
        "iteration": it,
        "branch": branch,
        "sha": sha,
        "tree": None,
        "parent_iteration": None if it == 0 else 0,
        "row_scores": rows,
        "comparison_set_score": sum(rows.values()) / len(rows),
        "train_baseline": None,
        "train_best": None,
    }


def _make_pop() -> Population:
    """Seed + two candidates with a known instance front.

    Scores per comparison row:
        seed (0): a=0.5, b=0.5, c=0.5
        cand (1): a=0.9, b=0.5, c=0.4
        cand (2): a=0.9, b=0.6, c=0.5

    Unpruned instance front: a -> {1, 2}, b -> {2}, c -> {0, 2}. After GEPA
    Algorithm 2 pruning candidates 1 ({a}) and 0 ({c}) are strict
    subsets of 2 ({a, b, c}) and are dropped: a -> {2}, b -> {2}, c -> {2}.
    Win counts: 2 -> 3.
    """
    pop = Population()
    pop.add(_cand(0, "seed", "s0", {"a": 0.5, "b": 0.5, "c": 0.5}))
    pop.add(_cand(1, "b1", "s1", {"a": 0.9, "b": 0.5, "c": 0.4}))
    pop.add(_cand(2, "b2", "s2", {"a": 0.9, "b": 0.6, "c": 0.5}))
    return pop


# --- _instance_front_counts ---------------------------------------------------


def test_instance_front_counts_match_pruned_instance_wins():
    counts = _instance_front_counts(_make_pop())
    assert counts == Counter({2: 3})


def test_instance_front_counts_ties_count_for_every_undominated_winner():
    pop = Population()
    pop.add(_cand(0, "seed", "s0", {"a": 0.9, "b": 0.1}))
    pop.add(_cand(1, "b1", "s1", {"a": 0.9, "b": 0.1}))  # twin of 0 -> one pruned
    pop.add(_cand(2, "b2", "s2", {"a": 0.1, "b": 0.9}))
    front = pop.get_pareto_front()
    # GEPA keeps one of two identical winning sets (tie on score -> lower key
    # pruned); 2 wins b uniquely and stays.
    assert front["a"] == {1} and front["b"] == {2}
    counts = _instance_front_counts(pop)
    assert counts == Counter({1: 1, 2: 1})


def test_instance_front_counts_missing_getter_returns_empty():
    class _NoFront:
        pass

    assert _instance_front_counts(_NoFront()) == Counter()


# --- _weighted_choice ---------------------------------------------------------


def test_weighted_choice_empty_or_zero_weights_returns_none():
    rng = random.Random(0)
    assert _weighted_choice({}, rng) is None
    assert _weighted_choice({1: 0, 2: 0}, rng) is None


def test_weighted_choice_proportional_sampling():
    rng = random.Random(42)
    weights = {0: 1, 1: 1, 2: 3}
    draws = Counter(_weighted_choice(weights, rng) for _ in range(10_000))
    total = sum(weights.values())
    for idx, w in weights.items():
        expected = w / total
        observed = draws[idx] / 10_000
        assert abs(observed - expected) < 0.03, (idx, observed, expected)


# --- _pareto_sample -----------------------------------------------------------


def test_pareto_sample_only_frontier_winners_selectable():
    pop = _make_pop()
    # A dominated candidate (wins nothing) must never be sampled.
    pop.add(_cand(3, "b3", "s3", {"a": 0.1, "b": 0.1, "c": 0.1}))
    rng = random.Random(7)
    picks = {_pareto_sample(pop, rng) for _ in range(500)}
    assert 3 not in picks
    assert picks <= {0, 1, 2}


def test_pareto_sample_seed_only_pool_returns_seed():
    pop = Population()
    pop.add(_cand(0, "seed", "s0", {"a": 0.5}))
    assert _pareto_sample(pop, random.Random(0)) == 0


def test_pareto_sample_rng_determinism():
    pop = _make_pop()
    picks_a = [_pareto_sample(pop, random.Random(123)) for _ in range(20)]
    picks_b = [_pareto_sample(pop, random.Random(123)) for _ in range(20)]
    assert picks_a == picks_b


def test_pareto_sample_fallback_to_frontier_members():
    class _EmptyFrontState:
        def get_pareto_front(self):
            return {}

        def get_frontier_members(self):
            return {2, 5}

        def get_best_program(self):  # pragma: no cover - not reached
            return 0

    pick = _pareto_sample(_EmptyFrontState(), random.Random(0))
    assert pick in {2, 5}


def test_pareto_sample_fallback_to_best_program():
    class _NoFrontierState:
        def get_pareto_front(self):
            return {}

        def get_frontier_members(self):
            return set()

        def get_best_program(self):
            return 4

    assert _pareto_sample(_NoFrontierState(), random.Random(0)) == 4


def test_pareto_sample_fallback_to_seed():
    class _BareState:
        pass

    assert _pareto_sample(_BareState(), random.Random(0)) == 0


# --- EVO merge note (driven by ctx.exploit_phase) -----------------------------


def _build_prompts(exploit_phase: bool):
    from src.engine.optimizer.v3.prompts import (
        build_architect_system_prompt,
        build_evo_architect_system_prompt,
        build_freeform_architect_system_prompt,
    )

    ctx = IterationContext(iteration_idx=1, exploit_phase=exploit_phase)
    kw = dict(
        additional_instructions=None,
        program_path=None,
        workspace_path="/workspace",
        context=ctx,
    )
    return (
        build_evo_architect_system_prompt(**kw),
        build_freeform_architect_system_prompt(**kw),
        build_architect_system_prompt(**kw),
    )


MERGE_HEADING = "### Combine Approaches (end of run)"


def test_evo_merge_note_absent_when_not_exploit_phase():
    evo, _, _ = _build_prompts(exploit_phase=False)
    assert MERGE_HEADING not in evo


def test_evo_merge_note_present_in_exploit_phase():
    from src.engine.optimizer.v3.prompts import EVO_MERGE_NOTE

    evo, _, _ = _build_prompts(exploit_phase=True)
    assert EVO_MERGE_NOTE in evo
    assert "keep your changes on the current branch" in evo


def test_merge_note_never_in_freeform_or_ee_prompts():
    # Even with the exploit flag set, only the EVO prompt has a merge slot.
    _, freeform, ee = _build_prompts(exploit_phase=True)
    assert MERGE_HEADING not in freeform
    assert MERGE_HEADING not in ee


# --- exploit-and-merge probability curve --------------------------------------


def test_exploit_probability_curve():
    from src.engine.optimizer.v3.pareto_evolution.select_candidate import (
        _exploit_probability,
    )

    def p(evals_used, max_eval_rows=1000):
        return _exploit_probability(
            IterationContext(
                iteration_idx=1,
                max_eval_rows=max_eval_rows,
                evals_used=evals_used,
            )
        )

    # Spec numbers for k=7, threshold=0.75 (defaults).
    assert abs(p(500) - 0.0293) < 1e-3     # 50% budget -> ~2.9%
    assert abs(p(600) - 0.1091) < 1e-3     # 60% budget -> ~10.9%
    assert abs(p(750) - 0.5) < 1e-9        # at threshold -> 0.5 (the cap)
    # Above the threshold the raw curve exceeds 0.5, so it stays capped.
    assert p(900) == 0.5
    assert p(1000) == 0.5
    # No budget configured -> never exploit.
    assert p(10_000, max_eval_rows=None) == 0.0


# --- exploit-phase parent choice (single shared draw) -------------------------

_SC = "src.engine.optimizer.v3.pareto_evolution.select_candidate"


def test_choose_parent_exploits_best_when_draw_fires():
    # In _make_pop the best program (highest comparison_set_score) is 2.
    pop = _make_pop()
    ctx = IterationContext(iteration_idx=5, max_eval_rows=1000, evals_used=900)
    # Force the probabilistic draw to fire, and prove the Pareto sampler is
    # bypassed by pointing it at a different candidate.
    with patch(f"{_SC}._exploit_probability", return_value=1.0), patch(
        f"{_SC}._pareto_sample", return_value=0
    ):
        assert choose_parent_iteration(pop, ctx, random.Random(9)) == 2
    assert ctx.exploit_phase is True


def test_choose_parent_uses_pareto_sampling_when_draw_misses():
    pop = _make_pop()
    ctx = IterationContext(iteration_idx=5, max_eval_rows=1000, evals_used=900)
    with patch(f"{_SC}._exploit_probability", return_value=0.0), patch(
        f"{_SC}._pareto_sample", return_value=0
    ):
        assert choose_parent_iteration(pop, ctx, random.Random(9)) == 0
    assert ctx.exploit_phase is False


def test_choose_parent_never_exploits_without_budget():
    # No budget -> probability 0.0 -> always Pareto sampling, exploit flag off.
    pop = _make_pop()
    ctx = IterationContext(iteration_idx=5, max_eval_rows=None, evals_used=10_000)
    with patch(f"{_SC}._pareto_sample", return_value=0):
        assert choose_parent_iteration(pop, ctx, random.Random(9)) == 0
    assert ctx.exploit_phase is False


def test_single_draw_couples_merge_note_and_preselect():
    """One draw drives both surfaces: when the exploit draw fires, the
    selection switches to the best candidate AND the EVO prompt built off the
    same ctx surfaces the merge note."""
    from src.engine.optimizer.v3.prompts import build_evo_architect_system_prompt

    pop = _make_pop()
    ctx = IterationContext(iteration_idx=5, max_eval_rows=1000, evals_used=900)
    with patch(f"{_SC}._exploit_probability", return_value=1.0), patch(
        f"{_SC}._pareto_sample", return_value=0
    ):
        assert choose_parent_iteration(pop, ctx, random.Random(9)) == 2

    assert ctx.exploit_phase is True
    evo = build_evo_architect_system_prompt(
        additional_instructions=None,
        program_path=None,
        workspace_path="/workspace",
        context=ctx,
    )
    assert MERGE_HEADING in evo


def test_evo_exploit_threshold_accessor(monkeypatch):
    from src.engine.paths import evo_exploit_threshold

    monkeypatch.delenv("CODEEVOLVER_EVO_EXPLOIT_THRESHOLD", raising=False)
    assert evo_exploit_threshold() == 0.75
    monkeypatch.setenv("CODEEVOLVER_EVO_EXPLOIT_THRESHOLD", "0.9")
    assert evo_exploit_threshold() == 0.9
    for garbage in ("not-a-float", "0", "-0.5", "1.5"):
        monkeypatch.setenv("CODEEVOLVER_EVO_EXPLOIT_THRESHOLD", garbage)
        assert evo_exploit_threshold() == 0.75, garbage


def test_evo_exploit_k_accessor(monkeypatch):
    from src.engine.paths import evo_exploit_k

    monkeypatch.delenv("CODEEVOLVER_EVO_EXPLOIT_K", raising=False)
    assert evo_exploit_k() == 7.0
    monkeypatch.setenv("CODEEVOLVER_EVO_EXPLOIT_K", "12.5")
    assert evo_exploit_k() == 12.5
    for garbage in ("not-a-float", "0", "-3"):
        monkeypatch.setenv("CODEEVOLVER_EVO_EXPLOIT_K", garbage)
        assert evo_exploit_k() == 7.0, garbage


# --- evo_select_parent --------------------------------------------------------


def _services(rng: random.Random):
    from src.engine.optimizer.v3.strategy import LoopServices

    return LoopServices(
        workspace="/tmp/ws",
        evaluator_config=None,  # type: ignore[arg-type]
        trainset_ids=[],
        comparison_ids=[0, 1, 2],
        comparison_timeout=None,
        run_timestamp="20260714000000",
        rng=rng,
    )


def test_evo_select_parent_forks_from_sampled_candidate_sha():
    pop = _make_pop()
    ctx = IterationContext(iteration_idx=3)
    expected = _pareto_sample(pop, random.Random(9))
    with patch(f"{_SC}.Population.load_or_none", return_value=pop), patch(
        f"{_SC}.fork_child_from"
    ) as mock_fork:
        evo_select_parent(state=None, ctx=ctx, services=_services(random.Random(9)))
    cand = pop.get(expected)
    mock_fork.assert_called_once_with(
        ctx=ctx,
        parent_branch=cand["branch"],
        parent_sha=cand["sha"],
        parent_iteration=expected,
        workspace="/tmp/ws",
        run_timestamp="20260714000000",
    )


def test_evo_select_parent_requires_seeded_population():
    import pytest

    ctx = IterationContext(iteration_idx=1)
    with patch(f"{_SC}.Population.load_or_none", return_value=None):
        with pytest.raises(RuntimeError, match="population.json"):
            evo_select_parent(state=None, ctx=ctx, services=_services(random.Random(0)))
