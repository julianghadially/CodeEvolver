"""Copyright © 2026 440 Labs LLC

Unit tests for the loop-owned pareto_evolution population
(`v3/pareto_evolution/population.py`) and the admission rule
(`pareto_optimizer.admission_exclusion` / `train_progressed`).
"""

from __future__ import annotations

import pytest

from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.pareto_evolution.pareto_optimizer import (
    ADMISSION_REASON_BY_STATE,
    admission_exclusion,
    train_progressed,
)
from src.engine.optimizer.v3.pareto_evolution.population import Population


def _cand(it, rows, branch=None, score=None):
    return {
        "iteration": it,
        "branch": branch or f"b{it}",
        "sha": f"sha{it}",
        "tree": None,
        "parent_iteration": None if it == 0 else 0,
        "row_scores": rows,
        "comparison_set_score": (
            score if score is not None else sum(rows.values()) / len(rows)
        ),
        "train_baseline": None,
        "train_best": None,
    }


# --- frontier ----------------------------------------------------------------


def test_front_prunes_strictly_dominated_ties_and_best_is_max_score():
    # GEPA Algorithm 2: ties credit every winner, THEN any candidate with no
    # row it wins uniquely among the survivors is pruned (lowest score
    # first). Candidate 2 wins {a, b, c}; 1 wins {a} and 0 wins {c} -> both
    # dominated and off the front.
    pop = Population()
    pop.add(_cand(0, {"a": 0.5, "b": 0.5, "c": 0.5}))
    pop.add(_cand(1, {"a": 0.9, "b": 0.5, "c": 0.4}))
    pop.add(_cand(2, {"a": 0.9, "b": 0.6, "c": 0.5}))
    assert pop.get_pareto_front() == {"a": {2}, "b": {2}, "c": {2}}
    assert pop.get_frontier_members() == {2}
    assert pop.get_best_program() == 2
    assert pop.best()["branch"] == "b2"
    assert not pop.is_on_frontier(1)


def test_front_prunes_one_of_two_equal_winning_sets_and_keeps_complementary_winners():
    pop = Population()
    pop.add(_cand(0, {"a": 0.9, "b": 0.1}))
    pop.add(_cand(1, {"a": 0.1, "b": 0.9}))
    pop.add(_cand(2, {"a": 0.9, "b": 0.1}))  # identical winning set + score to 0
    # GEPA removes the lower scorer of two twins; on a tie the lower key (0).
    assert pop.get_pareto_front() == {"a": {2}, "b": {1}}
    assert pop.get_frontier_members() == {1, 2}
    pop2 = Population()
    pop2.add(_cand(0, {"a": 0.9, "b": 0.1}, score=0.8))
    pop2.add(_cand(1, {"a": 0.1, "b": 0.9}))
    pop2.add(_cand(2, {"a": 0.9, "b": 0.1}, score=0.3))  # lower score -> pruned
    assert pop2.get_pareto_front() == {"a": {0}, "b": {1}}


def _fifteen_row_pop(incumbent_unique_row=False):
    """Incumbent (0) ties A (1) on r01-r05 and B (2) on r06-r10; A wins r11,
    B wins r12; r13-r15 are unsolved by everyone (three-way 0.0 ties)."""
    rows = [f"r{i:02d}" for i in range(1, 16)]
    inc = {r: (1.0 if i < 10 else 0.0) for i, r in enumerate(rows)}
    a = {r: (1.0 if i < 5 or r == "r11" else 0.0) for i, r in enumerate(rows)}
    b = {r: (1.0 if 5 <= i < 10 or r == "r12" else 0.0) for i, r in enumerate(rows)}
    if incumbent_unique_row:
        inc["r13"] = 1.0
    pop = Population()
    pop.add(_cand(0, inc))
    pop.add(_cand(1, a))
    pop.add(_cand(2, b))
    return pop


def test_union_covered_incumbent_is_pruned_15_rows():
    pop = _fifteen_row_pop()
    # The incumbent has the highest comparison score (10/15) but every row it
    # wins is also won by A or by B -> GEPA prunes it. (A strict-subset rule
    # would keep it: its winning set is not a subset of A's or of B's alone.)
    front = pop.get_pareto_front()
    assert pop.get_frontier_members() == {1, 2}
    assert not any(0 in winners for winners in front.values())
    weights = {}
    for winners in front.values():
        for it in winners:
            weights[it] = weights.get(it, 0) + 1
    assert weights == {1: 9, 2: 9}  # 5 shared + 1 unique + 3 all-zero ties each
    assert not pop.is_on_frontier(0)
    # The exploit draw is untouched by pruning: best is still by aggregate score.
    assert pop.get_best_program() == 0
    assert [c["on_frontier"] for c in pop.summary()] == [False, True, True]


def test_incumbent_survives_with_a_unique_win_15_rows():
    pop = _fifteen_row_pop(incumbent_unique_row=True)
    front = pop.get_pareto_front()
    assert pop.get_frontier_members() == {0, 1, 2}
    assert front["r13"] == {0}
    assert sum(1 for w in front.values() if 0 in w) == 13  # r01-r10, r13, r14, r15


def test_dominated_candidate_is_off_frontier():
    pop = Population()
    pop.add(_cand(0, {"a": 0.5, "b": 0.5}))
    pop.add(_cand(1, {"a": 0.1, "b": 0.1}))
    assert not pop.is_on_frontier(1)
    assert pop.get_frontier_members() == {0}


def test_best_tie_breaks_to_lowest_iteration():
    pop = Population()
    pop.add(_cand(0, {"a": 0.5}, score=0.7))
    pop.add(_cand(3, {"a": 0.9}, score=0.7))
    assert pop.get_best_program() == 0


def test_add_rejects_duplicate_iteration():
    pop = Population()
    pop.add(_cand(0, {"a": 0.5}))
    with pytest.raises(ValueError):
        pop.add(_cand(0, {"a": 0.6}))


def test_empty_population_has_no_best():
    with pytest.raises(ValueError):
        Population().get_best_program()


# --- persistence -------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "population.json")
    pop = Population()
    pop.add(_cand(0, {"a": 0.5, "b": 0.5}))
    pop.add(_cand(4, {"a": 0.9, "b": 0.2}))
    pop.save(path)
    loaded = Population.load_or_none(path)
    assert loaded is not None
    assert sorted(loaded.candidates) == [0, 4]
    assert loaded.get(4)["row_scores"] == {"a": 0.9, "b": 0.2}
    assert loaded.get_pareto_front() == pop.get_pareto_front()
    assert "admitted_at" in loaded.get(0)


def test_load_or_none_when_missing(tmp_path):
    assert Population.load_or_none(str(tmp_path / "nope.json")) is None


def test_default_path_lives_under_history_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEEVOLVER_HISTORY_DIR", str(tmp_path))
    assert Population.path().startswith(str(tmp_path))
    assert Population.path().endswith("population.json")


# --- admission ---------------------------------------------------------------


def _ctx(**kw) -> IterationContext:
    ctx = IterationContext(iteration_idx=2, parent_locked=True)
    ctx.changes_landed = True
    ctx.initial_subsample_score = 0.5
    ctx.best_subsample_score = 0.6
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


@pytest.mark.parametrize(
    "ctx_kw, rec, expected",
    [
        ({}, {}, None),
        ({"skipped": True}, {}, "skipped"),
        ({}, {"skipped": True}, "skipped"),
        ({}, {"constraint_violation": "no hardcoding"}, "constraint_violation"),
        ({"changes_landed": False}, {}, "no_change"),
        # Train evidence is NOT an exclusion: a tie, a regression
        # or a missing train reading leaves the content admissible — admission
        # is decided by the existence of a comparison eval.
        ({"best_subsample_score": None}, {}, None),
        ({"initial_subsample_score": None}, {}, None),
        ({"best_subsample_score": 0.5}, {}, None),   # tie
        ({"best_subsample_score": 0.4}, {}, None),   # regression
    ],
)
def test_admission_exclusion_table(ctx_kw, rec, expected):
    assert admission_exclusion(_ctx(**ctx_kw), rec) == expected


@pytest.mark.parametrize(
    "ctx_kw, expected",
    [
        ({}, True),                                   # 0.6 > 0.5
        ({"best_subsample_score": 0.5}, False),       # tie is not progress
        ({"best_subsample_score": 0.4}, False),
        ({"best_subsample_score": None}, False),
        ({"initial_subsample_score": None}, False),
    ],
)
def test_train_progressed_is_the_loop_fallback_predicate(ctx_kw, expected):
    assert train_progressed(_ctx(**ctx_kw)) is expected


def test_admission_reason_by_state_covers_every_comparison_state():
    assert set(ADMISSION_REASON_BY_STATE) == {
        "EXACT", "PROGRAM_EQUIVALENT", "REVERTED_TO_COMPARISON_POINT", "RERUN_BY_LOOP",
    }
    assert ADMISSION_REASON_BY_STATE["EXACT"] == ADMISSION_REASON_BY_STATE["PROGRAM_EQUIVALENT"] == "agent_comparison"
