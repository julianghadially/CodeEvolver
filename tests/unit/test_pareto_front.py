"""Pure instance-front computation + GEPA Algorithm 2 dominated pruning
(`v3/pareto_front.py`), shared by pareto_evolution and ASA."""

from src.engine.optimizer.v3.pareto_front import (
    pareto_counts,
    pareto_front,
    prune_dominated,
    winning_rows,
)


def test_front_ties_credit_every_winner():
    front = pareto_front({"x": {"a": 1.0, "b": 0.5}, "y": {"a": 1.0, "b": 0.7}})
    assert front == {"a": {"x", "y"}, "b": {"y"}}


def test_none_scores_and_unscored_rows_are_ignored():
    front = pareto_front({"x": {"a": None, "b": 0.5}, "y": {"a": 0.2}})
    assert front == {"a": {"y"}, "b": {"x"}}


def _check_front_invariants(original, pruned):
    # Every non-empty row keeps a winner, and no winner is invented.
    assert set(pruned) == {r for r, w in original.items() if w}
    for rid, winners in pruned.items():
        assert winners and winners <= original[rid]


def test_prune_removes_strict_subsets():
    front = {"a": {"x", "y"}, "b": {"y"}, "c": {"y", "z"}}
    pruned, dropped = prune_dominated(front)
    # x wins {a}; y wins {a,b,c}; z wins {c}. {a} < {a,b,c} and {c} < {a,b,c}.
    assert dropped == {"x", "z"}
    assert pruned == {"a": {"y"}, "b": {"y"}, "c": {"y"}}
    _check_front_invariants(front, pruned)


def test_prune_drops_lower_scoring_twin_and_keeps_complementary_winners():
    # x and z have identical winning sets: GEPA removes one of them — the
    # lower aggregate scorer, or on a score tie the lower key.
    front = {"a": {"x", "z"}, "b": {"y"}}
    pruned, dropped = prune_dominated(front, scores={"x": 0.9, "z": 0.2, "y": 0.5})
    assert dropped == {"z"}
    assert pruned == {"a": {"x"}, "b": {"y"}}
    pruned, dropped = prune_dominated(front)  # no scores -> tie -> lower key
    assert dropped == {"x"}
    assert pruned == {"a": {"z"}, "b": {"y"}}
    _check_front_invariants(front, pruned)
    # Complementary winners are never dominated.
    front = {"a": {"x"}, "b": {"y"}}
    assert prune_dominated(front) == (front, set())


def test_prune_removes_union_covered_candidate():
    # Incumbent ties A on rows 1-5 and B on rows 6-10; A wins 11, B wins 12.
    # No single candidate covers the incumbent (the old strict-subset rule
    # kept it) but A ∪ B does -> GEPA prunes it, even as the top scorer.
    front = {f"r{i:02d}": {"inc", "A"} for i in range(1, 6)}
    front |= {f"r{i:02d}": {"inc", "B"} for i in range(6, 11)}
    front |= {"r11": {"A"}, "r12": {"B"}}
    pruned, dropped = prune_dominated(front, scores={"inc": 0.9, "A": 0.4, "B": 0.4})
    assert dropped == {"inc"}
    assert dict(pareto_counts(pruned)) == {"A": 6, "B": 6}
    _check_front_invariants(front, pruned)
    # A unique row anywhere rescues it.
    front["r13"] = {"inc"}
    pruned, dropped = prune_dominated(front, scores={"inc": 0.9, "A": 0.4, "B": 0.4})
    assert dropped == set()
    assert dict(pareto_counts(pruned)) == {"inc": 11, "A": 6, "B": 6}


def test_prune_order_is_lowest_score_first():
    # P wins {x, y}; Q wins {x}; R wins {y}. Which side survives depends on
    # who is examined (and removed) first: the lowest aggregate scorer.
    front = {"x": {"P", "Q"}, "y": {"P", "R"}}
    pruned, dropped = prune_dominated(front, scores={"P": 0.9, "Q": 0.5, "R": 0.5})
    assert dropped == {"Q", "R"} and pruned == {"x": {"P"}, "y": {"P"}}
    pruned, dropped = prune_dominated(front, scores={"P": 0.1, "Q": 0.9, "R": 0.9})
    assert dropped == {"P"} and pruned == {"x": {"Q"}, "y": {"R"}}
    _check_front_invariants(front, pruned)


def test_prune_reports_candidates_on_no_row_when_keys_given():
    front = {"a": {"x"}}
    pruned, dropped = prune_dominated(front, keys={"x", "nobody"})
    assert dropped == {"nobody"}
    assert pruned == {"a": {"x"}}
    # ... but a lone candidate on no row is not dominated by anyone.
    _, dropped2 = prune_dominated({}, keys={"nobody"})
    assert dropped2 == set()


def test_counts_and_winning_rows():
    front = {"a": {"x", "y"}, "b": {"y"}}
    assert dict(pareto_counts(front)) == {"x": 1, "y": 2}
    assert winning_rows(front) == {"x": frozenset({"a"}), "y": frozenset({"a", "b"})}
