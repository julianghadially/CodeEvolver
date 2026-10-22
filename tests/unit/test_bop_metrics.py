"""Synthetic-fixture tests for scripts/best_of_population/metrics.py
(winners, confirmation isolation, final-as-winner zero contribution,
sums vs means, no systematic selection optimism)."""

from __future__ import annotations

import math
import random

import pytest

from scripts.best_of_population import metrics as M

pytestmark = pytest.mark.unit


def test_classify_and_normalize_binary_with_hover_max():
    kind, m = M.classify_score_type([0.0, 0.998, 0.998, 0.0])
    assert kind == "binary" and m == 0.998
    assert M.normalize_observation(0.998, kind, m) == 1.0
    assert M.normalize_observation(0.0, kind, m) == 0.0
    kind2, m2 = M.classify_score_type([0.0, 0.25, 1.0])
    assert kind2 == "fractional" and m2 == 1.0
    assert M.classify_score_type([0.0, 0.0]) == ("binary", 1.0)


def test_difficult_set_strict_threshold():
    obs = {0: [0, 0, 1, 1], 1: [0, 0, 0, 1], 2: [1, 1, 1], 3: [], 4: [0.4, 0.5]}
    assert M.difficult_set(obs) == [1, 4]  # row 0 is exactly 0.5 -> excluded; row 3 unclassifiable


def test_greedy_top_k_order_and_ties():
    rows = [0, 1, 2, 3]
    scores = {
        "a": {0: 1, 1: 1, 2: 0, 3: 0},   # mean .5
        "b": {0: 0, 1: 0, 2: 1, 3: 1},   # mean .5, tie on mean -> "a" first (ascending tree)
        "c": {0: 1, 1: 0, 2: 1, 3: 0},   # mean .5, adds nothing after a+b
        "d": {0: 0, 1: 0, 2: 0, 3: 1},   # mean .25
    }
    chosen, steps = M.greedy_top_k(scores, ["a", "b", "c", "d"], rows, 3)
    assert chosen == ["a", "b", "c"]
    assert steps[0]["rule"] == "top_mean" and steps[1]["gain"] == 2 and steps[2]["gain"] == 0
    # gains tie (c and d both add 0 after a+b) -> higher overall mean wins -> c
    chosen2, _ = M.greedy_top_k(scores, ["a", "b", "c", "d"], rows, 4)
    assert chosen2 == ["a", "b", "c", "d"]


def test_force_final_replaces_last_added_and_backfill():
    port, replaced = M.force_final(["a", "b", "c"], "z")
    assert port == ["a", "b", "z"] and replaced == "c"
    assert M.force_final(["a", "z"], "z") == (["a", "z"], None)
    scores = {"a": {0: 1}, "p": {0: 0.5}, "q": {0: 0.9}, "r": {0: 0.9}}
    port2, added = M.backfill(["a"], scores, ["p", "q", "r"], [0], 3)
    assert port2 == ["a", "q", "r"] and added == ["q", "r"]  # tie q/r -> ascending tree


def test_row_winners_per_row_max_with_overall_tiebreak():
    rows = [0, 1]
    select = {
        "a": {0: [1, 1, 1, 1, 0], 1: [0, 0, 0, 0, 0]},   # row0 .8, row1 0   overall .4
        "b": {0: [1, 1, 1, 1, 0], 1: [1, 0, 0, 0, 0]},   # row0 .8, row1 .2  overall .5
        "c": {0: [0, 0, 0, 0, 0], 1: [1, 0, 0, 0, 0]},   # row1 .2 ties b -> b (higher overall)
    }
    winners, means, overall = M.row_winners(select, rows)
    assert winners[0]["tree"] == "b" and sorted(winners[0]["tied"]) == ["a", "b"]
    assert winners[1]["tree"] == "b" and sorted(winners[1]["tied"]) == ["b", "c"]
    assert means["a"][0] == pytest.approx(0.8) and overall["b"] == pytest.approx(0.5)


def _fixture():
    rows = [0, 1, 2]
    winners = {0: "w", 1: "f", 2: "w"}          # final "f" wins row 1
    confirm = {0: [1, 0, 1, 1, 0], 1: [1, 1, 0, 1, 1], 2: [0, 0, 1, 0, 0]}
    final = {0: [0, 0, 0, 1, 0], 2: [0, 0, 0, 0, 0]}
    return rows, winners, confirm, final


def test_confirmation_metrics_rounds_sums_and_final_reuse():
    rows, winners, confirm, final = _fixture()
    m = M.confirmation_metrics(rows, winners, confirm, final, "f")
    assert m["B_rounds"] == pytest.approx([2/3, 1/3, 2/3, 2/3, 1/3])
    # F rounds: row1 reuses confirm (winner == final): [1,0,0,1,0]/3 ... etc
    assert m["F_rounds"] == pytest.approx([1/3, 1/3, 0, 2/3, 1/3])
    assert m["D_rounds"] == pytest.approx([1/3, 0, 2/3, 0, 0])
    assert m["B"] == pytest.approx(M.mean(m["B_rounds"]))
    assert m["sum_B"] == pytest.approx(3 * m["B"]) and m["sum_F"] == pytest.approx(3 * m["F"])
    assert m["SE_G"] == pytest.approx(M.sample_sd(m["D_rounds"]) / math.sqrt(5))
    assert m["t_crit"] == 2.776 and m["n_rows_final_is_winner"] == 1
    assert not m["degenerate_zero_variance_B"]
    # row 1 (final is the winner) contributes exactly zero to every D_r: B_r and F_r
    # both read confirm[1][k]; changing final[1] must not matter and is not needed
    m2 = M.confirmation_metrics(rows, winners, confirm, {**final, 1: [0, 0, 0, 0, 0]}, "f")
    assert m2["D_rounds"] == pytest.approx(m["D_rounds"])


def test_confirmation_isolated_from_selection_and_partial_round_raises():
    rows, winners, confirm, final = _fixture()
    sel = {"w": {0: 1.0, 1: 1.0, 2: 1.0}, "f": {0: 0.0, 1: 1.0, 2: 0.0}}
    m = M.confirmation_metrics(rows, winners, confirm, final, "f", sel)
    assert m["selection_to_confirmation_drop"] == pytest.approx(1.0 - m["B"])
    sel2 = {"w": {0: 0.2, 1: 0.2, 2: 0.2}, "f": {0: 0.0, 1: 0.2, 2: 0.0}}
    m2 = M.confirmation_metrics(rows, winners, confirm, final, "f", sel2)
    assert m2["B"] == m["B"] and m2["G"] == m["G"]  # selection scores never enter B/F/G
    with pytest.raises(ValueError):
        M.confirmation_metrics(rows, winners, {**confirm, 2: [0, 0]}, final, "f")
    with pytest.raises(ValueError):
        M.confirmation_metrics(rows, winners, confirm, {0: final[0]}, "f")  # row 2 final missing


def test_zero_variance_flag_and_cross_arm():
    rows = [0, 1]
    m = M.confirmation_metrics(rows, {0: "a", 1: "a"}, {0: [1] * 5, 1: [1] * 5}, {0: [0] * 5, 1: [0] * 5}, "f")
    assert m["degenerate_zero_variance_B"] and m["degenerate_zero_variance_G"] and m["SE_B"] == 0.0
    evo = {"B": 0.6, "SE_B": 0.03, "reps": 5}
    ff = {"B": 0.5, "SE_B": 0.04, "reps": 5}
    x = M.cross_arm(evo, ff)
    assert x["evo_minus_freeform_B"] == pytest.approx(0.1)
    assert x["SE"] == pytest.approx(0.05)


def test_no_systematic_selection_optimism_under_equal_candidates():
    rng = random.Random(7)
    p = [0.3] * 5
    Bs, drops, Gs = [], [], []
    for _ in range(1500):
        m = M.simulate_audit(rng, p, n_rows=20)
        Bs.append(m["B"]); drops.append(m["selection_to_confirmation_drop"]); Gs.append(m["G"])
    # confirmation is unbiased for the frozen winners' true (equal) probability ...
    assert abs(M.mean(Bs) - 0.3) < 0.01
    assert abs(M.mean(Gs)) < 0.01
    # ... while the selection means themselves are optimistic (winner's-curse diagnostic > 0)
    assert M.mean(drops) > 0.1


def test_simulate_audit_prefers_the_truly_better_candidate():
    rng = random.Random(3)
    m = M.simulate_audit(rng, [0.2, 0.9, 0.2], n_rows=40, final_index=0)
    assert m["true_B"] > 0.8 and m["B"] > 0.7 and m["G"] > 0.4
