"""ASA rule engine (`v3/asa/rules.py`, specs/archive-public/v3_asa.md §Rule set / §Window)."""

import pytest

from src.engine.optimizer.v3.asa.prompt_sections import (
    REQ_ELIGIBLE_NONPRIMARY,
    REQ_NEW_LINEAGE,
)
from src.engine.optimizer.v3.asa.rules import (
    allowance,
    is_free_tail,
    is_phase1,
    plan_iteration,
    primary_record_is_stale,
)


def _rec(i, *, new_lineage=False, eligible_at_start=(), parent_eligible=False,
         violation=False, skipped=False, branch="codeevolver-x", landed=True):
    return {
        "iteration": i,
        "skipped": skipped,
        "new_lineage": new_lineage,
        "asa_eligible_at_start": list(eligible_at_start),
        "asa_parent_is_eligible": parent_eligible,
        "asa_violation": violation,
        "branch": branch,
        "changes_landed": landed,
    }


def _plan(records, *, eligible=(), used=0, budget=1000, sub=10):
    return plan_iteration(records, eligible=list(eligible), evals_used=used,
                          max_eval_rows=budget, subsample_size=sub)


# --- rule 1 -----------------------------------------------------------------

def test_rule1_silent_before_three_closed():
    recs = [_rec(1), _rec(2)]
    assert _plan(recs).requirements == []


def test_rule1_fires_at_iteration_4_when_rate_below_40pct():
    recs = [_rec(1), _rec(2), _rec(3)]                  # 0/3
    assert _plan(recs).requirements == [REQ_NEW_LINEAGE]
    recs = [_rec(1, new_lineage=True), _rec(2), _rec(3)]  # 1/3 = 33% < 40%
    assert _plan(recs).requirements == [REQ_NEW_LINEAGE]
    recs = [_rec(1, new_lineage=True), _rec(2, new_lineage=True), _rec(3)]  # 66%
    assert _plan(recs).requirements == []


def test_rule1_pacing_over_ten_iterations_needs_four_lineages():
    # 4/10 = 40% is NOT below the floor; 3/10 is.
    recs = [_rec(i, new_lineage=(i <= 4)) for i in range(1, 11)]
    assert REQ_NEW_LINEAGE not in _plan(recs).requirements
    recs = [_rec(i, new_lineage=(i <= 3)) for i in range(1, 11)]
    assert REQ_NEW_LINEAGE in _plan(recs).requirements


def test_rule1_off_after_phase1():
    recs = [_rec(i) for i in range(1, 11)]  # 0% new lineages
    # budget past half AND >= 10 closed -> phase 1 over
    p = _plan(recs, used=600, budget=1000)
    assert not p.phase1 and p.requirements == []
    # only one of the two conditions -> still phase 1
    assert _plan(recs, used=400, budget=1000).phase1
    assert _plan([_rec(i) for i in range(1, 9)], used=600, budget=1000).phase1


def test_skipped_stubs_do_not_count_but_reverted_iterations_do():
    recs = [_rec(1, skipped=True), _rec(2, skipped=True), _rec(3, skipped=True), _rec(4)]
    assert _plan(recs).n_closed == 1
    assert _plan(recs).requirements == []
    recs = [_rec(1, landed=False), _rec(2, landed=False), _rec(3, landed=False)]
    assert _plan(recs).requirements == [REQ_NEW_LINEAGE]


# --- rule 2 -----------------------------------------------------------------

def test_rule2_needs_eligible_set_and_two_eligible_iterations():
    recs = [_rec(1), _rec(2), _rec(3, new_lineage=True), _rec(4, new_lineage=True)]
    # no eligible lineage -> inert regardless of history
    assert REQ_ELIGIBLE_NONPRIMARY not in _plan(recs, eligible=()).requirements
    # eligible now, but only one past iteration started with an eligible set
    recs[3]["asa_eligible_at_start"] = ["codeevolver-b"]
    assert REQ_ELIGIBLE_NONPRIMARY not in _plan(recs, eligible=["codeevolver-b"]).requirements
    # two eligible iterations, 0/2 satisfied -> fires
    recs[2]["asa_eligible_at_start"] = ["codeevolver-b"]
    p = _plan(recs, eligible=["codeevolver-b"])
    assert REQ_ELIGIBLE_NONPRIMARY in p.requirements and p.n_eligible_window == 2


def test_rule2_rate_at_30pct_still_fires_and_above_does_not():
    base = [_rec(1, new_lineage=True), _rec(2, new_lineage=True)]
    window = [_rec(i, new_lineage=True, eligible_at_start=["b"], parent_eligible=(i <= 5))
              for i in range(3, 13)]  # 3/10 = 30% -> "<= 30%" fires
    assert REQ_ELIGIBLE_NONPRIMARY in _plan(base + window, eligible=["b"]).requirements
    window[5]["asa_parent_is_eligible"] = True  # 4/10
    assert REQ_ELIGIBLE_NONPRIMARY not in _plan(base + window, eligible=["b"]).requirements


def test_window_pauses_and_resumes_without_reset():
    recs = [_rec(1, new_lineage=True), _rec(2, new_lineage=True),
            _rec(3, new_lineage=True, eligible_at_start=["b"]),
            _rec(4, new_lineage=True),                       # eligibility lapsed
            _rec(5, new_lineage=True, eligible_at_start=["b"])]
    p = _plan(recs, eligible=["b"])
    assert p.n_eligible_window == 2 and REQ_ELIGIBLE_NONPRIMARY in p.requirements


def test_both_rules_can_fire_together():
    recs = [_rec(1, eligible_at_start=["b"]), _rec(2, eligible_at_start=["b"]), _rec(3)]
    assert _plan(recs, eligible=["b"]).requirements == [REQ_NEW_LINEAGE, REQ_ELIGIBLE_NONPRIMARY]


# --- free tail / allowance ---------------------------------------------------

def test_free_tail_thresholds():
    # max(2*5*10=100, 0.1*1000=100) -> 100 rows remaining
    assert not is_free_tail(evals_used=899, max_eval_rows=1000, subsample_size=10)
    assert is_free_tail(evals_used=900, max_eval_rows=1000, subsample_size=10)
    # bigger subsample dominates the 10%
    assert is_free_tail(evals_used=700, max_eval_rows=1000, subsample_size=30)
    assert not is_free_tail(evals_used=700, max_eval_rows=None, subsample_size=30)


def test_free_tail_silences_every_rule():
    recs = [_rec(1, eligible_at_start=["b"]), _rec(2, eligible_at_start=["b"]), _rec(3)]
    p = _plan(recs, eligible=["b"], used=950, budget=1000)
    assert p.free_tail and p.requirements == []


def test_allowance_is_ceil_closed_over_ten_minus_violations():
    assert allowance(0) == 0 and allowance(1) == 1 and allowance(10) == 1
    assert allowance(11) == 2 and allowance(20) == 2 and allowance(21) == 3
    recs = [_rec(i, violation=(i == 2)) for i in range(1, 6)]
    assert _plan(recs).allowance_left == 0          # ceil(5/10)=1 - 1
    recs = [_rec(i, violation=(i == 2)) for i in range(1, 12)]
    assert _plan(recs).allowance_left == 1          # ceil(11/10)=2 - 1


def test_phase1_without_budget_never_ends():
    assert is_phase1(n_closed=50, evals_used=0, max_eval_rows=None)


# --- stale primary -----------------------------------------------------------

def test_primary_record_stale_counts_landed_iterations_on_primary_only():
    recs = [_rec(i, branch="codeevolver-main", landed=True) for i in range(1, 8)]
    recs.append(_rec(8, branch="codeevolver-other", landed=True))
    recs.append(_rec(9, branch="codeevolver-main", landed=False))
    assert not primary_record_is_stale(recs, primary="codeevolver-main", record_iteration=3, stale_after_landed=4)
    assert primary_record_is_stale(recs, primary="codeevolver-main", record_iteration=2, stale_after_landed=4)
    assert not primary_record_is_stale(recs, primary="codeevolver-main", record_iteration=None, stale_after_landed=4)


def test_scaling_sanity_15_20_100():
    for n_total, budget in ((15, 750), (20, 1000), (100, 5000)):
        recs = [_rec(i, new_lineage=(i % 3 == 0)) for i in range(1, n_total + 1)]
        p = _plan(recs, used=budget - 1, budget=budget, sub=10)
        assert p.free_tail  # last rows are always free
        p = _plan(recs[: n_total // 2], used=budget // 2 - 1, budget=budget, sub=10)
        assert p.phase1 or n_total // 2 >= 10
