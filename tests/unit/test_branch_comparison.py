"""ASA lineage representation, contribution, eligibility and the
`branch_comparison` table (`v3/asa/branch_comparison.py`)."""

import json

import pytest

from src.engine.optimizer.v3.asa.branch_comparison import (
    build_table,
    contribution_threshold,
    eligible_set,
    format_table,
    lineage_record,
    load_comparison_records,
    positive_contribution,
)


def _rec(branch, rows, *, tree=None, iteration=1, score=None):
    return {
        "iteration": iteration, "branch": branch, "sha": f"sha-{branch}-{iteration}",
        "tree": tree, "row_scores": rows,
        "score": score if score is not None else sum(rows.values()) / len(rows),
    }


# --- contribution / threshold --------------------------------------------------

def test_positive_contribution_sums_differences_not_row_counts():
    c = positive_contribution({"1": 0.9, "2": 0.2, "3": 0.5}, {"1": 0.4, "2": 0.6, "3": 0.5})
    assert c.positive_behavioral_contribution == pytest.approx(0.5)
    assert c.n_rows_ahead == 1 and c.n_rows_compared == 3 and c.n_rows_missing == 0
    # graded scores: two small wins add up to more than one big loss ignored
    c = positive_contribution({"1": 0.6, "2": 0.6, "3": 0.0}, {"1": 0.5, "2": 0.5, "3": 1.0})
    assert c.positive_behavioral_contribution == pytest.approx(0.2) and c.n_rows_ahead == 2


def test_positive_contribution_missing_rows_count_zero_and_are_reported():
    c = positive_contribution({"1": 1.0, "9": 1.0}, {"1": 0.0, "2": 0.0})
    assert c.positive_behavioral_contribution == 1.0 and c.n_rows_missing == 2


def test_threshold_examples_from_spec():
    kw = dict(max_score=1.0, constant=0.10, cap=0.05)
    assert contribution_threshold(primary_comparison_score=0.96, **kw) == pytest.approx(0.004)
    assert contribution_threshold(primary_comparison_score=0.70, **kw) == pytest.approx(0.03)
    assert contribution_threshold(primary_comparison_score=0.30, **kw) == pytest.approx(0.05)
    assert contribution_threshold(primary_comparison_score=0.90, **kw) == pytest.approx(0.01)
    assert contribution_threshold(primary_comparison_score=None, **kw) == pytest.approx(0.05)


# --- lineage record ----------------------------------------------------------

def test_lineage_record_prefers_tree_match_then_branch_fallback():
    recs = [_rec("a", {"1": 1.0}, tree="t1", iteration=1), _rec("a", {"1": 0.0}, tree="t2", iteration=2),
            _rec("b", {"1": 0.5}, tree="t3", iteration=3)]
    never = lambda x, y: False
    exact = lineage_record("a", recs, tip_tree="t1", equivalent=never)
    assert exact.up_to_date and exact.record["iteration"] == 1
    # tip moved past every record -> newest on the branch, flagged stale
    stale = lineage_record("a", recs, tip_tree="t9", equivalent=never)
    assert not stale.up_to_date and stale.record["iteration"] == 2
    # program-equivalent tree on ANOTHER branch represents this branch too (seed twins)
    twin = lineage_record("zzz", recs, tip_tree="t3x", equivalent=lambda x, y: {x, y} == {"t3", "t3x"})
    assert twin.up_to_date and twin.record["branch"] == "b"
    assert lineage_record("nope", recs, tip_tree=None, equivalent=never) is None


def test_load_records_tolerates_bad_and_partial_lines(tmp_path):
    p = tmp_path / "cmp.jsonl"
    p.write_text(json.dumps(_rec("a", {"1": 1.0})) + "\nnot json\n\n" + json.dumps({"branch": "b"}) + "\n")
    recs = load_comparison_records(str(p))
    assert [r.get("branch") for r in recs] == ["a", "b"]
    assert load_comparison_records(str(tmp_path / "missing.jsonl")) == []


# --- table / eligibility ------------------------------------------------------

def _lineages():
    from src.engine.optimizer.v3.asa.branch_comparison import LineageRecord
    primary = _rec("codeevolver-main", {"1": 1.0, "2": 1.0, "3": 0.0, "4": 0.0}, tree="tp", score=0.5)
    other = _rec("codeevolver-react", {"1": 0.0, "2": 1.0, "3": 1.0, "4": 0.0}, tree="to", score=0.5)
    worse = _rec("codeevolver-one", {"1": 0.0, "2": 1.0, "3": 0.0, "4": 0.0}, tree="tw", score=0.25)
    return {
        "codeevolver-main": LineageRecord(primary, True),
        "codeevolver-react": LineageRecord(other, True),
        "codeevolver-one": LineageRecord(worse, False),
        "codeevolver-new": None,
    }


def test_table_rows_eligibility_and_pruned_pareto():
    t = build_table(lineages=_lineages(), reference="codeevolver-main", primary="codeevolver-main",
                    comparison_set_size=4, threshold=0.05)
    by = {r.branch: r for r in t.rows}
    assert t.rows[0].is_reference and t.rows[0].branch == "codeevolver-main"
    react = by["codeevolver-react"]
    assert react.positive_behavioral_contribution == 1.0 and react.n_rows_ahead == 1
    assert react.positive_behavioral_contribution_normalized == 0.25 and react.eligible
    assert not by["codeevolver-main"].eligible
    # Row 4 ties at 0.0 for everyone (ties credit every winner). `one` wins
    # {2, 4}, both also won by survivors (GEPA union cover) -> dominated;
    # main is unique on 1 and react on 3, so both stay.
    assert by["codeevolver-one"].pareto_dominated and by["codeevolver-one"].pareto_score is None
    assert not by["codeevolver-one"].eligible  # 0 contribution
    assert by["codeevolver-main"].pareto_score == 3 and react.pareto_score == 3
    assert t.missing == ["codeevolver-new"]
    assert t.eligible == {"codeevolver-react": 0.25}


def test_twin_lineages_sharing_a_record_are_one_pareto_candidate():
    # Seed twins: main and seed resolve to the SAME record (same tree). GEPA's
    # pruning would drop one identical twin; the table treats them as one
    # candidate and both carry the representative's score, undominated.
    from src.engine.optimizer.v3.asa.branch_comparison import LineageRecord
    shared = _rec("codeevolver-main", {"1": 1.0, "2": 1.0, "3": 0.0}, tree="tp", score=0.67)
    other = _rec("codeevolver-react", {"1": 0.0, "2": 1.0, "3": 1.0}, tree="to", score=0.67)
    lin = {
        "codeevolver-main": LineageRecord(shared, True),
        "codeevolver-seed": LineageRecord(shared, True),
        "codeevolver-react": LineageRecord(other, True),
    }
    t = build_table(lineages=lin, reference="codeevolver-main", primary="codeevolver-main",
                    comparison_set_size=3, threshold=0.05)
    by = {r.branch: r for r in t.rows}
    assert not by["codeevolver-main"].pareto_dominated and by["codeevolver-main"].pareto_score == 2
    assert not by["codeevolver-seed"].pareto_dominated and by["codeevolver-seed"].pareto_score == 2
    assert not by["codeevolver-react"].pareto_dominated and by["codeevolver-react"].pareto_score == 2


def test_eligibility_is_always_vs_primary_even_with_other_reference():
    t = build_table(lineages=_lineages(), reference="codeevolver-react", primary="codeevolver-main",
                    comparison_set_size=4, threshold=0.05)
    assert all(not r.eligible for r in t.rows)  # not computed vs primary here
    assert t.rows[0].branch == "codeevolver-react" and t.rows[0].is_reference


def test_eligible_set_is_empty_without_primary_record_and_reference_error():
    lin = _lineages()
    lin["codeevolver-main"] = None
    assert eligible_set(lineages=lin, primary="codeevolver-main", comparison_set_size=4, threshold=0.05) == {}
    with pytest.raises(ValueError, match="no comparison-set record"):
        build_table(lineages=lin, reference="codeevolver-main", primary="codeevolver-main",
                    comparison_set_size=4, threshold=0.05)


def test_format_table_mentions_dominated_stale_and_missing():
    t = build_table(lineages=_lineages(), reference="codeevolver-main", primary="codeevolver-main",
                    comparison_set_size=4, threshold=0.05)
    text = format_table(t)
    assert "n/a - fully dominated by other programs" in text
    assert "`codeevolver-main` (reference)" in text
    assert "Out-of-date comparison records" in text and "`codeevolver-one`" in text
    assert "Not compared" in text and "`codeevolver-new`" in text
    assert "| true | true | false |" in text  # react: up_to_date, eligible, not primary
