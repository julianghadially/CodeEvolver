"""Phase-keyed store + planners: resuming never double-counts, fills only
replace failed rows, planners return only what is missing."""

from __future__ import annotations

import pytest

from scripts.best_of_population import planner as P
from scripts.best_of_population.store import AuditStore, row_line

pytestmark = pytest.mark.unit


def _row(tree: str, phase: str, rid: int, draw: int, score: float, failed: bool = False) -> dict:
    return row_line(bench="b", job_id="j", arm="freeform", tree=tree, sha=tree * 5, split="test",
                    row_id=rid, draw=draw, score=score, failed=failed, purpose=phase, eval_id="e",
                    testset_sha256="t", trainset_sha256="tr")


def _cand(tree: str, **kw) -> dict:
    return {"job_id": "j", "arm": "freeform", "sha": tree * 5, "tree": tree, "reqhash": "r", **kw}


def test_phase_keys_are_distinct_and_appends_are_idempotent(tmp_path):
    st = AuditStore(tmp_path / "rows.jsonl", tmp_path / "evals.jsonl")
    rows = [_row("aaaaaaaa", "screen", 0, 0, 1.0), _row("aaaaaaaa", "select", 0, 0, 0.0)]
    assert st.append_eval({"eval_id": "e1"}, rows) == 2
    assert st.append_eval({"eval_id": "e1"}, rows) == 0
    assert st.have("j", "aaaaaaaa", "screen", 0, 0) and st.have("j", "aaaaaaaa", "select", 0, 0)
    assert st.scores("j", "aaaaaaaa", "screen", 0) == {0: 1.0}
    st2 = AuditStore(tmp_path / "rows.jsonl", tmp_path / "evals.jsonl")  # reload
    assert len(st2.rows()) == 2 and len(st2.evals()) == 2


def test_fill_overrides_only_failed_rows(tmp_path):
    st = AuditStore(tmp_path / "rows.jsonl", tmp_path / "evals.jsonl")
    st.append_eval({"eval_id": "e1"}, [_row("a" * 8, "screen", 0, 0, 0.0, failed=True),
                                        _row("a" * 8, "screen", 1, 0, 1.0)])
    assert st.failed_rows("j", "a" * 8, "screen", 0) == [0]
    n = st.append_fill({"eval_id": "fill"}, [_row("a" * 8, "screen", 0, 0, 1.0),
                                              _row("a" * 8, "screen", 1, 0, 0.0)])
    assert n == 1
    assert st.scores("j", "a" * 8, "screen", 0) == {0: 1.0, 1: 1.0}
    assert st.record("j", "a" * 8, "screen", 0, 0)["fill"] is True
    st2 = AuditStore(tmp_path / "rows.jsonl", tmp_path / "evals.jsonl")
    assert st2.scores("j", "a" * 8, "screen", 0) == {0: 1.0, 1: 1.0}  # later line wins on reload


def test_planners_return_only_missing_rows(tmp_path):
    st = AuditStore(tmp_path / "rows.jsonl", tmp_path / "evals.jsonl")
    H = [0, 1, 2]
    a, b = _cand("a" * 8), _cand("b" * 8, unstable=True)
    tasks = P.plan_screen([a, b], H, st)
    assert [t.tree for t in tasks] == ["a" * 8] and tasks[0].draw == 0 and tasks[0].purpose == "screen"
    st.append_eval({"eval_id": "e"}, [_row("a" * 8, "screen", 0, 0, 1.0), _row("a" * 8, "screen", 1, 0, 1.0)])
    tasks = P.plan_screen([a], H, st)
    assert tasks[0].row_ids == [2]
    # select draws 1..5; a partial draw 1 is resumed, not restarted
    st.append_eval({"eval_id": "e"}, [_row("a" * 8, "select", r, 1, 1.0) for r in (0, 1)])
    tasks = P.plan_select([a], H, st, 5)
    assert [(t.draw, t.row_ids) for t in tasks][:2] == [(1, [2]), (2, [0, 1, 2])]
    assert len(tasks) == 5 and P.count_rows(tasks) == {"calls": 5, "rows": 13}
    # confirm/final share draws 6..10; final excludes the rows it won
    port = {"a" * 8: a, "f" * 8: _cand("f" * 8, is_final=True)}
    ctasks = P.plan_confirm(port, {"a" * 8: [0, 2], "f" * 8: [1]}, st, 5, 5)
    assert sorted({t.draw for t in ctasks}) == [6, 7, 8, 9, 10] and len(ctasks) == 10
    assert {t.purpose for t in ctasks} == {"confirm"}
    ftasks = P.plan_final(port["f" * 8], H, [1], st, 5, 5)
    assert all(t.row_ids == [0, 2] and t.purpose == "final" for t in ftasks) and len(ftasks) == 5
    assert P.plan_final(port["f" * 8], H, H, st, 5, 5) == []
    assert P.phase_matrix(st, "j", "a" * 8, "select", H, [1, 2]) == {0: [1.0], 1: [1.0]}
    assert ftasks[0].suffix == "holdout_final_ffffffff_d6"
