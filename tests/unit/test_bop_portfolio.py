"""Stage 2b/3/5 over a synthetic store: portfolio pruning + greedy + final
forcing + backfill + unstable exclusion; winners; results + summary."""

from __future__ import annotations

import json

import pytest

from scripts.best_of_population import planner as P
from scripts.best_of_population.config import AuditConfig
from scripts.best_of_population.portfolio import build_portfolio
from scripts.best_of_population.report import build_results, summary_rows, write_results, write_summary
from scripts.best_of_population.store import AuditStore, row_line
from scripts.best_of_population.winners import build_winners

pytestmark = pytest.mark.unit

H = [0, 1, 2, 3, 4, 5]


def _store(tmp_path) -> AuditStore:
    return AuditStore(tmp_path / "rows.jsonl", tmp_path / "evals.jsonl")


def _put(store, tree, phase, draw, scores: dict[int, float], failed: set[int] = frozenset()):
    rows = [row_line(bench="b", job_id="j", arm="freeform", tree=tree, sha=tree[:8] * 5, split="test", row_id=r,
                     draw=draw, score=s, failed=r in failed, purpose=phase, eval_id=f"e_{tree[:2]}_{phase}_{draw}",
                     testset_sha256="t", trainset_sha256="tr") for r, s in scores.items()]
    store.append_eval({"eval_id": f"e_{tree[:2]}_{phase}_{draw}", "job_id": "j", "tree": tree, "purpose": phase,
                       "draw": draw, "trace_dir": None}, rows)


def _cand_doc(trees, final):
    return {"bench": "b", "arm": "freeform", "run_id": "freeform-b-s1", "job_id": "j", "final_tree": final,
            "n_candidates": len(trees), "candidates": [
                {"job_id": "j", "arm": "freeform", "sha": t[:8] * 5, "tree": t, "reqhash": "r", "iteration": i,
                 "iterations": [i], "is_final": t == final} for i, t in enumerate(trees, 1)]}


def test_build_portfolio_prunes_greedy_forces_final_and_drops_unstable(tmp_path):
    st = _store(tmp_path)
    cfg = AuditConfig(out=str(tmp_path / "out"), top_k=3)
    trees = ["a" * 40, "b" * 40, "c" * 40, "d" * 40, "e" * 40, "f" * 40]
    _put(st, "a" * 40, "screen", 0, {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0})
    _put(st, "b" * 40, "screen", 0, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0})
    _put(st, "c" * 40, "screen", 0, {0: 1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0})  # union-covered by a+b -> pruned
    _put(st, "d" * 40, "screen", 0, {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0})
    _put(st, "e" * 40, "screen", 0, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0})  # wins nothing -> pruned
    _put(st, "f" * 40, "screen", 0, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0})        # final, but row 5 missing -> hard error
    with pytest.raises(RuntimeError, match="final program is unstable"):
        build_portfolio(cfg, "b", "freeform", _cand_doc(trees, "f" * 40), H, st, log=lambda *a: None)
    # make "f" complete with the worst score; "e" stays unstable via a missing row
    _put(st, "f" * 40, "screen", 0, {5: 0})
    doc = _cand_doc(trees, "f" * 40)
    doc["candidates"][4]["unstable"] = "venv build failed"
    out = build_portfolio(cfg, "b", "freeform", doc, H, st, log=lambda *a: None)
    assert "c" * 40 in out["pruned"] and "f" * 40 in out["pruned"]
    assert out["survivors"] == ["a" * 40, "b" * 40, "d" * 40]
    port = [c["tree"] for c in out["portfolio"]]
    assert port == ["a" * 40, "b" * 40, "f" * 40] and out["final_forced"] and out["replaced"] == "d" * 40
    assert any(u["reason"] == "venv build failed" for u in out["unstable"]) and out["n_screened"] == 5
    # k larger than the survivors -> backfill from the pruned by aggregate mean (c .333 before f 0)
    cfg5 = AuditConfig(out=str(tmp_path / "out"), top_k=5)
    out5 = build_portfolio(cfg5, "b", "freeform", doc, H, st, log=lambda *a: None)
    assert [c["tree"] for c in out5["portfolio"]] == ["a" * 40, "b" * 40, "d" * 40, "c" * 40, "f" * 40]
    assert out5["backfilled"] == ["c" * 40, "f" * 40] and not out5["final_forced"]


def _full_pipeline(tmp_path):
    st = _store(tmp_path)
    cfg = AuditConfig(out=str(tmp_path / "out"), top_k=2, reps_select=2, reps_confirm=3)
    a, f = "a" * 40, "f" * 40
    _put(st, a, "screen", 0, {r: 1 for r in H}); _put(st, f, "screen", 0, {r: 0 for r in H})
    port = build_portfolio(cfg, "b", "freeform", _cand_doc([a, f], f), H, st, log=lambda *a_: None)
    # selection: a wins rows 0-3, f wins 4-5 (a scores 0 there, f scores 1)
    for d in (1, 2):
        _put(st, a, "select", d, {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0})
        _put(st, f, "select", d, {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1}, failed={0})
    w = build_winners(cfg, "b", "freeform", port, st, log=lambda *a_: None)
    assert w["rows_by_winner"] == {a: [0, 1, 2, 3], f: [4, 5]} and w["n_rows_final_is_winner"] == 2
    return st, cfg, port, w, a, f


def test_winners_results_and_summary(tmp_path):
    st, cfg, port, w, a, f = _full_pipeline(tmp_path)
    sel = {"arms": {"freeform": {"final_test_score": 0.8, "final_test_scores": [0.8], "score_source": "status.json",
                                 "seed_test_score": 0.5}}}
    diff = {"label": "probability", "score_type": "binary", "testset_sha256": "t", "n_seed_observations": 3,
            "rows": {str(r): {"mean": 0.2, "n": 3} for r in H}}
    with pytest.raises(ValueError):  # confirmation not run yet -> never report a partial round
        build_results(cfg, "b", "freeform", sel, diff, port, w, st, log=lambda *a_: None)
    for d in (3, 4, 5):  # confirm draws = reps_select+1 ..
        _put(st, a, "confirm", d, {0: 1, 1: 1, 2: 1, 3: 0})
        _put(st, f, "confirm", d, {4: 1, 5: 0})
        _put(st, f, "final", d, {0: 0, 1: 0, 2: 1, 3: 0})
    res = build_results(cfg, "b", "freeform", sel, diff, port, w, st, log=lambda *a_: None)
    m = res["metrics"]
    # B_r = (1+1+1+0 + 1+0)/6 = 4/6 every round; F_r = (0+0+1+0 + 1+0)/6 = 2/6 (rows 4,5 reuse confirm)
    assert m["B_rounds"] == pytest.approx([4 / 6] * 3) and m["F_rounds"] == pytest.approx([2 / 6] * 3)
    assert m["G"] == pytest.approx(2 / 6) and m["degenerate_zero_variance_G"] and m["n_rows_final_is_winner"] == 2
    assert res["n_failed_rows_by_phase"]["select"] == 2  # f failed row 0 in both selection draws
    assert len(res["raw"]) == 12 + 24 + 18 + 12
    paths = write_results(cfg, "b", "freeform", res, diff, w, st)
    csv_text = paths[1].read_text().splitlines()
    assert csv_text[0].startswith("# b / freeform") and csv_text[1].split(",")[:3] == ["row", "seed_estimate", "n_seed_obs"]
    assert len(csv_text) == 2 + len(H)
    row4 = csv_text[2 + 4].split(",")
    assert row4[3:5] == ["0.0", "1.0"] and row4[5] == "c2" and row4[-2] == "1"   # final (c2) wins row 4
    rows = summary_rows([res], {"zzz": "empty H"})
    assert rows[0]["arm"] == "freeform" and rows[0]["evo_minus_freeform_B"] is None and rows[0]["sum_B"] == pytest.approx(4.0)
    assert rows[1]["bench"] == "zzz" and rows[1]["not_applicable"] == "empty H"
    out = write_summary(cfg, rows)
    assert "not applicable" in out["md"].read_text() and json.loads(out["json"].read_text())[0]["B_best_of_population"] == pytest.approx(4 / 6)
