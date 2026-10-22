"""Stage 0: best-final-score run per (bench, arm), void-aware, posthoc-aware."""

from __future__ import annotations

import pytest

from scripts.best_of_population.config import AuditConfig
from scripts.best_of_population.select_run import select_runs, selection_table
from tests.unit.bop_fixtures import make_run

pytestmark = pytest.mark.unit


def test_best_score_wins_void_dirs_ignored_ties_to_lower_seed(tmp_path):
    root = tmp_path / "ce-runs"
    make_run(root, "freeform-hover-s1", job_id="j1", final_test_score=0.80)
    make_run(root, "freeform-hover-s2", job_id="j2", final_test_score=0.85)
    make_run(root, "freeform-hover-s3", job_id="j3", final_test_score=0.85)      # tie -> s2
    make_run(root, "freeform-hover-s4", job_id="j4", final_test_score=0.99, status="failed")
    make_run(root, "freeform-hover-s5", job_id="j5", final_test_score=0.99, with_bundle=False)
    make_run(root, "pareto_evolution-hover-s1", job_id="p1", final_test_score=0.99)         # void (needs -redo)
    make_run(root, "pareto_evolution-hover-s1-redo", job_id="p1r", final_test_score=0.70)
    make_run(root, "pareto_evolution-hover-s2-redo", job_id="p2r", final_test_score=None,
             posthoc={"final_test_score": 0.75, "final_test_scores": [0.7, 0.8], "program_sha": "f" * 40})
    make_run(root, "pareto_evolution-hover-s3-redo", job_id="p3r", final_test_score=None,
             posthoc={"final_test_score": 0.95, "program_sha": "0" * 40})                   # sha mismatch
    cfg = AuditConfig(out=str(tmp_path / "out"), runs_roots=[str(root)], ce_home=str(tmp_path / "ce"))
    sel = select_runs(cfg, "hover", log=lambda *a: None)
    ff = sel["arms"]["freeform"]
    assert ff["run_id"] == "freeform-hover-s2" and ff["score_source"] == "status.json"
    reasons = {c["run_id"]: c["reason"] for c in ff["considered"]}
    assert "status failed" in reasons["freeform-hover-s4"] and "bundle" in reasons["freeform-hover-s5"]
    pe = sel["arms"]["pareto_evolution"]
    assert pe["run_id"] == "pareto_evolution-hover-s2-redo"
    assert pe["score_source"] == "posthoc_holdout_rescore.json" and pe["final_test_scores"] == [0.7, 0.8]
    pr = {c["run_id"]: c for c in pe["considered"]}
    assert pr["pareto_evolution-hover-s1"]["reason"].startswith("void")
    assert "program_sha" in pr["pareto_evolution-hover-s3-redo"]["reason"]
    assert "freeform-hover-s2" in selection_table(sel)


def test_no_eligible_run(tmp_path):
    root = tmp_path / "ce-runs"
    make_run(root, "freeform-hover-s1", job_id="j1", final_test_score=None)
    cfg = AuditConfig(out=str(tmp_path / "out"), runs_roots=[str(root)], ce_home=str(tmp_path / "ce"))
    sel = select_runs(cfg, "hover", log=lambda *a: None)
    assert sel["arms"]["freeform"]["run_id"] is None and sel["arms"]["pareto_evolution"]["run_id"] is None
