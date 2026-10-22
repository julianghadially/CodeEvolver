"""Stage 1: pooled seed held-out observations -> frozen difficult set H."""

from __future__ import annotations

import pytest

from scripts.best_of_population.config import AuditConfig
from scripts.best_of_population.difficult import build_difficult
from tests.unit.bop_fixtures import make_run

pytestmark = pytest.mark.unit


def _cfg(tmp_path, root, **kw):
    return AuditConfig(out=str(tmp_path / "out"), runs_roots=[str(root)], ce_home=str(tmp_path / "ce"), **kw)


def test_binary_hover_normalization_excludes_exact_half_and_pools_all_arms(tmp_path):
    root = tmp_path / "ce-runs"
    # row0: 1/2 -> exactly 0.5 (excluded); row1: 0/2 -> H; row2: 2/2 -> not; row3: failed in one run, 0 in other -> H
    make_run(root, "ee-hover-s1", job_id="e1", seed_rows={0: 0.998, 1: 0.0, 2: 0.998, 3: 0.0}, seed_failed=[3])
    make_run(root, "freeform-hover-s1", job_id="f1", seed_rows={0: 0.0, 1: 0.0, 2: 0.998, 3: 0.0})
    make_run(root, "pareto_evolution-hover-s1", job_id="p1", seed_rows={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})  # void
    make_run(root, "asa-hover-s1", job_id="a1", seed_rows=None)                                        # no seed line
    make_run(root, "simple_loop-hover-s1", job_id="s1", seed_rows={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
             models={"coding_harness_model": "other"})                                                  # incompatible
    out = build_difficult(_cfg(tmp_path, root), "hover", log=lambda *a: None)
    assert out["score_type"] == "binary" and out["label"] == "probability" and out["max_score_observed"] == 0.998
    assert out["H"] == [1, 3] and out["n_seed_observations"] == 2
    assert out["rows"]["0"]["mean"] == 0.5 and out["rows"]["0"]["n"] == 2
    assert out["rows"]["3"]["n"] == 1 and out["rows"]["3"]["n_failed"] == 1
    reasons = {e["run_id"]: e["reason"] for e in out["runs_excluded"]}
    assert reasons["pareto_evolution-hover-s1"].startswith("void")
    assert "no seed" in reasons["asa-hover-s1"] and "incompatible" in reasons["simple_loop-hover-s1"]
    assert out["sparse_rows"] == [0, 1, 2, 3] and out["missing_rows"] == [] and not out["not_applicable"]


def test_fractional_expected_score_missing_rows_and_not_applicable(tmp_path):
    root = tmp_path / "ce-runs"
    make_run(root, "ee-ragqa-s1", job_id="e1", bench="ragqa", seed_rows={0: 0.25, 1: 1.0, 2: 0.5}, testset_size=4)
    make_run(root, "ee-ragqa-s2", job_id="e2", bench="ragqa", seed_rows={0: 0.5, 1: 0.75, 2: 0.5}, testset_size=4)
    out = build_difficult(_cfg(tmp_path, root), "ragqa", log=lambda *a: None)
    assert out["score_type"] == "fractional" and out["label"] == "expected_score"
    assert out["H"] == [0] and out["missing_rows"] == [3] and out["testset_size"] == 4
    make_run(root, "ee-hover-s1", job_id="h1", seed_rows={0: 1.0, 1: 1.0})
    out2 = build_difficult(_cfg(tmp_path, root), "hover", log=lambda *a: None)
    assert out2["not_applicable"] and out2["H"] == []
    out3 = build_difficult(_cfg(tmp_path, root), "nope", log=lambda *a: None)
    assert out3["not_applicable"] and out3["runs_used"] == []
