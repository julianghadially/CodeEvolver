"""Finalization test-set sampling (`src/engine/evaluator/holdout.py`).

The held-out set is drawn `test_samples` times per program at finalization;
every reading is reported and the legacy `*_test_score` fields are the means.
Caching is NOT configured by the engine (the client repo owns it) — the engine
only detects identical samples and warns.
"""

from __future__ import annotations

import json
import logging
import types

import pytest

from src.engine.evaluator import holdout
from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.sandbox_client import EvalResult
from src.engine.optimizer.v3.optimization_config import OptimizationConfig
from src.engine.telemetry.trace_layout import write_eval_trace


def _evaluator() -> EvaluatorConfig:
    return EvaluatorConfig(
        program_factory="p:P", metric_fns=["m:m"], metric_names=["accuracy"],
        primary_metric="accuracy", trainset_path="train.json", testset_path="test.json",
    )


def _datasets():
    return types.SimpleNamespace(has_testset=True, testset_ids=[0, 1], testset_size=2, comparison_ids=[])


def _write_trace(trace_dir, outputs, failed=False):
    write_eval_trace(str(trace_dir), [
        {"example_id": rid, "score": 1.0, "output": out, "failed": failed, "trace": []}
        for rid, out in enumerate(outputs)
    ], max_line_chars=None)


def _result(score, trace_dir=""):
    return EvalResult(
        scores_by_metric={"accuracy": score}, primary_score=score, trace_meta=[],
        trace_dir=trace_dir, trace_index_file="", row_ids=[0, 1], raw={},
    )


def test_test_suffix_names_every_draw_with_the_holdout_fragment():
    assert holdout.test_suffix(0) == "holdout"
    assert holdout.test_suffix(1) == "holdout_2"
    assert holdout.test_suffix(2) == "holdout_3"
    # Every draw's trace directory carries the substring fence fragment.
    from src.engine.telemetry.trace_layout import HOLDOUT_DENY_FRAGMENT
    for k in range(3):
        assert HOLDOUT_DENY_FRAGMENT in f"iteration_1_{holdout.test_suffix(k)}"
    with pytest.raises(ValueError):
        holdout.test_suffix(-1)


def test_mean_or_none():
    assert holdout.mean_or_none([]) is None
    assert holdout.mean_or_none([0.5, 1.0]) == 0.75


def test_sample_test_scores_draws_n_times_with_distinct_suffixes():
    calls = []

    def fake(**kw):
        calls.append((kw["suffix"], kw["iteration_idx"], tuple(kw["row_ids"])))
        return _result(0.1 * (len(calls)))

    scores = holdout.sample_test_scores(
        workspace="/ws", config=_evaluator(), datasets=_datasets(),
        iteration_idx=1, num_samples=3, timeout=5, run_eval_fn=fake,
    )
    assert calls == [("holdout", 1, (0, 1)), ("holdout_2", 1, (0, 1)), ("holdout_3", 1, (0, 1))]
    assert scores == pytest.approx([0.1, 0.2, 0.3])


def test_sample_test_scores_skips_a_failed_draw_and_keeps_the_rest():
    n = {"k": 0}

    def fake(**kw):
        n["k"] += 1
        if n["k"] == 2:
            raise RuntimeError("provider blip")
        return _result(0.5)

    scores = holdout.sample_test_scores(
        workspace="/ws", config=_evaluator(), datasets=_datasets(),
        iteration_idx=0, num_samples=3, timeout=None, run_eval_fn=fake,
    )
    assert scores == [0.5, 0.5]


def test_sample_test_scores_rejects_zero_samples():
    with pytest.raises(ValueError):
        holdout.sample_test_scores(
            workspace="/ws", config=_evaluator(), datasets=_datasets(),
            iteration_idx=0, num_samples=0, timeout=None, run_eval_fn=lambda **k: None,
        )


def test_identical_samples_warn_but_varying_samples_do_not(tmp_path, caplog):
    same = [tmp_path / f"same_{i}" for i in range(3)]
    for p in same:
        _write_trace(p, ["alpha", "beta"])
    assert holdout.samples_are_identical([str(p) for p in same])

    varied = [tmp_path / f"var_{i}" for i in range(3)]
    for i, p in enumerate(varied):
        _write_trace(p, ["alpha", f"beta-{i}"])
    assert not holdout.samples_are_identical([str(p) for p in varied])

    # Failed rows (output null in every draw) are not evidence of a cache.
    failed = [tmp_path / f"fail_{i}" for i in range(3)]
    for p in failed:
        _write_trace(p, [None, None], failed=True)
    assert not holdout.samples_are_identical([str(p) for p in failed])

    # Missing / unreadable traces are "no evidence", never a warning.
    assert not holdout.samples_are_identical([str(same[0]), str(tmp_path / "nope")])
    assert not holdout.samples_are_identical([str(same[0])])

    files = iter(same)

    def fake(**kw):
        return _result(1.0, trace_dir=str(next(files)))

    with caplog.at_level(logging.WARNING, logger="src.engine.evaluator.holdout"):
        holdout.sample_test_scores(
            workspace="/ws", config=_evaluator(), datasets=_datasets(),
            iteration_idx=1, num_samples=3, timeout=None, run_eval_fn=fake,
        )
    assert any("identical across 3 draws" in r.getMessage() for r in caplog.records)


def test_load_dataset_resolves_every_test_suffix_to_the_holdout(tmp_path):
    from src.engine.evaluator.mounted.evaluate import _load_dataset

    (tmp_path / "train.json").write_text(json.dumps([{"q": "train"}]))
    (tmp_path / "test.json").write_text(json.dumps([{"q": "held-out"}]))
    extra = {"trainset_path": "train.json", "testset_path": "test.json"}
    for suffix in ("holdout", "holdout_2", "holdout_3"):
        assert _load_dataset(str(tmp_path), extra, suffix) == [{"q": "held-out"}]
    for suffix in ("test", "testing", "holdouts", "final_comparison", "comparison", "child"):
        assert _load_dataset(str(tmp_path), extra, suffix) == [{"q": "train"}]


def test_split_for_suffix_mirrors_the_evaluator():
    from src.engine.optimizer.v3.eval_log import split_for_suffix

    assert split_for_suffix("holdout") == "test"
    assert split_for_suffix("holdout_2") == "test"
    assert split_for_suffix("test") == "training"
    assert split_for_suffix("comparison") == "comparison"
    assert split_for_suffix("child") == "training"


def test_test_samples_config_default_roundtrip_and_validation():
    cfg = OptimizationConfig(program="p:P", metric="m:m")
    assert cfg.test_samples == 3
    assert cfg.to_dict()["test_samples"] == 3
    assert OptimizationConfig.from_dict({"program": "p:P", "metric": "m:m"}).test_samples == 3
    assert OptimizationConfig.from_dict(
        {"program": "p:P", "metric": "m:m", "test_samples": "5"}
    ).test_samples == 5
    with pytest.raises(ValueError):
        OptimizationConfig(program="p:P", metric="m:m", test_samples=0)
    assert cfg.seed_test_samples == 1
    assert OptimizationConfig.from_dict(
        {"program": "p:P", "metric": "m:m", "seed_test_samples": 2}
    ).to_dict()["seed_test_samples"] == 2
    with pytest.raises(ValueError):
        OptimizationConfig(program="p:P", metric="m:m", seed_test_samples=0)


def test_test_scores_field_is_whitelisted_everywhere():
    from src.db.db_schemas.db_schemas import IterationHistoryRecord, JobRecord
    from src.engine.schemas.iteration_history import ITERATION_RECORD_FIELDS

    assert "test_scores" in ITERATION_RECORD_FIELDS
    assert "test_scores" in IterationHistoryRecord.model_fields
    assert {"seed_test_scores", "final_test_scores"} <= set(JobRecord.model_fields)


def test_set_completed_carries_the_sample_lists():
    from src.engine.services.callbacks import JobCallbacks

    cb = JobCallbacks.__new__(JobCallbacks)
    sent = {}
    cb._put_status = lambda payload: sent.update(payload)  # type: ignore[attr-defined]
    cb.set_completed(
        final_candidate=None, total_metric_calls=1, num_branches=1, ce_state=None,
        seed_test_score=0.5, final_test_score=0.7,
        seed_test_scores=[0.4, 0.6], final_test_scores=[0.7, 0.7],
    )
    assert sent["seed_test_scores"] == [0.4, 0.6]
    assert sent["final_test_scores"] == [0.7, 0.7]
    sent.clear()
    cb.set_completed(final_candidate=None, total_metric_calls=1, num_branches=1, ce_state=None)
    assert "seed_test_scores" not in sent and "final_test_scores" not in sent


def test_v3_finalization_samples_both_branches_and_stamps_history(monkeypatch):
    from src.engine.optimizer.v3 import finalization
    from src.engine.schemas.iteration_history import IterationHistory

    calls = []
    checkouts = []
    scores = iter([0.4, 0.8, 1.0, 0.9])

    def fake(**kw):
        calls.append((kw["suffix"], kw["iteration_idx"]))
        return _result(next(scores))

    monkeypatch.setattr(finalization, "run_eval", fake)
    monkeypatch.setattr(
        finalization, "git_checkout", lambda _ws, ref, **_k: checkouts.append(ref),
    )
    monkeypatch.setattr(finalization, "clean_python_bytecode", lambda *_a: None)
    monkeypatch.setattr(
        finalization, "final_candidate",
        lambda _s: {"git_branch": "codeevolver-x-final", "git_sha": "abc"},
    )
    history = IterationHistory()
    history[0] = {"git_branch": "seed"}
    history[1] = {"git_branch": "codeevolver-x-final"}
    out = finalization.score_seed_and_final(
        workspace="/ws", datasets=_datasets(), evaluator_config=_evaluator(),
        state=None, history=history, seed_branch="seed", seed_sha="5eed5ha",
        comparison_eval_timeout=None, test_samples=3, seed_test_samples=1,
    )
    assert calls == [
        ("holdout", 0),
        ("holdout", 1), ("holdout_2", 1), ("holdout_3", 1),
    ]
    # The seed is scored at its COMMIT (detached), the final at its branch.
    assert checkouts == ["5eed5ha", "codeevolver-x-final"]
    assert out.seed_test_scores == pytest.approx([0.4])
    assert out.final_test_scores == pytest.approx([0.8, 1.0, 0.9])
    assert out.seed_test == pytest.approx(0.4)
    assert out.final_test == pytest.approx(0.9)
    assert history[0]["test_scores"] == pytest.approx([0.4])
    assert history[0]["test_score"] == pytest.approx(0.4)
    assert history[1]["test_scores"] == pytest.approx([0.8, 1.0, 0.9])
    assert history[1]["test_score"] == pytest.approx(0.9)



def test_v3_final_equals_seed_still_gets_its_own_draws_when_asked(monkeypatch):
    """Seed==final short-circuit only when the final needs no more draws
    than the seed already took; otherwise the final is sampled itself."""
    from src.engine.optimizer.v3 import finalization
    from src.engine.schemas.iteration_history import IterationHistory

    calls = []
    monkeypatch.setattr(
        finalization, "run_eval",
        lambda **kw: calls.append((kw["suffix"], kw["iteration_idx"])) or _result(0.5),
    )
    monkeypatch.setattr(finalization, "git_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(finalization, "clean_python_bytecode", lambda *_a: None)
    monkeypatch.setattr(
        finalization, "final_candidate", lambda _s: {"git_branch": "seed", "git_sha": "s"},
    )
    out = finalization.score_seed_and_final(
        workspace="/ws", datasets=_datasets(), evaluator_config=_evaluator(),
        state=None, history=IterationHistory(), seed_branch="seed", seed_sha="s",
        comparison_eval_timeout=None, test_samples=3, seed_test_samples=1,
    )
    assert calls == [("holdout", 0), ("holdout", 1), ("holdout_2", 1), ("holdout_3", 1)]
    assert len(out.seed_test_scores) == 1 and len(out.final_test_scores) == 3


def test_v3_seed_is_scored_at_seed_sha_even_when_final_lives_on_the_seed_branch(monkeypatch):
    """The freeform architect works ON the run-main branch, so checking the seed out by
    branch name scores the final program twice. The seed must be checked
    out at `seed_sha`; sharing a branch name with the final must not
    trigger the seed==final short-circuit either."""
    from src.engine.optimizer.v3 import finalization
    from src.engine.schemas.iteration_history import IterationHistory

    checkouts = []
    calls = []
    monkeypatch.setattr(
        finalization, "run_eval",
        lambda **kw: calls.append(kw["iteration_idx"]) or _result(0.5),
    )
    monkeypatch.setattr(
        finalization, "git_checkout", lambda _ws, ref, **_k: checkouts.append(ref),
    )
    monkeypatch.setattr(finalization, "clean_python_bytecode", lambda *_a: None)
    monkeypatch.setattr(
        finalization, "final_candidate",
        lambda _s: {"git_branch": "codeevolver-run-main", "git_sha": "f1na1"},
    )
    finalization.score_seed_and_final(
        workspace="/ws", datasets=_datasets(), evaluator_config=_evaluator(),
        state=None, history=IterationHistory(),
        seed_branch="codeevolver-run-main", seed_sha="5eed",
        comparison_eval_timeout=None, test_samples=1, seed_test_samples=1,
    )
    assert checkouts == ["5eed", "codeevolver-run-main"]
    assert calls == [0, 1]  # both scored — no short-circuit on branch name


def test_v3_finalization_failure_is_recorded_and_branch_restored(monkeypatch):
    """A scoring failure must surface as
    `finalization_error` instead of silently returning None scores, and the
    workspace must be restored to the final branch (the seed pass is a
    detached checkout)."""
    from src.engine.optimizer.v3 import finalization
    from src.engine.schemas.iteration_history import IterationHistory

    checkouts = []

    def checkout(_ws, ref, **kw):
        checkouts.append((ref, kw.get("force", False)))
        if ref == "5eed5ha":
            raise RuntimeError(
                "git checkout failed: local changes would be overwritten"
            )

    monkeypatch.setattr(finalization, "run_eval", lambda **_kw: _result(0.5))
    monkeypatch.setattr(finalization, "git_checkout", checkout)
    monkeypatch.setattr(finalization, "clean_python_bytecode", lambda *_a: None)
    monkeypatch.setattr(
        finalization, "final_candidate",
        lambda _s: {"git_branch": "codeevolver-x-final", "git_sha": "abc"},
    )
    datasets = types.SimpleNamespace(
        has_testset=True, testset_ids=[0, 1], testset_size=2, comparison_ids=[],
    )
    out = finalization.score_seed_and_final(
        workspace="/ws", datasets=datasets, evaluator_config=_evaluator(),
        state=None, history=IterationHistory(), seed_branch="seed",
        seed_sha="5eed5ha", comparison_eval_timeout=None,
        test_samples=1, seed_test_samples=1,
    )
    assert out.finalization_error is not None
    assert "RuntimeError" in out.finalization_error
    assert "overwritten" in out.finalization_error
    assert out.final_test is None and out.seed_test is None
    # Detached seed checkout, then the forced restore of the final branch.
    assert checkouts[0] == ("5eed5ha", True)
    assert checkouts[-1] == ("codeevolver-x-final", True)


def test_v3_finalization_success_has_no_error(monkeypatch):
    from src.engine.optimizer.v3 import finalization
    from src.engine.schemas.iteration_history import IterationHistory

    monkeypatch.setattr(finalization, "run_eval", lambda **_kw: _result(0.5))
    monkeypatch.setattr(finalization, "git_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(finalization, "clean_python_bytecode", lambda *_a: None)
    monkeypatch.setattr(
        finalization, "final_candidate", lambda _s: {"git_branch": "b", "git_sha": "f"},
    )
    out = finalization.score_seed_and_final(
        workspace="/ws", datasets=_datasets(), evaluator_config=_evaluator(),
        state=None, history=IterationHistory(), seed_branch="b", seed_sha="s",
        comparison_eval_timeout=None, test_samples=1, seed_test_samples=1,
    )
    assert out.finalization_error is None


def test_set_completed_carries_finalization_error():
    from src.engine.services.callbacks import JobCallbacks

    cb = JobCallbacks.__new__(JobCallbacks)
    sent = {}
    cb._put_status = lambda payload: sent.update(payload)  # type: ignore[attr-defined]
    cb.set_completed(
        final_candidate=None, total_metric_calls=1, num_branches=1, ce_state=None,
        finalization_error="RuntimeError: checkout blocked",
    )
    assert sent["status"] == "completed"
    assert sent["finalization_error"] == "RuntimeError: checkout blocked"
    sent.clear()
    cb.set_completed(final_candidate=None, total_metric_calls=1, num_branches=1, ce_state=None)
    assert "finalization_error" not in sent


def test_git_checkout_force_discards_dirty_tracked_file(tmp_path):
    """The finalization checkout path must survive a modified tracked file
    (codeevolver.md) blocking a plain `git checkout`."""
    import subprocess

    from src.engine.services.subprocess_service import git_checkout

    ws = str(tmp_path)

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=ws, check=True, capture_output=True,
        )

    git("init", "-b", "main")
    (tmp_path / "codeevolver.md").write_text("v1")
    git("add", "-A"); git("commit", "-m", "c1")
    git("checkout", "-b", "other")
    (tmp_path / "codeevolver.md").write_text("v2")
    git("add", "-A"); git("commit", "-m", "c2")
    # Dirty the tracked file so a plain checkout back to main conflicts.
    (tmp_path / "codeevolver.md").write_text("dirty local edit")
    with pytest.raises(RuntimeError):
        git_checkout(ws, "main")
    git_checkout(ws, "main", force=True)
    assert (tmp_path / "codeevolver.md").read_text() == "v1"
