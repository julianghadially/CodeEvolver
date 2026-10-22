"""subsample_eval_timeout: round-trips through OptimizationConfig and reaches
the architect's RunEvalBinding as a real eval-subprocess backstop (unset, an
aborted/orphaned eval would run with timeout=None forever)."""

from __future__ import annotations

from src.engine.optimizer.v3.optimization_config import OptimizationConfig


def test_subsample_eval_timeout_default_and_roundtrip():
    cfg = OptimizationConfig(program="pkg.p:P", metric="pkg.m:M")
    assert cfg.subsample_eval_timeout == 7200  # generous 2h backstop default

    assert cfg.to_dict()["subsample_eval_timeout"] == 7200

    overridden = OptimizationConfig.from_dict(
        {"program": "pkg.p:P", "metric": "pkg.m:M", "subsample_eval_timeout": 1800}
    )
    assert overridden.subsample_eval_timeout == 1800
    # Survives a full serialize -> deserialize cycle.
    assert (
        OptimizationConfig.from_dict(overridden.to_dict()).subsample_eval_timeout
        == 1800
    )
