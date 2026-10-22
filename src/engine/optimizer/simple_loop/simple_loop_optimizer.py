"""Copyright © 2026 440 Labs LLC

`run_simple_loop_optimization` — the simple loop's run entry.

Takes the shared `OptimizationConfig` (the same config the CodeEvolver arms
receive), resolves datasets, evaluator and callbacks, and delegates to
`loop.run_simple_loop`. `debug_max_iterations` is ignored: eval rows are the
only budget.
"""

from __future__ import annotations

import logging

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.dataset_config import DatasetConfig
from src.engine.optimizer.v3.coding_harness_config import normalize_coding_harness_config
from src.engine.optimizer.v3.lm_config import resolve_agent_model_for_harness
from src.engine.optimizer.v3.optimization_config import OptimizationConfig

from .loop import run_simple_loop

logger = logging.getLogger(__name__)


def run_simple_loop_optimization(*, config: OptimizationConfig) -> dict:
    """Run a full simple-loop optimization and return the result dict."""
    coding_harness_config = normalize_coding_harness_config(
        coding_harness=config.coding_harness,
        coding_harness_model=config.coding_harness_model,
    )
    agent_model = resolve_agent_model_for_harness(
        reflection_lm=config.reflection_lm,
        coding_harness=coding_harness_config.coding_harness,
    )
    metric_names = config.metric_names or ["accuracy"]
    primary_metric = config.primary_metric or metric_names[0]

    callbacks = config.job.make_callbacks()
    callbacks.set_running()

    datasets = DatasetConfig.prepare(
        config.workspace,
        trainset_json=config.trainset_json,
        trainset_path=config.trainset_path,
        comparison_set_size=config.comparison_set_size,
        testset_json=config.testset_json,
        testset_path=config.testset_path,
    )
    evaluator_config = EvaluatorConfig(
        program_factory=config.program,
        metric_fns=[config.metric],
        metric_names=metric_names[:1],
        primary_metric=primary_metric,
        num_threads=config.num_threads,
        trainset_path=datasets.trainset_rel_path,
        testset_path=datasets.testset_rel_path or None,
        input_keys=config.input_keys,
    )
    try:
        return run_simple_loop(
            config=config,
            workspace=config.workspace,
            datasets=datasets,
            evaluator_config=evaluator_config,
            callbacks=callbacks,
            coding_harness_config=coding_harness_config,
            agent_model=agent_model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("simple loop raised — reporting failure to API")
        callbacks.set_failed(f"{type(exc).__name__}: {exc}")
        raise


__all__ = ["run_simple_loop_optimization"]
