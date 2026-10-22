"""Copyright © 2026 440 Labs LLC

Unit tests for EvaluatorConfig validation and factory-path parsing.
"""

from __future__ import annotations

import pytest

from src.engine.evaluator.config import EvaluatorConfig, parse_factory_path


def test_parse_factory_path_colon_form():
    assert parse_factory_path("pkg.mod:factory_fn") == ("pkg.mod", "factory_fn")


def test_parse_factory_path_dotted_form():
    assert parse_factory_path("pkg.mod.factory_fn") == ("pkg.mod", "factory_fn")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "notamodule",
        "pkg..mod:fn",
        "pkg.mod:",
        ":fn",
        "pkg.1bad:fn",
    ],
)
def test_parse_factory_path_rejects_malformed(bad: str):
    with pytest.raises(ValueError):
        parse_factory_path(bad)


def test_evaluator_config_happy_path():
    cfg = EvaluatorConfig(
        program_factory="my.pkg:make",
        metric_fns=["my.pkg.metrics:accuracy", "my.pkg.metrics:f1"],
        metric_names=["accuracy", "f1"],
        primary_metric="accuracy",
    )
    assert cfg.primary_metric == "accuracy"


def test_evaluator_config_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        EvaluatorConfig(
            program_factory="my.pkg:make",
            metric_fns=["my.pkg.metrics:accuracy"],
            metric_names=["accuracy", "f1"],
            primary_metric="accuracy",
        )


def test_evaluator_config_rejects_unknown_primary_metric():
    with pytest.raises(ValueError, match="primary_metric"):
        EvaluatorConfig(
            program_factory="my.pkg:make",
            metric_fns=["my.pkg.metrics:accuracy"],
            metric_names=["accuracy"],
            primary_metric="latency",
        )


def test_evaluator_config_to_payload_round_trips_fields():
    cfg = EvaluatorConfig(
        program_factory="my.pkg:make",
        metric_fns=["my.pkg.metrics:accuracy"],
        metric_names=["accuracy"],
        primary_metric="accuracy",
    )
    payload = cfg.to_payload(
        row_ids=[1, 2, 3],
        iteration_idx=7,
        suffix="child",
        capture_traces=True,
    )
    assert payload["command"] == "evaluate"
    assert payload["row_ids"] == [1, 2, 3]
    assert payload["iteration_idx"] == 7
    assert payload["suffix"] == "child"
    assert payload["primary_metric"] == "accuracy"
