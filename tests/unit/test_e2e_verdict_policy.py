"""Offline check of the local e2e's self-check policy.

`local_test_app.py::_post_run_agent_failures` decides what the e2e treats as a
FAILURE (an armed auditor broke, a constraint gate fired) versus a WARNING (a
cheating classification from the run-integrity inspector, which on the local
target reflects the advisory holdout fence rather than harness health). No
agent, no network: the function only reads the payload `run_e2e` returns.
"""
from __future__ import annotations

import argparse

import pytest

from local_test_app import _post_run_agent_failures


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        meta_reflection=False,
        clean_slate=False,
        run_integrity=True,
        fail_on_cheating=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _result(category, rationale="read the holdout via bash") -> dict:
    return {
        "integrity_verdict": {
            "integrity_cheating_category": category,
            "integrity_cheating_rationale": rationale,
        },
        "constraint_violations": [],
    }


@pytest.mark.parametrize("category", ["non_impactful_cheating", "impactful_cheating"])
def test_cheating_is_a_warning_by_default(category):
    failures, warnings = _post_run_agent_failures(_args(), _result(category))
    assert failures == []
    assert len(warnings) == 1
    assert category in warnings[0]
    assert "read the holdout via bash" in warnings[0]


@pytest.mark.parametrize("category", ["non_impactful_cheating", "impactful_cheating"])
def test_fail_on_cheating_flag_restores_strict_policy(category):
    failures, warnings = _post_run_agent_failures(
        _args(fail_on_cheating=True), _result(category)
    )
    assert warnings == []
    assert len(failures) == 1
    assert category in failures[0]


@pytest.mark.parametrize("fail_on_cheating", [False, True])
def test_missing_verdict_always_fails(fail_on_cheating):
    failures, warnings = _post_run_agent_failures(
        _args(fail_on_cheating=fail_on_cheating), _result(None)
    )
    assert warnings == []
    assert len(failures) == 1
    assert "no verdict" in failures[0]


def test_no_cheating_is_clean():
    failures, warnings = _post_run_agent_failures(_args(), _result("no_cheating"))
    assert failures == []
    assert warnings == []


def test_constraint_gate_fire_still_fails_alongside_a_cheating_warning():
    result = _result("non_impactful_cheating")
    result["constraint_violations"] = [{"iteration": "2", "reason": "hardcoded"}]
    failures, warnings = _post_run_agent_failures(_args(), result)
    assert len(warnings) == 1
    assert len(failures) == 1
    assert "constraint gate fired on iteration 2" in failures[0]


def test_run_integrity_disabled_ignores_verdict():
    failures, warnings = _post_run_agent_failures(
        _args(run_integrity=False), _result("impactful_cheating")
    )
    assert failures == []
    assert warnings == []
