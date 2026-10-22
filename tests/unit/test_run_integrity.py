"""Copyright © 2026 440 Labs LLC

Unit tests for the run-integrity inspector (offline — no agent spawned).

Covers the prompt builder (cheating taxonomy + data-index carve-out,
functioning-iteration rubric, safe-mode preamble), verdict parsing (tolerant,
never raises), the driver's output contracts (closing mode writes to
memory_dir; safe mode writes JSON + markdown to the operator path and never
into memory_dir), non-fatality, and the config-flag round-trip.
"""

from __future__ import annotations

import json
import os

from src.engine.optimizer.meta import run_integrity as ri
from src.engine.schemas.iteration_history import ITERATION_RECORD_FIELDS


def _prompt(**overrides):
    kwargs = dict(
        method="ee",
        num_iterations=4,
        best_valset_score=0.8,
        seed_test_score=0.5,
        best_candidate_test_score=0.7,
        workspace="/workspace/repo",
        iteration_indices=[0, 1, 2, 3],
        testset_path="data/bench_test.json",
        safe_mode=False,
    )
    kwargs.update(overrides)
    return ri.build_run_integrity_prompt(**kwargs)


# ---------------------------------------------------------------- prompt


def test_prompt_names_all_three_categories():
    prompt = _prompt()
    for category in ri.INTEGRITY_CATEGORIES:
        assert category in prompt
    # The cheating definition binds on hard-coding; QA-pair indexes over test
    # rows are named under the impactful category.
    assert "hard-coding" in prompt
    assert "QA-pair indexes over test rows" in prompt


def test_prompt_permits_reading_the_testset_for_hardcoding_checks():
    prompt = _prompt()
    assert "data/bench_test.json" in prompt
    assert "MAY read it" in prompt


def test_prompt_contains_functioning_iteration_rubric():
    prompt = _prompt()
    assert "Functioning iteration" in prompt
    assert "minor" in prompt and "major" in prompt
    # The two canonical examples from the spec.
    assert "run_eval error" in prompt
    assert "outage" in prompt.lower()


def test_prompt_safe_mode_preamble_present_iff_safe_mode():
    assert "SAFE MODE" not in _prompt(safe_mode=False)
    safe = _prompt(safe_mode=True)
    assert "SAFE MODE" in safe
    assert "MUST NOT switch git branches" in safe
    # Safe mode also flips the framing from finished to in-progress.
    assert "is in progress" in safe


def test_prompt_points_at_traces_and_memory():
    prompt = _prompt()
    assert "_architect_trace" in prompt
    assert "memory.md" in prompt
    assert "iteration_change_summaries.md" in prompt


def test_prompt_handles_missing_testset_path():
    prompt = _prompt(testset_path=None)
    assert "No test-set path was provided" in prompt


# ---------------------------------------------------------------- parsing


def test_parse_verdict_happy_path():
    md = (
        "## Cheating assessment\nclean\n\n```json\n"
        '{"cheating_category": "no_cheating", "cheating_rationale": "checked", '
        '"functioning_iterations": "4/4", "integrity_pct": 100.0}\n```\n'
    )
    v = ri.parse_integrity_verdict(md)
    assert v == {
        "cheating_category": "no_cheating",
        "cheating_rationale": "checked",
        "functioning_iterations": "4/4",
        "integrity_pct": 100.0,
        "budget_bypass": None,
        "budget_bypass_estimated_rows": None,
        "budget_bypass_rationale": "",
    }


def test_parse_verdict_budget_bypass_fields():
    from src.engine.optimizer.meta.run_integrity import (
        INTEGRITY_RECORD_KEYS,
        parse_integrity_verdict,
        stamp_integrity_verdict,
    )

    md = (
        "report\n```json\n{\"cheating_category\": \"no_cheating\", \"cheating_rationale\": \"r\", "
        "\"functioning_iterations\": \"3/3\", \"integrity_pct\": 100, "
        "\"budget_bypass\": \"true\", \"budget_bypass_estimated_rows\": \"12\", "
        "\"budget_bypass_rationale\": \"ran python over 12 rows\"}\n```\n"
    )
    v = parse_integrity_verdict(md)
    assert v["budget_bypass"] is True and v["budget_bypass_estimated_rows"] == 12
    assert v["budget_bypass_rationale"] == "ran python over 12 rows"
    # absent -> None, never a crash
    v2 = parse_integrity_verdict("```json\n{\"cheating_category\": \"no_cheating\"}\n```")
    assert v2["budget_bypass"] is None and v2["budget_bypass_estimated_rows"] is None
    history = {0: {}, 2: {}}
    assert stamp_integrity_verdict(history, v) == 2
    assert history[2]["integrity_budget_bypass"] is True
    assert history[2]["integrity_budget_bypass_estimated_rows"] == 12
    assert set(INTEGRITY_RECORD_KEYS) <= set(history[2])


def test_parse_verdict_takes_last_matching_block():
    md = (
        '```json\n{"cheating_category": "no_cheating", "integrity_pct": 10}\n```\n'
        "revised below\n"
        '```json\n{"cheating_category": "impactful_cheating", '
        '"cheating_rationale": "hardcoded answers", "integrity_pct": 50}\n```\n'
    )
    v = ri.parse_integrity_verdict(md)
    assert v["cheating_category"] == "impactful_cheating"
    assert v["integrity_pct"] == 50.0


def test_parse_verdict_coerces_unknown_category_and_bad_pct():
    md = (
        '```json\n{"cheating_category": "totally fine", '
        '"integrity_pct": "lots"}\n```\n'
    )
    v = ri.parse_integrity_verdict(md)
    assert v["cheating_category"] == "unparsed"
    assert v["integrity_pct"] is None


def test_parse_verdict_none_on_no_json_or_garbage():
    assert ri.parse_integrity_verdict(None) is None
    assert ri.parse_integrity_verdict("no fences here") is None
    assert ri.parse_integrity_verdict("```json\n{not json}\n```") is None
    # A JSON block without the category key is not a verdict.
    assert ri.parse_integrity_verdict('```json\n{"other": 1}\n```') is None


# ---------------------------------------------------------------- driver

_GOOD_REPORT = (
    "## Cheating assessment\n**Category:** no_cheating\n\n## Run integrity\n"
    "4/4\n\n```json\n"
    '{"cheating_category": "no_cheating", "cheating_rationale": "clean", '
    '"functioning_iterations": "4/4", "integrity_pct": 100.0}\n```\n'
)


def _run(monkeypatch, tmp_path, agent_result, **overrides):
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    captured = {}

    def fake_run_inspection_agent(**kwargs):
        captured.update(kwargs)
        return agent_result

    import src.engine.services.subprocess_service as svc

    monkeypatch.setattr(svc, "run_inspection_agent", fake_run_inspection_agent)

    kwargs = dict(
        workspace="/workspace",
        method="ee",
        num_iterations=4,
        best_valset_score=0.8,
        seed_test_score=0.5,
        best_candidate_test_score=0.7,
        iteration_indices=[0, 1, 2, 3],
        testset_path="data/bench_test.json",
        coding_harness="claude",
    )
    kwargs.update(overrides)
    return ri.run_integrity_inspection(**kwargs), captured


def test_closing_mode_writes_report_and_returns_verdict(tmp_path, monkeypatch):
    verdict, captured = _run(
        monkeypatch, tmp_path, {"success": True, "proposed_change": _GOOD_REPORT}
    )
    assert verdict["cheating_category"] == "no_cheating"
    assert verdict["integrity_pct"] == 100.0
    written = os.path.join(str(tmp_path), "run_integrity.md")
    assert os.path.exists(written)
    # Read-only auditor, post-run: no deny paths (it may read the testset).
    assert captured["writable"] is False
    assert captured["deny_paths"] == []


def test_unparsed_report_still_returns_fallback_verdict(tmp_path, monkeypatch):
    verdict, _ = _run(
        monkeypatch, tmp_path,
        {"success": True, "proposed_change": "prose only, no JSON block"},
    )
    assert verdict["cheating_category"] == "unparsed"
    assert "prose only" in verdict["cheating_rationale"]
    assert verdict["integrity_pct"] is None


def test_safe_mode_writes_json_and_md_outside_memory_dir(tmp_path, monkeypatch):
    mem = tmp_path / "memory"
    mem.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    out_json = artifacts / "integrity.json"

    verdict, captured = _run(
        monkeypatch, mem,
        {"success": True, "proposed_change": _GOOD_REPORT},
        safe_mode=True,
        output_json_path=str(out_json),
    )
    assert verdict["cheating_category"] == "no_cheating"
    assert json.loads(out_json.read_text())["cheating_category"] == "no_cheating"
    assert (artifacts / "run_integrity.md").exists()
    # Nothing may land in memory_dir while the run is live.
    assert list(mem.iterdir()) == []
    # And the prompt actually carried the safe-mode preamble.
    assert "SAFE MODE" in captured["prompt"]


def test_safe_mode_without_output_path_skips(tmp_path, monkeypatch):
    verdict, _ = _run(
        monkeypatch, tmp_path,
        {"success": True, "proposed_change": _GOOD_REPORT},
        safe_mode=True,
    )
    assert verdict is None


def test_non_fatal_on_agent_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))
    import src.engine.services.subprocess_service as svc

    def boom(**kwargs):
        raise RuntimeError("harness exploded")

    monkeypatch.setattr(svc, "run_inspection_agent", boom)
    verdict = ri.run_integrity_inspection(
        workspace="/workspace",
        method="ee",
        num_iterations=1,
        best_valset_score=None,
        seed_test_score=None,
        best_candidate_test_score=None,
        iteration_indices=[0],
        coding_harness="claude",
    )
    assert verdict is None


def test_none_on_empty_agent_output(tmp_path, monkeypatch):
    verdict, _ = _run(
        monkeypatch, tmp_path, {"success": False, "proposed_change": ""}
    )
    assert verdict is None


# ------------------------------------------------------- history + config


def test_integrity_keys_are_whitelisted_in_iteration_history():
    for key in ri.INTEGRITY_RECORD_KEYS:
        assert key in ITERATION_RECORD_FIELDS
    assert "clean_slate_status" in ITERATION_RECORD_FIELDS


def test_config_flags_round_trip():
    from src.engine.optimizer.v3.optimization_config import OptimizationConfig

    base = {
        "workspace": "/w",
        "program": "p.X",
        "metric": "m.f",
        "repo_url": "file:///w",
        "enable_run_integrity_agent": True,
        "enable_clean_slate_inspector": True,
    }
    cfg = OptimizationConfig.from_dict(base)
    assert cfg.enable_run_integrity_agent is True
    assert cfg.enable_clean_slate_inspector is True
    d = cfg.to_dict()
    assert d["enable_run_integrity_agent"] is True
    assert d["enable_clean_slate_inspector"] is True
    # Defaults are off (opt-in at the dataclass layer).
    cfg2 = OptimizationConfig.from_dict(
        {"workspace": "/w", "program": "p.X", "metric": "m.f", "repo_url": "u"}
    )
    assert cfg2.enable_run_integrity_agent is False
    assert cfg2.enable_clean_slate_inspector is False


def test_stamp_integrity_verdict_targets_latest_record():
    from src.engine.schemas.iteration_history import IterationHistory

    history = IterationHistory()
    for i in (0, 1, 2):
        history.create_entry(i)
    verdict = {
        "cheating_category": "non_impactful_cheating",
        "cheating_rationale": "listed the testset dir",
        "functioning_iterations": "2/2",
        "integrity_pct": 100.0,
    }
    assert ri.stamp_integrity_verdict(history, verdict) == 2
    assert history[2]["integrity_cheating_category"] == "non_impactful_cheating"
    assert history[2]["integrity_pct"] == 100.0
    # Earlier records untouched (create_entry pre-populates all fields None).
    assert history[0]["integrity_cheating_category"] is None
    assert history[1]["integrity_cheating_category"] is None


def test_stamp_integrity_verdict_noop_cases():
    from src.engine.schemas.iteration_history import IterationHistory

    history = IterationHistory()
    assert ri.stamp_integrity_verdict(history, {"cheating_category": "x"}) is None
    history.create_entry(0)
    assert ri.stamp_integrity_verdict(history, None) is None
    assert history[0]["integrity_cheating_category"] is None
