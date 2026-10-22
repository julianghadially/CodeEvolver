"""Copyright © 2026 440 Labs LLC

Unit tests for the held-out test set + the agent-facing test-set lock.

Covers the pieces that run without a sandbox:
  - `DatasetConfig.prepare` resolves an optional test split (engine 3: no
    validation set; the comparison set is a fixed trainset subsample).
  - `_build_holdout_deny_paths` fences the testset (+ trace suffixes).
  - `create_deny_paths_hook` denies tool calls referencing a deny path.
  - `_make_holdout_guard_hook` denies access to the test set / meta-traces /
    client-supplied extra paths.
  - `build_agent_config` forwards `deny_paths` into the coding-agent config.

The finalization-test-eval path is exercised by the local e2e
(`tests/functional/test_e2e.py`); these are the fast, isolated checks.
"""

from __future__ import annotations

import pytest

from src.engine.evaluator.dataset_config import DatasetConfig

def _seed_trace_dir(trace_dir):
    """Minimal eval trace directory: the registry checks for `index.jsonl`."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "index.jsonl").write_text("{}\n")



_ROWS = [{"q": f"q{i}", "label": f"a{i}"} for i in range(6)]


def test_dataset_config_resolves_optional_testset(tmp_path):
    cfg = DatasetConfig.prepare(
        str(tmp_path),
        trainset_json=_ROWS,
        trainset_path=None,
        testset_json=_ROWS[:3],
        testset_path=None,
    )
    assert cfg.has_testset is True
    assert cfg.testset_ids == [0, 1, 2]
    assert cfg.testset_size == 3
    assert cfg.testset_rel_path.endswith(".codeevolver/testset.json")


def test_dataset_config_testset_absent_by_default(tmp_path):
    cfg = DatasetConfig.prepare(
        str(tmp_path),
        trainset_json=_ROWS,
        trainset_path=None,
    )
    assert cfg.has_testset is False
    assert cfg.testset_ids == []
    assert cfg.testset_rel_path == ""


def test_dataset_config_comparison_set_is_fixed_trainset_subsample(tmp_path):
    cfg = DatasetConfig.prepare(
        str(tmp_path),
        trainset_json=_ROWS,
        trainset_path=None,
        comparison_set_size=3,
    )
    assert len(cfg.comparison_ids) == 3
    assert set(cfg.comparison_ids) <= set(cfg.trainset_ids)
    # Deterministic (seed 42): the same dataset always yields the same set.
    again = DatasetConfig.prepare(
        str(tmp_path),
        trainset_json=_ROWS,
        trainset_path=None,
        comparison_set_size=3,
    )
    assert again.comparison_ids == cfg.comparison_ids

    # Unset/0 disables comparison evals.
    none = DatasetConfig.prepare(
        str(tmp_path), trainset_json=_ROWS, trainset_path=None
    )
    assert none.comparison_ids == []
    assert none.comparison_set_size == 0


def test_build_holdout_deny_paths_covers_testset():
    from src.engine.optimizer.v3.iteration_architect import _build_holdout_deny_paths

    deny = _build_holdout_deny_paths(".codeevolver/testset.json")
    assert ".codeevolver/testset.json" in deny
    assert "testset.json" in deny
    assert "_holdout" in deny


def test_build_holdout_deny_paths_without_testset_still_fences_test_traces():
    from src.engine.optimizer.v3.iteration_architect import _build_holdout_deny_paths

    deny = _build_holdout_deny_paths(None)
    # Even with no test file, the trace suffix is always fenced.
    assert "_holdout" in deny
    assert all("testset.json" not in d for d in deny)


def test_build_holdout_deny_paths_appends_additional_deny_paths():
    from src.engine.optimizer.v3.iteration_architect import _build_holdout_deny_paths

    deny = _build_holdout_deny_paths(
        ".codeevolver/testset.json",
        ["secrets.env", "data/corpus/"],
    )
    # Built-in fences are preserved...
    assert ".codeevolver/testset.json" in deny
    assert "_holdout" in deny
    # ...and the client-supplied extras are appended.
    assert "secrets.env" in deny
    assert "data/corpus/" in deny


def test_build_holdout_deny_paths_default_unchanged_without_extras():
    """Omitting the extras arg must be byte-identical to before (no extras leak)."""
    from src.engine.optimizer.v3.iteration_architect import _build_holdout_deny_paths

    with_none = _build_holdout_deny_paths(".codeevolver/testset.json", None)
    with_empty = _build_holdout_deny_paths(".codeevolver/testset.json", [])
    assert with_none == with_empty


def _pre_tool_use(tool_name: str, **tool_input):
    return {"tool_name": tool_name, "tool_input": tool_input}


async def test_create_deny_paths_hook_denies_and_allows():
    from src.engine.agent.mounted.environment_setup import create_deny_paths_hook

    hook = create_deny_paths_hook(
        [".codeevolver/testset.json", "_holdout"], log_prefix="AGENT"
    )

    denied = await hook(
        _pre_tool_use("Read", file_path="/workspace/.codeevolver/testset.json"),
        "id",
        None,
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    bash_denied = await hook(
        _pre_tool_use("Bash", command="cat .codeevolver/testset.json"), "id", None
    )
    assert bash_denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    allowed = await hook(
        _pre_tool_use("Read", file_path="/workspace/src/program.py"), "id", None
    )
    assert allowed == {}


async def test_create_deny_paths_hook_noop_when_empty():
    from src.engine.agent.mounted.environment_setup import create_deny_paths_hook

    hook = create_deny_paths_hook([], log_prefix="AGENT")
    out = await hook(
        _pre_tool_use("Read", file_path="/workspace/.codeevolver/testset.json"),
        "id",
        None,
    )
    assert out == {}


async def test_architect_guard_fences_testset():
    from src.engine.optimizer.v3.architect_hooks import _make_holdout_guard_hook

    guard = _make_holdout_guard_hook(3, testset_path="data/testset.jsonl")

    test = await guard(
        _pre_tool_use("Read", file_path="data/testset.jsonl"), "id", None
    )
    assert test["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "test set" in test["hookSpecificOutput"]["permissionDecisionReason"].lower()

    grepped = await guard(
        _pre_tool_use("Grep", path="data", pattern="testset.jsonl"), "id", None
    )
    assert grepped["hookSpecificOutput"]["permissionDecision"] == "deny"

    ok = await guard(_pre_tool_use("Read", file_path="src/program.py"), "id", None)
    assert ok == {}


async def test_architect_guard_fences_its_own_meta_traces():
    """The architect's own session must not read its `*_architect_trace/`
    meta-traces — only the end-of-run meta-reflection agent reads those."""
    from src.engine.optimizer.v3.architect_hooks import _make_holdout_guard_hook

    guard = _make_holdout_guard_hook(5, testset_path="data/testset.jsonl")

    denied = await guard(
        _pre_tool_use(
            "Read", file_path="/architect_traces/iteration_5_architect_trace/turn_0003.json"
        ),
        "id",
        None,
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert (
        "meta-trace"
        in denied["hookSpecificOutput"]["permissionDecisionReason"].lower()
    )

    # The fence is always armed, even when no testset is configured.
    guard_no_holdout = _make_holdout_guard_hook(5)
    grepped = await guard_no_holdout(
        _pre_tool_use("Grep", path="/architect_traces", pattern="_architect_trace"),
        "id",
        None,
    )
    assert grepped["hookSpecificOutput"]["permissionDecision"] == "deny"


async def test_architect_guard_fences_client_additional_deny_paths():
    """The architect's own session honors client-supplied extra deny paths."""
    from src.engine.optimizer.v3.architect_hooks import _make_holdout_guard_hook

    guard = _make_holdout_guard_hook(
        2,
        testset_path="data/testset.jsonl",
        extra_deny_paths=["secrets.env"],
    )

    denied = await guard(
        _pre_tool_use("Read", file_path="/workspace/secrets.env"), "id", None
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert (
        "additional_deny_paths"
        in denied["hookSpecificOutput"]["permissionDecisionReason"].lower()
    )

    # An unrelated path still passes.
    ok = await guard(
        _pre_tool_use("Read", file_path="/workspace/src/program.py"), "id", None
    )
    assert ok == {}


def test_build_agent_config_forwards_deny_paths():
    from src.engine.agent.agent_config import build_agent_config

    cfg = build_agent_config(
        workspace_path="/workspace",
        change_request="do a thing",
        deny_paths=[".codeevolver/testset.json", "_holdout"],
    )
    assert cfg["deny_paths"] == [".codeevolver/testset.json", "_holdout"]

    cfg_none = build_agent_config(
        workspace_path="/workspace", change_request="do a thing"
    )
    assert "deny_paths" not in cfg_none


# --- The IMPLEMENTED architect arms the fence on its sub-agents --------------
#
# The tests above check the guard-hook *helpers*. These check that a real
# ``IterationArchitect.build_tool_registry`` actually forwards the holdout
# deny-paths into the coding_agent and inspect_traces sub-agents it spawns —
# i.e. the implemented agent arms the fence, not just a hand-built copy. The
# architect's own SDK session guard is exercised behaviorally (real session)
# in ``functional/test_valset_guard_behavioral.py``.

_TESTSET = ".codeevolver/testset.json"


def _holdout_request(tool_call, additional_deny_paths=None):
    """Run a registry tool against a RecordingHarness; return the HarnessRequest."""
    from tests.controls._arch_helpers import RecordingHarness, make_architect
    from src.engine.optimizer.v3 import agent_tools as agent_tools_module

    harness = RecordingHarness()
    arch, ctx, _ = make_architect(
        testset_path=_TESTSET,
        additional_deny_paths=additional_deny_paths,
    )
    registry = arch.build_tool_registry(agent_harness=harness)
    # The context is already locked (engine 3 loop lock); record a baseline
    # so coding_agent passes its controls and reaches the harness call.
    ctx.initial_subsample_score = 0.40
    # Stub the post-success git commit so coding_agent stays offline.
    import pytest as _pytest  # local import keeps module top clean

    _mp = _pytest.MonkeyPatch()
    _mp.setattr(agent_tools_module, "git_add_and_commit", lambda ws, msg: False)
    # `inspect_traces(trace="parent")` refuses to spawn when the baseline
    # trace file is absent; give it one.
    import tempfile as _tempfile
    from pathlib import Path as _Path

    traces = _Path(_tempfile.mkdtemp(prefix="ce-traces-"))
    _seed_trace_dir(traces / f"iteration_{ctx.iteration_idx}_parent")
    _mp.setenv("CODEEVOLVER_TRACES_DIR", str(traces))
    _mp.delenv("CODEEVOLVER_JOB_ID", raising=False)
    try:
        tool_call(registry)
    finally:
        _mp.undo()
    assert harness.requests, "expected the sub-agent harness to be invoked"
    return harness.requests[-1]


def _assert_fences_holdout(deny_paths):
    assert deny_paths is not None
    assert _TESTSET in deny_paths
    assert "testset.json" in deny_paths
    assert "_holdout" in deny_paths


def test_architect_fences_inspect_traces_subagent_against_holdout():
    req = _holdout_request(
        lambda registry: registry["inspect_traces"](query="why low score?", trace="parent")
    )
    _assert_fences_holdout(req.raw_provider_metadata.get("deny_paths"))


def test_architect_fences_coding_subagent_against_holdout():
    req = _holdout_request(
        lambda registry: registry["coding_agent"](
            mode="execute", change_request="add re-ranking"
        )
    )
    _assert_fences_holdout(req.raw_provider_metadata.get("deny_paths"))


def test_architect_forwards_additional_deny_paths_to_subagents():
    """Client extras reach the coding / inspect sub-agents on top of the
    built-in holdout fence."""
    for tool_call in (
        lambda registry: registry["inspect_traces"](query="why low score?", trace="parent"),
        lambda registry: registry["coding_agent"](
            mode="execute", change_request="add re-ranking"
        ),
    ):
        req = _holdout_request(tool_call, additional_deny_paths=["secrets.env"])
        deny = req.raw_provider_metadata.get("deny_paths")
        _assert_fences_holdout(deny)  # built-ins still present
        assert "secrets.env" in deny  # extras appended


def test_optimization_config_roundtrips_additional_deny_paths():
    from src.engine.optimizer.v3.optimization_config import OptimizationConfig

    cfg = OptimizationConfig(
        program="pkg.mod:Prog",
        metric="pkg.metrics:acc",
        additional_deny_paths=["secrets.env", "data/corpus/"],
    )
    restored = OptimizationConfig.from_dict(cfg.to_dict())
    assert restored.additional_deny_paths == ["secrets.env", "data/corpus/"]

    # Default (unset) stays None across the round-trip.
    plain = OptimizationConfig(program="pkg.mod:Prog", metric="pkg.metrics:acc")
    assert OptimizationConfig.from_dict(plain.to_dict()).additional_deny_paths is None
