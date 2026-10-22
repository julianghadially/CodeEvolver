"""ASA seed / iteration-start / close hooks against a real git repo
(specs/archive-public/v3_asa.md §Seed, §Eligibility, §Rule set, §Definitions)."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest

from src.engine.evaluator.sandbox_client import EvalResult
from src.engine.optimizer.v3 import asa as asa_pkg  # noqa: F401  (import side effects none)
from src.engine.optimizer.v3.asa import asa_optimizer as AO
from src.engine.optimizer.v3.asa.prompt_sections import REQ_ELIGIBLE_NONPRIMARY, REQ_NEW_LINEAGE
from src.engine.optimizer.v3.ce_state import CEState
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.strategy import LoopServices
from src.engine.services.subprocess_service import git_get_sha, git_tree_signature

from tests.controls._arch_helpers import make_evaluator_config

MAIN = "codeevolver-t-main"
SEED = "codeevolver-t-seed"
REACT = "codeevolver-t-react"


def _git(ws: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ws), *args], check=True, capture_output=True, text=True).stdout.strip()


class _Cfg:
    max_score = 1.0
    subsample_size = 5
    asa_behavioral_contribution_constant = 0.10
    asa_behavioral_contribution_threshold_cap = 0.05
    asa_primary_stale_landed_iterations = 4
    job = None


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(tmp_path / "traces"))
    ws = tmp_path / "ws"
    ws.mkdir()
    _git(ws, "init", "-q", "-b", MAIN)
    _git(ws, "config", "user.email", "t@t")
    _git(ws, "config", "user.name", "t")
    (ws / "prog.py").write_text("x = 1\n")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-qm", "seed")
    state = CEState(seed_branch=MAIN, seed_sha=git_get_sha(str(ws)),
                    comparison_history_path=str(tmp_path / "history" / "comparison_set_history.jsonl"))
    services = LoopServices(
        workspace=str(ws), evaluator_config=make_evaluator_config(), trainset_ids=list(range(20)),
        comparison_ids=[1, 2, 3, 4], comparison_timeout=None, run_timestamp="t",
        rng=random.Random(0), max_eval_rows=1000, config=_Cfg(),
    )
    return ws, state, services


def _fake_loop_eval(scores_by_ref: dict[str, dict[str, float]], calls: list):
    """Stand-in for `run_loop_comparison_eval`: records an eval on ctx (with
    row scores) and appends the JSONL line, like the real binding does."""
    def fake(*, ctx, state, services, ref):
        calls.append(ref)
        rows = scores_by_ref[ref]
        ws = services.workspace
        subprocess.run(["git", "-C", ws, "checkout", "-q", ref], check=True)
        sha = git_get_sha(ws)
        tree = git_tree_signature(ws)
        score = sum(rows.values()) / len(rows)
        ctx.record_eval(sha, score, eval_sample="comparison", branch=ref, n_rows=len(rows), tree=tree,
                        row_scores=dict(rows), eval_id=f"e{len(calls)}", committed=True, by_loop=True)
        state.increment_evals(len(rows))
        state.record_comparison_eval(iteration=ctx.iteration_idx, branch=ref, sha=sha, score=score,
                                     n_rows=len(rows), row_scores=rows, eval_id=f"e{len(calls)}", tree=tree)
        return EvalResult(scores_by_metric={"accuracy": score}, primary_score=score, trace_meta=[],
                          trace_dir="", trace_index_file="", row_ids=list(range(len(rows))), raw={})
    return fake


def _ctx(state, i=2, **kw):
    base = dict(iteration_idx=i, subsample_size=5, comparison_set_size=4, max_eval_rows=1000,
                evals_used=int(state.total_num_evals), primary_branch_at_start=state.primary_branch,
                parent_locked=True)
    base.update(kw)
    return IterationContext(**base)


# --- seed ----------------------------------------------------------------------

def test_seed_registers_two_lineages_with_one_counted_eval(env, monkeypatch):
    ws, state, services = env
    calls = []
    monkeypatch.setattr(AO, "run_loop_comparison_eval",
                        _fake_loop_eval({MAIN: {"1": 1.0, "2": 0.0, "3": 1.0, "4": 0.0}}, calls))
    history = {0: {}}
    AO.asa_on_seed_complete(state, history, services)
    assert calls == [MAIN] and state.total_num_evals == 4
    assert set(state.branches) == {MAIN, SEED} and state.primary_branch == MAIN
    assert not state.branches[SEED]["is_primary"] and state.branches[SEED]["created_iteration"] == 0
    assert git_get_sha(str(ws), SEED) == git_get_sha(str(ws), MAIN)
    assert history[0]["comparison_score"] == 0.5
    # both lineages resolve to the single record by tree
    lin = AO.lineages_for(state, str(ws))
    assert lin[MAIN].up_to_date and lin[SEED].up_to_date
    assert lin[MAIN].record["eval_id"] == lin[SEED].record["eval_id"]
    assert len(Path(state.comparison_history_path).read_text().strip().splitlines()) == 1


def test_seed_requires_comparison_set(env):
    ws, state, services = env
    bad = LoopServices(**{**services.__dict__, "comparison_ids": []})
    with pytest.raises(RuntimeError, match="comparison set"):
        AO.asa_on_seed_complete(state, {0: {}}, bad)


# --- iteration start --------------------------------------------------------------

def _seeded(env, monkeypatch, react_rows):
    ws, state, services = env
    calls = []
    fake = _fake_loop_eval({MAIN: {"1": 1.0, "2": 0.0, "3": 1.0, "4": 0.0}, REACT: react_rows}, calls)
    monkeypatch.setattr(AO, "run_loop_comparison_eval", fake)
    AO.asa_on_seed_complete(state, {0: {}}, services)
    # a registered non-primary lineage with its own comparison record
    _git(ws, "checkout", "-qb", REACT)
    (ws / "prog.py").write_text("x = 2\n")
    _git(ws, "commit", "-qam", "react")
    fake(ctx=_ctx(state, 1), state=state, services=services, ref=REACT)
    state.register_lineage(REACT, sha=git_get_sha(str(ws)), iteration=1)
    state.iterations.append({"iteration": 1, "branch": REACT, "close_sha": git_get_sha(str(ws)),
                             "skipped": False, "new_lineage": True, "changes_landed": True,
                             "parent_branch": MAIN, "parent_is_primary": True})
    _git(ws, "checkout", "-q", MAIN)
    return ws, state, services, calls


def test_iteration_start_computes_eligible_set_requirements_and_table(env, monkeypatch):
    # react beats primary on rows 2 and 4 -> contribution 2/4 = 0.5 > threshold 0.05
    ws, state, services, _ = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    ctx = _ctx(state, 2)
    AO.asa_on_iteration_start(state, ctx, services)
    assert ctx.asa_eligible_at_start == [REACT]
    assert ctx.asa_eligible_contributions_at_start[REACT] == pytest.approx(0.5)
    assert ctx.asa_requirements == []            # 1 closed iteration: too early for either rule
    assert ctx.asa_phase1 and not ctx.asa_free_tail and not ctx.asa_primary_record_stale
    table = (Path(services.workspace).parent / "history" / "branch_comparison_iter2.md").read_text()
    assert f"`{REACT}`" in table and "| true | true | false |" in table
    assert f"`{SEED}`" in table


def test_iteration_start_dominated_lineage_is_not_eligible(env, monkeypatch):
    ws, state, services, _ = _seeded(env, monkeypatch, {"1": 1.0, "2": 0.0, "3": 0.0, "4": 0.0})
    ctx = _ctx(state, 2)
    AO.asa_on_iteration_start(state, ctx, services)
    assert ctx.asa_eligible_at_start == []


def test_branch_comparison_tool_reads_live_jsonl_and_errors_without_reference_record(env, monkeypatch):
    ws, state, services, _ = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    ctx = _ctx(state, 2)
    tools = AO.asa_extra_tools(ctx, state, str(ws), services)
    out = tools["branch_comparison"]()
    assert out["reference"] == MAIN and out["eligible"] == [REACT]
    rows = {r["branch"]: r for r in out["rows"]}
    assert rows[REACT]["n_rows_ahead"] == 2 and rows[REACT]["positive_behavioral_contribution"] == 2.0
    assert rows[REACT]["pareto_score"] == 2 and rows[MAIN]["pareto_score"] == 2
    assert rows[SEED]["pareto_dominated"] is False          # identical winning set to MAIN -> kept
    out2 = tools["branch_comparison"](reference=REACT)
    assert out2["rows"][0]["branch"] == REACT and out2["eligible"] == []
    from src.engine.optimizer.v3.errors import RunEvalError
    with pytest.raises(RunEvalError, match="no comparison-set record"):
        tools["branch_comparison"](reference="codeevolver-t-nothing")


# --- close -------------------------------------------------------------------------

def test_close_derives_satisfied_and_violation(env, monkeypatch):
    ws, state, services, calls = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    ctx = _ctx(state, 2, asa_requirements=[REQ_NEW_LINEAGE, REQ_ELIGIBLE_NONPRIMARY],
               asa_eligible_at_start=[REACT], asa_eligible_contributions_at_start={REACT: 0.5},
               asa_allowance_left=1, asa_phase1=True)
    ctx.child_branch = "codeevolver-t-react-v2"; ctx.new_lineage = True
    ctx.parent_branch = REACT; ctx.parent_is_primary = False
    rec = {"primary_branch_after": MAIN, "comparison_scores": []}
    AO.asa_on_iteration_closed(ctx, state, rec, services)
    assert rec["asa_satisfied"] and not rec["asa_violation"] and rec["asa_parent_is_eligible"]
    assert rec["asa_parent_positive_contribution_to_primary_normalized"] == 0.5
    assert rec["asa_loop_comparison_reason"] is None and calls[-1] != ctx.child_branch

    ctx2 = _ctx(state, 3, asa_requirements=[REQ_ELIGIBLE_NONPRIMARY], asa_eligible_at_start=[REACT])
    ctx2.child_branch = MAIN; ctx2.new_lineage = False; ctx2.parent_branch = MAIN; ctx2.parent_is_primary = True
    rec2 = {"primary_branch_after": MAIN, "comparison_scores": []}
    AO.asa_on_iteration_closed(ctx2, state, rec2, services)
    assert rec2["asa_violation"] and not rec2["asa_satisfied"]
    assert rec2["asa_parent_positive_contribution_to_primary_normalized"] is None


def test_close_promotion_without_tip_record_runs_one_counted_eval(env, monkeypatch):
    ws, state, services, calls = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    # move REACT past its record, then promote it
    _git(ws, "checkout", "-q", REACT)
    (ws / "prog.py").write_text("x = 3\n")
    _git(ws, "commit", "-qam", "react v2")
    ctx = _ctx(state, 2)
    ctx.child_branch = REACT; ctx.parent_branch = REACT; ctx.parent_is_primary = False
    rec = {"primary_branch_after": REACT, "comparison_scores": []}
    used = state.total_num_evals
    AO.asa_on_iteration_closed(ctx, state, rec, services)
    assert calls[-1] == REACT and rec["asa_loop_comparison_reason"] == "promotion"
    assert state.total_num_evals == used + 4 and state.loop_comparison_reruns == 1
    assert rec["comparison_scores"] and rec["comparison_scores"][-1]["branch"] == REACT


def test_close_promotion_with_program_equivalent_record_skips_eval(env, monkeypatch):
    ws, state, services, calls = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    _git(ws, "checkout", "-q", REACT)
    (ws / "codeevolver.md").write_text("docs only\n")            # non-program change
    _git(ws, "add", "-A"); _git(ws, "commit", "-qm", "docs")
    ctx = _ctx(state, 2); ctx.child_branch = REACT; ctx.parent_branch = REACT
    rec = {"primary_branch_after": REACT, "comparison_scores": []}
    n = len(calls)
    AO.asa_on_iteration_closed(ctx, state, rec, services)
    assert len(calls) == n and rec["asa_loop_comparison_reason"] is None


def test_close_promotion_skips_loop_eval_when_agent_ran_a_comparison(env, monkeypatch):
    """specs/archive-public/comparison_credits.md: the loop never triggers a second
    comparison run in an iteration where the agent already ran one."""
    ws, state, services, calls = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    _git(ws, "checkout", "-q", REACT)
    (ws / "prog.py").write_text("x = 3\n")
    _git(ws, "commit", "-qam", "react v2")
    ctx = _ctx(state, 2)
    ctx.child_branch = REACT; ctx.parent_branch = REACT; ctx.parent_is_primary = False
    # The agent's own (non-tip-matching) comparison run this iteration.
    ctx.record_eval("some_sha", 0.5, eval_sample="comparison", n_rows=4, tree="some_tree")
    rec = {"primary_branch_after": REACT, "comparison_scores": []}
    n = len(calls)
    AO.asa_on_iteration_closed(ctx, state, rec, services)
    assert len(calls) == n
    assert rec["asa_loop_comparison_reason"] == "skipped_agent_ran_comparison"


def test_close_stale_primary_refresh_skipped_when_agent_ran_a_comparison(env, monkeypatch):
    ws, state, services, calls = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    for i in range(2, 7):
        state.iterations.append({"iteration": i, "branch": MAIN, "close_sha": "s", "skipped": False,
                                 "changes_landed": True, "parent_branch": MAIN, "parent_is_primary": True})
    ctx = _ctx(state, 7)
    AO.asa_on_iteration_start(state, ctx, services)
    assert ctx.asa_primary_record_stale
    ctx.child_branch = MAIN; ctx.parent_branch = MAIN; ctx.parent_is_primary = True
    ctx.record_eval("some_sha", 0.5, eval_sample="comparison", n_rows=4, tree="some_tree")
    rec = {"primary_branch_after": MAIN, "comparison_scores": []}
    n = len(calls)
    AO.asa_on_iteration_closed(ctx, state, rec, services)
    assert len(calls) == n
    assert rec["asa_loop_comparison_reason"] == "skipped_agent_ran_comparison"


def test_close_after_constraint_revert_is_not_a_promotion(env, monkeypatch):
    ws, state, services, calls = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    _git(ws, "checkout", "-q", REACT)
    (ws / "prog.py").write_text("x = 3\n"); _git(ws, "commit", "-qam", "v2")
    ctx = _ctx(state, 2); ctx.child_branch = REACT; ctx.parent_branch = REACT
    # the loop resets primary_branch_after on a force-revert but leaves is_primary
    rec = {"primary_branch_after": MAIN, "is_primary": True, "comparison_scores": []}
    n = len(calls)
    AO.asa_on_iteration_closed(ctx, state, rec, services)
    assert len(calls) == n


def test_close_refreshes_stale_primary_but_not_in_free_tail(env, monkeypatch):
    ws, state, services, calls = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    # five landed iterations on the primary since its record (iteration 0)
    for i in range(2, 7):
        state.iterations.append({"iteration": i, "branch": MAIN, "close_sha": "s", "skipped": False,
                                 "changes_landed": True, "parent_branch": MAIN, "parent_is_primary": True})
    ctx = _ctx(state, 7)
    AO.asa_on_iteration_start(state, ctx, services)
    assert ctx.asa_primary_record_stale
    ctx.child_branch = MAIN; ctx.parent_branch = MAIN; ctx.parent_is_primary = True
    rec = {"primary_branch_after": MAIN, "comparison_scores": []}
    AO.asa_on_iteration_closed(ctx, state, rec, services)
    assert calls[-1] == MAIN and rec["asa_loop_comparison_reason"] == "stale_primary"
    # free tail: nothing
    ctx2 = _ctx(state, 8, evals_used=990)
    AO.asa_on_iteration_start(state, ctx2, services)
    assert ctx2.asa_free_tail
    ctx2.child_branch = MAIN; ctx2.parent_branch = MAIN; ctx2.parent_is_primary = True
    n = len(calls)
    AO.asa_on_iteration_closed(ctx2, state, {"primary_branch_after": MAIN, "comparison_scores": []}, services)
    assert len(calls) == n


def test_rules_fire_from_records_and_are_stamped(env, monkeypatch):
    ws, state, services, _ = _seeded(env, monkeypatch, {"1": 0.0, "2": 1.0, "3": 0.0, "4": 1.0})
    for i in range(2, 5):   # three more closes, none new, none on react
        state.iterations.append({"iteration": i, "branch": MAIN, "close_sha": "s", "skipped": False,
                                 "changes_landed": True, "new_lineage": False, "parent_branch": MAIN,
                                 "parent_is_primary": True, "asa_eligible_at_start": [REACT],
                                 "asa_parent_is_eligible": False})
    ctx = _ctx(state, 5)
    AO.asa_on_iteration_start(state, ctx, services)
    assert ctx.asa_requirements == [REQ_NEW_LINEAGE, REQ_ELIGIBLE_NONPRIMARY]   # 1/4 = 25%; 0/3
    assert ctx.asa_allowance_left == 1
