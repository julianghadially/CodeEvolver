"""Copyright © 2026 440 Labs LLC

Replay of a recorded hover run through the engine-3 closing logic.

The unit tests pin each mechanism; this pins the OUTCOME on a recorded run
whose eval sequences exercise reverted variants, doc-only closes, redundant
re-evals and iterations that land nothing. Every eval sequence below is
transcribed from that run's log (score, sha and tree as `run_eval` recorded
them), and every expectation is the correct close for that sequence.

What a sha-only `finalize()` records, and what the correct close is:

    iter | recorded             | truth
    -----+----------------------+-----------------------------------------
      1  | skipped, 0.7694      | landed bc742547
      6  | closed, 0.7730→0.7878| nothing landed; 0.7878 was reverted code
      7  | 0.9114 + a warning   | 0.9308, no re-eval, no warning
     11  | skipped              | closed, landed nothing (correct outcome,
         |                      | wrong category)

Program equivalence is faked here rather than run against git: these tests are
about the closing ARITHMETIC, and `tests/unit/test_program_paths.py` covers the
predicate itself against a real repo.
"""

from __future__ import annotations

import pytest

from src.engine.optimizer.v3.closing import build_close_record, derive_iteration_parent
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.run_metrics import derive_run_metrics
from tests.controls._arch_helpers import FakeState


def _equivalent(doc_only_pairs: set[frozenset[str]] | None = None):
    """Program equivalence: identical, or declared doc-only-different."""
    pairs = doc_only_pairs or set()

    def fn(a, b):
        if not a or not b:
            return False
        return a == b or frozenset({a, b}) in pairs

    return fn


def _replay(
    *,
    iteration: int,
    parent_sha: str,
    parent_tree: str,
    close_sha: str,
    close_tree: str,
    evals: list[tuple[str, float, str, str]],
    branch: str = "codeevolver-hover-react-gap",
    equivalent=None,
) -> IterationContext:
    """Run one iteration's recorded evals through the deterministic close."""
    ctx = IterationContext(
        iteration_idx=iteration,
        parent_locked=True,
        start_branch=branch,
        start_sha=parent_sha,
        primary_branch_at_start=branch,
    )
    ctx.child_branch = branch
    ctx.registered_branch = {
        "branch": branch, "was_primary": True, "is_primary": True,
        "promotion_evidence": None,
    }
    ctx.close_called = True
    ctx.parent_sha, ctx.parent_tree = parent_sha, parent_tree
    ctx.close_sha, ctx.close_tree = close_sha, close_tree
    for sample, score, sha, tree in evals:
        ctx.record_eval(sha, score, eval_sample=sample, tree=tree, branch=branch)
        if sample == "train" and ctx.initial_subsample_score is None:
            ctx.initial_subsample_score = score
    ctx.finalize(equivalent or _equivalent())
    ctx.skipped = ctx.child_branch is None
    return ctx


def test_iteration_1_landed_the_runs_biggest_improvement():
    """Recorded as `skipped` with score 0.7694 — the score of discarded code.

    The architect evaluated three variants, kept the second (0.8285) and
    committed it as bc742547, which became the run's primary. The trailing eval
    of the third, discarded variant carried the seed's sha (dirty tree); the
    newest-first sha walk hit it, found the seed reachable, and called the
    whole iteration a fall-back to the parent.
    """
    ctx = _replay(
        iteration=1,
        parent_sha="5ec84d8d", parent_tree="dba41ecc",
        close_sha="bc742547", close_tree="43b026c4",
        branch="codeevolver-hover-decomp-rrf",
        evals=[
            ("train", 0.3992, "5ec84d8d", "dba41ecc"),
            ("train", 0.7300, "5ec84d8d", "28b5b7a7"),
            ("train", 0.8285, "5ec84d8d", "43b026c4"),  # kept + committed
            ("train", 0.7694, "5ec84d8d", "08f86beb"),  # tried, reverted
        ],
    )

    assert ctx.skipped is False
    assert ctx.changes_landed is True
    assert ctx.close_sha == "bc742547"
    assert ctx.best_subsample_score == pytest.approx(0.8285)
    assert ctx.best_subsample_tree == "43b026c4"
    # The closing content is the kept variant — a by-content lookup agrees,
    # and the discarded trailing eval never becomes the outcome.
    assert ctx.score_for_content(ctx.close_tree, _equivalent()) == (
        pytest.approx(0.8285), "exact_tree"
    )


def test_iteration_2_parent_is_iteration_1_not_the_seed():
    """The cascade: iteration 1's false skip removed it as a parent candidate.

    Filtering `skipped` records in `derive_iteration_parent` would attribute
    iteration 2 to the SEED (`parent_derivation="seed"`) even though it built
    directly on iteration 1's commit.
    """
    past = [
        {
            "iteration": 1,
            "close_sha": "bc742547",
            "final_sha": "bc742547",
            "branch": "codeevolver-hover-decomp-rrf",
            "skipped": False,
            "changes_landed": True,
        }
    ]

    parent = derive_iteration_parent(
        close_sha="cc3fc162",
        past_iterations=past,
        seed_sha="5ec84d8d",
        seed_branch="codeevolver-20260101140000-main",
        is_ancestor=lambda a, d: a == "bc742547" and d == "cc3fc162",
        merge_base=lambda a, b: None,
    )

    assert parent.via == "ancestor"
    assert parent.parent_sha == "bc742547"
    assert parent.parent_iteration == 1


def test_iteration_6_landed_nothing_despite_an_advanced_branch_tip():
    """Recorded as `closed (subsample 0.7730 -> 0.7878)`. Nothing landed.

    The branch tip 6e53a130 was the loop's own post-close codeevolver.md commit
    from iteration 5, so it already differed from the derived parent and the
    sha comparison read that as a change. The 0.7878 belonged to tree
    `67c76fb0`, which the architect reverted — the final comparison eval ran on
    the parent's tree `70f58904`.
    """
    ctx = _replay(
        iteration=6,
        parent_sha="6e53a130", parent_tree="70f58904",
        close_sha="6e53a130", close_tree="70f58904",
        evals=[
            ("train", 0.7730, "6e53a130", "70f58904"),
            ("train", 0.7878, "6e53a130", "b2737f6a"),
            ("train", 0.7878, "6e53a130", "67c76fb0"),
            ("comparison", 0.8372, "6e53a130", "67c76fb0"),
            ("comparison", 0.8422, "6e53a130", "70f58904"),  # back on the parent
        ],
    )

    assert ctx.changes_landed is False
    assert ctx.skipped is False, "it closed — it just did not move the program"
    assert ctx.score_for_content(ctx.close_tree, _equivalent())[0] == pytest.approx(0.7730)
    # The version it tried and rejected is preserved, so the record still says
    # what the iteration cost and what it learned.
    assert ctx.best_subsample_score == pytest.approx(0.7878)
    # Comparison best on the registered branch: the parent's re-measurement.
    assert ctx.comparison_best_score == pytest.approx(0.8422)


def test_iteration_7_keeps_its_score_through_the_doc_commit():
    """Recorded 0.9114, which was a redundant 50-row re-eval of identical code.

    The close gate rejected the codeevolver.md-only commit as unevaluated, so
    the architect re-ran the eval purely to satisfy it — 50 rows spent, a
    noisier number recorded, and a spurious revert-check warning because 0.9114
    < the 0.9308 already measured. The replay omits that forced re-eval: with a
    program-aware gate it never happens.
    """
    ctx = _replay(
        iteration=7,
        parent_sha="6e53a130", parent_tree="70f58904",
        close_sha="a4c123b1", close_tree="6947ccf2",  # doc-only on top of code
        evals=[
            ("train", 0.8715, "6e53a130", "70f58904"),
            ("train", 0.9308, "6e53a130", "8bbf33fe"),
            ("comparison", 0.8816, "6e53a130", "8bbf33fe"),
            ("comparison", 0.8717, "6e53a130", "70f58904"),
        ],
        equivalent=_equivalent({frozenset({"8bbf33fe", "6947ccf2"})}),
    )

    assert ctx.changes_landed is True
    assert ctx.close_sha == "a4c123b1"
    assert ctx.best_subsample_score == pytest.approx(0.9308)
    assert ctx.score_for_content(
        ctx.close_tree, _equivalent({frozenset({"8bbf33fe", "6947ccf2"})})
    ) == (pytest.approx(0.9308), "program_equivalent")


def test_iteration_11_closed_having_landed_nothing():
    """Recorded `skipped`. The revert was correct; the category was not.

    The architect merged the iter-5 reranker onto the primary, measured +0.059
    on train, proved from the traces that the reranker promoted zero golds
    across 150 examples, and reverted. That is a well-reasoned negative result,
    and it earns a change summary and a place in the run's metrics.
    """
    ctx = _replay(
        iteration=11,
        parent_sha="a4c123b1", parent_tree="6947ccf2",
        close_sha="a4c123b1", close_tree="6947ccf2",
        evals=[
            ("train", 0.8717, "a4c123b1", "6947ccf2"),
            ("train", 0.9308, "a4c123b1", "f506b409"),
            ("comparison", 0.8716, "a4c123b1", "f506b409"),
        ],
    )

    assert ctx.skipped is False
    assert ctx.changes_landed is False
    assert ctx.score_for_content(ctx.close_tree, _equivalent())[0] == pytest.approx(0.8717)
    assert ctx.best_subsample_score == pytest.approx(0.9308)

    record = build_close_record(ctx, FakeState())
    assert record["skipped"] is False
    assert record["changes_landed"] is False
    assert record["subsample_best"] == pytest.approx(0.9308)


def test_run_metrics_cover_the_whole_run_not_half_of_it():
    """`claimed_explore_fraction` was 0.375, computed over 8 of 17 records.

    The nine reverted iterations were filtered out as `skipped`, which also
    halved the explore guard's denominator — so the guard never fired despite
    the run's real exploration share sitting near 47%.
    """
    modes = {
        1: "exploration", 2: "exploitation", 3: "exploration", 4: "exploitation",
        5: "exploration", 6: "exploitation", 7: "exploitation", 8: "exploration",
        9: "exploration", 10: "exploitation", 11: "exploitation",
        12: "exploration", 13: "exploration", 14: "exploration",
        15: "exploitation", 16: "exploitation", 17: "exploitation",
    }
    landed = {1, 2, 3, 5, 6, 7, 14, 15, 16}
    records = [
        {
            "iteration": i,
            "branch": "codeevolver-hover-react-gap",
            "close_sha": f"sha{i}",
            "skipped": False,
            "changes_landed": i in landed,
            "claimed_improvement_mode": mode,
            "primary_branch_at_start": "codeevolver-hover-react-gap",
            "parent_is_primary": True,
        }
        for i, mode in modes.items()
    ]

    metrics = derive_run_metrics(records)

    assert metrics["iterations_closed"] == 17
    assert metrics["claimed_explore_fraction"] == pytest.approx(8 / 17)
    # The rejection rate is now visible instead of being filtered away.
    assert metrics["iterations_landed"] == 9
    assert metrics["land_rate"] == pytest.approx(9 / 17)


def test_architect_crash_is_the_only_thing_still_called_skipped():
    stub = {"iteration": 3, "skipped": True, "changes_landed": False, "error": "boom"}
    records = [
        {"iteration": 1, "close_sha": "a", "skipped": False, "changes_landed": True,
         "branch": "b", "claimed_improvement_mode": "exploration"},
        stub,
    ]

    assert derive_run_metrics(records)["iterations_closed"] == 1

    # And it is not a parent candidate: it never touched git.
    parent = derive_iteration_parent(
        close_sha="z",
        past_iterations=records,
        seed_sha="seed",
        seed_branch="seed-branch",
        is_ancestor=lambda a, d: a == "a",
        merge_base=lambda a, b: None,
    )
    assert parent.parent_sha == "a"
    assert parent.parent_iteration == 1
