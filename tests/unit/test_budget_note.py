"""Copyright © 2026 440 Labs LLC

The architect budget note:
eval-row wording, the live remaining count, per-eval-type costs, and the
no-room-for-a-before/after endgame warning -- on the shared v3 builder only.
The simple loop keeps its own copy that states ONLY the percent used and the
rows remaining (Baseline 0 gets no planning guidance; specs/v3_simple_loop.md).
"""

from __future__ import annotations

from src.engine.optimizer.simple_loop.prompt import _budget_note as sl_note
from src.engine.optimizer.v3.prompts import _budget_note


# --- shared v3 builder (ee / freeform / pareto_evolution / asa) ---------------


def test_note_states_remaining_costs_and_room():
    note = _budget_note(2250, 1800, subsample_size=50, comparison_set_size=200)
    assert "450 of 2250 rows remain" in note
    assert "A train eval costs 50 rows" in note
    assert "a comparison eval costs 200 rows" in note
    assert "4×" in note or "≈4×" in note
    assert "`single` costs 1" in note
    assert "~9 more train eval(s)" in note
    assert "before/after" not in note  # plenty of room — no warning


def test_endgame_warning_when_no_room_for_a_pair():
    note = _budget_note(2250, 2180, subsample_size=50, comparison_set_size=200)
    assert "do not have room for a before/after train pair" in note
    assert "comparison set" in note


def test_comparison_clause_omitted_without_a_comparison_set():
    note = _budget_note(1000, 990, subsample_size=50, comparison_set_size=0)
    assert "comparison eval costs" not in note
    assert "do not have room for a before/after train pair" in note
    assert "comparison set" not in note


def test_v3_note_empty_without_budget():
    assert _budget_note(None, 5, subsample_size=50, comparison_set_size=0) == ""
    assert _budget_note(0, 0, subsample_size=50, comparison_set_size=0) == ""


# --- simple loop: percent used + rows remaining, nothing else -----------------

_GUIDANCE = ("train eval costs", "comparison eval", "before/after", "room for",
             "`single`")


def test_simple_loop_note_is_percent_and_rows_remaining_only():
    note = sl_note(2250, 1800)
    assert note == "You have used 80% of your evaluation budget; 450 of 2250 rows remain.\n\n"
    for phrase in _GUIDANCE:
        assert phrase not in note


def test_simple_loop_note_has_no_endgame_warning_when_nearly_spent():
    note = sl_note(1000, 990)
    assert "99% of your evaluation budget" in note
    assert "10 of 1000 rows remain" in note
    for phrase in _GUIDANCE:
        assert phrase not in note


def test_simple_loop_note_clamps_overshoot_to_zero():
    assert "0 of 100 rows remain" in sl_note(100, 130)


def test_simple_loop_note_empty_without_budget():
    assert sl_note(None, 5) == ""
    assert sl_note(0, 0) == ""
