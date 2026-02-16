"""Unit tests for GEPA state history change detection.

Tests that _detect_change correctly identifies seed, prompt, and code changes
by comparing each candidate to its parent using only the candidate JSON dicts.

Run:  pytest tests/unit_test.py -v
"""

import pytest

from src.optimizer.gepa_state import GEPAStateRecord
from tests.test_data.gepa_state_data import (
    CANDIDATE_0_SEED,
    CANDIDATE_1_PROMPT_QUERY,
    CANDIDATE_2_PROMPT_SUMMARIZE,
    CANDIDATE_3_CODE_CHANGE,
    CANDIDATE_SCORES,
    CODE_BRANCH,
    MAIN_BRANCH,
    PARENT_PROGRAMS,
    PROGRAM_CANDIDATES,
)


@pytest.fixture
def state_record() -> GEPAStateRecord:
    """Build a GEPAStateRecord from the test trajectory."""
    return GEPAStateRecord(
        program_candidates=PROGRAM_CANDIDATES,
        candidate_scores=CANDIDATE_SCORES,
        parent_programs=PARENT_PROGRAMS,
        num_iterations=4,
        total_evals=120,
    )


# ------------------------------------------------------------------
# _detect_change tests — one per candidate type
# ------------------------------------------------------------------


class TestGEPAStateDetectChange:
    """Verify GEPAState._detect_change returns the correct (type, description, branch)."""

    def test_seed_candidate(self, state_record: GEPAStateRecord):
        change_type, desc, branch = state_record._detect_change(
            0, CANDIDATE_0_SEED
        )
        assert change_type == "seed"
        assert "seed" in desc.lower() or "initial" in desc.lower()
        assert branch == MAIN_BRANCH

    def test_prompt_change_at_create_query_hop2(
        self, state_record: GEPAStateRecord
    ):
        change_type, desc, branch = state_record._detect_change(
            1, CANDIDATE_1_PROMPT_QUERY
        )
        assert change_type == "prompt"
        assert "program.create_query_hop2" in desc
        # Should NOT mention other components that didn't change
        assert "summarize1" not in desc
        assert "generate_answer" not in desc
        # Branch unchanged from seed
        assert branch == MAIN_BRANCH

    def test_prompt_change_at_summarize1(self, state_record: GEPAStateRecord):
        change_type, desc, branch = state_record._detect_change(
            2, CANDIDATE_2_PROMPT_SUMMARIZE
        )
        assert change_type == "prompt"
        assert "program.summarize1" in desc
        # Only summarize1 changed relative to parent (idx 1)
        assert "create_query_hop2" not in desc
        assert branch == MAIN_BRANCH

    def test_code_change_adds_module(self, state_record: GEPAStateRecord):
        change_type, desc, branch = state_record._detect_change(
            3, CANDIDATE_3_CODE_CHANGE
        )
        assert change_type == "code"
        # Description should contain the change_request content
        assert "ExtractKeyFacts" in desc
        # Branch should be the new code branch
        assert branch == CODE_BRANCH
    def test_missing_parent_index(self):
        """Parent index beyond candidate list → 'unknown'."""
        record = GEPAStateRecord(
            program_candidates=[CANDIDATE_0_SEED],
            candidate_scores=[0.45],
            parent_programs=[[99]],  # bogus parent
            num_iterations=1,
            total_evals=10,
        )
        # idx 0 with parent [99] but idx 0 is treated as seed
        # Let's use a non-zero idx scenario
        record.program_candidates = [CANDIDATE_0_SEED, CANDIDATE_1_PROMPT_QUERY]
        record.candidate_scores = [0.45, 0.48]
        record.parent_programs = [[None], [99]]  # idx 1 → parent 99 doesn't exist
        change_type, desc, _ = record._detect_change(1, CANDIDATE_1_PROMPT_QUERY)
        assert change_type == "unknown"
        assert "Parent not found" in desc

    def test_no_code_field(self):
        """Candidate missing _code entirely → seed if idx 0."""
        bare_candidate = {"program.summarize1": "Some instruction"}
        record = GEPAStateRecord(
            program_candidates=[bare_candidate],
            candidate_scores=[0.30],
            parent_programs=[[None]],
            num_iterations=1,
            total_evals=5,
        )
        change_type, desc, branch = record._detect_change(0, bare_candidate)
        assert change_type == "seed"
        assert branch is None

    def test_identical_candidate_to_parent(self):
        """Candidate identical to parent → 'unknown'."""
        record = GEPAStateRecord(
            program_candidates=[CANDIDATE_0_SEED, CANDIDATE_0_SEED],
            candidate_scores=[0.45, 0.45],
            parent_programs=[[None], [0]],
            num_iterations=2,
            total_evals=20,
        )
        change_type, desc, _ = record._detect_change(1, CANDIDATE_0_SEED)
        assert change_type == "unknown"
        assert "No detectable change" in desc


# ------------------------------------------------------------------
# to_history_dict tests — full trajectory structure
# ------------------------------------------------------------------


class TestGEPAStateToHistoryDict:
    """Verify to_history_dict builds correct per-candidate records."""

    def test_history_has_all_candidates(self, state_record: GEPAStateRecord):
        history = state_record.to_history_dict()
        assert set(history.keys()) == {"0", "1", "2", "3"}

    def test_history_scores(self, state_record: GEPAStateRecord):
        history = state_record.to_history_dict()
        for idx, expected_score in enumerate(CANDIDATE_SCORES):
            assert history[str(idx)]["score"] == expected_score

    def test_history_parent_chain(self, state_record: GEPAStateRecord):
        history = state_record.to_history_dict()
        assert history["0"]["parent_candidates"] == [None]
        assert history["1"]["parent_candidates"] == [0]
        assert history["2"]["parent_candidates"] == [1]
        assert history["3"]["parent_candidates"] == [2]

    def test_history_change_types(self, state_record: GEPAStateRecord):
        history = state_record.to_history_dict()
        assert history["0"]["change_type"] == "seed"
        assert history["1"]["change_type"] == "prompt"
        assert history["2"]["change_type"] == "prompt"
        assert history["3"]["change_type"] == "code"

    def test_history_branches(self, state_record: GEPAStateRecord):
        history = state_record.to_history_dict()
        # Seed and prompt changes share the main branch
        assert history["0"]["git_branch"] == MAIN_BRANCH
        assert history["1"]["git_branch"] == MAIN_BRANCH
        assert history["2"]["git_branch"] == MAIN_BRANCH
        # Code change has a new branch
        assert history["3"]["git_branch"] == CODE_BRANCH

    def test_history_includes_candidate_dict(
        self, state_record: GEPAStateRecord
    ):
        history = state_record.to_history_dict()
        assert history["0"]["candidate"] is CANDIDATE_0_SEED
        assert history["3"]["candidate"] is CANDIDATE_3_CODE_CHANGE

    def test_code_change_description_contains_change_request(
        self, state_record: GEPAStateRecord
    ):
        history = state_record.to_history_dict()
        desc = history["3"]["change_description"]
        assert "ExtractKeyFacts" in desc

    def test_prompt_change_description_names_component(
        self, state_record: GEPAStateRecord
    ):
        history = state_record.to_history_dict()
        assert "program.create_query_hop2" in history["1"]["change_description"]
        assert "program.summarize1" in history["2"]["change_description"]

