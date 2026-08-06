"""
test_exit_state.py — Unit tests for ExitPipelineState.

Tests the dict-like interface and all sub-group field mappings.
"""

import pytest
from sack_counter.exit.exit_state import (
    ExitPipelineState, ExitSackState, ExitCountState,
)


class TestExitPipelineStateDictInterface:

    def test_getitem_sack_field(self, state):
        state.sacks.confirmed_sacks.add(42)
        assert 42 in state["confirmed_sacks"]

    def test_getitem_count_field(self, state):
        state.counts.total_sacks_out = 7
        assert state["total_sacks_out"] == 7

    def test_setitem_sack_field(self, state):
        state["sack_positions"] = {1: (100, 200)}
        assert state.sacks.sack_positions == {1: (100, 200)}

    def test_setitem_count_field(self, state):
        state["total_sacks_out"] = 5
        assert state.counts.total_sacks_out == 5

    def test_contains_mapped_key(self, state):
        assert "confirmed_sacks" in state
        assert "total_sacks_out" in state
        assert "exit_log" in state

    def test_contains_unmapped_key(self, state):
        assert "does_not_exist" not in state

    def test_get_existing_key(self, state):
        assert state.get("total_sacks_out", -1) == 0

    def test_get_missing_key_returns_default(self, state):
        assert state.get("nonexistent_key", 99) == 99

    def test_writing_an_unknown_key_raises(self, state):
        """
        Unmapped keys used to land in an ``_extra`` overflow dict, so a
        misspelled field silently became a new one while the correctly
        spelled field kept reading its default forever.  Writing one is
        now an error at the assignment that made the mistake.
        """
        with pytest.raises(KeyError):
            state["landing_pekk_count"] = 3

    def test_missing_key_raises(self, state):
        with pytest.raises(KeyError):
            _ = state["this_key_does_not_exist"]

    def test_reset_clears_all(self, state):
        state["total_sacks_out"] = 10
        state["confirmed_sacks"].add(1)
        state.reset()
        assert state["total_sacks_out"] == 0
        assert len(state["confirmed_sacks"]) == 0
        assert state.get("my_custom_key") is None

    def test_exit_log_is_list(self, state):
        assert isinstance(state["exit_log"], list)

    def test_crossed_sacks_is_set(self, state):
        assert isinstance(state["crossed_sacks"], set)

    def test_tentative_crossings_is_dict(self, state):
        assert isinstance(state["tentative_crossings"], dict)

    def test_discrepancy_flags_is_list(self, state):
        assert isinstance(state["discrepancy_flags"], list)

    def test_increment_total_sacks_out(self, state):
        state["total_sacks_out"] = state["total_sacks_out"] + 1
        state["total_sacks_out"] = state["total_sacks_out"] + 1
        assert state["total_sacks_out"] == 2
