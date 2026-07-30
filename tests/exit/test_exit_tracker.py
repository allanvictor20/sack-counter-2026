"""
test_exit_tracker.py — Unit tests for exit sack track confirmation logic.
"""

import pytest
from sack_counter.exit.exit_tracker import update_exit_sacks, cleanup_stale_tentatives


class TestUpdateExitSacks:

    def test_sack_not_confirmed_on_first_frame(self, state, sack_det):
        det = [sack_det(sid=1, cx=200, cy=350)]
        confirmed = update_exit_sacks(state, det, fn=1, confirm_frames=3)
        assert 1 not in state["confirmed_sacks"]
        assert len(confirmed) == 0

    def test_sack_confirmed_after_enough_hits(self, state, sack_det):
        det = [sack_det(sid=1)]
        for fn in range(1, 5):
            confirmed = update_exit_sacks(state, det, fn=fn, confirm_frames=3)
        assert 1 in state["confirmed_sacks"]
        assert any(s[0] == 1 for s in confirmed)

    def test_sack_evicted_after_miss_streak(self, state, sack_det):
        det = [sack_det(sid=1)]
        # Confirm the sack
        for fn in range(1, 6):
            update_exit_sacks(state, det, fn=fn, confirm_frames=3)
        assert 1 in state["confirmed_sacks"]

        # Now miss it for miss_frames+1 frames
        for fn in range(6, 6 + 10):
            update_exit_sacks(state, [], fn=fn, miss_frames=8)

        assert 1 not in state["confirmed_sacks"]

    def test_position_updated(self, state, sack_det):
        det = [sack_det(sid=2, cx=150, cy=200)]
        update_exit_sacks(state, det, fn=1)
        assert state["sack_positions"][2] == (150, 200)

    def test_multiple_sacks_tracked_independently(self, state, sack_det):
        det = [sack_det(sid=1, cx=100), sack_det(sid=2, cx=200)]
        for fn in range(1, 5):
            update_exit_sacks(state, det, fn=fn, confirm_frames=3)
        assert 1 in state["confirmed_sacks"]
        assert 2 in state["confirmed_sacks"]

    def test_hit_counter_increments(self, state, sack_det):
        det = [sack_det(sid=1)]
        update_exit_sacks(state, det, fn=1)
        update_exit_sacks(state, det, fn=2)
        assert state["sack_hit"].get(1, 0) == 2

    def test_miss_counter_resets_on_redetection(self, state, sack_det):
        det = [sack_det(sid=1)]
        # Confirm
        for fn in range(1, 5):
            update_exit_sacks(state, det, fn=fn, confirm_frames=3)
        # Miss 3 frames
        for fn in range(5, 8):
            update_exit_sacks(state, [], fn=fn)
        miss_before = state["sack_miss"].get(1, 0)
        # Redetect
        update_exit_sacks(state, det, fn=8)
        assert state["sack_miss"].get(1, 0) == 0

    def test_crossed_sacks_not_cleared_on_eviction(self, state, sack_det):
        """Eviction must NOT remove a sid from crossed_sacks."""
        state["crossed_sacks"].add(1)
        det = [sack_det(sid=1)]
        # Confirm then evict
        for fn in range(1, 5):
            update_exit_sacks(state, det, fn=fn, confirm_frames=3)
        for fn in range(5, 20):
            update_exit_sacks(state, [], fn=fn, miss_frames=8)
        assert 1 in state["crossed_sacks"]

    def test_returns_empty_list_for_no_detections(self, state):
        confirmed = update_exit_sacks(state, [], fn=1)
        assert confirmed == []


class TestCleanupStaleTentatives:

    def test_stale_tentative_removed(self, state):
        state["tentative_crossings"][99] = 1   # added at frame 1
        cleanup_stale_tentatives(state, fn=200, max_age=90)
        assert 99 not in state["tentative_crossings"]

    def test_fresh_tentative_kept(self, state):
        state["tentative_crossings"][99] = 150   # added at frame 150
        cleanup_stale_tentatives(state, fn=160, max_age=90)
        assert 99 in state["tentative_crossings"]

    def test_exactly_at_boundary_removed(self, state):
        state["tentative_crossings"][5] = 1
        cleanup_stale_tentatives(state, fn=92, max_age=90)  # age = 91 > 90
        assert 5 not in state["tentative_crossings"]
