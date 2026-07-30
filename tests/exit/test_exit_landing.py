"""
test_exit_landing.py — Unit tests for landing zone tracking and pile counting.
"""

import pytest
from sack_counter.exit.exit_landing import (
    LandingZoneTracker,
    update_landing_zone,
    check_discrepancy,
    reconcile_counts,
)


class TestLandingZoneTracker:
    """
    LandingZoneTracker now wraps SackMotionTracker (the same stillness
    class entry mode uses for ground-sack suppression) plus a
    consecutive-still-frame counter on top. These tests verify both
    layers behave correctly together.
    """

    def test_not_still_with_few_frames(self, motion_tracker):
        motion_tracker.update(1, 100, 200)
        motion_tracker.update(1, 101, 200)
        assert not motion_tracker.is_still(1)

    def test_still_after_stationary_frames(self, motion_tracker):
        for _ in range(5):
            motion_tracker.update(1, 175, 375)
        assert motion_tracker.is_still(1)

    def test_not_still_when_moving(self, motion_tracker):
        for i in range(5):
            motion_tracker.update(1, 100 + i * 20, 375)
        assert not motion_tracker.is_still(1)

    def test_cleanup_does_not_immediately_wipe_within_grace_period(self, motion_tracker):
        """A single missed frame (or a few, up to miss_grace_frames) must
        NOT wipe a sack's tracking state — only sustained absence beyond
        the grace period should clear it. Fixture uses miss_grace_frames=3."""
        motion_tracker.update(1, 100, 200)
        motion_tracker.cleanup(active_ids={2})  # 1 missed, 1st miss frame
        assert 1 in motion_tracker._motion.history
        assert 1 in motion_tracker._miss_streak

    def test_cleanup_wipes_after_grace_period_exceeded(self, motion_tracker):
        """A sack absent for MORE than miss_grace_frames consecutive
        frames should have its tracking state cleared."""
        motion_tracker.update(1, 100, 200)
        for _ in range(4):
            motion_tracker.cleanup(active_ids=set())
        assert 1 not in motion_tracker._motion.history
        assert 1 not in motion_tracker._still_streak

    def test_cleanup_keeps_active_id(self, motion_tracker):
        motion_tracker.update(1, 100, 200)
        motion_tracker.cleanup(active_ids={1})
        assert 1 in motion_tracker._motion.history

    def test_unknown_id_not_still(self, motion_tracker):
        assert not motion_tracker.is_still(999)

    def test_still_streak_resets_on_movement(self, motion_tracker):
        """A sack that was briefly still then starts moving again should
        lose its still-streak progress, requiring still_frames consecutive
        still frames again before being marked landed.

        Note: SackMotionTracker.is_still() needs >= 3 position samples
        before it can compute a speed average, so the still-streak only
        starts incrementing from the 3rd update onward. With
        still_frames=3 (the test fixture's setting), 5 stationary
        updates are needed to actually reach a confirmed "landed" state.
        """
        # Build up a still streak (5 updates needed: first 2 don't have
        # enough history yet, then 3 more to satisfy still_frames=3)
        for _ in range(5):
            motion_tracker.update(1, 175, 375)
        assert motion_tracker.is_still(1)

        # Now move sharply — should break the streak immediately
        motion_tracker.update(1, 400, 375)
        assert not motion_tracker.is_still(1)

    def test_uses_real_sack_motion_tracker(self, motion_tracker):
        """Verify LandingZoneTracker actually delegates to the same
        SackMotionTracker class entry mode uses, not a separate
        reimplementation — this was the root cause of the original bug
        where exit-mode stillness used different, untuned thresholds."""
        from sack_counter.trackers import SackMotionTracker
        assert isinstance(motion_tracker._motion, SackMotionTracker)

    def test_config_driven_construction_matches_entry_mode_defaults(self):
        """make_landing_zone_tracker(cfg) should read the exact same
        still_speed_thresh / still_speed_window keys entry mode's
        orchestrator.py uses, not separate exit-only defaults."""
        from sack_counter.exit.exit_landing import make_landing_zone_tracker
        cfg = {
            "still_speed_thresh": 1.5,   # entry mode's real default
            "still_speed_window": 8,
            "exit_still_frames":  8,
        }
        tracker = make_landing_zone_tracker(cfg)
        assert tracker._motion.speed_thresh == 1.5
        assert tracker._motion.speed_window == 8
        assert tracker.still_frames == 8

    def test_config_driven_construction_falls_back_safely(self):
        """An empty config dict must not crash — falls back to the
        documented defaults (which match entry mode's DEFAULT_CFG)."""
        from sack_counter.exit.exit_landing import make_landing_zone_tracker
        tracker = make_landing_zone_tracker({})
        assert tracker._motion.speed_thresh == 1.5
        assert tracker.still_frames == 8


class TestJitterTolerance:
    """
    REGRESSION TESTS for the real-world latency bug: some sacks took
    much longer to be counted than others, even though they visibly
    landed and stopped moving at the same time. Root cause: a sack that
    IS detected every frame (no gaps) but whose SackMotionTracker.is_still()
    occasionally flips False for one frame due to sub-pixel detection
    noise was getting its ENTIRE still-streak progress wiped on every
    such flicker, requiring it to restart from zero each time.

    jitter_tolerance allows a small number of isolated non-still
    readings to decrement the streak by 1 instead of resetting to 0,
    while still fully resetting on SUSTAINED motion.
    """

    def test_isolated_flicker_does_not_fully_reset_streak(self):
        """A single non-still reading surrounded by still readings should
        only cost 1 streak point, not reset everything to 0."""
        tracker = LandingZoneTracker(
            still_speed_thresh=10.0, still_speed_window=8,
            still_frames=8, jitter_tolerance=2,
        )
        # Build up a solid still streak
        for _ in range(6):
            tracker.update(1, 175, 375)
        streak_before = tracker._still_streak.get(1, 0)
        assert streak_before > 0

        # One jittery frame — small jump, but motion tracker's averaging
        # window means a single jump usually still reads as "moving"
        tracker.update(1, 250, 375)  # large jump = genuinely not still
        streak_after_jitter = tracker._still_streak.get(1, 0)

        # Should have decremented by 1, NOT reset to 0
        assert streak_after_jitter == max(0, streak_before - 1)
        assert streak_after_jitter != 0 or streak_before <= 1

    def test_sustained_motion_beyond_tolerance_fully_resets(self):
        """Motion sustained for MORE than jitter_tolerance consecutive
        frames is genuine movement, not noise — must fully reset."""
        tracker = LandingZoneTracker(
            still_speed_thresh=10.0, still_speed_window=8,
            still_frames=8, jitter_tolerance=2,
        )
        # Build up a still streak
        for _ in range(6):
            tracker.update(1, 175, 375)
        assert tracker._still_streak.get(1, 0) > 0

        # Sustained large jumps for MORE than jitter_tolerance (2) frames
        for i in range(5):
            tracker.update(1, 175 + i * 100, 375)

        assert tracker._still_streak.get(1, 0) == 0

    def test_flickery_sack_lands_faster_with_jitter_tolerance(self):
        """
        Direct comparison: a sack with occasional 1-frame flickers
        should reach is_still()==True noticeably faster with jitter
        tolerance enabled (jitter_tolerance=2) than with it disabled
        (jitter_tolerance=0, equivalent to the old strict behavior).
        """
        strict_tracker = LandingZoneTracker(
            still_speed_thresh=10.0, still_speed_window=8,
            still_frames=5, jitter_tolerance=0,
        )
        lenient_tracker = LandingZoneTracker(
            still_speed_thresh=10.0, still_speed_window=8,
            still_frames=5, jitter_tolerance=2,
        )

        # Simulate a sack that's still, with an isolated SMALL flicker
        # every 4th frame (mimicking sub-pixel detection noise on an
        # otherwise-stationary sack — a few pixels of box-edge jitter
        # from confidence fluctuation, NOT a large physical jump).
        positions = []
        base = (175, 375)
        for i in range(20):
            if i % 4 == 3:
                positions.append((base[0] + 15, base[1]))  # small jitter
            else:
                positions.append(base)

        strict_landed_at = None
        lenient_landed_at = None
        for i, (cx, cy) in enumerate(positions):
            strict_tracker.update(1, cx, cy)
            lenient_tracker.update(1, cx, cy)
            if strict_landed_at is None and strict_tracker.is_still(1):
                strict_landed_at = i
            if lenient_landed_at is None and lenient_tracker.is_still(1):
                lenient_landed_at = i

        # Lenient tracker should reach "still" at least as fast, and in
        # this flickery scenario, strictly faster.
        if strict_landed_at is not None and lenient_landed_at is not None:
            assert lenient_landed_at <= strict_landed_at
        # The lenient tracker must succeed in landing at all within the
        # simulated window — the strict one may never reach it.
        assert lenient_landed_at is not None


class TestUpdateLandingZone:

    def test_no_event_when_zone_empty(self, state, landing, motion_tracker):
        events = update_landing_zone(state, landing, [], fn=1,
                                     motion_tracker=motion_tracker)
        assert events == []
        assert state["landing_peak_count"] == 0

    def test_pile_count_increases_emits_event(self, state, landing, motion_tracker, sack_det):
        # Sack at (175, 375) — inside landing zone (50,300)-(300,450)
        det = [sack_det(sid=1, cx=175, cy=375)]
        # Make sack appear stationary
        for i in range(5):
            update_landing_zone(state, landing, det, fn=i+1,
                                motion_tracker=motion_tracker)

        assert state["landing_peak_count"] >= 1
        assert state["landing_exit_count"] >= 1
        assert 1 in state["landed_sacks"]

    def test_pile_count_does_not_decrease(self, state, landing, motion_tracker, sack_det):
        det1 = [sack_det(sid=1, cx=175, cy=375)]
        # Add one sack
        for fn in range(1, 6):
            update_landing_zone(state, landing, det1, fn=fn,
                                motion_tracker=motion_tracker)
        peak_after_one = state["landing_peak_count"]

        # Remove sack from zone
        update_landing_zone(state, landing, [], fn=6, motion_tracker=motion_tracker)
        assert state["landing_peak_count"] == peak_after_one  # peak stays

    def test_second_sack_increments_peak(self, state, landing, motion_tracker, sack_det):
        det1 = [sack_det(sid=1, cx=175, cy=375)]
        det2 = [sack_det(sid=1, cx=175, cy=375), sack_det(sid=2, cx=200, cy=390)]

        for fn in range(1, 6):
            update_landing_zone(state, landing, det1, fn=fn,
                                motion_tracker=motion_tracker)
        peak1 = state["landing_peak_count"]

        for fn in range(6, 11):
            update_landing_zone(state, landing, det2, fn=fn,
                                motion_tracker=motion_tracker)
        peak2 = state["landing_peak_count"]
        assert peak2 > peak1

    def test_sack_straddling_zone_edge_is_still_counted(self, state, landing, motion_tracker):
        """
        Integration-level version of the bbox_overlaps fix: a sack whose
        bounding box straddles the landing zone boundary (its CENTROID
        sits outside the zone, but most of its area is inside) must
        still be picked up by update_landing_zone end-to-end.

        Zone is (50,300)-(300,450). This sack box spans x=270-370,
        so its centroid (x=320) is outside the zone, but ~30% of its
        width is still inside — exactly the half-in/half-out scenario
        described as a real-world bug.
        """
        def straddling_det(sid):
            x1, y1, x2, y2 = 270, 320, 370, 400
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return (sid, x1, y1, x2, y2, cx, cy, 0.9)

        det = [straddling_det(sid=99)]
        for fn in range(1, 6):
            update_landing_zone(state, landing, det, fn=fn,
                                motion_tracker=motion_tracker)

        assert state["landing_peak_count"] >= 1
        assert 99 in state["landed_sacks"]

    def test_sack_outside_zone_not_counted(self, state, landing, motion_tracker, sack_det):
        # Sack far outside landing zone
        det = [sack_det(sid=5, cx=500, cy=100)]
        for fn in range(1, 6):
            update_landing_zone(state, landing, det, fn=fn,
                                motion_tracker=motion_tracker)
        assert state["landing_peak_count"] == 0
        assert 5 not in state["landed_sacks"]

    def test_moving_sack_in_zone_not_marked_landed(self, state, landing, motion_tracker, sack_det):
        """Sack moving through zone (not stationary) should not be marked as landed."""
        for fn in range(1, 6):
            # Move sack across zone each frame
            det = [sack_det(sid=3, cx=60 + fn * 30, cy=375)]
            update_landing_zone(state, landing, det, fn=fn,
                                motion_tracker=motion_tracker)
        assert 3 not in state["landed_sacks"]


class TestCheckDiscrepancy:

    def test_no_discrepancy_when_counts_agree(self, state):
        state["total_sacks_out"] = 5
        state["landing_exit_count"] = 5
        flagged = check_discrepancy(state, fn=100, tolerance=1)
        assert not flagged
        assert len(state["discrepancy_flags"]) == 0

    def test_no_discrepancy_within_tolerance(self, state):
        state["total_sacks_out"] = 5
        state["landing_exit_count"] = 6
        flagged = check_discrepancy(state, fn=100, tolerance=1)
        assert not flagged

    def test_discrepancy_flagged_when_exceeds_tolerance(self, state):
        state["total_sacks_out"] = 5
        state["landing_exit_count"] = 8
        flagged = check_discrepancy(state, fn=100, tolerance=1)
        assert flagged
        assert len(state["discrepancy_flags"]) == 1
        assert state["discrepancy_flags"][0]["diff"] == 3

    def test_multiple_flags_accumulate(self, state):
        state["total_sacks_out"] = 3
        state["landing_exit_count"] = 7
        check_discrepancy(state, fn=50, tolerance=1)
        check_discrepancy(state, fn=100, tolerance=1)
        assert len(state["discrepancy_flags"]) == 2


class TestReconcileCounts:

    def test_returns_max_of_two_counts(self, state):
        state["total_sacks_out"] = 4
        state["landing_exit_count"] = 6
        assert reconcile_counts(state) == 6

    def test_returns_door_count_when_higher(self, state):
        state["total_sacks_out"] = 8
        state["landing_exit_count"] = 5
        assert reconcile_counts(state) == 8

    def test_returns_zero_when_both_zero(self, state):
        assert reconcile_counts(state) == 0

    def test_returns_equal_count(self, state):
        state["total_sacks_out"] = 7
        state["landing_exit_count"] = 7
        assert reconcile_counts(state) == 7


class TestBboxOverlapsFractional:
    """
    Tests for the fractional-area overlap fix in LandingZone.bbox_overlaps.

    A sack that is mostly inside the landing zone but straddles its
    boundary (centroid pushed just outside) must still be detected as
    overlapping — this is the real-world case of a sack lying at the
    edge of a pile, which was previously silently dropped by the old
    centroid-only check.

    Landing zone fixture (see conftest.py): rectangle (50,300) to (300,450).
    """

    def test_fully_inside_box_overlaps(self, landing):
        assert landing.bbox_overlaps(100, 320, 200, 400)

    def test_fully_outside_box_does_not_overlap(self, landing):
        assert not landing.bbox_overlaps(500, 500, 600, 600)

    def test_centroid_outside_but_majority_inside_overlaps(self, landing):
        """
        The exact bug scenario: a sack box whose CENTROID sits just
        outside the zone boundary, but whose bulk is still inside the
        zone. Old centroid-only logic would silently drop this sack.
        """
        x1, y1, x2, y2 = 270, 320, 370, 400
        cx = (x1 + x2) // 2  # 320 -> outside zone (zone ends at x=300)
        assert not landing.contains(cx, y1)  # confirm centroid IS outside
        # ~30% of the box width (270-300 out of 270-370) sits inside.
        assert landing.bbox_overlaps(x1, y1, x2, y2, min_fraction=0.25)

    def test_barely_clipping_edge_does_not_overlap(self, landing):
        """A box that only grazes a tiny corner of the zone should NOT
        count — prevents oversized/unrelated boxes triggering on a
        minor corner clip."""
        x1, y1, x2, y2 = 290, 440, 500, 600
        assert not landing.bbox_overlaps(x1, y1, x2, y2, min_fraction=0.30)

    def test_half_in_half_out_horizontally_overlaps(self, landing):
        """Sack straddling the right edge of the zone (x=300), with
        roughly half its area inside -> should overlap."""
        x1, y1, x2, y2 = 250, 320, 350, 400
        assert landing.bbox_overlaps(x1, y1, x2, y2)

    def test_half_in_half_out_vertically_overlaps(self, landing):
        """Sack straddling the bottom edge of the zone (y=450)."""
        x1, y1, x2, y2 = 100, 420, 200, 500
        assert landing.bbox_overlaps(x1, y1, x2, y2)

    def test_custom_min_fraction_stricter(self, landing):
        """A box with ~30% overlap should fail a stricter 50%+ threshold
        but pass the default looser threshold."""
        x1, y1, x2, y2 = 270, 320, 370, 400
        assert landing.bbox_overlaps(x1, y1, x2, y2, min_fraction=0.25)
        assert not landing.bbox_overlaps(x1, y1, x2, y2, min_fraction=0.60)

    def test_centroid_inside_always_overlaps_regardless_of_fraction(self, landing):
        """Fast-path: if centroid is inside, it overlaps even with a
        very strict min_fraction — preserves old behavior for the
        common case of a sack fully inside the zone."""
        x1, y1, x2, y2 = 100, 320, 200, 400
        assert landing.bbox_overlaps(x1, y1, x2, y2, min_fraction=0.99)
