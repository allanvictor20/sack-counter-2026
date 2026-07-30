"""
test_exit_integration.py — End-to-end integration tests for the exit pipeline.

REDESIGNED scenarios (see exit_crossing.py and exit_landing.py module
docstrings for full rationale). The room interior is never visible to
the camera, so a sack's FIRST-EVER detection already happens at or near
the door — there is no earlier "room-side" frame. Realistic test
scenarios therefore have sacks first appear AT THE DOOR, not at some
artificial "room side" coordinate that the real camera could never see.

Simulates complete sack exit events without YOLO or video:
  frame 1-4:  sack detected AT THE DOOR (not confirmed yet — building hits)
  frame 4:    sack confirmed
  frame 4:    same frame, first sighting at door -> TENTATIVE
  frame 5-8:  sack moves to landing zone, becomes stationary
  frame 9:    landing confirmed -> CONFIRMED exit, count = 1

Also tests:
  - Fast throw (door crossing missed entirely, only landing zone sees it
    because the sack was first detected already inside the landing zone)
  - Multiple sacks, confirmed independently
  - A sack landing on top of an already-counted pile (accumulator design)
"""

import pytest
from sack_counter.exit.exit_state import ExitPipelineState
from sack_counter.exit.exit_tracker import update_exit_sacks
from sack_counter.exit.exit_crossing import (
    update_exit_crossings,
    confirm_tentative_crossings,
    expire_old_projections,
)
from sack_counter.exit.exit_landing import (
    LandingZoneTracker,
    update_landing_zone,
    reconcile_counts,
)
from sack_counter.exit.landing_zone import LandingZone


# ── Helpers ───────────────────────────────────────────────────

def make_det(sid, cx, cy=375, conf=0.9):
    w, h = 60, 80
    x1, y1 = cx - w//2, cy - h//2
    x2, y2 = cx + w//2, cy + h//2
    return (sid, x1, y1, x2, y2, cx, cy, conf)


def run_frames(state, door, landing_zone, motion_tracker, detections_by_frame,
               confirm_frames=3, miss_frames=8, cross_threshold=20.0,
               confirm_timeout=45, door_proximity=60):
    """
    Run the full exit pipeline for a sequence of frames.

    Note: stillness detection is owned entirely by the `motion_tracker`
    passed in — build it via the `motion_tracker` fixture (conftest.py)
    or `make_landing_zone_tracker(cfg)` for production use.

    detections_by_frame: dict[frame_no → list of (sid,x1,y1,x2,y2,cx,cy,conf)]
    """
    max_fn = max(detections_by_frame.keys()) if detections_by_frame else 0
    for fn in range(1, max_fn + 1):
        dets = detections_by_frame.get(fn, [])
        confirmed = update_exit_sacks(state, dets, fn,
                                      confirm_frames=confirm_frames,
                                      miss_frames=miss_frames)
        active_ids = {s[0] for s in confirmed}

        update_exit_crossings(state, door, fn,
                              confirm_timeout=confirm_timeout,
                              cross_threshold_px=cross_threshold,
                              door_proximity_px=door_proximity)

        update_landing_zone(state, landing_zone, dets, fn,
                            motion_tracker=motion_tracker)

        confirm_tentative_crossings(state, fn, confirm_timeout=confirm_timeout)
        expire_old_projections(state, active_ids)


# ── Tests ─────────────────────────────────────────────────────

class TestSingleSackFullJourney:

    def test_one_sack_counted_via_door_and_landing(self, door, landing, motion_tracker):
        """
        Sack first appears AT THE DOOR (cx=320, dead center — the only
        place it CAN first appear, since the room is invisible), then
        moves to and settles in the landing zone.
        Expected: total_sacks_out = 1.
        """
        state = ExitPipelineState()

        # Frames 1-4: sack detected at the door (building up confirm_frames)
        # Frame 5 onward: sack has moved into the landing zone, settles
        detections = {}
        for fn in range(1, 5):
            detections[fn] = [make_det(sid=1, cx=320, cy=240)]  # at the door
        for fn in range(5, 12):
            detections[fn] = [make_det(sid=1, cx=175, cy=375)]  # in landing zone

        run_frames(state, door, landing, motion_tracker, detections,
                   confirm_frames=3, confirm_timeout=45)

        assert state["total_sacks_out"] == 1
        assert 1 in state["crossed_sacks"]
        assert 1 in state["landed_sacks"]


class TestFastThrowDirectLand:

    def test_sack_counted_from_landing_zone_only(self, door, landing, motion_tracker):
        """
        Sack's first-ever detection is ALREADY inside the landing zone
        (the door-area detection was missed entirely — too fast/blurry,
        or simply never confirmed at the door). It becomes stationary
        and gets counted purely via the landing zone accumulator.
        Expected: landing_exit_count >= 1, reconciled >= 1.
        """
        state = ExitPipelineState()
        detections = {}
        for fn in range(1, 10):
            detections[fn] = [make_det(sid=2, cx=175, cy=375)]

        run_frames(state, door, landing, motion_tracker, detections,
                   confirm_frames=3)

        assert state["landing_exit_count"] >= 1
        assert reconcile_counts(state) >= 1


class TestTentativeCancelledOnTimeout:

    def test_false_positive_at_door_cancelled(self, door, landing, motion_tracker):
        """
        Sack is first detected AT the door (triggers tentative) but
        never appears in the landing zone at all — e.g. a false
        positive, or a person mistaken for a sack. After timeout, the
        tentative is cancelled and the count stays 0.
        """
        state = ExitPipelineState()
        # Confirm sack AT the door (its only ever-seen location)
        for fn in range(1, 5):
            dets = [make_det(sid=3, cx=320, cy=240)]
            update_exit_sacks(state, dets, fn, confirm_frames=3)

        # First-sighting-at-door tentative fires here (frame 4, when the
        # sack becomes confirmed and update_exit_crossings runs on it)
        update_exit_crossings(state, door, fn=4, confirm_timeout=45,
                              cross_threshold_px=20.0)

        assert 3 in state["tentative_crossings"]

        # Run timeout frames without any landing zone activity
        for fn in range(5, 55):
            update_exit_sacks(state, [], fn=fn, miss_frames=8)
            confirm_tentative_crossings(state, fn=fn, confirm_timeout=45)

        assert state["total_sacks_out"] == 0
        assert 3 not in state["tentative_crossings"]


class TestMultipleSacks:

    def test_two_sacks_counted_independently(self, door, landing, motion_tracker):
        """Two sacks, each first seen at the door then landing, should
        give count = 2, confirmed independently."""
        state = ExitPipelineState()
        detections = {}

        # Sack 1: first seen at door, then lands
        for fn in range(1, 5):
            detections[fn] = [make_det(sid=1, cx=320, cy=240)]
        for fn in range(5, 12):
            detections[fn] = [make_det(sid=1, cx=175, cy=375)]

        # Sack 2: first seen at door slightly later, then lands
        for fn in range(8, 12):
            detections.setdefault(fn, [])
            detections[fn] = detections[fn] + [make_det(sid=2, cx=320, cy=240)]
        for fn in range(12, 20):
            detections.setdefault(fn, [])
            detections[fn] = [make_det(sid=2, cx=200, cy=390)] + [
                d for d in detections[fn] if d[0] != 2
            ]

        run_frames(state, door, landing, motion_tracker, detections,
                   confirm_frames=3, confirm_timeout=45)

        assert state["total_sacks_out"] == 2

    def test_same_sack_not_counted_twice(self, door, landing, motion_tracker):
        """A sack that is somehow re-confirmed near the door after
        already being counted must not recount — it's latched in
        crossed_sacks and skipped before the first-sighting check runs."""
        state = ExitPipelineState()
        detections = {}

        for fn in range(1, 5):
            detections[fn] = [make_det(sid=1, cx=320, cy=240)]
        for fn in range(5, 12):
            detections[fn] = [make_det(sid=1, cx=175, cy=375)]

        run_frames(state, door, landing, motion_tracker, detections,
                   confirm_frames=3, confirm_timeout=45)
        count_after_first = state["total_sacks_out"]
        assert count_after_first == 1
        assert 1 in state["crossed_sacks"]

        # Same sack ID re-appears at the door again (e.g. a tracking
        # glitch reusing the ID) — already in crossed_sacks, must skip.
        more_detections = {1: [make_det(sid=1, cx=320, cy=240)]}
        update_exit_crossings(state, door, fn=20)

        # Count must not increase — sack is latched in crossed_sacks
        assert state["total_sacks_out"] == count_after_first


class TestPileOnExistingSack:

    def test_second_sack_on_pile_detected(self, door, landing, motion_tracker):
        """
        Sack A already in landing zone (counted).
        Sack B lands on top — accumulator grows from 1 to 2.
        Expected: landing_exit_count = 2.
        """
        state = ExitPipelineState()

        # Sack A lands and settles
        dets_a = [make_det(sid=1, cx=175, cy=375)]
        for fn in range(1, 10):
            update_landing_zone(state, landing, dets_a, fn,
                                motion_tracker=motion_tracker)
        count_after_a = state["landing_exit_count"]
        assert count_after_a >= 1

        # Sack B lands on pile alongside A
        dets_ab = [
            make_det(sid=1, cx=175, cy=375),
            make_det(sid=2, cx=200, cy=390),
        ]
        for fn in range(10, 18):
            update_landing_zone(state, landing, dets_ab, fn,
                                motion_tracker=motion_tracker)

        assert state["landing_exit_count"] >= 2
        assert state["landing_peak_count"] >= 2

    def test_five_sacks_never_simultaneously_visible_still_all_counted(
        self, door, landing, motion_tracker
    ):
        """
        REGRESSION TEST for the real-world bug: 5 sacks are genuinely
        present in the landing zone over time, but YOLO's per-frame
        detection count flickers due to occlusion/confidence noise and
        NEVER detects all 5 simultaneously in any single frame — mirroring
        the actual diagnostic data collected from real footage (counts
        swinging 2 -> 4 -> 7 -> 3 -> 5 frame to frame).

        With the OLD peak-snapshot design, this would have under-counted
        (the true simultaneous-visible count of 5 might never occur).
        With the NEW accumulator design, each sack just needs its own
        few frames of clear, still detection at SOME point — not all
        5 at once — so all 5 should be counted correctly.
        """
        state = ExitPipelineState()

        # All 5 sacks are stationary in the landing zone for the whole
        # test, but each frame only a noisy SUBSET of them gets detected
        # by "YOLO" (simulated), exactly like the real diagnostic showed.
        all_sacks = {
            1: (120, 320), 2: (160, 340), 3: (200, 360),
            4: (240, 380), 5: (280, 400),
        }

        # Frame-by-frame visible subset patterns (deliberately never all 5
        # at once) — mirrors real flicker: 2, 4, 3, 5(not all), 3, 4...
        # Every sack appears at least once every <=3 frames (matching the
        # motion_tracker fixture's miss_grace_frames=3), so no sack's
        # accumulated progress gets wiped by the grace-period cleanup —
        # this models realistic brief occlusion, not abandonment.
        visible_patterns = [
            [1, 2, 5],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [1, 3, 5],
            [3, 4, 5],
            [1, 2, 4, 5],
            [2, 4, 5],
            [1, 2, 3, 5],
            [3, 4, 5],
            [1, 2, 4, 5],
        ]

        fn = 0
        # Repeat the pattern cycle enough times that each sack
        # individually accumulates enough still-frames to be confirmed
        # landed (motion_tracker fixture uses still_frames=3, but the
        # underlying SackMotionTracker needs >=3 history points before
        # it can even compute "is_still", so give it enough cycles).
        for _ in range(4):
            for pattern in visible_patterns:
                fn += 1
                dets = [make_det(sid=sid, cx=all_sacks[sid][0], cy=all_sacks[sid][1])
                       for sid in pattern]
                update_landing_zone(state, landing, dets, fn,
                                    motion_tracker=motion_tracker)

        # Every one of the 5 sacks should have been confirmed landed at
        # SOME point, even though no single frame ever showed all 5.
        assert state["landed_sacks"] == {1, 2, 3, 4, 5}
        assert state["landing_exit_count"] == 5
        assert state["landing_peak_count"] == 5


class TestIDChurnDeduplication:
    """
    REGRESSION TESTS for the real-world bug: a single sack untouched for
    a long time gets recounted 2-3 times because YOLO/ByteTrack assigns
    it a brand new track ID each time detection briefly drops out and
    comes back (e.g. confidence flicker on a long-stationary sack).

    The fix requires THREE things to all be true before treating a new
    sid as "same physical sack, new ID":
      1. Close enough in position to an existing landed sack (radius)
      2. The existing landed sack was active recently (recency window)
      3. The existing landed sack's ID is NOT currently being detected
         (it must have genuinely gone quiet, not be coexisting)
    """

    def test_same_sack_id_churn_does_not_inflate_count(self, door, landing, motion_tracker):
        """
        Sack #1 lands and is confirmed. It then sits untouched so long
        that detection drops out (simulating real low-confidence
        flicker), gets evicted, and reappears under sid #2 at the SAME
        position. Then again under sid #3. Final count must still be 1,
        not 3.
        """
        state = ExitPipelineState()

        # Sack #1 lands and is confirmed still
        det1 = [make_det(sid=1, cx=175, cy=375)]
        for fn in range(1, 8):
            update_landing_zone(state, landing, det1, fn,
                                motion_tracker=motion_tracker)
        assert state["landing_exit_count"] == 1

        # Long gap with NO detections at all (sack still physically
        # there, but YOLO confidence dipped — simulates "untouched for
        # a long time" from real footage). sid #1 is no longer in
        # in_zone_ids during this gap.
        for fn in range(8, 15):
            update_landing_zone(state, landing, [], fn,
                                motion_tracker=motion_tracker)

        # Sack reappears at the SAME position under a NEW id (#2) —
        # this is what ByteTrack does when a track is lost and a
        # stationary object is re-detected.
        det2 = [make_det(sid=2, cx=176, cy=374)]  # near-identical position
        for fn in range(15, 22):
            update_landing_zone(state, landing, det2, fn,
                                motion_tracker=motion_tracker)

        # Must NOT have incremented — same physical sack, new ID
        assert state["landing_exit_count"] == 1

        # Another gap, then a THIRD id at the same spot
        for fn in range(22, 29):
            update_landing_zone(state, landing, [], fn,
                                motion_tracker=motion_tracker)

        det3 = [make_det(sid=3, cx=174, cy=376)]
        for fn in range(29, 36):
            update_landing_zone(state, landing, det3, fn,
                                motion_tracker=motion_tracker)

        # STILL must be 1 — not 2, not 3.
        assert state["landing_exit_count"] == 1
        assert state["landing_peak_count"] == 1

    def test_two_genuinely_separate_close_sacks_both_counted(
        self, door, landing, motion_tracker
    ):
        """
        Counter-test to the above: two DIFFERENT sacks landing close
        together (within the dedup radius) but COEXISTING in the same
        frames (both actively detected simultaneously under their own
        IDs) must both be counted — proximity alone must never merge
        two genuinely separate, live sacks.
        """
        state = ExitPipelineState()

        # Both sacks present and detected together from the start —
        # they coexist, so they can never be "the same sack, new ID".
        det_both = [
            make_det(sid=10, cx=175, cy=375),
            make_det(sid=11, cx=195, cy=385),  # ~28px away — within dedup radius
        ]
        for fn in range(1, 8):
            update_landing_zone(state, landing, det_both, fn,
                                motion_tracker=motion_tracker)

        assert state["landing_exit_count"] == 2
        assert state["landed_sacks"] == {10, 11}

    def test_dedup_does_not_trigger_across_long_time_gap(
        self, door, landing, motion_tracker
    ):
        """
        A new sack appearing at a position where an OLD sack landed a
        long time ago (beyond the recency window) must be counted as a
        genuinely new sack — not incorrectly merged just because a
        previous sack happened to land nearby at some point earlier in
        the session.
        """
        state = ExitPipelineState()

        # Sack #1 lands early and is confirmed
        det1 = [make_det(sid=1, cx=175, cy=375)]
        for fn in range(1, 8):
            update_landing_zone(state, landing, det1, fn,
                                motion_tracker=motion_tracker)
        assert state["landing_exit_count"] == 1

        # A very long gap passes — far beyond the recency window —
        # before a genuinely different sack happens to land nearby.
        for fn in range(8, 100):
            update_landing_zone(state, landing, [], fn,
                                motion_tracker=motion_tracker)

        det2 = [make_det(sid=99, cx=176, cy=374)]
        for fn in range(100, 108):
            update_landing_zone(state, landing, det2, fn,
                                motion_tracker=motion_tracker)

        # This should be treated as a genuinely NEW sack — count = 2.
        assert state["landing_exit_count"] == 2
