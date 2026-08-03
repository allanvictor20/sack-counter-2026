"""
test_regressions.py — Regression tests for the code-review fixes.

Each test here pins a bug that was found by reading the code and was
silently wrong at runtime: no test covered it, and in most cases the
symptom was a miscount rather than a crash, so nothing surfaced it.

One class per bug, named for what it protects.
"""

import unittest
from unittest.mock import MagicMock

import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from sack_counter.config import DEFAULT_CFG
from sack_counter.confidence import ConfidenceTracker
from sack_counter.door_polygon import DoorPolygon
from sack_counter.trackers import GhostSacks, SackMotionTracker
from sack_counter.pipeline.state import PipelineState
from sack_counter.pipeline.state_machine import (
    SackState, SackStateMachineRegistry,
)
from sack_counter.pipeline.door_crossing import update_door_zone
from sack_counter.pipeline.counting import update_peak_counts
from sack_counter.pipeline.reporter import _int_keyed


def _make_door():
    """200x200 square centred on (200, 200). Room is below (+y)."""
    return DoorPolygon(
        points=[(100, 100), (300, 100), (300, 300), (100, 300)],
        room_point=(200, 400),
    )


def _make_session(door=None, cfg_overrides=None):
    cfg = dict(DEFAULT_CFG)
    cfg["door_cross_threshold_px"] = 30
    if cfg_overrides:
        cfg.update(cfg_overrides)

    state = PipelineState()
    state["confirmed_persons"].add(1)
    state["person_prev_cx"][1] = 200
    state["person_boxes"][1]   = (150, 150, 250, 250)

    session = MagicMock()
    session.state  = state
    session.door   = door or _make_door()
    session.cfg    = cfg
    session.logger = MagicMock()
    return session


def _sack(sid, cx, cy, conf=0.9):
    w, h = 40, 60
    return (sid, cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2, cx, cy, conf)


def _stamp(session, sid, pid, cx, cy):
    state = session.state
    state["raw_sacks"].append(_sack(sid, cx, cy))
    state["confirmed_sacks"].add(sid)
    state["sack_carrier_stamp"][sid] = pid
    state["sack_scores"][sid] = 0.9


class TestCrossingIsDirectional(unittest.TestCase):
    """
    The crossing test used abs(projection), so a sack sitting `threshold`
    px on the CORRIDOR side of the door also satisfied it — and
    is_on_approach_side() returns False for anything inside the polygon,
    so a carrier merely standing in a wide doorway was committed before
    they had gone through.  Only room-side travel may count.
    """

    def test_room_side_crossing_commits(self):
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        _stamp(session, sid=1, pid=1, cx=200, cy=290)   # proj = +90
        update_door_zone(session, fn=10)
        self.assertEqual(session.state["total_sacks_counted"], 2)

    def test_corridor_side_does_not_commit(self):
        """proj = -90: same magnitude, wrong side. Must NOT count."""
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        _stamp(session, sid=1, pid=1, cx=200, cy=110)
        update_door_zone(session, fn=10)
        self.assertEqual(session.state["total_sacks_counted"], 0)
        self.assertNotIn(1, session.state["persons_past_door"])


class TestDeliveredSacksRecorded(unittest.TestCase):
    """
    person_sack_delivered was read in four places but never written, so
    every per-carrier "delivered" figure in the HUD and the report was
    permanently 0.
    """

    def test_commit_records_delivered_sack_for_carrier(self):
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        _stamp(session, sid=4, pid=1, cx=200, cy=290)
        update_door_zone(session, fn=10)
        self.assertIn(4, session.state["person_sack_delivered"][1])


class TestOrphanRescue(unittest.TestCase):
    """
    timeout_persons snapshots orphaned_peaks / orphaned_sack_owner for a
    carrier whose track dies in the doorway, but nothing consumed them —
    the load was measured and then silently dropped.
    """

    def test_orphaned_peak_committed_when_its_sack_crosses(self):
        session = _make_session()
        state = session.state
        # Carrier P#7 has timed out: not confirmed, no stamp, but a peak
        # snapshot and a surviving sack.
        state["orphaned_peaks"][7] = {"peak": 3, "expires_at": 999}
        state["orphaned_sack_owner"][11] = 7
        state["raw_sacks"].append(_sack(11, 200, 290))
        state["confirmed_sacks"].add(11)

        update_door_zone(session, fn=20)

        self.assertEqual(state["total_sacks_counted"], 3)
        self.assertIn(7, state["persons_past_door"])
        self.assertEqual(state["delivery_log"][0]["trigger"], "orphan_rescue")

    def test_orphan_not_committed_before_crossing(self):
        session = _make_session()
        state = session.state
        state["orphaned_peaks"][7] = {"peak": 3, "expires_at": 999}
        state["orphaned_sack_owner"][11] = 7
        state["raw_sacks"].append(_sack(11, 200, 110))   # corridor side
        state["confirmed_sacks"].add(11)

        update_door_zone(session, fn=20)
        self.assertEqual(state["total_sacks_counted"], 0)

    def test_orphan_committed_only_once(self):
        session = _make_session()
        state = session.state
        state["orphaned_peaks"][7] = {"peak": 3, "expires_at": 999}
        state["orphaned_sack_owner"][11] = 7
        state["raw_sacks"].append(_sack(11, 200, 290))
        state["confirmed_sacks"].add(11)

        update_door_zone(session, fn=20)
        update_door_zone(session, fn=21)
        self.assertEqual(state["total_sacks_counted"], 3)


class TestStampTransfer(unittest.TestCase):
    """
    The stamp-eligibility test required the sack to be unstamped, which
    made the STAMP TRANSFER branch below it unreachable: a sack could
    never move to a second carrier, so a handover stayed credited to
    whoever picked it up first.
    """

    def _session_with_two_carriers(self):
        session = _make_session()
        state = session.state
        for pid, cx in ((1, 200), (2, 205)):
            state["confirmed_persons"].add(pid)
            state["person_prev_cx"][pid] = cx
            state["person_boxes"][pid]   = (cx - 50, 150, cx + 50, 250)
        return session

    def test_sack_can_transfer_to_a_new_owner(self):
        session = self._session_with_two_carriers()
        state = session.state
        state["sack_carrier_stamp"][5] = 1        # currently P#1's
        sacks = [_sack(5, 200, 200)]

        # Ownership has resolved to P#2 with a strong score.
        update_peak_counts(
            sacks, session, fn=10,
            sack_owner_override={5: 2},
            sack_scores_override={5: 0.95},
        )
        self.assertEqual(state["sack_carrier_stamp"][5], 2)

    def test_transfer_resets_the_previous_carriers_peak(self):
        session = self._session_with_two_carriers()
        state = session.state
        state["sack_carrier_stamp"][5] = 1
        state["person_peak_count"][1]  = 4
        sacks = [_sack(5, 200, 200)]

        update_peak_counts(
            sacks, session, fn=10,
            sack_owner_override={5: 2},
            sack_scores_override={5: 0.95},
        )
        self.assertEqual(state["person_peak_count"][1], 0)


class TestGhostsDoNotResurrect(unittest.TestCase):
    """
    update() recreated a ghost for any sid in sack_positions that was not
    active, and sack_positions was never pruned — so every sack ID ever
    seen got a fresh ghost the frame after its previous one aged out,
    forever.
    """

    def test_ghost_is_not_recreated_after_expiring(self):
        ghosts = GhostSacks(max_frames=3)
        ghosts.record_position(1, 100, 100)
        positions = {1: (100, 100)}
        boxes     = {1: (90, 90, 110, 110)}

        # Sack goes missing; ghost is created then ages out.
        for _ in range(6):
            ghosts.update(set(), {1: 7}, positions, boxes)

        self.assertEqual(list(ghosts.iter_ghosts()), [],
                         "expired ghost must not be recreated")

    def test_redetection_grants_a_fresh_ghost_budget(self):
        ghosts = GhostSacks(max_frames=2)
        ghosts.record_position(1, 100, 100)
        positions = {1: (100, 100)}
        boxes     = {1: (90, 90, 110, 110)}

        for _ in range(5):
            ghosts.update(set(), {1: 7}, positions, boxes)
        self.assertEqual(list(ghosts.iter_ghosts()), [])

        ghosts.update({1}, {1: 7}, positions, boxes)   # seen again
        ghosts.update(set(), {1: 7}, positions, boxes)  # lost again
        self.assertEqual(len(list(ghosts.iter_ghosts())), 1)

    def test_cleanup_drops_history_for_unknown_ids(self):
        ghosts = GhostSacks(max_frames=2)
        ghosts.record_position(1, 100, 100)
        ghosts.record_position(2, 50, 50)
        ghosts.cleanup({1})
        self.assertIn(1, ghosts._prev_pos)
        self.assertNotIn(2, ghosts._prev_pos)


class TestStateMachinePrunesLostRecords(unittest.TestCase):
    """
    cleanup() only pruned DELIVERED records, and nothing ever called
    mark_delivered — so no record could reach that state and the registry
    grew for every tracker ID of the whole session.
    """

    def test_lost_record_is_eventually_pruned(self):
        reg = SackStateMachineRegistry()
        reg.get_or_create(1, frame_no=0)
        reg.cleanup(set(), frame_no=1)
        self.assertEqual(reg.get(1).state, SackState.LOST)

        reg.cleanup(set(), frame_no=reg._LOST_KEEP_FRAMES + 1)
        self.assertIsNone(reg.get(1), "stale LOST record must be pruned")

    def test_delivered_record_survives_its_keep_window(self):
        reg = SackStateMachineRegistry()
        reg.get_or_create(1, frame_no=0)
        reg.mark_delivered(1, frame_no=10)

        reg.cleanup(set(), frame_no=20)
        self.assertIsNotNone(reg.get(1))

        reg.cleanup(set(), frame_no=10 + reg._DELIVERED_KEEP_FRAMES + 1)
        self.assertIsNone(reg.get(1))


class TestCrossingConfidenceIsRecorded(unittest.TestCase):
    """
    Nothing called record_gate_a/record_gate_b, so the crossing term —
    15% of the score — was permanently 0 and delivery_confidence was
    capped at 0.85.
    """

    def test_crossing_raises_delivery_confidence(self):
        tracker = ConfidenceTracker(DEFAULT_CFG)
        tracker.record_detection(1, 0.9)
        tracker.record_ownership(1, 0.9)
        before = tracker.delivery_confidence(1)

        tracker.record_crossing(1)
        after = tracker.delivery_confidence(1)

        self.assertAlmostEqual(
            after - before, DEFAULT_CFG["conf_weight_crossing"], places=6)

    def test_commit_records_crossing_for_delivered_sacks(self):
        session = _make_session()
        session.conf_tracker = ConfidenceTracker(DEFAULT_CFG)
        session.state["person_peak_count"][1] = 1
        _stamp(session, sid=3, pid=1, cx=200, cy=290)
        update_door_zone(session, fn=10)
        self.assertEqual(session.conf_tracker._crs.get(3), 1.0)


class TestReporterKeyNormalisation(unittest.TestCase):
    """
    AnalyticsEngine.summary() keys workers by int pid, but the report
    looked them up as str(pid) — every lookup missed, so Avg Load and Avg
    Conf printed 0.00 for everyone.
    """

    def test_int_keys_pass_through(self):
        self.assertEqual(_int_keyed({1: {"a": 1}}), {1: {"a": 1}})

    def test_str_keys_are_converted(self):
        self.assertEqual(_int_keyed({"1": {"a": 1}}), {1: {"a": 1}})

    def test_non_integer_keys_are_dropped(self):
        self.assertEqual(_int_keyed({"total": {}}), {})


class TestMotionTrackerSignature(unittest.TestCase):
    """
    The constructor accepted `window` and `thresh_px` and discarded both,
    which made still_window / still_thresh_ratio look like live tuning
    knobs when they did nothing.
    """

    def test_dead_parameters_are_gone(self):
        with self.assertRaises(TypeError):
            SackMotionTracker(speed_thresh=1.5, speed_window=8, window=10)

    def test_config_no_longer_advertises_dead_keys(self):
        self.assertNotIn("still_window", DEFAULT_CFG)
        self.assertNotIn("still_thresh_ratio", DEFAULT_CFG)


class TestEndToEndCountPath(unittest.TestCase):
    """
    Drives the real counting functions across simulated frames: a carrier
    walks the approach window holding two sacks, gets them stamped, then
    carries them through the door.  This is the path that produces the
    number the whole product exists to report, and nothing exercised it
    end to end before.
    """

    def _walk_carrier_through_door(self, n_sacks=2):
        # Landscape door across the top; room is below (+y). The carrier
        # approaches from above (negative projection) and passes through.
        door = DoorPolygon(
            points=[(100, 380), (700, 380), (700, 420), (100, 420)],
            room_point=(400, 900),
        )
        session = _make_session(door=door, cfg_overrides={
            "peak_window_px":       350,
            "peak_freeze_px":       40,
            "peak_stamp_score":     0.65,
            "peak_min_assoc_score": 0.55,
            "peak_conf_sack_floor": 0.55,
        })
        state = session.state
        state["confirmed_persons"] = {1}

        owner  = {sid: 1 for sid in range(1, n_sacks + 1)}
        scores = {sid: 0.9 for sid in range(1, n_sacks + 1)}

        # Approach: carrier walks from y=150 (corridor) toward the door.
        for fn, cy in enumerate(range(150, 400, 25), start=1):
            state["person_boxes"][1]   = (380, cy - 40, 420, cy + 40)
            state["person_prev_cx"][1] = 400
            sacks = [_sack(sid, 380 + 20 * sid, cy - 50)
                     for sid in range(1, n_sacks + 1)]
            state["raw_sacks"] = sacks
            state["confirmed_sacks"] = set(owner)
            update_peak_counts(sacks, session, fn,
                               sack_owner_override=owner,
                               sack_scores_override=scores)
            update_door_zone(session, fn)

        # Cross: sacks emerge on the room side, well past the threshold.
        fn = 100
        state["person_boxes"][1]   = (380, 460, 420, 540)
        state["person_prev_cx"][1] = 400
        state["raw_sacks"] = [_sack(sid, 380 + 20 * sid, 480)
                              for sid in range(1, n_sacks + 1)]
        update_door_zone(session, fn)
        return state

    def test_two_sack_load_is_counted_once(self):
        state = self._walk_carrier_through_door(n_sacks=2)
        self.assertEqual(state["total_sacks_counted"], 2)

    def test_carrier_is_marked_past_door(self):
        state = self._walk_carrier_through_door()
        self.assertIn(1, state["persons_past_door"])

    def test_delivery_is_logged_once(self):
        state = self._walk_carrier_through_door()
        self.assertEqual(len(state["delivery_log"]), 1)
        self.assertEqual(state["delivery_log"][0]["peak_count"], 2)

    def test_sacks_are_not_recounted_on_later_frames(self):
        """Re-running the crossing frame must not inflate the total."""
        state = self._walk_carrier_through_door()
        self.assertEqual(state["total_sacks_counted"], 2)

    def test_delivered_sacks_recorded_against_carrier(self):
        state = self._walk_carrier_through_door()
        self.assertEqual(len(state["person_sack_delivered"][1]), 2)


class TestConfigSingleSourceOfTruth(unittest.TestCase):
    """
    Exit-mode keys lived only as inline cfg.get() fallbacks, which drifted
    apart: door_cross_threshold_px defaulted to 30 in entry mode and 20 in
    exit mode, so the two modes disagreed about where the door was.
    """

    def test_exit_keys_present_in_defaults(self):
        for key in ("exit_model_path", "landing_zone_points",
                    "exit_confirm_timeout_frames", "exit_still_frames",
                    "exit_dedup_radius_px", "exit_dedup_recency_window",
                    "exit_landing_jitter_tolerance",
                    "exit_landing_min_overlap", "exit_stale_tentative_frames"):
            self.assertIn(key, DEFAULT_CFG)

    def test_shared_door_keys_present(self):
        self.assertIn("door_cross_threshold_px", DEFAULT_CFG)
        self.assertIn("reentry_grace_frames", DEFAULT_CFG)
        self.assertIn("peak_stamp_score", DEFAULT_CFG)


if __name__ == "__main__":
    unittest.main()
