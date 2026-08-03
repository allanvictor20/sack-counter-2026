"""
test_door_crossing.py — Unit tests for door-zone crossing logic (v22).

Rewritten to match the CURRENT update_door_zone() implementation.

History
-------
The original version of this test file targeted an older design where
``update_door_zone`` built a per-person ``DoorCandidateState`` state machine
(stored in ``state["door_candidates"]``) that progressed through
CORRIDOR -> APPROACHING -> IN_DOOR -> DISAPPEARING -> COUNTED states, with
a separate two-phase commit step (``process_door_disappearance``) applied
after a delay.

The current algorithm is simpler and is implemented entirely inside
``update_door_zone``:

    For each confirmed person not yet past the door:
      1. Find sacks stamped to that person (``sack_carrier_stamp``).
      2. If any stamped sack's centroid projection onto the door normal
         has magnitude >= door_cross_threshold_px AND the sack is not on
         the approach/corridor side -> the sack has crossed.
      3. On crossing: commit the carrier's peak count immediately
         (no delay, no disappearance phase), clear all sack stamps for
         that carrier, and mark the carrier as past_door / past_gate.

``door_candidates``, ``DoorCandidateState``, and ``DoorState`` are still
defined in door_crossing.py for backward-compatible imports (drawing.py
references DoorState for debug overlay labels), but they are no longer
populated by the counting algorithm itself. ``process_door_disappearance``
is now a no-op kept only for call-site compatibility.

These tests verify the REAL commit trigger: a stamped sack crossing the
door threshold.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.door_polygon import DoorPolygon
from sack_counter.pipeline.door_crossing import (
    DoorState,
    DoorCandidateState,
    update_door_zone,
    process_door_disappearance,
    check_door_reentry,
    cleanup_door_histories,
)
from sack_counter.pipeline.state import PipelineState
from sack_counter.config import DEFAULT_CFG


def _make_door(room_point=None):
    """200x200 square centred around (200, 200). Room is below (y=400)."""
    return DoorPolygon(
        points=[(100, 100), (300, 100), (300, 300), (100, 300)],
        room_point=room_point or (200, 400),
    )


def _make_session(door=None, cfg_overrides=None):
    cfg = dict(DEFAULT_CFG)
    cfg["door_cross_threshold_px"]   = 30
    cfg["direction_lookback_frames"] = 3
    cfg["reentry_margin_px"]         = 30
    cfg["reentry_grace_frames"]      = 5
    cfg["min_delivery_confidence"]   = 0.55
    cfg["conf_class_high"]           = 0.80
    cfg["conf_class_medium"]         = 0.60
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
    session.conf_tracker = MagicMock()
    session.conf_tracker.delivery_confidence.return_value = 0.7
    session.sack_sm = MagicMock()
    session.record_delivery = MagicMock()
    return session


def _stamp_sack(session, sid, pid, cx, cy, conf=0.9):
    """
    Add a confirmed sack stamped to *pid* at (cx, cy).

    Populates raw_sacks (the tuple format update_door_zone reads),
    confirmed_sacks, and sack_carrier_stamp — the three pieces of
    state the real algorithm actually inspects.
    """
    state = session.state
    w, h = 40, 60
    box = (sid, cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2, cx, cy, conf)
    state["raw_sacks"].append(box)
    state["confirmed_sacks"].add(sid)
    state["sack_carrier_stamp"][sid] = pid
    state["sack_scores"][sid] = conf
    return box


# ── DoorState enum — still defined for drawing.py compatibility ───

class TestDoorStateEnum(unittest.TestCase):
    """
    DoorState / DoorCandidateState remain importable even though the
    current algorithm no longer drives a per-candidate state machine.
    drawing.py still imports DoorState for debug overlay labels.
    """

    def test_enum_values_exist(self):
        for name in ("CORRIDOR", "APPROACHING", "IN_DOOR",
                     "DISAPPEARING", "COUNTED", "ABORTED"):
            self.assertTrue(hasattr(DoorState, name))

    def test_candidate_default_state(self):
        cand = DoorCandidateState(
            pid=1, entered_door_fn=0,
            peak_at_entry=2, direction_ok=True,
        )
        self.assertEqual(cand.door_state, DoorState.IN_DOOR)


# ── No-op compatibility shim ───────────────────────────────────────

class TestProcessDoorDisappearanceIsNoOp(unittest.TestCase):
    """
    process_door_disappearance is kept only so older call-sites don't
    break on import/call. It must always return 0 and must never raise,
    regardless of session/state content.
    """

    def test_returns_zero_with_empty_session(self):
        session = _make_session()
        result = process_door_disappearance(session, fn=10)
        self.assertEqual(result, 0)

    def test_returns_zero_after_a_real_commit(self):
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        _stamp_sack(session, sid=1, pid=1, cx=200, cy=290)  # past threshold
        update_door_zone(session, fn=10)
        # Even after a real commit happened via update_door_zone,
        # process_door_disappearance is still a no-op.
        result = process_door_disappearance(session, fn=20)
        self.assertEqual(result, 0)


# ── Core commit trigger: stamped sack crosses the door ─────────────

class TestSackCrossingCommit(unittest.TestCase):
    """
    The real trigger: a sack stamped to a confirmed carrier whose centroid
    projection onto the door normal has |proj| >= door_cross_threshold_px,
    AND is not on the approach/corridor side, causes an immediate commit
    of that carrier's peak count.
    """

    def test_no_commit_without_any_stamped_sack(self):
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        update_door_zone(session, fn=10)
        self.assertEqual(session.state["total_sacks_counted"], 0)
        self.assertNotIn(1, session.state["persons_past_door"])

    def test_no_commit_when_sack_inside_threshold(self):
        """Sack stamped but still near centroid (proj < threshold) — no commit."""
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        # Door centroid is (200, 200); sack right at centroid -> proj ~ 0
        _stamp_sack(session, sid=1, pid=1, cx=200, cy=200)
        update_door_zone(session, fn=10)
        self.assertEqual(session.state["total_sacks_counted"], 0)
        self.assertNotIn(1, session.state["persons_past_door"])

    def test_commit_when_sack_crosses_threshold_into_room(self):
        """
        Sack stamped to carrier, positioned past the door threshold on the
        room side (y=290, far enough from centroid y=200 to exceed the
        30px threshold) -> commit fires immediately, no delay.
        """
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        _stamp_sack(session, sid=1, pid=1, cx=200, cy=290)
        update_door_zone(session, fn=10)

        self.assertEqual(session.state["total_sacks_counted"], 2)
        self.assertIn(1, session.state["persons_past_door"])
        self.assertIn(1, session.state["persons_past_gate"])
        self.assertEqual(session.state["person_peak_delivery"][1], 2)

    def test_commit_is_immediate_no_delay_needed(self):
        """Unlike the old two-phase design, commit happens on the SAME frame
        the crossing is detected — no commit_delay_frames wait."""
        session = _make_session()
        session.state["person_peak_count"][1] = 1
        _stamp_sack(session, sid=1, pid=1, cx=200, cy=290)
        update_door_zone(session, fn=1)  # very first frame
        self.assertEqual(session.state["total_sacks_counted"], 1)

    def test_no_commit_if_peak_is_zero(self):
        """Sack crosses but the carrier's peak count is 0 -> not counted,
        but the carrier IS still marked past_door (abandons tracking)."""
        session = _make_session()
        session.state["person_peak_count"][1] = 0
        _stamp_sack(session, sid=1, pid=1, cx=200, cy=290)
        update_door_zone(session, fn=10)

        self.assertEqual(session.state["total_sacks_counted"], 0)
        self.assertIn(1, session.state["persons_past_door"])
        self.assertNotIn(1, session.state["person_peak_delivery"])

    def test_stamps_cleared_after_commit(self):
        """FIX 1: all sacks stamped to the carrier are cleared immediately
        after commit so they cannot trigger a second commit next frame."""
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        _stamp_sack(session, sid=1, pid=1, cx=200, cy=290)
        _stamp_sack(session, sid=2, pid=1, cx=190, cy=295)
        update_door_zone(session, fn=10)

        self.assertNotIn(1, session.state["sack_carrier_stamp"])
        self.assertNotIn(2, session.state["sack_carrier_stamp"])
        self.assertIn(1, session.state["delivered_sacks"])

    def test_person_already_past_door_is_skipped(self):
        """A person already marked persons_past_door is not re-processed,
        even if a new sack is (incorrectly) stamped to them."""
        session = _make_session()
        session.state["persons_past_door"].add(1)
        session.state["person_peak_count"][1] = 5
        _stamp_sack(session, sid=1, pid=1, cx=200, cy=290)
        update_door_zone(session, fn=10)
        self.assertEqual(session.state["total_sacks_counted"], 0)

    def test_ground_sack_does_not_trigger_commit(self):
        """A sack marked as a ground sack must not be eligible to trigger
        a crossing commit even if stamped and past the threshold."""
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        _stamp_sack(session, sid=1, pid=1, cx=200, cy=290)
        session.state["ground_sacks"].add(1)
        update_door_zone(session, fn=10)
        self.assertEqual(session.state["total_sacks_counted"], 0)

    def test_unconfirmed_sack_does_not_trigger_commit(self):
        """A sack not in confirmed_sacks must not trigger a commit even if
        present in raw_sacks and stamped."""
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        box = _stamp_sack(session, sid=1, pid=1, cx=200, cy=290)
        session.state["confirmed_sacks"].discard(1)
        update_door_zone(session, fn=10)
        self.assertEqual(session.state["total_sacks_counted"], 0)

    def test_delivery_log_entry_created_on_commit(self):
        session = _make_session()
        session.state["person_peak_count"][1] = 3
        _stamp_sack(session, sid=7, pid=1, cx=200, cy=290)
        update_door_zone(session, fn=42)

        self.assertEqual(len(session.state["delivery_log"]), 1)
        entry = session.state["delivery_log"][0]
        self.assertEqual(entry["person_id"], 1)
        self.assertEqual(entry["peak_count"], 3)
        self.assertEqual(entry["sack_id"], 7)
        self.assertEqual(entry["frame"], 42)
        self.assertEqual(entry["trigger"], "sack_crossing")


# ── Stale candidate eviction — feature no longer exists ────────────

class TestStaleCandidateEvictionRemoved(unittest.TestCase):
    """
    The old design evicted DoorCandidateState entries older than
    max_candidate_age_frames. The current algorithm has no notion of
    candidate "age" at all — door_candidates is never populated, so
    there is nothing to evict. This test documents that door_candidates
    stays empty regardless of how many frames pass.
    """

    def test_door_candidates_remains_empty_regardless_of_age(self):
        session = _make_session(cfg_overrides={"max_candidate_age_frames": 10})
        update_door_zone(session, fn=5)
        self.assertEqual(session.state["door_candidates"], {})

        update_door_zone(session, fn=200)
        self.assertEqual(session.state["door_candidates"], {})


# ── Memory-leak prevention ──────────────────────────────────────────

class TestCleanupDoorHistories(unittest.TestCase):
    """
    cleanup_door_histories intentionally preserves person_peak_count
    (so ReID can restore a peak for a person who briefly drops tracking)
    while clearing position history and any door_candidates entry.
    """

    def test_position_history_cleared_on_timeout(self):
        session = _make_session()
        state = session.state
        from collections import deque
        state.persons.person_position_history[42] = deque([(1, 2), (3, 4)])

        cleanup_door_histories(42, state)

        self.assertNotIn(42, state.persons.person_position_history)

    def test_door_candidates_entry_cleared_if_present(self):
        """Even though nothing populates door_candidates today, cleanup
        must still safely remove an entry if one is present (defensive)."""
        session = _make_session()
        state = session.state
        state["door_candidates"][42] = MagicMock()

        cleanup_door_histories(42, state)

        self.assertNotIn(42, state["door_candidates"])

    def test_person_peak_count_is_intentionally_preserved(self):
        """This is documented, deliberate behavior — NOT a bug.
        person_peak_count is preserved across cleanup so a ReID-restored
        pid does not lose its accumulated peak."""
        session = _make_session()
        state = session.state
        state["person_peak_count"][42] = 5

        cleanup_door_histories(42, state)

        self.assertIn(42, state["person_peak_count"])
        self.assertEqual(state["person_peak_count"][42], 5)

    def test_cleanup_safe_on_missing_pid(self):
        session = _make_session()
        try:
            cleanup_door_histories(999, session.state)
        except Exception as exc:
            self.fail(f"cleanup_door_histories raised unexpectedly: {exc}")


# ── Re-entry ─────────────────────────────────────────────────────────

class TestDoorReentry(unittest.TestCase):
    def test_reentry_resets_past_door(self):
        session = _make_session()
        state = session.state
        state["persons_past_door"].add(1)
        # Place person on corridor side (above the square, room is below)
        state["person_prev_cx"][1] = 200
        state["person_boxes"][1]   = (180, 10, 220, 60)   # y centroid = 35

        check_door_reentry(session, fn=50)
        self.assertNotIn(1, state["persons_past_door"])

    def test_no_reentry_if_still_inside(self):
        session = _make_session()
        state = session.state
        state["persons_past_door"].add(1)
        state["person_prev_cx"][1] = 200
        state["person_boxes"][1]   = (150, 150, 250, 250)  # inside polygon

        check_door_reentry(session, fn=50)
        self.assertIn(1, state["persons_past_door"])

    def test_reentry_respects_grace_period_after_commit(self):
        """FIX 2: a person committed within reentry_grace_frames must not
        be reset even if their last known position looks like corridor.

        The grace period is driven by state["last_commit_frame"], an index
        maintained by the commit path, rather than by rescanning the whole
        delivery_log on every frame.
        """
        session = _make_session()
        state = session.state
        state["persons_past_door"].add(1)
        state["person_prev_cx"][1] = 200
        state["person_boxes"][1]   = (180, 10, 220, 60)  # corridor side

        state["last_commit_frame"][1] = 48
        state["delivery_log"].append({
            "frame": 48, "person_id": 1, "peak_count": 2,
        })

        check_door_reentry(session, fn=50)  # only 2 frames after commit
        self.assertIn(1, state["persons_past_door"],
                      "Should not reset within grace period")

    def test_reentry_fires_after_grace_period_expires(self):
        session = _make_session()
        state = session.state
        state["persons_past_door"].add(1)
        state["person_prev_cx"][1] = 200
        state["person_boxes"][1]   = (180, 10, 220, 60)

        state["last_commit_frame"][1] = 10
        state["delivery_log"].append({
            "frame": 10, "person_id": 1, "peak_count": 2,
        })

        check_door_reentry(session, fn=50)  # 40 frames after commit, grace=5
        self.assertNotIn(1, state["persons_past_door"])

    def test_commit_records_last_commit_frame(self):
        """The commit path must populate the index check_door_reentry reads."""
        session = _make_session()
        session.state["person_peak_count"][1] = 2
        _stamp_sack(session, sid=1, pid=1, cx=200, cy=290)
        update_door_zone(session, fn=77)
        self.assertEqual(session.state["last_commit_frame"][1], 77)

    def test_reentry_clears_stale_sack_stamps(self):
        session = _make_session()
        state = session.state
        state["persons_past_door"].add(1)
        state["sack_carrier_stamp"][9] = 1
        state["person_prev_cx"][1] = 200
        state["person_boxes"][1]   = (180, 10, 220, 60)

        check_door_reentry(session, fn=50)
        self.assertNotIn(9, state["sack_carrier_stamp"])


if __name__ == "__main__":
    unittest.main()
