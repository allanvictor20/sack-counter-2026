"""
test_state_machine.py — Unit tests for the sack lifecycle state machine.

No main.py / OpenCV / YOLO imports.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.pipeline.state_machine import (
    SackState, SackRecord, SackStateMachineRegistry,
)


class TestSackStateTransitions(unittest.TestCase):
    def test_valid_detected_to_confirmed(self):
        rec = SackRecord(sack_id=1)
        self.assertTrue(rec.transition(SackState.CONFIRMED, 10))
        self.assertEqual(rec.state, SackState.CONFIRMED)

    def test_invalid_detected_to_delivered(self):
        rec = SackRecord(sack_id=1)
        self.assertFalse(rec.transition(SackState.DELIVERED, 10))
        self.assertEqual(rec.state, SackState.DETECTED)

    def test_history_recorded(self):
        rec = SackRecord(sack_id=1)
        rec.transition(SackState.CONFIRMED, 5)
        rec.transition(SackState.CARRIED, 10)
        self.assertEqual(len(rec.history), 2)
        self.assertEqual(rec.history[0], (5, SackState.DETECTED, SackState.CONFIRMED))

    def test_force_transition_bypasses_guard(self):
        rec = SackRecord(sack_id=1)
        rec.force_transition(SackState.DELIVERED, 99)
        self.assertEqual(rec.state, SackState.DELIVERED)

    def test_delivered_is_terminal(self):
        rec = SackRecord(sack_id=1)
        rec.force_transition(SackState.DELIVERED, 1)
        self.assertFalse(rec.transition(SackState.DETECTED, 2))

    def test_delivered_at_set_by_mark_delivered(self):
        """BUG #9: delivered_at must be set so the keep-window guard works."""
        reg = SackStateMachineRegistry()
        reg.get_or_create(1, 0)
        reg.mark_delivered(1, frame_no=50)
        rec = reg.get(1)
        self.assertEqual(rec.delivered_at, 50)


class TestSackStateMachineRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = SackStateMachineRegistry()

    def test_get_or_create(self):
        rec = self.reg.get_or_create(42, frame_no=1)
        self.assertEqual(rec.sack_id, 42)
        self.assertEqual(rec.first_seen, 1)

    def test_idempotent_create(self):
        rec1 = self.reg.get_or_create(42, 1)
        rec2 = self.reg.get_or_create(42, 5)
        self.assertIs(rec1, rec2)
        self.assertEqual(rec1.first_seen, 1)

    def test_transition(self):
        self.reg.get_or_create(1, 0)
        self.assertTrue(self.reg.transition(1, SackState.CONFIRMED, 5))

    def test_mark_delivered(self):
        self.reg.get_or_create(1, 0)
        self.reg.mark_delivered(1, 10)
        self.assertEqual(self.reg.get(1).state, SackState.DELIVERED)

    def test_cleanup_does_not_immediately_prune_delivered(self):
        """
        BUG #9 regression: DELIVERED records must persist for at least
        _DELIVERED_KEEP_FRAMES frames so ByteTrack ID reuse cannot silently
        resurrect the record in DETECTED state on the next frame.
        """
        self.reg.get_or_create(1, 0)
        self.reg.mark_delivered(1, frame_no=5)
        # Cleanup 1 frame later — must NOT prune
        self.reg.cleanup(set(), frame_no=6)
        self.assertIsNotNone(self.reg.get(1))
        self.assertEqual(self.reg.get(1).state, SackState.DELIVERED)

    def test_cleanup_prunes_delivered_after_keep_window(self):
        keep = SackStateMachineRegistry._DELIVERED_KEEP_FRAMES
        self.reg.get_or_create(1, 0)
        self.reg.mark_delivered(1, frame_no=0)
        self.reg.cleanup(set(), frame_no=keep)
        self.assertIsNone(self.reg.get(1))

    def test_cleanup_marks_missing_as_lost(self):
        self.reg.get_or_create(2, 0)
        self.reg.transition(2, SackState.CONFIRMED, 1)
        self.reg.cleanup(set(), frame_no=5)
        self.assertEqual(self.reg.get(2).state, SackState.LOST)

    def test_in_state(self):
        self.reg.get_or_create(1, 0)
        self.reg.get_or_create(2, 0)
        self.reg.transition(1, SackState.CONFIRMED, 1)
        self.assertIn(2, self.reg.in_state(SackState.DETECTED))
        self.assertIn(1, self.reg.in_state(SackState.CONFIRMED))


if __name__ == "__main__":
    unittest.main()
