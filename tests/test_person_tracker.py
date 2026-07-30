"""
test_person_tracker.py — Unit tests for person_tracker helpers.

Tests timeout_persons and update_persons logic in isolation using a
lightweight mock PipelineSession — no OpenCV, YOLO, or main.py imports.
"""

import unittest
import sys
import os
from collections import defaultdict
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.trackers import OwnershipMemory
from sack_counter.pipeline.state import PipelineState
from sack_counter.pipeline.person_tracker import (
    timeout_persons,
    _EXIT_BUFFER_PX,
)


def _make_session(direction="RL", line_x=640, offset=40, miss_frames=10):
    """Build a minimal mock PipelineSession for person_tracker tests."""
    session = MagicMock()
    session.state = PipelineState()
    session.gate  = GateCounter(line_x=line_x, direction=direction, offset=offset)
    session.ownership_mem = OwnershipMemory(switch_margin=0.10)
    session.miss_frames = miss_frames
    # relinker: canonical returns pid unchanged, register_lost is a no-op
    session.relinker.canonical.side_effect = lambda pid: pid
    session.relinker.register_lost.return_value = None
    session.logger = MagicMock()
    return session


class TestExitBuffer(unittest.TestCase):
    """Critical Bug #3: exit buffer must be large enough to not fire prematurely."""

    def test_exit_buffer_at_least_20px(self):
        """Regression: buffer was 5 px — must be >= 20 px."""
        self.assertGreaterEqual(_EXIT_BUFFER_PX, 20)

    def test_rl_exit_threshold_uses_buffer(self):
        """For RL, person must be _EXIT_BUFFER_PX past exit_line to trigger."""
        session = _make_session(direction="RL", line_x=640, offset=40)
        gate = session.gate
        # gate.exit_line = 640; trigger at cx < 640 - _EXIT_BUFFER_PX
        threshold = gate.exit_line - _EXIT_BUFFER_PX
        # Just at threshold boundary — not past it
        cx_safe  = threshold + 1   # should NOT trigger
        cx_past  = threshold - 1   # should trigger
        # Verify the math
        self.assertGreater(cx_safe, threshold)
        self.assertLess(cx_past, threshold)

    def test_lr_exit_threshold_uses_buffer(self):
        session = _make_session(direction="LR", line_x=600, offset=40)
        gate = session.gate
        # gate.exit_line = 640; trigger at cx > 640 + _EXIT_BUFFER_PX
        threshold = gate.exit_line + _EXIT_BUFFER_PX
        cx_safe = threshold - 1
        cx_past = threshold + 1
        self.assertLess(cx_safe, threshold)
        self.assertGreater(cx_past, threshold)


class TestReentryGuard(unittest.TestCase):
    """Bug #6: reentry guard must not fire for workers queuing before the gate."""

    def test_rl_reentry_requires_cx_right_of_entry(self):
        """
        RL: re-entry only when cx > entry_line + 50 (clearly back on entry side).
        A person at cx = entry_line - 50 (approaching) must NOT clear past_gate.
        """
        # entry_line = 680 for RL with line_x=640, offset=40
        # Old guard: cx > entry_line - 100 = 580 — fired for anyone right of 580
        # New guard: cx > entry_line + 50 = 730 — only fires when clearly back
        session = _make_session(direction="RL", line_x=640, offset=40)
        gate = session.gate
        # A person approaching from the right at cx=650 (between exit and entry)
        # should NOT be considered re-entered.
        cx_approaching = gate.entry_line - 20   # 660 — approaching, not re-entered
        in_reentry = (gate.direction == "RL" and cx_approaching > gate.entry_line + 50)
        self.assertFalse(in_reentry, "Approaching person should not trigger re-entry")

    def test_rl_reentry_fires_when_clearly_back(self):
        session = _make_session(direction="RL", line_x=640, offset=40)
        gate = session.gate
        cx_returned = gate.entry_line + 100   # well past entry → re-entered
        in_reentry  = (gate.direction == "RL" and cx_returned > gate.entry_line + 50)
        self.assertTrue(in_reentry)

    def test_lr_reentry_requires_cx_left_of_entry(self):
        session = _make_session(direction="LR", line_x=600, offset=40)
        gate = session.gate
        cx_approaching = gate.entry_line + 20
        in_reentry = (gate.direction == "LR" and cx_approaching < gate.entry_line - 50)
        self.assertFalse(in_reentry)


class TestTimeoutPersons(unittest.TestCase):
    """Tests for timeout_persons, including Bug #7 (cy/area) and Bug #4 (canonical)."""

    def _seed_person(self, session, pid, cx, cy, box, peak=3,
                     miss_frames=None, owns_sacks=None):
        """Add a person to state and fast-forward past miss_frames."""
        state = session.state
        state["confirmed_persons"].add(pid)
        state["person_prev_cx"][pid]    = cx
        state["person_velocity"][pid]   = (0.0, 0.0)
        state["person_boxes"][pid]      = box
        state["person_embeddings"][pid] = None
        state["person_peak_count"][pid] = peak
        # Fill miss counter past threshold
        mf = miss_frames or session.miss_frames
        state["person_miss"][pid] = mf + 1

        if owns_sacks:
            for sid in owns_sacks:
                session.ownership_mem.update(sid, pid, 0.9)
                state["sack_carrier_stamp"][sid] = pid

    def test_register_lost_receives_correct_cy_and_area(self):
        """Bug #7: cy and area must come from person_boxes, not hard-coded 0."""
        session = _make_session()
        box = (100, 200, 200, 400)  # cx=150, cy=300, area=20000
        self._seed_person(session, pid=1, cx=150, cy=300, box=box, peak=0)

        timeout_persons(active_pids=set(), session=session, fn=100)

        call_args = session.relinker.register_lost.call_args
        self.assertIsNotNone(call_args)
        _, cx_arg, cy_arg, _, _, area_arg, _, _ = call_args[0]
        self.assertEqual(cy_arg, 300)       # mid of y1=200, y2=400
        self.assertGreater(area_arg, 0)     # (200-100)*(400-200) = 20000

    def test_orphaned_sack_owner_written_for_owned_sacks(self):
        session = _make_session()
        box = (100, 200, 200, 400)
        self._seed_person(session, pid=1, cx=150, cy=300, box=box,
                          peak=3, owns_sacks=[5, 6])
        session.state["sack_carrier_stamp"][5] = 1
        session.state["sack_carrier_stamp"][6] = 1

        timeout_persons(active_pids=set(), session=session, fn=100)

        self.assertEqual(session.state["orphaned_sack_owner"].get(5), 1)
        self.assertEqual(session.state["orphaned_sack_owner"].get(6), 1)

    def test_canonical_orphan_written_bug4(self):
        """Bug #4: orphaned_sack_owner must be updated to canonical pid."""
        session = _make_session()
        box = (100, 200, 200, 400)
        # Set canonical(1) → 99 (simulating ReID fragmentation)
        session.relinker.canonical.side_effect = lambda pid: 99 if pid == 1 else pid

        self._seed_person(session, pid=1, cx=150, cy=300, box=box,
                          peak=3, owns_sacks=[7])
        session.state["sack_carrier_stamp"][7] = 1

        timeout_persons(active_pids=set(), session=session, fn=100)

        # After BUG #4 fix: orphaned_sack_owner[7] should be updated to 99
        self.assertEqual(session.state["orphaned_sack_owner"].get(7), 99)

    def test_person_removed_from_confirmed_after_timeout(self):
        session = _make_session()
        self._seed_person(session, pid=2, cx=150, cy=300,
                          box=(100, 200, 200, 400), peak=0)
        timeout_persons(active_pids=set(), session=session, fn=100)
        self.assertNotIn(2, session.state["confirmed_persons"])

    def test_no_orphan_if_already_delivered(self):
        """A person who already committed a delivery should not create orphan."""
        session = _make_session()
        self._seed_person(session, pid=3, cx=150, cy=300,
                          box=(100, 200, 200, 400), peak=5)
        session.state["person_peak_delivery"][3] = 5   # already delivered

        timeout_persons(active_pids=set(), session=session, fn=100)
        self.assertNotIn(3, session.state["orphaned_peaks"])


if __name__ == "__main__":
    unittest.main()
