"""
test_person_tracker.py — Unit tests for person_tracker helpers.

Tests timeout_persons and update_persons logic in isolation using a
lightweight mock PipelineSession — no OpenCV, YOLO, or main.py imports.

History
-------
This module used to also hold TestExitBuffer and TestReentryGuard, which
targeted the v19 gate design (``GateCounter``, ``_EXIT_BUFFER_PX``).  That
design was replaced by the door polygon in v21 and both symbols were
deleted from the package — but the tests were left behind, importing a
name that no longer existed.  Because pytest aborts on a collection
error, that single stale import made the ENTIRE suite unrunnable, not
just this file.  Door-side re-entry behaviour is covered by
TestDoorReentry in test_door_crossing.py.
"""

import unittest
import sys
import os
from collections import defaultdict
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.trackers import OwnershipMemory
from sack_counter.pipeline.state import PipelineState
from sack_counter.pipeline.person_tracker import timeout_persons


def _make_session(miss_frames=10):
    """Build a minimal mock PipelineSession for person_tracker tests."""
    session = MagicMock()
    session.state = PipelineState()
    session.ownership_mem = OwnershipMemory(switch_margin=0.10)
    session.miss_frames = miss_frames
    # relinker: canonical returns pid unchanged, register_lost is a no-op
    session.relinker.canonical.side_effect = lambda pid: pid
    session.relinker.register_lost.return_value = None
    session.logger = MagicMock()
    return session


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
