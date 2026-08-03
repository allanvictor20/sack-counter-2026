"""
test_assignment.py — Unit tests for the sack-to-person assignment module.

Imports only the modules under test (config, assignment, trackers).
No main.py, no drawing, no embedder, no OpenCV dependency.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.config import DEFAULT_CFG
from sack_counter.assignment import association_score, carry_zone_bounds, assign_sacks_hungarian
from sack_counter.trackers import SackMotionTracker, OwnershipMemory


def _cfg(**overrides):
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    return cfg


class TestCarryZoneBounds(unittest.TestCase):
    def test_basic_bounds(self):
        cfg = _cfg(carry_zone_top=-1.5, carry_zone_bot=0.6)
        person_box = (100, 200, 200, 400)   # 100px wide, 200px tall
        x1, y1, x2, y2 = carry_zone_bounds(person_box, cfg)
        self.assertEqual(x1, 75)
        self.assertEqual(x2, 225)
        self.assertEqual(y1, -100)   # 200 + 200*(-1.5)
        self.assertEqual(y2, 320)    # 200 + 200*0.6

    def test_symmetric_horizontal_margin(self):
        cfg = _cfg(carry_zone_top=0.0, carry_zone_bot=1.0)
        x1, y1, x2, y2 = carry_zone_bounds((0, 0, 100, 100), cfg)
        self.assertEqual(x2 - x1, 150)   # 100 + 2*25


class TestAssociationScore(unittest.TestCase):
    def setUp(self):
        self.cfg = _cfg(
            w_distance=0.35, w_overlap=0.25, w_height=0.20, w_appearance=0.20,
            carry_zone_top=-1.5, carry_zone_bot=0.6,
            max_assoc_dist=300,
        )

    def test_score_in_unit_range(self):
        score = association_score(
            110, 150, 190, 250, 150, 200,
            (100, 200, 200, 400), self.cfg,
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_close_scores_higher_than_far(self):
        person_box = (100, 200, 200, 400)
        close = association_score(110, 80, 190, 180, 150, 130, person_box, self.cfg)
        far   = association_score(400, 200, 500, 300, 450, 250, person_box, self.cfg)
        self.assertGreater(close, far)

    def test_beyond_max_dist_returns_zero(self):
        score = association_score(
            900, 200, 1000, 300, 950, 250,
            (100, 200, 200, 400), self.cfg,
        )
        self.assertEqual(score, 0.0)

    def test_above_head_scores_positively(self):
        """Sack directly above person's head should still get a positive score."""
        person_box = (100, 300, 200, 500)
        score = association_score(
            110, 150, 190, 290, 150, 220,   # scy=220 < py1=300
            person_box, self.cfg,
        )
        self.assertGreater(score, 0.0)


class TestOwnershipMemory(unittest.TestCase):
    """Tests for the ownership persistence / hysteresis logic."""

    def test_initial_assignment(self):
        mem = OwnershipMemory(switch_margin=0.15)
        result = mem.update(sid=1, new_pid=10, new_score=0.8)
        self.assertEqual(result, 10)
        self.assertEqual(mem.get(1), 10)
        self.assertAlmostEqual(mem.get_score(1), 0.8)

    def test_persists_when_new_score_insufficient(self):
        mem = OwnershipMemory(switch_margin=0.15)
        mem.update(1, 10, 0.8)
        result = mem.update(1, 20, 0.9)   # 0.9 >= 0.8 + 0.15? No → stays 10
        self.assertEqual(result, 10)

    def test_switches_when_new_score_exceeds_margin(self):
        mem = OwnershipMemory(switch_margin=0.15)
        mem.update(1, 10, 0.6)
        result = mem.update(1, 20, 0.76)  # 0.76 >= 0.6 + 0.15 → switches
        self.assertEqual(result, 20)

    def test_get_score_after_persistence(self):
        """
        CRITICAL BUG #1: sack_scores must reflect the *persisted* owner's
        score, not the new candidate's.  Verify get_score returns the
        persisted owner's score when the candidate fails to displace them.
        """
        mem = OwnershipMemory(switch_margin=0.15)
        mem.update(1, 10, 0.8)    # pid=10, score=0.8 persisted
        mem.update(1, 20, 0.85)   # pid=20 fails to displace (0.85 < 0.8+0.15)
        # persisted owner is still pid=10 with score=0.8
        self.assertEqual(mem.get(1), 10)
        self.assertAlmostEqual(mem.get_score(1), 0.8)

    def test_iter_by_owner(self):
        mem = OwnershipMemory(switch_margin=0.10)
        mem.update(1, 10, 0.7)
        mem.update(2, 10, 0.6)
        mem.update(3, 20, 0.9)
        pairs = dict(mem.iter_by_owner())
        self.assertEqual(pairs[1], 10)
        self.assertEqual(pairs[2], 10)
        self.assertEqual(pairs[3], 20)

    def test_iter_by_owner_is_snapshot(self):
        """iter_by_owner must be safe to call while mutating the registry."""
        mem = OwnershipMemory(switch_margin=0.10)
        mem.update(1, 10, 0.7)
        mem.update(2, 10, 0.6)
        # Should not raise RuntimeError: dictionary changed size during iteration
        for sid, pid in mem.iter_by_owner():
            if sid == 1:
                mem.clear(2)

    def test_cleanup_removes_inactive(self):
        mem = OwnershipMemory(switch_margin=0.10)
        mem.update(1, 10, 0.7)
        mem.update(2, 10, 0.6)
        mem.cleanup(active_sids={1})
        self.assertIsNone(mem.get(2))
        self.assertEqual(mem.get(1), 10)


class TestHungarianAssignment(unittest.TestCase):
    def setUp(self):
        self.cfg    = _cfg(max_sacks_per_person=4, min_assoc_score=0.05)
        self.motion = SackMotionTracker(speed_thresh=1.5, speed_window=8)
        self.ownership = OwnershipMemory(switch_margin=0.15)

    def _warm_motion(self, sid, cx=140, cy=130, n=12):
        # Move 5px per frame so mean speed (5.0) exceeds speed_thresh (1.5)
        for i in range(n):
            self.motion.update(sid, cx + i * 5, cy)

    def test_single_sack_single_person_assigned(self):
        self._warm_motion(1)
        # sack centroid after warm-up is at cx=140+(11*5)=195
        sacks   = [(1, 90, 80, 190, 180, 195, 130, 0.9)]
        persons = {10: (100, 200, 200, 400)}
        owner, scores, load, stats = assign_sacks_hungarian(
            sacks, persons, self.motion, self.ownership, self.cfg
        )
        self.assertIn(1, owner)
        self.assertEqual(owner[1], 10)
        self.assertEqual(load[10], 1)

    def test_sack_scores_use_persisted_score(self):
        """
        CRITICAL BUG #1 regression: sack_scores[sid] must equal
        ownership_mem.get_score(sid), not the raw candidate score.
        """
        # Pre-seed ownership with pid=10, score=0.9
        self.ownership.update(1, 10, 0.9)
        self._warm_motion(1)
        # Hungarian produces pid=10 with a lower raw score this frame
        sacks   = [(1, 90, 80, 190, 180, 195, 130, 0.3)]
        persons = {10: (100, 200, 200, 400), 20: (600, 200, 700, 400)}
        owner, scores, load, stats = assign_sacks_hungarian(
            sacks, persons, self.motion, self.ownership, self.cfg
        )
        # Whatever the resolved owner is, scores[sid] must match
        # ownership_mem.get_score(sid) — not the raw candidate value
        if 1 in scores:
            self.assertAlmostEqual(scores[1], self.ownership.get_score(1), places=5)

    def test_no_sacks_returns_empty(self):
        owner, scores, load, stats = assign_sacks_hungarian(
            [], {10: (100, 200, 200, 400)},
            self.motion, self.ownership, self.cfg
        )
        self.assertEqual(owner, {})
        self.assertEqual(load, {})

    def test_no_persons_returns_empty(self):
        self._warm_motion(1)
        sacks = [(1, 90, 80, 190, 180, 140, 130, 0.9)]
        owner, scores, load, stats = assign_sacks_hungarian(
            sacks, {}, self.motion, self.ownership, self.cfg
        )
        self.assertEqual(owner, {})

    def test_still_sack_excluded(self):
        """Sacks the motion tracker considers still must be excluded."""
        # All positions identical → speed=0 < speed_thresh → is_still=True
        for _ in range(20):
            self.motion.update(1, 140, 130)
        sacks   = [(1, 90, 80, 190, 180, 140, 130, 0.9)]
        persons = {10: (100, 200, 200, 400)}
        owner, scores, load, stats = assign_sacks_hungarian(
            sacks, persons, self.motion, self.ownership, self.cfg
        )
        self.assertNotIn(1, owner)
        self.assertEqual(stats["still"], 1)


if __name__ == "__main__":
    unittest.main()
