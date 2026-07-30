"""
test_ground_memory.py — Unit tests for GroundMemory.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.pipeline.ground_memory import GroundMemory


class TestGroundMemory(unittest.TestCase):
    def setUp(self):
        self.mem = GroundMemory(
            appearance_threshold=0.80,
            position_threshold_px=60,
            pickup_stable_frames=3,
        )

    def test_mark_grounded_and_is_known(self):
        self.mem.mark_grounded(1, (100, 200), frame_no=5)
        self.assertTrue(self.mem.is_known_ground_object(1))

    def test_unknown_object_not_ground(self):
        self.assertFalse(self.mem.is_known_ground_object(99))

    def test_pickup_confirmed_after_stable_frames(self):
        self.mem.mark_grounded(1, (100, 200), frame_no=1)
        self.mem.mark_moving(1, 2)
        self.mem.mark_moving(1, 3)
        confirmed = self.mem.mark_moving(1, 4)   # 3rd moving frame
        self.assertTrue(confirmed)
        self.assertFalse(self.mem.is_known_ground_object(1))

    def test_not_confirmed_before_stable_frames(self):
        self.mem.mark_grounded(1, (100, 200), frame_no=1)
        self.mem.mark_moving(1, 2)
        confirmed = self.mem.mark_moving(1, 3)   # only 2nd moving frame
        self.assertFalse(confirmed)
        self.assertTrue(self.mem.is_known_ground_object(1))

    def test_still_after_moving_resets_streak(self):
        self.mem.mark_grounded(1, (100, 200), frame_no=1)
        self.mem.mark_moving(1, 2)
        self.mem.mark_moving(1, 3)
        self.mem.mark_grounded(1, (100, 200), frame_no=4)   # went still again
        confirmed = self.mem.mark_moving(1, 5)              # streak reset
        self.assertFalse(confirmed)

    def test_find_by_position(self):
        self.mem.mark_grounded(1, (100, 200), frame_no=1)
        found = self.mem.find_by_position((105, 205))
        self.assertEqual(found, 1)

    def test_find_by_position_too_far(self):
        self.mem.mark_grounded(1, (100, 200), frame_no=1)
        found = self.mem.find_by_position((300, 400))
        self.assertIsNone(found)

    def test_find_by_position_with_appearance(self):
        emb_a = np.array([1.0, 0.0, 0.0])
        emb_b = np.array([0.0, 1.0, 0.0])
        self.mem.mark_grounded(1, (100, 200), frame_no=1, appearance=emb_a)
        # Matching appearance
        found_match = self.mem.find_by_position((102, 202), appearance=emb_a)
        self.assertEqual(found_match, 1)
        # Non-matching appearance (orthogonal vector)
        found_miss = self.mem.find_by_position((102, 202), appearance=emb_b)
        self.assertIsNone(found_miss)

    def test_grounded_ids_property(self):
        self.mem.mark_grounded(1, (100, 200), frame_no=1)
        self.mem.mark_grounded(2, (300, 400), frame_no=1)
        self.assertSetEqual(self.mem.grounded_ids, {1, 2})

    def test_cleanup_removes_delivered(self):
        self.mem.mark_grounded(1, (100, 200), frame_no=1)
        for i in range(3):
            self.mem.mark_moving(1, 2 + i)
        # sack 1 is now picked_up=True; cleanup with empty active set removes it
        self.mem.cleanup(active_ids=set())
        self.assertIsNone(self.mem._objects.get(1))


if __name__ == "__main__":
    unittest.main()
