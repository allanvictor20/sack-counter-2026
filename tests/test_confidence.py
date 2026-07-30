"""
test_confidence.py — Unit tests for confidence scoring.

No main.py / OpenCV / YOLO imports.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.config import DEFAULT_CFG
from sack_counter.confidence import ConfidenceTracker, confidence_class


def _cfg(**overrides):
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    return cfg


class TestConfidenceClass(unittest.TestCase):
    def setUp(self):
        self.cfg = _cfg(conf_class_high=0.80, conf_class_medium=0.60)

    def test_high(self):
        self.assertEqual(confidence_class(0.85, self.cfg), "HIGH")
        self.assertEqual(confidence_class(0.80, self.cfg), "HIGH")

    def test_medium(self):
        self.assertEqual(confidence_class(0.70, self.cfg), "MEDIUM")
        self.assertEqual(confidence_class(0.60, self.cfg), "MEDIUM")

    def test_low(self):
        self.assertEqual(confidence_class(0.50, self.cfg), "LOW")
        self.assertEqual(confidence_class(0.00, self.cfg), "LOW")


class TestConfidenceTracker(unittest.TestCase):
    def setUp(self):
        self.cfg = _cfg(
            conf_weight_detection=0.30,
            conf_weight_tracking=0.30,
            conf_weight_ownership=0.25,
            conf_weight_crossing=0.15,
            min_delivery_confidence=0.55,
        )
        self.tracker = ConfidenceTracker(self.cfg)

    def test_delivery_confidence_in_unit_range(self):
        self.tracker.record_detection(1, 0.8)
        self.tracker.record_ownership(1, 0.7)
        self.tracker.record_gate_b(1)
        conf = self.tracker.delivery_confidence(1)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_high_confidence_flag(self):
        for _ in range(30):
            self.tracker.record_detection(1, 0.95)
        self.tracker.record_ownership(1, 0.90)
        self.tracker.record_gate_b(1)
        self.assertTrue(self.tracker.is_high_confidence(1))

    def test_remove_clears_single_sack(self):
        """BUG #5: remove(sid) should clear only that sack, not others."""
        self.tracker.record_detection(1, 0.8)
        self.tracker.record_detection(2, 0.9)
        self.tracker.remove(1)
        # sack 1 gone
        self.assertNotIn(1, self.tracker._det)
        # sack 2 untouched
        self.assertIn(2, self.tracker._det)

    def test_remove_is_idempotent(self):
        """remove() on an unknown sid must not raise."""
        self.tracker.remove(999)

    def test_cleanup_keeps_active_removes_others(self):
        """cleanup(active_sids) keeps active sacks and removes the rest."""
        self.tracker.record_detection(1, 0.8)
        self.tracker.record_detection(2, 0.7)
        self.tracker.cleanup({1})   # keep only sack 1
        self.assertIn(1, self.tracker._det)
        self.assertNotIn(2, self.tracker._det)

    def test_cleanup_semantics_are_keep_not_remove(self):
        """Verify cleanup({sid}) removes everything EXCEPT sid — it's a keep-set."""
        self.tracker.record_detection(1, 0.8)
        self.tracker.record_detection(2, 0.7)
        self.tracker.cleanup({2})   # keep sack 2
        self.assertIn(2, self.tracker._det)
        self.assertNotIn(1, self.tracker._det)

    def test_gate_a_sets_half_score(self):
        self.tracker.record_gate_a(1)
        self.assertAlmostEqual(self.tracker._crs[1], 0.5)

    def test_gate_b_sets_full_score(self):
        self.tracker.record_gate_b(1)
        self.assertAlmostEqual(self.tracker._crs[1], 1.0)

    def test_unknown_sack_returns_default_confidence(self):
        conf = self.tracker.delivery_confidence(999)
        self.assertGreaterEqual(conf, 0.0)


if __name__ == "__main__":
    unittest.main()
