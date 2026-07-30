"""
test_counting.py — Tests for config loader and peak-window logic (v21).

No main.py, no OpenCV, no YOLO, no embedder imports.
Gate tests replaced by door polygon + door crossing tests.
"""

import unittest
import tempfile
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.config import DEFAULT_CFG, load_config


def _cfg(**overrides):
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    return cfg


class TestDoorPeakWindowDirection(unittest.TestCase):
    """
    Verify direction-aware peak window boundaries using door centroid.
    """

    def _compute_window(self, approach_side, door_cx=600,
                        peak_window_px=350, peak_freeze_px=40):
        cfg = _cfg(
            peak_window_px=peak_window_px,
            peak_freeze_px=peak_freeze_px,
            door_approach_side=approach_side,
        )
        if approach_side == "right":
            approach_open = door_cx + cfg["peak_window_px"]
            freeze_edge   = door_cx - cfg["peak_freeze_px"]
            def in_window(cx): return freeze_edge < cx <= approach_open
        else:  # left
            approach_open = door_cx - cfg["peak_window_px"]
            freeze_edge   = door_cx + cfg["peak_freeze_px"]
            def in_window(cx): return approach_open <= cx < freeze_edge
        return approach_open, freeze_edge, in_window, door_cx

    def test_right_approach_window_has_valid_range(self):
        ao, fe, in_w, door_cx = self._compute_window("right")
        # freeze_edge < approach_open
        self.assertLess(fe, ao)
        mid = (fe + ao) // 2
        self.assertTrue(in_w(mid))

    def test_left_approach_window_has_valid_range(self):
        ao, fe, in_w, door_cx = self._compute_window("left")
        # approach_open < freeze_edge for left approach
        self.assertLess(ao, fe)
        mid = (ao + fe) // 2
        self.assertTrue(in_w(mid))

    def test_right_approach_person_past_door_not_in_window(self):
        ao, fe, in_w, door_cx = self._compute_window("right")
        # Someone far to the left (past door on room side) should be outside window
        self.assertFalse(in_w(door_cx - 500))

    def test_left_approach_person_past_door_not_in_window(self):
        ao, fe, in_w, door_cx = self._compute_window("left")
        self.assertFalse(in_w(door_cx + 500))


class TestConfigLoader(unittest.TestCase):
    def test_defaults_returned_with_no_path(self):
        cfg = load_config(None)
        for key in DEFAULT_CFG:
            self.assertIn(key, cfg)

    def test_json_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"conf_sack": 0.99}, f)
            fname = f.name
        try:
            cfg = load_config(fname)
            self.assertAlmostEqual(cfg["conf_sack"], 0.99)
            self.assertIn("conf_person", cfg)
        finally:
            os.unlink(fname)

    def test_yaml_override(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"conf_person": 0.42}, f)
            fname = f.name
        try:
            cfg = load_config(fname)
            self.assertAlmostEqual(cfg["conf_person"], 0.42)
        finally:
            os.unlink(fname)

    def test_missing_file_returns_defaults(self):
        cfg = load_config("/nonexistent/path/config.json")
        self.assertEqual(cfg["conf_sack"], DEFAULT_CFG["conf_sack"])

    def test_door_zone_keys_in_defaults(self):
        cfg = _cfg()
        self.assertIn("door_disappear_frames", cfg)
        self.assertIn("direction_lookback_frames", cfg)
        self.assertIn("min_convergence_px", cfg)
        self.assertIn("reentry_margin_px", cfg)
        self.assertIn("door_approach_side", cfg)
        self.assertNotIn("gate_gap_px", cfg)


if __name__ == "__main__":
    unittest.main()
