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
        self.assertNotIn("gate_gap_px", cfg)
        # door_approach_side was a left/right flag from before the
        # normal-vector rewrite.  Its last two readers were the axis bug
        # that rewrite was meant to remove, so the key is gone rather
        # than left in the config looking like a live tuning knob.
        self.assertNotIn("door_approach_side", cfg)


if __name__ == "__main__":
    unittest.main()
