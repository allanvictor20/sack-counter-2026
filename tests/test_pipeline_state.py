"""
test_pipeline_state.py — Unit tests for the PipelineState typed container.

Verifies that the dict-like interface works correctly for all known keys,
and that sub-group typed attributes are accessible directly.

No main.py / OpenCV / YOLO imports.
"""

import unittest
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.pipeline.state import PipelineState


class TestPipelineStateDictInterface(unittest.TestCase):
    def setUp(self):
        self.state = PipelineState()

    def test_frame_read_write(self):
        self.assertIsNone(self.state["frame"])
        self.state["frame"] = "fake_frame"
        self.assertEqual(self.state["frame"], "fake_frame")

    def test_frame_no_read_write(self):
        self.assertEqual(self.state["_frame_no"], 0)
        self.state["_frame_no"] = 42
        self.assertEqual(self.state["_frame_no"], 42)

    def test_set_contains(self):
        self.assertIn("sack_owner", self.state)
        self.assertIn("confirmed_persons", self.state)
        self.assertNotIn("nonexistent_key", self.state)

    def test_get_default(self):
        self.assertIsNone(self.state.get("frame"))
        self.assertEqual(self.state.get("nonexistent_key", "default"), "default")

    def test_scalar_increment(self):
        """state['total_sacks_counted'] += n must work via __getitem__/__setitem__."""
        self.state["total_sacks_counted"] += 5
        self.assertEqual(self.state["total_sacks_counted"], 5)
        self.state["total_sacks_counted"] += 3
        self.assertEqual(self.state["total_sacks_counted"], 8)

    def test_dict_mutation_on_sack_owner(self):
        """Mutating the returned dict must be reflected on the next access."""
        self.state["sack_owner"][1] = 10
        self.assertEqual(self.state["sack_owner"][1], 10)
        self.state["sack_owner"].pop(1, None)
        self.assertNotIn(1, self.state["sack_owner"])

    def test_set_mutation_on_confirmed_persons(self):
        self.state["confirmed_persons"].add(99)
        self.assertIn(99, self.state["confirmed_persons"])
        self.state["confirmed_persons"].discard(99)
        self.assertNotIn(99, self.state["confirmed_persons"])

    def test_defaultdict_on_person_hit(self):
        """person_hit must behave as defaultdict(int) — missing keys return 0."""
        self.assertEqual(self.state["person_hit"][42], 0)
        self.state["person_hit"][42] += 1
        self.assertEqual(self.state["person_hit"][42], 1)

    def test_full_reassignment_of_dict(self):
        """Assigning a new dict to orphaned_sack_owner must work."""
        self.state["orphaned_sack_owner"][1] = 10
        self.state["orphaned_sack_owner"] = {s: p for s, p in
                                              self.state["orphaned_sack_owner"].items()
                                              if p != 10}
        self.assertNotIn(1, self.state["orphaned_sack_owner"])

    def test_n_anomalies_scalar(self):
        self.state["n_anomalies"] = 3
        self.assertEqual(self.state["n_anomalies"], 3)

    def test_delivery_log_append(self):
        self.state["delivery_log"].append({"frame": 1})
        self.assertEqual(len(self.state["delivery_log"]), 1)

    def test_persons_past_door_set_operations(self):
        self.state["persons_past_door"].add(5)
        self.assertIn(5, self.state["persons_past_door"])
        self.state["persons_past_door"].discard(5)
        self.assertNotIn(5, self.state["persons_past_door"])

    def test_unknown_key_raises_keyerror(self):
        with self.assertRaises(KeyError):
            _ = self.state["this_key_does_not_exist"]

    def test_pop_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.state.pop("frame")


class TestPipelineStateSubGroups(unittest.TestCase):
    """Verify typed sub-group attributes are accessible and consistent."""

    def setUp(self):
        self.state = PipelineState()

    def test_persons_subgroup(self):
        self.state.persons.person_boxes[1] = (0, 0, 100, 200)
        self.assertEqual(self.state["person_boxes"][1], (0, 0, 100, 200))

    def test_sacks_subgroup(self):
        self.state.sacks.sack_owner[2] = 10
        self.assertEqual(self.state["sack_owner"][2], 10)

    def test_delivery_subgroup(self):
        self.state.delivery.total_sacks_counted = 7
        self.assertEqual(self.state["total_sacks_counted"], 7)

    def test_guards_subgroup(self):
        self.state.guards.just_evicted_sacks.add(3)
        self.assertIn(3, self.state["just_evicted_sacks"])

    def test_frame_no_initialised_at_zero(self):
        """BUG #11: _frame_no must be in state from construction."""
        self.assertIn("_frame_no", self.state)
        self.assertEqual(self.state["_frame_no"], 0)


if __name__ == "__main__":
    unittest.main()
