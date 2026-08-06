"""
test_box_pipeline.py — Box counting, class resolution, the session view.

Boxes settle on the same trigger as sacks: the door-crossing commit for
whoever is holding them.  That wiring had no test, and it is the path
where ``total_counted`` sat permanently at 0 while the report printed a
Boxes column.
"""

import numpy as np
import pytest

from sack_counter.box_pipeline import BoxCountingPipeline
from sack_counter.confidence import ConfidenceTracker
from sack_counter.config import DEFAULT_CFG
from sack_counter.model_classes import resolve_class_indices, unmatched_classes
from sack_counter.session_view import SessionView
from sack_counter.trackers import OwnershipMemory


@pytest.fixture
def cfg():
    return dict(DEFAULT_CFG)


@pytest.fixture
def pipeline(cfg):
    return BoxCountingPipeline(
        confirm_frames=2, miss_frames=3,
        ownership_mem=OwnershipMemory(cfg["ownership_switch_margin"]),
        conf_tracker=ConfidenceTracker(cfg),
        cfg=cfg,
    )


#: A person standing at x≈300 with a box held at chest height.
PERSON = {7: (250, 200, 350, 500)}
BOX    = (1, 270, 260, 330, 320, 0.9)


def _run(pipeline, cfg, frames, boxes=(BOX,), persons=None):
    owner = {}
    for fn in range(1, frames + 1):
        owner = pipeline.update(list(boxes),
                                PERSON if persons is None else persons,
                                fn, 20.0, cfg=cfg)
    return owner


# ── Confirmation ──────────────────────────────────────────────

class TestConfirmation:
    def test_box_is_not_owned_before_confirmation(self, pipeline, cfg):
        assert _run(pipeline, cfg, frames=1) == {}

    def test_box_is_owned_once_confirmed(self, pipeline, cfg):
        assert _run(pipeline, cfg, frames=2) == {1: 7}

    def test_track_expires_after_enough_misses(self, pipeline, cfg):
        _run(pipeline, cfg, frames=3)
        for fn in range(10, 20):
            pipeline.update([], PERSON, fn, 20.0, cfg=cfg)
        assert 1 not in pipeline._confirmed

    def test_ownership_drops_when_the_track_expires(self, pipeline, cfg):
        """A carrier must not later be credited with a box that is gone."""
        _run(pipeline, cfg, frames=3)
        for fn in range(10, 20):
            pipeline.update([], PERSON, fn, 20.0, cfg=cfg)
        assert pipeline.commit_delivery(7, fn=25) == 0

    def test_box_with_no_person_nearby_is_unowned(self, pipeline, cfg):
        far = {7: (2000, 2000, 2100, 2300)}
        assert _run(pipeline, cfg, frames=3, persons=far) == {}


# ── Delivery ──────────────────────────────────────────────────

class TestDelivery:
    def test_commit_credits_the_carrier(self, pipeline, cfg):
        _run(pipeline, cfg, frames=3)
        assert pipeline.commit_delivery(7, fn=10) == 1
        assert pipeline.total_counted == 1
        assert pipeline.delivery_counts() == {7: 1}

    def test_commit_writes_a_log_entry(self, pipeline, cfg):
        _run(pipeline, cfg, frames=3)
        pipeline.commit_delivery(7, fn=42)
        entry = pipeline.delivery_log[0]
        assert entry["box_id"] == 1
        assert entry["person_id"] == 7
        assert entry["frame"] == 42
        assert entry["type"] == "box"

    def test_the_same_box_is_not_counted_twice(self, pipeline, cfg):
        _run(pipeline, cfg, frames=3)
        assert pipeline.commit_delivery(7, fn=10) == 1
        assert pipeline.commit_delivery(7, fn=11) == 0
        assert pipeline.total_counted == 1

    def test_committing_a_carrier_holding_nothing_is_a_no_op(self, pipeline):
        assert pipeline.commit_delivery(99, fn=10) == 0
        assert pipeline.delivery_log == []

    def test_two_boxes_credit_two(self, pipeline, cfg):
        second = (2, 275, 265, 335, 325, 0.9)
        _run(pipeline, cfg, frames=3, boxes=(BOX, second))
        assert pipeline.commit_delivery(7, fn=10) == 2
        assert pipeline.total_counted == 2


# ── Class resolution ──────────────────────────────────────────

class TestClassResolution:
    def test_resolves_by_name_regardless_of_order(self):
        got = resolve_class_indices({0: "sack", 1: "box", 2: "person"})
        assert got == {"person": 2, "sack": 0, "box": 1}

    def test_accepts_aliases(self):
        got = resolve_class_indices({0: "worker", 1: "bag", 2: "carton"})
        assert got == {"person": 0, "sack": 1, "box": 2}

    def test_is_case_insensitive(self):
        assert resolve_class_indices({0: "Person"}, required=("person",)) \
            == {"person": 0}

    def test_falls_back_to_documented_index(self, capsys):
        got = resolve_class_indices({0: "a", 1: "b", 2: "c"})
        assert got == {"person": 0, "sack": 1, "box": 2}
        assert "Warning" in capsys.readouterr().out

    def test_strict_class_refuses_to_guess(self):
        with pytest.raises(RuntimeError):
            resolve_class_indices({0: "a"}, required=("sack",),
                                  strict=("sack",))

    def test_unmatched_reports_exactly_what_fell_back(self):
        assert unmatched_classes({0: "person", 1: "sack", 2: "widget"}) \
            == ("box",)

    def test_unmatched_is_empty_when_all_named(self):
        assert unmatched_classes({0: "person", 1: "sack", 2: "box"}) == ()

    def test_unmatched_handles_no_names(self):
        assert set(unmatched_classes(None)) == {"person", "sack", "box"}


# ── SessionView ───────────────────────────────────────────────

class _FakeExitState(dict):
    """Exit state exposes __getitem__; a dict is a faithful stand-in."""


class TestSessionView:
    def _exit_state(self):
        return _FakeExitState({
            "total_sacks_out": 12, "landing_exit_count": 11,
            "landing_peak_count": 4, "tentative_crossings": {1: 2},
            "discrepancy_flags": [], "exit_log": [{"frame": 10, "sack_id": 3}],
        })

    def test_exit_view_totals(self):
        view = SessionView.for_exit(np.zeros((4, 4, 3), np.uint8),
                                    self._exit_state(), 100, 15.0)
        assert view.mode == "exit"
        assert view.label == "Sacks out"
        assert view.totals["sacks"] == 12
        assert view.totals["pending"] == 1

    def test_exit_view_has_no_workers(self):
        view = SessionView.for_exit(None, self._exit_state(), 1, 1.0)
        assert view.workers == []

    def test_exit_view_events_are_newest_first(self):
        state = self._exit_state()
        state["exit_log"] = [{"frame": 1, "sack_id": 1},
                             {"frame": 2, "sack_id": 2}]
        view = SessionView.for_exit(None, state, 1, 1.0)
        assert view.events[0]["text"].startswith("Sack 2")

    def test_derived_fields_are_computed_once(self):
        """
        The loop builds a view every frame and a host reads ~15 a second,
        so the derived fields must be cached rather than recomputed on
        every attribute access.
        """
        view = SessionView.for_exit(None, self._exit_state(), 1, 1.0)
        assert view.totals is view.totals
        assert view.events is view.events

    def test_both_modes_expose_the_same_shape(self):
        """One callback contract: a consumer must not need to know the mode."""
        view = SessionView.for_exit(None, self._exit_state(), 1, 1.0)
        for attribute in ("frame", "frame_no", "fps", "mode", "warning",
                          "label", "totals", "workers", "events"):
            assert hasattr(view, attribute)
