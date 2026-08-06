"""
test_geometry.py — Where a person is, relative to the door.

The stages that ask this question used to disagree: counting and
crossing projected onto the door normal, while peak-cleanup and the
sack state machine compared raw x against the door centroid with a
left/right flag.  For an angled doorway those are different questions,
so a carrier could be "past the door" to one stage and "still
approaching" to another.

The diagonal-door cases below are the ones the old x-only arithmetic
got wrong, and they are why this module exists.
"""

import pytest

from sack_counter.door_polygon import DoorPolygon
from sack_counter.pipeline.geometry import (
    has_approached, in_approach_window, person_centroid, person_projection,
)
from sack_counter.pipeline.state import PipelineState


def _vertical_door() -> DoorPolygon:
    """Door plane runs top-to-bottom; the room is to the right (+x)."""
    return DoorPolygon(
        points=[(500, 100), (560, 100), (560, 500), (500, 500)],
        room_point=(900, 300),
    )


def _diagonal_door() -> DoorPolygon:
    """A doorway seen at an angle — the case x-only comparison broke on."""
    return DoorPolygon(
        points=[(300, 100), (600, 250), (560, 330), (260, 180)],
        room_point=(500, 600),
    )


def _state_with(pid: int, box=None, cx=None) -> PipelineState:
    state = PipelineState()
    if box is not None:
        state["person_boxes"][pid] = box
    if cx is not None:
        state["person_prev_cx"][pid] = cx
    return state


# ── person_centroid ───────────────────────────────────────────

class TestPersonCentroid:
    def test_from_box(self):
        state = _state_with(1, box=(100, 200, 200, 400))
        assert person_centroid(state, 1) == (150, 300)

    def test_prev_cx_wins_over_the_box(self):
        """
        person_prev_cx survives a frame where the box is missing, so it
        is the authoritative horizontal position when the two disagree.
        """
        state = _state_with(1, box=(100, 200, 200, 400), cx=175)
        assert person_centroid(state, 1) == (175, 300)

    def test_cx_only_still_yields_a_point(self):
        state = _state_with(1, cx=175)
        assert person_centroid(state, 1) == (175, 0)

    def test_unknown_person_is_none(self):
        assert person_centroid(PipelineState(), 99) is None


# ── person_projection ─────────────────────────────────────────

class TestPersonProjection:
    def test_corridor_side_is_negative(self):
        door = _vertical_door()
        state = _state_with(1, box=(100, 250, 200, 350))   # left of door
        assert person_projection(door, state, 1) < 0

    def test_room_side_is_positive(self):
        door = _vertical_door()
        state = _state_with(1, box=(800, 250, 900, 350))   # right of door
        assert person_projection(door, state, 1) > 0

    def test_unknown_person_is_none(self):
        assert person_projection(_vertical_door(), PipelineState(), 9) is None

    def test_diagonal_door_disagrees_with_raw_x(self):
        """
        The heart of the bug.  This carrier is on the ROOM side of an
        angled doorway, but sits at a smaller x than the door centroid —
        so the old "past the door means cx < centroid_x" test called it
        approaching while the projection correctly calls it through.
        """
        door = _diagonal_door()
        # Below the door plane (room side, +y) but left of the centroid.
        state = _state_with(1, box=(300, 600, 400, 700))
        proj = person_projection(door, state, 1)
        assert proj > 0                       # genuinely through the door
        assert person_centroid(state, 1)[0] < door.centroid[0]


# ── Window predicates ─────────────────────────────────────────

class TestApproachWindow:
    def test_inside_the_window(self):
        assert in_approach_window(-100.0, window_px=350, freeze_px=40)

    def test_too_far_back_is_outside(self):
        assert not in_approach_window(-400.0, window_px=350, freeze_px=40)

    def test_past_the_freeze_edge_is_outside(self):
        """Past the freeze edge the peak is held, so the carrier leaves
        the counting window even though they are still visible."""
        assert not in_approach_window(60.0, window_px=350, freeze_px=40)

    def test_boundaries_are_inclusive(self):
        assert in_approach_window(-350.0, 350, 40)
        assert in_approach_window(40.0, 350, 40)

    def test_none_is_never_in_the_window(self):
        assert not in_approach_window(None, 350, 40)

    def test_has_approached_has_no_far_edge(self):
        """
        The lifecycle transition is one-way: once a carrier has reached
        the window they stay 'approached' through the doorway and out
        the other side.
        """
        assert has_approached(-350.0, 350)
        assert has_approached(0.0, 350)
        assert has_approached(5000.0, 350)

    def test_has_approached_false_before_the_window(self):
        assert not has_approached(-400.0, 350)

    def test_has_approached_none_is_false(self):
        assert not has_approached(None, 350)


# ── Cross-stage agreement ─────────────────────────────────────

class TestStagesAgree:
    @pytest.mark.parametrize("door", [_vertical_door(), _diagonal_door()])
    def test_window_membership_implies_approached(self, door):
        """
        Anything the counting window accepts must also read as having
        approached, or a carrier could be counted by one stage while the
        state machine still thinks they are in the corridor.
        """
        for proj in range(-500, 500, 25):
            if in_approach_window(float(proj), 350, 40):
                assert has_approached(float(proj), 350)
