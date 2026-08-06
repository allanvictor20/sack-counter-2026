"""
geometry.py — Where a person is, relative to the door.

Every stage that asks "has this carrier reached the door yet?" must ask
it the same way.  They did not: counting and crossing projected the
carrier onto the door normal, while the peak-cleanup and state-machine
stages compared raw ``person_prev_cx`` against ``door.centroid[0]`` with
a left/right ``door_approach_side`` flag.

A sign flip cannot fix a wrong axis.  For any doorway that is not close
to axis-aligned, the x-only comparison measured along a line that has
nothing to do with the real corridor/room boundary, so a carrier could
be "past the door" on one stage's arithmetic and "still approaching" on
another's.  ``_get_person_cy`` was also copied into two modules, which
is how the two halves drifted apart in the first place.

One helper, one axis, one definition of the approach window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..door_polygon import DoorPolygon


def person_centroid(state, pid: int) -> Optional[tuple[int, int]]:
    """
    Centroid of *pid*'s current box, or None if it has no box this frame.

    ``person_prev_cx`` is authoritative for the horizontal position (it
    survives a frame where the box is missing), so it wins when present;
    the vertical component can only come from the box.

    Args:
        state: PipelineState.
        pid:   Person tracker ID.

    Returns:
        ``(cx, cy)``, or None when the person has no known position.
    """
    box = state["person_boxes"].get(pid)
    cx = state["person_prev_cx"].get(pid)
    if box is None:
        return None if cx is None else (int(cx), 0)
    x1, y1, x2, y2 = box
    if cx is None:
        cx = (x1 + x2) // 2
    return int(cx), (y1 + y2) // 2


def person_projection(door: "DoorPolygon", state, pid: int) -> Optional[float]:
    """
    Signed projection of *pid* onto the door normal.

    Negative is the corridor side, positive is the room side — the same
    convention every other geometry test in the pipeline uses.

    Args:
        door:  Calibrated DoorPolygon.
        state: PipelineState.
        pid:   Person tracker ID.

    Returns:
        The projection, or None when the person has no known position.
    """
    point = person_centroid(state, pid)
    if point is None:
        return None
    return door.project_onto_normal(point[0], point[1])


def in_approach_window(proj: Optional[float], window_px: float,
                       freeze_px: float) -> bool:
    """
    True when a projection falls inside the peak-counting window.

    The window opens ``window_px`` back on the corridor side and closes
    ``freeze_px`` past the door plane, at which point the peak freezes so
    a carrier's load cannot change while they are in the doorway.

    Args:
        proj:      Signed projection, or None (never in the window).
        window_px: How far back on the corridor side the window opens.
        freeze_px: How far past the plane it closes.
    """
    if proj is None:
        return False
    return -float(window_px) <= proj <= float(freeze_px)


def has_approached(proj: Optional[float], window_px: float) -> bool:
    """
    True once a carrier has reached the approach window at all.

    Unlike :func:`in_approach_window` this has no far edge — it stays
    True through the doorway and beyond, which is what a one-way
    lifecycle transition (CARRIED → APPROACHING) needs.

    Args:
        proj:      Signed projection, or None (not approaching).
        window_px: How far back on the corridor side the window opens.
    """
    if proj is None:
        return False
    return proj >= -float(window_px)
