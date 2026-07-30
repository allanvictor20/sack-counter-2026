"""
conftest.py — Shared fixtures for exit counter tests.

All fixtures are self-contained — no YOLO model, no video file,
no GPU required.  Tests run purely on in-memory state.
"""

import pytest
import sys
import os

# Make sure the package root is on the path regardless of how pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sack_counter.exit.exit_state import ExitPipelineState
from sack_counter.exit.landing_zone import LandingZone
from sack_counter.exit.exit_landing import LandingZoneTracker


# ── Minimal DoorPolygon stub ───────────────────────────────────

class FakeDoorPolygon:
    """
    Minimal stub of DoorPolygon for use in tests.

    The door is a narrow vertical strip from x=310 to x=330, centered at
    x=320, in a 640×480 frame. normal_vec points LEFT (corridor is to
    the left/lower-x side, room is to the right/higher-x side).

    project_onto_normal(x, y):
        positive → room side     (x > 320)
        negative → corridor side (x < 320)

    contains(x, y):
        True only for points genuinely inside the door strip
        (310 <= x <= 330), matching the real DoorPolygon's
        point-in-polygon behavior.
    """

    def __init__(self):
        self.centroid   = (320, 240)
        self.normal_vec = (-1.0, 0.0)   # points corridor→room (left)
        self.points     = [(310, 100), (330, 100), (330, 380), (310, 380)]

    def project_onto_normal(self, x: int, y: int) -> float:
        cx, cy = self.centroid
        nx, ny = self.normal_vec
        return float((x - cx) * nx + (y - cy) * ny)

    def is_on_approach_side(self, x: int, y: int, margin_px: int = 50) -> bool:
        proj = self.project_onto_normal(x, y)
        return proj < -margin_px

    def contains(self, x: int, y: int) -> bool:
        """True if (x, y) is inside the door polygon's bounding strip."""
        x1 = min(p[0] for p in self.points)
        x2 = max(p[0] for p in self.points)
        y1 = min(p[1] for p in self.points)
        y2 = max(p[1] for p in self.points)
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw(self, frame, **kwargs):
        pass   # no-op in tests


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def door():
    """Fake door polygon — door at x=320, corridor to the left."""
    return FakeDoorPolygon()


@pytest.fixture
def landing():
    """Landing zone polygon: rectangle from (50,300) to (300,450)."""
    return LandingZone(points=[(50, 300), (300, 300), (300, 450), (50, 450)])


@pytest.fixture
def state():
    """Fresh ExitPipelineState for each test."""
    return ExitPipelineState()


@pytest.fixture
def motion_tracker():
    """
    Fresh LandingZoneTracker for each test.

    Uses a fast, test-friendly configuration: a high speed threshold
    (10.0 px/frame) so small synthetic position deltas easily register
    as "still" in unit tests, a short still_frames requirement (3) so
    tests don't need long detection sequences, and a small grace period
    (3 frames) so brief intentional detection gaps in flicker-simulation
    tests don't wipe progress before they're meant to.
    """
    return LandingZoneTracker(
        still_speed_thresh=10.0,
        still_speed_window=8,
        still_frames=3,
        miss_grace_frames=3,
    )


@pytest.fixture
def sack_det():
    """
    Factory that returns a sack detection tuple.
    Usage: sack_det(sid=1, cx=200, cy=350)
    """
    def _make(sid: int, cx: int = 200, cy: int = 350, conf: float = 0.9):
        w, h = 60, 80
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        return (sid, x1, y1, x2, y2, cx, cy, conf)
    return _make
