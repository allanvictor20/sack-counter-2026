"""
door_polygon.py — Door polygon geometry for Sack Counter v22.

Key improvements over v21:
  - Normal-vector-based projection (replaces centroid distance).
  - Room-side click during calibration (removes top-edge assumption).
  - Polygon validation: convex hull ordering + convexity check.
  - project_onto_normal() replaces distance_to_centroid() everywhere.
  - Enhanced draw() shows normal arrow, corridor/room labels, approach zone.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


def _get_display_cap() -> tuple[int, int]:
    """
    Detect the user's actual screen resolution and return a safe display
    window cap, leaving margin for the OS title bar, taskbar, and window
    borders. Used by calibration windows to keep the full frame visible
    regardless of screen size.

    Falls back to a conservative 1024x576 cap if screen size can't be
    detected (e.g. headless/SSH environments, or tkinter unavailable).
    """
    margin_w, margin_h = 80, 120
    fallback = (1024, 576)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        if screen_w <= 0 or screen_h <= 0:
            return fallback
        return (max(640, screen_w - margin_w), max(360, screen_h - margin_h))
    except Exception:
        return fallback



@dataclass
class DoorPolygon:
    """
    Quadrilateral representing the door entrance.

    The door is modelled as a *portal* with two sides:
        - corridor_side  : where carriers approach from
        - room_side      : where carriers disappear into

    All direction logic uses projection onto ``normal_vec`` rather than
    Euclidean distance to the centroid, which was unreliable for angled
    cameras and non-square doorways.

    Args:
        points     : Four (x, y) tuples defining the polygon corners.
                     They are automatically hull-sorted clockwise so
                     order does not matter.
        room_point : A point (x, y) clicked inside the room by the user.
                     Used to determine which side of the portal is the room.

    Attributes
    ----------
    points          : list of four (x, y) tuples (clockwise, validated)
    np_points       : numpy array shape (4, 1, 2) dtype int32
    centroid        : (cx, cy) centre of the polygon
    normal_vec      : unit vector pointing FROM corridor TOWARD room
    corridor_midpt  : representative point on the corridor side
    room_midpt      : representative point on the room side
    approach_vec    : alias = -normal_vec  (kept for backward compat)
    """

    points:     list
    room_point: tuple = field(default=None)

    def __post_init__(self):
        assert len(self.points) == 4, "DoorPolygon requires exactly 4 points"

        # ── Validate & sort convex hull clockwise ─────────────
        pts = np.array(self.points, dtype=np.float32)
        hull = cv2.convexHull(pts.reshape(-1, 1, 2), clockwise=True)
        if hull is None or len(hull) < 4:
            raise ValueError(
                "Door polygon is degenerate (collinear or self-intersecting). "
                "Please re-draw the door."
            )
        hull_pts = hull.reshape(-1, 2)
        if len(hull_pts) != 4:
            raise ValueError(
                f"Door polygon convex hull has {len(hull_pts)} points "
                "(expected 4). Please re-draw the door."
            )
        if not cv2.isContourConvex(hull_pts.reshape(-1, 1, 2).astype(np.int32)):
            raise ValueError(
                "Door polygon is not convex after hull computation. "
                "Please re-draw the door."
            )
        # Plain ints, not numpy int32: these get written to the calibration
        # sidecar, and numpy scalars serialise as opaque !!python/object
        # tags that yaml.safe_load then refuses to read back.
        self.points = [(int(p[0]), int(p[1])) for p in hull_pts]
        self.np_points = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        self.centroid = (int(np.mean(xs)), int(np.mean(ys)))

        # ── Bounding box ──────────────────────────────────────
        self.polygon_bbox = (min(xs), min(ys), max(xs), max(ys))

        # ── Compute normal vector ─────────────────────────────
        # The normal points from corridor → room.
        #
        # This used to take the perpendicular of the p0→p1 edge and then
        # flip its SIGN using room_point.  Flipping a sign cannot fix a
        # wrong AXIS: p0→p1 is whichever edge cv2.convexHull happened to
        # order first, so for a portrait doorway (taller than it is wide)
        # the normal came out vertical when the room was to the side —
        # 90° wrong, which silently measured every crossing, projection
        # and approach-window test along the wrong axis.  Every existing
        # test used a SQUARE polygon, where the two axes are equivalent,
        # so nothing caught it.
        #
        # The axes now come from the polygon's minimum-area rectangle, and
        # room_point selects among all four candidate directions at once
        # (both axes, both signs) rather than only correcting the sign.
        cx, cy = self.centroid
        axes = self._rect_axes(pts)

        if self.room_point is not None:
            rx, ry = self.room_point
            to_room = np.array([rx - cx, ry - cy], dtype=float)
            if np.linalg.norm(to_room) < 1e-6:
                raw_normal = axes[0]
            else:
                to_room = to_room / np.linalg.norm(to_room)
                candidates = [a for axis in axes for a in (axis, -axis)]
                raw_normal = max(candidates, key=lambda a: float(np.dot(a, to_room)))
        else:
            # No room hint: assume the door's LONG side is its plane, so
            # the normal is the short axis.  axes[1] is the shorter one.
            raw_normal = axes[1]

        self.normal_vec = tuple(raw_normal.tolist())

        # Corridor and room midpoints (used for visual debug)
        offset = 80
        self.corridor_midpt = (
            int(cx - raw_normal[0] * offset),
            int(cy - raw_normal[1] * offset),
        )
        self.room_midpt = (
            int(cx + raw_normal[0] * offset),
            int(cy + raw_normal[1] * offset),
        )

        # Backward-compat alias
        self.approach_vec = (-raw_normal[0], -raw_normal[1])

    # ── Geometry helpers ──────────────────────────────────────

    @staticmethod
    def _rect_axes(pts: np.ndarray) -> list:
        """
        Return the polygon's two unit axes, LONGEST first.

        Uses the minimum-area rectangle rather than an arbitrary polygon
        edge, so the axes describe the shape's real orientation even when
        the corners were clicked in an odd order or the doorway is drawn
        at an angle.

        Args:
            pts: (N, 2) float32 array of polygon corners.

        Returns:
            ``[long_axis, short_axis]`` as unit numpy vectors.  Falls back
            to the image axes if the rectangle is degenerate.
        """
        rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
        box  = cv2.boxPoints(rect).astype(float)

        e1 = box[1] - box[0]
        e2 = box[3] - box[0]
        len1, len2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
        if len1 < 1e-6 or len2 < 1e-6:
            return [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

        u1, u2 = e1 / len1, e2 / len2
        return [u1, u2] if len1 >= len2 else [u2, u1]

    # ── Geometry queries ──────────────────────────────────────

    def contains(self, x: int, y: int) -> bool:
        """Return True if (x, y) is inside the polygon."""
        result = cv2.pointPolygonTest(self.np_points, (float(x), float(y)), False)
        return result >= 0

    def project_onto_normal(self, x: int, y: int) -> float:
        """
        Project (x, y) relative to centroid onto the door normal.

        Positive  → room side.
        Negative  → corridor side.
        Zero      → exactly at centroid.

        This is the canonical direction query replacing distance_to_centroid.
        """
        cx, cy = self.centroid
        nx, ny = self.normal_vec
        return float((x - cx) * nx + (y - cy) * ny)

    # Kept for backward compatibility — but internally replaced everywhere.
    def distance_to_centroid(self, x: int, y: int) -> float:
        """Euclidean distance from (x, y) to the polygon centroid."""
        return float(np.hypot(x - self.centroid[0], y - self.centroid[1]))

    def is_on_approach_side(self, x: int, y: int, margin_px: int = 50) -> bool:
        """
        Return True if (x, y) is clearly on the corridor side.

        Uses projection onto normal_vec.  Positive projection = room side,
        negative = corridor side, so corridor requires projection < -margin_px.
        """
        if self.contains(x, y):
            return False
        proj = self.project_onto_normal(x, y)
        return proj < -margin_px

    def is_past_polygon(self, x: int, y: int) -> bool:
        """
        Return True if (x, y) is on the room side (past the door).

        Uses normal projection: positive = room side.
        """
        proj = self.project_onto_normal(x, y)
        return proj > 20

    def movement_toward_door(
        self,
        old_x: int, old_y: int,
        cur_x: int, cur_y: int,
        threshold: float = 0.0,
    ) -> bool:
        """
        Return True if the movement vector has a positive component
        along the door normal (i.e. moving corridor → room).

        This replaces the centroid-distance convergence check, which
        could give false positives when a carrier approaches from the side.

        Args:
            old_x, old_y  : Previous position.
            cur_x, cur_y  : Current position.
            threshold     : Minimum dot product to count as "toward door".

        Returns:
            True if the carrier is moving toward the room side.
        """
        nx, ny = self.normal_vec
        dx = cur_x - old_x
        dy = cur_y - old_y
        dot = dx * nx + dy * ny
        return dot > threshold

    # ── Drawing ───────────────────────────────────────────────

    def draw(
        self,
        frame,
        color=(0, 200, 255),
        thickness: int = 2,
        label: str = "DOOR",
        count: int = 0,
        candidates: dict = None,
        show_debug: bool = True,
    ):
        """
        Draw the door polygon and optional debug overlays.

        New in v22:
          - Normal vector arrow showing room direction.
          - "CORRIDOR" and "ROOM" labels on each side.
          - Candidate state annotations per pid.

        Args:
            frame      : BGR numpy array.
            color      : Polygon edge color (B, G, R).
            thickness  : Line thickness.
            label      : Text label drawn at centroid.
            count      : Current delivery count shown next to label.
            candidates : dict[pid, DoorCandidateState] for debug overlay.
            show_debug : If True, draw normal arrow and side labels.
        """
        cv2.polylines(frame, [self.np_points], isClosed=True,
                      color=color, thickness=thickness)

        cv2.putText(
            frame, f"{label}: {count}",
            (self.centroid[0] - 40, self.centroid[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        # Corner indices
        for i, (px, py) in enumerate(self.points):
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.putText(frame, str(i + 1), (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if not show_debug:
            return

        # Normal vector arrow  (corridor → room)
        cx, cy = self.centroid
        arrow_end = self.room_midpt
        cv2.arrowedLine(
            frame, (cx, cy), arrow_end,
            (0, 255, 128), 2, tipLength=0.25,
        )

        # Side labels
        cv2.putText(
            frame, "ROOM",
            (self.room_midpt[0] - 20, self.room_midpt[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1,
        )
        cv2.putText(
            frame, "CORRIDOR",
            (self.corridor_midpt[0] - 35, self.corridor_midpt[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1,
        )

        # Candidate states
        if candidates:
            for pid, cand in candidates.items():
                from .pipeline.door_crossing import DoorState
                state_lbl = cand.door_state.name if hasattr(cand, "door_state") else "?"
                color_map = {
                    "CORRIDOR":     (180, 180, 180),
                    "APPROACHING":  (0, 200, 255),
                    "IN_DOOR":      (0, 255, 200),
                    "DISAPPEARING": (0, 140, 255),
                    "COUNTED":      (0, 255, 0),
                    "ABORTED":      (0, 0, 200),
                }
                c = color_map.get(state_lbl, (200, 200, 200))
                lx = cand.last_seen_cx
                ly = cand.last_seen_cy
                if lx > 0 or ly > 0:
                    cv2.circle(frame, (lx, ly), 6, c, 2)
                    cv2.putText(
                        frame, f"P#{pid}:{state_lbl[:3]}",
                        (lx + 8, ly - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1,
                    )


# ── Calibration ────────────────────────────────────────────────

def calibrate_door_polygon(cap, cfg: dict = None) -> "DoorPolygon":
    """
    Interactive calibration: show first frame, let user click 4 corners,
    then click once inside the room to set room direction.

    New in v22:
      - Phase 2 asks user to click a point inside the room.
      - Convex hull validation happens before DoorPolygon is created.
      - If the hull is invalid an error is shown and the user can redo.

    Controls:
        Left-click    : Place a corner (phase 1) or room point (phase 2)
        Right-click   : Remove last corner (phase 1 only)
        r             : Reset all points
        q             : Quit without calibrating

    Args:
        cap: cv2.VideoCapture — seeks to frame 0 to grab the first frame.
        cfg: Optional config dict updated with door_polygon_points, etc.

    Returns:
        DoorPolygon with the 4 user-selected points and room direction set.

    Raises:
        RuntimeError: if the user quits without completing calibration.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame for calibration.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Fit large frames to a reasonable on-screen window size ─────
    # cv2.imshow displays at native pixel size by default, which on a
    # 4K/1080p+ video opens a window larger than the screen — the user
    # then only sees the top-left corner cropped in, looking like an
    # unwanted zoom.  We scale the frame down for DISPLAY ONLY and
    # convert every mouse click back to original-frame coordinates via
    # `disp_scale`, so calibration accuracy is unaffected.
    max_w, max_h = _get_display_cap()
    fh, fw = frame.shape[:2]
    disp_scale = min(max_w / fw, max_h / fh, 1.0)  # never upscale
    disp_w, disp_h = int(fw * disp_scale), int(fh * disp_scale)

    points:     list  = []
    room_point: tuple = None
    clone = frame.copy()
    error_msg: str    = ""

    WIN = "Calibrate Door | Enter=confirm | r=reset | q=quit"

    def _mouse(event, x, y, flags, param):
        nonlocal room_point
        # Convert from displayed/window pixel space back to original
        # frame pixel space before storing the point.
        ox, oy = int(x / disp_scale), int(y / disp_scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((ox, oy))
            elif room_point is None:
                room_point = (ox, oy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if room_point is not None:
                room_point = None
            elif points:
                points.pop()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.moveWindow(WIN, 0, 0)
    cv2.setMouseCallback(WIN, _mouse)

    phase = 1  # 1 = clicking corners, 2 = clicking room point

    while True:
        display = clone.copy()

        # Draw placed points and edges
        for i, (px, py) in enumerate(points):
            cv2.circle(display, (px, py), 5, (0, 200, 255), -1)
            cv2.putText(display, str(i + 1), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(display, points[i], points[i + 1], (0, 200, 255), 2)
        if len(points) == 4:
            cv2.line(display, points[3], points[0], (0, 200, 255), 2)

        # Room point
        if room_point is not None:
            cv2.circle(display, room_point, 8, (0, 255, 128), -1)
            cv2.putText(display, "ROOM", (room_point[0] + 10, room_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

        # Status / instruction
        if len(points) < 4:
            status = f"PHASE 1/2 — Click 4 door corners  ({len(points)}/4)"
        elif room_point is None:
            status = "PHASE 2/2 — Click once INSIDE the room"
        else:
            status = "Ready — Press Enter to confirm, r to reset"

        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if error_msg:
            cv2.putText(display, f"ERROR: {error_msg}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow(WIN, cv2.resize(display, (disp_w, disp_h)))
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32) and len(points) == 4 and room_point is not None:
            # Validate polygon
            try:
                pts = np.array(points, dtype=np.float32)
                hull = cv2.convexHull(pts.reshape(-1, 1, 2), clockwise=True)
                if hull is None or len(hull.reshape(-1, 2)) != 4:
                    raise ValueError("Hull is not a valid quadrilateral.")
                if not cv2.isContourConvex(
                    hull.reshape(-1, 2).astype(np.int32).reshape(-1, 1, 2)
                ):
                    raise ValueError("Resulting polygon is not convex.")
                break  # validation passed
            except ValueError as exc:
                error_msg = str(exc) + " — please redraw."
                points.clear()
                room_point = None

        elif key == ord("r"):
            points.clear()
            room_point = None
            error_msg = ""
        elif key == ord("q"):
            cv2.destroyWindow(WIN)
            raise RuntimeError("Door calibration cancelled by user.")

    cv2.destroyWindow(WIN)

    door = DoorPolygon(points=points, room_point=room_point)

    if cfg is not None:
        cfg["door_polygon_points"] = door.points
        cfg["door_room_point"]     = room_point
        # approach_side kept for peak_window compat (left/right of centroid)
        nx, ny = door.normal_vec
        cfg["door_approach_side"] = "right" if nx < 0 else "left"

    return door
