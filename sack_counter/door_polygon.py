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

from . import theme as T


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
        color=None,
        thickness: int | None = None,
        label: str = "DOOR",
        count: int = 0,
        show_debug: bool = True,
    ):
        """
        Draw the door polygon and optional debug overlays.

        Styled from the design's live view: an accent outline over a 10%
        accent wash, a letter-spaced uppercase label, and numbered accent
        squares on the corners matching the calibration screen.

        Args:
            frame      : BGR numpy array.
            color      : Polygon edge color (B, G, R).  Defaults to the
                         design accent.
            thickness  : Line thickness.  Defaults to the design's 3 px,
                         scaled to the frame.
            label      : Text label drawn at centroid.
            count      : Current delivery count shown next to label.
            show_debug : If True, draw normal arrow and side labels.
        """
        u = T.unit(frame)
        if color is None:
            color = T.ACCENT
        if thickness is None:
            thickness = T.px(3, u)

        T.fill_polygon(frame, self.np_points, color, alpha=0.10)
        cv2.polylines(frame, [self.np_points], isClosed=True,
                      color=color, thickness=thickness, lineType=cv2.LINE_AA)

        # Label sits above the doorway's highest corner, as in the design —
        # the centroid belongs to the room/corridor arrow.
        cx, cy = self.centroid
        kick_scale = T.fs(11, u)
        num_scale  = T.fs(20, u)
        kick_h = T.text_h(kick_scale, 1, T.FONT_BODY)
        num_h  = T.text_h(num_scale, 2, T.FONT_HEAD)
        anchor = min(self.points, key=lambda p: p[1])
        lx = anchor[0] + T.px(14, u)
        base = anchor[1] - T.px(8, u)
        if base - kick_h - num_h < 0:      # doorway hugs the frame top
            base = anchor[1] + T.px(10, u) + num_h + kick_h + T.px(6, u)
        T.text(frame, str(count), lx, base, num_scale, color, 2, T.FONT_HEAD)
        T.text(frame, label.upper(), lx, base - num_h - T.px(6, u),
               kick_scale, color, 1, T.FONT_BODY,
               tracking=max(1.0, 11 * u * 0.12))

        # Corner markers — small filled squares, numbered in reverse ink.
        marker = T.px(13, u)
        ink = T.contrast_ink(color)
        mark_scale = T.fs(11, u)
        for i, (px_, py_) in enumerate(self.points):
            T.fill(frame, px_ - marker // 2, py_ - marker // 2,
                   px_ - marker // 2 + marker, py_ - marker // 2 + marker,
                   color, 1.0)
            digit = str(i + 1)
            dw = T.text_w(digit, mark_scale, 1, T.FONT_HEAD)
            dh = T.text_h(mark_scale, 1, T.FONT_HEAD)
            T.text(frame, digit, px_ - dw // 2, py_ + dh // 2,
                   mark_scale, ink, 1, T.FONT_HEAD)

        if not show_debug:
            return

        # Normal vector arrow  (corridor → room)
        arrow_end = self.room_midpt
        cv2.arrowedLine(
            frame, (cx, cy), arrow_end,
            T.PAPER, T.px(2, u), cv2.LINE_AA, tipLength=0.25,
        )

        # Side labels — quiet, so they never compete with the door itself.
        side_scale = T.fs(10, u)
        side_track = max(1.0, 10 * u * 0.12)
        T.text(frame, "ROOM", self.room_midpt[0] - T.px(20, u),
               self.room_midpt[1] + T.px(18, u), side_scale, T.PAPER, 1,
               T.FONT_BODY, tracking=side_track)
        T.text(frame, "CORRIDOR", self.corridor_midpt[0] - T.px(35, u),
               self.corridor_midpt[1] + T.px(18, u), side_scale,
               T.NEUTRAL_300, 1, T.FONT_BODY, tracking=side_track)


# ── Calibration ────────────────────────────────────────────────

# Plain-English prompts, one per click, from the design's calibration
# screen.  Index = number of corners already placed.
_CAL_HINTS = (
    "Click the first corner of the doorway.",
    "Click the next corner, going around the doorway.",
    "Two more corners to go.",
    "One more corner and the doorway is outlined.",
    "Doorway outlined. Now click any spot inside the room, "
    "so the system knows which way is in.",
    "Done — press Enter to save. You will not need to do this "
    "again for this camera.",
)

_CAL_STEPS = (
    ("1", "Outline the doorway",
     "Click its four corners in order. A red box marks each click."),
    ("2", "Mark the inside", "Click one spot on the floor inside the room."),
    ("3", "Save", "The outline is remembered until the camera moves."),
)


def _draw_calibration_overlay(display, points, room_point, error_msg) -> None:
    """
    Render the calibration screen's overlay onto *display* in place.

    Purely presentational — it reads the click list and draws; the click
    handling and validation above are untouched.
    """
    u = T.unit(display)
    fh, fw = display.shape[:2]

    # ── Doorway outline in progress ───────────────────────────
    if len(points) >= 3:
        T.fill_polygon(display, np.array(points, dtype=np.int32),
                       T.ACCENT, alpha=0.12)
    for i in range(len(points) - 1):
        cv2.line(display, points[i], points[i + 1], T.ACCENT,
                 T.px(3, u), cv2.LINE_AA)
    if len(points) == 4:
        cv2.line(display, points[3], points[0], T.ACCENT,
                 T.px(3, u), cv2.LINE_AA)

    marker = T.px(26, u)
    mark_scale = T.fs(17, u)
    for i, (px_, py_) in enumerate(points):
        T.fill(display, px_ - marker // 2, py_ - marker // 2,
               px_ - marker // 2 + marker, py_ - marker // 2 + marker,
               T.ACCENT, 1.0)
        digit = str(i + 1)
        dw = T.text_w(digit, mark_scale, 2, T.FONT_HEAD)
        dh = T.text_h(mark_scale, 2, T.FONT_HEAD)
        T.text(display, digit, px_ - dw // 2, py_ + dh // 2,
               mark_scale, T.PAPER, 2, T.FONT_HEAD)

    # ── Room point ────────────────────────────────────────────
    if room_point is not None:
        dot = T.px(18, u)
        T.fill(display, room_point[0] - dot // 2, room_point[1] - dot // 2,
               room_point[0] + dot // 2, room_point[1] + dot // 2,
               T.PAPER, 1.0)
        T.chip(display, "INSIDE", room_point[0] + dot,
               room_point[1] - dot // 2, u, T.PAPER, T.INK,
               size=10.5, tracking=max(1.0, 10.5 * u * 0.12))

    # ── Step list, top-left ───────────────────────────────────
    done = len(points) if room_point is None else 5
    active = 0 if done < 4 else (1 if done < 5 else 2)
    pad_x, pad_y = T.px(14, u), T.px(12, u)
    step_w = T.px(300, u)
    title_scale = T.fs(13.5, u)
    body_scale  = T.fs(12, u)
    title_h = T.text_h(title_scale, 1, T.FONT_HEAD)
    body_h  = T.text_h(body_scale, 1, T.FONT_BODY)

    y = 0
    for i, (num, title, body) in enumerate(_CAL_STEPS):
        on = i == active
        lines = T.wrap(body, step_w - pad_x * 2 - T.px(24, u),
                       body_scale, 1, T.FONT_BODY, max_lines=2)
        card_h = pad_y * 2 + title_h + T.px(5, u) + \
            int(body_h * 1.5) * len(lines)
        T.fill(display, 0, y, step_w, y + card_h, T.SCRIM,
               T.SCRIM_ALPHA if on else 0.62)
        T.edge(display, 0, y, y + card_h,
               T.ACCENT if on else T.NEUTRAL_700, T.px(3, u))
        nx = T.px(3, u) + pad_x
        T.text(display, num, nx, y + pad_y + title_h, T.fs(13, u),
               T.ACCENT if on else T.NEUTRAL_500, 1, T.FONT_HEAD)
        tx = nx + T.px(18, u)
        T.text(display, title, tx, y + pad_y + title_h, title_scale,
               T.PAPER if on else T.NEUTRAL_400, 1, T.FONT_HEAD)
        by = y + pad_y + title_h + T.px(5, u) + body_h
        for line in lines:
            T.text(display, line, tx, by, body_scale, T.NEUTRAL_400,
                   1, T.FONT_BODY)
            by += int(body_h * 1.5)
        y += card_h + T.px(2, u)

    # ── Hint bar, bottom-left ─────────────────────────────────
    hint = _CAL_HINTS[min(done, len(_CAL_HINTS) - 1)]
    hint_scale = T.fs(12.5, u)
    hint_h = T.text_h(hint_scale, 1, T.FONT_BODY)
    hpad_x, hpad_y = T.px(16, u), T.px(9, u)
    bar_h = hpad_y * 2 + hint_h
    bar_w = min(fw, hpad_x * 2 + T.text_w(hint, hint_scale, 1, T.FONT_BODY))
    T.fill(display, 0, fh - bar_h, bar_w, fh, T.SCRIM, 0.85)
    T.text(display, hint, hpad_x, fh - hpad_y, hint_scale, T.PAPER,
           1, T.FONT_BODY)

    # ── Error callout, above the hint bar ─────────────────────
    if error_msg:
        err_scale = T.fs(12.5, u)
        err_h = T.text_h(err_scale, 1, T.FONT_BODY)
        card_h = hpad_y * 2 + err_h
        top = fh - bar_h - card_h - T.px(2, u)
        card_w = min(fw, hpad_x * 2 + T.px(3, u)
                     + T.text_w(error_msg, err_scale, 1, T.FONT_BODY))
        T.fill(display, 0, top, card_w, top + card_h, T.ACCENT_200, 0.94)
        T.edge(display, 0, top, top + card_h, T.ACCENT, T.px(3, u))
        T.text(display, error_msg, T.px(3, u) + hpad_x,
               top + hpad_y + err_h, err_scale, T.ACCENT_900,
               1, T.FONT_BODY)


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

    while True:
        display = clone.copy()

        _draw_calibration_overlay(display, points, room_point, error_msg)

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

    return door
